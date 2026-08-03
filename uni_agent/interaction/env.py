import re
import shlex
from pathlib import Path, PurePath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from swerex.exceptions import BashIncorrectSyntaxError, CommandTimeoutError
from swerex.runtime.abstract import (
    BashAction,
    BashInterruptAction,
    Command,
    CreateBashSessionRequest,
    ReadFileRequest,
    UploadRequest,
    WriteFileRequest,
)

from uni_agent.async_logging import get_logger
from uni_agent.deployment import DeployConfig
from uni_agent.skills.manager import SkillsManager
from uni_agent.tools.base import AbstractTool
from uni_agent.utils import auto_await


# A program awaiting input never prints the shell's PS1, so interactive sends must also
# expect the program's own prompt or they wait out the full timeout and return nothing.
# The sandbox returns the prompt it matched as part of the observation, so the model sees
# the real text ("In [3]: ", "[Y/n] ") rather than the pattern that matched it.
INTERACTIVE_PROMPTS = [
    r">>> $",            # python
    r"In \[\d+\]: $",    # ipython
    r"\.\.\. $",         # python continuation
    r"\(Pdb\) $",        # pdb
    r"ipdb> $",
    r"\[Y/n\]",          # apt and friends
    r"\[y/N\]",
    r"\(y/n\)",
    r"File to patch: $",
    r"\? $",
]

# Reads that only collect what is ALREADY buffered wait this long: peeking for a pending
# shell prompt, and getting past a prompt a timed-out command left behind. A hit is
# instant, so this only bounds the miss. Both send nothing, so neither disturbs the
# program, which is what lets the second one run at ANY prompt.
BUFFERED_READ_TIMEOUT = 0.5

# Every path that frees the session says so: without it the model is left thinking it
# still holds one, and spends its next turn on an is_input that can only be refused.
FREED_NOTE = "\n<NOTE>The program is no longer running: the session is free, so run a new command.</NOTE>"


class ActionTimeoutError(Exception):
    pass


class ActionIncorrectSyntaxError(Exception):
    pass


class TerminalNotAliveError(Exception):
    pass


class EnvAction(BaseModel):
    """A single action to run in the environment session.

    Attributes:
        command: The bash command to run; or, when ``is_input`` is True, the
            input to send to the currently running interactive program
            (``"C-c"`` interrupts it).
        is_input: Send ``command`` to the running interactive program instead of
            starting a new command.
        timeout: Optional per-action timeout in seconds; callers fall back to
            their own default when this is ``None``.
    """

    command: str
    is_input: bool = False
    timeout: int | None = None


class AgentEnvConfig(BaseModel):
    deployment: DeployConfig = Field(description="Deployment configuration")
    env_variables: dict[str, str] | None = Field(
        default=None, description="Optional environment variables to set after start"
    )
    post_setup_cmd: str | None = Field(default=None, description="Command to run after environment startup")
    privileged_setup_cmd: str | None = Field(
        default=None,
        description=(
            "Command run right after post_setup_cmd, in the same session. Its stdout is kept in "
            "memory as privileged context (the reflector's reference patch) and is never shown to "
            "the agent, so this is where task state the agent must not reach is read and destroyed."
        ),
    )
    tool_install_dir: Path = Field(
        default=Path("/usr/local/bin"), description="Directory where tool scripts are installed"
    )
    model_config = ConfigDict(extra="forbid")


class AgentEnv:
    def __init__(
        self,
        run_id: str,
        env_config: AgentEnvConfig,
    ):
        """
        This class represents the environment in which we solve the tasks.

        Args:
            run_id: Run ID for the environment
            env_config: environment configuration
        """
        super().__init__()
        self.deployment = env_config.deployment.get_deployment(run_id)
        self.env_variables = env_config.env_variables
        self.post_setup_cmd = env_config.post_setup_cmd
        self.privileged_setup_cmd = env_config.privileged_setup_cmd
        self.tool_install_dir = env_config.tool_install_dir
        # stdout of privileged_setup_cmd; read by the reflector, never sent to the agent
        self.privileged_context: str = ""
        self.logger = get_logger("environment", run_id)
        # command still attached to the session after a timeout; while set, only input
        # may be sent, so a new command is never typed into a running program
        self.attached_command: str | None = None
        # lazily created side session for reward-time commands (communicate_isolated)
        self._isolated_session: str | None = None
        # output of the attached command already shown: a timeout does not consume the
        # terminal buffer, so the next read replays this prefix
        self.attached_shown: str = ""
        # in-env seconds spent on the attached command, summed across yields and sends.
        # Wall clock would count the model's own thinking time and kill healthy REPL
        # sessions; this only grows while we are actually blocked on the terminal.
        self.attached_seconds: float = 0.0
        # the attached program last showed us its own prompt, so it is idle and alive
        self.attached_at_prompt: bool = False

    @auto_await
    async def start(self, max_retries: int = 5) -> None:
        """Start the environment"""

        self.logger.info("Beginning environment startup...")

        await self.deployment.start(max_retries=max_retries)
        self.logger.info("Runtime initialized")
        # each step logs on entry: a stalled call used to leave a silent gap between
        # "Runtime initialized" and the setup timeout, with no way to tell which one hung
        if self.env_variables:
            self.logger.info("Setting env variables...")
            await self.set_env_variables(self.env_variables)
        if self.post_setup_cmd:
            self.logger.info("Running post_setup_cmd...")
            await self.communicate(self.post_setup_cmd, check="raise")
        if self.privileged_setup_cmd:
            output = await self.communicate(self.privileged_setup_cmd, timeout=300, check="raise")
            self.privileged_context = output.strip()
            self.logger.info(f"Captured {len(self.privileged_context)} chars of privileged context")

    @auto_await
    async def install_tools(self, tools: list[AbstractTool]) -> None:
        self.logger.info(f"Installing {len(tools)} tools...")
        install_dir = self.tool_install_dir
        await self.communicate(f"export PATH={shlex.quote(install_dir.as_posix())}:$PATH", check="raise")
        for tool in tools:
            tool_name = tool.name
            if tool.copy_to_remote:
                local_tool_path = tool.local_path
                assert local_tool_path is not None and local_tool_path.is_file(), (
                    f"Tool {tool_name} has copy_to_remote=True but local_path={local_tool_path!r} is not a file"
                )
                container_tool_path = install_dir / tool_name
                await self.copy_to_container(
                    src=local_tool_path,
                    tgt=container_tool_path,
                )
                await self.communicate(f"chmod +x {container_tool_path.as_posix()}", check="raise")
            install_cmd = tool.get_install_command()
            if install_cmd:
                await self.communicate(install_cmd, check="raise")
            # check if tool is installed
            await self.communicate(f"which {tool_name}", check="raise", error_msg=f"Failed to install tool {tool_name}")
            self.logger.info(f"Tool {tool_name} successfully installed")

    @auto_await
    async def copy_to_container(self, src: Path, tgt: Path) -> None:
        await self.deployment.runtime.execute(Command(command=["mkdir", "-p", str(tgt.parent)]))
        await self.deployment.runtime.upload(UploadRequest(source_path=str(src), target_path=str(tgt)))

    @auto_await
    async def install_skills(self, skills_manager: "SkillsManager") -> None:
        """Resolve each skill's runtime path and (if needed) copy it in.

        Mutates ``skills_manager.runtime_paths`` so the subsequent
        ``build_manifest`` call renders the right ``<location>`` for each
        skill:

        - **Host-style runtime** (``HostDeployment`` / ``LocalNativeDeployment``):
          skills are read in place from their host ``source_dir``; no copy.
        - **Container runtime** (everything else): each skill directory is
          uploaded to ``/opt/uni-agent/skills/<name>``.
        """
        from uni_agent.deployment.host.deployment import HostDeployment

        host_types: tuple[type, ...] = (HostDeployment,)
        try:
            from uni_agent.deployment.local_native.deployment import LocalNativeDeployment

            host_types = host_types + (LocalNativeDeployment,)
        except ImportError:
            pass

        if isinstance(self.deployment, host_types):
            for skill in skills_manager.skills:
                skills_manager.runtime_paths[skill.name] = skill.source_dir
            names = "\n".join(s.name for s in skills_manager.skills)
            self.logger.info(f"Host runtime: {len(skills_manager.skills)} skill(s) read in place, no copy\n{names}")
            return

        for skill in skills_manager.skills:
            tgt = Path("/opt/uni-agent/skills") / skill.name
            await self.copy_to_container(src=skill.source_dir, tgt=tgt)
            skills_manager.runtime_paths[skill.name] = tgt
            self.logger.info(f"Skill {skill.name} installed at {tgt}")
        self.logger.info(f"Installed {len(skills_manager.skills)} skill(s) into runtime")

    @auto_await
    async def close(self) -> None:
        """Shutdown SWE-ReX deployment etc."""
        self.logger.info("Beginning environment shutdown...")
        try:
            await self.deployment.stop()
        except Exception as e:
            self.logger.error(f"Failed to stop environment deployment: {e}")
            return
        self.logger.info("Environment shutdown completed")

    @staticmethod
    def _format_observation(raw: str, max_observation_length: int, empty_message: str) -> str:
        """Strip control chars and wrap raw session output as an observation.

        Shared by :meth:`run_action` and :meth:`send_input`. Returns
        ``empty_message`` when there is no output, an ``Observation:``-prefixed
        block otherwise, and a clipped block with a NOTE when the output exceeds
        ``max_observation_length``.
        """
        cleaned = re.sub(r"\x1b\[[0-9;]*m|\r", "", raw or "").strip()
        if cleaned == "":
            return empty_message
        if len(cleaned) > max_observation_length:
            # keep both ends: a test run's verdict is its last lines, and head-only
            # truncation deletes exactly that
            head = max_observation_length // 2
            tail = max_observation_length - head
            elided = len(cleaned) - max_observation_length
            return (
                f"Observation:\n{cleaned[:head]}\n"
                f"<response clipped: {elided} characters elided from the middle>\n"
                f"{cleaned[-tail:]}\n"
                f"<NOTE>Observation exceeded {max_observation_length} characters. "
                "Run a command that produces less output, or pipe through head/tail/grep or redirect to a "
                "file. Do not use interactive pagers.</NOTE>"
            )
        return f"Observation:\n{cleaned}"

    def _detach(self) -> None:
        """Forget the attached command: the session is free for a new one."""
        self.attached_command = None
        self.attached_shown = ""
        self.attached_seconds = 0.0
        self.attached_at_prompt = False

    async def _settle(self) -> str:
        """Give the shell a prompt of its own, and return what it said at it.

        A program that ignores SIGINT is escalated to ``kill -9 %1``, and bash announces
        a job's fate at its NEXT prompt, not the one it is already at. Without spending a
        prompt here, "[1]+ Killed ..." lands on top of whatever the model runs next and
        reads as that command's output.
        """
        try:
            r = await self.deployment.runtime.run_in_session(
                BashAction(command=":", timeout=BUFFERED_READ_TIMEOUT, check="ignore")
            )
        except Exception:
            return ""
        late = (getattr(r, "output", "") or "").strip()
        return f"\n{late}" if late else ""

    async def _read_pending(self) -> str:
        """Whatever the attached program has printed but not shown yet, if anything."""
        try:
            r = await self.deployment.runtime.run_in_session(
                BashAction(
                    command="",
                    timeout=BUFFERED_READ_TIMEOUT,
                    is_interactive_command=True,
                    check="ignore",
                    expect=INTERACTIVE_PROMPTS,
                )
            )
        except CommandTimeoutError:
            return ""
        return self._unshown(r.output)

    async def _probe_attached(self) -> str | None:
        """Output left behind if the attached command already finished, else ``None``.

        An empty interactive command reads without typing anything, and expecting only
        the shell's own prompt means a program that is still running matches nothing and
        so has nothing consumed.
        """
        try:
            r = await self.deployment.runtime.run_in_session(
                BashAction(
                    command="",
                    timeout=BUFFERED_READ_TIMEOUT,
                    is_interactive_command=True,
                    check="ignore",
                    expect=[],
                )
            )
        except CommandTimeoutError:
            return None
        return self._unshown(r.output)

    @auto_await
    async def kill_attached(self) -> str:
        """Interrupt the attached command and free the session, whatever it was doing."""
        try:
            obs = await self.deployment.runtime.run_in_session(BashInterruptAction(timeout=2))
            output = self._unshown(getattr(obs, "output", "") or "")
        except Exception as e:
            self.logger.error(f"Failed to kill attached command: {e}")
            output = ""
        # same deferred job notice a C-c has to absorb: without this the wall's own kill
        # drops "[1]+ Killed ..." onto whatever the model runs next
        output += await self._settle()
        self._detach()
        return output.strip()

    @staticmethod
    def _without_prompt(output: str, matched: str) -> str:
        """What the program actually said, with the prompt it stopped at taken off.

        The sandbox appends the matched prompt to the output, so "did this read return
        anything" cannot be answered by looking at the output as a whole: a read that
        only re-consumed a stale prompt still comes back carrying that prompt.
        """
        return re.sub(matched, "", output or "")

    def _unshown(self, output: str) -> str:
        """Drop the prefix already shown for the attached command.

        A read that matches consumes the prompt, so its output can be *shorter* than
        what the timeout showed (which kept the prompt it never matched): that read
        carried nothing new.
        """
        shown = self.attached_shown
        if not shown:
            return output
        if output.startswith(shown):
            return output[len(shown) :]
        return "" if shown.startswith(output) else output

    def _partial_from(self, exc: Exception) -> str:
        """What a timed-out command printed but we have not shown yet.

        SWE-ReX carries it on ``extra_info`` (sandboxes built before that patch send
        nothing, which degrades to the old behaviour of yielding blind).
        """
        buffered = (getattr(exc, "extra_info", None) or {}).get("output") or ""
        new = self._unshown(buffered)
        self.attached_shown = buffered
        # it did not answer, so it is working rather than parked at its prompt
        self.attached_at_prompt = False
        return new

    @auto_await
    async def run_action(self, action_cmd: str, action_timeout: int, max_observation_length: int = 100_000) -> str:
        """Run a new bash command in the session and return its observation."""
        try:
            observation = await self.communicate(input=action_cmd, timeout=action_timeout, check="ignore")
        except CommandTimeoutError as e:
            # the command keeps running: killing it here would destroy the state that
            # is_input exists to reach. Mark the session attached instead; the next
            # non-input command is refused rather than typed into the running program.
            self._detach()
            self.attached_command = action_cmd
            observation = self._format_observation(
                self._partial_from(e),
                max_observation_length,
                empty_message="It has produced no output yet.",
            )
            raise ActionTimeoutError(observation) from None

        except BashIncorrectSyntaxError as e:
            # this should not happen, so add critical logs here
            self.logger.error("Action command has incorrect syntax")
            error_message = (
                "Your bash command contained syntax errors and was NOT executed. "
                "Please fix the syntax errors and try again. This can be the result "
                "of not adhering to the syntax for multi-line commands. Here is the output of `bash -n`:\n"
                f"{e.extra_info['bash_stdout']}\n{e.extra_info['bash_stderr']}"
            )
            raise ActionIncorrectSyntaxError(error_message) from None

        return self._format_observation(
            observation,
            max_observation_length,
            empty_message="Your command ran successfully and did not produce any output.",
        )

    @auto_await
    async def send_input(self, payload: str, action_timeout: int, max_observation_length: int = 100_000) -> str:
        """Send input to a program already running in the persistent session.

        ``payload == "C-c"`` interrupts the running process, ``"C-d"`` sends EOF (a
        REPL ignores SIGINT but exits on EOF); any other payload is sent as a line of
        input via SWE-ReX's interactive-command mode (no exit code / PS1 seek). A
        timeout here is benign: the program is still running or awaiting more input,
        and the session is NOT interrupted.

        The interactive mode waits for ``expect + [PS1]``, and a live program never
        prints the shell's PS1, so without ``INTERACTIVE_PROMPTS`` every send would
        wait out the full timeout and return nothing.
        """
        # A command that finished while the model was deciding leaves the shell's prompt
        # pending, and sending now would type the payload into the shell: "y" becomes a
        # bash command, and C-d closes the session. Nothing to check when the program is
        # parked at its own prompt, since it cannot have exited on its own from there.
        if payload.strip() and not self.attached_at_prompt:
            leftover = await self._probe_attached()
            if leftover is not None:
                finished = self.attached_command
                self._detach()
                return self._format_observation(
                    leftover,
                    max_observation_length,
                    empty_message="It produced no further output.",
                ) + (
                    f"\n<NOTE>'{finished}' had already finished, so your input was NOT sent to it. "
                    "The session is free again: run a new command.</NOTE>"
                )

        if payload.strip() == "C-d":
            try:
                r = await self.deployment.runtime.run_in_session(
                    BashAction(command="\x04", timeout=action_timeout, is_interactive_quit=True, check="ignore")
                )
            except CommandTimeoutError:
                # EOF does not always mean exit: ipython answers it with "Do you really
                # want to exit ([y]/n)?". Waiting for a shell prompt that is never coming
                # burns the whole timeout and reports a failure that did not happen, so
                # collect what it asked instead and leave it attached.
                return self._format_observation(
                    await self._read_pending(),
                    max_observation_length,
                    empty_message="Sent EOF (C-d), but the program has not exited and printed nothing.",
                ) + (
                    "\n<NOTE>It is still running and may be asking you to confirm. Answer it with "
                    'is_input=true, or send "C-c" to interrupt it.</NOTE>'
                )
            except Exception as e:
                self.logger.error(f"Failed to send EOF: {e}")
                return "Failed to send EOF (C-d) to the running process."
            output = self._unshown(getattr(r, "output", "") or "")
            self._detach()
            # EOF is sent as a keystroke followed by a newline, so a program that answers
            # it by asking "really exit?" gets that newline as its default and exits. The
            # question is in the output but was already dealt with: say so, or the model
            # reads it as still waiting and burns a turn answering it.
            return self._format_observation(
                output,
                max_observation_length,
                empty_message="Sent EOF (C-d); it exited without printing anything.",
            ) + (
                "\n<NOTE>The program exited on EOF, so any prompt still visible above has already "
                "been answered. The session is free, so run a new command.</NOTE>"
            )
        if payload.strip() == "C-c":
            try:
                obs = await self.deployment.runtime.run_in_session(BashInterruptAction(timeout=2))
            except Exception as e:
                self.logger.error(f"Failed to interrupt session: {e}")
                return "Failed to send interrupt (C-c) to the running process."
            output = self._unshown(getattr(obs, "output", "") or "") + await self._settle()
            self._detach()
            return self._format_observation(
                output,
                max_observation_length,
                empty_message="Sent interrupt (C-c); it printed nothing before stopping.",
            ) + FREED_NOTE

        try:
            r = await self.deployment.runtime.run_in_session(
                BashAction(
                    command=payload,
                    timeout=action_timeout,
                    is_interactive_command=True,
                    check="ignore",
                    expect=INTERACTIVE_PROMPTS,
                )
            )
        except CommandTimeoutError as e:
            observation = self._format_observation(
                self._partial_from(e),
                max_observation_length,
                empty_message="No new output yet.",
            )
            # the program is still there and still reachable, so this is a yield like any
            # other: the caller decides whether to keep waiting or give up on it
            raise ActionTimeoutError(observation) from None
        matched = getattr(r, "expect_string", "")
        output = self._unshown(r.output)
        # The prompt left unread by the timed-out command is still buffered, so this
        # first match consumed that one and returned everything before it: nothing.
        # One more read lands on the prompt that follows the actual output. It sends
        # nothing, so it is safe even where a newline would mean something ("..." ends
        # the block, "(Pdb)" repeats, "[Y/n]" answers the default), and it only has to
        # collect what is already buffered, so it does not wait long for it.
        if matched in INTERACTIVE_PROMPTS and not self._without_prompt(output, matched).strip():
            try:
                r2 = await self.deployment.runtime.run_in_session(
                    BashAction(
                        command="",
                        timeout=BUFFERED_READ_TIMEOUT,
                        is_interactive_command=True,
                        check="ignore",
                        expect=INTERACTIVE_PROMPTS,
                    )
                )
                output, matched = self._unshown(r2.output), getattr(r2, "expect_string", "")
            except CommandTimeoutError:
                pass
        # a read that matched consumed the buffer, so nothing is owed to the next one
        self.attached_shown = ""
        # a matched shell PS1 means the program exited and the session is free again;
        # matching one of INTERACTIVE_PROMPTS means it is still waiting on us
        exited = matched not in INTERACTIVE_PROMPTS
        if exited:
            self._detach()
        else:
            self.attached_at_prompt = True
        observation = self._format_observation(
            output,
            max_observation_length,
            empty_message=(
                "Your input made it exit without printing anything."
                if exited
                else "Your input was sent; the program produced no output yet. "
                "Send an empty command to collect more output."
            ),
        )
        # without this the model is told to keep polling a session it no longer holds,
        # and spends its next turn on an is_input that can only be refused
        if exited:
            observation += FREED_NOTE
        return observation

    @auto_await
    async def interrupt_session(self) -> str:
        """Interrupt whatever is running; returns the output it had produced so far.

        swe-rex retries the SIGINT ``n_retry`` times at this timeout before escalating,
        so a large value is paid in full against programs that ignore SIGINT (a REPL).
        """
        self.logger.info("Interrupting session")
        obs = await self.deployment.runtime.run_in_session(BashInterruptAction(timeout=2))
        return getattr(obs, "output", "") or ""

    @auto_await
    async def clear_attached(self) -> None:
        """Free the session if a timed-out command is still attached when the trajectory ends.

        The model no longer owns the session, but reward-side commands (patch
        extraction) still run in it, and a live leftover process swallows them until
        their own timeout. C-c first; a REPL ignores SIGINT, so C-d next; each attempt
        is verified with a no-op command because ``send_input`` clears the flag
        optimistically.
        """
        if self.attached_command is None:
            return
        cmd = self.attached_command
        self.logger.info(f"Trajectory ended with a command still attached; interrupting: '{cmd}'")
        for payload in ("C-c", "C-d"):
            await self.send_input(payload, action_timeout=5)
            try:
                await self.communicate("true", timeout=5, check="ignore")
                self.attached_command = None
                return
            except CommandTimeoutError:
                continue
        self.attached_command = None
        self.logger.warning(f"Session still busy after C-c/C-d; leftover command: '{cmd}'")

    @auto_await
    async def communicate_isolated(self, input: str, timeout: int | float = 60) -> str:
        """Run a command in a dedicated side session, immune to whatever the agent left
        running or broke in the main one (used for reward-side inspection like patch
        extraction). The session is created lazily on first use."""
        if self._isolated_session is None:
            await self.deployment.runtime.create_session(
                CreateBashSessionRequest(session="uniagent-reward", startup_timeout=30)
            )
            self._isolated_session = "uniagent-reward"
        r = await self.deployment.runtime.run_in_session(
            BashAction(command=input, timeout=timeout, check="ignore", session=self._isolated_session)
        )
        return r.output

    @auto_await
    async def communicate(
        self,
        input: str,
        timeout: int | float = 60,
        check: Literal["warn", "ignore", "raise"] = "ignore",
        error_msg: str = "Command failed",
    ) -> str:
        """Executes a command in the running shell. The details of this are handled by
        the SWE-ReX deployment/runtime.

        Args:
            input: input to send to container
            timeout_duration: duration to wait for output
            check: `ignore`: do not extract exit code (more stable), `warn`: extract exit code and log error if
                exit code is non-zero, `raise`: raise error if exit code is non-zero
            error_msg: error message to raise if the command fails

        Returns:
            output: output from container
        """
        self.logger.debug(f"Input:\n{input}")
        # `check` is a Literal string, so a truthiness test always picked "silent" and
        # paid swe-rex's extra exit-code round trip on every command
        rex_check = "ignore" if check == "ignore" else "silent"
        r = await self.deployment.runtime.run_in_session(BashAction(command=input, timeout=timeout, check=rex_check))
        output = r.output
        self.logger.debug(f"Output:\n{output}")
        if check != "ignore" and r.exit_code != 0:
            self.logger.error(f"{error_msg}:\n{output}")
            msg = f"Command {input!r} failed ({r.exit_code=}): {error_msg}"
            if check == "raise":
                await self.close()
                raise RuntimeError(msg)
        return output

    @auto_await
    async def read_file(self, path: str | PurePath, encoding: str | None = None, errors: str | None = None) -> str:
        """Read file contents from container

        Args:
            path: Absolute path to file
            encoding: Encoding to use when reading the file. None means default encoding.
                This is the same as the `encoding` argument of `Path.read_text()`
            errors: Error handling to use when reading the file. None means default error handling.
                This is the same as the `errors` argument of `Path.read_text()`

        Returns:
            file_contents: Contents of file as string
        """
        r = await self.deployment.runtime.read_file(ReadFileRequest(path=str(path), encoding=encoding, errors=errors))
        return r.content

    @auto_await
    async def write_file(self, path: str | PurePath, content: str) -> None:
        """Write content to file in container"""
        await self.deployment.runtime.write_file(WriteFileRequest(path=str(path), content=content))

    @auto_await
    async def set_env_variables(self, env_variables: dict[str, str]) -> None:
        """Set environment variables in the environment."""
        _env_setters = [f"export {k}={shlex.quote(str(v))}" for k, v in env_variables.items()]
        command = " && ".join(_env_setters)
        await self.communicate(command, check="raise")
