"""SWE-smith reward spec — the swe_bench.py counterpart for SWE-smith instances.

Mirrors :mod:`uni_agent.reward.swe_bench` in structure and behavior, but evaluates
through the **swesmith** library instead of swebench's repo-version spec maps (SWE-smith's
~128 repos are not in ``MAP_REPO_VERSION_TO_SPECS``). The differences are confined to how
the eval script is built and how its output is parsed/graded:

  * **Image layout.** Each SWE-smith image holds one repo@commit with a *branch per bug*
    (named by ``instance_id``). The branch has two commits: (1) the bug, (2) removal of the
    held-out F2P test files on top. So:
      - the **agent** runs at ``git checkout {instance_id}`` (HEAD = bug applied, tests hidden);
      - **eval** does ``checkout {instance_id}`` then ``checkout HEAD~1`` to bring the tests
        back, applies the agent's patch, reverts the test files, and runs the suite.
  * **Test command + parser** come from the repo's swesmith *profile*
    (``registry.get_from_inst(instance)`` -> ``rp.get_test_cmd`` / ``rp.log_parser``).
  * **Grading** uses swesmith's ``get_eval_tests_report`` + swebench's ``get_resolution_status``.
  * **Gold patch** = the (bug) ``patch`` reversed.

Feedback rendering is identical, so we reuse :class:`~uni_agent.reward.swe_bench.FeedbackConfig`.
"""

import re
import time
import uuid
from pathlib import Path

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    FAIL_TO_PASS,
    PASS_TO_PASS,
    TESTS_TIMEOUT,
    ResolvedStatus,
)
from swebench.harness.grading import get_resolution_status
from swesmith.constants import GIT_APPLY_CMDS, TEST_OUTPUT_END, TEST_OUTPUT_START
from swesmith.harness.grading import get_eval_tests_report
from swesmith.profiles import registry

from uni_agent.async_logging import get_logger
from uni_agent.interaction import AgentEnv
from uni_agent.reward.base import AbstractRewardSpec
from uni_agent.reward.registry import register_reward_spec
from uni_agent.reward.swe_bench import FeedbackConfig
from uni_agent.utils import auto_await

#: HEREDOC delimiter unlikely to appear in a diff (matches swe_bench's convention).
_HEREDOC_DELIMITER = "EOF_114329324912"


def _make_eval_script_list(instance_id, patch, test_command, test_files):
    """Build the SWE-smith eval script (run in a clean checkout of the instance image).

    Replicates ``swesmith.harness.utils.run_patch_in_container`` as a single bash
    script so it can run through ``env.communicate`` (like swe_bench's eval), rather
    than via the host docker SDK. ``set -uxo pipefail`` (no ``-e``) so a failed apply
    does not abort the script — we surface it via the ``APPLY_PATCH_FAIL`` sentinel,
    exactly as the library does.
    """
    repo_directory = "/testbed"
    # Apply the agent patch with swesmith's fallback ladder; echo the sentinel on total
    # failure so _get_logs_eval can flag patch_apply_failed (mirrors the library).
    apply_lines = ["_applied=0"]
    for cmd in GIT_APPLY_CMDS:
        apply_lines.append(f'if [ "$_applied" -eq 0 ] && {cmd} /tmp/swesmith_pred.diff; then _applied=1; fi')
    apply_lines.append(f"if [ \"$_applied\" -eq 0 ]; then echo '{APPLY_PATCH_FAIL}'; fi")

    revert_tests = f"git checkout -- {' '.join(test_files)}" if test_files else "echo 'no test files to reset'"

    return [
        f"cd {repo_directory}",
        f"git config --global --add safe.directory {repo_directory}",
        # Bring the instance's bug branch, then step back one commit so the held-out
        # F2P/P2P test files are present again (-f discards any working-tree state).
        f"git checkout -f {instance_id}",
        "git checkout -f HEAD~1",
        # Stage the agent's prediction and apply it.
        f"cat <<'{_HEREDOC_DELIMITER}' > /tmp/swesmith_pred.diff\n{patch}\n{_HEREDOC_DELIMITER}",
        *apply_lines,
        # Tests are graded from the repo's own copy — discard any agent edits to them.
        revert_tests,
        f": '{TEST_OUTPUT_START}'",
        test_command,
        f": '{TEST_OUTPUT_END}'",
    ]


@register_reward_spec("swe_smith")
class SWESmithRewardSpec(AbstractRewardSpec):
    def __init__(
        self,
        *,
        run_id: str,
        metadata: dict,
        env: AgentEnv,
        eval_timeout: int = 300,
        feedback: dict | FeedbackConfig | None = None,
        env_config: dict | None = None,
        isolate: bool = False,
    ):
        self.run_id = run_id
        self.metadata = metadata
        self.env = env
        self.logger = get_logger("reward_spec", run_id=run_id)
        self.eval_timeout = eval_timeout
        self.feedback = feedback if isinstance(feedback, FeedbackConfig) else FeedbackConfig(**(feedback or {}))
        self.env_config = env_config
        self.isolate = isolate

    @auto_await
    async def apply_gold_patch(self) -> None:
        """Apply the gold fix = the bug ``patch`` reversed (SWE-smith convention)."""
        await self._apply_patch(self.metadata["patch"], reverse=True)

    @auto_await
    async def compute_reward(self, **kwargs) -> tuple[bool, dict]:
        """Run the SWE-smith eval and grade the result. Returns ``(resolved, result)``.

        With ``isolate=True`` (recommended; the agent container is mid-trajectory on the
        bug branch) the eval runs in a fresh sibling deployment from the same image.
        """
        result = {
            "eval_completed": False,
            "eval_execution_time": None,
            "eval_report": None,
            "resolved": False,
        }

        instance = self.metadata
        instance_id = instance["instance_id"]
        rp = registry.get_from_inst(instance)
        # Whole-suite command (f2p_only would drop P2P tests in other files -> false regressions).
        test_command, _ = rp.get_test_cmd(instance, f2p_only=False)
        try:
            f2p_files, p2p_files = rp.get_test_files(instance)
            test_files = sorted(set(f2p_files + p2p_files))
        except Exception:
            test_files = []

        # The agent's diff IS the prediction we evaluate, so we always need it.
        patch = await self._get_interaction_env_patch()
        eval_script_list = _make_eval_script_list(instance_id, patch, test_command, test_files)
        eval_script = "\n".join(["#!/bin/bash", "set -uxo pipefail"] + eval_script_list) + "\n"

        output = ""
        eval_env = self.env
        sibling = None
        try:
            if self.isolate:
                env_config = kwargs.get("env_config") or self.env_config
                sibling = await self._start_sibling_env(env_config)
                eval_env = sibling

            eval_script_container = Path(f"/tmp/eval_script_{uuid.uuid4()}.sh")
            await eval_env.write_file(eval_script_container, eval_script)

            execution_t0 = time.perf_counter()
            output = await eval_env.communicate(
                f"bash {eval_script_container}",
                timeout=self.eval_timeout,
                check="ignore",
            )
            result["eval_execution_time"] = time.perf_counter() - execution_t0
            result["eval_completed"] = True

            # Strip ANSI escapes / carriage returns before parsing.
            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)

            eval_report = self._get_eval_report(output)
            result["eval_report"] = eval_report
            self.logger.info(f"Eval report: {eval_report}")
            result["resolved"] = eval_report["resolved"]
            if not eval_report["found_eval_status"] and APPLY_PATCH_FAIL in output:
                result["patch_apply_failed"] = True
        except Exception as e:
            self.logger.error(f"Failed to evaluate: {e}")
        finally:
            if sibling is not None:
                try:
                    await sibling.close()
                except Exception as e:
                    self.logger.error(f"Failed to close sibling eval env: {e}")

        if self.feedback.enabled:
            feedback = self.feedback.render(
                result=result,
                output=output,
                patch=patch,
                instance_id=instance.get("instance_id", ""),
            )
            result["reward_extra_info"] = {"feedback": feedback}

        return result["resolved"], result

    @auto_await
    async def _start_sibling_env(self, env_config: dict | None):
        """Start a fresh deployment from the same env config for isolated eval.

        The eval relies on the instance's pristine git history (checkout {id} -> HEAD~1
        restores the held-out F2P/P2P tests). The agent-side ``post_setup_cmd`` flattens
        that history to block git-archaeology reward-hacking, so it must NOT run in the
        eval env -- it would leave the tests deleted and force every rollout to score 0.
        """
        if not env_config:
            raise RuntimeError("isolate=True requires env_config; pass the agent-loop env config into compute_reward.")
        from uni_agent.interaction import AgentEnv, AgentEnvConfig

        sibling = AgentEnv(
            run_id=f"{self.run_id}-eval",
            env_config=AgentEnvConfig(**{**env_config, "post_setup_cmd": None}),
        )
        await sibling.start()
        self.logger.info("Started isolated sibling eval environment")
        return sibling

    @auto_await
    async def _get_interaction_env_patch(self) -> str:
        """Get the current staged diff in /testbed (interaction env state) as a patch string."""
        try:
            env_patch_file = Path(f"/tmp/patch_{uuid.uuid4()}.diff")
            await self.env.communicate(
                f"cd /testbed && git add -A && git diff --no-color --cached > {env_patch_file.as_posix()}",
                check="ignore",
            )
            return await self.env.read_file(env_patch_file)
        except Exception as e:
            self.logger.error(f"Failed to get interaction environment patch: {e}")
            return ""

    @auto_await
    async def _apply_patch(self, patch: str, reverse: bool = False) -> None:
        """Apply a patch string in the interaction env, trying swesmith's apply ladder;
        ``reverse=True`` reverts it (used for the gold = bug-reversed patch).
        """
        env = self.env
        if not patch or not patch.strip():
            self.logger.info("Empty patch, nothing to apply.")
            return
        patch_path = Path(f"/tmp/patch_{uuid.uuid4()}.diff")
        await env.write_file(patch_path, patch)
        rev = " --reverse" if reverse else ""
        last_error: Exception | None = None
        for git_apply_cmd in GIT_APPLY_CMDS:
            cmd = f"cd /testbed && {git_apply_cmd}{rev} {patch_path.as_posix()}"
            try:
                await env.communicate(cmd, check="raise")
                self.logger.info("Applied patch successfully!")
                return
            except RuntimeError as e:
                last_error = e
                continue
        raise RuntimeError("Failed to apply patch with any command") from last_error

    def _get_logs_eval(self, eval_output: str):
        """Extract the test-output section and parse it with the repo profile's log parser."""
        if APPLY_PATCH_FAIL in eval_output or TESTS_TIMEOUT in eval_output:
            return {}, False
        if TEST_OUTPUT_START in eval_output and TEST_OUTPUT_END in eval_output:
            test_content = eval_output.split(TEST_OUTPUT_START, 1)[1].split(TEST_OUTPUT_END, 1)[0]
            rp = registry.get_from_inst(self.metadata)
            return rp.log_parser(test_content), True
        return {}, False

    def _get_eval_report(self, eval_output: str):
        eval_report = {
            "resolved": False,
            "found_eval_status": False,
            "test_status": None,
        }

        status_map, found = self._get_logs_eval(eval_output)
        eval_report["found_eval_status"] = found
        if not found:
            return eval_report

        gold_results = {
            FAIL_TO_PASS: list(self.metadata.get(FAIL_TO_PASS, []) or []),
            PASS_TO_PASS: list(self.metadata.get(PASS_TO_PASS, []) or []),
        }
        report = get_eval_tests_report(status_map, gold_results)
        eval_report["test_status"] = report
        if get_resolution_status(report) == ResolvedStatus.FULL.value:
            eval_report["resolved"] = True
        return eval_report
