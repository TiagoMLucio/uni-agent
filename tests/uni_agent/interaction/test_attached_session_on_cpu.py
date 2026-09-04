"""A timed-out command stays attached to the session; only input may follow it."""

from __future__ import annotations

import asyncio
import types

import pytest
from swerex.exceptions import CommandTimeoutError

from uni_agent.interaction.env import ActionTimeoutError, AgentEnv


class _Obs:
    def __init__(self, output: str = "", expect_string: str = ""):
        self.output = output
        self.expect_string = expect_string
        self.exit_code = 0


def _is_peek(action) -> bool:
    """The liveness peek: interactive, sends nothing, expects only the shell prompt."""
    return (
        getattr(action, "is_interactive_command", False)
        and getattr(action, "command", "") == ""
        and not getattr(action, "expect", None)
    )


class _Runtime:
    def __init__(self):
        self.script: list = []
        self.finished: _Obs | None = None  # what the liveness peek finds; None = still running
        self.sent: list[str] = []  # every command actually typed into the session

    async def run_in_session(self, action):
        self.sent.append(getattr(action, "command", None))
        # the peek is the only read with an empty command AND no expect list, so a
        # running program matches nothing and times out
        if _is_peek(action):
            if self.finished is None:
                raise CommandTimeoutError("nothing pending")
            return self.finished
        if self.script:
            nxt = self.script.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt
        return _Obs("ok", "PS1")


def _timeout(output: str | None = None) -> CommandTimeoutError:
    exc = CommandTimeoutError("timeout")
    if output is not None:
        exc.extra_info = {"output": output}
    return exc


def _env():
    e = AgentEnv.__new__(AgentEnv)  # __init__ needs a real deployment
    e.attached_command = None
    e.attached_shown = ""
    e.attached_seconds = 0.0
    e.attached_at_prompt = False
    noop = lambda *a, **k: None  # noqa: E731
    e.logger = types.SimpleNamespace(info=noop, error=noop, critical=noop, debug=noop)
    e.deployment = types.SimpleNamespace(runtime=_Runtime())
    return e


def _run(coro):
    return asyncio.run(coro)


def test_timeout_attaches_instead_of_killing():
    env = _env()
    env.deployment.runtime.script = [CommandTimeoutError("timeout")]
    with pytest.raises(ActionTimeoutError) as excinfo:
        _run(AgentEnv.run_action.__wrapped__(env, "python", action_timeout=90))
    assert env.attached_command == "python"
    # the guidance is added a layer up, where the kill budget is known; here it is output
    assert "no output yet" in str(excinfo.value)


def test_input_landing_on_a_repl_prompt_stays_attached():
    env = _env()
    env.attached_command = "python"
    env.deployment.runtime.script = [_Obs("4", ">>> $")]
    out = _run(AgentEnv.send_input.__wrapped__(env, "print(2+2)", action_timeout=90))
    assert env.attached_command == "python"
    assert "4" in out


def test_input_returning_to_the_shell_detaches():
    env = _env()
    env.attached_command = "python"
    env.deployment.runtime.script = [_Obs("done", "PS1MARKER")]
    out = _run(AgentEnv.send_input.__wrapped__(env, "exit()", action_timeout=90))
    assert env.attached_command is None
    # "q" in a debugger used to be answered with "send an empty command to collect more
    # output", which the model can then only spend a turn being refused for
    assert "session is free" in out
    assert "empty command" not in out


@pytest.mark.parametrize("payload", ["exit()", "C-c", "C-d"])
def test_every_detach_path_says_the_session_is_free(payload):
    env = _env()
    env.attached_command = "python"
    env.attached_at_prompt = True  # skip the peek; we are testing the exit message
    env.deployment.runtime.script = [_Obs("goodbye", "PS1MARKER")]
    out = _run(AgentEnv.send_input.__wrapped__(env, payload, action_timeout=90))
    assert env.attached_command is None
    assert "session is free" in out


@pytest.mark.parametrize("key", ["C-c", "C-d"])
def test_control_keys_detach(key):
    env = _env()
    env.attached_command = "python"
    env.deployment.runtime.script = [_Obs("", "")]
    _run(AgentEnv.send_input.__wrapped__(env, key, action_timeout=90))
    assert env.attached_command is None


def test_stale_prompt_triggers_one_follow_up_read():
    # the timed-out command left its prompt unread, so the first match consumes that
    # one and returns nothing; a second read must pick up the real output
    env = _env()
    env.attached_command = "python"
    env.deployment.runtime.script = [_Obs("", ">>> $"), _Obs("42", ">>> $")]
    out = _run(AgentEnv.send_input.__wrapped__(env, "print(6*7)", action_timeout=15))
    assert "42" in out
    assert env.attached_command == "python"  # still in the REPL


@pytest.mark.parametrize(
    "prompt, payload",
    [
        (r"\[Y/n\]", "n"),        # a newline here would answer the default
        (r"\.\.\. $", "if True:"),  # a newline here ends the block
        (r"\(Pdb\) $", "n"),      # a newline here repeats the last command
        (r"ipdb> $", "n"),
    ],
)
def test_the_follow_up_read_runs_at_every_prompt(prompt, payload):
    # It used to be limited to ">>>"-style prompts because a bare newline MEANS something
    # at these. The read types nothing now, so it is safe everywhere; skipping it left
    # ipdb one turn behind on every single command.
    env = _env()
    env.attached_command = "prog"
    env.deployment.runtime.script = [_Obs("", prompt), _Obs("REAL OUTPUT", prompt)]
    out = _run(AgentEnv.send_input.__wrapped__(env, payload, action_timeout=15))
    assert "REAL OUTPUT" in out
    assert env.attached_command == "prog"
    # the safety property the old gate was protecting: the re-read must type nothing
    assert env.deployment.runtime.sent[-1] == ""


def test_timeout_shows_what_the_command_printed_so_far():
    env = _env()
    env.deployment.runtime.script = [_timeout("LINE1\nLINE2\n")]
    with pytest.raises(ActionTimeoutError) as excinfo:
        _run(AgentEnv.run_action.__wrapped__(env, "slow.sh", action_timeout=5))
    assert "LINE1" in str(excinfo.value)
    assert env.attached_shown == "LINE1\nLINE2\n"


def test_a_sandbox_without_the_patch_still_works():
    # older images send no extra_info at all; yielding blind is the documented fallback
    env = _env()
    env.deployment.runtime.script = [_timeout()]
    with pytest.raises(ActionTimeoutError) as excinfo:
        _run(AgentEnv.run_action.__wrapped__(env, "slow.sh", action_timeout=5))
    assert "no output yet" in str(excinfo.value)
    assert env.attached_shown == ""


def test_the_replayed_prefix_is_not_shown_twice():
    # pexpect does not consume the buffer on a timeout, so the read that finally
    # matches returns the partial output again with the rest appended
    env = _env()
    env.attached_command = "slow.sh"
    env.attached_shown = "LINE1\nLINE2\n"
    env.deployment.runtime.script = [_Obs("LINE1\nLINE2\nLINE3\nDONE", "PS1")]
    out = _run(AgentEnv.send_input.__wrapped__(env, "", action_timeout=5))
    assert "LINE3\nDONE" in out
    assert out.count("LINE1") == 0
    assert env.attached_shown == ""


def test_repeated_polls_only_show_the_increment():
    # a send that times out is a yield, not a result: it raises so the caller can
    # count it as such, and the program stays attached
    env = _env()
    env.attached_command = "slow.sh"
    env.attached_shown = "LINE1\n"
    env.deployment.runtime.script = [_timeout("LINE1\nLINE2\n")]
    with pytest.raises(ActionTimeoutError) as excinfo:
        _run(AgentEnv.send_input.__wrapped__(env, "", action_timeout=5))
    out = str(excinfo.value)
    assert "LINE2" in out
    assert out.count("LINE1") == 0
    assert env.attached_shown == "LINE1\nLINE2\n"
    assert env.attached_command == "slow.sh"


def test_an_action_cannot_outlive_what_is_left_of_the_kill_wall():
    # checking the wall only after an action returned meant a 10s request served on 2.9s
    # of remaining budget still ran the full 10s: the wall bounded nothing
    import types as _t

    from uni_agent.interaction.interaction import AgentInteraction

    seen = {}

    class _Env:
        attached_command, attached_seconds, attached_at_prompt = "cat", 2.1, False

        async def send_input(self, command, action_timeout):
            seen["timeout"] = action_timeout
            raise ActionTimeoutError("still running")

        async def kill_attached(self):
            pass

    it = AgentInteraction.__new__(AgentInteraction)
    noop = lambda *a, **k: None  # noqa: E731
    it.env, it.logger = _Env(), _t.SimpleNamespace(info=noop, error=noop, debug=noop)
    it.action_timeout, it.yield_timeout = 30, 5
    it.attached_kill_timeout, it.timeout_budget = 5.0, 1
    it.tools_manager = _t.SimpleNamespace(
        get_tool_action=lambda _tc: _t.SimpleNamespace(command="teste", is_input=True, timeout=10),
        format_args_example=lambda args: str(args),
    )
    tc = _t.SimpleNamespace(id="c", function=_t.SimpleNamespace(name="execute_bash"))
    _run(AgentInteraction._execute_tool_call(it, tc))
    assert seen["timeout"] == 3  # ceil(5.0 - 2.1), not the 10 that was asked for


def test_a_yield_reports_the_balance_but_a_kill_does_not_bother():
    import types as _t

    from uni_agent.interaction.interaction import AgentInteraction

    def _interaction(spent: float, wall: float):
        class _Env:
            attached_command, attached_at_prompt = "cat", False
            attached_seconds = spent

            async def send_input(self, command, action_timeout):
                raise ActionTimeoutError("partial output")

            async def kill_attached(self):
                pass

        noop = lambda *a, **k: None  # noqa: E731
        it = AgentInteraction.__new__(AgentInteraction)
        it.env, it.logger = _Env(), _t.SimpleNamespace(info=noop, error=noop, debug=noop)
        it.action_timeout, it.yield_timeout = 30, 5
        it.attached_kill_timeout, it.timeout_budget = wall, 1
        it.tools_manager = _t.SimpleNamespace(
            get_tool_action=lambda _tc: _t.SimpleNamespace(command="x", is_input=True, timeout=1),
            format_args_example=lambda args: str(args),
        )
        return it

    tc = _t.SimpleNamespace(id="c", function=_t.SimpleNamespace(name="execute_bash"))

    survived = _run(AgentInteraction._execute_tool_call(_interaction(2.0, 45.0), tc))
    assert "used before it is cancelled" in survived.observation
    assert survived.status == "yielded"
    assert survived.observation.count("<NOTE>") == 1, survived.observation

    # a kill never builds the yield note, so it cannot contradict itself
    killed = _run(AgentInteraction._execute_tool_call(_interaction(45.0, 45.0), tc))
    assert "used before it is cancelled" not in killed.observation
    assert 'send "C-d"' not in killed.observation
    assert killed.observation.count("<NOTE>") == 1, killed.observation
    assert killed.status == "timeout"


@pytest.mark.parametrize(
    "budget, expected",
    [
        (1, "One more cancellation ends the episode."),   # 1 -> 0, one kill still allowed
        (2, "2 more cancellations end the episode."),
        (0, "That was the last one allowed, so the episode ends here."),  # 0 -> -1, exhausted
    ],
)
def test_the_kill_note_states_the_real_remaining_allowance(budget, expected):
    # it used to say "repeating this ends the episode" on every kill, which is true only
    # for the first kill at budget 1: at budget 2 it understates, and on the kill that
    # actually ends the run it claims the run is still going
    import types as _t

    from uni_agent.interaction.interaction import AgentInteraction

    class _Env:
        attached_command, attached_seconds, attached_at_prompt = "cat", 99.0, False

        async def send_input(self, command, action_timeout):
            raise ActionTimeoutError("partial")

        async def kill_attached(self):
            return ""

    noop = lambda *a, **k: None  # noqa: E731
    it = AgentInteraction.__new__(AgentInteraction)
    it.env, it.logger = _Env(), _t.SimpleNamespace(info=noop, error=noop, debug=noop)
    it.action_timeout, it.yield_timeout = 30, 5
    it.attached_kill_timeout, it.timeout_budget = 45.0, budget
    it.tools_manager = _t.SimpleNamespace(
        get_tool_action=lambda _tc: _t.SimpleNamespace(command="x", is_input=True, timeout=1),
        format_args_example=lambda args: str(args),
    )
    tc = _t.SimpleNamespace(id="c", function=_t.SimpleNamespace(name="execute_bash"))
    r = _run(AgentInteraction._execute_tool_call(it, tc))
    assert r.status == "timeout"
    assert expected in r.observation, r.observation


def test_a_deferred_job_notice_lands_on_the_interrupt_not_the_next_command():
    # a program that ignores SIGINT gets escalated to `kill -9 %1`, and bash announces
    # the job's fate at its NEXT prompt: "[1]+ Killed ..." used to surface on top of
    # whatever the model ran afterwards and read as that command's output
    env = _env()
    env.attached_command = "ipython3"
    env.deployment.runtime.script = [
        _Obs("KeyboardInterrupt\n[1]+  Stopped   ipython3", "PS1"),  # the interrupt
        _Obs("[1]+  Killed    ipython3", "PS1"),  # what the settling prompt collects
    ]
    out = _run(AgentEnv.send_input.__wrapped__(env, "C-c", action_timeout=5))
    assert "Killed" in out
    assert env.attached_command is None
    # the settling prompt must be a no-op the model never has to reason about
    assert env.deployment.runtime.sent[-1] == ":"


def test_a_stale_prompt_is_not_mistaken_for_output():
    # the sandbox appends the prompt it matched, so a read that only re-consumed a stale
    # prompt comes back carrying that prompt. Judging emptiness on the whole output made
    # the follow-up read stop firing, and ipython went a turn behind again.
    env = _env()
    env.attached_command = "ipython3"
    env.attached_at_prompt = True
    env.deployment.runtime.script = [
        _Obs("In [2]: ", r"In \[\d+\]: $"),  # stale: the prompt and nothing else
        _Obs("Out[2]: 49\n\nIn [3]: ", r"In \[\d+\]: $"),
    ]
    out = _run(AgentEnv.send_input.__wrapped__(env, "7*7", action_timeout=5))
    assert "Out[2]: 49" in out


def test_a_read_that_only_reconsumed_the_prompt_counts_as_empty():
    # the timeout showed "banner\n>>> " (its prompt was never matched); the next read
    # matches that prompt, so its output is SHORTER. Treating it as new content showed
    # the banner twice and lagged every REPL answer by one turn.
    env = _env()
    env.attached_command = "python"
    env.attached_shown = "banner\n>>> "
    env.deployment.runtime.script = [_Obs("banner\n", ">>> $"), _Obs("42", ">>> $")]
    out = _run(AgentEnv.send_input.__wrapped__(env, "print(6*7)", action_timeout=5))
    assert "42" in out
    assert "banner" not in out


def test_unrelated_output_is_not_mistaken_for_a_prefix():
    # if the buffer does not start with what we showed, show all of it rather than slice
    env = _env()
    env.attached_command = "slow.sh"
    env.attached_shown = "STALE\n"
    env.deployment.runtime.script = [_Obs("TOTALLY DIFFERENT", "PS1")]
    out = _run(AgentEnv.send_input.__wrapped__(env, "", action_timeout=5))
    assert "TOTALLY DIFFERENT" in out


def test_input_is_not_typed_into_the_shell_when_the_command_already_finished():
    # apt finished while the model was deciding: sending "y" now would run it as a bash
    # command, and "C-d" would close the session outright
    env = _env()
    env.attached_command = "apt install foo"
    env.deployment.runtime.finished = _Obs("...done.\nSetting up foo", "PS1")
    out = _run(AgentEnv.send_input.__wrapped__(env, "y", action_timeout=5))
    assert "Setting up foo" in out
    assert "had already finished" in out
    assert "NOT sent" in out
    assert env.attached_command is None


@pytest.mark.parametrize("key", ["C-c", "C-d"])
def test_control_keys_are_also_withheld_once_the_command_finished(key):
    env = _env()
    env.attached_command = "apt install foo"
    env.deployment.runtime.finished = _Obs("", "PS1")
    out = _run(AgentEnv.send_input.__wrapped__(env, key, action_timeout=5))
    assert "had already finished" in out
    assert env.attached_command is None


def test_no_peek_while_the_program_sits_at_its_own_prompt():
    # a REPL parked at ">>> " cannot have exited on its own, so the peek is skipped and
    # a long debugging session pays nothing for this check
    env = _env()
    env.attached_command = "python"
    env.attached_at_prompt = True
    env.deployment.runtime.finished = _Obs("SHOULD-NOT-BE-REACHED", "PS1")
    env.deployment.runtime.script = [_Obs("42", ">>> $")]
    out = _run(AgentEnv.send_input.__wrapped__(env, "print(6*7)", action_timeout=5))
    assert "42" in out
    assert env.attached_command == "python"


def test_a_yield_clears_the_at_prompt_flag():
    # after a timeout the program is working, not parked, so the next send must peek
    env = _env()
    env.attached_command = "python"
    env.attached_at_prompt = True
    env.deployment.runtime.script = [_timeout("working...")]
    with pytest.raises(ActionTimeoutError):
        _run(AgentEnv.send_input.__wrapped__(env, "compute()", action_timeout=5))
    assert env.attached_at_prompt is False


@pytest.mark.parametrize("key", ["C-c", "C-d"])
def test_control_keys_do_not_reprint_what_was_already_shown(key):
    # sends that time out do not consume the terminal buffer, so it accumulates. C-c/C-d
    # then match the shell prompt and used to dump the whole thing back, replaying every
    # line the model had already been shown.
    env = _env()
    env.attached_command = "cat"
    env.attached_shown = "teste\nteste1\n"
    env.deployment.runtime.script = [_Obs("teste\nteste1\nEOF", "PS1")]
    out = _run(AgentEnv.send_input.__wrapped__(env, key, action_timeout=5))
    assert "EOF" in out
    assert "teste1" not in out


def test_a_confirmation_the_eof_already_answered_is_not_left_looking_open():
    # sendline("\x04") is EOF plus a newline, so ipython --simple-prompt asks "really
    # exit?" and then takes that newline as the default and goes. Showing the question
    # next to "the session is free" read as a contradiction.
    env = _env()
    env.attached_command = "ipython3"
    env.attached_at_prompt = True
    env.deployment.runtime.script = [_Obs("Do you really want to exit ([y]/n)? ", "PS1")]
    out = _run(AgentEnv.send_input.__wrapped__(env, "C-d", action_timeout=5))
    assert env.attached_command is None
    assert "already been answered" in out
    assert "session is free" in out


def test_eof_answered_by_a_confirmation_shows_the_question():
    # ipython answers C-d with "Do you really want to exit ([y]/n)?" rather than exiting;
    # waiting for a shell prompt that never comes burned the whole timeout and reported
    # "Failed to send EOF", hiding the question for another turn
    env = _env()
    env.attached_command = "ipython3"
    env.attached_at_prompt = True
    env.deployment.runtime.script = [
        _timeout(),  # the C-d itself never reaches a shell prompt
        _Obs("Do you really want to exit ([y]/n)? ", r"\? $"),
    ]
    out = _run(AgentEnv.send_input.__wrapped__(env, "C-d", action_timeout=8))
    assert "Do you really want to exit" in out
    assert "Failed" not in out
    assert env.attached_command == "ipython3"  # still there, still answerable


def test_the_prompt_the_sandbox_returns_reaches_the_model():
    # the sandbox appends the literal prompt it matched, so "[Y/n]" (not "[y/N]", and not
    # the r"\[Y/n\]" pattern) survives all the way through the dedup into the observation
    env = _env()
    env.attached_command = "apt install foo"
    env.attached_at_prompt = True
    env.deployment.runtime.script = [_Obs("Do you want to continue? [Y/n] ", r"\[Y/n\]")]
    out = _run(AgentEnv.send_input.__wrapped__(env, "", action_timeout=5))
    assert "Do you want to continue? [Y/n]" in out


def test_a_requested_send_timeout_is_honoured():
    # a hidden 8s cap used to override the model here, so asking to wait 20s for a slow
    # REPL statement silently got 8. The ceiling in interaction.py is the only bound.
    seen = {}

    class _Rt:
        async def run_in_session(self, action):
            if _is_peek(action):
                raise CommandTimeoutError("nothing pending")
            seen["timeout"] = action.timeout
            return _Obs("ok", "PS1")

    env = _env()
    env.attached_command = "cat"
    env.deployment.runtime = _Rt()
    _run(AgentEnv.send_input.__wrapped__(env, "hello", action_timeout=20))
    assert seen["timeout"] == 20


def test_format_args_example_follows_the_parser():
    from uni_agent.interaction.tools_manager import ToolsManager, ToolsManagerConfig

    tm = ToolsManager.__new__(ToolsManager)
    tm.tools_manager_config = ToolsManagerConfig.model_construct(tools=[], parser="hermes")
    assert tm.format_args_example({"command": "C-c", "is_input": True}) == '{"command": "C-c", "is_input": true}'
    tm.tools_manager_config = ToolsManagerConfig.model_construct(tools=[], parser="qwen3_coder")
    assert (
        tm.format_args_example({"command": "C-c", "is_input": True})
        == "<parameter=command>C-c</parameter> <parameter=is_input>true</parameter>"
    )


def test_attached_refusal_quotes_a_literal_cancel_call_and_clips_echoes():
    import types as _t

    from uni_agent.interaction.interaction import AgentInteraction

    class _Env:
        attached_command = "find /testbed -type f -name '*.py' " + "x" * 200
        attached_seconds = 0.0

    noop = lambda *a, **k: None  # noqa: E731
    it = AgentInteraction.__new__(AgentInteraction)
    it.env, it.logger = _Env(), _t.SimpleNamespace(info=noop, error=noop, debug=noop)
    it.action_timeout, it.yield_timeout = 30, 5
    it.attached_kill_timeout, it.timeout_budget = 45.0, 1
    long_cmd = "grep -rn pattern " + "y" * 200
    it.tools_manager = _t.SimpleNamespace(
        get_tool_action=lambda _tc: _t.SimpleNamespace(command=long_cmd, is_input=False, timeout=None),
        format_args_example=lambda args: '{"command": "C-c", "is_input": true}',
    )
    tc = _t.SimpleNamespace(id="c", function=_t.SimpleNamespace(name="execute_bash"))
    result = _run(AgentInteraction._execute_tool_call(it, tc))
    assert result.status == "syntax_error"
    assert '{"command": "C-c", "is_input": true}' in result.observation
    assert "not part of the command text" in result.observation
    assert "y" * 200 not in result.observation  # command echo is clipped
    assert "x" * 200 not in result.observation  # attached echo is clipped


def test_input_without_attached_quotes_a_literal_rerun_call():
    import types as _t

    from uni_agent.interaction.interaction import AgentInteraction

    class _Env:
        attached_command = None
        attached_seconds = 0.0

    noop = lambda *a, **k: None  # noqa: E731
    it = AgentInteraction.__new__(AgentInteraction)
    it.env, it.logger = _Env(), _t.SimpleNamespace(info=noop, error=noop, debug=noop)
    it.action_timeout, it.yield_timeout = 30, 5
    it.attached_kill_timeout, it.timeout_budget = 45.0, 1
    it.tools_manager = _t.SimpleNamespace(
        get_tool_action=lambda _tc: _t.SimpleNamespace(command="ls", is_input=True, timeout=None),
        format_args_example=lambda args: f"EXAMPLE({args['command']}, {args['is_input']})",
    )
    tc = _t.SimpleNamespace(id="c", function=_t.SimpleNamespace(name="execute_bash"))
    result = _run(AgentInteraction._execute_tool_call(it, tc))
    assert result.status == "syntax_error"
    assert "EXAMPLE(ls, False)" in result.observation


def test_yield_note_quotes_a_literal_cancel_call():
    import types as _t

    from uni_agent.interaction.interaction import AgentInteraction

    class _Env:
        attached_command, attached_at_prompt = "cat", False
        attached_seconds = 2.0

        async def send_input(self, command, action_timeout):
            raise ActionTimeoutError("partial output")

    noop = lambda *a, **k: None  # noqa: E731
    it = AgentInteraction.__new__(AgentInteraction)
    it.env, it.logger = _Env(), _t.SimpleNamespace(info=noop, error=noop, debug=noop)
    it.action_timeout, it.yield_timeout = 30, 5
    it.attached_kill_timeout, it.timeout_budget = 45.0, 1
    it.tools_manager = _t.SimpleNamespace(
        get_tool_action=lambda _tc: _t.SimpleNamespace(command="", is_input=True, timeout=1),
        format_args_example=lambda args: '{"command": "C-c", "is_input": true}',
    )
    tc = _t.SimpleNamespace(id="c", function=_t.SimpleNamespace(name="execute_bash"))
    result = _run(AgentInteraction._execute_tool_call(it, tc))
    assert result.status == "yielded"
    assert '{"command": "C-c", "is_input": true}' in result.observation
