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


class _Runtime:
    def __init__(self):
        self.script: list = []

    async def run_in_session(self, action):
        if self.script:
            nxt = self.script.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt
        return _Obs("ok", "PS1")


def _env():
    e = AgentEnv.__new__(AgentEnv)  # __init__ needs a real deployment
    e.attached_command = None
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
    assert "still running" in str(excinfo.value)
    assert "is_input" in str(excinfo.value)


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
    _run(AgentEnv.send_input.__wrapped__(env, "exit()", action_timeout=90))
    assert env.attached_command is None


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


def test_no_follow_up_read_after_a_confirmation_prompt():
    # a bare newline would be read as the default answer, so never re-read these
    env = _env()
    env.attached_command = "apt install foo"
    env.deployment.runtime.script = [_Obs("", r"\[Y/n\]"), _Obs("SHOULD-NOT-BE-READ", "PS1")]
    out = _run(AgentEnv.send_input.__wrapped__(env, "n", action_timeout=15))
    assert "SHOULD-NOT-BE-READ" not in out
    assert env.attached_command == "apt install foo"


def test_no_follow_up_read_at_a_continuation_prompt():
    # a newline at "..." ends the block: re-reading turned `if True:` into a SyntaxError
    env = _env()
    env.attached_command = "python"
    env.deployment.runtime.script = [_Obs("", r"\.\.\. $"), _Obs("SHOULD-NOT-BE-READ", ">>> $")]
    out = _run(AgentEnv.send_input.__wrapped__(env, "if True:", action_timeout=15))
    assert "SHOULD-NOT-BE-READ" not in out


def test_no_follow_up_read_at_a_pdb_prompt():
    # a newline at "(Pdb)" repeats the previous command
    env = _env()
    env.attached_command = "python"
    env.deployment.runtime.script = [_Obs("", r"\(Pdb\) $"), _Obs("SHOULD-NOT-BE-READ", ">>> $")]
    out = _run(AgentEnv.send_input.__wrapped__(env, "n", action_timeout=15))
    assert "SHOULD-NOT-BE-READ" not in out


def test_interactive_send_timeout_is_capped():
    # a program with no prompt can never match, so an uncapped send burns action_timeout
    from uni_agent.interaction.env import INTERACTIVE_SEND_TIMEOUT

    seen = {}

    class _Rt:
        async def run_in_session(self, action):
            seen["timeout"] = action.timeout
            return _Obs("ok", "PS1")

    env = _env()
    env.attached_command = "cat"
    env.deployment.runtime = _Rt()
    _run(AgentEnv.send_input.__wrapped__(env, "hello", action_timeout=900))
    assert seen["timeout"] == INTERACTIVE_SEND_TIMEOUT
