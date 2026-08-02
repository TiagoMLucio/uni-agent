"""privileged_setup_cmd runs after post_setup_cmd and its stdout stays out of the agent's reach."""

from __future__ import annotations

import asyncio
import types

from uni_agent.interaction.env import AgentEnv


class _Obs:
    def __init__(self, output: str = ""):
        self.output = output
        self.exit_code = 0


class _Runtime:
    def __init__(self, outputs: dict[str, str]):
        self.outputs = outputs
        self.sent: list[str] = []

    async def run_in_session(self, action):
        self.sent.append(action.command)
        return _Obs(self.outputs.get(action.command, ""))


def _env(post: str | None, privileged: str | None, outputs: dict[str, str]) -> AgentEnv:
    e = AgentEnv.__new__(AgentEnv)  # __init__ needs a real deployment
    e.env_variables = None
    e.post_setup_cmd = post
    e.privileged_setup_cmd = privileged
    e.privileged_context = ""
    noop = lambda *a, **k: None  # noqa: E731
    e.logger = types.SimpleNamespace(info=noop, error=noop, critical=noop, debug=noop)
    runtime = _Runtime(outputs)
    e.deployment = types.SimpleNamespace(
        runtime=runtime, start=lambda **kw: asyncio.sleep(0)
    )
    return e


def test_captures_stdout_after_post_setup():
    e = _env("checkout", "capture", {"capture": "  diff --git a/x b/x\n"})
    e.start()
    assert e.deployment.runtime.sent == ["checkout", "capture"]
    assert e.privileged_context == "diff --git a/x b/x"


def test_absent_when_unset():
    e = _env("checkout", None, {})
    e.start()
    assert e.deployment.runtime.sent == ["checkout"]
    assert e.privileged_context == ""
