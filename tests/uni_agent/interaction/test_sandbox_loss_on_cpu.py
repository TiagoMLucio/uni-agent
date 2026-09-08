"""A sandbox that stops answering ends the rollout under its own reason.

``TerminalNotAliveError`` lost its last producer when ``run_action`` was rewritten, so
``terminal_dead`` could not fire at all and a lost environment was reported as
``unknown_error``, the reason reserved for what only a harness bug can produce.
"""

from __future__ import annotations

import asyncio
import types

import aiohttp
import pytest
from swerex.exceptions import (
    BashIncorrectSyntaxError,
    CommandTimeoutError,
    DeploymentNotStartedError,
    SessionDoesNotExistError,
    SessionNotInitializedError,
)

from uni_agent.interaction.env import (
    ActionIncorrectSyntaxError,
    ActionTimeoutError,
    AgentEnv,
    TerminalNotAliveError,
)
from uni_agent.interaction.interaction import AgentInteraction, ToolResult, get_logger
from uni_agent.interaction.tool_parser import XMLToolParser
from uni_agent.interaction.tool_schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
)
from uni_agent.interaction.tools_manager import ToolsManager

GONE = [
    SessionNotInitializedError("shell not initialized"),
    SessionDoesNotExistError("session 'default' does not exist"),
    DeploymentNotStartedError(),
    aiohttp.ClientConnectionError("connection refused"),
    ConnectionError("connection reset"),
    TimeoutError("the sandbox never answered"),
]


class _Runtime:
    def __init__(self, exc):
        self.exc = exc

    async def run_in_session(self, action):
        raise self.exc


def _env(exc):
    e = AgentEnv.__new__(AgentEnv)  # __init__ needs a real deployment
    e.attached_command = None
    e.attached_shown = ""
    e.attached_seconds = 0.0
    e.attached_at_prompt = True
    noop = lambda *a, **k: None  # noqa: E731
    e.logger = types.SimpleNamespace(info=noop, error=noop, critical=noop, debug=noop, warning=noop)
    e.deployment = types.SimpleNamespace(runtime=_Runtime(exc))
    return e


def _run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("exc", GONE, ids=lambda e: type(e).__name__)
def test_a_command_against_a_lost_sandbox_is_not_a_harness_bug(exc):
    env = _env(exc)
    with pytest.raises(TerminalNotAliveError) as excinfo:
        _run(AgentEnv.run_action.__wrapped__(env, "ls", action_timeout=30))
    assert type(exc).__name__ in str(excinfo.value), "the underlying failure is the only record of which it was"


@pytest.mark.parametrize("exc", GONE, ids=lambda e: type(e).__name__)
def test_input_against_a_lost_sandbox_is_not_a_harness_bug(exc):
    env = _env(exc)
    env.attached_command = "python"
    with pytest.raises(TerminalNotAliveError):
        _run(AgentEnv.send_input.__wrapped__(env, "print(1)", action_timeout=30))


def test_a_command_timeout_is_still_a_yield():
    # CommandTimeoutError is itself a TimeoutError, so only the catch order keeps a slow
    # command from being read as a dead sandbox
    env = _env(CommandTimeoutError("timeout"))
    with pytest.raises(ActionTimeoutError):
        _run(AgentEnv.run_action.__wrapped__(env, "sleep 500", action_timeout=30))
    assert env.attached_command == "sleep 500"


def test_a_syntax_error_is_still_the_agents_own():
    exc = BashIncorrectSyntaxError("bad")
    exc.extra_info = {"bash_stdout": "", "bash_stderr": "unexpected EOF"}
    with pytest.raises(ActionIncorrectSyntaxError):
        _run(AgentEnv.run_action.__wrapped__(_env(exc), "for", action_timeout=30))


TOOL = OpenAIFunctionToolSchema(
    type="function",
    function=OpenAIFunctionSchema(
        name="execute_bash",
        description="bash",
        parameters=OpenAIFunctionParametersSchema(
            type="object",
            properties={"command": OpenAIFunctionPropertySchema(type="string")},
            required=["command"],
        ),
    ),
)

CALL = (
    "Looking.\n\n<tool_call>\n<function=execute_bash>\n"
    "<parameter=command>\nls /testbed\n</parameter>\n</function>\n</tool_call>"
)


class _Model:
    max_completion_tokens = 4096

    async def query(self, messages, rollout_cache):
        rollout_cache["response_mask"] = rollout_cache.get("response_mask", []) + [1] * 5
        return CALL, None, rollout_cache, {"prompt_tokens": 1, "completion_tokens": 5, "capped": False}

    async def append_messages_to_rollout_cache(self, messages, rollout_cache):
        return rollout_cache


def _interaction(status: str) -> AgentInteraction:
    it = AgentInteraction.__new__(AgentInteraction)
    it.logger = get_logger("interaction", "test")
    it.messages = [{"role": "user", "content": "fix it"}]
    it.rollout_cache = {"metrics": {}, "response_mask": [], "prompt_ids": []}
    it.condense_max_retries = 0
    it.condenser = None
    it.chat_mode = False
    it.observation_role = "tool"
    it.timeout_budget = 60.0
    it.trajectory = []
    it.model = _Model()
    tm = ToolsManager.__new__(ToolsManager)
    tm._tool_parser = XMLToolParser()
    tm.tools_schemas = [TOOL.model_dump()]
    it.tools_manager = tm

    async def _execute(tool_call):
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.function.name,
            action="",
            observation="the sandbox stopped answering",
            status=status,
            execution_time=0.0,
        )

    it._execute_tool_call = _execute
    return it


def test_a_lost_sandbox_ends_the_turn_as_terminal_dead():
    out = asyncio.run(_interaction("skipped").step(1))
    assert out.exit_reason == "terminal_dead"
    assert out.done


def test_a_healthy_call_is_untouched():
    out = asyncio.run(_interaction("ok").step(1))
    assert (out.exit_reason, out.done) == ("completed", False)
