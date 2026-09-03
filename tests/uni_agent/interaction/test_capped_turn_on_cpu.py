"""A turn cut by the per-turn cap gets the cut-off notice and executes nothing, even when
what survived the cut parses cleanly."""

from __future__ import annotations

import asyncio

from uni_agent.interaction.interaction import AgentInteraction, ToolResult, get_logger
from uni_agent.interaction.tool_parser import XMLToolParser
from uni_agent.interaction.tool_schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
)
from uni_agent.interaction.tools_manager import ToolsManager

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

COMPLETE = (
    "Running it.\n\n<tool_call>\n<function=execute_bash>\n"
    "<parameter=command>\nls /testbed\n</parameter>\n</function>\n</tool_call>"
)


class _Model:
    max_completion_tokens = 4096

    def __init__(self, output: str, capped: bool):
        self.output, self.capped = output, capped

    async def query(self, messages, rollout_cache):
        rollout_cache["response_mask"] = rollout_cache.get("response_mask", []) + [1] * 5
        info = {"prompt_tokens": 1, "completion_tokens": 5, "capped": self.capped}
        return self.output, None, rollout_cache, info

    async def append_messages_to_rollout_cache(self, messages, rollout_cache):
        return rollout_cache


def _interaction(output: str, capped: bool, executed: list) -> AgentInteraction:
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
    it.model = _Model(output, capped)
    tm = ToolsManager.__new__(ToolsManager)
    tm._tool_parser = XMLToolParser()
    tm.tools_schemas = [TOOL.model_dump()]
    it.tools_manager = tm

    async def _execute(tool_call):
        executed.append(tool_call.function.arguments)
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.function.name,
            action="",
            observation="ok",
            status="ok",
            execution_time=0.0,
        )

    it._execute_tool_call = _execute
    return it


def test_uncapped_complete_call_executes():
    executed: list = []
    it = _interaction(COMPLETE, capped=False, executed=executed)
    out = asyncio.run(it.step(1))
    assert out.exit_reason == "completed"
    assert executed == [{"command": "ls /testbed"}]


def test_capped_turn_executes_nothing_and_gets_the_notice():
    executed: list = []
    it = _interaction(COMPLETE, capped=True, executed=executed)
    out = asyncio.run(it.step(1))
    assert out.exit_reason == "format_error"
    assert executed == []
    assert "cut off at the per-turn output limit of 4096 tokens" in it.messages[-1]["content"]


def test_capped_turn_cut_inside_a_call_gets_the_notice():
    executed: list = []
    cut = COMPLETE[: COMPLETE.index("/testbed")]
    it = _interaction(cut, capped=True, executed=executed)
    out = asyncio.run(it.step(1))
    assert out.exit_reason == "format_error"
    assert executed == []
    assert "cut off at the per-turn output limit" in it.messages[-1]["content"]
