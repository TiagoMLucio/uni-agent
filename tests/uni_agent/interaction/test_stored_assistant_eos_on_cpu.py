"""The assistant content stored in ``messages`` carries no eos, so a re-render (the
condenser's reseat) does not double it in front of every past turn."""

from __future__ import annotations

import asyncio

import pytest

from uni_agent.interaction.interaction import AgentInteraction, ToolResult, get_logger
from uni_agent.interaction.tool_parser import XMLToolParser
from uni_agent.interaction.tool_schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
)
from uni_agent.interaction.tools_manager import ToolsManager

MODEL = "Qwen/Qwen3.5-4B"

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
    "Running it.\n\n<tool_call>\n<function=execute_bash>\n"
    "<parameter=command>\nls /testbed\n</parameter>\n</function>\n</tool_call>"
)


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL)
    except Exception as e:
        pytest.skip(f"{MODEL} tokenizer unavailable offline: {e!r}")


class _Model:
    max_completion_tokens = 4096

    def __init__(self, tokenizer, output: str):
        self.tokenizer, self.output = tokenizer, output

    async def query(self, messages, rollout_cache):
        rollout_cache["response_mask"] = rollout_cache.get("response_mask", []) + [1] * 5
        return self.output, None, rollout_cache, {"prompt_tokens": 1, "completion_tokens": 5}

    async def append_messages_to_rollout_cache(self, messages, rollout_cache):
        return rollout_cache


def _interaction(tokenizer, output: str) -> AgentInteraction:
    it = AgentInteraction.__new__(AgentInteraction)
    it.logger = get_logger("interaction", "test")
    it.messages = [
        {"role": "system", "content": "You are a software engineer."},
        {"role": "user", "content": "fix it"},
    ]
    it.rollout_cache = {"metrics": {}, "response_mask": [], "prompt_ids": []}
    it.condense_max_retries = 0
    it.condenser = None
    it.chat_mode = False
    it.observation_role = "tool"
    it.timeout_budget = 60.0
    it.trajectory = []
    it.model = _Model(tokenizer, output)
    tm = ToolsManager.__new__(ToolsManager)
    tm._tool_parser = XMLToolParser()
    tm.tools_schemas = [TOOL.model_dump()]
    it.tools_manager = tm

    async def _execute(tool_call):
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


def _decoded_response(tokenizer, text: str) -> str:
    """What the sampler's token ids decode to: the reply plus the eos that closed it."""
    ids = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
    return tokenizer.decode(ids)


def test_reseated_prefix_has_one_eos_per_assistant_turn(tokenizer):
    eos = tokenizer.eos_token
    response = _decoded_response(tokenizer, CALL)
    assert response.endswith(eos)

    it = _interaction(tokenizer, response)
    step = asyncio.run(it.step(1))
    asyncio.run(it.step(2))

    assert step.response == response, "the raw model output must stay untouched"

    rendered = tokenizer.apply_chat_template(
        it.messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    assert eos + eos not in rendered
    turns = rendered.split("<|im_start|>assistant\n")[1:]
    closed = [t for t in turns if eos in t]
    assert len(closed) == 2, "both stored assistant turns must be closed"
    for turn in closed:
        assert turn.split(eos)[0].endswith("</tool_call>")


def test_stored_content_drops_only_the_trailing_eos(tokenizer):
    eos = tokenizer.eos_token
    it = _interaction(tokenizer, _decoded_response(tokenizer, CALL))
    asyncio.run(it.step(1))

    stored = next(m for m in it.messages if m["role"] == "assistant")["content"]
    assert stored == CALL
    assert not stored.endswith(eos)
