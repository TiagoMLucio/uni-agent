"""The token stream built turn by turn decodes to the full chat-template render of the same
messages: after a tool turn, a multi-tool turn, a format-error turn and a turn cut at the
per-turn cap. Without the latter, the next turn continues a half-open assistant message."""

from __future__ import annotations

import asyncio
import types

import pytest

from uni_agent.interaction.interaction import AgentInteraction, ToolResult, get_logger
from uni_agent.interaction.model import AgentChatModel
from uni_agent.interaction.tool_parser import XMLToolParser
from uni_agent.interaction.tool_schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
)
from uni_agent.interaction.tools_manager import ToolsManager

MODEL = "Qwen/Qwen3.5-4B"
TEMPLATE_KWARGS = {"enable_thinking": False}
CAP = 64

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


def _call(command: str) -> str:
    return (
        f"<tool_call>\n<function=execute_bash>\n<parameter=command>\n{command}\n"
        "</parameter>\n</function>\n</tool_call>"
    )


TOOL_TURN = "Listing it.\n\n" + _call("ls /testbed")
MULTI_TOOL_TURN = "Two checks.\n\n" + _call("pwd") + "\n" + _call("git status")
FORMAT_ERROR_TURN = "I think the bug is in the parser."
CAPPED_TURN = "Writing the file.\n\n" + _call("cat > a.py <<'EOF'\n" + "print(1)\n" * 200 + "EOF")


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("verl.utils.chat_template")
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL)
    except Exception as e:
        pytest.skip(f"{MODEL} tokenizer unavailable offline: {e!r}")


def _rollout(tokenizer, turns: list[str]) -> AgentInteraction:
    """Run one step per scripted reply through the real model; a reply closes with eos
    unless the cap cuts it first."""
    replies = iter(turns)

    async def generate(request_id, prompt_ids, sampling_params):
        ids = tokenizer.encode(next(replies), add_special_tokens=False) + [tokenizer.eos_token_id]
        ids = ids[: sampling_params["max_tokens"]]
        return types.SimpleNamespace(
            token_ids=ids, log_probs=[-0.5] * len(ids), num_preempted=0, routed_experts=None, extra_fields={}
        )

    async def run():
        model = AgentChatModel(
            client=types.SimpleNamespace(generate=generate),
            tokenizer=tokenizer,
            max_model_len=32768,
            sampling_params={},
            max_completion_tokens=CAP,
            chat_template_kwargs=TEMPLATE_KWARGS,
        )
        it = AgentInteraction.__new__(AgentInteraction)
        it.logger = get_logger("interaction", "test")
        it.messages = [
            {"role": "system", "content": "You are a software engineer."},
            {"role": "user", "content": "fix it"},
        ]
        it.condense_max_retries = 0
        it.condenser = None
        it.chat_mode = False
        it.observation_role = "tool"
        it.timeout_budget = 60.0
        it.trajectory = []
        it.model = model
        tm = ToolsManager.__new__(ToolsManager)
        tm._tool_parser = XMLToolParser()
        tm.tools_schemas = [TOOL.model_dump()]
        it.tools_manager = tm

        async def _execute(tool_call):
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.function.name,
                action="",
                observation=f"ran {tool_call.function.arguments['command']}",
                status="ok",
                execution_time=0.0,
            )

        it._execute_tool_call = _execute
        it.rollout_cache = await model.prepare_rollout_cache(it.messages, include_tools=False)
        for step_idx in range(1, len(turns) + 1):
            it.trajectory.append(await it.step(step_idx))
        return it

    return asyncio.run(run())


def _assert_stream_is_the_render(tokenizer, it: AgentInteraction):
    rendered = tokenizer.apply_chat_template(
        it.messages, add_generation_prompt=True, tokenize=False, **TEMPLATE_KWARGS
    )
    cache = it.rollout_cache
    assert tokenizer.decode(cache["prompt_ids"]) == rendered
    assert len(cache["response_logprobs"]) == len(cache["response_mask"])


@pytest.mark.parametrize(
    "turns",
    [[TOOL_TURN], [MULTI_TOOL_TURN], [FORMAT_ERROR_TURN], [TOOL_TURN, MULTI_TOOL_TURN, FORMAT_ERROR_TURN, TOOL_TURN]],
    ids=["tool", "multi_tool", "format_error", "mixed"],
)
def test_stream_matches_full_render(tokenizer, turns):
    _assert_stream_is_the_render(tokenizer, _rollout(tokenizer, turns))


def test_capped_turn_is_closed_with_an_unmasked_eos(tokenizer):
    it = _rollout(tokenizer, [TOOL_TURN, CAPPED_TURN, TOOL_TURN])
    assert [s.exit_reason for s in it.trajectory] == ["completed", "format_error", "completed"]
    _assert_stream_is_the_render(tokenizer, it)

    cache = it.rollout_cache
    prompt_len = len(cache["prompt_ids"]) - len(cache["response_mask"])
    _, start, end = cache["turn_spans"][1]
    assert end - start == CAP
    closing = prompt_len + end
    assert cache["prompt_ids"][closing] == tokenizer.eos_token_id
    assert cache["response_mask"][end] == 0
    assert cache["response_logprobs"][end] == 0.0
    assert cache["metrics"]["capped_turns"] == 1


def test_naturally_stopped_turns_get_no_extra_eos(tokenizer):
    it = _rollout(tokenizer, [TOOL_TURN, TOOL_TURN])
    cache = it.rollout_cache
    eos = tokenizer.eos_token_id
    prompt_len = len(cache["prompt_ids"]) - len(cache["response_mask"])
    for _, _, end in cache["turn_spans"]:
        assert cache["prompt_ids"][prompt_len + end - 1] == eos
        assert cache["prompt_ids"][prompt_len + end] != eos
