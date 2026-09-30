"""A per-turn cap is counted; a generation that overflows the window is not, since it is thrown away
and regenerated in the condensed segment (where it is counted if it caps again)."""

import asyncio
import types

from uni_agent.interaction.model import AgentChatModel, MaxTokenExceededError


def _query(max_model_len: int, max_completion_tokens: int, prompt_tokens: int):
    """One real ``AgentChatModel.query`` against an engine that fills whatever it is allowed;
    returns the rollout cache and the error it raised, if any."""

    async def generate(request_id, prompt_ids, sampling_params):
        return types.SimpleNamespace(
            token_ids=[7] * sampling_params["max_tokens"], log_probs=None, num_preempted=0,
            routed_experts=None, extra_fields={},
        )

    async def run():
        model = types.SimpleNamespace(
            max_model_len=max_model_len, max_completion_tokens=max_completion_tokens, sampling_params={},
            client=types.SimpleNamespace(generate=generate),
            tokenizer=types.SimpleNamespace(decode=lambda ids: "x" * len(ids)),
            loop=asyncio.get_running_loop(),
        )
        cache = {"request_id": "r", "prompt_ids": [1] * prompt_tokens, "metrics": {},
                 "response_mask": [], "response_logprobs": [], "extra_fields": {}}
        try:
            await AgentChatModel.query(model, [{"role": "user", "content": "t"}], cache)
        except MaxTokenExceededError as e:
            return cache, e
        return cache, None

    return asyncio.run(run())


def test_a_turn_cut_by_its_own_cap_is_counted():
    cache, error = _query(max_model_len=1000, max_completion_tokens=16, prompt_tokens=60)
    assert error is None
    assert cache["metrics"]["capped_turns"] == 1
    assert len(cache["response_mask"]) == 16


def test_an_overflow_that_is_regenerated_is_not_counted():
    # the window leaves 40 tokens and the engine fills them: capped, and past the budget
    cache, error = _query(max_model_len=100, max_completion_tokens=4096, prompt_tokens=60)
    assert isinstance(error, MaxTokenExceededError)
    assert cache["metrics"].get("capped_turns", 0) == 0
    assert cache["response_mask"] == [], "the overflowing turn is not appended"
