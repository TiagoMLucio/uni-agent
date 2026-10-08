"""Prefix-cache hits per trajectory: the engine's cached prompt tokens next to every prompt token sent.

A multi-turn rollout resends its whole history each turn, so the hit rate says whether sticky
routing kept the prefix where it was cached. One turn the engine did not report makes the sum
unknown, which is -1 and not a smaller number, and an engine older than the field reads the same.
"""

import asyncio
import functools
import types

from uni_agent.agent_loop import UniAgentLoop
from uni_agent.interaction.model import AgentChatModel


def _queries(prompt_lengths, cached_per_turn):
    """Real ``AgentChatModel.query`` calls on one rollout cache; ``"absent"`` drops the field."""
    turns = iter(cached_per_turn)

    async def generate(request_id, prompt_ids, sampling_params):
        out = types.SimpleNamespace(token_ids=[7], log_probs=None, num_preempted=0, routed_experts=None,
                                    extra_fields={})
        cached = next(turns)
        if cached != "absent":
            out.num_cached_tokens = cached
        return out

    async def run():
        model = types.SimpleNamespace(
            max_model_len=10_000, max_completion_tokens=None, sampling_params={},
            client=types.SimpleNamespace(generate=generate),
            tokenizer=types.SimpleNamespace(decode=lambda ids: "x" * len(ids)),
            loop=asyncio.get_running_loop(),
        )
        cache = {"request_id": "r", "prompt_ids": [], "metrics": {},
                 "response_mask": [], "response_logprobs": [], "extra_fields": {}}
        for length in prompt_lengths:
            # the rollout's history grows to this length before the next call
            cache["prompt_ids"] += [1] * (length - len(cache["prompt_ids"]))
            cache["response_mask"] = [0] * len(cache["prompt_ids"])
            await AgentChatModel.query(model, [{"role": "user", "content": "t"}], cache)
        return cache["metrics"]

    return asyncio.run(run())


def test_hits_and_prompt_tokens_are_summed_over_the_turns():
    metrics = _queries([100, 150, 230], [0, 96, 144])
    assert metrics["num_cached_tokens"] == 240
    assert metrics["num_prompt_tokens"] == 480


def test_one_unreported_turn_makes_the_sum_unknown():
    assert _queries([100, 150, 230], [0, -1, 144])["num_cached_tokens"] == -1
    assert _queries([100, 150], [None, 96])["num_cached_tokens"] == -1


def test_an_engine_without_the_field_reads_as_unreported():
    metrics = _queries([100, 150], ["absent", "absent"])
    assert metrics["num_cached_tokens"] == -1
    assert metrics["num_prompt_tokens"] == 250


def _row(metrics):
    loop = types.SimpleNamespace(
        logger=types.SimpleNamespace(info=lambda _m: None, warning=lambda _m: None),
        config=types.SimpleNamespace(actor_rollout_ref=types.SimpleNamespace(
            rollout=types.SimpleNamespace(prompt_length=64, response_length=64))),
        opening_messages=[], mask_abnormal_exit_traj=False, emit_feedback=False,
        chat_model=types.SimpleNamespace(max_model_len=10**9),
    )
    loop._segment_to_output = functools.partial(UniAgentLoop._segment_to_output, loop)
    cache = {"prompt_ids": [1, 2, 3], "response_mask": [1], "response_logprobs": [], "turn_spans": [[1, 2, 3]]}
    result = {"trajectory": [types.SimpleNamespace(step_idx=1, tool_results=[], exit_reason="finished")],
              "segments": [{"rollout_cache": cache, "prompt_messages": None}],
              "rollout_cache": cache, "metrics": metrics}
    (row,) = asyncio.run(UniAgentLoop.convert_to_agent_output(loop, result))
    return row


def test_the_row_ships_the_counts_as_their_own_fields():
    row = _row({"num_cached_tokens": 240, "num_prompt_tokens": 480, "generate_sequences": 1.5})
    assert row.extra_fields["num_cached_tokens"] == 240
    assert row.extra_fields["num_prompt_tokens"] == 480
    assert row.extra_fields["timings"] == {"generate_sequences": 1.5}, "counts, not timings"


def test_a_rollout_that_never_generated_reports_nothing_cached():
    row = _row({})
    assert row.extra_fields["num_cached_tokens"] == -1
    assert row.extra_fields["num_prompt_tokens"] == 0
