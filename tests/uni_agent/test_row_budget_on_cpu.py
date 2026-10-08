"""A stored row ends at the rollout's context budget.

Every model call reads and writes within ``max_model_len``, so the only tokens past it are the ones the last
observation added after the segment's final call: masked out of the loss and after every scored token, yet a
227k-token row of a 131k-budget rollout ran the update out of memory.
"""

from __future__ import annotations

import asyncio
import functools
from types import SimpleNamespace

import pytest

pytest.importorskip("verl.experimental.agent_loop")

from uni_agent.agent_loop import UniAgentLoop  # noqa: E402


def _rows(segments, budget, warnings=None):
    loop = SimpleNamespace(
        logger=SimpleNamespace(info=lambda _m: None, warning=(warnings.append if warnings is not None else print)),
        config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(prompt_length=10**6, response_length=10**6))),
        opening_messages=[],
        mask_abnormal_exit_traj=False,
        emit_feedback=False,
        chat_model=SimpleNamespace(max_model_len=budget),
    )
    loop._segment_to_output = functools.partial(UniAgentLoop._segment_to_output, loop)
    trajectory = [SimpleNamespace(step_idx=i + 1, tool_results=[], exit_reason="finished") for i in range(len(segments))]
    result = {"trajectory": trajectory, "segments": segments, "rollout_cache": segments[-1]["rollout_cache"],
              "reward_score": 1.0}
    return asyncio.run(UniAgentLoop.convert_to_agent_output(loop, result))


def _segment(prompt_len, mask, spans):
    return {
        "rollout_cache": {
            "prompt_ids": list(range(prompt_len + len(mask))),
            "response_mask": list(mask),
            "response_logprobs": [-0.5 if m else 0.0 for m in mask],
            "turn_spans": spans,
        },
        "prompt_messages": None,
    }


def test_the_observation_past_the_budget_is_dropped():
    # prompt 4, a 3-token turn, a 2-token observation inside the budget of 10, then 6 observation tokens past it
    (row,) = _rows([_segment(4, [1, 1, 1, 0, 0] + [0] * 6, [[1, 0, 3]])], budget=10)
    assert len(row.prompt_ids) + len(row.response_ids) == 10
    assert row.response_mask == [1, 1, 1, 0, 0, 0]
    assert row.response_logprobs == [-0.5, -0.5, -0.5, 0.0, 0.0, 0.0]
    assert row.extra_fields["turn_spans"] == [[1, 0, 3]]
    assert row.reward_score == 1.0


def test_a_row_inside_the_budget_is_untouched():
    (row,) = _rows([_segment(4, [1, 1, 0, 0], [[1, 0, 2]])], budget=8)
    assert len(row.response_ids) == 4


def test_generated_tokens_past_the_budget_keep_the_row_whole():
    warnings = []
    (row,) = _rows([_segment(4, [1, 1, 0, 0, 1, 1], [[1, 0, 2], [2, 4, 6]])], budget=6, warnings=warnings)
    assert len(row.response_ids) == 6 and len(warnings) == 1


def test_a_validation_budget_keeps_a_row_training_would_cut():
    seg = _segment(4, [1, 1, 1] + [0] * 9, [[1, 0, 3]])
    (train,) = _rows([seg], budget=10)
    (val,) = _rows([seg], budget=20)
    assert len(train.response_ids) == 6 and len(val.response_ids) == 12


def test_a_condensed_segment_counts_its_condensed_prompt():
    # the second segment opens on a long condensed context, so less of its response fits
    first = _segment(4, [1, 1, 0, 0], [[1, 0, 2]])
    second = _segment(7, [1, 0, 0, 0, 0], [[2, 0, 1]])
    rows = _rows([first, second], budget=9)
    assert [len(r.prompt_ids) + len(r.response_ids) for r in rows] == [8, 9]
    assert rows[1].response_mask == [1, 0]
