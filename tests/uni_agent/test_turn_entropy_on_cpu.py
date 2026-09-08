"""Per-turn surprisal, over the tokens the model generated and nothing else.

`-log p(sampled)` is an unbiased single-sample estimate of the entropy of the distribution the
token was drawn from, so the sampler's own log-probs are enough and no second forward pass is
needed. The trap is that observation tokens are padded into the same list with a log-prob of
0.0, which reads as a perfectly predicted token: averaging them in would drag each turn toward
zero in proportion to how much its tools printed, which is the quantity that varies most along a
trajectory and would have looked like a real trend.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

pytest.importorskip("verl.experimental.agent_loop")

from uni_agent.agent_loop import turn_entropy_records  # noqa: E402


def _segment(spans, logprobs, mask):
    return {"rollout_cache": {"turn_spans": spans, "response_logprobs": logprobs, "response_mask": mask}}


def _step(idx, *tool_names):
    return SimpleNamespace(
        step_idx=idx,
        tool_results=[SimpleNamespace(name=name) for name in tool_names],
    )


def test_the_mean_is_over_the_generated_tokens_only():
    """One turn: three generated tokens, then four tokens of tool observation padded with 0.0."""
    logprobs = [-1.0, -2.0, -3.0] + [0.0, 0.0, 0.0, 0.0]
    mask = [1, 1, 1] + [0, 0, 0, 0]
    records = turn_entropy_records([_segment([[1, 0, 3]], logprobs, mask)], [_step(1, "execute_bash")])

    assert len(records) == 1
    assert records[0]["entropy"] == pytest.approx(2.0), "mean of 1, 2, 3 and not of 1, 2, 3, 0, 0, 0, 0"
    assert records[0]["tokens"] == 3


def test_the_padding_would_have_halved_it():
    """The size of the error the exclusion prevents, stated rather than implied."""
    logprobs = [-1.0, -2.0, -3.0] + [0.0] * 4
    mask = [1, 1, 1] + [0] * 4
    naive = sum(-lp for lp in logprobs) / len(logprobs)
    records = turn_entropy_records([_segment([[1, 0, 3]], logprobs, mask)], [_step(1)])
    assert naive == pytest.approx(6 / 7)
    assert records[0]["entropy"] > naive * 2


def test_a_later_turn_reads_its_own_span():
    """Turn 2's tokens sit after turn 1's observation, so the span is what separates them."""
    logprobs = [-1.0, -1.0] + [0.0, 0.0, 0.0] + [-4.0, -6.0]
    mask = [1, 1] + [0, 0, 0] + [1, 1]
    records = turn_entropy_records(
        [_segment([[1, 0, 2], [2, 5, 7]], logprobs, mask)],
        [_step(1, "execute_bash"), _step(2, "str_replace_editor")],
    )
    assert [r["turn"] for r in records] == [1, 2]
    assert [r["entropy"] for r in records] == [pytest.approx(1.0), pytest.approx(5.0)]
    assert [r["tokens"] for r in records] == [2, 2]


def test_a_run_without_logprobs_records_nothing():
    """Not zeros: a run that did not ask for them has nothing to say about entropy."""
    assert turn_entropy_records([_segment([[1, 0, 3]], [], [1, 1, 1])], [_step(1, "execute_bash")]) == []


def test_a_turn_with_no_tool_call_still_gets_a_record():
    records = turn_entropy_records([_segment([[1, 0, 2]], [-1.0, -3.0], [1, 1])], [_step(1)])
    assert records[0]["tools"] == []
    assert records[0]["entropy"] == pytest.approx(2.0)


def test_the_tools_of_a_turn_are_named_in_call_order():
    records = turn_entropy_records(
        [_segment([[1, 0, 2]], [-1.0, -1.0], [1, 1])], [_step(1, "execute_bash", "str_replace_editor")]
    )
    assert records[0]["tools"] == ["execute_bash", "str_replace_editor"]


def test_turns_are_numbered_from_one_and_counted_across_segments():
    """A condensation restarts the token buffer, so turn 3 indexes into the second segment."""
    first = _segment([[1, 0, 2], [2, 2, 4]], [-1.0, -1.0, -2.0, -2.0], [1, 1, 1, 1])
    second = _segment([[3, 0, 2]], [-3.0, -3.0], [1, 1])
    records = turn_entropy_records([first, second], [_step(1), _step(2), _step(3)])

    assert [r["turn"] for r in records] == [1, 2, 3], "1-based, and continuing across the boundary"
    assert {r["turns"] for r in records} == {3}, "every record carries the same denominator"
    assert [r["entropy"] for r in records] == [pytest.approx(1.0), pytest.approx(2.0), pytest.approx(3.0)]


def test_a_segment_without_logprobs_does_not_stop_the_others():
    records = turn_entropy_records(
        [_segment([[1, 0, 2]], [], [1, 1]), _segment([[2, 0, 2]], [-2.0, -2.0], [1, 1])],
        [_step(1), _step(2)],
    )
    assert [r["turn"] for r in records] == [2]


def test_a_surprisal_is_never_negative():
    """Sanity on the sign convention: log-probs are negative, surprisal is their negation."""
    records = turn_entropy_records([_segment([[1, 0, 3]], [-0.01, -5.0, -12.0], [1, 1, 1])], [_step(1)])
    assert records[0]["entropy"] > 0
    assert math.isclose(records[0]["entropy"], (0.01 + 5.0 + 12.0) / 3)


def test_the_record_reaches_the_dump_next_to_what_the_analysis_splits_on(tmp_path):
    """The shape the plotting script reads, taken from the real writer rather than described."""
    import json

    from uni_agent.agent_loop import UniAgentLoop
    from uni_agent.interaction.interaction import StepOutput, ToolResult

    call = ToolResult(tool_call_id="c", name="execute_bash", action="ls", observation="ok",
                      status="ok", execution_time=0.0)
    trajectory = [StepOutput(step_idx=1, exit_reason="completed", tool_results=[call]),
                  StepOutput(step_idx=2, exit_reason="finished", tool_results=[call])]
    cache = {"turn_spans": [[1, 0, 2], [2, 4, 6]],
             "response_logprobs": [-1.0, -1.0, 0.0, 0.0, -3.0, -3.0],
             "response_mask": [1, 1, 0, 0, 1, 1]}

    loop = UniAgentLoop.__new__(UniAgentLoop)
    loop.output_dir = tmp_path / "rollout"
    loop.identity = {"uid": "u1"}
    loop.env = SimpleNamespace(privileged_context="")
    loop._save_interaction_result({
        "rollout_cache": cache,
        "segments": [{"rollout_cache": cache, "prompt_messages": None}],
        "trajectory": trajectory,
        "execution_time": 1.0,
        "messages": [],
        "metrics": {},
        "reward_score": 1.0,
        "resolved": True,
    })

    dumped = json.loads((loop.output_dir / "interaction_result.json").read_text())
    assert dumped["turn_entropy"] == [
        {"turn": 1, "entropy": 1.0, "tokens": 2, "tools": ["execute_bash"], "turns": 2},
        {"turn": 2, "entropy": 3.0, "tokens": 2, "tools": ["execute_bash"], "turns": 2},
    ]
    # what the analysis splits on, in the same file
    assert dumped["resolved"] is True
    assert dumped["reward_score"] == 1.0
    assert dumped["trajectory"][-1]["exit_reason"] == "finished"


def test_a_dump_from_a_run_without_logprobs_has_no_entropy_key(tmp_path):
    import json

    from uni_agent.agent_loop import UniAgentLoop
    from uni_agent.interaction.interaction import StepOutput

    cache = {"turn_spans": [[1, 0, 2]], "response_logprobs": [], "response_mask": [1, 1]}
    loop = UniAgentLoop.__new__(UniAgentLoop)
    loop.output_dir = tmp_path / "rollout"
    loop.identity = {}
    loop.env = SimpleNamespace(privileged_context="")
    loop._save_interaction_result({
        "rollout_cache": cache,
        "segments": [{"rollout_cache": cache, "prompt_messages": None}],
        "trajectory": [StepOutput(step_idx=1, exit_reason="finished")],
        "execution_time": 1.0,
        "messages": [],
    })

    assert "turn_entropy" not in json.loads((loop.output_dir / "interaction_result.json").read_text())
