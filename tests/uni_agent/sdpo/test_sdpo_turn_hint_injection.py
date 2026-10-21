"""User-turn hint insertion in build_spliced_teacher_row: the hint lands before the turn's
assistant header, with a bare-splice fallback when the header isn't where expected. Each sub-row
the row stands for must end on its span's verbatim tokens."""

import pytest
import torch

from uni_agent.sdpo.hints import HintedTurn
from uni_agent.sdpo.splice import build_spliced_teacher_row
from verl.trainer.ppo.sdpo.batch import trace_weights
from verl.trainer.ppo.sdpo.teacher_meta import sub_row_ids, unpack_row

HEADER = torch.tensor([90, 91, 92], dtype=torch.int64)


def _subs(seq, meta):
    """The whole sequence each sub-row of the shipped row stands for."""
    base, offset, noted = unpack_row(seq, meta)
    return [sub_row_ids(base, offset, sub) for sub in noted]


def _spans_map_back(seq, meta, response_ids, hinted_turns):
    """Each sub-row ends on its scored span, which must reproduce the student's tokens verbatim."""
    subs = _subs(seq, meta)
    assert meta[0] == len(hinted_turns) == len(subs)
    for sub, hint in zip(subs, hinted_turns, strict=True):
        span = sub[sub.shape[0] - (hint.end - hint.start) :]
        assert torch.equal(span, response_ids[hint.start : hint.end]), f"span [{hint.start},{hint.end}) corrupted"


def test_hint_inserted_before_assistant_header():
    prompt = torch.arange(10, dtype=torch.int64)
    # response: turn0 [0:4), obs+header [4:12) with header at [9:12), turn1 [12:16)
    response = torch.tensor([0, 1, 2, 3, 50, 51, 52, 53, 54, 90, 91, 92, 10, 11, 12, 13], dtype=torch.int64)
    hinted = [HintedTurn(1, 12, 16, "hint")]
    hint = torch.tensor([70, 71], dtype=torch.int64)

    seq, meta, fallbacks, _ = build_spliced_teacher_row(prompt, response, hinted, [hint], 100, HEADER)

    assert fallbacks == 0
    expected = torch.cat([prompt, response[:9], hint, response[9:16]])
    assert torch.equal(torch.cat(_subs(seq, meta)), expected), "hint must sit before the assistant header, not after it"
    _spans_map_back(seq, meta, response, hinted)


def test_first_turn_hint_joins_prompt_tail():
    prompt = torch.cat([torch.arange(5, dtype=torch.int64), HEADER])
    response = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    hinted = [HintedTurn(0, 0, 4, "hint")]
    hint = torch.tensor([70, 71], dtype=torch.int64)

    seq, meta, fallbacks, _ = build_spliced_teacher_row(prompt, response, hinted, [hint], 100, HEADER)

    assert fallbacks == 0
    expected = torch.cat([prompt[:-3], hint, HEADER, response])
    assert torch.equal(torch.cat(_subs(seq, meta)), expected)
    assert meta == [1, 8, 5, 2, 0, 4], "the hint goes before the prompt's header tail"
    _spans_map_back(seq, meta, response, hinted)


def test_missing_header_falls_back_to_bare_splice():
    prompt = torch.arange(10, dtype=torch.int64)
    response = torch.tensor([0, 1, 2, 3, 50, 51, 52, 10, 11, 12], dtype=torch.int64)  # no header anywhere
    hinted = [HintedTurn(1, 7, 10, "hint")]
    hint = torch.tensor([70, 71], dtype=torch.int64)

    seq, meta, fallbacks, _ = build_spliced_teacher_row(prompt, response, hinted, [hint], 100, HEADER)

    assert fallbacks == 1
    expected = torch.cat([prompt, response[:7], hint, response[7:10]])
    assert torch.equal(torch.cat(_subs(seq, meta)), expected)
    _spans_map_back(seq, meta, response, hinted)


def test_cumulative_hints_and_truncation_after_last_span():
    prompt = torch.arange(4, dtype=torch.int64)
    response = torch.cat(
        [
            torch.tensor([0, 1], dtype=torch.int64),  # obs
            HEADER,
            torch.tensor([10, 11, 12], dtype=torch.int64),  # turn a: [5:8)
            torch.tensor([60, 61], dtype=torch.int64),  # obs
            HEADER,
            torch.tensor([20, 21], dtype=torch.int64),  # turn b: [13:15)
            torch.tensor([98, 99], dtype=torch.int64),  # trailing tokens beyond last span
        ]
    )
    hinted = [HintedTurn(0, 5, 8, "a"), HintedTurn(1, 13, 15, "b")]
    hints = [torch.tensor([70], dtype=torch.int64), torch.tensor([71], dtype=torch.int64)]

    seq, meta, fallbacks, _ = build_spliced_teacher_row(prompt, response, hinted, hints, 100, HEADER)

    assert fallbacks == 0
    expected = torch.cat(
        [prompt, response[:2], hints[0], response[2:8], prompt, response[:10], hints[1], response[10:15]]
    )
    assert torch.equal(torch.cat(_subs(seq, meta)), expected), "one sub-row per hint, each cut after its own span"
    assert torch.equal(seq, torch.cat([prompt, response[:15], *hints])), "the trajectory ships once"
    _spans_map_back(seq, meta, response, hinted)


def test_thought_opens_the_turn_after_its_header():
    prompt = torch.arange(10, dtype=torch.int64)
    response = torch.tensor([0, 1, 2, 3, 50, 51, 52, 53, 54, 90, 91, 92, 10, 11, 12, 13], dtype=torch.int64)
    hinted = [HintedTurn(1, 12, 16, "hint")]
    hint = torch.tensor([70, 71], dtype=torch.int64)

    seq, meta, fallbacks, _ = build_spliced_teacher_row(prompt, response, hinted, [hint], 100, HEADER, after_header=True)

    assert fallbacks == 0
    expected = torch.cat([prompt, response[:12], hint, response[12:16]])
    assert torch.equal(torch.cat(_subs(seq, meta)), expected), (
        "a thought sits after the assistant header, before what the agent wrote"
    )
    _spans_map_back(seq, meta, response, hinted)


def test_first_turn_thought_follows_the_prompt_tail():
    prompt = torch.cat([torch.arange(5, dtype=torch.int64), HEADER])
    response = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    hinted = [HintedTurn(0, 0, 4, "hint")]
    hint = torch.tensor([70, 71], dtype=torch.int64)

    seq, meta, fallbacks, _ = build_spliced_teacher_row(prompt, response, hinted, [hint], 100, HEADER, after_header=True)

    assert fallbacks == 0
    assert torch.equal(torch.cat(_subs(seq, meta)), torch.cat([prompt, hint, response]))
    assert meta == [1, 8, 8, 2, 0, 4], "the thought follows the prompt's header tail"
    _spans_map_back(seq, meta, response, hinted)


def test_thought_without_its_header_falls_back_to_the_span_start():
    prompt = torch.arange(10, dtype=torch.int64)
    response = torch.tensor([0, 1, 2, 3, 50, 51, 52, 10, 11, 12], dtype=torch.int64)
    hinted = [HintedTurn(1, 7, 10, "hint")]
    hint = torch.tensor([70, 71], dtype=torch.int64)

    seq, meta, fallbacks, _ = build_spliced_teacher_row(prompt, response, hinted, [hint], 100, HEADER, after_header=True)

    assert fallbacks == 1
    assert torch.equal(torch.cat(_subs(seq, meta)), torch.cat([prompt, response[:7], hint, response[7:10]]))
    _spans_map_back(seq, meta, response, hinted)


def test_cumulative_thoughts_each_open_their_own_turn():
    prompt = torch.arange(4, dtype=torch.int64)
    response = torch.cat(
        [
            torch.tensor([0, 1], dtype=torch.int64),
            HEADER,
            torch.tensor([10, 11, 12], dtype=torch.int64),  # turn a: [5:8)
            torch.tensor([60, 61], dtype=torch.int64),
            HEADER,
            torch.tensor([20, 21], dtype=torch.int64),  # turn b: [13:15)
            torch.tensor([98, 99], dtype=torch.int64),
        ]
    )
    hinted = [HintedTurn(0, 5, 8, "a"), HintedTurn(1, 13, 15, "b")]
    hints = [torch.tensor([70], dtype=torch.int64), torch.tensor([71], dtype=torch.int64)]

    seq, meta, fallbacks, _ = build_spliced_teacher_row(prompt, response, hinted, hints, 100, HEADER, after_header=True)

    assert fallbacks == 0
    expected = torch.cat(
        [prompt, response[:5], hints[0], response[5:8], prompt, response[:13], hints[1], response[13:15]]
    )
    assert torch.equal(torch.cat(_subs(seq, meta)), expected)
    _spans_map_back(seq, meta, response, hinted)


# --- trace weights -----------------------------------------------------------------------


def test_trace_weights_default_matches_one_per_supervised_row():
    w = trace_weights([10.0, 0.0, 5.0], [("a", "0"), ("a", "0"), ("b", "0")])
    assert sum(w) == pytest.approx(2.0), "weights renormalise to the supervised-row count"
    assert w[1] == 0.0, "unsupervised rows stay at zero"


def test_trace_weights_split_a_condensed_trajectory_by_supervision():
    # one trajectory, two segments: the 3:1 supervision split sets their relative weight,
    # and the renormalisation restores the supervised-row count (not the trajectory count)
    w = trace_weights([30.0, 10.0], [("a", "0"), ("a", "0")])
    assert w[0] / w[1] == pytest.approx(3.0)
    assert sum(w) == pytest.approx(2.0)
