"""User-turn hint insertion in build_spliced_teacher_row: the hint lands before the turn's
assistant header, with a bare-splice fallback when the header isn't where expected. Meta must
keep mapping each span's verbatim tokens on the body grid."""

import pytest
import torch

from uni_agent.sdpo.hints import HintedTurn
from uni_agent.sdpo.splice import build_spliced_teacher_row
from verl.trainer.ppo.sdpo.batch import trace_weights

HEADER = torch.tensor([90, 91, 92], dtype=torch.int64)


def _spans_map_back(seq, meta, response_ids, hinted_turns):
    """meta is [n_sub, (total_len, body_len, body_start, start, end) per sub-row]; each
    scored span must reproduce the student's tokens verbatim on the sub-row grid."""
    n_sub, rest = meta[0], meta[1:]
    assert n_sub == len(hinted_turns) and len(rest) == 5 * n_sub
    offset = 0
    for j in range(0, len(rest), 5):
        total, body_len, body_start, start, end = rest[j : j + 5]
        prefix_len = total - body_len
        span = seq[offset + prefix_len + body_start : offset + prefix_len + body_start + (end - start)]
        assert torch.equal(span, response_ids[start:end]), f"span [{start},{end}) corrupted"
        offset += total
    assert offset == seq.shape[0]


def test_hint_inserted_before_assistant_header():
    prompt = torch.arange(10, dtype=torch.int64)
    # response: turn0 [0:4), obs+header [4:12) with header at [9:12), turn1 [12:16)
    response = torch.tensor([0, 1, 2, 3, 50, 51, 52, 53, 54, 90, 91, 92, 10, 11, 12, 13], dtype=torch.int64)
    hinted = [HintedTurn(1, 12, 16, "hint")]
    hint = torch.tensor([70, 71], dtype=torch.int64)

    seq, meta, fallbacks, _ = build_spliced_teacher_row(prompt, response, hinted, [hint], 100, HEADER)

    assert fallbacks == 0
    expected = torch.cat([prompt, response[:9], hint, response[9:16]])
    assert torch.equal(seq, expected), "hint must sit before the assistant header, not after it"
    _spans_map_back(seq, meta, response, hinted)


def test_first_turn_hint_joins_prompt_tail():
    prompt = torch.cat([torch.arange(5, dtype=torch.int64), HEADER])
    response = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    hinted = [HintedTurn(0, 0, 4, "hint")]
    hint = torch.tensor([70, 71], dtype=torch.int64)

    seq, meta, fallbacks, _ = build_spliced_teacher_row(prompt, response, hinted, [hint], 100, HEADER)

    assert fallbacks == 0
    expected = torch.cat([prompt[:-3], hint, HEADER, response])
    assert torch.equal(seq, expected)
    assert meta == [1, seq.shape[0], 4, 0, 0, 4], "hint in the prefix must not count toward the body"
    _spans_map_back(seq, meta, response, hinted)


def test_missing_header_falls_back_to_bare_splice():
    prompt = torch.arange(10, dtype=torch.int64)
    response = torch.tensor([0, 1, 2, 3, 50, 51, 52, 10, 11, 12], dtype=torch.int64)  # no header anywhere
    hinted = [HintedTurn(1, 7, 10, "hint")]
    hint = torch.tensor([70, 71], dtype=torch.int64)

    seq, meta, fallbacks, _ = build_spliced_teacher_row(prompt, response, hinted, [hint], 100, HEADER)

    assert fallbacks == 1
    expected = torch.cat([prompt, response[:7], hint, response[7:10]])
    assert torch.equal(seq, expected)
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
    assert torch.equal(seq, expected), "one sub-row per hint, each cut after its own span"
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
