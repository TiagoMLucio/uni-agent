"""One teacher sub-row per hinted turn, each with a clean history.

A wrong meta encoding misaligns spans rather than raising, so the tokens each sub-row
carries are checked directly.

Runs under pytest, or as ``python tests/uni_agent/sdpo/test_teacher_subrows.py``.
"""

import pytest
import torch

from uni_agent.sdpo.hints import HintedTurn
from uni_agent.sdpo.splice import build_spliced_teacher_row
from verl.trainer.ppo.sdpo.teacher_meta import DEGENERATE_META, SubRowSpan, unpack
from verl.workers.utils.sdpo import explode_turn_teacher_rows, response_keep_positions

HEADER = torch.tensor([90, 91])
PROMPT = torch.arange(100, 110)
RESPONSE = torch.arange(0, 40)
# the same response with the assistant header ahead of the turns at 12 and 24
RESPONSE_WITH_HEADERS = RESPONSE.clone()
RESPONSE_WITH_HEADERS[10:12] = HEADER
RESPONSE_WITH_HEADERS[22:24] = HEADER


def _hint(value, length=3):
    return torch.full((length,), value, dtype=torch.int64)


def _build(hinted, hints):
    return build_spliced_teacher_row(PROMPT, RESPONSE, hinted, hints, max_prefix_len=64, header_ids=HEADER)


def test_one_subrow_per_hint():
    hinted = [
        HintedTurn(3, 12, 16, "a"),
        HintedTurn(5, 24, 28, "b"),
        HintedTurn(7, 33, 36, "c"),
    ]
    seq, meta, _, _ = _build(hinted, [_hint(900), _hint(901), _hint(902)])
    assert meta[0] == 3
    assert len(meta) == 1 + 5 * 3
    assert seq.shape[0] == sum(meta[1 + 5 * j] for j in range(3))


def test_a_subrow_carries_only_its_own_hint():
    """The whole point: sub-row 2 must not contain hint 1, or the teacher scores turn 2
    from a state where it gave advice that was then ignored."""
    hinted = [HintedTurn(3, 12, 16, "a"), HintedTurn(5, 24, 28, "b")]
    seq, meta, _, _ = _build(hinted, [_hint(900), _hint(901)])

    offset = 0
    carried = []
    for j in range(meta[0]):
        total_len = meta[1 + 5 * j]
        sub = seq[offset : offset + total_len]
        carried.append({900 in sub.tolist(), 901 in sub.tolist()})
        offset += total_len

    assert carried[0] == {True, False}, "sub-row 1 should carry only hint 900"
    assert carried[1] == {False, True}, "sub-row 2 should carry only hint 901"


def test_span_lands_at_body_start():
    """body_start must index the scored tokens inside the body, or the teacher's outputs
    scatter onto the wrong response positions."""
    hinted = [HintedTurn(3, 12, 16, "a"), HintedTurn(5, 24, 28, "b")]
    seq, meta, _, _ = _build(hinted, [_hint(900), _hint(901)])

    offset = 0
    for j in range(meta[0]):
        total_len, body_len, body_start, start, end = meta[1 + 5 * j : 6 + 5 * j]
        body = seq[offset : offset + total_len][-body_len:]
        torch.testing.assert_close(body[body_start : body_start + (end - start)], RESPONSE[start:end])
        offset += total_len


def test_history_before_the_hint_is_untouched():
    hinted = [HintedTurn(5, 24, 28, "b")]
    seq, meta, _, _ = _build(hinted, [_hint(901)])
    total_len, body_len, body_start, start, end = meta[1:6]
    body = seq[-body_len:]
    # everything before the hint is the verbatim trajectory
    torch.testing.assert_close(body[: start - HEADER.shape[0]], RESPONSE[: start - HEADER.shape[0]])


def test_exploder_returns_one_row_per_hint():
    hinted = [
        HintedTurn(3, 12, 16, "a"),
        HintedTurn(5, 24, 28, "b"),
        HintedTurn(7, 33, 36, "c"),
    ]
    seq, meta, _, _ = _build(hinted, [_hint(900), _hint(901), _hint(902)])
    seqs = torch.nested.nested_tensor([seq], layout=torch.jagged)
    metas = torch.nested.nested_tensor([torch.tensor(meta)], layout=torch.jagged)
    responses = torch.nested.nested_tensor([RESPONSE], layout=torch.jagged)

    sub_seqs, sub_resps, _, sub_spans = explode_turn_teacher_rows(seqs, metas, responses)
    assert [s.parent for s in sub_spans] == [0, 0, 0]
    assert [(s.start, s.end) for s in sub_spans] == [(12, 16), (24, 28), (33, 36)]
    for j, sub in enumerate(sub_seqs.unbind()):
        assert sub.shape[0] == meta[1 + 5 * j]


def test_student_keeps_the_union_of_spans():
    """The student scores every hinted span in one pass, so its keep positions are the union
    even though the teacher now splits them."""
    hinted = [HintedTurn(3, 12, 16, "a"), HintedTurn(5, 24, 28, "b")]
    _, meta, _, _ = _build(hinted, [_hint(900), _hint(901)])
    prompt_len = PROMPT.shape[0]
    input_ids = torch.nested.nested_tensor(
        [torch.cat([PROMPT, RESPONSE])], layout=torch.jagged
    )
    responses = torch.nested.nested_tensor([RESPONSE], layout=torch.jagged)
    metas = torch.nested.nested_tensor([torch.tensor(meta)], layout=torch.jagged)

    keep = response_keep_positions(input_ids, responses, metas).values().tolist()
    want = [prompt_len + p - 1 for p in list(range(12, 16)) + list(range(24, 28))]
    assert keep == want


def _split(seq, sub_rows):
    out, offset = [], 0
    for sub_row in sub_rows:
        out.append(seq[offset : offset + sub_row.total_len].tolist())
        offset += sub_row.total_len
    return out


def _swap(sub, old_value, new_ids):
    at = sub.index(old_value)
    return sub[:at] + new_ids.tolist() + sub[at + 3 :]


def test_contrast_subrow_is_the_hint_subrow_with_only_the_hint_swapped():
    """Per hinted turn a second sub-row follows the hint ones; it differs from its hint sub-row in the
    hint tokens alone, for a hint before the header, after it, and joining the prompt on the first turn."""
    contrast = _hint(800, length=5)
    cases = [
        (False, [HintedTurn(3, 12, 16, "a"), HintedTurn(5, 24, 28, "b")], PROMPT),
        (True, [HintedTurn(3, 12, 16, "a"), HintedTurn(5, 24, 28, "b")], PROMPT),
        (False, [HintedTurn(1, 0, 4, "a"), HintedTurn(5, 24, 28, "b")], torch.cat([PROMPT, HEADER])),
        (True, [HintedTurn(1, 0, 4, "a"), HintedTurn(5, 24, 28, "b")], torch.cat([PROMPT, HEADER])),
    ]
    for after_header, hinted, prompt in cases:
        hints = [_hint(900), _hint(901)]
        plain_seq, plain_meta, plain_fallbacks, plain_spans = build_spliced_teacher_row(
            prompt, RESPONSE_WITH_HEADERS, hinted, hints, 64, HEADER, after_header
        )
        seq, meta, fallbacks, spans = build_spliced_teacher_row(
            prompt, RESPONSE_WITH_HEADERS, hinted, hints, 64, HEADER, after_header, contrast_ids=contrast
        )
        assert (fallbacks, spans) == (plain_fallbacks, plain_spans) and fallbacks == 0
        hint_rows, contrast_rows = unpack(meta), unpack(meta, contrast=True)
        assert hint_rows == unpack(plain_meta) and unpack(plain_meta, contrast=True) == []
        assert len(meta) == 1 + 5 * 2 * len(hinted) and len(plain_meta) == 1 + 5 * len(hinted)
        torch.testing.assert_close(seq[: plain_seq.shape[0]], plain_seq)

        hint_subs = _split(plain_seq, hint_rows)
        contrast_subs = _split(seq[plain_seq.shape[0] :], contrast_rows)
        for hint_value, hint_sub, contrast_sub, hint_row, contrast_row in zip(
            (900, 901), hint_subs, contrast_subs, hint_rows, contrast_rows, strict=True
        ):
            assert contrast_sub == _swap(hint_sub, hint_value, contrast)
            assert (contrast_row.start, contrast_row.end) == (hint_row.start, hint_row.end)
            assert contrast_row.total_len - hint_row.total_len == contrast.shape[0] - 3
            body = torch.tensor(contrast_sub)[-contrast_row.body_len :]
            start, end = contrast_row.start, contrast_row.end
            torch.testing.assert_close(
                body[contrast_row.body_start : contrast_row.body_start + end - start], RESPONSE_WITH_HEADERS[start:end]
            )


PLACEMENTS = [
    (False, [HintedTurn(3, 12, 16, "a"), HintedTurn(5, 24, 28, "b")], PROMPT),
    (True, [HintedTurn(3, 12, 16, "a"), HintedTurn(5, 24, 28, "b")], PROMPT),
    (False, [HintedTurn(1, 0, 4, "a"), HintedTurn(5, 24, 28, "b")], torch.cat([PROMPT, HEADER])),
    (True, [HintedTurn(1, 0, 4, "a"), HintedTurn(5, 24, 28, "b")], torch.cat([PROMPT, HEADER])),
]


@pytest.mark.parametrize("after_header, hinted, prompt", PLACEMENTS)
def test_anchor_subrow_is_the_turn_with_no_note(after_header, hinted, prompt):
    """With ``anchor`` a third sub-row per hinted turn follows the contrast ones: the prompt and the
    response up to the turn's end with nothing spliced, scored on the same span."""
    hints, contrast = [_hint(900), _hint(901)], _hint(800, length=5)
    contrast_seq, contrast_meta, contrast_fallbacks, contrast_spans = build_spliced_teacher_row(
        prompt, RESPONSE_WITH_HEADERS, hinted, hints, 64, HEADER, after_header, contrast_ids=contrast
    )
    seq, meta, fallbacks, spans = build_spliced_teacher_row(
        prompt, RESPONSE_WITH_HEADERS, hinted, hints, 64, HEADER, after_header, contrast_ids=contrast, anchor=True
    )
    assert (fallbacks, spans) == (contrast_fallbacks, contrast_spans)
    assert unpack(meta) == unpack(contrast_meta)
    assert unpack(meta, contrast=True) == unpack(contrast_meta, contrast=True)
    assert len(meta) == 1 + 5 * 3 * len(hinted)
    torch.testing.assert_close(seq[: contrast_seq.shape[0]], contrast_seq)

    anchor_rows = unpack(meta, anchor=True)
    anchor_subs = _split(seq[contrast_seq.shape[0] :], anchor_rows)
    for hint, sub, row in zip(hinted, anchor_subs, anchor_rows, strict=True):
        assert sub == torch.cat([prompt, RESPONSE_WITH_HEADERS[: hint.end]]).tolist()
        assert (row.start, row.end, row.body_start, row.body_len) == (hint.start, hint.end, hint.start, hint.end)
    assert sum(row.total_len for row in anchor_rows) == seq.shape[0] - contrast_seq.shape[0]


def test_an_anchor_needs_a_contrast():
    hinted = [HintedTurn(3, 12, 16, "a")]
    with pytest.raises(ValueError, match="contrast_ids"):
        build_spliced_teacher_row(PROMPT, RESPONSE_WITH_HEADERS, hinted, [_hint(900)], 64, HEADER, anchor=True)


def test_contrast_ids_may_differ_per_hinted_turn():
    hinted = [HintedTurn(3, 12, 16, "a"), HintedTurn(5, 24, 28, "b")]
    hints, per_turn = [_hint(900), _hint(901)], [_hint(800, length=5), _hint(801, length=2)]
    plain_seq, plain_meta, _, _ = build_spliced_teacher_row(PROMPT, RESPONSE_WITH_HEADERS, hinted, hints, 64, HEADER)
    seq, meta, _, _ = build_spliced_teacher_row(
        PROMPT, RESPONSE_WITH_HEADERS, hinted, hints, 64, HEADER, contrast_ids=per_turn
    )
    hint_subs = _split(plain_seq, unpack(plain_meta))
    contrast_subs = _split(seq[plain_seq.shape[0] :], unpack(meta, contrast=True))
    for value, hint_sub, contrast_sub, contrast in zip((900, 901), hint_subs, contrast_subs, per_turn, strict=True):
        assert contrast_sub == _swap(hint_sub, value, contrast)


def test_contrast_leaves_the_students_spans_and_keep_positions_alone():
    """The exploder flags the contrast sub-rows after the hint ones, on the same spans, and the student
    still keeps each span once."""
    hinted = [HintedTurn(3, 12, 16, "a"), HintedTurn(5, 24, 28, "b")]
    seq, meta, _, _ = build_spliced_teacher_row(
        PROMPT, RESPONSE_WITH_HEADERS, hinted, [_hint(900), _hint(901)], 64, HEADER, contrast_ids=_hint(800, length=5)
    )
    seqs = torch.nested.nested_tensor([seq], layout=torch.jagged)
    metas = torch.nested.nested_tensor([torch.tensor(meta)], layout=torch.jagged)
    responses = torch.nested.nested_tensor([RESPONSE_WITH_HEADERS], layout=torch.jagged)

    _, sub_resps, _, sub_spans = explode_turn_teacher_rows(seqs, metas, responses)
    assert [s.contrast for s in sub_spans] == [False, False, True, True]
    assert [(s.parent, s.start, s.end) for s in sub_spans] == [(0, 12, 16), (0, 24, 28)] * 2
    for span, body in zip(sub_spans, sub_resps.unbind(), strict=True):
        scored = body[span.body_start : span.body_start + span.end - span.start]
        torch.testing.assert_close(scored, RESPONSE_WITH_HEADERS[span.start : span.end])

    input_ids = torch.nested.nested_tensor([torch.cat([PROMPT, RESPONSE_WITH_HEADERS])], layout=torch.jagged)
    keep = response_keep_positions(input_ids, responses, metas).values().tolist()
    assert keep == [PROMPT.shape[0] + p - 1 for p in list(range(12, 16)) + list(range(24, 28))]


def test_degenerate_row_encoding_round_trips():
    """The stub an un-hinted row ships must survive the exploder."""
    meta = torch.tensor(DEGENERATE_META)
    seqs = torch.nested.nested_tensor([torch.tensor([100, 0])], layout=torch.jagged)
    metas = torch.nested.nested_tensor([meta], layout=torch.jagged)
    responses = torch.nested.nested_tensor([RESPONSE], layout=torch.jagged)

    sub_seqs, sub_resps, _, sub_spans = explode_turn_teacher_rows(seqs, metas, responses)
    assert sub_spans == [SubRowSpan(parent=0, body_start=0, start=0, end=1)]
    assert sub_resps.unbind()[0].shape[0] == 1


def test_first_turn_hint_joins_the_prefix():
    """A turn starting at position 0 has its header in the prompt, so the hint goes there."""
    prompt = torch.cat([torch.arange(100, 108), HEADER])
    hinted = [HintedTurn(1, 0, 4, "a")]
    seq, meta, fallbacks, _ = build_spliced_teacher_row(
        prompt, RESPONSE, hinted, [_hint(900)], max_prefix_len=64, header_ids=HEADER
    )
    assert fallbacks == 0
    assert 900 in seq.tolist()
    total_len, body_len, body_start, start, end = meta[1:6]
    assert (start, end) == (0, 4)
    torch.testing.assert_close(seq[-body_len:][body_start : body_start + 4], RESPONSE[0:4])


def test_fallback_counted_when_no_header_precedes_the_span():
    hinted = [HintedTurn(3, 13, 17, "a")]  # 13-2=11,12 are not the header tokens
    _, meta, fallbacks, _ = _build(hinted, [_hint(900)])
    assert fallbacks == 1
    assert meta[0] == 1


if __name__ == "__main__":
    failures = 0
    for name, case in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            case()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    raise SystemExit(1 if failures else 0)
