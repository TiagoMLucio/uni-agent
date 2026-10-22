"""One teacher sub-row per hinted turn, each with a clean history.

A wrong meta encoding misaligns spans rather than raising, so the tokens each sub-row
stands for are checked directly.

Runs under pytest, or as ``python tests/uni_agent/sdpo/test_teacher_subrows.py``.
"""

import pytest
import torch

from uni_agent.sdpo.hints import HintedTurn
from uni_agent.sdpo.splice import build_spliced_teacher_row
from verl.trainer.ppo.sdpo.teacher_meta import DEGENERATE_META, SubRowSpan, sub_row_ids, unpack, unpack_row
from verl.workers.utils.sdpo import response_keep_positions, teacher_branches

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


def _subs(seq, meta, contrast=False, anchor=False):
    """The whole sequence each sub-row of one group stands for."""
    base, offset, noted = unpack_row(seq, meta)
    return [
        sub_row_ids(base, offset, sub).tolist() for sub in noted if (sub.contrast, sub.anchor) == (contrast, anchor)
    ]


def _branches(seq, meta, response):
    nested = [torch.nested.nested_tensor([value], layout=torch.jagged) for value in (seq, torch.tensor(meta), response)]
    return teacher_branches(*nested)[0]


def test_one_subrow_per_hint():
    hinted = [
        HintedTurn(3, 12, 16, "a"),
        HintedTurn(5, 24, 28, "b"),
        HintedTurn(7, 33, 36, "c"),
    ]
    seq, meta, _, _ = _build(hinted, [_hint(900), _hint(901), _hint(902)])
    assert meta[:2] == [3, PROMPT.shape[0]]
    assert len(meta) == 2 + 4 * 3


def test_the_trajectory_ships_once():
    """The base is the prompt and the response up to the last scored span, followed by the notes alone."""
    hinted = [HintedTurn(3, 12, 16, "a"), HintedTurn(5, 24, 28, "b")]
    seq, meta, _, _ = _build(hinted, [_hint(900), _hint(901, length=4)])
    assert seq.tolist() == torch.cat([PROMPT, RESPONSE[:28], _hint(900), _hint(901, length=4)]).tolist()
    assert [sub_row.note_len for sub_row in unpack(meta)] == [3, 4]


def test_a_subrow_carries_only_its_own_hint():
    """The whole point: sub-row 2 must not contain hint 1, or the teacher scores turn 2
    from a state where it gave advice that was then ignored."""
    hinted = [HintedTurn(3, 12, 16, "a"), HintedTurn(5, 24, 28, "b")]
    seq, meta, _, _ = _build(hinted, [_hint(900), _hint(901)])
    carried = [{900 in sub, 901 in sub} for sub in _subs(seq, meta)]
    assert carried[0] == {True, False}, "sub-row 1 should carry only hint 900"
    assert carried[1] == {False, True}, "sub-row 2 should carry only hint 901"


def test_a_subrow_ends_on_its_scored_span():
    hinted = [HintedTurn(3, 12, 16, "a"), HintedTurn(5, 24, 28, "b")]
    seq, meta, _, _ = _build(hinted, [_hint(900), _hint(901)])
    for sub, hint in zip(_subs(seq, meta), hinted, strict=True):
        assert sub[-(hint.end - hint.start) :] == RESPONSE[hint.start : hint.end].tolist()


def test_history_before_the_hint_is_untouched():
    hinted = [HintedTurn(5, 24, 28, "b")]
    seq, meta, _, _ = _build(hinted, [_hint(901)])
    (sub,) = _subs(seq, meta)
    # everything before the hint is the verbatim trajectory, up to the turn's header at least
    at = sub.index(901)
    assert at >= PROMPT.shape[0] + 24 - HEADER.shape[0]
    assert sub[:at] == torch.cat([PROMPT, RESPONSE]).tolist()[:at]


def test_a_branch_is_its_subrow_after_the_shared_base():
    """The cached teacher forwards the base up to a branch's point once and the branch on top of it, so
    the two together must be the whole sub-row, and the branch must score the span at body_start."""
    hinted = [HintedTurn(3, 12, 16, "a"), HintedTurn(5, 24, 28, "b"), HintedTurn(7, 33, 36, "c")]
    seq, meta, _, _ = build_spliced_teacher_row(
        PROMPT, RESPONSE_WITH_HEADERS, hinted, [_hint(900), _hint(901), _hint(902)], 64, HEADER,
        contrast_ids=_hint(800, length=5), anchor=True,
    )
    base, branches = _branches(seq, meta, RESPONSE_WITH_HEADERS)
    assert [branch.point for branch in branches] == sorted(branch.point for branch in branches)
    subs = _subs(seq, meta) + _subs(seq, meta, contrast=True) + _subs(seq, meta, anchor=True)
    by_span = {(b.span.start, b.span.contrast, b.span.anchor): b for b in branches}
    groups = [(False, False)] * 3 + [(True, False)] * 3 + [(False, True)] * 3
    for sub, hint, (contrast, anchor) in zip(subs, hinted * 3, groups, strict=True):
        branch = by_span[(hint.start, contrast, anchor)]
        assert base[: branch.point].tolist() + branch.ids.tolist() == sub
        scored = branch.ids[1:][branch.span.body_start : branch.span.body_start + hint.end - hint.start]
        torch.testing.assert_close(scored, RESPONSE_WITH_HEADERS[hint.start : hint.end])


def test_branches_follow_their_spans():
    hinted = [
        HintedTurn(3, 12, 16, "a"),
        HintedTurn(5, 24, 28, "b"),
        HintedTurn(7, 33, 36, "c"),
    ]
    seq, meta, _, _ = _build(hinted, [_hint(900), _hint(901), _hint(902)])
    _, branches = _branches(seq, meta, RESPONSE)
    assert [b.span.parent for b in branches] == [0, 0, 0]
    assert [(b.span.start, b.span.end) for b in branches] == [(12, 16), (24, 28), (33, 36)]


def test_student_keeps_the_union_of_spans():
    """The student scores every hinted span in one pass, so its keep positions are the union
    even though the teacher splits them."""
    hinted = [HintedTurn(3, 12, 16, "a"), HintedTurn(5, 24, 28, "b")]
    _, meta, _, _ = _build(hinted, [_hint(900), _hint(901)])
    prompt_len = PROMPT.shape[0]
    input_ids = torch.nested.nested_tensor([torch.cat([PROMPT, RESPONSE])], layout=torch.jagged)
    responses = torch.nested.nested_tensor([RESPONSE], layout=torch.jagged)
    metas = torch.nested.nested_tensor([torch.tensor(meta)], layout=torch.jagged)

    keep = response_keep_positions(input_ids, responses, metas).values().tolist()
    want = [prompt_len + p - 1 for p in list(range(12, 16)) + list(range(24, 28))]
    assert keep == want


def _swap(sub, old_value, new_ids):
    at = sub.index(old_value)
    return sub[:at] + new_ids.tolist() + sub[at + 3 :]


PLACEMENTS = [
    (False, [HintedTurn(3, 12, 16, "a"), HintedTurn(5, 24, 28, "b")], PROMPT),
    (True, [HintedTurn(3, 12, 16, "a"), HintedTurn(5, 24, 28, "b")], PROMPT),
    (False, [HintedTurn(1, 0, 4, "a"), HintedTurn(5, 24, 28, "b")], torch.cat([PROMPT, HEADER])),
    (True, [HintedTurn(1, 0, 4, "a"), HintedTurn(5, 24, 28, "b")], torch.cat([PROMPT, HEADER])),
]


@pytest.mark.parametrize("after_header, hinted, prompt", PLACEMENTS)
def test_contrast_subrow_is_the_hint_subrow_with_only_the_hint_swapped(after_header, hinted, prompt):
    """Per hinted turn a second sub-row follows the hint ones; it differs from its hint sub-row in the
    hint tokens alone, for a hint before the header, after it, and joining the prompt on the first turn."""
    contrast = _hint(800, length=5)
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
    assert len(meta) == 2 + 4 * 2 * len(hinted) and len(plain_meta) == 2 + 4 * len(hinted)
    torch.testing.assert_close(seq[: plain_seq.shape[0]], plain_seq)

    for hint_value, hint_sub, contrast_sub, hint_row, contrast_row in zip(
        (900, 901), _subs(plain_seq, plain_meta), _subs(seq, meta, contrast=True), hint_rows, contrast_rows, strict=True
    ):
        assert contrast_sub == _swap(hint_sub, hint_value, contrast)
        assert (contrast_row.split, contrast_row.start, contrast_row.end) == (
            hint_row.split, hint_row.start, hint_row.end
        )
        assert contrast_row.note_len == contrast.shape[0]


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
    assert len(meta) == 2 + 4 * 3 * len(hinted)
    # no note: the anchors add sub-rows to the meta and nothing to the tokens
    torch.testing.assert_close(seq, contrast_seq)

    anchor_rows = unpack(meta, anchor=True)
    for hint, sub, row, hint_row in zip(hinted, _subs(seq, meta, anchor=True), anchor_rows, unpack(meta), strict=True):
        assert sub == torch.cat([prompt, RESPONSE_WITH_HEADERS[: hint.end]]).tolist()
        assert (row.split, row.note_len, row.start, row.end) == (hint_row.split, 0, hint.start, hint.end)


def test_an_anchor_needs_a_contrast():
    hinted = [HintedTurn(3, 12, 16, "a")]
    with pytest.raises(ValueError, match="contrast_ids"):
        build_spliced_teacher_row(PROMPT, RESPONSE_WITH_HEADERS, hinted, [_hint(900)], 64, HEADER, anchor=True)


def test_contrast_groups_follow_in_order_before_the_anchor():
    """Several contrast teachers ship one group each, in order, every sub-row its hint sub-row with only the
    note swapped; the anchor group comes after them all, and the branches flag each group."""
    hinted = [HintedTurn(3, 12, 16, "a"), HintedTurn(5, 24, 28, "b")]
    hints = [_hint(900), _hint(901)]
    groups = [_hint(800, length=5), [_hint(810, length=2), _hint(811, length=4)], _hint(820, length=1)]
    plain_seq, plain_meta, _, _ = build_spliced_teacher_row(PROMPT, RESPONSE_WITH_HEADERS, hinted, hints, 64, HEADER)
    seq, meta, _, _ = build_spliced_teacher_row(
        PROMPT, RESPONSE_WITH_HEADERS, hinted, hints, 64, HEADER, contrast_groups=groups, anchor=True
    )
    assert len(meta) == 2 + 4 * 2 * (1 + len(groups) + 1)
    plain = _subs(plain_seq, plain_meta)
    contrasts = _subs(seq, meta, contrast=True)
    assert len(contrasts) == 2 * len(groups)
    for g, group in enumerate(groups):
        per_turn = group if isinstance(group, list) else [group, group]
        for t, (value, contrast) in enumerate(zip((900, 901), per_turn, strict=True)):
            assert contrasts[g * 2 + t] == _swap(plain[t], value, contrast)
    assert _subs(seq, meta, anchor=True) == [
        PROMPT.tolist() + RESPONSE_WITH_HEADERS[:16].tolist(),
        PROMPT.tolist() + RESPONSE_WITH_HEADERS[:28].tolist(),
    ]
    branches = _branches(seq, meta, RESPONSE_WITH_HEADERS)[1]
    assert sum(branch.span.contrast for branch in branches) == 2 * len(groups)
    assert sum(branch.span.anchor for branch in branches) == 2
    with pytest.raises(ValueError, match="not both"):
        build_spliced_teacher_row(
            PROMPT, RESPONSE_WITH_HEADERS, hinted, hints, 64, HEADER, contrast_ids=groups[0], contrast_groups=groups
        )


def test_contrast_ids_may_differ_per_hinted_turn():
    hinted = [HintedTurn(3, 12, 16, "a"), HintedTurn(5, 24, 28, "b")]
    hints, per_turn = [_hint(900), _hint(901)], [_hint(800, length=5), _hint(801, length=2)]
    plain_seq, plain_meta, _, _ = build_spliced_teacher_row(PROMPT, RESPONSE_WITH_HEADERS, hinted, hints, 64, HEADER)
    seq, meta, _, _ = build_spliced_teacher_row(
        PROMPT, RESPONSE_WITH_HEADERS, hinted, hints, 64, HEADER, contrast_ids=per_turn
    )
    for value, hint_sub, contrast_sub, contrast in zip(
        (900, 901), _subs(plain_seq, plain_meta), _subs(seq, meta, contrast=True), per_turn, strict=True
    ):
        assert contrast_sub == _swap(hint_sub, value, contrast)


def test_contrast_leaves_the_students_spans_and_keep_positions_alone():
    """The branches flag the contrast sub-rows, on the same spans as the hint ones, and the student
    still keeps each span once."""
    hinted = [HintedTurn(3, 12, 16, "a"), HintedTurn(5, 24, 28, "b")]
    seq, meta, _, _ = build_spliced_teacher_row(
        PROMPT, RESPONSE_WITH_HEADERS, hinted, [_hint(900), _hint(901)], 64, HEADER, contrast_ids=_hint(800, length=5)
    )
    _, branches = _branches(seq, meta, RESPONSE_WITH_HEADERS)
    spans = sorted((b.span.start, b.span.contrast) for b in branches)
    assert spans == [(12, False), (12, True), (24, False), (24, True)]

    responses = torch.nested.nested_tensor([RESPONSE_WITH_HEADERS], layout=torch.jagged)
    metas = torch.nested.nested_tensor([torch.tensor(meta)], layout=torch.jagged)
    input_ids = torch.nested.nested_tensor([torch.cat([PROMPT, RESPONSE_WITH_HEADERS])], layout=torch.jagged)
    keep = response_keep_positions(input_ids, responses, metas).values().tolist()
    assert keep == [PROMPT.shape[0] + p - 1 for p in list(range(12, 16)) + list(range(24, 28))]


def test_degenerate_row_encoding_round_trips():
    """The stub an un-hinted row ships is one branch from the start of its 2-token base."""
    base, (branch,) = _branches(torch.tensor([100, 0]), DEGENERATE_META, RESPONSE)
    assert branch.point == 0 and branch.ids.tolist() == [100, 0]
    assert branch.span == SubRowSpan(parent=0, body_start=0, start=0, end=1)


def test_first_turn_hint_joins_the_prefix():
    """A turn starting at position 0 has its header in the prompt, so the hint goes there."""
    prompt = torch.cat([torch.arange(100, 108), HEADER])
    hinted = [HintedTurn(1, 0, 4, "a")]
    seq, meta, fallbacks, _ = build_spliced_teacher_row(
        prompt, RESPONSE, hinted, [_hint(900)], max_prefix_len=64, header_ids=HEADER
    )
    assert fallbacks == 0
    (sub,) = _subs(seq, meta)
    assert sub == torch.cat([prompt[:-2], _hint(900), HEADER, RESPONSE[:4]]).tolist()


def test_fallback_counted_when_no_header_precedes_the_span():
    hinted = [HintedTurn(3, 13, 17, "a")]  # 13-2=11,12 are not the header tokens
    seq, meta, fallbacks, _ = _build(hinted, [_hint(900)])
    assert fallbacks == 1
    assert meta[0] == 1
    (sub,) = _subs(seq, meta)
    # at the span start instead
    assert sub == torch.cat([PROMPT, RESPONSE[:13], _hint(900), RESPONSE[13:17]]).tolist()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
