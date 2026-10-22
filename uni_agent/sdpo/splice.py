"""The spliced teacher row: the trajectory once and one note per sub-row, its meta packed by
:mod:`verl.trainer.ppo.sdpo.teacher_meta`, plus the per-token mask."""

from typing import Optional

import torch

from uni_agent.sdpo.hints import HintedTurn
from verl.trainer.ppo.sdpo.teacher_meta import SubRow, pack

__all__ = ["build_spliced_teacher_row", "turn_token_mask"]


def _split(
    prompt: torch.Tensor, response_ids: torch.Tensor, start: int, header_ids: torch.Tensor, after_header: bool
) -> tuple[int, bool]:
    """Where a turn's note goes in the base (the prompt, then the response) and whether that is a fallback:
    before the turn's assistant header, or with ``after_header`` right after it; at the span start when the
    header is not where the span says."""
    header, offset = header_ids.shape[0], prompt.shape[0]
    if start >= header and torch.equal(response_ids[start - header : start], header_ids):
        return offset + (start if after_header else start - header), False
    if start == 0 and offset >= header and torch.equal(prompt[-header:], header_ids):
        # first turn: its assistant header is the prompt tail
        return (offset if after_header else offset - header), False
    return offset + start, True


def build_spliced_teacher_row(
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    hinted_turns: list[HintedTurn],
    hint_ids_list: list[torch.Tensor],
    max_prefix_len: int,
    header_ids: torch.Tensor,
    after_header: bool = False,
    contrast_ids: Optional[torch.Tensor | list[torch.Tensor]] = None,
    anchor: bool = False,
    contrast_groups: Optional[list[torch.Tensor | list[torch.Tensor]]] = None,
) -> tuple[torch.Tensor, list[int], int, list[tuple[int, int]]]:
    """One teacher sub-row per hinted turn, shipped as the trajectory once and each sub-row's note.

    Each sub-row is the trajectory up to its own turn with only its own hint spliced in,
    and truncated after the scored span. The hint goes immediately before the turn's
    assistant header, or with ``after_header`` immediately after it, opening the turn as
    its own text; the whole turn is scored either way. When the header is not where the span
    says, the hint is spliced at the span start instead and counted as a fallback.

    Carrying every hint in one sequence would make the teacher score a later turn from a
    state it could not reach: it would see its own earlier advice followed by the student
    ignoring it.

    With ``contrast_ids`` (one tensor for every hinted turn, or one per hinted turn), one more
    sub-row per hinted turn follows them: the same sub-row with those tokens in place of the hint;
    ``contrast_groups`` gives several such groups in order, one per contrast teacher. With ``anchor``
    too, a last sub-row per hinted turn follows those: the same turn with no note spliced at all.

    The prompt is left-truncated to ``max_prefix_len``. Returns the base (the prompt, then the
    response up to the last scored span) followed by every sub-row's note, the packed meta (see
    :mod:`verl.trainer.ppo.sdpo.teacher_meta`), the number of fallback placements and the
    (start, end) spans for the distillation mask (== the meta spans).
    """
    if contrast_ids is not None and contrast_groups is not None:
        raise ValueError("pass contrast_ids (one contrast teacher) or contrast_groups (several), not both")
    contrasts = [contrast_ids] if contrast_ids is not None else list(contrast_groups or [])
    if anchor and not contrasts:
        raise ValueError("a no-note anchor sub-row only anchors a contrast: pass contrast_ids or contrast_groups")
    prompt = prompt_ids if prompt_ids.shape[0] <= max_prefix_len else prompt_ids[-max_prefix_len:]
    placed = [_split(prompt, response_ids, hint.start, header_ids, after_header) for hint in hinted_turns]
    notes = [ids.to(response_ids.dtype) for ids in hint_ids_list]
    for group in contrasts:
        notes += group if isinstance(group, list) else [group] * len(hinted_turns)
    if contrasts and anchor:
        notes += [response_ids.new_zeros(0)] * len(hinted_turns)
    turns = len(hinted_turns)
    groups = [
        [
            SubRow(split, note.shape[0], hint.start, hint.end)
            for (split, _), note, hint in zip(placed, notes[at : at + turns], hinted_turns, strict=True)
        ]
        for at in range(0, len(notes), turns)
    ]
    base = torch.cat([prompt, response_ids[: max(hint.end for hint in hinted_turns)]])
    return (
        torch.cat([base, *(note.to(base.dtype) for note in notes)]),
        pack(prompt.shape[0], *groups),
        sum(fallback for _, fallback in placed),
        [(hint.start, hint.end) for hint in hinted_turns],
    )


def turn_token_mask(response_len: int, spans: list[tuple[int, int]], skip: int = 0) -> torch.Tensor:
    """Per-token distillation mask: 1 on the scored spans past their first ``skip`` tokens, 0 elsewhere."""
    mask = torch.zeros(response_len, dtype=torch.float32)
    for start, end in spans:
        mask[min(start + skip, end):end] = 1.0
    return mask
