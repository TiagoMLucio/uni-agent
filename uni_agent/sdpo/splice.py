"""The spliced teacher row: one sub-row per hint, its meta packed by
:mod:`verl.trainer.ppo.sdpo.teacher_meta`, plus the per-token mask."""

from typing import Optional

import torch

from uni_agent.sdpo.hints import HintedTurn
from verl.trainer.ppo.sdpo.teacher_meta import SubRow, pack

__all__ = ["build_spliced_teacher_row", "turn_token_mask"]


def _sub_row(
    prefix: torch.Tensor, body: list[torch.Tensor], scored: int, span: tuple[int, int]
) -> tuple[torch.Tensor, SubRow]:
    """Concatenate one sub-row from its prefix and ordered body pieces; ``body[scored]`` is
    the span the teacher scores."""
    body_len = sum(part.shape[0] for part in body)
    body_start = sum(part.shape[0] for part in body[:scored])
    return torch.cat([prefix, *body]), SubRow(prefix.shape[0] + body_len, body_len, body_start, *span)


def _splice(
    base_prefix: torch.Tensor,
    response_ids: torch.Tensor,
    hinted_turns: list[HintedTurn],
    hint_ids_list: list[torch.Tensor],
    header_ids: torch.Tensor,
    after_header: bool,
) -> tuple[list[torch.Tensor], list[SubRow], int]:
    """The sub-row of each hinted turn with its own hint tokens, their meta records and the fallback count."""
    header = header_ids.shape[0]
    pieces: list[torch.Tensor] = []
    sub_rows: list[SubRow] = []
    fallbacks = 0

    for hint, hint_ids in zip(hinted_turns, hint_ids_list, strict=True):
        start, end = hint.start, hint.end
        hint_ids = hint_ids.to(response_ids.dtype)
        prefix = base_prefix
        if start >= header and torch.equal(response_ids[start - header : start], header_ids):
            insert_at = start if after_header else start - header
        elif start == 0 and prefix.shape[0] >= header and torch.equal(prefix[-header:], header_ids):
            # first turn: its assistant header is the prompt tail, so the hint joins the prefix
            if after_header:
                prefix = torch.cat([prefix, hint_ids])
            else:
                prefix = torch.cat([prefix[:-header], hint_ids, prefix[-header:]])
            insert_at = None
        else:
            insert_at = start
            fallbacks += 1

        if insert_at is None:
            body = [response_ids[:start], response_ids[start:end]]
            row, sub_row = _sub_row(prefix, body, scored=1, span=(start, end))
        else:
            body = [
                response_ids[:insert_at],  # untouched history, no other hints
                hint_ids,
                response_ids[insert_at:start],  # the turn's assistant header, empty when the hint follows it
                response_ids[start:end],  # the span the teacher scores
            ]
            row, sub_row = _sub_row(prefix, body, scored=3, span=(start, end))

        pieces.append(row)
        sub_rows.append(sub_row)

    return pieces, sub_rows, fallbacks


def _no_note(
    base_prefix: torch.Tensor, response_ids: torch.Tensor, hinted_turns: list[HintedTurn]
) -> tuple[list[torch.Tensor], list[SubRow]]:
    """The sub-row of each hinted turn with nothing spliced in: the trajectory up to the turn's end."""
    pieces, sub_rows = [], []
    for hint in hinted_turns:
        body = [response_ids[: hint.start], response_ids[hint.start : hint.end]]
        row, sub_row = _sub_row(base_prefix, body, scored=1, span=(hint.start, hint.end))
        pieces.append(row)
        sub_rows.append(sub_row)
    return pieces, sub_rows


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
) -> tuple[torch.Tensor, list[int], int, list[tuple[int, int]]]:
    """One teacher sub-row per hinted turn, concatenated into a single row.

    Each sub-row is the trajectory up to its own turn with only its own hint spliced in,
    and truncated after the scored span. The hint goes immediately before the turn's
    assistant header, or with ``after_header`` immediately after it, opening the turn as
    its own text; the whole turn is scored either way. When the header is not where the span
    says, the hint is spliced at the span start instead and counted as a fallback.

    Carrying every hint in one sequence would make the teacher score a later turn from a
    state it could not reach: it would see its own earlier advice followed by the student
    ignoring it.

    With ``contrast_ids`` (one tensor for every hinted turn, or one per hinted turn), one more
    sub-row per hinted turn follows them: the same sub-row with those tokens in place of the hint.
    With ``anchor`` too, a third sub-row per hinted turn follows those: the same turn with no note
    spliced at all.

    The prompt is left-truncated to ``max_prefix_len``. Returns the concatenation, the packed
    meta (see :mod:`verl.trainer.ppo.sdpo.teacher_meta`), the number of fallback placements
    and the (start, end) spans for the distillation mask (== the meta spans).
    """
    if anchor and contrast_ids is None:
        raise ValueError("a no-note anchor sub-row only anchors a contrast: pass contrast_ids")
    base_prefix = prompt_ids if prompt_ids.shape[0] <= max_prefix_len else prompt_ids[-max_prefix_len:]
    pieces, sub_rows, fallbacks = _splice(
        base_prefix, response_ids, hinted_turns, hint_ids_list, header_ids, after_header
    )
    contrast_rows: list[SubRow] = []
    anchor_rows: list[SubRow] = []
    if contrast_ids is not None:
        per_turn = contrast_ids if isinstance(contrast_ids, list) else [contrast_ids] * len(hinted_turns)
        contrast_pieces, contrast_rows, _ = _splice(
            base_prefix, response_ids, hinted_turns, per_turn, header_ids, after_header
        )
        pieces += contrast_pieces
        if anchor:
            anchor_pieces, anchor_rows = _no_note(base_prefix, response_ids, hinted_turns)
            pieces += anchor_pieces

    return (
        torch.cat(pieces),
        pack(sub_rows, contrast_rows, anchor_rows),
        fallbacks,
        [(sub_row.start, sub_row.end) for sub_row in sub_rows],
    )


def turn_token_mask(response_len: int, spans: list[tuple[int, int]], skip: int = 0) -> torch.Tensor:
    """Per-token distillation mask: 1 on the scored spans past their first ``skip`` tokens, 0 elsewhere."""
    mask = torch.zeros(response_len, dtype=torch.float32)
    for start, end in spans:
        mask[min(start + skip, end):end] = 1.0
    return mask
