"""The spliced teacher row: one sub-row per hint, its meta packed by
:mod:`verl.trainer.ppo.sdpo.teacher_meta`, plus the per-token mask."""

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


def build_spliced_teacher_row(
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    hinted_turns: list[HintedTurn],
    hint_ids_list: list[torch.Tensor],
    max_prefix_len: int,
    header_ids: torch.Tensor,
) -> tuple[torch.Tensor, list[int], int, list[tuple[int, int]]]:
    """One teacher sub-row per hinted turn, concatenated into a single row.

    Each sub-row is the trajectory up to its own turn with only its own hint spliced in,
    and truncated after the scored span. The hint goes immediately before the turn's
    assistant header and the whole turn is scored; when the header is not where the span
    says, the hint is spliced at the span start instead and counted as a fallback.

    Carrying every hint in one sequence would make the teacher score a later turn from a
    state it could not reach: it would see its own earlier advice followed by the student
    ignoring it.

    The prompt is left-truncated to ``max_prefix_len``. Returns the concatenation, the packed
    meta (see :mod:`verl.trainer.ppo.sdpo.teacher_meta`), the number of fallback placements
    and the (start, end) spans for the distillation mask (== the meta spans).
    """
    base_prefix = prompt_ids if prompt_ids.shape[0] <= max_prefix_len else prompt_ids[-max_prefix_len:]
    header = header_ids.shape[0]
    pieces: list[torch.Tensor] = []
    sub_rows: list[SubRow] = []
    fallbacks = 0

    for hint, hint_ids in zip(hinted_turns, hint_ids_list, strict=True):
        start, end = hint.start, hint.end
        hint_ids = hint_ids.to(response_ids.dtype)
        prefix = base_prefix
        if start >= header and torch.equal(response_ids[start - header : start], header_ids):
            insert_at = start - header
        elif start == 0 and prefix.shape[0] >= header and torch.equal(prefix[-header:], header_ids):
            # first turn: its assistant header is the prompt tail, so the hint joins the prefix
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
                response_ids[insert_at:start],  # the turn's assistant header
                response_ids[start:end],  # the span the teacher scores
            ]
            row, sub_row = _sub_row(prefix, body, scored=3, span=(start, end))

        pieces.append(row)
        sub_rows.append(sub_row)

    return torch.cat(pieces), pack(sub_rows), fallbacks, [(sub_row.start, sub_row.end) for sub_row in sub_rows]


def turn_token_mask(response_len: int, spans: list[tuple[int, int]]) -> torch.Tensor:
    """Per-token distillation mask: 1 on the scored spans, 0 elsewhere."""
    mask = torch.zeros(response_len, dtype=torch.float32)
    for start, end in spans:
        mask[start:end] = 1.0
    return mask
