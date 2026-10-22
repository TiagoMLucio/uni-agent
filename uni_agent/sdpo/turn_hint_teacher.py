"""The turn-hint teacher: hints only, no sibling solution, no response decode."""

from dataclasses import dataclass
from typing import Optional

import torch

from uni_agent.sdpo.hints import HintedTurn, assistant_header_ids, hint_user_turn_ids, select_hinted_turns
from uni_agent.sdpo.metrics import hint_metrics
from uni_agent.sdpo.splice import build_spliced_teacher_row, turn_token_mask
from verl.trainer.ppo.sdpo.batch import TeacherBatch, TeacherInputs
from verl.trainer.ppo.sdpo.teacher import SDPOTeacher
from verl.trainer.ppo.sdpo.teacher_meta import DEGENERATE_META


@dataclass(kw_only=True)
class TurnHintBatch(TeacherBatch):
    """A :class:`TeacherBatch` that also carries the hints spliced per row."""

    hinted_per_row: list[list[HintedTurn]]


_VOICES = ("user", "thought")
_CONTRAST_ANCHORS = ("student", "teacher")
_CONTRAST_SOURCES = ("text", "other_row")


def _validate_hint_template(name: str, template: str) -> None:
    try:
        template.format(hint="")
    except (KeyError, IndexError) as exc:
        raise ValueError(f"{name} must format on {{hint}} alone, got {template!r}") from exc
    if "{hint}" not in template:
        raise ValueError(
            f"{name} must contain the {{hint}} placeholder or the hint is silently dropped, got {template!r}"
        )


class TurnHintTeacher(SDPOTeacher):
    """Supervision is hints-only: a sample carrying reflection hints ships its trajectory once with
    every hint as a note (``teacher_seq_meta`` placing each before its turn and mapping the scored
    spans back to the response grid) and a per-token distillation mask over those spans. Un-hinted
    samples are not trained at all: a degenerate 2-token teacher row (``DEGENERATE_META``) with a
    zero mask, scored only so that dp-group collectives stay in lockstep.

    ``max_prefix_len`` caps the spliced prefix, which is the student's real prompt (segment
    rows reach ~24k): the student's own prompt budget, not the reprompt one. The keyword
    options are ``uni_agent/conf/sdpo_teacher/turn_hints.yaml``: ``turn_hint_template``
    (``{hint}`` is the only placeholder) and ``chat_template_kwargs`` (the rollout's
    ``apply_chat_template`` kwargs, so the header and hint fragments match the rollout tokens;
    the trainer's ``apply_chat_template_kwargs`` is the dataset's and is not used here),
    ``skip_first_tokens`` (the start of each hinted turn left out of the distillation mask) and
    ``turn_hint_voice`` (``user``: the hint is a user turn before the hinted turn; ``thought``: it
    opens the hinted turn after its header, as the agent's own words). Every hint the reflector
    shipped is spliced; its ``max_selected_turns`` is the one cap.

    ``contrast_hint`` (a text spliced through the same template and voice) adds a contrast teacher:
    every hinted turn ships a second sub-row with it in place of the hint, and the loss moves the
    student by the difference of the two teachers instead of to the hint teacher. With
    ``contrast_source: other_row`` the contrast is instead the hint of another task's hinted row in
    the batch (``contrast_hint`` when the batch has none), and with ``contrast_rows`` above 1 the
    hints of that many distinct other tasks, one contrast teacher each, whose probabilities the
    target averages (fewer when the batch has fewer). ``contrast_anchor: teacher`` adds that
    difference to the teacher's own distribution on the turn with no note (a third sub-row per hinted
    turn) rather than to the student's, so the target stays put while the student moves.
    """

    needs_prompts = True

    def __init__(
        self,
        tokenizer,
        *,
        max_prefix_len: int,
        apply_chat_template_kwargs=None,
        success_reward_threshold: Optional[float] = None,
        turn_hint_template: str,
        chat_template_kwargs: Optional[dict] = None,
        skip_first_tokens: int = 0,
        turn_hint_voice: str = "user",
        contrast_hint: Optional[str] = None,
        contrast_anchor: str = "student",
        contrast_source: str = "text",
        contrast_rows: int = 1,
    ):
        super().__init__(
            tokenizer,
            max_prefix_len=max_prefix_len,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
            success_reward_threshold=success_reward_threshold,
        )
        _validate_hint_template("turn_hint_template", turn_hint_template)
        if isinstance(skip_first_tokens, bool) or not isinstance(skip_first_tokens, int) or skip_first_tokens < 0:
            raise ValueError(f"skip_first_tokens must be a non-negative int, got {skip_first_tokens!r}")
        if turn_hint_voice not in _VOICES:
            raise ValueError(f"turn_hint_voice must be one of {_VOICES}, got {turn_hint_voice!r}")
        if contrast_hint is not None and (not isinstance(contrast_hint, str) or not contrast_hint.strip()):
            raise ValueError(f"contrast_hint must be null or a non-empty string, got {contrast_hint!r}")
        if contrast_anchor not in _CONTRAST_ANCHORS:
            raise ValueError(f"contrast_anchor must be one of {_CONTRAST_ANCHORS}, got {contrast_anchor!r}")
        if contrast_source not in _CONTRAST_SOURCES:
            raise ValueError(f"contrast_source must be one of {_CONTRAST_SOURCES}, got {contrast_source!r}")
        if contrast_hint is None and (contrast_anchor != "student" or contrast_source != "text"):
            raise ValueError(
                "contrast_anchor and contrast_source shape a contrast teacher: set contrast_hint "
                "(the contrast text, and other_row's fallback when the batch has no other task's hint)"
            )
        if isinstance(contrast_rows, bool) or not isinstance(contrast_rows, int) or contrast_rows < 1:
            raise ValueError(f"contrast_rows must be a positive int, got {contrast_rows!r}")
        if contrast_rows > 1 and contrast_source != "other_row":
            raise ValueError("contrast_rows above 1 takes other tasks' hints: set contrast_source: other_row")
        self.skip_first_tokens = skip_first_tokens
        self.turn_hint_voice = turn_hint_voice
        self.turn_hint_template = turn_hint_template
        self.template_kwargs = dict(chat_template_kwargs or {})
        self.header_ids = torch.tensor(
            assistant_header_ids(tokenizer, template_kwargs=self.template_kwargs), dtype=torch.int64
        )
        self.contrasts = contrast_hint is not None
        self.contrast_ids = self._text_ids(contrast_hint) if self.contrasts else None
        self.anchors = self.contrasts and contrast_anchor == "teacher"
        self.contrast_source = contrast_source
        self.contrast_rows = contrast_rows

    def hint_ids(self, hint: HintedTurn) -> torch.Tensor:
        return self._text_ids(hint.text)

    def _other_row_contrasts(self, hinted_per_row: list[list[HintedTurn]], uids: list) -> tuple[list, int]:
        """Per row, its contrast groups, one per donor: the next hinted rows (in batch order, wrapping) of up
        to ``contrast_rows`` distinct tasks other than its own, the k-th turn taking a donor's k-th hint
        (cycling); the fixed contrast text when no other task's row is hinted. Returns them and the number
        of fallbacks."""
        n_rows = len(hinted_per_row)
        per_row, fallbacks = [], 0
        for i, hinted in enumerate(hinted_per_row):
            if not hinted:
                per_row.append(None)
                continue
            donors, tasks = [], {uids[i]}
            for j in ((i + step) % n_rows for step in range(1, n_rows)):
                if len(donors) == self.contrast_rows:
                    break
                if hinted_per_row[j] and uids[j] not in tasks:
                    donors.append(hinted_per_row[j])
                    tasks.add(uids[j])
            if not donors:
                fallbacks += len(hinted)
                per_row.append([self.contrast_ids])
            else:
                per_row.append(
                    [[self._text_ids(other[k % len(other)].text) for k in range(len(hinted))] for other in donors]
                )
        return per_row, fallbacks

    def _text_ids(self, hint_text: str) -> torch.Tensor:
        text = self.turn_hint_template.format(hint=hint_text)
        if self.turn_hint_voice == "thought":
            # its own paragraph, a blank line before what the agent wrote
            ids = self.tokenizer.encode(text.strip() + "\n\n", add_special_tokens=False)
        else:
            ids = hint_user_turn_ids(self.tokenizer, text, self.template_kwargs)
        return torch.tensor(ids, dtype=torch.int64)

    def build(self, inputs: TeacherInputs) -> TurnHintBatch:
        from verl.utils.debug_breakpoints import should_break

        hinted_per_row = [
            select_hinted_turns(extra_fields, response.shape[0])
            for extra_fields, response in zip(inputs.extra_fields, inputs.responses, strict=True)
        ]
        teacher_seqs, seq_meta, mask_rows, loss_mask_rows = [], [], [], []
        hint_fallbacks = 0
        prefix_clips = 0
        contrast_per_row = [[self.contrast_ids] if self.contrasts else None] * len(hinted_per_row)
        contrast_fallbacks = 0
        if self.contrasts and self.contrast_source == "other_row":
            contrast_per_row, contrast_fallbacks = self._other_row_contrasts(hinted_per_row, list(inputs.uids))
        for prompt_ids, response_ids, response_mask, hinted, contrast_groups in zip(
            inputs.prompts, inputs.responses, inputs.response_mask, hinted_per_row, contrast_per_row, strict=True
        ):
            if hinted:
                if should_break("teacher_build_row"): breakpoint()  # noqa: E701
                # a clipped prefix is a teacher conditioned on less than the student wrote from
                prefix_clips += int(prompt_ids.shape[0] > self.max_prefix_len)
                seq, meta, fallbacks, spans = build_spliced_teacher_row(
                    prompt_ids,
                    response_ids,
                    hinted,
                    [self.hint_ids(hint) for hint in hinted],
                    self.max_prefix_len,
                    self.header_ids,
                    after_header=self.turn_hint_voice == "thought",
                    contrast_groups=contrast_groups,
                    anchor=self.anchors,
                )
                hint_fallbacks += fallbacks
                mask_row = turn_token_mask(response_ids.shape[0], spans, self.skip_first_tokens)
            else:
                seq = torch.cat([prompt_ids[-1:], response_ids[:1]])
                meta = DEGENERATE_META
                mask_row = torch.zeros(response_ids.shape[0], dtype=torch.float32)
            teacher_seqs.append(seq)
            seq_meta.append(torch.tensor(meta, dtype=torch.int64))
            mask_rows.append(mask_row)
            loss_mask_rows.append(response_mask * mask_row.to(response_mask.dtype))

        fields = {
            "teacher_input_ids": torch.nested.nested_tensor(teacher_seqs, layout=torch.jagged),
            "teacher_seq_meta": torch.nested.nested_tensor(seq_meta, layout=torch.jagged),
            "self_distillation_mask": torch.nested.nested_tensor(mask_rows, layout=torch.jagged),
            "loss_mask": torch.nested.nested_tensor(loss_mask_rows, layout=torch.jagged),
        }
        num_hinted = sum(1 for hinted in hinted_per_row if hinted)
        metrics = {
            "self_distillation/hinted_sample_fraction": num_hinted / len(inputs),
            "self_distillation/hinted_turns_per_sample": (
                sum(len(hinted) for hinted in hinted_per_row) / num_hinted if num_hinted else 0.0
            ),
            "self_distillation/hint_injection_fallbacks": hint_fallbacks,
            "self_distillation/teacher_prefix_clips": prefix_clips,
        }
        if self.contrasts and self.contrast_source == "other_row":
            metrics["self_distillation/contrast_source_fallbacks"] = contrast_fallbacks
            metrics["self_distillation/contrast_teachers_per_hinted_row"] = (
                sum(len(groups) for groups in contrast_per_row if groups) / num_hinted if num_hinted else 0.0
            )
        return TurnHintBatch(fields=fields, metrics=metrics, hinted_per_row=hinted_per_row)

    def trajectory_metrics(
        self, batch: TurnHintBatch, inputs: TeacherInputs, supervised_per_row: list[float], weights: list[float]
    ) -> dict:
        return hint_metrics(batch.hinted_per_row, inputs.extra_fields, inputs.traj_of_row)
