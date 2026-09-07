"""The turn-hint teacher: hints only, no sibling solution, no response decode."""

from typing import Optional

import torch

from uni_agent.sdpo.hints import assistant_header_ids, hint_token_ids, select_hinted_turns
from uni_agent.sdpo.splice import build_spliced_teacher_row, turn_token_mask
from verl.trainer.ppo.sdpo.batch import TeacherBatch, TeacherInputs
from verl.trainer.ppo.sdpo.teacher import SDPOTeacher
from verl.trainer.ppo.sdpo.teacher_meta import DEGENERATE_META


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
    """Supervision is hints-only: a sample carrying reflection hints ships one spliced teacher
    sequence (each hint inserted before its turn, ``teacher_seq_meta`` mapping the scored spans
    back to the response grid) and a per-token distillation mask over those spans. Un-hinted
    samples are not trained at all: a degenerate 2-token teacher row (1-token body, ``DEGENERATE_META``)
    with a zero mask, scored only so that dp-group collectives stay in lockstep.

    ``max_prefix_len`` caps the spliced prefix, which is the student's real prompt (segment
    rows reach ~24k): the student's own prompt budget, not the reprompt one. The keyword
    options are verl's ``trainer/config/sdpo_teacher/turn_hints.yaml``: the two hint templates
    (``{hint}`` is the only placeholder), ``chat_template_kwargs`` (the rollout's
    ``apply_chat_template`` kwargs, so the header and hint fragments match the rollout tokens;
    the trainer's ``apply_chat_template_kwargs`` is the dataset's and is not used here),
    ``max_hinted_turns`` (keeps the first ones; None hints every turn the reflector wrote for)
    and ``call_loss_weight`` (lambda in ``L = L_turn + lambda * L_call``, a row weight because a
    within-row scale would cancel in the token-mean).
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
        call_hint_template: str,
        chat_template_kwargs: Optional[dict] = None,
        max_hinted_turns: Optional[int] = None,
        call_loss_weight: float = 1.0,
    ):
        super().__init__(
            tokenizer,
            max_prefix_len=max_prefix_len,
            apply_chat_template_kwargs=apply_chat_template_kwargs,
            success_reward_threshold=success_reward_threshold,
        )
        _validate_hint_template("turn_hint_template", turn_hint_template)
        _validate_hint_template("call_hint_template", call_hint_template)
        if call_loss_weight < 0:
            raise ValueError(f"call_loss_weight must be >= 0, got {call_loss_weight}")
        self.turn_hint_template = turn_hint_template
        self.call_hint_template = call_hint_template
        self.template_kwargs = dict(chat_template_kwargs or {})
        self.max_hinted_turns = max_hinted_turns
        self.call_loss_weight = float(call_loss_weight)
        self.header_ids = torch.tensor(
            assistant_header_ids(tokenizer, template_kwargs=self.template_kwargs), dtype=torch.int64
        )
        # call-placed splices close the assistant turn and reopen it after the hint; the
        # call span starts at the template's tool-call opening token
        self.close_ids = torch.tensor(
            tokenizer.encode(tokenizer.eos_token + "\n", add_special_tokens=False), dtype=torch.int64
        )
        self.call_open_ids = torch.tensor(tokenizer.encode("<tool_call>", add_special_tokens=False), dtype=torch.int64)

    def hint_ids(self, hint) -> torch.Tensor:
        return hint_token_ids(
            self.tokenizer, hint, self.turn_hint_template, self.call_hint_template, self.template_kwargs
        )

    def build(self, inputs: TeacherInputs) -> TeacherBatch:
        from verl.utils.debug_breakpoints import should_break

        hinted_per_row = [
            select_hinted_turns(extra_fields, response.shape[0], self.max_hinted_turns)
            for extra_fields, response in zip(inputs.extra_fields, inputs.responses, strict=True)
        ]
        teacher_seqs, seq_meta, mask_rows, loss_mask_rows = [], [], [], []
        hint_fallbacks = 0
        for prompt_ids, response_ids, response_mask, hinted in zip(
            inputs.prompts, inputs.responses, inputs.response_mask, hinted_per_row, strict=True
        ):
            if hinted:
                if should_break("teacher_build_row"): breakpoint()  # noqa: E701
                seq, meta, fallbacks, spans = build_spliced_teacher_row(
                    prompt_ids,
                    response_ids,
                    hinted,
                    [self.hint_ids(hint) for hint in hinted],
                    self.max_prefix_len,
                    self.header_ids,
                    close_ids=self.close_ids,
                    call_open_ids=self.call_open_ids,
                )
                hint_fallbacks += fallbacks
                mask_row = turn_token_mask(response_ids.shape[0], spans)
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
            "self_distillation/call_loss_weight": self.call_loss_weight,
        }
        return TeacherBatch(fields=fields, metrics=metrics, hinted_per_row=hinted_per_row)
