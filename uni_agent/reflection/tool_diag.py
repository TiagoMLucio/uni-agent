"""Deterministic hint at the trajectory's first failed str_replace edit; nothing else.

The editor already diagnoses a failed edit in its observation (shifted indentation, collapsed
newlines, trailing whitespace, a near-miss window, an ambiguous or no-op match). This strategy
turns that diagnosis into a class-specific hint spliced immediately before the turn that made
the mistake, so the teacher conditions on the warning at the exact position the student wrote
the failing call. One hint per trajectory, at the first failure: measurement shows the failure
rate compounds from there, and the earliest placement is the one that moves behaviour.

Every other trace ships no hints, so under turn-feedback distillation this arm trains on
exactly the traces whose first failed edit was hinted; combine with a ``fallback`` block later
to integrate with an LLM reflector once the targeted hints prove out.
"""

import re
from typing import Any, ClassVar

from uni_agent.reflection.base import AbstractReflector, BaseReflectionConfig
from uni_agent.reflection.registry import build_reflection_config, load_reflector, register_reflector
from uni_agent.tracing import register_langfuse_op, rollout_trace_op

FAIL_MARK = "No replacement was performed"
EDIT_RE = re.compile(r"str_replace_editor\s+str_replace\b")

#: diagnosis-message fragment -> hint, checked in order; texts for the four measured classes
#: are the counterfactual-gate winners verbatim, the rest follow their style
CLASS_HINTS = [
    ("collapsed your newlines", "escape",
     "Every line break inside old_str must be a real newline in the JSON string; a literal "
     "backslash-n is two characters and will never match the file."),
    ("leading space", "indent",
     "When you copy lines into old_str, reproduce the file's leading whitespace exactly as "
     "the last view shows it; do not add or remove indentation."),
    ("trailing whitespace", "trailing",
     "Do not leave trailing spaces on any line of old_str; match the file's line endings "
     "exactly."),
    ("Multiple occurrences", "multiple",
     "Your old_str matches more than one place in the file; include enough surrounding lines "
     "to make the match unique before editing."),
    ("is the same as new_str", "same",
     "Your old_str and new_str are identical, so the edit changes nothing; write new_str as "
     "the code should read after the change."),
    ("Closest match", "window",
     "Before quoting code into old_str, re-read the exact lines in the most recent view and "
     "copy them verbatim; do not quote from memory."),
]
DEFAULT_HINT = ("verbatim",
                "Before editing, view the exact region you intend to change and copy old_str "
                "verbatim from that output, including every space and line break.")


def first_failed_edit(turns: list[dict]) -> tuple[int, str, str] | None:
    """``(step, class, hint)`` for the first str_replace whose observation reports a failure."""
    for turn in turns:
        for call in turn.get("tools") or []:
            obs = call.get("observation") or ""
            if call.get("name") != "str_replace_editor" or FAIL_MARK not in obs:
                continue
            if not EDIT_RE.match(call.get("action") or ""):
                continue
            cls, hint = next(((c, h) for frag, c, h in CLASS_HINTS if frag in obs), DEFAULT_HINT)
            return turn["step"], cls, hint
    return None


class ToolDiagReflectionConfig(BaseReflectionConfig):
    #: reflection block for traces with no failed edit; None leaves them unhinted
    fallback: dict[str, Any] | None = None

    def model_post_init(self, _ctx):
        if self.fallback is not None:
            build_reflection_config(self.fallback)


@register_reflector("tool_diag")
class ToolDiagReflector(AbstractReflector):
    Config: ClassVar[type[BaseReflectionConfig]] = ToolDiagReflectionConfig

    def __init__(self, model: Any, config: BaseReflectionConfig, run_id: str = "",
                 record_path=None, identity: dict | None = None):
        super().__init__(model, config, run_id=run_id, record_path=record_path, identity=identity)
        self._fallback = (
            load_reflector(model, build_reflection_config(config.fallback),
                           run_id=run_id, record_path=record_path, identity=identity)
            if config.fallback is not None else None
        )

    @rollout_trace_op
    async def reflect_trajectory(
        self, task: str, turns: list[dict], gold: str, feedback: str, outcome: str = "", agent_patch: str = ""
    ) -> dict[int, str]:
        hit = first_failed_edit(turns)
        if hit is None:
            if self._fallback is None:
                return {}
            return await self._fallback.reflect_trajectory(
                task=task, turns=turns, gold=gold, feedback=feedback,
                outcome=outcome, agent_patch=agent_patch)
        step, cls, hint = hit
        hints = self._keep_valid({step: hint}, turns)
        # no model call happens on this path, so the routing decision is logged here or nowhere
        self._record(
            "tool_diag", step,
            [{"role": "system", "content": "(deterministic: first failed str_replace)"},
             {"role": "user", "content": f"class: {cls}"}],
            hints.get(step, ""), None, None, None)
        return hints


register_langfuse_op("ToolDiagReflector.reflect_trajectory", name="reflection", as_type="evaluator")
