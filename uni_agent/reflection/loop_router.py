"""Deterministic routing for loop-pattern trajectories; every other trace falls through.

A loop is the same tool call issued ``min_repeats`` times over the trajectory, matched on the
rendered call text like the verbatim-edit check in ``facts``. Counterfactual measurement showed
the repeat attractor locks in after one failed attempt, so the one placement that helps is
before the loop's *first* occurrence; a fixed take-stock hint there beats every later or more
directive intervention. The hint is deterministic and carries no information about later turns,
so unlike an LLM-written hint it cannot leak the outcome.

``route`` decides what a loop-pattern trace gets: ``hint`` splices the fixed text at the loop's
first turn, ``drop`` returns no hints at all, which under turn-feedback distillation zero-masks
the trace out of training. Non-loop traces are delegated unchanged to the ``fallback`` reflector
block, so arms sharing a fallback differ only in how loop traces are treated.
"""

from typing import Any, ClassVar, Literal

from pydantic import model_validator

from uni_agent.reflection.base import AbstractReflector, BaseReflectionConfig
from uni_agent.reflection.registry import build_reflection_config, load_reflector, register_reflector
from uni_agent.tracing import register_langfuse_op, rollout_trace_op

REORIENT_HINT = (
    "Pause and take stock: list in your own words what you have already confirmed "
    "from the observations above, and what single question remains open. Then take "
    "the one action that answers that question."
)

#: matching the verbatim-edit dedup in ``facts``: whitespace-insensitive, and 300 chars of the
#: rendered call separate distinct edits while ignoring drift deep inside a long argument
SIGNATURE_CHARS = 300


def call_signature(call: dict) -> str:
    return " ".join(f"{call.get('name') or ''} {call.get('action') or ''}".split())[:SIGNATURE_CHARS]


def first_loop(turns: list[dict], min_repeats: int) -> tuple[int, str] | None:
    """``(step of the first occurrence, signature)`` of the earliest call to repeat
    ``min_repeats`` times, or ``None``. Scanned in turn order, so the loop that completes
    first wins even when another signature started earlier."""
    seen: dict[str, list[int]] = {}
    for turn in turns:
        for call in turn.get("tools") or []:
            occurrences = seen.setdefault(call_signature(call), [])
            occurrences.append(turn["step"])
            if len(occurrences) == min_repeats:
                return occurrences[0], call_signature(call)
    return None


class LoopRouterReflectionConfig(BaseReflectionConfig):
    #: identical calls that make a loop; 3 is the definition the placement was measured under
    min_repeats: int = 3
    route: Literal["hint", "drop"] = "hint"
    hint: str = REORIENT_HINT
    #: full ``reflection`` block for non-loop traces (its ``enabled``/``failed_only`` are moot:
    #: the loop already gated on this config before routing); None leaves them unhinted
    fallback: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _check(self):
        if self.min_repeats < 2:
            raise ValueError("min_repeats < 2 would route every trace as a loop")
        if self.fallback is not None:
            build_reflection_config(self.fallback)
        return self


@register_reflector("loop_router")
class LoopRouterReflector(AbstractReflector):
    Config: ClassVar[type[BaseReflectionConfig]] = LoopRouterReflectionConfig

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
        cfg = self.config
        hit = first_loop(turns, cfg.min_repeats)
        if hit is None:
            if self._fallback is None:
                return {}
            return await self._fallback.reflect_trajectory(
                task=task, turns=turns, gold=gold, feedback=feedback,
                outcome=outcome, agent_patch=agent_patch)
        step, signature = hit
        hints = {} if cfg.route == "drop" else self._keep_valid({step: cfg.hint}, turns)
        # no model call happens on this path, so the routing decision is logged here or nowhere
        self._record(
            "loop_router", step,
            [{"role": "system", "content": f"(deterministic: route={cfg.route}, min_repeats={cfg.min_repeats})"},
             {"role": "user", "content": f"loop signature: {signature}"}],
            hints.get(step, ""), None, None, None)
        return hints


register_langfuse_op("LoopRouterReflector.reflect_trajectory", name="reflection", as_type="evaluator")
