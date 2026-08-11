"""Several calls per rollout: one drafts hints, later ones select turns or repair each hint.

A call sees exactly the fields its ``user`` template names, so a stage is kept from the reference
patch by construction rather than by an instruction it can disregard: a repair stage naming only
``{task} {prefix} {turn} {hint}`` cannot reach the patch at all. That is what a single call cannot
express, since ``{prefix}`` is the trajectory truncated to the hinted turn.
"""

import asyncio
import json
import re
from collections import Counter
from string import Formatter
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from uni_agent.reflection.base import FINAL_MARKER, AbstractReflector, BaseReflectionConfig
from uni_agent.reflection.facts import turn_candidates
from uni_agent.reflection.registry import register_reflector
from uni_agent.tracing import register_langfuse_op, rollout_trace_op

TURNS_MARKER = "FINAL_TURNS_JSON:"

TRACE_FIELDS = frozenset({"task", "outcome", "gold", "agent_patch", "feedback", "turns", "candidates"})
TURN_FIELDS = frozenset({"prefix", "turn", "hint"})
#: the previous call's reply, so a stage can react to one without being handed the trajectory
PREV_FIELD = "prev"

_JSON_DECODER = json.JSONDecoder()
_DROP = {"DROP", "NONE", "REMOVE"}


def _fields(template: str) -> set[str]:
    return {name for _, name, _, _ in Formatter().parse(template) if name} - {"k"}


def _words(text: str) -> list[str]:
    """Words reduced to letters and digits, so a deletion may reflow punctuation at the seam."""
    return [w for w in (re.sub(r"[^a-z0-9]+", "", t.lower()) for t in (text or "").split()) if w]


def is_deletion_of(candidate: str, original: str) -> bool:
    """Whether ``candidate`` is ``original`` with whole words removed and nothing added."""
    it = iter(_words(original))
    return all(any(w == o for o in it) for w in _words(candidate))


class CallSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    system: str
    user: str
    per: Literal["trace", "turn"] = "trace"
    parse: Literal["text", "turns", "hints"] = "text"
    #: ``delete_only`` refuses a reply that introduces a word the draft did not have. The stage
    #: sees the draft, which was written with the patch, so an unconstrained rewrite can restate
    #: privileged content the stage itself never saw.
    edit: Literal["free", "delete_only"] = "free"
    max_output_tokens: int | None = None


class PipelineReflectionConfig(BaseReflectionConfig):
    calls: list[CallSpec]

    @model_validator(mode="after")
    def _check_calls(self):
        if not self.calls:
            raise ValueError("pipeline reflector needs at least one call")
        if self.calls[-1].parse != "hints":
            raise ValueError("the last call must have parse=hints, or the pipeline yields none")
        for index, call in enumerate(self.calls):
            allowed = set(TRACE_FIELDS)
            if index:
                allowed.add(PREV_FIELD)
            if call.per == "turn":
                allowed |= TURN_FIELDS
            unknown = _fields(call.user) - allowed
            if unknown:
                raise ValueError(f"call {call.id!r} references fields it cannot be given: {sorted(unknown)}")
            if call.per == "turn" and call.parse == "turns":
                raise ValueError(f"call {call.id!r}: a per-turn call cannot select the turns")
            if call.edit == "delete_only" and call.parse != "hints":
                raise ValueError(f"call {call.id!r}: edit=delete_only only applies to parse=hints")
        return self


@register_reflector("pipeline")
class PipelineReflector(AbstractReflector):
    Config: ClassVar[type[BaseReflectionConfig]] = PipelineReflectionConfig

    @rollout_trace_op
    async def reflect_trajectory(
        self, task: str, turns: list[dict], gold: str, feedback: str, outcome: str = "", agent_patch: str = ""
    ) -> dict[int, str]:
        cfg = self.config
        k = str(cfg.max_selected_turns)
        gold = self._clip(gold, cfg.max_patch_chars) if gold else gold
        agent_patch = self._clip(agent_patch, cfg.max_patch_chars) if agent_patch else agent_patch
        base: dict[str, Any] = {
            "task": task,
            "outcome": outcome or "(not available)",
            "agent_patch": agent_patch if cfg.include_agent_patch and agent_patch else "(not available)",
            "gold": gold if cfg.include_gold and gold else "(not available)",
            "feedback": feedback if cfg.include_exec_feedback and feedback else "(not available)",
            "candidates": turn_candidates(turns),
        }
        hints: dict[int, str] = {}
        selected: list[int] = []
        prev = ""

        for call in cfg.calls:
            if call.per == "turn":
                if not selected:
                    break
                hints = await self._run_per_turn(call, selected, turns, base, prev, k, hints)
                selected = sorted(hints)
                continue

            text = await self._ask(
                call.system.replace("{k}", k),
                lambda obs, resp, c=call, p=prev: self._render(c.user, k, base, turns, obs, resp, prev=p),
                call.max_output_tokens,
            )
            if text is None:
                return {}
            if call.parse == "turns":
                selected = [step for step in self._parse_turns(text) if step in {t["step"] for t in turns}]
                if not selected:
                    return {}
            elif call.parse == "hints":
                hints = self._keep_valid(self._parse(text), turns)
                selected = sorted(hints)
                if not hints:
                    return {}
            prev = text
        return hints

    async def _run_per_turn(self, call, selected, turns, base, prev, k, hints) -> dict[int, str]:
        replies = await asyncio.gather(
            *(self._ask(
                call.system.replace("{k}", k),
                lambda obs, resp, s=step: self._render(
                    call.user, k, base, turns, obs, resp, prev=prev, step=s, hint=hints.get(s, "")
                ),
                call.max_output_tokens,
            ) for step in selected)
        )
        if call.parse != "hints":
            return hints
        out, tally = dict(hints), Counter()
        for step, reply in zip(selected, replies, strict=True):
            # a failed call leaves nothing to judge the hint by, so it goes; a rewrite leaves the
            # drafted hint standing, which is the worst case this stage can produce
            if reply is None:
                out.pop(step, None)
                tally["failed"] += 1
                continue
            text, outcome = self._adopt(reply, hints.get(step, ""), call.edit)
            tally[outcome] += 1
            if text:
                out[step] = self._clip_diagnosis(text)
            else:
                out.pop(step, None)
        # "unchanged" is the stage saying there is nothing to cut; "rewrote" is the guard firing.
        # Reporting them together once hid a run where every reply was a rewrite.
        summary = ", ".join(f"{n} {k}" for k, n in sorted(tally.items()))
        self.logger.info(f"Reflection {call.id}: {summary}")
        return out

    def _render(self, template, k, base, turns, obs_cap, resp_cap, prev="", step=None, hint="") -> str:
        wanted = _fields(template)
        values = {name: base[name] for name in wanted if name in base}
        if "turns" in wanted:
            values["turns"] = self._render_turns(turns, obs_cap, resp_cap)
        if "prefix" in wanted:
            before = [turn for turn in turns if turn["step"] < step]
            values["prefix"] = self._render_turns(before, obs_cap, resp_cap) or "(nothing yet)"
        if "turn" in wanted:
            values["turn"] = str(step)
        if "hint" in wanted:
            values["hint"] = hint
        if PREV_FIELD in wanted:
            values[PREV_FIELD] = prev
        return template.replace("{k}", k).format(**values)

    def _adopt(self, raw: str, original: str, edit: str) -> tuple[str | None, str]:
        """The stage's hint for one turn, with what it did: the text, or ``original`` when it
        declines to edit, or None to drop.

        Without the marker the reply is taken as its last line: a stage that writes its audit and
        forgets the marker would otherwise ship the whole audit as the hint.
        """
        if FINAL_MARKER in raw:
            text = raw.rsplit(FINAL_MARKER, 1)[1]
        else:
            text = next((line for line in reversed((raw or "").splitlines()) if line.strip()), "")
        text = " ".join(text.split()).strip().strip('"')
        if not text:
            return (original or None), "empty"
        if text.upper() in _DROP:
            return None, "dropped"
        if text == " ".join(original.split()):
            return original, "unchanged"
        if edit == "delete_only" and not is_deletion_of(text, original):
            # the lab sees no rewrites at all on this prompt, so when training sees nothing but,
            # the reply itself is the only evidence of what the stage is really being asked
            self.logger.info(f"Reflection rewrote, draft kept | draft={original[:160]!r} "
                             f"| reply={text[:220]!r}")
            return (original or None), "rewrote"
        return text, "edited"

    @staticmethod
    def _parse_turns(text: str) -> list[int]:
        tail = text.rsplit(TURNS_MARKER, 1)[1] if TURNS_MARKER in text else text
        index = tail.find("{")
        while index != -1:
            try:
                obj = _JSON_DECODER.raw_decode(tail, index)[0]
            except json.JSONDecodeError:
                index = tail.find("{", index + 1)
                continue
            if isinstance(obj, dict) and isinstance(obj.get("turns"), list):
                return [int(x) for x in obj["turns"] if str(x).lstrip("-").isdigit()]
            index = tail.find("{", index + 1)
        found = re.findall(r"\[([\d,\s]*)\]", tail)
        return [int(x) for x in re.findall(r"\d+", found[-1])] if found else []


register_langfuse_op("PipelineReflector.reflect_trajectory", name="reflection", as_type="evaluator")
