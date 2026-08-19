"""Exact-call hint at the trajectory's first failed str_replace, spliced mid-turn.

The editor's own diagnosis makes the corrected call derivable: shifted indentation,
collapsed newlines and trailing whitespace are inverted mechanically from the message,
and a near-miss window is recovered from the file text the model previously viewed,
cross-checked against the line fragments the diagnosis quotes. The hint is the corrected
call verbatim, and it carries an ``at: call`` marker so the trainer splices it between
the turn's reasoning and its tool call and masks only the call tokens: measured against
alternatives, that construction flips the teacher's top-1 to the corrected token in ~92%
of failures, while defect descriptions and raw corrected strings move far less.

Corrections are trusted only above ``min_sim`` (the editor's own similarity ratio), and a
window recovered from the view blob must also match the diagnosis's quoted fragments.
Traces without a confident correction fall through to ``fallback`` (or ship no hints).
"""

import difflib
import json
import re
from typing import Any, ClassVar

from uni_agent.reflection.base import AbstractReflector, BaseReflectionConfig
from uni_agent.reflection.registry import build_reflection_config, load_reflector, register_reflector
from uni_agent.tracing import register_langfuse_op, rollout_trace_op

FAIL_MARK = "No replacement was performed"
EDIT_RE = re.compile(r"str_replace_editor\s+str_replace\b")
GUTTER = re.compile(r"^\s{0,6}\d+\t?", re.M)
INDENT_RE = re.compile(r"(\d+) (?:extra|missing) leading space")
SIM_RE = re.compile(r"\((\d+)% similar\)")
DIFF_LINE_RE = re.compile(r"line \d+: file has `(.*)`, your old_str has `(.*)`")

DEFAULT_HINT_TEMPLATE = ("The edit you are about to make must be exactly this call, "
                         "character for character:\n{corrected_call}")


def _transform(cls: str, text: str, diag: str) -> str | None:
    if cls == "indent":
        m = INDENT_RE.search(diag)
        if not m:
            return None
        k = int(m.group(1))
        return "\n".join(
            (line[k:] if "extra" in m.group(0) else " " * k + line) if line.strip() else line
            for line in text.splitlines())
    if cls == "escape":
        return text.replace("\\n", "\n")
    if cls == "trailing":
        return "\n".join(line.rstrip() for line in text.splitlines())
    return None


def _window(old: str, blob: str, diag: str) -> str | None:
    """The blob window closest to old_str, kept only when the diagnosis's quoted
    fragments all appear in it (the cross-check that makes a guess trustworthy)."""
    old_lines = old.split("\n")
    blob_lines = blob.split("\n")
    n = len(old_lines)
    if not blob_lines or n > len(blob_lines) or n > 60:
        return None
    sm = difflib.SequenceMatcher(None, autojunk=False)
    sm.set_seq2(old)
    best, best_r = None, 0.0
    for s in range(len(blob_lines) - n + 1):
        w = "\n".join(blob_lines[s: s + n])
        sm.set_seq1(w)
        if sm.real_quick_ratio() <= best_r or sm.quick_ratio() <= best_r:
            continue
        r = sm.ratio()
        if r > best_r:
            best, best_r = w, r
    if best is None or best_r < 0.7 or best == old:
        return None
    frags = [f for f, _ in DIFF_LINE_RE.findall(diag)]
    stripped = [line.strip() for line in best.split("\n")]
    if not frags or not all(any(line.startswith(f) for line in stripped) for f in frags):
        return None
    return best


def _classify(diag: str) -> str:
    for cls, frag in (("escape", "collapsed your newlines"), ("indent", "leading space"),
                      ("trailing", "trailing whitespace"), ("window", "Closest match")):
        if frag in diag:
            return cls
    return "other"


def _call_blocks(response: str):
    """(raw block, parsed arguments) for each str_replace call in the assistant text."""
    out = []
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response, re.S):
        try:
            d = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        args = d.get("arguments") or {}
        if d.get("name") == "str_replace_editor" and args.get("command") == "str_replace":
            out.append((m.group(0), args))
    return out


def corrected_call(turns: list[dict], min_sim: float) -> tuple[int, str, str] | None:
    """``(step, class, corrected call text)`` for the first confidently-fixable failed edit."""
    blob_parts: list[str] = []
    for turn in turns:
        hit = None
        for call in turn.get("tools") or []:
            obs = call.get("observation") or ""
            if (call.get("name") == "str_replace_editor" and FAIL_MARK in obs
                    and EDIT_RE.match(call.get("action") or "")):
                hit = obs
                break
        if hit is None:
            for call in turn.get("tools") or []:
                obs = call.get("observation") or ""
                if "cat -n" in obs or "\t" in obs:
                    blob_parts.append(GUTTER.sub("", obs))
            continue

        diag = hit[hit.find(FAIL_MARK):][:800]
        m = SIM_RE.search(diag)
        sim = int(m.group(1)) / 100 if m else (0.99 if "match exactly except" in diag else 0.0)
        if sim < min_sim:
            return None
        blocks = _call_blocks(turn.get("response") or "")
        if not blocks:
            return None
        raw, args = blocks[0]
        old, new = str(args.get("old_str", "")), str(args.get("new_str", ""))
        cls = _classify(diag)
        fixed = _transform(cls, old, diag)
        fixed_new = _transform(cls, new, diag) if fixed is not None else None
        if fixed is None and cls == "window":
            fixed = _window(old, "\n".join(blob_parts)[-40000:], diag)
        if fixed is None:
            return None
        fo = json.dumps(old)[1:-1]
        if fo not in raw:
            return None
        raw = raw.replace(fo, json.dumps(fixed)[1:-1])
        if fixed_new and fixed_new != new:
            fn = json.dumps(new)[1:-1]
            if fn in raw:
                raw = raw.replace(fn, json.dumps(fixed_new)[1:-1])
        return turn["step"], cls, raw
    return None


class ToolFixReflectionConfig(BaseReflectionConfig):
    #: below this editor-reported similarity the correction is a guess, not a fix
    min_sim: float = 0.8
    hint_template: str = DEFAULT_HINT_TEMPLATE
    #: reflection block for traces with no confident correction; None leaves them unhinted
    fallback: dict[str, Any] | None = None

    def model_post_init(self, _ctx):
        if "{corrected_call}" not in self.hint_template:
            raise ValueError("tool_fix hint_template must format on {corrected_call}")
        if self.fallback is not None:
            build_reflection_config(self.fallback)


@register_reflector("tool_fix")
class ToolFixReflector(AbstractReflector):
    Config: ClassVar[type[BaseReflectionConfig]] = ToolFixReflectionConfig

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
    ) -> dict[int, Any]:
        hit = corrected_call(turns, self.config.min_sim)
        if hit is None:
            if self._fallback is None:
                return {}
            return await self._fallback.reflect_trajectory(
                task=task, turns=turns, gold=gold, feedback=feedback,
                outcome=outcome, agent_patch=agent_patch)
        step, cls, call = hit
        text = self.config.hint_template.format(corrected_call=call)
        # `at: call` routes the trainer to the mid-turn splice (between reasoning and call)
        # with the distillation mask on the call tokens alone; never clipped, since a cut
        # corrected call teaches a truncation
        hints = {step: {"text": text, "at": "call"}} if step in {t["step"] for t in turns} else {}
        self._record(
            "tool_fix", step,
            [{"role": "system", "content": f"(deterministic: corrected call, min_sim={self.config.min_sim})"},
             {"role": "user", "content": f"class: {cls}"}],
            text if hints else "", None, None, None)
        return hints


register_langfuse_op("ToolFixReflector.reflect_trajectory", name="reflection", as_type="evaluator")
