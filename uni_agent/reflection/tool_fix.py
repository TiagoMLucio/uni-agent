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


OBS_REGION_MARK = "The matching region of the file reads exactly:\n"


def _obs_region(obs: str) -> str | None:
    """The editor-verified matching region printed into the failure observation (present
    when the sandbox runs with STR_REPLACE_DID_YOU_MEAN): file-true at failure time, so it
    beats any reconstruction."""
    if OBS_REGION_MARK not in obs:
        return None
    region = obs.split(OBS_REGION_MARK, 1)[1]
    region = region.split("\nResend the call", 1)[0]
    return region.strip("\n") or None


def _call_path(raw: str) -> str | None:
    """The edited path of a rendered ``<tool_call>`` block."""
    m = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", raw, re.S)
    if not m:
        return None
    try:
        return str((json.loads(m.group(1)).get("arguments") or {}).get("path", "")) or None
    except json.JSONDecodeError:
        return None


def retry_step(turns: list[dict], after_step: int, path: str | None) -> int | None:
    """The step of the retry that failed again on the same file, or None.

    Only a retry that ALSO failed is a recovery target: the file is then still what the
    correction was derived from, and the model is demonstrably stuck. A retry that
    succeeded needs no teaching (and would have changed the file underneath the
    correction), and an edit that moves to another file means the model moved on.
    """
    if path is None:
        return None
    for turn in turns:
        if turn["step"] <= after_step:
            continue
        for call in turn.get("tools") or []:
            if call.get("name") != "str_replace_editor" or not EDIT_RE.match(call.get("action") or ""):
                continue
            blocks = _call_blocks(turn.get("response") or "")
            if not blocks or str(blocks[0][1].get("path", "")) != path:
                return None
            return turn["step"] if FAIL_MARK in (call.get("observation") or "") else None
    return None


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _rebase_new(old: str, new: str, verified_old: str) -> str | None:
    """Re-base the model's intended change (old -> new) onto the verified region.

    Line-count-matching fast path only: each verified line corresponds to the same-index
    old line, so intent opcodes transfer positionally and inserted/replaced lines are
    re-indented by the local old->verified indent delta. Anything else returns None and
    the caller keeps the model's new_str untouched.
    """
    import difflib

    old_lines, ver_lines = old.split("\n"), verified_old.split("\n")
    if len(old_lines) != len(ver_lines):
        return None
    deltas = [_indent_of(v) - _indent_of(o) if o.strip() and v.strip() else 0
              for o, v in zip(old_lines, ver_lines, strict=True)]

    def shift(line: str, d: int) -> str:
        if not line.strip():
            return line
        if d >= 0:
            return " " * d + line
        return line[-d:] if line.startswith(" " * -d) else line

    out: list[str] = []
    sm = difflib.SequenceMatcher(None, old_lines, new.split("\n"), autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.extend(ver_lines[i1:i2])
        elif tag == "delete":
            continue
        else:  # replace / insert: the model's own lines, re-indented to the verified offset
            d = deltas[i1] if i1 < len(deltas) else (deltas[i1 - 1] if i1 else 0)
            out.extend(shift(line, d) for line in new.split("\n")[j1:j2])
    return "\n".join(out)


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
        # the editor-printed region (did-you-mean environments) is file-verified at failure
        # time: prefer it for every class, then the mechanical transforms, then the blob
        fixed = _obs_region(hit)
        fixed_new = None
        if fixed is not None:
            fixed_new = _rebase_new(old, new, fixed)
        else:
            fixed = _transform(cls, old, diag)
            fixed_new = _transform(cls, new, diag) if fixed is not None else None
            if fixed is None and cls == "window":
                fixed = _window(old, "\n".join(blob_parts)[-40000:], diag)
        if fixed is None or fixed == old:
            return None
        # a correction that collapses old_str into new_str teaches a call the editor rejects
        # outright ("old_str is the same as new_str"): the mechanical transforms hit this
        # whenever the edit's only intended change WAS the whitespace or the escaping
        if fixed == (fixed_new if fixed_new else new):
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
    #: also ship the corrected call as `target`, so the trainer can narrow the distillation
    #: mask to the tokens it changes (actor.self_distillation.call_mask=first|all)
    target_mask: bool = False
    hint_template: str = DEFAULT_HINT_TEMPLATE
    #: where the corrected call is taught: at the failed call (`call`), at the retry that
    #: failed again (`retry`, which trains recovery from the editor's error message), or both
    hint_at: str = "call"
    #: reflection block for traces with no confident correction; None leaves them unhinted
    fallback: dict[str, Any] | None = None

    def model_post_init(self, _ctx):
        if not any(k in self.hint_template for k in ("{corrected_call}", "{corrected_old_wire}")):
            raise ValueError(
                "tool_fix hint_template must format on {corrected_call} or {corrected_old_wire}")
        if self.hint_at not in ("call", "retry", "both"):
            raise ValueError(f"tool_fix hint_at must be call|retry|both, got {self.hint_at!r}")
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
        # the wire old_str (escaped JSON value) measured as the strongest conditioning form
        m = re.search(r'"old_str":\s*("(?:[^"\\]|\\.)*")', call)
        old_wire = m.group(1) if m else json.dumps("")
        text = self.config.hint_template.format(
            **{k: v for k, v in (("corrected_call", call), ("corrected_old_wire", old_wire))
               if "{" + k + "}" in self.config.hint_template})
        # `at: call` routes the trainer to the mid-turn splice (between reasoning and call)
        # with the distillation mask on the call tokens alone; never clipped, since a cut
        # corrected call teaches a truncation
        hint: dict[str, Any] = {"text": text, "at": "call"}
        if self.config.target_mask:
            hint["target"] = call
        known = {t["step"] for t in turns}
        steps = [step] if self.config.hint_at in ("call", "both") else []
        if self.config.hint_at in ("retry", "both"):
            again = retry_step(turns, step, _call_path(call))
            if again is not None:
                steps.append(again)
        hints = {s: dict(hint) for s in steps if s in known}
        self._record(
            "tool_fix", step,
            [{"role": "system", "content": f"(deterministic: corrected call, min_sim={self.config.min_sim})"},
             {"role": "user", "content": f"class: {cls}"}],
            text if hints else "", None, None, None)
        return hints


register_langfuse_op("ToolFixReflector.reflect_trajectory", name="reflection", as_type="evaluator")
