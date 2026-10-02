"""Whole-trajectory hindsight reflection: the policy re-prompted to coach its own rollout.

One call per rollout: the reflector sees every turn (compactly rendered) plus privileged
context (gold patch, execution feedback, outcome) the student never saw, selects the few
turns where better guidance would most have changed the outcome, and writes one coaching
hint per selected turn. Hints condition the distillation teacher and are never a training
target. Guidance only: the prompt forbids revealing the fix itself.
"""

import asyncio
import gzip
import json
import re
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from uni_agent.async_logging import get_logger
from uni_agent.interaction.model import MaxTokenExceededError
from uni_agent.tracing import rollout_trace_span

#: A reasoned prompt writes its audit before the answer, so the object that matters is the one
#: after the last marker. Parsing the first decodable ``{...}`` instead lets a brace quoted in
#: the audit shadow the real hints, which costs the rollout its supervision silently.
FINAL_MARKER = "FINAL_HINTS_JSON:"

#: the ``_record`` error that marks a render the ladder had to shrink past
OVER_BUDGET = "over budget"
#: what the reflector cost one trajectory, reported by the agent loop
CALL_METRICS = ("reflect_calls", "reflect_redraws", "reflect_over_budget", "reflect_rung")
#: where the search for a response cut starts from above; no single response comes near it
_RESPONSE_SEARCH_TOP = 1_000_000

class ReflectionFailed(RuntimeError):
    """The pipeline could not run as designed: a call raised, or no rung of the ladder fit the
    budget. Its hints are dropped, an earlier stage's included, since they are not what the
    pipeline decided."""


TURN_TEMPLATE = "### Turn {step}\nASSISTANT:\n{response}\n{tools}"
# the response is the model's raw output, so it already carries the tool call and its arguments;
# rendering the parsed action too duplicated whole written files in the prompt
TOOL_TEMPLATE = "TOOL {name}:\n{observation}"


def neutralised_atomic(token: str) -> str:
    """``<|im_end|>`` -> ``[im_end]``, ``</tool_call>`` -> ``[/tool_call]``.

    Same thing to a reader, but no longer the single atomic token the tokenizer emits.
    """
    return "[" + token.strip("<>").strip("|") + "]"

_JSON_DECODER = json.JSONDecoder()
#: raw control characters inside a string are routine when a hint quotes code
_LENIENT_DECODER = json.JSONDecoder(strict=False)
#: a hint key as the model writes it: "turn7", 'turn_7', turn 7
_TURN_KEY_RE = re.compile(r"[\"']?turn[_\s]*(\d+)[\"']?\s*:\s*[\"']")


class BaseReflectionConfig(BaseModel):
    """Settings shared by every reflector (the agent config's ``reflection`` block).

    ``name`` picks the implementation from the registry; each one validates the block against
    its own subclass, so a key that belongs to another strategy is rejected rather than ignored.
    """

    #: a misspelled key used to be dropped in silence, leaving the default in force
    model_config = ConfigDict(extra="forbid")

    #: the strategy whose prompts the block carries; there is no default prompt in code
    name: str
    enabled: bool = False
    failed_only: bool = True
    #: terminations (`stuck`, `max_step_limit`, ...) left unhinted; skips the reflector calls too
    skip_exit_reasons: list[str] = []
    #: apply_chat_template kwargs for reflector calls only; None inherits the rollout's.
    #: The reflector is a separate, untrained call whose prompt asks for staged reasoning,
    #: so it can need reasoning on where the rollout deliberately has it off.
    chat_template_kwargs: dict | None = None
    include_gold: bool = True
    # what the attempt actually produced; the other half of "what they did vs what was needed".
    # Captured by the reward spec (``reward.agent_patch_context`` sets its width).
    include_agent_patch: bool = True
    include_exec_feedback: bool = True
    max_selected_turns: int = 3
    # Serving ceiling for the reflector's one-shot read of a whole trajectory; None inherits the
    # agent's context budget. Cannot exceed what the engine serves.
    max_model_len: int | None = None
    max_observation_chars: int = 1000
    max_diagnosis_chars: int = 4000
    # None reads both patches whole: the attempt's comes as facts.patch_view, which lists the files
    # it created instead of showing them. A cap middle-cuts both patches.
    max_patch_chars: int | None = None
    # Room for the reflector's own reply, reasoning included: a reply cut before its JSON closes
    # parses to nothing and the rollout loses every hint without an error.
    max_output_tokens: int = 16384
    # Steps tried in order when the whole render overflows the serving context, as floors
    # (observations, responses); None leaves that part uncut. A step uses the largest cut at or
    # above its floors that fits, so only items longer than the cut lose their middle and the
    # largest go first. Tool outputs give way before the agent's responses, which hold the
    # decisions the hints are about. Over the overflows measured, cutting the largest outputs
    # alone fit every one, at 70k-600k chars, without hiding a sighting of the defect.
    shrink_ladder: list[tuple[int | None, int | None]] = [(10_000, None), (10_000, 3_800)]
    #: extra draws on the same rung when a reply is unusable, before the render shrinks
    redraws_per_rung: int = Field(default=1, ge=0)


class AbstractReflector(ABC):
    """One reflector strategy: turns a finished trajectory into hints for its pivotal turns.

    ``model`` is any client exposing ``prepare_rollout_cache``/``query``.
    """

    Config: ClassVar[type[BaseReflectionConfig]] = BaseReflectionConfig

    def __init__(self, model: Any, config: BaseReflectionConfig, run_id: str = "",
                 record_path: Path | None = None, identity: dict | None = None):
        self.model = model
        self.config = config
        self.logger = get_logger("reflection", run_id=run_id)
        self._record_path = Path(record_path) if record_path else None
        # a pipeline stage's per-turn calls run concurrently and append to the same file
        self._record_lock = asyncio.Lock()
        self.identity = {k: v for k, v in (identity or {}).items()
                         if k in ("uid", "instance_id", "run_id")}
        # the agent loop builds one reflector per rollout, so these count the trajectory in flight
        self._counts: Counter = Counter()

    @abstractmethod
    async def reflect_trajectory(
        self, task: str, turns: list[dict], gold: str, feedback: str, outcome: str = "", agent_patch: str = ""
    ) -> dict[int, str]:
        """Hints keyed by step index; empty when the reflector found nothing to hint, and
        ``ReflectionFailed`` when the pipeline could not run."""

    async def _ask(self, system: str, render_user, max_output_tokens: int | None = None,
                   stage: str = "", step: int | None = None, accept=None) -> str:
        """One call, re-drawn on an unusable reply and shrunk down the ladder on an over-budget
        render. ``render_user(obs_cap, resp_cap) -> str``. Raises ``ReflectionFailed`` when a call
        raises or no rung fits.

        ``accept(text) -> bool`` decides whether a reply is usable. A reply that parses to
        nothing is a wasted rollout, and contract failures are drawn per sample rather than
        being properties of the trace (measured: zero traces failed in all three repeats of
        one experiment, Cohen's kappa about 0), so re-drawing recovers most of them where rewording
        the prompt does not.
        """
        cfg = self.config
        draws = 1 + cfg.redraws_per_rung if accept is not None else 1
        max_tokens = max_output_tokens or cfg.max_output_tokens
        # every call reaches Langfuse as an identical model_call, so a pipeline's stages are
        # indistinguishable there without a span naming the one they belong to
        label = "reflect:{}".format(stage or "call") + ("" if step is None else f"@turn{step}")
        rejected = None
        async for rung, obs_cap, resp_cap, messages, cache, prompt_tokens in self._renders(
            system, render_user, max_tokens, stage, step
        ):
            for draw in range(draws):
                try:
                    with rollout_trace_span(label, metadata={"obs_cap": obs_cap, "resp_cap": resp_cap}):
                        sampling_params = {
                            **(getattr(self.model, "sampling_params", None) or {}),
                            "max_tokens": max_tokens,
                        }
                        text, _, _, _ = await self.model.query(
                            messages=messages, rollout_cache=cache, sampling_params=sampling_params,
                            max_model_len=cfg.max_model_len,
                        )
                    self.logger.info(f"Reflection call ok: prompt_tokens={prompt_tokens} "
                                     f"obs_cap={obs_cap} resp_cap={resp_cap} out={len(text or '')}c")
                    await self._record(stage, step, messages, text, prompt_tokens, obs_cap, resp_cap, draw=draw)
                    if accept is None or accept(text):
                        self._counts["reflect_rung"] = max(self._counts["reflect_rung"], rung)
                        return text
                    rejected = text
                    self.logger.info(f"Reflection reply unusable (draw {draw + 1}/{draws}, "
                                     f"obs_cap={obs_cap})")
                except MaxTokenExceededError as exc:
                    # the client's own ceiling, when none was known to measure against
                    self.logger.info(f"Reflection render over budget (obs_cap={obs_cap}, resp_cap={resp_cap}): {exc}")
                    await self._record(stage, step, messages, None, None, obs_cap, resp_cap,
                                       error=OVER_BUDGET, draw=draw)
                    # deterministic at this cut: the redraws would render the same prompt
                    break
                except Exception as exc:
                    self.logger.warning(f"Reflection call failed; no hints for this rollout: {exc}")
                    await self._record(stage, step, messages, None, None, obs_cap, resp_cap,
                                       error=repr(exc), draw=draw)
                    raise ReflectionFailed(f"{stage or 'call'} failed: {exc!r}") from exc
            else:
                # the ladder is there for a render that does not fit; a reply the parser could not
                # use is no reason to ask again from a deliberately smaller view of the trajectory
                break
        if rejected is not None:
            # hand back the last reply anyway: the caller's own parse is the arbiter, and a
            # reply it cannot use is no worse than the None this used to return
            self.logger.warning("Reflection: no usable reply in %d draws", draws)
            return rejected
        self.logger.warning("Reflection skipped: render over budget at every shrink level")
        raise ReflectionFailed(f"{stage or 'call'}: render over budget at every shrink level")

    async def _tokenized(self, system: str, render_user, obs_cap: int | None, resp_cap: int | None):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": render_user(obs_cap, resp_cap)}]
        # omit tool schemas: they bias the model toward a tool call instead of the requested JSON
        cache = await self.model.prepare_rollout_cache(
            messages, include_tools=False, chat_template_kwargs=self.config.chat_template_kwargs
        )
        # the engine is sized for this call, not for a rollout: rollouts condense to the agent's
        # budget while a reflector prompt is the whole trajectory at once, so its prefill is what
        # sets the peak activation the rollout engine has to fit
        return messages, cache, len(cache.get("prompt_ids") or ())

    async def _renders(self, system: str, render_user, max_tokens: int, stage: str, step: int | None):
        """The renders worth sending, in order: (rung, obs_cap, resp_cap, messages, cache, prompt_tokens).

        Rung 0 is the whole view; each ladder step follows only if everything before it overflowed,
        at the largest cut at or above its floors that fits. A step that overflows even at its floors
        is recorded once and skipped. Without a known ceiling nothing can be measured, so a step is
        tried at its floors and the client's own MaxTokenExceededError moves on to the next.
        """
        cfg = self.config
        # the same ceiling query() enforces: the config's, else the client's own
        limit = cfg.max_model_len or getattr(self.model, "max_model_len", None)
        top = cfg.max_observation_chars

        def fits(n: int) -> bool:
            # a prompt that fits but leaves less than the reply's room is over budget too: the
            # staged reply cannot close, and the same prefill would be paid again
            return not limit or n + max_tokens <= limit

        tried = set()
        for rung, (obs_floor, resp_floor) in enumerate([(top, None), *cfg.shrink_ladder]):
            caps = (top if obs_floor is None else min(obs_floor, top), resp_floor)
            if caps in tried:
                continue
            tried.add(caps)
            messages, cache, n = await self._tokenized(system, render_user, *caps)
            if not fits(n):
                self.logger.info(f"Reflection render over budget (obs_cap={caps[0]}, resp_cap={caps[1]}): "
                                 f"prompt_tokens {n} + max_tokens {max_tokens} exceeds max_model_len {limit}")
                await self._record(stage, step, messages, None, None, *caps, error=OVER_BUDGET)
                continue
            if limit and rung:
                # the largest cut that still fits: the part this step cuts is the responses once
                # it names a response floor, the observations otherwise
                part = 1 if resp_floor is not None else 0
                lo, hi = caps[part], top if part == 0 else _RESPONSE_SEARCH_TOP
                while hi - lo > max(256, lo // 20):
                    mid = (lo + hi) // 2
                    probe = (mid, None) if part == 0 else (caps[0], mid)
                    rendered = await self._tokenized(system, render_user, *probe)
                    if fits(rendered[2]):
                        lo, caps, (messages, cache, n) = mid, probe, rendered
                    else:
                        hi = mid
            yield rung, *caps, messages, cache, n

    def call_metrics(self) -> dict[str, float]:
        """What the reflector cost this trajectory, every stage summed: calls, re-draws after an
        unusable reply, over-budget renders, and the worst rung an accepted answer was written
        from (0 is the full view, and under overflow-only shrinking only overflow raises it)."""
        return {key: float(self._counts[key]) for key in CALL_METRICS}

    async def _record(self, stage, step, messages, text, prompt_tokens, obs_cap, resp_cap, error="", draw=0):
        """Tally one call, and append it to the rollout's reflection log if the loop asked for one.

        What each stage was shown and answered is not recoverable from anything else the
        rollout writes, so it is captured here or not at all. Never raises: a reflector
        that dies over its own bookkeeping would cost the rollout its supervision. The write
        runs off the event loop, where a stall on the shared filesystem would hold every rollout.
        """
        if error == OVER_BUDGET:
            self._counts["reflect_over_budget"] += 1
        elif not error:
            self._counts["reflect_calls"] += 1
            if draw:
                self._counts["reflect_redraws"] += 1
        if self._record_path is None:
            return
        async with self._record_lock:
            # an earlier write may have failed while this one waited
            if self._record_path is None:
                return
            try:
                row = {
                    **self.identity,
                    "stage": stage,
                    "step": step,
                    "obs_cap": obs_cap,
                    "resp_cap": resp_cap,
                    "prompt_tokens": prompt_tokens,
                    "system": messages[0]["content"],
                    "user": messages[1]["content"],
                    "output": text,
                    "error": error,
                }
                await asyncio.to_thread(self._append_record, self._record_path, row)
            except Exception as exc:
                self.logger.warning(f"Reflection record not written: {exc!r}")
                self._record_path = None

    @staticmethod
    def _append_record(path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "at", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def _keep_valid(self, hints: dict[int, str], turns: list[dict]) -> dict[int, str]:
        """Hints for real turns only, capped at the configured budget, earliest first."""
        valid = {turn["step"] for turn in turns}
        kept = {step: self._clip_diagnosis(text) for step, text in hints.items() if step in valid}
        if len(kept) > self.config.max_selected_turns:
            kept = dict(sorted(kept.items())[: self.config.max_selected_turns])
        return kept

    def _clip_diagnosis(self, text: str) -> str:
        """Suffix-cut an over-long hint, marking the cut so the teacher knows it is incomplete."""
        cap = self.config.max_diagnosis_chars
        return text if len(text) <= cap else text[:cap] + " [... clipped ...]"

    def _atomic_strings(self) -> list[str]:
        """Every string the tokenizer turns into one atomic token, longest first so an
        overlapping pair rewrites cleanly. Empty when there is no client or it exposes no
        tokenizer -- the render helpers build a Reflector without one -- which costs the
        sanitising, never the call."""
        cached = getattr(self, "_atomic_cache", None)
        if cached is None:
            tokenizer = getattr(getattr(self, "model", None), "tokenizer", None)
            try:
                vocab = tokenizer.get_added_vocab() if tokenizer is not None else {}
            except Exception:
                vocab = {}
            cached = sorted(vocab, key=len, reverse=True)
            self._atomic_cache = cached
        return cached

    def _strip_atomic(self, text: str) -> str:
        """Neutralise atomic tokens carried in from a trajectory.

        A turn's ``response`` is the model's raw output, so it still holds the real
        ``<tool_call>`` and the ``<|im_end|>`` that closed it. Embedded verbatim these
        tokenize as themselves, and a rendered trajectory then reaches the reflector as a
        live conversation with one turn boundary per turn (measured: ~175 atomic tokens
        per prompt over SWE-smith) rather than as the transcript it is meant to read.
        """
        for token in self._atomic_strings():
            if token in text:
                text = text.replace(token, neutralised_atomic(token))
        return text

    def _render_turns(self, turns: list[dict], obs_cap: int, resp_cap: int | None) -> str:
        return "\n\n".join(
            TURN_TEMPLATE.format(
                step=turn["step"],
                response=(
                    # mark breakdown turns so the reflector coaches recovery instead of inventing content (rule 3)
                    f"(degenerate turn: the model emitted almost no output and no tool call) "
                    f"{self._strip_atomic(turn['response'])!r}"
                    if not turn["tools"] and len(turn["response"].strip()) < 20
                    else self._clip_response(self._strip_atomic(turn["response"]), resp_cap)
                ),
                tools="\n".join(
                    TOOL_TEMPLATE.format(
                        name=r["name"], observation=self._clip(self._strip_atomic(r["observation"] or ""), obs_cap)
                    )
                    for r in turn["tools"]
                )
                or "(no tool calls)",
            )
            for turn in turns
        )

    def _clip_response(self, text: str, cap: int | None) -> str:
        """Cut a response to ``cap``: the call's long arguments (a created file's text, an edit's
        strings) lose their middle first, longest first; the agent's own words go last."""
        if cap is None or len(text) <= cap:
            return text
        marker = next((m for m in ("<tool_call>", neutralised_atomic("<tool_call>")) if m in text), None)
        if marker is None:
            return self._clip(text, cap)
        words, call = text.split(marker, 1)
        args = list(re.finditer(r"(<parameter=\w+>\n?)(.*?)(\n?</parameter>)", call, re.S))

        def cut(level: int) -> str:
            # every argument longer than ``level`` keeps its two ends, the shorter ones stay whole
            pieces, last = [], 0
            for m in args:
                pieces += [call[last:m.start(2)], self._clip(m[2], level)]
                last = m.end(2)
            return "".join(pieces) + call[last:]

        if args:
            room = cap - len(words) - len(marker)
            lo, hi = 200, max(len(m[2]) for m in args)
            while hi - lo > 50 and len(cut(lo)) <= room:
                mid = (lo + hi) // 2
                lo, hi = (mid, hi) if len(cut(mid)) <= room else (lo, mid)
            call = cut(lo)
        if len(words) + len(marker) + len(call) > cap:
            words = self._clip(words, max(cap - len(marker) - len(call), 200))
        text = words + marker + call
        return text if len(text) <= cap else self._clip(text, cap)

    def _clip(self, text: str, cap: int | None) -> str:
        """Middle-out truncation: a failing turn's signal is often at the observation's tail
        (traceback, assertion), so keep both ends and elide the middle. ``None`` is uncapped."""
        if cap is None or len(text) <= cap:
            return text
        head = cap // 2
        return f"{text[:head]}\n[... {len(text) - cap} chars elided ...]\n{text[-(cap - head):]}"

    @staticmethod
    def _extract_json_object(text: str, strict: bool = True) -> Any:
        """First ``{...}`` that decodes as JSON, tolerating surrounding prose or ```json fences.

        ``strict=False`` additionally allows raw control characters inside strings, which a
        hint quoting a traceback or a diff routinely contains.
        """
        decoder = _JSON_DECODER if strict else _LENIENT_DECODER
        idx = text.find("{")
        while idx != -1:
            try:
                return decoder.raw_decode(text, idx)[0]
            except json.JSONDecodeError:
                idx = text.find("{", idx + 1)
        return None

    @staticmethod
    def _salvage_hints(text: str) -> dict[int, str]:
        """Turn-keyed hints recovered from an object the decoder cannot accept.

        The object is all-or-nothing to ``json``: one unescaped quote inside one hint costs
        the rollout every hint in the reply, and hints quote code, so that happens. This reads
        the keys directly and takes each value up to the next key, which recovers the text
        verbatim without trusting the delimiters in between. It deliberately does NOT mine the
        prose analysis -- measured, that invents hints where the model declined.
        """
        hints: dict[int, str] = {}
        anchors = list(_TURN_KEY_RE.finditer(text))
        for i, match in enumerate(anchors):
            stop = anchors[i + 1].start() if i + 1 < len(anchors) else len(text)
            value = text[match.end():stop].strip()
            value = value.rstrip("\"},;. \n\t")
            # a fragment shorter than this is a truncated key or a stray delimiter, not a hint
            if len(value) >= 15:
                hints[int(match.group(1))] = value
        return hints

    @classmethod
    def _parse(cls, text: str) -> dict[int, str]:
        # After the last marker first, so an audit's own braces cannot shadow the answer; the
        # unanchored scan stays as the fallback for a reply that omitted the marker entirely.
        text = text or ""
        tail = text.rsplit(FINAL_MARKER, 1)[1] if FINAL_MARKER in text else ""
        raw = None
        for strict in (True, False):
            if tail:
                raw = cls._extract_json_object(tail, strict=strict)
            if not isinstance(raw, dict):
                raw = cls._extract_json_object(text, strict=strict)
            if isinstance(raw, dict):
                break
        if not isinstance(raw, dict):
            # last resort: read the keys out of an object no decoder will take
            return cls._salvage_hints(tail or text)
        hints: dict[int, str] = {}
        for key, value in raw.items():
            digits = "".join(c for c in str(key) if c.isdigit())
            if digits and isinstance(value, str) and value.strip():
                hints[int(digits)] = value.strip()
        return hints
