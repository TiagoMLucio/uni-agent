"""Whole-trajectory hindsight reflection: the policy re-prompted to coach its own rollout.

One call per rollout: the reflector sees every turn (compactly rendered) plus privileged
context (gold patch, execution feedback, outcome) the student never saw, selects the few
turns where better guidance would most have changed the outcome, and writes one coaching
hint per selected turn. Hints condition the distillation teacher and are never a training
target. Guidance only: the prompt forbids revealing the fix itself.
"""

import gzip
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from uni_agent.async_logging import get_logger
from uni_agent.interaction.model import MaxTokenExceededError
from uni_agent.tracing import rollout_trace_span

DEFAULT_SYSTEM_TEMPLATE = (
    "You are a hindsight coach reviewing a software-engineering agent's failed attempt at a "
    "task. You see the whole attempt and its outcome; the agent does not. Select up to {k} "
    "turns where a different action would most have changed the outcome, and write one hint "
    "per selected turn telling the agent exactly what to do next from that state.\n"
    "Rules:\n"
    "1. Be direct and concrete: name the action to take now (the command to run, the file to "
    "open, the edit to make, the test to run) and what it should accomplish. Do not ask the "
    "agent to verify, double-check, or confirm things: tell it what to do.\n"
    "2. Use only information the agent had already seen at that turn. Never name files, "
    "functions, paths, or values it had not yet discovered; when the right action depends on "
    "something undiscovered, direct the agent to the action that discovers it.\n"
    "3. The agent is at that exact state and has not acted yet: never mention this or any "
    "other attempt, and never refer to anything that happens after the selected turn.\n"
    '4. Output only valid JSON, no other text: {"turn<index>": "<hint>", ...} with entries '
    "only for the selected turns, using the given turn indices."
)

DEFAULT_USER_TEMPLATE = (
    "Task:\n{task}\n\n"
    "Outcome of the attempt:\n{outcome}\n\n"
    "Patch the attempt produced:\n{agent_patch}\n\n"
    "Reference patch (privileged, never reveal its content):\n{gold}\n\n"
    "Execution feedback from the attempt:\n{feedback}\n\n"
    "Full trajectory:\n{turns}\n\n"
    "Return the JSON with one hint per selected turn (at most {k})."
)

#: A reasoned prompt writes its audit before the answer, so the object that matters is the one
#: after the last marker. Parsing the first decodable ``{...}`` instead lets a brace quoted in
#: the audit shadow the real hints, which costs the rollout its supervision silently.
FINAL_MARKER = "FINAL_HINTS_JSON:"

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
#: The reflectors that produce the best hints answer one note per line, "54: the condition needs
#: both clauses to hold". Nothing here read that shape, so such a reply yielded no hints at all,
#: which reads as a reflector with nothing to say rather than as a format mismatch. A colon is
#: required rather than any separator, so a numbered list ("1. First, open the file") and a diff
#: body cannot be mined as hints.
_TURN_LINE_RE = re.compile(r"^[\s>*_`#-]*\[?\(?<?\s*(?:turn\s*)?(\d+)\s*>?\)?\]?[`*_]*\s*:\s+(.+?)\s*$",
                           re.I | re.M)

#: The str_replace editor's own error messages, one stable fragment per message (interpolated
#: values elided; both wording variants where the tool has two). Verbatim substrings only: a
#: fuzzy pattern would also catch file content that merely resembles an error.
EDITOR_ERROR_MARKS = (
    # str_replace failures
    "did not appear verbatim in",
    "No replacement was performed. Multiple occurrences of",
    "your old_str and new_str are byte-identical",
    "is the same as new_str",
    # argument validation
    "Parameter `file_text` is required for command: create",
    "Parameter `old_str` is required for command: str_replace",
    "Parameter `insert_line` is required for command: insert",
    "Parameter `new_str` is required for command: insert",
    "Unrecognized command `",
    # path validation
    "does not exist. Please provide a valid path.",
    "Cannot overwrite files using command",
    "is a directory and only the `view` command can be used on directories",
    # view / create / insert / undo_edit / file io
    "The `view_range` parameter is not allowed when `path` points to a directory.",
    "Invalid `view_range`",
    "does not exist. Please create it first.",
    "Invalid `insert_line` parameter:",
    "No edit history found for",
    "Ran into UnicodeDecodeError",
    "while trying to write to",
)


def first_editor_error_step(turns: list[dict]) -> int | None:
    """Step of the first str_replace_editor call whose observation is one of the editor's errors."""
    for turn in turns:
        for call in turn.get("tools") or []:
            observation = call.get("observation") or ""
            if call.get("name") == "str_replace_editor" and any(mark in observation for mark in EDITOR_ERROR_MARKS):
                return turn["step"]
    return None


class BaseReflectionConfig(BaseModel):
    """Settings shared by every reflector (the agent config's ``reflection`` block).

    ``name`` picks the implementation from the registry; each one validates the block against
    its own subclass, so a key that belongs to another strategy is rejected rather than ignored.
    """

    #: a misspelled key used to be dropped in silence, leaving the default in force
    model_config = ConfigDict(extra="forbid")

    name: str = "single"
    enabled: bool = False
    failed_only: bool = True
    #: terminations (`stuck`, `max_step_limit`, ...) left unhinted; skips the reflector calls too
    skip_exit_reasons: list[str] = []
    #: drop every hint at or after the first failed str_replace_editor call (its own error strings)
    hint_cutoff_on_editor_error: bool = False
    #: apply_chat_template kwargs for reflector calls only; None inherits the rollout's.
    #: The reflector is a separate, untrained call whose prompt asks for staged reasoning,
    #: so it can need reasoning on where the rollout deliberately has it off.
    chat_template_kwargs: dict | None = None
    include_gold: bool = True
    # what the attempt actually produced; the other half of "what they did vs what was needed".
    # Captured by the reward spec (``reward.agent_patch_context`` sets its width).
    include_agent_patch: bool = True
    include_exec_feedback: bool = True
    max_selected_turns: int = 5
    # The reflector reads a finished trajectory in one shot, so it needs far more room than the
    # agent's own context budget, which the condenser reseats against. Left unset it inherits
    # that budget, and long trajectories then get hints written from a shrink-laddered render:
    # measured over SWE-smith, a render with observations already capped at 1000 chars reaches
    # ~47k tokens, well past a 32k agent budget. Cannot exceed what the engine serves.
    max_model_len: int | None = None
    max_observation_chars: int = 1000
    max_diagnosis_chars: int = 4000
    # the shrink ladder only trims turns, so an outsized patch overflows at every level and
    # drops the rollout's hints entirely. Over SWE-smith 16k spares 99.2% of tasks.
    max_patch_chars: int = 16000
    # Room for the reflector's own reply. A reasoned prompt writes an audit before its JSON, and
    # 2048 truncates the tail on long trajectories: the object never closes and the rollout loses
    # every hint without an error, since a cut reply simply parses to nothing.
    max_output_tokens: int = 2048
    # Retries when the render overflows the serving context, as (observation cap, response cap);
    # None means uncapped. The first attempt always uses max_observation_chars, so these are what
    # it falls back to. Deriving them from that field instead made the first rungs no-ops
    # whenever it was set high.
    # Rungs are token budgets approximated in chars at ~3.8 chars/token: observations shrink
    # first (2k -> 1k tokens), then the agent's own responses (2k -> 1k), since the responses
    # are what the hints are about. Responses drive the worst case (4k tok/turn x 50 turns vs
    # a measured 77k-token observation ceiling), so only the last rung provably fits 131k.
    shrink_ladder: list[tuple[int | None, int | None]] = [
        (7600, None), (3800, None), (3800, 7600), (3800, 3800),
    ]


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
        self.identity = {k: v for k, v in (identity or {}).items()
                         if k in ("uid", "instance_id", "run_id")}

    @abstractmethod
    async def reflect_trajectory(
        self, task: str, turns: list[dict], gold: str, feedback: str, outcome: str = "", agent_patch: str = ""
    ) -> dict[int, str]:
        """Hints keyed by step index; empty on any failure, since hints are optional supervision."""

    async def _ask(self, system: str, render_user, max_output_tokens: int | None = None,
                   stage: str = "", step: int | None = None, accept=None) -> str | None:
        """One call, retried down the shrink ladder. ``render_user(obs_cap, resp_cap) -> str``.

        ``accept(text) -> bool`` decides whether a reply is usable. A reply that parses to
        nothing is a wasted rollout, and contract failures are drawn per sample rather than
        being properties of the trace (measured: zero traces failed in all three repeats of
        one arm, Cohen's kappa about 0), so re-drawing recovers most of them where rewording
        the prompt does not. Without it, only an over-budget render was ever retried.
        """
        cfg = self.config
        rungs = [(cfg.max_observation_chars, None), *cfg.shrink_ladder]
        # a rejected reply re-draws on the same rung once before shrinking, since the shrink
        # is there for prompts that do not fit, not for replies that came out malformed
        ladder = [rung for rung in rungs for _ in (range(2) if accept is not None else range(1))]
        rejected = None
        for attempt, (obs_cap, resp_cap) in enumerate(ladder):
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": render_user(obs_cap, resp_cap)},
            ]
            # every call reaches Langfuse as an identical model_call, so a pipeline's stages are
            # indistinguishable there without a span naming the one they belong to
            label = "reflect:{}".format(stage or "call") + ("" if step is None else f"@turn{step}")
            try:
                with rollout_trace_span(label, metadata={"obs_cap": obs_cap, "resp_cap": resp_cap}):
                    # omit tool schemas: they bias the model toward a tool call instead of the requested JSON
                    cache = await self.model.prepare_rollout_cache(
                        messages, include_tools=False, chat_template_kwargs=cfg.chat_template_kwargs
                    )
                    # the engine is sized for this call, not for a rollout: rollouts condense to the
                    # agent's budget while a reflector prompt is the whole trajectory at once, so its
                    # prefill is what sets the peak activation the rollout engine has to fit
                    prompt_tokens = len(cache.get("prompt_ids") or ())
                    sampling_params = {
                        **(getattr(self.model, "sampling_params", None) or {}),
                        "max_tokens": max_output_tokens or cfg.max_output_tokens,
                    }
                    text, _, _, _ = await self.model.query(
                        messages=messages, rollout_cache=cache, sampling_params=sampling_params,
                        max_model_len=cfg.max_model_len,
                    )
                self.logger.info(f"Reflection call ok: prompt_tokens={prompt_tokens} "
                                 f"obs_cap={obs_cap} resp_cap={resp_cap} out={len(text or '')}c")
                self._record(stage, step, messages, text, prompt_tokens, obs_cap, resp_cap)
                if accept is None or accept(text):
                    return text
                rejected = text
                self.logger.info(f"Reflection reply unusable (attempt {attempt + 1}/{len(ladder)}, "
                                 f"obs_cap={obs_cap}); re-drawing")
                continue
            except MaxTokenExceededError as exc:
                self.logger.info(f"Reflection render over budget (obs_cap={obs_cap}, resp_cap={resp_cap}): {exc}")
                self._record(stage, step, messages, None, None, obs_cap, resp_cap, error="over budget")
                continue
            except Exception as exc:
                self.logger.warning(f"Reflection call failed; no hints for this rollout: {exc}")
                self._record(stage, step, messages, None, None, obs_cap, resp_cap, error=repr(exc))
                return None
        if rejected is not None:
            # hand back the last reply anyway: the caller's own parse is the arbiter, and a
            # reply it cannot use is no worse than the None this used to return
            self.logger.warning("Reflection: no usable reply in %d attempts", len(ladder))
            return rejected
        self.logger.warning("Reflection skipped: render over budget at every shrink level")
        return None

    def _record(self, stage, step, messages, text, prompt_tokens, obs_cap, resp_cap, error=""):
        """Append one call to the rollout's reflection log, if the loop asked for one.

        What each stage was shown and answered is not recoverable from anything else the
        rollout writes, so it is captured here or not at all. Never raises: a reflector
        that dies over its own bookkeeping would cost the rollout its supervision.
        """
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
            self._record_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(self._record_path, "at", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        except Exception as exc:
            self.logger.warning(f"Reflection record not written: {exc!r}")
            self._record_path = None

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
                tokens=turn.get("tokens", "?"),
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
        if cap is None:
            return text
        return self._clip(text, cap)

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

    @staticmethod
    def _parse_turn_lines(text: str) -> dict[int, str]:
        """Hints from a reply that answers one note per line instead of a JSON object."""
        hints: dict[int, str] = {}
        for step, note in _TURN_LINE_RE.findall(text or ""):
            note = note.strip().strip('"').strip()
            if len(note) >= 15:
                hints[int(step)] = note
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
            salvaged = cls._salvage_hints(tail or text)
            return salvaged or cls._parse_turn_lines(tail or text)
        hints: dict[int, str] = {}
        for key, value in raw.items():
            digits = "".join(c for c in str(key) if c.isdigit())
            if digits and isinstance(value, str) and value.strip():
                hints[int(digits)] = value.strip()
        return hints


