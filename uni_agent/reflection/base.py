"""Whole-trajectory hindsight reflection: the policy re-prompted to coach its own rollout.

One call per rollout: the reflector sees every turn (compactly rendered) plus privileged
context (gold patch, execution feedback, outcome) the student never saw, selects the few
turns where better guidance would most have changed the outcome, and writes one coaching
hint per selected turn. Hints condition the distillation teacher and are never a training
target. Guidance only: the prompt forbids revealing the fix itself.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from uni_agent.async_logging import get_logger
from uni_agent.interaction.model import MaxTokenExceededError

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

_JSON_DECODER = json.JSONDecoder()


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
    shrink_ladder: list[tuple[int | None, int | None]] = [
        (2000, None), (1000, None), (1000, 2000), (1000, 1000),
    ]


class AbstractReflector(ABC):
    """One reflector strategy: turns a finished trajectory into hints for its pivotal turns.

    ``model`` is any client exposing ``prepare_rollout_cache``/``query``.
    """

    Config: ClassVar[type[BaseReflectionConfig]] = BaseReflectionConfig

    def __init__(self, model: Any, config: BaseReflectionConfig, run_id: str = ""):
        self.model = model
        self.config = config
        self.logger = get_logger("reflection", run_id=run_id)

    @abstractmethod
    async def reflect_trajectory(
        self, task: str, turns: list[dict], gold: str, feedback: str, outcome: str = "", agent_patch: str = ""
    ) -> dict[int, str]:
        """Hints keyed by step index; empty on any failure, since hints are optional supervision."""

    async def _ask(self, system: str, render_user, max_output_tokens: int | None = None) -> str | None:
        """One call, retried down the shrink ladder. ``render_user(obs_cap, resp_cap) -> str``."""
        cfg = self.config
        for obs_cap, resp_cap in [(cfg.max_observation_chars, None), *cfg.shrink_ladder]:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": render_user(obs_cap, resp_cap)},
            ]
            try:
                # omit tool schemas: they bias the model toward a tool call instead of the requested JSON
                cache = await self.model.prepare_rollout_cache(messages, include_tools=False)
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
                return text
            except MaxTokenExceededError as exc:
                self.logger.info(f"Reflection render over budget (obs_cap={obs_cap}, resp_cap={resp_cap}): {exc}")
                continue
            except Exception as exc:
                self.logger.warning(f"Reflection call failed; no hints for this rollout: {exc}")
                return None
        self.logger.warning("Reflection skipped: render over budget at every shrink level")
        return None

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

    def _render_turns(self, turns: list[dict], obs_cap: int, resp_cap: int | None) -> str:
        return "\n\n".join(
            TURN_TEMPLATE.format(
                step=turn["step"],
                tokens=turn.get("tokens", "?"),
                response=(
                    # mark breakdown turns so the reflector coaches recovery instead of inventing content (rule 3)
                    f"(degenerate turn: the model emitted almost no output and no tool call) {turn['response']!r}"
                    if not turn["tools"] and len(turn["response"].strip()) < 20
                    else self._clip_response(turn["response"], resp_cap)
                ),
                tools="\n".join(
                    TOOL_TEMPLATE.format(name=r["name"], observation=self._clip(r["observation"] or "", obs_cap))
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
    def _extract_json_object(text: str) -> Any:
        """First ``{...}`` that decodes as JSON, tolerating surrounding prose or ```json fences."""
        idx = text.find("{")
        while idx != -1:
            try:
                return _JSON_DECODER.raw_decode(text, idx)[0]
            except json.JSONDecodeError:
                idx = text.find("{", idx + 1)
        return None

    @classmethod
    def _parse(cls, text: str) -> dict[int, str]:
        # After the last marker first, so an audit's own braces cannot shadow the answer; the
        # unanchored scan stays as the fallback for a reply that omitted the marker entirely.
        raw = None
        if FINAL_MARKER in (text or ""):
            raw = cls._extract_json_object(text.rsplit(FINAL_MARKER, 1)[1])
        if not isinstance(raw, dict):
            raw = cls._extract_json_object(text)
        if not isinstance(raw, dict):
            return {}
        hints: dict[int, str] = {}
        for key, value in raw.items():
            digits = "".join(c for c in str(key) if c.isdigit())
            if digits and isinstance(value, str) and value.strip():
                hints[int(digits)] = value.strip()
        return hints


