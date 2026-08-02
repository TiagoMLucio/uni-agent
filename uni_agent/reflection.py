"""Whole-trajectory hindsight reflection: the policy re-prompted to coach its own rollout.

One call per rollout: the reflector sees every turn (compactly rendered) plus privileged
context (gold patch, execution feedback, outcome) the student never saw, selects the few
turns where better guidance would most have changed the outcome, and writes one coaching
hint per selected turn. Hints condition the distillation teacher and are never a training
target. Guidance only: the prompt forbids revealing the fix itself.
"""

import json
from typing import Any

from pydantic import BaseModel

from uni_agent.async_logging import get_logger
from uni_agent.interaction.model import MaxTokenExceededError
from uni_agent.tracing import register_langfuse_op, rollout_trace_op

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

TURN_TEMPLATE = "### Turn {step} ({tokens} tokens)\nASSISTANT:\n{response}\n{tools}"
# the response is the model's raw output, so it already carries the tool call and its arguments;
# rendering the parsed action too duplicated whole written files in the prompt
TOOL_TEMPLATE = "TOOL {name}:\n{observation}"

_JSON_DECODER = json.JSONDecoder()


class ReflectionConfig(BaseModel):
    """Hindsight-reflector settings (off by default; the agent config's ``reflection`` block)."""

    enabled: bool = False
    failed_only: bool = True
    include_gold: bool = True
    # what the attempt actually produced; the other half of "what they did vs what was needed".
    # Captured by the reward spec (``reward.agent_patch_context`` sets its width).
    include_agent_patch: bool = True
    include_exec_feedback: bool = True
    max_selected_turns: int = 5
    max_observation_chars: int = 1000
    max_diagnosis_chars: int = 4000
    # the shrink ladder only trims turns, so an outsized patch overflows at every level and
    # drops the rollout's hints entirely. Over SWE-smith 16k spares 99.2% of tasks.
    max_patch_chars: int = 16000
    system_template: str = DEFAULT_SYSTEM_TEMPLATE
    user_template: str = DEFAULT_USER_TEMPLATE


class Reflector:
    """Policy-as-reflector: one call per rollout through the given chat model.

    ``model`` is any client exposing ``prepare_rollout_cache``/``query``.
    """

    def __init__(self, model: Any, config: ReflectionConfig, run_id: str = ""):
        self.model = model
        self.config = config
        self.logger = get_logger("reflection", run_id=run_id)

    @rollout_trace_op
    async def reflect_trajectory(
        self, task: str, turns: list[dict], gold: str, feedback: str, outcome: str = "", agent_patch: str = ""
    ) -> dict[int, str]:
        """Select and hint the pivotal turns of one full trajectory; empty on any failure.

        The render is retried down a shrink ladder (smaller observation caps, then middle-cut
        responses) when the prompt exceeds the serving context; the overflow check is client-
        side, so retries cost no server call.
        """
        cfg = self.config
        k = str(cfg.max_selected_turns)
        system = cfg.system_template.replace("{k}", k)
        obs = cfg.max_observation_chars
        gold = self._clip(gold, cfg.max_patch_chars) if gold else gold
        agent_patch = self._clip(agent_patch, cfg.max_patch_chars) if agent_patch else agent_patch
        for obs_cap, resp_cap in ((obs, None), (obs // 2, None), (obs // 2, 800)):
            user = cfg.user_template.replace("{k}", k).format(
                task=task,
                outcome=outcome or "(not available)",
                agent_patch=agent_patch if cfg.include_agent_patch and agent_patch else "(not available)",
                gold=gold if cfg.include_gold and gold else "(not available)",
                feedback=feedback if cfg.include_exec_feedback and feedback else "(not available)",
                turns=self._render_turns(turns, obs_cap, resp_cap),
            )
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            try:
                # omit tool schemas: they bias the model toward a tool call instead of the requested JSON
                cache = await self.model.prepare_rollout_cache(messages, include_tools=False)
                # the rollout request's response-length formula clamps to 1 token on long prompts, truncating the JSON
                sampling_params = {**(getattr(self.model, "sampling_params", None) or {}), "max_tokens": 2048}
                text, _, _, _ = await self.model.query(
                    messages=messages, rollout_cache=cache, sampling_params=sampling_params
                )
            except MaxTokenExceededError as exc:
                self.logger.info(f"Reflection render over budget (obs_cap={obs_cap}, resp_cap={resp_cap}): {exc}")
                continue
            except Exception as exc:
                self.logger.warning(f"Reflection call failed; no hints for this rollout: {exc}")
                return {}
            valid_steps = {turn["step"] for turn in turns}
            hints = {
                step: self._clip_diagnosis(diagnosis)
                for step, diagnosis in self._parse(text).items()
                if step in valid_steps
            }
            # over-selection guard: keep the earliest K
            if len(hints) > cfg.max_selected_turns:
                hints = dict(sorted(hints.items())[: cfg.max_selected_turns])
            return hints
        self.logger.warning("Reflection skipped: render over budget at every shrink level")
        return {}

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

    def _clip(self, text: str, cap: int) -> str:
        """Middle-out truncation: a failing turn's signal is often at the observation's tail
        (traceback, assertion), so keep both ends and elide the middle."""
        if len(text) <= cap:
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

    @staticmethod
    def _parse(text: str) -> dict[int, str]:
        raw = Reflector._extract_json_object(text)
        if not isinstance(raw, dict):
            return {}
        hints: dict[int, str] = {}
        for key, value in raw.items():
            digits = "".join(c for c in str(key) if c.isdigit())
            if digits and isinstance(value, str) and value.strip():
                hints[int(digits)] = value.strip()
        return hints


register_langfuse_op("Reflector.reflect_trajectory", name="reflection", as_type="evaluator")
