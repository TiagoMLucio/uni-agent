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
    "You are a hindsight critic reviewing a software-engineering agent's complete attempt at "
    "a task. You see every turn, the attempt's outcome, the reference (gold) patch, and the "
    "execution feedback: privileged information the agent never had. Select at most {k} "
    "turns where better guidance would most have changed the outcome, and write one coaching "
    "hint for each. Each hint is shown to an agent that is at that exact state but has NOT "
    "acted yet: it has not made the mistake, opened the files, or seen anything from that "
    "turn onward. Every hint is policy-facing guidance for what to do next, never a "
    "retrospective explanation of a trajectory.\n"
    "Rules:\n"
    "1. Select at most {k} turns: decisive mistakes, wasteful detours, moments where the "
    "right macro step (reproduce / localize / edit / test / submit) was not taken, or the "
    "moment a correct idea was abandoned. Do not select turns that were correct and "
    "efficient. Prefer the earliest turn where the attempt left a viable path over the later "
    "turns that merely suffered the consequences. When the attempt degenerates into empty or "
    "looping turns, select only the first turn of the breakdown. Selecting fewer than {k} "
    "turns, or none (return {}), is better than writing a hint the trajectory gives no "
    "evidence for.\n"
    "2. Each hint, at most three sentences, phrased only in the imperative about what to do "
    "next. Never mention this or any other attempt, and never describe an action as already "
    "taken or a mistake as already made: from the agent's view nothing after this state has "
    "happened, so any reference to past actions reads as a contradiction. Turn the observed "
    "mistake into an avoidance rule anchored to its warning sign ('avoid X until Y', 'verify "
    "Z before W'), then give the corrective direction: which macro step to take now and "
    "which files, functions, or tests to target. End every hint with the observation the "
    "corrected action should produce (so the agent can internalize action to expected "
    "result).\n"
    "3. Assert only what some visible observation supports; if you are not certain of a "
    "claim, prescribe the action that would verify it instead of stating it as fact. A "
    "correction that could be misread (which value, which direction) is worse than no hint: "
    "if you cannot state it unambiguously in one sentence, prescribe the check instead of "
    "the edit. For a degenerate turn (marked in the render), never invent what it did: coach "
    "how to proceed productively from that state.\n"
    "4. State only conclusions the agent could itself derive from evidence visible before "
    "that turn (or from the action you suggest). Never reveal what only the reference patch "
    "could tell you: its code, its values, or identifiers, functions, and file paths that no "
    "visible observation mentions. Never mention the reference patch in a hint, and never "
    "advise against its fix: the tests reward exactly that fix, however unusual it looks.\n"
    "5. You may use later-trajectory evidence to identify delayed consequences, but never "
    "reveal future actions or observations verbatim.\n"
    "6. Each turn shows its token cost. Wasteful token spending (re-reading whole files, "
    "redundant searches, repeated confirmations) is a valid reason to select a turn, "
    "especially when the attempt died at the token limit.\n"
    '7. Output only valid JSON, no other text: {"turn<index>": "<hint>", ...} with entries '
    "only for the selected turns, using the given turn indices."
)

DEFAULT_USER_TEMPLATE = (
    "Task:\n{task}\n\n"
    "Outcome of the attempt:\n{outcome}\n\n"
    "Reference patch (privileged, never reveal its content):\n{gold}\n\n"
    "Execution feedback from the attempt:\n{feedback}\n\n"
    "Full trajectory:\n{turns}\n\n"
    "Return the JSON with one hint per selected turn (at most {k})."
)

TURN_TEMPLATE = "### Turn {step} ({tokens} tokens)\nASSISTANT:\n{response}\n{tools}"
TOOL_TEMPLATE = "TOOL {name}({action}):\n{observation}"

_JSON_DECODER = json.JSONDecoder()


class ReflectionConfig(BaseModel):
    """Hindsight-reflector settings (off by default; the agent config's ``reflection`` block)."""

    enabled: bool = False
    failed_only: bool = True
    include_gold: bool = True
    include_exec_feedback: bool = True
    max_selected_turns: int = 5
    max_observation_chars: int = 300
    max_diagnosis_chars: int = 4000
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
        self, task: str, turns: list[dict], gold: str, feedback: str, outcome: str = ""
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
        for obs_cap, resp_cap in ((obs, None), (obs // 2, None), (obs // 2, 800)):
            user = cfg.user_template.replace("{k}", k).format(
                task=task,
                outcome=outcome or "(not available)",
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
                    TOOL_TEMPLATE.format(
                        name=r["name"], action=r["action"], observation=self._clip(r["observation"] or "", obs_cap)
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
