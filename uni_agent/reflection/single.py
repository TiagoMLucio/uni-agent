"""One call per rollout: the whole trajectory in, hints for its pivotal turns out."""

from typing import ClassVar

from uni_agent.reflection.base import (
    DEFAULT_SYSTEM_TEMPLATE,
    DEFAULT_USER_TEMPLATE,
    AbstractReflector,
    BaseReflectionConfig,
)
from uni_agent.reflection.registry import register_reflector
from uni_agent.tracing import register_langfuse_op, rollout_trace_op


class SingleCallReflectionConfig(BaseReflectionConfig):
    system_template: str = DEFAULT_SYSTEM_TEMPLATE
    user_template: str = DEFAULT_USER_TEMPLATE


@register_reflector("single")
class Reflector(AbstractReflector):
    """Policy-as-reflector: the reflector sees every turn plus the privileged context the student
    never saw, and writes one hint per selected turn."""

    Config: ClassVar[type[BaseReflectionConfig]] = SingleCallReflectionConfig

    @rollout_trace_op
    async def reflect_trajectory(
        self, task: str, turns: list[dict], gold: str, feedback: str, outcome: str = "", agent_patch: str = ""
    ) -> dict[int, str]:
        cfg = self.config
        k = str(cfg.max_selected_turns)
        gold = self._clip(gold, cfg.max_patch_chars) if gold else gold
        agent_patch = self._clip(agent_patch, cfg.max_patch_chars) if agent_patch else agent_patch

        def render_user(obs_cap, resp_cap):
            return cfg.user_template.replace("{k}", k).format(
                task=task,
                outcome=outcome or "(not available)",
                agent_patch=agent_patch if cfg.include_agent_patch and agent_patch else "(not available)",
                gold=gold if cfg.include_gold and gold else "(not available)",
                feedback=feedback if cfg.include_exec_feedback and feedback else "(not available)",
                turns=self._render_turns(turns, obs_cap, resp_cap),
            )

        text = await self._ask(cfg.system_template.replace("{k}", k), render_user)
        if text is None:
            return {}
        return self._keep_valid(self._parse(text), turns)


register_langfuse_op("Reflector.reflect_trajectory", name="reflection", as_type="evaluator")
