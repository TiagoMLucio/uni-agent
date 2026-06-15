"""Abstract base for reward specs."""

from abc import ABC, abstractmethod


class AbstractRewardSpec(ABC):
    """Reward spec: computes reward from interaction result and optional env eval."""

    @abstractmethod
    async def compute_reward(self, interaction_result: dict, **kwargs) -> tuple:
        """
        Compute reward (and optionally run eval in env) from the interaction result.

        Returns:
            A 2-tuple whose first element is the reward score (or eval report) and
            whose second element is auxiliary info; the concrete element types
            depend on the reward spec.

        Reward-extra-info convention:
            If the second element is a dict and contains a ``"reward_extra_info"``
            key, ``UniAgentLoop`` surfaces it on the trajectory's
            ``extra_fields["reward_extra_info"]``, where downstream training code can
            consume it (for example, textual ``feedback`` describing the attempt).
            Specs that emit such info should populate ``result["reward_extra_info"]``
            with the relevant keys (e.g. ``{"feedback": <str | None>}``; ``None`` when
            there is nothing to report, such as on success).
        """
        ...
