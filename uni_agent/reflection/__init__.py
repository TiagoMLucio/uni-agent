"""Whole-trajectory hindsight reflection: the policy re-prompted to coach its own rollout.

The reflector sees every turn plus privileged context (gold patch, execution feedback, outcome)
the student never saw, selects the few turns where better guidance would most have changed the
outcome, and writes one hint per selected turn. Hints condition the distillation teacher and are
never a training target.

``reflection.name`` picks the strategy; ``pipeline`` runs one or more calls, including per-turn
calls whose context is truncated to that turn's prefix. Every prompt is in the config block.
"""

from uni_agent.reflection.base import (
    FINAL_MARKER,
    TOOL_TEMPLATE,
    TURN_TEMPLATE,
    AbstractReflector,
    BaseReflectionConfig,
)
from uni_agent.reflection.pipeline import CallSpec, PipelineReflectionConfig, PipelineReflector
from uni_agent.reflection.registry import (
    REFLECTOR_REGISTRY,
    build_reflection_config,
    load_reflector,
    register_reflector,
)

__all__ = [
    "FINAL_MARKER",
    "REFLECTOR_REGISTRY",
    "TOOL_TEMPLATE",
    "TURN_TEMPLATE",
    "AbstractReflector",
    "CallSpec",
    "PipelineReflectionConfig",
    "PipelineReflector",
    "BaseReflectionConfig",
    "build_reflection_config",
    "load_reflector",
    "register_reflector",
]
