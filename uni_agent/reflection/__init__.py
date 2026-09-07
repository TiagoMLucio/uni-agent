"""Whole-trajectory hindsight reflection: the policy re-prompted to coach its own rollout.

The reflector sees every turn plus privileged context (gold patch, execution feedback, outcome)
the student never saw, selects the few turns where better guidance would most have changed the
outcome, and writes one hint per selected turn. Hints condition the distillation teacher and are
never a training target.

``reflection.name`` picks the strategy: ``single`` is one call per rollout, ``pipeline`` runs
several, including per-turn calls whose context is truncated to that turn's prefix.
"""

from uni_agent.reflection.base import (
    DEFAULT_SYSTEM_TEMPLATE,
    DEFAULT_USER_TEMPLATE,
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
from uni_agent.reflection.single import Reflector, SingleCallReflectionConfig

#: the block a plain `reflection:` config validates against, kept as the historical name
ReflectionConfig = SingleCallReflectionConfig

__all__ = [
    "DEFAULT_SYSTEM_TEMPLATE",
    "DEFAULT_USER_TEMPLATE",
    "FINAL_MARKER",
    "REFLECTOR_REGISTRY",
    "TOOL_TEMPLATE",
    "TURN_TEMPLATE",
    "AbstractReflector",
    "CallSpec",
    "PipelineReflectionConfig",
    "PipelineReflector",
    "BaseReflectionConfig",
    "ReflectionConfig",
    "Reflector",
    "SingleCallReflectionConfig",
    "build_reflection_config",
    "load_reflector",
    "register_reflector",
]
