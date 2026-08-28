"""Whole-trajectory hindsight reflection: the policy re-prompted to coach its own rollout.

The reflector sees every turn plus privileged context (gold patch, execution feedback, outcome)
the student never saw, selects the few turns where better guidance would most have changed the
outcome, and writes one hint per selected turn. Hints condition the distillation teacher and are
never a training target.

``reflection.name`` picks the strategy: ``single`` is one call per rollout, ``pipeline`` runs
several, including per-turn calls whose context is truncated to that turn's prefix, and
``loop_router`` deterministically handles loop-pattern trajectories, delegating the rest.
"""

from uni_agent.reflection.base import (
    DEFAULT_SYSTEM_TEMPLATE,
    DEFAULT_USER_TEMPLATE,
    EDITOR_ERROR_MARKS,
    FINAL_MARKER,
    TOOL_TEMPLATE,
    TURN_TEMPLATE,
    AbstractReflector,
    BaseReflectionConfig,
    first_editor_error_step,
)
from uni_agent.reflection.loop_router import LoopRouterReflectionConfig, LoopRouterReflector
from uni_agent.reflection.pipeline import CallSpec, PipelineReflectionConfig, PipelineReflector
from uni_agent.reflection.registry import (
    REFLECTOR_REGISTRY,
    build_reflection_config,
    load_reflector,
    register_reflector,
)
from uni_agent.reflection.single import Reflector, SingleCallReflectionConfig
from uni_agent.reflection.tool_diag import ToolDiagReflectionConfig, ToolDiagReflector
from uni_agent.reflection.tool_fix import ToolFixReflectionConfig, ToolFixReflector

#: the block a plain `reflection:` config validates against, kept as the historical name
ReflectionConfig = SingleCallReflectionConfig

__all__ = [
    "DEFAULT_SYSTEM_TEMPLATE",
    "DEFAULT_USER_TEMPLATE",
    "EDITOR_ERROR_MARKS",
    "FINAL_MARKER",
    "REFLECTOR_REGISTRY",
    "TOOL_TEMPLATE",
    "TURN_TEMPLATE",
    "AbstractReflector",
    "CallSpec",
    "LoopRouterReflectionConfig",
    "LoopRouterReflector",
    "PipelineReflectionConfig",
    "PipelineReflector",
    "BaseReflectionConfig",
    "ReflectionConfig",
    "Reflector",
    "SingleCallReflectionConfig",
    "ToolDiagReflectionConfig",
    "ToolDiagReflector",
    "ToolFixReflectionConfig",
    "ToolFixReflector",
    "build_reflection_config",
    "first_editor_error_step",
    "load_reflector",
    "register_reflector",
]
