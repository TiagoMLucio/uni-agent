"""Reflector registry: register by name and load by config (mirrors reward/registry)."""

from collections.abc import Callable
from importlib import import_module
from typing import Any

from uni_agent.reflection.base import AbstractReflector, BaseReflectionConfig

REFLECTOR_REGISTRY: dict[str, type[AbstractReflector]] = {}

REFLECTOR_MODULES: dict[str, str] = {
    "single": "uni_agent.reflection.single",
    "pipeline": "uni_agent.reflection.pipeline",
    "loop_router": "uni_agent.reflection.loop_router",
}


def register_reflector(name: str) -> Callable[[type[AbstractReflector]], type[AbstractReflector]]:
    """Decorator to register a reflector class with a given name."""

    def decorator(cls: type[AbstractReflector]) -> type[AbstractReflector]:
        if name in REFLECTOR_REGISTRY and REFLECTOR_REGISTRY[name] != cls:
            raise ValueError(f"Reflector {name} has already been registered: {REFLECTOR_REGISTRY[name]} vs {cls}")
        REFLECTOR_REGISTRY[name] = cls
        return cls

    return decorator


def _reflector_class(name: str) -> type[AbstractReflector]:
    if name not in REFLECTOR_REGISTRY and name in REFLECTOR_MODULES:
        import_module(REFLECTOR_MODULES[name])
    if name not in REFLECTOR_REGISTRY:
        available = sorted(set(REFLECTOR_REGISTRY) | set(REFLECTOR_MODULES))
        raise ValueError(f"Unknown reflector: {name}. Available: {available}")
    return REFLECTOR_REGISTRY[name]


def build_reflection_config(config: dict[str, Any] | None) -> BaseReflectionConfig:
    """Validate the ``reflection`` block against the strategy its ``name`` selects.

    The caller reads ``enabled`` and ``failed_only`` before deciding to reflect at all, so the
    config is built separately from the reflector itself.
    """
    config = dict(config or {})
    return _reflector_class(config.get("name", "single")).Config(**config)


def load_reflector(model: Any, config: BaseReflectionConfig, run_id: str = "",
                   record_path=None, identity: dict | None = None) -> AbstractReflector:
    """Instantiate the reflector a validated config selects."""
    return _reflector_class(config.name)(
        model, config, run_id=run_id, record_path=record_path, identity=identity)
