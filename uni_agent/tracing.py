"""Rollout tracing, re-exported from verl with no-op fallbacks.

Keeps verl (torch, ray) an optional import-time dependency of the inference-only path:
without verl the agent simply runs untraced.
"""

try:
    from verl.utils.rollout_trace import (
        register_langfuse_op,
        rollout_trace_event,
        rollout_trace_generation,
        rollout_trace_op,
        rollout_trace_score,
        rollout_trace_set_attr,
    )
except ImportError:

    def rollout_trace_op(func):
        return func

    def register_langfuse_op(*args, **kwargs):
        pass

    def rollout_trace_event(*args, **kwargs):
        pass

    def rollout_trace_generation(*args, **kwargs):
        pass

    def rollout_trace_score(*args, **kwargs):
        pass

    def rollout_trace_set_attr(*args, **kwargs):
        pass


__all__ = [
    "register_langfuse_op",
    "rollout_trace_event",
    "rollout_trace_generation",
    "rollout_trace_op",
    "rollout_trace_score",
    "rollout_trace_set_attr",
]
