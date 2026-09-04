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
        rollout_trace_span,
        rollout_trace_update_span,
        rollout_trace_update_trace,
        trace_clip,
    )
except ImportError:
    import contextlib

    def rollout_trace_op(func):
        return func

    def register_langfuse_op(*args, **kwargs):
        pass

    def rollout_trace_event(*args, **kwargs):
        pass

    def trace_clip(text, cap=8000):
        return text

    def rollout_trace_generation(*args, **kwargs):
        pass

    def rollout_trace_score(*args, **kwargs):
        pass

    def rollout_trace_set_attr(*args, **kwargs):
        pass

    @contextlib.contextmanager
    def rollout_trace_span(*args, **kwargs):
        yield None

    def rollout_trace_update_span(*args, **kwargs):
        pass

    def rollout_trace_update_trace(*args, **kwargs):
        pass


# Trace payload caps (middle-clipped by trace_clip): a failing pytest log carries its
# setup errors at the top and its failure summary at the bottom, so both ends survive.
TRACE_TEST_OUTPUT_CHARS = 16000
TRACE_PATCH_CHARS = 8000
TRACE_FEEDBACK_CHARS = 16000

__all__ = [
    "TRACE_FEEDBACK_CHARS",
    "TRACE_PATCH_CHARS",
    "TRACE_TEST_OUTPUT_CHARS",
    "register_langfuse_op",
    "rollout_trace_event",
    "rollout_trace_generation",
    "rollout_trace_op",
    "rollout_trace_score",
    "rollout_trace_set_attr",
    "rollout_trace_span",
    "rollout_trace_update_span",
    "rollout_trace_update_trace",
    "trace_clip",
]
