"""Pluggable context condensers for the agent interaction loop.

A condenser shortens the message history when a rollout would otherwise overflow
``max_model_len``. It is a pure ``messages -> condensed messages`` transform; the
interaction loop owns *when* to call it (reactively, on overflow) and how the token
cache is split into segments afterwards.

The default :class:`TruncateMaskCondenser` uses a budget-based reactive algorithm
(truncate long observations middle-out, then mask old observations, then mask bulky
edit arguments) over uni-agent's message shape (observations are ``role="tool"``;
the tool call is inline in the assistant ``content``).
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Callable

#: An ``arg_masker`` masks the values of named tool-call args in a message's content,
#: returning ``(new_content, chars_removed)``. Supplied by the caller (typically
#: ``ToolsManager.mask_tool_args``) so the condenser need not know the tool-call format.
ArgMasker = Callable[[str, list[str], str], tuple[str, int]]

DEFAULT_TRUNCATED_NOTICE = "\n[The observation was truncated in the middle as it was too long.]"
DEFAULT_MASKED_NOTICE = "[The observation of the above action has been masked out to shorten the context.]"

#: Default tool-call argument names to mask in old assistant messages (step 3).
#: These are *names*, not patterns — the actual format-specific masking is delegated to
#: the tool parser via the ``arg_masker`` passed to :meth:`condense`, so the condenser
#: stays format-agnostic (works for any tool-call format the parser supports).
DEFAULT_MASK_ARG_NAMES = ["old_str", "new_str", "file_text"]


class CondensationFailed(Exception):
    """Raised when the condenser cannot shave the requested budget of characters."""


class AbstractCondenser(ABC):
    """Condenser interface: a pure transform over the message history."""

    @abstractmethod
    def condense(
        self, messages: list[dict], budget_chars: int, arg_masker: ArgMasker | None = None
    ) -> list[dict]:
        """Return a condensed copy of ``messages`` that masks/truncates at least
        ``budget_chars`` characters of old content. Raise :class:`CondensationFailed`
        if it cannot reach the budget. Must not mutate ``messages``.

        ``arg_masker`` (optional) masks named tool-call arguments in a message's
        content in the active tool-call format; condensers that reclaim from
        assistant tool calls delegate to it instead of knowing the format.
        """
        ...


CONDENSER_REGISTRY: dict[str, type[AbstractCondenser]] = {}


def register_condenser(name: str) -> Callable[[type[AbstractCondenser]], type[AbstractCondenser]]:
    def decorator(cls: type[AbstractCondenser]) -> type[AbstractCondenser]:
        if name in CONDENSER_REGISTRY and CONDENSER_REGISTRY[name] is not cls:
            raise ValueError(f"Condenser {name!r} already registered: {CONDENSER_REGISTRY[name]} vs {cls}")
        CONDENSER_REGISTRY[name] = cls
        return cls

    return decorator


def load_condenser(config: dict | None) -> AbstractCondenser | None:
    """Build a condenser from config, or ``None`` when disabled.

    ``config`` is ``{"name": <registered>, ...kwargs}``; ``None``/empty disables
    condensation (the rollout aborts on overflow, as before).
    """
    if not config:
        return None
    cfg = dict(config)
    name = cfg.pop("name", None)
    if name is None:
        raise ValueError("Condenser config must contain 'name'")
    if name not in CONDENSER_REGISTRY:
        raise ValueError(f"Unknown condenser: {name}. Available: {sorted(CONDENSER_REGISTRY)}")
    return CONDENSER_REGISTRY[name](**cfg)


@register_condenser("truncate_mask")
class TruncateMaskCondenser(AbstractCondenser):
    """Reactive, budget-based truncate+mask condenser over uni-agent messages.

    Order: (1) truncate long observations middle-out (always); (2) mask old
    observations oldest-first up to the budget; (3) if still short, mask bulky edit
    arguments in old assistant messages. The first ``keep_start`` messages
    (system + task) and the last ``keep_last_n_messages`` are never touched.
    """

    def __init__(
        self,
        *,
        truncate_char_length: int = 5000,
        keep_last_n_messages: int = 10,
        keep_start: int = 2,
        condense_assistant_messages: bool = True,
        observation_roles: tuple[str, ...] = ("tool",),
        mask_arg_names: list[str] | None = None,
        truncated_notice: str = DEFAULT_TRUNCATED_NOTICE,
        masked_notice: str = DEFAULT_MASKED_NOTICE,
    ):
        self.truncate_char_length = truncate_char_length
        self.keep_last_n_messages = keep_last_n_messages
        self.keep_start = keep_start
        self.condense_assistant_messages = condense_assistant_messages
        self.observation_roles = tuple(observation_roles)
        self.mask_arg_names = list(mask_arg_names) if mask_arg_names is not None else list(DEFAULT_MASK_ARG_NAMES)
        self.truncated_notice = truncated_notice
        self.masked_notice = masked_notice

    def _maskable(self, idx: int, n: int) -> bool:
        return self.keep_start <= idx < n - self.keep_last_n_messages

    def condense(
        self, messages: list[dict], budget_chars: int, arg_masker: ArgMasker | None = None
    ) -> list[dict]:
        res = copy.deepcopy(messages)
        n = len(res)

        left = budget_chars

        # (1) Cap every over-long observation (keep head + tail, drop the middle).
        #     Applied to ALL observations past keep_start, *including recent ones*: a
        #     huge recent observation is often what overflowed the context, so it must be
        #     reclaimable. This is the mild reclaim (only the middle is lost) and its
        #     savings count toward the budget. Masking (steps 2-3) is the aggressive op
        #     and is the one that protects the last keep_last_n_messages.
        for idx, msg in enumerate(res):
            if idx < self.keep_start or msg.get("role") not in self.observation_roles:
                continue
            content = msg.get("content") or ""
            if len(content) > self.truncate_char_length:
                half = self.truncate_char_length // 2
                truncated = content[:half] + " [... TRUNCATED ...] " + content[-half:] + self.truncated_notice
                left -= len(content) - len(truncated)
                msg["content"] = truncated

        # (2) Mask old observations oldest-first up to the budget.
        for idx, msg in enumerate(res):
            if left <= 0:
                break
            if msg.get("role") in self.observation_roles and self._maskable(idx, n):
                content = msg.get("content") or ""
                if content == self.masked_notice:
                    continue
                left -= len(content)
                msg["content"] = self.masked_notice

        # (3) If still short, mask bulky edit arguments in old assistant messages.
        #     Delegated to ``arg_masker`` (the tool parser knows the format); skipped
        #     when no masker is provided.
        if self.condense_assistant_messages and arg_masker is not None and self.mask_arg_names:
            for idx, msg in enumerate(res):
                if left <= 0:
                    break
                if msg.get("role") != "assistant" or not self._maskable(idx, n):
                    continue
                content = msg.get("content") or ""
                new_content, removed = arg_masker(content, self.mask_arg_names, "<MASKED>")
                if removed:
                    msg["content"] = new_content
                    left -= removed

        if left > 0:
            raise CondensationFailed(
                f"Could not condense {budget_chars} chars (still {left} short after truncate+mask)."
            )

        return res
