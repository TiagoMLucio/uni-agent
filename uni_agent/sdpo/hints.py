"""Reflector hints paired with the turns they land on, and their chat-template rendering."""

from typing import NamedTuple, Optional

__all__ = [
    "HintedTurn",
    "assistant_header_ids",
    "hint_user_turn_ids",
    "select_hinted_turns",
]


class HintedTurn(NamedTuple):
    """One reflection hint paired with the turn it lands on: ``[start, end)`` on the response
    grid, spliced as a user turn before the turn's assistant header."""

    step: int
    start: int
    end: int
    text: str


def select_hinted_turns(
    extra_fields: dict, response_len: int, max_hinted_turns: Optional[int] = None
) -> list[HintedTurn]:
    """Pair a sample's turn spans with its hints. A ``turn_hints`` entry is ``[step, text]``.

    Spans are clamped to the (possibly truncated) response; with a cap, the first
    ``max_hinted_turns`` turns are kept (earliest, before the trajectory loses coherence).
    """
    hint_by_step = {int(entry[0]): entry[1] for entry in (extra_fields.get("turn_hints") or [])}
    hinted = []
    for step, start, end in extra_fields.get("turn_spans") or []:
        step, start, end = int(step), int(start), min(int(end), response_len)
        if step in hint_by_step and start < end:
            hinted.append(HintedTurn(step, start, end, hint_by_step[step]))
    if max_hinted_turns is not None and len(hinted) > max_hinted_turns:
        hinted = hinted[:max_hinted_turns]
    return hinted


# Render-suffix over this probe yields the exact mid-conversation fragment (auto system blocks cancel in the prefix).
_TEMPLATE_PROBE = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
# User-turn fragments use a user-only probe: templates that re-render a no-longer-final
# assistant turn (Qwen3.5 drops its empty think block) break the two-turn probe's prefix
# property, while a trailing user turn never changes how the probe itself renders.
_TEMPLATE_PROBE_USER = [{"role": "user", "content": "x"}]


def _template_suffix(
    tokenizer, messages=(), add_generation_prompt=False, probe=_TEMPLATE_PROBE, template_kwargs=None
) -> str:
    kwargs = dict(template_kwargs or {})
    base = tokenizer.apply_chat_template(list(probe), tokenize=False, add_generation_prompt=False, **kwargs)
    full = tokenizer.apply_chat_template(
        list(probe) + list(messages), tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs
    )
    assert full.startswith(base), "chat template does not render conversations as extendable prefixes"
    return full[len(base) :]


def assistant_header_ids(tokenizer, template_kwargs=None) -> list[int]:
    """Token ids of the template's assistant generation header (e.g. ``<|im_start|>assistant\\n``).

    ``template_kwargs`` must match the kwargs the rollout passed to ``apply_chat_template``
    (e.g. ``{"enable_thinking": False}``), or the header will not match the rollout tokens.
    """
    return tokenizer.encode(
        _template_suffix(tokenizer, add_generation_prompt=True, template_kwargs=template_kwargs),
        add_special_tokens=False,
    )


def hint_user_turn_ids(tokenizer, hint_text: str, template_kwargs=None) -> list[int]:
    """Token ids of ``hint_text`` rendered as a full user turn of the tokenizer's chat template."""
    return tokenizer.encode(
        _template_suffix(
            tokenizer,
            messages=[{"role": "user", "content": hint_text}],
            probe=_TEMPLATE_PROBE_USER,
            template_kwargs=template_kwargs,
        ),
        add_special_tokens=False,
    )
