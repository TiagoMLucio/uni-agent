"""Unit tests for the TruncateMaskCondenser."""

from __future__ import annotations

import copy

import pytest

from uni_agent.interaction.condenser import (
    CondensationFailed,
    TruncateMaskCondenser,
    load_condenser,
)
from uni_agent.interaction.tool_parser import HermesToolParser, XMLToolParser


def _msgs(n_turns: int, obs_len: int = 50) -> list[dict]:
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    for i in range(n_turns):
        msgs.append({"role": "assistant", "content": f"act{i}"})
        msgs.append({"role": "tool", "content": "O" * obs_len + f"#{i}"})
    return msgs


def test_truncate_long_observation_middle_out():
    c = TruncateMaskCondenser(truncate_char_length=20, keep_last_n_messages=0)
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "act"},
        {"role": "tool", "content": "HEAD" + "x" * 100 + "TAIL"},
    ]
    out = c.condense(msgs, budget_chars=0)
    obs = out[-1]["content"]
    assert "[... TRUNCATED ...]" in obs
    assert obs.startswith("HEAD") and "TAIL" in obs
    assert len(obs) < len("HEAD" + "x" * 100 + "TAIL")


def test_truncate_does_not_touch_task_or_short_obs():
    c = TruncateMaskCondenser(truncate_char_length=5, keep_last_n_messages=0)
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "a long task description that exceeds five chars"},
        {"role": "assistant", "content": "act"},
        {"role": "tool", "content": "tiny"},
    ]
    out = c.condense(msgs, budget_chars=0)
    assert out[1]["content"] == msgs[1]["content"]  # task untouched (idx < keep_start)
    assert out[3]["content"] == "tiny"  # short obs untouched


def test_truncation_counts_toward_budget():
    # Truncating the long observation frees more than the budget, so step 2 masking
    # must NOT additionally fire (truncation savings count -> no over-masking).
    c = TruncateMaskCondenser(truncate_char_length=20, keep_last_n_messages=0)
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "a0"},
        {"role": "tool", "content": "O" * 500},  # idx3: truncated, frees >> budget
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "short obs"},  # idx5
    ]
    out = c.condense(msgs, budget_chars=100)
    assert "[... TRUNCATED ...]" in out[3]["content"]  # truncated
    assert out[3]["content"] != c.masked_notice  # but NOT masked: budget already met by truncation
    assert out[5]["content"] == "short obs"  # untouched


def test_mask_old_observations_oldest_first_within_budget():
    c = TruncateMaskCondenser(keep_last_n_messages=2, truncate_char_length=10_000)
    msgs = _msgs(3, obs_len=50)  # [sys,task,a0,t0,a1,t1,a2,t2], n=8; maskable tools: t0(idx3), t1(idx5)
    out = c.condense(msgs, budget_chars=40)  # < len(t0) ~ 52 -> masks only t0
    assert out[3]["content"] == c.masked_notice
    assert out[5]["content"] != c.masked_notice  # t1 not yet masked
    assert out[7]["content"] != c.masked_notice  # t2 protected (last n)


def test_keep_last_n_protects_recent_observations():
    c = TruncateMaskCondenser(keep_last_n_messages=4, truncate_char_length=10_000)
    msgs = _msgs(3, obs_len=50)  # n=8; maskable idx in [2, 4) -> only t0(idx3)
    # only t0 is maskable; a huge budget cannot be met -> CondensationFailed
    with pytest.raises(CondensationFailed):
        c.condense(msgs, budget_chars=10_000_000)
    # a small budget masks just t0; t1 (idx5) and t2 (idx7) are protected by keep_last_n
    out = c.condense(msgs, budget_chars=10)
    assert out[3]["content"] == c.masked_notice
    assert out[5]["content"] != c.masked_notice and out[7]["content"] != c.masked_notice


def test_edit_arg_masking_qwen_via_parser():
    c = TruncateMaskCondenser(keep_last_n_messages=0, truncate_char_length=10_000)
    big = "X" * 200
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": f"<function=str_replace_editor><parameter=old_str>{big}</parameter>"},
        {"role": "tool", "content": "obs"},
    ]
    # masking is delegated to the parser (qwen3_coder XML form)
    out = c.condense(msgs, budget_chars=100, arg_masker=XMLToolParser().mask_arguments)
    assert "<parameter=old_str><MASKED></parameter>" in out[2]["content"]


def test_edit_arg_masking_hermes_via_parser():
    c = TruncateMaskCondenser(keep_last_n_messages=0, truncate_char_length=10_000)
    big = "X" * 200
    tool_call = '<tool_call>\n{"name": "str_replace_editor", "arguments": {"old_str": "%s"}}\n</tool_call>' % big
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": tool_call},
        {"role": "tool", "content": "obs"},
    ]
    out = c.condense(msgs, budget_chars=100, arg_masker=HermesToolParser().mask_arguments)
    assert '"old_str": "<MASKED>"' in out[2]["content"]
    assert big not in out[2]["content"]


def test_step3_skipped_without_arg_masker():
    # Without an arg_masker, step 3 cannot run; if observation masking is insufficient,
    # condensation fails rather than touching assistant content.
    c = TruncateMaskCondenser(keep_last_n_messages=0, truncate_char_length=10_000)
    big = "X" * 200
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": f"<function=str_replace_editor><parameter=old_str>{big}</parameter>"},
        {"role": "tool", "content": "obs"},
    ]
    with pytest.raises(CondensationFailed):
        c.condense(msgs, budget_chars=100)  # no arg_masker -> step 3 skipped


def test_condensation_failed_when_budget_unreachable():
    c = TruncateMaskCondenser(keep_last_n_messages=0, truncate_char_length=10_000)
    msgs = _msgs(1, obs_len=10)  # tiny, little to mask
    with pytest.raises(CondensationFailed):
        c.condense(msgs, budget_chars=10_000)


def test_does_not_mutate_input():
    c = TruncateMaskCondenser(keep_last_n_messages=0, truncate_char_length=5)
    msgs = _msgs(2, obs_len=100)
    snapshot = copy.deepcopy(msgs)
    c.condense(msgs, budget_chars=50)
    assert msgs == snapshot


def test_registry_load():
    assert load_condenser(None) is None
    assert load_condenser({}) is None
    c = load_condenser({"name": "truncate_mask", "truncate_char_length": 123})
    assert isinstance(c, TruncateMaskCondenser) and c.truncate_char_length == 123
    with pytest.raises(ValueError):
        load_condenser({"name": "nope"})
    with pytest.raises(ValueError):
        load_condenser({"truncate_char_length": 1})  # missing name
