"""Unit tests for AgentEnv._format_observation (shared observation formatting)."""

from __future__ import annotations

from uni_agent.interaction.env import AgentEnv

EMPTY = "no output"


def test_empty_returns_empty_message():
    assert AgentEnv._format_observation("", 1000, EMPTY) == EMPTY


def test_whitespace_and_ansi_only_returns_empty_message():
    # ANSI color codes + carriage returns + whitespace strip to nothing.
    assert AgentEnv._format_observation("\x1b[31m\r  \r\n", 1000, EMPTY) == EMPTY


def test_normal_output_is_prefixed():
    assert AgentEnv._format_observation("hello", 1000, EMPTY) == "Observation:\nhello"


def test_ansi_is_stripped_from_normal_output():
    assert AgentEnv._format_observation("\x1b[32mok\x1b[0m\r", 1000, EMPTY) == "Observation:\nok"


def test_over_limit_keeps_both_ends_with_positive_elided_count():
    raw = "a" * 25 + "b" * 25
    out = AgentEnv._format_observation(raw, 10, EMPTY)
    assert out.startswith("Observation:\n" + "a" * 5)
    assert "40 characters elided from the middle" in out  # 50 - 10, never negative
    assert out.split("<response clipped")[1].endswith("b" * 5 + "\n<NOTE>" + out.split("<NOTE>")[1])
    assert "<NOTE>" in out


def test_over_limit_preserves_the_tail():
    # a test run's verdict is its last line; head-only truncation deletes it
    raw = "noise\n" * 500 + "3 failed, 12 passed"
    out = AgentEnv._format_observation(raw, 200, EMPTY)
    assert "3 failed, 12 passed" in out
