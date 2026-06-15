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


def test_over_limit_is_clipped_with_positive_elided_count():
    raw = "a" * 50
    out = AgentEnv._format_observation(raw, 10, EMPTY)
    assert out.startswith("Observation:\n" + "a" * 10 + "<response clipped>")
    assert "40 were elided" in out  # 50 - 10, never negative
    assert "<NOTE>" in out
