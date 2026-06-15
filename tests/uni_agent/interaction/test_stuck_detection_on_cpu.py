"""Unit tests for AgentInteraction stuck detection (exact-repeat loop)."""

from __future__ import annotations

from uni_agent.interaction.interaction import AgentInteraction, StepOutput


def _interaction(stuck_threshold: int, responses: list[str | None]) -> AgentInteraction:
    """Build a bare AgentInteraction (no env/model) exercising only check_stuck."""
    it = AgentInteraction.__new__(AgentInteraction)
    it.stuck_threshold = stuck_threshold
    trajectory = []
    for response in responses:
        step = StepOutput(step_idx=0)
        step.response = response or ""
        trajectory.append(step)
    it.trajectory = trajectory
    return it


def test_stuck_when_tail_responses_identical():
    it = _interaction(3, ["a", "b", "x", "x", "x"])
    assert it.check_stuck() is True


def test_not_stuck_when_tail_differs():
    it = _interaction(3, ["x", "x", "y"])
    assert it.check_stuck() is False


def test_exactly_threshold_is_stuck():
    it = _interaction(3, ["x", "x", "x"])
    assert it.check_stuck() is True


def test_fewer_than_threshold_not_stuck():
    it = _interaction(3, ["x", "x"])
    assert it.check_stuck() is False


def test_threshold_zero_disables_detection():
    it = _interaction(0, ["x", "x", "x", "x", "x"])
    assert it.check_stuck() is False


def test_empty_responses_are_skipped():
    it = _interaction(3, [None, None, "x", "x", "x"])
    assert it.check_stuck() is True
