"""The per-trajectory behaviour counters, on a hand-built trajectory: one status key per
``ToolStatus``, an edit counted refused because the editor printed a refusal and exited like a
success, and the repetition and idle-tail counts read off the turn table."""

from __future__ import annotations

from typing import get_args

from uni_agent.interaction.behaviour import behaviour_metrics
from uni_agent.interaction.interaction import StepOutput, ToolResult, ToolStatus

MOD = "/testbed/pkg/mod.py"
EDIT = f"str_replace_editor str_replace --path {MOD} --old_str 'a' --new_str 'b'"
APPLIED = f"The file {MOD} has been edited. Review the changes and make sure they are as expected."
REFUSED = f"No replacement was performed, old_str `a` did not appear verbatim in {MOD}."
RUN = "python reproduce_bug.py"


def call(name: str, action: str, observation: str = "", status: ToolStatus = "ok") -> ToolResult:
    return ToolResult(tool_call_id="c", name=name, action=action, observation=observation, status=status)


def step(idx: int, *tool_results: ToolResult, exit_reason: str = "completed") -> StepOutput:
    return StepOutput(step_idx=idx, tool_results=list(tool_results), exit_reason=exit_reason)


#: an edit that applies, a malformed turn, the same edit refused, a reproducer written, it run
#: twice, a step whose second call the loop skipped, a bad command and a submit
TRAJECTORY = [
    step(1, call("str_replace_editor", EDIT, APPLIED)),
    step(2, exit_reason="format_error"),
    step(3, call("str_replace_editor", EDIT, REFUSED)),
    step(4, call("str_replace_editor", "str_replace_editor create --path /testbed/reproduce_bug.py",
                 "File created successfully at: /testbed/reproduce_bug.py")),
    step(5, call("execute_bash", RUN, "1 failed")),
    step(6, call("execute_bash", RUN, "still running", status="yielded")),
    step(7, call("execute_bash", "sleep 600", "cancelled", status="timeout"),
         call("str_replace_editor", "", "Skipped: the bash session died mid-step.", status="skipped")),
    step(8, call("execute_bash", "ls", "is_input was set", status="syntax_error")),
    step(9, call("submit", "echo '<<<Finished>>>'", "<<<Finished>>>"), exit_reason="finished"),
    # the cap sentinel reuses the last real step's index and must not count as a turn
    step(9, exit_reason="max_step_limit"),
]


def test_behaviour_metrics_count_outcomes_edits_repeats_and_the_idle_tail():
    metrics = behaviour_metrics(TRAJECTORY)

    assert metrics == {
        "tool_calls": 9.0,
        "tool_ok": 5.0,
        "tool_yielded": 1.0,
        "tool_timeout": 1.0,
        "tool_syntax_error": 1.0,
        "tool_skipped": 1.0,
        # the skipped call carries no action, so it is not an edit that was attempted
        "edit_attempts": 3.0,
        "edit_calls_run": 3.0,
        "edit_failures": 1.0,
        "format_errors": 1.0,
        "acting_turns": 8.0,
        # turn 3 repeats turn 1's edit, turn 6 turn 5's command
        "repeated_turns": 2.0,
        # only turn 1 changed source: turn 3's edit was refused and the reproducer is not source
        "source_edit_turns": 1.0,
        "source_edited": 1.0,
        "idle_turns_after_edit": 8.0,
    }


def test_every_tool_status_gets_its_own_key():
    keys = set(behaviour_metrics([]))
    assert {f"tool_{status}" for status in get_args(ToolStatus)} < keys
    assert keys == set(behaviour_metrics(TRAJECTORY)) - {"idle_turns_after_edit"}


def test_the_idle_tail_is_absent_when_nothing_edited_source():
    only_tests = "str_replace_editor str_replace --path /testbed/tests/test_mod.py"
    metrics = behaviour_metrics([
        step(1, call("execute_bash", "ls", "mod.py")),
        step(2, call("str_replace_editor", only_tests, "The file /testbed/tests/test_mod.py has been edited. ")),
    ])
    assert "idle_turns_after_edit" not in metrics
    # the population the idle tail would be read over, and this trajectory is not in it
    assert (metrics["source_edited"], metrics["source_edit_turns"]) == (0.0, 0.0)
    assert (metrics["edit_attempts"], metrics["edit_calls_run"], metrics["edit_failures"]) == (1.0, 1.0, 0.0)


def test_an_edit_the_loop_never_ran_is_an_attempt_but_not_an_editor_refusal():
    """The attached-session guard refuses the call before the editor sees it: the turn is an
    attempt and a ``syntax_error``, not an edit the editor rejected and not a change to source."""
    metrics = behaviour_metrics([
        step(1, call("str_replace_editor", EDIT, "Your command is NOT executed.", status="syntax_error")),
    ])
    # the refusal rate is read out of edit_calls_run, which this call never entered
    assert (metrics["edit_attempts"], metrics["edit_calls_run"], metrics["edit_failures"]) == (1.0, 0.0, 0.0)
    assert metrics["tool_syntax_error"] == 1.0
    assert (metrics["source_edited"], "idle_turns_after_edit" in metrics) == (0.0, False)
