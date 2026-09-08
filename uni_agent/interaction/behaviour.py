"""What the agent did, counted off a finished trajectory.

Reported under ``agent/``, a prefix the trainer aggregates per step without reading the names.
Everything here is a per-trajectory count, so a pooled rate is the ratio of two step means
(failed edits per attempt, repeated turns per acting turn, one tool status per call).
"""

import re
from typing import TYPE_CHECKING, get_args

from uni_agent.interaction.interaction import ToolStatus
from uni_agent.interaction.tools_manager import destructive_git_subcommand

if TYPE_CHECKING:
    from uni_agent.interaction.interaction import StepOutput

ARG_PATH_RE = re.compile(r"--path\s+(\S+)")
CMD_RE = re.compile(r"str_replace_editor\s+(\w+)")
EDIT_CMDS = frozenset({"str_replace", "insert", "create"})

#: A refused edit prints its reason and exits 0 exactly like an applied one, so it is legible
#: only as the absence of what an applied one prints.
EDIT_DONE = (" has been edited. ", "File created successfully at: ")


def edit_command(action: str) -> str | None:
    """The editor subcommand an action runs, or ``None`` when it is not an editor call."""
    match = CMD_RE.match(action)
    return match[1] if match else None


def edit_path(action: str) -> str | None:
    match = ARG_PATH_RE.search(action)
    return match[1] if match else None


def is_source_path(path: str | None) -> bool:
    """Whether a path is the code under repair rather than a test or a reproducer.

    On the basename, not the path: every task lives under ``/testbed/``, so matching "test"
    anywhere marks every file a test.
    """
    name = path.rsplit("/", 1)[-1] if path else ""
    return bool(name) and "reproduce" not in name and "test" not in name


def behaviour_metrics(trajectory: list["StepOutput"]) -> dict[str, float]:
    """Tool-call outcomes, edit attempts and refusals, malformed turns, turns whose calls
    repeat an earlier turn's, and the tail after the last turn that changed source.

    ``idle_turns_after_edit`` is absent when no edit to source ever landed, so its step mean is
    read over the trajectories that changed source and ``source_edited`` is that share. Without
    the share, a run that drives trajectories to stop editing reads as a shorter idle tail.

    ``edit_failures`` is out of ``edit_calls_run``, not out of ``edit_attempts``: the difference
    is the calls the harness refused before the editor saw them.

    ``git_refusals`` is out of ``tool_calls``: blocking ``checkout`` takes a move the policy makes
    today, so whether it fires has to be measured rather than assumed.
    """
    steps: dict[int, StepOutput] = {}
    for step in trajectory:
        steps.setdefault(step.step_idx, step)  # the cap and stuck sentinels reuse the last index
    ordered = [steps[idx] for idx in sorted(steps)]

    out = dict.fromkeys(
        ("tool_calls", "edit_attempts", "edit_calls_run", "edit_failures", "format_errors",
         "acting_turns", "repeated_turns", "source_edit_turns", "source_edited", "git_refusals"), 0
    )
    out.update({f"tool_{status}": 0 for status in get_args(ToolStatus)})
    seen: set[int] = set()
    last_source_edit = None
    for step in ordered:
        if step.exit_reason == "format_error":
            out["format_errors"] += 1
        if not step.tool_results:
            continue
        out["acting_turns"] += 1
        # the calls themselves, not the reasoning around them: hashed so the trajectory's
        # own strings are the only copy kept
        key = hash("\n".join(f"{call.name} {call.action.strip()}" for call in step.tool_results))
        if key in seen:
            out["repeated_turns"] += 1
        seen.add(key)
        edited_source = False
        for call in step.tool_results:
            out["tool_calls"] += 1
            out[f"tool_{call.status}"] += 1
            if destructive_git_subcommand(call.action):
                out["git_refusals"] += 1
            if edit_command(call.action) not in EDIT_CMDS:
                continue
            out["edit_attempts"] += 1
            # a call the harness itself refused never reached the editor, so it is not an edit
            # outcome; its own tool status already carries it
            if call.status != "ok":
                continue
            out["edit_calls_run"] += 1
            if not any(done in call.observation for done in EDIT_DONE):
                out["edit_failures"] += 1
            else:
                edited_source |= is_source_path(edit_path(call.action))
        if edited_source:
            out["source_edit_turns"] += 1
            last_source_edit = step.step_idx
    if last_source_edit is not None:
        out["source_edited"] = 1
        out["idle_turns_after_edit"] = sum(1 for step in ordered if step.step_idx > last_source_edit)
    return {name: float(value) for name, value in out.items()}
