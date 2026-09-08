"""A git subcommand that discards the working tree never reaches the session.

The graded patch is taken from the working tree, so `git commit` hides a correct fix behind HEAD
and stash, reset, checkout, restore and clean throw it away. Agents reach for them: across 8000
reference rollouts, 14% ran `git stash` and 10% ran `git checkout`.

The block used to live in a script (`BLOCKED_BASH_COMMANDS`) that the dispatch short-circuits and
never ran, and it read only the first word of the line, which is why `cd /testbed && git stash`
would have passed it twice over.
"""

from __future__ import annotations

import asyncio
import types

import pytest

from uni_agent.interaction.behaviour import behaviour_metrics
from uni_agent.interaction.interaction import AgentInteraction, ToolResult
from uni_agent.interaction.tools_manager import (
    DESTRUCTIVE_GIT_SUBCOMMANDS,
    destructive_git_subcommand,
)

ALLOWED = ["git diff", "git log --oneline -5", "git show HEAD", "git status", "git blame f.py",
           "git diff --cached", "git add -A", "git apply /tmp/p.diff", "git rev-parse HEAD"]


@pytest.mark.parametrize("sub", sorted(DESTRUCTIVE_GIT_SUBCOMMANDS))
def test_every_destructive_subcommand_is_caught(sub):
    assert destructive_git_subcommand(f"git {sub}") == sub


@pytest.mark.parametrize("command", ALLOWED)
def test_reading_the_repository_stays_allowed(command):
    assert destructive_git_subcommand(command) is None


def test_git_diff_survives_next_to_a_blocked_one():
    assert destructive_git_subcommand("git diff") is None
    assert destructive_git_subcommand("git diff && git checkout .") == "checkout"


@pytest.mark.parametrize("command", [
    "cd /testbed && git stash",
    "cd /testbed; git reset --hard",
    "git diff || git checkout -- .",
    "ls | grep py; git clean -fd",
    "cd /testbed\ngit restore src/a.py",
    "git diff --name-only | xargs git checkout --",
    "find . -name '*.py' -exec git checkout {} ;",
    "echo $(git stash)",
    "echo `git commit -m wip`",
    "GIT_PAGER=cat git checkout .",
    "git -C /testbed stash",
    "git -c user.name=x commit -m wip",
])
def test_composition_does_not_get_past_it(command):
    assert destructive_git_subcommand(command) in DESTRUCTIVE_GIT_SUBCOMMANDS


@pytest.mark.parametrize("command", [
    "cd /testbed && git diff",
    "cd /testbed && python -m pytest",
    "grep -rn checkout .",
    "echo 'git stash'",
    "python -c \"print('git reset')\"",
])
def test_the_ordinary_command_is_left_alone(command):
    assert destructive_git_subcommand(command) is None


def _interaction():
    it = AgentInteraction.__new__(AgentInteraction)
    noop = lambda *a, **k: None  # noqa: E731
    it.logger = types.SimpleNamespace(info=noop, error=noop, debug=noop, warning=noop)
    it.env = types.SimpleNamespace(attached_command=None, attached_seconds=0.0, attached_at_prompt=False)
    it.action_timeout, it.yield_timeout, it.attached_kill_timeout = 30, 5, 60.0
    it.max_observation_length, it.timeout_budget = 100_000, 60.0
    return it


def _call(command: str, is_input: bool = False):
    it = _interaction()
    it.tools_manager = types.SimpleNamespace(
        get_tool_action=lambda _tc: types.SimpleNamespace(command=command, is_input=is_input, timeout=None),
        format_args_example=lambda args: str(args),
    )
    tc = types.SimpleNamespace(id="c1", function=types.SimpleNamespace(name="execute_bash"))
    return asyncio.run(AgentInteraction._execute_tool_call(it, tc))


def test_the_refusal_is_an_observation_the_model_can_act_on():
    result = _call("cd /testbed && git stash")
    assert result.status == "syntax_error", "skipped would trip the terminal_dead abort"
    assert "NOT executed" in result.observation
    assert "git stash" in result.observation, "it has to name what was refused"
    assert "working tree" in result.observation, "and why"
    assert "editor" in result.observation, "and what to do instead"
    assert "git diff" in result.observation, "and that reading the repository still works"


def test_a_refused_call_never_reaches_the_session():
    env_calls = []
    it = _interaction()
    it.env.run_action = lambda *a, **k: env_calls.append(a)
    it.tools_manager = types.SimpleNamespace(
        get_tool_action=lambda _tc: types.SimpleNamespace(command="git reset --hard", is_input=False, timeout=None),
        format_args_example=lambda args: str(args),
    )
    tc = types.SimpleNamespace(id="c1", function=types.SimpleNamespace(name="execute_bash"))
    asyncio.run(AgentInteraction._execute_tool_call(it, tc))
    assert env_calls == []


def _step(status: str, action: str, idx: int = 1):
    return types.SimpleNamespace(
        step_idx=idx, exit_reason="completed_with_tool_errors",
        tool_results=[ToolResult(tool_call_id="c", name="execute_bash", action=action,
                                 observation="refused", status=status, execution_time=0.0)],
    )


def test_the_refusals_are_counted_per_trajectory():
    metrics = behaviour_metrics([_step("syntax_error", "git stash"), _step("ok", "git diff", idx=2)])
    assert metrics["git_refusals"] == 1.0
    assert metrics["tool_calls"] == 2.0


def test_the_counter_is_reported_even_when_nothing_fired():
    """Never conditional: a run that stopped reaching for git has to read as zero, not as absent."""
    assert behaviour_metrics([_step("ok", "git diff")])["git_refusals"] == 0.0


def test_the_refusals_do_not_enter_the_edit_accounting():
    """`edit_calls_run` is the editor-call denominator; a refused git call must not touch it."""
    metrics = behaviour_metrics([_step("syntax_error", "git checkout .")])
    assert (metrics["edit_attempts"], metrics["edit_calls_run"], metrics["edit_failures"]) == (0.0, 0.0, 0.0)


REWARD_EXTRACTION = (
    "cd /testbed && printf '*.py diff=python\\n' > /tmp/.uniagent_gitattributes && git add -A && "
    "(git diff --cached --numstat | awk -F'\\t' '$1==\"-\"{print $3}' "
    "| xargs -r -d '\\n' git reset -q --) ; "
    "git -c core.attributesFile=/tmp/.uniagent_gitattributes diff --no-color --cached > /tmp/patch.diff"
)


def test_the_graders_own_command_would_be_refused_if_it_ever_took_this_path():
    """It contains `git reset`, so the guard must stay on the tool-call path and nowhere else.

    The reward runs this through ``communicate_isolated``, a side session that never touches
    ``ToolsManager``; ``get_tool_action`` has one caller, ``_execute_tool_call``.
    """
    assert destructive_git_subcommand(REWARD_EXTRACTION) == "reset"


def test_the_side_session_runs_a_command_verbatim():
    from uni_agent.interaction.env import AgentEnv

    sent = []

    class _Runtime:
        async def create_session(self, request):
            return None

        async def run_in_session(self, action):
            sent.append(action.command)
            return types.SimpleNamespace(output="")

    env = AgentEnv.__new__(AgentEnv)
    env._isolated_session = None
    env.deployment = types.SimpleNamespace(runtime=_Runtime())
    asyncio.run(AgentEnv.communicate_isolated.__wrapped__(env, REWARD_EXTRACTION))
    assert sent == [REWARD_EXTRACTION]
