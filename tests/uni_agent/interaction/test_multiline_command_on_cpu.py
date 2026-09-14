"""A heredoc must never reach the shell's pty as several lines.

bashlex joins every other multi-line command with ``;``, but a heredoc stays multi-line, and
bash then prints one prompt more than the runtime consumes. The agent's commands run with
``check="ignore"``, which returns after the first prompt, so the extra one stays buffered and
every later observation is the previous command's output for the rest of the session. In the
reference run that was 7.2% of all ``view`` calls and 27% of trajectories, and all 72 desynced
observations followed that trajectory's first heredoc.
"""

from __future__ import annotations

import asyncio
import types

import pytest
from swerex.runtime.abstract import BashAction, CreateBashSessionRequest

from uni_agent.deployment.local_native.runtime import BashSession
from uni_agent.interaction.env import MULTILINE_COMMAND_PATH, AgentEnv, as_single_line

HEREDOC = "python3 << 'EOF'\nimport sys\n\nprint('HEREDOC')\nEOF"


def test_single_line_commands_are_left_alone():
    assert as_single_line("cd /tmp && echo X") is None
    assert as_single_line("printf 'a\\nb\\n'") is None  # an escaped newline is not a newline


def test_a_heredoc_becomes_a_file_and_one_line():
    content, line = as_single_line(HEREDOC)
    assert content == HEREDOC + "\n"
    assert "\n" not in line
    assert line == f"bash -n {MULTILINE_COMMAND_PATH} && source {MULTILINE_COMMAND_PATH}"


class _Runtime:
    def __init__(self):
        self.sent = []
        self.written = {}

    async def run_in_session(self, action):
        self.sent.append(action.command)
        return types.SimpleNamespace(output="ok", exit_code=0)

    async def write_file(self, request):
        self.written[request.path] = request.content


def _env():
    e = AgentEnv.__new__(AgentEnv)  # __init__ needs a real deployment
    noop = lambda *a, **k: None  # noqa: E731
    e.logger = types.SimpleNamespace(info=noop, error=noop, critical=noop, debug=noop)
    e.deployment = types.SimpleNamespace(runtime=_Runtime())
    return e


def test_communicate_sends_the_heredoc_through_the_file_api():
    env = _env()
    asyncio.run(AgentEnv.communicate.__wrapped__(env, HEREDOC, check="ignore"))
    rt = env.deployment.runtime
    assert rt.written == {MULTILINE_COMMAND_PATH: HEREDOC + "\n"}
    assert rt.sent == [f"bash -n {MULTILINE_COMMAND_PATH} && source {MULTILINE_COMMAND_PATH}"]


def test_communicate_leaves_single_lines_untouched():
    env = _env()
    asyncio.run(AgentEnv.communicate.__wrapped__(env, "echo X", check="ignore"))
    rt = env.deployment.runtime
    assert rt.written == {} and rt.sent == ["echo X"]


async def _observations(commands, path):
    """What the real bash session answers to each command, as the tool would send them."""
    s = BashSession(CreateBashSessionRequest(), run_id="test")
    await s.start()
    out = []
    try:
        for c in commands:
            rewritten = as_single_line(c, path)
            if rewritten is not None:
                content, c = rewritten
                open(path, "w").write(content)
            out.append((await s.run(BashAction(command=c, timeout=10, check="ignore"))).output)
    finally:
        await s.close()
    return out


@pytest.mark.timeout(60)
def test_the_session_stays_in_sync_after_a_heredoc(tmp_path):
    path = str(tmp_path / "cmd.sh")
    obs = asyncio.run(_observations([HEREDOC, "echo AFTER-1", "echo AFTER-2", "cd /tmp && pwd"], path))
    assert "HEREDOC" in obs[0]
    assert "AFTER-1" in obs[1] and "AFTER-2" in obs[2]
    assert obs[3].strip().endswith("/tmp")  # sourcing keeps `cd` in the session
    assert not any("SHELLPS1PREFIX" in o for o in obs)


@pytest.mark.timeout(60)
def test_without_the_rewrite_the_session_desyncs():
    # documents the failure the rewrite exists for; if this starts passing, swe-rex fixed it
    async def raw():
        s = BashSession(CreateBashSessionRequest(), run_id="test")
        await s.start()
        try:
            out = [(await s.run(BashAction(command=c, timeout=10, check="ignore"))).output
                   for c in [HEREDOC, "echo AFTER-1", "echo AFTER-2"]]
        finally:
            await s.close()
        return out
    obs = asyncio.run(raw())
    assert "AFTER-1" not in obs[1] and "AFTER-1" in obs[2]
