"""Every rollout's run.log sink goes when the rollout does.

A sink is a writer thread, a queue and an open file, and every log call on the worker evaluates
every sink's filter on the event loop. The agent loop workers live for the whole job, so a sink
left behind per rollout is thousands by the late steps. Removing it must not cost run.log its tail.
"""

from __future__ import annotations

import asyncio
import threading
import types

import pytest

from uni_agent import agent_loop as agent_loop_module
from uni_agent import async_logging
from uni_agent.agent_loop import UniAgentLoop
from uni_agent.async_logging import get_logger

TAIL = 500


def _sinks():
    """Our registry's runs, and loguru's writer threads: one per enqueued sink still open."""
    writers = sum(t.name.startswith("loguru-writer") for t in threading.enumerate())
    return len(async_logging._handler_registry), writers


class _Env:
    """Logs a burst on close, as the real env does, so the sink's queue is full when it is removed."""

    def __init__(self, loop):
        self.loop = loop
        self.sinks_open = None

    async def clear_attached(self):
        pass

    async def close(self):
        self.sinks_open = _sinks()
        env_logger = get_logger("env", run_id=self.loop.run_id)
        for i in range(TAIL):
            env_logger.info(f"shutdown line {i}")


class _Interaction:
    # the bounds the episode backstop is derived from
    max_turns, action_timeout, attached_kill_timeout, episode_timeout = 1, 1, 1, None
    fail = False

    def __init__(self, **_kwargs):
        pass

    async def run(self):
        if self.fail:
            raise RuntimeError("the agent loop broke")
        return {"trajectory": [], "execution_time": 0.0, "messages": [], "rollout_cache": {"response_mask": []}}


def _loop(tmp_path, monkeypatch, fail):
    monkeypatch.setattr(UniAgentLoop, "_semaphore", None)
    monkeypatch.setattr(agent_loop_module, "AgentInteraction", type("I", (_Interaction,), {"fail": fail}))
    loop = UniAgentLoop.__new__(UniAgentLoop)
    loop.config = types.SimpleNamespace(
        actor_rollout_ref=types.SimpleNamespace(
            model=types.SimpleNamespace(path="m"),
            rollout=types.SimpleNamespace(agent=types.SimpleNamespace(num_workers=1)),
        )
    )
    config = {"model": {}, "tools": [], "env": {}, "interaction": {}, "reward": None, "log_dir": str(tmp_path)}

    async def done(*_args, **_kwargs):
        return ["output"]

    async def failed(*_args, **_kwargs):
        return "failed"

    async def no_op(*_args, **_kwargs):
        pass

    loop._init_config = lambda *_args, **_kwargs: config
    loop._init_chat_model = lambda _cfg: types.SimpleNamespace(max_completion_tokens=8, max_model_len=64)
    loop._init_tools_manager = lambda **_kwargs: None
    loop._init_skills_manager = lambda _cfg: None
    loop._init_condense = lambda _cfg: (None, {})
    loop._init_env = lambda _cfg: _Env(loop)
    loop._start_env = no_op
    loop.convert_to_agent_output = done
    loop._failed_output = failed
    return loop


@pytest.mark.parametrize("fail", [False, True], ids=["success", "exception"])
def test_the_sink_is_removed_and_run_log_keeps_its_tail(tmp_path, monkeypatch, fail):
    before = _sinks()
    loop = _loop(tmp_path, monkeypatch, fail)

    out = asyncio.run(loop.run({}, raw_prompt=[]))

    assert out == (["failed"] if fail else ["output"])
    assert loop.env.sinks_open == (before[0] + 1, before[1] + 1), "the rollout had its own sink"
    assert _sinks() == before, "the rollout's sink outlived it"
    lines = (tmp_path / loop.run_id / "run.log").read_text().splitlines()
    assert lines[-1].endswith(f"shutdown line {TAIL - 1}"), "run.log lost its tail"
    assert sum("shutdown line" in line for line in lines) == TAIL


def test_a_failed_removal_does_not_cost_the_rollout(tmp_path, monkeypatch):
    def broken(_run_id):
        raise OSError("disk full")

    monkeypatch.setattr(agent_loop_module, "cleanup_handlers", broken)
    loop = _loop(tmp_path, monkeypatch, fail=False)
    try:
        assert asyncio.run(loop.run({}, raw_prompt=[])) == ["output"]
    finally:
        async_logging.cleanup_handlers(loop.run_id)
