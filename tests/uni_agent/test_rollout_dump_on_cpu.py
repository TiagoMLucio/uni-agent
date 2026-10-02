"""The rollout dump is best effort and written off the event loop.

It goes to the shared filesystem, where a write can stall or fail. On the event loop a stall
holds every other rollout of the worker; inside the rollout's ``try`` a failure turned a scored,
hinted trajectory into an ``agent_loop_failed`` dummy row.
"""

from __future__ import annotations

import asyncio
import json
import threading
import types

from uni_agent import agent_loop as agent_loop_module
from uni_agent import async_logging
from uni_agent.agent_loop import UniAgentLoop


class _Env:
    privileged_context = ""

    async def clear_attached(self):
        pass

    async def close(self):
        pass


class _Interaction:
    def __init__(self, **_kwargs):
        pass

    async def run(self):
        return {"trajectory": [], "execution_time": 0.0, "messages": [], "rollout_cache": {"response_mask": []}}


def _loop(tmp_path, monkeypatch):
    monkeypatch.setattr(UniAgentLoop, "_semaphore", None)
    monkeypatch.setattr(agent_loop_module, "AgentInteraction", _Interaction)
    loop = UniAgentLoop.__new__(UniAgentLoop)
    loop.config = types.SimpleNamespace(
        actor_rollout_ref=types.SimpleNamespace(
            model=types.SimpleNamespace(path="m"),
            rollout=types.SimpleNamespace(agent=types.SimpleNamespace(num_workers=1)),
        )
    )
    config = {"model": {}, "tools": [], "env": {}, "interaction": {}, "reward": None, "log_dir": str(tmp_path)}

    async def converted(interaction_result):
        return [("scored", interaction_result["reward_score"])]

    async def no_op(*_args, **_kwargs):
        pass

    loop._init_config = lambda *_args, **_kwargs: config
    loop._init_chat_model = lambda _cfg: None
    loop._init_tools_manager = lambda **_kwargs: types.SimpleNamespace(tools=[])
    loop._init_skills_manager = lambda _cfg: None
    loop._init_condense = lambda _cfg: (None, {})
    loop._init_env = lambda _cfg: _Env()
    loop._start_env = no_op
    loop.convert_to_agent_output = converted
    return loop


def _run(loop):
    try:
        return asyncio.run(loop.run({}, raw_prompt=[]))
    finally:
        async_logging.cleanup_handlers(loop.run_id)


def test_the_dump_is_written_off_the_event_loop(tmp_path, monkeypatch):
    loop = _loop(tmp_path, monkeypatch)
    writers = []
    save = UniAgentLoop._save_interaction_result

    def recording_save(self, interaction_result):
        writers.append(threading.current_thread() is threading.main_thread())
        save(self, interaction_result)

    monkeypatch.setattr(UniAgentLoop, "_save_interaction_result", recording_save)
    assert _run(loop) == [("scored", -100)]
    assert writers == [False]
    dumped = json.loads((tmp_path / loop.run_id / "interaction_result.json").read_text())
    assert dumped["reward_score"] == -100


def test_a_failed_dump_keeps_the_scored_row(tmp_path, monkeypatch):
    def full_disk(self, interaction_result):
        raise OSError(122, "Disk quota exceeded")

    monkeypatch.setattr(UniAgentLoop, "_save_interaction_result", full_disk)
    loop = _loop(tmp_path, monkeypatch)
    assert _run(loop) == [("scored", -100)], "the dump failure must not replace the row with a dummy"
    assert "Rollout dump not written" in (tmp_path / loop.run_id / "run.log").read_text()
