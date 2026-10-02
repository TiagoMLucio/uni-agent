"""A row that breaks the rollout's own setup costs a masked row, never the step.

Before the sandbox exists, a bad row (an unknown config key, a prompt template the row cannot
fill, a reward spec that will not load, a log dir that cannot be opened) used to raise out of
``run`` and take the whole gathered step with it. And the dummy row itself renders the
conversation, so when that fails too there is a last, 1-token masked row (``build_failed``).
"""

from __future__ import annotations

import asyncio
import types

import pytest

from uni_agent import agent_loop as agent_loop_module
from uni_agent import async_logging
from uni_agent.agent_loop import UniAgentLoop


class _Env:
    closed = 0

    async def clear_attached(self):
        pass

    async def close(self):
        _Env.closed += 1


class _Interaction:
    def __init__(self, messages, **_kwargs):
        self.messages = messages

    async def run(self):
        raise AssertionError("a failed setup never reaches the turn loop")


def _loop(tmp_path, monkeypatch, **config_overrides):
    """The real ``run`` over stubbed builders; ``config_overrides`` shape the row's config."""
    monkeypatch.setattr(UniAgentLoop, "_semaphore", None)
    monkeypatch.setattr(agent_loop_module, "AgentInteraction", _Interaction)
    _Env.closed = 0
    loop = UniAgentLoop.__new__(UniAgentLoop)
    loop.config = types.SimpleNamespace(
        actor_rollout_ref=types.SimpleNamespace(
            model=types.SimpleNamespace(path="m"),
            rollout=types.SimpleNamespace(
                agent=types.SimpleNamespace(num_workers=1), prompt_length=16, response_length=8
            ),
        )
    )
    loop.tokenizer = types.SimpleNamespace(pad_token_id=7, eos_token_id=9)
    config = {"model": {}, "tools": [], "env": {}, "interaction": {}, "reward": None, "log_dir": str(tmp_path)}
    config.update(config_overrides)

    async def cache(_messages):
        return {"prompt_ids": [1, 2, 3], "extra_fields": {}}

    loop._init_config = lambda *_args, **_kwargs: config
    loop._init_chat_model = lambda _cfg: types.SimpleNamespace(
        set_tools_schemas=lambda _s: None, prepare_rollout_cache=cache
    )
    loop._init_tools_manager = lambda **_kwargs: types.SimpleNamespace(tools_schemas=[], tools=[])
    loop._init_skills_manager = lambda _cfg: None
    loop._init_condense = lambda _cfg: (None, {})
    loop._init_env = lambda _cfg: _Env()
    return loop


def _run(loop, **kwargs):
    try:
        return asyncio.run(loop.run({}, raw_prompt=[{"role": "user", "content": "fix it"}], **kwargs))
    finally:
        async_logging.cleanup_handlers(loop.run_id)


def _assert_masked(row):
    assert row.response_mask == [0] * len(row.response_mask)
    assert row.reward_score == 0


def test_a_config_that_will_not_build_is_a_build_failed_row(tmp_path, monkeypatch):
    loop = _loop(tmp_path, monkeypatch)

    def bad_config(*_args, **_kwargs):
        raise ValueError("Unknown top-level agent config key(s): envv")

    loop._init_config = bad_config
    (row,) = _run(loop)
    # nothing was built to render a dummy prompt from, so the floor row ships
    assert row.extra_fields["traj_exit_reason"] == "build_failed"
    assert row.prompt_ids == [7] and row.response_ids == [7]
    _assert_masked(row)
    # and it still says why the rollout failed, not only that its row could not be built
    assert row.extra_fields["failed_exit_reason"] == "agent_loop_failed"
    assert "Unknown top-level agent config key(s): envv" in row.extra_fields["failure"]


def test_a_prompt_the_row_cannot_fill_is_a_masked_row(tmp_path, monkeypatch):
    prompts = {"system": "S", "task": "{problem_statement}"}
    loop = _loop(tmp_path, monkeypatch, prompts=prompts)
    (row,) = _run(loop, extra_info={})
    _assert_masked(row)
    assert _Env.closed == 0, "no sandbox was started, so there is nothing to close"


def test_a_reward_spec_that_will_not_load_keeps_the_ordinary_dummy_row(tmp_path, monkeypatch):
    def broken(_cfg):
        raise ImportError("no reward module for this row")

    monkeypatch.setattr(agent_loop_module, "load_reward_spec", broken)
    loop = _loop(tmp_path, monkeypatch, reward={"name": "nope"})
    (row,) = _run(loop)
    assert row.extra_fields["traj_exit_reason"] == "agent_loop_failed"
    assert row.prompt_ids == [1, 2, 3]
    _assert_masked(row)


def test_a_log_dir_that_cannot_be_opened_is_a_masked_row(tmp_path, monkeypatch):
    def unwritable(*_args, **_kwargs):
        raise PermissionError("read-only log dir")

    monkeypatch.setattr(agent_loop_module, "add_file_handler", unwritable)
    loop = _loop(tmp_path, monkeypatch)
    (row,) = _run(loop)
    assert row.extra_fields["traj_exit_reason"] == "agent_loop_failed"
    _assert_masked(row)
    assert _Env.closed == 1


@pytest.mark.parametrize("emit_feedback", [False, True])
def test_the_floor_row_keeps_the_columns_every_row_carries(emit_feedback):
    loop = UniAgentLoop.__new__(UniAgentLoop)
    loop.tokenizer = types.SimpleNamespace(pad_token_id=None, eos_token_id=9)
    loop.logger = types.SimpleNamespace(critical=lambda _m: None)
    loop.setup_attempts = 2
    loop.emit_feedback = emit_feedback

    async def broken(**_kwargs):
        raise RuntimeError("the chat template would not render")

    loop._build_empty_agent_output = broken
    row = asyncio.run(loop._failed_output("setup_timeout"))
    assert row.prompt_ids == [9] and row.response_ids == [9]
    assert row.response_mask == [0] and row.response_logprobs == [0.0]
    assert row.extra_fields["traj_exit_reason"] == "build_failed"
    assert row.extra_fields["timings"] == {"agent/setup_attempts": 2.0, "agent/setup_retried": 1.0}
    assert row.extra_fields["turn_spans"] == [] and row.extra_fields["turn_hints"] == []
    assert ("reward_extra_info" in row.extra_fields) is emit_feedback


def test_the_floor_row_keeps_the_reason_it_replaced(monkeypatch):
    traced = []
    monkeypatch.setattr(agent_loop_module, "rollout_trace_update_trace", lambda **kw: traced.append(kw))
    loop = UniAgentLoop.__new__(UniAgentLoop)
    loop.tokenizer = types.SimpleNamespace(pad_token_id=7, eos_token_id=9)
    loop.logger = types.SimpleNamespace(critical=lambda _m: None)
    loop.setup_attempts = 3

    async def broken(**_kwargs):
        raise RuntimeError("the chat template would not render")

    loop._build_empty_agent_output = broken
    row = asyncio.run(loop._failed_output("setup_timeout", TimeoutError("env.start took 90s")))
    assert row.extra_fields["traj_exit_reason"] == "build_failed"
    assert row.extra_fields["failed_exit_reason"] == "setup_timeout"
    assert row.extra_fields["failure"] == repr(TimeoutError("env.start took 90s"))
    # the trace's last word is the row that shipped, with the cause beside it
    assert traced[-1]["output"] == {
        "termination": "build_failed",
        "failed_exit_reason": "setup_timeout",
        "failure": repr(TimeoutError("env.start took 90s")),
    }
