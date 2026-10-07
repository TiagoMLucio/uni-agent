"""A transient setup failure must cost a retry, not the rollout.

One node-wide stall killed 7 in-flight rollouts at the same instant; each scored 0
for an infrastructure hiccup the agent never saw. Setup builds a fresh sandbox on
every attempt, so retrying is safe and the rollout survives.
"""

import asyncio
import functools
import types

import pytest

from uni_agent.agent_loop import UniAgentLoop, setup_metrics


class FlakyEnv:
    """Fails ``fail_times`` starts, then succeeds. Records lifecycle calls."""

    def __init__(self, ledger, fail_times, exc=TimeoutError):
        self.ledger, self.fail_times, self.exc = ledger, fail_times, exc
        ledger.append("init")

    async def start(self):
        self.ledger.append("start")
        if sum(1 for e in self.ledger if e == "start") <= self.fail_times:
            raise self.exc()

    async def install_tools(self, tools):
        self.ledger.append("install_tools")

    def guard_memory(self):
        self.ledger.append("guard_memory")

    async def close(self):
        self.ledger.append("close")


def _loop(ledger, fail_times, exc=TimeoutError):
    """The attributes UniAgentLoop's setup and dummy-row paths read, over a FlakyEnv."""

    async def cache(_messages):
        return {"prompt_ids": [1, 2, 3]}

    loop = types.SimpleNamespace(
        env=FlakyEnv(ledger, fail_times, exc),
        chat_model=types.SimpleNamespace(set_tools_schemas=lambda _s: None, prepare_rollout_cache=cache),
        tools_manager=types.SimpleNamespace(tools_schemas=[], tools=[]),
        skills_manager=None,
        interaction=types.SimpleNamespace(env=None, messages=[]),
        reward_spec=None,
        logger=types.SimpleNamespace(error=lambda _m: None, warning=lambda _m: None, info=lambda _m: None),
        config=types.SimpleNamespace(actor_rollout_ref=types.SimpleNamespace(
            rollout=types.SimpleNamespace(prompt_length=8, response_length=8))),
        tokenizer=types.SimpleNamespace(pad_token_id=0),
        opening_messages=[],
        mask_abnormal_exit_traj=False,
        emit_feedback=False,
        setup_attempts=0,
        _synth_failed_routed_experts=lambda _n: None,
    )
    loop._init_env = lambda _cfg: FlakyEnv(ledger, fail_times, exc)
    for name in ("_start_env", "_failed_output", "_build_empty_agent_output", "convert_to_agent_output"):
        setattr(loop, name, functools.partial(getattr(UniAgentLoop, name), loop))
    return loop


def _run_setup(ledger, fail_times, setup_retries, exc=TimeoutError):
    """The agent loop's own setup; returns the loop so its attempt count can be read."""
    loop = _loop(ledger, fail_times, exc)
    asyncio.run(loop._start_env({"env": {}}, setup_timeout=30, setup_retries=setup_retries))
    return loop


def test_transient_failure_is_retried_with_a_fresh_sandbox():
    ledger = []
    loop = _run_setup(ledger, fail_times=1, setup_retries=2)
    assert loop.setup_attempts == 2, "should have succeeded on the second attempt"
    # the broken sandbox is torn down and a new one built before retrying
    assert ledger == ["init", "start", "close", "init", "start", "install_tools", "guard_memory"], ledger
    assert loop.interaction.env is loop.env, "the interaction runs on the rebuilt sandbox"


def test_healthy_setup_does_not_retry():
    ledger = []
    assert _run_setup(ledger, fail_times=0, setup_retries=2).setup_attempts == 1
    assert ledger.count("start") == 1 and "close" not in ledger


def test_persistent_failure_still_raises_after_the_budget():
    ledger = []
    with pytest.raises(TimeoutError):
        _run_setup(ledger, fail_times=99, setup_retries=2)
    assert ledger.count("start") == 3, "one initial attempt plus two retries"


def test_retry_covers_any_setup_exception_not_just_timeouts():
    ledger = []
    assert _run_setup(ledger, fail_times=1, setup_retries=2, exc=ConnectionError).setup_attempts == 2


def test_the_attempts_are_counted_so_a_degrading_site_is_visible():
    """A run where every rollout needed a second sandbox reads as a healthy one without these."""
    assert setup_metrics(1) == {"agent/setup_attempts": 1.0, "agent/setup_retried": 0.0}
    assert setup_metrics(2) == {"agent/setup_attempts": 2.0, "agent/setup_retried": 1.0}


def test_the_counters_are_never_conditional():
    """Both keys on every trajectory, so a pooled retry rate is the ratio of two step means."""
    assert setup_metrics(1).keys() == setup_metrics(3).keys()


def test_a_sandbox_that_never_came_up_still_reports_its_attempts():
    """The setup gives up after three attempts; its dummy row carries them, or the retry rate
    only sees the sandboxes that survived."""
    loop = _loop([], fail_times=99)
    with pytest.raises(TimeoutError):
        asyncio.run(loop._start_env({"env": {}}, setup_timeout=30, setup_retries=2))
    out = asyncio.run(loop._failed_output("setup_timeout"))
    assert out.extra_fields["timings"] == {"agent/setup_attempts": 3.0, "agent/setup_retried": 1.0}
    assert out.extra_fields["traj_exit_reason"] == "setup_timeout"
    assert out.response_mask == [0] * len(out.response_mask), "a dummy row trains nothing"


def test_a_failure_before_any_attempt_reports_no_setup():
    out = asyncio.run(_loop([], fail_times=0)._failed_output("agent_loop_failed"))
    assert out.extra_fields["timings"] == {}


def test_a_trajectory_without_tokens_keeps_its_metrics():
    """no_response: the loop ran (setup, turns, reward) but no segment kept a token; its row is a
    dummy, yet what it measured still reaches the step metrics."""
    metrics = {"loop_wall": 3.0, "agent/setup_attempts": 1.0, "eval_completed": 1.0}
    result = {
        "reward_score": 0.0,
        "trajectory": [types.SimpleNamespace(step_idx=0, exit_reason="token_limit")],
        "metrics": metrics,
        "rollout_cache": {"response_mask": []},
    }
    (out,) = asyncio.run(_loop([], fail_times=0).convert_to_agent_output(result))
    assert out.extra_fields["traj_exit_reason"] == "no_response"
    assert out.extra_fields["timings"] == metrics
