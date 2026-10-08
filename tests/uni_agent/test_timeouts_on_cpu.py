"""A rollout always ends, and a rollout that ends any way at all takes its sandbox with it.

A dead rollout engine used to hang the synchronous step for the whole job (twice for the full 8 h):
nothing bounded a generate call. Each call now has a deadline its own token budget sets, the turn
loop has a wall-clock backstop, and both end as masked rows the trainer counts as infra failures.
Teardown is shielded from cancellation and bounded, so neither a cancel nor a stalled node leaks
the sandbox or holds the caller.
"""

from __future__ import annotations

import asyncio
import subprocess
import types

import pytest

from uni_agent import agent_loop as agent_loop_module
from uni_agent import async_logging
from uni_agent.agent_loop import UniAgentLoop, derived_episode_timeout
from uni_agent.interaction import env as env_module
from uni_agent.interaction import model as model_module
from uni_agent.interaction.interaction import AgentInteraction
from uni_agent.interaction.model import AgentChatModel, GenerationTimeoutError, generation_timeout

# --- the generate deadline -------------------------------------------------------------------


def _model(generate, max_completion_tokens=16):
    return types.SimpleNamespace(
        max_model_len=1000, max_completion_tokens=max_completion_tokens, sampling_params={},
        client=types.SimpleNamespace(generate=generate),
        tokenizer=types.SimpleNamespace(decode=lambda ids: "x" * len(ids)),
        loop=asyncio.get_running_loop(),
    )


def _cache():
    return {"request_id": "r", "prompt_ids": [1] * 10, "metrics": {},
            "response_mask": [], "response_logprobs": [], "extra_fields": {}}


def test_the_deadline_grows_with_what_the_call_may_generate():
    assert generation_timeout(0) == model_module.GENERATION_TIMEOUT_BASE_S
    # the base agent's per-turn cap: about 17 minutes, against seconds for a healthy turn
    assert generation_timeout(4096) == pytest.approx(600 + 409.6)


def test_a_generate_that_never_returns_raises_after_its_deadline(monkeypatch):
    monkeypatch.setattr(model_module, "GENERATION_TIMEOUT_BASE_S", 0.05)
    monkeypatch.setattr(model_module, "GENERATION_TIMEOUT_S_PER_TOKEN", 0.0)
    seen = {}

    async def dead_engine(request_id, prompt_ids, sampling_params):
        seen["max_tokens"] = sampling_params["max_tokens"]
        await asyncio.Event().wait()

    async def run():
        cache = _cache()
        with pytest.raises(GenerationTimeoutError, match="max_tokens=16"):
            await AgentChatModel.query(_model(dead_engine), [{"role": "user", "content": "t"}], cache)
        return cache

    cache = asyncio.run(run())
    assert seen["max_tokens"] == 16
    assert cache["response_mask"] == [] and cache["prompt_ids"] == [1] * 10, "nothing appended"
    assert cache["metrics"]["generate_sequences"] > 0, "the wait is still timed"


def test_a_timeout_the_client_raises_itself_is_not_relabelled():
    async def client_timeout(request_id, prompt_ids, sampling_params):
        raise TimeoutError("ray rpc")

    async def run():
        with pytest.raises(TimeoutError, match="ray rpc") as info:
            await AgentChatModel.query(_model(client_timeout), [{"role": "user", "content": "t"}], _cache())
        assert not isinstance(info.value, GenerationTimeoutError)

    asyncio.run(run())


# --- the turn ends, the row is masked, the reflector stays away -------------------------------


def test_a_generation_timeout_ends_the_trajectory_as_its_own_exit():
    async def timed_out(**_kwargs):
        raise GenerationTimeoutError("generate returned nothing in 1010s")

    interaction = AgentInteraction.__new__(AgentInteraction)
    interaction.model = types.SimpleNamespace(query=timed_out)
    interaction.messages = [{"role": "user", "content": "fix it"}]
    interaction.rollout_cache = {"response_mask": [], "prompt_ids": [1, 2], "metrics": {}}
    interaction.condense_max_retries = 2
    interaction.logger = types.SimpleNamespace(info=lambda *_a: None, error=lambda *_a: None)

    step = asyncio.run(AgentInteraction.step.__wrapped__(interaction, step_idx=3))
    assert (step.exit_reason, step.done) == ("generation_timeout", True)
    assert interaction.rollout_cache["prompt_ids"] == [1, 2]


def _converted(exit_reason):
    import functools

    loop = types.SimpleNamespace(
        logger=types.SimpleNamespace(info=lambda _m: None, warning=lambda _m: None),
        config=types.SimpleNamespace(actor_rollout_ref=types.SimpleNamespace(
            rollout=types.SimpleNamespace(prompt_length=64, response_length=64))),
        opening_messages=[], mask_abnormal_exit_traj=False, emit_feedback=False,
        chat_model=types.SimpleNamespace(max_model_len=10**9),
    )
    loop._segment_to_output = functools.partial(UniAgentLoop._segment_to_output, loop)
    cache = {"prompt_ids": [1, 2, 3, 4], "response_mask": [1, 1], "response_logprobs": [], "turn_spans": [[1, 2, 4]]}
    result = {"trajectory": [types.SimpleNamespace(step_idx=1, tool_results=[], exit_reason=exit_reason)],
              "segments": [{"rollout_cache": cache, "prompt_messages": None}],
              "rollout_cache": cache, "metrics": {}, "reward_score": 1.0}
    (row,) = asyncio.run(UniAgentLoop.convert_to_agent_output(loop, result))
    return row


def test_a_generation_timeout_row_is_scored_but_masked():
    row = _converted("generation_timeout")
    assert row.extra_fields["traj_exit_reason"] == "generation_timeout"
    assert row.response_mask == [0, 0]
    assert row.reward_score == 1.0, "the grade of what it did is kept, as for any partial trajectory"
    assert _converted("max_step_limit").response_mask == [1, 1]


def test_the_reflector_never_runs_on_a_generation_timeout(monkeypatch):
    def no_reflector(*_args, **_kwargs):
        raise AssertionError("the reflector would wait on the same engine, for a row that is masked")

    monkeypatch.setattr(agent_loop_module, "load_reflector", no_reflector)
    loop = types.SimpleNamespace(env=types.SimpleNamespace(privileged_context="g"),
                                 logger=types.SimpleNamespace(critical=lambda _m: None))
    step = types.SimpleNamespace(step_idx=1, exit_reason="generation_timeout", done=True, response="r",
                                 tool_results=[])
    result = {"trajectory": [step], "reward_score": 0,
              "rollout_cache": {"turn_spans": [[1, 0, 2]]}, "messages": []}
    hints = asyncio.run(UniAgentLoop._maybe_reflect(loop, result, {"reflection": {"enabled": True}}, validate=False))
    assert hints == {}


# --- the episode backstop ---------------------------------------------------------------------


def test_the_derived_backstop_allows_every_turn_its_own_bounds():
    # the base agent: 200 turns at the 4096-token cap, 30 s actions, 45 s kill wall
    assert derived_episode_timeout(200, 4096, 30, 45) == pytest.approx(200 * (1009.6 + 30 + 45))


def _interaction(**kwargs):
    return AgentInteraction(run_id="r", env=None, model=None, tools_manager=None, messages=[], **kwargs)


@pytest.mark.parametrize("seconds", [7200, 7200.0])
def test_the_episode_timeout_is_an_interaction_setting(seconds):
    assert _interaction(episode_timeout=seconds).episode_timeout == seconds
    assert _interaction().episode_timeout is None, "unset keeps the derived bound"


@pytest.mark.parametrize("seconds", [0, -5, "7200", True])
def test_an_episode_timeout_that_is_not_a_positive_number_is_refused(seconds):
    with pytest.raises(ValueError, match="episode_timeout must be a positive number"):
        _interaction(episode_timeout=seconds)


class _Env:
    def __init__(self, ledger):
        self.ledger = ledger

    async def clear_attached(self):
        pass

    async def close(self):
        self.ledger.append("close")


def _hung_loop(tmp_path, monkeypatch, ledger, configured=None):
    class _Hung:
        max_turns, action_timeout, attached_kill_timeout, episode_timeout = 1, 1, 1, configured

        def __init__(self, messages, **_kwargs):
            self.messages = messages

        async def run(self):
            ledger.append("run")
            await asyncio.Event().wait()

    monkeypatch.setattr(UniAgentLoop, "_semaphore", None)
    monkeypatch.setattr(agent_loop_module, "AgentInteraction", _Hung)
    loop = UniAgentLoop.__new__(UniAgentLoop)
    loop.config = types.SimpleNamespace(actor_rollout_ref=types.SimpleNamespace(
        model=types.SimpleNamespace(path="m"),
        rollout=types.SimpleNamespace(agent=types.SimpleNamespace(num_workers=1), prompt_length=16,
                                      response_length=8)))
    loop.tokenizer = types.SimpleNamespace(pad_token_id=7, eos_token_id=9)
    config = {"model": {}, "tools": [], "env": {}, "interaction": {}, "reward": None, "log_dir": str(tmp_path)}

    async def cache(_messages):
        return {"prompt_ids": [1, 2, 3], "extra_fields": {}}

    async def started(*_args, **_kwargs):
        pass

    loop._init_config = lambda *_args, **_kwargs: config
    loop._init_chat_model = lambda _cfg: types.SimpleNamespace(
        set_tools_schemas=lambda _s: None, prepare_rollout_cache=cache, max_completion_tokens=8, max_model_len=64)
    loop._init_tools_manager = lambda **_kwargs: types.SimpleNamespace(tools_schemas=[], tools=[])
    loop._init_skills_manager = lambda _cfg: None
    loop._init_condense = lambda _cfg: (None, {})
    loop._init_env = lambda _cfg: _Env(ledger)
    loop._start_env = started
    return loop


@pytest.mark.parametrize("configured", [0.05, None], ids=["configured", "derived"])
def test_a_turn_loop_past_its_backstop_is_an_episode_timeout_row(tmp_path, monkeypatch, configured):
    def derived(*_args):
        if configured is not None:
            raise AssertionError("a configured episode_timeout replaces the derived bound")
        return 0.05

    monkeypatch.setattr(agent_loop_module, "derived_episode_timeout", derived)
    ledger = []
    loop = _hung_loop(tmp_path, monkeypatch, ledger, configured)
    try:
        (row,) = asyncio.run(loop.run({}, raw_prompt=[{"role": "user", "content": "fix it"}]))
    finally:
        async_logging.cleanup_handlers(loop.run_id)
    assert row.extra_fields["traj_exit_reason"] == "episode_timeout"
    assert row.response_mask == [0] * len(row.response_mask) and row.reward_score == 0
    assert ledger == ["run", "close"]


def test_a_cancelled_rollout_still_closes_its_sandbox(tmp_path, monkeypatch):
    ledger = []
    loop = _hung_loop(tmp_path, monkeypatch, ledger)

    async def run():
        task = asyncio.ensure_future(loop.run({}, raw_prompt=[{"role": "user", "content": "fix it"}]))
        while "run" not in ledger:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    try:
        asyncio.run(run())
    finally:
        async_logging.cleanup_handlers(loop.run_id)
    assert ledger == ["run", "close"]


# --- teardown: shielded and bounded -------------------------------------------------------------


def _env(stop):
    env = env_module.AgentEnv.__new__(env_module.AgentEnv)
    env.deployment = types.SimpleNamespace(stop=stop)
    env.logger = types.SimpleNamespace(info=lambda _m: None, error=lambda _m: None)
    return env


def test_a_cancel_during_close_does_not_stop_the_teardown():
    done = []

    async def slow_stop():
        await asyncio.sleep(0.05)
        done.append("stopped")

    async def run():
        closing = asyncio.ensure_future(_env(slow_stop).close())
        await asyncio.sleep(0)
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        assert done == [], "the caller is released at once"
        await asyncio.sleep(0.1)

    asyncio.run(run())
    assert done == ["stopped"], "the stop ran on after the cancel"


def test_a_stalled_teardown_releases_its_caller(monkeypatch):
    monkeypatch.setattr(env_module, "CLOSE_TIMEOUT_S", 0.05)

    async def stalled_stop():
        await asyncio.Event().wait()

    async def run():
        await asyncio.wait_for(_env(stalled_stop).close(), 1.0)

    asyncio.run(run())


def test_a_stop_cancelled_while_the_runtime_closes_still_kills_the_sandbox():
    from uni_agent.deployment.local.deployment import LocalDeployment

    process = subprocess.Popen(["sleep", "60"])
    deployment = LocalDeployment(run_id="test", type="local", container_runtime="apptainer")

    class _HungRuntime:
        async def close(self):
            await asyncio.Event().wait()

    deployment._runtime = _HungRuntime()
    deployment._server_process = process
    deployment._stopped = False

    async def run():
        stopping = asyncio.ensure_future(deployment.stop())
        await asyncio.sleep(0.05)
        stopping.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stopping

    try:
        asyncio.run(run())
        assert process.wait(timeout=5) == -9
    finally:
        process.kill()


@pytest.mark.parametrize("spec", ["swe_bench", "swe_smith"])
def test_a_sibling_that_fails_to_start_is_closed(monkeypatch, spec):
    import importlib

    import uni_agent.interaction as interaction_module

    ledger = []

    class _Sibling:
        def __init__(self, run_id, env_config):
            pass

        async def start(self):
            ledger.append("start")
            raise RuntimeError("post_setup_cmd failed")

        async def close(self):
            ledger.append("close")

    monkeypatch.setattr(interaction_module, "AgentEnv", _Sibling)
    monkeypatch.setattr(interaction_module, "AgentEnvConfig", lambda **kw: kw)
    module = importlib.import_module(f"uni_agent.reward.{spec}")
    cls = module.SWEBenchRewardSpec if spec == "swe_bench" else module.SWESmithRewardSpec
    reward = cls.__new__(cls)
    reward.run_id = "r"
    reward.logger = types.SimpleNamespace(info=lambda _m: None)
    with pytest.raises(RuntimeError, match="post_setup_cmd failed"):
        asyncio.run(reward._start_sibling_env({"deployment": {}}))
    assert ledger == ["start", "close"]
