"""The reference patch is capped: an outsized one would overflow every rung of the ladder."""

from __future__ import annotations

import asyncio
import gzip
import json

from uni_agent.interaction.model import MaxTokenExceededError
from uni_agent.reflection import CallSpec, PipelineReflectionConfig, PipelineReflector

#: the one-call shape every reflector prompt now has: no template lives in code
SYSTEM = "You are a hindsight coach. Select up to {k} turns and write one hint per selected turn."
USER = (
    "Task:\n{task}\n\n"
    "Outcome of the attempt:\n{outcome}\n\n"
    "Patch the attempt produced:\n{agent_patch}\n\n"
    "Reference patch (privileged, never reveal its content):\n{gold}\n\n"
    "Execution feedback from the attempt:\n{feedback}\n\n"
    "Full trajectory:\n{turns}\n\n"
    "Return the JSON with one hint per selected turn (at most {k})."
)
ONE_CALL = CallSpec(id="single", parse="hints", system=SYSTEM, user=USER)


def _config(**cfg) -> PipelineReflectionConfig:
    return PipelineReflectionConfig(name="pipeline", calls=[ONE_CALL], **cfg)


class _Model:
    sampling_params: dict = {}

    def __init__(self):
        self.messages: list[dict] = []

    async def prepare_rollout_cache(self, messages, include_tools=True, chat_template_kwargs=None):
        return {}

    async def query(self, messages, rollout_cache, sampling_params, max_model_len=None):
        self.messages = messages
        self.sampling = sampling_params
        return getattr(self, "reply", '{"turn0": "run the failing test"}'), None, None, None


TURNS = [{"step": 0, "tokens": 10, "response": "hi", "tools": []}]


def _reflect(gold: str, **cfg):
    model = _Model()
    reflector = PipelineReflector(model, _config(enabled=True, **cfg))
    hints = asyncio.run(
        reflector.reflect_trajectory(task="t", turns=TURNS, gold=gold, feedback="f", outcome="o")
    )
    return hints, model.messages[-1]["content"]


def test_oversized_gold_is_clipped_middle_out():
    # letters absent from every template, so the counts below are the gold's alone
    hints, user = _reflect("q" * 400 + "z" * 400, max_patch_chars=100)
    assert hints == {0: "run the failing test"}
    assert "[... 700 chars elided ...]" in user
    assert user.count("q") == 50 and user.count("z") == 50


def test_gold_under_the_cap_is_untouched():
    _, user = _reflect("diff --git a/x b/x", max_patch_chars=100)
    assert "diff --git a/x b/x" in user and "elided" not in user


def test_default_cap():
    assert _config().max_patch_chars == 16000


def test_agent_patch_is_included_and_capped():
    model = _Model()
    reflector = PipelineReflector(model, _config(enabled=True, max_patch_chars=100))
    asyncio.run(
        reflector.reflect_trajectory(
            task="t", turns=TURNS, gold="g", feedback="f", outcome="o", agent_patch="q" * 400 + "z" * 400
        )
    )
    user = model.messages[-1]["content"]
    assert "Patch the attempt produced:" in user
    assert user.count("q") == 50 and user.count("z") == 50


def test_agent_patch_can_be_switched_off():
    model = _Model()
    reflector = PipelineReflector(model, _config(enabled=True, include_agent_patch=False))
    asyncio.run(
        reflector.reflect_trajectory(task="t", turns=TURNS, gold="g", feedback="f", agent_patch="diff --git a/x b/x")
    )
    assert "diff --git a/x b/x" not in model.messages[-1]["content"]


def test_tool_call_arguments_are_not_duplicated():
    """The raw response already carries the call, so the parsed action must not be rendered again."""
    script = "x = 1\n" * 400
    turns = [
        {
            "step": 0,
            "tokens": 10,
            "response": f'writing a repro\n<tool_call>\n{{"file_text": "{script}"}}\n</tool_call>',
            "tools": [
                {"name": "str_replace_editor", "action": f"str_replace_editor create --file_text '{script}'",
                 "observation": "File created successfully"},
            ],
        }
    ]
    model = _Model()
    reflector = PipelineReflector(model, _config(enabled=True))
    asyncio.run(reflector.reflect_trajectory(task="t", turns=turns, gold="", feedback=""))
    user = model.messages[-1]["content"]
    assert user.count("x = 1") == 400  # once, from the response, not twice
    assert "TOOL str_replace_editor:" in user
    assert "File created successfully" in user


def test_audit_braces_do_not_shadow_the_final_object():
    """A reasoned prompt writes its audit first; the object after the marker is the answer."""
    model = _Model()
    model.reply = 'AUDIT turn=3 the file defines {"a": 1}\nFINAL_HINTS_JSON:\n{"turn7": "open the parser"}'
    reflector = PipelineReflector(model, _config(enabled=True))
    turns = [{"step": 7, "tokens": 10, "response": "hi", "tools": []}]
    hints = asyncio.run(reflector.reflect_trajectory(task="t", turns=turns, gold="", feedback=""))
    assert hints == {7: "open the parser"}


def test_a_reply_without_the_marker_still_parses():
    model = _Model()
    model.reply = 'no marker here {"turn0": "run the failing test"}'
    reflector = PipelineReflector(model, _config(enabled=True))
    hints = asyncio.run(reflector.reflect_trajectory(task="t", turns=TURNS, gold="", feedback=""))
    assert hints == {0: "run the failing test"}


def test_output_budget_is_configurable():
    model = _Model()
    reflector = PipelineReflector(model, _config(enabled=True, max_output_tokens=8192))
    asyncio.run(reflector.reflect_trajectory(task="t", turns=TURNS, gold="", feedback=""))
    assert model.sampling["max_tokens"] == 8192
    assert _config().max_output_tokens == 16384


def test_shrink_ladder_is_configurable_and_starts_uncapped():
    cfg = _config(enabled=True, max_observation_chars=50, shrink_ladder=[(10, None)])
    assert cfg.shrink_ladder == [(10, None)]
    # token-denominated rungs, char-approximated at ~3.8 chars/token
    assert _config().shrink_ladder[0] == (7600, None)
    assert len(_config().shrink_ladder) == 4


def test_an_unknown_key_is_rejected():
    import pytest
    with pytest.raises(Exception):
        _config(max_selected_turn=3)  # typo: singular


# --- pipeline -----------------------------------------------------------------------------

PIPE_TURNS = [
    {"step": 1, "tokens": 5, "response": "look at parser.py", "tools": []},
    {"step": 4, "tokens": 5, "response": "edit it", "tools": []},
    {"step": 9, "tokens": 5, "response": "submit", "tools": []},
]


class _ScriptedModel:
    """Replies chosen by a tag in the system prompt, since per-turn calls run concurrently."""

    sampling_params: dict = {}

    def __init__(self, replies):
        self.replies = replies
        self.seen: list[dict] = []

    async def prepare_rollout_cache(self, messages, include_tools=True, chat_template_kwargs=None):
        return {}

    async def query(self, messages, rollout_cache, sampling_params, max_model_len=None):
        system, user = messages[0]["content"], messages[1]["content"]
        self.seen.append({"system": system, "user": user})
        for tag, reply in self.replies.items():
            if tag in system:
                return (reply(user) if callable(reply) else reply), None, None, None
        raise AssertionError(f"no scripted reply for {system[:30]!r}")


def _pipeline(replies, calls, **cfg):
    model = _ScriptedModel(replies)
    config = PipelineReflectionConfig(enabled=True, name="pipeline", calls=calls, **cfg)
    reflector = PipelineReflector(model, config)
    hints = asyncio.run(
        reflector.reflect_trajectory(task="t", turns=PIPE_TURNS, gold="GOLDPATCH", feedback="f", outcome="o")
    )
    return hints, model


DRAFT = CallSpec(id="draft", parse="hints", system="DRAFT {k}",
                 user="{task}\n{gold}\n{turns}")
REPAIR = CallSpec(id="repair", per="turn", parse="hints", edit="delete_only", system="REPAIR",
                  user="{task}\nprefix:\n{prefix}\nturn {turn}\nhint: {hint}")


def test_a_per_turn_stage_replaces_each_hint_with_what_it_returns():
    replies = {
        "DRAFT": 'FINAL_HINTS_JSON:\n{"turn1": "open parser.py because the gold says so", '
                 '"turn4": "run the reproduction script first"}',
        "REPAIR": lambda user: ("FINAL_HINTS_JSON:\nopen parser.py" if "turn 1" in user
                                else "FINAL_HINTS_JSON:\nrun the reproduction script first"),
    }
    hints, _ = _pipeline(replies, [DRAFT, REPAIR])
    assert hints == {1: "open parser.py", 4: "run the reproduction script first"}


def test_a_rewrite_is_refused_and_the_draft_stands():
    """The stage sees a draft written with the patch, so a rewrite can restate what it never saw."""
    replies = {
        "DRAFT": 'FINAL_HINTS_JSON:\n{"turn1": "open parser.py"}',
        "REPAIR": "FINAL_HINTS_JSON:\nthe reference patch shows you should edit the compiler",
    }
    hints, _ = _pipeline(replies, [DRAFT, REPAIR])
    assert hints == {1: "open parser.py"}


def test_a_failed_repair_call_drops_the_hint_rather_than_keeping_it():
    class _Flaky(_ScriptedModel):
        async def query(self, messages, rollout_cache, sampling_params, max_model_len=None):
            if "REPAIR" in messages[0]["content"]:
                raise RuntimeError("boom")
            return await super().query(messages, rollout_cache, sampling_params, max_model_len)

    model = _Flaky({"DRAFT": 'FINAL_HINTS_JSON:\n{"turn1": "open parser.py"}'})
    reflector = PipelineReflector(model, PipelineReflectionConfig(enabled=True, name="pipeline",
                                                                 calls=[DRAFT, REPAIR]))
    hints = asyncio.run(reflector.reflect_trajectory(task="t", turns=PIPE_TURNS, gold="g", feedback="f"))
    assert hints == {}


def test_a_stage_that_says_drop_removes_the_hint():
    replies = {"DRAFT": 'FINAL_HINTS_JSON:\n{"turn1": "open parser.py"}',
               "REPAIR": "nothing here is quotable\nFINAL_HINTS_JSON:\nDROP"}
    hints, _ = _pipeline(replies, [DRAFT, REPAIR])
    assert hints == {}


def test_a_stage_that_forgets_the_marker_yields_its_last_line():
    replies = {"DRAFT": 'FINAL_HINTS_JSON:\n{"turn1": "open parser.py"}',
               "REPAIR": "clause one: fine\nclause two: unsupported\nopen parser.py"}
    hints, _ = _pipeline(replies, [DRAFT, REPAIR])
    assert hints == {1: "open parser.py"}


def test_a_per_turn_call_sees_only_its_own_prefix():
    replies = {
        "DRAFT": 'FINAL_HINTS_JSON:\n{"turn4": "edit it"}',
        "REPAIR": "FINAL_HINTS_JSON:\nedit it",
    }
    _, model = _pipeline(replies, [DRAFT, REPAIR])
    repair_user = next(c["user"] for c in model.seen if "turn 4" in c["user"])
    assert "look at parser.py" in repair_user      # turn 1 is before turn 4
    assert "submit" not in repair_user             # turn 9 is after it
    assert "GOLDPATCH" not in repair_user          # the template never names {gold}


def test_a_failing_repair_drops_rather_than_ships_the_draft():
    class _Flaky(_ScriptedModel):
        async def query(self, messages, rollout_cache, sampling_params, max_model_len=None):
            if "REPAIR" in messages[0]["content"]:
                raise RuntimeError("boom")
            return await super().query(messages, rollout_cache, sampling_params, max_model_len)

    model = _Flaky({"DRAFT": 'FINAL_HINTS_JSON:\n{"turn1": "open parser.py"}'})
    reflector = PipelineReflector(model, PipelineReflectionConfig(enabled=True, name="pipeline",
                                                                 calls=[DRAFT, REPAIR]))
    hints = asyncio.run(reflector.reflect_trajectory(task="t", turns=PIPE_TURNS, gold="g", feedback="f"))
    assert hints == {}


def test_a_call_cannot_reference_a_field_it_is_not_given():
    import pytest
    with pytest.raises(Exception, match="cannot be given"):
        PipelineReflectionConfig(name="pipeline", calls=[
            CallSpec(id="draft", parse="hints", system="s", user="{task} {prefix}"),
        ])


def test_the_last_call_must_produce_hints():
    import pytest
    with pytest.raises(Exception, match="parse=hints"):
        PipelineReflectionConfig(name="pipeline", calls=[
            CallSpec(id="draft", parse="text", system="s", user="{task}"),
        ])


def test_chat_control_tokens_do_not_look_like_a_rewrite():
    """The raw completion keeps <|im_end|>; leaving it in made every reply fail the edit check."""
    replies = {"DRAFT": 'FINAL_HINTS_JSON:\n{"turn1": "open parser.py and run the snippet"}',
               "REPAIR": "FINAL_HINTS_JSON:\nopen parser.py<|im_end|>"}
    hints, _ = _pipeline(replies, [DRAFT, REPAIR])
    assert hints == {1: "open parser.py"}


def test_a_drop_carrying_a_control_token_still_drops():
    replies = {"DRAFT": 'FINAL_HINTS_JSON:\n{"turn1": "open parser.py"}',
               "REPAIR": "FINAL_HINTS_JSON:\nDROP<|im_end|>"}
    hints, _ = _pipeline(replies, [DRAFT, REPAIR])
    assert hints == {}


def test_records_every_call_with_its_prompt_and_reply(tmp_path):
    path = tmp_path / "reflection.jsonl.gz"
    model = _Model()
    reflector = PipelineReflector(model, _config(enabled=True), record_path=path,
                                  identity={"uid": "u1", "instance_id": "repo.abc", "junk": "dropped"})
    asyncio.run(reflector.reflect_trajectory(task="t", turns=TURNS, gold="g", feedback="f"))

    rows = [json.loads(l) for l in gzip.open(path, "rt")]
    assert len(rows) == 1
    row = rows[0]
    assert (row["uid"], row["instance_id"], row["stage"]) == ("u1", "repo.abc", "single")
    assert "junk" not in row
    assert row["output"] == '{"turn0": "run the failing test"}'
    assert "Full trajectory:" in row["user"] and row["system"].startswith("You are a hindsight coach")
    assert row["error"] == ""


def test_recording_failure_never_costs_the_rollout_its_hints(tmp_path):
    model = _Model()
    # a directory where the file should be: opening it raises on every call
    (tmp_path / "reflection.jsonl.gz").mkdir()
    reflector = PipelineReflector(model, _config(enabled=True),
                                  record_path=tmp_path / "reflection.jsonl.gz")
    hints = asyncio.run(reflector.reflect_trajectory(task="t", turns=TURNS, gold="g", feedback="f"))
    assert hints == {0: "run the failing test"}


def test_no_record_path_writes_nothing(tmp_path):
    reflector = PipelineReflector(_Model(), _config(enabled=True))
    asyncio.run(reflector.reflect_trajectory(task="t", turns=TURNS, gold="g", feedback="f"))
    assert not list(tmp_path.iterdir())


def test_a_pipeline_records_each_stage_separately(tmp_path):
    path = tmp_path / "reflection.jsonl.gz"
    model = _ScriptedModel({
        "DRAFT": 'FINAL_HINTS_JSON:\n{"turn1": "open parser.py", "turn4": "run the repro"}',
        "REPAIR": "FINAL_HINTS_JSON:\nDELETE",
    })
    reflector = PipelineReflector(
        model, PipelineReflectionConfig(enabled=True, name="pipeline", calls=[DRAFT, REPAIR]),
        record_path=path, identity={"uid": "u9"})
    asyncio.run(reflector.reflect_trajectory(task="t", turns=PIPE_TURNS, gold="G", feedback="f"))

    rows = [json.loads(l) for l in gzip.open(path, "rt")]
    # one draft over the whole trajectory, then one repair call per selected turn
    assert [r["stage"] for r in rows] == ["draft", "repair", "repair"]
    assert [r["step"] for r in rows] == [None, 1, 4]
    assert all(r["uid"] == "u9" for r in rows)
    assert rows[0]["user"].startswith("t\nG\n") and rows[1]["output"] == "FINAL_HINTS_JSON:\nDELETE"


def test_the_shrink_ladder_still_retries_through_the_trace_span(tmp_path):
    """The ladder depends on MaxTokenExceededError escaping the span wrapping each call."""
    path = tmp_path / "reflection.jsonl.gz"

    class _Overflows(_Model):
        calls = 0

        async def query(self, messages, rollout_cache, sampling_params, max_model_len=None):
            self.calls += 1
            if self.calls == 1:
                raise MaxTokenExceededError("prompt_ids length 200000 exceeds max_model_len")
            return '{"turn0": "run the failing test"}', None, None, None

    model = _Overflows()
    # no same-rung re-draw, so the second call is the second rung
    reflector = PipelineReflector(model, _config(enabled=True, redraws_per_rung=0), record_path=path,
                                  identity={"uid": "u1"})
    hints = asyncio.run(reflector.reflect_trajectory(task="t", turns=TURNS, gold="g", feedback="f"))

    assert hints == {0: "run the failing test"}, "the second rung's hints were lost"
    assert model.calls == 2, "the ladder did not retry after the overflow"
    rows = [json.loads(l) for l in gzip.open(path, "rt")]
    # both attempts are on record: the rung that overflowed and the one that answered
    assert [r["error"] for r in rows] == ["over budget", ""]
    assert [r["obs_cap"] for r in rows] == [1000, 7600]


# --- _maybe_reflect filters: skip_exit_reasons ----------------------------------------------

import types  # noqa: E402

from uni_agent.reflection import BaseReflectionConfig, build_reflection_config  # noqa: E402


def _traj_step(idx, tools=(), exit_reason="turn_done"):
    return types.SimpleNamespace(
        step_idx=idx,
        response="r",
        tool_results=[types.SimpleNamespace(name=n, action=a, observation=o) for n, a, o in tools],
        exit_reason=exit_reason,
        done=False,
    )


def _maybe_reflect(tmp_path, reflection, trajectory, reply=None, reward_score=0.0, resolved=False):
    """The real ``UniAgentLoop._maybe_reflect`` over a fake loop; errors fail the test."""
    from uni_agent.agent_loop import UniAgentLoop

    model = _Model()
    if reply is not None:
        model.reply = reply

    def _raise(msg):
        raise AssertionError(msg)

    loop = types.SimpleNamespace(
        env=types.SimpleNamespace(privileged_context="gold"),
        chat_model=model,
        run_id="run",
        output_dir=tmp_path,
        identity={},
        logger=types.SimpleNamespace(critical=_raise),
    )
    result = {
        "reward_score": reward_score,
        "resolved": resolved,
        "reward_extra_info": {"feedback": "f", "agent_patch": ""},
        "metrics": {},
        "messages": [{"role": "user", "content": "t"}],
        "trajectory": trajectory,
        "rollout_cache": {"turn_spans": [[s.step_idx, i * 2, i * 2 + 2] for i, s in enumerate(trajectory)]},
    }
    block = {"name": "pipeline", "calls": [ONE_CALL.model_dump()], **reflection}
    hints = asyncio.run(UniAgentLoop._maybe_reflect(loop, result, {"reflection": block}, validate=False))
    return hints, model


def test_outcome_reads_resolved_from_the_reward_result(tmp_path, monkeypatch):
    """The reward sets ``resolved`` on its result, never inside reward_extra_info."""
    from uni_agent import agent_loop as agent_loop_mod
    from uni_agent.agent_loop import UniAgentLoop

    trajectory = [_traj_step(0), _traj_step(1)]
    _, model = _maybe_reflect(
        tmp_path, {"enabled": True, "failed_only": False}, trajectory, reward_score=1.0, resolved=True
    )
    assert "resolved: True" in model.messages[1]["content"]

    captured = {}
    monkeypatch.setattr(agent_loop_mod, "rollout_trace_update_trace", lambda **kw: captured.update(kw))
    base = {"reward_score": 1.0, "trajectory": trajectory, "segments": None}
    UniAgentLoop._record_trace_outcome(None, dict(base, resolved=True))
    assert captured["metadata"]["outcome"]["resolved"] is True
    UniAgentLoop._record_trace_outcome(None, dict(base, reward_extra_info={"resolved": True}))
    assert captured["metadata"]["outcome"]["resolved"] is False


def test_filter_fields_default_off_and_validate():
    import pytest
    from pydantic import ValidationError

    config = _config()
    assert config.skip_exit_reasons == [] and config.failed_only and config.redraws_per_rung == 1
    block = {"name": "pipeline", "calls": [ONE_CALL.model_dump()], "skip_exit_reasons": ["stuck"]}
    assert build_reflection_config(block).skip_exit_reasons == ["stuck"]
    with pytest.raises(ValueError, match="reflection.name is required"):
        build_reflection_config({"enabled": True})
    with pytest.raises(ValidationError):
        BaseReflectionConfig(enabled=True)  # name has no default
    with pytest.raises(ValidationError):
        _config(redraws_per_rung=-1)


def test_skip_exit_reasons_skips_the_reflector_call(tmp_path):
    trajectory = [_traj_step(0, exit_reason="stuck")]
    hints, model = _maybe_reflect(tmp_path, {"enabled": True, "skip_exit_reasons": ["stuck"]}, trajectory)
    assert hints == {} and model.messages == [], "the reflector must not even be queried"
    # unset, the same trajectory is hinted as before
    hints, model = _maybe_reflect(tmp_path, {"enabled": True}, trajectory)
    assert hints == {0: "run the failing test"} and model.messages
    # a termination outside the list is untouched
    trajectory = [_traj_step(0, exit_reason="max_step_limit")]
    hints, _ = _maybe_reflect(tmp_path, {"enabled": True, "skip_exit_reasons": ["stuck"]}, trajectory)
    assert hints == {0: "run the failing test"}


def test_the_cap_sentinel_does_not_shadow_the_last_real_turn(tmp_path):
    """max_step_limit and stuck append a bare StepOutput under the last real step's index; the
    reflector must render that step's response and tool call, and count the turns without it."""
    from uni_agent.agent_loop import UniAgentLoop
    from uni_agent.interaction.interaction import StepOutput

    def _turn(idx, response, command, observation):
        # the call reaches the reflector inside the raw response; TOOL_TEMPLATE renders name and observation
        step = _traj_step(idx, tools=[("execute_bash", command, observation)])
        step.response = (
            f"{response}\n<tool_call>\n<function=execute_bash>\n"
            f"<parameter=command>{command}</parameter>\n</function>\n</tool_call>"
        )
        return step

    last = _turn(2, "the real last turn", "pytest -x", "1 failed")
    first = _turn(1, "listing the repo", "ls /testbed", "a.py")
    trajectory = [first, last, StepOutput(step_idx=2, exit_reason="max_step_limit")]
    model = _Model()
    loop = types.SimpleNamespace(
        env=types.SimpleNamespace(privileged_context="gold"), chat_model=model, run_id="run",
        output_dir=tmp_path, identity={}, logger=None,
    )
    result = {
        "reward_score": 0.0, "resolved": False, "reward_extra_info": {"feedback": "f", "agent_patch": ""},
        "metrics": {}, "messages": [{"role": "user", "content": "t"}], "trajectory": trajectory,
        "rollout_cache": {"turn_spans": [[1, 0, 2], [2, 2, 4]]},
    }
    block = {"name": "pipeline", "calls": [ONE_CALL.model_dump()], "enabled": True}
    asyncio.run(UniAgentLoop._maybe_reflect(loop, result, {"reflection": block}, validate=False))

    rendered = model.messages[1]["content"]
    assert "the real last turn" in rendered and "pytest -x" in rendered and "1 failed" in rendered
    assert "degenerate turn" not in rendered
    assert "termination: max_step_limit | turns: 2" in rendered
