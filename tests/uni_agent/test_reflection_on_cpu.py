"""The reference patch is capped: an outsized one would overflow every rung of the ladder."""

from __future__ import annotations

import asyncio
import gzip
import json

from uni_agent.interaction.model import MaxTokenExceededError
from uni_agent.reflection import ReflectionConfig, Reflector


class _Model:
    sampling_params: dict = {}

    def __init__(self):
        self.messages: list[dict] = []

    async def prepare_rollout_cache(self, messages, include_tools=True):
        return {}

    async def query(self, messages, rollout_cache, sampling_params, max_model_len=None):
        self.messages = messages
        self.sampling = sampling_params
        return getattr(self, "reply", '{"turn0": "run the failing test"}'), None, None, None


TURNS = [{"step": 0, "tokens": 10, "response": "hi", "tools": []}]


def _reflect(gold: str, **cfg):
    model = _Model()
    reflector = Reflector(model, ReflectionConfig(enabled=True, **cfg))
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
    assert ReflectionConfig().max_patch_chars == 16000


def test_agent_patch_is_included_and_capped():
    model = _Model()
    reflector = Reflector(model, ReflectionConfig(enabled=True, max_patch_chars=100))
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
    reflector = Reflector(model, ReflectionConfig(enabled=True, include_agent_patch=False))
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
    reflector = Reflector(model, ReflectionConfig(enabled=True))
    asyncio.run(reflector.reflect_trajectory(task="t", turns=turns, gold="", feedback=""))
    user = model.messages[-1]["content"]
    assert user.count("x = 1") == 400  # once, from the response, not twice
    assert "TOOL str_replace_editor:" in user
    assert "File created successfully" in user


def test_audit_braces_do_not_shadow_the_final_object():
    """A reasoned prompt writes its audit first; the object after the marker is the answer."""
    model = _Model()
    model.reply = 'AUDIT turn=3 the file defines {"a": 1}\nFINAL_HINTS_JSON:\n{"turn7": "open the parser"}'
    reflector = Reflector(model, ReflectionConfig(enabled=True))
    turns = [{"step": 7, "tokens": 10, "response": "hi", "tools": []}]
    hints = asyncio.run(reflector.reflect_trajectory(task="t", turns=turns, gold="", feedback=""))
    assert hints == {7: "open the parser"}


def test_a_reply_without_the_marker_still_parses():
    model = _Model()
    model.reply = 'no marker here {"turn0": "run the failing test"}'
    reflector = Reflector(model, ReflectionConfig(enabled=True))
    hints = asyncio.run(reflector.reflect_trajectory(task="t", turns=TURNS, gold="", feedback=""))
    assert hints == {0: "run the failing test"}


def test_output_budget_is_configurable():
    model = _Model()
    reflector = Reflector(model, ReflectionConfig(enabled=True, max_output_tokens=8192))
    asyncio.run(reflector.reflect_trajectory(task="t", turns=TURNS, gold="", feedback=""))
    assert model.sampling["max_tokens"] == 8192
    assert ReflectionConfig().max_output_tokens == 2048


def test_shrink_ladder_is_configurable_and_starts_uncapped():
    cfg = ReflectionConfig(enabled=True, max_observation_chars=50, shrink_ladder=[(10, None)])
    assert cfg.shrink_ladder == [(10, None)]
    # token-denominated rungs, char-approximated at ~3.8 chars/token
    assert ReflectionConfig().shrink_ladder[0] == (7600, None)
    assert len(ReflectionConfig().shrink_ladder) == 4


def test_an_unknown_key_is_rejected():
    import pytest
    with pytest.raises(Exception):
        ReflectionConfig(max_selected_turn=3)  # typo: singular


# --- pipeline -----------------------------------------------------------------------------

from uni_agent.reflection import CallSpec, PipelineReflectionConfig, PipelineReflector  # noqa: E402

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

    async def prepare_rollout_cache(self, messages, include_tools=True):
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
    reflector = Reflector(model, ReflectionConfig(enabled=True), record_path=path,
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
    reflector = Reflector(model, ReflectionConfig(enabled=True),
                          record_path=tmp_path / "reflection.jsonl.gz")
    hints = asyncio.run(reflector.reflect_trajectory(task="t", turns=TURNS, gold="g", feedback="f"))
    assert hints == {0: "run the failing test"}


def test_no_record_path_writes_nothing(tmp_path):
    reflector = Reflector(_Model(), ReflectionConfig(enabled=True))
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
    reflector = Reflector(model, ReflectionConfig(enabled=True), record_path=path,
                          identity={"uid": "u1"})
    hints = asyncio.run(reflector.reflect_trajectory(task="t", turns=TURNS, gold="g", feedback="f"))

    assert hints == {0: "run the failing test"}, "the second rung's hints were lost"
    assert model.calls == 2, "the ladder did not retry after the overflow"
    rows = [json.loads(l) for l in gzip.open(path, "rt")]
    # both attempts are on record: the rung that overflowed and the one that answered
    assert [r["error"] for r in rows] == ["over budget", ""]
    assert [r["obs_cap"] for r in rows] == [1000, 7600]


# --- loop router --------------------------------------------------------------------------

import pytest  # noqa: E402

from uni_agent.reflection import LoopRouterReflectionConfig, LoopRouterReflector, build_reflection_config  # noqa: E402
from uni_agent.reflection.loop_router import REORIENT_HINT, first_loop  # noqa: E402


def _edit(action):
    return {"name": "str_replace_editor", "action": action, "observation": "No replacement was performed"}


def _loop_turns(steps=(2, 5, 8), action="str_replace --path /testbed/x.py --old_str a --new_str b"):
    turns = [{"step": s, "tokens": 5, "response": "explore", "tools": [_edit(f"view --path /f{s}.py")]}
             for s in range(2)]
    turns += [{"step": s, "tokens": 5, "response": "edit", "tools": [_edit(action)]} for s in steps]
    return sorted(turns, key=lambda t: t["step"])


def _route(turns, **cfg):
    # no model: a routed trace must never cost an LLM call
    reflector = LoopRouterReflector(None, LoopRouterReflectionConfig(enabled=True, name="loop_router", **cfg))
    return asyncio.run(reflector.reflect_trajectory(task="t", turns=turns, gold="g", feedback="f"))


def test_a_loop_gets_the_reorient_hint_at_its_first_turn():
    assert _route(_loop_turns()) == {2: REORIENT_HINT}


def test_whitespace_drift_still_counts_as_the_same_call():
    turns = _loop_turns(steps=(2,)) + [
        {"step": s, "tokens": 5, "response": "edit",
         "tools": [_edit("str_replace  --path /testbed/x.py --old_str a\n--new_str b")]}
        for s in (5, 8)
    ]
    assert _route(turns) == {2: REORIENT_HINT}


def test_distinct_calls_are_not_a_loop():
    turns = _loop_turns(steps=(2,)) + [
        {"step": s, "tokens": 5, "response": "edit", "tools": [_edit(f"str_replace --old_str a{s}")]}
        for s in (5, 8)
    ]
    assert _route(turns) == {}


def test_the_earliest_completing_loop_wins():
    slow, fast = "str_replace --old_str slow", "str_replace --old_str fast"
    turns = [{"step": s, "tokens": 5, "response": "r", "tools": [_edit(a)]}
             for s, a in [(0, slow), (1, fast), (2, slow), (3, fast), (4, fast), (6, slow)]]
    assert first_loop(turns, 3) == (1, "str_replace_editor str_replace --old_str fast")


def test_route_drop_returns_no_hints_for_a_loop():
    assert _route(_loop_turns(), route="drop") == {}


def test_no_loop_and_no_fallback_yields_no_hints():
    assert _route(_loop_turns(steps=(2,))) == {}


def test_no_loop_delegates_to_the_fallback():
    model = _Model()
    config = LoopRouterReflectionConfig(enabled=True, name="loop_router", fallback={"enabled": True})
    reflector = LoopRouterReflector(model, config)
    hints = asyncio.run(
        reflector.reflect_trajectory(task="t", turns=TURNS, gold="g", feedback="f", outcome="o"))
    assert hints == {0: "run the failing test"}
    assert "Full trajectory:" in model.messages[-1]["content"]


def test_a_loop_never_reaches_the_fallback():
    class _Bomb(_Model):
        async def query(self, messages, rollout_cache, sampling_params, max_model_len=None):
            raise AssertionError("a routed trace must not consult the fallback reflector")

    config = LoopRouterReflectionConfig(enabled=True, name="loop_router", fallback={"enabled": True})
    reflector = LoopRouterReflector(_Bomb(), config)
    hints = asyncio.run(reflector.reflect_trajectory(task="t", turns=_loop_turns(), gold="g", feedback="f"))
    assert hints == {2: REORIENT_HINT}


def test_the_registry_builds_a_loop_router():
    config = build_reflection_config({"name": "loop_router", "enabled": True, "min_repeats": 2})
    assert isinstance(config, LoopRouterReflectionConfig) and config.min_repeats == 2


def test_min_repeats_below_two_is_rejected():
    with pytest.raises(ValueError, match="min_repeats"):
        LoopRouterReflectionConfig(name="loop_router", min_repeats=1)


def test_a_fallback_typo_is_rejected_at_config_time():
    with pytest.raises(ValueError):
        LoopRouterReflectionConfig(name="loop_router", fallback={"enabled": True, "sytem_template": "x"})


def test_the_route_is_recorded(tmp_path):
    path = tmp_path / "reflection.jsonl.gz"
    reflector = LoopRouterReflector(
        None, LoopRouterReflectionConfig(enabled=True, name="loop_router"),
        record_path=path, identity={"uid": "u2"})
    asyncio.run(reflector.reflect_trajectory(task="t", turns=_loop_turns(), gold="g", feedback="f"))

    rows = [json.loads(line) for line in gzip.open(path, "rt")]
    assert [(r["stage"], r["step"], r["uid"]) for r in rows] == [("loop_router", 2, "u2")]
    assert rows[0]["output"] == REORIENT_HINT and "loop signature:" in rows[0]["user"]


# --- tool diag ----------------------------------------------------------------------------

from uni_agent.reflection import ToolDiagReflectionConfig, ToolDiagReflector  # noqa: E402
from uni_agent.reflection.tool_diag import first_failed_edit  # noqa: E402

FAIL_OBS = "No replacement was performed, old_str `x` did not appear verbatim in /f.py."


def _sr(obs, action="str_replace_editor str_replace --path /f.py"):
    return {"name": "str_replace_editor", "action": action, "observation": obs}


def _diag(turns):
    reflector = ToolDiagReflector(None, ToolDiagReflectionConfig(enabled=True, name="tool_diag"))
    return asyncio.run(reflector.reflect_trajectory(task="t", turns=turns, gold="g", feedback="f"))


def test_the_first_failed_edit_gets_its_class_hint():
    turns = [
        {"step": 0, "tokens": 5, "response": "view", "tools": [_sr("ok", "str_replace_editor view --path /f.py")]},
        {"step": 3, "tokens": 5, "response": "edit",
         "tools": [_sr(FAIL_OBS + " Closest match: lines 10-14 (90% similar).")]},
        {"step": 5, "tokens": 5, "response": "edit",
         "tools": [_sr(FAIL_OBS + " has a real line break -- the escaping collapsed your newlines")]},
    ]
    hints = _diag(turns)
    assert list(hints) == [3] and "most recent view" in hints[3]


def test_each_diagnosis_class_maps_to_its_own_hint():
    cases = {
        "the escaping collapsed your newlines": "real newline",
        "every line of your old_str has 2 extra leading space(s)": "leading whitespace",
        "your old_str has trailing whitespace the file does not have": "trailing spaces",
        "Multiple occurrences of old_str": "unique",
        "old_str `x` is the same as new_str": "identical",
        "Closest match: lines 1-3 (80% similar)": "most recent view",
    }
    for frag, expect in cases.items():
        hints = _diag([{"step": 1, "tokens": 5, "response": "e", "tools": [_sr(FAIL_OBS + " " + frag)]}])
        assert expect in hints[1], (frag, hints)


def test_an_unclassified_failure_gets_the_generic_hint():
    hints = _diag([{"step": 2, "tokens": 5, "response": "e", "tools": [_sr(FAIL_OBS)]}])
    assert "copy old_str" in hints[2]


def test_a_failed_view_is_not_a_failed_edit():
    turns = [{"step": 1, "tokens": 5, "response": "v",
              "tools": [_sr(FAIL_OBS, "str_replace_editor view --path /f.py")]}]
    assert _diag(turns) == {}


def test_no_failed_edit_and_no_fallback_yields_no_hints():
    assert _diag([{"step": 0, "tokens": 5, "response": "ok", "tools": [_sr("edited fine")]}]) == {}


def test_tool_diag_delegates_clean_traces_to_the_fallback():
    model = _Model()
    config = ToolDiagReflectionConfig(enabled=True, name="tool_diag", fallback={"enabled": True})
    reflector = ToolDiagReflector(model, config)
    hints = asyncio.run(reflector.reflect_trajectory(task="t", turns=TURNS, gold="g", feedback="f"))
    assert hints == {0: "run the failing test"}


def test_first_failed_edit_reports_the_class():
    step, cls, _, _ = first_failed_edit(
        [{"step": 4, "tokens": 5, "response": "e",
          "tools": [_sr(FAIL_OBS + " every line of your old_str has 1 missing leading space(s)")]}])
    assert (step, cls) == (4, "indent")


def _llm_diag(reply, turns):
    model = _ScriptedModel({"TDIAG": reply})
    config = ToolDiagReflectionConfig(enabled=True, name="tool_diag", source="llm",
                                      system="TDIAG", user="{action}\n{observation}")
    reflector = ToolDiagReflector(model, config)
    return asyncio.run(reflector.reflect_trajectory(task="t", turns=turns, gold="g", feedback="f")), model


def test_llm_source_uses_the_reply_after_the_marker():
    turns = [{"step": 2, "tokens": 5, "response": "e",
              "tools": [_sr(FAIL_OBS + " Closest match: lines 10-14 (90% similar).")]}]
    hints, model = _llm_diag("the window is off\nFINAL_HINT:\nCopy lines 10-14 exactly as the view shows them.", turns)
    assert hints == {2: "Copy lines 10-14 exactly as the view shows them."}
    assert "Closest match" in model.seen[0]["user"]


def test_a_hindsight_worded_llm_hint_falls_back_to_the_class_hint():
    turns = [{"step": 1, "tokens": 5, "response": "e",
              "tools": [_sr(FAIL_OBS + " Closest match: lines 1-3 (80% similar).")]}]
    hints, _ = _llm_diag("FINAL_HINT:\nYour previous attempt went wrong on whitespace.", turns)
    assert "most recent view" in hints[1]


def test_an_llm_reply_without_the_hint_marker_falls_back():
    turns = [{"step": 3, "tokens": 5, "response": "e", "tools": [_sr(FAIL_OBS)]}]
    hints, _ = _llm_diag("no marker here", turns)
    assert "copy old_str" in hints[3]


def test_a_dead_llm_call_still_ships_the_class_hint():
    class _Dead(_Model):
        async def query(self, messages, rollout_cache, sampling_params, max_model_len=None):
            raise RuntimeError("engine gone")

    config = ToolDiagReflectionConfig(enabled=True, name="tool_diag", source="llm")
    reflector = ToolDiagReflector(_Dead(), config)
    hints = asyncio.run(reflector.reflect_trajectory(
        task="t", turns=[{"step": 0, "tokens": 5, "response": "e", "tools": [_sr(FAIL_OBS)]}],
        gold="g", feedback="f"))
    assert "copy old_str" in hints[0]


# --- tool fix -----------------------------------------------------------------------------

from uni_agent.reflection import ToolFixReflectionConfig, ToolFixReflector  # noqa: E402
from uni_agent.reflection.tool_fix import corrected_call  # noqa: E402


def _fix_call(old, new):
    return ('<tool_call>\n{"name": "str_replace_editor", "arguments": {"command": "str_replace", '
            '"path": "/f.py", "old_str": ' + json.dumps(old) + ', "new_str": ' + json.dumps(new) + '}}\n</tool_call>')


def _fix_turn(step, old, new, diag, response_prefix="fixing it now:\n"):
    return {"step": step, "tokens": 5, "response": response_prefix + _fix_call(old, new),
            "tools": [{"name": "str_replace_editor",
                       "action": "str_replace_editor str_replace --path /f.py",
                       "observation": "No replacement was performed, " + diag}]}


def _fix(turns, **cfg):
    reflector = ToolFixReflector(None, ToolFixReflectionConfig(enabled=True, name="tool_fix", **cfg))
    return asyncio.run(reflector.reflect_trajectory(task="t", turns=turns, gold="g", feedback="f"))


def test_indent_correction_fixes_both_strings_and_marks_call_placement():
    old, new = "    a()\n    b()", "    a()\n    c()"
    turns = [_fix_turn(2, old, new, "Closest match: lines 1-2 match exactly except every line "
                                    "of your old_str has 4 extra leading space(s).")]
    hints = _fix(turns)
    assert list(hints) == [2] and hints[2]["at"] == "call"
    assert json.dumps("a()\nb()")[1:-1] in hints[2]["text"], "old_str must be shifted"
    assert json.dumps("a()\nc()")[1:-1] in hints[2]["text"], "new_str must be shifted too"


def test_escape_correction_restores_real_newlines():
    old = "x = 1\\ny = 2"
    turns = [_fix_turn(0, old, "z", "Closest match: lines 1-2 match exactly except your old_str contains "
                                    "the 2-character sequence `\\n` (1x) where the file has a real line "
                                    "break -- the escaping collapsed your newlines.")]
    hints = _fix(turns)
    assert json.dumps("x = 1\ny = 2")[1:-1] in hints[0]["text"]


def test_window_correction_comes_from_the_viewed_file_and_needs_the_diag_check():
    view = {"step": 0, "tokens": 5, "response": "look",
            "tools": [{"name": "str_replace_editor", "action": "str_replace_editor view --path /f.py",
                       "observation": "Here's the result of running `cat -n` on /f.py:\n"
                                      "     1\tdef f():\n     2\t    return 2\n     3\tprint(f())\n"}]}
    fail = _fix_turn(1, "def f():\n    return 3\nprint(f())", "new",
                     "old_str (3 lines) did not appear verbatim in /f.py. Closest match: lines 1-3 "
                     "(92% similar). Differing lines:\n  line 2: file has `return 2`, your old_str has `return 3`")
    hints = _fix([view, fail])
    assert json.dumps("def f():\n    return 2\nprint(f())")[1:-1] in hints[1]["text"]


def test_low_similarity_ships_no_hint():
    turns = [_fix_turn(0, "    a()", "b", "Closest match: lines 1-1 (72% similar). Differing lines:\n"
                                          "  line 1: file has `q`, your old_str has `a()`")]
    assert _fix(turns) == {}


def test_clean_traces_delegate_to_the_fallback():
    reflector = ToolFixReflector(_Model(), ToolFixReflectionConfig(
        enabled=True, name="tool_fix", fallback={"enabled": True}))
    hints = asyncio.run(reflector.reflect_trajectory(task="t", turns=TURNS, gold="g", feedback="f"))
    assert hints == {0: "run the failing test"}


def test_hint_template_must_carry_the_corrected_call():
    with pytest.raises(ValueError, match="corrected_call"):
        ToolFixReflectionConfig(name="tool_fix", hint_template="be careful")


def test_corrected_call_reports_class():
    old = "a \nb"
    turns = [_fix_turn(4, old, "n", "Closest match: lines 1-2 match exactly except your old_str has "
                                    "trailing whitespace the file does not have.")]
    step, cls, call = corrected_call(turns, 0.8)
    assert (step, cls) == (4, "trailing") and json.dumps("a\nb")[1:-1] in call


def test_tool_fix_ships_the_target_when_configured():
    old, new = "    a()", "    b()"
    turns = [_fix_turn(1, old, new, "Closest match: lines 1-1 match exactly except every line "
                                    "of your old_str has 4 extra leading space(s).")]
    reflector = ToolFixReflector(None, ToolFixReflectionConfig(enabled=True, name="tool_fix", target_mask=True))
    hints = asyncio.run(reflector.reflect_trajectory(task="t", turns=turns, gold="g", feedback="f"))
    h = hints[1]
    assert h["at"] == "call" and "target" in h
    assert json.dumps("a()")[1:-1] in h["target"], "target is the corrected call"
    without = ToolFixReflector(None, ToolFixReflectionConfig(enabled=True, name="tool_fix"))
    h2 = asyncio.run(without.reflect_trajectory(task="t", turns=turns, gold="g", feedback="f"))[1]
    assert "target" not in h2, "default ships no target (p1toolfix3 reproducible)"


def test_tool_fix_prefers_the_editor_printed_region_and_rebases_new_str():
    old = "        a()\n        b()"
    new = "        a()\n        c()"
    obs_tail = ("Closest match: lines 1-2 match exactly except every line of your old_str has "
                "4 extra leading space(s).\nThe matching region of the file reads exactly:\n"
                "    a()\n    b()\nResend the call with this, copied character for character, as your old_str.")
    turns = [_fix_turn(0, old, new, obs_tail)]
    reflector = ToolFixReflector(None, ToolFixReflectionConfig(enabled=True, name="tool_fix", target_mask=True))
    h = asyncio.run(reflector.reflect_trajectory(task="t", turns=turns, gold="g", feedback="f"))[0]
    assert json.dumps("    a()\n    b()")[1:-1] in h["target"], "old_str from the printed region"
    assert json.dumps("    a()\n    c()")[1:-1] in h["target"], "new_str rebased to the verified indent"


def test_rebase_falls_back_when_line_counts_differ():
    from uni_agent.reflection.tool_fix import _rebase_new
    assert _rebase_new("a\nb", "a\nc", "x\ny\nz") is None


def test_rebase_transfers_an_insertion_with_reindent():
    from uni_agent.reflection.tool_fix import _rebase_new
    out = _rebase_new("    a()\n    b()", "    a()\n    mid()\n    b()", "  a()\n  b()")
    assert out == "  a()\n  mid()\n  b()"


def test_tool_fix_old_wire_hint_template():
    old, new = "    a()\n    b()", "    a()\n    c()"
    turns = [_fix_turn(3, old, new, "Closest match: lines 1-2 match exactly except every line "
                                    "of your old_str has 4 extra leading space(s).")]
    cfg = ToolFixReflectionConfig(
        enabled=True, name="tool_fix", target_mask=True,
        hint_template='Your next edit must use exactly this value:\n"old_str": {corrected_old_wire}')
    h = asyncio.run(ToolFixReflector(None, cfg).reflect_trajectory(
        task="t", turns=turns, gold="g", feedback="f"))[3]
    assert h["text"].endswith(json.dumps("a()\nb()")), h["text"]
    assert "new_str" not in h["text"], "old-wire hint carries no new_str"
    assert "target" in h and json.dumps("a()\nb()")[1:-1] in h["target"], "mask target still the full call"


def test_tool_fix_template_validation_accepts_either_placeholder():
    ToolFixReflectionConfig(name="tool_fix", hint_template="x {corrected_old_wire}")
    with pytest.raises(ValueError):
        ToolFixReflectionConfig(name="tool_fix", hint_template="no placeholder")


def test_correction_that_collapses_old_into_new_ships_no_hint():
    """The escape/whitespace transforms hit both strings, so an edit whose only intended
    change WAS the escaping corrects into old_str == new_str: a call the editor rejects."""
    old, new = "x = 1\\ny = 2", "x = 1\ny = 2"
    diag = ("Closest match: lines 1-2 match exactly except your old_str contains the 2-character "
            "sequence `\\n` (1x) where the file has a real line break -- the escaping collapsed "
            "your newlines.")
    assert corrected_call([_fix_turn(0, old, new, diag)], 0.8) is None
    # the same edit with a real change still gets its hint
    step, _, call = corrected_call([_fix_turn(0, old, "x = 1\nY = 2", diag)], 0.8)
    assert step == 0 and json.dumps("x = 1\ny = 2")[1:-1] in call


def test_trailing_correction_that_collapses_ships_no_hint():
    old, new = "a \nb", "a\nb"
    diag = ("Closest match: lines 1-2 match exactly except your old_str has trailing whitespace "
            "the file does not have.")
    assert corrected_call([_fix_turn(0, old, new, diag)], 0.8) is None


def _fix_retry_turn(step, old, new, diag=None):
    """A later edit on the same file; `diag=None` makes it succeed."""
    obs = "No replacement was performed, " + diag if diag else "The file /f.py has been edited."
    return {"step": step, "tokens": 5, "response": "trying again:\n" + _fix_call(old, new),
            "tools": [{"name": "str_replace_editor",
                       "action": "str_replace_editor str_replace --path /f.py", "observation": obs}]}


INDENT_DIAG = ("Closest match: lines 1-1 match exactly except every line of your old_str has "
               "4 extra leading space(s).")


def test_hint_at_retry_targets_the_retry_that_failed_again():
    turns = [_fix_turn(0, "    a()", "    b()", INDENT_DIAG),
             _fix_retry_turn(1, "    a()", "    b()", INDENT_DIAG)]
    assert list(_fix(turns, hint_at="call")) == [0]
    assert list(_fix(turns, hint_at="retry")) == [1], "recovery is taught at the retry turn"
    assert sorted(_fix(turns, hint_at="both")) == [0, 1]


def test_hint_at_retry_skips_a_retry_that_succeeded():
    turns = [_fix_turn(0, "    a()", "    b()", INDENT_DIAG),
             _fix_retry_turn(1, "a()", "b()")]
    assert _fix(turns, hint_at="retry") == {}, "a recovered model needs no correction"
    assert list(_fix(turns, hint_at="both")) == [0], "the failed call is still taught"


def test_hint_at_retry_skips_when_the_model_moved_to_another_file():
    other = {"step": 1, "tokens": 5,
             "response": 'trying:\n<tool_call>\n{"name": "str_replace_editor", "arguments": '
                         '{"command": "str_replace", "path": "/other.py", "old_str": "q", '
                         '"new_str": "r"}}\n</tool_call>',
             "tools": [{"name": "str_replace_editor",
                        "action": "str_replace_editor str_replace --path /other.py",
                        "observation": "No replacement was performed, " + INDENT_DIAG}]}
    turns = [_fix_turn(0, "    a()", "    b()", INDENT_DIAG), other]
    assert _fix(turns, hint_at="retry") == {}


def test_hint_at_retry_reaches_past_a_view_turn():
    view = {"step": 1, "tokens": 5, "response": "let me look",
            "tools": [{"name": "str_replace_editor", "action": "str_replace_editor view --path /f.py",
                       "observation": "Here's the result of running `cat -n` on /f.py:\n     1\ta()\n"}]}
    turns = [_fix_turn(0, "    a()", "    b()", INDENT_DIAG), view,
             _fix_retry_turn(2, "    a()", "    b()", INDENT_DIAG)]
    assert list(_fix(turns, hint_at="retry")) == [2]


def test_hint_at_retry_ships_the_same_corrected_call_and_target():
    turns = [_fix_turn(0, "    a()", "    b()", INDENT_DIAG),
             _fix_retry_turn(1, "    a()", "    b()", INDENT_DIAG)]
    hints = _fix(turns, hint_at="both", target_mask=True)
    assert hints[0]["text"] == hints[1]["text"] and hints[0]["at"] == hints[1]["at"] == "call"
    assert hints[0]["target"] == hints[1]["target"]
    assert hints[0] is not hints[1], "each step carries its own hint dict"


def test_hint_at_validation():
    with pytest.raises(ValueError, match="hint_at"):
        ToolFixReflectionConfig(name="tool_fix", hint_at="somewhere")


# --- tool fix: which failures get hinted --------------------------------------------------

INDENT_DIAG = ("Closest match: lines 1-2 match exactly except every line of your old_str "
               "has 4 extra leading space(s).")


def test_hint_failures_first_keeps_one_hint_at_the_earliest_failure():
    turns = [_fix_turn(1, "    a()\n    b()", "    a()\n    c()", INDENT_DIAG),
             _fix_turn(3, "    x()\n    y()", "    x()\n    z()", INDENT_DIAG)]
    assert list(_fix(turns)) == [1]


def test_hint_failures_all_reaches_the_later_failures():
    turns = [_fix_turn(1, "    a()\n    b()", "    a()\n    c()", INDENT_DIAG),
             _fix_turn(3, "    x()\n    y()", "    x()\n    z()", INDENT_DIAG),
             _fix_turn(5, "    p()\n    q()", "    p()\n    r()", INDENT_DIAG)]
    hints = _fix(turns, hint_failures="all")
    assert list(hints) == [1, 3, 5]
    assert json.dumps("x()\ny()")[1:-1] in hints[3]["text"], "each hint carries its own correction"


def test_max_hints_caps_the_all_mode():
    turns = [_fix_turn(s, "    a()\n    b()", f"    a()\n    c{s}()", INDENT_DIAG) for s in (1, 2, 3, 4)]
    assert list(_fix(turns, hint_failures="all", max_hints=2)) == [1, 2]


def test_hint_failures_all_skips_uncorrectable_failures_instead_of_stopping():
    weak = _fix_turn(1, "    a()", "b", "Closest match: lines 1-1 (72% similar). Differing lines:\n"
                                        "  line 1: file has `q`, your old_str has `a()`")
    good = _fix_turn(4, "    x()\n    y()", "    x()\n    z()", INDENT_DIAG)
    assert list(_fix([weak, good], hint_failures="all")) == [4]
    assert _fix([weak, good]) == {}, "first mode still stops at an uncorrectable first failure"


def test_hint_failures_repeat_targets_the_loop_point():
    old, new = "    a()\n    b()", "    a()\n    c()"
    turns = [_fix_turn(1, old, new, INDENT_DIAG),          # first attempt
             _fix_turn(2, "    q()\n    w()", "    q()\n    e()", INDENT_DIAG),
             _fix_turn(3, old, new, INDENT_DIAG)]          # same call again: the loop point
    hints = _fix(turns, hint_failures="repeat")
    assert list(hints) == [3], "the hint lands where the model repeated itself"


def test_hint_failures_repeat_ships_nothing_without_a_repeat():
    turns = [_fix_turn(1, "    a()\n    b()", "    a()\n    c()", INDENT_DIAG),
             _fix_turn(3, "    x()\n    y()", "    x()\n    z()", INDENT_DIAG)]
    assert _fix(turns, hint_failures="repeat") == {}


def test_repeat_mode_sees_repeats_of_uncorrectable_calls():
    weak = "Closest match: lines 1-1 (72% similar). Differing lines:\n  line 1: file has `q`, your old_str has `a()`"
    turns = [_fix_turn(1, "    a()", "b", weak),
             _fix_turn(2, "    a()", "b", weak),
             _fix_turn(3, "    x()\n    y()", "    x()\n    z()", INDENT_DIAG),
             _fix_turn(4, "    x()\n    y()", "    x()\n    z()", INDENT_DIAG)]
    assert list(_fix(turns, hint_failures="repeat")) == [4]


def test_hint_failures_is_validated():
    with pytest.raises(ValueError, match="hint_failures"):
        ToolFixReflectionConfig(name="tool_fix", hint_failures="everywhere")
    with pytest.raises(ValueError, match="max_hints"):
        ToolFixReflectionConfig(name="tool_fix", max_hints=0)


def test_obs_region_reads_the_escaped_json_form():
    from uni_agent.reflection.tool_fix import _obs_region
    obs = ("No replacement was performed, old_str did not appear verbatim in /f.py.\n"
           "Closest match: lines 1-2 match exactly except your old_str has trailing whitespace "
           "the file does not have.\n"
           "The matching region of the file is exactly this JSON value:\n"
           '"old_str": "    x = (0,)\\n    return x"\n'
           "Resend the call with this as your old_str, copied character for character.")
    assert _obs_region(obs) == "    x = (0,)\n    return x"


def test_obs_region_still_reads_the_older_raw_form():
    from uni_agent.reflection.tool_fix import _obs_region
    obs = ("No replacement was performed.\n"
           "The matching region of the file reads exactly:\n"
           "    x = (0,)\n    return x\n"
           "Resend the call with this, copied character for character, as your old_str.")
    assert _obs_region(obs) == "    x = (0,)\n    return x"
