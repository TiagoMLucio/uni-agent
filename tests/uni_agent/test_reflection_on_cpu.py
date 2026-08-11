"""The reference patch is capped: an outsized one would overflow every rung of the ladder."""

from __future__ import annotations

import asyncio

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
    assert ReflectionConfig().shrink_ladder[0] == (2000, None)


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
