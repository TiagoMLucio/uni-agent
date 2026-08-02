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

    async def query(self, messages, rollout_cache, sampling_params):
        self.messages = messages
        return '{"turn0": "run the failing test"}', None, None, None


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
