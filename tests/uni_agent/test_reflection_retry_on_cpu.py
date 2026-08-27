"""A reflector reply that parses to nothing is re-drawn, and no stage may lose hints.

Contract failures are drawn per sample rather than being properties of the trace: across
three repeats of one arm, zero traces failed in all three (Cohen's kappa about 0). So a
re-draw recovers most of them, while rewording the prompt does not. These tests pin the
three behaviours that follow: re-draw on an unusable reply, keep the hints an earlier
stage earned, and read a hint out of an object no JSON decoder will take.
"""

import asyncio

from uni_agent.reflection.base import AbstractReflector
from uni_agent.reflection.pipeline import PipelineReflector

MARKER = "FINAL_HINTS_JSON:"
TURNS = [{"step": 1, "response": "a", "tools": []}, {"step": 2, "response": "b", "tools": []}]


class Model:
    """Serves canned replies in order; records how many calls it saw."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.sampling_params = {}
        self.tokenizer = None

    async def prepare_rollout_cache(self, messages, include_tools=True, chat_template_kwargs=None):
        return {"prompt_ids": [0]}

    async def query(self, messages, rollout_cache, **kwargs):
        self.calls += 1
        return self.replies[min(self.calls - 1, len(self.replies) - 1)], None, None, None


def config(**over):
    cfg = PipelineReflector.Config(
        name="pipeline",
        calls=[{"id": "cascade", "per": "trace", "parse": "hints",
                "system": 'emit {k} hints as {"turn<index>": "<hint>"}',
                "user": "{task}\n{turns}"}],
        **over,
    )
    return cfg


def run(replies):
    model = Model(replies)
    r = PipelineReflector(model, config())
    hints = asyncio.run(
        r.reflect_trajectory(task="t", turns=TURNS, gold="g", feedback="f", outcome="o", agent_patch="p")
    )
    return hints, model.calls


def test_an_unusable_reply_is_redrawn():
    good = MARKER + '\n{"turn1": "look at the parser in foo.py before editing it"}'
    hints, calls = run(["no object here at all", good])
    assert hints == {1: "look at the parser in foo.py before editing it"}, hints
    assert calls >= 2, f"expected a re-draw, saw {calls} call(s)"


def test_a_usable_reply_is_not_redrawn():
    good = MARKER + '\n{"turn1": "look at the parser in foo.py before editing it"}'
    hints, calls = run([good])
    assert hints and calls == 1, (hints, calls)


def test_an_unescaped_quote_still_yields_its_hint():
    # one stray quote used to cost the rollout every hint in the reply
    reply = MARKER + '\n{"turn2": "the call passes "utf-8" positionally, so it lands in errors="}'
    assert AbstractReflector._parse(reply).get(2, "").startswith("the call passes")


def test_a_control_character_inside_a_hint_is_tolerated():
    reply = MARKER + '\n{"turn1": "run this:\nmake test\nand read the failure"}'
    assert 1 in AbstractReflector._parse(reply)


def test_prose_alone_is_not_mined_for_hints():
    # mining the analysis was measured to invent hints where the model declined
    assert AbstractReflector._parse("TARGET: turn 3 looks wrong. COVERAGE: none.") == {}


def test_an_explicit_decline_stays_a_decline():
    assert AbstractReflector._parse(MARKER + "\n{}") == {}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
