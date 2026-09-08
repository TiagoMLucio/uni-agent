"""A reflector reply that parses to nothing is re-drawn, and no stage may lose hints.

Contract failures are drawn per sample rather than being properties of the trace: across
three repeats of one experiment, zero traces failed in all three (Cohen's kappa about 0). So a
re-draw recovers most of them, while rewording the prompt does not. These tests pin the
behaviours that follow: re-draw on an unusable reply at the rung it was drawn from, shrink the
render only when it does not fit, keep the hints an earlier stage earned, read a hint out of an
object no JSON decoder will take, and count what the whole thing cost.
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
        self.rendered = []
        self.sampling_params = {}
        self.tokenizer = None

    async def prepare_rollout_cache(self, messages, include_tools=True, chat_template_kwargs=None):
        self.rendered.append(messages[1]["content"])
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


def reflect(replies, turns=TURNS, **over):
    model = Model(replies)
    r = PipelineReflector(model, config(**over))
    hints = asyncio.run(
        r.reflect_trajectory(task="t", turns=turns, gold="g", feedback="f", outcome="o", agent_patch="p")
    )
    return hints, model, r


def run(replies, **over):
    hints, model, _ = reflect(replies, **over)
    return hints, model.calls


def expected_metrics(calls, redraws, over_budget, rung):
    return {"reflect_calls": float(calls), "reflect_redraws": float(redraws),
            "reflect_over_budget": float(over_budget), "reflect_rung": float(rung)}


def test_an_unusable_reply_is_redrawn():
    good = MARKER + '\n{"turn1": "look at the parser in foo.py before editing it"}'
    hints, calls = run(["no object here at all", good])
    assert hints == {1: "look at the parser in foo.py before editing it"}, hints
    assert calls >= 2, f"expected a re-draw, saw {calls} call(s)"


def test_a_usable_reply_is_not_redrawn():
    good = MARKER + '\n{"turn1": "look at the parser in foo.py before editing it"}'
    hints, calls = run([good])
    assert hints and calls == 1, (hints, calls)


def test_redraws_per_rung_is_the_draw_count_on_one_rung():
    # one rung only, so every extra call is a same-rung re-draw
    for redraws, calls in ((0, 1), (1, 2), (3, 4)):
        _, seen = run(["no object here at all"], shrink_ladder=[], redraws_per_rung=redraws)
        assert seen == calls, (redraws, seen)


class SizedModel(Model):
    """Prompt length = chars // 4; over-budget prompts raise where the real client does."""

    def __init__(self, replies, max_model_len):
        super().__init__(replies)
        self.max_model_len = max_model_len
        self.renders = []
        self.queries = []

    async def prepare_rollout_cache(self, messages, include_tools=True, chat_template_kwargs=None):
        n = sum(len(m["content"]) for m in messages) // 4
        self.renders.append(n)
        return {"prompt_ids": list(range(n))}

    async def query(self, messages, rollout_cache, sampling_params=None, max_model_len=None, **kwargs):
        from uni_agent.interaction.model import MaxTokenExceededError

        n = len(rollout_cache["prompt_ids"])
        if n >= (max_model_len or self.max_model_len):
            raise MaxTokenExceededError(f"{n} >= {max_model_len}")
        self.queries.append((n, sampling_params["max_tokens"]))
        return await super().query(messages, rollout_cache)


def turns_with_observations(n_turns, obs_chars):
    return [
        {"step": i + 1, "response": "resp " * 50,
         "tools": [{"name": "execute_bash", "action": "python x.py", "observation": "o" * obs_chars}]}
        for i in range(n_turns)
    ]


def run_ladder(n_turns, obs_chars, max_model_len=262144, replies=None, **over):
    good = MARKER + '\n{"turn2": "run the snippet you printed at turn 1 before editing"}'
    model = SizedModel(replies or [good], max_model_len)
    r = PipelineReflector(model, config(max_model_len=max_model_len, max_observation_chars=1_000_000,
                                       max_output_tokens=16384, **over))
    hints = asyncio.run(r.reflect_trajectory(task="t", turns=turns_with_observations(n_turns, obs_chars),
                                             gold="g", feedback="f"))
    return hints, model, r


def test_an_over_budget_rung_is_rendered_once():
    """Rungs 0 and 1 overflow, rung 2 fits: three renders however many redraws a rung allows,
    since an over-budget render is deterministic and the redraws would repeat it."""
    for redraws in (0, 1, 3):
        hints, model, _ = run_ladder(230, 100_000, redraws_per_rung=redraws)
        assert hints == {2: "run the snippet you printed at turn 1 before editing"}
        assert len(model.renders) == 3, (redraws, model.renders)
        assert len(model.queries) == 1 and model.queries[0][1] == 16384


def test_a_prompt_without_reply_room_is_over_budget():
    """A prompt that fits but leaves less than max_output_tokens of room is not sent: the staged
    reply could not close, and the shrink ladder moves on without paying the prefill."""
    hints, model, _ = run_ladder(250, 100_000, redraws_per_rung=1)
    assert model.queries == [], "every rung leaves under 16384 tokens of room"
    assert hints == {}
    assert len(model.renders) == 5, "one render per rung, none repeated"
    assert all(n < 262144 for n in model.renders[2:]), "the last rungs fit by prompt length alone"
    assert all(n + 16384 > 262144 for n in model.renders[2:])


GOOD = MARKER + '\n{"turn1": "look at the parser in foo.py before editing it"}'
BAD = "no object here at all"


def test_a_rejected_reply_never_buys_a_smaller_view():
    """The ladder is for a render that does not fit, so an unusable reply is re-drawn where it
    was drawn: 1 + redraws_per_rung calls at rung 0, not one per rung down the ladder."""
    hints, model, _ = reflect([BAD], turns=turns_with_observations(3, 20_000),
                              max_observation_chars=50_000, redraws_per_rung=1)
    assert hints == {}
    assert model.calls == 2 and len(model.rendered) == 2, (model.calls, len(model.rendered))
    assert model.rendered[0] == model.rendered[1], "the re-draw saw the same view"
    assert "chars elided" not in model.rendered[0], "rung 0 renders the observations uncapped"


def test_an_over_budget_rung_advances_where_a_rejected_reply_does_not():
    """Rung 0 does not fit and costs a rung; the two draws that follow are the rung that fits,
    and the rungs below it are never rendered."""
    hints, model, _ = run_ladder(20, 100_000, replies=[BAD], redraws_per_rung=1)
    assert hints == {}
    assert len(model.renders) == 3, model.renders
    assert model.renders[1] == model.renders[2], "both draws at the rung that fit"
    assert len(model.queries) == 2, model.queries


def test_the_metrics_count_a_call_answered_first_time():
    _, _, r = reflect([GOOD])
    assert r.call_metrics() == expected_metrics(calls=1, redraws=0, over_budget=0, rung=0)


def test_the_metrics_count_a_reply_recovered_by_a_re_draw():
    _, _, r = reflect([BAD, GOOD])
    assert r.call_metrics() == expected_metrics(calls=2, redraws=1, over_budget=0, rung=0)


def test_the_metrics_count_a_reply_that_never_parses():
    _, _, r = reflect([BAD])
    assert r.call_metrics() == expected_metrics(calls=2, redraws=1, over_budget=0, rung=0)


def test_the_metrics_name_the_rung_the_answer_came_from():
    # 20 turns of 100k chars overflow the full view and fit once the observations are capped
    hints, _, r = run_ladder(20, 100_000, redraws_per_rung=1)
    assert hints
    assert r.call_metrics() == expected_metrics(calls=1, redraws=0, over_budget=1, rung=1)


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
