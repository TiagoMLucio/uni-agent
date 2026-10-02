"""A reflector reply that parses to nothing is re-drawn, and a call that cannot run fails.

Contract failures are drawn per sample rather than being properties of the trace: across
three repeats of one experiment, zero traces failed in all three (Cohen's kappa about 0). So a
re-draw recovers most of them, while rewording the prompt does not. These tests pin the
behaviours that follow: re-draw on an unusable reply at the rung it was drawn from, shrink the
render only when it does not fit, fail rather than come up empty when no call could run, read a hint out of an
object no JSON decoder will take, and count what the whole thing cost.
"""

import asyncio

import pytest

from uni_agent.reflection.base import AbstractReflector, ReflectionFailed
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


def test_an_over_budget_view_is_rendered_once():
    """The whole view overflows and the output step fits: the same renders however many redraws
    a view allows, since an over-budget render is deterministic and the redraws would repeat it."""
    seen = set()
    for redraws in (0, 1, 3):
        hints, model, _ = run_ladder(60, 100_000, redraws_per_rung=redraws)
        assert hints == {2: "run the snippet you printed at turn 1 before editing"}
        assert len(model.queries) == 1 and model.queries[0][1] == 16384
        seen.add(tuple(model.renders))
    assert len(seen) == 1, seen


def test_the_output_cut_is_the_largest_that_fits():
    """Only outputs longer than the cut lose their middle, so the cut is searched, not stepped:
    the prompt sent fills the room left for the reply to within a few percent."""
    hints, model, _ = run_ladder(60, 100_000)
    assert hints
    sent = model.queries[0][0]
    assert sent + 16384 <= 262144 and sent + 16384 > 262144 * 0.95, sent


def test_a_prompt_without_reply_room_is_over_budget():
    """A prompt that fits but leaves less than max_output_tokens of room is not sent: the staged
    reply could not close, and the shrink ladder moves on without paying the prefill. With no step
    left the reflector never saw the trajectory, so the reflection failed rather than came up empty."""
    good = MARKER + '\n{"turn2": "run the snippet you printed at turn 1 before editing"}'
    model = SizedModel([good], 262144)
    r = PipelineReflector(model, config(max_model_len=262144, max_observation_chars=1_000_000,
                                       max_output_tokens=16384, redraws_per_rung=1))
    with pytest.raises(ReflectionFailed, match="over budget at every shrink level"):
        asyncio.run(r.reflect_trajectory(task="t", turns=turns_with_observations(100, 100_000),
                                         gold="g", feedback="f"))
    assert r.call_metrics()["reflect_over_budget"] == 3 and r.call_metrics()["reflect_calls"] == 0
    assert model.queries == [], "every step leaves under 16384 tokens of room"
    assert len(model.renders) == 3, "one render per step, none repeated"
    assert all(262144 - 16384 < n < 262144 for n in model.renders[1:]), "the steps fit by prompt length alone"


GOOD = MARKER + '\n{"turn1": "look at the parser in foo.py before editing it"}'
BAD = "no object here at all"


def test_a_rejected_reply_never_buys_a_smaller_view():
    """The ladder is for a render that does not fit, so an unusable reply is re-drawn where it
    was drawn: 1 + redraws_per_rung calls at rung 0, not one per rung down the ladder."""
    hints, model, _ = reflect([BAD], turns=turns_with_observations(3, 20_000),
                              max_observation_chars=50_000, redraws_per_rung=1)
    assert hints == {}
    assert model.calls == 2 and len(model.rendered) == 1, (model.calls, len(model.rendered))
    assert "chars elided" not in model.rendered[0], "the whole view renders the observations uncapped"


def test_an_over_budget_view_advances_where_a_rejected_reply_does_not():
    """The whole view does not fit; both draws are the output step's view, and the response step
    after it is never rendered."""
    hints, model, _ = run_ladder(20, 100_000, replies=[BAD], redraws_per_rung=1)
    assert hints == {}
    assert len(model.queries) == 2 and model.queries[0] == model.queries[1], model.queries


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


def test_a_call_that_raises_fails_the_reflection():
    """An API error, a rate limit or a timeout is not the reflector finding nothing to hint."""

    class _Down(Model):
        async def query(self, *args, **kwargs):
            raise ConnectionError("503 from the reflector endpoint")

    r = PipelineReflector(_Down([GOOD]), config())
    with pytest.raises(ReflectionFailed, match="503"):
        asyncio.run(r.reflect_trajectory(task="t", turns=TURNS, gold="g", feedback="f"))


def test_a_reply_unusable_after_every_redraw_is_empty_not_failed():
    """The model answered and the parser found nothing: the pipeline ran as designed."""
    hints, calls = run([BAD], redraws_per_rung=1)
    assert hints == {} and calls == 2


class SeeingModel(SizedModel):
    """A SizedModel that also keeps what it was last asked to answer."""

    async def query(self, messages, rollout_cache, **kwargs):
        self.sent = messages[-1]["content"]
        return await super().query(messages, rollout_cache, **kwargs)


def reflect_sized(turns, max_model_len, feedback="f", **over):
    model = SeeingModel([GOOD], max_model_len)
    r = PipelineReflector(model, PipelineReflector.Config(
        name="pipeline", max_model_len=max_model_len, max_output_tokens=1000, max_observation_chars=100_000,
        calls=[{"id": "test", "per": "trace", "parse": "hints", "system": "emit hints",
                "user": "{task}\n{turns}\n{feedback}"}], **over,
    ))
    hints = asyncio.run(r.reflect_trajectory(task="t", turns=turns, gold="g", feedback=feedback))
    return hints, model, r


def test_the_largest_outputs_give_way_first():
    """One runaway output and ten file views: only the runaway loses its middle."""
    views = [f"view {i} " + "v" * 8_000 for i in range(10)]
    turns = [{"step": 1, "response": "run it", "tools": [{"name": "execute_bash", "observation": "r" * 90_000}]}]
    turns += [{"step": i + 2, "response": "look", "tools": [{"name": "str_replace_editor", "observation": v}]}
              for i, v in enumerate(views)]
    hints, model, r = reflect_sized(turns, 40_000)
    assert hints and r.call_metrics()["reflect_rung"] == 1
    assert all(v in model.sent for v in views), "a file view was cut"
    assert model.sent.count("chars elided") == 1


def test_responses_give_way_after_outputs_and_the_call_text_before_the_words():
    words = "The parser drops the last token because the loop stops one short, so " * 20
    call = "<tool_call>\n<function=str_replace_editor>\n<parameter=command>\ncreate\n</parameter>\n" \
           "<parameter=file_text>\n" + "x = 1\n" * 15_000 + "\n</parameter>\n</function>\n</tool_call>"
    turns = [{"step": 1, "response": words + call, "tools": [{"name": "str_replace_editor", "observation": "ok"}]}]
    hints, model, r = reflect_sized(turns, 12_000)
    assert hints and r.call_metrics()["reflect_rung"] == 2
    assert words in model.sent, "the agent's own words were cut"
    assert "chars elided" in model.sent


def test_feedback_is_never_cut_by_the_reflector():
    feedback = "[Diagnostic test feedback begins]\n" + "E AssertionError: case\n" * 2_000 + "[Diagnostic test feedback ends]"
    turns = [{"step": 1, "response": "run it", "tools": [{"name": "execute_bash", "observation": "r" * 90_000}]}]
    hints, model, _ = reflect_sized(turns, 30_000, feedback=feedback)
    assert hints and feedback in model.sent


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
