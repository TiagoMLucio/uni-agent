"""The hosted reflector: its best turns in the model's order, token use counted, the render measured with
the policy's tokenizer and cut down the shared ladder, the feedback never cut, and hints citing names the
agent never saw logged, not dropped."""

import asyncio
import json
import sys
import types

import pytest

from uni_agent.reflection.api import ApiReflector, unseen_names
from uni_agent.reflection.base import ReflectionFailed

TURNS = [
    {"step": 0, "response": "Let me look.\n<tool_call>\n<function=execute_bash>\n<parameter=command>\ncat a.py\n"
                            "</parameter>\n</function>\n</tool_call>", "tools": [{"observation": "def parse_row(x): ..."}]},
    {"step": 1, "response": "parse_row looks fine.", "tools": [{"observation": "ok"}]},
    {"step": 2, "response": "Done.", "tools": []},
]
TASK = "preamble <issue_description>\nrows are dropped\n</issue_description> postscript"


class Policy:
    """The policy's client as the reflector measures with it: a token per four characters."""

    def __init__(self, max_model_len=None):
        self.max_model_len = max_model_len

    async def prepare_rollout_cache(self, messages, include_tools=False, chat_template_kwargs=None):
        return {"prompt_ids": [0] * (sum(len(m["content"]) for m in messages) // 4)}


def answer(*top):
    return json.dumps({"problems": [{"id": "P1", "where": "a.py", "why": "w", "timeline": []}], "turns": [],
                       "top": [{"turn": t, "problem": "P1", "hint": h, "why": "y"} for t, h in top]})


class APIStatusError(Exception):
    def __init__(self, message="", status_code=500, code=None, param=None):
        super().__init__(message)
        self.status_code, self.code, self.param = status_code, code, param


class BadRequestError(APIStatusError):
    def __init__(self, message="", code=None, param=None):
        super().__init__(message, 400, code, param)


class RateLimitError(APIStatusError):
    def __init__(self, message="", code=None):
        super().__init__(message, 429, code)


seen_clients = []
#: a reply that never comes
STALL = object()
#: the ChatGPT plan's requests (a client whose key is a plan's access token), and which plan's token each used
plan_seen = []
plan_keys = []


def plan_usage():
    return types.SimpleNamespace(input_tokens=500, output_tokens=60, input_tokens_details=types.SimpleNamespace(cached_tokens=0),
                                 output_tokens_details=types.SimpleNamespace(reasoning_tokens=50))


def fake_openai(replies, seen, plan=()):
    async def plan_stream(reply):
        if isinstance(reply, tuple):
            yield types.SimpleNamespace(type="response.failed", response=types.SimpleNamespace(
                error=types.SimpleNamespace(code=reply[1], message="failed")))
            return
        # as the plan streams it: the answer only in the deltas, an unstored response completing with no output
        for start in range(0, len(reply), 7):
            yield types.SimpleNamespace(type="response.output_text.delta", delta=reply[start:start + 7])
        yield types.SimpleNamespace(type="response.output_text.done", text=reply)
        yield types.SimpleNamespace(type="response.completed", response=types.SimpleNamespace(
            output_text="", usage=plan_usage(), status="completed"))

    class PlanResponses:
        async def create(self, **kwargs):
            plan_seen.append(kwargs)
            reply = plan[min(len(plan_seen) - 1, len(plan) - 1)]
            if isinstance(reply, Exception):
                raise reply
            return plan_stream(reply)

    class Responses:
        async def create(self, **kwargs):
            seen.append(kwargs)
            reply = replies[min(len(seen) - 1, len(replies) - 1)]
            if reply is STALL:
                await asyncio.Event().wait()
            if isinstance(reply, Exception):
                raise reply
            usage = types.SimpleNamespace(
                input_tokens=100, output_tokens=40,
                input_tokens_details=types.SimpleNamespace(cached_tokens=10),
                output_tokens_details=types.SimpleNamespace(reasoning_tokens=30))
            return types.SimpleNamespace(output_text=reply, usage=usage, status="completed")

    class Completions:
        async def create(self, **kwargs):
            seen.append(kwargs)
            reply = replies[min(len(seen) - 1, len(replies) - 1)]
            if isinstance(reply, Exception):
                raise reply
            usage = types.SimpleNamespace(prompt_tokens=200, completion_tokens=80, prompt_tokens_details=None,
                                          completion_tokens_details=None)
            choice = types.SimpleNamespace(message=types.SimpleNamespace(content=reply), finish_reason="stop")
            return types.SimpleNamespace(choices=[choice], usage=usage)

    class AsyncOpenAI:
        def __init__(self, **kwargs):
            seen_clients.append(kwargs)
            on_plan = str(kwargs.get("api_key")).startswith("plan-token")
            if on_plan:
                plan_keys.append(kwargs["api_key"])
            self.responses = PlanResponses() if on_plan else Responses()
            self.chat = types.SimpleNamespace(completions=Completions())

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    return types.SimpleNamespace(AsyncOpenAI=AsyncOpenAI, APIStatusError=APIStatusError, BadRequestError=BadRequestError,
                                 RateLimitError=RateLimitError)


def reflect(monkeypatch, replies, turns=TURNS, feedback="1 failed", agent_patch="diff --git a/a.py b/a.py\n+y",
            policy=None, raises=None, plan=(), reflector=None, **over):
    seen = []
    plan_seen.clear()
    plan_keys.clear()
    monkeypatch.setitem(sys.modules, "openai", fake_openai(replies, seen, plan))
    cfg = ApiReflector.Config(**{
        "name": "openai", "model": "gpt-6-luna", "enabled": True,
        "calls": [{"id": "luna", "system": "sys", "parse": "hints",
                   "user": "{task}|{first}-{last}|{turns}|{agent_patch}|{gold}|{feedback}"}],
        **over})
    r = reflector or ApiReflector(policy or Policy(), cfg)

    def run():
        return asyncio.run(r.reflect_trajectory(task=TASK, turns=turns, gold="diff --git a/a.py b/a.py\n+x",
                                                feedback=feedback, agent_patch=agent_patch))

    if raises is None:
        return run(), seen, r
    with pytest.raises(raises):
        run()
    return None, seen, r


def test_best_turn_first_and_render(monkeypatch):
    hints, seen, r = reflect(monkeypatch, [answer((2, "later"), (1, "check `parse_row`"))], max_selected_turns=1)
    assert hints == {2: "later"}
    user = seen[0]["input"]
    assert user.startswith("rows are dropped|0-2|")
    assert "### Turn 0\nLet me look.\n<tool_call>\n<function=execute_bash>" in user
    assert "### Output of turn 0\ndef parse_row(x): ...\n\n### Turn 1\nparse_row looks fine." in user
    assert user.count("### Turn ") == 3 and "### Output of turn 2\n(no tool call)" in user
    assert seen[0]["model"] == "gpt-6-luna" and seen[0]["reasoning"] == {"effort": "high"}
    metrics = r.call_metrics()
    assert metrics["reflect_input_tokens"] == 100 and metrics["reflect_reasoning_tokens"] == 30


def test_attempt_patch_lists_created_files(monkeypatch):
    patch = ("diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n"
             "diff --git a/reproduce.py b/reproduce.py\nnew file mode 100644\n--- /dev/null\n+++ b/reproduce.py\n"
             "@@ -0,0 +1,2 @@\n+import a\n+print(a)\n")
    _, seen, _ = reflect(monkeypatch, [answer((1, "ok"))], agent_patch=patch)
    user = seen[0]["input"]
    assert "+++ b/a.py" in user and "Files the attempt created, not shown:\n- reproduce.py (2 lines)" in user
    assert "print(a)" not in user
    _, seen, _ = reflect(monkeypatch, [answer((1, "ok"))], agent_patch="")
    assert "(empty: no change was extracted from the attempt)" in seen[0]["input"]


def test_overflow_cuts_the_largest_outputs_and_never_the_feedback(monkeypatch):
    turns = [{**TURNS[0], "tools": [{"observation": "head " + "o" * 40_000 + " tail"}]}, *TURNS[1:]]
    feedback = "".join(f"test_{i} failed: ValueError {i}\n" for i in range(300))
    hints, seen, r = reflect(monkeypatch, [answer((1, "ok"))], turns=turns, feedback=feedback,
                             policy=Policy(max_model_len=10_000), max_output_tokens=2_000,
                             max_observation_chars=100_000)
    assert hints == {1: "ok"} and len(seen) == 1
    user = seen[0]["input"]
    assert feedback.strip() in user and "chars elided" in user and "head " in user and " tail" in user
    assert len(user) // 4 + 2_000 <= 10_000 and len(user) // 4 + 2_000 > 9_000
    assert r.call_metrics()["reflect_over_budget"] == 1 and r.call_metrics()["reflect_rung"] == 1


def test_feedback_alone_over_budget_fails_without_a_call(monkeypatch):
    _, seen, r = reflect(monkeypatch, [answer((1, "ok"))], feedback="f" * 50_000,
                         policy=Policy(max_model_len=10_000), max_output_tokens=2_000, raises=ReflectionFailed)
    assert seen == []
    assert r.call_metrics()["reflect_over_budget"] == 2


def test_context_error_moves_to_the_next_step(monkeypatch):
    replies = [BadRequestError("context_length_exceeded"), answer((1, "ok"))]
    hints, seen, r = reflect(monkeypatch, replies, max_observation_chars=10**6)
    assert hints == {1: "ok"} and len(seen) == 2
    assert r.call_metrics()["reflect_over_budget"] == 1


def test_response_cut_keeps_the_reasoning_and_the_agents_words():
    cfg = ApiReflector.Config(name="openai", model="gpt-6-luna",
                              calls=[{"id": "luna", "system": "sys", "parse": "hints", "user": "{turns}"}])
    response = ("<think>" + "t" * 5_000 + "</think>I will write the module.\n<tool_call>\n"
                "<function=str_replace_editor>\n<parameter=command>\ncreate\n</parameter>\n<parameter=path>\n"
                "/testbed/a.py\n</parameter>\n<parameter=file_text>\n" + "y" * 10_000 + "\n</parameter>\n"
                "</function>\n</tool_call>")
    out = ApiReflector(Policy(), cfg)._render_attempt([{"step": 0, "response": response, "tools": []}], None, 3_800)
    thought, _, rest = out.partition("</think>")
    assert "t" * 5_000 in thought and "chars elided" not in thought
    assert "I will write the module." in rest and "<function=str_replace_editor>" in rest and "chars elided" in rest
    assert len(rest) < 3_900


def test_reasoning_cut_off_by_the_token_limit_stays_whole():
    cfg = ApiReflector.Config(name="openai", model="gpt-6-luna",
                              calls=[{"id": "luna", "system": "sys", "parse": "hints", "user": "{turns}"}])
    out = ApiReflector(Policy(), cfg)._render_attempt([{"step": 1, "response": "r" * 5_000, "tools": []}], None, 1_000)
    assert "r" * 5_000 in out and "chars elided" not in out


def test_turn_is_rendered_as_generated():
    cfg = ApiReflector.Config(name="openai", model="gpt-6-luna",
                              calls=[{"id": "luna", "system": "sys", "parse": "hints", "user": "{turns}"}])
    turns = [{"step": 3, "response": "Check the parser.</think>\nI will read it.<|im_end|>", "tools": [{"observation": "x"}]},
             {"step": 4, "response": "No reasoning here.", "tools": []}]
    out = ApiReflector(Policy(), cfg)._render_attempt(turns, None, None)
    assert out == ("### Turn 3\n<think>\nCheck the parser.</think>\nI will read it.\n\n### Output of turn 3\nx\n\n"
                   "### Turn 4\nNo reasoning here.\n\n### Output of turn 4\n(no tool call)")


def turn_hint(turn, hint):
    return json.dumps({"problem": "p", "turn": turn, "evidence": "e", "hint": hint})


def test_turn_hint_answer(monkeypatch):
    hints, seen, _ = reflect(monkeypatch, [turn_hint(9, "no such turn"), turn_hint(1, "check `parse_row`")],
                             calls=[{"id": "luna", "system": "sys", "parse": "turn_hint",
                                     "user": "{task}|{first}-{last}|{turns}|{agent_patch}|{gold}|{feedback}"}])
    assert hints == {1: "check `parse_row`"} and len(seen) == 2
    assert seen[0]["text"]["format"]["name"] == "hint_turn" and seen[0]["text"]["format"]["strict"]


def test_parse_belongs_to_its_reflector():
    import pytest
    from uni_agent.reflection.pipeline import PipelineReflectionConfig

    with pytest.raises(ValueError, match="parses its answer"):
        ApiReflector.Config(name="openai", model="m", calls=[{"id": "x", "system": "s", "user": "{turns}"}])
    with pytest.raises(ValueError, match="api reflector's answer"):
        PipelineReflectionConfig(name="pipeline", calls=[
            {"id": "a", "system": "s", "user": "{turns}", "parse": "turn_hint"},
            {"id": "b", "system": "s", "user": "{turns}", "parse": "hints"}])


def test_responses_request_carries_the_extra_parameters(monkeypatch):
    hints, seen, _ = reflect(monkeypatch, [answer((1, "ok"))], sampling={"service_tier": "flex"})
    assert hints == {1: "ok"} and seen[0]["service_tier"] == "flex" and seen[0]["reasoning"] == {"effort": "high"}


def test_chat_completions_endpoint(monkeypatch):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "k")
    hints, seen, r = reflect(monkeypatch, [answer((1, "ok"))], base_url="https://api.deepinfra.com/v1/openai",
                             api_key_env="DEEPINFRA_API_KEY", sampling={"temperature": 1.0, "top_p": 0.95})
    assert hints == {1: "ok"}
    assert seen_clients[-1]["base_url"] == "https://api.deepinfra.com/v1/openai" and seen_clients[-1]["api_key"] == "k"
    req = seen[0]
    assert req["messages"][0] == {"role": "system", "content": "sys"} and req["response_format"] == {"type": "json_object"}
    assert req["extra_body"] == {"reasoning_effort": "high"} and req["temperature"] == 1.0 and req["top_p"] == 0.95
    assert r.call_metrics()["reflect_input_tokens"] == 200 and r.call_metrics()["reflect_output_tokens"] == 80


def test_invalid_turns_skipped_and_unusable_redrawn(monkeypatch):
    hints, seen, _ = reflect(monkeypatch, ["not json", answer((9, "no such turn"), (1, "ok"))], max_selected_turns=1)
    assert hints == {1: "ok"} and len(seen) == 2


def test_rate_limit_waited_out(monkeypatch):
    from uni_agent.reflection import api
    monkeypatch.setattr(api.random, "uniform", lambda a, b: 0.0)
    replies = [RateLimitError("Rate limit reached ... Please try again in 10ms."), answer((1, "ok"))]
    hints, seen, r = reflect(monkeypatch, replies)
    assert hints == {1: "ok"} and len(seen) == 2
    assert r.call_metrics()["reflect_rate_limited"] == 1 and r.call_metrics()["reflect_calls"] == 1


def test_a_call_error_is_a_failed_reflection(monkeypatch):
    _, seen, _ = reflect(monkeypatch, [RuntimeError("401")], raises=ReflectionFailed)
    assert len(seen) == 1


@pytest.mark.parametrize("message", [
    "Error code: 429 - {'error': {'message': 'You exceeded your current quota', 'code': 'insufficient_quota'}}",
    "Error code: 429 - Request too large for gpt-6-luna on tokens per min (TPM): Limit 200000, Requested 210000.",
])
def test_a_429_that_never_clears_is_not_waited_out(monkeypatch, message):
    _, seen, r = reflect(monkeypatch, [RateLimitError(message), answer((1, "ok"))], raises=ReflectionFailed)
    assert len(seen) == 1 and r.call_metrics()["reflect_rate_limited"] == 0


def test_the_asked_wait_is_read_in_any_unit():
    from uni_agent.reflection.api import retry_after
    assert retry_after("Please try again in 2s.") == 2.0 and retry_after("try again in 20ms") == 0.02
    assert retry_after("try again in 6m0s.") == 360.0 and retry_after("try again in 1h2m3.5s") == 3723.5
    assert retry_after("Rate limit reached for requests") is None


def test_a_per_day_limit_is_not_waited_out_but_a_per_minute_one_is(monkeypatch):
    from uni_agent.reflection import api
    monkeypatch.setattr(api.random, "uniform", lambda a, b: 0.0)
    day = RateLimitError("Rate limit reached ... on requests per day (RPD): Limit 10000. Please try again in 7h12m0s.")
    _, seen, r = reflect(monkeypatch, [day, answer((1, "ok"))], raises=ReflectionFailed)
    assert len(seen) == 1 and r.call_metrics()["reflect_rate_limited"] == 0
    minute = RateLimitError("Rate limit reached ... on tokens per min (TPM): Limit 200000. Please try again in 15ms.")
    hints, seen, r = reflect(monkeypatch, [minute, answer((1, "ok"))])
    assert hints == {1: "ok"} and len(seen) == 2 and r.call_metrics()["reflect_rate_limited"] == 1


def test_a_stalled_call_ends_at_the_wall_budget(monkeypatch):
    from uni_agent.reflection import api
    monkeypatch.setattr(api, "WALL_S", 0.05)
    _, seen, _ = reflect(monkeypatch, [STALL], raises=ReflectionFailed)
    assert len(seen) == 1


def test_hints_with_chat_control_text_are_dropped_and_counted(monkeypatch):
    hints, seen, r = reflect(monkeypatch, [answer((1, "print <|im_end|> after the header")), answer((1, "ok"))],
                             max_selected_turns=1)
    assert hints == {1: "ok"} and len(seen) == 2
    assert r.call_metrics()["reflect_unsafe_hints"] == 1
    hints, _, r = reflect(monkeypatch, [answer((1, "close </tool_call> early")), answer((1, "broken \ud83d"))],
                          max_selected_turns=1)
    assert hints == {} and r.call_metrics()["reflect_unsafe_hints"] == 2


def test_leaks_logged_not_dropped(monkeypatch):
    hints, _, r = reflect(monkeypatch, [answer((1, "raise `SystemExit` in parse_row"))])
    assert hints == {1: "raise `SystemExit` in parse_row"}
    assert r.call_metrics()["reflect_leaky_hints"] == 1


def test_unseen_names():
    seen = "class DotEnv:\n    def get(self): ...\nparse_row"
    assert unseen_names("use `DotEnv.get()` in parse_row", seen) == []
    assert unseen_names("raise `SystemExit` from `load_config`", seen) == ["SystemExit", "load_config"]


@pytest.fixture
def plan(monkeypatch):
    from uni_agent.reflection import api, chatgpt_plan

    monkeypatch.setattr(chatgpt_plan, "access_token", lambda path: f"plan-token:{path}")
    monkeypatch.setattr(api, "PLAN_BUSY_WAIT_S", 0.0)
    monkeypatch.setattr(api, "_PLAN_REST", {})
    monkeypatch.setattr(api.random, "uniform", lambda a, b: 0.0)
    return {"plan_credentials": "/creds/chatgpt_plan.json", "sampling": {"service_tier": "flex"}}


def test_the_plan_serves_the_call_in_its_required_form(monkeypatch, plan):
    hints, seen, r = reflect(monkeypatch, [answer((1, "api"))], plan=[answer((1, "plan"))], **plan)
    assert hints == {1: "plan"} and seen == [] and len(plan_seen) == 1
    req = plan_seen[0]
    assert req["store"] is False and req["stream"] is True and req["input"][0]["role"] == "user"
    assert "max_output_tokens" not in req and "service_tier" not in req and req["reasoning"] == {"effort": "high"}
    assert req["instructions"] == "sys" and req["text"]["format"]["strict"]
    metrics = r.call_metrics()
    assert metrics["reflect_plan_calls"] == 1 and metrics["reflect_plan_fallbacks"] == 0
    assert metrics["reflect_input_tokens"] == 500


def test_a_spent_plan_rests_and_the_api_serves_on_flex(monkeypatch, plan):
    from uni_agent.reflection import api
    spent = RateLimitError("weekly cap", code="subscription_sharing_usage_limit_exceeded")
    hints, seen, r = reflect(monkeypatch, [answer((1, "api"))], plan=[spent], **plan)
    assert hints == {1: "api"} and len(plan_seen) == 1 and seen[0]["service_tier"] == "flex"
    assert r.call_metrics()["reflect_plan_fallbacks"] == 1 and "/creds/chatgpt_plan.json" in api._PLAN_REST
    # the next rollout's reflector skips the resting plan
    hints, seen, r = reflect(monkeypatch, [answer((2, "api again"))], plan=[answer((1, "plan"))], **plan)
    assert hints == {2: "api again"} and plan_seen == [] and r.call_metrics()["reflect_plan_fallbacks"] == 1


def test_a_rested_plan_is_tried_again(monkeypatch, plan):
    from uni_agent.reflection import api
    monkeypatch.setattr(api, "PLAN_REST_S", 0.0)
    spent = RateLimitError("weekly cap", code="subscription_sharing_usage_limit_exceeded")
    hints, _, _ = reflect(monkeypatch, [answer((1, "api"))], plan=[spent], **plan)
    assert hints == {1: "api"}
    hints, seen, _ = reflect(monkeypatch, [answer((1, "api"))], plan=[answer((1, "plan"))], **plan)
    assert hints == {1: "plan"} and seen == []


def test_plans_are_tried_in_order_before_the_api(monkeypatch, plan):
    spent = RateLimitError("5-hour cap", code="subscription_sharing_usage_limit_exceeded")
    plans = {**plan, "plan_credentials": ["/creds/main.json", "/creds/spare.json"]}
    hints, seen, r = reflect(monkeypatch, [answer((1, "api"))], plan=[spent, answer((1, "spare"))], **plans)
    assert hints == {1: "spare"} and seen == []
    assert plan_keys == ["plan-token:/creds/main.json", "plan-token:/creds/spare.json"]
    m = r.call_metrics()
    assert m["reflect_plan_calls"] == 1 and m["reflect_plan_backup_calls"] == 1 and m["reflect_plan_fallbacks"] == 0
    # the main plan rests: the next rollout goes straight to the spare, then to the API once that is spent too
    hints, seen, r = reflect(monkeypatch, [answer((1, "api"))], plan=[spent], **plans)
    assert hints == {1: "api"} and plan_keys == ["plan-token:/creds/spare.json"]
    assert r.call_metrics()["reflect_plan_fallbacks"] == 1


@pytest.mark.parametrize("failure", [("failed", "subscription_sharing_usage_limit_exceeded"),
                                     RateLimitError("", code="subscription_sharing_invalid_user")])
def test_a_refused_plan_rests_whether_the_request_or_the_stream_says_so(monkeypatch, plan, failure):
    from uni_agent.reflection import api
    hints, _, _ = reflect(monkeypatch, [answer((1, "api"))], plan=[failure], **plan)
    assert hints == {1: "api"} and "/creds/chatgpt_plan.json" in api._PLAN_REST


def test_a_busy_plan_is_retried_before_the_api_serves(monkeypatch, plan):
    from uni_agent.reflection import api
    busy = APIStatusError("busy", 503, "subscription_sharing_usage_unavailable")
    hints, seen, r = reflect(monkeypatch, [answer((1, "api"))], plan=[busy, answer((1, "plan"))], **plan)
    assert hints == {1: "plan"} and len(plan_seen) == 2 and seen == [] and not api._PLAN_REST
    hints, seen, r = reflect(monkeypatch, [answer((1, "api"))], plan=[busy], **plan)
    assert hints == {1: "api"} and len(plan_seen) == 3 and not api._PLAN_REST


def test_a_refused_answer_schema_is_dropped_once(monkeypatch, plan):
    unsupported = BadRequestError("no schema", code="subscription_sharing_unsupported_capability", param="text.format")
    hints, seen, r = reflect(monkeypatch, [answer((1, "api"))], plan=[unsupported, answer((1, "plan"))], **plan)
    assert hints == {1: "plan"} and "text" in plan_seen[0] and "text" not in plan_seen[1] and seen == []


def test_a_context_error_from_the_plan_moves_down_the_ladder(monkeypatch, plan):
    too_long = BadRequestError("maximum context length exceeded")
    hints, seen, r = reflect(monkeypatch, [answer((1, "api"))], plan=[too_long, answer((1, "plan"))], **plan)
    assert hints == {1: "plan"} and len(plan_seen) == 2 and seen == []
    assert r.call_metrics()["reflect_plan_fallbacks"] == 0


def test_a_refused_refresh_rests_the_plan(monkeypatch, plan):
    from uni_agent.reflection import api, chatgpt_plan

    def refused(path):
        raise chatgpt_plan.PlanUnavailable("token endpoint refused (400): invalid_grant")

    monkeypatch.setattr(chatgpt_plan, "access_token", refused)
    hints, seen, r = reflect(monkeypatch, [answer((1, "api"))], plan=[answer((1, "plan"))], **plan)
    assert hints == {1: "api"} and plan_seen == [] and "/creds/chatgpt_plan.json" in api._PLAN_REST


def test_plan_calls_need_the_responses_api():
    with pytest.raises(ValueError, match="needs no base_url"):
        ApiReflector.Config(name="openai", model="m", base_url="https://x/v1", plan_credentials="/c.json",
                            calls=[{"id": "x", "system": "s", "user": "{turns}", "parse": "hints"}])


PLACE = {"id": "place", "system": "sys", "parse": "placement",
         "user": "{task}|{turns}|{facts}|{agent_patch}|{gold}|{feedback}"}
WRITE = {"id": "write", "system": "sys", "parse": "note",
         "user": "{turn}|{turns_before}|{mistake}|{type}|{type_rule}|{seen_before}|{task}"}


def placement(turn, kind="incomplete fix"):
    return json.dumps({"mistake": "the fix covers one branch", "type": kind, "statements": [] if turn is None else [turn],
                       "turn": turn, "quote": "q", "seen_before": "turn 0 showed both branches", "later_turn": None,
                       "later_quote": "", "later_evidence": "", "no_note_reason": "hidden" if turn is None else ""})


def note(text="The other branch of `parse_row` may also drop rows.", check="Run it on an empty row."):
    return json.dumps({"note": text, "check": check})


def test_two_calls_place_then_write_from_the_turns_before(monkeypatch):
    hints, seen, _ = reflect(monkeypatch, [placement(1), note()], calls=[PLACE, WRITE])
    assert hints == {1: "The other branch of `parse_row` may also drop rows."} and len(seen) == 2
    assert seen[0]["text"]["format"]["name"] == "hint_placement" and seen[1]["text"]["format"]["name"] == "hint_note"
    writer = seen[1]["input"]
    assert writer.startswith("1|### Turn 0\n") and "### Turn 1" not in writer and "### Turn 2" not in writer
    assert "the fix covers one branch|incomplete fix|the other place or case, and what it does now.|turn 0 showed" in writer
    assert "a different way forward" not in writer and "- test runs:" in seen[0]["input"]


def test_check_version_is_the_hint(monkeypatch):
    hints, _, _ = reflect(monkeypatch, [placement(2, "weak check"), note()], calls=[PLACE, {**WRITE, "parse": "check"}])
    assert hints == {2: "Run it on an empty row."}


def test_no_turn_means_no_note_and_no_writer_call(monkeypatch):
    hints, seen, _ = reflect(monkeypatch, [placement(None), note()], calls=[PLACE, WRITE])
    assert hints == {} and len(seen) == 1


def test_a_turn_outside_the_attempt_is_drawn_again(monkeypatch):
    hints, seen, _ = reflect(monkeypatch, [placement(9), placement(1), note()], calls=[PLACE, WRITE])
    assert hints == {1: "The other branch of `parse_row` may also drop rows."} and len(seen) == 3


def test_two_call_configs_are_checked():
    import pytest

    base = {"name": "openai", "model": "m"}
    with pytest.raises(ValueError, match="a placement, then a note or a check"):
        ApiReflector.Config(**base, calls=[WRITE, PLACE])
    with pytest.raises(ValueError, match="cannot be given"):
        ApiReflector.Config(**base, calls=[PLACE, {**WRITE, "user": "{turns}"}])
    with pytest.raises(ValueError, match="one call per trace"):
        ApiReflector.Config(**base, calls=[PLACE, WRITE, WRITE])


def test_attempt_facts():
    from uni_agent.reflection.facts import attempt_facts

    turns = [{"step": 1, "tools": [{"name": "str_replace_editor", "action": "str_replace_editor view --path /testbed/a.py",
                                    "observation": ""}]},
             {"step": 2, "tools": [{"name": "str_replace_editor", "observation": "",
                                    "action": "str_replace_editor str_replace --path /testbed/a.py --old_str x --new_str y"}]},
             {"step": 3, "tools": [{"name": "execute_bash", "action": "python -m pytest tests", "observation": ""}]},
             {"step": 4, "tools": [{"name": "submit", "action": "submit", "observation": ""}]}]
    facts = attempt_facts(turns, "diff --git a/a.py b/a.py\n", "diff --git a/a.py b/a.py\n")
    assert facts.splitlines() == ["- source edits (files in the final patch): a.py: turns [2]", "- test runs: turns [3]",
                                  "- script runs: turns none", "- first turn that names each gold file: a.py: turn 1",
                                  "- submitted at turn 4"]
