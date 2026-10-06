"""A reflector served by a hosted model instead of the policy: the OpenAI Responses API, or any
OpenAI-compatible Chat Completions endpoint (``base_url``).

One call per failed trace. The prompt is a labeling brief answered as a JSON object under a fixed
schema: the v4 brief grades every turn and names the best ones, which become the hints in the
model's order (``parse: hints``); the protocol brief names one turn and its hint (``parse:
turn_hint``). The trajectory is rendered in the tagged form those briefs were measured on: each
turn shows the tool output the agent saw right before writing it, then what the agent wrote and
the tool it called.
"""

import asyncio
import os
import random
import re
import time
from typing import ClassVar

from pydantic import model_validator

from uni_agent.reflection import chatgpt_plan
from uni_agent.reflection.base import OVER_BUDGET, AbstractReflector, BaseReflectionConfig, ReflectionFailed
from uni_agent.reflection.facts import patch_view
from uni_agent.reflection.pipeline import CallSpec, PipelineReflectionConfig, _fields
from uni_agent.reflection.registry import register_reflector
from uni_agent.tracing import register_langfuse_op, rollout_trace_op, rollout_trace_span

#: the fields a brief may name; {first} and {last} are the attempt's turn range
FIELDS = frozenset({"task", "outcome", "gold", "agent_patch", "feedback", "turns", "first", "last"})
#: a high-effort read of a 150-turn trace takes minutes
TIMEOUT_S = 1200.0
#: the client's own retries, with backoff, on transient server errors
MAX_RETRIES = 8
#: how long one reflection keeps waiting out the account's tokens-per-minute limit: the failed
#: rollouts of a step all ask at once, and the client's own backoff gives up within seconds
RATE_LIMIT_WAIT_S = 1800.0
#: one whole reflection, retries and rate-limit waits included: the step waits for every rollout,
#: and a stalled request is resent MAX_RETRIES times at TIMEOUT_S each
WALL_S = 1800.0
#: "try again in 1.2s", "20ms", "6m0s", "7h12m0s": a per-day limit asks for hours
RETRY_IN = re.compile(r"try again in ((?:[\d.]+(?:ms|h|m|s))+)")
RETRY_UNIT = re.compile(r"([\d.]+)(ms|h|m|s)")
SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
#: chat-template control tokens and lone surrogates: the teacher encodes a hint with special tokens parsed
UNSAFE = re.compile(r"<\|[^<>|\n]*\|>|</?(?:tool_call|tool_response|think)>|[\ud800-\udfff]")
#: token use per call, summed over the trajectory's calls
USAGE = ("reflect_input_tokens", "reflect_cached_tokens", "reflect_output_tokens", "reflect_reasoning_tokens")
#: ChatGPT plan errors: a spent allowance or a refused user ends plan calls for the run; a busy service is retried
PLAN_OFF = frozenset({"subscription_sharing_usage_limit_exceeded", "subscription_sharing_user_not_eligible",
                      "subscription_sharing_invalid_user", "subscription_sharing_route_not_supported"})
PLAN_BUSY = frozenset({"subscription_sharing_usage_unavailable", "subscription_sharing_user_unavailable"})
PLAN_TRIES = 3
PLAN_BUSY_WAIT_S = 20.0

CALL = re.compile(r"<function=(\w+)>(.*?)</function>", re.S)
PARAM = re.compile(r"<parameter=(\w+)>\n?(.*?)\n?</parameter>", re.S)
ISSUE = re.compile(r"<issue_description>(.*?)</issue_description>", re.S)
#: what a hint may cite by name: every identifier in backticks, and bare dotted paths, snake_case, CamelCase
QUOTED = re.compile(r"`([^`\n]+)`")
IDENT = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\b")
BARE = re.compile(r"\b(?:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+|[a-z]\w*_\w+|[A-Z][a-z0-9]+[A-Z]\w*)\b")

_PROBLEMS = {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["id", "where", "why", "timeline"], "properties": {
        "id": {"type": "string"}, "where": {"type": "string"}, "why": {"type": "string"},
        "timeline": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["turn", "what"],
            "properties": {"turn": {"type": "integer"}, "what": {"type": "string"}}}}}}}
_TURNS = {"type": "array", "items": {
    "type": "object", "additionalProperties": False,
    "required": ["turn", "seen", "doing", "grade", "problem", "reason", "hint"], "properties": {
        "turn": {"type": "integer"}, "seen": {"type": "string"}, "doing": {"type": "string"},
        "grade": {"type": "integer", "enum": [-1, 0, 1, 2, 3]}, "problem": {"type": ["string", "null"]},
        "reason": {"type": "string"}, "hint": {"type": ["string", "null"]}}}}
_TOP = {"type": "array", "maxItems": 3, "items": {
    "type": "object", "additionalProperties": False, "required": ["turn", "problem", "hint", "why"], "properties": {
        "turn": {"type": "integer"}, "problem": {"type": "string"}, "hint": {"type": "string"}, "why": {"type": "string"}}}}
ANSWER_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["problems", "turns", "top"],
                 "properties": {"problems": _PROBLEMS, "turns": _TURNS, "top": _TOP}}
#: written in this order, so the turn is chosen after naming the problem and the hint after its evidence
TURN_HINT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["problem", "turn", "evidence", "hint"],
                    "properties": {"problem": {"type": "string"}, "turn": {"type": "integer"},
                                   "evidence": {"type": "string"}, "hint": {"type": "string"}}}
SCHEMAS = {"hints": ("hint_positions", ANSWER_SCHEMA), "turn_hint": ("hint_turn", TURN_HINT_SCHEMA)}


def fence(text: str, lang: str = "") -> str:
    ticks = "`" * max(3, max((len(m) for m in re.findall(r"`+", text)), default=0) + 1)
    return f"{ticks}{lang}\n{text.rstrip()}\n{ticks}"


def own_part(response: str) -> str:
    """What the agent wrote and the tool it called, in a neutral form (raw call markup invites the reader to continue it)."""
    match = CALL.search(response)
    fn, p = (match.group(1), dict(PARAM.findall(match.group(2)))) if match else (None, {})
    if fn == "str_replace_editor":
        head = f"str_replace_editor {p.get('command', '')} {p.get('path', '')}" + (
            f" {p['view_range']}" if p.get("view_range") else "")
        call = "\n".join([head] + [f"<{k}>\n{p[k]}\n</{k}>"
                                   for k in ("old_str", "new_str", "file_text", "insert_line") if p.get(k)])
    elif fn == "execute_bash":
        call = f"bash\n$ {p.get('command', '')}"
    elif fn:
        call = fn + "".join(f" {k}={v}" for k, v in p.items())
    else:
        call = "(no tool call)"
    text = response.split("</think>")[-1].split("<tool_call>")[0].strip()
    return (f"<agent_wrote>\n{text}\n</agent_wrote>\n" if text else "") + f"<agent_called>\n{call}\n</agent_called>"


def unseen_names(hint: str, seen: str) -> list[str]:
    """Names the hint cites that appear nowhere in what the agent had seen before its turn."""
    names = {name for quoted in QUOTED.findall(hint) for name in IDENT.findall(quoted) if len(name) >= 3}
    names |= set(BARE.findall(QUOTED.sub(" ", hint)))
    return sorted(name for name in names if name not in seen and not all(part in seen for part in name.split(".")))


def retry_after(message: str) -> float | None:
    """The wait a 429 asks for, in seconds; ``None`` when it names none."""
    match = RETRY_IN.search(message)
    if match is None:
        return None
    return sum(float(n) * SECONDS[unit] for n, unit in RETRY_UNIT.findall(match.group(1)))


class ApiReflectionConfig(BaseReflectionConfig):
    """One call to a hosted model; the brief's prompts come in ``calls`` as for the pipeline."""

    calls: list[CallSpec]
    #: the hosted model's id, and its reasoning effort
    model: str
    reasoning_effort: str = "high"
    #: None is the OpenAI Responses API; a URL is an OpenAI-compatible Chat Completions endpoint there
    base_url: str | None = None
    #: the environment variable holding the endpoint's key
    api_key_env: str = "OPENAI_API_KEY"
    #: extra request parameters: a Chat Completions endpoint's recommended sampling, or the Responses API's
    #: service_tier (flex bills half and may queue; its capacity 429s are waited out like rate limits)
    sampling: dict = {}
    #: a ChatGPT plan credentials file (chatgpt_plan login): calls go to the user's plan, and to the API key
    #: with ``sampling`` when the plan cannot serve them
    plan_credentials: str | None = None

    @model_validator(mode="after")
    def _check_calls(self):
        if len(self.calls) != 1:
            raise ValueError("the api reflector makes exactly one call per trace")
        if self.plan_credentials and self.base_url is not None:
            raise ValueError("ChatGPT plan calls go to the OpenAI Responses API: plan_credentials needs no base_url")
        if self.calls[0].parse not in SCHEMAS:
            raise ValueError(f"the api reflector's call parses its answer as one of {sorted(SCHEMAS)}")
        unknown = _fields(self.calls[0].user) - FIELDS
        if unknown:
            raise ValueError(f"call {self.calls[0].id!r} references fields it cannot be given: {sorted(unknown)}")
        if self.enabled:
            PipelineReflectionConfig._check_inputs(self)
        return self


@register_reflector("openai")
class ApiReflector(AbstractReflector):
    """The render is measured with the policy's tokenizer, which over the hosted calls measured
    counted 1-16% more tokens than the endpoint billed, so the shared ladder cuts on the safe side;
    a context error from the endpoint moves on to the next step all the same."""

    Config: ClassVar[type[BaseReflectionConfig]] = ApiReflectionConfig
    #: set once the plan's allowance is spent or its sign-in refused; the API key serves the rest of the run
    _plan_off: bool = False
    #: dropped if the plan refuses the strict answer schema; the parser reads the JSON from the text alone
    _plan_schema: bool = True

    @rollout_trace_op
    async def reflect_trajectory(
        self, task: str, turns: list[dict], gold: str, feedback: str, outcome: str = "", agent_patch: str = ""
    ) -> dict[int, str]:
        from openai import AsyncOpenAI, BadRequestError, RateLimitError

        cfg = self.config
        call = cfg.calls[0]
        max_tokens = call.max_output_tokens or cfg.max_output_tokens
        issue = ISSUE.search(task or "")
        attempt = (fence(self._clip(patch_view(agent_patch, gold), cfg.max_patch_chars), "diff") if agent_patch
                   else "(empty: no change was extracted from the attempt)")
        values = {
            "task": issue.group(1).strip() if issue else task,
            "outcome": outcome or "(not available)",
            "agent_patch": attempt if cfg.include_agent_patch else "(not available)",
            "gold": fence(self._clip(gold, cfg.max_patch_chars), "diff") if cfg.include_gold and gold else "(not available)",
            "feedback": fence(feedback) if cfg.include_exec_feedback and feedback else "(not available)",
            "first": str(turns[0]["step"]),
            "last": str(turns[-1]["step"]),
        }

        def render_user(obs_cap, resp_cap):
            return call.user.format(**values, turns=self._render_attempt(turns, obs_cap, resp_cap))

        deadline = time.monotonic() + WALL_S
        async with AsyncOpenAI(api_key=os.environ.get(cfg.api_key_env), base_url=cfg.base_url,
                               timeout=TIMEOUT_S, max_retries=MAX_RETRIES) as client:
            async for rung, obs_cap, resp_cap, messages, _, _ in self._renders(
                call.system, render_user, max_tokens, call.id, None
            ):
                for draw in range(1 + cfg.redraws_per_rung):
                    start = time.monotonic()
                    try:
                        with rollout_trace_span(f"reflect:{call.id}", metadata={
                                "model": cfg.model, "obs_cap": obs_cap, "resp_cap": resp_cap}):
                            text, usage, status = await asyncio.wait_for(
                                self._request(client, RateLimitError, messages, max_tokens, call.parse),
                                timeout=deadline - start)
                    except BadRequestError as exc:
                        if "context" not in str(exc).lower():
                            self.logger.warning(f"Reflection call failed; no hints for this rollout: {exc}")
                            await self._record(call.id, None, messages, None, None, obs_cap, resp_cap, error=repr(exc))
                            raise ReflectionFailed(f"reflection call failed: {exc!r}") from exc
                        self.logger.info(f"Reflection render over budget (obs_cap={obs_cap}, resp_cap={resp_cap}): {exc}")
                        await self._record(call.id, None, messages, None, None, obs_cap, resp_cap, error=OVER_BUDGET)
                        break
                    except asyncio.TimeoutError as exc:
                        await self._record(call.id, None, messages, None, None, obs_cap, resp_cap,
                                           error=f"over the {WALL_S:.0f} s wall budget")
                        raise ReflectionFailed(f"reflection over its {WALL_S:.0f} s wall budget") from exc
                    except Exception as exc:
                        self.logger.warning(f"Reflection call failed; no hints for this rollout: {exc}")
                        await self._record(call.id, None, messages, None, None, obs_cap, resp_cap, error=repr(exc))
                        raise ReflectionFailed(f"reflection call failed: {exc!r}") from exc
                    hints, leaks = self._hints(text, turns, task, call.parse)
                    await self._record(call.id, None, messages, text, usage.get("input_tokens"), obs_cap, resp_cap,
                                       draw=draw, extra={"model": cfg.model, "status": status,
                                                         "seconds": round(time.monotonic() - start, 1),
                                                         "usage": usage, "hints": hints, "unseen_names": leaks})
                    if hints:
                        self._counts["reflect_rung"] = max(self._counts["reflect_rung"], rung)
                        self._counts["reflect_leaky_hints"] += sum(1 for step in hints if leaks.get(step))
                        self.logger.info(f"Reflection ok: {usage} hints at {sorted(hints)}")
                        return hints
                    self.logger.info(f"Reflection reply unusable (draw {draw + 1}, status {status})")
                else:
                    # a reply the parser could not use is no reason to ask again from a smaller view
                    return {}
        raise ReflectionFailed("render over budget at every shrink level")

    async def _request(self, client, rate_limit_error, messages, max_tokens, parse) -> tuple[str, dict, str]:
        """One call on either API: (reply text, token use, completion status)."""
        cfg = self.config
        if cfg.base_url is None:
            name, schema = SCHEMAS[parse]
            if cfg.plan_credentials and not self._plan_off:
                served = await self._plan_request(messages, name, schema)
                if served is not None:
                    return served
            response = await self._create(
                client.responses.create, rate_limit_error,
                model=cfg.model, instructions=messages[0]["content"], input=messages[1]["content"],
                reasoning={"effort": cfg.reasoning_effort}, max_output_tokens=max_tokens,
                text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
                **cfg.sampling,
            )
            u = response.usage
            usage = {} if u is None else {
                "input_tokens": u.input_tokens or 0,
                "cached_tokens": getattr(u.input_tokens_details, "cached_tokens", 0) or 0,
                "output_tokens": u.output_tokens or 0,
                "reasoning_tokens": getattr(u.output_tokens_details, "reasoning_tokens", 0) or 0,
            }
            return response.output_text or "", self._tally(usage), response.status
        response = await self._create(
            client.chat.completions.create, rate_limit_error,
            model=cfg.model, messages=messages, max_tokens=max_tokens, response_format={"type": "json_object"},
            extra_body={"reasoning_effort": cfg.reasoning_effort}, **cfg.sampling,
        )
        u = response.usage
        usage = {} if u is None else {
            "input_tokens": u.prompt_tokens or 0,
            "cached_tokens": getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0,
            "output_tokens": u.completion_tokens or 0,
            "reasoning_tokens": getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0,
        }
        choice = response.choices[0]
        return choice.message.content or "", self._tally(usage), choice.finish_reason

    async def _plan_request(self, messages, name, schema) -> tuple[str, dict, str] | None:
        """One call on the user's ChatGPT plan, in the form that path requires (streamed, not stored, no output
        cap); None when the plan cannot serve it, so the API key does. A context error goes back to the ladder."""
        from openai import APIStatusError, AsyncOpenAI, BadRequestError

        cfg = self.config
        for attempt in range(PLAN_TRIES):
            code = None
            try:
                token = await asyncio.to_thread(chatgpt_plan.access_token, cfg.plan_credentials)
                body = {"model": cfg.model, "instructions": messages[0]["content"],
                        "input": [{"role": "user", "content": messages[1]["content"]}],
                        "reasoning": {"effort": cfg.reasoning_effort}, "store": False, "stream": True}
                if self._plan_schema:
                    body["text"] = {"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}}
                final = None
                async with AsyncOpenAI(api_key=token, timeout=TIMEOUT_S, max_retries=0) as plan:
                    async for event in await plan.responses.create(**body):
                        if event.type == "response.completed":
                            final = event.response
                        elif event.type in ("response.failed", "error"):
                            failed = getattr(getattr(event, "response", None), "error", None) or event
                            code = getattr(failed, "code", None)
                            raise RuntimeError(f"plan call failed: {code}: {getattr(failed, 'message', '')}")
                if final is None:
                    raise RuntimeError("plan stream ended without response.completed")
                u = final.usage
                usage = {} if u is None else {
                    "input_tokens": u.input_tokens or 0,
                    "cached_tokens": getattr(u.input_tokens_details, "cached_tokens", 0) or 0,
                    "output_tokens": u.output_tokens or 0,
                    "reasoning_tokens": getattr(u.output_tokens_details, "reasoning_tokens", 0) or 0,
                }
                self._counts["reflect_plan_calls"] += 1
                return final.output_text or "", self._tally(usage), final.status
            except chatgpt_plan.PlanUnavailable as exc:
                self._plan_off = True
                self.logger.warning(f"ChatGPT plan unavailable for the rest of the run, using the API key: {exc}")
                break
            except BadRequestError as exc:
                if "context" in str(exc).lower():
                    raise
                if exc.code == "subscription_sharing_unsupported_capability" and self._plan_schema:
                    self._plan_schema = False
                    self.logger.warning(f"ChatGPT plan refused the answer schema ({exc.param}); asking without it")
                    continue
                self.logger.warning(f"ChatGPT plan call refused, the API key serves this one: {exc}")
                code = exc.code
                break
            except APIStatusError as exc:
                code = exc.code
                self.logger.warning(f"ChatGPT plan call failed ({exc.status_code} {code}): {exc}")
            except Exception as exc:
                self.logger.warning(f"ChatGPT plan call failed: {exc!r}")
            if code in PLAN_OFF:
                self._plan_off = True
                self.logger.warning(f"ChatGPT plan off for the rest of the run ({code}); the API key serves it")
                break
            if code not in PLAN_BUSY or attempt + 1 == PLAN_TRIES:
                break
            await asyncio.sleep(PLAN_BUSY_WAIT_S + random.uniform(0.0, 10.0))
        self._counts["reflect_plan_fallbacks"] += 1
        return None

    async def _create(self, create, rate_limit_error, **request):
        """One request, waiting out 429s for as long as the API asks, up to RATE_LIMIT_WAIT_S in all;
        a wait that would pass it (a per-day limit) fails at once."""
        deadline = time.monotonic() + RATE_LIMIT_WAIT_S
        while True:
            try:
                return await create(**request)
            except rate_limit_error as exc:
                # a spent quota, or one request above the per-minute limit, never clears by waiting
                if "insufficient_quota" in str(exc) or "Request too large" in str(exc):
                    raise
                wait = retry_after(str(exc))
                wait = (20.0 if wait is None else wait) + random.uniform(1.0, 10.0)
                if time.monotonic() + wait > deadline:
                    raise
                self._counts["reflect_rate_limited"] += 1
                await asyncio.sleep(wait)

    def call_metrics(self) -> dict[str, float]:
        return {**super().call_metrics(), **{key: float(self._counts[key]) for key in (
            *USAGE, "reflect_leaky_hints", "reflect_rate_limited", "reflect_unsafe_hints", "reflect_plan_calls",
            "reflect_plan_fallbacks")}}

    def _tally(self, usage: dict[str, int]) -> dict[str, int]:
        for key, value in usage.items():
            self._counts[f"reflect_{key}"] += value
        return usage

    def _hints(self, text: str, turns: list[dict], task: str,
               parse: str = "hints") -> tuple[dict[int, str], dict[int, list[str]]]:
        """The best turns in the model's order, capped at the budget, with the names each hint
        cites that the agent had not seen before that turn (logged, never used to drop a hint)."""
        answer = self._extract_json_object(text, strict=False)
        if not isinstance(answer, dict):
            top = []
        elif parse == "turn_hint":
            top = [{"turn": answer.get("turn"), "hint": answer.get("hint")}]
        else:
            top = answer.get("top")
        valid = {turn["step"] for turn in turns}
        hints: dict[int, str] = {}
        for pick in top if isinstance(top, list) else []:
            if len(hints) >= self.config.max_selected_turns:
                break
            step, hint = (pick.get("turn"), pick.get("hint")) if isinstance(pick, dict) else (None, None)
            if step in valid and step not in hints and isinstance(hint, str) and hint.strip():
                if UNSAFE.search(hint):
                    self._counts["reflect_unsafe_hints"] += 1
                    self.logger.warning(f"Hint for turn {step} dropped, it carries chat control text: {hint!r}")
                    continue
                hints[step] = self._clip_diagnosis(hint.strip())
        leaks = {}
        for step in hints:
            seen = "\n".join([task] + [turn["response"] + "\n" + "\n".join(t["observation"] or "" for t in turn["tools"])
                                       for turn in turns if turn["step"] < step])
            names = unseen_names(hints[step], seen)
            if names:
                leaks[step] = names
        return hints, leaks

    def _render_attempt(self, turns: list[dict], obs_cap: int | None, resp_cap: int | None) -> str:
        parts, previous = [], None
        for turn in turns:
            shown = (f"<tool_output>\n{self._clip(previous, obs_cap)}\n</tool_output>\n" if previous is not None else "")
            # the cut counts what the reader is shown: the reply after its reasoning
            own = own_part(self._clip_response(turn["response"].split("</think>")[-1], resp_cap))
            parts.append(f'<turn n="{turn["step"]}">\n{shown}{own}\n</turn>')
            previous = "\n".join(t["observation"] or "" for t in turn["tools"])
        return "\n\n".join(parts)


register_langfuse_op("ApiReflector.reflect_trajectory", name="reflection", as_type="evaluator")
