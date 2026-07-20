import time
from typing import Literal

import orjson
from pydantic import BaseModel, Field

from uni_agent.async_logging import get_logger
from uni_agent.skills.manager import SkillsManager
from uni_agent.tracing import (
    register_langfuse_op,
    rollout_trace_event,
    rollout_trace_op,
    rollout_trace_set_attr,
)
from uni_agent.utils import auto_await, simple_timer

from .condenser import AbstractCondenser, CondensationFailed
from .env import ActionIncorrectSyntaxError, ActionTimeoutError, AgentEnv, TerminalNotAliveError
from .model import AgentChatModel, MaxTokenExceededError
from .tool_parser import FunctionCallFormatError
from .tool_schemas import OpenAIFunctionToolCall
from .tools_manager import ToolsManager

ToolStatus = Literal["ok", "timeout", "syntax_error", "skipped"]

# Heuristic budget for reactive condensation: how many characters to free when the
# context overflows, escalated per retry (~chars_per_token x tokens_over + margin).
CONDENSE_CHARS_PER_TOKEN = 4
CONDENSE_MARGIN_TOKENS = 1024
CONDENSE_MIN_CHARS = 2000


class ToolResult(BaseModel):
    """Per-tool-call result inside a single step. ``observation`` is the
    string sent back to the model as the ``role="tool"`` content (error
    text included).
    """

    tool_call_id: str
    name: str
    action: str = ""
    observation: str = ""
    status: ToolStatus
    execution_time: float | None = None


class StepOutput(BaseModel):
    step_idx: int

    response: str = ""
    thought: str = ""
    tool_results: list[ToolResult] = Field(default_factory=list)
    done: bool = False
    exit_reason: str = ""


def fast_deepcopy(obj):
    return orjson.loads(orjson.dumps(obj))


def _should_break(name: str) -> bool:
    try:
        from verl.utils.debug_breakpoints import should_break
    except ImportError:
        return False
    return should_break(name)


def _step_span_update(result):
    """Compact span output for a step: outcome only (messages/tools are their own observations)."""
    if not hasattr(result, "model_dump"):
        return {"output": result}
    dumped = result.model_dump()
    tool_results = dumped.get("tool_results") or []
    output = {
        "exit_reason": dumped.get("exit_reason"),
        "done": dumped.get("done"),
        "n_tools": len(tool_results),
        "tools": [t.get("name") for t in tool_results],
        "response_chars": len(dumped.get("response") or ""),
    }
    return {"output": output}


def _tool_span_update(result):
    """Span rendering for one executed tool call (errors surface as ERROR level)."""
    update = {
        "input": result.action,
        "output": result.observation,
        "metadata": {"status": result.status, "execution_time": result.execution_time},
    }
    if result.status != "ok":
        update["level"] = "ERROR"
        update["status_message"] = str(result.status)
    return update


register_langfuse_op("AgentInteraction.step", as_type="chain", name="step", update_fn=_step_span_update)
register_langfuse_op(
    "AgentInteraction._execute_tool_call",
    as_type="tool",
    name_fn=lambda inputs: f"tool:{inputs['tool_call'].function.name}",
    update_fn=_tool_span_update,
)


class AgentInteraction:
    def __init__(
        self,
        run_id: str,
        env: AgentEnv,
        model: AgentChatModel,
        tools_manager: ToolsManager,
        messages: list[dict[str, str]],
        action_timeout: int = 60,
        timeout_budget: int = 3,
        max_turns: int = 50,
        skills_manager: SkillsManager | None = None,
        chat_mode: bool = False,
        stuck_threshold: int = 0,
        condenser: AbstractCondenser | None = None,
        condense_max_retries: int = 5,
        condense_chars_per_token: int = CONDENSE_CHARS_PER_TOKEN,
        condense_margin_tokens: int = CONDENSE_MARGIN_TOKENS,
        condense_min_chars: int = CONDENSE_MIN_CHARS,
    ):
        """:param chat_mode: how to treat an assistant message with no
        tool calls. ``False`` (default, training / code-eval) raises
        ``format_error`` so the loop continues. ``True`` (long-running
        chat) marks the step ``turn_done`` so the caller can wait for
        the next user message.
        :param stuck_threshold: abort the rollout once this many consecutive
        assistant responses are identical (a stuck loop). ``0`` disables the
        check. The aborted trajectory exits with ``exit_reason="stuck"`` and is
        scored normally (masked only if ``mask_abnormal_exit_traj`` is set).
        :param condenser: optional context condenser. When set, a context overflow
        is handled reactively: the current token buffer is materialized as a
        trajectory *segment*, the message history is condensed, the buffer is
        re-seated from it, and generation retries. ``None`` keeps the legacy
        behavior (overflow ends the rollout with ``token_limit``).
        :param condense_max_retries: max condense+retry attempts per overflow.
        """
        self.env = env
        self.model = model
        self.tools_manager = tools_manager
        self.skills_manager = skills_manager
        self.messages = messages
        self.action_timeout = action_timeout
        self.timeout_budget = timeout_budget
        self.max_turns = max_turns
        self.chat_mode = chat_mode
        self.stuck_threshold = stuck_threshold
        self.condenser = condenser
        self.condense_max_retries = condense_max_retries
        self.condense_chars_per_token = condense_chars_per_token
        self.condense_margin_tokens = condense_margin_tokens
        self.condense_min_chars = condense_min_chars
        self.logger = get_logger("interaction", run_id)

    def inject_skills_manifest(self) -> None:
        """Append the skills manifest to the first system message.

        The manifest lists each discovered skill (name + description +
        path to its SKILL.md) so the model knows what is available and
        how to load it on demand. Skill *bodies* are not in the prompt --
        they live as real files on disk (read lazily, progressive
        disclosure).

        Call this exactly once, after ``AgentEnv.install_skills`` has
        populated ``runtime_paths``. The method is **not** idempotent --
        calling it twice will append the manifest twice. The single
        in-tree caller (``UniAgentLoop.run``) already enforces this.
        """
        if self.skills_manager is None:
            return
        manifest = self.skills_manager.build_manifest()
        if not manifest:
            return

        block = "\n\n" + manifest
        for msg in self.messages:
            if msg.get("role") == "system":
                content = msg.get("content") or ""
                msg["content"] = content + block
                return
        self.messages.insert(0, {"role": "system", "content": manifest})

    @rollout_trace_op
    async def step(self, step_idx: int):
        """Run one model-call + tool-execution cycle.

        Outcome is reported at two levels:

        * **Tool**: per-call :class:`ToolResult` on ``step_output.tool_results``
          with ``status`` in ``{ok, timeout, syntax_error, skipped}``.
        * **Step**: ``step_output.exit_reason`` + ``done``:

          - terminal (``done=True``): ``finished``, ``turn_done``,
            ``token_limit``, ``terminal_dead``, ``timeout_budget_exhausted``.
          - non-terminal (``done=False``): ``completed``,
            ``completed_with_tool_errors``, ``format_error``.
          - set by :meth:`run`: ``max_step_limit``, ``stuck``, ``unknown_error``.

        ``turn_done`` is gated on ``self.chat_mode`` (see ``__init__``).
        """
        # step index start from 1
        step_output = StepOutput(step_idx=step_idx)
        self.logger.info(f"{'=' * 25} STEP {step_idx} {'=' * 25}")

        # step 1: prepare template
        self.logger.info(f"🤖 MODEL INPUT\n{self.messages[-1]['content']}")

        # step 2: generate response and update rollout cache. On context overflow,
        # either condense + re-seat + retry (if a condenser is configured) or end.
        model_output = tool_calls = rollout_cache = generation_info = None
        for attempt in range(self.condense_max_retries + 1):
            try:
                model_output, tool_calls, rollout_cache, generation_info = await self.model.query(
                    messages=self.messages,
                    rollout_cache=self.rollout_cache,
                )
                break
            except MaxTokenExceededError as e:
                _msg = (
                    f"[step{step_idx}] MaxTokenExceededError: "
                    f"response_mask_len_before={len(self.rollout_cache.get('response_mask', []))} "
                    f"prompt_ids_len={len(self.rollout_cache.get('prompt_ids', []))} "
                    f"detail: {str(e)}"
                )
                if self.condenser is None or attempt == self.condense_max_retries:
                    self.logger.error("{}", _msg)
                    step_output.exit_reason = "token_limit"
                    step_output.done = True
                    return step_output
                self.logger.warning("{}", _msg)
                try:
                    await self._condense_and_reseat(attempt)
                    self.logger.info(f"Condensed context (attempt {attempt + 1}); retrying generation.")
                except CondensationFailed as ce:
                    self.logger.error(f"Condensation failed: {ce}")
                    step_output.exit_reason = "token_limit"
                    step_output.done = True
                    return step_output

        step_output.response = model_output
        # turn table: this step's response span within the active segment buffer
        span_end = len(rollout_cache["response_mask"])
        rollout_cache.setdefault("turn_spans", []).append(
            [step_idx, span_end - generation_info["completion_tokens"], span_end]
        )
        self.logger.info(
            f"Prompt Tokens: {generation_info['prompt_tokens']}, "
            f"Completion Tokens: {generation_info['completion_tokens']}"
        )
        self.logger.debug(f"Model Output:\n{model_output}")

        # step 3: parse model response to actions
        self.rollout_cache = rollout_cache

        # Persist the assistant message in api-shape (with tool_calls)
        # so replay preserves the assistant<->tool linkage.
        assistant_msg: dict[str, object] = {"role": "assistant", "content": model_output}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        self.messages.append(assistant_msg)

        try:
            if tool_calls:
                content, tool_calls = await self.tools_manager.parse_structured_action(
                    content=model_output,
                    tool_calls_data=tool_calls,
                )
            else:
                content, tool_calls = await self.tools_manager.parse_action(model_output=model_output)
            if not tool_calls and not self.chat_mode:
                raise FunctionCallFormatError("No function call found in the response.")
        except FunctionCallFormatError as e:
            if tool_calls:
                error_msgs: list[dict[str, object]] = [
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": tc["function"]["name"],
                        "content": str(e),
                    }
                    for tc in tool_calls
                ]
            else:
                error_msgs = [{"role": "tool", "content": str(e)}]
            self.messages.extend(error_msgs)
            self.rollout_cache = await self.model.append_messages_to_rollout_cache(error_msgs, self.rollout_cache)
            step_output.exit_reason = "format_error"
            model_output_preview = "\n".join(model_output.splitlines()[:20])
            _msg = (
                f"Fail to parse thought and action from model output.\n"
                f"Error Message: {str(e)}\n"
                f"Model Output (first 20 lines): {model_output_preview}"
            )
            self.logger.error("{}", _msg)
            return step_output

        step_output.thought = content
        self.logger.info(f"💭 THOUGHT:\n{content}")

        # step 4: chat_mode-only end-of-turn (single-shot already raised above).
        if not tool_calls:
            step_output.done = True
            step_output.exit_reason = "turn_done"
            self.logger.info(f"💬 TURN DONE (no tool call): {model_output}")
            return step_output

        # step 5: run each tool call sequentially in the shared bash session
        tool_results: list[ToolResult] = []
        tool_messages: list[dict[str, object]] = []
        saw_finish = False
        terminal_dead = False

        with simple_timer("tool_calls", self.rollout_cache["metrics"]):
            for idx, tool_call in enumerate(tool_calls):
                tool_call: OpenAIFunctionToolCall  # type: ignore[no-redef]
                result = await self._execute_tool_call(tool_call)
                if result.status == "ok" and result.name in ("finish", "submit"):
                    saw_finish = True
                elif result.status == "skipped":
                    terminal_dead = True

                tool_results.append(result)
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "name": result.name,
                        "content": result.observation,
                    }
                )

                # On hard failure (dead session / budget out), synthesize
                # `skipped` results for remaining tool calls to keep the
                # assistant<->tool N:N invariant, then break.
                budget_exhausted = self.timeout_budget < 0
                if terminal_dead or budget_exhausted:
                    skipped_reason = (
                        "Skipped: the bash session died mid-step; no further tool calls ran."
                        if terminal_dead
                        else "Skipped: timeout budget exhausted mid-step; no further tool calls ran."
                    )
                    for remaining in tool_calls[idx + 1 :]:
                        tool_results.append(
                            ToolResult(
                                tool_call_id=remaining.id,
                                name=remaining.function.name,
                                action="",
                                observation=skipped_reason,
                                status="skipped",
                                execution_time=None,
                            )
                        )
                        tool_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": remaining.id,
                                "name": remaining.function.name,
                                "content": skipped_reason,
                            }
                        )
                    break

        # step 6: commit collected tool messages
        self.messages.extend(tool_messages)
        self.rollout_cache = await self.model.append_messages_to_rollout_cache(tool_messages, self.rollout_cache)
        step_output.tool_results = tool_results

        # step 7: step-level exit_reason (precedence: terminal_dead >
        # timeout_budget_exhausted > finished > completed_with_tool_errors > completed)
        if terminal_dead:
            step_output.done = True
            step_output.exit_reason = "terminal_dead"
            return step_output
        if self.timeout_budget < 0:
            step_output.done = True
            step_output.exit_reason = "timeout_budget_exhausted"
            self.logger.info("Exit step: timeout budget exhausted.")
            return step_output
        if saw_finish:
            step_output.done = True
            step_output.exit_reason = "finished"
            return step_output
        if any(tr.status in ("timeout", "syntax_error") for tr in tool_results):
            step_output.done = False
            step_output.exit_reason = "completed_with_tool_errors"
            return step_output
        step_output.done = False
        step_output.exit_reason = "completed"
        return step_output

    def check_stuck(self) -> bool:
        """Return True when the last ``stuck_threshold`` assistant responses are
        all identical (a degenerate repeat loop). Disabled when threshold <= 0.

        Scans the trajectory tail and short-circuits: stops at the first response
        that differs (not stuck) or once ``stuck_threshold`` identical ones are
        seen (stuck), so it never materializes the full history.
        """
        if not self.stuck_threshold or self.stuck_threshold <= 0:
            return False
        last = None
        count = 0
        for step in reversed(self.trajectory):
            response = step.response
            if not response:
                continue
            if last is None:
                last = response
            elif response != last:
                return False
            count += 1
            if count >= self.stuck_threshold:
                return True
        return False

    def _materialize_segment(self) -> None:
        """Freeze the current token buffer as a finished trajectory segment, paired
        with the message context that formed its prompt (used downstream for
        per-segment teacher reconstruction)."""
        self.segments.append(
            {"rollout_cache": self.rollout_cache, "prompt_messages": self.segment_start_messages}
        )

    def _condense_budget(self, attempt: int) -> int:
        over_tokens = max(0, len(self.rollout_cache.get("prompt_ids", [])) - self.model.max_model_len)
        tokens_to_free = (over_tokens + self.condense_margin_tokens) * (attempt + 1)
        return max(self.condense_min_chars, tokens_to_free * self.condense_chars_per_token)

    @rollout_trace_op
    async def _execute_tool_call(self, tool_call: OpenAIFunctionToolCall) -> ToolResult:
        """Run one tool call in the env; errors become the observation (status marks the kind)."""
        action = self.tools_manager.get_tool_action(tool_call)
        self.logger.info(f"🎬 ACTION ({tool_call.function.name}):\n{action.command}")
        action_timeout = action.timeout or self.action_timeout

        tool_t0 = time.perf_counter()
        status: ToolStatus
        try:
            if _should_break("tool"):
                breakpoint()
            if action.is_input:
                observation = await self.env.send_input(action.command, action_timeout=action_timeout)
            else:
                observation = await self.env.run_action(action.command, action_timeout=action_timeout)
            status = "ok"
        except ActionTimeoutError as e:
            observation = str(e)
            status = "timeout"
            self.timeout_budget -= 1
            self.logger.error(f"{observation} (timeout_budget left: {self.timeout_budget})")
        except ActionIncorrectSyntaxError as e:
            observation = str(e)
            status = "syntax_error"
            self.logger.error(observation)
        except TerminalNotAliveError as e:
            observation = str(e)
            status = "skipped"
            self.logger.error(observation)
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.function.name,
            action=action.command,
            observation=observation,
            status=status,
            execution_time=time.perf_counter() - tool_t0,
        )

    async def _condense_and_reseat(self, attempt: int) -> None:
        """Materialize the overflowing buffer as a segment (when it produced tokens),
        condense the history, and re-seat a fresh buffer from the condensed messages so
        the next generation continues into a new segment."""
        if _should_break("condense"):
            breakpoint()
        if len(self.rollout_cache.get("response_mask", [])) > 0:
            self._materialize_segment()
        budget = self._condense_budget(attempt)
        n_messages_before = len(self.messages)
        n_tokens_before = len(self.rollout_cache.get("prompt_ids", []))
        self.messages = self.condenser.condense(
            self.messages, budget, arg_masker=self.tools_manager.mask_tool_args
        )
        self.rollout_cache = await self.model.prepare_rollout_cache(self.messages)
        self.segment_start_messages = list(self.messages)
        # mark the condensation boundary; later op spans carry the new segment index
        seg_idx = len(self.segments)
        rollout_trace_set_attr("segment_index", seg_idx)
        n_after = len(self.rollout_cache.get("prompt_ids", []))
        rollout_trace_event(
            "condensation",
            metadata={"segment_index": seg_idx, "attempt": attempt, "budget": budget},
            input=f"context overflow on attempt {attempt} at {n_messages_before} messages / "
            f"{n_tokens_before} prompt tokens; freeing ~{budget} chars",
            output=f"re-seated to {len(self.messages)} messages / {n_after} prompt tokens (segment {seg_idx})",
        )

    @auto_await
    async def run(self):
        self.trajectory: list[StepOutput] = []

        self.logger.info("Inital Prompt:")
        for message in self.messages:
            self.logger.info(f"{message['role'].upper()} PROMPT:\n{message['content']}")

        rollout_cache = await self.model.prepare_rollout_cache(self.messages)
        self.rollout_cache: dict[str, str] = rollout_cache
        # Trajectory segments: a new one starts after each condensation. With no
        # condensation this stays a single segment == the whole rollout.
        self.segments: list[dict] = []
        self.segment_start_messages: list[dict] = list(self.messages)

        done = False
        step_idx = 0
        execution_time = time.perf_counter()
        while not done:
            # we start from 1
            step_idx += 1
            try:
                step_output = await self.step(step_idx=step_idx)
                self.trajectory.append(step_output)
                done = step_output.done
                if done:
                    break
                if self.check_stuck():
                    self.logger.error(f"Exit due to stuck loop: {self.stuck_threshold} identical responses")
                    rollout_trace_event(
                        "stuck_abort",
                        metadata={"threshold": self.stuck_threshold, "step_idx": step_idx},
                        input=f"{self.stuck_threshold} consecutive identical responses",
                        output=(step_output.response or "")[:500],
                    )
                    step_output = StepOutput(step_idx=step_idx, exit_reason="stuck")
                    self.trajectory.append(step_output)
                    break
                if step_idx >= self.max_turns:
                    self.logger.error(f"Exit due to max step limit: {self.max_turns}")
                    step_output = StepOutput(step_idx=step_idx, exit_reason="max_step_limit")
                    self.trajectory.append(step_output)
                    break
            except Exception as e:
                # this should not happen, if it happens, we should fix the code
                _msg = (
                    f"[step{step_idx}] unknown_error: {type(e).__name__}: {e} "
                    f"response_mask_len_before={len(self.rollout_cache.get('response_mask', []))} "
                    f"prompt_ids_len={len(self.rollout_cache.get('prompt_ids', []))}"
                )
                self.logger.opt(exception=True).critical("{}", _msg)
                step_output = StepOutput(step_idx=step_idx, exit_reason="unknown_error")
                self.trajectory.append(step_output)
                break

        # Freeze the final buffer as the last segment, unless it was already materialized
        # (on CondensationFailed the buffer isn't re-seated, so it would be duplicated).
        if not self.segments or self.segments[-1]["rollout_cache"] is not self.rollout_cache:
            self._materialize_segment()

        execution_time = time.perf_counter() - execution_time
        result = {
            "trajectory": self.trajectory,
            "rollout_cache": self.rollout_cache,  # final buffer (kept for save/back-compat)
            "segments": self.segments,
            "execution_time": execution_time,
            "messages": self.messages,
        }
        return result
