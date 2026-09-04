import asyncio
import uuid
from functools import cached_property
from typing import Any

from uni_agent.tracing import rollout_trace_generation
from uni_agent.async_logging import get_logger
from uni_agent.utils import get_event_loop, simple_timer


class MaxTokenExceededError(Exception):
    pass


class AgentChatModel:
    client: Any
    """AsyncLLM server manager"""

    tokenizer: Any
    """Tokenizer for the model"""

    max_model_len: int
    """Max model context length"""

    sampling_params: dict[str, Any]
    """Sampling parameters for the model"""

    max_completion_tokens: int | None = None
    """Per-turn completion cap; None leaves the window as the only bound."""

    chat_template_kwargs: dict[str, Any] | None = None
    """Extra apply_chat_template kwargs (e.g. {"enable_thinking": False}); must match
    what the trainer uses to derive template fragments."""

    tools_schemas: list[dict] = None

    def __init__(self, **data):
        for key, value in data.items():
            setattr(self, key, value)
        self.loop = asyncio.get_running_loop()

    def set_tools_schemas(self, tools_schemas: list[dict]) -> None:
        self.tools_schemas = tools_schemas

    async def prepare_rollout_cache(
        self,
        messages: list[dict[str, str]],
        include_tools: bool = True,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """``chat_template_kwargs`` overrides the model's own for this call only: the
        reflector is a separate, untrained call and may need a different template mode
        (reasoning on, say) from the rollout it is diagnosing."""
        from verl.utils.tokenizer import normalize_token_ids

        tools = self.tools_schemas if include_tools else None
        template_kwargs = self.chat_template_kwargs if chat_template_kwargs is None else chat_template_kwargs
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                tools=tools,
                **(template_kwargs or {}),
            ),
        )
        prompt_ids = normalize_token_ids(prompt_ids)
        return {
            "request_id": str(uuid.uuid4()),
            "prompt_ids": prompt_ids,
            "response_mask": [],
            "response_logprobs": [],
            "routed_experts": None,
            "metrics": {},
            "extra_fields": {},
        }

    async def append_messages_to_rollout_cache(
        self,
        new_messages: list[dict[str, str]],
        rollout_cache: dict[str, Any] | None,
    ):
        """Append newly added user/tool messages to the rollout cache."""

        valid_roles = {"user", "tool"}
        invalid_roles = [message["role"] for message in new_messages if message["role"] not in valid_roles]
        assert not invalid_roles, f"New messages must be user or tool, but got invalid roles: {invalid_roles}"

        # encode tool response
        tool_response_ids = await self._get_new_message_ids(new_messages)

        # append tool response to prompt
        rollout_cache["prompt_ids"] += tool_response_ids
        rollout_cache["response_mask"] += [0] * len(tool_response_ids)
        if rollout_cache["response_logprobs"]:
            rollout_cache["response_logprobs"] += [0.0] * len(tool_response_ids)

        return rollout_cache

    async def query(
        self,
        messages: list[dict[str, str]],
        rollout_cache: dict[str, Any] | None,
        **kwargs,
    ) -> tuple[str, list[dict], dict[str, Any], dict[str, int]]:
        """Run one model call. Returns ``(text, tool_calls, rollout_cache,
        generation_info)``. ``tool_calls`` is always ``[]`` on the training
        path -- verl returns token ids, so callers must parse ``text``.
        """
        request_id = rollout_cache["request_id"]
        prompt_ids = rollout_cache["prompt_ids"]
        metrics = rollout_cache["metrics"]

        # `max_model_len` is the agent's own context budget, which is what the condenser reseats
        # against; a caller that is not building the agent's context (the reflector reads a whole
        # finished trajectory in one shot) may raise its own ceiling up to what the engine serves.
        limit = kwargs.get("max_model_len") or self.max_model_len
        if len(prompt_ids) >= limit:
            raise MaxTokenExceededError(
                f"prompt_ids length {len(rollout_cache['prompt_ids'])} exceeds max_model_len {limit}\n"
                f"Last tool response: {messages[-1]['content']}"
            )

        sampling_params = kwargs.get("sampling_params", self.sampling_params)
        # Cap one turn's completion. Uncapped, a runaway turn fills the whole remaining
        # window (measured: 0.15% of turns, ~14% of all generated tokens, ~4min of
        # single-stream decode each, and the overflow usually kills the segment).
        # min() with the window keeps a capped call from ever overflowing max_model_len.
        turn_limit = None
        if self.max_completion_tokens:
            room = max(1, limit - len(prompt_ids))
            # A caller that states its own budget keeps it: the reflector asks for far more
            # than one agent turn because it writes a staged analysis before its hints, and
            # this used to clamp it to max_completion_tokens regardless, so replies died at
            # exactly 4096 tokens with the object unwritten. The window clamp still applies.
            asked = sampling_params.get("max_tokens")
            ceiling = int(asked) if asked else int(self.max_completion_tokens)
            turn_limit = min(ceiling, room)
            sampling_params = {**sampling_params, "max_tokens": turn_limit}

        with simple_timer("generate_sequences", metrics):
            token_output = await self.client.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = token_output.num_preempted if token_output.num_preempted is not None else -1
        else:
            metrics["num_preempted"] += token_output.num_preempted if token_output.num_preempted is not None else 0
        if turn_limit and len(token_output.token_ids) > turn_limit:
            # Should be impossible (the server clamps to the requested max_tokens), yet
            # observed once, immediately after a condensation retry (run 2985518,
            # 5137 > 4096). Enforce the invariant here and log enough to find the path.
            get_logger("model").warning(
                f"generation returned {len(token_output.token_ids)} tokens despite max_tokens={turn_limit} "
                f"(prompt={len(prompt_ids)}, request_id={request_id}); truncating to the cap"
            )
            token_output.token_ids = token_output.token_ids[:turn_limit]
            if token_output.log_probs is not None:
                token_output.log_probs = token_output.log_probs[:turn_limit]
        generation_info = {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(token_output.token_ids),
            # the turn was cut mid-generation by the per-turn cap (not a natural stop);
            # the interaction layer tells the model, or it misreads the parse error
            "capped": bool(turn_limit and len(token_output.token_ids) >= turn_limit),
        }
        metrics["capped_turns"] = metrics.get("capped_turns", 0) + int(generation_info["capped"])
        response_ids = token_output.token_ids
        rollout_cache["prompt_ids"] += response_ids
        rollout_cache["response_mask"] += [1] * len(response_ids)
        if token_output.log_probs is not None:
            expected = len(rollout_cache["response_mask"]) - len(response_ids)
            if len(rollout_cache["response_logprobs"]) != expected:
                raise RuntimeError(
                    f"response_logprobs ({len(rollout_cache['response_logprobs'])}) out of step with "
                    f"response_mask ({expected}): an earlier turn returned no log-probs"
                )
            rollout_cache["response_logprobs"] += token_output.log_probs
        if token_output.routed_experts is not None:
            rollout_cache["routed_experts"] = token_output.routed_experts
        if not rollout_cache["extra_fields"]:
            rollout_cache["extra_fields"].update(token_output.extra_fields)
        else:
            max_global_steps = token_output.extra_fields.get("max_global_steps", None)
            if max_global_steps is not None:
                rollout_cache["extra_fields"]["max_global_steps"] = max_global_steps
        response_str = await self.loop.run_in_executor(None, lambda: self.tokenizer.decode(response_ids))
        # omit tool_calls: null breaks Langfuse chat rendering; calls are their own tool spans
        rollout_trace_generation(
            "model_call",
            model=getattr(self.tokenizer, "name_or_path", None),
            input=list(messages),
            output={"role": "assistant", "content": response_str},
            usage={
                "input": generation_info["prompt_tokens"],
                "output": generation_info["completion_tokens"],
            },
            num_preempted=token_output.num_preempted,
        )

        if len(rollout_cache["prompt_ids"]) >= limit:
            raise MaxTokenExceededError(
                f"prompt_ids length {len(rollout_cache['prompt_ids'])} exceeds max_model_len {limit}\n"
                f"Generated response:\n{response_str}"
            )

        return response_str, [], rollout_cache, generation_info

    async def _get_new_message_ids(self, new_messages: list[dict[str, Any]]) -> list[int]:
        from verl.utils.chat_template import apply_chat_template
        from verl.utils.tokenizer import normalize_token_ids

        tokenized_prompt = await self.loop.run_in_executor(
            None,
            lambda: apply_chat_template(
                self.tokenizer,
                new_messages,
                add_generation_prompt=True,
                tokenize=True,
                **(self.chat_template_kwargs or {}),
            ),
        )
        return self.message_boundary_tokens + normalize_token_ids(tokenized_prompt)

    @cached_property
    def message_boundary_tokens(self) -> list[int]:
        from verl.utils.chat_template import apply_chat_template
        from verl.utils.tokenizer import normalize_token_ids

        dummy_history = [
            {"role": "user", "content": "dummy user"},
            {"role": "assistant", "content": "dummy assistant"},
        ]
        dummy_next_message = {"role": "user", "content": "dummy user"}

        try:
            standalone_ids = normalize_token_ids(
                apply_chat_template(
                    self.tokenizer,
                    [dummy_next_message],
                    add_generation_prompt=True,
                    tokenize=True,
                    **(self.chat_template_kwargs or {}),
                )
            )
            with_boundary_ids = normalize_token_ids(
                apply_chat_template(
                    self.tokenizer,
                    dummy_history + [dummy_next_message],
                    add_generation_prompt=True,
                    tokenize=True,
                    **(self.chat_template_kwargs or {}),
                )
            )
        except Exception:
            return []

        if not standalone_ids or with_boundary_ids[-len(standalone_ids) :] != standalone_ids:
            return []

        text_before_message_ids = with_boundary_ids[: -len(standalone_ids)]
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            return []

        for i in range(len(text_before_message_ids) - 1, -1, -1):
            if text_before_message_ids[i] == eos_id:
                return text_before_message_ids[i + 1 :]

        return []


# this class is only used for Inference-Only Scenario
class OpenAICompatibleChatModel:
    base_url: str
    """OpenAI-compatible API base URL, for example http://127.0.0.1:8000/v1"""

    api_key: str
    """API key for the chat completion endpoint"""

    model_name: str
    """Model name sent to the OpenAI-compatible endpoint"""

    sampling_params: dict[str, Any]
    """Default sampling parameters passed to the endpoint"""

    timeout: int | float
    """HTTP timeout in seconds"""

    tools_schemas: list[dict] = None

    def __init__(self, **data):
        for key, value in data.items():
            setattr(self, key, value)
        if not hasattr(self, "sampling_params"):
            self.sampling_params = {}
        if not hasattr(self, "timeout"):
            self.timeout = 300
        self.base_url = self.base_url.rstrip("/")
        self.loop = get_event_loop()

        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def set_tools_schemas(self, tools_schemas: list[dict]) -> None:
        self.tools_schemas = tools_schemas

    async def prepare_rollout_cache(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Stateless: caller owns ``messages`` and re-passes them every
        :meth:`query`. Cache holds only metrics.
        """
        return {"metrics": {}}

    async def append_messages_to_rollout_cache(
        self,
        new_messages: list[dict[str, Any]],
        rollout_cache: dict[str, Any] | None,
    ):
        """No-op; kept so :class:`AgentInteraction` can dispatch uniformly
        across training and inference paths.
        """
        return rollout_cache

    def _normalize_messages_for_api(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Strip locally-added fields the OpenAI API doesn't accept.
        Tool messages missing ``tool_call_id`` (format-error fallbacks)
        pass through as-is.
        """
        normalized_messages = []
        for message in messages:
            normalized_message = {"role": message["role"]}
            if message.get("content") is not None:
                normalized_message["content"] = message["content"]
            if message["role"] == "assistant" and message.get("tool_calls"):
                normalized_message["tool_calls"] = message["tool_calls"]
            if message["role"] == "tool":
                tool_call_id = message.get("tool_call_id")
                if tool_call_id is not None:
                    normalized_message["tool_call_id"] = tool_call_id
                if message.get("name") is not None:
                    normalized_message["name"] = message["name"]
            normalized_messages.append(normalized_message)
        return normalized_messages

    # OpenAI ChatCompletion top-level sampling fields.
    _OPENAI_TOP_LEVEL_SAMPLING_FIELDS: frozenset[str] = frozenset(
        {
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "max_tokens",
            "max_completion_tokens",
            "stop",
            "n",
            "seed",
            "logprobs",
            "top_logprobs",
            "logit_bias",
            "user",
        }
    )

    async def query(
        self,
        messages: list[dict[str, str]],
        rollout_cache: dict[str, Any] | None,
        **kwargs,
    ) -> tuple[str, list[dict], dict[str, Any], dict[str, int]]:
        """Run one chat-completion call. Returns ``(text, tool_calls,
        rollout_cache, generation_info)``. ``tool_calls`` is the OpenAI
        ``{"id", "type", "function": {"name", "arguments"}}`` shape (one
        entry per parallel call; ``[]`` if the model returned plain text).
        """
        sampling_params = kwargs.get("sampling_params", self.sampling_params) or {}
        api_messages = self._normalize_messages_for_api(messages)

        top_level = {k: v for k, v in sampling_params.items() if k in self._OPENAI_TOP_LEVEL_SAMPLING_FIELDS}
        extra_body = {k: v for k, v in sampling_params.items() if k not in self._OPENAI_TOP_LEVEL_SAMPLING_FIELDS}

        with simple_timer("generate_sequences", rollout_cache["metrics"]):
            chat_completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=api_messages,
                tools=self.tools_schemas,
                extra_body=extra_body or None,
                **top_level,
            )

        response_message = chat_completion.choices[0].message
        response_content = response_message.content or ""
        response_tool_calls = list(response_message.tool_calls or [])

        serialized_tool_calls: list[dict] = [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in response_tool_calls
        ]

        usage = chat_completion.usage
        completion_tokens = usage.completion_tokens if usage is not None else max(len(response_content.split()), 1)
        prompt_tokens = usage.prompt_tokens if usage is not None else 0
        return (
            response_content,
            serialized_tool_calls,
            rollout_cache,
            {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )
