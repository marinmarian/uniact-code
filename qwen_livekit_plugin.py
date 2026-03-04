"""
Custom LiveKit LLM plugin wrapping the fine-tuned Qwen 2.5 VL model.

Implements ``livekit.agents.llm.LLM`` so Qwen can serve as the conversational
LLM in the unified agent.  GPU access is serialised with MotionEngine via a
shared ``asyncio.Lock``.

Tool calling is implemented via prompt engineering: tool schemas are injected
into the system prompt, and the model's output is parsed for JSON tool-call
patterns.
"""

import asyncio
import json
import re
import uuid
from typing import Any

import torch

from livekit.agents import llm
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

from infer_robot import MOTION_TOKEN_CONFIG


# ---------------------------------------------------------------------------
# Tool-call prompt injection
# ---------------------------------------------------------------------------

_TOOL_PREAMBLE = """\
You have access to the following tools. To call a tool, output EXACTLY one \
JSON object on its own line in this format (no markdown, no extra text around it):
{"tool_call": {"name": "<tool_name>", "arguments": {<args>}}}

Available tools:
"""


def _build_tool_schema_text(tools: list[llm.Tool]) -> str:
    """Render tool definitions as plain-text schema for the system prompt."""
    if not tools:
        return ""
    parts = [_TOOL_PREAMBLE]
    for tool in tools:
        if not hasattr(tool, "name"):
            continue
        entry = {"name": tool.name, "description": tool.description or ""}
        if hasattr(tool, "parameters") and tool.parameters:
            entry["parameters"] = tool.parameters
        parts.append(json.dumps(entry, indent=2))
    parts.append(
        "\nIf you do NOT need to call a tool, just respond normally with text."
    )
    return "\n".join(parts)


# Regex to find tool_call JSON in model output
_TOOL_CALL_RE = re.compile(
    r'\{\s*"tool_call"\s*:\s*\{.*?\}\s*\}', re.DOTALL
)


def _try_parse_tool_call(text: str) -> dict | None:
    """Try to extract a tool_call JSON from accumulated text."""
    m = _TOOL_CALL_RE.search(text)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
        tc = parsed.get("tool_call")
        if tc and "name" in tc:
            return tc
    except json.JSONDecodeError:
        pass
    return None


# ---------------------------------------------------------------------------
# QwenLLMStream
# ---------------------------------------------------------------------------

class QwenLLMStream(llm.LLMStream):
    """Streaming wrapper that runs Qwen generation in an executor."""

    def __init__(
        self,
        llm_instance: "QwenLLM",
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        conn_options: APIConnectOptions,
    ):
        super().__init__(
            llm_instance,
            chat_ctx=chat_ctx,
            tools=tools,
            conn_options=conn_options,
        )
        self._qwen_llm = llm_instance

    async def _run(self) -> None:
        request_id = str(uuid.uuid4())
        loop = asyncio.get_event_loop()

        # Build Qwen chat-template input from ChatContext
        input_text = self._build_input_text()

        # Tokenize
        model = self._qwen_llm._model
        tokenizer = self._qwen_llm._tokenizer

        async with self._qwen_llm._gpu_lock:
            input_ids, past_kv, prompt_length = await loop.run_in_executor(
                None, self._encode_input, input_text, model, tokenizer
            )

        # Autoregressive generation
        accumulated = ""
        tool_call_emitted = False
        max_tokens = 512  # chat responses are short
        prompt_tokens = prompt_length
        completion_tokens = 0

        for _step in range(max_tokens):
            async with self._qwen_llm._gpu_lock:
                next_token_id, past_kv = await loop.run_in_executor(
                    None, self._generate_one, input_ids, past_kv, model,
                    prompt_length + _step,
                )

            token_id = next_token_id.item()

            # Stop on EOS or motion tokens (model shouldn't generate these in chat)
            if token_id in (
                tokenizer.eos_token_id,
                151645,  # <|im_end|>
                151644,  # <|im_start|>
            ) or token_id >= MOTION_TOKEN_CONFIG["code_base_id"]:
                break

            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            accumulated += token_text
            completion_tokens += 1

            # Check for tool call in accumulated text
            tc = _try_parse_tool_call(accumulated)
            if tc and not tool_call_emitted:
                tool_call_emitted = True
                call_id = f"call_{uuid.uuid4().hex[:12]}"
                self._event_ch.send_nowait(
                    llm.ChatChunk(
                        id=request_id,
                        delta=llm.ChoiceDelta(
                            role="assistant",
                            tool_calls=[
                                llm.FunctionToolCall(
                                    name=tc["name"],
                                    arguments=json.dumps(tc.get("arguments", {})),
                                    call_id=call_id,
                                )
                            ],
                        ),
                    )
                )
                break

            # Emit text delta (skip if we're still accumulating a potential tool call)
            if not accumulated.lstrip().startswith('{"tool_call"'):
                self._event_ch.send_nowait(
                    llm.ChatChunk(
                        id=request_id,
                        delta=llm.ChoiceDelta(role="assistant", content=token_text),
                    )
                )

            # Prepare for next step
            input_ids = next_token_id

        # If we accumulated text that looked like it might be a tool call but
        # wasn't, flush it as regular text
        if accumulated.lstrip().startswith('{"tool_call"') and not tool_call_emitted:
            self._event_ch.send_nowait(
                llm.ChatChunk(
                    id=request_id,
                    delta=llm.ChoiceDelta(role="assistant", content=accumulated),
                )
            )

        # Usage
        self._event_ch.send_nowait(
            llm.ChatChunk(
                id=request_id,
                usage=llm.CompletionUsage(
                    completion_tokens=completion_tokens,
                    prompt_tokens=prompt_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_input_text(self) -> str:
        """Convert ChatContext to Qwen chat template string."""
        im_start = "<|im_start|>"
        im_end = "<|im_end|>"
        parts = []

        # Inject tool definitions into system message
        tool_text = _build_tool_schema_text(self._tools) if self._tools else ""

        for msg in self._chat_ctx.messages():
            role = msg.role
            content = msg.text_content or ""
            if role == "system" and tool_text:
                content = content + "\n\n" + tool_text
                tool_text = ""  # only inject once
            parts.append(f"{im_start}{role}\n{content}{im_end}")

        # If tool text wasn't injected (no system message), prepend it
        if tool_text:
            parts.insert(0, f"{im_start}system\n{tool_text}{im_end}")

        parts.append(f"{im_start}assistant\n")
        return "\n".join(parts)

    @staticmethod
    def _encode_input(input_text: str, model, tokenizer):
        """Tokenize input and run first forward pass.

        Returns (last_token_ids, past_kv, prompt_length).
        Qwen 2.5 VL requires explicit 3D position_ids (shape [3, B, seq])
        to avoid in-place expand() aliasing errors in its RoPE code.
        """
        device = model.device
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        seq_len = input_ids.shape[1]

        # Qwen 2.5 VL uses 3D RoPE — provide explicit position_ids
        # For pure text (no vision), all 3 dims use the same sequential positions
        pos = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
        position_ids = pos.unsqueeze(0).expand(3, -1, -1).clone()  # [3, 1, seq_len]

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=True,
                past_key_values=None,
                cache_position=torch.arange(seq_len, device=device),
            )
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
        return next_token, outputs.past_key_values, seq_len

    @staticmethod
    def _generate_one(input_ids, past_kv, model, step):
        """Single autoregressive step for chat generation."""
        device = model.device
        cache_len = past_kv[0][0].shape[2] if past_kv else 0
        cache_position = torch.tensor([cache_len], device=device)

        # Explicit 3D position_ids for Qwen 2.5 VL
        position_ids = torch.tensor([[[step]]], device=device).expand(3, 1, 1).clone()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=True,
                past_key_values=past_kv,
                cache_position=cache_position,
            )
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
        return next_token, outputs.past_key_values


# ---------------------------------------------------------------------------
# QwenLLM
# ---------------------------------------------------------------------------

class QwenLLM(llm.LLM):
    """LiveKit LLM plugin that uses a shared Qwen 2.5 VL model.

    Parameters
    ----------
    model : torch model
        The loaded Qwen model (shared with MotionEngine).
    tokenizer : transformers tokenizer
        The Qwen tokenizer.
    gpu_lock : asyncio.Lock
        Shared lock to serialise GPU access with motion generation.
    """

    def __init__(self, model, tokenizer, gpu_lock: asyncio.Lock):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._gpu_lock = gpu_lock

    @property
    def model(self) -> str:
        return "qwen2.5-vl-finetuned"

    @property
    def provider(self) -> str:
        return "qwen"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> QwenLLMStream:
        return QwenLLMStream(
            self,
            chat_ctx=chat_ctx,
            tools=tools or [],
            conn_options=conn_options,
        )
