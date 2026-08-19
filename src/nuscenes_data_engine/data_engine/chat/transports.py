"""Chat transports: one OpenAI-compatible (Ollama / vLLM), one Claude API.

The agent speaks the OpenAI wire shape throughout — messages as
``{role, content, tool_calls?, tool_call_id?}`` dicts, tools as
``{type: "function", function: {...}}`` — and ``complete()`` returns the assistant
message dict. The Anthropic transport translates that shape to and from Claude
content blocks, so the provider is a pure config flip.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from typing import Any, Protocol

logger = logging.getLogger("nuscenes_data_engine")

MAX_COMPLETION_TOKENS = 4096


class ChatTransport(Protocol):
    """One model turn: messages + tools in, assistant message out."""

    @property
    def model(self) -> str: ...

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]: ...


class TransportError(RuntimeError):
    """The model server could not be reached or returned a hard error."""


def assemble_openai_stream(
    lines: Iterable[str], on_token: Callable[[str], None]
) -> dict[str, Any]:
    """Pure SSE assembler: OpenAI-compatible ``/chat/completions`` stream lines in,
    one assistant message out — shaped exactly like the non-stream ``complete()``
    message. No I/O; the transport supplies the decoded line iterator.

    Content deltas are appended and forwarded to ``on_token`` as they arrive.
    ``tool_calls`` fragments are merged by their ``index`` (id/name are set once,
    ``function.arguments`` is concatenated across chunks), since providers split a
    single tool call's JSON arguments across many SSE events.
    """
    content_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or not line.startswith("data:"):
            continue  # blank lines and SSE comments (e.g. ": keepalive") are framing
        # SSE allows multi-line `data:` fields; compact JSON from OpenAI-compatible
        # servers never spans lines — single-line assumed.
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        piece = delta.get("content")
        if piece:
            content_parts.append(piece)
            on_token(piece)
        for fragment in delta.get("tool_calls") or []:
            index = fragment.get("index", 0)
            if index not in tool_calls:
                tool_calls[index] = {
                    "id": fragment.get("id", ""),
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
                order.append(index)
            call = tool_calls[index]
            if fragment.get("id"):
                call["id"] = fragment["id"]
            fn = fragment.get("function") or {}
            if fn.get("name"):
                call["function"]["name"] = fn["name"]
            if fn.get("arguments"):
                call["function"]["arguments"] += fn["arguments"]
    return {
        "role": "assistant",
        "content": "".join(content_parts) or None,
        "tool_calls": [tool_calls[index] for index in order],
    }


def stream_with_fallback(
    transport: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    on_token: Callable[[str], None],
) -> dict[str, Any]:
    """Stream through ``transport.complete_stream`` when the transport offers one;
    otherwise fall back to ``complete`` and forward its whole answer as one token.

    ``complete_stream`` is deliberately NOT added to the ``ChatTransport`` Protocol
    (it stays a single-``complete`` surface) so existing test fakes that implement
    only ``complete`` keep working untouched — this checks for the optional method
    with ``getattr`` instead of widening the Protocol.
    """
    stream_fn = getattr(transport, "complete_stream", None)
    if stream_fn is not None:
        result: dict[str, Any] = stream_fn(messages, tools, on_token=on_token)
        return result
    reply: dict[str, Any] = transport.complete(messages, tools)
    content = reply.get("content") or ""
    if content:
        on_token(content)
    return reply


class OpenAICompatTransport:
    """Any OpenAI-compatible /chat/completions server (Ollama, vLLM, LM Studio)."""

    def __init__(self, base_url: str, model: str, timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    @property
    def model(self) -> str:
        return self._model

    def _payload(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, stream: bool = False
    ) -> dict[str, Any]:
        """The request body shared by ``complete`` and ``complete_stream`` — built in
        one place so the two paths cannot drift apart."""
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": MAX_COMPLETION_TOKENS,
        }
        if stream:
            payload["stream"] = True
        if tools:
            payload["tools"] = tools
        return payload

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        import httpx

        payload = self._payload(messages, tools)
        try:
            with httpx.Client(timeout=httpx.Timeout(self._timeout, connect=5.0)) as client:
                response = client.post(f"{self.base_url}/chat/completions", json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise TransportError(
                f"Chat model at {self.base_url} unavailable ({exc}). "
                "Is the local model server (e.g. `ollama serve`) running?"
            ) from exc
        message: dict[str, Any] = response.json()["choices"][0]["message"]
        return message

    def complete_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_token: Callable[[str], None],
    ) -> dict[str, Any]:
        import httpx

        payload = self._payload(messages, tools, stream=True)
        try:
            with (
                httpx.Client(timeout=httpx.Timeout(self._timeout, connect=5.0)) as client,
                client.stream("POST", f"{self.base_url}/chat/completions", json=payload) as response,
            ):
                response.raise_for_status()
                return assemble_openai_stream(response.iter_lines(), on_token)
        except httpx.HTTPError as exc:
            raise TransportError(
                f"Chat model at {self.base_url} unavailable ({exc}). "
                "Is the local model server (e.g. `ollama serve`) running?"
            ) from exc


class AnthropicTransport:
    """Claude API with tool use, translated to/from the OpenAI wire shape."""

    def __init__(self, api_key: str, model: str) -> None:
        import anthropic

        if not api_key:
            raise TransportError("ANTHROPIC_API_KEY is not configured.")
        # Any at the SDK boundary: requests are built as plain dicts (as autolabel does).
        self._client: Any = anthropic.Anthropic(api_key=api_key)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        import anthropic

        system, converted = convert_messages(messages)
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=MAX_COMPLETION_TOKENS,
                thinking={"type": "adaptive"},
                system=system,
                messages=converted,
                tools=[convert_tool(tool) for tool in tools],
            )
        except anthropic.AnthropicError as exc:
            raise TransportError(f"Claude API error: {exc}") from exc

        return self._convert_message(response)

    def complete_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_token: Callable[[str], None],
    ) -> dict[str, Any]:
        import anthropic

        system, converted = convert_messages(messages)
        try:
            with self._client.messages.stream(
                model=self._model,
                max_tokens=MAX_COMPLETION_TOKENS,
                thinking={"type": "adaptive"},
                system=system,
                messages=converted,
                tools=[convert_tool(tool) for tool in tools],
            ) as stream:
                for text in stream.text_stream:
                    on_token(text)
                final_message = stream.get_final_message()
        except anthropic.AnthropicError as exc:
            raise TransportError(f"Claude API error: {exc}") from exc

        return self._convert_message(final_message)

    def _convert_message(self, response: Any) -> dict[str, Any]:
        """Claude content blocks -> the OpenAI-shaped assistant message. Shared by
        ``complete`` and ``complete_stream`` so the two conversions cannot drift
        apart. Thinking blocks are excluded by the ``type == "text"`` filter below —
        only text and tool_use blocks ever reach the model-visible message."""
        texts = [block.text for block in response.content if block.type == "text"]
        tool_calls = [
            {
                "id": block.id,
                "type": "function",
                "function": {"name": block.name, "arguments": json.dumps(block.input)},
            }
            for block in response.content
            if block.type == "tool_use"
        ]
        message: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts) or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message


def make_transport(
    settings: Any,
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> ChatTransport:
    """Build the configured transport (CLI/API entry point); overrides win."""
    resolved = (provider or settings.chat_provider).lower()
    if resolved == "local":
        return OpenAICompatTransport(
            base_url or settings.chat_base_url, model or settings.chat_model
        )
    if resolved == "anthropic":
        return AnthropicTransport(
            settings.anthropic_api_key, model or settings.chat_anthropic_model
        )
    raise ValueError(f"Unknown chat provider {resolved!r} (expected 'local' or 'anthropic')")


def convert_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """OpenAI function-tool spec -> Anthropic tool spec."""
    fn = tool["function"]
    return {
        "name": fn["name"],
        "description": fn.get("description", ""),
        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
    }


def convert_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """OpenAI-shaped messages -> (system string, Anthropic messages).

    Assistant ``tool_calls`` become ``tool_use`` blocks; consecutive ``tool``-role
    results merge into one user message of ``tool_result`` blocks (Claude requires
    results directly after the tool_use turn).
    """
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        if role == "system":
            system_parts.append(str(message.get("content") or ""))
        elif role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id", ""),
                "content": str(message.get("content") or ""),
            }
            if converted and converted[-1]["role"] == "user" and _is_tool_results(converted[-1]):
                converted[-1]["content"].append(block)
            else:
                converted.append({"role": "user", "content": [block]})
        elif role == "assistant" and message.get("tool_calls"):
            blocks: list[dict[str, Any]] = []
            if message.get("content"):
                blocks.append({"type": "text", "text": str(message["content"])})
            for call in message["tool_calls"]:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["function"]["name"],
                        "input": json.loads(call["function"]["arguments"] or "{}"),
                    }
                )
            converted.append({"role": "assistant", "content": blocks})
        else:
            converted.append({"role": role, "content": str(message.get("content") or "")})
    return "\n\n".join(part for part in system_parts if part), converted


def _is_tool_results(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return isinstance(content, list) and all(
        isinstance(block, dict) and block.get("type") == "tool_result" for block in content
    )
