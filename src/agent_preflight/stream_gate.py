from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Mapping


class StreamTruncationError(RuntimeError):
    """Raised fail-closed when a tool-call stream truncates, errors, or yields invalid JSON."""


@dataclass(frozen=True, slots=True)
class ToolCallDraft:
    """Verified, completely buffered tool call ready for execution."""

    call_id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str
    protocol: str


@dataclass(frozen=True, slots=True)
class StreamGateResult:
    """Non-throwing audit outcome of a buffered tool-call stream."""

    safe_to_execute: bool
    protocol: str
    status: str
    reason: str
    tool_calls: tuple[ToolCallDraft, ...] = ()


@dataclass
class _PendingToolCall:
    key: str
    call_id: str = ""
    name: str = ""
    chunks: list[str] = field(default_factory=list)
    explicit_input: dict[str, Any] | None = None
    has_deltas: bool = False
    completed: bool = False


_VALID_CHAT_FINISH_REASONS = frozenset({"tool_calls", "stop", "function_call"})
_VALID_ANTHROPIC_STOP_REASONS = frozenset({"tool_use", "end_turn", "stop_sequence"})


def _as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, Mapping):
        return {str(k): v for k, v in obj.items()}
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        dumped = obj.model_dump()
        if isinstance(dumped, Mapping):
            return {str(k): v for k, v in dumped.items()}
    if hasattr(obj, "__dict__"):
        return {str(k): v for k, v in vars(obj).items() if not str(k).startswith("_")}
    raise TypeError(f"Unsupported stream event type: {type(obj).__name__}")


def _parse_json_object(raw: str, context: str) -> dict[str, Any]:
    stripped = raw.strip()
    if not stripped:
        raise StreamTruncationError(f"Empty tool arguments payload in {context}.")
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise StreamTruncationError(f"Malformed or truncated JSON arguments in {context}: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise StreamTruncationError(f"Tool arguments in {context} must decode to a JSON object, got {type(parsed).__name__}.")
    return {str(k): v for k, v in parsed.items()}


class StreamToolCallGate:
    """Fail-closed buffer gate for OpenAI Chat, Responses API, Anthropic, and MCP tool-call streams."""

    def __init__(self, protocol: str = "auto") -> None:
        normalized = protocol.strip().lower()
        if normalized not in ("auto", "openai_chat", "openai_responses", "anthropic", "mcp"):
            raise ValueError(f"Unsupported stream protocol: {protocol!r}")
        self._requested_protocol = normalized
        self._detected_protocol: str | None = None if normalized == "auto" else normalized
        self._pending: dict[str, _PendingToolCall] = {}
        self._order: list[str] = []
        self._event_count: int = 0
        self._terminal_seen: bool = False
        self._terminal_reason: str = ""
        self._failure_reason: str | None = None
        self._sse_event_type: str | None = None
        self._sse_data_lines: list[str] = []
        self._sse_done_marker: bool = False

    @property
    def protocol(self) -> str:
        return self._detected_protocol or self._requested_protocol

    def _get_or_create(self, key: str) -> _PendingToolCall:
        if key not in self._pending:
            self._pending[key] = _PendingToolCall(key=key)
            self._order.append(key)
        return self._pending[key]

    def _fail(self, reason: str) -> None:
        if self._failure_reason is None:
            self._failure_reason = reason

    def _detect_protocol(self, event: dict[str, Any]) -> str:
        if self._detected_protocol is not None:
            return self._detected_protocol
        ev_type = str(event.get("type", ""))
        if ev_type.startswith("response.") or (ev_type == "error" and "response" in event):
            self._detected_protocol = "openai_responses"
        elif ev_type in ("message_start", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop", "ping"):
            self._detected_protocol = "anthropic"
        elif "choices" in event or event.get("object") == "chat.completion.chunk":
            self._detected_protocol = "openai_chat"
        elif "jsonrpc" in event or "method" in event or "result" in event or "toolCalls" in event or "isError" in event:
            self._detected_protocol = "mcp"
        elif ev_type == "error":
            self._detected_protocol = "openai_responses"
        else:
            raise StreamTruncationError(f"Unable to auto-detect stream protocol for event keys: {sorted(event.keys())}")
        return self._detected_protocol

    def feed_event(self, raw_event: Mapping[str, Any] | str | Any) -> None:
        """Ingest one parsed or serialized JSON event frame."""
        if isinstance(raw_event, str):
            stripped = raw_event.strip()
            if not stripped:
                return
            if stripped == "[DONE]":
                self._sse_done_marker = True
                return
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError as exc:
                self._fail(f"Truncated JSON frame in stream: {exc.msg}")
                return
            if not isinstance(decoded, dict):
                self._fail(f"Stream frame must be a JSON object, got {type(decoded).__name__}.")
                return
            event: dict[str, Any] = {str(k): v for k, v in decoded.items()}
        else:
            event = _as_dict(raw_event)

        self._event_count += 1
        proto = self._detect_protocol(event)
        if proto == "openai_chat":
            self._ingest_openai_chat(event)
        elif proto == "openai_responses":
            self._ingest_openai_responses(event)
        elif proto == "anthropic":
            self._ingest_anthropic(event)
        elif proto == "mcp":
            self._ingest_mcp(event)

    def feed_sse_line(self, line: str) -> None:
        """Ingest a single raw SSE (Server-Sent Events) or NDJSON line."""
        clean = line.rstrip("\r\n")
        if clean == "":
            self._flush_sse_buffer()
            return
        if clean.startswith(":"):
            return
        if clean.startswith("event:"):
            self._sse_event_type = clean[6:].strip()
            return
        if clean.startswith("data:"):
            self._sse_data_lines.append(clean[5:].lstrip())
            return
        if clean.lstrip().startswith("{"):
            self.feed_event(clean)
            return
        self._fail(f"Unexpected non-JSON SSE fragment: {clean[:80]!r}")

    def feed_sse_lines(self, lines: Iterable[str]) -> None:
        for line in lines:
            self.feed_sse_line(line)
        if self._sse_data_lines:
            self._flush_sse_buffer()

    def _flush_sse_buffer(self) -> None:
        if not self._sse_data_lines:
            self._sse_event_type = None
            return
        payload = "\n".join(self._sse_data_lines).strip()
        ev_type = self._sse_event_type
        self._sse_data_lines.clear()
        self._sse_event_type = None
        if payload == "[DONE]":
            self._sse_done_marker = True
            return
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            self._fail(f"Truncated SSE data payload: {exc.msg}")
            return
        if isinstance(decoded, dict):
            frame: dict[str, Any] = {str(k): v for k, v in decoded.items()}
            if ev_type and "type" not in frame:
                frame["type"] = ev_type
            self.feed_event(frame)
        else:
            self._fail(f"SSE payload must be JSON object, got {type(decoded).__name__}.")

    def _ingest_openai_chat(self, event: dict[str, Any]) -> None:
        if "error" in event and event["error"] is not None:
            self._fail(f"OpenAI Chat stream error: {event['error']}")
            return
        choices = event.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            delta = choice.get("delta")
            if isinstance(delta, Mapping):
                tool_calls = delta.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        if not isinstance(tc, Mapping):
                            continue
                        key = f"idx:{tc.get('index', 0)}"
                        pending = self._get_or_create(key)
                        if tc.get("id"):
                            pending.call_id = str(tc["id"])
                        fn = tc.get("function")
                        if isinstance(fn, Mapping):
                            if fn.get("name"):
                                pending.name += str(fn["name"])
                            if "arguments" in fn and fn["arguments"] is not None:
                                pending.chunks.append(str(fn["arguments"]))
                                pending.has_deltas = True
                legacy_fn = delta.get("function_call")
                if isinstance(legacy_fn, Mapping):
                    pending = self._get_or_create("legacy:0")
                    if legacy_fn.get("name"):
                        pending.name += str(legacy_fn["name"])
                    if "arguments" in legacy_fn and legacy_fn["arguments"] is not None:
                        pending.chunks.append(str(legacy_fn["arguments"]))
                        pending.has_deltas = True
            if (finish_reason := choice.get("finish_reason")) is not None:
                reason_str = str(finish_reason)
                if reason_str in _VALID_CHAT_FINISH_REASONS:
                    self._terminal_seen = True
                    self._terminal_reason = reason_str
                    for p in self._pending.values():
                        p.completed = True
                else:
                    self._fail(f"OpenAI Chat stream terminated abnormally with finish_reason={reason_str!r}.")

    def _ingest_openai_responses(self, event: dict[str, Any]) -> None:
        ev_type = str(event.get("type", ""))
        if ev_type in ("response.incomplete", "response.failed", "error"):
            resp_obj = event.get("response")
            incomplete = resp_obj.get("incomplete_details") if isinstance(resp_obj, Mapping) else None
            self._fail(f"OpenAI Responses stream aborted ({ev_type}): {event.get('error') or incomplete or ev_type}")
            return
        if ev_type == "response.output_item.added":
            item = event.get("item")
            if isinstance(item, Mapping) and str(item.get("type", "")) in ("function_call", "mcp_call", "custom_tool_call"):
                item_id = str(item.get("id") or f"out:{event.get('output_index', 0)}")
                pending = self._get_or_create(item_id)
                pending.call_id = str(item.get("call_id") or item.get("id") or item_id)
                if item.get("name"):
                    pending.name = str(item["name"])
                if item.get("arguments"):
                    pending.chunks = [str(item["arguments"])]
        elif ev_type in ("response.function_call_arguments.delta", "response.mcp_call_arguments.delta"):
            item_id = str(event.get("item_id") or f"out:{event.get('output_index', 0)}")
            pending = self._get_or_create(item_id)
            if (delta := event.get("delta")) is not None:
                pending.chunks.append(str(delta))
                pending.has_deltas = True
        elif ev_type in ("response.function_call_arguments.done", "response.mcp_call_arguments.done"):
            item_id = str(event.get("item_id") or f"out:{event.get('output_index', 0)}")
            pending = self._get_or_create(item_id)
            if event.get("name"):
                pending.name = str(event["name"])
            if "arguments" in event and event["arguments"] is not None:
                pending.chunks = [str(event["arguments"])]
            pending.completed = True
        elif ev_type == "response.output_item.done":
            item = event.get("item")
            if isinstance(item, Mapping) and str(item.get("type", "")) in ("function_call", "mcp_call", "custom_tool_call"):
                item_id = str(item.get("id") or f"out:{event.get('output_index', 0)}")
                pending = self._get_or_create(item_id)
                pending.call_id = str(item.get("call_id") or item.get("id") or pending.call_id or item_id)
                if item.get("name"):
                    pending.name = str(item["name"])
                if "arguments" in item and item["arguments"] is not None:
                    pending.chunks = [str(item["arguments"])]
                if str(item.get("status", "completed")) in ("incomplete", "failed"):
                    self._fail(f"Tool item {pending.name or item_id!r} ended with status={item.get('status')!r}.")
                    return
                pending.completed = True
        elif ev_type == "response.completed":
            resp = event.get("response")
            if isinstance(resp, Mapping) and str(resp.get("status", "completed")) != "completed":
                self._fail(f"OpenAI Responses completion had non-completed status={resp.get('status')!r}.")
                return
            self._terminal_seen = True
            self._terminal_reason = "response.completed"

    def _ingest_anthropic(self, event: dict[str, Any]) -> None:
        ev_type = str(event.get("type", ""))
        if ev_type == "error":
            self._fail(f"Anthropic stream error: {event.get('error')}")
            return
        if ev_type == "content_block_start":
            idx = event.get("index", 0)
            block = event.get("content_block")
            if isinstance(block, Mapping) and str(block.get("type", "")) in ("tool_use", "server_tool_use", "mcp_tool_use"):
                key = f"block:{idx}"
                pending = self._get_or_create(key)
                pending.call_id = str(block.get("id") or key)
                pending.name = str(block.get("name") or "")
                if isinstance((inp := block.get("input")), Mapping):
                    pending.explicit_input = {str(k): v for k, v in inp.items()}
        elif ev_type == "content_block_delta":
            delta = event.get("delta")
            if isinstance(delta, Mapping) and delta.get("type") == "input_json_delta":
                pending = self._get_or_create(f"block:{event.get('index', 0)}")
                if (partial := delta.get("partial_json")) is not None:
                    pending.chunks.append(str(partial))
                    pending.has_deltas = True
        elif ev_type == "content_block_stop":
            key = f"block:{event.get('index', 0)}"
            if key in self._pending:
                self._pending[key].completed = True
        elif ev_type == "message_delta":
            delta = event.get("delta")
            if isinstance(delta, Mapping) and "stop_reason" in delta:
                sr = delta.get("stop_reason")
                if sr is None:
                    self._fail("Anthropic stream ended with null stop_reason.")
                elif str(sr) not in _VALID_ANTHROPIC_STOP_REASONS:
                    self._fail(f"Anthropic stream terminated abnormally with stop_reason={sr!r}.")
                else:
                    self._terminal_reason = str(sr)
        elif ev_type == "message_stop":
            if not self._terminal_reason:
                self._fail("Anthropic message_stop arrived without valid stop_reason in message_delta.")
                return
            self._terminal_seen = True

    def _ingest_mcp(self, event: dict[str, Any]) -> None:
        if event.get("error") is not None:
            self._fail(f"MCP JSON-RPC error: {event['error']}")
            return
        method = str(event.get("method", ""))
        if method == "notifications/cancelled":
            self._fail("MCP tool invocation was cancelled.")
            return
        params = event.get("params")
        if isinstance(params, Mapping) and (method.startswith("notifications/") or event.get("done") is False or params.get("partial") is True):
            pending = self._get_or_create(str(params.get("callId") or params.get("id") or "mcp:0"))
            if params.get("name"):
                pending.name = str(params["name"])
            if "delta" in params and params["delta"] is not None:
                pending.chunks.append(str(params["delta"]))
                pending.has_deltas = True
            elif "arguments" in params and isinstance(params["arguments"], str):
                pending.chunks.append(params["arguments"])
                pending.has_deltas = True
            return

        raw_res = event.get("result")
        res: dict[str, Any] = {str(k): v for k, v in raw_res.items()} if isinstance(raw_res, Mapping) else event
        if res.get("isError") is True:
            self._fail("MCP tool execution returned isError=True.")
            return
        if res.get("done") is False:
            pending = self._get_or_create(str(res.get("id") or "mcp:0"))
            if res.get("name"):
                pending.name = str(res["name"])
            if "arguments_delta" in res:
                pending.chunks.append(str(res["arguments_delta"]))
                pending.has_deltas = True
            return

        tool_entries: list[Mapping[str, Any]] = []
        for list_key in ("toolCalls", "tool_calls"):
            if isinstance((raw_list := res.get(list_key)), list):
                tool_entries.extend(item for item in raw_list if isinstance(item, Mapping))
        if not tool_entries and isinstance((raw_content := res.get("content")), list):
            tool_entries.extend(item for item in raw_content if isinstance(item, Mapping) and item.get("type") in ("tool_use", "tool_call"))
        if not tool_entries and "name" in res:
            tool_entries.append(res)

        for idx, entry in enumerate(tool_entries):
            call_key = str(entry.get("id") or entry.get("callId") or f"mcp:{idx}")
            pending = self._get_or_create(call_key)
            pending.call_id = str(entry.get("id") or entry.get("callId") or call_key)
            pending.name = str(entry.get("name") or pending.name)
            args_val = entry.get("arguments") if "arguments" in entry else entry.get("input")
            if isinstance(args_val, Mapping):
                pending.explicit_input = {str(k): v for k, v in args_val.items()}
            elif isinstance(args_val, str):
                pending.chunks = [args_val]
            pending.completed = True

        self._terminal_seen = True
        self._terminal_reason = "mcp.result"

    def finalize(self) -> list[ToolCallDraft]:
        """Verify stream completeness and return all tool calls, or raise StreamTruncationError."""
        if self._sse_data_lines:
            self._flush_sse_buffer()
        if self._failure_reason is not None:
            raise StreamTruncationError(self._failure_reason)
        if self._event_count == 0:
            raise StreamTruncationError("Stream ended with zero events received (fail-closed).")
        if not self._terminal_seen:
            raise StreamTruncationError(f"Stream truncated before completion event (protocol={self.protocol}, done_marker={self._sse_done_marker}).")

        verified: list[ToolCallDraft] = []
        for key in self._order:
            pending = self._pending[key]
            if not pending.completed:
                raise StreamTruncationError(f"Tool call {pending.name or key!r} did not receive a completion event.")
            if not pending.name.strip():
                raise StreamTruncationError(f"Tool call {key!r} is missing a tool name.")
            if pending.has_deltas or pending.chunks:
                raw_args = "".join(pending.chunks)
                parsed_args = _parse_json_object(raw_args, f"{self.protocol}:{pending.name}")
            elif pending.explicit_input is not None:
                parsed_args = pending.explicit_input
                raw_args = json.dumps(parsed_args, separators=(",", ":"))
            else:
                raise StreamTruncationError(f"Tool call {pending.name!r} completed with missing arguments payload.")
            verified.append(ToolCallDraft(pending.call_id or key, pending.name, parsed_args, raw_args, self.protocol))
        return verified

    def evaluate(self) -> StreamGateResult:
        """Non-throwing counterpart to finalize()."""
        try:
            calls = self.finalize()
            return StreamGateResult(True, self.protocol, "verified", self._terminal_reason or "completed", tuple(calls))
        except StreamTruncationError as exc:
            return StreamGateResult(False, self.protocol, "blocked", str(exc), ())


def guard_tool_stream(events: Iterable[Mapping[str, Any] | str | Any], protocol: str = "auto") -> Iterator[ToolCallDraft]:
    """Buffer an entire tool-call stream and yield verified calls only after completion."""
    gate = StreamToolCallGate(protocol=protocol)
    for event in events:
        if isinstance(event, str) and (event.startswith(("data:", "event:", ":")) or event.strip() == ""):
            gate.feed_sse_line(event)
        else:
            gate.feed_event(event)
    yield from gate.finalize()
