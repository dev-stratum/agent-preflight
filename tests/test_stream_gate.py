from __future__ import annotations

import pytest

from agent_preflight.stream_gate import (
    StreamToolCallGate,
    StreamTruncationError,
    guard_tool_stream,
)


def test_openai_chat_sse_complete_stream() -> None:
    sse_lines = [
        'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"run_query","arguments":"{\\"sql\\":"}}]},"finish_reason":null}]}',
        "",
        'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"SELECT 1\\"}"}}]},"finish_reason":null}]}',
        "",
        'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
        "",
        "data: [DONE]",
        "",
    ]
    calls = list(guard_tool_stream(sse_lines))
    assert len(calls) == 1
    assert calls[0].protocol == "openai_chat"
    assert calls[0].name == "run_query"
    assert calls[0].arguments == {"sql": "SELECT 1"}


def test_openai_chat_fail_closed_on_length_or_missing_finish_reason() -> None:
    gate_len = StreamToolCallGate()
    gate_len.feed_event({
        "choices": [{
            "index": 0,
            "delta": {"tool_calls": [{"index": 0, "id": "call_cut", "function": {"name": "delete_file", "arguments": '{"path": "/tmp/'}}]},
            "finish_reason": "length",
        }]
    })
    result = gate_len.evaluate()
    assert not result.safe_to_execute
    assert result.status == "blocked"
    assert "length" in result.reason

    gate_premature = StreamToolCallGate(protocol="openai_chat")
    gate_premature.feed_sse_lines([
        'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c2","function":{"name":"ping","arguments":"{}"}}]},"finish_reason":null}]}',
        "",
        "data: [DONE]",
        "",
    ])
    with pytest.raises(StreamTruncationError):
        gate_premature.finalize()


def test_openai_responses_complete_and_incomplete_streams() -> None:
    ok_events = [
        {"type": "response.created"},
        {"type": "response.output_item.added", "output_index": 0, "item": {"id": "fc_1", "call_id": "call_r1", "type": "function_call", "name": "deploy_service"}},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": '{"env": '},
        {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "delta": '"staging"}'},
        {"type": "response.function_call_arguments.done", "item_id": "fc_1", "arguments": '{"env": "staging"}'},
        {"type": "response.output_item.done", "item_id": "fc_1", "item": {"id": "fc_1", "type": "function_call", "status": "completed"}},
        {"type": "response.completed", "response": {"status": "completed"}},
    ]
    calls = list(guard_tool_stream(ok_events))
    assert len(calls) == 1
    assert calls[0].protocol == "openai_responses"
    assert calls[0].arguments == {"env": "staging"}

    bad_gate = StreamToolCallGate()
    for ev in ok_events[:3]:
        bad_gate.feed_event(ev)
    bad_gate.feed_event({"type": "response.incomplete", "response": {"incomplete_details": {"reason": "max_output_tokens"}}})
    res = bad_gate.evaluate()
    assert not res.safe_to_execute
    assert "response.incomplete" in res.reason


def test_anthropic_tool_use_complete_and_truncated() -> None:
    ok_events = [
        {"type": "message_start"},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "lookup"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"key": "val"}'}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        {"type": "message_stop"},
    ]
    calls = list(guard_tool_stream(ok_events))
    assert len(calls) == 1
    assert calls[0].protocol == "anthropic"
    assert calls[0].arguments == {"key": "val"}

    truncated = ok_events[:3] + [{"type": "message_delta", "delta": {"stop_reason": "max_tokens"}}, {"type": "message_stop"}]
    with pytest.raises(StreamTruncationError):
        list(guard_tool_stream(truncated))


def test_mcp_tool_call_complete_cancelled_and_truncated() -> None:
    ok_events = [
        {"jsonrpc": "2.0", "method": "notifications/tools/call_delta", "params": {"callId": "mcp_1", "name": "write_note", "delta": '{"path": '}},
        {"jsonrpc": "2.0", "id": 1, "result": {"id": "mcp_1", "name": "write_note", "arguments": '{"path": "notes.md"}'}},
    ]
    calls = list(guard_tool_stream(ok_events))
    assert len(calls) == 1
    assert calls[0].protocol == "mcp"
    assert calls[0].arguments == {"path": "notes.md"}

    cancelled_gate = StreamToolCallGate()
    cancelled_gate.feed_event(ok_events[0])
    cancelled_gate.feed_event({"jsonrpc": "2.0", "method": "notifications/cancelled"})
    assert not cancelled_gate.evaluate().safe_to_execute

    assert not StreamToolCallGate().evaluate().safe_to_execute
