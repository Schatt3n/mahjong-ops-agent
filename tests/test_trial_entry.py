from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from mahjong_agent.models import ChannelType
from mahjong_agent.trial_entry import TrialControlledEntryAdapter, TrialControlledRequestBuilder


TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 6, 30, 18, 0, tzinfo=TZ)


def make_builder() -> TrialControlledRequestBuilder:
    return TrialControlledRequestBuilder(
        trace_id_factory=lambda: "trace_generated",
        now_factory=lambda: NOW,
        parse_datetime=lambda value: datetime.fromisoformat(value) if value else None,
    )


def test_trial_controlled_request_builder_creates_web_console_message() -> None:
    request = make_builder().build(
        {
            "text": " 0.5无烟人齐开 ",
            "sender_id": "zhang",
            "sender_name": "张哥",
            "conversationId": "conv_1",
            "trace_id": "trace_1",
            "now": "2026-06-30T19:00:00+08:00",
        }
    )

    assert request.text == "0.5无烟人齐开"
    assert request.trace_id == "trace_1"
    assert request.conversation_id == "conv_1"
    assert request.now.isoformat() == "2026-06-30T19:00:00+08:00"
    assert request.message.text == "0.5无烟人齐开"
    assert request.message.sender_id == "zhang"
    assert request.message.sender_name == "张哥"
    assert request.message.channel_id == "conv_1"
    assert request.message.channel_type == ChannelType.WEB_CONSOLE
    assert request.message.metadata["conversation_id"] == "conv_1"
    assert request.message.metadata["trace_id"] == "trace_1"
    assert request.message.metadata["source"] == "boss_trial_controlled"


def test_trial_controlled_request_builder_rejects_empty_message() -> None:
    with pytest.raises(ValueError, match="消息不能为空"):
        make_builder().build({"text": "   "})


def test_trial_controlled_entry_runs_lifecycle_service_and_response_adapter() -> None:
    calls: list[tuple[str, object]] = []

    class FakeWorkflowService:
        def handle_message(self, message, *, now, trace_id):
            calls.append(("service", {"message": message, "now": now, "trace_id": trace_id}))
            return {"workflow": "result"}

    class FakeResponseAdapter:
        def build(self, **kwargs):
            calls.append(("response", kwargs))
            return {"ok": True, "trace_id": kwargs["trace_id"]}

    adapter = TrialControlledEntryAdapter(
        workflow_service=FakeWorkflowService(),
        response_adapter=FakeResponseAdapter(),
        request_builder=make_builder(),
        customer_reloader=lambda: calls.append(("reload", None)),
        lifecycle_runner=lambda now: calls.append(("lifecycle", now)),
    )

    result = adapter.analyze(
        {
            "text": "组",
            "sender_id": "zhang",
            "sender_name": "张哥",
            "conversation_id": "conv_2",
        }
    )

    assert result == {"ok": True, "trace_id": "trace_generated"}
    assert [name for name, _ in calls] == ["reload", "lifecycle", "service", "response"]
    service_call = calls[2][1]
    response_call = calls[3][1]
    assert service_call["message"].channel_id == "conv_2"
    assert service_call["message"].text == "组"
    assert service_call["now"] == NOW
    assert service_call["trace_id"] == "trace_generated"
    assert response_call["workflow_result"] == {"workflow": "result"}
    assert response_call["source_text"] == "组"
    assert response_call["sender_id"] == "zhang"
    assert response_call["sender_name"] == "张哥"
    assert response_call["now"] == NOW
