from __future__ import annotations

import time
from datetime import datetime
from tempfile import TemporaryDirectory
from pathlib import Path
from zoneinfo import ZoneInfo

from mahjong_agent import (
    AgentResponder,
    AgentRuntime,
    ChannelType,
    CustomerProfile,
    Message,
    ReplyAction,
    RuntimeConfig,
)


TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 6, 16, 12, 0, tzinfo=TZ)


def make_message(text: str, sender_id: str = "host") -> Message:
    return Message(
        text=text,
        sender_id=sender_id,
        sender_name=sender_id,
        channel_id="group",
        channel_type=ChannelType.WECHAT_GROUP,
    )


def test_runtime_processes_message_and_records_context() -> None:
    responder = AgentResponder()
    responder.core.upsert_customer(
        CustomerProfile(
            id="amy",
            display_name="Amy",
            preferred_levels=["0.5"],
            tags=["无烟"],
            smoke_free_preference=True,
            usual_start_hours=[17],
        )
    )
    runtime = AgentRuntime(responder, RuntimeConfig(max_recent_messages_per_context=2))

    result = runtime.process_message(make_message("今晚5点 0.5 三缺一 无烟"), now=NOW)

    assert result.ok is True
    assert result.decision.action in {ReplyAction.QUEUE_INVITES, ReplyAction.CREATE_GAME}
    assert result.context is not None
    assert result.context["turn_count"] == 1
    assert runtime.metrics.total_messages == 1
    runtime.shutdown()


def test_runtime_rejects_too_long_message_fail_closed() -> None:
    runtime = AgentRuntime(config=RuntimeConfig(max_text_chars=5))

    result = runtime.process_message(make_message("这是一条很长的消息"), now=NOW)

    assert result.ok is False
    assert result.error == "validation_failed"
    assert result.decision.action == ReplyAction.HUMAN_REVIEW
    assert result.decision.needs_human_review is True
    runtime.shutdown()


def test_runtime_allows_empty_text_when_metadata_has_transcript() -> None:
    runtime = AgentRuntime()

    result = runtime.process_message(
        Message(
            text="",
            sender_id="voice_user",
            sender_name="语音客",
            channel_id="group",
            channel_type=ChannelType.WECHAT_GROUP,
            metadata={"message_type": "audio", "audio_transcript": "下班想搓一把，有局吗"},
        ),
        now=NOW,
    )

    assert result.decision.action == ReplyAction.ASK_CLARIFICATION
    assert result.decision.should_reply is True
    runtime.shutdown()


class ExplodingResponder:
    def respond(self, message: Message, now: datetime | None = None):
        raise RuntimeError("boom")


def test_runtime_catches_exceptions_fail_closed() -> None:
    runtime = AgentRuntime(ExplodingResponder(), RuntimeConfig())  # type: ignore[arg-type]

    result = runtime.process_message(make_message("今晚5点 0.5 三缺一"), now=NOW)

    assert result.ok is False
    assert result.decision.action == ReplyAction.HUMAN_REVIEW
    assert "RuntimeError" in (result.error or "")
    assert runtime.metrics.total_errors == 1
    runtime.shutdown()


class SlowResponder:
    def respond(self, message: Message, now: datetime | None = None):
        time.sleep(0.05)
        return AgentResponder().respond(message, now)


def test_runtime_times_out_fail_closed() -> None:
    runtime = AgentRuntime(SlowResponder(), RuntimeConfig(timeout_seconds=0.01))  # type: ignore[arg-type]

    result = runtime.process_message(make_message("今晚5点 0.5 三缺一"), now=NOW)

    assert result.ok is False
    assert result.timed_out is True
    assert result.error == "timeout"
    assert result.decision.action == ReplyAction.HUMAN_REVIEW
    assert runtime.metrics.total_timeouts == 1
    runtime.shutdown()


def test_runtime_writes_jsonl_log() -> None:
    with TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "events.jsonl"
        runtime = AgentRuntime(config=RuntimeConfig(log_path=log_path))

        runtime.process_message(make_message("今天路上有点堵"), now=NOW)

        assert log_path.exists()
        assert "message_processed" in log_path.read_text(encoding="utf-8")
        runtime.shutdown()
