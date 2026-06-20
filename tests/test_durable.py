from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

from mahjong_agent import (
    AgentResponder,
    AgentRuntime,
    ChannelType,
    CustomerProfile,
    DurableAgentProcessor,
    IncomingEnvelope,
    Message,
    ReplyAction,
    RuntimeConfig,
    SQLiteDurableStore,
)


TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 6, 16, 12, 0, tzinfo=TZ)


def make_processor(path: Path) -> DurableAgentProcessor:
    responder = AgentResponder(invite_limit=3)
    for customer in [
        CustomerProfile(
            id="amy",
            display_name="Amy",
            preferred_levels=["0.5"],
            tags=["无烟"],
            smoke_free_preference=True,
            usual_start_hours=[17],
        ),
        CustomerProfile(
            id="chen",
            display_name="陈姐",
            preferred_levels=["0.5"],
            tags=["无烟"],
            smoke_free_preference=True,
            usual_start_hours=[17],
        ),
    ]:
        responder.core.upsert_customer(customer)
    return DurableAgentProcessor(
        AgentRuntime(responder, RuntimeConfig(log_path=None)),
        SQLiteDurableStore(path),
        processing_lease_seconds=1,
    )


def make_message(
    text: str,
    sender_id: str = "host",
    source_id: str = "m1",
    channel_id: str = "group",
) -> Message:
    return Message(
        id=source_id,
        text=text,
        sender_id=sender_id,
        sender_name=sender_id,
        channel_id=channel_id,
        channel_type=ChannelType.WECHAT_GROUP,
    )


def envelope(
    text: str,
    source_id: str,
    sequence: int,
    sender_id: str = "host",
    channel_id: str = "group",
    received_at: datetime | None = None,
) -> IncomingEnvelope:
    return IncomingEnvelope(
        message=make_message(text, sender_id=sender_id, source_id=source_id, channel_id=channel_id),
        tenant_id="shop",
        source_message_id=source_id,
        sequence=sequence,
        received_at=received_at or datetime.now(TZ),
    )


def test_duplicate_source_message_is_idempotent() -> None:
    with TemporaryDirectory() as tmp:
        processor = make_processor(Path(tmp) / "agent.sqlite3")

        first = processor.process(
            envelope("今晚5点 0.5 三缺一 无烟", "msg-1", 1),
            now=NOW,
        )
        second = processor.process(
            envelope("今晚5点 0.5 三缺一 无烟", "msg-1", 1),
            now=NOW,
        )

        assert first.runtime_result is not None
        assert second.duplicate is True
        assert second.runtime_result is not None
        assert second.runtime_result.decision.action == first.runtime_result.decision.action
        snapshot = processor.snapshot()["durable"]
        assert snapshot["counts"]["inbound_messages"] == 1
        assert snapshot["counts"]["outbox_events"] == 4
        processor.shutdown()


def test_semantic_duplicate_same_sender_short_window_skips_side_effects() -> None:
    with TemporaryDirectory() as tmp:
        processor = make_processor(Path(tmp) / "agent.sqlite3")
        first_received_at = datetime(2026, 6, 16, 12, 0, tzinfo=TZ)

        first = processor.process(
            envelope(
                "今晚5点 0.5 三缺一 无烟",
                "msg-1",
                1,
                received_at=first_received_at,
            ),
            now=NOW,
        )
        second = processor.process(
            envelope(
                "今晚5点 0.5 三缺一 无烟",
                "msg-2",
                2,
                received_at=first_received_at + timedelta(seconds=5),
            ),
            now=NOW,
        )

        assert first.runtime_result is not None
        assert second.runtime_result is not None
        assert second.runtime_result.decision.action == ReplyAction.IGNORE
        assert second.runtime_result.decision.should_reply is False
        assert any("语义重复" in note for note in second.runtime_result.decision.notes)
        snapshot = processor.snapshot()["durable"]
        assert snapshot["counts"]["inbound_messages"] == 2
        assert snapshot["counts"]["outbox_events"] == 4
        assert snapshot["offsets"][0]["last_sequence"] == 2
        assert snapshot["message_statuses"]["processed"] == 2
        assert any(item["event_type"] == "semantic_duplicate_skipped" for item in snapshot["recent_audit"])
        processor.shutdown()


def test_semantic_duplicate_does_not_cross_senders() -> None:
    with TemporaryDirectory() as tmp:
        processor = make_processor(Path(tmp) / "agent.sqlite3")
        first_received_at = datetime(2026, 6, 16, 12, 0, tzinfo=TZ)

        first = processor.process(
            envelope("今天路上有点堵", "msg-1", 1, sender_id="amy", received_at=first_received_at),
            now=NOW,
        )
        second = processor.process(
            envelope(
                "今天路上有点堵",
                "msg-2",
                2,
                sender_id="chen",
                received_at=first_received_at + timedelta(seconds=5),
            ),
            now=NOW,
        )

        assert first.runtime_result is not None
        assert second.runtime_result is not None
        assert not any("语义重复" in note for note in second.runtime_result.decision.notes)
        snapshot = processor.snapshot()["durable"]
        assert snapshot["message_statuses"]["processed"] == 2
        assert all(item["event_type"] != "semantic_duplicate_skipped" for item in snapshot["recent_audit"])
        processor.shutdown()


def test_semantic_duplicate_does_not_cross_time_window() -> None:
    with TemporaryDirectory() as tmp:
        processor = make_processor(Path(tmp) / "agent.sqlite3")
        first_received_at = datetime(2026, 6, 16, 12, 0, tzinfo=TZ)

        first = processor.process(
            envelope("今天路上有点堵", "msg-1", 1, received_at=first_received_at),
            now=NOW,
        )
        second = processor.process(
            envelope(
                "今天路上有点堵",
                "msg-2",
                2,
                received_at=first_received_at + timedelta(minutes=5),
            ),
            now=NOW,
        )

        assert first.runtime_result is not None
        assert second.runtime_result is not None
        assert not any("语义重复" in note for note in second.runtime_result.decision.notes)
        snapshot = processor.snapshot()["durable"]
        assert snapshot["message_statuses"]["processed"] == 2
        assert all(item["event_type"] != "semantic_duplicate_skipped" for item in snapshot["recent_audit"])
        processor.shutdown()


def test_out_of_order_message_waits_then_drains() -> None:
    with TemporaryDirectory() as tmp:
        processor = make_processor(Path(tmp) / "agent.sqlite3")

        second = processor.process(
            envelope("今天路上有点堵", "msg-2", 2, sender_id="passerby"),
            now=NOW,
        )
        assert second.waiting_for_sequence is True
        assert second.runtime_result is None

        first = processor.process(
            envelope("今晚5点 0.5 三缺一 无烟", "msg-1", 1),
            now=NOW,
        )
        assert first.runtime_result is not None
        snapshot = processor.snapshot()["durable"]
        assert snapshot["offsets"][0]["last_sequence"] == 2
        assert snapshot["message_statuses"]["processed"] == 2
        processor.shutdown()


def test_concurrent_same_conversation_eventually_preserves_sequence() -> None:
    with TemporaryDirectory() as tmp:
        processor = make_processor(Path(tmp) / "agent.sqlite3")

        def submit(seq: int):
            return processor.process(
                envelope(f"无关消息 {seq}", f"msg-{seq}", seq, sender_id=f"user-{seq}"),
                now=NOW,
            )

        with ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(submit, range(1, 11)))

        assert len(results) == 10
        snapshot = processor.snapshot()["durable"]
        assert snapshot["offsets"][0]["last_sequence"] == 10
        assert snapshot["message_statuses"]["processed"] == 10
        processor.shutdown()


def test_concurrent_different_conversations_do_not_duplicate_private_invites() -> None:
    with TemporaryDirectory() as tmp:
        processor = make_processor(Path(tmp) / "agent.sqlite3")

        def submit(index: int):
            return processor.process(
                envelope(
                    "今晚5点 0.5 三缺一 无烟",
                    f"msg-{index}",
                    1,
                    sender_id=f"host-{index}",
                    channel_id=f"group-{index}",
                ),
                now=NOW,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(submit, [1, 2]))

        private_invitees = [
            invitation.customer_id
            for result in results
            if result.runtime_result is not None
            for invitation in result.runtime_result.decision.invitation_drafts
        ]

        assert private_invitees
        assert len(private_invitees) == len(set(private_invitees))
        snapshot = processor.snapshot()["durable"]
        private_outbox_targets = [
            item["target_id"]
            for item in snapshot["outbox"]
            if item["target_type"] == "private"
        ]
        assert len(private_outbox_targets) == len(set(private_outbox_targets))
        processor.shutdown()


def test_restart_restores_business_state_for_followup_message() -> None:
    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "agent.sqlite3"
        first_node = make_processor(db_path)
        opening = first_node.process(
            envelope("今晚5点 0.5 三缺一 无烟", "msg-1", 1),
            now=NOW,
        )
        assert opening.runtime_result is not None
        assert opening.runtime_result.decision.action == ReplyAction.QUEUE_INVITES
        first_node.shutdown()

        second_node = make_processor(db_path)
        signup = second_node.process(
            envelope("我来", "msg-2", 2, sender_id="amy"),
            now=NOW,
        )

        assert signup.runtime_result is not None
        assert signup.runtime_result.decision.action == ReplyAction.ACCEPT_SEAT
        assert "人数已齐" in signup.runtime_result.decision.reply_text
        second_node.shutdown()


def test_restart_restores_room_availability_state() -> None:
    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "agent.sqlite3"
        first_node = make_processor(db_path)
        first_node.runtime.responder.core.configure_room_capacity(1)
        first_node.runtime.responder.core.add_room_hold(
            start_at=datetime(2026, 6, 16, 16, 0, tzinfo=TZ),
            end_at=datetime(2026, 6, 16, 18, 0, tzinfo=TZ),
            room_id="room-1",
            source="room_schedule",
        )
        seed = first_node.process(
            envelope("今天天气不错", "room-seed", 1, sender_id="passerby"),
            now=datetime(2026, 6, 16, 15, 55, tzinfo=TZ),
        )
        assert seed.runtime_result is not None
        first_node.shutdown()

        second_node = make_processor(db_path)
        request = second_node.process(
            envelope("今晚5点 0.5 三缺一 无烟", "room-msg-1", 2),
            now=datetime(2026, 6, 16, 16, 0, tzinfo=TZ),
        )

        assert request.runtime_result is not None
        decision = request.runtime_result.decision
        assert decision.action == ReplyAction.ASK_CLARIFICATION
        assert "满房" in decision.reply_text
        assert "18:00" in decision.reply_text
        second_node.shutdown()
