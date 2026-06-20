from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

from mahjong_agent import (
    AgentResponder,
    AgentRuntime,
    ChannelAddress,
    ChannelType,
    CustomerProfile,
    DurableAgentProcessor,
    IncomingEnvelope,
    Message,
    OutboundMessage,
    OutboundResult,
    OutputRouter,
    RuntimeConfig,
    SQLiteDurableStore,
    dispatch_pending_outbox,
)


TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 6, 16, 12, 0, tzinfo=TZ)


class CaptureAdapter:
    channel = "wechat"

    def __init__(self) -> None:
        self.messages: list[OutboundMessage] = []

    def send(self, message: OutboundMessage) -> OutboundResult:
        self.messages.append(message)
        return OutboundResult(ok=True, adapter=self.channel, external_id=f"sent:{message.id}")


def make_processor(path: Path) -> DurableAgentProcessor:
    responder = AgentResponder(invite_limit=2)
    responder.core.upsert_customer(
        CustomerProfile(
            id="amy",
            display_name="Amy",
            preferred_levels=["0.5"],
            tags=["无烟"],
            smoke_free_preference=True,
            usual_start_hours=[19],
        )
    )
    return DurableAgentProcessor(
        AgentRuntime(responder, RuntimeConfig(log_path=None)),
        SQLiteDurableStore(path),
    )


def envelope(text: str, source_id: str = "msg-1") -> IncomingEnvelope:
    return IncomingEnvelope(
        message=Message(
            id=source_id,
            text=text,
            sender_id="host",
            sender_name="张哥",
            channel_id="console_main",
            channel_type=ChannelType.MANUAL,
            metadata={"output_channel": "wechat"},
        ),
        tenant_id="shop",
        source_message_id=source_id,
        sequence=1,
    )


def test_reply_text_is_persisted_to_outbox() -> None:
    with TemporaryDirectory() as tmp:
        processor = make_processor(Path(tmp) / "agent.sqlite3")

        result = processor.process(envelope("今天下班有人打麻将吗"), now=NOW)

        assert result.runtime_result is not None
        rows = processor.store.pending_outbox_events()
        assert len(rows) == 1
        assert rows[0]["output_channel"] == "wechat"
        assert rows[0]["target_type"] == "reply"
        assert rows[0]["target_id"] == "shop:console_main"
        assert "几点到" in rows[0]["message_text"]
        processor.shutdown()


def test_output_router_can_redirect_any_outbox_to_test_wechat_recipient() -> None:
    with TemporaryDirectory() as tmp:
        processor = make_processor(Path(tmp) / "agent.sqlite3")
        processor.process(envelope("今天下班有人打麻将吗"), now=NOW)
        capture = CaptureAdapter()
        router = OutputRouter(
            adapters={"wechat": capture},
            default_channel="wechat",
            test_redirect=ChannelAddress("wechat", "private", "radon_1"),
        )

        dispatched = dispatch_pending_outbox(processor.store, router)

        assert len(dispatched) == 1
        assert capture.messages[0].target.channel == "wechat"
        assert capture.messages[0].target.target_type == "private"
        assert capture.messages[0].target.target_id == "radon_1"
        assert capture.messages[0].original_target is not None
        assert capture.messages[0].original_target.target_id == "shop:console_main"
        assert processor.store.pending_outbox_events() == []
        snapshot = processor.snapshot()["durable"]
        assert snapshot["outbox"][0]["status"] == "sent"
        processor.shutdown()
