from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Any, Protocol

from .durable import IncomingEnvelope, SQLiteDurableStore
from .models import DEFAULT_TZ, ChannelType, Invitation, Message


@dataclass(slots=True)
class ChannelAddress:
    channel: str
    target_type: str
    target_id: str
    display_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        name = f" / {self.display_name}" if self.display_name else ""
        return f"{self.channel}:{self.target_type}:{self.target_id}{name}"


@dataclass(slots=True)
class OutboundMessage:
    id: str
    trace_id: str
    tenant_id: str
    conversation_id: str
    text: str
    target: ChannelAddress
    original_target: ChannelAddress | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OutboundResult:
    ok: bool
    adapter: str
    external_id: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MessageSource(Protocol):
    """Any inbound platform adapter should emit durable envelopes."""

    def fetch_new_messages(self) -> list[IncomingEnvelope]:
        ...


class OutboundAdapter(Protocol):
    channel: str

    def send(self, message: OutboundMessage) -> OutboundResult:
        ...


class Outbox(Protocol):
    """Legacy boundary for concrete adapters that send approved drafts."""

    def send_private_invite(self, invitation: Invitation) -> str:
        ...

    def send_group_message(self, channel_id: str, text: str) -> str:
        ...


class HumanApprovalOutbox:
    """Default safe outbox: collect drafts for a human operator to approve."""

    def __init__(self) -> None:
        self.pending: list[tuple[str, str]] = []

    def send_private_invite(self, invitation: Invitation) -> str:
        text = invitation.message_text or ""
        self.pending.append((invitation.customer_id, text))
        return "queued_for_human_approval"

    def send_group_message(self, channel_id: str, text: str) -> str:
        self.pending.append((channel_id, text))
        return "queued_for_human_approval"


class ConsoleInboundSource:
    """Small helper for converting console text into platform-neutral envelopes."""

    def __init__(
        self,
        tenant_id: str = "default",
        channel_id: str = "console",
        channel_type: ChannelType = ChannelType.MANUAL,
        sender_id: str = "console_user",
        sender_name: str = "Console User",
    ) -> None:
        self.tenant_id = tenant_id
        self.channel_id = channel_id
        self.channel_type = channel_type
        self.sender_id = sender_id
        self.sender_name = sender_name
        self._sequence = 0

    def envelope_for_text(self, text: str, metadata: dict[str, Any] | None = None) -> IncomingEnvelope:
        self._sequence += 1
        message = Message(
            text=text,
            sender_id=self.sender_id,
            sender_name=self.sender_name,
            channel_id=self.channel_id,
            channel_type=self.channel_type,
            metadata=metadata or {},
        )
        return IncomingEnvelope(
            message=message,
            tenant_id=self.tenant_id,
            source_message_id=message.id,
            sequence=self._sequence,
        )

    def fetch_new_messages(self) -> list[IncomingEnvelope]:
        return []


class ConsoleOutboundAdapter:
    channel = "console"

    def send(self, message: OutboundMessage) -> OutboundResult:
        original = message.original_target.label() if message.original_target else message.target.label()
        print("\n=== OUTBOUND ===")
        print(f"id: {message.id}")
        print(f"to: {message.target.label()}")
        print(f"original: {original}")
        print(message.text)
        print("=== END OUTBOUND ===\n")
        return OutboundResult(ok=True, adapter=self.channel, external_id=f"console:{message.id}")


class CommandOutboundAdapter:
    """Sends an outbound message by invoking an external command.

    The command receives JSON on stdin. This keeps platform senders outside the
    core process and lets WeChat/XHS/Douyin senders evolve independently.
    """

    def __init__(self, channel: str, command: list[str], timeout_seconds: float = 30.0) -> None:
        self.channel = channel
        self.command = command
        self.timeout_seconds = timeout_seconds

    def send(self, message: OutboundMessage) -> OutboundResult:
        payload = {
            "id": message.id,
            "trace_id": message.trace_id,
            "tenant_id": message.tenant_id,
            "conversation_id": message.conversation_id,
            "text": message.text,
            "target": {
                "channel": message.target.channel,
                "target_type": message.target.target_type,
                "target_id": message.target.target_id,
                "display_name": message.target.display_name,
                "metadata": message.target.metadata,
            },
            "original_target": {
                "channel": message.original_target.channel,
                "target_type": message.original_target.target_type,
                "target_id": message.original_target.target_id,
                "display_name": message.original_target.display_name,
                "metadata": message.original_target.metadata,
            }
            if message.original_target
            else None,
            "metadata": message.metadata,
        }
        try:
            result = subprocess.run(
                self.command,
                input=_json_dumps(payload),
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except Exception as exc:  # pragma: no cover - defensive boundary
            return OutboundResult(ok=False, adapter=self.channel, error=f"{type(exc).__name__}: {exc}")
        if result.returncode != 0:
            return OutboundResult(
                ok=False,
                adapter=self.channel,
                error=(result.stderr or result.stdout or f"exit {result.returncode}").strip(),
                metadata={"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode},
            )
        external_id = (result.stdout or "").strip() or f"{self.channel}:{message.id}"
        return OutboundResult(
            ok=True,
            adapter=self.channel,
            external_id=external_id,
            metadata={"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode},
        )


class WeChatTestOutboundAdapter:
    """Safe test adapter: route every platform's output to one WeChat test user."""

    channel = "wechat"

    def __init__(
        self,
        test_recipient_id: str = "radon_1",
        command: list[str] | None = None,
        dry_run: bool = True,
    ) -> None:
        self.test_recipient_id = test_recipient_id
        self.dry_run = dry_run
        self.command_adapter = CommandOutboundAdapter("wechat", command) if command else None

    @classmethod
    def from_env(cls) -> "WeChatTestOutboundAdapter":
        command_text = os.getenv("MAHJONG_WECHAT_SEND_COMMAND", "").strip()
        command = shlex.split(command_text) if command_text else None
        return cls(
            test_recipient_id=os.getenv("MAHJONG_WECHAT_TEST_RECIPIENT", "radon_1"),
            command=command,
            dry_run=os.getenv("MAHJONG_WECHAT_DRY_RUN", "1") != "0",
        )

    def send(self, message: OutboundMessage) -> OutboundResult:
        redirected = OutboundMessage(
            id=message.id,
            trace_id=message.trace_id,
            tenant_id=message.tenant_id,
            conversation_id=message.conversation_id,
            text=self._format_test_message(message),
            target=ChannelAddress("wechat", "private", self.test_recipient_id),
            original_target=message.original_target or message.target,
            metadata={**message.metadata, "test_redirect": True},
        )
        if self.dry_run or self.command_adapter is None:
            print(f"[WECHAT TEST DRY-RUN -> {self.test_recipient_id}]\n{redirected.text}\n")
            return OutboundResult(
                ok=True,
                adapter="wechat_test_dry_run",
                external_id=f"dry_run:{redirected.id}",
                metadata={"target_id": self.test_recipient_id},
            )
        return self.command_adapter.send(redirected)

    def _format_test_message(self, message: OutboundMessage) -> str:
        original = message.original_target or message.target
        return (
            "[模拟发送]\n"
            f"原通道：{original.channel}\n"
            f"原目标：{original.target_type}:{original.target_id}\n"
            f"Trace：{message.trace_id}\n\n"
            f"{message.text}"
        )


class OutputRouter:
    def __init__(
        self,
        adapters: dict[str, OutboundAdapter],
        default_channel: str = "console",
        test_redirect: ChannelAddress | None = None,
    ) -> None:
        self.adapters = adapters
        self.default_channel = default_channel
        self.test_redirect = test_redirect

    def dispatch(self, message: OutboundMessage) -> OutboundResult:
        routed = self._route(message)
        adapter = self.adapters.get(routed.target.channel) or self.adapters.get(self.default_channel)
        if adapter is None:
            return OutboundResult(
                ok=False,
                adapter="none",
                error=f"no outbound adapter for channel {routed.target.channel}",
            )
        return adapter.send(routed)

    def _route(self, message: OutboundMessage) -> OutboundMessage:
        if self.test_redirect is None:
            return message
        return OutboundMessage(
            id=message.id,
            trace_id=message.trace_id,
            tenant_id=message.tenant_id,
            conversation_id=message.conversation_id,
            text=message.text,
            target=self.test_redirect,
            original_target=message.original_target or message.target,
            metadata={**message.metadata, "router_redirect": True},
        )


def outbound_from_outbox_row(row: dict[str, Any]) -> OutboundMessage:
    target = ChannelAddress(
        channel=str(row.get("output_channel") or "console"),
        target_type=str(row.get("target_type") or "reply"),
        target_id=str(row.get("target_id") or row.get("conversation_id") or "unknown"),
    )
    original_target_id = row.get("original_target_id") or row.get("target_id")
    original = ChannelAddress(
        channel=str(row.get("output_channel") or "console"),
        target_type=str(row.get("target_type") or "reply"),
        target_id=str(original_target_id or "unknown"),
    )
    return OutboundMessage(
        id=str(row["id"]),
        trace_id=str(row["trace_id"]),
        tenant_id=str(row["tenant_id"]),
        conversation_id=str(row["conversation_id"]),
        text=str(row.get("message_text") or ""),
        target=target,
        original_target=original,
        metadata={
            "idempotency_key": row.get("idempotency_key"),
            "status": row.get("status"),
            "attempt_count": row.get("attempt_count"),
        },
    )


def dispatch_pending_outbox(
    store: SQLiteDurableStore,
    router: OutputRouter,
    limit: int = 50,
) -> list[tuple[OutboundMessage, OutboundResult]]:
    dispatched: list[tuple[OutboundMessage, OutboundResult]] = []
    for row in store.pending_outbox_events(limit=limit):
        message = outbound_from_outbox_row(row)
        result = router.dispatch(message)
        if result.ok:
            store.mark_outbox_sent(message.id, external_id=result.external_id)
        else:
            store.mark_outbox_failed(message.id, result.error or "unknown outbound error")
        dispatched.append((message, result))
    return dispatched


def _json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)
