from __future__ import annotations

import json
import logging
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import DEFAULT_TZ, Message
from .responder import AgentResponder, ReplyAction, ReplyDecision
from .signals import has_intent_content


@dataclass(slots=True)
class RuntimeConfig:
    timeout_seconds: float = 3.0
    max_text_chars: int = 500
    max_recent_messages_per_context: int = 50
    max_workers: int = 4
    log_path: Path | None = None
    fail_closed_reply: str = "这个我先转人工确认一下。"


@dataclass(slots=True)
class ContextTurn:
    message_id: str
    sender_id: str
    sender_name: str
    text: str
    decision_action: str | None
    reply_text: str | None
    created_at: datetime = field(default_factory=lambda: datetime.now(DEFAULT_TZ))

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "text": self.text,
            "decision_action": self.decision_action,
            "reply_text": self.reply_text,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(slots=True)
class ConversationContext:
    channel_id: str
    turns: deque[ContextTurn]
    created_at: datetime = field(default_factory=lambda: datetime.now(DEFAULT_TZ))
    updated_at: datetime = field(default_factory=lambda: datetime.now(DEFAULT_TZ))

    def append(self, turn: ContextTurn) -> None:
        self.turns.append(turn)
        self.updated_at = datetime.now(DEFAULT_TZ)

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_id": self.channel_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "turn_count": len(self.turns),
            "recent_turns": [turn.to_dict() for turn in list(self.turns)[-10:]],
        }


@dataclass(slots=True)
class RuntimeMetrics:
    total_messages: int = 0
    total_errors: int = 0
    total_timeouts: int = 0
    total_human_reviews: int = 0
    total_ignored: int = 0
    action_counts: Counter[str] = field(default_factory=Counter)
    latency_ms_total: float = 0.0
    latency_ms_max: float = 0.0
    started_at: datetime = field(default_factory=lambda: datetime.now(DEFAULT_TZ))

    def observe(self, decision: ReplyDecision, latency_ms: float, error: bool = False, timed_out: bool = False) -> None:
        self.total_messages += 1
        self.latency_ms_total += latency_ms
        self.latency_ms_max = max(self.latency_ms_max, latency_ms)
        self.action_counts[decision.action.value] += 1
        if error:
            self.total_errors += 1
        if timed_out:
            self.total_timeouts += 1
        if decision.needs_human_review:
            self.total_human_reviews += 1
        if decision.should_reply is False:
            self.total_ignored += 1

    def to_dict(self) -> dict[str, Any]:
        avg = self.latency_ms_total / self.total_messages if self.total_messages else 0.0
        return {
            "started_at": self.started_at.isoformat(),
            "total_messages": self.total_messages,
            "total_errors": self.total_errors,
            "total_timeouts": self.total_timeouts,
            "total_human_reviews": self.total_human_reviews,
            "total_ignored": self.total_ignored,
            "action_counts": dict(self.action_counts),
            "latency_ms_avg": round(avg, 2),
            "latency_ms_max": round(self.latency_ms_max, 2),
        }


@dataclass(slots=True)
class RuntimeResult:
    ok: bool
    decision: ReplyDecision
    latency_ms: float
    timed_out: bool = False
    error: str | None = None
    context: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = self.decision.to_dict()
        data["runtime"] = {
            "ok": self.ok,
            "latency_ms": round(self.latency_ms, 2),
            "timed_out": self.timed_out,
            "error": self.error,
            "context": self.context,
        }
        return data


class JsonlEventLogger:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.logger = logging.getLogger("mahjong_agent.runtime")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict[str, Any]) -> None:
        event = {"logged_at": datetime.now(DEFAULT_TZ).isoformat(), **event}
        self.logger.info(json.dumps(event, ensure_ascii=False))
        if not self.path:
            return
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")


class AgentRuntime:
    """Production-like wrapper for context, logging, metrics, exceptions, and timeouts."""

    def __init__(self, responder: AgentResponder | None = None, config: RuntimeConfig | None = None) -> None:
        self.responder = responder or AgentResponder()
        self.config = config or RuntimeConfig()
        self.contexts: dict[str, ConversationContext] = {}
        self.metrics = RuntimeMetrics()
        self.logger = JsonlEventLogger(self.config.log_path)
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)

    def process_message(self, message: Message, now: datetime | None = None) -> RuntimeResult:
        started = time.monotonic()
        validation = self._validate(message)
        if validation:
            latency_ms = self._elapsed_ms(started)
            result = RuntimeResult(
                ok=False,
                decision=validation,
                latency_ms=latency_ms,
                error="validation_failed",
                context=self._record_context(message, validation),
            )
            self._observe_and_log(message, result, validation_failed=True)
            return result

        future = self.executor.submit(self.responder.respond, message, now)
        try:
            decision = future.result(timeout=self.config.timeout_seconds)
            latency_ms = self._elapsed_ms(started)
            result = RuntimeResult(
                ok=True,
                decision=decision,
                latency_ms=latency_ms,
                context=self._record_context(message, decision),
            )
            self._observe_and_log(message, result)
            return result
        except TimeoutError:
            future.cancel()
            latency_ms = self._elapsed_ms(started)
            decision = self._fallback_decision(
                note=f"单轮处理超过 {self.config.timeout_seconds} 秒，已中断自动流程。"
            )
            result = RuntimeResult(
                ok=False,
                decision=decision,
                latency_ms=latency_ms,
                timed_out=True,
                error="timeout",
                context=self._record_context(message, decision),
            )
            self._observe_and_log(message, result)
            return result
        except Exception as exc:
            latency_ms = self._elapsed_ms(started)
            decision = self._fallback_decision(note=f"{type(exc).__name__}: {exc}")
            result = RuntimeResult(
                ok=False,
                decision=decision,
                latency_ms=latency_ms,
                error=f"{type(exc).__name__}: {exc}",
                context=self._record_context(message, decision),
            )
            self._observe_and_log(message, result)
            return result

    def snapshot(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(),
            "contexts": {
                channel_id: context.to_dict()
                for channel_id, context in sorted(self.contexts.items(), key=lambda item: item[0])
            },
        }

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)

    def _validate(self, message: Message) -> ReplyDecision | None:
        if len(message.text) > self.config.max_text_chars:
            return self._fallback_decision(
                note=f"消息长度 {len(message.text)} 超过限制 {self.config.max_text_chars}。"
            )
        if not has_intent_content(message):
            return ReplyDecision(
                action=ReplyAction.IGNORE,
                reply_text="",
                confidence=1.0,
                should_reply=False,
                notes=["没有可用于判断的文字、转写、OCR 或表情描述，已忽略。"],
            )
        return None

    def _fallback_decision(self, note: str) -> ReplyDecision:
        return ReplyDecision(
            action=ReplyAction.HUMAN_REVIEW,
            reply_text=self.config.fail_closed_reply,
            confidence=1.0,
            needs_human_review=True,
            notes=[note],
        )

    def _record_context(self, message: Message, decision: ReplyDecision) -> dict[str, Any]:
        context_key = str(message.metadata.get("conversation_id") or message.channel_id)
        context = self.contexts.get(context_key)
        if context is None:
            context = ConversationContext(
                channel_id=context_key,
                turns=deque(maxlen=self.config.max_recent_messages_per_context),
            )
            self.contexts[context_key] = context
        context.append(
            ContextTurn(
                message_id=message.id,
                sender_id=message.sender_id,
                sender_name=message.sender_name,
                text=message.text,
                decision_action=decision.action.value,
                reply_text=decision.reply_text if decision.should_reply else None,
            )
        )
        return context.to_dict()

    def _observe_and_log(
        self,
        message: Message,
        result: RuntimeResult,
        validation_failed: bool = False,
    ) -> None:
        self.metrics.observe(
            result.decision,
            result.latency_ms,
            error=result.error is not None and not validation_failed,
            timed_out=result.timed_out,
        )
        self.logger.write(
            {
                "event": "message_processed",
                "message_id": message.id,
                "channel_id": message.channel_id,
                "sender_id": message.sender_id,
                "action": result.decision.action.value,
                "ok": result.ok,
                "timed_out": result.timed_out,
                "error": result.error,
                "latency_ms": round(result.latency_ms, 2),
                "needs_human_review": result.decision.needs_human_review,
                "should_reply": result.decision.should_reply,
                "llm_context_digest": result.decision.llm_context_digest,
                "notes": result.decision.notes,
            }
        )

    def _elapsed_ms(self, started: float) -> float:
        return (time.monotonic() - started) * 1000
