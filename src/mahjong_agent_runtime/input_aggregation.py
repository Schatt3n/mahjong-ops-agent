"""聚合用户碎片化输入，并在静默窗口结束后重新触发语义判断。

"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from time import monotonic
from typing import Any, Callable

from .models import DEFAULT_TZ, PendingInputBatch, QuotedMessageRef, UserMessage, now


BACKGROUND_INPUT_DISPATCH_MODES = frozenset(
    {
        "background_after_immediate_ack",
        "background_after_accept",
    }
)


def is_background_input_dispatch(decision: dict[str, Any] | None) -> bool:
    """返回是否有持久化批次已排队等待语义处理。"""

    return str((decision or {}).get("dispatch_mode") or "") in BACKGROUND_INPUT_DISPATCH_MODES


@dataclass(slots=True)
class InputBatchDispatch:
    """输入边界协调器返回给通道适配器的结果。"""

    status: str
    batch: PendingInputBatch
    message: UserMessage | None = None
    decision: dict[str, Any] | None = None
    trace_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转换为只包含可序列化字段的字典表示。"""
        return {
            "status": self.status,
            "trace_id": self.trace_id,
            "batch": self.batch.to_dict(),
            "message": self.message.to_dict() if self.message else None,
            "decision": dict(self.decision or {}),
        }


def aggregate_pending_input_batch(
    batch: PendingInputBatch,
    *,
    quiet_period_elapsed: bool,
    trigger: str,
) -> UserMessage:
    """合并有序片段，同时保留其源 ID 以用于审计。"""

    fragments = [dict(item) for item in batch.fragments]
    latest = fragments[-1] if fragments else {}
    texts = [str(item.get("text") or "").strip() for item in fragments]
    joined_text = "\n".join(item for item in texts if item)
    metadata = dict(latest.get("metadata") or {}) if isinstance(latest.get("metadata"), dict) else {}
    # Keep observations from authenticated media fragments when later text updates the batch.
    visual_observations = []
    metadata.pop("visual_observations", None)
    metadata.pop("_trusted_visual_observation_only", None)
    for fragment in fragments:
        attributes = fragment.get("metadata")
        if not isinstance(attributes, dict) or attributes.get("_trusted_visual_observation_only") is not True:
            continue
        observations = attributes.get("visual_observations")
        if not isinstance(observations, list) or len(observations) != 1:
            continue
        observation = observations[0]
        if (isinstance(observation, dict) and observation.get("kind") in ("image", "sticker")
                and observation.get("source_message_id") == fragment.get("message_id")
                and isinstance(observation.get("description"), str)
                and isinstance(observation.get("visible_text"), str)):
            visual_observations.append(observation)
    if visual_observations:
        metadata["visual_observations"] = visual_observations[-8:]
        metadata["_trusted_visual_observation_only"] = len(visual_observations) == len(fragments)
    metadata["input_window"] = {
        "batch_id": batch.batch_id,
        "batch_version": batch.version,
        "fragment_count": len(fragments),
        "source_message_ids": [str(item.get("message_id") or "") for item in fragments],
        "fragments": [
            {
                "message_id": str(item.get("message_id") or ""),
                "text": str(item.get("text") or ""),
                "sent_at": str(item.get("sent_at") or ""),
            }
            for item in fragments
        ],
        "quiet_period_elapsed": bool(quiet_period_elapsed),
        "quiet_deadline": batch.quiet_deadline.isoformat(),
        "trigger": trigger,
    }
    return UserMessage(
        conversation_id=batch.conversation_id,
        sender_id=batch.sender_id,
        sender_name=batch.sender_name,
        text=joined_text,
        message_id=f"{batch.batch_id}:v{batch.version}",
        sent_at=_datetime_from_value(latest.get("sent_at")),
        quoted_message=_quoted_message_from_value(latest.get("quoted_message")),
        metadata=metadata,
    )


class PendingInputScheduler:
    """并行消费不同会话的到期批次，同一会话只分配一个后台槽位。

    导航：``MahjongAgentApplication.from_environment`` 构造本类，并把
    ``handle_due_pending_input_batch`` 注册为 ``handler``。本类只扫描到期项；真正的
    语义判断、批次 CAS 和 Runtime 调用发生在该回调中。
    """

    def __init__(
        self,
        *,
        store: Any,
        handler: Callable[[PendingInputBatch, str], None],
        trace_recorder: Any,
        poll_interval_seconds: float = 0.5,
        batch_limit: int = 50,
        max_workers: int = 4,
    ) -> None:
        """限制后台并行度，真正的版本校验和批次领取仍由应用层回调负责。"""
        self.store = store
        self.handler = handler
        self.trace_recorder = trace_recorder
        self.poll_interval_seconds = max(0.05, float(poll_interval_seconds))
        self.batch_limit = max(1, int(batch_limit))
        self.max_workers = max(1, int(max_workers))
        self._stop_event = threading.Event()
        self._selection_lock = threading.Lock()
        self._active_conversations: set[str] = set()
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        """启动固定数量的后台线程，在旧线程完全退出前不重复启动。"""
        with self._selection_lock:
            if any(thread.is_alive() for thread in self._threads):
                return
            self._stop_event.clear()
            self._threads = []
            for index in range(self.max_workers):
                thread = threading.Thread(target=self._run, name=f"pending-input-scheduler-{index}", daemon=True)
                thread.start()
                self._threads.append(thread)

    def stop(self, timeout_seconds: float = 2.0) -> None:
        """停止领取，所有线程共用一次限时退出窗口。"""
        deadline = monotonic() + max(0.0, float(timeout_seconds))
        with self._selection_lock:
            self._stop_event.set()
            threads = list(self._threads)
        for thread in threads:
            if thread is not threading.current_thread():
                thread.join(max(0.0, deadline - monotonic()))

    def run_due_once(self, *, at: datetime | None = None) -> int:
        """保留单次同步扫描的顺序；正在后台处理的会话留给原线程。"""
        due = self.store.due_pending_input_batches(at=at or now(), limit=self.batch_limit)
        handled = 0
        for batch in due:
            with self._selection_lock:
                if batch.conversation_id in self._active_conversations:
                    continue
                self._active_conversations.add(batch.conversation_id)
            self._dispatch_reserved_batch(batch)
            handled += 1
        return handled

    def _dispatch_reserved_batch(self, batch: PendingInputBatch) -> None:
        """锁外运行回调，结束后释放会话槽位。"""
        try:
            trace_id = f"trace_input_wait_{uuid.uuid4().hex[:12]}"
            background_queued = is_background_input_dispatch(batch.decision)
            self.trace_recorder.record(
                trace_id,
                "input_background_dispatch_due" if background_queued else "input_quiet_period_elapsed",
                {
                    "batch_id": batch.batch_id,
                    "batch_version": batch.version,
                    "conversation_id": batch.conversation_id,
                    "sender_id": batch.sender_id,
                    "fragment_count": len(batch.fragments),
                    "origin_trace_id": str(batch.decision.get("origin_trace_id") or ""),
                },
            )
            try:
                self.handler(batch, trace_id)
            except Exception as exc:
                self.trace_recorder.record(
                    trace_id,
                    "input_quiet_period_handler_error",
                    {"error_type": type(exc).__name__, "error": str(exc), "batch_id": batch.batch_id},
                    level="ERROR",
                )
        finally:
            with self._selection_lock:
                self._active_conversations.remove(batch.conversation_id)

    def _take_due_batch(self, handled: set[str]) -> PendingInputBatch | None:
        """跳过活跃会话后再选取到期项，避免同一会话的待办挡住其他会话。"""
        with self._selection_lock:
            if self._stop_event.is_set():
                return None
            due = self.store.due_pending_input_batches(
                at=now(), limit=1, exclude_conversations=self._active_conversations | handled,
            )
            if not due:
                return None
            batch = due[0]
            self._active_conversations.add(batch.conversation_id)
            return batch

    def _run(self) -> None:
        """每轮消费有限批次，回调异常时避免同一线程连续重试同一会话。"""
        while not self._stop_event.is_set():
            handled: set[str] = set()
            for _ in range(self.batch_limit):
                batch = self._take_due_batch(handled)
                if batch is None:
                    break
                handled.add(batch.conversation_id)
                self._dispatch_reserved_batch(batch)
            self._stop_event.wait(self.poll_interval_seconds)


def _datetime_from_value(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value)
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=DEFAULT_TZ)
        except ValueError:
            pass
    return now()


def _quoted_message_from_value(value: Any) -> QuotedMessageRef | None:
    if not isinstance(value, dict) or not value.get("message_id"):
        return None
    return QuotedMessageRef(
        message_id=str(value.get("message_id") or ""),
        sender_id=str(value.get("sender_id") or "") or None,
        sender_name=str(value.get("sender_name") or "") or None,
        text=str(value.get("text") or ""),
        conversation_id=str(value.get("conversation_id") or "") or None,
        business_ref_type=str(value.get("business_ref_type") or "") or None,
        business_ref_id=str(value.get("business_ref_id") or "") or None,
        metadata=dict(value.get("metadata") or {}) if isinstance(value.get("metadata"), dict) else {},
    )
