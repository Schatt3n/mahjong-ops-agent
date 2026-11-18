"""实现内存存储层的输入aggregation数据访问能力，主要用于本地运行和测试。

"""

from __future__ import annotations

from typing import Any
import copy
from datetime import datetime
from datetime import timedelta
from ...models import (
    PendingInputBatch,
    PendingInputBatchStatus,
    StateTransition,
    new_id,
    now,
)
from ...domains import (
    PENDING_INPUT_PROCESSING_LEASE_SECONDS,
    pending_input_batch_key,
)

class InMemoryInputAggregationStoreMixin:
    """从兼容性存储中提取的后端特定操作。"""

    __slots__ = ()

    def upsert_pending_input_fragment(
        self,
        message,
        *,
        trace_id: str,
        quiet_deadline: datetime,
    ) -> tuple[PendingInputBatch, StateTransition | None, bool]:
        """新增或更新pending输入消息片段，使重复写入收敛到同一条记录。

        """

        key = pending_input_batch_key(message.conversation_id, message.sender_id)
        fragment = message.to_dict()
        with self._lock:
            existing = self.pending_input_batches.get(key)
            message_id = str(fragment.get("message_id") or "")
            if existing is not None and message_id and any(
                str(item.get("message_id") or "") == message_id for item in existing.fragments
            ):
                return copy.deepcopy(existing), None, False
            if existing is None or existing.status in {
                PendingInputBatchStatus.COMPLETED,
                PendingInputBatchStatus.IGNORED,
                PendingInputBatchStatus.FAILED,
            }:
                batch = PendingInputBatch(
                    batch_id=new_id("input_batch"),
                    conversation_id=message.conversation_id,
                    sender_id=message.sender_id,
                    sender_name=message.sender_name,
                    fragments=[fragment],
                    quiet_deadline=quiet_deadline,
                    source_channel=str(message.metadata.get("channel") or ""),
                )
                previous_status = "absent"
            else:
                batch = existing
                previous_status = batch.status.value
                batch.fragments.append(fragment)
                batch.sender_name = message.sender_name or batch.sender_name
                batch.version += 1
                batch.status = PendingInputBatchStatus.PENDING
                batch.quiet_deadline = quiet_deadline
                batch.updated_at = now()
                batch.decision = {}
                if message.metadata.get("channel"):
                    batch.source_channel = str(message.metadata["channel"])
            # 只对新增来源推进信号；聚合快照后续进入 Runtime 时会再取得自己的接收版。
            version_key = message.conversation_id or "default"
            previous_version = int(self.conversation_versions.get(version_key, 0))
            self.conversation_versions[version_key] = previous_version + 1
            self.transitions.append(StateTransition("conversation_version", version_key, str(previous_version),
                str(previous_version + 1), "input_fragment_received", trace_id))
            self.pending_input_batches[key] = batch
            transition = StateTransition(
                entity_type="pending_input_batch",
                entity_id=batch.batch_id,
                from_status=previous_status,
                to_status=batch.status.value,
                reason="input_fragment_buffered",
                trace_id=trace_id,
            )
            self.transitions.append(transition)
            return copy.deepcopy(batch), transition, True

    def pending_input_batch(self, conversation_id: str, sender_id: str) -> PendingInputBatch | None:
        with self._lock:
            batch = self.pending_input_batches.get(pending_input_batch_key(conversation_id, sender_id))
            return copy.deepcopy(batch) if batch is not None else None

    def due_pending_input_batches(
        self, *, at: datetime, limit: int = 100, exclude_conversations: set[str] | None = None,
    ) -> list[PendingInputBatch]:
        """选取到期或租期已过的批次，等待入站回执交接的会话不占用名额。"""
        with self._lock:
            stale_before = at - timedelta(seconds=PENDING_INPUT_PROCESSING_LEASE_SECONDS)
            excluded = exclude_conversations or set()
            pending_acknowledgements = {
                (record.get("conversation_id"), record["trace_id"])
                for record in self.channel_observations.values()
                if record.get("channel") == "agent-wechat"
                and record.get("route_mode") in {"ingress_queue", "ingress_reply_pending"}
                and record.get("trace_id")
            }
            due = [
                item
                for item in self.pending_input_batches.values()
                if item.conversation_id not in excluded and (
                    (item.status == PendingInputBatchStatus.PENDING and item.quiet_deadline <= at)
                    or (item.status == PendingInputBatchStatus.PROCESSING and item.updated_at <= stale_before)
                )
                and not (
                    item.source_channel == "agent-wechat"
                    and item.decision.get("dispatch_mode") == "background_after_immediate_ack"
                    and (item.conversation_id, item.decision.get("origin_trace_id")) in pending_acknowledgements
                )
            ]
            return copy.deepcopy(sorted(due, key=lambda item: item.quiet_deadline)[: max(1, int(limit))])

    def claim_pending_input_batch(
        self,
        *,
        batch_id: str,
        expected_version: int,
        trace_id: str,
    ) -> tuple[PendingInputBatch | None, StateTransition | None]:
        """原子领取pending输入批次，防止并发消费者重复处理。

"""

        with self._lock:
            batch = next((item for item in self.pending_input_batches.values() if item.batch_id == batch_id), None)
            stale_before = now() - timedelta(seconds=PENDING_INPUT_PROCESSING_LEASE_SECONDS)
            if (
                batch is None
                or batch.version != int(expected_version)
                or (
                    batch.status != PendingInputBatchStatus.PENDING
                    and not (
                        batch.status == PendingInputBatchStatus.PROCESSING
                        and batch.updated_at <= stale_before
                    )
                )
            ):
                return None, None
            old = batch.status.value
            batch.status = PendingInputBatchStatus.PROCESSING
            batch.updated_at = now()
            transition = StateTransition(
                "pending_input_batch",
                batch.batch_id,
                old,
                batch.status.value,
                "input_batch_claimed",
                trace_id,
            )
            self.transitions.append(transition)
            return copy.deepcopy(batch), transition

    def record_pending_input_decision(
        self,
        *,
        batch_id: str,
        expected_version: int,
        decision: dict[str, Any],
    ) -> PendingInputBatch | None:
        """只保存待领取批次的分类，分类阶段不续租或改写已领取批次。"""

        with self._lock:
            batch = next((item for item in self.pending_input_batches.values() if item.batch_id == batch_id), None)
            if (
                batch is None
                or batch.version != int(expected_version)
                or batch.status != PendingInputBatchStatus.PENDING
            ):
                return None
            batch.decision = dict(decision)
            batch.updated_at = now()
            return copy.deepcopy(batch)

    def queue_pending_input_batch(
        self,
        *,
        batch_id: str,
        expected_version: int,
        decision: dict[str, Any],
        due_at: datetime,
    ) -> PendingInputBatch | None:
        """原子地使一个精确批次版本进入待后台分发状态。"""

        with self._lock:
            batch = next((item for item in self.pending_input_batches.values() if item.batch_id == batch_id), None)
            if (
                batch is None
                or batch.version != int(expected_version)
                or batch.status != PendingInputBatchStatus.PENDING
            ):
                return None
            batch.decision = dict(decision)
            batch.quiet_deadline = due_at
            batch.updated_at = now()
            return copy.deepcopy(batch)

    def finish_pending_input_batch(
        self,
        *,
        batch_id: str,
        expected_version: int,
        status: PendingInputBatchStatus,
        trace_id: str,
        decision: dict[str, Any] | None = None,
    ) -> tuple[PendingInputBatch | None, StateTransition | None]:
        with self._lock:
            batch = next((item for item in self.pending_input_batches.values() if item.batch_id == batch_id), None)
            if batch is None or batch.version != int(expected_version):
                return None, None
            old = batch.status.value
            batch.status = status
            batch.decision = dict(decision or {})
            batch.updated_at = now()
            transition = StateTransition(
                "pending_input_batch",
                batch.batch_id,
                old,
                status.value,
                "input_batch_finished",
                trace_id,
            )
            self.transitions.append(transition)
            return copy.deepcopy(batch), transition
