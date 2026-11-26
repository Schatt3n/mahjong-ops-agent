"""托管群分类超时的独立接手和重试凭证，不把群原话送入私聊聚合。"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from datetime import datetime, timedelta

from ..models import PendingInputBatch, PendingInputBatchStatus, UserMessage
from .models import GroupHandleResult, GroupMessage


GROUP_TOPIC_RECOVERY_ROUTE = "managed_group_topic"


class GroupTopicRecoverySuperseded(RuntimeError):
    """分类重试已被更新的来话、接手或现存来源回执替代。"""


def group_ingress_key(message: GroupMessage) -> str:
    """按同群同发送人隔离入口顺序，原始标识不会成为主 Agent 的私聊内容。"""
    return json.dumps([message.channel, message.room_id, message.sender_external_id], ensure_ascii=False)


def _scope(message: GroupMessage) -> str:
    return "group-topic-recovery:" + hashlib.sha256(group_ingress_key(message).encode()).hexdigest()


def load_group_topic_recovery(batch: PendingInputBatch) -> GroupMessage:
    """只从独立入口的单条不可合并片段还原原消息和已验证的通道凭证。"""
    if batch.source_channel != GROUP_TOPIC_RECOVERY_ROUTE or len(batch.fragments) != 1:
        raise ValueError("invalid group topic recovery input")
    fragment = batch.fragments[0]
    payload = dict(fragment["metadata"]["group_message"])
    payload["sent_at"] = datetime.fromisoformat(payload["sent_at"])
    message = GroupMessage(**payload)
    if (batch.conversation_id != _scope(message) or batch.sender_id != message.sender_external_id
            or fragment["message_id"] != message.message_id or fragment["text"] != message.text):
        raise ValueError("group topic recovery source mismatch")
    return message


def pending_group_topic_recovery(store, message: GroupMessage) -> PendingInputBatch | None:
    """适配器领取的候选仍必须是本次同一条群来话。"""
    batch = store.pending_input_batch(_scope(message), message.sender_external_id)
    return batch if batch is not None and load_group_topic_recovery(batch).message_id == message.message_id else None


def _source_identity(store, message: GroupMessage):
    """重放只允许由当前同一托管群和原发送人身份承接。"""
    policy = store.get_group_room_policy(message.room_id)
    if not policy or not policy.managed or policy.channel != message.channel:
        return None
    return store.get_channel_identity(message.channel, message.sender_external_id)


def protect_group_topic_source(store, message: GroupMessage, *, trace_id: str) -> GroupHandleResult | None:
    """新群来话终止同人旧待办；同一来源重复到达沿用原分类重试或结算。"""
    scope = _scope(message)
    batch = store.pending_input_batch(scope, message.sender_external_id)
    if batch is None:
        return None
    source = load_group_topic_recovery(batch)
    recovery = batch.decision.get("gate_recovery") or {}
    if source.message_id == message.message_id:
        if (source.text, source.sent_at, source.quoted_message_id) != (
                message.text, message.sent_at, message.quoted_message_id):
            raise ValueError("group topic recovery message id conflict")
        return GroupHandleResult(action="group_topic_retry_existing", detail={"batch_id": batch.batch_id})
    if recovery.get("open") or batch.status in {PendingInputBatchStatus.PENDING, PendingInputBatchStatus.PROCESSING}:
        if message.sent_at < source.sent_at:
            return GroupHandleResult(action="group_topic_retry_superseded")
        store.advance_conversation_version(scope, trace_id=trace_id, reason="new_managed_group_source")
        store.finish_pending_input_batch(batch_id=batch.batch_id, expected_version=batch.version,
            status=PendingInputBatchStatus.IGNORED, trace_id=trace_id,
            decision={"gate_recovery": {**recovery, "open": False}, "reason": "new_managed_group_source"})
    return None


def queue_group_topic_recovery(store, message: GroupMessage, *, trace_id: str, at: datetime) -> PendingInputBatch | None:
    """同一故障建立一次到期重试和一个立即可见的内部接手项，不向群广播等待词。"""
    identity = _source_identity(store, message)
    if identity is None or not message.message_id:
        return None
    scope = _scope(message)
    due = at + timedelta(seconds=1)
    fragment = UserMessage(conversation_id=scope, sender_id=message.sender_external_id,
        sender_name=message.sender_name, text=message.text, message_id=message.message_id, sent_at=message.sent_at,
        metadata={"channel": GROUP_TOPIC_RECOVERY_ROUTE, "group_message": message.to_dict(),
                  "origin_trace_id": trace_id, "customer_id": identity.customer_id})
    batch, _, added = store.upsert_pending_input_fragment(fragment, trace_id=trace_id, quiet_deadline=due)
    if not added:
        return batch
    return _open_recovery(store, batch, at=due)


def _open_recovery(store, batch: PendingInputBatch, *, at: datetime) -> PendingInputBatch | None:
    """将来源片段与接手版本绑定，崩溃中断建单后可从片段继续同一个批次。"""
    metadata = batch.fragments[0]["metadata"]
    recovery = {"attempt": 1, "conversation_version": store.conversation_version(batch.conversation_id),
                "origin_trace_id": metadata["origin_trace_id"], "route": GROUP_TOPIC_RECOVERY_ROUTE,
                "open": True, "wait_delivered": False}
    return store.queue_pending_input_batch(batch_id=batch.batch_id, expected_version=batch.version,
        due_at=at, decision={"gate_recovery": recovery})


def execute_group_topic_recovery(store, batch: PendingInputBatch, *, route, apply, trace_recorder=None,
                                 authorize: Callable[[GroupMessage], bool] | None = None) -> GroupHandleResult:
    """同一群入口锁内分类，话题落库前以版本栅栏与店家接手竞争。"""
    source = load_group_topic_recovery(batch)
    current = store.pending_input_batch(batch.conversation_id, batch.sender_id)
    if current is None or (current.batch_id, current.version) != (batch.batch_id, batch.version):
        return GroupHandleResult(action="group_topic_retry_superseded")
    batch = current
    if batch.status is PendingInputBatchStatus.PENDING and not batch.decision.get("gate_recovery"):
        batch = _open_recovery(store, batch, at=batch.quiet_deadline) or batch
    recovery = batch.decision.get("gate_recovery") or {}
    if (not recovery.get("open") or recovery.get("route") != GROUP_TOPIC_RECOVERY_ROUTE
            or recovery.get("conversation_version") != store.conversation_version(batch.conversation_id)):
        return GroupHandleResult(action="group_topic_retry_superseded")
    trace_id = recovery["origin_trace_id"]

    def finish(status, reason, *, open_for_owner, routing=None):
        """自动步骤耗尽保留人工项，结束后沿用同一来源和版本。"""
        result, transition = store.finish_pending_input_batch(batch_id=batch.batch_id,
            expected_version=batch.version, status=status, trace_id=trace_id,
            decision={"gate_recovery": {**recovery, "attempt": 2, "open": open_for_owner}, "reason": reason})
        if trace_recorder is not None and transition:
            trace_recorder.record(trace_id, "group_topic_retry_finished", {"batch_id": batch.batch_id,
                "batch_version": batch.version, "source_message_id": source.message_id, "room_id": source.room_id,
                "status": status.value, "reason": reason, "route_action": routing.action if routing else None,
                **({"private_conversation_id": routing.detail.get("private_conversation_id")}
                   if routing and routing.action == "private_switch" else {})})
        return result

    # 领过的旧租约可能已写话题或执行业务。即使结果未知，也只交给店家，禁止整轮重做。
    if (batch.status is PendingInputBatchStatus.PROCESSING
            or (batch.status is PendingInputBatchStatus.PENDING and recovery.get("attempt") == 2)):
        finish(PendingInputBatchStatus.FAILED, "group_topic_previous_claim_outcome_unknown", open_for_owner=True)
        return GroupHandleResult(action="group_topic_needs_human")
    if batch.status is not PendingInputBatchStatus.PENDING:
        return GroupHandleResult(action="group_topic_retry_existing", detail={"batch_id": batch.batch_id})

    def check_permission():
        """模型调用期间通道权限或绑定可能变化，写入前再核对一次。"""
        identity = _source_identity(store, source)
        if (not identity or identity.customer_id != batch.fragments[0]["metadata"]["customer_id"]
                or (authorize is not None and not authorize(source))):
            raise PermissionError("managed group source is no longer authorized")

    def claim(*, already_recorded):
        """在首个话题写事务前使用当前版本；店家拿到接手凭证后本回调不能领取。"""
        if already_recorded:
            finish(PendingInputBatchStatus.IGNORED, "group_topic_source_already_recorded", open_for_owner=False)
            raise GroupTopicRecoverySuperseded("group topic source already recorded")
        check_permission()
        if store.conversation_version(batch.conversation_id) != recovery["conversation_version"]:
            raise GroupTopicRecoverySuperseded("group topic conversation version advanced")
        claimed, _ = store.claim_pending_input_batch(batch_id=batch.batch_id, expected_version=batch.version, trace_id=trace_id)
        if claimed is None:
            raise GroupTopicRecoverySuperseded("group topic retry already superseded")

    try:
        # 分类前也必须检查授权，观察模式不会因此重新获得群管理权。
        check_permission()
        # 网络调用前保存尝试次数。宕机后新的进程不用第三次请求碰运气，继续由店家接手。
        started = store.record_pending_input_decision(batch_id=batch.batch_id, expected_version=batch.version,
            decision={"gate_recovery": {**recovery, "attempt": 2}})
        if started is None:
            raise GroupTopicRecoverySuperseded("group topic retry already superseded")
        routing = route(source, trace_id=trace_id, before_record=claim)
        result = apply(source, routing, trace_id=trace_id)
    except GroupTopicRecoverySuperseded:
        return GroupHandleResult(action="group_topic_retry_superseded")
    except Exception as error:
        finish(PendingInputBatchStatus.FAILED, type(error).__name__, open_for_owner=True)
        return GroupHandleResult(action="group_topic_needs_human", detail={"batch_id": batch.batch_id})
    finish(PendingInputBatchStatus.COMPLETED, "group_topic_retry_routed", open_for_owner=False, routing=result)
    return result
