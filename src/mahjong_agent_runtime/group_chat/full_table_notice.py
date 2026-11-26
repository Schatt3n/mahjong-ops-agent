"""通过正式系统通知、租约和稳定发送身份交付私聊成局回告。"""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import re
import unicodedata

from ..customer_visible_contract import customer_visible_text_contract_violations
from ..customer_visible_review import CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME, customer_visible_content_review_approved
from ..domains.full_table_notice import (
    FULL_TABLE_NOTICE_TASK_TYPE, full_table_notice_facts, full_table_private_contact, ready_table_fingerprint,
)
from ..models import GameStatus, MessageReference, ScheduledAgentTask, SystemTriggerMessage, now
from ..scheduled_tasks import ScheduledTaskDeferred


# 稳定规则只识别满桌回告中的独立准点断言。计划、带条件的说法和疑问由语义审查判断。
_START_GUARANTEE_RULE = "full_table_notice.unsupported_start_guarantee"
_START_REVIEW_GUIDANCE = (
    "只告知该桌人齐了和已知开场安排。计划时刻可以照实转告。实际开场尚未发生，不能声称已开场，"
    "也不能承诺准时开打。审查必须拒绝把计划写成准时保证的稿件，让主 Agent 按计划事实重新表达。"
    "不提档位、时长、包间、其他顾客或后续承诺。"
)
_CLOCK_LABEL = r"(?:\d{1,2}:\d{2}|[零一二两三四五六七八九十\d]{1,3}点(?:半|[零一二两三四五六七八九十\d]{1,3}分)?)"
_PUNCTUAL_START_DECLARATION = re.compile(
    rf"(?:(?:这桌|这局)?人(?:到)?齐了[,。.!]\s*)?"
    rf"(?:(?:今天|今晚|明天|明晚|晚上|下午)?\s*{_CLOCK_LABEL}\s*)?"
    r"(?:一定|保证)?\s*(?:准时|准点)\s*开(?:打|局|桌|场)[。.!]?"
)


def _declares_guaranteed_start(text: str) -> bool:
    """只拦清楚独立地保证准时开的满桌回告，避免误杀否定、条件及原定安排。"""
    return _PUNCTUAL_START_DECLARATION.fullmatch(unicodedata.normalize("NFKC", text).strip()) is not None


def _current_notice_target(store, task: ScheduledAgentTask):
    """确认仍是原时段、原成局和同一位已占席的私聊好友。"""
    if datetime.fromisoformat(task.payload["expires_at"]) <= now():
        return None
    game = store.games.get(task.aggregate_id)
    if (game is None or game.status != GameStatus.READY or game.remaining_seats() != 0
            or game.actual_started_at is not None or ready_table_fingerprint(game) != task.payload["fingerprint"]):
        return None
    room_id = task.payload["room_id"]
    policy = store.get_group_room_policy(room_id)
    if policy is None or not policy.managed or not policy.board_enabled or not policy.outbound_enabled:
        raise ScheduledTaskDeferred("full_table_notice_room_paused")
    identity = full_table_private_contact(store, game, room_id=room_id, customer_id=task.subject_id)
    if (identity is None or identity.channel != task.payload["channel"]
            or identity.private_conversation_id != task.conversation_id):
        return None
    return game, identity


def verify_full_table_notice_claim(store, task: ScheduledAgentTask):
    """在声明交付和实际传给桥接前，重读同一满桌快照、收件人和未过期领取凭证。"""
    latest = _current_notice_target(store, task)
    if latest is None:
        raise ScheduledTaskDeferred("full_table_notice_colocated_facts_changed", delay_seconds=5)
    claim = store.scheduled_tasks.get(task.task_id)
    if (claim is None or task.lease_until is None or task.lease_until <= now()
            or not claim.matches_claim(expected_attempts=task.attempts, expected_lease_until=task.lease_until)):
        raise ScheduledTaskDeferred("full_table_notice_claim_changed", delay_seconds=5)
    return latest[1]


def handle_full_table_notice(runtime, messenger, task: ScheduledAgentTask, trace_id: str, *, room_paused=None) -> None:
    """只发经过审查的本局满桌消息；失败重新领取，通道重试复用同一个发送键。"""
    from ..runtime import message_idempotency_key

    store = runtime.store
    room_id = task.payload["room_id"]

    def target():
        contact = _current_notice_target(store, task)
        if contact is not None and room_paused is not None and room_paused(room_id):
            raise ScheduledTaskDeferred("full_table_notice_room_paused")
        return contact

    source = store.resolve_message_reference(conversation_id=task.conversation_id, message_id=task.idempotency_key)
    if (source is not None and source.business_ref_type == FULL_TABLE_NOTICE_TASK_TYPE
            and source.business_ref_id == task.aggregate_id and source.recipient_id == task.subject_id):
        store.mark_assistant_reply_delivered(task.conversation_id, trace_id=source.metadata["trace_id"])
        return
    contact = target()
    if contact is None:
        runtime.trace_recorder.record(trace_id, "full_table_notice_skipped", {"task_id": task.task_id})
        return
    game, identity = contact
    # 满桌只证明席位齐备。按本触发的有效性条件，还没有真实开场，也没有准点保证。
    facts = full_table_notice_facts(game) | {"actual_started_at": None, "on_time_start_guaranteed": False}
    payload = {"trigger_summary": "你将通知这位已占席的顾客，他参与的牌局现已满员。", "game": facts,
               "reply_guidelines": _START_REVIEW_GUIDANCE}
    if task.last_error == f"RuntimeError: {_START_GUARANTEE_RULE}":
        payload["delivery_review_feedback"] = {
            "rule_id": _START_GUARANTEE_RULE,
            "instruction": "上次正文保证准时开场，未外发。请保留真实安排，按开场计划重新写一条简短通知。",
        }
    for attempt in range(1, task.attempts + 1):
        # 若上一轮已经生成且仍可发，复用同一正文。只有模型未成功或交付版本已过期才重新生成。
        trigger_key = sha256(f"{task.idempotency_key}:{attempt}".encode()).hexdigest()
        trigger = SystemTriggerMessage(
            trigger_id=f"group_game_full:{trigger_key}", trigger_type="group_game_full",
            conversation_id=task.conversation_id, sender_id=identity.customer_id, sender_name=identity.public_name,
            payload=payload,
        )
        if attempt < task.attempts:
            result = store.idempotent_message_result(message_idempotency_key(trigger.to_user_message()))
        else:
            result = runtime.handle_system_trigger(trigger, trace_id=trace_id)
        if (result is None or result.status != "completed" or not result.final_reply
                or customer_visible_text_contract_violations(result.final_reply)):
            continue
        review = next((tool for tool in reversed(result.tool_results)
                       if tool.name == CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME), None)
        if review is None or not customer_visible_content_review_approved(review):
            continue
        if _declares_guaranteed_start(result.final_reply):
            if attempt == task.attempts:
                runtime.trace_recorder.record(trace_id, "full_table_notice_content_rejected", {
                    "rule_id": _START_GUARANTEE_RULE, "task_id": task.task_id, "candidate_trace_id": result.trace_id,
                })
                # 现有任务按有限次数重试。不能在此改写原稿，或以同一送达凭证替换正文。
                raise RuntimeError(_START_GUARANTEE_RULE)
            continue
        with runtime.input_receipt_lock(task.conversation_id):
            identity = verify_full_table_notice_claim(store, task)
            if not store.claim_assistant_reply_delivery(task.conversation_id, trace_id=result.trace_id):
                # 若上次桥接实际已发送但回执丢失，此后绝不能用同一业务键替换正文再补记成功。
                raise RuntimeError("reviewed full table notice is no longer deliverable")
        # 交付已在输入栅栏声明，桥接网络不占用顾客接收新消息的锁。
        receipt = messenger.send_private_message(identity.external_user_id, result.final_reply, metadata={
            "trace_id": result.trace_id, "draft_id": task.idempotency_key, "source_message_id": task.idempotency_key,
            "origin_room_id": room_id, "conversation_id": task.conversation_id,
            "message_type": FULL_TABLE_NOTICE_TASK_TYPE,
        })
        if not receipt or receipt.startswith("suppressed:"):
            raise RuntimeError("full table notice delivery was not acknowledged")
        store.register_message_reference(MessageReference(
            message_id=task.idempotency_key, conversation_id=task.conversation_id,
            business_ref_type=FULL_TABLE_NOTICE_TASK_TYPE, business_ref_id=game.game_id,
            text=result.final_reply, channel=identity.channel, recipient_id=identity.customer_id,
            recipient_name=identity.public_name, metadata={"delivery_id": receipt, "trace_id": result.trace_id},
        ))
        store.mark_assistant_reply_delivered(task.conversation_id, trace_id=result.trace_id)
        runtime.trace_recorder.record(trace_id, "full_table_notice_sent",
                                      {"task_id": task.task_id, "game_id": game.game_id, "delivery_id": receipt})
        return
    raise RuntimeError("full table notice had no deliverable reviewed reply")
