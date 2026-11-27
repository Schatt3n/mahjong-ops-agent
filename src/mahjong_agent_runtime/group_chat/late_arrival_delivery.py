"""交付晚到征询和回告，只把受审文本送给持久责任指定的私聊收件人。"""

from __future__ import annotations

import json

from ..budget import TokenBudget
from ..customer_visible_contract import customer_visible_text_contract_violations
from ..customer_visible_review import customer_visible_content_review_approved
from ..domains.table_availability_authorization import stated_arrival_estimate
from ..late_arrival import (
    LATE_ARRIVAL_QUESTION_TASK_TYPE,
    LATE_ARRIVAL_UPDATE_TASK_TYPE,
    late_arrival_delivery_context,
)
from ..models import AgentAction, ConversationRole, MessageReference, ScheduledAgentTask, UserMessage, now
from ..scheduled_tasks import ScheduledTaskDeferred
from .projections import public_game_start_display


LATE_ARRIVAL_DELIVERY_TYPES = frozenset({LATE_ARRIVAL_QUESTION_TASK_TYPE, LATE_ARRIVAL_UPDATE_TASK_TYPE})


def verify_late_arrival_delivery_claim(store, task: ScheduledAgentTask, *, room_paused=None):
    """在声明交付及调用桥接前检查真实桌况、收件范围和仍有效的领取代次。"""
    at = now()
    target = late_arrival_delivery_context(store, task, at=at)
    if target is None:
        return None
    if room_paused is not None and room_paused(task.payload["room_id"]):
        raise ScheduledTaskDeferred("late_arrival_delivery_paused")
    claim = store.scheduled_tasks.get(task.task_id)
    if (claim is None or task.lease_until is None or task.lease_until <= at
            or not claim.matches_claim(expected_attempts=task.attempts, expected_lease_until=task.lease_until)):
        raise ScheduledTaskDeferred("late_arrival_delivery_claim_changed", delay_seconds=5)
    return target


def _review_message(runtime, task, target, trace_id: str) -> str:
    """只提供这桌和到店时间，牌友看不到询局人身份、私聊原文或其他人的答复。"""
    game, recipient = target["game"], target["recipient"]
    with_question = task.task_type == LATE_ARRIVAL_QUESTION_TASK_TYPE
    approximate = target.get("arrival_is_estimate") is True
    facts = {
        "table_start": public_game_start_display(game),
        "table_type": game.requirement.get("ruleset") or game.requirement.get("game_type"),
        "candidate_arrival_time": target["arrival_at"].astimezone(now().tzinfo).strftime("%H:%M"),
        "seat_reserved": False,
    }
    if with_question:
        timing_question = "这桌有个人大概×点到，能等吗？" if approximate else "这桌有个人要到×点才来，能等吗？"
        scope = (
            "当前收件人就是这桌已经报名的牌友。直接问收件人是否接受有位顾客在这个钟点到店。"
            f"请采用「{timing_question}」的直接问法，清楚区分原定开场与到店时间。"
            "禁止写「问下这桌」「我帮你问」或声称要转问第三方，本条就是直接发给牌友的问题。"
        )
        goal = "向同桌牌友确认能否接受顾客晚到"
        status = "waiting_user"
    elif target["decision"] == "accepted":
        scope = "告知这桌接受顾客在这个钟点到店，请本人确认还来不来。不得说已留位、保证还有空位或已经改了开场。"
        goal = "告知询局人这桌接受这个到店时间"
        status = "waiting_user"
        facts["late_arrival_decision"] = "accepted"
    else:
        scope = "告知这桌目前没谈妥这个到店时间，简短收尾。不可转述或归因给某个牌友，也不承诺后续另找或留位。"
        goal = "回告询局人这个到店时间未能谈妥"
        status = "completed"
        facts["late_arrival_decision"] = target["decision"]
    if approximate:
        facts["candidate_arrival_is_estimate"] = True
        scope += "这是本人给出的到店估计，提到该钟点时说明大概或左右，不说确定会到或到此时才来。"
    boundary = "像麻将馆老板发微信，简短自然。使用给定事实，不提其他人的身份，不说已经占位、开打或改开场，不附带无关问题。"
    messages = [
        {"role": "system", "content": scope + boundary + "只输出一个 JSON 对象，唯一字段是字符串 text。"},
        {"role": "user", "content": json.dumps(facts, ensure_ascii=False)},
    ]
    system_message = UserMessage(task.conversation_id, recipient.customer_id, recipient.public_name, f"系统任务：{goal}。{scope}")
    for attempt in range(2):
        runtime.trace_recorder.record(trace_id, "late_arrival_message_prompt",
                                      {"messages": messages, "task_type": task.task_type, "attempt": attempt + 1})
        raw = runtime.llm_client.complete(messages, trace_id=trace_id, timeout_seconds=min(20, runtime.llm_timeout_seconds))
        payload = json.loads(str(raw))
        text = payload.get("text") if isinstance(payload, dict) and set(payload) == {"text"} else None
        if not isinstance(text, str) or not 0 < len(text.strip()) <= 100 or customer_visible_text_contract_violations(text):
            raise RuntimeError("late arrival message failed the text contract")
        text = text.strip()
        if with_question and approximate and not stated_arrival_estimate(
            text, sent_at=now(), estimate=target["arrival_at"],
        ):
            raise RuntimeError("late arrival question omitted the customer's estimated arrival qualification")
        action = AgentAction(goal=goal, objective_status=status, reasoning_summary=goal, reply_to_user=text)
        review = runtime.action_processor.review_customer_visible_content(
            message=system_message, trace_id=trace_id, action=action,
            review_items=[{"item_id": task.task_type, "source": "reply_to_user",
                           "recipient_id": recipient.customer_id, "text": text}],
            context_payload={
                "active_game_visible_summaries": [facts],
                "customer_visibility_contract": {
                    "audience": "private", "private_identities_allowed": False, "message_scope": scope + boundary,
                },
            },
            # 这是同桌协商的主动消息，审查仍走统一链路，但不把系统任务说明当成收件人的来话。
            turn_budget=TokenBudget(max_calls_per_turn=1), review_scope=task.task_type,
        )
        if review is not None and customer_visible_content_review_approved(review):
            return text
        rejected = review.result if review is not None else {}
        violations = set(rejected.get("violations") or ())
        if (attempt or rejected.get("needs_human") or not violations
                or violations - {"semantic_contradiction", "unverified_external_action"}):
            break
        # 只回灌受控错误类别，由生成器按原真实事实重写并重新审查。
        messages = [*messages, {"role": "user", "content": json.dumps({
            "violations": sorted(violations),
            "instruction": "重写正文，直接与当前收件人说话。删除对第三方询问、联络或执行业务的声称。",
        }, ensure_ascii=False)}]
    raise RuntimeError("late arrival message content review did not approve delivery")


def handle_late_arrival_delivery(runtime, messenger, task: ScheduledAgentTask, trace_id: str, *, room_paused=None):
    """发送已审查的同一份文本，通道回执才能建立待答引用或证明结果已回告。"""
    store = runtime.store
    target = verify_late_arrival_delivery_claim(store, task, room_paused=room_paused)
    if target is None:
        runtime.trace_recorder.record(trace_id, "late_arrival_delivery_skipped", {"task_id": task.task_id})
        return
    if not runtime.reply_self_review_enabled:
        raise ScheduledTaskDeferred("late_arrival_delivery_review_disabled")
    existing = store.resolve_message_reference(conversation_id=task.conversation_id, message_id=task.idempotency_key)
    if existing is not None:
        if not (existing.conversation_id == task.conversation_id and existing.business_ref_type == task.task_type
                and existing.business_ref_id == task.aggregate_id and existing.recipient_id == task.subject_id
                and existing.metadata.get("negotiation_id") == task.payload["negotiation_id"]
                and existing.metadata.get("delivery_id") and existing.metadata.get("trace_id")):
            raise RuntimeError("late arrival delivery reference conflicts with its task")
        store.mark_assistant_reply_delivered(task.conversation_id, trace_id=existing.metadata["trace_id"])
        return

    game, recipient = target["game"], target["recipient"]
    conversation_version = store.conversation_version(task.conversation_id)
    # 发送失败后沿用原有正文和业务键，避免网络超时、桥接已发时产生第二条消息。
    prior = next((turn for turn in reversed(store.recent_turns(task.conversation_id, limit=60))
                  if turn.role == ConversationRole.ASSISTANT
                  and turn.metadata.get("source_message_id") == task.idempotency_key), None)
    if prior is None:
        text, reply_trace = _review_message(runtime, task, target, trace_id), trace_id
    else:
        text, reply_trace = prior.content, prior.trace_id

    with runtime.input_receipt_lock(task.conversation_id):
        latest = verify_late_arrival_delivery_claim(store, task, room_paused=room_paused)
        if latest is None:
            return
        if latest != target or store.conversation_version(task.conversation_id) != conversation_version:
            raise ScheduledTaskDeferred("late_arrival_delivery_context_changed", delay_seconds=5)
        if prior is None:
            store.append_assistant_turn(task.conversation_id, text, reply_trace, metadata={
                "delivery_status": "pending_operator_send",
                "conversation_version": conversation_version,
                "source_message_id": task.idempotency_key,
            })
        if not store.claim_assistant_reply_delivery(task.conversation_id, trace_id=reply_trace):
            raise RuntimeError("late arrival reviewed reply is no longer deliverable")
    receipt = messenger.send_private_message(recipient.external_user_id, text, metadata={
        "trace_id": reply_trace,
        "draft_id": task.idempotency_key,
        "source_message_id": task.idempotency_key,
        "origin_room_id": task.payload["room_id"],
        "conversation_id": task.conversation_id,
        "message_type": task.task_type,
    })
    if not isinstance(receipt, str) or not receipt or receipt.startswith("suppressed:"):
        raise RuntimeError("late arrival delivery was not acknowledged")
    store.register_message_reference(MessageReference(
        message_id=task.idempotency_key, conversation_id=task.conversation_id,
        business_ref_type=task.task_type, business_ref_id=game.game_id, text=text, channel=recipient.channel,
        recipient_id=recipient.customer_id, recipient_name=recipient.public_name,
        metadata={"delivery_id": receipt, "trace_id": reply_trace, "negotiation_id": task.payload["negotiation_id"]},
    ))
    if receipt != task.idempotency_key:
        store.link_message_reference(conversation_id=task.conversation_id, message_id=receipt,
                                     source_message_id=task.idempotency_key)
    store.mark_assistant_reply_delivered(task.conversation_id, trace_id=reply_trace)
    runtime.trace_recorder.record(trace_id, "late_arrival_delivery_sent",
                                  {"task_id": task.task_id, "game_id": game.game_id, "delivery_id": receipt})
