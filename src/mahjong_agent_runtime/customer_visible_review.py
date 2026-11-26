"""构建并解析客户可见内容审查结果，将模型审查与后端安全合同衔接。"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from typing import Any

from .action_attribution import (
    action_attribution_contract,
    build_action_attribution_context,
    verified_action_attribution_spans,
)
from .customer_visible_contract import (
    customer_visible_action_claim_violations,
    customer_visible_seat_claim_violations,
    customer_visible_text_contract_violations,
)
from .models import AgentAction, ToolResult, UserMessage, now
from .domains.delivered_game_offer import DELIVERED_GAME_OFFER
from .domains.query_contract import game_query_strategy_contract
from .domains.context_builders.tool_results import (
    model_game_tool_evidence,
    reference_duplicate_latest_tool_results,
    reference_shared_query_contracts,
)
from .group_chat.topic_models import PUBLIC_TOPIC_FIELDS, TOPIC_INTENTS, validate_public_topic_summary


CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME = "customer_visible_content_review"


def external_action_evidence_from_tool_results(
    tool_results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """从后端 ToolResults 提取交付事实，用于最终回复审查。"""

    relevant_names = {
        "create_invite_drafts",
        "create_outbound_message_drafts",
        "update_invite_delivery_status",
        "request_late_arrival",
    }
    source_tool_names: list[str] = []
    draft_statuses: list[str] = []
    late_arrival_question_delivered = False
    for raw in tool_results:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "")
        if name not in relevant_names or not raw.get("called") or not raw.get("allowed") or raw.get("error"):
            continue
        source_tool_names.append(name)
        result = raw.get("result") if isinstance(raw.get("result"), dict) else {}
        if name == "request_late_arrival":
            late_arrival_question_delivered |= result.get("contact_started") is True
        for draft in result.get("drafts") or []:
            if isinstance(draft, dict) and draft.get("status"):
                draft_statuses.append(str(draft["status"]))
        recorded = result.get("recorded_status")
        if recorded:
            draft_statuses.append(str(recorded))
    if not source_tool_names:
        return None
    contact_started_statuses = {"sent", "confirmed", "declined", "negotiating", "no_reply"}
    return {
        "source": "backend_tool_results",
        "source_tool_names": source_tool_names,
        "draft_statuses": draft_statuses,
        "contact_started": late_arrival_question_delivered or any(status in contact_started_statuses for status in draft_statuses),
    }


def customer_visible_content_review_approved(result: ToolResult) -> bool:
    """返回审查工具是否批准了所有客户可见项目。"""

    return (
        result.name == CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME
        and result.called
        and result.allowed
        and result.error is None
        and bool(result.result.get("approved"))
    )


def customer_visible_items_for_action(action: AgentAction) -> list[dict[str, Any]]:
    """在执行前从工具参数中提取出站文本。"""

    items: list[dict[str, Any]] = []
    for call_index, call in enumerate(action.tool_calls, start=1):
        if call.name == "create_invite_drafts":
            for item_index, raw in enumerate(call.arguments.get("invitations") or [], start=1):
                if not isinstance(raw, dict):
                    continue
                text = str(raw.get("message_text") or "").strip()
                if text:
                    items.append(
                        {
                            "item_id": f"tool_calls[{call_index}].arguments.invitations[{item_index}].message_text",
                            "source": "create_invite_drafts",
                            "recipient_id": str(raw.get("customer_id") or ""),
                            "recipient_name": str(raw.get("display_name") or raw.get("customer_id") or ""),
                            "text": text,
                        }
                    )
        if call.name == "create_outbound_message_drafts":
            for item_index, raw in enumerate(call.arguments.get("drafts") or [], start=1):
                if not isinstance(raw, dict):
                    continue
                text = str(raw.get("message_text") or "").strip()
                if text:
                    items.append(
                        {
                            "item_id": f"tool_calls[{call_index}].arguments.drafts[{item_index}].message_text",
                            "source": "create_outbound_message_drafts",
                            "recipient_id": str(raw.get("recipient_id") or ""),
                            "recipient_name": str(raw.get("recipient_name") or raw.get("recipient_id") or ""),
                            "channel": str(raw.get("channel") or ""),
                            "purpose": str(raw.get("purpose") or ""),
                            "text": text,
                        }
                    )
    return items


def normalize_item_reviews(
    review: dict[str, Any], review_items: list[dict[str, Any]], *,
    attribution_context: dict[str, Any] | None = None,
    follow_up_evidence: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """规范化itemreviews，消除等价输入之间的表示差异。

"""

    raw_reviews = review.get("item_reviews")
    if not isinstance(raw_reviews, list):
        legacy_safe_text = str(review.get("final_reply") or "").strip()
        normalized: list[dict[str, Any]] = []
        for item in review_items:
            original = str(item.get("text") or "")
            suggested = legacy_safe_text or original
            violations = review.get("violations")
            normalized.append(
                normalize_review_item_contract(
                    item_id=str(item.get("item_id") or ""),
                    approved=bool(review.get("approved"))
                    and (not legacy_safe_text or legacy_safe_text == original),
                    suggested_safe_text=suggested,
                    reasoning_summary=str(review.get("reasoning_summary") or ""),
                    violations=list(violations) if isinstance(violations, list) else [],
                    action_evidence=(
                        dict(item.get("action_evidence") or {})
                        if isinstance(item.get("action_evidence"), dict)
                        else None
                    ),
                    follow_up_evidence=follow_up_evidence,
                    seat_claim_evidence=item.get("seat_claim_evidence"),
                )
            )
        return normalized

    by_item_id = {
        str(item.get("item_id") or ""): item for item in raw_reviews if isinstance(item, dict)
    }
    # 本人私聊通常只有一条。重复评审不能覆盖另一条已识别的承诺。
    unique_assessments = len(by_item_id) == len(raw_reviews) == len(review_items)
    normalized = []
    for item in review_items:
        item_id = str(item.get("item_id") or "")
        raw = by_item_id.get(item_id, {})
        suggested = str(
            raw.get("suggested_safe_text")
            or raw.get("safe_text")
            or raw.get("final_text")
            or item.get("text")
            or ""
        )
        violations = raw.get("violations")
        normalized_item = normalize_review_item_contract(
            item_id=item_id,
            approved=bool(raw.get("approved")),
            suggested_safe_text=suggested,
            reasoning_summary=str(raw.get("reasoning_summary") or ""),
            violations=list(violations) if isinstance(violations, list) else [],
            action_evidence=(
                dict(item.get("action_evidence") or {})
                if isinstance(item.get("action_evidence"), dict)
                else None
            ),
            external_action_attributions=raw.get("external_action_attributions"),
            recipient_id=str(item.get("recipient_id") or ""),
            attribution_context=attribution_context,
            follow_up_evidence=follow_up_evidence,
            follow_up_commitments=raw.get("follow_up_commitments") if unique_assessments else None,
            seat_claim_evidence=item.get("seat_claim_evidence"),
        )
        # 只绑定原样放行的正文。讨论一桌与明确提议入席是不同证据，不能当成同一种报名提议。
        offer = item.get("game_offer_candidate")
        if (unique_assessments and isinstance(offer, dict) and offer.get("game_id")
                and normalized_item["approved"] and suggested == item.get("text")):
            offer_id = raw.get("reviewed_game_offer_id")
            reference_id = raw.get("reviewed_game_reference_id")
            if offer_id == offer["game_id"] and reference_id is None:
                normalized_item["reviewed_game_offer_id"] = offer_id
            elif reference_id == offer["game_id"] and offer_id is None:
                normalized_item["reviewed_game_reference_id"] = reference_id
        normalized.append(normalized_item)
    return normalized


def normalize_review_item_contract(
    *,
    item_id: str,
    approved: bool,
    suggested_safe_text: str,
    reasoning_summary: str,
    violations: list[str],
    action_evidence: dict[str, Any] | None = None,
    external_action_attributions: Any = None,
    recipient_id: str = "",
    attribution_context: dict[str, Any] | None = None,
    follow_up_evidence: list[dict[str, Any]] | None = None,
    follow_up_commitments: Any = None,
    seat_claim_evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """规范化审查item结构化合同，消除等价输入之间的表示差异。

"""

    contract_violations = customer_visible_text_contract_violations(suggested_safe_text)
    contract_violations.extend(customer_visible_seat_claim_violations(suggested_safe_text, seat_claim_evidence or []))
    attributed_spans, attribution_violations = verified_action_attribution_spans(
        suggested_safe_text, external_action_attributions, recipient_id=recipient_id,
        attribution_context=attribution_context,
    )
    action_claim_violations = customer_visible_action_claim_violations(
        suggested_safe_text,
        action_evidence,
        attributed_claim_spans=attributed_spans,
    )
    normalized_violations = list(violations)
    normalized_violations.extend(
        f"customer_visible_contract:{violation}" for violation in contract_violations
    )
    normalized_violations.extend(
        f"customer_visible_contract:{violation}" for violation in action_claim_violations
    )
    normalized_violations.extend(
        f"customer_visible_contract:{violation}" for violation in attribution_violations
    )
    # 仅对拟放行文本执行绑定。已经拒绝的建议仍须由主 Agent 重写后重新审查。
    follow_up_violations = _follow_up_commitment_violations(
        suggested_safe_text, follow_up_commitments, follow_up_evidence,
    ) if approved and follow_up_evidence is not None else []
    normalized_violations.extend(follow_up_violations)
    return {
        "item_id": item_id,
        "approved": bool(approved) and not (
            contract_violations or action_claim_violations or attribution_violations or follow_up_violations
        ),
        "suggested_safe_text": suggested_safe_text,
        "reasoning_summary": reasoning_summary,
        "violations": normalized_violations,
        # 保留精确引用供末端再次核验，绝不持久化可被模型伪造的“已验证”标志。
        "external_action_attributions": deepcopy(external_action_attributions or []),
        **({"follow_up_commitments": deepcopy(follow_up_commitments)} if follow_up_evidence is not None else {}),
    }


def item_reviews_approved(
    item_reviews: list[dict[str, Any]], review_items: list[dict[str, Any]], *,
    attribution_context: dict[str, Any] | None = None,
    follow_up_evidence: list[dict[str, Any]] | None = None,
) -> bool:
    """要求每个项目都有一对一、未更改、无违规的批准。"""

    original_by_id = {
        str(item.get("item_id") or ""): str(item.get("text") or "") for item in review_items
    }
    evidence_by_id = {
        str(item.get("item_id") or ""): (
            dict(item.get("action_evidence") or {})
            if isinstance(item.get("action_evidence"), dict)
            else None
        )
        for item in review_items
    }
    recipients_by_id = {str(item.get("item_id") or ""): str(item.get("recipient_id") or "")
                        for item in review_items}
    seat_facts_by_id = {str(item.get("item_id") or ""): item.get("seat_claim_evidence") or [] for item in review_items}
    if len(item_reviews) != len(review_items):
        return False
    for item in item_reviews:
        item_id = str(item.get("item_id") or "")
        suggested = str(item.get("suggested_safe_text") or "")
        if item_id not in original_by_id or not bool(item.get("approved")):
            return False
        if suggested != original_by_id[item_id]:
            return False
        if (customer_visible_text_contract_violations(suggested)
                or customer_visible_seat_claim_violations(suggested, seat_facts_by_id.get(item_id, []))):
            return False
        attributed_spans, attribution_violations = verified_action_attribution_spans(
            suggested, item.get("external_action_attributions"),
            recipient_id=recipients_by_id.get(item_id, ""), attribution_context=attribution_context,
        )
        if attribution_violations or customer_visible_action_claim_violations(
            suggested, evidence_by_id.get(item_id), attributed_claim_spans=attributed_spans,
        ):
            return False
        if follow_up_evidence is not None and _follow_up_commitment_violations(
            suggested, item.get("follow_up_commitments"), follow_up_evidence,
        ):
            return False
    return True


def _current_follow_up_summaries(context_payload: dict[str, Any], message: UserMessage) -> list[dict[str, Any]]:
    """仅投影当前客户、会话和业务目标仍有效的复查证据，不复制任务私有载荷。

    旧目标的任务不能替新承诺作证；无明确目标标识时不猜测归属。这里只校验结构和
    身份边界，承诺是否与记录的目标一致仍交给模型审查，不解析客户自然语言。
    """
    window = context_payload.get("task_context_window")
    context_id = window.get("task_context_id") if isinstance(window, dict) else None
    tasks = context_payload.get("active_follow_ups")
    if not isinstance(context_id, str) or not context_id or not isinstance(tasks, list):
        return []
    summaries: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        detail = task.get("payload")
        if (not isinstance(task.get("task_id"), str) or not task["task_id"]
                or task.get("task_type") != "conversation_follow_up"
                or task.get("conversation_id") != message.conversation_id
                or task.get("subject_id") != message.sender_id
                or task.get("status") not in {"pending", "processing"}
                or not isinstance(detail, dict) or detail.get("task_context_id") != context_id):
            continue
        try:
            expires_at = datetime.fromisoformat(detail["expires_at"])
            if expires_at.tzinfo is None or expires_at <= now():
                continue
        except (KeyError, TypeError, ValueError):
            continue
        summary = {key: task[key] for key in (
            "task_id", "task_type", "status", "due_at", "aggregate_type", "aggregate_id",
        ) if isinstance(task.get(key), str)}
        summary.update({key: detail[key] for key in (
            "task_context_id", "goal", "waiting_for", "expires_at",
        ) if isinstance(detail.get(key), str)})
        summaries.append(summary)
    return summaries


_FOLLOW_UP_KINDS = {"finite_recheck", "match_notification"}


def _active_commitment_evidence(
    context_payload: dict[str, Any], message: UserMessage, follow_ups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """只提供已落库且在本会话有效的安排，不从画像、模型计划或提议推断通知能力。"""
    evidence = [{
        "ref": f"follow_up:{task['task_id']}", "kind": "finite_recheck",
        **{key: task[key] for key in ("due_at", "expires_at", "goal", "waiting_for") if key in task},
    } for task in follow_ups if _ordered_follow_up_window(task)]
    stamp = now()
    for request in context_payload.get("current_late_arrival_requests") or []:
        if request.get("status") not in {"queued", "pending"}:
            continue
        evidence.append({
            "ref": f"late_arrival:{request['game_id']}", "kind": "finite_recheck", "expires_at": request["arrival_at"],
            "planned_start_at": request["planned_start_at"], "goal": "在到店时间前收到这桌明确决定后回告本人",
        })
    for record in context_payload.get("active_waiting_demands") or []:
        if (not isinstance(record, dict) or record.get("sender_id") != message.sender_id
                or record.get("conversation_id") != message.conversation_id or record.get("status") != "active"
                or not isinstance(record.get("demand_id"), str) or not record["demand_id"]
                or not isinstance(record.get("demand"), dict)):
            continue
        try:
            expires = datetime.fromisoformat(record["expires_at"])
            if expires.tzinfo is None or expires <= stamp:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        terms = record["demand"]
        evidence.append({
            "ref": f"waiting:{record['demand_id']}", "kind": "match_notification", "expires_at": expires.isoformat(),
            "terms": {key: terms[key] for key in (
                "game_type", "game_variant", "stake", "stake_range", "cap_score", "smoke_preference",
                "time_preference", "match_remaining_seats",
            ) if terms.get(key) is not None},
        })
    return evidence


def _ordered_follow_up_window(task: dict[str, Any]) -> bool:
    """有期限的复查必须具备可解析的实际执行时刻及有效截止。"""
    try:
        due_at = datetime.fromisoformat(task["due_at"])
        expires_at = datetime.fromisoformat(task["expires_at"])
        return due_at.tzinfo is not None and expires_at.tzinfo is not None and due_at <= expires_at
    except (KeyError, TypeError, ValueError):
        return False


def _follow_up_commitment_violations(text: str, claims: Any, evidence: list[dict[str, Any]]) -> list[str]:
    """核验模型识别出的承诺原文和安排引用，禁止用自报标签替代可执行证据。"""
    invalid = "unverified_follow_up_commitment_assessment"
    unsupported = "unsupported_follow_up_commitment"
    if not isinstance(claims, list):
        return [invalid]
    kinds = {row["ref"]: row["kind"] for row in evidence}
    quotes: set[str] = set()
    violations: set[str] = set()
    for claim in claims:
        if (not isinstance(claim, dict) or set(claim) != {"quote", "kind", "arrangement_ref"}
                or not isinstance(claim["quote"], str) or not claim["quote"].strip()
                or claim["quote"] not in text or claim["quote"] in quotes
                or not isinstance(claim["kind"], str) or claim["kind"] not in _FOLLOW_UP_KINDS
                or (claim["arrangement_ref"] is not None and not isinstance(claim["arrangement_ref"], str))):
            violations.add(invalid)
            continue
        quotes.add(claim["quote"])
        if kinds.get(claim["arrangement_ref"]) != claim["kind"]:
            violations.add(unsupported)
    return sorted(violations)


def _request_scope_proposal(action: AgentAction, context_payload: dict[str, Any], message: UserMessage) -> dict[str, Any] | None:
    """只投影可与本人来源核对的范围解释；模型计划不构成授权或撤销委托的事实。"""
    scope = action.objective_state.get('request_scope')
    if not isinstance(scope, dict) or scope.get('kind') not in {'query_only', 'delegated_matching', 'specific_game_choice', 'not_applicable'}:
        return None
    known_sources = {message.message_id} if message.message_id else set()
    topic = context_payload.get('group_topic_context') or {}
    for row in topic.get('own_condition_sources', []) if isinstance(topic, dict) else []:
        if isinstance(row, dict) and isinstance(row.get('source_message_id'), str):
            known_sources.add(row['source_message_id'])
    for row in context_payload.get('recent_conversation') or []:
        if isinstance(row, dict) and row.get('role') == 'user' and row.get('sender_id') == message.sender_id:
            source = (row.get('metadata') or {}).get('source_message_id')
            if isinstance(source, str):
                known_sources.add(source)
    sources, limits = scope.get('source_message_ids'), scope.get('explicit_limits')
    return {'authority':'model_proposal_only', 'kind':scope['kind'],
        'source_message_ids':[value for value in sources if isinstance(value, str) and value in known_sources][:6] if isinstance(sources, list) else [],
        'explicit_limits':[value for value in limits if isinstance(value, str) and len(value) <= 160][:6] if isinstance(limits, list) else []}


def _current_topic_summary(context_payload: dict[str, Any]) -> dict[str, Any] | None:
    """压缩上游已核验的本人话题来源；群原文和任意自由字段不能进入审查上下文。

    本函数不重新判意图或赋予授权。结构条件只供审查核对，实际原话、任务状态及
    工具证据仍优先；调用方仅可在给本人审查回复时使用此投影。
    """
    topic = context_payload.get('group_topic_context')
    if not isinstance(topic, dict) or topic.get('intent') not in TOPIC_INTENTS:
        return None
    result = {'intent': topic['intent'], 'own_condition_sources': [], 'related_game_ids': []}
    for key in ('topic_id', 'goal_relation'):
        value = topic.get(key)
        if isinstance(value, str) and len(value) <= 200:
            result[key] = value
    game_ids = topic.get('related_game_ids')
    result['related_game_ids'] = [value for value in game_ids if isinstance(value, str) and len(value) <= 200][:20] if isinstance(game_ids, list) else []
    sources = topic.get('own_condition_sources')
    for source in sources[-6:] if isinstance(sources, list) else []:
        if (not isinstance(source, dict) or not isinstance(source.get('source_message_id'), str)
                or not 1 <= len(source['source_message_id']) <= 200):
            continue
        values = source.get('constraints')
        if not isinstance(values, dict):
            continue
        try:
            constraints = validate_public_topic_summary({key: values[key] for key in PUBLIC_TOPIC_FIELDS if key in values})
        except (TypeError, ValueError):
            continue
        if type(values.get('party_size')) is int and 1 <= values['party_size'] <= 4:
            constraints['party_size'] = values['party_size']
        for key in ('planned_start_at', 'latest_arrival_at', 'latest_wait_at', 'end_at'):
            value = values.get(key)
            try:
                if isinstance(value, str) and datetime.fromisoformat(value).tzinfo is not None:
                    constraints[key] = value
            except ValueError:
                continue
        projected = {'source_message_id': source['source_message_id'], 'constraints': constraints}
        try:
            if isinstance(source.get('sent_at'), str) and datetime.fromisoformat(source['sent_at']).tzinfo is not None:
                projected['sent_at'] = source['sent_at']
        except ValueError:
            pass
        result['own_condition_sources'].append(projected)
    return result


def _current_participation_deadline_summaries(context_payload: dict[str, Any], message: UserMessage) -> list[dict[str, Any]]:
    """仅有效的本人同目标截止任务证明到点退出责任，记忆和招募任务不作替代。

    截止时只在未成局时退出本人，因此必须保留条件和动作范围；不复制其他客户的
    截止信息、自由备注或整份调度载荷，也不把已失效任务作为未来执行保证。
    """
    window = context_payload.get('task_context_window')
    context_id = window.get('task_context_id') if isinstance(window, dict) else None
    tasks = context_payload.get('active_participation_deadlines')
    if not isinstance(context_id, str) or not context_id or not isinstance(tasks, list):
        return []
    summaries = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        detail = task.get('payload')
        if (not isinstance(task.get('task_id'), str) or not task['task_id']
                or task.get('task_type') != 'participation_deadline'
                or task.get('aggregate_type') != 'game_participant'
                or task.get('conversation_id') != message.conversation_id
                or task.get('subject_id') != message.sender_id
                or task.get('status') not in {'pending', 'processing'}
                or not isinstance(detail, dict) or detail.get('customer_id') != message.sender_id
                or detail.get('task_context_id') != context_id
                # 业务效果可能已提交，而调度器尚未确认完成；processing 不等于仍待执行。
                or detail.get('execution_outcome')
                or detail.get('condition') != 'game_not_ready' or detail.get('effect') != 'withdraw_self'
                or not isinstance(detail.get('game_id'), str) or not detail['game_id']):
            continue
        try:
            deadline = datetime.fromisoformat(detail['deadline_at'])
            if deadline.tzinfo is None or deadline <= now() or deadline != datetime.fromisoformat(task['due_at']):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        summaries.append({'task_id': task['task_id'], 'status': task['status'], 'game_id': detail['game_id'],
            'task_context_id': context_id, 'deadline_at': detail['deadline_at'],
            'condition': detail['condition'], 'effect': detail['effect']})
    return summaries


def _tool_evidence_without_response_audit(raw: Any) -> Any:
    """为可见文本审查复制业务事实，剔除主规划纠错的身份、事实来源与技术诊断。

    只处理工具结果的确定结构，不递归改写业务值或客户原话；原 trace 和主模型
    回灌证据保持不变，审查仍能看到成功、拒绝、业务结果和修复要求。
    """
    projected = deepcopy(raw)
    if not isinstance(projected, dict):
        return projected
    for key in ("response_source", "completion_diagnostics", "reference_fact_sources", "validation_error"):
        projected.pop(key, None)
    result = projected.get("result")
    if "result" in projected:
        result = projected["result"] = model_game_tool_evidence(projected.get("name"), result)
    if isinstance(result, dict):
        for key in ("response_source", "completion_diagnostics", "reference_fact_sources", "validation_error"):
            result.pop(key, None)
    return projected


def _review_conversation_without_response_audit(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """工具历史可能是 JSON 字符串，须按结构投影；用户原文不做关键词替换。"""
    projected = []
    for turn in turns:
        item = deepcopy(turn)
        if isinstance(item, dict) and item.get("role") == "tool":
            content = item.get("content")
            if isinstance(content, str):
                try:
                    evidence = json.loads(content)
                except json.JSONDecodeError:
                    # 无法验证的工具原文不应成为审查的事实源或内部诊断旁路。
                    item["content"] = "[]"
                else:
                    if isinstance(evidence, list):
                        item["content"] = json.dumps(
                            [_tool_evidence_without_response_audit(raw) for raw in evidence], ensure_ascii=False,
                        )
                    elif isinstance(evidence, dict):
                        item["content"] = json.dumps(_tool_evidence_without_response_audit(evidence), ensure_ascii=False)
            elif isinstance(content, list):
                item["content"] = [_tool_evidence_without_response_audit(raw) for raw in content]
            elif isinstance(content, dict):
                item["content"] = _tool_evidence_without_response_audit(content)
        projected.append(item)
    return projected


def _reply_turn_state_semantics(status: str) -> dict[str, Any] | None:
    """只投影当前枚举的协议要求，不从话术、标点或未成局状态推断是否真正待答。"""
    answer_requirement = {"waiting_user": "required", "completed": "absent"}.get(status)
    if answer_requirement is None:
        return None
    return {"scope": "current_conversation_turn", "current_user_answer_requirement": answer_requirement,
        "business_goal_completion_implied": False, "mismatch_violation": "semantic_contradiction"}


def _review_seat_count_evidence(
    visible_games: list[dict[str, Any]], turn_tool_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """按实际局 ID 收集最近的结构化席位，查询条件和自由文本均不作为人数来源。"""

    facts_by_game: dict[str, dict[str, Any]] = {}

    def record(snapshot: Any) -> None:
        """仅接受席位数加总有效的局快照，同一时刻的不同局仍各自保留。"""
        if not isinstance(snapshot, dict):
            return
        game_id = snapshot.get("game_id")
        if not isinstance(game_id, str) or not game_id:
            return
        if snapshot.get("status") in ("cancelled", "finished"):
            facts_by_game.pop(game_id, None)
            return
        seats = snapshot.get("seat_summary")
        if not isinstance(seats, dict):
            return
        claimed, remaining, total = (seats.get(key) for key in ("claimed_seats", "remaining_seats", "seats_total"))
        if (any(type(count) is not int or count < 0 for count in (claimed, remaining, total)) or
                total == 0 or claimed + remaining != total):
            return
        previous = facts_by_game.get(game_id, {})
        fact = {
            "game_id": game_id,
            "planned_start_at": snapshot.get("planned_start_at", previous.get("planned_start_at")),
            "claimed_seats": claimed,
            "remaining_seats": remaining,
            "seats_total": total,
            "remaining_seats_display": f"还缺{remaining}人" if remaining else "满座",
        }
        if total == 4 and 1 <= remaining <= 3:
            fact["four_seat_shorthand"] = ("371 / 三缺一", "272 / 二缺二", "173 / 一缺三")[remaining - 1]
        facts_by_game[game_id] = fact

    for game in visible_games:
        record(game)
    for tool in turn_tool_evidence:
        if not isinstance(tool, dict) or not tool.get("called") or not tool.get("allowed") or tool.get("error"):
            continue
        result = tool.get("result")
        if not isinstance(result, dict):
            continue
        continuation = result.get("continuation")
        if isinstance(continuation, dict):
            record(continuation.get("authoritative_facts"))
        record(result.get("game"))
        if tool.get("name") == "search_current_games":
            for key in ("matches", "alternatives", "nearby_options"):
                for option in result.get(key) or []:
                    if isinstance(option, dict):
                        record(option.get("game"))
    return list(facts_by_game.values())


def _review_game_availability_evidence(
    turn_tool_evidence: list[dict[str, Any]], seat_count_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """分开投影检索有没有返回选项，以及哪张具体桌的有效局况确实为零席。"""

    queries = [tool for tool in turn_tool_evidence
               if isinstance(tool, dict) and tool.get("name") == "search_current_games"]
    if not queries and not seat_count_evidence:
        return {}
    evidence: dict[str, Any] = {
        "zero_seat_game_ids": [game["game_id"] for game in seat_count_evidence if game["remaining_seats"] == 0],
    }
    searches_by_scope = {}
    for query in queries:
        result = query.get("result")
        outcome = _review_search_outcome(query)
        evidence["latest_search"] = outcome
        scope = result.get("lookup_scope") if isinstance(result, dict) else None
        if scope in ("specific_table", "personal"):
            searches_by_scope[scope] = {**outcome, "call_id": query.get("call_id")}
    if searches_by_scope:
        evidence["searches_by_scope"] = searches_by_scope
    return evidence


def _review_search_outcome(query: dict[str, Any]) -> dict[str, Any]:
    """只从本次实际查局提取命中数和临近选项，失败或未执行的调用保留为未知。"""
    result = query.get("result")
    matches = result.get("matches") if isinstance(result, dict) else None
    if not (query.get("called") and query.get("allowed") and not query.get("error") and isinstance(matches, list)):
        return {"status": "unknown"}
    timing_options = [
        {"game_id": option["game"].get("game_id"), "planned_start_at": option["game"].get("planned_start_at")}
        for key in ("matches", "alternatives", "nearby_options")
        for option in result.get(key) or ()
        if isinstance(option, dict) and option.get("requires_timing_confirmation") is True
        and isinstance(option.get("game"), dict)
    ]
    empty_search_filters = _review_empty_personal_search_filters(result) if not matches else {}
    return {"status": "completed", "matched_count": len(matches),
            **({"submitted_filters_without_results": empty_search_filters} if empty_search_filters else {}),
            **({"table_reference_evidence": result["table_reference_evidence"]}
               if result.get("lookup_scope") == "specific_table" and result.get("table_reference_evidence") else {}),
            **({"options_requiring_timing_confirmation": timing_options} if timing_options else {})}


def _review_empty_personal_search_filters(result: dict[str, Any]) -> dict[str, Any]:
    """只整理完整个人查局为空的实查过滤值，避免把缺数据、备选或截断当成空。"""
    if (result.get("lookup_scope") != "personal" or result.get("selection_truncated")
            or any(result.get(key) not in (None, []) for key in ("alternatives", "nearby_options"))
            or result.get("match_count", 0) != 0):
        return {}
    contract = result.get("customer_reply_contract")
    if not isinstance(contract, dict):
        return {}
    scope, semantics = contract.get("evidence_scope"), contract.get("search_result_semantics")
    if (not isinstance(scope, dict) or not isinstance(semantics, dict)
            or scope.get("kind") != "submitted_filters_only"
            or scope.get("supports_narrower_empty_claim") is not False
            or semantics.get("status") != "no_actionable_match"
            or semantics.get("backend_retrieval_policy_applied") is not True
            or any(type(semantics.get(key)) is not int or semantics[key] != 0
                   for key in ("returned_game_count", "actionable_match_count", "alternative_count"))):
        return {}
    requirement = result.get("requirement")
    applied = scope.get("applied_filter_fields")
    if (not isinstance(requirement, dict) or not isinstance(applied, list)
            or any(not isinstance(field, str) for field in applied)):
        return {}
    # 过滤协议包含座位数路径，平铺该路径只传真实座位数，不携带请求方身份。
    party = requirement.get("requesting_party")
    values = ({**requirement, "requesting_party.seat_count": party.get("seat_count")}
              if isinstance(party, dict) else requirement)
    if any(values.get(field) is None for field in applied):
        return {}
    return {field: deepcopy(values[field]) for field in applied}


def _blocked_seat_registration(
    turn_tool_evidence: list[dict[str, Any]], memberships: list[dict[str, Any]],
) -> dict[str, str] | None:
    """只从同轮实际工具拒绝恢复未执行的占席；已确认席位或混合执行仍用原审查。"""
    participation = [row for row in turn_tool_evidence
                     if isinstance(row, dict) and row.get("name") in {"join_game", "record_candidate_reply"}]
    if any(row.get("called") is True and row.get("allowed") is True and not row.get("error") for row in participation):
        return None
    denied_tables = set()
    for row in participation:
        result = row.get("result")
        validation = result.get("validation_error") if isinstance(result, dict) else None
        if (row.get("called") is False and row.get("allowed") is False and row.get("error")
                and isinstance(validation, dict) and validation.get("code") == "venue_write_not_authorized"
                and validation.get("scope") == "participate" and isinstance(game_id := validation.get("game_id"), str)
                and game_id):
            denied_tables.add(game_id)
    if len(denied_tables) != 1:
        return None
    game_id = next(iter(denied_tables))
    if any(isinstance(row, dict) and row.get("game_id") == game_id and row.get("participation_already_recorded") is True
           for row in memberships):
        return None
    return {"game_id": game_id, "write_status": "not_executed", "authorization_status": "not_verified"}


def build_reply_self_review_payload(
    *,
    message: UserMessage,
    action: AgentAction,
    review_items: list[dict[str, Any]],
    context_payload: dict[str, Any],
    review_scope: str,
) -> dict[str, Any]:
    """投影审查所需的原话、执行证据与当前需求解释，避免把旧查询当成新约束。

    主模型解释仅帮助审查理解回复目的，不替代用户原话或数据库事实；不在这里
    解析自然语言、决定查询范围或发起运营动作。
    """

    turn_tool_evidence = (
        context_payload.get("turn_tool_evidence")
        or context_payload.get("previous_tool_results")
        or []
    )
    visible_games = context_payload.get("active_game_visible_summaries") or []
    seat_count_evidence = _review_seat_count_evidence(visible_games, turn_tool_evidence)
    current_customer_reply = review_scope == "reply_to_user" and all(
        item.get("recipient_id") == message.sender_id for item in review_items
    )
    metadata = message.metadata if isinstance(message.metadata, dict) else {}
    game_availability_evidence = (
        _review_game_availability_evidence(turn_tool_evidence, seat_count_evidence) if current_customer_reply else {}
    )
    has_submitted_empty_search = any(
        search.get("submitted_filters_without_results")
        for search in (game_availability_evidence.get("latest_search", {}),
                       *game_availability_evidence.get("searches_by_scope", {}).values())
    )
    private_customer_reply = (
        current_customer_reply and bool(review_items) and not any(item.get("channel") == "group" for item in review_items)
        and metadata.get("source") != "group" and metadata.get("is_room") is not True and not metadata.get("room_id")
        and "reply_constraints" not in metadata
    )
    has_game_offer_candidate = private_customer_reply and any(item.get("game_offer_candidate") for item in review_items)
    blocked_registration = (_blocked_seat_registration(
        context_payload.get("turn_tool_evidence") or [], context_payload.get("sender_active_game_memberships") or [],
    ) if private_customer_reply else None)
    follow_ups = _current_follow_up_summaries(context_payload, message) if current_customer_reply else []
    # 私聊历史只参加给本人的私聊审查，工具文字和群回复可能面向其他人。
    owner_conversation_context = (
        context_payload.get("owner_conversation_context")
        if current_customer_reply and review_items and not any(item.get("channel") == "group" for item in review_items)
        else None
    )
    allow_known_echo_omission = current_customer_reply
    objective_state = action.objective_state
    payload = {
        "current_message": message.to_dict(),
        **({DELIVERED_GAME_OFFER: context_payload[DELIVERED_GAME_OFFER]}
           if private_customer_reply and context_payload.get(DELIVERED_GAME_OFFER) else {}),
        **({"current_late_arrival_requests": context_payload["current_late_arrival_requests"]}
           if private_customer_reply and context_payload.get("current_late_arrival_requests") else {}),
        **({"current_sender_invites": context_payload["current_sender_invites"]}
           if current_customer_reply and review_items
           and not any(item.get("channel") == "group" for item in review_items)
           and context_payload.get("current_sender_invites") else {}),
        # 仅在给本人的私聊回复中核对本人是否占席。空列表也保留，避免把已读取的无席位误作缺证据。
        **({"sender_active_game_memberships": [
                {key: membership[key] for key in ("game_id", "participant_status", "participation_already_recorded")}
                for membership in context_payload["sender_active_game_memberships"]
            ]}
           if private_customer_reply and "sender_active_game_memberships" in context_payload else {}),
        "game_query_strategy_contract": game_query_strategy_contract(),
        "customer_visibility_contract": context_payload.get("customer_visibility_contract") or {},
        "sender_profile": context_payload.get("sender_profile"),
        "sender_relationships": context_payload.get("sender_relationships") or [],
        "active_games": context_payload.get("active_games") or [],
        "active_follow_up_summaries": follow_ups,
        **({"follow_up_commitment_evidence": _active_commitment_evidence(context_payload, message, follow_ups)}
           if private_customer_reply else {}),
        "group_topic_context": _current_topic_summary(context_payload) if current_customer_reply else None,
        "active_participation_deadline_summaries": (
            _current_participation_deadline_summaries(context_payload, message) if current_customer_reply else []
        ),
        "active_game_visible_summaries": visible_games,
        "previous_tool_results": [
            _tool_evidence_without_response_audit(raw) for raw in context_payload.get("previous_tool_results") or []
        ],
        "turn_tool_evidence": [_tool_evidence_without_response_audit(raw) for raw in turn_tool_evidence],
        "recent_conversation_tail": _review_conversation_without_response_audit(
            list(context_payload.get("recent_conversation") or [])[-8:]
        ),
        **({"owner_conversation_context": owner_conversation_context} if owner_conversation_context else {}),
        "action_boundary": {
            "objective_status": action.objective_status,
            "turn_state_semantics": _reply_turn_state_semantics(action.objective_status) if current_customer_reply else None,
            "needs_human": action.needs_human,
            "tool_call_names": [call.name for call in action.tool_calls],
            "has_reply_to_user": bool(action.reply_to_user.strip()),
            "goal": action.goal,
            "request_scope": _request_scope_proposal(action, context_payload, message) if current_customer_reply else None,
            "reply_scope": action.objective_state.get("reply_scope") or {},
            "known_facts": objective_state.get("known_facts") if isinstance(objective_state.get("known_facts"), dict) else {},
            "missing_facts": objective_state.get("missing_facts") if isinstance(objective_state.get("missing_facts"), list) else [],
            "fact_assertions": objective_state.get("fact_assertions") if isinstance(objective_state.get("fact_assertions"), list) else [],
            "customer_visible_item_count": len(review_items),
        },
        "customer_service_review_contract": {
            "offer_is_not_execution": True,
            "proposal_scope_is_whole_utterance": True,
            "separate_assertions_need_independent_evidence": True,
            "commitment_evidence_requires_accepted_responsibility": True,
            "customer_acceptance_condition_differs_from_event_trigger": True,
            "communication_act_review_order": [
                "先保留整句疑问、条件、否定及并列从句的作用域",
                "再区分未承接的服务提议、已承接责任、声称已执行",
                "最后仅按实际承接或声称的内容核验安排与执行证据",
            ],
            "proposed_next_steps_are_not_authoritative": True,
            "existing_delegation_must_not_be_reset_to_proposal": True,
            "missing_execution_requires_replan_not_textual_acknowledgement": True,
            "waiting_memory_is_not_executable_deadline": True,
            "recruitment_window_is_not_participation_deadline": True,
            "latest_user_constraints_override_query_filters": True,
            "allow_known_condition_omission_without_game_option": allow_known_echo_omission,
            "read_only_lookup_is_not_durable_follow_up": True,
            "open_game_is_not_durable_follow_up": True,
            "short_acknowledgement_is_not_future_notification": True,
            "commitment_requires_matching_mechanism_scope": True,
            "conversation_follow_up_capability": (
                "有效同目标复查任务可在期限内重新判断，并按实际进展准备给同一客户的反馈草稿；"
                "无必要进展可静默。这支持任务范围内有进展再反馈，不证明已发送、保证有结果或无限跟进。"
            ),
            "current_clock_is_not_availability_evidence": True,
            "model_missing_facts_are_not_authority": True,
            "known_constraints_require_source_recheck": True,
            "profile_does_not_override_inventory_scope": True,
            "reply_scope_requires_actual_request_support": True,
        },
        "review_scope": review_scope,
        "review_items": review_items,
        "external_action_attribution_context": build_action_attribution_context(
            message, context_payload,
            enabled=current_customer_reply and not any(
                item.get("channel") == "group" for item in review_items
            ),
        ),
        "external_action_attribution_contract": action_attribution_contract(),
        "review_goal": (
            "一次性审查 review_items 中的客户可见文本是否泄露系统信息、后台流程、其他用户信息、"
            "未发生动作、结论超出证据范围，或与权威工具结果/客户可见局摘要、当前沟通意图及状态发生矛盾；"
            "主 Agent 已直接生成最终回复，不做业务规划，不决定工具调用，不另作话术润色。"
        ),
        "semantic_fidelity_contract": {
            **({"seat_count_evidence": seat_count_evidence} if seat_count_evidence else {}),
            **({"seat_registration_evidence": blocked_registration} if blocked_registration else {}),
            **({
                "game_availability_evidence": game_availability_evidence,
                "availability_claim_interpretation": (
                    "先按整段对话分清客户问有无任何合适空位，还是问某张已经点名的桌。"
                    "客户用‘那桌’、编号或时段及档位指具体桌，答复承接其指称并肯定式说这桌‘已经没位了’时，"
                    "正文是在说这张桌确实存在且现无空席。不得把这项桌况换读成查询范围内没找到。"
                    "例如问‘八点半那个还有位吗’，回答‘八点半那个现在没位了’，是具体桌席位声明；"
                    "回答‘八点半没查到符合的桌’才只是查询未返回。"
                    "只有同一张桌的结构化零席事实支持前一句；即使支持更窄查询范围为空，也不支持具体桌已满。"
                    "泛问这会儿有没有位置时，可自然说暂时没找到适合上的位置，无须盘点全店或列举查询条件。"
                ),
            } if game_availability_evidence else {}),
            "allow_known_user_echo_omission": allow_known_echo_omission,
            # 事实权威性与本次应答范围分开；完整摘要不能把单项问题升级为重新推销整桌。
            "response_scope_before_anchor_preservation": True,
            "visible_facts_are_not_mandatory_response_facts": True,
            "progress_queries_require_all_decision_anchors": True,
            "narrow_reply_must_not_add_unrequested_status_or_confirmation": True,
            "response_scope_source": (
                "先核对真实 current_message 与相关用户上下文，再确定本次需要回答或合法提案的信息；"
                "action_boundary.goal/reply_scope、source_text 和审查建议本身不能扩大用户请求。"
            ),
            "source_order": [
                "成功的 turn_tool_evidence 是本轮完整工具事实的最高优先级来源",
                "previous_tool_results 只表示最近一步反馈，用于定位最新错误或续办要求",
                "active_game_visible_summaries 是可对当前客户披露的局况事实来源",
                "该局的 seat_summary.claimed_seats 和 remaining_seats 优先于旧公开摘要中的人数、缺口和短码。与实时座位一致的回复不能因旧摘要不同而被拒绝",
                "review_items[].source_text 是话术改写前的语义基线",
                "review_items[].text 是待发送文本，不得反向覆盖上述事实",
                "当前需求以用户最新有效表达及同一任务上下文为准；工具参数只证明当时查了什么，不冻结客户后续放宽的条件；action_boundary 的模型解释须与原话核对",
            ],
            "rules": [
                "待审文本不得反转存在/不存在、成功/失败、已确认/未确认、已发送/未发送等事实极性",
                *(("seat_registration_evidence 记载本轮对这桌占席的实际拒绝，仅说明这次没有登记，"
                   "不证明顾客未提出请求，也不判定『先记上』与旁边的等候条件是否独立。"
                   "顾客意愿及能否等仍按完整原话核对。明确说明这次未确认席位已经回应占席请求，"
                   "可补充真实局况或只澄清是否即使无法等也要报名，不因未完成该受阻写入判 semantic_fact_loss 或 delegation_not_advanced。"
                   "若原话已明确分别提出报名和等待，不重问已清楚的报名。不得让任何改写说已记上、已留位或已答应等待。",)
                  if blocked_registration else ()),
                *(("game_availability_evidence.searches_by_scope.specific_table.table_reference_evidence 显示查桌用的缺位线索与候选余位冲突。"
                   "没有已核实的同桌引用或其他充分线索时，当前回信不能直接把冲突候选认成顾客所指。"
                   "即使候选玩法和档位全部属实，也要就所指作条件式说明或追问，同时答复已经查明的其他问题；无条件认桌以 semantic_contradiction 拒绝。",)
                  if game_availability_evidence.get("searches_by_scope", {}).get("specific_table", {}).get("table_reference_evidence") else ()),
                "结论强度不能超过证据：局池查询只证明已提交条件下的登记局况；空条件查询且客户需求未知时，不支持有无适合该客户的局，也不支持候选人有无的断言。主 Agent 原稿也可能过度推断，不能因改写忠于原稿就放行",
                "找局策略遵守 game_query_strategy_contract，结合原话、上下文和真实工具证据判断是否合理，不以固定追问或唯一语义路径拒绝；未知个人适配仍不得伪造。越过证据范围使用 unsupported_conclusion，approved=false，并提供不新增事实的安全文本",
                "empty_personal_matching_default 是主 Agent 的服务规划策略。审查空结果正文时核对真实查询范围和事实。仅未附加服务提议不构成 semantic_fact_loss。用户明确要求的答复、待决事项或实际已获委托仍按各自证据和完成要求审查。",
                "先按 request_scope_contract 核对本人原话、限制及同目标委托，再按 delegation_acceptance.progress_review 核验实际执行阶段，最后判断剩余提议或承诺。action_boundary.request_scope仅是有来源的模型解释，新goal、reply_scope或空查询不能撤销既有委托。有效完整委托且事实足够，但尚无相应持久承接及实际推进，却停在重新询问是否开始同一服务，以 delegation_not_advanced 拒绝并要求重新规划；短确认也不能掩盖缺执行。已建立同目标待组局并完成当前可执行找人步骤是真实进展，候选为空不撤销执行；工具允许本轮结束时，另征询持续复查或个人通知且整句仍待客户接受，不是重置原委托，不要求预先调度。实际承担未来责任仍须有效对应机制，不得借提议放过无条件通知、独立已发送声明、未完成的明确要求或工具续办义务。真正缺项、具体局参加确认、只找现成或明确撤销仍保留；看板未配置不能谎称已安排发布。",
                *(("通过 game_availability_evidence.latest_search.options_requiring_timing_confirmation 逐桌对照原定开场。"
                   "若介绍这些临近桌，须保留顾客赶不赶得上的未知；已有空席不代表会晚开或同桌愿意等候。"
                   "可直接省略这些备选；无真实本人或同桌确认的肯定保证，以 unsupported_conclusion 拒绝。",)
                  if game_availability_evidence.get("latest_search", {}).get("options_requiring_timing_confirmation") else ()),
                "明确问有无现成局且已按有效条件完整查过时，可以直接说当前范围内没有可搭的现成桌。保留会改变结论的时间、玩法和档位，允许对话自然承接已知条件；不强制使用‘没查到’或报告查询过程。局况仍待核实、搜索未完成或失败时，不能称为当前无桌。提议组桌仍是等本人决定，不表示已开始。",
                "按同一局最新的真实查询结果或 active_game_visible_summaries.public_requirement 核对 start_time_kind 与原定时间。"
                "固定开场的桌可以说‘原定七点半，人还没齐，到点还缺人得再商量’，也可以说明晚到不保证等候。"
                "‘七点半那桌……不过人齐才开’把时间当作桌子的称呼，随后把开场方式说成了人齐开。"
                "已有同桌固定时间且无已核实的改期时，这类把原定开场改说成人齐开的正文以 semantic_contradiction 拒绝。"
                "实际是人齐开的桌可以这样描述。仅有查询筛选条件或另一桌的摘要不能给所问的桌定开场方式。",
                *(("分别核对未查到符合条件的现成局、确定存在且满座的具体桌。game_availability_evidence.latest_search.matched_count=0 只证明本次查询没返回匹配，不证明某桌存在、已满或先有位后被占。正文若把具体桌说成没位或满座，必须由这张桌自己的 zero_seat_game_ids 和 seat_count_evidence 支持；别桌、历史话术、房间无空、失败查询均不能代替。缺证据以 unsupported_conclusion 拒绝。自然转述、询问或当前范围内没查到合适桌不等于断言具体桌满座。",)
                  if game_availability_evidence else ()),
                *(("submitted_filters_without_results 是这一次成功检索实际应用的条件，结果未返回正选或备选。先把正文里的结论跟这组条件对照。"
                   "‘按你说的条件，暂时没找到能搭的现成桌’与‘暂时没找到符合这些条件的现成桌’都是限定结论；范围对应且其他事实正确时，两种表达均通过。"
                   "不能把它读成‘目前所有时段都没有这种桌’，也不要求把上下文已清楚的条件逐项念一遍。"
                   "这份凭据不补足未知玩法，也不授予建局、邀约或预约权限。"
                   "包间是否可用及其时段另核对查房回执；空的查局回执不支持包间可用、已有预约或可以保证开打。",)
                  if has_submitted_empty_search else ()),
                "客户明确盘点全店时，画像不改变其范围；过滤查询不能支撑全店结论。显式限定的盘点保留限制，范围不符使用 unsupported_conclusion 回灌纠正查询，不因主模型自称全局目标而放行",
                "只有真实查询的 customer_reply_contract.evidence_scope.supports_narrower_empty_claim=true，且本次成功查询确为空时，才可依据后端已证实的未过滤可用局池为空推断更窄"
                + ("检索范围为空。该权限只覆盖没有查到可供选择的局，不覆盖已指明具体桌的席位或存在情况。"
                   if game_availability_evidence else "范围为空。")
                + "该字段为 false 或缺失时，空列表只证明本次检索未返回；评分阈值可能使新增过滤后出现结果，不能凭条件包含关系自动推断未查子范围为空。本轮是否显式呈现结果按 game_query_strategy_contract.query_result_reporting 审查；非空、截断、不完整或失败结果均不能当空集，不得外推未知个人适配、潜在客户或房态",
                "missing_facts 和 reply_scope 是待核解释；原话、任务事实或可靠唯一画像仍确认的条件，不能因本条未重复或中间查房而重问。无真实冲突的重复澄清以 known_constraint_reask 拒绝",
                "当前目标由真实原话和相关上下文确定；非业务交流附带招揽、业务提醒或业务回流尾句属于 unsolicited_goal_expansion，即使无工具也须拒绝，安全改写可去掉未请求背景；正常找局咨询的服务提议仍允许",
                "客户补发短词修正本人引用的原话时，核对回复是否自然承接修正后的意思。简短确认或顺着改后的意思聊均可，无须复述错词。只有引用与新词清楚支持更正、回复却遗漏或误读这层意思时，以 semantic_fact_loss 反馈。",
                "明确名单询问可使用 active_game_visible_summaries.public_participant_names 的后端私聊授权白名单，不把已授权公开值当私有备注；字段缺失时不能从全局画像/内部参与记录补名，角色/邀约状态/关系仍保密，不能顺带招揽或确认",
                "未知个人适配不能先宣判有无适合的局再补问；缺少 supports_narrower_empty_claim=true 时只支持本次有效条件下的查询结果，不能自动宣判未查范围为空。不得因相关服务征询或必要追问未复述空结果就拒绝，按 game_query_strategy_contract.query_result_reporting 核对上下文合理性；不得扩大查询范围、冒充个人适配或由登记局为空推断没有潜在客户；active_games 仅是客户相关局，不是全店库存",
                "不得篡改仍在表达的业务事实；要区分必需决策信息和用户已知条件的冗余回声",
                *(('分桌回答核对 assigned_room.access_window_started。为 false 时，把预约开始时间明确关联到用房；单说牌局几点开不能给随后「到了直接过去」加上未来条件。能说明稍后用房，不补写没有证据的大厅、前台或其他等候位置，不保证准时开打。',)
                  if private_customer_reply and any(game.get("assigned_room") for game in visible_games if isinstance(game, dict)) else ()),
                "回复断言一张已经登记的桌目前缺几人时，核对同一 game_id 最新成功结果的 seat_summary.remaining_seats。"
                "若整句明确在征询一项未定的方案，『你们三个人来，那还要找一位』可以是方案中的人数推算，不能当成当前席位回执。"
                "条件句之外另行断言已经登记、已留位或已有缺口时，仍单独核对。真实现桌的缺口以 seat_count_evidence 的 remaining_seats_display 为准。"
                "claimed_seats 是已占位数，不能拿来判断缺几人。173/一缺三=缺3，272/二缺二=缺2，371/三缺一=缺1。"
                "同局的结构化席位优先于自由填写的 user_visible_summary、历史话术和查询需求。冲突以 semantic_contradiction 拒绝。"
                "已有准确的缺口不转成另一种说法，修正不确定时先保留其他有证据的桌况。",
                "allow_known_user_echo_omission=true 时，包括真实局选项在内的当前客户回复都默认省去已明确且未变化的条件，不能仅因未复述档位、烟况等字段判 semantic_fact_loss",
                "按实际收件人的已知范围判断省略，不能把发起人的条件套给另一候选人；对他尚未知、真正不同或新变化且影响决定的条件仍须说明，不能以简短掩盖差异",
                "介绍选项或答局况时须保留真实新变化、条件差异、辨认所指局必要的锚点、客户明确要求核对的信息和本目标仍需的决策问题；原稿或公开摘要包含某字段不产生逐项复述义务",
                "偶尔复述准确条件是风格瑕疵，不因此判事实、隐私或执行错误，不阻断；但真实矛盾、无依据结论、权限与隐私风险仍须拒绝",
                "本轮状态必须满足 action_boundary.turn_state_semantics：真实待答问题或选择须 waiting_user，无待答的纯告知须 completed；本轮结束不等于原业务完成。按完整语义核验，不凭问号、礼貌尾句或未成局推断，错配以 semantic_contradiction 拒绝；自然表达不绑定唯一结尾。",
                "status_query_reply_contract 的 all_decision_anchors 先核对其 applicability，再区分已知条件与真正新信息；已知且未变化的字段不须复读，识别新选项或理解变化所必需的锚点仍须保留",
                "仅名单及不表示局况进展的窄属性可省未问邻接背景，省去无关时间、缺口等不构成 semantic_fact_loss；原稿或审查建议曾加入背景不产生保留义务。为识别用户正在问的对象确实必要的信息仍须保留，不能机械按字段删减",
                "可披露事实不是本轮必答清单；名单或窄属性待审文本擅自附带未请求的局况、进展、招揽或参加确认，且不是理解所问对象必要的信息时，必须以 unsolicited_goal_expansion 拒绝，安全改写仅保留所问信息。背景真实、原稿含背景或主模型自述扩大后的目标都不能豁免",
                "成功查询返回非空 matches 时，待审文本不得声称没有匹配结果",
                "审查发现语义不保真时 approved=false，并在 violations 标注 semantic_contradiction 或 semantic_fact_loss",
                "Agent 声称已问、已联系、已发送或已通知必须有后端 contact_started=true 证据；字段缺失同样没有外发证明。客户本人做过的工作不冒充 Agent 动作，可按 external_action_attribution_contract 同次返回严格原话引用，后端核验主体和来源后仅豁免相应分句；不靠话题标签或模型 approved 放行。",
                *(('回复申请晚到的本人时，先区分单纯询局与同桌有条件报名。给出有效到店时间并同意条件满足就参加，已授权核实。'
                   '查到原桌且能核实时，缺少同桌有效征询就以 delegation_not_advanced 退回办理；无机制的「我给你问一下」不是征询顾客要不要。'
                   '已安排的状态以 late_arrival_request 为准。queued 支持告知能否晚到还待确认，不支持声称已联系；等牌友的答复可以结束本轮。'
                   '用 remaining_play_minutes 核对本人时长，不拿原桌全程与晚到者可打时长比较。没有同桌同意不能承诺等候或留位。',)
                  if private_customer_reply else ()),
                "search_customers 只查潜在人选意向，零结果也不能说没问着人；顾客给出的等候截止不等于店家已安排继续询问、等回信或届时反馈。",
                "按完整句子的疑问或条件作用域先识别各独立沟通行为，再分类为服务提议、已承接责任、声称执行，最后核验证据；不能先按动作内容索要安排再倒推它已经承诺",
                "整句征询是否需要服务时，逗号后的打听、关注或将来通知仍可处在提议范围；尚待客户接受的条件不等于有消息就通知等业务事件触发条件；后者不自动豁免已接手责任。尚未承接不要求后续安排，不适用 unsupported_follow_up_commitment，不能因没有任务或含持续动作就拒绝提议",
                "本轮真实建局、找候选人或准备邀约可正常短承接，不等于承担持续复查或未来回信。仅已承接持续责任才需覆盖当前客户、目标、有效期及所承诺动作的真实机制，缺少时使用 unsupported_follow_up_commitment。开放局、一次查询、已有草稿或任意任务都不自动支持持续通知；无安排不能反向证明征询已变成承诺",
                "active_follow_up_summaries 中的有效同目标任务支持有限复查，并可按实际进展准备给同一客户的反馈草稿，覆盖该范围内有进展再反馈的承接；真实有效候补登记支持相应匹配事件后的通知准备；招募窗口任务只支持届时开启招募，不自动支持通知发起人。有限安排不证明无限跟进、一定凑齐或已经外发",
                "到点未成就退出本人须有同客户、同目标、同局有效 participation_deadline 工具结果或 active_participation_deadline_summaries；核对 deadline_at、condition=game_not_ready、effect=withdraw_self。latest_wait_at记忆、普通复查或招募开始任务不能替代；任务只支持到点未成退出本人，不支持取消整局、退出同行或已成局仍退出。缺证据以 unsupported_participation_deadline 拒绝并要求重新规划，不能删去承诺换短确认掩盖尚未完成的明确截止要求。",
                "独立的已执行声明或无条件承诺分别核验，不因邻接提议放行；声称联系/发送/通知须真实外发事实，实际接手有安排但尚未发送不能误判成虚构已联系",
                "同为无外发无安排的对照：需不需要替你打听，消息有了再告诉你？为提议；我替你打听，消息有了再告诉你。为已承接，后者需拒绝。"
                "群里或私聊说烟况还没定，随后另起一句「稍后问下」，仍明确承担这次询问，需要同桌的实际安排。只有烟况未知时可照实说，或询问顾客是否需要代问。"
                "条件式征询可通过，但另起我已经联系过其他人了须证据；提议后另起我也会持续替你盯着仍是独立责任，无安排拒绝",
                "安全改写只修违规，不新增换档位、换时间等业务安排；原稿里的下一步建议不是权威事实，不能因为它曾出现就强制保留；若业务依据有问题，反馈原因给主模型重新规划",
            ],
        },
        "review_contract": {
            "format": "json_object",
            "required_keys": ["approved", "needs_human", "reasoning_summary", "violations", "item_reviews"],
            "field_types": {
                "approved": "boolean",
                "needs_human": "boolean",
                "reasoning_summary": "string",
                "violations": "array of string labels",
                "item_reviews": "array; one item per review_items entry",
                "item_reviews[].item_id": "must equal original review_items[].item_id",
                "item_reviews[].approved": "boolean",
                "item_reviews[].suggested_safe_text": "string; same as original text when approved=true, safe rewrite when approved=false",
                **({
                    "item_reviews[].reviewed_game_offer_id": (
                        "仅当正文原样放行并明确让本收件人选择是否参加这唯一的一桌时，填写 game_offer_candidate.game_id。"
                        "仍须确认是否赶得上或桌上可否等候时，改用 reviewed_game_reference_id。"
                    ),
                    "item_reviews[].reviewed_game_reference_id": (
                        "正文原样放行，且正文与当前来话明确讨论查询证实的一桌，但只是介绍、拒绝或条件待核对时，"
                        "填写 game_offer_candidate.game_id。已入席回执、其他桌或指代不唯一时留空。"
                        "两种字段最多填一种；有候选时须判断是否适用，均不适用可留空或填 null。"
                    ),
                } if has_game_offer_candidate else {}),
                "item_reviews[].external_action_attributions": "optional JSON array conforming to external_action_attribution_contract.json_schema; actor is the fixed literal current_customer, not a customer ID; each *_span is a JSON array of two integers, never an interval string",
                "item_reviews[].follow_up_commitments": (
                    "When follow_up_commitment_evidence is present, approved items require a JSON array of "
                    "{quote, kind, arrangement_ref}; quote is exact original text, kind is finite_recheck or match_notification, "
                    "arrangement_ref is an exact evidence ref that covers the promise or null when none. No commitments means []."
                ),
            },
            "invariants": [
                "approved=true means every review item has no information leakage, unverified external action, semantic contradiction, or required fact loss",
                "approved=true requires every item_reviews[].approved=true and suggested_safe_text equals the original text",
                "the top-level approved value must equal the aggregate of item_reviews; an unchanged, violation-free, approved item must not be rejected only because it safely refuses a user's requested disclosure format",
                "approved=false and needs_human=false means at least one item_reviews entry contains a safe rewrite",
                "needs_human=true means the main agent should use a short human-handoff reply or stop the unsafe tool action",
                "suggested_safe_text must not expose actual credentials, this system's private protocol identifiers or implementation state, or unauthorized customer identities/roles; ordinary JSON, technical, draft and approval discussions are allowed",
            ],
            "available_tools": [],
        },
    }
    return project_review_transport_payload(payload)


def project_review_transport_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """仅引用审查输入中的精确副本，原上下文、原工具证据与业务值保持不变。

    最新反馈保留顺序及执行外壳，完整有序证据只序列化一次；共享查询合同仅在
    完全相等时引用顶层定义。这里不摘要、不选择证据，也不解析用户自然语言。
    """
    projected = deepcopy(payload)
    latest = projected.get("previous_tool_results")
    evidence = projected.get("turn_tool_evidence")
    if isinstance(latest, list) and isinstance(evidence, list):
        projected["previous_tool_results"], _ = reference_duplicate_latest_tool_results(latest, evidence)
    projected, _ = reference_shared_query_contracts(
        projected, contract_path=("game_query_strategy_contract",),
        conversation_field="recent_conversation_tail",
    )
    return projected


def expand_review_transport_references(payload: dict[str, Any]) -> dict[str, Any]:
    """为验真展开审查的受控引用，拒绝越界、错源、篡改外壳和循环引用。

    只处理后端约定的工具路径；不遍历客户正文。完整事实和列表顺序均来自原
    发送载荷，不能从当前代码版本补合同或从预期标签恢复缺失证据。
    """
    expanded = deepcopy(payload)
    canonical = expanded.get("game_query_strategy_contract")

    def expand_contracts(results: list[Any]) -> None:
        """在固定工具结果路径恢复合同，历史不同版本的完整合同保留原样。"""
        for raw in results:
            result = raw.get("result") if isinstance(raw, dict) else None
            reply = result.get("customer_reply_contract") if isinstance(result, dict) else None
            contract = reply.get("game_query_strategy_contract") if isinstance(reply, dict) else None
            if not isinstance(contract, dict) or "contract_reference" not in contract:
                continue
            if (contract != {"contract_reference": "game_query_strategy_contract"}
                    or not isinstance(canonical, dict) or not canonical
                    or "contract_reference" in canonical):
                raise ValueError("invalid review query contract reference")
            reply["game_query_strategy_contract"] = deepcopy(canonical)

    evidence = expanded.get("turn_tool_evidence") or []
    latest = expanded.get("previous_tool_results") or []
    if not isinstance(evidence, list) or not isinstance(latest, list):
        raise ValueError("review tool evidence must be ordered lists")
    if any(isinstance(raw, dict) and "result_reference" in raw for raw in evidence):
        raise ValueError("review authoritative evidence cannot reference another result")
    expand_contracts(evidence)
    expand_contracts(latest)
    restored = []
    claimed_indexes: set[int] = set()
    shell_keys = ("name", "called", "allowed", "call_id", "error", "deduplicated")
    for raw in latest:
        reference = raw.get("result_reference") if isinstance(raw, dict) else None
        if reference is None:
            restored.append(raw)
            continue
        if (not isinstance(reference, dict) or set(reference) != {"source", "index", "call_id"}
                or reference["source"] != "turn_tool_evidence" or type(reference["index"]) is not int
                or not 0 <= reference["index"] < len(evidence)
                or reference["index"] in claimed_indexes
                or set(raw) != {*shell_keys, "result_reference"}):
            raise ValueError("invalid review tool result reference")
        target = evidence[reference["index"]]
        if (not isinstance(target, dict) or "result_reference" in target
                or reference["call_id"] != target.get("call_id")
                or any(type(raw[key]) is not type(target.get(key, False if key == "deduplicated" else None))
                       or raw[key] != target.get(key, False if key == "deduplicated" else None)
                       for key in shell_keys)):
            raise ValueError("review tool reference differs from its authoritative evidence")
        restored.append(deepcopy(target))
        claimed_indexes.add(reference["index"])
    expanded["previous_tool_results"] = restored
    for turn in expanded.get("recent_conversation_tail") or []:
        if not isinstance(turn, dict) or turn.get("role") != "tool":
            continue
        content = turn.get("content")
        if isinstance(content, str):
            try:
                results = json.loads(content)
            except json.JSONDecodeError:
                continue
            expand_contracts(results if isinstance(results, list) else [results])
            turn["content"] = json.dumps(results, ensure_ascii=False)
        elif isinstance(content, (list, dict)):
            expand_contracts(content if isinstance(content, list) else [content])
    return expanded


def parse_reply_self_review(raw_response: str) -> tuple[dict[str, Any], list[str]]:
    """解析replyself审查，并把非结构化输入转换为稳定合同。

"""

    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return {}, [f"reply self review is not valid JSON: {exc.msg}"]
    if not isinstance(payload, dict):
        return {}, ["reply self review JSON root must be object"]
    errors: list[str] = []
    for key in ["approved", "needs_human", "reasoning_summary", "violations"]:
        if key not in payload:
            errors.append(f"missing required review key: {key}")
    if "approved" in payload and not isinstance(payload.get("approved"), bool):
        errors.append("approved must be boolean")
    if "needs_human" in payload and not isinstance(payload.get("needs_human"), bool):
        errors.append("needs_human must be boolean")
    if "reasoning_summary" in payload and not isinstance(payload.get("reasoning_summary"), str):
        errors.append("reasoning_summary must be string")
    if "violations" in payload and not isinstance(payload.get("violations"), list):
        errors.append("violations must be array")
    if "item_reviews" in payload and not isinstance(payload.get("item_reviews"), list):
        errors.append("item_reviews must be array")
    if "item_reviews" in payload and isinstance(payload.get("item_reviews"), list):
        for index, item in enumerate(payload.get("item_reviews") or [], start=1):
            if not isinstance(item, dict):
                errors.append(f"item_reviews[{index}] must be object")
                continue
            if not isinstance(item.get("item_id"), str) or not item.get("item_id"):
                errors.append(f"item_reviews[{index}].item_id is required")
            if "approved" in item and not isinstance(item.get("approved"), bool):
                errors.append(f"item_reviews[{index}].approved must be boolean")
            if not isinstance(item.get("suggested_safe_text"), str):
                errors.append(f"item_reviews[{index}].suggested_safe_text must be string")
            if "external_action_attributions" in item and not isinstance(item["external_action_attributions"], list):
                errors.append(f"item_reviews[{index}].external_action_attributions must be array")
    return payload, errors


__all__ = [
    "CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME",
    "build_reply_self_review_payload",
    "project_review_transport_payload",
    "expand_review_transport_references",
    "customer_visible_content_review_approved",
    "customer_visible_items_for_action",
    "item_reviews_approved",
    "normalize_item_reviews",
    "normalize_review_item_contract",
    "parse_reply_self_review",
]
