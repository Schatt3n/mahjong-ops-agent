from __future__ import annotations

"""Pure contracts and normalization helpers for customer-visible review."""

import json
from typing import Any

from .customer_visible_contract import (
    customer_visible_action_claim_violations,
    customer_visible_text_contract_violations,
)
from .models import AgentAction, ToolResult, UserMessage


CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME = "customer_visible_content_review"


def external_action_evidence_from_tool_results(
    tool_results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Project delivery facts from backend ToolResults for final-reply review."""

    relevant_names = {
        "create_invite_drafts",
        "create_outbound_message_drafts",
        "update_invite_delivery_status",
    }
    source_tool_names: list[str] = []
    draft_statuses: list[str] = []
    for raw in tool_results:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "")
        if name not in relevant_names or not raw.get("called") or not raw.get("allowed") or raw.get("error"):
            continue
        source_tool_names.append(name)
        result = raw.get("result") if isinstance(raw.get("result"), dict) else {}
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
        "contact_started": any(status in contact_started_statuses for status in draft_statuses),
    }


def customer_visible_content_review_approved(result: ToolResult) -> bool:
    """Return whether the review tool approved every customer-visible item."""

    return (
        result.name == CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME
        and result.called
        and result.allowed
        and result.error is None
        and bool(result.result.get("approved"))
    )


def customer_visible_items_for_action(action: AgentAction) -> list[dict[str, Any]]:
    """Extract outbound text from tool arguments before execution."""

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
    review: dict[str, Any], review_items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Normalize current and legacy model responses into one stable item contract."""

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
                )
            )
        return normalized

    by_item_id = {
        str(item.get("item_id") or ""): item for item in raw_reviews if isinstance(item, dict)
    }
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
        normalized.append(
            normalize_review_item_contract(
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
            )
        )
    return normalized


def normalize_review_item_contract(
    *,
    item_id: str,
    approved: bool,
    suggested_safe_text: str,
    reasoning_summary: str,
    violations: list[str],
    action_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply deterministic text-contract checks to one model review item."""

    contract_violations = customer_visible_text_contract_violations(suggested_safe_text)
    action_claim_violations = customer_visible_action_claim_violations(
        suggested_safe_text,
        action_evidence,
    )
    normalized_violations = list(violations)
    normalized_violations.extend(
        f"customer_visible_contract:{violation}" for violation in contract_violations
    )
    normalized_violations.extend(
        f"customer_visible_contract:{violation}" for violation in action_claim_violations
    )
    return {
        "item_id": item_id,
        "approved": bool(approved) and not contract_violations and not action_claim_violations,
        "suggested_safe_text": suggested_safe_text,
        "reasoning_summary": reasoning_summary,
        "violations": normalized_violations,
    }


def item_reviews_approved(
    item_reviews: list[dict[str, Any]], review_items: list[dict[str, Any]]
) -> bool:
    """Require a one-to-one, unchanged, violation-free approval for every item."""

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
    if len(item_reviews) != len(review_items):
        return False
    for item in item_reviews:
        item_id = str(item.get("item_id") or "")
        suggested = str(item.get("suggested_safe_text") or "")
        if item_id not in original_by_id or not bool(item.get("approved")):
            return False
        if suggested != original_by_id[item_id]:
            return False
        if customer_visible_text_contract_violations(suggested):
            return False
        if customer_visible_action_claim_violations(suggested, evidence_by_id.get(item_id)):
            return False
    return True


def build_reply_self_review_payload(
    *,
    message: UserMessage,
    action: AgentAction,
    review_items: list[dict[str, Any]],
    context_payload: dict[str, Any],
    review_scope: str,
) -> dict[str, Any]:
    """Build the minimal review-only context and output contract."""

    turn_tool_evidence = (
        context_payload.get("turn_tool_evidence")
        or context_payload.get("previous_tool_results")
        or []
    )
    return {
        "current_message": message.to_dict(),
        "customer_visibility_contract": context_payload.get("customer_visibility_contract") or {},
        "sender_profile": context_payload.get("sender_profile"),
        "sender_relationships": context_payload.get("sender_relationships") or [],
        "active_games": context_payload.get("active_games") or [],
        "active_game_visible_summaries": context_payload.get("active_game_visible_summaries") or [],
        "previous_tool_results": context_payload.get("previous_tool_results") or [],
        "turn_tool_evidence": turn_tool_evidence,
        "recent_conversation_tail": list(context_payload.get("recent_conversation") or [])[-8:],
        "action_boundary": {
            "objective_status": action.objective_status,
            "needs_human": action.needs_human,
            "tool_call_names": [call.name for call in action.tool_calls],
            "has_reply_to_user": bool(action.reply_to_user.strip()),
            "customer_visible_item_count": len(review_items),
        },
        "review_scope": review_scope,
        "review_items": review_items,
        "review_goal": (
            "一次性审查 review_items 中的客户可见文本是否泄露系统信息、后台流程、其他用户信息、"
            "未发生动作，或与权威工具结果/客户可见局摘要/改写前文本发生语义矛盾或关键事实丢失；"
            "不做业务规划，不决定工具调用，不负责润色文风。"
        ),
        "semantic_fidelity_contract": {
            "source_order": [
                "成功的 turn_tool_evidence 是本轮完整工具事实的最高优先级来源",
                "previous_tool_results 只表示最近一步反馈，用于定位最新错误或续办要求",
                "active_game_visible_summaries 是可对当前客户披露的局况事实来源",
                "review_items[].source_text 是话术改写前的语义基线",
                "review_items[].text 是待发送文本，不得反向覆盖上述事实",
            ],
            "rules": [
                "待审文本不得反转存在/不存在、成功/失败、已确认/未确认、已发送/未发送等事实极性",
                "待审文本不得修改人数、缺口、时间、档位、烟况、时长、玩法或下一步决策问题",
                "如果 source_text 或客户可见摘要把多个字段组成一个可识别选项，待审文本必须保留完整决策锚点",
                "成功查询返回非空 matches 时，待审文本不得声称没有匹配结果",
                "审查发现语义不保真时 approved=false，并在 violations 标注 semantic_contradiction 或 semantic_fact_loss",
                "当 review_items[].action_evidence.contact_started=false 时，不得声称已经问了、联系了、发送了或通知了；外部动作只能以该后端证据为准",
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
            },
            "invariants": [
                "approved=true means every review item has no information leakage, unverified external action, semantic contradiction, or required fact loss",
                "approved=true requires every item_reviews[].approved=true and suggested_safe_text equals the original text",
                "the top-level approved value must equal the aggregate of item_reviews; an unchanged, violation-free, approved item must not be rejected only because it safely refuses a user's requested disclosure format",
                "approved=false and needs_human=false means at least one item_reviews entry contains a safe rewrite",
                "needs_human=true means the main agent should use a short human-handoff reply or stop the unsafe tool action",
                "suggested_safe_text must not expose tool names, JSON, traceId, internal process, draft/approval state, or other customer identities/roles",
            ],
            "available_tools": [],
        },
    }


def parse_reply_self_review(raw_response: str) -> tuple[dict[str, Any], list[str]]:
    """Parse and validate the review model's JSON response."""

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
    return payload, errors


__all__ = [
    "CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME",
    "build_reply_self_review_payload",
    "customer_visible_content_review_approved",
    "customer_visible_items_for_action",
    "item_reviews_approved",
    "normalize_item_reviews",
    "normalize_review_item_contract",
    "parse_reply_self_review",
]
