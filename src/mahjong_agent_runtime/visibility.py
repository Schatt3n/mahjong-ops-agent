from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .budget import TokenBudget
from .copywriting import (
    DEFAULT_CUSTOMER_VISIBLE_TEXT_PROMPT_PATH,
    build_customer_visible_text_generation_payload,
    parse_customer_visible_text_generation,
)
from .customer_visible_contract import customer_visible_text_contract_violations
from .llm import AgentLLMClient
from .models import AgentAction, ToolResult, UserMessage


DEFAULT_REPLY_SELF_REVIEW_PROMPT_PATH = Path(__file__).with_name("prompts").joinpath("agent_runtime_reply_self_review.md")
CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME = "customer_visible_content_review"
REPLY_SELF_REVIEW_TOOL_NAME = CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME
CUSTOMER_VISIBLE_TEXT_GENERATION_NAME = "customer_visible_text_generation"


@dataclass(slots=True)
class CustomerVisibleProcessor:
    llm_client: AgentLLMClient
    trace_recorder: Any
    timeout_seconds: float
    text_generation_enabled: bool = False
    text_generation_client: AgentLLMClient | None = None
    text_generation_prompt_path: Path = DEFAULT_CUSTOMER_VISIBLE_TEXT_PROMPT_PATH
    review_enabled: bool = False
    review_client: AgentLLMClient | None = None
    review_prompt_path: Path = DEFAULT_REPLY_SELF_REVIEW_PROMPT_PATH

    def run_text_generation(
        self,
        *,
        message: UserMessage,
        trace_id: str,
        action: AgentAction,
        items: list[dict[str, Any]],
        context_payload: dict[str, Any],
        turn_budget: TokenBudget,
        generation_scope: str,
    ) -> ToolResult | None:
        if not self.text_generation_enabled or not items:
            return None
        client = self.text_generation_client or self.llm_client
        generation_payload = build_customer_visible_text_generation_payload(
            message=message,
            action=action,
            items=items,
            context_payload=context_payload,
            generation_scope=generation_scope,
        )
        messages = [
            {"role": "system", "content": self.text_generation_prompt_path.read_text(encoding="utf-8")},
            {"role": "user", "content": json.dumps(generation_payload, ensure_ascii=False, sort_keys=True)},
        ]
        self.trace_recorder.record(trace_id, "customer_visible_text_generation_prompt", {"messages": messages})
        budget = turn_budget.reserve(messages)
        self.trace_recorder.record(trace_id, "customer_visible_text_generation_budget_checked", budget.to_dict())
        if not budget.allowed:
            self.trace_recorder.record(
                trace_id,
                "customer_visible_text_generation_skipped",
                {"reason": budget.reason, "generation_scope": generation_scope, "items": items},
                level="WARN",
            )
            return None
        started = time.perf_counter()
        try:
            raw_response = client.complete(messages, trace_id=trace_id, timeout_seconds=self.timeout_seconds)
        except Exception as exc:
            self.trace_recorder.record(
                trace_id,
                "customer_visible_text_generation_error",
                {
                    "generation_scope": generation_scope,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                },
                level="ERROR",
            )
            return None
        self.trace_recorder.record(
            trace_id,
            "customer_visible_text_generation_response",
            {"content": raw_response, "elapsed_ms": int((time.perf_counter() - started) * 1000), "generation_scope": generation_scope},
        )
        generation, errors = parse_customer_visible_text_generation(raw_response, items)
        if errors:
            self.trace_recorder.record(
                trace_id,
                "customer_visible_text_generation_contract_error",
                {"errors": errors, "raw_response": raw_response, "generation_scope": generation_scope},
                level="WARN",
            )
            return None
        result = ToolResult(
            name=CUSTOMER_VISIBLE_TEXT_GENERATION_NAME,
            called=True,
            allowed=True,
            result={
                "generation_scope": generation_scope,
                "items": items,
                "reasoning_summary": generation.reasoning_summary,
                "item_rewrites": generation.item_rewrites,
            },
        )
        self.trace_recorder.record(trace_id, "customer_visible_text_generation_result", result.to_dict())
        return result

    def run_content_review(
        self,
        *,
        message: UserMessage,
        trace_id: str,
        action: AgentAction,
        review_items: list[dict[str, Any]],
        context_payload: dict[str, Any],
        turn_budget: TokenBudget,
        review_scope: str,
    ) -> ToolResult | None:
        if not self.review_enabled or not review_items:
            return None
        client = self.review_client or self.llm_client
        review_payload = build_reply_self_review_payload(
            message=message,
            action=action,
            review_items=review_items,
            context_payload=context_payload,
            review_scope=review_scope,
        )
        messages = [
            {"role": "system", "content": self.review_prompt_path.read_text(encoding="utf-8")},
            {"role": "user", "content": json.dumps(review_payload, ensure_ascii=False, sort_keys=True)},
        ]
        self.trace_recorder.record(trace_id, "customer_visible_content_review_prompt", {"messages": messages})
        budget = turn_budget.reserve(messages)
        self.trace_recorder.record(trace_id, "customer_visible_content_review_budget_checked", budget.to_dict())
        if not budget.allowed:
            self.trace_recorder.record(trace_id, "customer_visible_content_review_failed", {"reason": budget.reason}, level="WARN")
            return ToolResult(
                name=CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME,
                called=False,
                allowed=False,
                result={
                    "review_scope": review_scope,
                    "items": review_items,
                    "instruction": "Customer-visible content review could not run because the turn budget was exhausted. Use needs_human with a short safe handoff reply.",
                },
                error=budget.reason,
            )
        started = time.perf_counter()
        try:
            raw_response = client.complete(messages, trace_id=trace_id, timeout_seconds=self.timeout_seconds)
        except Exception as exc:
            self.trace_recorder.record(
                trace_id,
                "customer_visible_content_review_error",
                {"error_type": type(exc).__name__, "error": str(exc), "elapsed_ms": int((time.perf_counter() - started) * 1000)},
                level="ERROR",
            )
            return ToolResult(
                name=CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME,
                called=False,
                allowed=False,
                result={
                    "review_scope": review_scope,
                    "items": review_items,
                    "instruction": "Customer-visible content review failed. Use needs_human with a short safe handoff reply.",
                },
                error=f"{type(exc).__name__}: {exc}",
            )
        self.trace_recorder.record(
            trace_id,
            "customer_visible_content_review_response",
            {"content": raw_response, "elapsed_ms": int((time.perf_counter() - started) * 1000)},
        )
        review, errors = parse_reply_self_review(raw_response)
        if errors:
            self.trace_recorder.record(trace_id, "customer_visible_content_review_contract_error", {"errors": errors}, level="WARN")
            return ToolResult(
                name=CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME,
                called=True,
                allowed=False,
                result={
                    "review_scope": review_scope,
                    "items": review_items,
                    "errors": list(errors),
                    "raw_response": raw_response,
                    "instruction": "Customer-visible content review returned an invalid contract. Use needs_human with a short safe handoff reply.",
                },
                error="Customer-visible content review contract invalid: " + "; ".join(errors),
            )
        item_reviews = normalize_item_reviews(review, review_items)
        approved = bool(review.get("approved")) and not bool(review.get("needs_human")) and item_reviews_approved(item_reviews, review_items)
        needs_human = bool(review.get("needs_human"))
        instruction = (
            "Review approved all customer-visible content. You may continue with the original action."
            if approved
            else "Review rejected customer-visible content. Read violations and suggested_safe_text values, then generate a corrected AgentAction; do not expose review details to any customer."
        )
        self.trace_recorder.record(
            trace_id,
            "customer_visible_content_review_result",
            {
                "approved": approved,
                "raw_approved": bool(review.get("approved")),
                "needs_human": needs_human,
                "review_scope": review_scope,
                "item_reviews": item_reviews,
                "reasoning_summary": str(review.get("reasoning_summary") or ""),
                "violations": list(review.get("violations") or []) if isinstance(review.get("violations"), list) else [],
            },
            level="WARN" if not approved else "INFO",
        )
        return ToolResult(
            name=CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME,
            called=True,
            allowed=True,
            result={
                "approved": approved,
                "raw_approved": bool(review.get("approved")),
                "needs_human": needs_human,
                "review_scope": review_scope,
                "items": review_items,
                "item_reviews": item_reviews,
                "reasoning_summary": str(review.get("reasoning_summary") or ""),
                "violations": list(review.get("violations") or []) if isinstance(review.get("violations"), list) else [],
                "instruction": instruction,
            },
        )


def customer_visible_content_review_approved(result: ToolResult) -> bool:
    return (
        result.name == CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME
        and result.called
        and result.allowed
        and result.error is None
        and bool(result.result.get("approved"))
    )


def customer_visible_items_for_action(action: AgentAction) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for call_index, call in enumerate(action.tool_calls, start=1):
        if call.name == "create_invite_drafts":
            for item_index, raw in enumerate(call.arguments.get("invitations") or [], start=1):
                if not isinstance(raw, dict):
                    continue
                text = str(raw.get("message_text") or "").strip()
                if not text:
                    continue
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
                if not text:
                    continue
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


def normalize_item_reviews(review: dict[str, Any], review_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_reviews = review.get("item_reviews")
    if not isinstance(raw_reviews, list):
        legacy_safe_text = str(review.get("final_reply") or "").strip()
        normalized: list[dict[str, Any]] = []
        for item in review_items:
            suggested = legacy_safe_text or str(item.get("text") or "")
            base_violations = list(review.get("violations") or []) if isinstance(review.get("violations"), list) else []
            normalized.append(
                normalize_review_item_contract(
                    item_id=str(item.get("item_id") or ""),
                    approved=bool(review.get("approved")) and (not legacy_safe_text or legacy_safe_text == str(item.get("text") or "")),
                    suggested_safe_text=suggested,
                    reasoning_summary=str(review.get("reasoning_summary") or ""),
                    violations=base_violations,
                )
            )
        return normalized
    by_item_id = {str(item.get("item_id") or ""): item for item in raw_reviews if isinstance(item, dict)}
    normalized: list[dict[str, Any]] = []
    for item in review_items:
        item_id = str(item.get("item_id") or "")
        raw = by_item_id.get(item_id, {})
        suggested = str(raw.get("suggested_safe_text") or raw.get("safe_text") or raw.get("final_text") or item.get("text") or "")
        violations = raw.get("violations")
        normalized.append(
            normalize_review_item_contract(
                item_id=item_id,
                approved=bool(raw.get("approved")),
                suggested_safe_text=suggested,
                reasoning_summary=str(raw.get("reasoning_summary") or ""),
                violations=list(violations) if isinstance(violations, list) else [],
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
) -> dict[str, Any]:
    contract_violations = customer_visible_text_contract_violations(suggested_safe_text)
    normalized_violations = list(violations)
    for violation in contract_violations:
        normalized_violations.append(f"customer_visible_contract:{violation}")
    return {
        "item_id": item_id,
        "approved": bool(approved) and not contract_violations,
        "suggested_safe_text": suggested_safe_text,
        "reasoning_summary": reasoning_summary,
        "violations": normalized_violations,
    }


def item_reviews_approved(item_reviews: list[dict[str, Any]], review_items: list[dict[str, Any]]) -> bool:
    original_by_id = {str(item.get("item_id") or ""): str(item.get("text") or "") for item in review_items}
    if len(item_reviews) != len(review_items):
        return False
    for item in item_reviews:
        item_id = str(item.get("item_id") or "")
        if item_id not in original_by_id:
            return False
        if not bool(item.get("approved")):
            return False
        suggested_safe_text = str(item.get("suggested_safe_text") or "")
        if suggested_safe_text != original_by_id[item_id]:
            return False
        if customer_visible_text_contract_violations(suggested_safe_text):
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
    return {
        "current_message": message.to_dict(),
        "sender_profile": context_payload.get("sender_profile"),
        "sender_relationships": context_payload.get("sender_relationships") or [],
        "active_games": context_payload.get("active_games") or [],
        "active_game_visible_summaries": context_payload.get("active_game_visible_summaries") or [],
        "previous_tool_results": context_payload.get("previous_tool_results") or [],
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
        "review_goal": "一次性审查 review_items 中的客户可见文本是否泄露系统信息、后台流程、其他用户信息或未发生动作；不做业务规划，不决定工具调用，不负责润色文风。",
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
                "approved=true means every review item has no information leakage or unverified external action",
                "approved=true requires every item_reviews[].approved=true and suggested_safe_text equals the original text",
                "approved=false and needs_human=false means at least one item_reviews entry contains a safe rewrite",
                "needs_human=true means the main agent should use a short human-handoff reply or stop the unsafe tool action",
                "suggested_safe_text must not expose tool names, JSON, traceId, internal process, draft/approval state, or other customer identities/roles",
            ],
            "available_tools": [],
        },
    }


def parse_reply_self_review(raw_response: str) -> tuple[dict[str, Any], list[str]]:
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
