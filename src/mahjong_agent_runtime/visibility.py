from __future__ import annotations

"""客户可见文本生成与审查。

设计理念：
- 主 agent 可以提出回复和邀约草稿，但凡是客户可能看到的文本，都必须走统一处理链路。
- 话术生成负责把结构化、僵硬的文本改得更像老板说话。
- 内容审查负责信息泄露、未验证动作和语义一致性风险，不做业务规划，不决定工具调用。
- 这一层是生产安全边界，不应该散落在主 loop 或具体工具里。
"""

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
from .customer_visible_review import (
    CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME,
    build_reply_self_review_payload,
    customer_visible_content_review_approved,
    customer_visible_items_for_action,
    item_reviews_approved,
    normalize_item_reviews,
    normalize_review_item_contract,
    parse_reply_self_review,
)
from .llm import AgentLLMClient
from .models import AgentAction, ToolResult, UserMessage


DEFAULT_REPLY_SELF_REVIEW_PROMPT_PATH = Path(__file__).with_name("prompts").joinpath("agent_runtime_reply_self_review.md")
REPLY_SELF_REVIEW_TOOL_NAME = CUSTOMER_VISIBLE_CONTENT_REVIEW_TOOL_NAME
CUSTOMER_VISIBLE_TEXT_GENERATION_NAME = "customer_visible_text_generation"


@dataclass(slots=True)
class CustomerVisibleProcessor:
    """客户可见文本处理器。

    它把“话术生成”和“安全审查”从主 runtime 中拆出来。
    runtime 只负责判断什么时候调用它；它负责构建专用上下文、调用模型、校验返回合同和写 trace。
    """

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
        """运行客户可见话术生成。

        generation_scope 用来区分是最终回复还是工具参数里的邀约草稿。
        如果开关关闭、没有文本或预算不足，就不阻断主流程，直接返回 None。
        """

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
        """运行客户可见内容审查。

        审查结果以 ToolResult 形式返回给主 loop：通过则继续执行，失败则作为工具结果回喂主模型修正。
        当审查模型异常或预算不足时，返回 allowed=False，要求主流程走安全兜底或人工。
        """

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
        needs_human = bool(review.get("needs_human"))
        top_level_violations = (
            list(review.get("violations") or [])
            if isinstance(review.get("violations"), list)
            else []
        )
        item_level_approved = item_reviews_approved(item_reviews, review_items)
        # Item reviews are the recipient-specific decisions and therefore the
        # authoritative aggregate source. Models occasionally emit an
        # inconsistent top-level approved=false while every item is approved,
        # unchanged, and violation-free. Derive the aggregate deterministically
        # so a redundant Boolean cannot make the loop retry until max_steps.
        approved = item_level_approved and not needs_human and not top_level_violations
        aggregate_repaired = approved != bool(review.get("approved"))
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
                "aggregate_repaired": aggregate_repaired,
                "needs_human": needs_human,
                "review_scope": review_scope,
                "item_reviews": item_reviews,
                "reasoning_summary": str(review.get("reasoning_summary") or ""),
                "violations": top_level_violations,
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
                "aggregate_repaired": aggregate_repaired,
                "needs_human": needs_human,
                "review_scope": review_scope,
                "items": review_items,
                "item_reviews": item_reviews,
                "reasoning_summary": str(review.get("reasoning_summary") or ""),
                "violations": top_level_violations,
                "instruction": instruction,
            },
        )
