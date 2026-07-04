from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import AgentAction, ToolResult, UserMessage


DEFAULT_CUSTOMER_VISIBLE_TEXT_PROMPT_PATH = Path(__file__).with_name("prompts").joinpath(
    "customer_visible_text_generation.md"
)


@dataclass(slots=True)
class CustomerVisibleTextGeneration:
    reasoning_summary: str = ""
    item_rewrites: list[dict[str, Any]] = field(default_factory=list)

    def rewrite_by_item_id(self) -> dict[str, str]:
        return {
            str(item.get("item_id") or ""): str(item.get("final_text") or "").strip()
            for item in self.item_rewrites
            if str(item.get("item_id") or "").strip() and str(item.get("final_text") or "").strip()
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "reasoning_summary": self.reasoning_summary,
            "item_rewrites": [dict(item) for item in self.item_rewrites],
        }


def build_customer_visible_text_generation_payload(
    *,
    message: UserMessage,
    action: AgentAction,
    items: list[dict[str, Any]],
    context_payload: dict[str, Any],
    generation_scope: str,
) -> dict[str, Any]:
    _ = message, context_payload
    public_items = [
        {
            "item_id": str(item.get("item_id") or ""),
            "source": str(item.get("source") or ""),
            "text": str(item.get("text") or ""),
        }
        for item in items
    ]
    return {
        "generation_scope": generation_scope,
        "items": public_items,
        "action_boundary": {
            "objective_status": action.objective_status,
            "needs_human": action.needs_human,
            "tool_call_names": [call.name for call in action.tool_calls],
        },
        "generation_goal": (
            "Rewrite only the customer-visible text in items into natural mahjong-shop owner WeChat wording. "
            "This is a semantic-preserving surface rewrite stage, not a business reasoning stage."
        ),
        "semantic_source_of_truth": "Only items[].text is allowed as factual source for the rewrite.",
        "allowed_changes": [
            "Normalize awkward visible wording already present in the item text, such as 1 -> 1块 when it is a stake.",
            "Translate internal enum-like words already present in the item text into natural customer wording.",
            "Shorten customer-service wording into natural WeChat wording while preserving the same meaning.",
            "Remove a pure leading salutation when it does not carry business meaning.",
        ],
        "style_quality_contract": {
            "voice": "mahjong_shop_owner_wechat",
            "target": "short, direct, decision-focused Chinese that a mahjong-shop owner would send in WeChat",
            "forbidden_customer_service_phrases": [
                "为您",
                "请耐心等待",
                "是否方便",
                "是否加入",
                "要加入吗",
                "要不要加入",
                "要一起吗",
                "请问还有什么可以帮您",
            ],
            "preferred_short_phrases": [
                "打吗？",
                "来吗？",
                "可以不？",
                "来不？",
                "好，我帮你问问。",
                "有消息跟你说。",
            ],
            "must_preserve_if_present": [
                "time",
                "public nickname/group nickname",
                "seat count or shortage code",
                "stake",
                "smoking condition",
                "duration",
                "next decision question",
            ],
        },
        "forbidden_changes": [
            "Do not add people, counts, missing seats, relationships, confirmed status, invite status, or tool execution facts.",
            "Do not infer from active games, customer profiles, conversation history, or tool results; they are intentionally not provided.",
            "Do not turn an uncertain or multi-option reply into a definite operational promise.",
            "Do not expose internal process details such as drafts, approval, tools, candidate counts, or backend state labels.",
        ],
        "output_contract": {
            "format": "json_object",
            "required_keys": ["reasoning_summary", "item_rewrites"],
            "item_rewrites_contract": {
                "one_item_per_input_item": True,
                "item_id": "must equal input items[].item_id",
                "final_text": "required non-empty customer-visible Chinese text",
                "semantic_preserved": "required boolean true; false means backend will discard the rewrite",
                "used_facts": "array of public facts used",
                "withheld_facts": "array of facts intentionally not disclosed",
                "style_checks": "array of style/self-check notes",
                "change_summary": "short description of wording-only changes",
            },
            "invariants": [
                "Do not output tool calls.",
                "Do not output Markdown.",
                "Only use facts explicitly present in input items[].text.",
                "Never repair missing business facts in this stage.",
                "Do not invent external actions such as already sent or already asked.",
                "Do not expose internal enum values or backend state labels.",
                "Keep customer-visible wording short and natural.",
                "Every rewrite must set semantic_preserved=true.",
            ],
            "available_tools": [],
        },
    }


def parse_customer_visible_text_generation(
    raw_response: str,
    items: list[dict[str, Any]],
) -> tuple[CustomerVisibleTextGeneration, list[str]]:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        return CustomerVisibleTextGeneration(), [f"text generation is not valid JSON: {exc.msg}"]
    if not isinstance(payload, dict):
        return CustomerVisibleTextGeneration(), ["text generation JSON root must be object"]
    errors: list[str] = []
    if not isinstance(payload.get("reasoning_summary"), str):
        errors.append("reasoning_summary must be string")
    raw_rewrites = payload.get("item_rewrites")
    if not isinstance(raw_rewrites, list):
        errors.append("item_rewrites must be array")
        raw_rewrites = []
    input_ids = [str(item.get("item_id") or "") for item in items]
    rewrite_ids: list[str] = []
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rewrites, start=1):
        if not isinstance(raw, dict):
            errors.append(f"item_rewrites[{index}] must be object")
            continue
        item_id = str(raw.get("item_id") or "")
        final_text = str(raw.get("final_text") or "").strip()
        semantic_preserved = raw.get("semantic_preserved")
        rewrite_ids.append(item_id)
        if not item_id:
            errors.append(f"item_rewrites[{index}].item_id is required")
        if item_id not in input_ids:
            errors.append(f"item_rewrites[{index}].item_id must match an input item_id")
        if not final_text:
            errors.append(f"item_rewrites[{index}].final_text is required")
        if semantic_preserved is not True:
            errors.append(f"item_rewrites[{index}].semantic_preserved must be true")
        normalized.append(
            {
                "item_id": item_id,
                "final_text": final_text,
                "semantic_preserved": semantic_preserved is True,
                "used_facts": normalize_string_list(raw.get("used_facts")),
                "withheld_facts": normalize_string_list(raw.get("withheld_facts")),
                "style_checks": normalize_string_list(raw.get("style_checks")),
                "change_summary": str(raw.get("change_summary") or "").strip(),
            }
        )
    if len(raw_rewrites) != len(items):
        errors.append("item_rewrites must contain exactly one entry per input item")
    missing_ids = [item_id for item_id in input_ids if item_id not in rewrite_ids]
    if missing_ids:
        errors.append(f"item_rewrites missing item_id(s): {', '.join(missing_ids)}")
    return CustomerVisibleTextGeneration(str(payload.get("reasoning_summary") or ""), normalized), errors


def normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def action_with_customer_visible_rewrites(action: AgentAction, rewrites: dict[str, str]) -> AgentAction:
    if not rewrites:
        return action
    payload = action.to_dict()
    if "reply_to_user" in rewrites:
        payload["reply_to_user"] = rewrites["reply_to_user"]
    for call_index, call in enumerate(payload.get("tool_calls") or [], start=1):
        if not isinstance(call, dict):
            continue
        arguments = call.get("arguments")
        if not isinstance(arguments, dict):
            continue
        for item_index, invitation in enumerate(arguments.get("invitations") or [], start=1):
            if not isinstance(invitation, dict):
                continue
            item_id = f"tool_calls[{call_index}].arguments.invitations[{item_index}].message_text"
            if item_id in rewrites:
                invitation["message_text"] = rewrites[item_id]
        for item_index, draft in enumerate(arguments.get("drafts") or [], start=1):
            if not isinstance(draft, dict):
                continue
            item_id = f"tool_calls[{call_index}].arguments.drafts[{item_index}].message_text"
            if item_id in rewrites:
                draft["message_text"] = rewrites[item_id]
    return AgentAction.from_payload(payload)


def generation_feedback_tool_result(
    *,
    generation: CustomerVisibleTextGeneration,
    items: list[dict[str, Any]],
) -> ToolResult:
    return ToolResult(
        name="customer_visible_text_generation",
        called=True,
        allowed=True,
        result={
            "items": items,
            "reasoning_summary": generation.reasoning_summary,
            "item_rewrites": generation.item_rewrites,
        },
    )
