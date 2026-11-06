"""用同客户原话验证动作主体引用；不推断意图，也不替 Agent 制造外发证据。"""

from __future__ import annotations

from typing import Any

from .models import UserMessage


_CLAUSE_SEPARATORS = frozenset("，,。.!！?？;；\n\r")
_ATTRIBUTION_FIELDS = frozenset({
    "reviewed_text", "claim_span", "actor", "actor_span", "source_ref",
    "source_quote", "source_claim_span", "source_actor_span",
})


def build_action_attribution_context(
    message: UserMessage, context_payload: dict[str, Any], *, enabled: bool,
) -> dict[str, Any]:
    """只为本人私聊回复投影本人原话；群输出及系统事件不能获得引用豁免。

    历史池来自已隔离的会话上下文，再按作者核验。源记录不携带私有备注、工具、
    其他客户原话或任意模型摘要。引用证明只支持本人首个完整分句的精确人称替换。
    """
    metadata = message.metadata if isinstance(message.metadata, dict) else {}
    if (not enabled or metadata.get("room_id")
            or metadata.get("input_source") == "system_trigger"):
        return {}
    sources = [{"source_ref": "current_message", "author_id": message.sender_id, "text": message.text}]
    for index, turn in enumerate(list(context_payload.get("recent_conversation") or [])[-8:]):
        if (not isinstance(turn, dict) or turn.get("role") != "user"
                or turn.get("sender_id") != message.sender_id
                or turn.get("conversation_id", message.conversation_id) != message.conversation_id
                or not isinstance(turn.get("content"), str)):
            continue
        sources.append({"source_ref": f"recent_user_{index}",
                        "author_id": message.sender_id, "text": turn["content"]})
    # 不使用 sender_name，它可能来自渠道私有备注；本轮仅支持无歧义的人称替换。
    # 即使登记公开名为“我”或“老板”，也不能借名称把 Agent 的自述变成本人动作。
    return {"current_customer_id": message.sender_id, "sources": sources,
            "target_actor_literals": ["你"], "source_actor_literals": ["我"]}


def _span(value: Any, text: str) -> tuple[int, int] | None:
    """只接受 Unicode 原文偏移，不接受布尔、负数或越界的伪造范围。"""
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(index) is not int for index in value)):
        return None
    start, end = value
    return (start, end) if 0 <= start < end <= len(text) else None


def _complete_clause(text: str, span: tuple[int, int]) -> bool:
    """引用覆盖完整分句，禁止只圈住动作词而排除主体或相邻独立动作。"""
    start, end = span
    return (start == 0 or text[start - 1] in _CLAUSE_SEPARATORS) and (
        end == len(text) or text[end] in _CLAUSE_SEPARATORS
    ) and not any(char in _CLAUSE_SEPARATORS for char in text[start:end])


def verified_action_attribution_spans(
    text: str, attributions: Any, *, recipient_id: str,
    attribution_context: dict[str, Any] | None,
) -> tuple[list[tuple[int, int]], list[str]]:
    """核验精确引用后才返回可豁免的原文范围，不能信任模型自报主体。

    当前仅支持“本人原话首分句：我＋原谓语 → 给本人的回复：你＋同一谓语”。
    名称别名、第三方、自由转述及嵌套人称没有本协议的独立身份凭据，
    本轮保持拒绝而交主模型自然重写。此窄协议不是自然语言主体分类器。
    """
    if attributions is None or attributions == []:
        return [], []
    invalid = ([], ["unverified_external_action_attribution"])
    context = attribution_context if isinstance(attribution_context, dict) else {}
    if (not isinstance(attributions, list) or not context.get("current_customer_id")
            or recipient_id != context["current_customer_id"]):
        return invalid
    sources = {source.get("source_ref"): source for source in context.get("sources", [])
               if isinstance(source, dict) and isinstance(source.get("source_ref"), str)}
    verified: list[tuple[int, int]] = []
    for attribution in attributions:
        if (not isinstance(attribution, dict) or set(attribution) != _ATTRIBUTION_FIELDS
                or attribution.get("reviewed_text") != text
                or attribution.get("actor") != "current_customer"
                or not isinstance(attribution.get("source_ref"), str)):
            return invalid
        source = sources.get(attribution["source_ref"])
        if (not source or source.get("author_id") != recipient_id
                or not isinstance(source.get("text"), str)):
            return invalid
        source_text = source["text"]
        claim = _span(attribution.get("claim_span"), text)
        actor = _span(attribution.get("actor_span"), text)
        source_claim = _span(attribution.get("source_claim_span"), source_text)
        source_actor = _span(attribution.get("source_actor_span"), source_text)
        if not all((claim, actor, source_claim, source_actor)):
            return invalid
        assert claim and actor and source_claim and source_actor
        if (not _complete_clause(text, claim) or not _complete_clause(source_text, source_claim)
                or source_claim[0] != 0 or actor[0] != claim[0]
                or source_actor[0] != source_claim[0] or actor[1] >= claim[1]
                or source_actor[1] >= source_claim[1]
                or text[actor[0]:actor[1]] not in context.get("target_actor_literals", [])
                or source_text[source_actor[0]:source_actor[1]] not in context.get("source_actor_literals", [])
                or attribution.get("source_quote") != source_text[source_claim[0]:source_claim[1]]):
            return invalid
        predicate = text[actor[1]:claim[1]]
        if predicate != source_text[source_actor[1]:source_claim[1]]:
            return invalid
        # 精确人称替换协议不支持残余“我”或引号包裹的嵌套说话者；否则同字谓语
        # 仍可能把来源中的本人错当回复中的 Agent。这里只拒绝歧义，不猜测主体。
        if any(marker in predicate for marker in ("我", "本人", "“", "”", '"', "‘", "’", "「", "」", "：", ":")):
            return invalid
        verified.append(claim)
    return verified, []


def action_attribution_contract() -> dict[str, Any]:
    """以明确 JSON 类型声明引用证明，避免身份枚举和偏移表示被自然语言误读。

    此 schema 只描述同次审查输出，后端仍按原严格引用校验，不把客户 ID 自动
    转换成主体枚举，也不把区间字符串转换成数组。示例只说明协议，不提供本轮事实。
    """
    span_schema = {
        "type": "array", "minItems": 2, "maxItems": 2,
        "items": {"type": "integer", "minimum": 0},
        "description": "Exactly two JSON integers: [start_inclusive, end_exclusive]. Never a string. Count Unicode code points in the original text.",
    }
    properties = {
        "reviewed_text": {"type": "string", "minLength": 1,
                          "description": "Exact complete suggested_safe_text for this review item."},
        "actor": {"type": "string", "enum": ["current_customer"],
                  "description": "Fixed literal current_customer; never substitute a customer ID, source author_id, nickname, or another actor label."},
        "source_ref": {"type": "string", "minLength": 1,
                       "description": "Exact source_ref from external_action_attribution_context.sources."},
        "source_quote": {"type": "string", "minLength": 1,
                         "description": "Exact source text slice selected by source_claim_span, excluding its trailing punctuation."},
        **{field: dict(span_schema) for field in (
            "claim_span", "actor_span", "source_claim_span", "source_actor_span",
        )},
    }
    return {
        "optional_item_field": "external_action_attributions",
        "json_schema": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": False,
                      "required": sorted(_ATTRIBUTION_FIELDS), "properties": properties},
        },
        "span_encoding": {
            "json_type": "array", "length": 2, "element_type": "integer",
            "start_inclusive": True, "end_exclusive": True,
            "offset_unit": "Unicode code point", "valid_example": [0, 1],
            "invalid_examples": ["[0,1)", "[0,1]", ["0", "1"], {"start": 0, "end": 1}],
        },
        "rules": [
            "Only when an apparent completed external action belongs to the current customer, supply attribution on that item; unrelated text needs no field.",
            "Emit the fixed JSON string actor=\"current_customer\" even if current_customer_id or source author_id has a different value. All four *_span fields must be JSON arrays containing two integers, for example [0,1], never interval strings.",
            "Use external_action_attribution_context sources only. Bind reviewed_text to the complete suggested_safe_text and source_ref to one trusted source.",
            "actor must be current_customer. claim_span covers a complete punctuation-delimited clause; actor_span is its initial 你. source_claim_span covers the first complete clause of that source; source_actor_span is its initial 我. source_quote equals that source clause exactly.",
            "After replacing only the actor, the rest of both clauses must be exactly identical. Nested self-reference, quoted speakers, third parties and free paraphrases are not supported by this narrow proof. Do not fabricate proofs; rewrite naturally within the current topic if needed.",
            "No attribution can prove an Agent action, enable another recipient or public group, waive privacy checks, or cover a second independent clause. Agent completed actions still require actual backend delivery evidence.",
        ],
        "format_example_only": {
            "instruction": "This camping example demonstrates JSON types only; do not copy its facts into the current reply.",
            "source_text": "我已经联系了露营领队，等他回复就好。",
            "item_review": {
                "item_id": "reply_to_user", "approved": True,
                "suggested_safe_text": "你已经联系了露营领队，等他回复就好。",
                "reasoning_summary": "按本人原话核验动作归属，没有声称 Agent 外发。", "violations": [],
                "external_action_attributions": [{
                    "reviewed_text": "你已经联系了露营领队，等他回复就好。",
                    "actor": "current_customer", "claim_span": [0, 10], "actor_span": [0, 1],
                    "source_ref": "current_message", "source_quote": "我已经联系了露营领队",
                    "source_claim_span": [0, 10], "source_actor_span": [0, 1],
                }],
            },
        },
    }
