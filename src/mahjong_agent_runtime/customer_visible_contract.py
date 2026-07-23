from __future__ import annotations

import unicodedata


FORBIDDEN_CUSTOMER_SERVICE_PHRASES: tuple[str, ...] = (
    "为您",
    "请耐心等待",
    "是否方便",
    "是否加入",
    "要加入吗",
    "要不要加入",
    "要一起吗",
    "请问还有什么可以帮您",
)

FORBIDDEN_IMPLEMENTATION_IDENTITY_TERMS: tuple[str, ...] = (
    "AI",
    "ai",
    "Agent",
    "agent",
    "大模型",
    "机器人",
    "智能助手",
    "系统",
    "模型",
    "测试账号",
    "后台",
)

FORBIDDEN_INTERNAL_PROCESS_TERMS: tuple[str, ...] = (
    "工具",
    "trace",
    "traceId",
    "日志",
    "数据库",
    "prompt",
    "预算",
    "审批",
    "草稿",
)

INTERNAL_ENUM_EXAMPLES: tuple[str, ...] = (
    "asap_when_full",
    "pending_approval",
    "forming",
    "inviting",
    "hangzhou_mahjong",
)

PREFERRED_REQUESTER_CURRENT_GAME_PHRASES: tuple[str, ...] = (
    "可以不？",
    "可以吗？",
)

PREFERRED_CANDIDATE_INVITE_PHRASES: tuple[str, ...] = (
    "打吗？",
    "来吗？",
    "来不？",
)

PREFERRED_OPERATION_ACK_PHRASES: tuple[str, ...] = (
    "ok",
    "好的",
    "行",
    "好，我帮你问问。",
    "有消息跟你说。",
)

# These phrases assert that an external contact/delivery already happened. They
# are guarded by backend evidence instead of prompt wording because a model or
# reviewer can otherwise turn a pending draft into a false customer promise.
EXTERNAL_ACTION_COMPLETION_PHRASES: tuple[str, ...] = (
    "问了",
    "在问",
    "联系了",
    "在联系",
    "发了",
    "已发送",
    "通知了",
    "已通知",
    "邀了",
    "已邀",
)


def customer_visible_contract_snapshot() -> dict[str, tuple[str, ...]]:
    return {
        "forbidden_customer_service_phrases": FORBIDDEN_CUSTOMER_SERVICE_PHRASES,
        "forbidden_implementation_identity_terms": FORBIDDEN_IMPLEMENTATION_IDENTITY_TERMS,
        "forbidden_internal_process_terms": FORBIDDEN_INTERNAL_PROCESS_TERMS,
        "internal_enum_examples": INTERNAL_ENUM_EXAMPLES,
        "preferred_requester_current_game_phrases": PREFERRED_REQUESTER_CURRENT_GAME_PHRASES,
        "preferred_candidate_invite_phrases": PREFERRED_CANDIDATE_INVITE_PHRASES,
        "preferred_operation_ack_phrases": PREFERRED_OPERATION_ACK_PHRASES,
    }


def customer_visible_text_contract_violations(text: str) -> list[str]:
    content = str(text or "")
    compact_content = compact_customer_visible_text(content)
    checks = (
        ("customer_service_phrase", FORBIDDEN_CUSTOMER_SERVICE_PHRASES),
        ("implementation_identity_term", FORBIDDEN_IMPLEMENTATION_IDENTITY_TERMS),
        ("internal_process_term", FORBIDDEN_INTERNAL_PROCESS_TERMS),
        ("internal_enum", INTERNAL_ENUM_EXAMPLES),
    )
    violations: list[str] = []
    seen: set[tuple[str, str]] = set()
    for category, terms in checks:
        for term in terms:
            compact_term = compact_customer_visible_text(term)
            violation_key = (category, compact_term or term)
            if term and violation_key not in seen and (term in content or (compact_term and compact_term in compact_content)):
                seen.add(violation_key)
                violations.append(f"{category}:{term}")
    return violations


def customer_visible_action_claim_violations(
    text: str,
    action_evidence: dict[str, object] | None,
) -> list[str]:
    """Reject completed external-action claims without durable execution proof.

    The evidence is produced by the backend from ToolResults.  An absent
    evidence object means this contract is not applicable to the current item;
    a present object with ``contact_started=false`` means drafts may exist, but
    no customer has actually been contacted yet.
    """

    if action_evidence is None or bool(action_evidence.get("contact_started")):
        return []
    content = str(text or "")
    compact_content = compact_customer_visible_text(content)
    for phrase in EXTERNAL_ACTION_COMPLETION_PHRASES:
        compact_phrase = compact_customer_visible_text(phrase)
        if phrase in content or (compact_phrase and compact_phrase in compact_content):
            return [f"unverified_external_action:{phrase}"]
    return []


def compact_customer_visible_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return "".join(char for char in normalized if char.isalnum())
