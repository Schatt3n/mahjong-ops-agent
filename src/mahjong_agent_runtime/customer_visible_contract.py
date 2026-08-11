"""定义客户可见文本的确定性安全合同，并校验回复中的事实与执行声明。"""

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

# 这些短语声称外部联系或发送已经发生，必须由后端执行证据约束，
# 不能只依赖提示词；否则生成模型或审查模型可能把待发草稿变成虚假承诺。
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
    """拒绝缺乏持久执行证明的已完成外部动作声明。

    证据由后端从 ToolResults 生成。证据对象缺失表示此契约不适用于当前项目；若存在对象且 ``contact_started=false``，则可能存在草稿，但尚未实际联系客户。
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
    """压缩客户客户可见text，保留决策所需字段并限制载荷大小。"""
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return "".join(char for char in normalized if char.isalnum())
