from __future__ import annotations


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
