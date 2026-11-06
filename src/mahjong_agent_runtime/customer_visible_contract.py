"""定义客户可见文本的确定性安全合同，并校验回复中的事实与执行声明。"""

from __future__ import annotations

import unicodedata
import re

# 技术失败只说明尚未处理好；人工状态和具体故障留在内部，不能冒充已执行或承诺通知。
CUSTOMER_EXECUTION_FAILURE_REPLY = "这边暂时没处理好。"

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

# 稳定规则 internal_identifier：只枚举本系统的专用协议与提示源标识。
# 普通技术、办公名词不是秘密；是否在披露真实部署或他人资料由最小投影和语义审查判断。
# 命中后回灌规则要求重写，不删除业务事实；专用字段被无关文本带出也不能直接外发。
FORBIDDEN_INTERNAL_IDENTIFIERS: tuple[str, ...] = (
    "agent_runtime_system.md",
    "agent_runtime_reply_self_review.md",
    "task_context_id",
    "turn_tool_evidence",
    "previous_tool_results",
    "active_game_visible_summaries",
    "customer_visible_content_review",
    "create_invite_drafts",
    "record_candidate_reply",
)

# 稳定规则 internal_tool_identifier：当前正式工具的专用标识，不包含“工具/查询”等普通概念。
# 清单由测试与真实注册表逐项对齐；运行时不为文本检查实例化 Store 或导入业务执行器。
FORBIDDEN_INTERNAL_TOOL_NAMES: tuple[str, ...] = (
    "set_participation_deadline", "clear_participation_deadline", "schedule_follow_up", "cancel_follow_up",
    "check_room_availability", "reserve_room", "search_current_games", "search_customers",
    "register_waiting_demand", "cancel_waiting_demand", "create_game", "join_game", "create_invite_drafts",
    "update_game_requirement", "record_registered_member_withdrawal", "create_outbound_message_drafts",
    "record_candidate_reply", "update_game_status", "record_badcase", "record_user_memory", "update_context_checkpoint",
)

INTERNAL_ENUM_EXAMPLES: tuple[str, ...] = (
    "asap_when_full",
    "pending_approval",
    "hangzhou_mahjong",
)

# forming/inviting 也是普通英语，只在状态字段赋值协议中识别；不按裸子串封锁整个话题。
_INTERNAL_STATUS_PATTERN = re.compile(r"status[\"']*[:=][\"']*(?:forming|inviting)(?![a-z])", re.IGNORECASE)

# 稳定规则 internal_response_source.identifier / .field：新回灌来源仅用于审计，
# 不能被模型抄成客户回复。限定后端 ID 格式和专用字段，避免误杀普通邀约、时间或人数。
_RESPONSE_SOURCE_ID_PATTERN = re.compile(r"llm(?:call|attempt)[0-9a-f]{32}")
_RESPONSE_SOURCE_FIELDS = ("response_source", "terminal_attempt_id", "raw_response_sha256", "http_response_complete")
# 稳定规则 internal_task_fact_feedback.field：只拦内部纠错字段，不以来源词汇判断业务意图。
# 误杀处理是重写不含内部字段的对客文本；不修改原时间、档位、人数或审计证据。
_TASK_FACT_FEEDBACK_FIELDS = ("reference_fact_sources", "validation_error")
# 稳定规则 internal_recruitment_metadata.identifier：新权限、方案关联及事实范围仅供内部执行。
# 限定专用标识，不过滤“两个方案/报名/退出”等业务词；误杀时回喂重写，不修改人数或时间。
_RECRUITMENT_INTERNAL_MARKERS = (
    "registered_by_customer_id", "option_root_game_id", "alternative_to_game_id",
    "record_registered_member_withdrawal", "bound_assertions", "binding_call_ids",
    "scope.call_id", "scope.game_id",
)

# 稳定规则 credential.*：识别凭证格式而非讨论凭证的意图，不回显匹配值。
# 统一全半角并移除空白/零宽字符后核验，防止逐字分隔绕过；正常“API key如何保管”不命中。
# 不解码任意文本或猜测普通长字符串；其他秘密仍须由上下文最小化与语义审查阻断。
_CREDENTIAL_PATTERNS = (
    ("credential.api_key", re.compile(r"sk-[a-z0-9_-]{16,}", re.IGNORECASE)),
    ("credential.authorization", re.compile(r"authorization[\"']*[:=][\"']*bearer[a-z0-9._~+/-]{16,}={0,2}", re.IGNORECASE)),
    ("credential.assigned_secret", re.compile(
        r"(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret|cookie)[\"']*[:=][\"']*[a-z0-9._~+/-]{8,}",
        re.IGNORECASE)),
    ("credential.private_key", re.compile(r"-----begin(?:rsa|ec|openssh)?privatekey-----", re.IGNORECASE)),
    ("credential.cookie", re.compile(
        r"(?:set-)?cookie[\"']*[:=][\"']*(?:[a-z0-9_-]+=[a-z0-9._~+/%=-]*;)*[a-z0-9_-]+=[a-z0-9._~+/%=-]{8,}",
        re.IGNORECASE)),
)
# 独立Bearer凭证保留值中的分词边界，不把“Bearer tokens are ...”整句合并成假token。
# 标签可逐字分隔，值可全半角或夹零宽；完整Authorization头另用上面的协议规则核验。
_BEARER_VALUE_PATTERN = re.compile(
    r"(?<![a-z0-9])b[ \t]*e[ \t]*a[ \t]*r[ \t]*e[ \t]*r[ \t]+[a-z0-9._~+/-]{16,}={0,2}(?![a-z0-9._~+/-])",
    re.IGNORECASE,
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
# 稳定规则 unverified_external_action 仅触发输出证据核验，不分类用户意图，不声称穷尽自然语言；
# 主体与语境由独立审查解释、后端核对来源，不能靠增加人称关键词来放行整句。
EXTERNAL_ACTION_COMPLETION_PHRASES: tuple[str, ...] = (
    "问了",
    "问过了",
    "在问",
    "联系了",
    "在联系",
    "发了",
    "发过消息",
    "已发送",
    "通知了",
    "已通知",
    "邀了",
    "已邀",
)


def customer_visible_contract_snapshot() -> dict[str, tuple[str, ...]]:
    """向各审查入口提供同一明确风险清单，普通技术概念不再作为禁词传播。"""
    return {
        "forbidden_customer_service_phrases": FORBIDDEN_CUSTOMER_SERVICE_PHRASES,
        "forbidden_internal_identifiers": FORBIDDEN_INTERNAL_IDENTIFIERS,
        "forbidden_internal_tool_names": FORBIDDEN_INTERNAL_TOOL_NAMES,
        "credential_rule_ids": tuple(rule for rule, _ in _CREDENTIAL_PATTERNS),
        "internal_enum_examples": INTERNAL_ENUM_EXAMPLES,
        "preferred_requester_current_game_phrases": PREFERRED_REQUESTER_CURRENT_GAME_PHRASES,
        "preferred_candidate_invite_phrases": PREFERRED_CANDIDATE_INVITE_PHRASES,
        "preferred_operation_ack_phrases": PREFERRED_OPERATION_ACK_PHRASES,
    }


def customer_visible_text_contract_violations(text: str) -> list[str]:
    """枚举可见文本的内部信息风险；只返回规则证据，不改写业务事实或代替意图判断。"""
    content = str(text or "")
    compact_content = compact_customer_visible_text(content)
    identifier_content = unicodedata.normalize("NFKC", content).casefold()
    checks = (
        ("customer_service_phrase", FORBIDDEN_CUSTOMER_SERVICE_PHRASES),
        ("internal_identifier", FORBIDDEN_INTERNAL_IDENTIFIERS),
        ("internal_tool_identifier", FORBIDDEN_INTERNAL_TOOL_NAMES),
        ("internal_enum", INTERNAL_ENUM_EXAMPLES),
    )
    violations: list[str] = []
    seen: set[tuple[str, str]] = set()
    for category, terms in checks:
        for term in terms:
            compact_term = compact_customer_visible_text(term)
            violation_key = (category, compact_term or term)
            matched = term in content or (compact_term and compact_term in compact_content)
            if category == "internal_tool_identifier":
                # 工具标识按完整协议词匹配；“create games”等普通英语句子不因压缩后含子串误杀。
                matched = bool(re.search(r"(?<![a-z0-9_])" + re.escape(term) + r"(?![a-z0-9_])", identifier_content)
                    or re.search(r"(?<![a-z0-9])" + re.escape(compact_term) + r"(?![a-z0-9])", compact_content))
            if term and violation_key not in seen and matched:
                seen.add(violation_key)
                violations.append(f"{category}:{term}")
    if _RESPONSE_SOURCE_ID_PATTERN.search(compact_content):
        violations.append("internal_response_source.identifier")
    if any(compact_customer_visible_text(field) in compact_content for field in _RESPONSE_SOURCE_FIELDS):
        violations.append("internal_response_source.field")
    if any(compact_customer_visible_text(field) in compact_content for field in _TASK_FACT_FEEDBACK_FIELDS):
        violations.append("internal_task_fact_feedback.field")
    if any(compact_customer_visible_text(marker) in compact_content for marker in _RECRUITMENT_INTERNAL_MARKERS):
        violations.append("internal_recruitment_metadata.identifier")
    normalized_content = "".join(char for char in unicodedata.normalize("NFKC", content)
                                 if unicodedata.category(char) != "Cf")
    credential_content = "".join(char for char in normalized_content if not char.isspace())
    if _INTERNAL_STATUS_PATTERN.search(credential_content):
        violations.append("internal_enum.status")
    violations.extend(rule for rule, pattern in _CREDENTIAL_PATTERNS if pattern.search(credential_content))
    if _BEARER_VALUE_PATTERN.search(normalized_content) and "credential.authorization" not in violations:
        violations.append("credential.authorization")
    return violations


def customer_visible_action_claim_violations(
    text: str,
    action_evidence: dict[str, object] | None,
    *,
    attributed_claim_spans: list[tuple[int, int]] | None = None,
) -> list[str]:
    """拒绝缺乏持久执行证明的已完成外部动作声明。

    缺少证据同样不证明已经发送。归属例外只能由审查后端验证原话、主体及来源后传入
    原文[start,end)；此函数核对精确覆盖范围，不自行解析人称或把整句列入白名单。
    同一回复另一处未获证明的动作仍须阻断，隐私与专用标识规则不受此例外影响。
    """

    if isinstance(action_evidence, dict) and action_evidence.get("contact_started") is True:
        return []
    content = str(text or "")
    supplied_spans = attributed_claim_spans if isinstance(attributed_claim_spans, list) else []
    spans = [tuple(span) for span in supplied_spans
        if isinstance(span, (tuple, list)) and len(span) == 2
        and all(type(value) is int for value in span) and 0 <= span[0] < span[1] <= len(content)]
    # 动作词均为中文；逐字符规范化并映射原文位置，确保分隔混淆匹配也只能由完整span豁免。
    characters, positions = [], []
    for index, character in enumerate(content):
        for normalized in unicodedata.normalize("NFKC", character).casefold():
            if normalized.isalnum():
                characters.append(normalized)
                positions.append(index)
    compact_content = "".join(characters)
    for phrase in EXTERNAL_ACTION_COMPLETION_PHRASES:
        compact_phrase = compact_customer_visible_text(phrase)
        offset = compact_content.find(compact_phrase)
        while offset >= 0:
            start, end = positions[offset], positions[offset + len(compact_phrase) - 1] + 1
            if not any(left <= start and end <= right for left, right in spans):
                return [f"unverified_external_action:{phrase}"]
            offset = compact_content.find(compact_phrase, offset + 1)
    return []


def compact_customer_visible_text(text: str) -> str:
    """压缩客户客户可见text，保留决策所需字段并限制载荷大小。"""
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return "".join(char for char in normalized if char.isalnum())
