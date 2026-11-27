"""定义客户可见文本的确定性安全合同，并校验回复中的事实与执行声明。"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Literal, TypedDict

from .knowledge import default_terminology_repository
from .models import DEFAULT_TZ

# 技术失败只说明尚未处理好；人工状态和具体故障留在内部，不能冒充已执行或承诺通知。
CUSTOMER_EXECUTION_FAILURE_REPLY = "稍等一下"

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
    "active_waiting_demands",
    "follow_up_commitment_evidence",
    "follow_up_commitments",
    "delivered_game_offer",
    "game_offer_candidate",
    "reviewed_game_offer_id",
    "reviewed_game_reference_id",
    "game_reference_only",
    "pending_late_arrival_decisions",
    "current_late_arrival_requests",
    "late_arrival_request",
    "late_arrival_decision",
    "remaining_play_minutes",
    "seat_claim_evidence",
    "seat_registration_evidence",
    "same_table_membership_evidence",
    "table_duration_fit_evidence",
    "remaining_seats_display",
    "assigned_room",
    "table_availability_evidence",
    "exploratory_game_type",
    "pending_board_candidate_decisions",
    "board_candidate_reply_evidence",
    "decision_task_id",
    "evidence_message_id",
    "board-candidate-decision:",
    "board-candidate-update:",
    "customer_visible_content_review",
    "create_invite_drafts",
    "record_candidate_reply",
)

# 稳定规则 internal_tool_identifier：当前正式工具的专用标识，不包含“工具/查询”等普通概念。
# 清单由测试与真实注册表逐项对齐；运行时不为文本检查实例化 Store 或导入业务执行器。
FORBIDDEN_INTERNAL_TOOL_NAMES: tuple[str, ...] = (
    "set_participation_deadline", "clear_participation_deadline", "schedule_follow_up", "cancel_follow_up",
    "request_board_stake_confirmation", "request_board_candidate_confirmation", "record_board_candidate_decision",
    "request_late_arrival", "record_late_arrival_decision",
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
)

# 稳定规则 seat_count.inconsistent_pair 只检查同一句中带时刻的明确牌局缺口和紧邻的重复计数。
# 比较两张桌、转述、条件以及单独的缺口表述仍须按语义和实际局况审查；命中只拒绝，不代改人数。
_SEAT_COUNT_PAIR_PATTERN = re.compile(
    r"(?:^|[,，。;；!！\n])\s*(?:[01]?\d|2[0-3]):[0-5]\d\s*(?:这桌|那桌)?\s*"
    r"(?P<shorthand>173|272|371|一缺三|二缺二|两缺两|三缺一)\s*[,，、]\s*"
    r"还差\s*(?P<count>[一二两三123])\s*(?:个?人|位)\s*(?=[,，。;；!！\n]|$)"
)
_REMAINING_BY_SEAT_LABEL = {
    "173": 3, "272": 2, "371": 1, "一缺三": 3, "二缺二": 2, "两缺两": 2, "三缺一": 1,
    "一": 1, "二": 2, "两": 2, "三": 3, "1": 1, "2": 2, "3": 3,
}

# 稳定规则 seat_count.current_table_mismatch 只核对明确绑定到当前一桌或逐桌列举的缺口。
# 明确的时间、桌况字段和人数协议可绑定；其他表达留给语义审查，拒绝时不自动更改客户话术。
_SEAT_LABEL = r"173|272|371|一缺三|二缺二|两缺两|三缺一"
_SEAT_LABEL_PATTERN = re.compile(_SEAT_LABEL)
_SEAT_CURRENT_PREFIX = r"(?:(?:目前|现在)\s*)?"
_SEAT_CURRENT_LABEL = re.compile(rf"{_SEAT_CURRENT_PREFIX}(?P<label>{_SEAT_LABEL})")
_SEAT_PEOPLE_UNIT = r"(?:个\s*(?:人|位)?|人|位|席)"
_SEAT_MENTION_NUMBER = r"(?:[零一二两三四五六七八九十]+|\d+)"
# 只用于区别没有已知人数标记与有标记但无法绑定；命中本身不给任何业务结论。
_SEAT_CLAIM_CUE = re.compile(
    r"(?<!\d)(?<!\d\.)(?:173|272|371)(?!\d|\.\d)|[一二两三]缺[一二两三]|"
    rf"(?:缺|差|占|剩|坐)\s*{_SEAT_MENTION_NUMBER}\s*{_SEAT_PEOPLE_UNIT}|"
    rf"{_SEAT_MENTION_NUMBER}\s*(?:个\s*)?(?:人|位|席)|满座|满员"
)
_SEAT_LIST_TABLE = r"(?:一|1|这|那|该)(?:桌|局)"
_SEAT_LIST_OPEN = r"开(?:场|打)?"
_SEAT_CLAIM_QUOTE_MARKS = frozenset(("“", "”", "「", "」", "『", "』", "‘", "’", '"', "'"))
_SEAT_HOUR_LABELS = {
    **dict(zip("零一二三四五六七八九十", range(11))),
    "两": 2,
    **dict(zip("十一 十二 十三 十四 十五 十六 十七 十八 十九 二十 二十一 二十二 二十三".split(), range(11, 24))),
}
_SEAT_CLAIM_CLOCK = re.compile(
    r"(?:(?:这|那|该)(?:桌|局)\s*)?(?:(?P<date>\d{4}-\d{2}-\d{2})[ T]\s*)?"
    r"(?:(?P<hour>[01]?\d|2[0-3])\s*:\s*(?P<minute>[0-5]\d)(?!\d)|"
    r"(?:(?P<day_part>早上|上午|下午|傍晚|晚上)\s*)?"
    r"(?P<spoken_hour>[01]?\d|2[0-3]|[零一二两三四五六七八九十]{1,3})\s*点"
    rf"(?:(?P<part>半|一刻|三刻)|(?P<spoken_minute>[0-5]?\d)分|"
    rf"(?P<bare_minute>[0-5]\d)(?=\s*(?:{_SEAT_LIST_TABLE}|{_SEAT_LIST_OPEN}|的|$)))?)"
)
# 只拦已绑定同桌且独立陈述开打的语法。问句、转述、其他桌及未来推测留给语义审查。
_DEFINITE_TABLE_START = re.compile(
    r"(?:^|[,。;!\n])\s*(?:这桌|那桌|人家|他们)\s*(?:已经|已)\s*"
    r"(?:开打|开始打|开局)(?:了)?\s*(?=[,。;!\n]|$)"
)
_START_CLAIM_OTHER_SCOPE = re.compile(
    r"如果|假如|万一|假设|可能|估计|恐怕|说不定|到时|届时|等你到|另一桌|别桌|隔壁|其他桌|"
    r"听说|据说|你说|他说|你是说|不能说|不代表"
)
# 玩法只允许按已有核验术语精确绑定，不用子串推断另一桌或门店自定义玩法。
_SEAT_GAME_TYPES = {
    alias: term.canonical["requested_game"]
    for term in default_terminology_repository().terms
    if term.category == "game_type" and term.confidence == "verified"
    for alias in (*term.aliases, term.canonical["requested_game"], term.canonical["game_type"])
}
_SEAT_GAME_LABEL = "|".join(re.escape(alias) for alias in sorted(_SEAT_GAME_TYPES, key=len, reverse=True))
_SEAT_CLAIM_ATTRIBUTE_FORM = (
    r"(?:(?:底(?:注)?\s*)?\d+(?:\.\d+)?"
    r"(?:\s*(?:到|至|[-~—])\s*\d+(?:\.\d+)?)?\s*(?:块|元|底)?\s*(?:无烟|有烟|禁烟)?|"
    r"无烟|有烟|禁烟|烟都可)"
)
_SEAT_CLAIM_ATTRIBUTE = re.compile(_SEAT_CLAIM_ATTRIBUTE_FORM)
_SEAT_CLAIM_INVITE = re.compile(r"(?:你\s*)?(?:打|来|可以)(?:吗|不)")
_SEAT_COLLECTIVE_STAKE_PAIR = r"(?:\d+(?:\.\d+)?\s*(?:块|元)\s*){2}"
_SEAT_COUNTED_STAKE = (
    rf"(?:{_SEAT_COLLECTIVE_STAKE_PAIR}|(?:底(?:注)?\s*)?\d+(?:\.\d+)?"
    r"(?:\s*(?:到|至|[-~—])\s*\d+(?:\.\d+)?)?\s*(?:块|元|底))"
)
_SEAT_CLAIM_HEADING = re.compile(
    rf"(?:(?:(?:这|那|该)(?:一)?|有(?:一|1))(?:桌|局)\s*)?(?:{_SEAT_LIST_OPEN}\s*)?"
    rf"(?:的\s*)?(?:{_SEAT_COUNTED_STAKE}\s*)?(?:(?P<game_type>{_SEAT_GAME_LABEL})\s*)?"
    rf"(?:(?:还在凑人?|仍在凑人?|正在凑人?|目前|现在)\s*)?(?P<label>{_SEAT_LABEL})?"
)
_SEAT_LIST_HEADING = re.compile(
    rf"(?:(?:老板好|你好|您好)\s*,\s*)?(?:有|查到)\s*"
    rf"(?:(?P<count>[两二三四五六七八九十]|[2-9]|10)\s*(?:桌|局)\s*"
    rf"(?:(?P<counted_type>{_SEAT_GAME_LABEL})\s*)?(?:{_SEAT_COUNTED_STAKE}\s*)?"
    rf"(?:的\s*)?(?P<heading_label>{_SEAT_LABEL})?(?:\s*(?:可(?:以)?|能)拼)?|"
    rf"(?:(?P<game_type>{_SEAT_GAME_LABEL})\s*)?(?:{_SEAT_CLAIM_ATTRIBUTE_FORM}\s*)?"
    r"(?:的\s*)?(?:现成)?(?:桌|局))\s*[:,]"
)
_SEAT_LIST_PREFIX = re.compile(rf"{_SEAT_LIST_TABLE}\s*")
_SEAT_DURATION_FORM = r"(?:打\s*)?(?:\d+(?:\.\d+)?|[一二两三四五六七八九十])\s*个?\s*(?:小时|钟头)"
_SEAT_LIST_SUFFIX = re.compile(
    rf"(?:{_SEAT_LIST_TABLE}\s*)?(?:{_SEAT_LIST_OPEN}\s*(?:{_SEAT_DURATION_FORM})?)?"
)
_SEAT_LIST_EACH_TABLE = re.compile(r"\s*各\s*(?:一|1)\s*(?:桌|局)$")
_SEAT_LIST_CLAIM = re.compile(r"\s*(?:这几(?:桌|局)\s*)?(?:都(?:是)?|每(?:桌|局)(?:都)?是)\s*(?P<fields>.+)$")
_SEAT_ROW_SHARED_CLAIM = re.compile(rf"(?:^|(?<=[,、])){_SEAT_LIST_CLAIM.pattern}")
_SEAT_LIST_MISSING_PEOPLE = re.compile(
    rf"{_SEAT_CURRENT_PREFIX}(?:还\s*)?(?:差|缺)\s*(?P<count>[一二两三123])\s*{_SEAT_PEOPLE_UNIT}"
)
_SEAT_LIST_CHOICE = re.compile(r"(?:你\s*)?(?:想\s*|要\s*)?(?:打|报|选|赶)\s*哪(?:一)?(?:桌|局)")
# 附带属性仅限定逐桌或共同字段的边界，不把报价和时长当作新的余位证据。
_SEAT_LIST_ATTRIBUTE = re.compile(
    rf"{_SEAT_CLAIM_ATTRIBUTE_FORM}|{_SEAT_COLLECTIVE_STAKE_PAIR}"
    rf"|(?:每(?:桌|局)\s*)?{_SEAT_DURATION_FORM}"
)
_SEAT_LIST_GAME_TERMS = re.compile(rf"(?P<game_type>{_SEAT_GAME_LABEL})(?:\s*{_SEAT_COUNTED_STAKE})?")

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


def credential_text_contract_violations(text: str) -> list[str]:
    """复用凭证格式规则，便于对客封禁和已鉴权工作记录脱敏使用同一口径。"""
    normalized = "".join(char for char in unicodedata.normalize("NFKC", text)
                         if unicodedata.category(char) != "Cf")
    compact = "".join(char for char in normalized if not char.isspace())
    violations = [rule for rule, pattern in _CREDENTIAL_PATTERNS if pattern.search(compact)]
    if _BEARER_VALUE_PATTERN.search(normalized) and "credential.authorization" not in violations:
        violations.append("credential.authorization")
    return violations


def customer_visible_text_contract_violations(text: str) -> list[str]:
    """枚举内部信息风险和明确协议的文本矛盾；只返回证据，不改写或推断业务事实。"""
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
    status_content = "".join(char for char in identifier_content
                             if not char.isspace() and unicodedata.category(char) != "Cf")
    if _INTERNAL_STATUS_PATTERN.search(status_content):
        violations.append("internal_enum.status")
    if any(
        _REMAINING_BY_SEAT_LABEL[match["shorthand"]] != _REMAINING_BY_SEAT_LABEL[match["count"]]
        for match in _SEAT_COUNT_PAIR_PATTERN.finditer(identifier_content)
    ):
        violations.append("seat_count.inconsistent_pair")
    violations.extend(credential_text_contract_violations(content))
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


_SEAT_REGISTRATION_ACK = re.compile(
    r"(?:^|[。.!！?？\n])\s*(?:好(?:的|嘞)?\s*[，,]\s*)?(?:我\s*)?(?:先\s*)?(?:给|帮)你(?:先)?\s*"
    r"(?:记上|报上|留上|留好|登记上)了(?=\s*(?:[，,。.!！；;]|$))"
)


def customer_visible_completed_seat_ack(text: str) -> bool:
    """识别回复中不带其它对象的报名完成短句；含条件和时间收讫的说法留给语义审查。"""
    content = unicodedata.normalize("NFKC", text.strip())
    return not _SEAT_CLAIM_QUOTE_MARKS.intersection(content) and bool(_SEAT_REGISTRATION_ACK.search(content))


class VerifiedSeatClaim(TypedDict):
    """一项已绑定的四人桌声明及权威数字，不保留原文或任何参与者身份。"""

    game_id: str | None
    planned_start_at: str | None
    source_span: tuple[int, int]
    seats_total: int
    declared_claimed_seats: int
    declared_remaining_seats: int
    actual_claimed_seats: int
    actual_remaining_seats: int
    consistent: bool


class SeatClaimAssessment(TypedDict):
    """确定性核验仅覆盖约定的人数语法，独立语义结论仍由调用方单独保存。"""

    status: Literal["absent", "unparsed", "matched", "mismatch"]
    verified_claims: list[VerifiedSeatClaim]


def customer_visible_seat_claim_evidence(text: str, facts: list[dict[str, object]]) -> SeatClaimAssessment:
    """逐桌核验回复声明；source_span 指向 NFKC 文本中的整个已绑定陈述。

    absent 表示没有已知人数标记，matched 表示所有已知标记均落入完整绑定的正确声明。
    缺证、引语或未完整绑定时返回 unparsed，可同时包含已经核验的其他声明。
    一处证实的错报即为 mismatch，不能由另一处正确声明或未决片段覆盖。
    """
    content = unicodedata.normalize("NFKC", text)
    markers = list(_SEAT_CLAIM_CUE.finditer(content))
    if not facts or _SEAT_CLAIM_QUOTE_MARKS.intersection(content):
        return {"status": "unparsed" if markers else "absent", "verified_claims": []}
    verified: list[VerifiedSeatClaim] = []
    for sentence in re.finditer(r"([^。!?]+)([。!?]|$)", content):
        body = sentence[1].strip()
        start = sentence.start(1) + len(sentence[1]) - len(sentence[1].lstrip())
        statements = re.split(r";|\n", body)
        listed = _current_listed_seat_declarations(statements, question=sentence[2] == "?")
        if listed is not None:
            tables, declared_counts = listed
            verified.extend(_verified_listed_seats(tables, declared_counts, facts, (start, start + len(body))))
            continue
        for index, statement in enumerate(statements):
            span = (start, start + len(statement))
            start = span[1] + 1
            question = index == len(statements) - 1 and sentence[2] == "?"
            collective = _current_collective_seat_declaration(statement, question=question)
            if collective is not None:
                clocks, declared_counts, game_type = collective
                verified.extend(_verified_listed_seats(
                    [(clock, game_type) for clock in clocks], [declared_counts] * len(clocks), facts, span,
                ))
                continue
            declaration = _current_seat_declaration(statement, question=question)
            # 分号可连接同一句的条件或例子。上一分句不属于事实协议时，不猜下一分句是否仍有作用域。
            if declaration is None:
                break
            clock, declared_remaining, game_type = declaration
            # 无钟点时只核对回复开头直接报出的缺口，避免把后文例子接到恰好唯一的候选桌。
            if clock is None and (sentence.start() != 0 or index != 0):
                continue
            if declared_remaining is not None:
                verified.extend(_verified_listed_seats([(clock, game_type)], [[declared_remaining]], facts, span))
    if any(not claim["consistent"] for claim in verified):
        return {"status": "mismatch", "verified_claims": verified}
    if not markers and not verified:
        return {"status": "absent", "verified_claims": []}
    covered = {claim["source_span"] for claim in verified if claim["game_id"] is not None}
    if covered and all(any(left <= marker.start() and marker.end() <= right for left, right in covered)
                       for marker in markers):
        return {"status": "matched", "verified_claims": verified}
    return {"status": "unparsed", "verified_claims": verified}


def customer_visible_seat_claim_violations(text: str, facts: list[dict[str, object]]) -> list[str]:
    """继续用同一权威核验拦截已证实错报；无确定结论则交给语义审查。"""
    evidence = customer_visible_seat_claim_evidence(text, facts)
    return ["seat_count.current_table_mismatch"] if evidence["status"] == "mismatch" else []


_CURRENT_OWN_SEAT_DECLARATION = re.compile(
    r"(?:(?:现在|目前)\s*(?:还是|只有|就(?:是)?)?|只有|就(?:是)?)\s*你\s*(?:[一1]个(?:人)?|[一1]位)\s*[,、]\s*"
    r"(?:还|仍)?\s*(?:差|缺)\s*[一二两三123]\s*(?:个(?:人)?|位|人)(?=\s*[,。;!\n]|\s*$)"
)
_OWN_SEAT_LOOKUP_HEADING = re.compile(
    rf"(?P<game_type>{_SEAT_GAME_LABEL})\s*(?P<smoke>无烟|有烟)?\s*的?\s*(?:查|找)到了\s*,\s*"
    rf"(?:(?P<stake>{_SEAT_COUNTED_STAKE})\s*,\s*)?"
)


def customer_visible_own_seat_violations(text: str, fact: dict[str, object] | None) -> list[str]:
    """只拦明确把本人说成同桌占席的断言，匿名余位、询问与可能参加交给原合同。"""
    if not fact or fact.get("participation_already_recorded") is not False:
        return []
    content = unicodedata.normalize("NFKC", text.strip())
    if _SEAT_CLAIM_QUOTE_MARKS.intersection(content):
        return []
    if clock := _SEAT_CLAIM_CLOCK.match(content):
        content = content[clock.end():].lstrip()
        if not content.startswith(("这桌", "那桌")) or _seat_fact_for_clock(clock, [fact]) is None:
            return []
    if content.startswith(("这桌", "那桌")):
        content = content[2:].lstrip()
        if clock is not None and (heading := _OWN_SEAT_LOOKUP_HEADING.match(content)):
            game_type, smoke, stake = heading.group("game_type", "smoke", "stake")
            # 带明确桌况的陈述只沿用同一时间、玩法及已核实附文，不把另一张类似桌接进来。
            if (_SEAT_GAME_TYPES.get(game_type) != _SEAT_GAME_TYPES.get(str(fact.get("game_type")))
                    or (smoke and {"无烟": "no_smoking", "有烟": "smoking"}[smoke] != fact.get("smoke_preference"))
                    or (stake and unicodedata.normalize("NFKC", str(fact.get("stake_label"))).strip() != stake.strip())):
                return []
            content = content[heading.end():]
    claim = _CURRENT_OWN_SEAT_DECLARATION.match(content)
    if not claim or re.match(r"\s*[,]\s*(?:对吗|是不是|还是|是吗|对不对)", content[claim.end():]):
        return []
    return ["seat_claim.customer_membership_unverified"]


_TABLE_DURATION_FIT_DECLARATION = re.compile(
    r"(?:^|[。.!！?？\n])\s*你\s*"
    r"(?P<arrival>[^,，、。;；!！?？\n]{1,20}?)\s*(?:到店|到)\s*[,，、]\s*(?:能|可以)?[打玩](?:到|至)\s*"
    r"(?P<departure>[^,，、。;；!！?？\n]{1,20}?)\s*(?:这个|这段|这种)\s*(?:时长|时间安排)\s*"
    r"(?:肯定|当然)?\s*(?:没问题|对得上|合适|可以)(?=\s*(?:[,，。;；!！]|$))"
)


def customer_visible_duration_fit_violations(text: str, facts: dict[str, object] | None) -> list[str]:
    """未知时长只拦明确肯定的到店至离店搭配；复述、询问、条件句和引用交给语义审查。"""
    if not facts or "agreed_duration_hours" not in facts or facts["agreed_duration_hours"] is not None:
        return []
    content = unicodedata.normalize("NFKC", text.strip())
    if _SEAT_CLAIM_QUOTE_MARKS.intersection(content):
        return []
    for match in _TABLE_DURATION_FIT_DECLARATION.finditer(content):
        if all(_SEAT_CLAIM_CLOCK.fullmatch(match[field].strip()) for field in ("arrival", "departure")):
            return ["table_duration.unverified_fit"]
    return []


def customer_visible_table_start_violations(text: str, facts: list[dict[str, object]]) -> list[str]:
    """仅用本轮唯一候选的实时局况拦错误开打声明，原定钟点从不证明实际开打。"""
    if len(facts) != 1:
        return []
    fact = facts[0]
    if (fact.get("status") not in {"forming", "inviting"}
            or "actual_started_at" not in fact or fact["actual_started_at"] is not None):
        return []
    content = unicodedata.normalize("NFKC", text).strip()
    if _SEAT_CLAIM_QUOTE_MARKS.intersection(content):
        return []
    # 省略桌名的断言须紧接本轮候选的开场钟点；开头直接说「那桌」是唯一候选的私聊指代。
    named_table = content.removeprefix("有一桌").lstrip()
    clock = _SEAT_CLAIM_CLOCK.match(named_table)
    clock_tail = named_table[clock.end():].lstrip() if clock else ""
    bound_by_clock = bool(clock and clock_tail.startswith(("开的", "开场的", "这桌", "那桌"))
                          and _seat_fact_for_clock(clock, facts) is not None)
    if bound_by_clock:
        content = clock_tail
    claim = _DEFINITE_TABLE_START.search(content)
    if claim is None:
        return []
    prefix = content[:claim.start()]
    if (_START_CLAIM_OTHER_SCOPE.search(prefix)
            or re.match(r"[,;]\s*(?:还是|是不是|对不对|是吗)", content[claim.end():])):
        return []
    bound_by_reference = not prefix and content.startswith(("这桌", "那桌"))
    return ["game_start.current_table_not_started"] if bound_by_clock or bound_by_reference else []


def _current_listed_seat_declarations(
    statements: list[str], *, question: bool,
) -> tuple[list[tuple[re.Match[str], str | None]], list[list[int]]] | None:
    """按各行钟点绑定桌，行内、标题和末尾明示共用的缺口可在同一列举中并存。"""
    if len(statements) < 2:
        return None
    heading = _SEAT_LIST_HEADING.match(statements[0].strip())
    if heading is None or heading["count"] is None:
        return None
    rows = [statements[0].strip()[heading.end():], *(row.strip() for row in statements[1:])]
    if question:
        final, separator, choice = rows[-1].rpartition(",")
        if not separator or not _SEAT_LIST_CHOICE.fullmatch(choice.strip()):
            return None
        rows[-1] = final
    game_type = _SEAT_GAME_TYPES.get(heading["counted_type"])
    shared_counts = []
    if heading["heading_label"]:
        shared_counts.append(_REMAINING_BY_SEAT_LABEL[heading["heading_label"]])
    if claim := _SEAT_ROW_SHARED_CLAIM.search(rows[-1]):
        shared = _shared_seat_list_fields(claim["fields"])
        if shared is None:
            return None
        remaining, shared_type = shared
        if game_type is not None and shared_type is not None and game_type != shared_type:
            return None
        game_type = game_type or shared_type
        if remaining is not None:
            shared_counts.append(remaining)
        last_entry = rows.pop()[:claim.start()].rstrip(" ,、")
        if last_entry:
            rows.append(last_entry)
    count = heading["count"]
    if len(rows) != (int(count) if count.isdecimal() else _SEAT_HOUR_LABELS[count]):
        return None

    tables, counts = [], []
    for row in rows:
        declaration = _current_seat_declaration(row, question=False, listed=True)
        if declaration is None:
            return None
        clock, remaining, row_type = declaration
        row_counts = shared_counts if remaining is None else [*shared_counts, remaining]
        if clock is None or not row_counts or (game_type is not None and row_type is not None and game_type != row_type):
            return None
        tables.append((clock, row_type or game_type))
        counts.append(row_counts)
    return tables, counts


def _shared_seat_list_fields(text: str) -> tuple[int | None, str | None] | None:
    """拆解明确共用的稳定字段；陌生附文或冲突的玩法及余位不作确定性归属。"""
    remaining, game_type = None, None
    for field in re.split(r"[,、]", text):
        value = field.strip()
        label = _SEAT_CURRENT_LABEL.fullmatch(value)
        missing = _SEAT_LIST_MISSING_PEOPLE.fullmatch(value)
        if label or missing:
            if remaining is not None:
                return None
            remaining = _REMAINING_BY_SEAT_LABEL[label["label"] if label else missing["count"]]
        elif terms := _SEAT_LIST_GAME_TERMS.fullmatch(value):
            declared_type = _SEAT_GAME_TYPES[terms["game_type"]]
            if game_type is not None and game_type != declared_type:
                return None
            game_type = declared_type
        elif not _SEAT_LIST_ATTRIBUTE.fullmatch(value):
            return None
    return remaining, game_type


def _current_collective_seat_declaration(
    text: str, *, question: bool,
) -> tuple[list[re.Match[str]], list[int], str | None] | None:
    """完整列出开场才核对标题人数及共同余位；独立的选桌问句不改变前文陈述。"""
    text = text.strip().rstrip(".")
    if question:
        text, separator, choice = text.rpartition(",")
        if not separator or not _SEAT_LIST_CHOICE.fullmatch(choice.strip()):
            return None
    heading = _SEAT_LIST_HEADING.match(text)
    game_type = _SEAT_GAME_TYPES.get(heading["game_type"] or heading["counted_type"]) if heading else None
    heading_label = heading["heading_label"] if heading else None
    declaration = text[heading.end():] if heading else text
    claim = _SEAT_LIST_CLAIM.search(declaration)
    if not heading_label and claim is None:
        return None
    listing = declaration[:claim.start()].rstrip(" ,、") if claim else declaration
    listing = _SEAT_LIST_EACH_TABLE.sub("", listing.strip())
    fields = re.split(r"[,、和]", listing)
    count = heading["count"] if heading else None
    expected_count = (int(count) if count.isdecimal() else _SEAT_HOUR_LABELS[count]) if count else None
    if len(fields) < 2 or expected_count is not None and len(fields) != expected_count:
        return None
    clocks = []
    for field in fields:
        value = field.strip()
        table = _SEAT_LIST_PREFIX.match(value)
        clock = _SEAT_CLAIM_CLOCK.match(value, table.end() if table else 0)
        if clock is None or not _SEAT_LIST_SUFFIX.fullmatch(value[clock.end():].strip()):
            return None
        clocks.append(clock)
    counts = [_REMAINING_BY_SEAT_LABEL[heading_label]] if heading_label else []
    if claim:
        shared = _shared_seat_list_fields(claim["fields"])
        if shared is None:
            return None
        remaining, shared_type = shared
        if game_type is not None and shared_type is not None and game_type != shared_type:
            return None
        game_type = game_type or shared_type
        if remaining is not None:
            counts.append(remaining)
    return (clocks, counts, game_type) if counts else None


def _verified_listed_seats(
    tables: list[tuple[re.Match[str] | None, str | None]], declared_counts: list[list[int]],
    facts: list[dict[str, object]], span: tuple[int, int],
) -> list[VerifiedSeatClaim]:
    """各行绑定到可核验的同桌席位；并列声明有歧义、缺余位或重复指代时整组不采信。"""
    verified: list[VerifiedSeatClaim] = []
    game_ids: set[str | None] = set()
    for (clock, game_type), counts in zip(tables, declared_counts):
        fact = _seat_fact_for_clock(clock, facts)
        seat_counts = _verified_seat_counts(fact, game_type) if fact is not None else None
        if seat_counts is None or fact is None:
            return []
        raw_id = fact.get("game_id")
        game_id = raw_id if isinstance(raw_id, str) and raw_id else None
        if len(tables) > 1 and (game_id is None or game_id in game_ids):
            return []
        game_ids.add(game_id)
        start_at = fact.get("planned_start_at")
        claimed, remaining, seats_total = seat_counts
        for declared in dict.fromkeys(counts):
            verified.append({
                "game_id": game_id,
                "planned_start_at": start_at if isinstance(start_at, str) else None,
                "source_span": span,
                "seats_total": seats_total,
                "declared_claimed_seats": seats_total - declared,
                "declared_remaining_seats": declared,
                "actual_claimed_seats": claimed,
                "actual_remaining_seats": remaining,
                "consistent": declared == remaining,
            })
    return verified


def _verified_seat_counts(fact: dict[str, object], game_type: str | None) -> tuple[int, int, int] | None:
    """只使用可核验的四人桌数据，显式玩法和已占数字也须与同桌事实一致。"""
    observed_game_type = fact.get("game_type")
    if game_type is not None and (not isinstance(observed_game_type, str)
                                  or _SEAT_GAME_TYPES.get(observed_game_type) != game_type):
        return None
    remaining, capacity = fact.get("remaining_seats"), fact.get("seats_total")
    if type(remaining) is not int or not 0 <= remaining <= 4 or type(capacity) is not int or capacity != 4:
        return None
    claimed = capacity - remaining
    if "claimed_seats" in fact and (type(fact["claimed_seats"]) is not int or fact["claimed_seats"] != claimed):
        return None
    return claimed, remaining, capacity


def _current_seat_declaration(
    text: str, *, question: bool, listed: bool = False,
) -> tuple[re.Match[str] | None, int | None, str | None] | None:
    """提取明确当前桌况；逐行列举也可只报钟点与已知附文，再与共用余位合并。"""
    lead, *fields = (field.strip() for field in re.split(r"[,、]", text.strip().rstrip(".")))
    clock = _SEAT_CLAIM_CLOCK.match(lead)
    game_type = None
    if clock is None:
        label = lead if _SEAT_LABEL_PATTERN.fullmatch(lead) else None
    else:
        heading = _SEAT_CLAIM_HEADING.fullmatch(lead[clock.end():].strip())
        if heading is None:
            return None
        label = heading["label"]
        game_type = _SEAT_GAME_TYPES.get(heading["game_type"])
        if label is None:
            for index, field in enumerate(fields):
                if declared := _SEAT_CURRENT_LABEL.fullmatch(field):
                    label, fields = declared["label"], fields[index + 1:]
                    break
                field_type = _SEAT_GAME_TYPES.get(field)
                if field_type is not None:
                    if game_type is not None and game_type != field_type:
                        return None
                    game_type = field_type
                elif not (_SEAT_LIST_ATTRIBUTE if listed else _SEAT_CLAIM_ATTRIBUTE).fullmatch(field):
                    return None
    if listed and (label is None or all(_SEAT_LIST_ATTRIBUTE.fullmatch(field) for field in fields)):
        fields = []
    if label is None and not (listed and clock is not None):
        return None
    if (fields or question) and not (len(fields) == 1 and _SEAT_CLAIM_INVITE.fullmatch(fields[0])):
        return None
    return clock, _REMAINING_BY_SEAT_LABEL.get(label), game_type


def _seat_fact_for_clock(
    clock: re.Match[str] | None, facts: list[dict[str, object]],
) -> dict[str, object] | None:
    """只有唯一对客桌或唯一吻合的钟点能绑定；同钟点和不合格时间不猜。"""
    if clock is None:
        return facts[0] if len(facts) == 1 else None
    if clock["hour"] is not None:
        hours, minute = {int(clock["hour"])}, int(clock["minute"])
    else:
        raw_hour = clock["spoken_hour"]
        hour = int(raw_hour) if raw_hour.isdecimal() else _SEAT_HOUR_LABELS.get(raw_hour)
        if hour is None:
            return None
        day_part = clock["day_part"]
        minute = {"半": 30, "一刻": 15, "三刻": 45}.get(
            clock["part"], int(clock["spoken_minute"] or clock["bare_minute"] or 0),
        )
        # 没有上下午限定时，七点及十二点都可能对应相隔半天的两个钟点。
        hours = {hour, (hour + 12) % 24} if day_part is None and 1 <= hour <= 12 else {hour}
        if day_part == "晚上" and hour == 12:
            return None  # 午夜可能已跨日，未能确定显示日期时交语义审查。
        if day_part in {"下午", "傍晚", "晚上"} and 1 <= hour < 12:
            hours = {hour + 12}
    matches = []
    for fact in facts:
        planned = fact.get("planned_start_at")
        if planned is None:
            continue
        if not isinstance(planned, str):
            return None
        try:
            stamp = datetime.fromisoformat(planned)
        except ValueError:
            return None
        if stamp.utcoffset() is None:
            return None
        local = stamp.astimezone(DEFAULT_TZ)
        if (local.hour in hours and local.minute == minute
                and (clock["date"] is None or clock["date"] == local.date().isoformat())):
            matches.append(fact)
    return matches[0] if len(matches) == 1 else None


def compact_customer_visible_text(text: str) -> str:
    """压缩客户客户可见text，保留决策所需字段并限制载荷大小。"""
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return "".join(char for char in normalized if char.isalnum())
