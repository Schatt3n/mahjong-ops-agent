from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import importlib.util
import sys
from pathlib import Path

import pytest

from mahjong_agent_runtime.copywriting import build_customer_visible_text_generation_payload
from mahjong_agent_runtime.customer_visible_contract import (
    FORBIDDEN_CUSTOMER_SERVICE_PHRASES,
    PREFERRED_CANDIDATE_INVITE_PHRASES,
    PREFERRED_OPERATION_ACK_PHRASES,
    PREFERRED_REQUESTER_CURRENT_GAME_PHRASES,
    compact_customer_visible_text,
    customer_visible_action_claim_violations,
    customer_visible_contract_snapshot,
    customer_visible_duration_fit_violations,
    customer_visible_seat_claim_evidence,
    customer_visible_seat_claim_violations,
    customer_visible_text_contract_violations,
)
from mahjong_agent_runtime.copywriting import parse_customer_visible_text_generation
from mahjong_agent_runtime.customer_visible_review import build_reply_self_review_payload
from mahjong_agent_runtime.models import AgentAction, UserMessage


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_customer_visible_contract.py"


@pytest.mark.parametrize("text", [
    "你六点半到、打到十一点半，我知道了。这桌还没约多久。",
    "你六点半到、打到十一点半这个时长没问题吗？",
    "要是大家同意，你六点半到、打到十一点半这个时长没问题。",
    "你六点半到、打到十一点半这个时长没问题的话，等桌上确认再说。",
    "他说「你六点半到、打到十一点半这个时长没问题」，桌上还得核实。",
])
def test_duration_fit_boundary_preserves_acknowledgments_conditions_and_quotes(text):
    """确定性检查只拦截直接肯定，按句保留讨论条件和顾客本人的时间。"""
    assert customer_visible_duration_fit_violations(text, {"agreed_duration_hours": None}) == []


@pytest.mark.parametrize("text", [
    "AI 可以帮忙整理消息，但最终效果得实际试试。",
    "Agent 和大模型不是一回事，机器人也不一定用大模型。",
    "数据库的日志有助于排查问题，prompt 也需要明确任务。",
    "JSON 是常用的数据格式，系统之间交换信息挺方便。",
    "后台工作今天挺忙，预算审批完了我再改草稿。",
    "智能助手只能辅助判断，测试账号别用真实客户资料。",
    "周末下雨，去看看展也挺好。",
    "这份合同先留个副本，明天再和同事讨论。",
    "API key 不要写到日志里，token 也不要发给别人。",
    "Forming a team 和 inviting friends 都是普通英语表达。",
    "Performing well 通常就是表现不错的意思。",
    "Bearer tokens are commonly used for authorization.",
    "We can create games and join games with friends.",
    "Cookie 保存登录状态，示例里用 session=<redacted> 就好。",
])
def test_general_technical_work_and_life_topics_are_not_implementation_leaks(text):
    """术语本身不代表暴露当前部署；语义真实性与授权由独立审查核对。"""
    assert customer_visible_text_contract_violations(text) == []


@pytest.mark.parametrize("text,rule", [
    ("凭证是 sk-0123456789abcdef0123456789abcdef", "credential.api_key"),
    ("ＡＰＩ＿ＫＥＹ＝abcdefgh12345678", "credential.assigned_secret"),
    ("Authorization: Bearer abcdefgh0123456789", "credential.authorization"),
    ("access_token=abcdefgh0123456789", "credential.assigned_secret"),
    ("-----BEGIN PRIVATE KEY-----", "credential.private_key"),
    ("s k - 0 1 2 3 4 5 6 7 8 9 a b c d e f 0 1 2 3 4 5 6 7 8 9 a b c d e f", "credential.api_key"),
    ("sk-01234567\u200b89abcdef0123456789abcdef", "credential.api_key"),
])
def test_concrete_credentials_remain_blocked_without_echoing_the_value(text, rule):
    """只返回稳定规则号，防止拒绝反馈自身复制凭证；大小写/宽度/分隔混淆不能绕过。"""
    violations = customer_visible_text_contract_violations(text)
    assert rule in violations
    assert text not in str(violations)


@pytest.mark.parametrize('text', ['status=forming', '"status": "inviting"', 'ＳＴＡＴＵＳ＝ＦＯＲＭＩＮＧ'])
def test_common_english_enum_word_only_blocks_as_internal_state_assignment(text):
    assert 'internal_enum.status' in customer_visible_text_contract_violations(text)


@pytest.mark.parametrize('text,rule', [
    ('我调用 search_current_games 查过了。', 'internal_tool_identifier:search_current_games'),
    ('我调用 record_board_candidate_decision。', 'internal_tool_identifier:record_board_candidate_decision'),
    ('这项补位的 decision_task_id 已记下。', 'internal_identifier:decision_task_id'),
    ('下一步执行 create_game。', 'internal_tool_identifier:create_game'),
    ('ＣＲＥＡＴＥ＿ＧＡＭＥ', 'internal_tool_identifier:create_game'),
    ('search current games', 'internal_tool_identifier:search_current_games'),
    ('search_\u200bcurrent_games', 'internal_tool_identifier:search_current_games'),
    ('Bearer abcdefgh0123456789abcdefgh01234567', 'credential.authorization'),
    ('Ｂｅａｒｅｒ ａｂｃｄｅｆｇｈ０１２３４５６７８９ａｂｃｄｅｆｇｈ０１２３４５６７', 'credential.authorization'),
    ('Bearer abcd\u200befgh0123456789abcdefgh01234567', 'credential.authorization'),
    ('B e a r e r abcdefgh0123456789abcdefgh01234567', 'credential.authorization'),
    ('Cookie: session=abcdefgh0123456789', 'credential.cookie'),
    ('Ｃｏｏｋｉｅ： ｓｅｓｓｉｏｎ＝ａｂｃｄｅｆｇｈ０１２３４５６７８９', 'credential.cookie'),
    ('Cookie: session=abcd\u200befgh0123456789', 'credential.cookie'),
])
def test_real_tool_identifiers_and_credential_values_override_wrong_review_approval(text, rule):
    """误批的语义审查不能绕过末端可枚举信息防线，反馈只含规则而不含凭证。"""
    from mahjong_agent_runtime.customer_visible_review import item_reviews_approved, normalize_item_reviews

    items = [{'item_id': 'reply_to_user', 'text': text}]
    raw_review = {'approved': True, 'needs_human': False, 'violations': [], 'item_reviews': [
        {'item_id': 'reply_to_user', 'approved': True, 'suggested_safe_text': text, 'violations': []}]}
    normalized = normalize_item_reviews(raw_review, items)
    assert rule in customer_visible_text_contract_violations(text)
    assert normalized[0]['approved'] is False
    assert item_reviews_approved(normalized, items) is False
    assert text not in str(normalized[0]['violations'])


@pytest.mark.parametrize("text", (
    "好嘞，这桌给你记上了，19:30三缺一，还差三个人，我这边接着凑。",
    "19:30这桌 173、还差一位。",
    "１９：３０ ２７２，还差３人",
))
def test_conflicting_same_table_seat_labels_override_wrong_review_approval(text):
    """同一时刻紧邻的缺口表述自相矛盾时，拒绝放行并原样交还审查反馈。"""
    from mahjong_agent_runtime.customer_visible_review import item_reviews_approved, normalize_item_reviews

    items = [{"item_id": "reply_to_user", "text": text}]
    review_item = {"item_id": "reply_to_user", "approved": True, "suggested_safe_text": text, "violations": []}
    raw_review = {"approved": True, "needs_human": False, "violations": [], "item_reviews": [review_item]}
    normalized = normalize_item_reviews(raw_review, items)

    assert customer_visible_text_contract_violations(text) == ["seat_count.inconsistent_pair"]
    assert normalized[0]["suggested_safe_text"] == text
    assert normalized[0]["violations"] == ["customer_visible_contract:seat_count.inconsistent_pair"]
    assert not item_reviews_approved(normalized, items)
    assert not item_reviews_approved([review_item], items)


@pytest.mark.parametrize("text", (
    "19:30 371，还差一位。",
    "19:30一缺三，还差3个人。",
    "19:30两缺两，还差二人。",
    "19:30三缺一，20:00还差三个人。",
    "19:30三缺一，还差三个人的另一桌可以等等。",
    "如果19:30三缺一，还差三个人的安排就另聊。",
    "19:30三缺一，还差三个人？",
    "你是说「19:30三缺一，还差三个人」这句吗？",
))
def test_seat_pair_check_leaves_table_comparisons_and_quoting_to_semantic_review(text):
    """人数一致或语义未明确绑定到同桌时，末端规则不提前代审。"""
    assert customer_visible_text_contract_violations(text) == []


@pytest.mark.parametrize("text,planned,game_type", [
    ("17:30那桌还在凑，底2到3、三缺一；你18:30才到、最多打三个半小时，"
     "这桌能不能等到那时候我这边还没答复，先帮你登记问一下。", "2026-11-26T17:30:00+08:00", "hangzhou_mahjong"),
    ("七点有一桌红中，1块无烟，三缺一，你打吗？", "2026-11-26T19:00:00+08:00", "red_center_mahjong"),
    ("１７：３０这桌，３７１，来不？", "2026-11-26T09:30:00Z", "hangzhou_mahjong"),
    ("三缺一，打吗？", None, "hangzhou_mahjong"),
    ("19:00有一桌，红中麻将，三缺一。", "2026-11-26T19:00:00+08:00", "红中麻将"),
])
def test_declared_current_seats_must_match_the_verified_table(text, planned, game_type):
    """归档错报和稳定短码只有绑定到当前这桌，才会以固定理由被拒绝。"""
    facts = [{"game_id": "table", "planned_start_at": planned, "remaining_seats": 3, "seats_total": 4,
              "game_type": game_type}]

    assert customer_visible_seat_claim_violations(text, facts) == ["seat_count.current_table_mismatch"]


@pytest.mark.parametrize("text", [
    "17:30那桌一缺三，来吗？",
    "173。",
    "你是说「17:30那桌，三缺一」吗？",
    "＂17:30三缺一＂",
    "如果17:30那桌三缺一，我就来。",
    "如果再找到两位；17:30那桌三缺一。",
    "17:30这桌加上你们就是三缺一。",
    "17:30那桌，另一桌三缺一。",
    "举个例子。三缺一，打吗？",
    "三缺一？",
    "18:00那桌三缺一。",
    "17:30有一桌红中，1块无烟，三缺一，你打吗？",
    "17:30这桌，红中，三缺一。",
    "17:30有一桌杭麻，四川麻将，三缺一。",
    "17:30这桌门店自定义麻将，三缺一。",
])
def test_declared_seat_check_defers_unbound_or_nonfactual_wording(text):
    """条件、引用、其他玩法或无关钟点不能借唯一查局结果校验。"""
    facts = [{"game_id": "table", "planned_start_at": "2026-11-26T17:30:00+08:00",
              "remaining_seats": 3, "seats_total": 4, "game_type": "hangzhou_mahjong"}]

    assert customer_visible_seat_claim_violations(text, facts) == []


@pytest.mark.parametrize("text,second_time,expected", [
    ("19:00这桌371。", "2026-11-26T20:00:00+08:00", ["seat_count.current_table_mismatch"]),
    ("20:00这桌272；19:00这桌371。", "2026-11-26T20:00:00+08:00", ["seat_count.current_table_mismatch"]),
    ("三缺一，打吗？", "2026-11-26T20:00:00+08:00", []),
    ("19:00这桌红中371。", "2026-11-26T19:00:00+08:00", []),
    ("七点三缺一，打吗？", "2026-11-26T07:00:00+08:00", []),
    ("晚上七点三缺一，打吗？", "2026-11-26T07:00:00+08:00", ["seat_count.current_table_mismatch"]),
])
def test_multiple_tables_need_a_unique_explicit_clock_for_seat_validation(text, second_time, expected):
    """同钟点不同桌不猜，未注明早晚的钟点也不能跨候选强行选一桌。"""
    facts = [
        {"game_id": "red", "planned_start_at": "2026-11-26T19:00:00+08:00",
         "remaining_seats": 3, "seats_total": 4, "game_type": "red_center_mahjong"},
        {"game_id": "hangzhou", "planned_start_at": second_time,
         "remaining_seats": 2, "seats_total": 4, "game_type": "hangzhou_mahjong"},
    ]

    assert customer_visible_seat_claim_violations(text, facts) == expected


@pytest.mark.parametrize("facts", [
    [],
    [{"game_id": "table", "planned_start_at": "2026-11-26T17:30:00", "remaining_seats": 3, "seats_total": 4}],
    [{"game_id": "table", "planned_start_at": "2026-11-26T17:30:00+08:00", "remaining_seats": False, "seats_total": 4}],
    [{"game_id": "table", "planned_start_at": "2026-11-26T17:30:00+08:00", "remaining_seats": 3, "seats_total": 3}],
    [{"game_id": "table", "planned_start_at": "2026-11-26T17:30:00+08:00", "remaining_seats": 5, "seats_total": 4}],
])
def test_missing_or_invalid_current_facts_cannot_prove_a_conflict(facts):
    """没有可核验的四人桌局况时，个人回复中的短码仍由语义审查处理。"""
    assert customer_visible_seat_claim_violations("17:30那桌371。", facts) == []


_COLLECTIVE_SEAT_FACTS = [
    {"game_id": f"table-{clock}", "planned_start_at": f"2026-11-26T{clock}:00+08:00",
     "remaining_seats": 3, "seats_total": 4, "game_type": "red_center_mahjong"}
    for clock in ("12:15", "16:00", "21:30")
]
_ARCHIVED_COLLECTIVE_SEAT_REPLY = "有红中2-3块的现成桌：12点15一桌、16点一桌、21点30一桌，都是三缺一。你想打哪桌？"
_ARCHIVED_COUNTED_TABLE_REPLY = (
    "有两桌红中的，一桌12点15开、一桌16点开，都是2块3块、三缺一，每桌打2小时。"
    "你想报哪桌？都不合适的话我帮你另凑一桌。"
)
_ARCHIVED_HEADING_SEAT_REPLY = "有两桌红中2块3块的三缺一：12:15和16:00各一桌，都差三人，你打哪桌？"
_EVENING_LISTED_SEAT_FACTS = [
    {"game_id": f"table-{clock}", "planned_start_at": f"2026-11-26T{clock}:00+08:00",
     "remaining_seats": 3, "seats_total": 4, "game_type": "hangzhou_mahjong"}
    for clock in ("19:30", "20:20")
]
_ARCHIVED_EVENING_SEAT_REPLY = (
    "有两桌可以拼：19:30的1-3块杭麻三缺一，打3小时；"
    "20:20的1-3块杭麻三缺一，打2小时。你赶哪桌？"
)
_ARCHIVED_EVENING_SHARED_SEAT_REPLY = (
    "老板好，有两桌能拼：19:30开、打三小时；20:20开、打两小时，都是杭麻1到3块、目前三缺一。"
    "你19:15能到的话19:30那桌正好赶上，要哪桌？"
)
_ARCHIVED_EVENING_EXPLICIT_REMAINING_REPLY = (
    "有两桌杭麻1到3块的：19:30开打三小时，20:20开打两小时，都还缺三个。"
    "你19:15到都赶得上，打哪桌？"
)


def test_seat_claim_evidence_verifies_each_declared_held_and_remaining_value():
    """受评判的实际送达回复可按局逐条对照，输出同时保留已占和余位的依据。"""
    facts = [{**fact, "claimed_seats": 1} for fact in _EVENING_LISTED_SEAT_FACTS]
    actual = customer_visible_seat_claim_evidence(_ARCHIVED_EVENING_EXPLICIT_REMAINING_REPLY, facts)
    assert actual == {"status": "matched", "verified_claims": [
        {
            "game_id": fact["game_id"],
            "planned_start_at": fact["planned_start_at"],
            "source_span": (0, _ARCHIVED_EVENING_EXPLICIT_REMAINING_REPLY.index("。")),
            "seats_total": 4,
            "declared_claimed_seats": 1,
            "declared_remaining_seats": 3,
            "actual_claimed_seats": 1,
            "actual_remaining_seats": 3,
            "consistent": True,
        }
        for fact in facts
    ]}
    wrong = _ARCHIVED_EVENING_EXPLICIT_REMAINING_REPLY.replace("缺三个", "缺一个")
    rejected = customer_visible_seat_claim_evidence(wrong, facts)
    assert rejected["status"] == "mismatch"
    assert [(claim["declared_claimed_seats"], claim["declared_remaining_seats"], claim["consistent"])
            for claim in rejected["verified_claims"]] == [(3, 1, False), (3, 1, False)]
    assert customer_visible_seat_claim_violations(wrong, facts) == ["seat_count.current_table_mismatch"]


@pytest.mark.parametrize("text", [
    "19:30开打三小时，20:20开打两小时。你打哪桌？",
    "19:30这桌现在还有空位，打吗？",
    "两桌都是1.371元。",
])
def test_seat_claim_evidence_reports_absent_only_without_detected_count_cues(text):
    """开场、时长、价格和泛称还有位置不提供可比较的人数声明。"""
    assert customer_visible_seat_claim_evidence(text, _EVENING_LISTED_SEAT_FACTS) == {
        "status": "absent", "verified_claims": [],
    }


@pytest.mark.parametrize("text,facts", [
    ("「19:30三缺一」你问的是这个吗？", _EVENING_LISTED_SEAT_FACTS),
    ("19:30是三缺一？", _EVENING_LISTED_SEAT_FACTS),
    ("现在好像已经坐了3人。", _EVENING_LISTED_SEAT_FACTS),
    ("19:30这桌一缺三。", []),
    ("19:30这桌一缺三。", [
        _EVENING_LISTED_SEAT_FACTS[0],
        {**_EVENING_LISTED_SEAT_FACTS[0], "game_id": "another-at-1930"},
    ]),
    ("19:30这桌一缺三。", [{**_EVENING_LISTED_SEAT_FACTS[0], "claimed_seats": 3}]),
])
def test_seat_claim_evidence_reports_unparsed_when_count_cues_have_no_reliable_binding(text, facts):
    """标记、问句和歧义范围不会提升成已核验事实，矛盾的来源数字也不采信。"""
    assert customer_visible_seat_claim_evidence(text, facts) == {"status": "unparsed", "verified_claims": []}


@pytest.mark.parametrize("shortage,status", [("一缺三", "unparsed"), ("三缺一", "mismatch")])
def test_seat_claim_evidence_preserves_verified_partial_claim_and_cannot_hide_mismatch(shortage, status):
    """一处数字已核验但另一处仅为问句时，保留部分证据与实际核验状态。"""
    text = f"19:30这桌{shortage}。你问的是20:20那桌三缺一吗？"
    result = customer_visible_seat_claim_evidence(text, _EVENING_LISTED_SEAT_FACTS)
    claim, = result["verified_claims"]
    assert result["status"] == status and claim["game_id"] == _EVENING_LISTED_SEAT_FACTS[0]["game_id"]
    assert claim["source_span"] == (0, text.index("。"))


@pytest.mark.parametrize("text", [
    _ARCHIVED_EVENING_SEAT_REPLY,
    _ARCHIVED_EVENING_SHARED_SEAT_REPLY,
    "查到2局杭麻：19:30那局一缺三，打3小时；20:20那局371，打2小时，你选哪局？",
    "有两桌杭麻371：19:30那桌一缺三，打3小时；20:20那桌一缺三，打2小时。",
    "有两桌可拼：晚上七点半的1到3元杭麻三缺一，三个钟头\n20点20的1到3块杭麻173，打两个小时。",
    "有两桌杭麻：19:30开、一缺三，打三小时；20:20开、打两小时；这几桌都是目前三缺一。",
    "有两桌杭麻：19:30开、371，打三小时；20:20开、打两小时，都是现在一缺三，你打哪桌？",
    "您好，有两桌可拼：19:30开、打三小时；20:20开、打两小时，都是杭麻1-3元、现在三缺一。",
])
def test_listed_table_seat_assertions_check_each_verified_current_game(text):
    """晚间逐桌报出的缺口各自核对，任一错报都不能因为另一桌人数正确而放行。"""
    assert customer_visible_seat_claim_violations(text, _EVENING_LISTED_SEAT_FACTS) == [
        "seat_count.current_table_mismatch",
    ]
    correct = text.replace("三缺一", "一缺三").replace("371", "173")
    assert customer_visible_seat_claim_violations(correct, _EVENING_LISTED_SEAT_FACTS) == []


@pytest.mark.parametrize("text", [
    "如果有两桌可以拼：19:30的1-3块杭麻三缺一，打3小时；20:20的1-3块杭麻三缺一，打2小时。",
    "刚刚是说「有两桌可以拼：19:30的1-3块杭麻三缺一；20:20的1-3块杭麻三缺一」？",
    "有三桌可以拼：19:30的1-3块杭麻三缺一；20:20的1-3块杭麻三缺一。",
    "有两桌可以拼：19:30这桌加上你们就三缺一；20:20的1-3块杭麻三缺一。",
    "有两桌可以拼：19:30的1-3块杭麻三缺一；20:20的1-3块杭麻三缺一？",
    "有两桌可以拼：19:30的1-3块杭麻三缺一；19:30的1-3块杭麻三缺一。",
    "有两桌可以拼：19:30的1-3块杭麻三缺一；22:20的1-3块杭麻三缺一。",
    "有两桌杭麻：19:30的1-3块杭麻三缺一；20:20的1-3块红中三缺一。",
    "有两桌红中：19:30三缺一；20:20三缺一。",
    "如果最后还缺两人，有两桌能拼：19:30开、打三小时；20:20开、打两小时，都是杭麻三缺一。",
    "有两桌能拼：19:30开、打三小时；20:20开、打两小时，如果再来两人都是杭麻、三缺一。",
    "有两桌能拼：19:30开、打三小时；20:20开、打两小时，都是红中1到3块、目前三缺一。",
    "有两桌杭麻：19:30开、打三小时；20:20开、打两小时，都是川麻、目前三缺一。",
    "有两桌杭麻：19:30开、打三小时；20:20开、打两小时，都是目前三缺一吗？",
    "有两桌杭麻：19:30开或20:20开；20:20开、打两小时，都是目前三缺一。",
])
def test_listed_table_seat_assertions_do_not_guess_scope_or_mixed_identity(text):
    """标题冲突、条件、引语、重复和未核验的逐桌归属交语义审查。"""
    assert customer_visible_seat_claim_violations(text, _EVENING_LISTED_SEAT_FACTS) == []


@pytest.mark.parametrize("facts", [
    _EVENING_LISTED_SEAT_FACTS[:1],
    [*_EVENING_LISTED_SEAT_FACTS, {**_EVENING_LISTED_SEAT_FACTS[1], "game_id": "another-2020"}],
    [_EVENING_LISTED_SEAT_FACTS[0], {**_EVENING_LISTED_SEAT_FACTS[1], "remaining_seats": None}],
    [_EVENING_LISTED_SEAT_FACTS[0], {**_EVENING_LISTED_SEAT_FACTS[1], "game_type": "sichuan_mahjong"}],
])
def test_listed_table_seat_assertions_require_complete_verified_evidence(facts):
    """真实查询没有为每张桌唯一提供人数和玩法时，不能把另一桌的数字套上。"""
    assert customer_visible_seat_claim_violations(_ARCHIVED_EVENING_SEAT_REPLY, facts) == []
    assert customer_visible_seat_claim_violations(_ARCHIVED_EVENING_SHARED_SEAT_REPLY, facts) == []


@pytest.mark.parametrize("text", [
    _ARCHIVED_COLLECTIVE_SEAT_REPLY,
    _ARCHIVED_COUNTED_TABLE_REPLY,
    _ARCHIVED_HEADING_SEAT_REPLY,
    "１２：１５、１６：００、２１：３０都是３７１。",
    "12点15、16点那桌、21点30都是三缺一。",
    "查到红中麻将的局：12点15分一局、16:00一局、21点半一局，每局都是三缺一。",
    "查到2局红中：一局12:15开场、16点那局开打，每局都是2到3块、两个小时、三缺一。",
])
def test_collective_seat_shortcut_is_compared_against_each_current_table(text):
    """原始误报与技术短码必须分别核对每张真实桌，不能把已报一人理解成只缺一人。"""
    assert customer_visible_seat_claim_violations(text, _COLLECTIVE_SEAT_FACTS) == [
        "seat_count.current_table_mismatch",
    ]
    correct = text.replace("三缺一", "一缺三").replace("３７１", "１７３")
    assert customer_visible_seat_claim_violations(correct, _COLLECTIVE_SEAT_FACTS) == []
    mixed = [{**row, "remaining_seats": 1} if index == 0 else row
             for index, row in enumerate(_COLLECTIVE_SEAT_FACTS)]
    assert customer_visible_seat_claim_violations(text, mixed) == ["seat_count.current_table_mismatch"]


@pytest.mark.parametrize(("text", "expected"), [
    ("有两桌红中2块3块的173：12:15和16:00各一桌，都差两个人，你打哪桌？", ["seat_count.current_table_mismatch"]),
    ("查到2局红中371：12:15和16:00各1局，你选哪一局？", ["seat_count.current_table_mismatch"]),
    ("12:15和16:00各一桌，都还差1位，你想报哪桌？", ["seat_count.current_table_mismatch"]),
    ("有两桌红中2块3块的173：12:15和16:00各一桌，都差三人，你打哪桌？", []),
    ("有两桌红中一缺三：12:15和16:00各一桌。", []),
    ("12:15、16:00每桌都是还差3个人。", []),
])
def test_collective_heading_and_explicit_shared_person_counts_are_independently_verified(text, expected):
    """标题与共同余位分别核对，任一说法不实都不能由另一处正确人数冲销。"""
    assert customer_visible_seat_claim_violations(text, _COLLECTIVE_SEAT_FACTS) == expected


@pytest.mark.parametrize("text", [
    "如果12:15、16:00、21:30都是三缺一，那就问问。",
    "如果能另招两个人；12:15、16:00、21:30都是三缺一。",
    "「12:15、16:00、21:30都是三缺一」",
    "１２：１５、１６：００、２１：３０都是三缺一？",
    "12:15一桌或16点一桌，都是三缺一。",
    "12:15到16:00都是三缺一。",
    "12:15、16:00、21:30加两个人都是三缺一。",
    "16点、16:00都是三缺一。",
    "12:15、16:00、21:00都是三缺一。",
    "有杭麻2-3块的现成桌：12:15、16:00、21:30，都是三缺一。",
    "有三桌红中的，一桌12点15开、一桌16点开，都是2块3块、三缺一，每桌打2小时。",
    "有两桌杭麻的，一桌12点15开、一桌16点开，都是2块3块、三缺一，每桌打2小时。",
    "一桌12点15开、一桌16点开，都是2块3块、三缺一，每桌再加人就能开。",
    "一桌12点15开、一桌16点开，都是2块3块、三缺一、二缺二，每桌打2小时。",
    "两桌如果到时凑齐的话，一桌12点15开、一桌16点开，都是2块3块、三缺一。",
    "有两桌红中三缺一：12:15和16:00各一桌，都差三人？",
    "有三桌红中三缺一：12:15和16:00各一桌，都差三人，你打哪桌？",
    "有两桌红中三缺一：12:15或16:00各一桌，都差三人，你打哪桌？",
    "如果有两桌红中三缺一：12:15和16:00各一桌，都差三人，你打哪桌？",
    "老板说「有两桌红中三缺一：12:15和16:00各一桌，都差三人」，你打哪桌？",
    "有两桌红中三缺一：12:15和16:00各一桌，如果有人走都差三人，你打哪桌？",
])
def test_collective_seat_shortcut_defers_unverified_scope_and_meaning(text):
    """引语、提问、加入新人员、时间范围和只匹配到一部分的陈述仍交独立语义审查。"""
    assert customer_visible_seat_claim_violations(text, _COLLECTIVE_SEAT_FACTS) == []


@pytest.mark.parametrize("facts", [
    _COLLECTIVE_SEAT_FACTS[:2],
    [*_COLLECTIVE_SEAT_FACTS, {**_COLLECTIVE_SEAT_FACTS[1], "game_id": "another-at-16"}],
    [*_COLLECTIVE_SEAT_FACTS, {**_COLLECTIVE_SEAT_FACTS[0], "game_id": "midnight",
                             "planned_start_at": "2026-11-26T00:15:00+08:00"}],
    [*_COLLECTIVE_SEAT_FACTS[:2], {**_COLLECTIVE_SEAT_FACTS[2], "remaining_seats": None}],
    [*_COLLECTIVE_SEAT_FACTS[:2], {**_COLLECTIVE_SEAT_FACTS[2], "seats_total": 3}],
    [*_COLLECTIVE_SEAT_FACTS[:2], {**_COLLECTIVE_SEAT_FACTS[2], "game_type": None}],
    [*_COLLECTIVE_SEAT_FACTS[:2], {**_COLLECTIVE_SEAT_FACTS[2], "game_id": _COLLECTIVE_SEAT_FACTS[1]["game_id"]}],
])
def test_collective_seat_shortcut_needs_complete_unique_current_facts(facts):
    """只有每个列出的钟点分别对上不同且数字有效的牌局，才能拒绝这条并列人数声明。"""
    assert customer_visible_seat_claim_violations(_ARCHIVED_COLLECTIVE_SEAT_REPLY, facts) == []


@pytest.mark.parametrize("facts", [
    _COLLECTIVE_SEAT_FACTS[:1],
    [*_COLLECTIVE_SEAT_FACTS, {**_COLLECTIVE_SEAT_FACTS[1], "game_id": "other-1600", "game_type": "hangzhou_mahjong"}],
    [*_COLLECTIVE_SEAT_FACTS, {**_COLLECTIVE_SEAT_FACTS[1], "game_id": "other-day-1600",
                             "planned_start_at": "2026-11-27T16:00:00+08:00"}],
])
def test_collective_heading_does_not_guess_missing_or_ambiguous_tables(facts):
    """没有逐桌唯一来源时，标题和共同余位都不能套到未确认的桌上。"""
    assert customer_visible_seat_claim_violations(_ARCHIVED_HEADING_SEAT_REPLY, facts) == []


@pytest.mark.parametrize("latest_query,expected_outcome", (
    ({"called": True, "allowed": True, "result": {"matches": [], "alternatives": []}},
     {"status": "completed", "matched_count": 0}),
    ({"called": True, "allowed": False, "error": "timeout", "result": {"matches": []}}, {"status": "unknown"}),
    ({"called": True, "allowed": True, "result": {}}, {"status": "unknown"}),
))
def test_empty_query_is_separate_from_scoped_zero_seat_evidence(latest_query, expected_outcome):
    """查空、失败和缺陷结果都不能冒充某一桌售罄，其他时段的真实零席按局身份保留。"""
    message = UserMessage("test_group", "customer", "顾客", "七点半那个0.5还有位吗？")
    latest = {"name": "search_current_games", **latest_query}
    older = {"name": "search_current_games", "called": True, "allowed": True, "result": {"matches": [
        {"game": {"game_id": "target_table", "planned_start_at": "19:30", "status": "forming",
                  "seat_summary": {"claimed_seats": 2, "remaining_seats": 2, "seats_total": 4}}},
    ]}}
    zero_seats = {"claimed_seats": 4, "remaining_seats": 0, "seats_total": 4}
    context = {
        "active_game_visible_summaries": [
            {"game_id": "other_table", "planned_start_at": "20:45", "status": "ready", "seat_summary": zero_seats},
            {"game_id": "cancelled_table", "planned_start_at": "19:30", "seat_summary": zero_seats},
            {"game_id": "unverified_claim", "planned_start_at": "19:30", "user_visible_summary": "这桌满了"},
        ],
        "turn_tool_evidence": [older, {"name": "update_game_status", "called": True, "allowed": True,
                                      "result": {"game": {"game_id": "cancelled_table", "status": "cancelled"}}}, latest],
    }
    original = deepcopy(context)
    action = AgentAction("核对顾客问的座位", "completed", "查询当前桌况", reply_to_user="七点半这桌没位了。")
    items = [{"item_id": "reply_to_user", "recipient_id": "customer", "channel": "group", "text": action.reply_to_user}]
    evidence = build_reply_self_review_payload(
        message=message, action=action, review_items=items, context_payload=context, review_scope="reply_to_user",
    )["semantic_fidelity_contract"]

    assert evidence["game_availability_evidence"] == {
        "latest_search": expected_outcome,
        "zero_seat_game_ids": ["other_table"],
    }
    assert any(game["game_id"] == "target_table" and game["remaining_seats"] == 2
               for game in evidence["seat_count_evidence"])
    interpretation = evidence["availability_claim_interpretation"]
    assert "具体桌席位声明" in interpretation and "泛问" in interpretation
    assert any("别桌" in rule and "unsupported_conclusion" in rule for rule in evidence["rules"])
    assert any("未定的方案" in rule and "条件句之外另行断言" in rule and "remaining_seats_display" in rule
               for rule in evidence["rules"])
    assert context == original


@pytest.mark.parametrize("mode,reply", [
    ("scheduled", "七点半那桌173还有位，现在一个人，还差三个，0.5无烟。你到得了就来。不过人齐才开，到点也不一定还等着，尽量早点过来。"),
    ("scheduled", "原定七点半开，现在还缺三个人。到点还缺人，得和同桌再商量。"),
    ("asap_when_full", "现在还缺三个人，人齐就开。"),
])
def test_review_keeps_same_table_opening_mode_separate_from_attendance(mode, reply):
    """真实桌的开场方式留给同桌核验，查空和另一桌的条件都不能覆盖。"""
    source = "看板刚出了173 七点半0.5无烟的 还有位吗 能到" if mode == "scheduled" else "看板里人齐开那桌还有位吗？"
    message = UserMessage("dm", "customer", "顾客", source)
    action = AgentAction("回答顾客所指的桌", "completed", "按本人所指桌核对", reply_to_user=reply)
    seats = {"claimed_seats": 1, "remaining_seats": 3, "seats_total": 4}
    public_terms = {"start_time_kind": mode, "stake": "0.5", "smoke_preference": "no_smoking"}
    if mode == "scheduled":
        public_terms["start_time"] = "19:30"
    context = {"active_game_visible_summaries": [
        {"game_id": "table", "status": "inviting", "public_requirement": public_terms, "seat_summary": seats},
        {"game_id": "other", "status": "inviting", "public_requirement": {"start_time_kind": "asap_when_full"},
         "seat_summary": seats},
    ], "turn_tool_evidence": [
        {"name": "search_current_games", "call_id": "board_table_relookup", "called": True, "allowed": True,
         "result": {"lookup_scope": "specific_table", "requirement": {"start_time_kind": "scheduled", "game_type": "hangzhou_mahjong"},
                    "matches": [], "alternatives": []}},
    ]}
    payload = build_reply_self_review_payload(message=message, action=action, context_payload=context, review_scope="reply_to_user",
        review_items=[{"item_id": "reply_to_user", "recipient_id": "customer", "text": reply}])
    assert payload["review_items"][0]["text"] == reply
    assert payload["active_game_visible_summaries"][0]["public_requirement"] == public_terms
    fidelity = payload["semantic_fidelity_contract"]
    assert fidelity["game_availability_evidence"]["searches_by_scope"]["specific_table"]["matched_count"] == 0
    rule, = [rule for rule in fidelity["rules"] if "原定七点半，人还没齐" in rule]
    assert all(term in rule for term in ("start_time_kind", "semantic_contradiction", "实际是人齐开的桌", "另一桌的摘要"))


def test_internal_tool_identifier_inventory_matches_actual_registered_tools():
    """注册新工具必须同步末端标识清单，避免保密边界与执行入口各自漂移。"""
    from mahjong_agent_runtime.customer_visible_contract import FORBIDDEN_INTERNAL_TOOL_NAMES
    from mahjong_agent_runtime.domains.tools.registry import default_tool_definitions
    from mahjong_agent_runtime.stores.memory import InMemoryAgentStore

    assert set(FORBIDDEN_INTERNAL_TOOL_NAMES) == set(default_tool_definitions(InMemoryAgentStore()))


def load_contract_verifier_module():
    spec = importlib.util.spec_from_file_location("verify_customer_visible_contract_for_test", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_customer_visible_contract_verifier_passes_current_prompts() -> None:
    module = load_contract_verifier_module()

    assert module.verify_prompts() == []


def test_customer_visible_copywriting_payload_uses_shared_contract_terms() -> None:
    payload = build_customer_visible_text_generation_payload(
        message=UserMessage(conversation_id="c1", sender_id="zhang", sender_name="张哥", text="现在有人吗"),
        action=AgentAction(
            goal="测试客户可见话术合同",
            objective_status="completed",
            reasoning_summary="",
            reply_to_user="七点三缺一，打吗？",
            stop_reason={"can_stop": True, "why": "test", "pending_work": [], "depends_on_tool_results": False},
        ),
        items=[{"item_id": "reply_to_user", "source": "reply_to_user", "text": "七点三缺一，打吗？"}],
        context_payload={},
        generation_scope="reply_to_user",
    )

    style_contract = payload["style_quality_contract"]
    assert style_contract["forbidden_customer_service_phrases"] == list(FORBIDDEN_CUSTOMER_SERVICE_PHRASES)
    assert style_contract["preferred_short_phrases"] == [
        *PREFERRED_REQUESTER_CURRENT_GAME_PHRASES,
        *PREFERRED_CANDIDATE_INVITE_PHRASES,
        *PREFERRED_OPERATION_ACK_PHRASES,
    ]
    style_examples_text = "\n".join(item["good"] for item in payload["style_examples"])
    assert "七点三缺一，可以不？" in style_examples_text
    assert "还没有，还差俩" in style_examples_text
    assert "两个人，18.30 星月的局，371 她，打吗？" in style_examples_text
    assert payload["current_request"] == {"text": "现在有人吗", "quoted_text": ""}
    assert "Only items[].text is allowed as factual source" in payload["semantic_source_of_truth"]
    assert "current_request is only used to decide" in payload["semantic_source_of_truth"]
    assert payload["reply_relevance_contract"]["applies_when"] == "generation_scope=reply_to_user"
    assert "shortest text that fully answers current_request" in payload["reply_relevance_contract"]["rule"]
    assert "Preserving a coherent customer decision summary overrides brevity" in payload["reply_relevance_contract"]["priority"]
    assert "treat those clauses as one coherent option summary" in payload["reply_relevance_contract"]["coherent_option_summary_rule"]
    assert "tone only" in payload["style_examples_boundary"]
    assert "Never copy facts from examples" in payload["style_examples_boundary"]


def test_customer_visible_contract_snapshot_is_grouped_for_eval_and_docs() -> None:
    snapshot = customer_visible_contract_snapshot()

    assert snapshot["forbidden_customer_service_phrases"] == FORBIDDEN_CUSTOMER_SERVICE_PHRASES
    assert snapshot["preferred_requester_current_game_phrases"] == PREFERRED_REQUESTER_CURRENT_GAME_PHRASES
    assert snapshot["preferred_candidate_invite_phrases"] == PREFERRED_CANDIDATE_INVITE_PHRASES
    assert snapshot["preferred_operation_ack_phrases"] == PREFERRED_OPERATION_ACK_PHRASES


def test_customer_visible_text_contract_violations_group_shared_terms() -> None:
    violations = customer_visible_text_contract_violations("task_context_id 写进了 turn_tool_evidence，局为 asap_when_full。")

    assert "internal_identifier:task_context_id" in violations
    assert "internal_identifier:turn_tool_evidence" in violations
    assert "internal_enum:asap_when_full" in violations


def test_customer_visible_text_contract_normalizes_spacing_width_and_case() -> None:
    assert compact_customer_visible_text("Ａ I / 智能 助手 / asap when full") == "ai智能助手asapwhenfull"

    violations = customer_visible_text_contract_violations("ＴＡＳＫ＿ＣＯＮＴＥＸＴ＿ＩＤ 和 turn tool evidence，asap when full。")

    assert "internal_identifier:task_context_id" in violations
    assert "internal_identifier:turn_tool_evidence" in violations
    assert "internal_enum:asap_when_full" in violations


def test_customer_visible_action_claim_requires_backend_delivery_evidence() -> None:
    pending_only = {
        "contact_started": False,
        "draft_statuses": ["pending_approval", "pending_approval"],
    }

    assert customer_visible_action_claim_violations("还差一个，在问了，人齐开。", pending_only) == [
        "unverified_external_action:问了"
    ]
    assert customer_visible_action_claim_violations("还差一个，我帮你问问。", pending_only) == []
    assert customer_visible_action_claim_violations(
        "还差一个，在问了，人齐开。",
        {"contact_started": True, "draft_statuses": ["sent"]},
    ) == []


@pytest.mark.parametrize("text", [
    "要我帮你问问有没有人吗？",
    "目前没有现成局，要不要我帮你问问？",
    "我帮你问问有没有人。",
])
def test_service_offer_and_future_intent_do_not_claim_contact_has_happened(text: str) -> None:
    assert customer_visible_action_claim_violations(
        text, {"contact_started": False, "draft_statuses": []}
    ) == []


@pytest.mark.parametrize("text, expected_phrase", [
    ("已经问了，有人会告诉你。", "问了"),
    ("正在问其他人。", "在问"),
    ("邀请已发送。", "已发送"),
])
def test_completed_or_ongoing_contact_still_requires_delivery_evidence(text: str, expected_phrase: str) -> None:
    assert customer_visible_action_claim_violations(
        text, {"contact_started": False, "draft_statuses": ["pending_approval"]}
    ) == [f"unverified_external_action:{expected_phrase}"]
    assert customer_visible_action_claim_violations(
        text, {"contact_started": True, "draft_statuses": ["sent"]}
    ) == []


@pytest.mark.parametrize('evidence', [None, {}, {'contact_started': False}, {'contact_started': 'true'}])
@pytest.mark.parametrize('text', ['我已经发消息问过了。', '我发过消息给其他人。', '我已经发了邀请。'])
def test_missing_delivery_evidence_cannot_approve_agent_contact_claim(evidence, text):
    assert customer_visible_action_claim_violations(text, evidence)
    assert customer_visible_action_claim_violations(text, {'contact_started': True}) == []


@pytest.mark.parametrize('clause', ['你问过了同事', '你发过消息给同事', '你已经发了邮件'])
def test_completed_forms_keep_verified_customer_attribution_exception(clause):
    text = clause + '，可以先等回复。'
    assert customer_visible_action_claim_violations(text, None, attributed_claim_spans=[(0, len(clause))]) == []
    assert customer_visible_action_claim_violations(text + '我问过了其他牌友。', None,
        attributed_claim_spans=[(0, len(clause))])


@pytest.mark.parametrize('text', ['你问了同事，可以先等等回复。', '你问 了同事，可以先等等回复。'])
def test_verified_attribution_exempts_only_fully_covered_action_occurrence(text):
    span = (0, text.index('，'))
    assert customer_visible_action_claim_violations(text, None, attributed_claim_spans=[span]) == []
    assert customer_visible_action_claim_violations(text, None, attributed_claim_spans=[(0, 2)])


def test_attributing_customer_clause_does_not_hide_another_agent_claim():
    text = '你问了同事，可以等等。老板这边我问了其他牌友。'
    assert customer_visible_action_claim_violations(text, None, attributed_claim_spans=[(0, 5)]) == [
        'unverified_external_action:问了']


@pytest.mark.parametrize('spans', [None, [], [(0, 2)], [(-1, 10)], [(0, 100)], [(True, 8)],
    [('0', 8)], [(8, 0)], [{'start': 0, 'end': 8}], 'all', 42])
def test_invalid_or_partial_attribution_span_never_exempts_contact(spans):
    assert customer_visible_action_claim_violations('你问了同事，可以等回复。', None, attributed_claim_spans=spans)


def test_review_payload_preserves_latest_constraints_and_their_conversation_evidence() -> None:
    message = UserMessage(
        conversation_id="time-flexible-chat", sender_id="customer", sender_name="客户",
        text="1块的杭麻，时间都行", message_id="latest-message",
    )
    known_facts = {"game_type": "hangzhou_mahjong", "stake": "1", "time_preference": "any"}
    fact_assertions = [{
        "field": "time_preference", "value": "any", "role": "current",
        "source_message_id": "latest-message",
    }]
    recent_conversation = [
        {"role": "user", "content": "下午有人吗", "message_id": "earlier-message"},
        {"role": "assistant", "content": "想打多少的？"},
    ]
    earlier_query = {
        "name": "search_current_games", "called": True, "allowed": True,
        "result": {"requirement": {"stake": "1", "time_preference": "下午"}, "matches": []},
    }
    action = AgentAction(
        goal="沿用一块杭麻，按客户最新表达放宽时间并询问是否需要找人",
        objective_status="waiting_user", reasoning_summary="当前时间偏好由最新消息说明",
        objective_state={
            "known_facts": known_facts, "missing_facts": ["当前人数"],
            "fact_assertions": fact_assertions, "unrelated_debug_notes": "not-for-review",
        },
        reply_to_user="目前没有现成局，要我帮你问问有没有人吗？",
        stop_reason={"can_stop": True, "why": "等客户决定", "pending_work": [], "depends_on_tool_results": True},
    )
    payload = build_reply_self_review_payload(
        message=message, action=action,
        review_items=[{
            "item_id": "reply_to_user", "recipient_id": "customer",
            "source_text": action.reply_to_user, "text": action.reply_to_user,
        }],
        context_payload={"recent_conversation": recent_conversation, "turn_tool_evidence": [earlier_query]},
        review_scope="reply_to_user",
    )
    assert payload["current_message"]["text"] == message.text
    assert payload["recent_conversation_tail"] == recent_conversation
    assert payload["action_boundary"]["known_facts"] == known_facts
    assert payload["action_boundary"]["missing_facts"] == ["当前人数"]
    assert payload["action_boundary"]["fact_assertions"] == fact_assertions
    assert "unrelated_debug_notes" not in payload["action_boundary"]
    assert payload["turn_tool_evidence"] == [earlier_query]
    assert payload["turn_tool_evidence"][0]["result"]["requirement"]["time_preference"] == "下午"
    service_contract = payload["customer_service_review_contract"]
    assert service_contract["offer_is_not_execution"] is True
    assert service_contract["proposed_next_steps_are_not_authoritative"] is True
    assert service_contract["latest_user_constraints_override_query_filters"] is True
    assert service_contract["allow_known_condition_omission_without_game_option"] is True


@pytest.mark.parametrize("review_scope, recipient_id", [
    ("tool_calls", "candidate"), ("reply_to_user", "another-customer"),
])
def test_review_contract_does_not_allow_candidate_invitation_condition_omission(review_scope: str, recipient_id: str) -> None:
    message = UserMessage(conversation_id="c1", sender_id="customer", sender_name="客户", text="一块杭麻")
    action = AgentAction(
        goal="保留被邀约客户做决定所需的条件", objective_status="waiting_user", reasoning_summary="",
        reply_to_user="1块杭麻，人齐开，打吗？",
        stop_reason={"can_stop": True, "why": "等回应", "pending_work": [], "depends_on_tool_results": False},
    )
    payload = build_reply_self_review_payload(
        message=message, action=action,
        review_items=[{"item_id": "invitation", "recipient_id": recipient_id, "text": action.reply_to_user}],
        context_payload={}, review_scope=review_scope,
    )
    assert payload["customer_service_review_contract"]["allow_known_condition_omission_without_game_option"] is False
    assert payload["action_boundary"]["known_facts"] == {}
    assert payload["action_boundary"]["missing_facts"] == []
    assert payload["action_boundary"]["fact_assertions"] == []


def test_review_boundary_rejects_wrong_fact_container_types_without_interpreting_text() -> None:
    message = UserMessage(conversation_id="c1", sender_id="customer", sender_name="客户", text="时间都行")
    action = AgentAction(
        goal="审查载荷只接受模型事实容器的既定类型", objective_status="waiting_user", reasoning_summary="",
        objective_state={"known_facts": ["时间都行"], "missing_facts": "人数", "fact_assertions": {"stake": "1"}},
        reply_to_user="要我帮你问问吗？",
    )
    payload = build_reply_self_review_payload(
        message=message, action=action,
        review_items=[{"item_id": "reply_to_user", "recipient_id": "customer", "text": action.reply_to_user}],
        context_payload={}, review_scope="reply_to_user",
    )
    assert payload["action_boundary"]["known_facts"] == {}
    assert payload["action_boundary"]["missing_facts"] == []
    assert payload["action_boundary"]["fact_assertions"] == []


@pytest.mark.parametrize("text,goal,missing,query", [
    ("可以，帮我问问", "先问玩法", ["玩法"], None),
    ("只看全店有哪些在组局，先不找人", "用画像做全店浏览", [], {"game_type": "national_standard_mahjong"}),
    ("一致", "承接闲聊并引导客户约局", [], None),
])
def test_review_receives_original_scope_and_profile_despite_conflicting_model_interpretation(text, goal, missing, query):
    """保留原话、权威画像与实际窄查询供审查核对，不能被主模型自述替换。"""
    message = UserMessage(conversation_id="review-source", sender_id="customer", sender_name="客户", text=text)
    profile = {"customer_id": "customer", "preferred_games": ["national_standard_mahjong"]}
    evidence = [] if query is None else [{"name": "search_current_games", "called": True, "allowed": True,
        "result": {"requirement": query, "matches": [], "customer_reply_contract": {
            "evidence_scope": {"kind": "submitted_filters_only"}}}}]
    action = AgentAction(goal=goal, objective_status="waiting_user", reasoning_summary="模型解释需外部证据核对",
        objective_state={"missing_facts": missing, "reply_scope": {"requested_information": goal}}, reply_to_user="待审文本")
    payload = build_reply_self_review_payload(message=message, action=action,
        review_items=[{"item_id": "reply_to_user", "recipient_id": "customer", "text": action.reply_to_user}],
        context_payload={"sender_profile": profile, "turn_tool_evidence": evidence}, review_scope="reply_to_user")
    assert payload["current_message"]["text"] == text
    assert payload["sender_profile"] == profile
    assert payload["turn_tool_evidence"] == evidence
    contract = payload["customer_service_review_contract"]
    assert contract["model_missing_facts_are_not_authority"] is True
    assert contract["known_constraints_require_source_recheck"] is True
    assert contract["profile_does_not_override_inventory_scope"] is True
    assert contract["reply_scope_requires_actual_request_support"] is True


def _follow_up_review_payload(context, *, recipient="customer", scope="reply_to_user"):
    """构造同一客户的服务提议，单独核对复查证据的权限与目标投影。"""
    message = UserMessage(conversation_id="c1", sender_id="customer", sender_name="客户", text="现在有人吗")
    action = AgentAction(goal="回应查局", objective_status="waiting_user", reasoning_summary="", reply_to_user="要我帮你问问吗？")
    return build_reply_self_review_payload(
        message=message, action=action,
        review_items=[{"item_id": "reply_to_user", "recipient_id": recipient, "text": action.reply_to_user}],
        context_payload=context, review_scope=scope,
    )


@pytest.fixture
def current_follow_up_context(monkeypatch):
    """固定服务器时钟，让过期任务与仍有效的目标证据可确定重放。"""
    monkeypatch.setattr("mahjong_agent_runtime.customer_visible_review.now", lambda: datetime.fromisoformat("2030-01-01T12:00:00+00:00"))
    return {
        "task_context_window": {"task_context_id": "goal-1"},
        "active_follow_ups": [{
            "task_id": "follow-1", "task_type": "conversation_follow_up", "status": "pending",
            "conversation_id": "c1", "subject_id": "customer", "subject_name": "私有备注",
            "due_at": "2030-01-01T12:05:00+00:00", "aggregate_type": "conversation_goal", "aggregate_id": "goal-1",
            "idempotency_key": "内部幂等标识", "last_error": "不应投影的日志",
            "payload": {"task_context_id": "goal-1", "goal": "继续查看合适局", "waiting_for": "局况变化",
                        "expires_at": "2030-01-01T13:00:00+00:00", "private_notes": "不可复制的私有内容"},
        }],
    }


def test_review_follow_up_evidence_is_current_and_minimal(current_follow_up_context):
    payload = _follow_up_review_payload(current_follow_up_context)
    assert payload["active_follow_up_summaries"] == [{
        "task_id": "follow-1", "task_type": "conversation_follow_up", "status": "pending",
        "due_at": "2030-01-01T12:05:00+00:00", "aggregate_type": "conversation_goal", "aggregate_id": "goal-1",
        "task_context_id": "goal-1", "goal": "继续查看合适局", "waiting_for": "局况变化",
        "expires_at": "2030-01-01T13:00:00+00:00",
    }]
    assert "private_notes" not in str(payload) and "内部幂等标识" not in str(payload)
    assert payload["customer_service_review_contract"]["read_only_lookup_is_not_durable_follow_up"] is True
    assert payload["customer_service_review_contract"]["current_clock_is_not_availability_evidence"] is True


@pytest.mark.parametrize("field,value", [
    ("conversation_id", "another-conversation"), ("subject_id", "another-customer"),
    ("task_type", "game_recruitment"), ("status", "cancelled"), ("status", "completed"),
    ("task_id", ""), ("payload", []),
    ("payload.task_context_id", "old-goal"), ("payload.task_context_id", None),
    ("payload.expires_at", "2030-01-01T11:59:59+00:00"),
    ("payload.expires_at", "2030-01-01T13:00:00"), ("payload.expires_at", "invalid"),
])
def test_unrelated_or_expired_follow_up_cannot_support_current_promise(current_follow_up_context, field, value):
    context = deepcopy(current_follow_up_context)
    task = context["active_follow_ups"][0]
    if field.startswith("payload."):
        task["payload"][field.split(".", 1)[1]] = value
    else:
        task[field] = value
    assert _follow_up_review_payload(context)["active_follow_up_summaries"] == []


@pytest.mark.parametrize("recipient,scope", [("other", "reply_to_user"), ("customer", "tool_calls")])
def test_follow_up_goal_is_not_projected_into_another_recipient_review(current_follow_up_context, recipient, scope):
    assert _follow_up_review_payload(current_follow_up_context, recipient=recipient, scope=scope)["active_follow_up_summaries"] == []


def test_customer_visible_text_generation_rejects_rewrite_with_contract_terms() -> None:
    _, errors = parse_customer_visible_text_generation(
        """
        {
          "reasoning_summary": "模型误保留了内部词。",
          "item_rewrites": [
            {
              "item_id": "reply_to_user",
              "final_text": "当前 task_context_id 已写入 turn_tool_evidence。",
              "semantic_preserved": true,
              "used_facts": [],
              "withheld_facts": [],
              "style_checks": ["误判安全"],
              "change_summary": "错误改写"
            }
          ]
        }
        """,
        [{"item_id": "reply_to_user", "source": "reply_to_user", "text": "好，我帮你看看。"}],
    )

    assert any("violates customer-visible contract" in error for error in errors)
    assert any("task_context_id" in error and "turn_tool_evidence" in error for error in errors)
