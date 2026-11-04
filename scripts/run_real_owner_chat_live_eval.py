#!/usr/bin/env python3
"""使用真实模型回放老板私聊黄金场景，检验决策与话术。"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import sys
import tempfile
from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, Callable


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from mahjong_agent_runtime import (  # noqa: E402
    AgentRuntime,
    CustomerProfile,
    InMemoryTraceRecorder,
    OpenAICompatibleAgentClient,
    QuotedMessageRef,
    SQLiteAgentStore,
    TokenBudget,
    ToolCall,
    ToolGateway,
    UserMessage,
)
from mahjong_agent_runtime.customer_visible_contract import (  # noqa: E402
    FORBIDDEN_CUSTOMER_SERVICE_PHRASES,
    FORBIDDEN_IMPLEMENTATION_IDENTITY_TERMS,
    FORBIDDEN_INTERNAL_PROCESS_TERMS,
    customer_visible_text_contract_violations,
)
from mahjong_agent_runtime.env import load_dotenv_defaults  # noqa: E402
from mahjong_agent_runtime.domains.tools.schemas import memory_item_schema  # noqa: E402
from mahjong_agent_runtime.models import now  # noqa: E402
from mahjong_agent_runtime.domains.value_utils import normalize_smoke_preference  # noqa: E402
from run_inquiry_grounding_eval import assess_reply_semantics  # noqa: E402
from reply_money_eval import check_reply_money_facts, validate_money_fact_check  # noqa: E402
from reply_state_eval import (  # noqa: E402
    assess_reply_communication, communication_state_is_consistent, validate_reply_state_assessment,
)
from eval_product_contracts import (  # noqa: E402
    resolve_product_contract_revisions, read_product_records,
    apply_reply_presentation_to_scenarios, apply_reply_presentation_to_cases,
    apply_query_option_policy_to_scenarios, apply_query_option_policy_to_cases,
)


DEFAULT_DB_PATH = ROOT / "runtime_data" / "real_owner_chat_live_eval.sqlite3"
GOLDEN_REPLY_PATH = ROOT / "eval" / "golden" / "real_owner_chat_golden.jsonl"
SEMANTIC_CALIBRATION_PATH = ROOT / "eval" / "golden" / "reply_semantic_calibration.jsonl"
# 旧黄金案例与已有真实夹具的对应关系；保留原始数据，不能另写一套期望绕过它。
LEGACY_GOLDEN_SCENARIOS = {
    "resume_game_status_after_casual_chat": "resume_status_after_casual_chat",
    "duration_negotiation_should_persist": "duration_limit_update_should_persist",
    "later_people_count_query_should_search_or_answer_current_status": "later_people_count_query",
    "reject_smoking_game_updates_preference": "reject_smoking_game_updates_preference",
    "casual_chat_should_not_pollute_business_state": "casual_chat_should_not_pollute_business_state",
    "initial_request_uses_profile_defaults_and_searches_pool": "profile_default_matched_game",
    "who_are_the_players_can_show_public_nickname_only": "public_nickname_lookup",
    "group_duration_mismatch_records_exit": "duration_rejection_not_chitchat",
    "human_likeness_reply_should_be_short_and_decision_focused": "profile_default_matched_game",
}
# 原始黄金把拒绝五小时与后续明确四小时写在一个案例里，必须按真实回合分别验收。
LEGACY_GOLDEN_LATER_FIELDS = {
    "group_duration_mismatch_records_exit": {"profile_updates": "duration_limit_update_should_persist"},
}
IMPLEMENTATION_DETAIL_FORBIDDEN_REPLY_FRAGMENTS = [
    *FORBIDDEN_IMPLEMENTATION_IDENTITY_TERMS,
    *FORBIDDEN_INTERNAL_PROCESS_TERMS,
    "JSON",
    "个人微信测试",
    "候选人",
    "已发送",
    "邀约草稿",
    "老板审批",
    "问了",
]
CUSTOMER_SERVICE_FORBIDDEN_REPLY_FRAGMENTS = [
    *FORBIDDEN_CUSTOMER_SERVICE_PHRASES,
    "请问还有什么可以帮",
]


@dataclass(slots=True)
class LiveEvalScenario:
    """真实老板聊天在线评测的一条场景、初始化逻辑与验收条件。"""

    scenario_id: str
    name: str
    message: UserMessage
    setup: Callable[[SQLiteAgentStore], None]
    required_tool_names: list[str] = field(default_factory=list)
    required_tool_name_any: list[str] = field(default_factory=list)
    # 未入局且无待处理邀约的约束反馈，可用本次真实写入并读回的任务记忆等价承接。
    equivalent_state_tracking_memory: dict[str, Any] = field(default_factory=dict)
    forbidden_tool_names: list[str] = field(default_factory=list)
    required_reply_any: list[list[str]] = field(default_factory=list)
    required_reply_contains: list[str] = field(default_factory=list)
    forbidden_reply_contains: list[str] = field(default_factory=list)
    forbidden_model_context_contains: list[str] = field(default_factory=list)
    forbidden_trace_steps: list[str] = field(default_factory=lambda: ["action_contract_error", "llm_error"])
    expected_tool_result_paths: dict[str, dict[str, Any] | list[dict[str, Any]]] = field(default_factory=dict)
    expected_tool_result_unordered_rows: dict[str, dict[str, list[dict[str, Any]]]] = field(default_factory=dict)
    optional_query_result_fields: dict[str, dict[str, Any]] = field(default_factory=dict)
    expected_status_without_tool: dict[str, list[str]] = field(default_factory=dict)
    expected_any_tool_result_paths: dict[str, dict[str, Any]] = field(default_factory=dict)
    expected_active_game_count: int | None = None
    expected_active_game_status: str | None = None
    expected_active_game_seat_summary: dict[str, Any] = field(default_factory=dict)
    expected_active_game_requirement: dict[str, Any] = field(default_factory=dict)
    expected_task_memory_contains: list[dict[str, Any]] = field(default_factory=list)
    # 标签不替代实际生效范围；显式启用时还须读回同客户、同当前任务绑定的记忆。
    require_task_memory_current_context: bool = False
    expected_checkpoint_contains: list[str] = field(default_factory=list)
    expected_status_any: list[str] = field(default_factory=list)
    max_tool_call_counts: dict[str, int] = field(default_factory=dict)
    max_reply_chars: int | None = 80
    require_help_offer: bool = False
    require_board_recovery_delivery: bool = False
    # 仅由已审计的纯查询policy启用；真实回复的沟通行为须独立盲判后与同步终态硬比较。
    requires_query_reply_state_check: bool = False
    excluded_search_filter_fields: list[str] = field(default_factory=list)
    excluded_query_requirement_values: dict[str, list[Any]] = field(default_factory=dict)


def build_empty_store(db_path: pathlib.Path) -> SQLiteAgentStore:
    """构建empty存储，只组合调用方允许进入当前作用域的数据。"""
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteAgentStore(db_path)


def future_planned_start_at(clock: str, *, anchor: dt.datetime | None = None) -> str:
    """按显式基准或当前时刻生成未来夹具时间；同一时间线可固定基准避免跨日漂移。"""

    hour_text, _, minute_text = str(clock).partition(":")
    now = anchor or dt.datetime.now().astimezone()
    target = now.replace(hour=int(hour_text), minute=int(minute_text or 0), second=0, microsecond=0)
    # 未给锚的既有夹具仍放次日；显式时间线锚则取其后的最近钟点，不一律多加一天。
    if anchor is None or target <= now:
        target += dt.timedelta(days=1)
    return target.isoformat()


# 只捕获一次未来15:42作为消息本身的时间，库存19:00据此生成；禁止二次滚动消息日期。
PROFILE_DEFAULT_FIXTURE_ANCHOR = dt.datetime.fromisoformat(
    future_planned_start_at("15:42", anchor=dt.datetime.now().astimezone())
)
# 多个回放复用同一合成时间线；跨轮或跨日执行setup不能重新滚动历史与当前消息的日期。
FLEXIBLE_TIME_FIXTURE_ANCHOR = dt.datetime.fromisoformat(future_planned_start_at("13:02"))


def today_at(clock: str) -> dt.datetime:
    """保留源聊天的时刻，用于口语化时间评测。"""

    hour_text, _, minute_text = str(clock).partition(":")
    return dt.datetime.now().astimezone().replace(
        hour=int(hour_text),
        minute=int(minute_text or 0),
        second=0,
        microsecond=0,
    )


def seed_default_profiles(store: SQLiteAgentStore) -> None:
    store.upsert_customer(
        CustomerProfile(
            customer_id="owner_real_customer",
            display_name="常客",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["0.5", "1"],
            profile_facts=[
                "真实老板聊天画像：95% 情况打 0.5，打 1 块会单独说。",
                "真实老板聊天画像：95% 情况是一个人来，带人会单独说。",
                "真实老板聊天画像：偏好无烟。",
            ],
            smoke_preference="no_smoke",
            response_score=0.9,
            notes="内部备注：不进入模型上下文。",
        )
    )
    store.upsert_customer(
        CustomerProfile(
            customer_id="summer",
            display_name="夏日-老板备注-可室外抽烟",
            public_name="夏日",
            private_remark="老板备注：可室外抽烟",
            notes="内部画像：熟人带局。",
        )
    )
    store.upsert_customer(
        CustomerProfile(
            customer_id="smile",
            display_name="笑脸-老板备注-高响应",
            public_name="笑脸",
            private_remark="老板备注：高响应",
            notes="内部画像：经常喊人。",
        )
    )
    store.upsert_customer(
        CustomerProfile(
            customer_id="kexing",
            display_name="可星-老板备注-偶尔迟到",
            public_name="可星",
            private_remark="老板备注：偶尔迟到",
            notes="内部画像：不稳定。",
        )
    )


def setup_profile_default_matched_game(store: SQLiteAgentStore) -> None:
    """按同一未来下午基准提供七点备选，保持原口语时间与库存日期一致。"""
    seed_default_profiles(store)
    store.create_game(
        conversation_id="owner_real_pool",
        organizer_id="summer",
        organizer_name="夏日",
        requirement={
            "game_type": "hangzhou_mahjong",
            "stake": "0.5",
            "smoke_preference": "no_smoke",
            "start_time_kind": "scheduled",
            "start_time": "19:00",
            "planned_start_at": future_planned_start_at("19:00", anchor=PROFILE_DEFAULT_FIXTURE_ANCHOR),
            "duration_hours": 4,
            "needed_seats": 1,
            "user_visible_summary": "七点三缺一",
        },
        known_players=[
            {"customer_id": "smile", "display_name": "笑脸-老板备注-高响应"},
            {"customer_id": "kexing", "display_name": "可星-老板备注-偶尔迟到"},
        ],
        trace_id="trace_owner_real_live_eval_setup",
    )


def setup_public_nickname_lookup(store: SQLiteAgentStore) -> None:
    seed_default_profiles(store)
    store.append_user_turn(
        UserMessage(
            conversation_id="owner_real_customer_chat",
            sender_id="owner_real_customer",
            sender_name="常客",
            text="帮我约个6.30无烟的",
            message_id="msg_owner_real_public_nickname_previous_user",
        ),
        "trace_owner_real_public_nickname_previous_user",
    )
    store.append_assistant_turn(
        "owner_real_customer_chat",
        "七点三缺一，打吗？",
        "trace_owner_real_public_nickname_previous_assistant",
    )
    store.create_game(
        conversation_id="owner_real_customer_chat",
        organizer_id="summer",
        organizer_name="夏日-老板备注-可室外抽烟",
        requirement={
            "game_type": "hangzhou_mahjong",
            "stake": "0.5",
            "smoke_preference": "no_smoke",
            "start_time_kind": "scheduled",
            "start_time": "19:00",
            "planned_start_at": future_planned_start_at("19:00"),
            "duration_hours": 4,
            "needed_seats": 1,
            "user_visible_summary": "七点三缺一",
        },
        known_players=[
            {"customer_id": "smile", "display_name": "笑脸-老板备注-高响应"},
            {"customer_id": "kexing", "display_name": "可星-老板备注-偶尔迟到"},
        ],
        trace_id="trace_owner_real_public_nickname_setup",
    )


def setup_accept_existing_offer(store: SQLiteAgentStore) -> None:
    seed_default_profiles(store)
    store.append_user_turn(
        UserMessage(
            conversation_id="owner_real_customer_chat",
            sender_id="owner_real_customer",
            sender_name="常客",
            text="帮我约个6.30无烟的",
            message_id="msg_owner_real_accept_previous_user",
        ),
        "trace_owner_real_accept_previous_user",
    )
    store.append_assistant_turn(
        "owner_real_customer_chat",
        "七点三缺一，可以不",
        "trace_owner_real_accept_previous_assistant",
    )
    store.create_game(
        conversation_id="owner_real_customer_chat",
        organizer_id="summer",
        organizer_name="夏日",
        requirement={
            "game_type": "hangzhou_mahjong",
            "stake": "0.5",
            "smoke_preference": "no_smoke",
            "start_time_kind": "scheduled",
            "start_time": "19:00",
            "planned_start_at": future_planned_start_at("19:00"),
            "duration_hours": 4,
            "needed_seats": 1,
            "user_visible_summary": "七点三缺一",
        },
        known_players=[
            {"customer_id": "summer", "display_name": "夏日", "status": "joined", "source": "requester"},
            {"customer_id": "smile", "display_name": "笑脸", "status": "confirmed"},
            {"customer_id": "kexing", "display_name": "可星", "status": "confirmed"},
        ],
        trace_id="trace_owner_real_accept_offer_setup",
    )
    store.create_game(
        conversation_id="owner_real_parallel_option",
        organizer_id="parallel_organizer",
        organizer_name="",
        requirement={
            "game_type": "hangzhou_mahjong",
            "stake": "0.5",
            "smoke_preference": "no_smoke",
            "start_time_kind": "scheduled",
            "start_time": "18:30",
            "planned_start_at": future_planned_start_at("18:30"),
            "duration_hours": 4,
            "needed_seats": 2,
            "user_visible_summary": "18:30待组局",
        },
        known_players=[
            {"customer_id": "parallel_organizer", "display_name": "", "status": "joined", "source": "requester"},
            {"customer_id": "owner_real_customer", "display_name": "", "status": "confirmed"},
        ],
        trace_id="trace_owner_real_parallel_option_setup",
    )


def setup_duration_rejection(store: SQLiteAgentStore) -> None:
    seed_default_profiles(store)
    store.append_user_turn(
        UserMessage(
            conversation_id="owner_real_customer_chat",
            sender_id="owner_real_customer",
            sender_name="常客",
            text="也可以",
            message_id="msg_owner_real_duration_previous_user",
        ),
        "trace_owner_real_duration_previous_user",
    )
    store.append_assistant_turn(
        "owner_real_customer_chat",
        "5小时也不行吗",
        "trace_owner_real_duration_previous_assistant",
    )
    store.create_game(
        conversation_id="owner_real_customer_chat",
        organizer_id="summer",
        organizer_name="夏日",
        requirement={
            "game_type": "hangzhou_mahjong",
            "stake": "0.5",
            "smoke_preference": "no_smoke",
            "start_time_kind": "scheduled",
            "start_time": "19:00",
            "planned_start_at": future_planned_start_at("19:00"),
            "duration_hours": 5,
            "needed_seats": 0,
            "user_visible_summary": "七点三缺一，5小时",
        },
        known_players=[
            {"customer_id": "owner_real_customer", "display_name": "常客", "status": "confirmed"},
            {"customer_id": "smile", "display_name": "笑脸", "status": "confirmed"},
            {"customer_id": "kexing", "display_name": "可星", "status": "confirmed"},
        ],
        trace_id="trace_owner_real_duration_setup",
    )


def setup_duration_limit_update(store: SQLiteAgentStore) -> None:
    seed_default_profiles(store)
    store.append_user_turn(
        UserMessage(
            conversation_id="owner_real_customer_chat",
            sender_id="owner_real_customer",
            sender_name="常客",
            text="所以现在有人了吗",
            message_id="msg_owner_real_duration_limit_status_query",
        ),
        "trace_owner_real_duration_limit_status_query",
    )
    store.append_assistant_turn("owner_real_customer_chat", "还没有，还差俩", "trace_owner_real_duration_limit_status_reply")
    store.create_game(
        conversation_id="owner_real_customer_chat",
        organizer_id="owner_real_customer",
        organizer_name="常客",
        requirement={
            "game_type": "hangzhou_mahjong",
            "stake": "0.5",
            "smoke_preference": "no_smoke",
            "start_time_kind": "scheduled",
            "start_time": "19:00",
            "planned_start_at": future_planned_start_at("19:00"),
            "duration_hours": None,
            "needed_seats": 2,
            "user_visible_summary": "还没有，还差俩",
        },
        known_players=[
            {"customer_id": "owner_real_customer", "display_name": "常客", "status": "confirmed"},
            {"customer_id": "smile", "display_name": "笑脸", "status": "confirmed"},
        ],
        trace_id="trace_owner_real_duration_limit_setup",
    )


def setup_casual_chat_should_not_pollute_business_state(store: SQLiteAgentStore) -> None:
    seed_default_profiles(store)
    store.append_user_turn(
        UserMessage(
            conversation_id="owner_real_customer_chat",
            sender_id="owner_real_customer",
            sender_name="常客",
            text="帮我约个6.30无烟的",
            message_id="msg_owner_real_casual_initial_request",
        ),
        "trace_owner_real_casual_initial_request",
    )
    store.append_assistant_turn("owner_real_customer_chat", "七点三缺一，可以不", "trace_owner_real_casual_initial_offer")
    store.create_game(
        conversation_id="owner_real_customer_chat",
        organizer_id="summer",
        organizer_name="夏日",
        requirement={
            "game_type": "hangzhou_mahjong",
            "stake": "0.5",
            "smoke_preference": "no_smoke",
            "start_time_kind": "scheduled",
            "start_time": "19:00",
            "planned_start_at": future_planned_start_at("19:00"),
            "duration_hours": 4,
            "needed_seats": 1,
            "user_visible_summary": "七点三缺一",
        },
        known_players=[
            {"customer_id": "smile", "display_name": "笑脸", "status": "confirmed"},
            {"customer_id": "kexing", "display_name": "可星", "status": "confirmed"},
        ],
        trace_id="trace_owner_real_casual_game_setup",
    )


def setup_resume_status_after_casual_chat(store: SQLiteAgentStore) -> None:
    seed_default_profiles(store)
    store.append_user_turn(
        UserMessage(
            conversation_id="owner_real_customer_chat",
            sender_id="owner_real_customer",
            sender_name="常客",
            text="帮我约个6.30无烟的",
            message_id="msg_owner_real_resume_initial_request",
        ),
        "trace_owner_real_resume_initial_request",
    )
    store.append_assistant_turn("owner_real_customer_chat", "七点三缺一，可以不", "trace_owner_real_resume_initial_offer")
    store.append_user_turn(
        UserMessage(
            conversation_id="owner_real_customer_chat",
            sender_id="owner_real_customer",
            sender_name="常客",
            text="也可以",
            message_id="msg_owner_real_resume_customer_accepts_offer",
        ),
        "trace_owner_real_resume_customer_accepts_offer",
    )
    store.append_assistant_turn("owner_real_customer_chat", "okk", "trace_owner_real_resume_ok")
    store.append_assistant_turn("owner_real_customer_chat", "5小时也不行吗", "trace_owner_real_resume_duration_check")
    store.append_user_turn(
        UserMessage(
            conversation_id="owner_real_customer_chat",
            sender_id="owner_real_customer",
            sender_name="常客",
            text="不行啊",
            message_id="msg_owner_real_resume_duration_rejected",
        ),
        "trace_owner_real_resume_duration_rejected",
    )
    store.append_assistant_turn("owner_real_customer_chat", "好吧，好吧", "trace_owner_real_resume_duration_ack")
    store.append_user_turn(
        UserMessage(
            conversation_id="owner_real_customer_chat",
            sender_id="owner_real_customer",
            sender_name="常客",
            text="好奇的问一下哈，你们是不是每天都要人工找人组局啊",
            message_id="msg_owner_real_resume_chitchat_ai",
        ),
        "trace_owner_real_resume_chitchat_ai",
    )
    store.append_assistant_turn(
        "owner_real_customer_chat",
        "感觉还是人工的话，会回复得比较精细点，而且粘合度也高一点",
        "trace_owner_real_resume_chitchat_boss",
    )
    store.append_user_turn(
        UserMessage(
            conversation_id="owner_real_customer_chat",
            sender_id="owner_real_customer",
            sender_name="常客",
            text="不过我作为一个常来的客人，最重要还是看你能不能帮我组上局",
            message_id="msg_owner_real_resume_chitchat_business_value",
        ),
        "trace_owner_real_resume_chitchat_business_value",
    )
    store.append_assistant_turn("owner_real_customer_chat", "停车也方便", "trace_owner_real_resume_chitchat_parking")
    store.create_game(
        conversation_id="owner_real_customer_chat",
        organizer_id="owner_real_customer",
        organizer_name="常客",
        requirement={
            "game_type": "hangzhou_mahjong",
            "stake": "0.5",
            "smoke_preference": "no_smoke",
            "start_time_kind": "scheduled",
            "start_time": "19:00",
            "planned_start_at": future_planned_start_at("19:00"),
            "duration_hours": 4,
            "needed_seats": 2,
            "user_visible_summary": "还没有，还差俩",
        },
        known_players=[
            {"customer_id": "owner_real_customer", "display_name": "常客", "status": "confirmed"},
            {"customer_id": "smile", "display_name": "笑脸", "status": "confirmed"},
        ],
        trace_id="trace_owner_real_resume_game_setup",
    )


def setup_later_people_count_query(store: SQLiteAgentStore) -> None:
    seed_default_profiles(store)
    store.append_user_turn(
        UserMessage(
            conversation_id="owner_real_customer_chat",
            sender_id="owner_real_customer",
            sender_name="常客",
            text="七点我也ok，但是我只能打四个小时",
            message_id="msg_owner_real_later_duration_limit",
        ),
        "trace_owner_real_later_duration_limit",
    )
    store.append_assistant_turn("owner_real_customer_chat", "ok", "trace_owner_real_later_duration_ack")
    store.create_game(
        conversation_id="owner_real_customer_chat",
        organizer_id="xingyue",
        organizer_name="星月",
        requirement={
            "game_type": "hangzhou_mahjong",
            "stake": "0.5",
            "smoke_preference": "smoking",
            "start_time_kind": "scheduled",
            "start_time": "18:30",
            "planned_start_at": future_planned_start_at("18:30"),
            "needed_seats": 2,
            "user_visible_summary": "两个人，18.30 星月的局，371 她",
        },
        known_players=[
            {"customer_id": "xingyue", "display_name": "星月", "status": "confirmed"},
            {"customer_id": "friend_of_xingyue", "display_name": "她", "status": "confirmed"},
        ],
        trace_id="trace_owner_real_later_people_game_setup",
    )


def setup_reject_smoking_game(store: SQLiteAgentStore) -> None:
    setup_later_people_count_query(store)
    store.append_user_turn(
        UserMessage(
            conversation_id="owner_real_customer_chat",
            sender_id="owner_real_customer",
            sender_name="常客",
            text="现在几个人了啊",
            message_id="msg_owner_real_reject_smoking_status_query",
        ),
        "trace_owner_real_reject_smoking_status_query",
    )
    store.append_assistant_turn(
        "owner_real_customer_chat",
        "两个人，18.30 星月的局，你要打吗，371 她",
        "trace_owner_real_reject_smoking_offer",
    )


def setup_configured_room_inventory(store: SQLiteAgentStore) -> None:
    """为实时回归提供真实的房间清单，而不是空的fixture。"""

    seed_default_profiles(store)
    store.configure_rooms([f"live_eval_room_{index:02d}" for index in range(1, 9)])


def setup_explicit_stake_no_match(store: SQLiteAgentStore) -> None:
    """复现因将冲突的赌注视为可操作而导致的循环。"""

    store.upsert_customer(
        CustomerProfile(
            customer_id="explicit_stake_requester",
            display_name="常客",
            preferred_games=["hangzhou_mahjong"],
            smoke_preference="no_smoke",
        )
    )
    for index, name in enumerate(["候选甲", "候选乙", "候选丙", "候选四"], start=1):
        store.upsert_customer(
            CustomerProfile(
                customer_id=f"stake_candidate_{index}",
                display_name=name,
                preferred_games=["hangzhou_mahjong"],
                preferred_stakes=["1"],
                smoke_preference="no_smoke",
                response_score=0.8,
            )
        )
    store.create_game(
        conversation_id="explicit_stake_existing_pool",
        organizer_id="existing_half_stake_owner",
        organizer_name="",
        requirement={
            "game_type": "hangzhou_mahjong",
            "stake": "0.5",
            "smoke_preference": "no_smoke",
            "start_time_kind": "asap_when_full",
            "needed_seats": 3,
            "user_visible_summary": "0.5无烟，人齐开",
        },
        known_players=[{"customer_id": "existing_half_stake_owner", "display_name": ""}],
        trace_id="trace_explicit_stake_existing_pool",
    )
    store.append_user_turn(
        UserMessage(
            conversation_id="explicit_stake_request",
            sender_id="explicit_stake_requester",
            sender_name="常客",
            text="三缺一，帮我约个无烟局",
            message_id="msg_explicit_stake_previous_user",
        ),
        "trace_explicit_stake_previous_user",
    )
    store.append_assistant_turn(
        "explicit_stake_request",
        "打什么麻将、什么档位？",
        "trace_explicit_stake_previous_assistant",
    )


def build_runtime(client: OpenAICompatibleAgentClient, store: SQLiteAgentStore, trace: InMemoryTraceRecorder, args: argparse.Namespace) -> AgentRuntime:
    """构建运行时，只组合调用方允许进入当前作用域的数据。"""
    return AgentRuntime(
        llm_client=client,
        store=store,
        tool_gateway=ToolGateway(store=store),
        trace_recorder=trace,
        token_budget=TokenBudget(max_tokens_per_call=args.max_tokens_per_call, max_calls_per_turn=args.max_calls_per_turn),
        customer_visible_text_generation_token_budget=TokenBudget(
            max_tokens_per_call=args.max_tokens_per_call,
            max_calls_per_turn=args.max_calls_per_turn,
        ),
        review_token_budget=TokenBudget(max_tokens_per_call=args.max_tokens_per_call, max_calls_per_turn=args.max_calls_per_turn),
        max_steps=args.max_steps,
        llm_timeout_seconds=args.timeout_seconds,
        customer_visible_text_generation_enabled=not args.skip_text_generation,
        customer_visible_text_generation_client=client,
        reply_self_review_enabled=not args.skip_review,
        reply_self_review_client=client,
    )


def setup_single_game_profile(store: SQLiteAgentStore) -> None:
    """复现熟客仅有唯一玩法偏好，其余本次条件仍需客户提供的边界。"""

    store.configure_rooms(["profile_eval_room"])
    store.upsert_customer(CustomerProfile(customer_id="profile_customer", display_name="常客",
                                         preferred_games=["national_standard_mahjong"]))


def setup_unanswered_party_question(store: SQLiteAgentStore) -> None:
    """保留已问但未答的人数与时间，不能把一句同意转成完整槽位。"""

    setup_single_game_profile(store)
    store.append_user_turn(UserMessage(conversation_id="profile_followup", sender_id="profile_customer",
                                     sender_name="常客", text="帮我组个0.5无烟的", message_id="profile_setup_message"), "profile_setup")
    store.append_assistant_turn("profile_followup", "你这边几个人？什么时候打？", "profile_setup")


def setup_mixed_inventory_profile(store: SQLiteAgentStore) -> None:
    """同一国标画像面对混合全店库存，检验盘点范围不受个人偏好覆盖。

    两局均属于独立公开局池，与查询客户没有参加或邀约关联；预置公开摘要，
    让评测只比较盘点所得的玩法、档位和缺口，不依赖内部编号或成员身份。
    """
    setup_single_game_profile(store)
    for game_type, stake, player_count, summary in (
        ("hangzhou_mahjong", "1", 3, "杭麻1块，三缺一"),
        ("national_standard_mahjong", "2", 2, "国标2块，两缺二"),
    ):
        owner_id = f"inventory_owner_{game_type}"
        store.create_game(
            conversation_id=f"inventory_pool_{game_type}", organizer_id=owner_id, organizer_name="发起人",
            requirement={"game_type": game_type, "stake": stake, "smoke_preference": "any",
                         "start_time_kind": "asap_when_full", "user_visible_summary": summary},
            known_players=[{"customer_id": owner_id, "display_name": "发起人", "seat_count": player_count}],
            trace_id="inventory_mixed_pool_setup",
        )


def setup_three_person_capacity_pool(store: SQLiteAgentStore) -> None:
    """同条件两桌只在剩余容量上不同，三人请求方只能匹配可容三座的一桌。"""
    setup_single_game_profile(store)
    for owner_id, seats, summary in (("capacity_open_owner", 1, "国标1块无烟，一缺三"),
                                     ("capacity_tight_owner", 3, "国标1块无烟，三缺一")):
        store.create_game(
            conversation_id=f"pool_{owner_id}", organizer_id=owner_id, organizer_name="发起人",
            requirement={"game_type": "national_standard_mahjong", "stake": "1", "smoke_preference": "no_smoke",
                         "start_time_kind": "asap_when_full", "user_visible_summary": summary},
            known_players=[{"customer_id": owner_id, "display_name": "发起人", "seat_count": seats}],
            trace_id="three_person_capacity_setup",
        )


def setup_custom_game_profile(store: SQLiteAgentStore) -> None:
    """用仅存在于测试配置的新玩法标识检验开放集合，不能依赖程序内置玩法分支。"""

    setup_single_game_profile(store)
    store.upsert_customer(CustomerProfile(customer_id="profile_customer", display_name="常客",
                                         preferred_games=["club_custom_mahjong"],
                                         profile_facts=["测试门店登记的常玩玩法名为竹影牌，类型标识 club_custom_mahjong。"] ))


def setup_no_game_profile(store: SQLiteAgentStore) -> None:
    """模拟没有玩法画像的客户，其他条件由本次消息明确提供。"""

    store.configure_rooms(["profile_eval_room"])
    store.upsert_customer(CustomerProfile(customer_id="profile_customer", display_name="新客"))


def setup_flexible_time_inquiry(store: SQLiteAgentStore) -> None:
    """保存先问下午、再被追问档位和时间的历史，当前消息将明确放宽时间。"""
    from mahjong_agent_runtime.models import ConversationRole, ConversationTurn

    store.configure_rooms(["flexible_time_eval_room"])
    store.upsert_customer(CustomerProfile(
        customer_id="flexible_customer", display_name="客户A", preferred_games=["hangzhou_mahjong"],
        preferred_stakes=["1", "2"], smoke_preference="any",
    ))
    store.append_user_turn(UserMessage(
        conversation_id="flexible_time_inquiry", sender_id="flexible_customer", sender_name="客户A",
        text="下午有人打吗", message_id="flexible_time_initial",
        sent_at=FLEXIBLE_TIME_FIXTURE_ANCHOR-dt.timedelta(minutes=2),
    ), "flexible_time_history")
    # assistant追加接口使用真实now；夹具需显式位于前文user和固定未来current之间。
    store.append_turn("flexible_time_inquiry", ConversationTurn(role=ConversationRole.ASSISTANT,
        content="打多大的，几点方便？", trace_id="flexible_time_history",
        occurred_at=FLEXIBLE_TIME_FIXTURE_ANCHOR-dt.timedelta(minutes=1)))


def setup_flexible_time_with_evening_game(store: SQLiteAgentStore) -> None:
    """提供同档晚间局，只有撤销旧下午限制后才应作为可用现成局提供。"""
    setup_flexible_time_inquiry(store)
    store.create_game(
        conversation_id="evening_game_pool", organizer_id="evening_owner", organizer_name="发起人",
        requirement={
            "game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "any",
            "start_time_kind": "scheduled", "start_time": future_planned_start_at("19:00", anchor=FLEXIBLE_TIME_FIXTURE_ANCHOR),
            "user_visible_summary": "晚上七点1块，三缺一",
        },
        known_players=[{"customer_id": "evening_owner", "display_name": "发起人", "seat_count": 3}],
        trace_id="flexible_time_evening_game",
    )


def setup_range_profile(store: SQLiteAgentStore) -> None:
    """保留用户人工输入的自由文本画像，让真实主模型解释可接受区间。"""
    store.configure_rooms(["range_eval_room"])
    store.upsert_customer(CustomerProfile(customer_id="range_customer", display_name="客户A",
        preferred_games=["red_center_mahjong"], preferred_stakes=["1-3块都行"],
        preferred_time_tags=["晚上"], smoke_preference="any"))


def setup_range_profile_with_game(store: SQLiteAgentStore) -> None:
    """提供区间中间值的现成局，检验模型能否直接查询而非重复询价。"""
    setup_range_profile(store)
    store.create_game(conversation_id="range_existing", organizer_id="range_owner", organizer_name="发起人",
        requirement={"game_type":"red_center_mahjong", "stake":"2", "start_time_kind":"asap_when_full",
                     "smoke_preference":"any", "user_visible_summary":"红中2块，三缺一"},
        known_players=[{"customer_id":"range_owner", "display_name":"发起人", "seat_count":3}],
        trace_id="range-existing-setup")


def setup_range_profile_with_mixed_inventory(store: SQLiteAgentStore) -> None:
    """以同等容量的区间内、区间外及其他玩法库存区分个人过滤与全店盘点。

    保留原画像和旧单局夹具；本受控对照只预置领域事实，不预写主模型动作或通过结论。
    三桌均为三缺一，避免人数或房态差异替代玩法与区间能力的验证。
    """
    setup_range_profile(store)
    for tag, game_type, stake, summary in (
        ("inside", "red_center_mahjong", "2", "红中2块，三缺一"),
        ("outside", "red_center_mahjong", "5", "红中5块，三缺一"),
        ("other_game", "national_standard_mahjong", "2", "国标2块，三缺一"),
    ):
        host = f"range_mixed_{tag}_host"
        store.create_game(conversation_id=f"range_mixed_existing_{tag}", organizer_id=host, organizer_name="牌友",
            requirement={"game_type": game_type, "stake": stake, "start_time_kind": "asap_when_full",
                         "smoke_preference": "any", "user_visible_summary": summary},
            known_players=[{"customer_id": host, "display_name": "牌友", "seat_count": 3}],
            trace_id="range-mixed-inventory-setup")


def setup_range_candidates(store: SQLiteAgentStore) -> None:
    """提供范围内外候选，查询结果必须保留区间并排除明确不相交的底注。"""
    setup_range_profile(store)
    for customer_id, stake in (("range_inside", "2"), ("range_outside", "5")):
        store.upsert_customer(CustomerProfile(customer_id=customer_id, display_name=customer_id,
            preferred_games=["red_center_mahjong"], preferred_stakes=[stake], smoke_preference="any", response_score=1))


def setup_base_cap_profile(store: SQLiteAgentStore) -> None:
    """用日常小底注画像支持档位解释，后端不按玩法名规定其自然语言含义。"""
    store.configure_rooms(["range_eval_room"])
    store.upsert_customer(CustomerProfile(customer_id="cap_customer", display_name="常客",
        preferred_games=["national_standard_mahjong"], preferred_stakes=["1", "2"], smoke_preference="any"))


def setup_base_cap_profile_with_board(store: SQLiteAgentStore) -> None:
    """保留原画像，追加明确授权的沙盒群，以真实发布责任验证私聊组局闭环。"""
    setup_base_cap_profile(store)
    configure_authorized_eval_board(store)


def configure_authorized_eval_board(store: SQLiteAgentStore) -> None:
    """追加显式沙盒群授权；原黄金画像和setup来源保持原样，不改原输入或验收期望。"""
    from mahjong_agent_runtime.group_chat import GroupRoomPolicy

    store.upsert_group_room_policy(GroupRoomPolicy(room_id="owner-eval-authorized-board", managed=True,
        board_enabled=True, outbound_enabled=True, merge_window_seconds=30))


def setup_custom_base_cap_profile(store: SQLiteAgentStore) -> None:
    """在另一门店玩法下提供真实规则背景，检验解释不绑定某个内置玩法名。"""
    store.configure_rooms(["range_eval_room"])
    store.upsert_customer(CustomerProfile(customer_id="cap_customer", display_name="常客",
        preferred_games=["club_custom_mahjong"], preferred_stakes=["1", "2"], smoke_preference="any",
        profile_facts=["过去交流中该客户一直用底注-封顶说本店的档位，常打1块或2块底注。"] ))


def live_eval_scenarios() -> list[LiveEvalScenario]:
    """组合真实模型场景，以工具参数和落库结果验证语义理解与业务边界。"""
    common_forbidden = [
        *IMPLEMENTATION_DETAIL_FORBIDDEN_REPLY_FRAGMENTS,
        *CUSTOMER_SERVICE_FORBIDDEN_REPLY_FRAGMENTS,
    ]
    scenarios = [
        LiveEvalScenario(
            scenario_id="profile_range_empty_inquiry",
            name="画像接受1至3块，下午咨询复用范围且不擅自建局",
            setup=setup_range_profile,
            message=UserMessage(conversation_id="range_empty",sender_id="range_customer",sender_name="客户A",
                text="下午有人打吗",message_id="range_empty_message"),
            required_tool_names=["search_current_games"],
            forbidden_tool_names=["create_game", "search_customers", "create_invite_drafts"],
            expected_tool_result_paths={"search_current_games": [
                {"result.requirement.stake_range": {"min": 1, "max": 3}},
                {"result.matches": [], "result.alternatives": [],
                 "result.customer_reply_contract.evidence_scope.kind": "unfiltered_inventory",
                 "result.customer_reply_contract.evidence_scope.applied_filter_fields": [],
                 "result.customer_reply_contract.evidence_scope.unfiltered_available_pool_empty": True,
                 "result.customer_reply_contract.evidence_scope.supports_narrower_empty_claim": True},
            ]},
            forbidden_reply_contains=["打多大", "几块", "什么档位", "多少的", *common_forbidden],
            expected_active_game_count=0, require_help_offer=True,
        ),
        LiveEvalScenario(
            scenario_id="profile_range_matches_middle_stake",
            name="范围画像能查到2块局，保持咨询边界并自然给出选项",
            setup=setup_range_profile_with_game,
            message=UserMessage(conversation_id="range_match",sender_id="range_customer",sender_name="客户A",
                text="现在有人打吗",message_id="range_match_message"),
            required_tool_names=["search_current_games"],
            forbidden_tool_names=["create_game", "search_customers", "create_invite_drafts"],
            expected_tool_result_paths={"search_current_games":{"result.requirement.stake_range":{"min":1,"max":3},
                "result.customer_reply_contract.search_result_semantics.actionable_match_count":1}},
            forbidden_reply_contains=["打多大", "几块", "什么档位", "多少的", *common_forbidden],
        ),
        LiveEvalScenario(
            scenario_id="profile_range_mixed_inventory_personal_query",
            name="明确个人匹配在混合库存中复用红中与1至3块范围，不能用全店空池捷径",
            setup=setup_range_profile_with_mixed_inventory,
            message=UserMessage(conversation_id="range_mixed_personal", sender_id="range_customer", sender_name="客户A",
                text="我一个人，按我平时打的范围，现在有合适的局吗？", message_id="range_mixed_personal_message"),
            required_tool_names=["search_current_games"],
            forbidden_reply_contains=[*common_forbidden],
            forbidden_tool_names=["create_game", "join_game", "record_candidate_reply", "update_game_requirement",
                "update_game_status", "reserve_room", "search_customers", "create_invite_drafts",
                "create_outbound_message_drafts", "register_waiting_demand", "cancel_waiting_demand",
                "schedule_follow_up", "cancel_follow_up"],
            expected_tool_result_paths={"search_current_games": {
                "result.requirement.game_type": "red_center_mahjong",
                "result.requirement.stake_range": {"min": 1, "max": 3},
                "result.requirement.requesting_party.seat_count": 1,
                "result.customer_reply_contract.search_result_semantics.actionable_match_count": 1,
                "result.matches.0.join_projection.requested_seats": 1,
                "result.matches.0.join_projection.sender_already_joined": False,
                "result.matches.0.join_projection.remaining_seats_after_join": 0,
                "result.matches.0.join_projection.would_fill_game": True,
                "result.alternatives": [],
            }},
            expected_tool_result_unordered_rows={"search_current_games": {"result.matches": [{
                "game.requirement.game_type": "red_center_mahjong", "game.requirement.stake": "2",
                "game.seat_summary.claimed_seats": 3, "game.seat_summary.remaining_seats": 1,
            }]}},
            expected_active_game_count=0,
            expected_status_any=["completed", "waiting_user"],
            forbidden_trace_steps=["action_contract_error", "llm_error"],
            max_reply_chars=None,
        ),
        LiveEvalScenario(
            scenario_id="explicit_range_creates_pending_game",
            name="明确组局授权下保留范围建局，查找范围内候选并准备邀约",
            setup=setup_range_candidates,
            message=UserMessage(conversation_id="range_create",sender_id="range_customer",sender_name="客户A",
                text="红中，底注1到3块都可以，人齐开，我一个人，烟不限，帮我组一桌",message_id="range_create_message"),
            required_tool_names=["search_current_games", "create_game", "search_customers", "create_invite_drafts"],
            expected_tool_result_paths={
                "create_game":{"result.game.requirement.stake_range":{"min":1,"max":3}},
                "search_customers":{"result.requirement.stake_range":{"min":1,"max":3},
                    "result.candidates.0.customer.customer_id":"range_inside"}},
            expected_active_game_count=1,
            expected_active_game_requirement={"stake_range":{"min":1,"max":3}, "stake":None},
            forbidden_reply_contains=["打多大", "几块", "什么档位", *common_forbidden],
        ),
        LiveEvalScenario(
            scenario_id="profile_context_infers_base_and_cap",
            name="结合常打1或2块的画像解释1-256为底注与封顶",
            setup=setup_base_cap_profile,
            require_board_recovery_delivery=True,
            message=UserMessage(conversation_id="cap_infer",sender_id="cap_customer",sender_name="常客",
                text="我一个人，人齐开，烟不限，我打1-256，帮我组一桌",message_id="cap_infer_message"),
            required_tool_names=["search_current_games", "create_game"],
            expected_tool_result_paths={"create_game":{"result.game.requirement.stake":"1", "result.game.requirement.cap_score":256}},
            expected_active_game_count=1,
            expected_active_game_requirement={"stake":"1", "cap_score":256, "stake_range":None},
            forbidden_reply_contains=["底注范围", "1到256", "打多大", *common_forbidden],
        ),
        LiveEvalScenario(
            scenario_id="custom_rules_infer_base_and_cap",
            name="非内置玩法结合门店历史表达解释档位，不依赖玩法字符串分支",
            setup=setup_custom_base_cap_profile,
            message=UserMessage(conversation_id="custom_cap_infer",sender_id="cap_customer",sender_name="常客",
                text="我一个人，人齐开，烟不限，1-256都行，帮我组一桌",message_id="custom_cap_infer_message"),
            required_tool_names=["search_current_games", "create_game"],
            expected_tool_result_paths={"create_game":{"result.game.requirement.stake":"1", "result.game.requirement.cap_score":256}},
            expected_active_game_requirement={"stake":"1", "cap_score":256, "stake_range":None},
            forbidden_reply_contains=["底注范围", "1到256", "打多大", *common_forbidden],
        ),
        LiveEvalScenario(
            scenario_id="explicit_range_overrides_fixed_profile",
            name="本次明确的底注区间覆盖平时固定偏好，封顶仍独立保留",
            setup=setup_base_cap_profile,
            message=UserMessage(conversation_id="range_override",sender_id="cap_customer",sender_name="常客",
                text="这次底注1到3块都能打，256封顶，我一个人，烟不限，人齐开，帮我组",message_id="range_override_message"),
            required_tool_names=["search_current_games", "create_game"],
            expected_tool_result_paths={"create_game":{"result.game.requirement.stake_range":{"min":1,"max":3},
                "result.game.requirement.cap_score":256}},
            expected_active_game_requirement={"stake_range":{"min":1,"max":3},"cap_score":256,"stake":None},
            forbidden_reply_contains=["打多大", "几块", "什么档位", *common_forbidden],
        ),
        LiveEvalScenario(
            scenario_id="explicit_wide_range_is_not_a_cap",
            name="明确的宽底注区间覆盖日常偏好，不能按数字大小误判封顶",
            setup=setup_base_cap_profile,
            message=UserMessage(conversation_id="wide_range",sender_id="cap_customer",sender_name="常客",
                text="这次我说的是底注1元到256元之间都能接受，没有约定封顶。我一个人，烟不限，人齐开，帮我组",message_id="wide_range_message"),
            required_tool_names=["search_current_games", "create_game"],
            expected_tool_result_paths={"create_game":{"result.game.requirement.stake_range":{"min":1,"max":256}}},
            expected_active_game_requirement={"stake_range":{"min":1,"max":256},"cap_score":None,"stake":None},
            forbidden_reply_contains=["256封顶", "打多大", "几块", "什么档位", *common_forbidden],
        ),
        LiveEvalScenario(
            scenario_id="complete_query_is_not_creation_consent",
            name="完整打牌咨询无现成局时提议帮忙问人，等待客户决定而不擅自建局",
            setup=setup_single_game_profile,
            message=UserMessage(conversation_id="complete_query", sender_id="profile_customer", sender_name="常客",
                                text="现在0.5无烟国标有人吗？我们两个人，人齐开，最多四小时。", message_id="complete_query_message"),
            required_tool_names=["search_current_games"],
            forbidden_tool_names=["create_game", "search_customers", "create_invite_drafts"],
            expected_active_game_count=0,
            expected_status_any=["waiting_user"],
            require_help_offer=True,
            forbidden_reply_contains=["已经联系", "联系过了", "问过了", "已经问过", "已经发", *common_forbidden],
        ),
        LiveEvalScenario(
            scenario_id="flexible_time_followup_matches_evening_game",
            name="先问下午后明确时间都行，保留1块并放宽旧时间限制查询晚间局",
            setup=setup_flexible_time_with_evening_game,
            message=UserMessage(
                conversation_id="flexible_time_inquiry", sender_id="flexible_customer", sender_name="客户A",
                text="1块的吧，时间都行的", message_id="flexible_time_current",
                sent_at=FLEXIBLE_TIME_FIXTURE_ANCHOR,
            ),
            required_tool_names=["search_current_games"],
            forbidden_tool_names=["create_game", "search_customers", "create_invite_drafts", "register_waiting_demand"],
            expected_tool_result_paths={"search_current_games": {
                "result.requirement.stake": "1",
                "result.customer_reply_contract.search_result_semantics.actionable_match_count": 1,
            }},
            # 元数据记录字段不等于施加时间限制；拒绝收窄取值，允许省略或明确不限。
            excluded_query_requirement_values={
                "start_time_kind": ["scheduled", "asap_when_full", "下午", "afternoon"],
                "time_preference": ["下午", "afternoon"], "time_range": ["下午", "afternoon"],
            },
            required_reply_any=[["七点", "7点", "19:00"], ["三缺一", "缺一个", "缺一", "还差一", "371"]],
            expected_active_game_count=0,
            expected_status_any=["waiting_user"],
            forbidden_reply_contains=["要不要换", "要不换", "愿意换", "换个时间吗", "换档位吗", "打多大", *common_forbidden],
        ),
        LiveEvalScenario(
            scenario_id="flexible_time_empty_pool_offers_help",
            name="放宽时间后仍无局，提议帮忙问人而不凭空承诺已联系或强迫改条件",
            setup=setup_flexible_time_inquiry,
            message=UserMessage(
                conversation_id="flexible_time_inquiry", sender_id="flexible_customer", sender_name="客户A",
                text="1块的吧，时间都行的", message_id="flexible_time_empty_current",
                sent_at=FLEXIBLE_TIME_FIXTURE_ANCHOR,
            ),
            required_tool_names=["search_current_games"],
            forbidden_tool_names=["create_game", "search_customers", "create_invite_drafts", "register_waiting_demand"],
            expected_tool_result_paths={"search_current_games": {
                "result.requirement.stake": "1",
                "result.customer_reply_contract.search_result_semantics.status": "no_actionable_match",
            }},
            excluded_query_requirement_values={
                "start_time_kind": ["scheduled", "asap_when_full", "下午", "afternoon"],
                "time_preference": ["下午", "afternoon"], "time_range": ["下午", "afternoon"],
            },
            expected_active_game_count=0,
            expected_status_any=["waiting_user"],
            require_help_offer=True,
            forbidden_reply_contains=["要不要换", "要不换", "愿意换", "换个时间吗", "换档位吗",
                                      "已经联系", "联系过了", "问过了", "已经问过", "已经发", *common_forbidden],
        ),
        LiveEvalScenario(
            scenario_id="inventory_only_does_not_offer_recruitment",
            name="明确只盘点且不想招呼人时回答局池，不扩张为招募服务",
            setup=setup_single_game_profile,
            message=UserMessage(
                conversation_id="inventory_only_query", sender_id="profile_customer", sender_name="常客",
                text="只想看一下现在有哪些在组的局，先别帮我问人，也别新组。", message_id="inventory_only_message",
            ),
            required_tool_names=["search_current_games"],
            forbidden_tool_names=["create_game", "search_customers", "create_invite_drafts", "register_waiting_demand"],
            expected_tool_result_paths={"search_current_games": {
                "result.customer_reply_contract.evidence_scope.kind": "unfiltered_inventory",
            }},
            expected_active_game_count=0,
            expected_status_any=["completed"],
            forbidden_reply_contains=["要我帮", "帮你问问", "换时间", "换档", *common_forbidden],
        ),
        LiveEvalScenario(
            scenario_id="inventory_all_games_ignores_personal_profile",
            name="全店盘点读取两种玩法，不以唯一国标画像缩窄范围",
            setup=setup_mixed_inventory_profile,
            message=UserMessage(conversation_id="inventory_all_games", sender_id="profile_customer", sender_name="常客",
                                text="只盘点一下全店现在有哪些在组的局，不限玩法，先别帮我问人，也别新组。",
                                message_id="inventory_all_games_message"),
            required_tool_names=["search_current_games"],
            forbidden_tool_names=["create_game", "search_customers", "create_invite_drafts", "register_waiting_demand",
                                  "check_room_availability", "reserve_room", "join_game", "record_candidate_reply",
                                  "update_game_status", "record_user_memory"],
            expected_tool_result_paths={"search_current_games": {
                "result.customer_reply_contract.evidence_scope.kind": "unfiltered_inventory",
                "result.customer_reply_contract.evidence_scope.applied_filter_fields": [],
            }},
            expected_tool_result_unordered_rows={"search_current_games": {"result.matches": [
                {"game.requirement.game_type": "hangzhou_mahjong", "game.requirement.stake": "1"},
                {"game.requirement.game_type": "national_standard_mahjong", "game.requirement.stake": "2"},
            ]}},
            expected_active_game_count=0, expected_status_any=["completed"],
            max_tool_call_counts={"search_current_games": 1},
            forbidden_reply_contains=["要我帮", "帮你问问", "换时间", "换档", *common_forbidden],
        ),
        LiveEvalScenario(
            scenario_id="inventory_explicit_game_overrides_profile",
            name="客户只盘点杭麻时保留显式范围，不套国标画像也不清空过滤",
            setup=setup_mixed_inventory_profile,
            message=UserMessage(conversation_id="inventory_explicit_game", sender_id="profile_customer", sender_name="常客",
                                text="只盘点一下现在在组的杭麻局，其他玩法先不看，别帮我问人，也别新组。",
                                message_id="inventory_explicit_game_message"),
            required_tool_names=["search_current_games"],
            forbidden_tool_names=["create_game", "search_customers", "create_invite_drafts", "register_waiting_demand",
                                  "check_room_availability", "reserve_room", "join_game", "record_candidate_reply",
                                  "update_game_status", "record_user_memory"],
            expected_tool_result_paths={"search_current_games": {
                "result.requirement.game_type": "hangzhou_mahjong",
                "result.customer_reply_contract.evidence_scope.kind": "submitted_filters_only",
                "result.customer_reply_contract.evidence_scope.applied_filter_fields": ["game_type"],
            }},
            expected_tool_result_unordered_rows={"search_current_games": {"result.matches": [
                {"game.requirement.game_type": "hangzhou_mahjong", "game.requirement.stake": "1"},
            ]}},
            expected_active_game_count=0, expected_status_any=["completed"],
            max_tool_call_counts={"search_current_games": 1},
            forbidden_reply_contains=["要我帮", "帮你问问", "换时间", "换档", *common_forbidden],
        ),
        LiveEvalScenario(
            scenario_id="known_three_person_party_matches_capacity",
            name="己方三人查现成局，只返回可容三座的局且不代为加入或新组",
            setup=setup_three_person_capacity_pool,
            message=UserMessage(conversation_id="three_person_capacity_query", sender_id="profile_customer", sender_name="常客",
                text="我们这边3个人，国标1块无烟人齐开，先看看有能一起坐下的现成局吗？只查一下，不要帮我新组或问人。",
                message_id="three_person_capacity_query_message"),
            required_tool_names=["search_current_games"],
            forbidden_tool_names=["create_game", "search_customers", "create_invite_drafts", "register_waiting_demand",
                                  "check_room_availability", "reserve_room", "join_game", "record_candidate_reply"],
            expected_tool_result_paths={"search_current_games": {
                "result.requirement.requesting_party.seat_count": 3,
                "result.requirement.game_type": "national_standard_mahjong", "result.requirement.stake": "1",
                "result.matches.0.game.requirement.known_player_count": 1,
                "result.matches.0.game.seat_summary.remaining_seats": 3,
                "result.matches.0.join_projection.requested_seats": 3,
                "result.matches.0.join_projection.remaining_seats_after_join": 0,
                "result.matches.0.join_projection.would_fill_game": True,
                "result.matches.0.join_projection.would_overfill_game": False,
                "result.alternatives": [],
            }},
            expected_tool_result_unordered_rows={"search_current_games": {"result.matches": [
                {"game.requirement.game_type": "national_standard_mahjong", "game.requirement.stake": "1",
                 "game.requirement.known_player_count": 1, "game.requirement.needed_seats": 3},
            ]}},
            expected_active_game_count=0, expected_status_any=["completed", "waiting_user"],
            max_tool_call_counts={"search_current_games": 1},
            forbidden_reply_contains=["要我帮", "帮你问问", "换时间", "换档", *common_forbidden],
        ),
        LiveEvalScenario(
            scenario_id="custom_profile_game_preserved",
            name="仅在配置里定义的新玩法原样用于查局，不套回内置玩法示例",
            setup=setup_custom_game_profile,
            message=UserMessage(conversation_id="custom_profile_query", sender_id="profile_customer",
                                sender_name="常客", text="现在最快几点能开", message_id="custom_profile_query_message"),
            forbidden_tool_names=["create_game", "search_customers", "create_invite_drafts", "register_waiting_demand"],
            optional_query_result_fields={"search_current_games": {"result.requirement.game_type": "club_custom_mahjong"}},
            expected_status_without_tool={"search_current_games": ["waiting_user"]},
            expected_active_game_count=0,
            forbidden_reply_contains=["杭麻", "川麻", "打什么麻将", "打什么牌", "什么玩法", *common_forbidden],
        ),
        LiveEvalScenario(
            scenario_id="unknown_game_requires_open_question",
            name="无玩法画像时合理澄清或真实查询公开选项，不凭空限定玩法",
            setup=setup_no_game_profile,
            message=UserMessage(conversation_id="unknown_game_query", sender_id="profile_customer",
                                sender_name="新客", text="我一个人，0.5无烟，人齐开，四小时，帮我约一桌",
                                message_id="unknown_game_query_message"),
            forbidden_tool_names=["create_game", "search_customers", "create_invite_drafts"],
            expected_active_game_count=0, expected_status_any=["waiting_user"],
            forbidden_reply_contains=[*common_forbidden],
        ),
        LiveEvalScenario(
            scenario_id="single_profile_game_used_for_query",
            name="国标熟客一般查局使用唯一画像玩法，不追问杭麻还是川麻",
            setup=setup_single_game_profile,
            message=UserMessage(conversation_id="profile_query", sender_id="profile_customer",
                                sender_name="常客", text="现在最快几点能开", message_id="profile_query_message"),
            forbidden_tool_names=["create_game", "search_customers", "create_invite_drafts", "register_waiting_demand"],
            optional_query_result_fields={"search_current_games": {"result.requirement.game_type": "national_standard_mahjong"}},
            expected_status_without_tool={"search_current_games": ["waiting_user"]},
            expected_active_game_count=0,
            forbidden_reply_contains=["杭麻", "川麻", "打什么麻将", "打什么牌", "什么玩法", *common_forbidden],
        ),
        LiveEvalScenario(
            scenario_id="single_profile_game_complete_query",
            name="唯一国标画像且本次条件充分时必须查局，空池先征询是否帮忙问人",
            setup=setup_single_game_profile,
            message=UserMessage(conversation_id="single_profile_game_complete_query", sender_id="profile_customer",
                                sender_name="常客", text="我一个人，1块，烟不限，现在有局吗？",
                                message_id="single_profile_game_complete_query_message"),
            required_tool_names=["search_current_games"],
            forbidden_tool_names=["create_game", "search_customers", "create_invite_drafts", "register_waiting_demand"],
            expected_tool_result_paths={"search_current_games": {
                "result.requirement.game_type": "national_standard_mahjong", "result.requirement.stake": "1",
                "result.matches": [], "result.alternatives": [],
                "result.customer_reply_contract.search_result_semantics.status": "no_actionable_match",
            }},
            expected_active_game_count=0, expected_status_any=["waiting_user"], require_help_offer=True,
            max_tool_call_counts={"search_current_games": 1},
            forbidden_reply_contains=["杭麻", "川麻", "打什么", "什么玩法", "打多大", "几个人", "有烟无烟",
                                      "已经联系", "联系过了", "问过了", "已经问过", *common_forbidden],
        ),
        LiveEvalScenario(
            scenario_id="custom_profile_game_complete_query",
            name="唯一自定义画像且本次条件充分时必须查局，空池先征询是否帮忙问人",
            setup=setup_custom_game_profile,
            message=UserMessage(conversation_id="custom_profile_game_complete_query", sender_id="profile_customer",
                                sender_name="常客", text="我一个人，1块，烟不限，现在有局吗？",
                                message_id="custom_profile_game_complete_query_message"),
            required_tool_names=["search_current_games"],
            forbidden_tool_names=["create_game", "search_customers", "create_invite_drafts", "register_waiting_demand"],
            expected_tool_result_paths={"search_current_games": {
                "result.requirement.game_type": "club_custom_mahjong", "result.requirement.stake": "1",
                "result.matches": [], "result.alternatives": [],
                "result.customer_reply_contract.search_result_semantics.status": "no_actionable_match",
            }},
            expected_active_game_count=0, expected_status_any=["waiting_user"], require_help_offer=True,
            max_tool_call_counts={"search_current_games": 1},
            forbidden_reply_contains=["杭麻", "川麻", "打什么", "什么玩法", "打多大", "几个人", "有烟无烟",
                                      "已经联系", "联系过了", "问过了", "已经问过", *common_forbidden],
        ),
        LiveEvalScenario(
            scenario_id="single_profile_game_explicit_override",
            name="本次明确川麻覆盖长期国标偏好，已知条件完整时推进建局",
            setup=setup_single_game_profile,
            message=UserMessage(conversation_id="profile_override", sender_id="profile_customer",
                                sender_name="常客", text="我一个人，人齐开，0.5无烟，四小时，帮我组个川麻",
                                message_id="profile_override_message"),
            required_tool_names=["search_current_games", "create_game", "search_customers"],
            expected_tool_result_paths={"create_game": {"result.game.requirement.game_type": "sichuan_mahjong"}},
            expected_active_game_count=1,
            expected_status_any=["completed", "waiting_user"],
            forbidden_reply_contains=["国标", "杭麻", *common_forbidden],
        ),
        LiveEvalScenario(
            scenario_id="confirmation_does_not_answer_missing_party",
            name="帮我问问不补全未回答的人数时间，已知国标偏好不再重问",
            setup=setup_unanswered_party_question,
            message=UserMessage(conversation_id="profile_followup", sender_id="profile_customer",
                                sender_name="常客", text="可以，帮我问问", message_id="profile_followup_message"),
            forbidden_tool_names=["create_game", "search_customers", "create_invite_drafts"],
            expected_active_game_count=0,
            expected_status_any=["waiting_user"],
            required_reply_any=[["几个人", "几位", "多少人", "几人", "一个人"], ["几点", "什么时候", "时间", "人齐开"]],
            forbidden_reply_contains=["杭麻", "川麻", "什么玩法", *common_forbidden],
        ),
        LiveEvalScenario(
            scenario_id="configured_room_inventory_query",
            name="房态查询必须读取已配置库存，不得把空 fixture 说成还没录",
            setup=setup_configured_room_inventory,
            message=UserMessage(
                conversation_id="configured_room_inventory_chat",
                sender_id="owner_real_customer",
                sender_name="常客",
                text="今天房间满了吗",
                message_id="msg_configured_room_inventory_query",
            ),
            required_tool_names=["check_room_availability"],
            expected_tool_result_paths={
                "check_room_availability": {
                    "result.configured": True,
                    "result.room_count": 8,
                    "result.available_count": 8,
                }
            },
            expected_status_any=["completed", "waiting_user"],
            max_tool_call_counts={"check_room_availability": 1},
            forbidden_reply_contains=["还没录", "没录", "未录", "未配置", "不确定满没满", *common_forbidden],
        ),
        LiveEvalScenario(
            scenario_id="explicit_stake_conflict_must_not_loop",
            name="明确要 1 块时不得把 0.5 局当匹配并重复搜索",
            setup=setup_explicit_stake_no_match,
            message=UserMessage(
                conversation_id="explicit_stake_request",
                sender_id="explicit_stake_requester",
                sender_name="常客",
                text="杭麻1块无烟马上开",
                message_id="msg_explicit_stake_current_user",
            ),
            required_tool_names=["search_current_games", "create_game", "search_customers"],
            expected_tool_result_paths={
                "search_current_games": {
                    "result.requirement.requesting_party.seat_count": 3,
                    "result.matches": [],
                    "result.alternatives": [],
                    "result.customer_reply_contract.search_result_semantics.status": "no_actionable_match",
                },
                "create_game": {
                    "result.game.requirement.known_player_count": 3,
                    "result.game.requirement.needed_seats": 1,
                    "result.game.requirement.seat_format": "371",
                    "result.game.seat_summary.claimed_seats": 3,
                    "result.game.seat_summary.remaining_seats": 1,
                },
                "search_customers": {
                    "result.requirement.known_player_count": 3,
                    "result.requirement.needed_seats": 1,
                    "result.exclude_customer_ids": ["explicit_stake_requester"],
                },
            },
            expected_status_any=["completed", "waiting_user"],
            expected_active_game_count=1,
            expected_active_game_seat_summary={
                "claimed_seats": 3,
                "remaining_seats": 1,
            },
            expected_active_game_requirement={
                "known_player_count": 3,
                "needed_seats": 1,
                "seat_format": "371",
                "stake": "1",
            },
            max_tool_call_counts={
                "search_current_games": 1,
                "create_game": 1,
                "search_customers": 1,
                "create_invite_drafts": 1,
            },
            forbidden_reply_contains=["转人工", "0.5的局", "五毛的局", *common_forbidden],
            forbidden_trace_steps=[
                "action_contract_error",
                "llm_error",
                "agent_loop_detected",
                "agent_loop_aborted",
            ],
        ),
        LiveEvalScenario(
            scenario_id="profile_default_matched_game",
            name="画像默认 0.5/一人，先查七点三缺一，再短句确认",
            setup=setup_profile_default_matched_game,
            message=UserMessage(
                conversation_id="owner_real_customer_chat",
                sender_id="owner_real_customer",
                sender_name="常客",
                text="帮我约个6.30无烟的",
                message_id="msg_owner_real_live_eval_profile_default",
                sent_at=PROFILE_DEFAULT_FIXTURE_ANCHOR,
            ),
            required_tool_names=["search_current_games"],
            expected_tool_result_paths={
                "search_current_games": {
                    "result.requirement.game_type": "hangzhou_mahjong",
                    "result.requirement.stake": "0.5",
                    "result.requirement.smoke_preference": "no_smoking",
                    "result.requirement.start_time": "18:30",
                    "result.matches.0.join_projection.requested_seats": 1,
                    "result.matches.0.join_projection.remaining_seats_after_join": 0,
                    "result.matches.0.join_projection.would_fill_game": True,
                }
            },
            required_reply_any=[
                ["七点", "7点", "19:00"],
                ["三缺一", "371", "缺一", "还差一个"],
                ["可以不", "行不", "可以吗"],
            ],
            forbidden_reply_contains=["打多大", "几个人", "0.5", "无烟", "打吗", "来吗", "来不", *common_forbidden],
        ),
        LiveEvalScenario(
            scenario_id="future_reservation_waits_for_recruitment_window",
            name="明天预约立即入局列表，但在 T-2 小时前不私聊候选人",
            setup=seed_default_profiles,
            message=UserMessage(
                conversation_id="owner_future_reservation_chat",
                sender_id="owner_real_customer",
                sender_name="常客",
                text="帮我约明天下午一点的财敲，1块无烟，272",
                message_id="msg_owner_real_live_eval_future_reservation",
            ),
            required_tool_names=["create_game"],
            forbidden_tool_names=["search_customers", "create_invite_drafts"],
            expected_tool_result_paths={
                "create_game": {
                    "result.recruitment_policy.status": "scheduled",
                }
            },
            required_reply_any=[
                ["好", "行", "记上", "安排", "留意", "帮你看着"],
            ],
            forbidden_reply_contains=[
                "已经问",
                "问了",
                "候选人",
                "两小时",
                "T-2",
                "定时",
                "调度",
                *common_forbidden,
            ],
            expected_active_game_count=1,
            expected_active_game_requirement={
                "game_type": "hangzhou_mahjong",
                "game_variant": "caiqiao",
                "stake": "1",
                "smoke_preference": "no_smoking",
                "start_time_kind": "scheduled",
                "known_player_count": 2,
                "needed_seats": 2,
            },
            max_reply_chars=40,
        ),
        LiveEvalScenario(
            scenario_id="public_nickname_lookup",
            name="用户追问哪些人时只给公开昵称，不暴露私有备注",
            setup=setup_public_nickname_lookup,
            message=UserMessage(
                conversation_id="owner_real_customer_chat",
                sender_id="owner_real_customer",
                sender_name="常客",
                text="哪些人啊",
                message_id="msg_owner_real_live_eval_public_nickname",
            ),
            required_reply_contains=["夏日", "笑脸"],
            forbidden_reply_contains=[
                "发起人",
                "组织者",
                "老板备注",
                "客户画像",
                "高响应",
                "可室外抽烟",
                "偶尔迟到",
                "熟人带局",
                "经常喊人",
                "不稳定",
                "私有",
                "三缺一",
                "371",
                "还差",
                "还缺",
                "打吗",
                "来吗",
                "可以不",
                *common_forbidden,
            ],
            forbidden_model_context_contains=[
                "夏日-老板备注",
                "笑脸-老板备注",
                "可星-老板备注",
                "可室外抽烟",
                "偶尔迟到",
                "熟人带局",
                "经常喊人",
                "不稳定",
            ],
            expected_active_game_count=1,
            expected_active_game_seat_summary={
                "claimed_seats": 3,
                "remaining_seats": 1,
            },
            expected_active_game_requirement={
                "stake": "0.5",
                "smoke_preference": "no_smoke",
                "start_time": "19:00",
                "needed_seats": 1,
                "user_visible_summary": "七点三缺一",
            },
        ),
        LiveEvalScenario(
            scenario_id="accept_existing_offer_marks_game_ready",
            name="用户回复也可以，应确认加入现成局并把局推到人齐",
            setup=setup_accept_existing_offer,
            message=UserMessage(
                conversation_id="owner_real_customer_chat",
                sender_id="owner_real_customer",
                sender_name="常客",
                text="也可以",
                message_id="msg_owner_real_live_eval_accept_existing_offer",
            ),
            required_tool_name_any=["join_game", "record_candidate_reply"],
            expected_any_tool_result_paths={
                "join_game": {
                    "result.cross_game_commitment.released_participations.0.customer_id": "owner_real_customer",
                },
                "record_candidate_reply": {
                    "result.cross_game_commitment.released_participations.0.customer_id": "owner_real_customer",
                },
            },
            forbidden_tool_names=["search_current_games", "search_customers", "create_game", "create_invite_drafts"],
            required_reply_any=[["ok", "okk", "好", "可以"]],
            forbidden_reply_contains=[
                "打多大",
                "几个人",
                "什么玩法",
                "要组",
                "帮你问问",
                "七点",
                "7点",
                "无烟",
                "0.5",
                "人齐",
                "到时候",
                "来。",
                "候选人",
                "邀约草稿",
                *common_forbidden,
            ],
            expected_active_game_count=1,
            expected_active_game_status="ready",
            expected_active_game_seat_summary={
                "claimed_seats": 4,
                "remaining_seats": 0,
            },
            expected_active_game_requirement={
                "stake": "0.5",
                "smoke_preference": "no_smoke",
                "start_time": "19:00",
                "needed_seats": 0,
                "user_visible_summary": "七点三缺一",
            },
        ),
        LiveEvalScenario(
            scenario_id="duration_rejection_not_chitchat",
            name="已拉群后用户说 5 小时不行，按时长协商失败/退出处理",
            setup=setup_duration_rejection,
            message=UserMessage(
                conversation_id="owner_real_customer_chat",
                sender_id="owner_real_customer",
                sender_name="常客",
                text="不行啊",
                message_id="msg_owner_real_live_eval_duration_rejection",
            ),
            required_tool_name_any=["record_candidate_reply", "update_context_checkpoint"],
            expected_tool_result_paths={
                "record_candidate_reply": {
                    "result.recorded_status": "declined",
                    "result.game.status": "forming",
                    "result.game.seat_summary.claimed_seats": 3,
                    "result.game.seat_summary.remaining_seats": 1,
                    "result.game.requirement.duration_hours": 5,
                    "result.game.requirement.seat_claims.1.contact_id": "owner_real_customer",
                    "result.game.requirement.seat_claims.1.status": "declined",
                }
            },
            required_reply_any=[["好吧", "好的", "行", "那先不算你", "那这桌先不排你"]],
            forbidden_reply_contains=[
                "我帮你问问",
                "要组",
                "有消息",
                "再看看",
                "5小时可以",
                "能打多久",
                "想打多久",
                "其他想法",
                "？",
                "?",
                *common_forbidden,
            ],
        ),
        LiveEvalScenario(
            scenario_id="duration_limit_update_should_persist",
            name="用户补充只能打四小时，应写入上下文约束而不是当普通闲聊",
            setup=setup_duration_limit_update,
            message=UserMessage(
                conversation_id="owner_real_customer_chat",
                sender_id="owner_real_customer",
                sender_name="常客",
                text="七点我也ok，但是我只能打四个小时",
                message_id="msg_owner_real_live_eval_duration_limit",
            ),
            required_tool_names=["record_user_memory"],
            forbidden_tool_names=[
                "record_candidate_reply",
                "search_current_games",
                "search_customers",
                "create_game",
                "create_invite_drafts",
            ],
            required_reply_any=[["ok", "好的", "好", "行", "收到", "记下", "知道了", "没问题"]],
            forbidden_reply_contains=[
                "要组",
                "打多大",
                "几个人",
                "5小时",
                "五小时",
                "候选人",
                *common_forbidden,
            ],
            expected_active_game_count=1,
            expected_active_game_requirement={
                "stake": "0.5",
                "smoke_preference": "no_smoke",
                "start_time": "19:00",
                "needed_seats": 2,
                "user_visible_summary": "还没有，还差俩",
            },
            expected_task_memory_contains=[
                {
                    "conversation_id": "owner_real_customer_chat",
                    "customer_id": "owner_real_customer",
                    "field": "max_duration_hours",
                    "value": 4,
                    "status": "active",
                }
            ],
            require_task_memory_current_context=True,
        ),
        LiveEvalScenario(
            scenario_id="casual_chat_should_not_pollute_business_state",
            name="AI/运营闲聊可回复，但不能污染当前组局状态",
            setup=setup_casual_chat_should_not_pollute_business_state,
            message=UserMessage(
                conversation_id="owner_real_customer_chat",
                sender_id="owner_real_customer",
                sender_name="常客",
                text="好奇问一下，你们是不是每天都要人工找人组局啊，如果给你一个AI帮你组局你会用吗",
                message_id="msg_owner_real_live_eval_casual_chat",
            ),
            forbidden_tool_names=[
                "search_current_games",
                "search_customers",
                "create_game",
                "create_invite_drafts",
                "record_candidate_reply",
                "update_game_status",
            ],
            required_reply_any=[["费脑", "条件", "确实", "麻烦", "精细"]],
            # 运营闲聊是否被改成个人约局由独立语义判定；身份与泄露硬检查继续执行。
            forbidden_reply_contains=[
                "我是AI",
                *common_forbidden,
            ],
            expected_active_game_count=1,
            expected_active_game_requirement={
                "stake": "0.5",
                "smoke_preference": "no_smoke",
                "start_time_kind": "scheduled",
                "start_time": "19:00",
                "needed_seats": 1,
                "user_visible_summary": "七点三缺一",
            },
        ),
        LiveEvalScenario(
            scenario_id="quoted_correction_should_not_pollute_business_state",
            name="引用闲聊更正片段时不能污染当前组局状态",
            setup=setup_casual_chat_should_not_pollute_business_state,
            message=UserMessage(
                conversation_id="owner_real_customer_chat",
                sender_id="owner_real_customer",
                sender_name="常客",
                text="一致",
                message_id="msg_owner_real_live_eval_quoted_correction",
                quoted_message=QuotedMessageRef(
                    message_id="msg_owner_real_live_eval_quoted_correction_source",
                    sender_id="owner_real_customer",
                    sender_name="常客",
                    text="真人回复跟这种回复还是有区别的，除非回复能做到语气什么的和人一直",
                ),
            ),
            forbidden_tool_names=[
                "search_current_games",
                "search_customers",
                "create_game",
                "create_invite_drafts",
                "record_candidate_reply",
                "update_game_status",
                "update_context_checkpoint",
            ],
            required_reply_any=[["对", "嗯", "是", "确实", "哈哈", "懂", "明白", "ok", "OK"]],
            forbidden_reply_contains=[
                "要组一个吗",
                "打多大",
                "几个人",
                "什么玩法",
                "七点",
                "7点",
                "三缺一",
                "371",
                "打吗",
                "来吗",
                *common_forbidden,
            ],
            expected_active_game_count=1,
            expected_active_game_requirement={
                "stake": "0.5",
                "smoke_preference": "no_smoke",
                "start_time_kind": "scheduled",
                "start_time": "19:00",
                "needed_seats": 1,
                "user_visible_summary": "七点三缺一",
            },
        ),
        LiveEvalScenario(
            scenario_id="resume_status_after_casual_chat",
            name="长闲聊后用户问局况，应接回当前组局状态",
            setup=setup_resume_status_after_casual_chat,
            message=UserMessage(
                conversation_id="owner_real_customer_chat",
                sender_id="owner_real_customer",
                sender_name="常客",
                text="所以现在有人了吗",
                message_id="msg_owner_real_live_eval_resume_status",
            ),
            forbidden_tool_names=["create_game", "create_invite_drafts"],
            required_reply_any=[
                ["还没有", "还差俩", "还差两个", "差俩", "差两个"],
            ],
            forbidden_reply_contains=[
                "你要组吗",
                "要组一个吗",
                "打吗",
                "来吗",
                "可以不",
                "0.5",
                "无烟",
                "打多大",
                "几个人",
                "什么玩法",
                "重新",
                *common_forbidden,
            ],
            expected_active_game_count=1,
            expected_active_game_seat_summary={
                "claimed_seats": 2,
                "remaining_seats": 2,
            },
            expected_active_game_requirement={
                "stake": "0.5",
                "smoke_preference": "no_smoke",
                "start_time_kind": "scheduled",
                "start_time": "19:00",
                "duration_hours": 4,
                "needed_seats": 2,
                "user_visible_summary": "还没有，还差俩",
            },
        ),
        LiveEvalScenario(
            scenario_id="later_people_count_query",
            name="用户再次问几个人，应回答当前局况，不重新建局",
            setup=setup_later_people_count_query,
            message=UserMessage(
                conversation_id="owner_real_customer_chat",
                sender_id="owner_real_customer",
                sender_name="常客",
                text="现在几个人了啊",
                message_id="msg_owner_real_live_eval_later_people",
            ),
            forbidden_tool_names=["create_game", "create_invite_drafts"],
            required_reply_any=[
                ["两个人", "2个人", "两个"],
                ["18.30", "18:30", "六点半"],
                ["星月", "打吗", "要打吗", "可以不", "还差2", "还差两个"],
            ],
            forbidden_reply_contains=[
                "要组一个吗",
                "你几个人",
                "打什么档位",
                "已发送",
                "候选人",
                "邀约草稿",
                *common_forbidden,
            ],
            expected_active_game_count=1,
            expected_active_game_seat_summary={
                "claimed_seats": 2,
                "remaining_seats": 2,
            },
            expected_active_game_requirement={
                "stake": "0.5",
                "smoke_preference": "smoking",
                "start_time_kind": "scheduled",
                "start_time": "18:30",
                "needed_seats": 2,
                "user_visible_summary": "两个人，18.30 星月的局，371 她",
            },
        ),
        LiveEvalScenario(
            scenario_id="reject_smoking_game_updates_preference",
            name="用户拒绝非无烟局，应记录反馈和无烟限制",
            setup=setup_reject_smoking_game,
            message=UserMessage(
                conversation_id="owner_real_customer_chat",
                sender_id="owner_real_customer",
                sender_name="常客",
                text="不打哈，我女朋友让我打无烟的",
                message_id="msg_owner_real_live_eval_reject_smoking",
            ),
            required_tool_name_any=["record_candidate_reply", "update_context_checkpoint"],
            forbidden_tool_names=["search_current_games", "search_customers", "create_game", "create_invite_drafts"],
            equivalent_state_tracking_memory={"memory_type":"preference","field":"smoke_preference","value":"no_smoking"},
            required_reply_any=[["okk", "ok", "好", "好的", "行", "先不排", "不排你"]],
            forbidden_reply_contains=[
                "有烟也可以",
                "再考虑",
                "我已经加你",
                "帮你问问",
                "要组",
                "候选人",
                *common_forbidden,
            ],
        ),
    ]


    records = read_product_records()
    return apply_query_option_policy_to_scenarios(
        apply_reply_presentation_to_scenarios(scenarios, records), records)


def _recovery_diagnostic_identities(diagnostics: Any) -> tuple[list[str], list[str]]:
    """收集真实调用叶节点身份；fallback包装与primary共用ID，不能把包装误计为重放。"""
    calls: list[str] = []
    attempts: list[str] = []
    if not isinstance(diagnostics, dict):
        return calls, attempts
    if diagnostics.get("route") == "fallback":
        for branch in ("primary", "fallback"):
            branch_calls, branch_attempts = _recovery_diagnostic_identities(diagnostics.get(branch))
            calls.extend(branch_calls)
            attempts.extend(branch_attempts)
    else:
        if isinstance(diagnostics.get("call_id"), str):
            calls.append(diagnostics["call_id"])
        for attempt in diagnostics.get("attempts", []) if isinstance(diagnostics.get("attempts"), list) else []:
            if isinstance(attempt, dict) and isinstance(attempt.get("attempt_id"), str):
                attempts.append(attempt["attempt_id"])
    return calls, attempts


def _recovery_feedback_in_context(payload: Any, feedback: dict[str, Any], *, source_required: bool) -> bool:
    """核对最小反馈投影；同文案的两次错误必须由各自来源区分，不能借旧指令冒充回灌。"""
    if not isinstance(payload, dict):
        return False
    projected = feedback.get("result") or {}
    for field in ("turn_tool_evidence", "previous_tool_results"):
        evidence = payload.get(field)
        for item in evidence if isinstance(evidence, list) else []:
            if not isinstance(item, dict) or not isinstance(item.get("result"), dict):
                continue
            result = item["result"]
            if (item.get("name") == "agent_action_contract" and item.get("called") is False
                    and item.get("allowed") is False and item.get("error") == feedback.get("error")
                    and bool(result.get("instruction")) and result["instruction"] == projected.get("instruction")
                    and (not source_required or result.get("response_source") == projected.get("response_source"))):
                return True
    return False


def _recovery_prompt_payload(prompt: Any) -> dict[str, Any] | None:
    """读取实际发送的最后一条user JSON；不能以尚未发送的context_built代替模型输入。"""
    if not isinstance(prompt, dict) or not isinstance(prompt.get("messages"), list):
        return None
    for message in reversed(prompt["messages"]):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        try:
            payload = json.loads(message.get("content", ""))
        except (ValueError, TypeError):
            return None
        return payload if isinstance(payload, dict) else None
    return None


def assess_action_contract_recovery(
    trace_events: list[Any],
    *,
    runtime_status: str,
    final_reply: str,
    business_checks_passed: bool,
    semantic_passed: bool,
) -> dict[str, Any]:
    """以真实执行边界证明动作合同错误被安全纠正，不以最终成功或模型自评代替证据。

    错误响应所在步骤不得调用业务工具、改变业务状态或输出客户回复；拒绝反馈必须
    回灌给后续模型。后续需有通过原合同的动作与成功工具反馈/终态输出。其他业务、
    语义或工具错误不属于这项豁免，缺失追踪证据也不能算作已恢复。
    """
    from collections import Counter
    from mahjong_agent_runtime.action_contract import validate_action_contract
    from mahjong_agent_runtime.llm import validate_model_response_source

    internal_results = {"agent_action_contract", "agent_progress_guard", "customer_visible_text_generation", "customer_visible_content_review"}
    errors = [(index, event) for index, event in enumerate(trace_events) if event.step == "action_contract_error"]
    metrics: dict[str, Any] = {
        "action_contract_error_count": len(errors), "safe_recovered_count": 0,
        "unrecovered_count": 0, "unsafe_rejected_step_count": 0, "unresolved_tool_error_count": 0,
        "quality_status": "passed", "recovery_classification": "clean", "events": [], "unresolved_tool_errors": [],
    }
    # 其他工具错误与权限拒绝不因“没有动作合同错误”而被算作 clean。
    # 当前只承认下文可证明的动作合同恢复，不自动推断其他拒绝已被处置。
    for index, event in enumerate(trace_events):
        content = event.content if isinstance(event.content, dict) else {}
        failed_tool = event.step == "tool_result" and content.get("name") not in internal_results and (
            bool(content.get("error")) or content.get("allowed") is False)
        denied = event.step in {"tool_permission_checked", "tool_authorization_checked"} and content.get("allowed") is False
        if failed_tool or denied:
            metrics["unresolved_tool_errors"].append({"event_index": index, "step": event.step,
                "tool_name": content.get("name") or content.get("tool_name"), "error": content.get("error")})
    metrics["unresolved_tool_error_count"] = len(metrics["unresolved_tool_errors"])
    if not errors:
        if not business_checks_passed or not semantic_passed or metrics["unresolved_tool_error_count"]:
            metrics["quality_status"] = "failed"
        if metrics["unresolved_tool_error_count"]:
            metrics["recovery_classification"] = "unresolved_tool_error"
        return metrics

    responses = [(index, event) for index, event in enumerate(trace_events) if event.step == "llm_response"]
    # 历史非空trace不补造新来源；一旦已有来源字段，本次链路不得通过删掉某一环退回旧口径。
    source_required = any(isinstance(event.content, dict) and (
        "response_source" in event.content or (event.step == "contract_error_feedback"
        and isinstance(event.content.get("result"), dict) and "response_source" in event.content["result"]))
        for event in trace_events)
    call_counts: Counter[str] = Counter()
    attempt_counts: Counter[str] = Counter()
    for item in trace_events:
        content = item.content if isinstance(item.content, dict) else {}
        call_ids, attempt_ids = _recovery_diagnostic_identities(content.get("completion_diagnostics"))
        call_counts.update(call_ids)
        attempt_counts.update(attempt_ids)
    duplicate_identity = any(count > 1 for count in call_counts.values()) or any(count > 1 for count in attempt_counts.values())
    response_trace = getattr(responses[0][1], "trace_id", None) if responses else None
    response_steps = [event.content.get("step_index") if isinstance(event.content, dict) else None for _, event in responses]
    response_boundary_invalid = (not isinstance(response_trace, str) or any(getattr(event, "trace_id", None) != response_trace
        for _, event in responses) or any(type(step) is not int or step <= 0 for step in response_steps)
        or any(left >= right for left, right in zip(response_steps, response_steps[1:])))
    invalid_source = source_required and any(not isinstance(event.content, dict) or not validate_model_response_source(
        event.content.get("response_source"), raw_response=event.content.get("content"),
        trace_id=getattr(event, "trace_id", None), step_index=event.content.get("step_index"),
        # 缺失已落盘诊断明确传空字典，不借内存字符串对象残留的隐藏属性补证。
        diagnostics=event.content.get("completion_diagnostics", {})) for _, event in responses)
    segments = []
    for offset, (start, response) in enumerate(responses):
        end = responses[offset + 1][0] if offset + 1 < len(responses) else len(trace_events)
        request_start = responses[offset - 1][0] + 1 if offset > 0 else 0
        requests = [item for item in trace_events[request_start:start]
                    if item.step in {"llm_prompt", "llm_prompt_after_context_compaction"}]
        request = requests[-1] if requests else None
        request_proven = (request is not None and getattr(request, "trace_id", None) == getattr(response, "trace_id", None)
            and isinstance(request.content, dict) and isinstance(response.content, dict)
            and request.content.get("step_index") == response.content.get("step_index")
            and _recovery_prompt_payload(request.content) is not None)
        events = trace_events[start:end]
        proposed = [item.content for item in events if item.step == "action_proposed" and isinstance(item.content, dict)]
        # 服务端AgentAction序列化会补全可选call_id/depends_on为None；JSON合同则用字段
        # 是否存在识别依赖图模式。只移除这些缺省值后复核，不修改实际动作或放宽执行合同。
        checked_action = dict(proposed[0]) if len(proposed) == 1 else {}
        if isinstance(checked_action.get("tool_calls"), list):
            checked_action["tool_calls"] = [{key: value for key, value in call.items()
                if not (key in {"call_id", "depends_on"} and value is None)} if isinstance(call, dict) else call
                for call in checked_action["tool_calls"]]
        legal = len(proposed) == 1 and not validate_action_contract(checked_action) and not any(item.step == "action_contract_error" for item in events)
        action = proposed[0] if legal else None
        successful_results = [item.content for item in events if item.step == "tool_result" and isinstance(item.content, dict)
            and item.content.get("name") not in internal_results and item.content.get("called") is True
            and item.content.get("allowed") is True and not item.content.get("error")]
        called = [item.content.get("call", {}) for item in events if item.step == "tool_called" and isinstance(item.content, dict)]
        trace_complete = Counter(item.get("name") for item in called) == Counter(item.get("name") for item in successful_results)
        for item in events:
            if item.step in {"state_transition", "state_transition_replayed"}:
                trace_complete = trace_complete and any(item.content in (result.get("state_transitions") or []) for result in successful_results)
        tool_feedback = False
        terminal = False
        if action and action.get("objective_status") == "needs_tool":
            wanted = list(action.get("tool_calls") or [])
            success_counts = Counter(item.get("name") for item in successful_results)
            called_counts = Counter(item.get("name") for item in called)
            tool_feedback = bool(wanted) and all(success_counts[name] >= count and called_counts[name] >= count
                for name, count in Counter(item.get("name") for item in wanted).items())
            for call in wanted:
                call_id = call.get("call_id")
                if call_id:
                    tool_feedback = tool_feedback and any(item.get("name") == call.get("name") and item.get("call_id") == call_id for item in successful_results)
        elif action and action.get("objective_status") == runtime_status and not called:
            terminal = any(item.step == "final_output" and isinstance(item.content, dict)
                and item.content.get("reply") == final_reply and item.content.get("objective_status") == runtime_status for item in events)
        segments.append({"start": start, "end": end, "response": response, "events": events,
                         "legal": bool(legal), "completed": bool(tool_feedback or terminal), "terminal": terminal,
                         "tool_trace_complete": bool(trace_complete), "request_proven": bool(request_proven)})

    final_legal = any(segment["legal"] and segment["terminal"] for segment in segments)

    for index, event in errors:
        detail = event.content if isinstance(event.content, dict) else {}
        segment = next((part for part in segments if part["start"] <= index < part["end"]), None)
        reasons: list[str] = []
        unsafe: list[str] = []
        feedback = False
        matched_feedback: dict[str, Any] | None = None
        returned = False
        raw = ""
        if segment is None:
            reasons.append("missing_rejected_model_response")
        else:
            response_event = segment["response"]
            response = response_event.content if isinstance(response_event.content, dict) else {}
            raw = response.get("content")
            strict_source = source_required or raw == ""
            source = response.get("response_source")
            if (type(detail.get("step_index")) is not int or detail["step_index"] != response.get("step_index")
                    or not isinstance(raw, str) or not isinstance(detail.get("errors"), list) or not detail["errors"]
                    or getattr(event, "trace_id", None) != getattr(response_event, "trace_id", None)):
                reasons.append("invalid_rejected_response_identity")
            if strict_source and (invalid_source or not validate_model_response_source(source, raw_response=raw,
                    trace_id=getattr(response_event, "trace_id", None), step_index=response.get("step_index"),
                    diagnostics=response.get("completion_diagnostics", {})) or detail.get("response_source") != source):
                reasons.append("rejected_response_source_not_proven")
            if sum(item.step == "action_contract_error" for item in segment["events"]) != 1:
                reasons.append("rejected_response_has_ambiguous_error_events")
            if not strict_source and sum(isinstance(part["response"].content, dict)
                    and part["response"].content.get("content") == raw
                    and any(item.step == "action_contract_error" for item in part["events"]) for part in segments) != 1:
                reasons.append("legacy_rejected_response_identity_ambiguous")
            if duplicate_identity:
                reasons.append("model_call_or_attempt_identity_reused")
            if response_boundary_invalid:
                reasons.append("model_response_trace_or_step_boundary_invalid")
            if any(not part["request_proven"] for part in segments):
                reasons.append("model_response_request_not_proven")
            for item in segment["events"]:
                content = item.content if isinstance(item.content, dict) else {}
                if item.step == "tool_called":
                    unsafe.append("rejected_step_called_tool")
                if item.step == "tool_result" and content.get("name") not in internal_results and content.get("called") is True:
                    unsafe.append("rejected_step_executed_business_tool")
                if item.step in {"state_transition", "state_transition_replayed"}:
                    unsafe.append("rejected_step_changed_business_state")
                if item.step == "final_output" and content.get("reply"):
                    unsafe.append("rejected_step_emitted_reply")
                if item.step == "contract_error_feedback":
                    result = content.get("result") if isinstance(content.get("result"), dict) else {}
                    matched = (content.get("name") == "agent_action_contract" and content.get("called") is False
                        and content.get("allowed") is False and result.get("raw_response") == raw and result.get("errors") == detail.get("errors")
                        and getattr(item, "trace_id", None) == getattr(response_event, "trace_id", None)
                        and (not strict_source or result.get("response_source") == source))
                    feedback = feedback or matched
                    if matched:
                        matched_feedback = content
            if matched_feedback is not None:
                next_segment = next((part for part in segments if part["start"] == segment["end"]), None)
                next_step = next_segment["response"].content.get("step_index") if next_segment is not None else None
                contexts = [item for item in segment["events"] if item.step == "context_built"]
                prompts = [item for item in segment["events"] if item.step in {"llm_prompt", "llm_prompt_after_context_compaction"}]
                # 压缩后prompt才是真正的请求；早期包含反馈的草稿不能掩盖最终请求已丢失反馈。
                prompt = prompts[-1] if prompts else None
                returned = (next_segment is not None and type(next_step) is int and type(detail.get("step_index")) is int
                    and next_step > detail["step_index"] and len(contexts) > 0
                    and getattr(contexts[-1], "trace_id", None) == getattr(response_event, "trace_id", None)
                    and _recovery_feedback_in_context(contexts[-1].content, matched_feedback, source_required=strict_source)
                    and prompt is not None and getattr(prompt, "trace_id", None) == getattr(response_event, "trace_id", None)
                    and isinstance(prompt.content, dict) and prompt.content.get("step_index") == next_step
                    and _recovery_feedback_in_context(_recovery_prompt_payload(prompt.content), matched_feedback, source_required=strict_source))
        if not feedback or not returned:
            reasons.append("rejection_feedback_not_proven_in_later_model_context")
        later = [part for part in segments if part["start"] > index and part["legal"] and part["completed"]]
        if not later or not final_legal:
            reasons.append("missing_later_legal_execution_and_terminal_output")
        if any(part["start"] > index and not part["tool_trace_complete"] for part in segments):
            reasons.append("later_tool_execution_or_state_feedback_not_proven")
        if metrics["unresolved_tool_error_count"]:
            reasons.append("other_tool_or_permission_error_not_exempt")
        if not business_checks_passed or not semantic_passed:
            reasons.append("business_state_or_semantic_checks_failed")
        reasons.extend(unsafe)
        recovered = not reasons
        metrics["events"].append({"step_index": detail.get("step_index"), "errors": detail.get("errors"),
            "rejected_response_sha256": hashlib.sha256(raw.encode("utf-8", errors="surrogatepass")).hexdigest() if isinstance(raw, str) else None,
            "safely_rejected": not unsafe and segment is not None, "feedback_returned_to_model": bool(feedback and returned),
            "recovered": recovered, "recovered_at_step": later[0]["response"].content.get("step_index") if recovered else None,
            "blocking_reasons": sorted(set(reasons))})
        metrics["safe_recovered_count"] += int(recovered)
        metrics["unsafe_rejected_step_count"] += int(bool(unsafe))
    metrics["unrecovered_count"] = len(errors) - metrics["safe_recovered_count"]
    metrics["quality_status"] = "passed" if not metrics["unrecovered_count"] and not metrics["unresolved_tool_error_count"] else "failed"
    metrics["recovery_classification"] = "safely_recovered" if metrics["quality_status"] == "passed" else "unrecovered"
    return metrics


def summarize_action_contract_recovery(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """逐例加总恢复质量，不把缺少指标的报告当作零错误通过。"""
    fields = ("action_contract_error_count", "safe_recovered_count", "unrecovered_count", "unsafe_rejected_step_count", "unresolved_tool_error_count")
    metrics = [report.get("recovery_metrics") for report in reports]
    valid = all(isinstance(item, dict) and all(type(item.get(field)) is int and item[field] >= 0 for field in fields) for item in metrics)
    return {"case_count": len(reports), "cases_with_action_contract_errors": sum(isinstance(item, dict) and item.get("action_contract_error_count", 0) > 0 for item in metrics),
        "cases_with_safe_recovery": sum(isinstance(item, dict) and item.get("safe_recovered_count", 0) > 0 for item in metrics),
        **{field: sum(item[field] for item in metrics) if valid else None for field in fields},
        "quality_status": "not_run" if not reports else ("passed" if valid and all(item.get("quality_status") == "passed" for item in metrics) else "failed")}


def verified_constraint_memory_write(result: Any, scenario: LiveEvalScenario, trace_events: list[Any], store: SQLiteAgentStore) -> dict[str, Any] | None:
    """只为未占座、无待处理邀约的限制反馈认可同客户同任务的真实记忆副作用。

    不用工具名、旧画像或模型自述代替证据；本次网关完成、工具返回和Store记录须一致。
    参加状态或邀约尚需处置时，记忆不能代替领域状态写入，原有工具路径仍保留。
    """
    expected=scenario.equivalent_state_tracking_memory
    if not expected:return None
    message=scenario.message
    task=store.current_task_context(message.conversation_id,message.sender_id)
    if task is None:return None
    from mahjong_agent_runtime.models import OPEN_INVITE_STATUSES
    related={g.game_id for g in store.games.values() if g.conversation_id==message.conversation_id}
    if any(p.customer_id==message.sender_id and p.status in {'joined','confirmed'}
           for g in store.games.values() if g.game_id in related for p in g.participants):return None
    if any(d.game_id in related and d.customer_id==message.sender_id and d.status in OPEN_INVITE_STATUSES
           for d in store.invite_drafts.values()):return None
    writes=[t for t in result.tool_results if t.name=='record_user_memory' and t.called and t.allowed and not t.error]
    completed=[e.content for e in trace_events if e.step=='tool_gateway_completed' and e.trace_id==result.trace_id
               and e.content.get('tool_name')=='record_user_memory' and e.content.get('called') is True
               and e.content.get('allowed') is True and e.content.get('outcome')=='executed' and not e.content.get('error')]
    for memory in store.task_memory_context(message.conversation_id,message.sender_id):
        if (memory.get('source_trace_id')!=result.trace_id or memory.get('conversation_id')!=message.conversation_id
                or memory.get('customer_id')!=message.sender_id or memory.get('status')!='active'
                or memory.get('scope')!='current_task' or memory.get('metadata',{}).get('task_context_id')!=task.task_context_id
                or not all(eval_values_equal(memory.get(k),v,field_path=expected.get('field','') if k=='value' else k)
                           for k,v in expected.items())):continue
        for write in writes:
            if (memory in write.result.get('task_memories',[])
                    and isinstance(write.idempotency_key,str) and bool(write.idempotency_key)
                    and any(c.get('idempotency_key')==write.idempotency_key for c in completed)
                    and any(e.step=='tool_result' and e.trace_id==result.trace_id
                            and json.dumps(e.content,ensure_ascii=False,sort_keys=True)==json.dumps(write.to_dict(),ensure_ascii=False,sort_keys=True)
                            for e in trace_events)):
                return memory
    return None


def validate_result(
    result: Any,
    scenario: LiveEvalScenario,
    trace_events: list[Any],
    store: SQLiteAgentStore,
    semantic_assessment: dict[str, Any] | None = None,
    money_fact_check: dict[str, Any] | None = None,
    query_reply_state_assessment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """联合检查执行、条件性查询和独立语义，缺失必查或语义证据均不能通过。

    可选查询字段仅约束实际执行结果；允许澄清的场景可另行声明未查工具时的合法
    终态。这里不按话术判定意图，真正缺项是否有效仍由既有人工语义合同验收。
    运行器传入的独立金额检查还须匹配真实工具与原文，不能覆盖原有失败或冒充完整覆盖。
    """
    final_reply = str(result.final_reply or "")
    trace_steps = [item.step for item in trace_events]
    model_context_text = customer_supplied_model_context_text(trace_events)
    customer_visible_violations = customer_visible_text_contract_violations(final_reply)
    tool_names = [item.name for item in result.tool_results if item.name not in {"customer_visible_text_generation", "customer_visible_content_review"}]
    checks: list[dict[str, Any]] = []
    checks.append(
        {
            "name": "final_reply_should_pass_customer_visible_contract",
            "passed": not customer_visible_violations,
            "violations": customer_visible_violations,
            "actual": final_reply,
        }
    )
    checks.append({
        "name": "reply_should_match_expected_meaning",
        "passed": bool(semantic_assessment) and semantic_assessment.get("passed") is True and not semantic_assessment.get("error"),
        "actual": semantic_assessment or {"passed": False, "error": "semantic_assessment_not_run"},
    })
    if scenario.requires_query_reply_state_check:
        # 告知和征询都可合法，但必须盲读真实回复后硬比状态；一般语义通过不能替代此证据。
        state_identity = f"{getattr(result, 'trace_id', '')}_query_reply_state"
        checks.append({"name": "query_reply_state_assessment_should_match_original_evidence",
            "passed": validate_reply_state_assessment(query_reply_state_assessment,
                evidence=reply_semantic_evidence(result, scenario, trace_events, store), trace_id=state_identity),
            "actual": query_reply_state_assessment})
        checks.append({"name": "query_reply_and_runtime_status_should_be_consistent",
            "passed": communication_state_is_consistent(result.status, query_reply_state_assessment),
            "actual": {"runtime_status": result.status, "assessment": query_reply_state_assessment}})
    if money_fact_check is not None:
        # 金额作用域只分类原文；事实比较重读真实工具结果，不以一般语义判定替代金额证据。
        checks.append({"name": "reply_money_fact_check_should_match_original_evidence",
            "passed": validate_money_fact_check(money_fact_check, reply=final_reply,
                tool_results=result.tool_results, trace_id=f"{result.trace_id}_money_fact_check"),
            "actual": money_fact_check})
        checks.append({"name": "reply_money_facts_should_not_conflict",
            "passed": money_fact_check.get("blocks") is False,
            "actual": {key: money_fact_check.get(key) for key in ("status", "coverage", "blocks")}})
    if scenario.max_reply_chars is not None:
        checks.append(
            {
                "name": "final_reply_should_be_short_owner_wechat_style",
                "passed": len(final_reply.strip()) <= scenario.max_reply_chars,
                "max_reply_chars": scenario.max_reply_chars,
                "actual_length": len(final_reply.strip()),
                "actual": final_reply,
            }
        )
    checks.append(
        {
            "name": "final_reply_should_be_single_message",
            "passed": "\n" not in final_reply.strip(),
            "actual": final_reply,
        }
    )
    for tool_name in scenario.required_tool_names:
        checks.append(
            {
                "name": f"should_call_{tool_name}",
                "passed": tool_name in tool_names,
                "actual": tool_names,
            }
        )
    if scenario.required_tool_name_any:
        memory_evidence=verified_constraint_memory_write(result,scenario,trace_events,store)
        checks.append(
            {
                "name": "should_call_any_state_tracking_tool",
                "passed": any(tool_name in tool_names for tool_name in scenario.required_tool_name_any) or memory_evidence is not None,
                "expected_any": scenario.required_tool_name_any,
                "actual": tool_names,
                "equivalent_current_task_memory": memory_evidence,
            }
        )
    for tool_name in scenario.forbidden_tool_names:
        checks.append(
            {
                "name": f"should_not_call_{tool_name}",
                "passed": tool_name not in tool_names,
                "forbidden": tool_name,
                "actual": tool_names,
            }
        )
    allowed_statuses = scenario.expected_status_any or ["completed", "waiting_user"]
    checks.append({
        "name": "runtime_status_should_be_expected",
        "passed": getattr(result, "status", None) in allowed_statuses,
        "expected_any": allowed_statuses, "actual": getattr(result, "status", None),
    })
    for tool_name, maximum in scenario.max_tool_call_counts.items():
        actual_count = tool_names.count(tool_name)
        checks.append(
            {
                "name": f"{tool_name}_call_count_should_not_exceed_{maximum}",
                "passed": actual_count <= maximum,
                "maximum": maximum,
                "actual": actual_count,
            }
        )
    if scenario.excluded_search_filter_fields or scenario.excluded_query_requirement_values:
        search_result = next((
            item for item in reversed(result.tool_results)
            if item.name == "search_current_games" and item.called and item.allowed and not item.error
        ), None)
        applied_fields = value_at_path(
            search_result.to_dict() if search_result else None,
            "result.customer_reply_contract.evidence_scope.applied_filter_fields",
        )
        if scenario.excluded_search_filter_fields:
            checks.append({
                "name": "relaxed_constraints_should_not_remain_in_applied_search_filters",
                "passed": isinstance(applied_fields, list) and not set(scenario.excluded_search_filter_fields) & set(applied_fields),
                "excluded_fields": scenario.excluded_search_filter_fields,
                "actual": applied_fields,
            })
        requirement = (search_result.result.get("requirement") if search_result else None)
        for path, excluded_values in scenario.excluded_query_requirement_values.items():
            actual = value_at_path(requirement, path)
            # 只比较场景声明的旧值；省略或不限表达无需统一成某个固定字符串。
            normalized_actual = actual.strip().casefold() if isinstance(actual, str) else actual
            stale_value = any(
                normalized_actual == (value.strip().casefold() if isinstance(value, str) else value)
                for value in excluded_values
            )
            checks.append({
                "name": f"relaxed_query_should_not_keep_excluded_value_{path}",
                "passed": isinstance(requirement, dict) and not stale_value,
                "excluded_values": excluded_values,
                "actual": actual,
            })
    for tool_name, expected_paths in scenario.expected_tool_result_paths.items():
        matching_tool_result = next(
            (
                item
                for item in reversed(result.tool_results)
                if item.name == tool_name and item.called and item.allowed and not item.error
            ),
            None,
        )
        if isinstance(expected_paths, list):
            # OR是评测协议：每个分支必须完整匹配同一个最近成功结果，不能跨调用拼证。
            if not expected_paths or any(not isinstance(paths, dict) or not paths for paths in expected_paths):
                raise ValueError("tool result alternatives require a non-empty list of non-empty path objects")
            alternatives = []
            for paths in expected_paths:
                actual = {path: value_at_path(matching_tool_result.to_dict() if matching_tool_result else None, path)
                          for path in paths}
                alternatives.append({"expected": paths, "actual": actual,
                    "passed": matching_tool_result is not None and all(eval_values_equal(actual[path], expected, field_path=path)
                                                                       for path, expected in paths.items())})
            checks.append({"name": f"{tool_name}_result_should_match_one_declared_alternative",
                           "passed": any(item["passed"] for item in alternatives), "alternatives": alternatives})
            continue
        if not isinstance(expected_paths, dict):
            raise ValueError("tool result paths require an object or non-empty list of objects")
        for path, expected in expected_paths.items():
            actual = value_at_path(matching_tool_result.to_dict() if matching_tool_result else None, path)
            checks.append(
                {
                    "name": f"{tool_name}_result_should_keep_{path}",
                    "passed": eval_values_equal(actual, expected, field_path=path),
                    "expected": expected,
                    "actual": actual,
                }
            )
    for tool_name, expected_paths in scenario.expected_tool_result_unordered_rows.items():
        matching_result = next((item for item in reversed(result.tool_results)
                                if item.name == tool_name and item.called and item.allowed and not item.error), None)
        for path, expected in expected_paths.items():
            actual_items = value_at_path(matching_result.to_dict() if matching_result else None, path)
            fields = sorted({field for row in expected for field in row})
            actual = ([{field: value_at_path(item, field) for field in fields} for item in actual_items]
                      if isinstance(actual_items, list) else None)
            # 同一行保留玩法与档位的配对；多重集合不约束排序，也不让重复项掩盖遗漏。
            checks.append({
                "name": f"{tool_name}_result_should_have_unordered_rows_{path}",
                "passed": actual is not None and sorted(json.dumps(value, sort_keys=True, ensure_ascii=False) for value in actual)
                          == sorted(json.dumps(value, sort_keys=True, ensure_ascii=False) for value in expected),
                "expected": expected, "actual": actual,
            })
    # 可选查询允许模型先澄清真正缺项；一旦执行查询，每次结果仍须满足显式字段合同。
    # 必查 expected_tool_result_paths 保持原逻辑，不能因新增可选合同绕过缺失工具的失败。
    for tool_name, expected_paths in scenario.optional_query_result_fields.items():
        executed = [item for item in result.tool_results if item.name == tool_name and item.called]
        if not executed:
            continue
        for path, expected in expected_paths.items():
            actual_values = [value_at_path(item.to_dict(), path) for item in executed]
            checks.append({
                "name": f"optional_{tool_name}_result_should_keep_{path}",
                "passed": all(item.allowed and not item.error and eval_values_equal(actual, expected, field_path=path)
                              for item, actual in zip(executed, actual_values)),
                "expected": expected, "actual": actual_values,
            })
    for tool_name, statuses in scenario.expected_status_without_tool.items():
        if not any(item.name == tool_name and item.called for item in result.tool_results):
            checks.append({
                "name": f"without_{tool_name}_status_should_be_expected",
                "passed": result.status in statuses, "expected_any": statuses, "actual": result.status,
            })
    if scenario.expected_any_tool_result_paths:
        candidate_evidence: dict[str, dict[str, Any]] = {}
        any_candidate_passed = False
        for tool_name, expected_paths in scenario.expected_any_tool_result_paths.items():
            matching_tool_result = next(
                (
                    item
                    for item in reversed(result.tool_results)
                    if item.name == tool_name and item.called and item.allowed and not item.error
                ),
                None,
            )
            actual_paths = {
                path: value_at_path(matching_tool_result.to_dict() if matching_tool_result else None, path)
                for path in expected_paths
            }
            candidate_passed = matching_tool_result is not None and all(
                eval_values_equal(actual_paths[path], expected, field_path=path)
                for path, expected in expected_paths.items()
            )
            candidate_evidence[tool_name] = {
                "called": matching_tool_result is not None,
                "passed": candidate_passed,
                "expected": expected_paths,
                "actual": actual_paths,
            }
            any_candidate_passed = any_candidate_passed or candidate_passed
        checks.append(
            {
                "name": "equivalent_tool_result_should_preserve_required_state",
                "passed": any_candidate_passed,
                "candidates": candidate_evidence,
            }
        )
    for item in scenario.forbidden_reply_contains:
        checks.append(
            {
                "name": f"reply_should_not_contain_{item}",
                "passed": item not in final_reply,
                "forbidden": item,
                "actual": final_reply,
            }
        )
    for item in scenario.forbidden_model_context_contains:
        checks.append(
            {
                "name": f"model_context_should_not_contain_{item}",
                "passed": item not in model_context_text,
                "forbidden": item,
                "actual_preview": context_preview_around(model_context_text, item),
            }
        )
    for step in scenario.forbidden_trace_steps:
        if step == "action_contract_error":
            # 用户验收口径允许安全拒绝后纠正，仍以真实执行证据单独核验，不能删原trace。
            continue
        checks.append(
            {
                "name": f"trace_should_not_have_{step}",
                "passed": step not in trace_steps,
                "forbidden": step,
                "actual": trace_steps,
            }
        )
    if scenario.expected_active_game_count is not None:
        active_games = store.active_games(scenario.message.conversation_id)
        checks.append(
            {
                "name": "active_game_count_should_match",
                "passed": len(active_games) == scenario.expected_active_game_count,
                "expected": scenario.expected_active_game_count,
                "actual": len(active_games),
            }
        )
    if scenario.expected_active_game_status is not None:
        active_games = store.active_games(scenario.message.conversation_id)
        active_status = active_games[0].status if active_games else None
        checks.append(
            {
                "name": "active_game_status_should_match",
                "passed": active_status == scenario.expected_active_game_status,
                "expected": scenario.expected_active_game_status,
                "actual": active_status,
            }
        )
    if scenario.expected_active_game_seat_summary:
        active_games = store.active_games(scenario.message.conversation_id)
        seat_summary = active_games[0].seat_summary() if active_games else {}
        for key, expected in scenario.expected_active_game_seat_summary.items():
            checks.append(
                {
                    "name": f"active_game_seat_summary_should_keep_{key}",
                    "passed": seat_summary.get(key) == expected,
                    "expected": expected,
                    "actual": seat_summary.get(key),
                }
            )
    if scenario.expected_active_game_requirement:
        active_games = store.active_games(scenario.message.conversation_id)
        active_requirement = active_games[0].requirement if active_games else {}
        for key, expected in scenario.expected_active_game_requirement.items():
            checks.append(
                {
                    "name": f"active_game_requirement_should_keep_{key}",
                    "passed": eval_values_equal(eval_requirement_value(active_requirement, key), expected, field_path=key),
                    "expected": expected,
                    "actual": active_requirement.get(key),
                }
            )
    if scenario.expected_task_memory_contains:
        task_memories = store.task_memory_context(
            scenario.message.conversation_id,
            scenario.message.sender_id,
        )
        current_memory_task = (store.current_task_context(scenario.message.conversation_id, scenario.message.sender_id)
                               if scenario.require_task_memory_current_context else None)
        for index, expected in enumerate(scenario.expected_task_memory_contains, start=1):
            matching_memories = [memory for memory in task_memories
                                 if all(memory.get(key) == value for key, value in expected.items())]
            current_task_id = current_memory_task.task_context_id if current_memory_task is not None else None
            if scenario.require_task_memory_current_context:
                # 验证存在一条完整可信记录，而非依赖返回顺序拼接不同记录或先取不合格首项。
                matching_memories = [memory for memory in matching_memories if current_task_id
                    and isinstance(memory.get("metadata"), dict)
                    and memory["metadata"].get("task_context_id") == current_task_id
                    and memory.get("conversation_id") == scenario.message.conversation_id
                    and memory.get("customer_id") == scenario.message.sender_id
                    and memory.get("status") == "active" and isinstance(memory.get("scope"), str)
                    and memory["scope"] in memory_item_schema["properties"]["scope"]["enum"]]
            matched_memory = next(iter(matching_memories), None)
            checks.append(
                {
                    "name": f"task_memory_should_match_{index}",
                    "passed": matched_memory is not None,
                    "expected": expected,
                    "actual": matched_memory if matched_memory is not None else task_memories,
                }
            )
            if scenario.require_task_memory_current_context:
                # 候选必须来自本次真实 Store 读回；不能靠工具自报成功或任意 scope 标签放行。
                checks.append({"name": f"task_memory_current_context_readback_{index}", "passed": matched_memory is not None,
                    "expected": {"task_context_id": current_task_id, "read_source": "store.task_memory_context",
                                 "same_customer_and_conversation": True, "status": "active", "schema_legal_scope": True},
                    "actual": matched_memory})
    elif scenario.require_task_memory_current_context:
        checks.append({"name": "task_memory_current_context_readback_required", "passed": False,
                       "expected": "at least one declared task-memory fact", "actual": []})
    if scenario.expected_checkpoint_contains:
        checkpoint = store.get_conversation_checkpoint(scenario.message.conversation_id)
        checkpoint_text = json.dumps(checkpoint.to_dict() if checkpoint else {}, ensure_ascii=False, sort_keys=True)
        for expected in scenario.expected_checkpoint_contains:
            checks.append(
                {
                    "name": f"checkpoint_should_contain_{expected}",
                    "passed": expected in checkpoint_text,
                    "expected": expected,
                    "actual": checkpoint_text,
                }
            )
    recovery = assess_action_contract_recovery(trace_events, runtime_status=result.status, final_reply=final_reply,
        business_checks_passed=all(item["passed"] for item in checks),
        semantic_passed=bool(semantic_assessment) and semantic_assessment.get("passed") is True and not semantic_assessment.get("error"))
    checks.append({"name": "action_contract_errors_should_be_safely_recovered", "passed": recovery["quality_status"] == "passed", "actual": recovery})
    return {
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "final_reply": final_reply,
        "tool_names": tool_names,
        "recovery_metrics": recovery,
    }


def customer_supplied_model_context_text(trace_events: list[Any]) -> str:
    payloads: list[Any] = []
    for event in trace_events:
        if event.step == "context_built":
            payloads.append(event.content)
            continue
        if event.step == "llm_prompt":
            user_messages: list[dict[str, Any]] = []
            for message in event.content.get("messages") or []:
                if isinstance(message, dict) and message.get("role") != "system":
                    user_messages.append(message)
            payloads.append({"messages": user_messages, "step_index": event.content.get("step_index")})
    return json.dumps(payloads, ensure_ascii=False, sort_keys=True)


def context_preview_around(text: str, needle: str, window: int = 120) -> str:
    if not needle:
        return ""
    index = text.find(needle)
    if index < 0:
        return ""
    start = max(index - window, 0)
    end = min(index + len(needle) + window, len(text))
    return text[start:end]


def eval_requirement_value(requirement: dict[str, Any], field: str) -> Any:
    """比较现有合同允许的别名及四人桌计数等价表示，不解释任何自然语言。

    一缺几人和已有几人受四人桌不变量约束。不得为严格字段快照要求模型重复
    提交等价的派生字段；冲突显式值、未知值和错误子玩法仍然不能通过。
    """
    if field == "requesting_party":
        # 只等价读取协议中显式的己方座位，绝不从目标桌况或缺省一人补出请求方人数。
        party = requirement.get("requesting_party")
        sources = [party, requirement] if isinstance(party, dict) else [requirement]
        counts = [source[key] for source in sources for key in ("seat_count", "party_size")
                  if key in source and source[key] is not None]
        if not counts or any(type(value) is not int or not 1 <= value <= 4 for value in counts) or len(set(counts)) != 1:
            return None
        return {**(party if isinstance(party, dict) else {}), "seat_count": counts[0]}
    if field in {"variant", "game_variant"} and all(key in requirement for key in ("variant", "game_variant")):
        if requirement["variant"] != requirement["game_variant"]:
            return None
    if field in requirement:
        return requirement[field]
    if field in {"variant", "game_variant"}:
        return requirement.get("variant" if field == "game_variant" else "game_variant")
    if "seats_total" in requirement and (type(requirement["seats_total"]) is not int or requirement["seats_total"] != 4):
        return None
    known, needed = requirement.get("known_player_count"), requirement.get("needed_seats")
    if field == "known_player_count" and type(needed) is int and 1 <= needed <= 3:
        return 4 - needed
    if field == "needed_seats" and type(known) is int and 1 <= known <= 3:
        return 4 - known
    if field == "seat_format":
        if known is None and type(needed) is int and 1 <= needed <= 3:
            known = 4 - needed
        if needed is None and type(known) is int and 1 <= known <= 3:
            needed = 4 - known
        if type(known) is int and type(needed) is int and 1 <= known <= 3 and known + needed == 4:
            return f"{known}7{needed}"
    return None


def value_at_path(payload: Any, dotted_path: str) -> Any:
    """读取结果路径；需求字段按已声明领域合同比较等价表示。"""
    current = payload
    parts = dotted_path.split(".")
    for index, raw_part in enumerate(parts):
        if isinstance(current, list):
            try:
                current = current[int(raw_part)]
            except (ValueError, IndexError):
                return None
            continue
        if isinstance(current, dict):
            current = eval_requirement_value(current, raw_part) if index and parts[index - 1] in {"requirement", "public_requirement"} else current.get(raw_part)
            continue
        return None
    return current


def eval_values_equal(actual: Any, expected: Any, *, field_path: str = "") -> bool:
    """仅在已知字段合同内比较等价表示；烟况相反、未知或缺失仍然失败。"""

    # 布尔证明不能由Python的1==True或0==False冒充，其他既有等价规则保持。
    if isinstance(expected, bool):
        return actual is expected
    if actual == expected:
        return True
    if isinstance(actual, str) and isinstance(expected, str):
        if field_path.rsplit(".", 1)[-1] == "smoke_preference" and actual.strip() and expected.strip():
            normalized = normalize_smoke_preference(actual)
            return normalized in {"smoking", "no_smoking", "any"} and normalized == normalize_smoke_preference(expected)
        if _looks_like_clock(expected):
            try:
                return dt.datetime.fromisoformat(actual).strftime("%H:%M") == expected
            except ValueError:
                return False
    return False


def _looks_like_clock(value: str) -> bool:
    try:
        parsed = dt.datetime.strptime(value, "%H:%M")
    except ValueError:
        return False
    return parsed.strftime("%H:%M") == value


def golden_reply_cases(path: pathlib.Path = GOLDEN_REPLY_PATH) -> list[dict[str, Any]]:
    """读取既有黄金期望及显式场景关联，未关联资产单独报告而非记作通过。"""
    cases: list[dict[str, Any]] = []
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    catalog = {scene.scenario_id: {"required_reply_any": scene.required_reply_any,
               "forbidden_reply_contains": scene.forbidden_reply_contains,
               "expected_tool_result_paths": scene.expected_tool_result_paths} for scene in live_eval_scenarios()}
    effective = resolve_product_contract_revisions(records, scenario_contracts=catalog)
    for record in records:
        for case in record.get("eval_cases", []):
            linked_case = {
                "record_id": record["id"], "case_id": case["id"],
                "scenario_id": case.get("live_scenario_id") or LEGACY_GOLDEN_SCENARIOS.get(case["id"]),
                "description": case.get("description") or case.get("name") or case["id"],
                "expected": dict(effective["owner"].get(case["id"]) or {}),
                "product_contract_revision_ids": effective["revision_ids"].get("owner:" + case["id"], []),
            }
            for field_name, later_scenario in LEGACY_GOLDEN_LATER_FIELDS.get(case["id"], {}).items():
                if field_name in linked_case["expected"]:
                    cases.append({**linked_case, "scenario_id": later_scenario,
                                  "expected": {field_name: linked_case["expected"].pop(field_name)},
                                  "coverage_note": "四小时上限来自原始父对话第56轮，不能提前泄露给第14轮拒五小时的输入。"})
            cases.append(linked_case)
    return apply_query_option_policy_to_cases(apply_reply_presentation_to_cases(cases, records), records)


def reply_semantic_expectation(scenario: LiveEvalScenario) -> dict[str, Any]:
    """仅从明确关联案例取得业务期望，缺来源直接失败，不让通用规则猜测场景目标。"""
    linked_cases = [case for case in golden_reply_cases() if case["scenario_id"] == scenario.scenario_id]
    if not linked_cases or any(not case["expected"] for case in linked_cases):
        raise ValueError(f"live scenario lacks explicit golden semantic expectation: {scenario.scenario_id}")
    required: list[dict[str, str]] = []
    forbidden = common_forbidden_semantic_meanings()
    # 保留旧正向验收的业务信息强度，但把措辞变体交给独立判定器。
    for index, alternatives in enumerate(scenario.required_reply_any):
        required.append({"id": f"scenario_reply_alternatives_{index}", "description": f"回复应表达这一组允许表达中至少一种的业务含义，允许同义措辞：{json.dumps(alternatives, ensure_ascii=False)}。"})
    for index, item in enumerate(scenario.required_reply_contains):
        required.append({"id": f"scenario_reply_fact_{index}", "description": f"回复应保留业务事实或应答含义：{item}；允许同义表达，不得换成其他时间、金额或人数。"})
    sources: list[dict[str, str]] = []
    references: list[str] = []
    for case in linked_cases:
        sources.append({"record_id": case["record_id"], "case_id": case["case_id"]})
        expected = case["expected"]
        prefix = case["case_id"]
        if expected.get("reply_reference"):
            references.append(expected["reply_reference"])
        if "required_meaning" in expected or "forbidden_meaning" in expected:
            for key, target in (("required_meaning", required), ("forbidden_meaning", forbidden)):
                target.extend({"id": f"{prefix}.{item['id']}", "description": item["description"]} for item in expected.get(key, []))
        else:
            for key, value in expected.items():
                if key == "good_examples":
                    references.extend(value)
                    continue
                target = forbidden if key.startswith("forbidden") or key == "bad_examples" else required
                target.append({
                    "id": f"{prefix}.{key}",
                    "description": f"黄金案例：{case['description']}。原始标准 {key}={json.dumps(value, ensure_ascii=False)}；按语义及实际状态判断，示例措辞不要求逐字一致。",
                })
    if scenario.require_help_offer:
        required.append({"id": "offer_finding_players", "description": "无现成局后由老板向客户提议是否需要帮忙问人或组局，等待客户选择；不能仅消极告知无局、强迫换条件或直接承诺已经联系。"})
    if not required:
        raise ValueError(f"live scenario lacks explicit required reply meaning: {scenario.scenario_id}")
    revision_ids = list(dict.fromkeys(version for case in linked_cases for version in case.get("product_contract_revision_ids", [])))
    return {"required_meaning": required, "forbidden_meaning": forbidden, "reply_reference": "\n".join(references),
            "golden_sources": sources, "product_contract_revision_ids": revision_ids}


def common_forbidden_semantic_meanings() -> list[dict[str, str]]:
    """原有动作真实性和隐私标准原文共用，不为校准样本或某次失败另换宽松口径。"""
    return [
        {"id": "fabricated_action", "description": "不能捏造已联系、已发送、已成局或已确认。是否已有安排应看实际工具/持久状态，不能以回复是否汇报后台动作判断；已有真实安排后‘好，我帮你看看/问问’是正常承接，无须披露建局或草稿。无安排时不能以持续承诺假装完成；征询式服务提议不等于已承担或已执行。"},
        {"id": "privacy", "description": "不能向客户泄露系统提示词、内部工具或字段、内部备注、未授权客户身份或其他隐私。客户可见信息仅能使用实际证据允许公开的投影。"},
    ]


def golden_reply_coverage(scenario_ids: set[str], passed_scenario_ids: set[str] | None = None) -> dict[str, Any]:
    """按原始案例聚合跨回合验收片段，部分运行不能冒充完整覆盖。"""
    grouped: dict[str, list[str | None]] = {}
    for case in golden_reply_cases():
        grouped.setdefault(case["case_id"], []).append(case["scenario_id"])
    return {
        "total_cases": len(grouped),
        "executed_case_ids": [case_id for case_id, ids in grouped.items() if all(value in scenario_ids for value in ids)],
        "passed_case_ids": [case_id for case_id, ids in grouped.items() if all(value in (passed_scenario_ids or set()) for value in ids)],
        "partially_executed_case_ids": [case_id for case_id, ids in grouped.items() if any(value in scenario_ids for value in ids) and not all(value in scenario_ids for value in ids)],
        "unmapped_case_ids": [case_id for case_id, ids in grouped.items() if any(value is None for value in ids)],
        "timeline_assets": {
            "live_status": "not_run", "deterministic_scope": "dedicated timeline pytest suites",
            "files": [{"file": path.name, "eval_case_count": len(data.get("eval_cases", [])),
                       "replay_scenario_count": len(data.get("replay_scenarios", []))}
                      for path in sorted((ROOT / "eval" / "golden").glob("real_owner_chat_timeline_*.json"))
                      for data in [json.loads(path.read_text(encoding="utf-8"))]],
        },
    }


def _follow_up_semantic_state(store: SQLiteAgentStore, message: UserMessage) -> dict[str, Any]:
    """只读当前客户、会话和目标的复查末态，不把历史工具成功当作仍有效的安排。

    状态、期限和次数原样提供，包括已取消、已完成或时限已过的记录；判定器结合
    观测时间与实际回复判断承诺边界。未找到当前目标时不猜归属，也不复制其他任务载荷。
    """
    observed_at = now().isoformat()
    context = store.current_task_context(message.conversation_id, message.sender_id)
    context_id = context.task_context_id if context else None
    tasks: list[dict[str, Any]] = []
    if context_id:
        for task in sorted(store.scheduled_tasks.values(), key=lambda item: (item.created_at, item.task_id)):
            if (task.task_type != "conversation_follow_up" or task.aggregate_type != "conversation_goal"
                    or task.conversation_id != message.conversation_id or task.subject_id != message.sender_id
                    or task.aggregate_id != context_id or task.payload.get("task_context_id") != context_id):
                continue
            tasks.append({
                "task_id": task.task_id, "task_type": task.task_type, "status": task.status.value,
                "conversation_id": task.conversation_id, "subject_id": task.subject_id,
                "aggregate_type": task.aggregate_type, "aggregate_id": task.aggregate_id,
                "due_at": task.due_at.isoformat(),
                **{key: task.payload.get(key) for key in (
                    "task_context_id", "goal", "waiting_for", "expires_at", "review_budget", "review_count",
                )},
            })
    return {
        "observed_at": observed_at, "conversation_id": message.conversation_id,
        "customer_id": message.sender_id, "current_task_context_id": context_id, "tasks": tasks,
    }


def reply_semantic_evidence(result: Any, scenario: LiveEvalScenario, trace_events: list[Any], store: SQLiteAgentStore) -> dict[str, Any]:
    """投影回合前上下文与回合后真实状态，不用旧画像掩盖末态，也不输入模型计划。"""
    initial_context = next((event.content for event in trace_events if event.step == "context_built"), {})
    business_results = [item for item in result.tool_results if item.name not in {"customer_visible_text_generation", "customer_visible_content_review"}]
    sender_profile = store.customers.get(scenario.message.sender_id)
    return {
        "request": scenario.message.to_dict(), "reply": result.final_reply,
        "recent_conversation": [turn for turn in initial_context.get("recent_conversation", []) if turn.get("role") in {"user", "assistant"}],
        "known_profile": initial_context.get("sender_profile"),
        "tool_results": [{key: value for key, value in item.items() if key != "checkpoint"}
                         for item in summarize_tool_results(business_results)],
        "persisted_state": {
            "sender_profile": sender_profile.to_model_context() if sender_profile else None,
            "related_active_games": [game.to_dict() for game in store.active_games(scenario.message.conversation_id)],
            "games": [game.to_dict() for game in store.games.values()],
            "sender_task_memories": store.task_memory_context(scenario.message.conversation_id, scenario.message.sender_id),
            "invite_statuses": [draft.status.value for draft in store.invite_drafts.values()],
            "follow_up_state": _follow_up_semantic_state(store, scenario.message),
        },
        "runtime_status": result.status,
        "execution_lifecycle": {
            "turn": {"status": result.status, "scope": "synchronous_turn_only"},
            "activities": [{"game_id": game.game_id, "status": game.status.value,
                "recruitment_status": game.recruitment_status.value,
                "board_arrangement": project_board_arrangement(store.game_board_arrangement(game.game_id))}
                for game in store.active_games(scenario.message.conversation_id)],
        },
    }


def project_board_arrangement(arrangement: dict[str, Any]) -> dict[str, Any]:
    """只投影后端的安全标量，不把群身份、任务载荷或模型自述当作发布证据。"""
    return {key: arrangement[key] for key in (
        "status", "linked_room_count", "pending_task_count", "outbound_authorized_room_count", "publication_sent",
    ) if key in arrangement}


def summarize_tool_results(tool_results: list[Any]) -> list[dict[str, Any]]:
    """保留实际工具执行及查询范围证据，仅投影判定需要的合同与业务摘要。"""
    summaries: list[dict[str, Any]] = []
    for item in tool_results:
        result = dict(item.result or {})
        summary: dict[str, Any] = {
            "name": item.name,
            "called": item.called,
            "allowed": item.allowed,
            "error": item.error,
            "deduplicated": item.deduplicated,
            "idempotency_key": item.idempotency_key,
        }
        continuation = result.get("continuation")
        if isinstance(continuation, dict):
            summary["continuation"] = {key: continuation[key] for key in (
                "can_stop", "allowed_terminal_statuses", "pending_capabilities",
            ) if key in continuation}
        if isinstance(result.get("board_arrangement"), dict):
            summary["board_arrangement"] = project_board_arrangement(result["board_arrangement"])
        # 工具名只证明调用类型，房态结论还需要真实配置、时间区间和库存数量。
        # 只读查询与实际预定分别投影，避免丢证据误拒，也不把有空房升格为已占房。
        if item.name == "check_room_availability":
            availability = {key: result[key] for key in (
                "configured", "start_at", "end_at", "room_count", "available_count") if key in result}
            if availability:
                summary["room_availability"] = availability
        if item.name == "reserve_room" and isinstance(result.get("reservation"), dict):
            summary["reservation"] = {key: result["reservation"][key] for key in (
                "reservation_id", "room_id", "start_at", "end_at", "status") if key in result["reservation"]}
        reply_contract = result.get("customer_reply_contract")
        if isinstance(reply_contract, dict):
            # 判定器需分清全店盘点与定向检索，仅透传这两类权威范围证据。
            summary["customer_reply_contract"] = {
                key: reply_contract[key] for key in ("evidence_scope", "search_result_semantics")
                if key in reply_contract
            }
        if "requirement" in result:
            summary["requirement"] = result["requirement"]
        if "matches" in result:
            matches = list(result.get("matches") or [])
            summary["match_count"] = len(matches)
            summary["matched_result_summaries"] = (
                result.get("customer_reply_contract", {}).get("matched_result_summaries", [])
                if isinstance(result.get("customer_reply_contract"), dict)
                else []
            )
            summary["join_projections"] = [
                match.get("join_projection")
                for match in matches[:3]
                if isinstance(match, dict) and match.get("join_projection")
            ]
        if "candidates" in result:
            candidates = list(result.get("candidates") or [])
            summary["candidate_count"] = len(candidates)
            summary["candidate_ids"] = [
                candidate.get("customer", {}).get("customer_id")
                for candidate in candidates[:5]
                if isinstance(candidate, dict)
            ]
        if "game" in result and isinstance(result["game"], dict):
            game = result["game"]
            summary["game"] = {
                "game_id": game.get("game_id"),
                "status": game.get("status"),
                "requirement": game.get("requirement"),
                "seat_summary": game.get("seat_summary"),
            }
        if "checkpoint" in result and isinstance(result["checkpoint"], dict):
            checkpoint = result["checkpoint"]
            summary["checkpoint"] = {
                "summary": checkpoint.get("summary"),
                "facts": checkpoint.get("facts"),
                "open_questions": checkpoint.get("open_questions"),
            }
        summaries.append(summary)
    return summaries


def scenario_db_path(base_path: pathlib.Path, scenario_id: str) -> pathlib.Path:
    return base_path.with_name(f"{base_path.stem}_{scenario_id}{base_path.suffix}")


def decision_trace_snapshots(trace_events: list[Any]) -> list[dict[str, Any]]:
    """在评测报告中保留模型/工具决策边界，而不复制完整提示。"""

    high_signal_steps = {
        "llm_response",
        "action_contract_error",
        "contract_error_feedback",
        "objective_plan_proposed",
        "tool_called",
        "tool_result",
        "state_transition",
        "state_transition_replayed",
        "llm_error",
        "customer_visible_text_generation_error",
        "customer_visible_content_review_error",
        "action_proposed",
        "customer_visible_text_generation_result",
        "action_after_customer_visible_text_generation",
        "customer_visible_content_review_result",
        "final_output",
    }
    return [
        {
            "step": event.step,
            "level": event.level,
            "content": event.content,
        }
        for event in trace_events
        if event.step in high_signal_steps
    ]


def scenario_after_seeded_history(
    scenario: LiveEvalScenario, store: SQLiteAgentStore, *, received_at: dt.datetime | None = None,
) -> LiveEvalScenario:
    """克隆本次回放输入并置于种子历史之后，不改共享案例、原文、ID或既有期望。

    场景可在全量执行前很早构造；setup里的now历史却稍后才写入。合成接收时刻必须
    同时晚于这些历史和实际接收时间，显式未来锚仍晚于两者时原样保留。嵌套元数据
    与引用一起复制，防止同一场景的并发轮次互相写入Runtime接收凭证。
    """
    original = scenario.message
    history = store.recent_turns(original.conversation_id, limit=1000)
    latest = max((turn.occurred_at for turn in history), default=original.sent_at-dt.timedelta(microseconds=1))
    stamp = max(original.sent_at, received_at or dt.datetime.now().astimezone(), latest+dt.timedelta(microseconds=1))
    message = replace(original, sent_at=stamp, metadata=deepcopy(original.metadata), quoted_message=deepcopy(original.quoted_message))
    return replace(scenario, message=message)


def run_scenario(
    client: OpenAICompatibleAgentClient,
    args: argparse.Namespace,
    scenario: LiveEvalScenario,
    *,
    iteration: int = 1,
) -> dict[str, Any]:
    """按独立数据库与因果有序的合成输入回放，保留原始执行与独立语义证据。"""
    # 先验证人工期望来源，不能执行完主模型才发现该场景没有验收标准。
    expectation = reply_semantic_expectation(scenario)
    db_path = scenario_db_path(args.db_path, scenario.scenario_id)
    # 重复回放保留每轮运行库；上一轮快照和任务是证据，不能被下一轮重建覆盖。
    if iteration > 1:
        db_path = db_path.with_name(f"{db_path.stem}_iteration_{iteration:02d}{db_path.suffix}")
    store = build_empty_store(db_path)
    scenario.setup(store)
    if scenario.require_board_recovery_delivery:
        configure_authorized_eval_board(store)
    scenario = scenario_after_seeded_history(scenario, store)
    trace = InMemoryTraceRecorder()
    runtime = build_runtime(client, store, trace, args)
    trace_id = f"trace_owner_real_live_eval_{scenario.scenario_id}_{iteration:02d}"
    result = runtime.handle_user_message(scenario.message, trace_id=trace_id)
    trace_events = trace.get_trace(trace_id)
    # 复杂语义判定有独立传输预算；主流程及两个盲分类器继续使用各自原来的时限。
    assessment = assess_reply_semantics(client,
        evidence=reply_semantic_evidence(result, scenario, trace_events, store), expectation=expectation,
        trace_id=f"{trace_id}_semantic_judge", judge_timeout_seconds=getattr(args, "judge_timeout_seconds", None))
    query_reply_state_assessment = None
    if scenario.requires_query_reply_state_check:
        query_reply_state_assessment = assess_reply_communication(client,
            evidence=reply_semantic_evidence(result, scenario, trace_events, store),
            trace_id=f"{trace_id}_query_reply_state", timeout_seconds=args.timeout_seconds)
        trace.record(trace_id, "query_reply_state_assessment", deepcopy(query_reply_state_assessment))
    # 一般判定保留原始输入和响应；独立金额核验另存事件，不回写或修正一般判定结论。
    money_fact_check = check_reply_money_facts(client, reply=str(result.final_reply or ""),
        tool_results=result.tool_results, trace_id=f"{trace_id}_money_fact_check", timeout_seconds=args.timeout_seconds)
    trace.record(trace_id, "reply_money_fact_check", deepcopy(money_fact_check))
    trace_events = trace.get_trace(trace_id)
    trace_steps = [item.step for item in trace_events]
    trace_dir = db_path.parent / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    # 独立新文件保存完整原始事件，复跑不覆盖既往证据；客户回复判定仍只接最小投影。
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=f"{scenario.scenario_id}_{iteration:02d}_",
                                     suffix=".trace.json", dir=trace_dir, delete=False) as output:
        json.dump({"trace_id": result.trace_id, "scenario_id": scenario.scenario_id, "iteration": iteration,
                   "events": [event.to_dict() for event in trace_events]}, output, ensure_ascii=False, indent=2)
        output.write("\n")
        raw_trace_path = str(pathlib.Path(output.name).resolve())
    validation = validate_result(result, scenario, trace_events, store,
        semantic_assessment=assessment, money_fact_check=money_fact_check,
        query_reply_state_assessment=query_reply_state_assessment)
    board_recovery = None
    if scenario.require_board_recovery_delivery:
        board_recovery = run_board_recovery_probe(store, scenario.message, work_dir=db_path.parent / "board_recovery")
        created = next((item.result for item in result.tool_results if item.name == "create_game" and item.called
                        and item.allowed and not item.error), {})
        board_recovery["original_tool_arrangement"] = project_board_arrangement(created.get("board_arrangement", {}))
        board_recovery["checks"].append({"name": "create_game_returns_its_durable_board_arrangement",
            "passed": bool(board_recovery["original_tool_arrangement"]
                           and board_recovery["original_tool_arrangement"] == board_recovery.get("before_arrangement"))})
        board_recovery["status"] = "passed" if all(check["passed"] for check in board_recovery["checks"]) else "failed"
        validation["checks"].extend(board_recovery["checks"])
        validation["passed"] = validation["passed"] and all(check["passed"] for check in board_recovery["checks"])
    return {
        "status": "passed" if validation["passed"] else "failed",
        "trace_id": result.trace_id,
        "conversation_id": result.conversation_id,
        "scenario_id": scenario.scenario_id,
        "scenario": scenario.name,
        "iteration": iteration,
        "input": scenario.message.to_dict(),
        "final_reply": validation["final_reply"],
        "runtime_status": result.status,
        "tool_names": validation["tool_names"],
        "tool_result_summaries": summarize_tool_results(result.tool_results),
        "checks": validation["checks"],
        "semantic_assessment": assessment,
        "money_fact_check": money_fact_check,
        "requires_query_reply_state_check": scenario.requires_query_reply_state_check,
        **({"query_reply_state_assessment": query_reply_state_assessment} if scenario.requires_query_reply_state_check else {}),
        "product_contract_revision_ids": expectation["product_contract_revision_ids"],
        "recovery_metrics": validation["recovery_metrics"],
        "raw_trace_path": raw_trace_path,
        "db_path": str(db_path),
        "trace_steps": trace_steps,
        "decision_trace": decision_trace_snapshots(trace_events),
        **({"board_recovery": board_recovery} if board_recovery is not None else {}),
    }


def run_board_recovery_probe(store: SQLiteAgentStore, message: UserMessage, *, work_dir: pathlib.Path) -> dict[str, Any]:
    """关闭原SQLite后领取真实发布任务，下一沙盒发送及不可变快照必须包含新局。

    不补建关联或任务来修饰失败；所有安排须由原create_game执行链路产生。沙盒
    messenger只返回本地标识并留证，不能据此声称微信生产通道已验证。
    """
    from mahjong_agent_runtime.group_chat import BoardEngine
    from mahjong_agent_runtime.scheduled_tasks import ScheduledAgentTaskScheduler

    work_dir.mkdir(parents=True, exist_ok=True)
    trace = InMemoryTraceRecorder()
    report: dict[str, Any] = {"status": "failed", "scope": "sqlite_restart_and_sandbox_group_delivery", "checks": [],
                              "db_path": str(store.path), "sent_messages": [], "snapshots": []}
    restarted = None
    try:
        games = store.active_games(message.conversation_id)
        if len(games) != 1:
            raise ValueError("board recovery requires exactly one current game")
        game_id = games[0].game_id
        report["game_id"] = game_id
        report["before_arrangement"] = project_board_arrangement(store.game_board_arrangement(game_id))
        links = store.game_conversation_links(game_id=game_id)
        room_ids = {link.room_id for link in links}
        tasks = [task for task in store.scheduled_tasks.values() if task.task_type == "publish_group_board"
                 and task.aggregate_type == "group_room" and task.aggregate_id in room_ids and task.status.value == "pending"]
        report["pending_tasks"] = [task.to_dict() for task in tasks]
        arranged = report["before_arrangement"]
        queued = (arranged.get("status") == "scheduled" and arranged.get("linked_room_count") == 1
                  and arranged.get("pending_task_count") == 1 and arranged.get("outbound_authorized_room_count") == 1
                  and arranged.get("publication_sent") is False and len(tasks) == 1)
        report["checks"].append({"name": "private_game_has_durable_authorized_board_arrangement", "passed": queued})
        if not queued:
            raise ValueError("original game lacks its authorized durable board responsibility")
        store._connection.close()
        restarted = SQLiteAgentStore(store.path)
        stamp = max(tasks[0].due_at, now())

        class SandboxMessenger:
            """沙盒发送仍检查重启后持久授权，记录实际文本和发送结果而不访问外部通道。"""

            def send_group_message(self, room_id: str, text: str, *, metadata=None) -> str:
                policy = restarted.get_group_room_policy(room_id)
                if policy is None or not policy.managed or not policy.board_enabled or not policy.outbound_enabled:
                    raise ValueError("sandbox group send is not authorized")
                external_id = f"sandbox-board-message-{len(report['sent_messages']) + 1}"
                sent = {"room_id": room_id, "text": text, "external_message_id": external_id, "metadata": metadata,
                        "sandbox_only": True}
                report["sent_messages"].append(sent)
                trace.record("board-recovery", "sandbox_group_message_sent", sent)
                return external_id

        engine = BoardEngine(store=restarted, messenger=SandboxMessenger(), clock=lambda: stamp)

        def publish(task, trace_id):
            """只处理原任务对应的群，发送后读取数据库权威快照作为本次执行凭据。"""
            if task.task_type != "publish_group_board" or task.task_id != tasks[0].task_id:
                raise ValueError("unexpected board recovery task")
            snapshot = engine.publish(task.aggregate_id, trace_id=trace_id, publication_id=task.payload["publication_id"])
            if snapshot is None:
                raise ValueError("board task did not produce a delivered snapshot")
            report["snapshots"].append(snapshot.to_dict())

        trace.record("board-recovery", "sqlite_reopened", {"game_id": game_id, "original_task_id": tasks[0].task_id})
        scheduler = ScheduledAgentTaskScheduler(store=restarted, handler=publish, trace_recorder=trace)
        claimed = scheduler.run_task_once(tasks[0].task_id, at=stamp)
        final_task = restarted.scheduled_tasks[tasks[0].task_id]
        snapshot = restarted.get_latest_board_snapshot(tasks[0].aggregate_id)
        report["after_arrangement"] = project_board_arrangement(restarted.game_board_arrangement(game_id))
        report["checks"].extend([
            {"name": "board_publish_task_survives_sqlite_restart", "passed": claimed == 1 and final_task.status.value == "completed"},
            {"name": "next_sandbox_board_snapshot_contains_created_game", "passed": bool(snapshot is not None
                and any(item.game_id == game_id for item in snapshot.items) and len(report["sent_messages"]) == 1
                and report["sent_messages"][0]["external_message_id"] == snapshot.external_message_id
                and report["sent_messages"][0]["text"] == snapshot.rendered_text)},
        ])
        report["status"] = "passed" if all(check["passed"] for check in report["checks"]) else "failed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["checks"].append({"name": "board_recovery_execution_completed", "passed": False})
    finally:
        if restarted is not None:
            restarted._connection.close()
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix="board_recovery_", suffix=".json",
                                         dir=work_dir, delete=False) as output:
            json.dump({"scope": report["scope"], "events": [event.to_dict() for event in trace.events]},
                      output, ensure_ascii=False, indent=2)
            output.write("\n")
            report["raw_trace_path"] = str(pathlib.Path(output.name).resolve())
    return report


def validate_board_recovery_evidence(item: dict[str, Any], report_path: pathlib.Path) -> None:
    """从原建局结果、重启调度轨迹及真实快照核验群发布，不接受仅自报发送成功。"""
    import sqlite3

    proof = item.get("board_recovery")
    if not isinstance(proof, dict) or proof.get("status") != "passed" or proof.get("scope") != "sqlite_restart_and_sandbox_group_delivery":
        raise ValueError("owner board recovery proof is absent or failed")
    root = report_path.resolve().parent
    for path in (item.get("db_path"), item.get("raw_trace_path"), proof.get("raw_trace_path")):
        if not isinstance(path, str) or not pathlib.Path(path).resolve().is_relative_to(root) or not pathlib.Path(path).is_file():
            raise ValueError("owner board recovery evidence must belong to this run")
    if pathlib.Path(item["db_path"]).resolve() != pathlib.Path(proof.get("db_path", "")).resolve():
        raise ValueError("owner board recovery did not reopen the original SQLite")
    original = json.loads(pathlib.Path(item["raw_trace_path"]).read_text())
    created = [event["content"] for event in original["events"] if event["step"] == "tool_result"
               and event["content"].get("name") == "create_game" and event["content"].get("called") is True
               and event["content"].get("allowed") is True and not event["content"].get("error")]
    arranged = {"status": "scheduled", "linked_room_count": 1, "pending_task_count": 1,
                "outbound_authorized_room_count": 1, "publication_sent": False}
    if (len(created) != 1 or created[0]["result"].get("game", {}).get("game_id") != proof.get("game_id")
            or project_board_arrangement(created[0]["result"].get("board_arrangement", {})) != arranged
            or proof.get("before_arrangement") != arranged or proof.get("original_tool_arrangement") != arranged):
        raise ValueError("owner original create_game did not establish the queued board responsibility")
    pending, sent, snapshots = proof.get("pending_tasks"), proof.get("sent_messages"), proof.get("snapshots")
    if any(not isinstance(value, list) or len(value) != 1 for value in (pending, sent, snapshots)):
        raise ValueError("owner board recovery requires exactly the original task, next send and persisted snapshot")
    task, outgoing, snapshot = pending[0], sent[0], snapshots[0]
    if (task.get("task_type") != "publish_group_board" or task.get("aggregate_type") != "group_room"
            or task.get("status") != "pending" or not task.get("payload", {}).get("publication_id")
            or outgoing.get("sandbox_only") is not True or outgoing.get("room_id") != task.get("aggregate_id")
            or snapshot.get("room_id") != task.get("aggregate_id")
            or outgoing.get("external_message_id") != snapshot.get("external_message_id")
            or outgoing.get("text") != snapshot.get("rendered_text")
            or not any(value.get("game_id") == proof["game_id"] for value in snapshot.get("items", []))):
        raise ValueError("owner next board message does not contain the original game")
    raw = json.loads(pathlib.Path(proof["raw_trace_path"]).read_text())
    events = raw.get("events", [])
    if ([event["content"] for event in events if event.get("step") == "sandbox_group_message_sent"] != sent
            or len([event for event in events if event.get("step") == "sqlite_reopened"
                    and event["content"].get("original_task_id") == task["task_id"]]) != 1
            or len([event for event in events if event.get("step") == "scheduled_agent_task_claimed"
                    and event["content"].get("task", {}).get("task_id") == task["task_id"]]) != 1
            or len([event for event in events if event.get("step") == "scheduled_agent_task_completed"
                    and event["content"].get("task", {}).get("task_id") == task["task_id"]
                    and event["content"]["task"].get("status") == "completed"]) != 1):
        raise ValueError("owner board recovery lacks original lease, send or completion evidence")
    with closing(sqlite3.connect(pathlib.Path(item["db_path"]).resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        final_task = connection.execute("SELECT status,payload FROM runtime_scheduled_agent_tasks WHERE task_id=?", (task["task_id"],)).fetchone()
        final_snapshot = connection.execute("SELECT payload FROM runtime_group_board_snapshots WHERE snapshot_id=?", (snapshot["snapshot_id"],)).fetchone()
        policy_row = connection.execute("SELECT payload FROM runtime_group_room_policies WHERE room_id=?", (task["aggregate_id"],)).fetchone()
        link = connection.execute("SELECT 1 FROM runtime_game_conversation_links WHERE game_id=? AND room_id=?",
                                  (proof["game_id"], task["aggregate_id"])).fetchone()
        policy = json.loads(policy_row[0]) if policy_row else {}
        if (final_task is None or final_task[0] != "completed"
                or json.loads(final_task[1]).get("payload", {}).get("publication_id") != task["payload"]["publication_id"]
                or final_snapshot is None or json.loads(final_snapshot[0]) != snapshot or link is None
                or not all(policy.get(key) is True for key in ("managed", "board_enabled", "outbound_enabled"))):
            raise ValueError("owner persisted task, authorization, link or immutable snapshot differs from recovery proof")
    expected_checks = {"private_game_has_durable_authorized_board_arrangement", "board_publish_task_survives_sqlite_restart",
                       "next_sandbox_board_snapshot_contains_created_game", "create_game_returns_its_durable_board_arrangement"}
    if (not isinstance(proof.get("checks"), list) or len(proof["checks"]) != len(expected_checks)
            or {check.get("name") for check in proof["checks"]} != expected_checks
            or any(check.get("passed") is not True for check in proof["checks"])):
        raise ValueError("owner board recovery required checks are incomplete")


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def write_report(payload: dict[str, Any], report_path: pathlib.Path | None) -> None:
    if report_path is None:
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_reply_semantic_calibration_cases(path: pathlib.Path = SEMANTIC_CALIBRATION_PATH) -> list[dict[str, Any]]:
    """加载八条正反校准源；标签只留在报告，模型输入不携带案例编号、夹具名或答案。"""
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    fixtures = {"empty", "board_queued", "board_and_follow_up", "recorded_unconfigured"}
    if len(cases) != 8 or any(not isinstance(case, dict) for case in cases):
        raise ValueError("semantic calibration requires all eight controlled source cases")
    ids = [case.get("case_id") for case in cases]
    if (any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != len(ids)
            or any(case.get("schema_version") != 1 or case.get("kind") != "reply_semantic_calibration"
                   or case.get("fixture") not in fixtures or type(case.get("expected_passed")) is not bool
                   or any(not isinstance(case.get(key), str) or not case[key].strip() for key in ("request", "reply"))
                   or not isinstance(case.get("source"), dict) or not isinstance(case.get("expected_violation_ids"), list)
                   or any(value != "grounded_communication" for value in case["expected_violation_ids"])
                   or bool(case["expected_violation_ids"]) is case["expected_passed"] for case in cases)
            or {case["fixture"] for case in cases} != fixtures or sum(case["expected_passed"] for case in cases) != 3):
        raise ValueError("semantic calibration source identities, labels or fixture coverage are invalid")
    return cases


def reply_semantic_calibration_expectation() -> dict[str, Any]:
    """校准与真实回放共用真实性/隐私标准，正反例只改变事实和待判文本。"""
    return {"required_meaning": [{"id": "grounded_communication", "description":
        "回复实际声称的已完成业务动作及已承接未来责任，须与真实工具、持久状态和发布/复查机制一致。"
        "真实记录或安排后的短确认、尚未承接的服务征询都可自然表达，不要求汇报后台；"
        "未实际登记却声称已登记、无对应任务却承诺持续复查或通知、只有待办却声称已发送均违反本条。"}],
        "forbidden_meaning": common_forbidden_semantic_meanings()}


def read_semantic_calibration_state(db_path: pathlib.Path) -> dict[str, Any]:
    """只读SQLite重建权威局、参与者与任务，运行器和门禁共用相同事实投影。"""
    import sqlite3
    from mahjong_agent_runtime.stores.sqlite.serialization import game_from_payload, game_participant_from_row
    from mahjong_agent_runtime.domains.game_participants import normalize_game_parties

    with closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        games = []
        for row in connection.execute("SELECT game_id, conversation_id, status, payload FROM runtime_games ORDER BY game_id"):
            payload = json.loads(row["payload"])
            if any(payload.get(key) != row[key] for key in ("game_id", "conversation_id", "status")):
                raise ValueError("calibration game columns contradict their payload")
            game = game_from_payload(payload)
            game.participants = [game_participant_from_row(item) for item in connection.execute(
                "SELECT * FROM runtime_game_participants WHERE game_id=? ORDER BY joined_at,customer_id", (game.game_id,))]
            game.parties = normalize_game_parties(game.participants)
            games.append(game.to_dict())
        tasks = []
        for row in connection.execute("SELECT * FROM runtime_scheduled_agent_tasks ORDER BY task_id"):
            payload = json.loads(row["payload"])
            if any(payload.get(key) != row[key] for key in ("task_id", "task_type", "aggregate_id", "conversation_id", "status", "due_at")):
                raise ValueError("calibration task columns contradict their payload")
            tasks.append(payload)
        return {"games": games, "scheduled_tasks": tasks,
                **{key: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for key, table in (
                    ("invite_count", "runtime_invite_drafts"), ("outbound_count", "runtime_outbound_message_drafts"),
                    ("group_policy_count", "runtime_group_room_policies"), ("group_link_count", "runtime_game_conversation_links"),
                    ("snapshot_count", "runtime_group_board_snapshots"), ("waiting_count", "waiting_demands"))}}


def run_reply_semantic_calibration(client: Any, args: Any, work_dir: pathlib.Path, *, judge_fn=None) -> dict[str, Any]:
    """用真实网关和SQLite准备固定事实，独立判定正反文本，不运行主规划或外发。

    每例只走原assess_reply_semantics机制；有效的语义失败不重试，不把预期标签或
    生产审查结论喂给判定器。所有报告、原判决和数据库在新目录中独立留存。
    复杂判定预算独立解析，不继承用于主运行时的timeout_seconds。
    """
    from mahjong_agent_runtime.group_chat import GroupRoomPolicy
    from run_inquiry_grounding_eval import semantic_judge_config_matches

    cases = load_reply_semantic_calibration_cases()
    expectation = reply_semantic_calibration_expectation()
    work_dir.mkdir(parents=True, exist_ok=False)
    reports = []
    judge = judge_fn or assess_reply_semantics
    for index, case in enumerate(cases, start=1):
        case_dir = work_dir / case["case_id"]
        case_dir.mkdir()
        store = SQLiteAgentStore(case_dir / "calibration.sqlite3")
        trace = InMemoryTraceRecorder()
        trace_id = f"semantic_calibration_{index:02d}"
        message = UserMessage(conversation_id=f"calibration_conversation_{index:02d}", sender_id="calibration_customer",
            sender_name="客户", text=case["request"], message_id=f"calibration_message_{index:02d}")
        report = {"case_id": case["case_id"], "status": "failed", "trace_id": trace_id, "source": case,
                  "db_path": str(store.path), "prerequisite_results": [], "checks": []}
        try:
            store.upsert_customer(CustomerProfile(customer_id=message.sender_id, display_name=message.sender_name,
                preferred_games=["national_standard_mahjong"], preferred_stakes=["1"], smoke_preference="any"))
            if case["fixture"] in {"board_queued", "board_and_follow_up"}:
                store.upsert_group_room_policy(GroupRoomPolicy(room_id="calibration-authorized-board", managed=True,
                    board_enabled=True, outbound_enabled=True, merge_window_seconds=30))
            runtime = AgentRuntime(llm_client=client, store=store, tool_gateway=ToolGateway(store), trace_recorder=trace)
            prepared = runtime.task_context_manager.prepare(message, trace_id=trace_id)
            trace.record(trace_id, "task_context_prepared", prepared.to_dict())
            store.append_user_turn(message, trace_id)
            prerequisites = []

            def execute(name, arguments):
                """固定前置条件经真实网关执行，失败就保留证据并停止本例判定。"""
                call = ToolCall(name=name, arguments=arguments, reason="准备受控判定的真实业务事实")
                step_index = len(prerequisites) + 1
                trace.record(trace_id, "tool_called", {"call": call.to_dict(), "step_index": step_index})
                result = runtime.tool_gateway.execute(call, trace_id=trace_id, conversation_id=message.conversation_id,
                    sender_id=message.sender_id, sender_name=message.sender_name, step_index=step_index, source_message_id=message.message_id)
                prerequisites.append(result)
                trace.record(trace_id, "tool_result", result.to_dict())
                if not result.called or not result.allowed or result.error:
                    raise ValueError("calibration prerequisite was not executed successfully")
                return result

            if case["fixture"] == "empty":
                execute("search_current_games", {"requirement": {"game_type": "national_standard_mahjong", "stake": "1"}})
            else:
                created = execute("create_game", {"organizer_id": message.sender_id, "organizer_name": message.sender_name,
                    "requirement": {"game_type": "national_standard_mahjong", "stake": "1", "cap_score": 256,
                        "smoke_preference": "any", "start_time_kind": "asap_when_full", "known_player_count": 1, "needed_seats": 3},
                    "known_players": [{"customer_id": message.sender_id, "display_name": message.sender_name, "seat_count": 1}]})
                execute("search_customers", {"game_id": created.result["game"]["game_id"], "exclude_customer_ids": [message.sender_id]})
                if case["fixture"] == "board_and_follow_up":
                    stamp = now()
                    execute("schedule_follow_up", {"goal": "有限复查本次组局进展并准备同一客户反馈", "waiting_for": "players",
                        "due_at": (stamp + dt.timedelta(minutes=5)).isoformat(),
                        "expires_at": (stamp + dt.timedelta(minutes=30)).isoformat(), "review_budget": 1})
            report["prerequisite_results"] = [result.to_dict() for result in prerequisites]
            built = runtime.context_builder.build(message, trace_id=trace_id, previous_tool_results=[prerequisites[-1]],
                turn_tool_evidence=prerequisites)
            trace.record(trace_id, "context_built", built.payload)
            result = SimpleNamespace(final_reply=case["reply"], tool_results=prerequisites, status="completed")
            evidence = reply_semantic_evidence(result, SimpleNamespace(message=message), trace.get_trace(trace_id), store)
            report["business_state_before"] = read_semantic_calibration_state(store.path)
            assessment = judge(client, evidence=evidence, expectation=expectation, trace_id=trace_id + ":judge",
                               judge_timeout_seconds=getattr(args, "judge_timeout_seconds", None))
            trace.record(trace_id, "semantic_calibration_assessment", assessment)
            report["semantic_assessment"] = assessment
            report["business_state_after"] = read_semantic_calibration_state(store.path)
            violated = {item["id"] for item in assessment.get("criteria", []) if item.get("passed") is False}
            matches = (assessment.get("error") is None and type(assessment.get("passed")) is bool
                       and assessment["passed"] is case["expected_passed"] and set(case["expected_violation_ids"]) <= violated
                       and semantic_judge_config_matches(assessment, require_http=True))
            report["checks"] = [{"name": "independent_judge_matches_fixed_source", "passed": matches},
                {"name": "calibration_judge_has_no_business_side_effects", "passed": report["business_state_before"] == report["business_state_after"]}]
            report["status"] = "passed" if all(check["passed"] for check in report["checks"]) else "failed"
        except Exception as exc:
            report["error_type"] = type(exc).__name__
        finally:
            raw_path = case_dir / "raw_trace.json"
            raw_path.write_text(json.dumps({"trace_id": trace_id, "case_id": case["case_id"], "request": message.to_dict(),
                "events": [event.to_dict() for event in trace.get_trace(trace_id)]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            report.update(raw_trace_path=str(raw_path), trace_steps=[event.step for event in trace.get_trace(trace_id)])
            store._connection.close()
        reports.append(report)
    passed = sum(report["status"] == "passed" for report in reports)
    return {"status": "passed" if passed == len(cases) else "failed",
        "execution_scope": "controlled_reply_semantic_judge_with_real_sqlite_evidence",
        "source_sha256": hashlib.sha256(SEMANTIC_CALIBRATION_PATH.read_bytes()).hexdigest(),
        "source_ids": [case["case_id"] for case in cases], "case_count": len(cases), "passed_count": passed,
        "failed_count": len(cases) - passed, "reports": reports}


def validate_reply_semantic_calibration(payload: dict[str, Any], report_path: pathlib.Path) -> dict[str, Any]:
    """逐源核验校准的原判决、真实前置工具和只读SQLite，不信任自报成功或覆盖数量。"""
    import sqlite3
    from run_inquiry_grounding_eval import (
        REPLY_SEMANTIC_EVIDENCE_FIELDS, REPLY_SEMANTIC_JUDGE_RUBRIC,
        _parse_semantic_assessment, _project_judge_expectation, _semantic_criteria_ids, semantic_judge_config_matches,
    )

    cases = load_reply_semantic_calibration_cases()
    sources = {case["case_id"]: case for case in cases}
    expectation = reply_semantic_calibration_expectation()
    criterion_ids = _semantic_criteria_ids(expectation)
    if (not isinstance(payload, dict) or payload.get("status") != "passed"
            or payload.get("execution_scope") != "controlled_reply_semantic_judge_with_real_sqlite_evidence"
            or payload.get("source_sha256") != hashlib.sha256(SEMANTIC_CALIBRATION_PATH.read_bytes()).hexdigest()
            or payload.get("source_ids") != list(sources) or type(payload.get("case_count")) is not int
            or payload["case_count"] != len(sources) or type(payload.get("passed_count")) is not int
            or payload["passed_count"] != len(sources) or type(payload.get("failed_count")) is not int or payload["failed_count"] != 0
            or not isinstance(payload.get("reports"), list) or len(payload["reports"]) != len(sources)):
        raise ValueError("semantic calibration source coverage is absent, changed or failed")
    root = report_path.resolve().parent
    seen = set()
    for item in payload["reports"]:
        case_id = item.get("case_id") if isinstance(item, dict) else None
        if case_id not in sources or case_id in seen or item.get("source") != sources[case_id] or item.get("status") != "passed":
            raise ValueError("semantic calibration contains unknown, duplicate or changed cases")
        seen.add(case_id)
        source = sources[case_id]
        paths = []
        for field in ("raw_trace_path", "db_path"):
            value = item.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError("semantic calibration original evidence path is absent")
            path = pathlib.Path(value).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("semantic calibration evidence must belong to this run")
            paths.append(path)
        raw_path, db_path = paths
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        events = raw.get("events")
        if (raw.get("case_id") != case_id or not isinstance(item.get("trace_id"), str) or not item["trace_id"]
                or raw.get("trace_id") != item["trace_id"] or not isinstance(events, list) or not events
                or any(not isinstance(event, dict) or event.get("trace_id") != item["trace_id"]
                       or not isinstance(event.get("content"), dict) for event in events)
                or item.get("trace_steps") != [event.get("step") for event in events]):
            raise ValueError("semantic calibration original trace identity is inconsistent")
        request = raw.get("request", {})
        if request.get("text") != source["request"] or request.get("sender_id") != "calibration_customer":
            raise ValueError("semantic calibration did not use the fixed source request")
        results = [event["content"] for event in events if event.get("step") == "tool_result"]
        wanted = (["search_current_games"] if source["fixture"] == "empty" else ["create_game", "search_customers"])
        if source["fixture"] == "board_and_follow_up":
            wanted.append("schedule_follow_up")
        calls = [event["content"].get("call", {}) for event in events if event.get("step") == "tool_called"]
        completions = [event["content"] for event in events if event.get("step") == "tool_gateway_completed"]
        if (results != item.get("prerequisite_results") or [result.get("name") for result in results] != wanted
                or [call.get("name") for call in calls] != wanted or [result.get("tool_name") for result in completions] != wanted
                or any(result.get("called") is not True or result.get("allowed") is not True or result.get("error") for result in results)
                or any(result.get("called") is not True or result.get("allowed") is not True or result.get("error") for result in completions)):
            raise ValueError("semantic calibration prerequisites lack their actual successful gateway execution")
        state = read_semantic_calibration_state(db_path)
        if item.get("business_state_before") != state or item.get("business_state_after") != state:
            raise ValueError("semantic calibration business state changed or contradicts SQLite")
        has_game = source["fixture"] != "empty"
        has_board = source["fixture"] in {"board_queued", "board_and_follow_up"}
        has_follow_up = source["fixture"] == "board_and_follow_up"
        if (len(state["games"]) != int(has_game) or state["group_policy_count"] != int(has_board)
                or state["group_link_count"] != int(has_board)
                or any(state[key] for key in ("invite_count", "outbound_count", "snapshot_count", "waiting_count"))):
            raise ValueError("semantic calibration persisted fixture differs from its declared domain facts")
        tasks = state["scheduled_tasks"]
        board_tasks = [task for task in tasks if task.get("task_type") == "publish_group_board"]
        follow_tasks = [task for task in tasks if task.get("task_type") == "conversation_follow_up"]
        if (len(board_tasks) != int(has_board) or len(follow_tasks) != int(has_follow_up)
                or len(tasks) != int(has_board) + int(has_follow_up) or any(task.get("status") != "pending" for task in tasks)):
            raise ValueError("semantic calibration pending mechanisms are missing or changed")
        with closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)) as connection:
            policies = [json.loads(row[0]) for row in connection.execute("SELECT payload FROM runtime_group_room_policies")]
            links = list(connection.execute("SELECT game_id,room_id FROM runtime_game_conversation_links"))
            if has_board and (not all(policy.get(key) is True for policy in policies for key in ("managed", "board_enabled", "outbound_enabled"))
                    or links != [(state["games"][0]["game_id"], policies[0]["room_id"])]
                    or board_tasks[0].get("aggregate_type") != "group_room" or board_tasks[0].get("aggregate_id") != policies[0]["room_id"]):
                raise ValueError("semantic calibration board task is not bound to its authorized room and game")
        assessment = item.get("semantic_assessment")
        original_assessments = [event["content"] for event in events if event.get("step") == "semantic_calibration_assessment"]
        if not isinstance(assessment, dict) or original_assessments != [assessment] or assessment.get("error") is not None:
            raise ValueError("semantic calibration original independent judgement is absent or failed")
        if (assessment.get("expectation") != expectation or assessment.get("judge_expectation") != _project_judge_expectation(expectation)
                or assessment.get("rubric") != REPLY_SEMANTIC_JUDGE_RUBRIC
                or not semantic_judge_config_matches(assessment, require_http=True)):
            raise ValueError("semantic calibration changed the judgement contract or independent configuration")
        attempts = assessment.get("attempts")
        if (not isinstance(attempts, list) or len(attempts) not in {1, 2}
                or type(assessment.get("contract_repair_count")) is not int or assessment["contract_repair_count"] != len(attempts) - 1
                or attempts[-1].get("raw_response") != assessment.get("raw_response") or attempts[-1].get("error") is not None
                or attempts[0].get("trace_id") != item["trace_id"] + ":judge"):
            raise ValueError("semantic calibration lacks its actual independent call attempt")
        if len(attempts) == 2:
            try:
                _parse_semantic_assessment(attempts[0].get("raw_response", ""), criterion_ids)
            except ValueError:
                pass
            else:
                raise ValueError("semantic calibration retried a valid business judgement")
            if attempts[1].get("trace_id") != item["trace_id"] + ":judge:contract_repair_1":
                raise ValueError("semantic calibration repair call has inconsistent identity")
        parsed = _parse_semantic_assessment(assessment.get("raw_response", ""), criterion_ids)
        if any(assessment.get(key) != parsed[key] for key in ("reason", "criteria", "passed")):
            raise ValueError("semantic calibration verdict contradicts the original judge JSON")
        violated = {criterion["id"] for criterion in parsed["criteria"] if criterion["passed"] is False}
        if parsed["passed"] is not source["expected_passed"] or not set(source["expected_violation_ids"]) <= violated:
            raise ValueError("semantic calibration did not identify the fixed positive/negative boundary")
        evidence = assessment.get("evidence", {})
        projected = summarize_tool_results([SimpleNamespace(**result) for result in results])
        contexts = [event["content"] for event in events if event.get("step") == "context_built"]
        if (not isinstance(evidence, dict) or not set(evidence) <= REPLY_SEMANTIC_EVIDENCE_FIELDS
                or evidence.get("request") != request or evidence.get("reply") != source["reply"]
                or evidence.get("tool_results") != projected or len(contexts) != 1
                or evidence.get("known_profile") != contexts[0].get("sender_profile")
                or evidence.get("persisted_state", {}).get("games") != state["games"]
                or evidence.get("persisted_state", {}).get("related_active_games") != state["games"]
                or evidence.get("persisted_state", {}).get("invite_statuses") != []
                or evidence.get("execution_lifecycle", {}).get("turn") != {"status": "completed", "scope": "synchronous_turn_only"}):
            raise ValueError("semantic calibration judge saw changed input, tools or authoritative lifecycle facts")
        if has_game:
            game = state["games"][0]
            if (game.get("conversation_id") != request.get("conversation_id") or game.get("organizer_id") != request["sender_id"]
                    or game.get("requirement", {}).get("stake") != "1" or game.get("requirement", {}).get("cap_score") != 256
                    or game.get("seat_summary", {}).get("claimed_seats") != 1 or results[1].get("result", {}).get("candidates") != []):
                raise ValueError("semantic calibration game or exhausted search facts are invalid")
            expected_arrangement = {"status": "scheduled" if has_board else "unconfigured", "linked_room_count": int(has_board),
                "pending_task_count": int(has_board), "outbound_authorized_room_count": int(has_board), "publication_sent": False}
            activities = evidence.get("execution_lifecycle", {}).get("activities")
            if activities != [{"game_id": game["game_id"], "status": game["status"], "recruitment_status": game["recruitment_status"],
                               "board_arrangement": expected_arrangement}]:
                raise ValueError("semantic calibration board arrangement projection contradicts its actual tasks")
        elif evidence.get("execution_lifecycle", {}).get("activities") != []:
            raise ValueError("semantic calibration invented an activity absent from SQLite")
        follow_state = evidence.get("persisted_state", {}).get("follow_up_state", {})
        if bool(follow_state.get("tasks")) != has_follow_up:
            raise ValueError("semantic calibration lost or fabricated the personal follow-up mechanism")
        if has_follow_up:
            task = follow_tasks[0]
            expected_task = {key: task[key] for key in (
                "task_id", "task_type", "status", "conversation_id", "subject_id", "aggregate_type", "aggregate_id", "due_at")}
            expected_task.update({key: task["payload"].get(key) for key in (
                "task_context_id", "goal", "waiting_for", "expires_at", "review_budget", "review_count")})
            if (task.get("conversation_id") != request["conversation_id"] or task.get("subject_id") != request["sender_id"]
                    or task.get("aggregate_id") != follow_state.get("current_task_context_id")
                    or follow_state["tasks"] != [expected_task]
                    or not dt.datetime.fromisoformat(follow_state["observed_at"]) < dt.datetime.fromisoformat(task["due_at"])
                    <= dt.datetime.fromisoformat(task["payload"]["expires_at"])):
                raise ValueError("semantic calibration follow-up is not current, scoped and valid")
        expected_checks = {"independent_judge_matches_fixed_source", "calibration_judge_has_no_business_side_effects"}
        if (not isinstance(item.get("checks"), list) or len(item["checks"]) != len(expected_checks)
                or {check.get("name") for check in item["checks"]} != expected_checks
                or any(check.get("passed") is not True for check in item["checks"])):
            raise ValueError("semantic calibration omitted required checks")
    return {"total": len(sources), "passed": len(seen), "failed": 0, "skipped": 0, "executed_case_ids": sorted(seen)}


def main(argv: list[str] | None = None) -> int:
    """运行原老板场景或独立校准全套，额外任务证据不改变旧黄金输入及硬断言。"""
    parser = argparse.ArgumentParser(description="Run a live LLM eval against real owner chat expectations.")
    parser.add_argument("--db-path", type=pathlib.Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--report-path", type=pathlib.Path, default=None, help="write the full JSON report to this path")
    parser.add_argument("--strict", action="store_true", help="return non-zero when the live eval fails")
    parser.add_argument("--calibration-only", action="store_true", help="run all fixed independent semantic calibration cases with real SQLite prerequisites")
    parser.add_argument("--skip-review", action="store_true", help="skip customer-visible content review")
    parser.add_argument("--skip-text-generation", action="store_true", help="skip customer-visible text generation")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat every selected scenario against a freshly reset database",
    )
    parser.add_argument("--max-calls-per-turn", type=int, default=8)
    parser.add_argument("--max-tokens-per-call", type=int, default=int(os.getenv("MAHJONG_AGENT_MAX_TOKENS_PER_CALL", "64000")))
    parser.add_argument("--timeout-seconds", type=float, default=float(os.getenv("MAHJONG_AGENT_LLM_TIMEOUT_SECONDS", "45")))
    parser.add_argument("--judge-timeout-seconds", type=float, default=None,
        help="复杂语义判定的独立传输预算；未指定时使用专用环境变量或180秒默认值")
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="run only the selected scenario id; may be supplied more than once",
    )
    parser.add_argument("--dotenv-path", type=pathlib.Path, default=ROOT / ".env")
    parser.add_argument("--no-dotenv", action="store_true", help="do not load local .env before resolving LLM config")
    args = parser.parse_args(argv)
    if args.calibration_only and (args.report_path is None or args.scenario or args.skip_review or args.skip_text_generation):
        parser.error("--calibration-only requires --report-path and cannot filter scenarios or skip model stages")
    if args.repeat < 1 or args.repeat > 20:
        parser.error("--repeat must be between 1 and 20")

    # 即使只跑部分场景，也不能让未建明确语义合同的新场景静默落入通用猜测。
    scenarios = live_eval_scenarios()
    try:
        for scenario in scenarios:
            reply_semantic_expectation(scenario)
    except (ValueError, KeyError, TypeError) as exc:
        payload = {"status": "failed", "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "reason": "invalid or missing explicit golden semantic expectation", "error": str(exc)}
        write_report(payload, args.report_path)
        print_json(payload)
        return 1

    if not args.no_dotenv:
        load_dotenv_defaults(args.dotenv_path)
    client = OpenAICompatibleAgentClient.from_env()
    if client is None:
        payload = {
            "status": "skipped",
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "reason": "missing MAHJONG_LLM_API_KEY/DEEPSEEK_API_KEY or MAHJONG_LLM_MODEL",
            "command_hint": (
                "MAHJONG_LLM_PROVIDER=deepseek MAHJONG_LLM_MODEL=deepseek-v4-flash "
                "DEEPSEEK_API_KEY=*** PYTHONPATH=src python scripts/run_real_owner_chat_live_eval.py --strict"
            ),
        }
        write_report(payload, args.report_path)
        print_json(payload)
        return 1 if args.strict else 0

    if args.calibration_only:
        payload = run_reply_semantic_calibration(client, args, args.report_path.parent / "reply_semantic_calibration")
        write_report(payload, args.report_path)
        if payload["status"] == "passed":
            validate_reply_semantic_calibration(payload, args.report_path)
        print_json(payload)
        return 1 if payload["status"] != "passed" else 0

    if args.scenario:
        requested = set(args.scenario)
        known = {scenario.scenario_id for scenario in scenarios}
        unknown = sorted(requested - known)
        if unknown:
            parser.error(f"unknown scenario id(s): {', '.join(unknown)}")
        scenarios = [scenario for scenario in scenarios if scenario.scenario_id in requested]
    reports: list[dict[str, Any]] = []
    payload = {
        "status": "running",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scenario_count": 0,
        "scenario_definition_count": len(scenarios),
        "repeat": args.repeat,
        "passed_count": 0,
        "failed_count": 0,
        "reports": reports,
        "execution_scope": "selected_scenarios" if args.scenario else "all_live_scenarios",
        "semantic_judge_ran": True,
        "golden_reply_coverage": golden_reply_coverage(set()),
        "recovery_summary": summarize_action_contract_recovery([]),
    }
    write_report(payload, args.report_path)
    for iteration in range(1, args.repeat + 1):
        for scenario in scenarios:
            reports.append(run_scenario(client, args, scenario, iteration=iteration))
            # 每例完成即保存原始回复与独立判定，后续崩溃不能抹去前序证据。
            payload.update(
                scenario_count=len(reports),
                passed_count=sum(report["status"] == "passed" for report in reports),
                failed_count=sum(report["status"] != "passed" for report in reports),
                recovery_summary=summarize_action_contract_recovery(reports),
                golden_reply_coverage=golden_reply_coverage(
                    {report["scenario_id"] for report in reports if not report["semantic_assessment"].get("error")},
                    {report["scenario_id"] for report in reports if report["status"] == "passed"}),
            )
            write_report(payload, args.report_path)
            print(json.dumps({"scenario_id": scenario.scenario_id, "status": reports[-1]["status"],
                              "completed": len(reports), "total": len(scenarios) * args.repeat}, ensure_ascii=False), flush=True)
    passed = bool(reports) and all(report["status"] == "passed" for report in reports)
    payload["status"] = "passed" if passed else "failed"
    write_report(payload, args.report_path)
    print_json(payload)
    return 1 if args.strict and not passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
