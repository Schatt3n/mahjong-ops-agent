#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any, Callable


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mahjong_agent_runtime import (  # noqa: E402
    AgentRuntime,
    CustomerProfile,
    InMemoryTraceRecorder,
    OpenAICompatibleAgentClient,
    QuotedMessageRef,
    SQLiteAgentStore,
    TokenBudget,
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


DEFAULT_DB_PATH = ROOT / "runtime_data" / "real_owner_chat_live_eval.sqlite3"
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
    scenario_id: str
    name: str
    message: UserMessage
    setup: Callable[[SQLiteAgentStore], None]
    required_tool_names: list[str] = field(default_factory=list)
    required_tool_name_any: list[str] = field(default_factory=list)
    forbidden_tool_names: list[str] = field(default_factory=list)
    required_reply_any: list[list[str]] = field(default_factory=list)
    required_reply_contains: list[str] = field(default_factory=list)
    forbidden_reply_contains: list[str] = field(default_factory=list)
    forbidden_model_context_contains: list[str] = field(default_factory=list)
    forbidden_trace_steps: list[str] = field(default_factory=lambda: ["action_contract_error", "llm_error"])
    expected_tool_result_paths: dict[str, dict[str, Any]] = field(default_factory=dict)
    expected_active_game_count: int | None = None
    expected_active_game_status: str | None = None
    expected_active_game_seat_summary: dict[str, Any] = field(default_factory=dict)
    expected_active_game_requirement: dict[str, Any] = field(default_factory=dict)
    expected_task_memory_contains: list[dict[str, Any]] = field(default_factory=list)
    expected_checkpoint_contains: list[str] = field(default_factory=list)
    max_reply_chars: int | None = 80


def build_empty_store(db_path: pathlib.Path) -> SQLiteAgentStore:
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteAgentStore(db_path)


def future_planned_start_at(clock: str) -> str:
    """Keep fixture games active regardless of the wall-clock time when evals run."""

    hour_text, _, minute_text = str(clock).partition(":")
    now = dt.datetime.now().astimezone()
    target_day = now + dt.timedelta(days=1)
    return target_day.replace(hour=int(hour_text), minute=int(minute_text or 0), second=0, microsecond=0).isoformat()


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
            "planned_start_at": future_planned_start_at("19:00"),
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
            "needed_seats": 1,
            "user_visible_summary": "七点三缺一",
        },
        known_players=[
            {"customer_id": "summer", "display_name": "夏日", "status": "confirmed"},
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


def build_runtime(client: OpenAICompatibleAgentClient, store: SQLiteAgentStore, trace: InMemoryTraceRecorder, args: argparse.Namespace) -> AgentRuntime:
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


def live_eval_scenarios() -> list[LiveEvalScenario]:
    common_forbidden = [
        *IMPLEMENTATION_DETAIL_FORBIDDEN_REPLY_FRAGMENTS,
        *CUSTOMER_SERVICE_FORBIDDEN_REPLY_FRAGMENTS,
    ]
    return [
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
            ),
            required_tool_names=["search_current_games"],
            expected_tool_result_paths={
                "search_current_games": {
                    "result.requirement.game_type": "hangzhou_mahjong",
                    "result.requirement.stake": "0.5",
                    "result.requirement.smoke_preference": "no_smoke",
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
            required_tool_names=["record_candidate_reply"],
            expected_tool_result_paths={
                "record_candidate_reply": {
                    "result.cross_game_commitment.released_participations.0.customer_id": "owner_real_customer",
                }
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
                    "scope": "current_task",
                    "field": "max_duration_hours",
                    "value": 4,
                    "status": "active",
                }
            ],
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
            forbidden_reply_contains=[
                "要组一个吗",
                "打多大",
                "几个人",
                "什么玩法",
                "这个先不聊",
                "打牌直接说",
                "七点",
                "7点",
                "三缺一",
                "371",
                "打吗",
                "来吗",
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
            required_reply_any=[["对", "嗯", "是", "确实", "哈哈", "懂", "明白"]],
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


def validate_result(
    result: Any,
    scenario: LiveEvalScenario,
    trace_events: list[Any],
    store: SQLiteAgentStore,
) -> dict[str, Any]:
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
        checks.append(
            {
                "name": "should_call_any_state_tracking_tool",
                "passed": any(tool_name in tool_names for tool_name in scenario.required_tool_name_any),
                "expected_any": scenario.required_tool_name_any,
                "actual": tool_names,
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
    for tool_name, expected_paths in scenario.expected_tool_result_paths.items():
        matching_tool_result = next((item for item in result.tool_results if item.name == tool_name), None)
        for path, expected in expected_paths.items():
            actual = value_at_path(matching_tool_result.to_dict() if matching_tool_result else None, path)
            checks.append(
                {
                    "name": f"{tool_name}_result_should_keep_{path}",
                    "passed": eval_values_equal(actual, expected),
                    "expected": expected,
                    "actual": actual,
                }
            )
    for index, alternatives in enumerate(scenario.required_reply_any, start=1):
        checks.append(
            {
                "name": f"reply_should_contain_any_{index}",
                "passed": any(reply_contains_required_fragment(final_reply, item) for item in alternatives),
                "expected_any": alternatives,
                "actual": final_reply,
            }
        )
    for item in scenario.required_reply_contains:
        checks.append(
            {
                "name": f"reply_should_contain_{item}",
                "passed": item in final_reply,
                "expected": item,
                "actual": final_reply,
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
                    "passed": active_requirement.get(key) == expected,
                    "expected": expected,
                    "actual": active_requirement.get(key),
                }
            )
    if scenario.expected_task_memory_contains:
        task_memories = store.task_memory_context(
            scenario.message.conversation_id,
            scenario.message.sender_id,
        )
        for index, expected in enumerate(scenario.expected_task_memory_contains, start=1):
            matched_memory = next(
                (
                    memory
                    for memory in task_memories
                    if all(memory.get(key) == value for key, value in expected.items())
                ),
                None,
            )
            checks.append(
                {
                    "name": f"task_memory_should_match_{index}",
                    "passed": matched_memory is not None,
                    "expected": expected,
                    "actual": matched_memory if matched_memory is not None else task_memories,
                }
            )
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
    return {
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "final_reply": final_reply,
        "tool_names": tool_names,
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


def value_at_path(payload: Any, dotted_path: str) -> Any:
    current = payload
    for raw_part in dotted_path.split("."):
        if isinstance(current, list):
            try:
                current = current[int(raw_part)]
            except (ValueError, IndexError):
                return None
            continue
        if isinstance(current, dict):
            current = current.get(raw_part)
            continue
        return None
    return current


def eval_values_equal(actual: Any, expected: Any) -> bool:
    """Compare semantically equivalent fixture values without weakening checks."""

    if actual == expected:
        return True
    if isinstance(actual, str) and isinstance(expected, str):
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


def reply_contains_required_fragment(reply: str, fragment: str) -> bool:
    return fragment in reply or fragment.casefold() in reply.casefold()


def summarize_tool_results(tool_results: list[Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for item in tool_results:
        result = dict(item.result or {})
        summary: dict[str, Any] = {
            "name": item.name,
            "called": item.called,
            "allowed": item.allowed,
            "error": item.error,
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
    """Keep the model/tool decision boundary in eval reports without copying full prompts."""

    high_signal_steps = {
        "action_contract_error",
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


def run_scenario(client: OpenAICompatibleAgentClient, args: argparse.Namespace, scenario: LiveEvalScenario) -> dict[str, Any]:
    db_path = scenario_db_path(args.db_path, scenario.scenario_id)
    store = build_empty_store(db_path)
    scenario.setup(store)
    trace = InMemoryTraceRecorder()
    runtime = build_runtime(client, store, trace, args)
    trace_id = f"trace_owner_real_live_eval_{scenario.scenario_id}"
    result = runtime.handle_user_message(scenario.message, trace_id=trace_id)
    trace_events = trace.get_trace(trace_id)
    trace_steps = [item.step for item in trace_events]
    validation = validate_result(result, scenario, trace_events, store)
    return {
        "status": "passed" if validation["passed"] else "failed",
        "trace_id": result.trace_id,
        "conversation_id": result.conversation_id,
        "scenario_id": scenario.scenario_id,
        "scenario": scenario.name,
        "input": scenario.message.to_dict(),
        "final_reply": validation["final_reply"],
        "tool_names": validation["tool_names"],
        "tool_result_summaries": summarize_tool_results(result.tool_results),
        "checks": validation["checks"],
        "db_path": str(db_path),
        "trace_steps": trace_steps,
        "decision_trace": decision_trace_snapshots(trace_events),
    }


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def write_report(payload: dict[str, Any], report_path: pathlib.Path | None) -> None:
    if report_path is None:
        return
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a live LLM eval against real owner chat expectations.")
    parser.add_argument("--db-path", type=pathlib.Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--report-path", type=pathlib.Path, default=None, help="write the full JSON report to this path")
    parser.add_argument("--strict", action="store_true", help="return non-zero when the live eval fails")
    parser.add_argument("--skip-review", action="store_true", help="skip customer-visible content review")
    parser.add_argument("--skip-text-generation", action="store_true", help="skip customer-visible text generation")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-calls-per-turn", type=int, default=8)
    parser.add_argument("--max-tokens-per-call", type=int, default=int(os.getenv("MAHJONG_AGENT_MAX_TOKENS_PER_CALL", "24000")))
    parser.add_argument("--timeout-seconds", type=float, default=float(os.getenv("MAHJONG_AGENT_LLM_TIMEOUT_SECONDS", "45")))
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="run only the selected scenario id; may be supplied more than once",
    )
    parser.add_argument("--dotenv-path", type=pathlib.Path, default=ROOT / ".env")
    parser.add_argument("--no-dotenv", action="store_true", help="do not load local .env before resolving LLM config")
    args = parser.parse_args(argv)

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
        return 0

    scenarios = live_eval_scenarios()
    if args.scenario:
        requested = set(args.scenario)
        known = {scenario.scenario_id for scenario in scenarios}
        unknown = sorted(requested - known)
        if unknown:
            parser.error(f"unknown scenario id(s): {', '.join(unknown)}")
        scenarios = [scenario for scenario in scenarios if scenario.scenario_id in requested]
    reports = [run_scenario(client, args, scenario) for scenario in scenarios]
    passed = all(report["status"] == "passed" for report in reports)
    payload = {
        "status": "passed" if passed else "failed",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scenario_count": len(reports),
        "passed_count": sum(1 for report in reports if report["status"] == "passed"),
        "failed_count": sum(1 for report in reports if report["status"] != "passed"),
        "reports": reports,
    }
    write_report(payload, args.report_path)
    print_json(payload)
    return 1 if args.strict and not passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
