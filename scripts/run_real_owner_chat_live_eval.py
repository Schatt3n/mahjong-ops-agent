#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    SQLiteAgentStore,
    TokenBudget,
    ToolGateway,
    UserMessage,
)


DEFAULT_DB_PATH = ROOT / "runtime_data" / "real_owner_chat_live_eval.sqlite3"


@dataclass(slots=True)
class LiveEvalScenario:
    scenario_id: str
    name: str
    message: UserMessage
    setup: Callable[[SQLiteAgentStore], None]
    required_tool_names: list[str] = field(default_factory=list)
    required_tool_name_any: list[str] = field(default_factory=list)
    required_reply_any: list[list[str]] = field(default_factory=list)
    required_reply_contains: list[str] = field(default_factory=list)
    forbidden_reply_contains: list[str] = field(default_factory=list)
    forbidden_trace_steps: list[str] = field(default_factory=lambda: ["action_contract_error", "llm_error"])


def build_empty_store(db_path: pathlib.Path) -> SQLiteAgentStore:
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return SQLiteAgentStore(db_path)


def seed_default_profiles(store: SQLiteAgentStore) -> None:
    store.upsert_customer(
        CustomerProfile(
            customer_id="owner_real_customer",
            display_name="常客",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["0.5", "1"],
            smoke_preference="no_smoke",
            response_score=0.9,
            notes=(
                "真实老板聊天画像：95% 情况打 0.5，打 1 块会单独说；"
                "95% 情况是一个人来，带人会单独说；偏好无烟。"
            ),
        )
    )
    store.upsert_customer(CustomerProfile(customer_id="summer", display_name="夏日"))
    store.upsert_customer(CustomerProfile(customer_id="smile", display_name="笑脸"))
    store.upsert_customer(CustomerProfile(customer_id="kexing", display_name="可星"))


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
            "duration_hours": 4,
            "needed_seats": 1,
            "user_visible_summary": "七点三缺一",
        },
        known_players=[
            {"customer_id": "smile", "display_name": "笑脸"},
            {"customer_id": "kexing", "display_name": "可星"},
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
        organizer_name="夏日",
        requirement={
            "game_type": "hangzhou_mahjong",
            "stake": "0.5",
            "smoke_preference": "no_smoke",
            "start_time_kind": "scheduled",
            "start_time": "19:00",
            "duration_hours": 4,
            "needed_seats": 1,
            "user_visible_summary": "七点三缺一",
        },
        known_players=[
            {"customer_id": "smile", "display_name": "笑脸"},
            {"customer_id": "kexing", "display_name": "可星"},
        ],
        trace_id="trace_owner_real_public_nickname_setup",
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
        "候选人",
        "邀约草稿",
        "老板审批",
        "系统",
        "模型",
        "AI",
        "agent",
        "已发送",
        "问了",
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
            required_reply_any=[
                ["七点", "7点", "19:00"],
                ["三缺一", "371", "缺一"],
                ["可以不", "可以吗", "打吗", "来吗"],
            ],
            forbidden_reply_contains=["打多大", "几个人", "0.5", "无烟", *common_forbidden],
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
                "私有",
                *common_forbidden,
            ],
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
    ]


def validate_result(result: Any, scenario: LiveEvalScenario, trace_steps: list[str]) -> dict[str, Any]:
    final_reply = str(result.final_reply or "")
    tool_names = [item.name for item in result.tool_results if item.name not in {"customer_visible_text_generation", "customer_visible_content_review"}]
    checks: list[dict[str, Any]] = []
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
    for index, alternatives in enumerate(scenario.required_reply_any, start=1):
        checks.append(
            {
                "name": f"reply_should_contain_any_{index}",
                "passed": any(item in final_reply for item in alternatives),
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
    for step in scenario.forbidden_trace_steps:
        checks.append(
            {
                "name": f"trace_should_not_have_{step}",
                "passed": step not in trace_steps,
                "forbidden": step,
                "actual": trace_steps,
            }
        )
    return {
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "final_reply": final_reply,
        "tool_names": tool_names,
    }


def scenario_db_path(base_path: pathlib.Path, scenario_id: str) -> pathlib.Path:
    return base_path.with_name(f"{base_path.stem}_{scenario_id}{base_path.suffix}")


def run_scenario(client: OpenAICompatibleAgentClient, args: argparse.Namespace, scenario: LiveEvalScenario) -> dict[str, Any]:
    db_path = scenario_db_path(args.db_path, scenario.scenario_id)
    store = build_empty_store(db_path)
    scenario.setup(store)
    trace = InMemoryTraceRecorder()
    runtime = build_runtime(client, store, trace, args)
    trace_id = f"trace_owner_real_live_eval_{scenario.scenario_id}"
    result = runtime.handle_user_message(scenario.message, trace_id=trace_id)
    trace_steps = [item.step for item in trace.get_trace(trace_id)]
    validation = validate_result(result, scenario, trace_steps)
    return {
        "status": "passed" if validation["passed"] else "failed",
        "trace_id": result.trace_id,
        "conversation_id": result.conversation_id,
        "scenario_id": scenario.scenario_id,
        "scenario": scenario.name,
        "input": scenario.message.to_dict(),
        "final_reply": validation["final_reply"],
        "tool_names": validation["tool_names"],
        "checks": validation["checks"],
        "db_path": str(db_path),
        "trace_steps": trace_steps,
    }


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a live LLM eval against real owner chat expectations.")
    parser.add_argument("--db-path", type=pathlib.Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--strict", action="store_true", help="return non-zero when the live eval fails")
    parser.add_argument("--skip-review", action="store_true", help="skip customer-visible content review")
    parser.add_argument("--skip-text-generation", action="store_true", help="skip customer-visible text generation")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-calls-per-turn", type=int, default=8)
    parser.add_argument("--max-tokens-per-call", type=int, default=int(os.getenv("MAHJONG_AGENT_MAX_TOKENS_PER_CALL", "24000")))
    parser.add_argument("--timeout-seconds", type=float, default=float(os.getenv("MAHJONG_AGENT_LLM_TIMEOUT_SECONDS", "45")))
    args = parser.parse_args(argv)

    client = OpenAICompatibleAgentClient.from_env()
    if client is None:
        print_json(
            {
                "status": "skipped",
                "reason": "missing MAHJONG_LLM_API_KEY/DEEPSEEK_API_KEY or MAHJONG_LLM_MODEL",
                "command_hint": (
                    "MAHJONG_LLM_PROVIDER=deepseek MAHJONG_LLM_MODEL=deepseek-v4-flash "
                    "DEEPSEEK_API_KEY=*** PYTHONPATH=src python scripts/run_real_owner_chat_live_eval.py --strict"
                ),
            }
        )
        return 0

    reports = [run_scenario(client, args, scenario) for scenario in live_eval_scenarios()]
    passed = all(report["status"] == "passed" for report in reports)
    print_json({"status": "passed" if passed else "failed", "scenario_count": len(reports), "reports": reports})
    return 1 if args.strict and not passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
