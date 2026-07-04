#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any


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


def build_store(db_path: pathlib.Path) -> SQLiteAgentStore:
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    store = SQLiteAgentStore(db_path)
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
    return store


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


def validate_result(result: Any) -> dict[str, Any]:
    final_reply = str(result.final_reply or "")
    tool_names = [item.name for item in result.tool_results if item.name not in {"customer_visible_text_generation", "customer_visible_content_review"}]
    required_reply_any = [
        ["七点", "7点", "19:00"],
        ["三缺一", "371", "缺一"],
        ["可以不", "可以吗", "打吗", "来吗"],
    ]
    forbidden_reply_contains = [
        "打多大",
        "几个人",
        "0.5",
        "无烟",
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
    checks: list[dict[str, Any]] = []
    checks.append(
        {
            "name": "should_call_search_current_games",
            "passed": "search_current_games" in tool_names,
            "actual": tool_names,
        }
    )
    for index, alternatives in enumerate(required_reply_any, start=1):
        checks.append(
            {
                "name": f"reply_should_contain_any_{index}",
                "passed": any(item in final_reply for item in alternatives),
                "expected_any": alternatives,
                "actual": final_reply,
            }
        )
    for item in forbidden_reply_contains:
        checks.append(
            {
                "name": f"reply_should_not_contain_{item}",
                "passed": item not in final_reply,
                "forbidden": item,
                "actual": final_reply,
            }
        )
    return {
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "final_reply": final_reply,
        "tool_names": tool_names,
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

    store = build_store(args.db_path)
    trace = InMemoryTraceRecorder()
    runtime = build_runtime(client, store, trace, args)
    message = UserMessage(
        conversation_id="owner_real_customer_chat",
        sender_id="owner_real_customer",
        sender_name="常客",
        text="帮我约个6.30无烟的",
        message_id="msg_owner_real_live_eval_profile_default",
    )
    trace_id = "trace_owner_real_live_eval"
    result = runtime.handle_user_message(message, trace_id=trace_id)
    validation = validate_result(result)
    print_json(
        {
            "status": "passed" if validation["passed"] else "failed",
            "trace_id": result.trace_id,
            "conversation_id": result.conversation_id,
            "scenario": "真实老板聊天：画像默认 0.5/一人，先查七点三缺一，再短句确认",
            "input": message.to_dict(),
            "final_reply": validation["final_reply"],
            "tool_names": validation["tool_names"],
            "checks": validation["checks"],
            "db_path": str(args.db_path),
            "trace_steps": [item.step for item in trace.get_trace(trace_id)],
        }
    )
    return 1 if args.strict and not validation["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
