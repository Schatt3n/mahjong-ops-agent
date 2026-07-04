from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from mahjong_agent_runtime import (
    AgentRuntime,
    CustomerProfile,
    InMemoryAgentStore,
    InMemoryTraceRecorder,
    StaticAgentClient,
    UserMessage,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "eval" / "golden" / "real_owner_chat_golden.jsonl"


def read_records() -> list[dict]:
    return [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_real_owner_chat_golden_transcript_is_structured() -> None:
    records = read_records()

    assert len(records) == 2
    record = next(item for item in records if item["kind"] == "real_owner_chat_golden")
    assert record["kind"] == "real_owner_chat_golden"
    assert record["id"] == "owner_chat_ai_chitchat_resume_20260704_001"

    messages = record["messages"]
    assert len(messages) == 67
    assert {message["role"] for message in messages} == {"customer", "boss"}
    assert all(message["text"] for message in messages)
    assert all("source_image" in message for message in messages)


def test_real_owner_chat_golden_covers_context_resume_cases() -> None:
    record = next(item for item in read_records() if item["kind"] == "real_owner_chat_golden")
    eval_cases = {item["id"]: item for item in record["eval_cases"]}
    facts = {item["id"]: item for item in record["business_facts"]}

    assert "resume_game_status_after_casual_chat" in eval_cases
    assert "later_people_count_query_should_search_or_answer_current_status" in eval_cases
    assert "reject_smoking_game_updates_preference" in eval_cases
    assert "casual_chat_interruption" in facts
    assert "resume_status_query" in facts

    resume_case = eval_cases["resume_game_status_after_casual_chat"]
    assert resume_case["expected"]["must_use_existing_context"] is True
    assert resume_case["expected"]["should_not_treat_as_new_game"] is True
    assert "我是AI" in resume_case["expected"]["forbidden_reply_contains"]


def test_real_owner_chat_supplement_captures_profile_defaults_and_privacy_boundary() -> None:
    supplement = next(item for item in read_records() if item["kind"] == "real_owner_chat_eval_supplement")
    eval_cases = {item["id"]: item for item in supplement["eval_cases"]}
    facts = {item["id"]: item for item in supplement["business_facts"]}

    assert supplement["parent_id"] == "owner_chat_ai_chitchat_resume_20260704_001"
    assert supplement["hidden_context"][0]["event"].startswith("老板已单独拉群")
    assert "局群" in supplement["hidden_context"][1]["event"]
    assert "退出" in supplement["hidden_context"][1]["event"]
    assert "95% 情况打 0.5" in supplement["customer_profile_assumptions"]["profile_facts"][0]
    assert "profile_defaults_fill_missing_slots" in facts
    assert "public_nickname_allowed_private_remark_forbidden" in facts

    initial_case = eval_cases["initial_request_uses_profile_defaults_and_searches_pool"]
    assert initial_case["expected"]["required_tool_sequence_prefix"] == ["search_current_games"]
    assert initial_case["expected"]["search_requirement_should_include"]["stake"] == "0.5"
    assert "打多大" in initial_case["expected"]["forbidden_reply_contains"]
    assert "几个人" in initial_case["expected"]["forbidden_reply_contains"]

    people_case = eval_cases["who_are_the_players_can_show_public_nickname_only"]
    assert "公开微信昵称" in people_case["expected"]["reply_allowed_content"]
    assert "老板微信备注" in people_case["expected"]["forbidden_reply_content"]


def test_real_owner_chat_agent_flow_uses_profile_defaults_to_query_pool() -> None:
    store = InMemoryAgentStore()
    store.upsert_customer(
        CustomerProfile(
            customer_id="owner_real_customer",
            display_name="常客",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["0.5", "1"],
            smoke_preference="no_smoke",
            response_score=0.9,
            notes="画像默认：95%打0.5，打1块会单独说；95%是一个人来，带人会单独说。",
        )
    )
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
            "needed_seats": 1,
            "user_visible_summary": "七点三缺一",
        },
        known_players=[
            {"customer_id": "smile", "display_name": "笑脸"},
            {"customer_id": "kexing", "display_name": "可星"},
        ],
        trace_id="trace_owner_real_pool_setup",
    )
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="老客户画像高置信默认0.5和1人，先查6:30附近无烟局。",
                tool_calls=[
                    {
                        "name": "search_current_games",
                        "arguments": {
                            "requirement": {
                                "game_type": "hangzhou_mahjong",
                                "stake": "0.5",
                                "smoke_preference": "no_smoke",
                                "start_time_kind": "scheduled",
                                "start_time": "18:30",
                                "known_player_count": 1,
                                "needed_seats": 3,
                            },
                            "limit": 5,
                        },
                        "reason": "用画像默认槽位查当前局。",
                    }
                ],
            ),
            action_json(
                objective_status="completed",
                reasoning_summary="查到七点三缺一，给用户短句确认。",
                reply_to_user="七点三缺一，可以不",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorder())

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="owner_real_customer_chat",
            sender_id="owner_real_customer",
            sender_name="常客",
            text="帮我约个6.30无烟的",
            message_id="msg_owner_real_profile_default",
        ),
        trace_id="trace_owner_real_profile_default",
    )

    assert result.final_reply == "七点三缺一，可以不"
    assert [item.name for item in result.tool_results] == ["search_current_games"]
    search_result = result.tool_results[0].result
    assert search_result["requirement"]["stake"] == "0.5"
    assert search_result["requirement"]["known_player_count"] == 1
    assert search_result["matches"][0]["game"]["remaining_seats"] == 1
    assert search_result["matches"][0]["join_projection"]["would_fill_game"] is True
    assert len(store.games) == 1

    first_prompt_payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert "95%打0.5" in first_prompt_payload["sender_profile"]["notes"]
    assert "打多大" not in result.final_reply
    assert "几个人" not in result.final_reply
    assert "0.5" not in result.final_reply
    assert "无烟" not in result.final_reply


def test_real_owner_chat_live_eval_skips_without_llm_env(monkeypatch, capsys) -> None:
    for name in ("MAHJONG_LLM_API_KEY", "DEEPSEEK_API_KEY", "MAHJONG_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    script_path = ROOT / "scripts" / "run_real_owner_chat_live_eval.py"
    spec = importlib.util.spec_from_file_location("run_real_owner_chat_live_eval", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    exit_code = module.main([])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "skipped"
    assert "MAHJONG_LLM_MODEL" in payload["reason"]


def test_real_owner_live_eval_seed_games_keep_expected_seat_counts(tmp_path: Path) -> None:
    script_path = ROOT / "scripts" / "run_real_owner_chat_live_eval.py"
    spec = importlib.util.spec_from_file_location("run_real_owner_chat_live_eval_for_seed_test", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    resume_store = module.SQLiteAgentStore(tmp_path / "resume.sqlite3")
    module.setup_resume_status_after_casual_chat(resume_store)
    resume_game = resume_store.active_games("owner_real_customer_chat")[0]
    assert resume_game.seat_summary()["claimed_seats"] == 2
    assert resume_game.seat_summary()["remaining_seats"] == 2
    assert resume_game.requirement["user_visible_summary"] == "还没有，还差俩"

    later_store = module.SQLiteAgentStore(tmp_path / "later.sqlite3")
    module.setup_later_people_count_query(later_store)
    later_game = later_store.active_games("owner_real_customer_chat")[0]
    assert later_game.seat_summary()["claimed_seats"] == 2
    assert later_game.seat_summary()["remaining_seats"] == 2
    assert later_game.requirement["user_visible_summary"] == "两个人，18.30 星月的局，371 她"


def action_json(
    *,
    objective_status: str,
    reasoning_summary: str = "test",
    reply_to_user: str = "",
    tool_calls: list[dict] | None = None,
) -> str:
    if objective_status == "needs_tool":
        stop_reason = {
            "can_stop": False,
            "why": "还需要调用工具。",
            "pending_work": [call.get("name", "tool") for call in tool_calls or []],
            "depends_on_tool_results": False,
        }
    else:
        stop_reason = {
            "can_stop": True,
            "why": "已经可以回复用户。",
            "pending_work": [],
            "depends_on_tool_results": False,
        }
    return json.dumps(
        {
            "goal": "真实老板聊天评测",
            "objective_status": objective_status,
            "reasoning_summary": reasoning_summary,
            "reply_to_user": reply_to_user,
            "tool_calls": tool_calls or [],
            "needs_human": False,
            "stop_reason": stop_reason,
            "badcase": None,
        },
        ensure_ascii=False,
    )
