from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from mahjong_agent_runtime.customer_visible_contract import (
    FORBIDDEN_CUSTOMER_SERVICE_PHRASES,
    FORBIDDEN_IMPLEMENTATION_IDENTITY_TERMS,
    FORBIDDEN_INTERNAL_PROCESS_TERMS,
)
from mahjong_agent_runtime import (
    AgentRuntime,
    CustomerProfile,
    InMemoryAgentStore,
    InMemoryTraceRecorder,
    StaticAgentClient,
    UserMessage,
)
from mahjong_agent_runtime.env import load_dotenv_defaults


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "eval" / "golden" / "real_owner_chat_golden.jsonl"


def future_planned_start_at(clock: str) -> str:
    """Return a future ISO timestamp while keeping the human-facing clock label stable."""

    hour_text, _, minute_text = clock.partition(":")
    now = dt.datetime.now().astimezone()
    target_day = now + dt.timedelta(days=1)
    return target_day.replace(hour=int(hour_text), minute=int(minute_text or 0), second=0, microsecond=0).isoformat()


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
    assert "打吗" in resume_case["expected"]["forbidden_reply_contains"]


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
    assert "打吗" in people_case["expected"]["forbidden_reply_content"]


def test_real_owner_chat_supplement_pins_duration_exit_and_human_style() -> None:
    supplement = next(item for item in read_records() if item["kind"] == "real_owner_chat_eval_supplement")
    eval_cases = {item["id"]: item for item in supplement["eval_cases"]}
    facts = {item["id"]: item for item in supplement["business_facts"]}

    assert "group_invite_then_duration_mismatch" in facts
    assert "客户拒绝并退群" in facts["group_invite_then_duration_mismatch"]["fact"]

    duration_case = eval_cases["group_duration_mismatch_records_exit"]
    assert duration_case["expected"]["intent"] == "decline_current_group_due_duration"
    assert "max_duration_hours=4" in duration_case["expected"]["profile_updates"]
    assert "customer_removed_or_not_joined_current_group" in duration_case["expected"]["state_updates"]
    assert "5小时局" in duration_case["expected"]["forbidden_next_recommendation"]

    style_case = eval_cases["human_likeness_reply_should_be_short_and_decision_focused"]
    assert "七点三缺一，可以不" in style_case["expected"]["good_examples"]
    assert "还没有，还差俩" in style_case["expected"]["good_examples"]
    assert "根据您的画像和当前局池，我找到一个0.5无烟局" in style_case["expected"]["bad_examples"]


def test_real_owner_chat_agent_flow_uses_profile_defaults_to_query_pool() -> None:
    store = InMemoryAgentStore()
    store.upsert_customer(
        CustomerProfile(
            customer_id="owner_real_customer",
            display_name="常客",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["0.5", "1"],
            profile_facts=[
                "画像默认：95%打0.5，打1块会单独说。",
                "画像默认：95%是一个人来，带人会单独说。",
            ],
            smoke_preference="no_smoke",
            response_score=0.9,
            notes="内部备注：这段不进入模型上下文。",
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
            "planned_start_at": future_planned_start_at("19:00"),
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
    assert "95%打0.5" in "\n".join(first_prompt_payload["sender_profile"]["profile_facts"])
    assert "内部备注" not in json.dumps(first_prompt_payload["sender_profile"], ensure_ascii=False)
    assert "打多大" not in result.final_reply
    assert "几个人" not in result.final_reply
    assert "0.5" not in result.final_reply
    assert "无烟" not in result.final_reply


def test_runtime_dotenv_loader_uses_defaults_without_overriding_existing_values(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# local secrets",
                "MAHJONG_LLM_MODEL=deepseek-v4-flash",
                "DEEPSEEK_API_KEY='from-file'",
                "EXISTING_VALUE=from-file",
                "INVALID_LINE",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EXISTING_VALUE", "from-env")
    monkeypatch.delenv("MAHJONG_LLM_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    loaded = load_dotenv_defaults(env_file)

    assert loaded == {"MAHJONG_LLM_MODEL": "deepseek-v4-flash", "DEEPSEEK_API_KEY": "from-file"}
    assert "INVALID_LINE" not in loaded
    assert os.environ["MAHJONG_LLM_MODEL"] == "deepseek-v4-flash"
    assert os.environ["DEEPSEEK_API_KEY"] == "from-file"
    assert os.environ["EXISTING_VALUE"] == "from-env"


def test_real_owner_chat_live_eval_skips_without_llm_env(monkeypatch, capsys, tmp_path: Path) -> None:
    for name in ("MAHJONG_LLM_API_KEY", "DEEPSEEK_API_KEY", "MAHJONG_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    script_path = ROOT / "scripts" / "run_real_owner_chat_live_eval.py"
    spec = importlib.util.spec_from_file_location("run_real_owner_chat_live_eval", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    report_path = tmp_path / "real_owner_chat_live_eval_report.json"

    exit_code = module.main(["--no-dotenv", "--report-path", str(report_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "skipped"
    assert "MAHJONG_LLM_MODEL" in payload["reason"]
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_payload["status"] == "skipped"
    assert "generated_at" in report_payload


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

    accept_store = module.SQLiteAgentStore(tmp_path / "accept.sqlite3")
    module.setup_accept_existing_offer(accept_store)
    accept_game = accept_store.active_games("owner_real_customer_chat")[0]
    assert accept_game.status == "forming"
    assert accept_game.seat_summary()["claimed_seats"] == 3
    assert accept_game.seat_summary()["remaining_seats"] == 1
    assert accept_game.requirement["user_visible_summary"] == "七点三缺一"


def test_real_owner_live_eval_accept_existing_offer_scenario_marks_ready() -> None:
    script_path = ROOT / "scripts" / "run_real_owner_chat_live_eval.py"
    spec = importlib.util.spec_from_file_location("run_real_owner_chat_live_eval_for_accept_test", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    scenario = next(
        item
        for item in module.live_eval_scenarios()
        if item.scenario_id == "accept_existing_offer_marks_game_ready"
    )
    assert scenario.required_tool_name_any == ["join_game", "record_candidate_reply"]
    assert set(scenario.expected_any_tool_result_paths) == {"join_game", "record_candidate_reply"}
    assert "search_current_games" in scenario.forbidden_tool_names
    assert "create_game" in scenario.forbidden_tool_names
    assert scenario.expected_active_game_status == "ready"
    assert scenario.expected_active_game_seat_summary == {"claimed_seats": 4, "remaining_seats": 0}
    assert scenario.expected_active_game_requirement["needed_seats"] == 0
    assert "帮你问问" in scenario.forbidden_reply_contains


def test_real_owner_live_eval_casual_chat_scenario_preserves_active_game(tmp_path: Path) -> None:
    script_path = ROOT / "scripts" / "run_real_owner_chat_live_eval.py"
    spec = importlib.util.spec_from_file_location("run_real_owner_chat_live_eval_for_casual_test", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    scenario_ids = {scenario.scenario_id for scenario in module.live_eval_scenarios()}
    assert "casual_chat_should_not_pollute_business_state" in scenario_ids

    casual_scenario = next(
        scenario
        for scenario in module.live_eval_scenarios()
        if scenario.scenario_id == "casual_chat_should_not_pollute_business_state"
    )
    assert "search_current_games" in casual_scenario.forbidden_tool_names
    assert ["费脑", "条件", "确实", "麻烦", "精细"] in casual_scenario.required_reply_any
    assert "这个先不聊" in casual_scenario.forbidden_reply_contains
    assert "打牌直接说" in casual_scenario.forbidden_reply_contains
    assert casual_scenario.expected_active_game_requirement["user_visible_summary"] == "七点三缺一"

    store = module.SQLiteAgentStore(tmp_path / "casual.sqlite3")
    module.setup_casual_chat_should_not_pollute_business_state(store)
    casual_game = store.active_games("owner_real_customer_chat")[0]
    assert casual_game.requirement["stake"] == "0.5"
    assert casual_game.requirement["smoke_preference"] == "no_smoke"
    assert casual_game.requirement["start_time"] == "19:00"
    assert casual_game.requirement["user_visible_summary"] == "七点三缺一"


def test_real_owner_live_eval_duration_limit_scenario_requires_task_memory(tmp_path: Path) -> None:
    script_path = ROOT / "scripts" / "run_real_owner_chat_live_eval.py"
    spec = importlib.util.spec_from_file_location("run_real_owner_chat_live_eval_for_duration_limit_test", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    scenario = next(
        item
        for item in module.live_eval_scenarios()
        if item.scenario_id == "duration_limit_update_should_persist"
    )
    assert scenario.required_tool_names == ["record_user_memory"]
    assert "收到" in scenario.required_reply_any[0]
    assert "record_candidate_reply" in scenario.forbidden_tool_names
    assert "search_current_games" in scenario.forbidden_tool_names
    assert "create_game" in scenario.forbidden_tool_names
    assert scenario.expected_checkpoint_contains == []
    assert scenario.expected_task_memory_contains == [
        {
            "conversation_id": "owner_real_customer_chat",
            "customer_id": "owner_real_customer",
            "scope": "current_task",
            "field": "max_duration_hours",
            "value": 4,
            "status": "active",
        }
    ]

    store = module.SQLiteAgentStore(tmp_path / "duration_limit.sqlite3")
    module.setup_duration_limit_update(store)
    active_game = store.active_games("owner_real_customer_chat")[0]
    assert active_game.requirement["user_visible_summary"] == "还没有，还差俩"
    assert active_game.requirement["duration_hours"] is None

    store.record_task_memory(
        conversation_id="owner_real_customer_chat",
        customer_id="owner_real_customer",
        memory_type="preference",
        field="max_duration_hours",
        value=4,
        evidence="七点我也ok，但是我只能打四个小时",
        confidence=1.0,
        risk_level="low",
        scope="current_task",
        trace_id="trace_duration_limit_test",
    )
    memories = store.task_memory_context("owner_real_customer_chat", "owner_real_customer")
    assert any(
        memory["field"] == "max_duration_hours"
        and memory["value"] == 4
        and memory["scope"] == "current_task"
        for memory in memories
    )


def test_real_owner_transcript_replay_context_keeps_business_state_after_chitchat(tmp_path: Path) -> None:
    script_path = ROOT / "scripts" / "run_real_owner_chat_live_eval.py"
    spec = importlib.util.spec_from_file_location("run_real_owner_chat_live_eval_for_replay_context_test", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    store = module.SQLiteAgentStore(tmp_path / "owner_replay.sqlite3")
    module.setup_resume_status_after_casual_chat(store)
    client = StaticAgentClient(
        [
            action_json(
                objective_status="completed",
                reasoning_summary="长闲聊后用户问当前局况，直接使用 active_game_visible_summaries 的老板式摘要。",
                reply_to_user="还没有，还差俩",
            )
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorder())

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="owner_real_customer_chat",
            sender_id="owner_real_customer",
            sender_name="常客",
            text="所以现在有人了吗",
            message_id="msg_owner_real_transcript_replay_resume_status",
        ),
        trace_id="trace_owner_real_transcript_replay_resume_status",
    )

    assert result.final_reply == "还没有，还差俩"
    assert "打多大" not in result.final_reply
    assert "几个人" not in result.final_reply
    assert "要组" not in result.final_reply
    assert "AI" not in result.final_reply

    prompt_payload = json.loads(client.calls[0]["messages"][1]["content"])
    recent_conversation_text = json.dumps(prompt_payload["recent_conversation"], ensure_ascii=False)
    assert "七点三缺一，可以不" in recent_conversation_text
    assert "也可以" in recent_conversation_text
    assert "5小时也不行吗" in recent_conversation_text
    assert "不行啊" in recent_conversation_text
    assert "好奇的问一下" in recent_conversation_text
    assert "最重要还是看你能不能帮我组上局" in recent_conversation_text

    active_games = prompt_payload["active_games"]
    assert len(active_games) == 1
    assert active_games[0]["requirement"]["stake"] == "0.5"
    assert active_games[0]["requirement"]["smoke_preference"] == "no_smoke"
    assert active_games[0]["requirement"]["user_visible_summary"] == "还没有，还差俩"
    assert active_games[0]["seat_summary"]["claimed_seats"] == 2
    assert active_games[0]["seat_summary"]["remaining_seats"] == 2

    visible_summaries = prompt_payload["active_game_visible_summaries"]
    assert len(visible_summaries) == 1
    assert visible_summaries[0]["status_query_reply_contract"]["preferred_reply_text"] == "还没有，还差俩"
    assert "不要只根据 seat_summary 重新概括" in visible_summaries[0]["status_query_reply_contract"]["rule"]
    assert "95% 情况打 0.5" in "\n".join(prompt_payload["sender_profile"]["profile_facts"])


def test_real_owner_live_eval_forbids_customer_service_tone_globally() -> None:
    script_path = ROOT / "scripts" / "run_real_owner_chat_live_eval.py"
    spec = importlib.util.spec_from_file_location("run_real_owner_chat_live_eval_for_style_contract", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    customer_service_fragments = set(module.CUSTOMER_SERVICE_FORBIDDEN_REPLY_FRAGMENTS)
    assert set(FORBIDDEN_CUSTOMER_SERVICE_PHRASES) <= customer_service_fragments
    assert {"为您", "请耐心等待", "是否加入", "要一起吗"} <= customer_service_fragments

    for scenario in module.live_eval_scenarios():
        forbidden = set(scenario.forbidden_reply_contains)
        missing = sorted(customer_service_fragments - forbidden)
        assert not missing, f"{scenario.scenario_id} missing customer-service tone forbids: {missing}"


def test_real_owner_live_eval_required_reply_match_is_case_insensitive_for_short_english_ack() -> None:
    script_path = ROOT / "scripts" / "run_real_owner_chat_live_eval.py"
    spec = importlib.util.spec_from_file_location("run_real_owner_chat_live_eval_for_reply_match_test", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.reply_contains_required_fragment("OK", "ok")
    assert module.reply_contains_required_fragment("Okk", "okk")
    assert module.reply_contains_required_fragment("好的", "好")


def test_real_owner_live_eval_forbids_implementation_details_globally() -> None:
    script_path = ROOT / "scripts" / "run_real_owner_chat_live_eval.py"
    spec = importlib.util.spec_from_file_location("run_real_owner_chat_live_eval_for_boundary_contract", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    implementation_fragments = set(module.IMPLEMENTATION_DETAIL_FORBIDDEN_REPLY_FRAGMENTS)
    assert set(FORBIDDEN_IMPLEMENTATION_IDENTITY_TERMS) <= implementation_fragments
    assert set(FORBIDDEN_INTERNAL_PROCESS_TERMS) <= implementation_fragments
    assert {"AI", "ai", "Agent", "agent", "机器人", "智能助手", "工具", "后台", "trace"} <= implementation_fragments

    for scenario in module.live_eval_scenarios():
        forbidden = set(scenario.forbidden_reply_contains)
        missing = sorted(implementation_fragments - forbidden)
        assert not missing, f"{scenario.scenario_id} missing implementation detail forbids: {missing}"


def test_real_owner_live_eval_validate_result_enforces_customer_visible_contract_and_short_reply(tmp_path: Path) -> None:
    script_path = ROOT / "scripts" / "run_real_owner_chat_live_eval.py"
    spec = importlib.util.spec_from_file_location("run_real_owner_chat_live_eval_for_generic_reply_gate", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    scenario = module.LiveEvalScenario(
        scenario_id="generic_reply_gate",
        name="通用客户可见回复门禁",
        setup=lambda store: None,
        message=UserMessage(
            conversation_id="owner_real_customer_chat",
            sender_id="owner_real_customer",
            sender_name="常客",
            text="帮我组一个",
            message_id="msg_generic_reply_gate",
        ),
        max_reply_chars=12,
    )
    result = SimpleNamespace(
        final_reply="我是智能助手，已经生成草稿，等老板审批后发送。",
        tool_results=[],
    )

    validation = module.validate_result(
        result,
        scenario,
        trace_events=[],
        store=module.SQLiteAgentStore(tmp_path / "generic_reply_gate.sqlite3"),
    )

    checks = {item["name"]: item for item in validation["checks"]}
    assert checks["final_reply_should_pass_customer_visible_contract"]["passed"] is False
    assert checks["final_reply_should_be_short_owner_wechat_style"]["passed"] is False
    assert any("智能助手" in item for item in checks["final_reply_should_pass_customer_visible_contract"]["violations"])
    assert any("草稿" in item for item in checks["final_reply_should_pass_customer_visible_contract"]["violations"])
    assert any("审批" in item for item in checks["final_reply_should_pass_customer_visible_contract"]["violations"])
    assert validation["passed"] is False


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
