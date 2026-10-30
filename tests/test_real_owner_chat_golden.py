from __future__ import annotations

import datetime as dt
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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
from mahjong_agent_runtime.models import DEFAULT_TZ


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


def resolved_hard_snapshots():
    """测试与数据校验器共用唯一修订解析器，不跳过原始案例或直接改写历史快照。"""
    path = ROOT / "scripts" / "validate_real_owner_chat_golden.py"
    spec = importlib.util.spec_from_file_location("owner_hard_revision_test_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.resolve_hard_assertion_snapshots(read_records())


def test_real_owner_chat_golden_transcript_is_structured() -> None:
    records = read_records()

    assert len(records) == 11
    assert {record["id"] for record in records} == {
        "owner_chat_ai_chitchat_resume_20260704_001",
        "owner_chat_profile_defaults_and_privacy_20260704_001",
        "owner_chat_flexible_time_and_help_offer_20261029_001",
        "owner_chat_live_semantic_regression_20261029_001",
        "owner_chat_inventory_scope_regression_20261029_001",
        "owner_chat_party_capacity_regression_20261029_001",
        "owner_chat_proactive_help_contract_review_20261029_001",
        "owner_chat_casual_semantic_regression_20261030_001",
        "owner_chat_casual_hard_assertion_review_20261030_001",
        "owner_chat_user_query_strategy_revision_20261030_001",
        "owner_chat_profile_empty_response_revision_20261031_001",
    }
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
            sent_at=dt.datetime(2026, 7, 4, 15, 42, tzinfo=DEFAULT_TZ),
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


def _live_eval_for_inquiry_tests():
    spec = importlib.util.spec_from_file_location("run_real_owner_live_eval_inquiry_test", ROOT / "scripts" / "run_real_owner_chat_live_eval.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_real_owner_live_eval_flexible_time_scenarios_preserve_history_and_service_boundary(tmp_path: Path) -> None:
    module = _live_eval_for_inquiry_tests()
    scenarios = {item.scenario_id: item for item in module.live_eval_scenarios()}
    matched = scenarios["flexible_time_followup_matches_evening_game"]
    empty = scenarios["flexible_time_empty_pool_offers_help"]
    assert matched.message.text == empty.message.text == "1块的吧，时间都行的"
    for scenario in (matched, empty):
        assert scenario.excluded_search_filter_fields == []
        assert scenario.excluded_query_requirement_values["start_time_kind"] == ["scheduled", "asap_when_full", "下午", "afternoon"]
        assert scenario.excluded_query_requirement_values["time_preference"] == ["下午", "afternoon"]
        assert scenario.excluded_query_requirement_values["time_range"] == ["下午", "afternoon"]
        assert scenario.expected_status_any == ["waiting_user"]
        assert {"create_game", "search_customers", "create_invite_drafts"} <= set(scenario.forbidden_tool_names)
    assert empty.require_help_offer
    assert scenarios["complete_query_is_not_creation_consent"].require_help_offer
    assert not scenarios["inventory_only_does_not_offer_recruitment"].require_help_offer
    store = module.SQLiteAgentStore(tmp_path / "flexible-time.sqlite3")
    try:
        matched.setup(store)
        history = store.recent_turns(matched.message.conversation_id)
        assert [turn.content for turn in history] == ["下午有人打吗", "打多大的，几点方便？"]
        games = store.search_current_games({"game_type": "hangzhou_mahjong", "stake": "1"})
        assert len(games) == 1
        assert "七点" in games[0]["game"]["requirement"]["user_visible_summary"]
        assert dt.datetime.fromisoformat(games[0]["game"]["requirement"]["start_time"]).hour == 19
    finally:
        store._connection.close()


def test_real_owner_live_eval_service_offer_rejects_empty_promise_and_stale_time_filter(tmp_path: Path) -> None:
    from mahjong_agent_runtime.domains.tools.shared import current_game_search_reply_contract
    from mahjong_agent_runtime.models import ToolResult

    module = _live_eval_for_inquiry_tests()
    scenario = next(item for item in module.live_eval_scenarios() if item.scenario_id == "flexible_time_empty_pool_offers_help")
    store = module.SQLiteAgentStore(tmp_path / "help-offer.sqlite3")
    try:
        scenario.setup(store)
        requirement = {"game_type": "hangzhou_mahjong", "stake": "1"}
        tool = ToolResult(name="search_current_games", called=True, allowed=True, result={
            "requirement": requirement,
            "customer_reply_contract": current_game_search_reply_contract(requirement, []),
        })
        result = SimpleNamespace(final_reply="暂时没现成的，要我帮你问问有没有人吗？", status="waiting_user", tool_results=[tool])
        semantic_pass = {"passed": True, "error": None}
        semantic_fail = {"passed": False, "error": None, "reason": "没有提供征询式找人服务"}
        assert not module.validate_result(result, scenario, [], store)["passed"]
        assert module.validate_result(result, scenario, [], store, semantic_pass)["passed"]
        for reply in ("我帮你问问。", "现在没现成的。", "要不要换个时间？", "这边暂时没处理好。"):
            result.final_reply = reply
            assert not module.validate_result(result, scenario, [], store, semantic_fail)["passed"]
        for reply in ("要不要给你找一下人？", "要我替你问一圈吗？", "要帮你组一个不？"):
            result.final_reply = reply
            assert module.validate_result(result, scenario, [], store, semantic_pass)["passed"]
        result.final_reply = "要不要给你找一下人？"
        for field in ("time_preference", "time_range", "start_time_kind"):
            stale_values = ["下午", "afternoon", " Afternoon "]
            if field == "start_time_kind":
                stale_values.extend(["scheduled", "asap_when_full", " Scheduled "])
            for stale in stale_values:
                tool.result["requirement"][field] = stale
                tool.result["customer_reply_contract"] = current_game_search_reply_contract(tool.result["requirement"], [])
                checks = module.validate_result(result, scenario, [], store, semantic_pass)["checks"]
                assert not next(item for item in checks if item["name"] == f"relaxed_query_should_not_keep_excluded_value_{field}")["passed"]
            for unconstrained in (None, "不限", "any", "时间都行", "flexible"):
                tool.result["requirement"][field] = unconstrained
                tool.result["customer_reply_contract"] = current_game_search_reply_contract(tool.result["requirement"], [])
                assert module.validate_result(result, scenario, [], store, semantic_pass)["passed"]
            tool.result["requirement"].pop(field)
    finally:
        store._connection.close()


def test_real_owner_live_eval_flexible_time_matches_scheduled_evening_without_forcing_query_shape(tmp_path: Path) -> None:
    """用真实局池证明不限时间可以保留字段，同时拒绝擅自固定或恢复旧下午限制。"""
    from mahjong_agent_runtime.domains.tools.shared import current_game_search_reply_contract
    from mahjong_agent_runtime.models import ToolResult

    module = _live_eval_for_inquiry_tests()
    scenario = next(item for item in module.live_eval_scenarios() if item.scenario_id == "flexible_time_followup_matches_evening_game")
    store = module.SQLiteAgentStore(tmp_path / "flexible-evening.sqlite3")
    try:
        scenario.setup(store)
        requirement = {"game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "any", "start_time_kind": "flexible"}
        matches = store.search_current_games(requirement)
        assert len(matches) == 1
        assert matches[0]["game"]["requirement"]["start_time_kind"] == "scheduled"
        assert dt.datetime.fromisoformat(matches[0]["game"]["requirement"]["start_time"]).hour == 19
        tool = ToolResult(name="search_current_games", called=True, allowed=True, result={
            "requirement": requirement,
            "customer_reply_contract": current_game_search_reply_contract(requirement, matches),
        })
        result = SimpleNamespace(final_reply="晚上七点，1块，三缺一，可以不？", status="waiting_user", tool_results=[tool])
        semantic_pass = {"passed": True, "error": None}
        assert "start_time_kind" in tool.result["customer_reply_contract"]["evidence_scope"]["applied_filter_fields"]
        assert module.validate_result(result, scenario, [], store, semantic_pass)["passed"]
        for narrowed in ("scheduled", "asap_when_full", "下午", "afternoon"):
            requirement["start_time_kind"] = narrowed
            tool.result["customer_reply_contract"] = current_game_search_reply_contract(requirement, matches)
            checks = module.validate_result(result, scenario, [], store, semantic_pass)["checks"]
            assert not next(item for item in checks if item["name"] == "relaxed_query_should_not_keep_excluded_value_start_time_kind")["passed"]
    finally:
        store._connection.close()


def test_flexible_time_query_score_matches_omitted_time_for_canonical_game_starts() -> None:
    """限定现有 scheduled/asap 领域取值，验证 flexible 不改变实际排序分数和理由。"""
    from mahjong_agent_runtime.domains.customer_domain import score_requirement

    requirement = {"game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "any"}
    for target_kind in ("scheduled", "asap_when_full"):
        target = {**requirement, "start_time_kind": target_kind}
        omitted_score = score_requirement(requirement, target)
        flexible_score = score_requirement({**requirement, "start_time_kind": "flexible"}, target)
        assert flexible_score == omitted_score
        assert flexible_score == (70, ["game_type_matched", "stake_matched", "smoke_preference_matched"])


def test_live_reply_semantics_load_existing_golden_cases_and_fail_without_judgement() -> None:
    module = _live_eval_for_inquiry_tests()
    cases = module.golden_reply_cases()
    assert len({case["case_id"] for case in cases}) == 36
    assert all(case["scenario_id"] for case in cases)
    scenarios = {scenario.scenario_id: scenario for scenario in module.live_eval_scenarios()}
    original = module.reply_semantic_expectation(scenarios["profile_default_matched_game"])
    assert {source["case_id"] for source in original["golden_sources"]} == {
        "initial_request_uses_profile_defaults_and_searches_pool", "human_likeness_reply_should_be_short_and_decision_focused",
    }
    offer = module.reply_semantic_expectation(scenarios["flexible_time_empty_pool_offers_help"])
    assert offer["reply_reference"]
    assert any(item["id"] == "offer_finding_players" for item in offer["required_meaning"])
    assert offer["golden_sources"]
    ids = [item["id"] for key in ("required_meaning", "forbidden_meaning") for item in offer[key]]
    assert len(ids) == len(set(ids))
    style_id = "human_likeness_reply_should_be_short_and_decision_focused.bad_examples"
    assert any(item["id"] == style_id for item in original["forbidden_meaning"])
    assert not any(item["id"].endswith("good_examples") for item in original["required_meaning"])
    duration = "group_duration_mismatch_records_exit"
    partial = module.golden_reply_coverage({"duration_rejection_not_chitchat"})
    assert duration in partial["partially_executed_case_ids"] and duration not in partial["executed_case_ids"]
    complete = module.golden_reply_coverage({"duration_rejection_not_chitchat", "duration_limit_update_should_persist"})
    assert duration in complete["executed_case_ids"]
    refusal_expectation = module.reply_semantic_expectation(scenarios["duration_rejection_not_chitchat"])
    assert not any("max_duration_hours=4" in item["description"] for item in refusal_expectation["required_meaning"])
    four_hour = module.reply_semantic_expectation(scenarios["duration_limit_update_should_persist"])
    assert any("max_duration_hours=4" in item["description"] for item in four_hour["required_meaning"])


def test_synthetic_live_semantics_preserve_original_bytes_and_match_every_fixture_input() -> None:
    """新增语义资产只追加合成来源，所有输入和引用必须与既有 live 夹具一致。"""
    raw_lines = DATASET_PATH.read_bytes().splitlines(keepends=True)
    assert hashlib.sha256(b"".join(raw_lines[:3])).hexdigest() == "e14b4efd86129662f0d38dc526b342c204861ec2ef83d83506f60a8984c3729d"
    record = read_records()[3]
    assert record["kind"] == "owner_chat_live_semantic_regression"
    assert record["source"]["kind"] == "synthetic_regression_fixture"
    assert record["source"]["not_original_human_transcript"] is True
    assert not {"draft_status", "coverage_draft", "suggested_patch_plan", "messages", "parent_id"} & record.keys()
    module = _live_eval_for_inquiry_tests()
    scenarios = {scene.scenario_id: scene for scene in module.live_eval_scenarios()}
    all_cases = module.golden_reply_cases()
    old_mapped = {case["scenario_id"] for case in all_cases if case["record_id"] != record["id"]}
    assert len(record["eval_cases"]) == 19
    assert {case["live_scenario_id"] for case in record["eval_cases"]} == set(scenarios) - old_mapped
    for case in record["eval_cases"]:
        scene = scenarios[case["live_scenario_id"]]
        message = dataclasses.asdict(scene.message)
        assert case["input"]["text"] == message["text"]
        assert case["input"].get("quoted_message") == message["quoted_message"]
        for field in ("sender_id", "sender_name", "conversation_id", "message_id"):
            assert case["input"][field] == message[field]
        assert case["context"]["fixture_ref"] == f"scripts/run_real_owner_chat_live_eval.py::{scene.setup.__name__}"
        assert case["context"]["known"] and case["context"]["unknown"]
        for field, value in resolved_hard_snapshots()[(case["id"], case["live_scenario_id"])].items():
            assert getattr(scene, field) == value
        assert case["expected"]["reply_reference_is_example"] is True
    assert len(scenarios) == 34
    assert {case["scenario_id"] for case in all_cases} == set(scenarios)
    assert len({case["case_id"] for case in all_cases}) == 36
    for scene in scenarios.values():
        expected = module.reply_semantic_expectation(scene)
        assert expected["golden_sources"] and expected["required_meaning"]
        assert all(item["id"] != "current_request" for item in expected["required_meaning"])
        assert {"fabricated_action", "privacy"} <= {item["id"] for item in expected["forbidden_meaning"]}


def test_live_runner_rejects_missing_explicit_case_before_any_model_call(monkeypatch, tmp_path: Path) -> None:
    """即使请求部分场景，缺少另一个场景的来源也必须在模型调用前失败。"""
    module = _live_eval_for_inquiry_tests()
    cases = module.golden_reply_cases()
    monkeypatch.setattr(module, "golden_reply_cases", lambda: [case for case in cases if case["scenario_id"] != "custom_profile_game_preserved"])
    scene = next(scene for scene in module.live_eval_scenarios() if scene.scenario_id == "custom_profile_game_preserved")
    with pytest.raises(ValueError, match="lacks explicit golden semantic expectation"):
        module.reply_semantic_expectation(scene)
    def forbidden_client_creation():
        raise AssertionError("缺少评测标准时不应构建真实模型客户端")
    monkeypatch.setattr(module.OpenAICompatibleAgentClient, "from_env", forbidden_client_creation)
    report_path = tmp_path / "missing-semantic-source.json"
    assert module.main(["--no-dotenv", "--scenario", "profile_default_matched_game", "--report-path", str(report_path)]) == 1
    report = json.loads(report_path.read_text())
    assert report["status"] == "failed" and "custom_profile_game_preserved" in report["error"]


def test_live_reply_semantic_judge_error_cannot_pass_structural_success(tmp_path: Path) -> None:
    module = _live_eval_for_inquiry_tests()
    scenario = module.LiveEvalScenario(
        scenario_id="judge_unavailable", name="语义评测缺失不能报业务通过", setup=lambda store: None,
        message=UserMessage(conversation_id="judge-test", sender_id="c", sender_name="客户", text="有人吗"),
    )
    result = SimpleNamespace(final_reply="要帮你组一个不？", tool_results=[], status="completed")
    store = module.SQLiteAgentStore(tmp_path / "judge-unavailable.sqlite3")
    try:
        for assessment in (None, {"passed": False, "error": "invalid judge JSON"}, {"passed": True, "error": "timeout"}):
            checks = module.validate_result(result, scenario, [], store, assessment)["checks"]
            assert not next(check for check in checks if check["name"] == "reply_should_match_expected_meaning")["passed"]
        for status in ("needs_help", "needs_human", "failed", "needs_tool"):
            result.status = status
            assert not module.validate_result(result, scenario, [], store, {"passed": True, "error": None})["passed"]
    finally:
        store._connection.close()


def test_live_reply_explicit_case_accepts_direct_missing_information_questions(tmp_path: Path) -> None:
    """直接追问由对应夹具的明确标准验收，不依赖通用规则猜测场景义务。"""
    module = _live_eval_for_inquiry_tests()
    scenarios = {scenario.scenario_id: scenario for scenario in module.live_eval_scenarios()}
    scenario = scenarios["confirmation_does_not_answer_missing_party"]
    expectation = module.reply_semantic_expectation(scenario)
    case = next(case for case in module.golden_reply_cases() if case["scenario_id"] == scenario.scenario_id)
    for item in case["expected"]["required_meaning"]:
        assert {"id": f"{case['case_id']}.{item['id']}", "description": item["description"]} in expectation["required_meaning"]
    assert not any(item["id"] == "current_request" for item in expectation["required_meaning"])
    assert scenario.expected_status_any == ["waiting_user"]
    assert scenario.expected_active_game_count == 0
    assert scenario.forbidden_tool_names == ["create_game", "search_customers", "create_invite_drafts"]
    assert scenario.required_reply_any == [
        ["几个人", "几位", "多少人", "几人", "一个人"], ["几点", "什么时候", "时间", "人齐开"],
    ]
    assert {"杭麻", "川麻", "什么玩法"} <= set(scenario.forbidden_reply_contains)
    store = module.SQLiteAgentStore(tmp_path / "direct-clarification.sqlite3")
    try:
        scenario.setup(store)
        assert [turn.content for turn in store.recent_turns(scenario.message.conversation_id)] == [
            "帮我组个0.5无烟的", "你这边几个人？什么时候打？",
        ]
        assert store.customers[scenario.message.sender_id].preferred_games == ["national_standard_mahjong"]
        result = SimpleNamespace(final_reply="你那边几个人？几点打，还是人齐开？", status="waiting_user", tool_results=[])
        assert module.validate_result(result, scenario, [], store, {"passed": True, "error": None})["passed"]
        result.status = "completed"
        assert not module.validate_result(result, scenario, [], store, {"passed": True, "error": None})["passed"]
    finally:
        store._connection.close()
    for scenario_id in ("profile_context_infers_base_and_cap", "custom_rules_infer_base_and_cap", "explicit_wide_range_is_not_a_cap"):
        inferred = module.reply_semantic_expectation(scenarios[scenario_id])
        case = next(case for case in module.golden_reply_cases() if case["scenario_id"] == scenario_id)
        assert inferred["reply_reference"] == case["expected"]["reply_reference"]
        assert any(item["id"].startswith(case["case_id"] + ".") for item in inferred["required_meaning"])
        assert {"fabricated_action", "privacy"} <= {item["id"] for item in inferred["forbidden_meaning"]}


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
    # 回避闲聊与定向追问都改用明确语义条目，不能再靠裸词判断开放语言。
    assert "打多大" not in casual_scenario.forbidden_reply_contains
    expected = module.reply_semantic_expectation(casual_scenario)
    assert any(item["id"].endswith(".unsolicited_resume_or_evasion") for item in expected["forbidden_meaning"])
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


def test_real_owner_live_eval_required_reply_match_is_case_insensitive_for_short_english_ack(monkeypatch) -> None:
    script_path = ROOT / "scripts" / "run_real_owner_chat_live_eval.py"
    spec = importlib.util.spec_from_file_location("run_real_owner_chat_live_eval_for_reply_match_test", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    scenario = module.LiveEvalScenario(
        scenario_id="acknowledgement", name="简短确认收到", setup=lambda store: None,
        message=UserMessage(conversation_id="ack", sender_id="c", sender_name="客户", text="知道了"),
        required_reply_any=[["ok", "okk", "好"]],
    )
    monkeypatch.setattr(module, "golden_reply_cases", lambda: [{
        "record_id": "ack_fixture", "case_id": "explicit_ack", "scenario_id": "acknowledgement",
        "description": "收到信息的简短确认", "expected": {
            "reply_reference": "好。", "required_meaning": [{"id": "ack", "description": "简短确认已收到。"}],
            "forbidden_meaning": [{"id": "no_new_action", "description": "不能凭收到信息编造新业务动作。"}],
        },
    }])
    expected = module.reply_semantic_expectation(scenario)
    assert any(item["id"] == "scenario_reply_alternatives_0" for item in expected["required_meaning"])
    for reply in ("OK", "Okk", "好的"):
        result = SimpleNamespace(final_reply=reply, tool_results=[], status="completed")
        assert module.validate_result(result, scenario, [], InMemoryAgentStore(), {"passed": True, "error": None})["passed"]
        assert not module.validate_result(result, scenario, [], InMemoryAgentStore(), {"passed": False, "error": None})["passed"]


def test_real_owner_live_eval_smoke_aliases_preserve_constraint_strength() -> None:
    script_path = ROOT / "scripts" / "run_real_owner_chat_live_eval.py"
    spec = importlib.util.spec_from_file_location("run_real_owner_chat_live_eval_smoke_alias_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    path = "result.requirement.smoke_preference"
    assert module.eval_values_equal("无烟", "no_smoking", field_path=path)
    assert module.eval_values_equal("no_smoke", "no_smoking", field_path=path)
    for actual in ("烟", "不限", "未知", "", None):
        assert not module.eval_values_equal(actual, "no_smoking", field_path=path)
    assert not module.eval_values_equal("", "any", field_path=path)
    assert not module.eval_values_equal("无烟", "no_smoking", field_path="result.requirement.game_type")
    assert not module.eval_values_equal({"min": 1, "max": 2}, {"min": 1, "max": 3}, field_path="stake_range")


def test_live_eval_requirement_equivalence_preserves_variant_and_seat_invariants() -> None:
    module = _live_eval_for_inquiry_tests()
    assert module.value_at_path({"result": {"requirement": {"variant": "caiqiao"}}}, "result.requirement.game_variant") == "caiqiao"
    assert module.value_at_path({"result": {"requirement": {"variant": "other"}}}, "result.requirement.game_variant") != "caiqiao"
    assert module.eval_requirement_value({"variant": "other", "game_variant": "caiqiao"}, "game_variant") is None
    assert module.eval_requirement_value({"variant": "caiqiao", "game_variant": "other"}, "variant") is None
    assert module.eval_requirement_value({"needed_seats": 1}, "known_player_count") == 3
    assert module.eval_requirement_value({"needed_seats": 1}, "seat_format") == "371"
    assert module.eval_requirement_value({"known_player_count": 3}, "needed_seats") == 1
    assert module.eval_requirement_value({"known_player_count": 2, "needed_seats": 1}, "known_player_count") == 2
    assert module.eval_requirement_value({"known_player_count": 2, "needed_seats": 1}, "seat_format") is None
    for malformed in (True, "1", 0, 4):
        assert module.eval_requirement_value({"needed_seats": malformed}, "known_player_count") is None
    for capacity in (3, 5, None, "4", True):
        assert module.eval_requirement_value({"needed_seats": 1, "seats_total": capacity}, "known_player_count") is None
        assert module.eval_requirement_value({"needed_seats": 1, "seats_total": capacity}, "seat_format") is None
    assert module.eval_requirement_value({"needed_seats": 1, "seats_total": 4}, "seat_format") == "371"
    assert module.eval_requirement_value({}, "seat_format") is None
    assert module.value_at_path({"other": {"needed_seats": 1}}, "other.known_player_count") is None


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
        status="completed",
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


@pytest.fixture(params=["memory", "sqlite"])
def runtime_contract_recovery_case(request, tmp_path):
    """走真实Runtime拒绝/回灌/工具链，模型输出仅用明确标记的确定性替身。"""
    module = _live_eval_for_inquiry_tests()
    store = InMemoryAgentStore() if request.param == "memory" else module.SQLiteAgentStore(tmp_path / "safe-recovery.sqlite3")
    trace = InMemoryTraceRecorder()
    message = UserMessage(conversation_id="contract-recovery", sender_id="customer", sender_name="客户",
                          text="只看看当前登记局概况", message_id="contract-recovery-input")
    call = {"name": "search_current_games", "arguments": {"requirement": {}, "limit": 5}, "reason": "读取登记局概况"}
    client = StaticAgentClient(outputs=[
        action_json(objective_status="needs_tool", reply_to_user="好，我查下。", tool_calls=[call]),
        action_json(objective_status="needs_tool", tool_calls=[call]),
        action_json(objective_status="completed", reply_to_user="现在没有登记中的局。"),
    ])
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace,
        token_budget=module.TokenBudget(max_tokens_per_call=64000, max_calls_per_turn=8))
    result = runtime.handle_user_message(message, trace_id="trace-contract-recovery")
    scenario = module.LiveEvalScenario(scenario_id="contract-recovery", name="合同安全恢复机制", message=message,
        setup=lambda store: None, required_tool_names=["search_current_games"], expected_active_game_count=0,
        expected_status_any=["completed"])
    yield module, store, result, trace.get_trace(result.trace_id), scenario
    if isinstance(store, module.SQLiteAgentStore):
        store._connection.close()


def test_runtime_contract_rejection_is_quality_recovery_only_after_real_feedback(runtime_contract_recovery_case):
    module, store, result, events, scenario = runtime_contract_recovery_case
    validation = module.validate_result(result, scenario, events, store, {"passed": True, "error": None})
    assert validation["passed"] is True, validation["checks"]
    metrics = validation["recovery_metrics"]
    assert metrics["action_contract_error_count"] == metrics["safe_recovered_count"] == 1
    assert metrics["unrecovered_count"] == metrics["unsafe_rejected_step_count"] == metrics["unresolved_tool_error_count"] == 0
    assert metrics["quality_status"] == "passed"
    assert metrics["events"][0]["feedback_returned_to_model"] is True
    assert metrics["events"][0]["recovered_at_step"] == 2
    assert any(event.step == "action_contract_error" for event in events)
    assert all(event.content["step_index"] != 1 for event in events if event.step == "tool_called")


@pytest.mark.parametrize("fault", ["tool_called", "tool_result", "state_transition", "final_output", "feedback", "context", "tool_feedback", "legal_terminal"])
def test_successful_final_status_cannot_hide_missing_or_unsafe_recovery_evidence(runtime_contract_recovery_case, fault):
    module, _, result, original, _ = runtime_contract_recovery_case
    import copy
    from mahjong_agent_runtime.tracing import TraceEvent

    events = copy.deepcopy(original)
    error_index = next(index for index, event in enumerate(events) if event.step == "action_contract_error")
    inserted = {
        "tool_called": {"call": {"name": "create_game"}, "step_index": 1},
        "tool_result": {"name": "create_game", "called": True, "allowed": True, "error": None},
        "state_transition": {"entity_type": "game", "entity_id": "unexpected"},
        "final_output": {"reply": "错误动作已经办好了", "objective_status": "completed"},
    }
    if fault in inserted:
        events.insert(error_index + 1, TraceEvent(trace_id=result.trace_id, step=fault, content=inserted[fault]))
    elif fault == "feedback":
        events = [event for event in events if event.step != "contract_error_feedback"]
    elif fault == "context":
        events = [event for index, event in enumerate(events) if event.step != "context_built" or index < error_index]
    elif fault == "tool_feedback":
        events = [event for event in events if not (event.step == "tool_result" and event.content.get("name") == "search_current_games")]
    elif fault == "legal_terminal":
        terminal = next(event for event in reversed(events) if event.step == "action_proposed")
        terminal.content["objective_status"] = "needs_tool"
    metrics = module.assess_action_contract_recovery(events, runtime_status=result.status, final_reply=result.final_reply,
        business_checks_passed=True, semantic_passed=True)
    assert metrics["quality_status"] == "failed"
    assert metrics["safe_recovered_count"] == 0 and metrics["unrecovered_count"] == 1


@pytest.mark.parametrize("business_passed,semantic_passed", [(False, True), (True, False)])
def test_recovery_does_not_exempt_business_or_semantic_failure(runtime_contract_recovery_case, business_passed, semantic_passed):
    module, _, result, events, _ = runtime_contract_recovery_case
    metrics = module.assess_action_contract_recovery(events, runtime_status=result.status, final_reply=result.final_reply,
        business_checks_passed=business_passed, semantic_passed=semantic_passed)
    assert metrics["quality_status"] == "failed" and metrics["unrecovered_count"] == 1


def test_missing_contract_error_recovery_still_blocks_permission_or_tool_errors():
    module = _live_eval_for_inquiry_tests()
    events = [SimpleNamespace(step="tool_permission_checked", content={"tool_name": "create_game", "allowed": False, "error": "denied"})]
    metrics = module.assess_action_contract_recovery(events, runtime_status="completed", final_reply="好。",
        business_checks_passed=True, semantic_passed=True)
    assert metrics["action_contract_error_count"] == 0
    assert metrics["unresolved_tool_error_count"] == 1 and metrics["quality_status"] == "failed"


def test_recovery_summary_cannot_count_missing_case_metrics_as_clean(runtime_contract_recovery_case):
    module, _, result, events, _ = runtime_contract_recovery_case
    metrics = module.assess_action_contract_recovery(events, runtime_status=result.status, final_reply=result.final_reply,
        business_checks_passed=True, semantic_passed=True)
    summary = module.summarize_action_contract_recovery([{"recovery_metrics": metrics}])
    assert summary["action_contract_error_count"] == summary["safe_recovered_count"] == 1
    assert summary["cases_with_safe_recovery"] == 1 and summary["quality_status"] == "passed"
    assert module.summarize_action_contract_recovery([{}])["quality_status"] == "failed"
    assert module.summarize_action_contract_recovery([])["quality_status"] == "not_run"


@pytest.mark.parametrize("scenario_id,game_type", [
    ("single_profile_game_used_for_query", "national_standard_mahjong"),
    ("custom_profile_game_preserved", "club_custom_mahjong"),
])
def test_unique_profile_missing_stake_allows_clarification_but_keeps_query_contract(scenario_id, game_type) -> None:
    """未知底注澄清与真实查询均合法；已查结果、未查终态和越权写仍各自强验。"""
    from mahjong_agent_runtime.models import ToolResult

    module = _live_eval_for_inquiry_tests()
    scenario = next(scene for scene in module.live_eval_scenarios() if scene.scenario_id == scenario_id)
    assert scenario.required_tool_names == [] and scenario.expected_tool_result_paths == {}
    assert scenario.expected_status_any == []  # 保持原默认 completed/waiting_user 范围。
    assert scenario.optional_query_result_fields == {"search_current_games": {"result.requirement.game_type": game_type}}
    assert scenario.expected_status_without_tool == {"search_current_games": ["waiting_user"]}
    store = InMemoryAgentStore()
    scenario.setup(store)
    semantics = {"passed": True, "error": None}
    result = SimpleNamespace(final_reply="你打多大的？", status="waiting_user", tool_results=[])
    assert module.validate_result(result, scenario, [], store, semantics)["passed"]
    result.final_reply = "你打什么档位？"
    assert module.validate_result(result, scenario, [], store, semantics)["passed"]
    assert "打什么" not in scenario.forbidden_reply_contains
    assert {"打什么麻将", "打什么牌", "什么玩法"} <= set(scenario.forbidden_reply_contains)
    assert not module.validate_result(result, scenario, [], store, {"passed": False, "error": None})["passed"]
    result.status = "completed"
    checks = module.validate_result(result, scenario, [], store, semantics)["checks"]
    assert not next(check for check in checks if check["name"] == "without_search_current_games_status_should_be_expected")["passed"]
    query = ToolResult(name="search_current_games", called=True, allowed=True, result={"requirement": {"game_type": game_type}})
    result.tool_results = [query]
    result.final_reply = "现在没有登记中的局。"
    assert module.validate_result(result, scenario, [], store, semantics)["passed"]
    for wrong in ("hangzhou_mahjong", None):
        query.result["requirement"]["game_type"] = wrong
        checks = module.validate_result(result, scenario, [], store, semantics)["checks"]
        assert not next(check for check in checks if check["name"] == "optional_search_current_games_result_should_keep_result.requirement.game_type")["passed"]
    # 后一次改对不得掩盖先前已执行的错误查询；失败/被拒查询也不能充当正确证据。
    correct_query = ToolResult(name="search_current_games", called=True, allowed=True, result={"requirement": {"game_type": game_type}})
    result.tool_results.append(correct_query)
    assert not module.validate_result(result, scenario, [], store, semantics)["passed"]
    result.tool_results = [correct_query]
    for name in scenario.forbidden_tool_names:
        result.tool_results = [correct_query, ToolResult(name=name, called=True, allowed=True, result={})]
        assert not module.validate_result(result, scenario, [], store, semantics)["passed"]
    result.tool_results = [correct_query]
    correct_query.error = "query_failed"
    assert not module.validate_result(result, scenario, [], store, semantics)["passed"]


def test_optional_query_contract_does_not_relax_required_tool_result_paths() -> None:
    """可选字段通过不能替代另一条必查结果；既有严格合同继续因工具缺失而失败。"""
    module = _live_eval_for_inquiry_tests()
    scenario = module.LiveEvalScenario(
        scenario_id="required-query-not-optional", name="必查不被可选字段豁免",
        message=UserMessage(conversation_id="query", sender_id="customer", sender_name="客户", text="看看"),
        setup=lambda store: None, required_tool_names=["search_current_games"],
        expected_tool_result_paths={"search_current_games": {"result.requirement.stake": "1"}},
        optional_query_result_fields={"search_current_games": {"result.requirement.game_type": "national_standard_mahjong"}},
    )
    result = SimpleNamespace(final_reply="你打多大的？", status="waiting_user", tool_results=[])
    validation = module.validate_result(result, scenario, [], InMemoryAgentStore(), {"passed": True, "error": None})
    assert not validation["passed"]
    checks = {check["name"]: check for check in validation["checks"]}
    assert not checks["should_call_search_current_games"]["passed"]
    assert not checks["search_current_games_result_should_keep_result.requirement.stake"]["passed"]


@pytest.mark.parametrize("scenario_id,game_type", [
    ("single_profile_game_complete_query", "national_standard_mahjong"),
    ("custom_profile_game_complete_query", "club_custom_mahjong"),
])
def test_unique_profile_complete_conditions_must_query_then_offer_help(scenario_id, game_type) -> None:
    """信息充分时仍需真实查询，空池提议须待答；错误玩法、档位或越权动作不得通过。"""
    from mahjong_agent_runtime.models import ToolResult
    from mahjong_agent_runtime.domains.tools.shared import current_game_search_reply_contract

    module = _live_eval_for_inquiry_tests()
    scenario = next(scene for scene in module.live_eval_scenarios() if scene.scenario_id == scenario_id)
    store = InMemoryAgentStore()
    scenario.setup(store)
    assert store.customers["profile_customer"].preferred_games == [game_type]
    assert scenario.message.text == "我一个人，1块，烟不限，现在有局吗？"
    assert scenario.required_tool_names == ["search_current_games"]
    assert scenario.expected_status_any == ["waiting_user"] and scenario.require_help_offer
    result = SimpleNamespace(final_reply="现在没现成的，要我帮你问问有没有人吗？", status="waiting_user", tool_results=[])
    semantic_pass = {"passed": True, "error": None}
    assert not module.validate_result(result, scenario, [], store, semantic_pass)["passed"]
    requirement = {"game_type": game_type, "stake": "1", "smoke_preference": "any", "known_player_count": 1}
    assert store.search_current_games(requirement) == []
    query = ToolResult(name="search_current_games", called=True, allowed=True, result={
        "requirement": requirement, "matches": [], "alternatives": [],
        "customer_reply_contract": current_game_search_reply_contract(requirement, []),
    })
    result.tool_results = [query]
    assert module.validate_result(result, scenario, [], store, semantic_pass)["passed"]
    for field,wrong in (("game_type", "hangzhou_mahjong"), ("stake", "0.5")):
        previous = requirement[field]
        requirement[field] = wrong
        assert not module.validate_result(result, scenario, [], store, semantic_pass)["passed"]
        requirement[field] = previous
    # 烟不限不要求固定提交 any；省略不约束字段的等价查询形态同样合法。
    requirement.pop("smoke_preference")
    assert module.validate_result(result, scenario, [], store, semantic_pass)["passed"]
    result.status = "completed"
    assert not module.validate_result(result, scenario, [], store, semantic_pass)["passed"]
    result.status = "waiting_user"
    for name in scenario.forbidden_tool_names:
        result.tool_results = [query, ToolResult(name=name, called=True, allowed=True, result={})]
        assert not module.validate_result(result, scenario, [], store, semantic_pass)["passed"]
    result.tool_results = [query]
    assert not module.validate_result(result, scenario, [], store, {"passed": False, "error": None})["passed"]


def test_missing_stake_oracle_revision_preserves_input_semantics_and_original_hard_evidence() -> None:
    """只修正已证实冲突的硬验收，原输入、人工语义、参考句和旧强制查询要求均可回溯。"""
    original_meaning_hashes = {'single_profile_game_used_for_query': '359ffad53a913f86b45182ffefd01e4523e3171cb28e7466ae3b7067aebec681', 'custom_profile_game_preserved': 'b5a0c27da165ca6a05c5691525f9adce0d2706d98aba85a1220a6c6ffa1c74fd'}
    record = next(record for record in read_records() if record["kind"] == "owner_chat_live_semantic_regression")
    for case in record["eval_cases"]:
        scenario_id = case["live_scenario_id"]
        if scenario_id not in original_meaning_hashes:
            continue
        meaning = json.dumps({"input": case["input"], "expected": case["expected"]}, ensure_ascii=False, sort_keys=True)
        assert hashlib.sha256(meaning.encode()).hexdigest() == original_meaning_hashes[scenario_id]
        history = case["hard_assertions_history"]
        assert len(history) == 1
        assert history[0]["source_run_id"] == "20261029T113641255733Z_dkhyk6n4"
        previous = history[0]["previous_snapshot"]
        assert previous["required_tool_names"] == ["search_current_games"]
        assert previous["expected_tool_result_paths"] == case["existing_hard_assertions_snapshot"]["optional_query_result_fields"]
        assert previous["forbidden_tool_names"] == ["create_game", "create_invite_drafts"]
        assert "打什么" in previous["forbidden_reply_contains"]
        current_forbidden = case["existing_hard_assertions_snapshot"]["forbidden_reply_contains"]
        assert "打什么" not in current_forbidden
        assert {"打什么麻将", "打什么牌", "什么玩法"} <= set(current_forbidden)
        assert set(previous["forbidden_reply_contains"]) - {"打什么"} <= set(current_forbidden)
        assert history[0]["reason"]


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("scenario_id,requirement,reply", [
    ("inventory_all_games_ignores_personal_profile", {}, "杭麻1块三缺一，国标2块两缺二。"),
    ("inventory_explicit_game_overrides_profile", {"game_type": "hangzhou_mahjong"}, "杭麻1块三缺一。"),
])
def test_inventory_scope_reads_mixed_pool_without_personalizing_or_writing(backend, scenario_id, requirement, reply, tmp_path) -> None:
    """真实网关读取混合池：全店完整、定向不越界，排序无关且任何业务副作用都阻断。"""
    from copy import deepcopy
    from mahjong_agent_runtime import ToolCall, ToolGateway
    from mahjong_agent_runtime.models import ToolResult

    module = _live_eval_for_inquiry_tests()
    scenario = next(scene for scene in module.live_eval_scenarios() if scene.scenario_id == scenario_id)
    store = InMemoryAgentStore() if backend == "memory" else module.SQLiteAgentStore(tmp_path / "mixed-pool.sqlite3")
    try:
        scenario.setup(store)
        assert store.customers[scenario.message.sender_id].preferred_games == ["national_standard_mahjong"]
        assert len(store.games) == 2 and store.active_games(scenario.message.conversation_id) == []
        before = [game.to_dict() for game in store.games.values()]
        gateway = ToolGateway(store=store)
        def query(filters, limit=8):
            return gateway.execute(ToolCall(name="search_current_games", arguments={"requirement": filters, "limit": limit}),
                trace_id="inventory-scope-test", conversation_id=scenario.message.conversation_id,
                sender_id=scenario.message.sender_id, sender_name=scenario.message.sender_name, step_index=1)
        tool = query(requirement)
        assert tool.called and tool.allowed and not tool.error
        result = SimpleNamespace(final_reply=reply, status="completed", tool_results=[tool])
        semantics = {"passed": True, "error": None}
        assert module.validate_result(result, scenario, [], store, semantics)["passed"]
        assert [game.to_dict() for game in store.games.values()] == before
        assert not store.invite_drafts
        tool.result["matches"].reverse()
        assert module.validate_result(result, scenario, [], store, semantics)["passed"]
        original = deepcopy(tool.result["matches"])
        for corrupted in (original[:-1], original + [original[0]], []):
            tool.result["matches"] = corrupted
            assert not module.validate_result(result, scenario, [], store, semantics)["passed"]
        tool.result.pop("matches")
        tool.result["alternatives"] = original
        assert not module.validate_result(result, scenario, [], store, semantics)["passed"]
        tool.result["alternatives"] = []
        tool.result["matches"] = deepcopy(original)
        tool.result["matches"][0]["game"]["requirement"].pop("stake")
        assert not module.validate_result(result, scenario, [], store, semantics)["passed"]
        tool.result["matches"] = deepcopy(original)
        tool.result["matches"][0]["game"]["requirement"]["stake"] = "9"
        assert not module.validate_result(result, scenario, [], store, semantics)["passed"]
        if len(original) == 2:
            swapped = deepcopy(original)
            swapped[0]["game"]["requirement"]["stake"], swapped[1]["game"]["requirement"]["stake"] = (
                swapped[1]["game"]["requirement"]["stake"], swapped[0]["game"]["requirement"]["stake"])
            tool.result["matches"] = swapped
            assert not module.validate_result(result, scenario, [], store, semantics)["passed"]
            result.tool_results = [query({}, limit=1)]
            assert not module.validate_result(result, scenario, [], store, semantics)["passed"]
        tool.result["matches"] = original
        for filters in ({"game_type": "national_standard_mahjong"}, {} if requirement else {"game_type": "hangzhou_mahjong"}):
            result.tool_results = [query(filters)]
            assert not module.validate_result(result, scenario, [], store, semantics)["passed"]
        result.tool_results = [tool]
        for forbidden in scenario.forbidden_tool_names:
            result.tool_results = [tool, ToolResult(name=forbidden, called=True, allowed=True, result={})]
            assert not module.validate_result(result, scenario, [], store, semantics)["passed"]
        result.tool_results = [tool]
        assert not module.validate_result(result, scenario, [], store, {"passed": False, "error": None})["passed"]
    finally:
        if backend == "sqlite":
            store._connection.close()


def test_inventory_scope_sources_append_without_rewriting_existing_evidence() -> None:
    """新增假设样本只追加；前四行含既有修订历史的字节必须保持原样。"""
    raw_lines = DATASET_PATH.read_bytes().splitlines(keepends=True)
    assert hashlib.sha256(b"".join(raw_lines[:4])).hexdigest() == "3b9ca4aef4e3d0c4b4d3a9c41362be7ecb27dee073fb16aab513e955fa53954b"
    record = read_records()[4]
    assert record["id"] == "owner_chat_inventory_scope_regression_20261029_001"
    assert record["source"]["derivation_kind"] == "hypothetical_real_failure_derivation"
    module = _live_eval_for_inquiry_tests()
    scenes = {scene.scenario_id: scene for scene in module.live_eval_scenarios()}
    assert len(record["eval_cases"]) == 2
    for case in record["eval_cases"]:
        scene = scenes[case["live_scenario_id"]]
        assert case["source"]["derivation_kind"] == "hypothetical_real_failure_derivation"
        assert case["source"]["not_original_human_transcript"] is True
        assert case["context"]["fixture_ref"] == "scripts/run_real_owner_chat_live_eval.py::setup_mixed_inventory_profile"
        for field, value in case["input"].items():
            assert getattr(scene.message, field) == value
        for field, value in resolved_hard_snapshots()[(case["id"], case["live_scenario_id"])].items():
            assert getattr(scene, field) == value
        assert case["expected"]["reply_reference_is_example"] is True
    original = scenes["inventory_only_does_not_offer_recruitment"]
    assert original.setup.__name__ == "setup_single_game_profile"
    assert original.expected_tool_result_paths == {"search_current_games": {"result.customer_reply_contract.evidence_scope.kind": "unfiltered_inventory"}}


def test_semantic_evidence_preserves_before_and_after_sender_profile_without_private_data() -> None:
    """判定器明确区分画像前后态，只看当前客户安全投影，不获得他人或私有备注。"""
    module = _live_eval_for_inquiry_tests()
    scenario = next(scene for scene in module.live_eval_scenarios() if scene.scenario_id == "single_profile_game_used_for_query")
    store = InMemoryAgentStore()
    scenario.setup(store)
    initial = store.customers[scenario.message.sender_id].to_model_context()
    store.upsert_customer(CustomerProfile(customer_id=scenario.message.sender_id, display_name="常客",
        preferred_games=["national_standard_mahjong"], smoke_preference="no_smoke", private_remark="SENDER_PRIVATE_SENTINEL", notes="PRIVATE_NOTES_SENTINEL"))
    store.upsert_customer(CustomerProfile(customer_id="other-customer", display_name="OTHER_CUSTOMER_SENTINEL", private_remark="OTHER_PRIVATE_SENTINEL"))
    result = SimpleNamespace(final_reply="好。", tool_results=[], status="completed")
    evidence = module.reply_semantic_evidence(result, scenario, [SimpleNamespace(step="context_built", content={"sender_profile": initial})], store)
    assert evidence["known_profile"]["smoke_preference"] is None
    assert evidence["persisted_state"]["sender_profile"]["smoke_preference"] == "no_smoke"
    assert initial["smoke_preference"] is None
    serialized = json.dumps(evidence, ensure_ascii=False)
    for secret in ("SENDER_PRIVATE_SENTINEL", "PRIVATE_NOTES_SENTINEL", "OTHER_CUSTOMER_SENTINEL", "OTHER_PRIVATE_SENTINEL"):
        assert secret not in serialized


def test_semantic_tool_summary_preserves_only_declared_query_scope_contract() -> None:
    """定向空结果不可丢失范围；合同其余字段不借判定证据入口被额外透传。"""
    from mahjong_agent_runtime.models import ToolResult
    module = _live_eval_for_inquiry_tests()
    contract = {"evidence_scope": {"kind": "submitted_filters_only", "applied_filter_fields": ["game_type"]},
                "search_result_semantics": {"status": "no_actionable_match", "returned_game_count": 0},
                "private_unrelated_contract_field": "EXTRA_CONTRACT_SENTINEL"}
    tool = ToolResult(name="search_current_games", called=True, allowed=True,
                      result={"matches": [], "customer_reply_contract": contract})
    summary = module.summarize_tool_results([tool])[0]
    assert summary["customer_reply_contract"] == {key: contract[key] for key in ("evidence_scope", "search_result_semantics")}
    assert "EXTRA_CONTRACT_SENTINEL" not in json.dumps(summary)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_known_three_person_query_carries_party_and_only_returns_capacity_match(backend, tmp_path) -> None:
    """实际网关与容量策略区分己方三人及目标桌况，只返回能容三座的现成局。"""
    from copy import deepcopy
    from mahjong_agent_runtime import ToolCall, ToolGateway

    module = _live_eval_for_inquiry_tests()
    scenario = next(scene for scene in module.live_eval_scenarios() if scene.scenario_id == "known_three_person_party_matches_capacity")
    store = InMemoryAgentStore() if backend == "memory" else module.SQLiteAgentStore(tmp_path / "three-person-capacity.sqlite3")
    try:
        scenario.setup(store)
        requirement = {"game_type": "national_standard_mahjong", "stake": "1", "smoke_preference": "no_smoke",
                       "start_time_kind": "asap_when_full", "requesting_party": {"contact_id": "profile_customer", "seat_count": 3}}
        tool = ToolGateway(store=store).execute(ToolCall(name="search_current_games", arguments={"requirement": requirement}),
            trace_id="capacity-oracle-test", conversation_id=scenario.message.conversation_id,
            sender_id=scenario.message.sender_id, sender_name=scenario.message.sender_name, step_index=1)
        assert tool.called and tool.allowed and not tool.error
        assert len(tool.result["matches"]) == 1
        projection = tool.result["matches"][0]["join_projection"]
        assert projection["requested_seats"] == 3 and projection["would_fill_game"] is True and projection["would_overfill_game"] is False
        result = SimpleNamespace(final_reply="有个国标1块无烟的一缺三，你们三个人来刚好。", status="completed", tool_results=[tool])
        assert module.validate_result(result, scenario, [], store, {"passed": True, "error": None})["passed"]
        assert len(store.games) == 2 and store.active_games(scenario.message.conversation_id) == [] and not store.invite_drafts
        saved = deepcopy(tool.result["requirement"])
        tool.result["requirement"].pop("requesting_party")
        tool.result["requirement"].update(known_player_count=3, needed_seats=1, seat_format="371")
        assert not module.validate_result(result, scenario, [], store, {"passed": True, "error": None})["passed"]
        tool.result["requirement"] = saved
        projection["would_overfill_game"] = True
        assert not module.validate_result(result, scenario, [], store, {"passed": True, "error": None})["passed"]
    finally:
        if backend == "sqlite":
            store._connection.close()


def test_query_party_aliases_never_use_target_table_or_default_single_person() -> None:
    module = _live_eval_for_inquiry_tests()
    path = "result.requirement.requesting_party.seat_count"
    for requirement in ({"requesting_party": {"seat_count": 3}}, {"seat_count": 3}, {"party_size": 3}):
        assert module.value_at_path({"result": {"requirement": requirement}}, path) == 3
    for requirement in ({}, {"known_player_count": 3, "needed_seats": 1, "seat_format": "371"},
                        {"requesting_party": {"seat_count": 3}, "party_size": 1}, {"seat_count": True}):
        assert module.value_at_path({"result": {"requirement": requirement}}, path) is None


def test_capacity_sources_append_revision_chain_without_overwriting_original_snapshots() -> None:
    """前五行字节及旧错误要求保留，追加修订只改变当前硬验收，不新增语义案例。"""
    raw = DATASET_PATH.read_bytes().splitlines(keepends=True)
    assert hashlib.sha256(b"".join(raw[:5])).hexdigest() == "193afe4ed72d3158dcf97a7b15cd44e5bd3da67b46d271459d6d28c1f5e84929"
    record = read_records()[5]
    assert record["source"]["derivation_kind"] == "hypothetical_real_failure_derivation"
    assert len(record["eval_cases"]) == 1 and len(record["hard_assertion_revisions"]) == 1
    original = next(case for previous in read_records()[:5] for case in previous.get("eval_cases", [])
                    if case.get("live_scenario_id") == "explicit_stake_conflict_must_not_loop")
    revision = record["hard_assertion_revisions"][0]
    assert revision["previous_snapshot"] == original["existing_hard_assertions_snapshot"]
    assert revision["source_run_id"] == "20261029T120208253377Z_g48m0xvp"
    old_query = original["existing_hard_assertions_snapshot"]["expected_tool_result_paths"]["search_current_games"]
    assert old_query["result.requirement.known_player_count"] == 3 and old_query["result.requirement.needed_seats"] == 1
    current = resolved_hard_snapshots()[(original["id"], original["live_scenario_id"])]
    query = current["expected_tool_result_paths"]["search_current_games"]
    assert query["result.requirement.requesting_party.seat_count"] == 3
    assert not {"result.requirement.known_player_count", "result.requirement.needed_seats", "result.requirement.seat_format"} & set(query)
    assert current["expected_tool_result_paths"]["create_game"] == revision["previous_snapshot"]["expected_tool_result_paths"]["create_game"]
    assert current["expected_active_game_requirement"] == revision["previous_snapshot"]["expected_active_game_requirement"]
    scene = next(scene for scene in _live_eval_for_inquiry_tests().live_eval_scenarios() if scene.scenario_id == "known_three_person_party_matches_capacity")
    case = record["eval_cases"][0]
    assert case["context"]["fixture_ref"] == "scripts/run_real_owner_chat_live_eval.py::setup_three_person_capacity_pool"
    assert case["input"]["text"] == scene.message.text
    for field, expected in case["existing_hard_assertions_snapshot"].items():
        assert getattr(scene, field) == expected



def test_proactive_help_review_is_metadata_only_and_preserves_original_optional_meaning() -> None:
    """主动服务义务来自独立明确 flag，旧“可提出”含义与前六行原始证据不改写。"""
    raw = DATASET_PATH.read_bytes().splitlines(keepends=True)
    assert hashlib.sha256(b"".join(raw[:6])).hexdigest() == "733693289a8cc53ee1d1850e3821f6f80f03c4bcbbf3419ec1f6ccdd99c12b6c"
    record = read_records()[6]
    assert record["kind"] == "owner_chat_hard_assertion_revision"
    assert not {"eval_cases", "messages", "input", "expected"} & record.keys()
    assert record["source"]["statement"] == "查询了没有就应该问，要我帮你问问吗"
    original = next(case for r in read_records()[:6] for case in r.get("eval_cases", [])
                    if case.get("live_scenario_id") == "profile_range_empty_inquiry")
    optional = next(item for item in original["expected"]["required_meaning"] if item["id"] == "optional_help_offer")
    assert optional["description"] == "查无现成局时，可提出帮忙问人或组局的征询式服务提议，等待客户选择；本轮不要求自动建局或实际邀约。"
    revision = record["hard_assertion_revisions"][0]
    assert revision["previous_snapshot"] == original["existing_hard_assertions_snapshot"]
    assert "require_help_offer" not in revision["previous_snapshot"]
    assert revision["current_snapshot"] == {**revision["previous_snapshot"], "require_help_offer": True}
    module = _live_eval_for_inquiry_tests()
    scene = next(scene for scene in module.live_eval_scenarios() if scene.scenario_id == "profile_range_empty_inquiry")
    expectation = module.reply_semantic_expectation(scene)
    assert scene.require_help_offer
    assert sum(item["id"] == "offer_finding_players" for item in expectation["required_meaning"]) == 1
    assert {"id": f"{original['id']}.optional_help_offer", "description": optional["description"]} in expectation["required_meaning"]
    for field, value in resolved_hard_snapshots()[(original["id"], original["live_scenario_id"])].items():
        assert getattr(scene, field) == value
    assert len(module.live_eval_scenarios()) == 34
    assert len({case["case_id"] for case in module.golden_reply_cases()}) == 36


@pytest.mark.parametrize("hour,minute", [(10, 0), (16, 0), (20, 0), (23, 59), (0, 1), (15, 42)])
def test_profile_default_fixture_uses_one_future_day_for_message_and_inventory(hour, minute, monkeypatch, tmp_path) -> None:
    """晚执行的 setup 复用同一基准，不能把今天6:30请求与明天七点局混在一起。"""
    from mahjong_agent_runtime.domains.context_builders.task_facts import project_explicit_task_facts

    module = _live_eval_for_inquiry_tests()
    original = module.future_planned_start_at
    capture = (dt.datetime.now().astimezone() + dt.timedelta(days=2)).replace(hour=hour, minute=minute, second=0, microsecond=0)
    base = dt.datetime.fromisoformat(original("15:42", anchor=capture))
    assert base > capture
    assert base.date() == capture.date() + dt.timedelta(days=int((hour, minute) >= (15, 42)))
    monkeypatch.setattr(module, "PROFILE_DEFAULT_FIXTURE_ANCHOR", base)
    calls = []
    def delayed_default_clock(clock, *, anchor=None):
        calls.append((clock, anchor))
        # 模拟未显式传锚的调用发生在几天后；正确夹具的消息与库存不受影响。
        return original(clock, anchor=anchor if anchor is not None else base + dt.timedelta(days=3))
    monkeypatch.setattr(module, "future_planned_start_at", delayed_default_clock)
    scene = next(scene for scene in module.live_eval_scenarios() if scene.scenario_id == "profile_default_matched_game")
    store = module.SQLiteAgentStore(tmp_path / "aligned-profile-time.sqlite3")
    try:
        scene.setup(store)
        game = store.active_games()[0]
        assert scene.message.text == "帮我约个6.30无烟的"
        assert scene.message.sent_at == base
        assert scene.message.sent_at.date() == game.planned_start_at.date() == base.date()
        assert scene.message.sent_at.strftime("%H:%M") == "15:42"
        assert game.planned_start_at.strftime("%H:%M") == "19:00"
        assert ("19:00", base) in calls
        assert not any(clock == "15:42" for clock, _ in calls)
        facts = project_explicit_task_facts([], scene.message.to_dict(), None)
        requested = dt.datetime.fromisoformat(facts["candidates"]["planned_start_at"])
        assert requested.date() == game.planned_start_at.date()
        assert requested.strftime("%H:%M") == "18:30"
        assert game.planned_start_at - requested == dt.timedelta(minutes=30)
    finally:
        store._connection.close()


def test_casual_semantic_revision_keeps_all_hard_boundaries_and_replays_original_false_positive(tmp_path):
    """旧泛论回复不再被业务词误杀，语义未运行/拒绝及状态变化仍阻断。"""
    module = _live_eval_for_inquiry_tests()
    scene = next(scene for scene in module.live_eval_scenarios() if scene.scenario_id == 'casual_chat_should_not_pollute_business_state')
    review = next(record for record in read_records() if record['id'] == 'owner_chat_casual_hard_assertion_review_20261030_001')
    revision = review['hard_assertion_revisions'][0]
    before, after = revision['previous_snapshot'], revision['current_snapshot']
    assert {key: value for key, value in before.items() if key != 'forbidden_reply_contains'} == {
        key: value for key, value in after.items() if key != 'forbidden_reply_contains'
    }
    assert set(before['forbidden_reply_contains']) - set(after['forbidden_reply_contains']) == {
        '要组一个吗', '打多大', '几个人', '什么玩法', '这个先不聊', '打牌直接说',
        '七点', '7点', '三缺一', '371', '打吗', '来吗',
    }
    assert set(after['forbidden_reply_contains']) <= set(before['forbidden_reply_contains'])
    assert '我是AI' in scene.forbidden_reply_contains and 'prompt' in scene.forbidden_reply_contains
    semantic = module.reply_semantic_expectation(scene)
    assert {source['case_id'] for source in semantic['golden_sources']} == {
        'casual_chat_should_not_pollute_business_state', 'synthetic_live_casual_operational_discussion_boundary',
    }
    assert any(item['id'] == 'scenario_reply_alternatives_0' for item in semantic['required_meaning'])
    store = module.SQLiteAgentStore(tmp_path / 'casual_boundary.sqlite3')
    try:
        scene.setup(store)
        reply = '组局确实费脑子，谁几点有空、打多大、抽不抽烟，一堆事都得记着，有人帮着张罗肯定省心不少。'
        result = SimpleNamespace(final_reply=reply, status='completed', tool_results=[])
        assert module.validate_result(result, scene, [], store, {'passed': True, 'error': None})['passed']
        for assessment in (None, {'passed': False, 'error': None}, {'passed': True, 'error': 'judge failed'}):
            validation = module.validate_result(result, scene, [], store, assessment)
            assert not validation['passed']
            assert not next(item for item in validation['checks'] if item['name'] == 'reply_should_match_expected_meaning')['passed']
        game = store.active_games(scene.message.conversation_id)[0]
        # 用领域接口生成不同的权威状态，不直接改测试预期来适配输出。
        store.update_game_requirement(game_id=game.game_id, requirement_patch={'stake': '2'}, reason='测试状态边界', trace_id='mutation-boundary')
        validation = module.validate_result(result, scene, [], store, {'passed': True, 'error': None})
        assert not validation['passed']
    finally:
        store._connection.close()
