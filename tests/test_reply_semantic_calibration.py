"""独立判定校准必须来自真实持久事实，不能靠通过标签或修订旧黄金期望放行。"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_real_owner_chat_live_eval as owner
import run_inquiry_grounding_eval as inquiry
from mahjong_agent_runtime import AgentLLMConfig, InMemoryTraceRecorder, SQLiteAgentStore, ToolCall
from tests.semantic_http_stub import semantic_http_completion


def build_calibration_stub(work_dir):
    """只替换独立模型边界；夹具准备、网关、SQLite和判决解析均使用真实代码。"""
    cases = owner.load_reply_semantic_calibration_cases()
    calls = []

    class FixedJudge:
        """固定输出仅验证证据接线，不能宣称这些离线结果是真实模型语义通过。"""

        def __init__(self, config):
            self.config = config
            assert all(getattr(config, key) == value for key, value in inquiry.SEMANTIC_JUDGE_FIXED_CONFIG.items())

        def complete(self, messages, **kwargs):
            assert kwargs["timeout_seconds"] == self.config.attempt_timeout_seconds
            request = json.loads(messages[-1]["content"])
            case = cases[len(calls)]
            calls.append(request)
            assert request["evidence"]["reply"] == case["reply"]
            raw = json.dumps({"reason": "离线替身按固定标签检查校准连线，不作语义结论", "criteria": [
                {"id": criterion["id"], "violation_detected": criterion["id"] in case["expected_violation_ids"],
                 "reason": "仅模拟独立模型的逐项判定输出"} for criterion in request["expectation"]["checks"]]}, ensure_ascii=False)
            return semantic_http_completion(raw, config=self.config, trace_id=kwargs["trace_id"])

    def forbidden_main_call(*args, **kwargs):
        raise AssertionError("calibration must never invoke main-agent planning")

    client = SimpleNamespace(config=AgentLLMConfig(api_key="unit-only-key", model="unit-model", provider="deepseek", base_url="https://example.invalid",
        attempt_timeout_seconds=60),
                             complete=forbidden_main_call)
    with patch.object(inquiry, "OpenAICompatibleAgentClient", FixedJudge):
        payload = owner.run_reply_semantic_calibration(client, SimpleNamespace(timeout_seconds=1), work_dir)
    return payload, calls


@pytest.fixture
def calibration(tmp_path):
    """为每例单独准备本次目录，失败注入只能改临时证据，不能读写历史运行资产。"""
    root = tmp_path / "current-run"
    root.mkdir()
    payload, calls = build_calibration_stub(root / "calibration")
    return payload, calls, root / "report.json"


def test_fixed_calibration_has_all_positive_negative_cases_and_no_label_leak(calibration):
    payload, calls, report_path = calibration
    assert payload["status"] == "passed", [(item["case_id"], item.get("error_type"), item.get("checks")) for item in payload["reports"]]
    coverage = owner.validate_reply_semantic_calibration(payload, report_path)
    assert coverage["total"] == coverage["passed"] == len(calls) == 8
    assert sum(item["semantic_assessment"]["passed"] for item in payload["reports"]) == 3
    for item in payload["reports"]:
        assessment = item["semantic_assessment"]
        assert assessment["judge_contract_version"] == inquiry.SEMANTIC_JUDGE_CONTRACT_VERSION
        assert inquiry.semantic_judge_config_matches(assessment)
    for source, request in zip(owner.load_reply_semantic_calibration_cases(), calls):
        serialized = json.dumps(request, ensure_ascii=False)
        assert source["case_id"] not in serialized and '"fixture":' not in serialized
        if source["fixture"] != "empty":
            assert source["fixture"] not in serialized
        assert "expected_passed" not in serialized and "expected_violation_ids" not in serialized
        assert "production_review_approved" not in serialized and "reasoning_summary" not in request["evidence"]
        assert request["expectation"]["communication_obligation_contract"] == inquiry.communication_obligation_contract()
        if source["fixture"] != "empty":
            assert request["evidence"]["execution_lifecycle"]["activities"][0]["status"] == "forming"
            assert request["evidence"]["execution_lifecycle"]["turn"]["scope"] == "synchronous_turn_only"


@pytest.mark.parametrize("fault", ["missing_case", "duplicate_case", "unknown_case", "changed_source", "missing_trace", "outside_trace",
    "missing_gateway", "missing_result", "changed_request", "no_raw_judge", "forged_pass", "hidden_write", "deleted_task",
    "disabled_outbound", "changed_board_evidence", "fake_follow_up", "missing_check", "numeric_check", "changed_hash"])
def test_calibration_gate_rejects_incomplete_or_self_certified_evidence(calibration, tmp_path, fault):
    payload, _, report_path = calibration
    item = payload["reports"][4]
    raw_path = Path(item["raw_trace_path"])
    raw = json.loads(raw_path.read_text())
    if fault == "missing_case":
        payload["reports"].pop()
        payload["case_count"] = payload["passed_count"] = len(payload["reports"])
    elif fault == "duplicate_case":
        payload["reports"][-1] = deepcopy(item)
    elif fault == "unknown_case":
        item["case_id"] = "not-a-source"
    elif fault == "changed_source":
        item["source"]["expected_passed"] = False
    elif fault == "missing_trace":
        item.pop("raw_trace_path")
    elif fault == "outside_trace":
        old = tmp_path / "old-trace.json"
        old.write_text(raw_path.read_text())
        item["raw_trace_path"] = str(old)
    elif fault in {"missing_gateway", "missing_result"}:
        removed = "tool_gateway_completed" if fault == "missing_gateway" else "tool_result"
        raw["events"] = [event for event in raw["events"] if event["step"] != removed]
        item["trace_steps"] = [event["step"] for event in raw["events"]]
    elif fault == "changed_request":
        raw["request"]["text"] = "更容易的输入"
    elif fault == "no_raw_judge":
        item["semantic_assessment"]["raw_response"] = ""
    elif fault == "forged_pass":
        negative = payload["reports"][3]
        negative["semantic_assessment"]["passed"] = True
    elif fault == "hidden_write":
        with sqlite3.connect(item["db_path"]) as connection:
            connection.execute("INSERT INTO runtime_outbound_message_drafts VALUES(?,?,?,?,?,?,?)",
                ("hidden", "unit", "unit", "wechat", "pending", "{}", "2026-11-02"))
    elif fault == "deleted_task":
        with sqlite3.connect(item["db_path"]) as connection:
            connection.execute("DELETE FROM runtime_scheduled_agent_tasks WHERE task_type='publish_group_board'")
    elif fault == "disabled_outbound":
        with sqlite3.connect(item["db_path"]) as connection:
            row = connection.execute("SELECT room_id,payload FROM runtime_group_room_policies").fetchone()
            policy = json.loads(row[1]);policy["outbound_enabled"] = False
            connection.execute("UPDATE runtime_group_room_policies SET payload=? WHERE room_id=?", (json.dumps(policy), row[0]))
    elif fault == "changed_board_evidence":
        item["semantic_assessment"]["evidence"]["execution_lifecycle"]["activities"][0]["board_arrangement"]["publication_sent"] = True
    elif fault == "fake_follow_up":
        item["semantic_assessment"]["evidence"]["persisted_state"]["follow_up_state"]["tasks"] = []
    elif fault == "missing_check":
        item["checks"].pop()
    elif fault == "numeric_check":
        item["checks"][0]["passed"] = 1
    elif fault == "changed_hash":
        payload["source_sha256"] = "0" * 64
    raw_path.write_text(json.dumps(raw))
    with pytest.raises((ValueError, json.JSONDecodeError)):
        owner.validate_reply_semantic_calibration(payload, report_path)


@pytest.mark.parametrize("fault", ["missing_version", "old_version", "disabled", "high_effort", "small_budget",
    "temperature", "response_format", "numeric_type", "contradictory_http", "missing_http", "empty_http"])
def test_old_eight_calibration_requires_new_reasoned_config_and_actual_http_when_present(calibration, fault):
    """同步修改报告与原事件仍须拒绝旧配置，正式HTTP矛盾不能降级成无诊断替身。"""
    payload, _, report_path = calibration
    item = payload["reports"][0]
    assessment = item["semantic_assessment"]
    if fault == "missing_version":
        assessment.pop("judge_contract_version")
    elif fault == "old_version":
        assessment["judge_contract_version"] = "disabled_v1"
    elif fault in {"missing_http", "empty_http"}:
        if fault == "missing_http":
            assessment["attempts"][0].pop("completion_diagnostics")
        else:
            assessment["attempts"][0]["completion_diagnostics"] = {}
    elif fault == "contradictory_http":
        assessment["attempts"][0]["completion_diagnostics"] = {"route": "primary",
            "provider": assessment["judge_config"]["provider"], "requested_model": assessment["judge_config"]["model"],
            "request": {"max_tokens": 16384, "response_format": "json_object", "thinking_mode": "disabled", "reasoning_effort": None}}
    else:
        field, value = {"disabled": ("thinking_mode", "disabled"), "high_effort": ("reasoning_effort", "high"),
            "small_budget": ("max_tokens", 4000), "temperature": ("temperature", 1),
            "response_format": ("response_format", "text"), "numeric_type": ("temperature", 0.0)}[fault]
        assessment["judge_config"][field] = value
    path = Path(item["raw_trace_path"])
    raw = json.loads(path.read_text())
    next(event for event in raw["events"] if event["step"] == "semantic_calibration_assessment")["content"] = deepcopy(assessment)
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="independent configuration"):
        owner.validate_reply_semantic_calibration(payload, report_path)


def test_old_eight_judge_uses_own_default_or_environment_budget_instead_of_main_one_second(monkeypatch, tmp_path):
    """主调用1秒/单尝试60秒都不约束独立判定；默认和专用环境预算明确记入原始证据。"""
    monkeypatch.delenv("MAHJONG_SEMANTIC_JUDGE_TIMEOUT_SECONDS", raising=False)
    for value, expected, source in ((None, 180.0, "default"), ("210", 210.0, "environment")):
        if value is not None:
            monkeypatch.setenv("MAHJONG_SEMANTIC_JUDGE_TIMEOUT_SECONDS", value)
        payload, _ = build_calibration_stub(tmp_path / source)
        assert payload["status"] == "passed"
        for report in payload["reports"]:
            assessment = report["semantic_assessment"]
            assert assessment["judge_transport"] == {"total_timeout_seconds": expected,
                "attempt_timeout_seconds": expected, "retry_attempts": 3, "budget_source": source}
            assert all(attempt["timeout_seconds"] == expected for attempt in assessment["attempts"])


def test_existing_golden_expectation_is_unchanged_and_new_evidence_has_no_private_room_ids(tmp_path):
    payload, _ = build_calibration_stub(tmp_path / "run")
    example = payload["reports"][1]["semantic_assessment"]["evidence"]
    arrangement = example["execution_lifecycle"]["activities"][0]["board_arrangement"]
    assert set(arrangement) == {"status", "linked_room_count", "pending_task_count", "outbound_authorized_room_count", "publication_sent"}
    assert "calibration-authorized-board" not in json.dumps(example, ensure_ascii=False)
    source = next(case for case in owner.golden_reply_cases() if case["scenario_id"] == "profile_context_infers_base_and_cap")
    assert source["expected"]["reply_reference"] == "好，我帮你看看。"
    assert owner.reply_semantic_expectation(next(scene for scene in owner.live_eval_scenarios()
        if scene.scenario_id == "profile_context_infers_base_and_cap"))["forbidden_meaning"][:2] == owner.common_forbidden_semantic_meanings()


def test_original_profile_fixture_board_task_survives_restart_and_next_fake_send_contains_game(tmp_path):
    store = SQLiteAgentStore(tmp_path / "original.sqlite3")
    owner.setup_base_cap_profile_with_board(store)
    scene = next(scene for scene in owner.live_eval_scenarios() if scene.scenario_id == "profile_context_infers_base_and_cap")
    trace = InMemoryTraceRecorder()
    from mahjong_agent_runtime import ToolGateway
    gateway = ToolGateway(store, trace_recorder=trace)
    call = ToolCall(name="create_game", arguments={"organizer_id": scene.message.sender_id, "organizer_name": scene.message.sender_name,
        "requirement": {"game_type": "national_standard_mahjong", "stake": "1", "cap_score": 256,
            "start_time_kind": "asap_when_full", "smoke_preference": "any", "known_player_count": 1, "needed_seats": 3},
        "known_players": [{"customer_id": scene.message.sender_id, "display_name": scene.message.sender_name, "seat_count": 1}]})
    created = gateway.execute(call, trace_id="original", conversation_id=scene.message.conversation_id,
        sender_id=scene.message.sender_id, sender_name=scene.message.sender_name, step_index=1)
    trace.record("original", "tool_result", created.to_dict())
    assert created.called and created.allowed and not created.error
    raw_path = tmp_path / "original_trace.json"
    raw_path.write_text(json.dumps({"events": [event.to_dict() for event in trace.events]}))
    proof = owner.run_board_recovery_probe(store, scene.message, work_dir=tmp_path / "recovery")
    proof["original_tool_arrangement"] = owner.project_board_arrangement(created.result["board_arrangement"])
    proof["checks"].append({"name": "create_game_returns_its_durable_board_arrangement", "passed": True})
    assert proof["status"] == "passed", proof
    item = {"board_recovery": proof, "db_path": str(store.path), "raw_trace_path": str(raw_path)}
    owner.validate_board_recovery_evidence(item, tmp_path / "report.json")
    assert proof["before_arrangement"]["publication_sent"] is False
    assert proof["after_arrangement"]["publication_sent"] is True
    assert proof["sent_messages"][0]["sandbox_only"] is True
