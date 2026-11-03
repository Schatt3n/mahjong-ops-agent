"""金额核验连线只替换模型响应，真实查询与原始trace不能被一般语义通过替代。"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from mahjong_agent_runtime import AgentLLMConfig, ToolCall, ToolGateway, ToolResult

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import reply_money_eval as money
import run_evals as gate
import run_real_owner_chat_live_eval as owner


def owner_case(monkeypatch, tmp_path, *, reply="晚上七点1块三缺一，你打吗？", scope="existing_option_fact", multiple=False, general_passed=True):
    """主流程替身只隔离本次接线测试；Gateway和SQLite仍实际执行，不宣称真实模型通过。"""
    scene = next(s for s in owner.live_eval_scenarios() if s.scenario_id == "flexible_time_followup_matches_evening_game")
    # 本文件隔离金额核验；纯查询状态guard另由独立正反测试验证，不以金额分类替代它。
    scene = replace(scene, requires_query_reply_state_check=False)
    if multiple:
        original_setup = scene.setup

        def setup(store):
            original_setup(store)
            game = next(iter(store.games.values()))
            store.create_game(conversation_id="another_pool", organizer_id="another-host", organizer_name="牌友",
                requirement=deepcopy(game.requirement), known_players=[{"customer_id": "another-host", "seat_count": 3}],
                trace_id="money-another-game")

        scene = replace(scene, setup=setup, expected_tool_result_paths={})
    calls = []

    class ScopeJudge:
        """只回受控作用域；分类输入不得含查询答案或通过标签。"""
        def __init__(self, config):
            assert config.temperature == 0 and config.thinking_mode == "disabled"

        def complete(self, messages, **kwargs):
            projected = json.loads(messages[-1]["content"])
            calls.append(projected)
            assert set(projected) == {"reply", "spans"}
            if scope == "raise_error":
                raise RuntimeError("受控分类调用故障")
            return json.dumps({"items": [{"span_id": item["span_id"], "scope": scope, "role": "stake",
                "form": "single", "reason": "受控单测分类，不是真实模型结论"} for item in projected["spans"]]})

    class LocalRuntime:
        """仅连接已有查询和最终回复，避免把预写主模型动作冒充本次端到端能力。"""
        def __init__(self, store, trace):
            self.store, self.trace = store, trace

        def handle_user_message(self, message, *, trace_id):
            result = ToolGateway(self.store, trace_recorder=self.trace).execute(ToolCall("search_current_games", {
                "requirement": {"game_type": "hangzhou_mahjong", "stake": "1"}, "limit": 10}), trace_id=trace_id,
                conversation_id=message.conversation_id, sender_id=message.sender_id, sender_name=message.sender_name,
                source_message_id=message.message_id, step_index=1)
            self.trace.record(trace_id, "tool_result", result.to_dict())
            # 正式主回复总会追加可见审查结果；它是无写副作用的模型证据，不能让金额核验失效。
            review = ToolResult("customer_visible_content_review", called=True, allowed=True, result={"approved": True})
            self.trace.record(trace_id, "tool_result", review.to_dict())
            self.trace.record(trace_id, "final_output", {"reply": reply, "objective_status": "waiting_user"})
            return SimpleNamespace(final_reply=reply, status="waiting_user", trace_id=trace_id,
                conversation_id=message.conversation_id, tool_results=[result, review])

    general = {"passed": general_passed, "error": None, "raw_response": "原始一般判定保留，不因金额检查改写"}
    monkeypatch.setattr(owner, "build_runtime", lambda client, store, trace, args: LocalRuntime(store, trace))
    monkeypatch.setattr(owner, "assess_reply_semantics", lambda *args, **kwargs: general)
    monkeypatch.setattr(money, "OpenAICompatibleAgentClient", ScopeJudge)
    client = SimpleNamespace(config=AgentLLMConfig(api_key="unit-only", model="unit-model", base_url="https://example.invalid"))
    report = owner.run_scenario(client, SimpleNamespace(db_path=tmp_path / "owner.sqlite3", timeout_seconds=1), scene)
    assert report["semantic_assessment"] is general and general["passed"] is general_passed
    return report, calls


@pytest.mark.parametrize("reply,scope,status,overall,calls", [
    ("晚上七点1块三缺一，你打吗？", "existing_option_fact", "verified", "passed", 1),
    ("晚上七点2块三缺一，你打吗？", "existing_option_fact", "conflict", "failed", 1),
    ("晚上七点三缺一，你打吗？", "existing_option_fact", "not_applicable", "passed", 0),
    ("晚上七点1块三缺一，你打吗？", "uncertain", "unproven", "failed", 1),
    ("晚上七点1块三缺一，你打吗？", "raise_error", "error", "failed", 1),
])
def test_owner_money_check_preserves_original_general_judge_and_controls_overall(monkeypatch, tmp_path, reply, scope, status, overall, calls):
    report, actual_calls = owner_case(monkeypatch, tmp_path, reply=reply, scope=scope)
    assert report["status"] == overall
    assert report["money_fact_check"]["status"] == status and len(actual_calls) == calls
    check = next(item for item in report["checks"] if item["name"] == "reply_money_facts_should_not_conflict")
    assert check["passed"] is (overall == "passed")
    raw = json.loads(Path(report["raw_trace_path"]).read_text())
    events = raw["events"]
    assert report["trace_steps"] == [event["step"] for event in events]
    assert events[-1]["step"] == "reply_money_fact_check" and events[-1]["content"] == report["money_fact_check"]
    assert events[-2]["step"] == "final_output" and events[-2]["content"]["reply"] == reply
    if overall == "passed":
        gate.validate_owner_recovery_evidence(report, {"report_path": str(tmp_path / "report.json")})


def test_money_match_never_overrides_a_failed_original_general_judgement(monkeypatch, tmp_path):
    report, _ = owner_case(monkeypatch, tmp_path, general_passed=False)
    assert report["status"] == "failed" and report["money_fact_check"]["status"] == "verified"
    assert next(check for check in report["checks"] if check["name"] == "reply_should_match_expected_meaning")["passed"] is False


def test_multiple_verified_candidates_are_uncovered_not_falsely_money_verified(monkeypatch, tmp_path):
    report, calls = owner_case(monkeypatch, tmp_path, multiple=True)
    check = report["money_fact_check"]
    assert report["status"] == "passed" and calls == []
    assert check["status"] == "uncovered" and check["blocks"] is False
    assert check["coverage"] == "no_unique_verified_candidate" and check["candidate"] is None
    gate.validate_owner_recovery_evidence(report, {"report_path": str(tmp_path / "report.json")})


@pytest.mark.parametrize("fault", ["missing_check", "missing_event", "changed_reply", "changed_scope_identity",
    "wrong_candidate_price", "money_before_final", "extra_money_event", "blocked_but_claimed_passed"])
def test_owner_gate_recomputes_money_from_original_reply_and_tools(monkeypatch, tmp_path, fault):
    reply = "晚上七点2块三缺一，你打吗？" if fault == "blocked_but_claimed_passed" else "晚上七点1块三缺一，你打吗？"
    report, _ = owner_case(monkeypatch, tmp_path, reply=reply)
    raw_path = Path(report["raw_trace_path"])
    raw = json.loads(raw_path.read_text())
    event = next(event for event in raw["events"] if event["step"] == "reply_money_fact_check")
    if fault == "missing_check":
        report.pop("money_fact_check")
    elif fault == "missing_event":
        raw["events"].remove(event)
    elif fault == "changed_reply":
        report["final_reply"] = "晚上七点2块三缺一，你打吗？"
    elif fault == "changed_scope_identity":
        report["money_fact_check"]["scope_assessment"]["trace_id"] = "other_call:scope"
        event["content"] = deepcopy(report["money_fact_check"])
    elif fault == "wrong_candidate_price":
        report["money_fact_check"]["candidate"]["money_facts"]["stake"]["values"] = ["2"]
        event["content"] = deepcopy(report["money_fact_check"])
    elif fault == "money_before_final":
        raw["events"].remove(event)
        raw["events"].insert(-1, event)
    elif fault == "extra_money_event":
        raw["events"].append(deepcopy(event))
    elif fault == "blocked_but_claimed_passed":
        report["status"] = "passed"
        for check in report["checks"]:
            check["passed"] = True
    report["trace_steps"] = [event["step"] for event in raw["events"]]
    raw_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="money fact check"):
        gate.validate_owner_recovery_evidence(report, {"report_path": str(tmp_path / "report.json")})
