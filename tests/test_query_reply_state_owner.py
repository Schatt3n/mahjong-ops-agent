"""两条已有选项纯查询复用盲分类：状态错配不能靠一般判定、报告或旧回复放行。"""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from mahjong_agent_runtime import AgentContextBuilder, AgentLLMConfig, ToolCall, ToolGateway, ToolResult

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import reply_state_eval as state_eval
import run_evals as gate
import run_real_owner_chat_live_eval as owner


SCENARIOS = ("flexible_time_followup_matches_evening_game", "profile_range_matches_middle_stake")


def run_query(monkeypatch, tmp_path, scenario_id, *, act="inform_only", status="completed", question=False, classifier_error=False):
    """只替换主模型和分类输出；SQLite、最小上下文、真实查询、原始证据及验收仍实际执行。"""
    scene = next(scene for scene in owner.live_eval_scenarios() if scene.scenario_id == scenario_id)
    assert scene.requires_query_reply_state_check is True
    reply = ("晚上七点三缺一" if scenario_id == SCENARIOS[0] else "有个红中两块三缺一的") + ("，你打吗？" if question else "。")
    calls = []

    class Classifier:
        """不读取状态、工具或标签，只返回本测试声明的沟通行为，不能冒称真实分类通过。"""
        def __init__(self, config):
            assert config.temperature == 0 and config.thinking_mode == "disabled"

        def complete(self, messages, **kwargs):
            projected = json.loads(messages[-1]["content"])
            calls.append(projected)
            if classifier_error:
                raise RuntimeError("受控沟通分类故障")
            return json.dumps({"communication_act": act, "reason": "受控单测沟通行为，不是真实模型判定"})

    class Runtime:
        """本地连线替身保留真实Store/Gateway；不把预写动作当作主Agent能力证据。"""
        def __init__(self, store, trace):
            self.store, self.trace = store, trace

        def handle_user_message(self, message, *, trace_id):
            self.store.append_user_turn(message, trace_id)
            gateway = ToolGateway(self.store, trace_recorder=self.trace)
            context = AgentContextBuilder(self.store, gateway).build(message, trace_id=trace_id).payload
            self.trace.record(trace_id, "context_built", context)
            requirement = ({"game_type": "hangzhou_mahjong", "stake": "1"} if scenario_id == SCENARIOS[0]
                else {"game_type": "red_center_mahjong", "stake_range": {"min": 1, "max": 3}})
            result = gateway.execute(ToolCall("search_current_games", {"requirement": requirement, "limit": 10}),
                trace_id=trace_id, conversation_id=message.conversation_id, sender_id=message.sender_id,
                sender_name=message.sender_name, source_message_id=message.message_id, step_index=1)
            self.trace.record(trace_id, "tool_result", result.to_dict())
            review = ToolResult("customer_visible_content_review", called=True, allowed=True, result={"approved": True})
            self.trace.record(trace_id, "tool_result", review.to_dict())
            self.trace.record(trace_id, "final_output", {"reply": reply, "objective_status": status})
            return SimpleNamespace(trace_id=trace_id, conversation_id=message.conversation_id,
                final_reply=reply, status=status, tool_results=[result, review])

    monkeypatch.setattr(owner, "build_runtime", lambda client, store, trace, args: Runtime(store, trace))
    general = {"passed": True, "error": None, "raw_response": "一般判定原文仍保留"}
    def general_judge(*args, **kwargs):
        """主流程此处仅有1秒，复杂判定必须交独立默认预算，不能继承给小分类器的时限。"""
        assert "timeout_seconds" not in kwargs and kwargs["judge_timeout_seconds"] is None
        return general
    monkeypatch.setattr(owner, "assess_reply_semantics", general_judge)
    monkeypatch.setattr(state_eval, "OpenAICompatibleAgentClient", Classifier)
    client = SimpleNamespace(config=AgentLLMConfig(api_key="unit-only", model="unit-model", base_url="https://example.invalid"))
    report = owner.run_scenario(client, SimpleNamespace(db_path=tmp_path / "query.sqlite3", timeout_seconds=1), scene)
    assert report["semantic_assessment"] is general and general["passed"] is True
    return report, scene, calls


@pytest.mark.parametrize("scenario_id", SCENARIOS)
@pytest.mark.parametrize("act,status,question,passed", [
    ("inform_only", "completed", False, True),
    ("requires_user_response", "waiting_user", True, True),
    ("inform_only", "waiting_user", False, False),
    ("requires_user_response", "completed", True, False),
    ("other", "completed", False, False),
])
def test_existing_query_option_requires_blinded_communication_and_matching_status(monkeypatch, tmp_path, scenario_id, act, status, question, passed):
    report, scene, calls = run_query(monkeypatch, tmp_path, scenario_id, act=act, status=status, question=question)
    assert report["status"] == ("passed" if passed else "failed")
    assert report["requires_query_reply_state_check"] is True and len(calls) == 1
    projected = calls[0]
    assert set(projected) == {"current_request", "recent_conversation", "reply"}
    assert projected["current_request"]["text"] == scene.message.text and projected["reply"] == report["final_reply"]
    assert all(set(turn) == {"role", "content"} for turn in projected["recent_conversation"])
    serialized = json.dumps(projected, ensure_ascii=False)
    for excluded in ("runtime_status", "requires_query_reply_state_check", "tool_results", scenario_id):
        assert excluded not in serialized
    raw = json.loads(Path(report["raw_trace_path"]).read_text())
    assert sum(event["step"] == "query_reply_state_assessment" for event in raw["events"]) == 1
    if passed:
        gate.validate_owner_recovery_evidence(report, {"report_path": str(tmp_path / "report.json")},
            require_query_reply_state_check=True, expected_query_request=scene.message.to_dict())


def test_query_communication_classifier_failure_blocks_even_when_general_judge_passed(monkeypatch, tmp_path):
    report, _, calls = run_query(monkeypatch, tmp_path, SCENARIOS[0], classifier_error=True)
    assert len(calls) == 1 and report["status"] == "failed"
    assert report["query_reply_state_assessment"]["error"]


@pytest.mark.parametrize("fault", ["missing", "disabled_flag", "changed_reply", "wrong_raw", "wrong_identity", "changed_history",
    "changed_user", "changed_quoted_user", "changed_config", "missing_attempt", "reused_valid_attempt", "missing_event", "mismatched_status",
    "changed_terminal_status"])
def test_query_guard_revalidates_original_input_reply_classification_and_backend_mapping(monkeypatch, tmp_path, fault):
    report, scene, _ = run_query(monkeypatch, tmp_path, SCENARIOS[0])
    path = Path(report["raw_trace_path"])
    raw = json.loads(path.read_text())
    state_check = report["query_reply_state_assessment"]
    state_event = next(event for event in raw["events"] if event["step"] == "query_reply_state_assessment")
    if fault == "missing":
        report.pop("query_reply_state_assessment")
    elif fault == "disabled_flag":
        report["requires_query_reply_state_check"] = False
    elif fault == "changed_reply":
        report["final_reply"] = "晚上七点三缺一，你打吗？"
    elif fault == "wrong_raw":
        state_check["raw_response"] = json.dumps({"communication_act": "requires_user_response", "reason": "不同原文"})
    elif fault == "wrong_identity":
        state_check["trace_id"] = "another_call"
    elif fault == "changed_history":
        state_check["input"]["recent_conversation"] = []
    elif fault in {"changed_user", "changed_quoted_user"}:
        request = next(event for event in raw["events"] if event["step"] == "context_built")["content"]["current_message"]
        if fault == "changed_user":
            request["text"] = "另一条用户需求"
        else:
            request["quoted_message"] = {"text": "伪造引用来改变沟通行为"}
    elif fault == "changed_config":
        state_check["judge_config"]["thinking_mode"] = "enabled"
    elif fault == "missing_attempt":
        state_check["attempts"] = []
    elif fault == "reused_valid_attempt":
        first = deepcopy(state_check["attempts"][0]); first["error"] = "伪称合法输出无效"
        last = deepcopy(state_check["attempts"][0]); last["trace_id"] += ":contract_repair_1"
        state_check.update(attempts=[first, last], contract_repair_count=1)
    elif fault == "missing_event":
        raw["events"].remove(state_event)
    elif fault == "mismatched_status":
        report["runtime_status"] = "waiting_user"
    elif fault == "changed_terminal_status":
        next(event for event in raw["events"] if event["step"] == "final_output")["content"]["objective_status"] = "waiting_user"
    if fault not in {"missing", "missing_event"}:
        state_event["content"] = deepcopy(state_check)
    report["trace_steps"] = [event["step"] for event in raw["events"]]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="query reply state"):
        gate.validate_owner_recovery_evidence(report, {"report_path": str(tmp_path / "report.json")},
            require_query_reply_state_check=True, expected_query_request=scene.message.to_dict())


def test_only_the_two_policy_declared_pure_queries_require_the_new_guard():
    contract = gate.load_owner_semantic_contract()
    assert set(contract["query_reply_state_requests"]) == set(SCENARIOS)
