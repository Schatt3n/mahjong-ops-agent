"""盲分类只解释真实沟通行为，状态比较、来源绑定和拒绝重判由后端验收。"""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from mahjong_agent_runtime import AgentLLMConfig

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import reply_state_eval as state_eval


def evidence():
    """夹具混入禁止给分类器的执行/标签信息，只有本客户自然对话可投影。"""
    return {"request": {"text": "只看看明晚七点国标2块有没有现成局", "message_id": "private-fixture-id",
                        "runtime_status": "completed", "metadata": {"expected_passed": False}},
            "recent_conversation": [{"role": "user", "content": "我一个人，烟不限"},
                {"role": "assistant", "content": "我帮你看看。", "runtime_status": "completed"},
                {"role": "tool", "content": "runtime_status=completed"}],
            "reply": "明晚七点有个国标2块三缺一的，你打吗？", "runtime_status": "completed",
            "execution_lifecycle": {"turn": {"status": "completed"}}, "expected_passed": False,
            "fixture_id": "private-fixture-id", "objective_plan": ["do-not-show-plan"],
            "tool_results": [{"private": "do-not-show-tool"}], "persisted_state": {"private": "do-not-show-state"}}


def assess(monkeypatch, outputs, *, value=None):
    """固定模型输出仅验证真实投影、解析及调用证据，不宣称语义能力已通过。"""
    calls = []
    outputs = list(outputs)

    class Judge:
        def __init__(self, *, config):
            assert config.temperature == 0 and config.thinking_mode == "disabled"

        def complete(self, messages, **kwargs):
            calls.append(deepcopy(messages))
            output = outputs.pop(0)
            if isinstance(output, Exception):
                raise output
            return output

    monkeypatch.setattr(state_eval, "OpenAICompatibleAgentClient", Judge)
    client = SimpleNamespace(config=AgentLLMConfig(api_key="unit-only", model="unit-model", base_url="https://example.invalid"))
    current = value if value is not None else evidence()
    result = state_eval.assess_reply_communication(client, evidence=current, trace_id="classification-call", timeout_seconds=1)
    return result, calls, current


def test_blind_communication_classifier_omits_runtime_status_and_backend_rejects_question_act_completed(monkeypatch):
    """本次真实校准误判回归：模型不见状态，询问行为配completed由后端必然拒绝。"""
    result, calls, current = assess(monkeypatch, [json.dumps({"communication_act": "requires_user_response", "reason": "回复在征询客户选择"})])
    assert len(calls) == 1
    payload = json.loads(calls[0][-1]["content"])
    assert set(payload) == {"current_request", "recent_conversation", "reply"}
    assert payload["current_request"] == {"text": current["request"]["text"], "quoted_text": ""}
    assert payload["recent_conversation"] == [{"role": "user", "content": "我一个人，烟不限"},
                                              {"role": "assistant", "content": "我帮你看看。"}]
    encoded = json.dumps(calls, ensure_ascii=False)
    for marker in ("runtime_status", "expected_passed", "private-fixture-id", "objective_plan", "do-not-show"):
        assert marker not in encoded
    assert state_eval.validate_reply_state_assessment(result, evidence=current, trace_id="classification-call")
    assert state_eval.communication_state_is_consistent("completed", result) is False
    assert state_eval.communication_state_is_consistent("waiting_user", result) is True


@pytest.mark.parametrize("act,status,expected", [
    ("inform_only", "completed", True), ("inform_only", "waiting_user", False),
    ("requires_user_response", "completed", False), ("requires_user_response", "waiting_user", True),
    ("other", "completed", False), ("other", "waiting_user", False),
])
def test_communication_state_comparison_uses_only_allowed_act_enum(act, status, expected):
    assert state_eval.communication_state_is_consistent(status, {"error": None, "communication_act": act}) is expected
    assert state_eval.communication_state_is_consistent(status, {"error": "unavailable", "communication_act": act}) is False


@pytest.mark.parametrize("raw", ["{}", "not JSON", '{"communication_act":"other","reason":"x","runtime_status":"completed"}',
    '{"communication_act":"other","communication_act":"inform_only","reason":"x"}'])
def test_invalid_classification_contract_can_be_repaired_once(monkeypatch, raw):
    result, calls, current = assess(monkeypatch, [raw, json.dumps({"communication_act": "requires_user_response", "reason": "需客户回答"})])
    assert len(calls) == 2 and result["contract_repair_count"] == 1
    assert result["attempts"][0]["raw_response"] == raw
    assert result["attempts"][0]["error"]
    assert state_eval.validate_reply_state_assessment(result, evidence=current, trace_id="classification-call")


def test_valid_but_wrong_classification_and_other_are_not_retried(monkeypatch):
    """有效分类可能有语义错误，必须由校准暴露，不能为标签重试到正确。"""
    for act in ("inform_only", "other"):
        result, calls, _ = assess(monkeypatch, [json.dumps({"communication_act": act, "reason": "受控错误分类"})])
        assert len(calls) == 1 and result["error"] is None and result["communication_act"] == act


def test_classifier_call_error_is_not_retried_or_converted_to_other(monkeypatch):
    result, calls, _ = assess(monkeypatch, [TimeoutError("unit-only timeout")])
    assert len(calls) == 1 and result["error"] and result["communication_act"] is None


@pytest.mark.parametrize("fault", ["input", "hash", "rubric", "config", "trace", "attempt_raw", "attempt_trace",
                                         "repair_count", "raw_act", "valid_retry"])
def test_classification_validation_rejects_unbound_or_rejudged_evidence(monkeypatch, fault):
    result, _, current = assess(monkeypatch, [json.dumps({"communication_act": "requires_user_response", "reason": "需回应"})])
    if fault == "input":
        result["input"]["runtime_status"] = "completed"
    elif fault == "hash":
        result["input_sha256"] = "0" * 64
    elif fault == "rubric":
        result["rubric"] = "changed"
    elif fault == "config":
        result["judge_config"]["temperature"] = 1
    elif fault == "trace":
        result["trace_id"] = "another-call"
    elif fault == "attempt_raw":
        result["attempts"][0]["raw_response"] = "{}"
    elif fault == "attempt_trace":
        result["attempts"][0]["trace_id"] = "another-call"
    elif fault == "repair_count":
        result["contract_repair_count"] = 1
    elif fault == "raw_act":
        result["communication_act"] = "inform_only"
    else:
        first = deepcopy(result["attempts"][0])
        first["error"] = "claimed contract error"
        result["attempts"] = [first, deepcopy(result["attempts"][0])]
        result["attempts"][1]["trace_id"] += ":contract_repair_1"
        result["contract_repair_count"] = 1
    assert not state_eval.validate_reply_state_assessment(result, evidence=current, trace_id="classification-call")


@pytest.mark.parametrize("reply,act,status", [
    ("明晚七点三缺一，这桌你选好后回我一声。", "requires_user_response", "waiting_user"),
    ("“你打吗？”这句话是在问你要不要参加。", "inform_only", "completed"),
])
def test_backend_comparison_does_not_use_question_marks_or_quoted_question_words(monkeypatch, reply, act, status):
    """仅用固定分类标签验证协议边界；无问号待选、引用问句解释不由后端关键词决定。"""
    current = evidence()
    current["reply"] = reply
    result, calls, _ = assess(monkeypatch, [json.dumps({"communication_act": act, "reason": "固定语义标签"})], value=current)
    assert len(calls) == 1
    assert state_eval.communication_state_is_consistent(status, result)
    opposite = "completed" if status == "waiting_user" else "waiting_user"
    assert not state_eval.communication_state_is_consistent(opposite, result)
