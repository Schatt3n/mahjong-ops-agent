"""盲分类只解释真实沟通行为，状态比较、来源绑定和拒绝重判由后端验收。"""
from copy import deepcopy
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from mahjong_agent_runtime import AgentLLMConfig
from mahjong_agent_runtime.llm import eval_transport_failure, external_call_deadline

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import reply_state_eval as state_eval
import eval_retry


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


def assess(monkeypatch, outputs, *, value=None, frozen_assessment=None):
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
    result = state_eval.assess_reply_communication(client, evidence=current, trace_id="classification-call", timeout_seconds=1,
                                                  frozen_assessment=frozen_assessment)
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


@contextmanager
def retry_window(monkeypatch, folder, index):
    """注入评测窗口与确定性退避，验证请求冻结和账本，无需真模型。"""
    monkeypatch.setattr(eval_retry, "sleep", lambda _seconds: None)
    budget = eval_retry.CaseBudget(folder, index)
    budget_token = eval_retry._budget.set(budget)
    deadline_token = external_call_deadline.set(None)
    try:
        yield budget
    finally:
        external_call_deadline.reset(deadline_token)
        eval_retry._budget.reset(budget_token)


def typed_timeout():
    """使用客户端已验证的结构化 HTTP 失败证据，普通超时文本没有此资格。"""
    failure = RuntimeError("test transport timed out")
    failure._completion_diagnostics = {"schema_version": 1, "route": "primary", "error_type": "RuntimeError",
        "attempts": [{"failure_reason": "timeout_reading_response_body"}]}
    return failure


def test_auxiliary_judge_resumes_exact_contract_repair_without_repeating_prior_response(monkeypatch, tmp_path):
    """等待传输失败不会填进生效判定；下个窗口只补原先被冻结的合同修复请求。"""
    with retry_window(monkeypatch, tmp_path, 0):
        frozen, calls, current = assess(monkeypatch, ["{}", *(typed_timeout() for _ in range(4))])
    assert len(calls) == 5 and calls[1:] == [calls[1]] * 4
    assert frozen["eval_inconclusive"]["category"] == "unattributed_timeout"
    assert len(frozen["attempts"]) == 1 and frozen["attempts"][0]["raw_response"] == "{}"
    assert len(frozen["transport_failures"]) == 4 and frozen["contract_repair_count"] == 1
    answer = json.dumps({"communication_act": "inform_only", "reason": "固定有效否定结果"})
    with retry_window(monkeypatch, tmp_path, 1):
        result, resumed_calls, _ = assess(monkeypatch, [answer], frozen_assessment=frozen)
    assert resumed_calls == [calls[1]] and len(frozen["attempts"]) == 1
    assert result["transport_failures"] == frozen["transport_failures"] and "eval_inconclusive" not in result
    assert state_eval.validate_reply_state_assessment(result, evidence=current, trace_id="classification-call")
    assert state_eval.communication_state_is_consistent("waiting_user", result) is False


def test_auxiliary_judge_defers_without_any_http_and_verifies_previous_shared_budget_record(monkeypatch, tmp_path):
    """挂起的判定没有实际HTTP；下一窗口须核验此前扣减及保持原冻结输入。"""
    with retry_window(monkeypatch, tmp_path, 0) as budget:
        assert budget.begin_retry(eval_transport_failure(typed_timeout()), stage="agent")
        budget.deadline = eval_retry.monotonic() - 1
        frozen, calls, current = assess(monkeypatch, [])
    assert calls == [] and frozen["attempts"] == [] and "transport_failures" not in frozen
    assert frozen["eval_inconclusive"] == eval_retry.DEFERRED_AUXILIARY_JUDGE
    assert frozen["deferred_judge"]["retry_trigger"]["stage"] == "agent"
    answer = json.dumps({"communication_act": "requires_user_response", "reason": "待用户回应"})
    with retry_window(monkeypatch, tmp_path, 1):
        result, calls, _ = assess(monkeypatch, [answer], frozen_assessment=frozen)
    assert calls == [frozen["frozen_judge_messages"]]
    assert state_eval.validate_reply_state_assessment(result, evidence=current, trace_id="classification-call")


def test_auxiliary_judge_rejects_tampered_frozen_requests_without_calling_model(monkeypatch, tmp_path):
    with retry_window(monkeypatch, tmp_path, 0):
        frozen, _, _ = assess(monkeypatch, [typed_timeout() for _ in range(4)])
    frozen["frozen_judge_messages"][-1]["content"] = "tampered"
    with retry_window(monkeypatch, tmp_path, 1):
        result, calls, _ = assess(monkeypatch, [], frozen_assessment=frozen)
    assert calls == [] and result["error"] and "eval_inconclusive" not in result


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
