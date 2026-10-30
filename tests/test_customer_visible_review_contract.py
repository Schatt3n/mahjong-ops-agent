"""受控审查评测复用真实网关、SQLite和生产审查器，模型替身仅验证调用合同。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from mahjong_agent_runtime import AgentLLMConfig

ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    spec = importlib.util.spec_from_file_location("concurrency_review_contract_test", ROOT / "scripts" / "run_concurrency_eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ReviewClient:
    """确定性审查响应只测试事实投影和失败传播，不能称为真实语义通过。"""
    def __init__(self, cases, *, fault=None):
        self.config = AgentLLMConfig(api_key="review-unit-credential", model="unit-model", base_url="https://example.invalid")
        self.verdicts = {case["text"]: case["expected_approved"] for case in cases}
        self.fault = fault
        self.calls = []

    def complete(self, messages, *, trace_id, timeout_seconds):
        self.calls.append(messages)
        payload = json.loads(messages[1]["content"])
        text = payload["review_items"][0]["text"]
        if len(self.calls) == 1 and self.fault == "provider":
            raise RuntimeError("offline provider refusal")
        if len(self.calls) == 1 and self.fault == "contract":
            return "not-json"
        approved = not self.verdicts[text] if self.fault == "verdict" else self.verdicts[text]
        violations = [] if approved else ["unsupported_follow_up_commitment"]
        return json.dumps({"approved": approved, "needs_human": False, "reasoning_summary": "仅测试替身",
            "violations": violations, "item_reviews": [{"item_id": "reply_to_user", "approved": approved,
                "suggested_safe_text": text if approved else "下午暂时没现成的。", "reasoning_summary": "逐项测试替身",
                "violations": violations}]}, ensure_ascii=False)


def args():
    return SimpleNamespace(timeout_seconds=1, max_tokens_per_call=64000)


def test_controlled_review_cases_use_actual_sqlite_read_and_never_send_expected_labels(tmp_path):
    module = load_runner()
    cases = module.load_customer_visible_review_contract_cases()
    client = ReviewClient(cases)
    report = module.run_live_customer_visible_review_contract(client, args(), tmp_path / "review-contract")
    assert report["status"] == "passed" and report["case_count"] == report["passed_count"] == 8
    assert report["source_ids"] == [case["case_id"] for case in cases]
    assert len(client.calls) == 8
    for case, actual, messages in zip(cases, report["reports"], client.calls):
        payload = json.loads(messages[1]["content"])
        serialized = json.dumps(payload, ensure_ascii=False)
        assert not any(label in serialized for label in ["expected_approved", "source_ids", "contrast", "observed_failure_regression"])
        assert case["case_id"] not in serialized
        assert payload["review_items"][0]["text"] == case["text"]
        assert payload["turn_tool_evidence"][0]["name"] == "search_current_games"
        assert payload["turn_tool_evidence"][0]["called"] is True
        assert all(payload["turn_tool_evidence"][0]["result"]["requirement"].get(key) == value for key, value in case["requirement"].items())
        assert payload["active_follow_up_summaries"] == []
        assert "按完整句子的疑问或条件作用域" in messages[0]["content"]
        assert payload["customer_service_review_contract"]["proposal_scope_is_whole_utterance"] is True
        assert payload["customer_service_review_contract"]["separate_assertions_need_independent_evidence"] is True
        assert payload["customer_service_review_contract"]["commitment_evidence_requires_accepted_responsibility"] is True
        order = payload["customer_service_review_contract"]["communication_act_review_order"]
        assert len(order) == 3 and "作用域" in order[0] and "未承接的服务提议" in order[1] and "最后" in order[2]
        assert "仅在已承接责任却无安排时" in messages[0]["content"]
        assert "业务事件触发条件本身不是客户接受条件" in messages[0]["content"]
        assert "无任务证据不能反向把第一类服务提议判成第二类承诺" in messages[0]["content"]
        assert "需不需要替你打听，消息有了再告诉你？" in messages[0]["content"]
        assert "我也会持续替你盯着" in messages[0]["content"]
        assert any("独立的已执行声明或无条件承诺" in rule for rule in payload["semantic_fidelity_contract"]["rules"])
        assert actual["raw_approved"] is actual["final_approved"] is case["expected_approved"]
        assert json.loads(actual["raw_response"])["approved"] is case["expected_approved"]
        raw = json.loads(Path(actual["raw_trace_path"]).read_text())
        assert raw["source_id"] == raw["case_id"] == case["case_id"]
        assert raw["trace_id"] == actual["trace_id"]
        assert [event["step"] for event in raw["events"]] == actual["trace_steps"]
        assert len([event for event in raw["events"] if event["step"] == "customer_visible_content_review_response"]) == 1
        assert actual["tool_result"]["called"] and actual["tool_result"]["allowed"]
        assert actual["tool_result"]["result"]["matches"] == []
        assert all(value == 0 for value in actual["persisted_state"].values())
    assert client.config.api_key not in json.dumps(report)


@pytest.mark.parametrize("fault,failed_count", [("provider", 1), ("contract", 1), ("verdict", 8)])
def test_review_failure_or_wrong_valid_verdict_is_not_retried_or_hidden(tmp_path, fault, failed_count):
    module = load_runner()
    client = ReviewClient(module.load_customer_visible_review_contract_cases(), fault=fault)
    report = module.run_live_customer_visible_review_contract(client, args(), tmp_path / fault)
    assert report["status"] == "failed" and report["failed_count"] == failed_count
    assert len(client.calls) == len(report["reports"]) == len(report["source_ids"]) == 8
    if fault == "provider":
        assert "customer_visible_content_review_error" in report["reports"][0]["trace_steps"]
    if fault == "contract":
        assert "customer_visible_content_review_contract_error" in report["reports"][0]["trace_steps"]
    assert report["reports"][0]["status"] == "failed"


def test_review_fixture_validation_fails_before_any_model_call(tmp_path):
    module = load_runner()
    cases = module.load_customer_visible_review_contract_cases()
    cases[-1]["case_id"] = cases[0]["case_id"]
    path = tmp_path / "duplicate.jsonl"
    path.write_text("\n".join(json.dumps(case, ensure_ascii=False) for case in cases))
    with pytest.raises(ValueError, match="unique ids"):
        module.load_customer_visible_review_contract_cases(path)
