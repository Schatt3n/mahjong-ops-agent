"""扩展受控审查评测的真实准备、证据保存与失败传播；替身仅检查机制，不宣称语义通过。"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from mahjong_agent_runtime import AgentLLMConfig


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "eval/golden/follow_up_commitment_review.jsonl"
CASE_IDS = [
    "review_open_game_short_ack", "review_open_game_service_offer", "review_open_game_future_notification",
    "review_scheduled_finite_recheck", "review_scheduled_but_fabricated_send", "review_cancelled_follow_up_future_notification",
]


def load_runner():
    """从当前源码副本装载同一运行器，避免环境安装指向另一仓库。"""
    spec = importlib.util.spec_from_file_location("follow_up_live_contract_test_runner", ROOT / "scripts/run_concurrency_eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ControlledReviewResponses:
    """按预声明序列提供测试输出，验证单次调用和完整失败证据，不模拟模型语义能力。"""

    def __init__(self, cases, fault=None):
        self.config = AgentLLMConfig(api_key="review-test-credential", model="unit-review", base_url="https://example.invalid")
        self.cases = cases
        self.fault = fault
        self.calls = []

    def complete(self, messages, *, trace_id, timeout_seconds):
        index = len(self.calls)
        self.calls.append(messages)
        case = self.cases[index]
        payload = json.loads(messages[1]["content"])
        text = payload["review_items"][0]["text"]
        assert text == case["text"]
        if index == 10 and self.fault == "provider":
            raise RuntimeError("controlled provider unavailable")
        if index == 10 and self.fault == "contract":
            return "not-json"
        approved = case["expected_approved"]
        if index == 10 and self.fault == "wrong_valid_verdict":
            approved = not approved
        violations = [] if approved else ["unsupported_follow_up_commitment"]
        return json.dumps({"approved": approved, "needs_human": False, "reasoning_summary": "仅验证评测机制",
            "violations": violations, "item_reviews": [{"item_id": "reply_to_user", "approved": approved,
                "suggested_safe_text": text if approved else "好。", "reasoning_summary": "仅验证评测机制", "violations": violations}]}, ensure_ascii=False)


def full_cases(module):
    return module.load_customer_visible_review_contract_cases() + module.load_follow_up_commitment_review_cases()


def test_follow_up_dataset_keeps_six_independent_fixed_contrasts():
    """旧八例另存原样，新资产不嵌造工具结果、持久状态或审查响应。"""
    cases = load_runner().load_follow_up_commitment_review_cases()
    assert [case["case_id"] for case in cases] == CASE_IDS
    assert [case["expected_approved"] for case in cases] == [True, True, False, True, False, False]
    assert [case["fixture"] for case in cases] == ["open_game_empty_candidates"] * 3 + ["open_game_follow_up"] * 2 + ["open_game_cancelled_follow_up"]
    assert cases[2]["text"] == "行，我帮你留意着，有合适的叫你。"
    assert cases[2]["source"]["verbatim_observed_text"] is True
    assert cases[2]["source"]["origin_scenario_id"] == "custom_rules_infer_base_and_cap"
    for case in cases:
        assert set(case) == {"case_id", "text", "expected_approved", "fixture", "requirement", "source"}
        assert set(case).isdisjoint({"tool_results", "state", "persisted_state", "raw_response"})
    assert all(value not in DATASET.read_text() for value in ["Bearer ", "/Users/", "/var/folders/"])


def test_extended_review_scope_uses_real_tools_and_never_sends_expected_labels(tmp_path):
    """同一生产审查入口覆盖完整十四例，前置事实由真实网关生成，审查不改业务状态。"""
    module = load_runner()
    cases = full_cases(module)
    client = ControlledReviewResponses(cases)
    report = module.run_live_customer_visible_review_contract(client, SimpleNamespace(timeout_seconds=1, max_tokens_per_call=64000),
        tmp_path / "fourteen", True)
    assert report["status"] == "passed"
    assert report["case_count"] == report["passed_count"] == 14
    assert len(client.calls) == 14
    assert report["source_ids"] == [case["case_id"] for case in cases]
    assert report["dataset_paths"] == ["eval/golden/customer_visible_review_contract.jsonl", "eval/golden/follow_up_commitment_review.jsonl"]
    for index, (case, actual, messages) in enumerate(zip(cases, report["reports"], client.calls)):
        payload = json.loads(messages[1]["content"])
        serialized = json.dumps(payload, ensure_ascii=False)
        assert all(field not in serialized for field in ["expected_approved", "case_id", "source_id", "fixture", "contrast", "origin_scenario_id"])
        assert case["case_id"] not in serialized
        assert payload["review_items"][0]["text"] == case["text"]
        assert actual["raw_approved"] is actual["final_approved"] is case["expected_approved"]
        assert actual["business_state"]["before"] == actual["business_state"]["after"]
        raw = json.loads(Path(actual["raw_trace_path"]).read_text())
        assert raw["source_id"] == raw["case_id"] == case["case_id"]
        assert raw["trace_id"] == actual["trace_id"]
        assert sum(event["step"] == "customer_visible_content_review_prompt" for event in raw["events"]) == 1
        assert sum(event["step"] == "customer_visible_content_review_response" for event in raw["events"]) == 1
        actual_tools = [event["content"] for event in raw["events"] if event["step"] == "tool_result"
            and event["content"]["name"] != "customer_visible_content_review"]
        assert actual["prerequisite_results"] == actual_tools
        assert all(result["called"] and result["allowed"] and not result["error"] for result in actual_tools)
        if index < 8:
            assert actual["fixture"] == "empty_inventory"
            assert actual["asset_path"] == report["dataset_paths"][0]
            assert [item["name"] for item in actual_tools] == ["search_current_games"]
            assert actual["tool_result"]["result"]["matches"] == []
            assert all(value == 0 for value in actual["persisted_state"].values())
            assert payload["active_follow_up_summaries"] == []
            continue
        assert actual["asset_path"] == report["dataset_paths"][1]
        expected_tools = ["create_game", "search_customers"]
        if case["fixture"] != "open_game_empty_candidates":
            expected_tools.append("schedule_follow_up")
        if case["fixture"] == "open_game_cancelled_follow_up":
            expected_tools.append("cancel_follow_up")
        assert [item["name"] for item in actual_tools] == expected_tools
        assert actual["tool_result"]["name"] == "search_customers"
        assert actual["tool_result"]["result"]["candidates"] == []
        assert actual["persisted_state"]["game_count"] == 1
        assert actual["persisted_state"]["invite_count"] == actual["persisted_state"]["outbound_count"] == 0
        state = actual["business_state"]["after"]
        assert state["waiting_count"] == 0
        assert [task["task_id"] for task in state["scheduled_tasks"]] == sorted(task["task_id"] for task in state["scheduled_tasks"])
        if case["fixture"] == "open_game_follow_up":
            assert state["scheduled_count"] == 1
            assert [task["status"] for task in state["scheduled_tasks"]] == ["pending"]
            assert len(payload["active_follow_up_summaries"]) == 1
            assert payload["active_follow_up_summaries"][0]["task_id"] == state["scheduled_tasks"][0]["task_id"]
        else:
            assert payload["active_follow_up_summaries"] == []
            assert state["scheduled_count"] == (2 if case["fixture"] == "open_game_cancelled_follow_up" else 0)
            assert all(task["status"] == "cancelled" for task in state["scheduled_tasks"])


@pytest.mark.parametrize("fault", ["provider", "contract", "wrong_valid_verdict"])
def test_follow_up_review_failure_does_not_retry_or_disappear(tmp_path, fault):
    """传输、格式和有效但错误判决都仅审一次并使完整集合失败。"""
    module = load_runner()
    client = ControlledReviewResponses(full_cases(module), fault=fault)
    report = module.run_live_customer_visible_review_contract(client, SimpleNamespace(timeout_seconds=1, max_tokens_per_call=64000),
        tmp_path / fault, True)
    assert report["status"] == "failed"
    assert report["failed_count"] == 1 and report["passed_count"] == 13
    assert len(client.calls) == len(report["reports"]) == 14
    actual = report["reports"][10]
    assert actual["status"] == "failed"
    assert sum(step == "customer_visible_content_review_prompt" for step in actual["trace_steps"]) == 1
    if fault == "wrong_valid_verdict":
        assert actual["raw_approved"] is actual["final_approved"] is True
        assert actual["expected_approved"] is False


@pytest.mark.parametrize("change", ["duplicate_id", "unknown_fixture", "missing_field", "string_label"])
def test_follow_up_asset_rejects_invalid_structure_before_model_call(tmp_path, change):
    """夹具类型和期望类型在调用前固定，不能用宽松资产绕过应执行的准备步骤。"""
    module = load_runner()
    cases = module.load_follow_up_commitment_review_cases()
    if change == "duplicate_id":
        cases[-1]["case_id"] = cases[0]["case_id"]
    elif change == "unknown_fixture":
        cases[0]["fixture"] = "invented_success"
    elif change == "missing_field":
        del cases[0]["requirement"]
    else:
        cases[0]["expected_approved"] = "true"
    path = tmp_path / "invalid.jsonl"
    path.write_text("\n".join(json.dumps(case, ensure_ascii=False) for case in cases))
    with pytest.raises(ValueError):
        module.load_follow_up_commitment_review_cases(path)
