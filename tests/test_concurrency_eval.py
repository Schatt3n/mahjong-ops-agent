from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from mahjong_agent_runtime import AgentLLMConfig, SQLiteAgentStore


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "run_concurrency_eval.py"
    spec = importlib.util.spec_from_file_location("run_concurrency_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_deterministic_concurrency_suite_preserves_business_invariants(tmp_path) -> None:
    module = load_module()

    results = module.run_deterministic_suite(tmp_path, operations=12, workers=6)

    assert len(results) == 8
    assert all(result.passed for result in results), [result.to_dict() for result in results if not result.passed]
    by_name = {result.name: result for result in results}
    assert by_name["duplicate_message_idempotency"].checks[1]["actual"] == 1
    assert by_name["parallel_conversation_isolation"].metrics["model_max_concurrency"] >= 2
    assert by_name["last_seat_race"].checks[0]["actual"] == 1
    shared_race = by_name["shared_participant_first_ready_wins_race"]
    assert shared_race.checks[1]["actual"] == 1
    assert shared_race.checks[2]["actual"] == 1
    assert shared_race.checks[4]["actual"] == 11
    evidence = shared_race.metrics["evidence"]
    assert evidence["test_kind"] == "deterministic_sqlite_concurrency"
    assert evidence["production_entrypoint"] == "SQLiteAgentStore.record_candidate_reply"
    assert len(evidence["fixture"]["initial_games"]) == 12
    assert len(evidence["simulated_candidate_replies"]) == 12
    assert evidence["outcome"]["winner"]["status"] == "ready"
    assert len(evidence["outcome"]["losers"]) == 11
    assert evidence["outcome"]["released_shared_participation_count"] == 11
    assert evidence["state_transitions"]
    assert by_name["room_inventory_race"].checks[0]["actual"] == 1
    assert by_name["duplicate_invite_race"].checks[0]["actual"] == 1


def test_sqlite_trace_context_lookup_waits_for_store_lock(tmp_path) -> None:
    store = SQLiteAgentStore(tmp_path / "trace_context_lock.sqlite3")
    lock_acquired = threading.Event()
    release_lock = threading.Event()

    def hold_store_lock() -> None:
        with store._lock:
            lock_acquired.set()
            assert release_lock.wait(timeout=2)

    with ThreadPoolExecutor(max_workers=2) as pool:
        holder = pool.submit(hold_store_lock)
        assert lock_acquired.wait(timeout=2)
        lookup = pool.submit(store._task_context_id_for_trace, "conversation", "trace")

        time.sleep(0.05)
        assert lookup.done() is False

        release_lock.set()
        holder.result(timeout=2)
        assert lookup.result(timeout=2) == ""


def test_observed_client_transparently_exposes_configuration_without_a_setter() -> None:
    module = load_module()
    configuration = object()
    observed = module.ObservedLLMClient(SimpleNamespace(config=configuration))

    assert observed.config is configuration
    with pytest.raises(AttributeError):
        observed.config = object()
    assert observed.config is configuration
    assert observed.metrics()["measurement_scope"] == "shared_runtime_client_only"
    assert observed.metrics()["independent_semantic_judge_included"] is False


@pytest.mark.parametrize("semantic_passed", [True, False])
def test_live_suite_keeps_golden_identity_and_runs_independent_judgement_with_delegate_configuration(
    tmp_path, monkeypatch, semantic_passed,
) -> None:
    """用进程内替身检验调用边界，不发网络请求，也不把替身判定声明为真实模型结果。"""
    module = load_module()
    import run_inquiry_grounding_eval as semantic_eval
    import run_real_owner_chat_live_eval as owner_eval

    config = AgentLLMConfig(
        api_key="unit-test-credential", model="unit-test-model", base_url="https://example.invalid",
        max_concurrency=2,
    )
    barrier = threading.Barrier(2)

    class PrimaryClient:
        def __init__(self):
            self.config = config

        def complete(self, messages, *, trace_id, timeout_seconds):
            # 两个真实工作线程同步抵达，证明共享客户端的重叠指标仍只量运行时请求。
            barrier.wait(timeout=3)
            return "还没现成的，要我帮你问问有没有人吗？"

    judge_configs = []
    expectations = []

    class IndependentJudge:
        def __init__(self, *, config):
            judge_configs.append(config)

        def complete(self, messages, **kwargs):
            expectation = json.loads(messages[1]["content"])["expectation"]
            criteria = [
                {"id": item["id"], "violation_detected": not semantic_passed, "reason": "仅测试输出合同和失败传播"}
                for item in expectation["checks"]
            ]
            return json.dumps({"reason": "进程内判定替身", "criteria": criteria})

    monkeypatch.setattr(module.OpenAICompatibleAgentClient, "from_env", classmethod(lambda cls: PrimaryClient()))
    monkeypatch.setattr(semantic_eval, "OpenAICompatibleAgentClient", IndependentJudge)
    chosen_ids = {"profile_default_matched_game", "flexible_time_empty_pool_offers_help"}
    scenarios = [scenario for scenario in owner_eval.live_eval_scenarios() if scenario.scenario_id in chosen_ids]
    monkeypatch.setattr(owner_eval, "live_eval_scenarios", lambda: scenarios)

    def run_scenario(client, args, scenario, *, iteration=1):
        assert client.config is config
        reply = client.complete([], trace_id=f"primary_{scenario.scenario_id}_{iteration}", timeout_seconds=1)
        expectation = owner_eval.reply_semantic_expectation(scenario)
        expectations.append((scenario.scenario_id, expectation["golden_sources"]))
        assessment = owner_eval.assess_reply_semantics(
            client, evidence={"reply": reply}, expectation=expectation,
            trace_id=f"judge_{scenario.scenario_id}_{iteration}", timeout_seconds=1,
        )
        db_path = owner_eval.scenario_db_path(args.db_path, scenario.scenario_id)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # 原 scenario_id、轮次与实际 owner runner 的路径规则共同决定唯一证据位置。
        assert not db_path.exists()
        db_path.write_text("isolated unit-test fixture", encoding="utf-8")
        return {
            "scenario_id": scenario.scenario_id, "iteration": iteration, "db_path": str(db_path),
            "status": "passed" if assessment["passed"] and not assessment["error"] else "failed",
            "semantic_assessment": assessment,
            "recovery_metrics": owner_eval.assess_action_contract_recovery([], runtime_status="waiting_user",
                final_reply=reply, business_checks_passed=True, semantic_passed=assessment["passed"]),
        }

    monkeypatch.setattr(owner_eval, "run_scenario", run_scenario)
    monkeypatch.setattr(module, "run_live_active_input_revision", lambda *args: {"status": "passed", "case_id": module.load_active_revision_fixture()["id"]})
    monkeypatch.setattr(module, "run_live_customer_visible_review_contract", lambda *args: {"status": "passed", "case_count": 8})
    report = module.run_live_suite(
        tmp_path, workers=2, repeats=2, timeout_seconds=1, max_steps=3,
        max_calls_per_turn=3, max_tokens_per_call=1000, skip_review=False, skip_text_generation=False,
    )

    assert report["scenario_count"] == 4
    assert report["status"] == ("passed" if semantic_passed else "failed"), [
        item["semantic_assessment"].get("error") for item in report["reports"]
    ]
    assert report["passed_count"] == (4 if semantic_passed else 0)
    assert report["failed_count"] == (0 if semantic_passed else 4)
    assert {(item["scenario_id"], item["iteration"]) for item in report["reports"]} == {
        (scenario_id, iteration) for scenario_id in chosen_ids for iteration in (1, 2)
    }
    assert len({item["execution_id"] for item in report["reports"]}) == 4
    assert len({item["db_path"] for item in report["reports"]}) == 4
    assert len(judge_configs) == 4
    assert all(copy is not config and copy.api_key == config.api_key for copy in judge_configs)
    assert all(copy.temperature == 0 and copy.response_format == "json_object" for copy in judge_configs)
    assert all(sources for _, sources in expectations)
    assert all(item["semantic_assessment"]["error"] is None for item in report["reports"]), [
        item["semantic_assessment"]["error"] for item in report["reports"]
    ]
    assert report["metrics"]["call_count"] == 4
    assert report["metrics"]["max_concurrent_client_calls"] == 2
    judge_metrics = report["metrics"]["independent_semantic_judge"]
    assert judge_metrics["attempt_count"] == 4
    assert judge_metrics["included_in_shared_runtime_metrics"] is False
    assert judge_metrics["shares_runtime_provider_semaphore"] is False
    assert judge_metrics["max_concurrent_case_bound"] == 2
    assert config.api_key not in json.dumps(report)
    assert report["recovery_summary"] == owner_eval.summarize_action_contract_recovery(report["reports"])
    assert report["recovery_summary"]["action_contract_error_count"] == 0


def test_active_revision_fixture_preserves_both_inputs_and_human_semantics():
    """改向回放独立保存输入和期望，不能从本轮主模型输出临时生成通过标准。"""
    module = load_module()
    fixture = module.load_active_revision_fixture()
    assert fixture["source"]["original_human_transcript"] is False
    assert fixture["messages"][0]["text"] == "红中1块，人齐开，我一个人，烟不限，帮我组一桌"
    assert fixture["messages"][1]["text"] == "先不组了，改成明天晚上七点国标2块，只看看有没有现成局"
    assert fixture["expected"]["latest_requirement"] == {"game_type": "national_standard_mahjong", "stake": "2"}
    assert fixture["expected"]["required_meaning"] and fixture["expected"]["forbidden_meaning"]


def test_live_revision_failure_blocks_the_whole_live_suite(tmp_path, monkeypatch):
    """即使所有旧场景通过，新增运行中改目标的失败也不能被旧通过率掩盖。"""
    module = load_module()
    import run_real_owner_chat_live_eval as owner_eval
    config = AgentLLMConfig(api_key="unit-test-credential", model="unit-test-model", base_url="https://example.invalid")
    primary = SimpleNamespace(config=config, complete=lambda *args, **kwargs: "unit-test-only")
    monkeypatch.setattr(module.OpenAICompatibleAgentClient, "from_env", classmethod(lambda cls: primary))
    monkeypatch.setattr(owner_eval, "live_eval_scenarios", lambda: [SimpleNamespace(scenario_id="stub")])
    def successful_case(client, *args, **kwargs):
        client.complete([], trace_id="unit-stub", timeout_seconds=1)
        return {"scenario_id": "stub", "iteration": 1, "status": "passed",
            "recovery_metrics": owner_eval.assess_action_contract_recovery([], runtime_status="waiting_user",
                final_reply="测试回复", business_checks_passed=True, semantic_passed=True)}
    monkeypatch.setattr(owner_eval, "run_scenario", successful_case)
    monkeypatch.setattr(module, "run_live_active_input_revision", lambda *args: {"status": "failed", "case_id": "active-revision"})
    monkeypatch.setattr(module, "run_live_customer_visible_review_contract", lambda *args: {"status": "passed", "case_count": 8})
    report = module.run_live_suite(tmp_path, workers=1, repeats=1, timeout_seconds=1, max_steps=3,
        max_calls_per_turn=3, max_tokens_per_call=1000, skip_review=False, skip_text_generation=False)
    assert report["passed_count"] == 1
    assert report["status"] == "failed"
    assert next(item for item in report["checks"] if item["name"] == "active_goal_revision_passed")["passed"] is False
    assert [item["name"] for item in report["checks"] if not item["passed"]] == ["active_goal_revision_passed"]


def test_live_review_contract_failure_blocks_whole_suite_without_hiding_existing_scenarios(tmp_path, monkeypatch):
    """审查对照失败独立阻断；原有case数量、语义结果和运行中改向证据不被替代。"""
    module = load_module()
    import run_real_owner_chat_live_eval as owner_eval
    config = AgentLLMConfig(api_key="unit-test-credential", model="unit-test-model", base_url="https://example.invalid")
    primary = SimpleNamespace(config=config, complete=lambda *args, **kwargs: "unit-test-only")
    monkeypatch.setattr(module.OpenAICompatibleAgentClient, "from_env", classmethod(lambda cls: primary))
    monkeypatch.setattr(owner_eval, "live_eval_scenarios", lambda: [SimpleNamespace(scenario_id="unchanged-stub")])
    def successful_case(client, *args, **kwargs):
        client.complete([], trace_id="unit-stub", timeout_seconds=1)
        return {"scenario_id": "unchanged-stub", "iteration": 1, "status": "passed", "recovery_metrics":
            owner_eval.assess_action_contract_recovery([], runtime_status="waiting_user", final_reply="测试回复",
                business_checks_passed=True, semantic_passed=True)}
    monkeypatch.setattr(owner_eval, "run_scenario", successful_case)
    monkeypatch.setattr(module, "run_live_active_input_revision", lambda *args: {"status": "passed", "case_id": "active-revision"})
    monkeypatch.setattr(module, "run_live_customer_visible_review_contract", lambda *args: {"status": "failed", "case_count": 8, "failed_count": 1})
    report = module.run_live_suite(tmp_path, workers=1, repeats=1, timeout_seconds=1, max_steps=3,
        max_calls_per_turn=3, max_tokens_per_call=1000, skip_review=False, skip_text_generation=False)
    assert report["scenario_count"] == report["passed_count"] == 1
    assert report["reports"][0]["scenario_id"] == "unchanged-stub"
    assert report["active_input_revision"]["status"] == "passed"
    assert report["customer_visible_review_contract"]["failed_count"] == 1
    assert report["status"] == "failed"
    assert [item["name"] for item in report["checks"] if not item["passed"]] == ["customer_visible_review_contract_passed"]
