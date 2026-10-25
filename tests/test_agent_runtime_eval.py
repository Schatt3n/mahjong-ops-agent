from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_agent_runtime_eval.py"
DATASET = ROOT / "eval" / "regression" / "agent_runtime_regression.jsonl"


def load_eval_module():
    spec = importlib.util.spec_from_file_location("run_agent_runtime_eval", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_agent_runtime_regression_dataset_loads_and_has_unique_ids() -> None:
    module = load_eval_module()
    scenarios = module.load_scenarios(DATASET)
    ids = [scenario.id for scenario in scenarios]

    assert len(scenarios) == 9
    assert len(ids) == len(set(ids))
    assert module.count_checks(scenarios) == 139
    assert all(scenario.id.startswith("runtime_") for scenario in scenarios)


def test_agent_runtime_regression_dataset_passes() -> None:
    module = load_eval_module()
    scenarios = module.load_scenarios(DATASET)
    passed = 0
    failed = 0
    for scenario in scenarios:
        scenario_passed, scenario_failed, errors = module.run_scenario(scenario)
        assert errors == []
        passed += scenario_passed
        failed += scenario_failed

    assert passed == 139
    assert failed == 0


def test_regression_client_resolves_compacted_evidence_before_binding_real_tool_ids() -> None:
    module = load_eval_module()
    client = module.RegressionAgentClient([])
    payload = {
        "previous_tool_results": [{"name": "search_customers", "result_reference": {"source": "turn_tool_evidence", "index": 1}}],
        "turn_tool_evidence": [
            {"name": "other", "result": {}},
            {"name": "search_customers", "result": {"game": {"game_id": "actual-game-id"},
              "candidates": [{"customer": {"customer_id": "actual-customer-id", "display_name": "测试客户"}}]}},
        ],
    }
    client._collect_bindings([{"role": "user", "content": json.dumps(payload)}])
    assert client.bindings["$last_game_id"] == "actual-game-id"
    assert client.bindings["$first_candidate_customer_id"] == "actual-customer-id"
