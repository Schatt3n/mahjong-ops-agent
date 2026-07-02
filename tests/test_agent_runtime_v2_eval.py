from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_agent_runtime_v2_eval.py"
DATASET = ROOT / "eval" / "regression" / "agent_runtime_v2_regression.jsonl"


def load_eval_module():
    spec = importlib.util.spec_from_file_location("run_agent_runtime_v2_eval", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_agent_runtime_v2_regression_dataset_loads_and_has_unique_ids() -> None:
    module = load_eval_module()
    scenarios = module.load_scenarios(DATASET)
    ids = [scenario.id for scenario in scenarios]

    assert len(scenarios) == 7
    assert len(ids) == len(set(ids))
    assert module.count_checks(scenarios) == 81


def test_agent_runtime_v2_regression_dataset_passes() -> None:
    module = load_eval_module()
    scenarios = module.load_scenarios(DATASET)
    passed = 0
    failed = 0
    for scenario in scenarios:
        scenario_passed, scenario_failed, errors = module.run_scenario(scenario)
        assert errors == []
        passed += scenario_passed
        failed += scenario_failed

    assert passed == 81
    assert failed == 0
