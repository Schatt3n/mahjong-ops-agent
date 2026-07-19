from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_privacy_isolation_live_eval.py"


def load_runner_module():
    spec = importlib.util.spec_from_file_location("run_privacy_isolation_live_eval", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_privacy_adversarial_dataset_is_independent_and_valid() -> None:
    module = load_runner_module()
    probes = module.load_privacy_probes(module.DEFAULT_CASES_PATH)

    assert len(probes) == 10
    assert len({probe.case_id for probe in probes}) == len(probes)
    assert len({probe.attack_type for probe in probes}) == len(probes)
    assert all(probe.must_review for probe in probes)
    assert all(not probe.allow_human_fallback for probe in probes)
    assert all(probe.expected_behavior for probe in probes)


def test_privacy_adversarial_dataset_rejects_duplicate_case_ids(tmp_path: pathlib.Path) -> None:
    module = load_runner_module()
    duplicate = (
        '{"schema_version":1,"kind":"privacy_isolation_adversarial","case_id":"same",'
        '"attack_type":"first","input":"input one","expected":{"behavior":"safe",'
        '"must_review":true,"allow_human_fallback":false,"forbidden_output":[]}}\n'
        '{"schema_version":1,"kind":"privacy_isolation_adversarial","case_id":"same",'
        '"attack_type":"second","input":"input two","expected":{"behavior":"safe",'
        '"must_review":true,"allow_human_fallback":false,"forbidden_output":[]}}\n'
    )
    path = tmp_path / "duplicate.jsonl"
    path.write_text(duplicate, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate case_id"):
        module.load_privacy_probes(path)
