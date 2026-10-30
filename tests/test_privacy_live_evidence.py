"""隐私运行器的失败原因和原始轨迹必须能回溯，复跑不能覆盖旧证据。"""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def test_provider_failure_keeps_complete_original_trace_across_replays(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/run_privacy_isolation_live_eval.py"
    spec = importlib.util.spec_from_file_location("privacy_evidence_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    args = SimpleNamespace(db_path=tmp_path / "privacy.sqlite3", max_tokens_per_call=64000,
        max_calls_per_turn=8, max_steps=4, timeout_seconds=1)
    probe = module.load_privacy_probes(module.DEFAULT_CASES_PATH)[0]

    class FailingClient:
        def __init__(self, marker):
            self.marker = marker

        def complete(self, *args, **kwargs):
            raise RuntimeError(self.marker)

    first = module.run_case(FailingClient("provider_failure_first"), args, probe)
    original = Path(first["raw_trace_path"]).read_bytes()
    second = module.run_case(FailingClient("provider_failure_second"), args, probe)
    assert first["status"] == second["status"] == "failed"
    assert first["raw_trace_path"] != second["raw_trace_path"]
    assert Path(first["raw_trace_path"]).read_bytes() == original
    for report, marker in ((first, "provider_failure_first"), (second, "provider_failure_second")):
        raw = json.loads(Path(report["raw_trace_path"]).read_text())
        errors = [event for event in raw["events"] if event["step"] == "llm_error"]
        assert errors and marker in json.dumps(errors)
        assert marker in json.dumps(report["decision_trace"])
        assert report["checks"]["no_model_or_contract_error"] is False
