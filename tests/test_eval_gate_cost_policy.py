"""用本地子进程验证失败优先调度，确保每次成功验收都有本次完整执行证据。"""

import json
from pathlib import Path
import sys

import pytest

from scripts import run_evals as gate


GROUP = "real_group_all_scenarios"
SCOPES = ["runtime_boundary", "all_pytest", gate.LIVE_SCOPES[0], f"{GROUP}_20260722", f"{GROUP}_20260723"]


@pytest.fixture
def run_full_gate(tmp_path, monkeypatch):
    """用真实CLI调度本地替身子进程，保留每次新产生的报告与日志。"""
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    failing = set()
    archive = tmp_path / "runtime_data" / "eval_runs"

    def plan(args, run_dir):
        result = []
        for scope in SCOPES:
            phase = {"scope": scope, "status": "pending", "command": [
                sys.executable, "-c", "import sys; print(sys.argv[1]); sys.exit(int(sys.argv[2]))",
                scope, str(int(scope in failing)),
            ]}
            if scope.startswith(GROUP):
                phase["scope_family"] = GROUP
            result.append(phase)
        return result

    monkeypatch.setattr(gate, "build_eval_plan", plan)

    def run(failures=(), *, collect_all=False):
        failing.clear()
        failing.update(failures)
        flags = ["--full", "--collect-all"] if collect_all else ["--full"]
        monkeypatch.setattr(sys, "argv", ["run_evals.py", *flags])
        before = set(archive.glob("*/gate_report.json"))
        assert gate.main() == int(bool(failing))
        path, = set(archive.glob("*/gate_report.json")) - before
        return path, json.loads(path.read_text())

    return run


def test_full_gate_prioritizes_previous_failure_and_reexecutes_all_scopes(run_full_gate):
    """同族的不同数据集单独排序；通过历史只能影响顺序，不能免除任何执行。"""
    original_path, _ = run_full_gate([SCOPES[-1]])
    archive = original_path.parent.parent
    original_bytes = original_path.read_bytes()

    # 较新的局部运行、未完成记录或损坏的旁路报告都不能取代最近一次完整验收。
    for suffix, content in (("partial", '{"full":false,"status":"passed"}'),
                            ("running", '{"full":true,"status":"running"}'), ("torn", "{")):
        history_dir = archive / (original_path.parent.name + suffix)
        history_dir.mkdir()
        (history_dir / "gate_report.json").write_text(content)

    retry_path, retry = run_full_gate([SCOPES[-1]])
    assert retry["priority_source_run_id"] == original_path.parent.name
    assert retry["failed_scopes"] == [SCOPES[-1]]
    assert [(phase["scope"], phase["status"]) for phase in retry["scopes"]] == [
        ("runtime_boundary", "passed"), ("all_pytest", "passed"), (SCOPES[-1], "failed"),
        (SCOPES[2], "not_run"), (SCOPES[3], "not_run"),
    ]
    assert all("log_path" not in phase for phase in retry["scopes"] if phase["status"] == "not_run")

    passed_path, passed = run_full_gate()
    expected = [*SCOPES[:2], SCOPES[-1], *SCOPES[2:-1]]
    assert passed["failure_policy"] == "fail_fast"
    assert passed["priority_source_run_id"] == retry_path.parent.name
    assert passed["prioritized_live_scopes"] == [SCOPES[-1]]
    assert passed["passed_scopes"] == expected
    for phase in passed["scopes"]:
        log = Path(phase["log_path"])
        assert log.parent == passed_path.parent
        assert log.read_text().strip() == phase["scope"]
    assert original_path.read_bytes() == original_bytes

    _, after_success = run_full_gate()
    assert after_success["prioritized_live_scopes"] == []
    assert after_success["passed_scopes"] == SCOPES


def test_early_stop_retains_unchecked_failures_but_retires_repaired_priorities(run_full_gate):
    """原有失败在新尝试未覆盖时保留优先级，经本次实际复测通过后退出队首。"""
    owner, group_a, group_b = SCOPES[2:]
    _, census = run_full_gate([owner, group_a, group_b], collect_all=True)
    assert census["failed_scopes"] == [owner, group_a, group_b]

    _, partial = run_full_gate([group_a])
    assert partial["prioritized_live_scopes"] == [owner, group_a, group_b]
    assert partial["passed_scopes"] == [*SCOPES[:2], owner]
    assert partial["failed_scopes"] == [group_a]
    assert partial["scopes"][-1]["status"] == "not_run"

    _, offline_blocked = run_full_gate(["all_pytest"])
    assert offline_blocked["prioritized_live_scopes"] == [group_a, group_b]
    assert offline_blocked["failed_scopes"] == ["all_pytest"]

    _, full = run_full_gate()
    assert full["prioritized_live_scopes"] == [group_a, group_b]
    assert full["passed_scopes"] == [*SCOPES[:2], group_a, group_b, owner]


def test_collect_all_requires_explicit_full_gate(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_evals.py", "--collect-all"])
    with pytest.raises(SystemExit) as exc:
        gate.parse_args()
    assert exc.value.code == 2
