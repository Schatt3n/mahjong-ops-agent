"""评测建议按已知影响给出定向入口，未知变化留给人工核对。"""

from __future__ import annotations

from pathlib import Path
import shlex
import subprocess

import pytest


@pytest.fixture
def impact(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import eval_impact

    return eval_impact


def test_change_aware_layers_keep_semantic_judge_and_runtime_separate(impact):
    judge_change = "scripts/semantic_judge_context.py"
    files, commands = impact.assess_eval_impact([judge_change])
    assert files["独立判定"] == {judge_change}
    assert not files["主 Agent 真模型"]
    assert commands["独立判定"] == {
        f"{impact.SELECT} reply_semantic_calibration",
        f"{impact.SELECT} reply_scope_calibration",
    }
    assert any("test_reply_semantic_judge.py" in command for command in commands["定向离线"])

    prompt_change = impact.RUNTIME + "prompts/agent_runtime_system.md"
    files, commands = impact.assess_eval_impact([judge_change, prompt_change])
    assert files["主 Agent 真模型"] == {prompt_change}
    assert commands["主 Agent 真模型"] == {
        f"{impact.SELECT} inquiry_grounding_six_cases",
        f"{impact.SELECT} real_owner_all_scenarios_with_reply_semantics",
    }
    assert files["独立判定"] == {judge_change}
    assert files["定向离线"] == {judge_change, prompt_change}
    assert not files["文档校验"]


def test_dataset_suggestions_use_current_source_specific_scopes(impact):
    group = "eval/golden/real_group_chat_20260722.jsonl"
    group_topic = "eval/golden/group_topic_recovery_20261105.jsonl"
    judge = "eval/golden/reply_scope_calibration_revisions.jsonl"
    files, commands = impact.assess_eval_impact([group, group_topic, judge])
    assert files["主 Agent 真模型"] == {group, group_topic}
    assert files["独立判定"] == {judge}
    assert f"{impact.SELECT} group_flow_deterministic_20260722" in commands["定向离线"]
    assert commands["主 Agent 真模型"] == {
        f"{impact.SELECT} real_group_all_scenarios_20260722",
        f"{impact.SELECT} group_topic_recovery_20261105",
    }
    assert commands["独立判定"] == {f"{impact.SELECT} reply_scope_calibration"}


def test_reviewer_prompt_can_be_verified_without_rerunning_main_agent(impact):
    """仅对客审查合同改动时，给出审查专项与离线边界，而不启动主模型。"""
    prompt = impact.RUNTIME + "prompts/agent_runtime_reply_self_review.md"
    files, commands = impact.assess_eval_impact([prompt])
    assert files["对客审查"] == {prompt}
    assert not files["主 Agent 真模型"] and not files["独立判定"]
    assert any("test_action_attribution.py" in command for command in commands["定向离线"])


def test_docs_tests_and_shared_risk_do_not_request_live_runs(impact):
    test = "tests/test_eval_impact.py"
    unknown = impact.RUNTIME + "prompts/unknown.md"
    store = impact.RUNTIME + "stores/sqlite/schema.py"
    paths = ["README.md", test, unknown, store, ".github/workflows/live-eval.yml"]
    files, commands = impact.assess_eval_impact(paths)
    assert files["文档校验"] == {"README.md"}
    assert commands["文档校验"] == {"git diff --check", "git diff --cached --check"}
    assert files["需人工核对"] == {store, unknown, ".github/workflows/live-eval.yml"}
    assert any(test in shlex.split(command) for command in commands["定向离线"])
    assert f"{impact.SELECT} concurrency_deterministic" in commands["定向离线"]
    assert not commands["独立判定"] and not commands["主 Agent 真模型"]
    assert not any("--full" in command for group in commands.values() for command in group)


def test_git_impact_keeps_index_worktree_and_rename_paths_without_ignored_files(impact, tmp_path):
    def git(*args):
        return subprocess.check_output(("git", *args), cwd=tmp_path, stderr=subprocess.DEVNULL).decode().strip()

    git("init")
    git("config", "user.name", "Eval Impact Test")
    git("config", "user.email", "impact@example.invalid")
    (tmp_path / ".gitignore").write_text(".env\nruntime_data/\n")
    (tmp_path / "staged").write_text("base")
    (tmp_path / "renamed old").write_text("rename me")
    git("add", ".")
    git("-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", "commit", "-m", "fixture")
    (tmp_path / "staged").write_text("in the index")
    git("add", "staged")
    (tmp_path / "staged").write_text("base")
    git("mv", "renamed old", "renamed new")
    (tmp_path / "untracked").write_text("only in worktree")
    (tmp_path / ".env").write_text("ignored")
    (tmp_path / "runtime_data").mkdir()
    (tmp_path / "runtime_data" / "report.json").write_text("{}")
    assert impact.changed_paths(tmp_path, "HEAD") == (
        git("rev-parse", "HEAD"), ["renamed new", "renamed old", "staged", "untracked"],
    )

    with pytest.raises(ValueError, match="无法解析 Git 基线"):
        impact.changed_paths(tmp_path, "--not-an-option")


def test_print_impact_is_advisory_and_leaves_reports_untouched(impact, monkeypatch, tmp_path, capsys):
    relative_script = "scripts/replay_archived_owner_judge.py"
    script = tmp_path / relative_script
    script.parent.mkdir()
    script.write_text("do not execute")
    report_path = tmp_path / "runtime_data" / "prior.json"
    report_path.parent.mkdir()
    report_path.write_text("previous evidence")
    before = set(tmp_path.rglob("*"))
    monkeypatch.setattr(impact, "changed_paths", lambda root, base: ("abc", [relative_script]))

    impact.print_eval_impact(tmp_path, "HEAD")

    output = capsys.readouterr().out
    assert "abc" in output and "独立判定" in output
    assert "--scope reply_semantic_calibration" in output
    assert "配置" in output and "显式 --full" in output
    assert set(tmp_path.rglob("*")) == before
    assert report_path.read_text() == "previous evidence"



def test_mixed_runners_and_deleted_targets_are_flagged_for_review(impact):
    mixed_runners = {"scripts/run_inquiry_grounding_eval.py", "scripts/run_real_owner_chat_live_eval.py"}
    removed = frozenset({"tests/test_removed.py", "eval/golden/real_group_chat_20260722.jsonl"})
    files, commands = impact.assess_eval_impact(sorted(mixed_runners | removed), removed_paths=removed)
    assert files["需人工核对"] == mixed_runners | removed
    assert files["主 Agent 真模型"] == {"scripts/run_real_owner_chat_live_eval.py"}
    assert files["独立判定"] == {"scripts/run_inquiry_grounding_eval.py"}
    assert not any(
        "test_removed.py" in command or "20260722" in command for group in commands.values() for command in group
    )


def test_suggested_tests_and_scopes_exist_for_current_checkout(impact):
    import run_evals

    root = Path(__file__).resolve().parents[1]
    paths = [path.relative_to(root).as_posix() for path in (root / "eval" / "golden").glob("*.jsonl")]
    paths.extend(pattern.replace("*", "sample") for rule in impact.RULES for pattern in rule.paths)
    _, commands = impact.assess_eval_impact(paths)
    for group in commands.values():
        for command in group:
            if command.startswith(impact.PYTEST):
                assert all((root / path).exists() for path in shlex.split(command)[5:])

    full_args = run_evals.argparse.Namespace(
        full=True, scope=None, live_real_owner=False, live_real_group=False, live_concurrency=False,
    )
    expected = {phase["scope"] for phase in run_evals.build_eval_plan(full_args, root / "unwritten")}
    # 局部门禁还支持全量计划中按日期展开的两个默认入口。
    for name in ("group_flow_deterministic", "real_group_all_scenarios"):
        options = run_evals.argparse.Namespace(
            full=False, scope=name, live_real_owner=False, live_real_group=False, live_concurrency=False, case=None,
        )
        expected.update(phase["scope"] for phase in run_evals.build_eval_plan(options, root / "unwritten"))
    suggested = {command.split()[-1] for group in commands.values() for command in group if command.startswith(impact.SELECT)}
    assert suggested <= expected
