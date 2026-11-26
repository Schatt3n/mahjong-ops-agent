"""用 Git 文件变化生成评测建议，沿用现有测试和门禁范围。"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess


RUNTIME = "src/mahjong_agent_runtime/"
SELECT = "PYTHONPATH=src .venv/bin/python scripts/run_evals.py --scope"
OFFLINE = "PYTHONPATH=src .venv/bin/python scripts/run_evals.py"
PYTEST = "PYTHONPATH=src .venv/bin/python -m pytest -q"
LAYERS = ("文档校验", "定向离线", "主 Agent 真模型", "对客审查", "独立判定", "需人工核对")


@dataclass(frozen=True)
class ImpactRule:
    """把相关文件、验证层和已有入口绑定为一条评测建议。"""

    paths: tuple[str, ...]
    layer: str
    tests: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()

    def matches(self, path: str) -> bool:
        """按仓库相对路径匹配，允许同一变化影响多层。"""
        return any(fnmatchcase(path, pattern) for pattern in self.paths)


RULES = (
    ImpactRule(
        ("scripts/run_evals.py", "scripts/eval_gate_reporting.py", "scripts/eval_retry.py",
         "scripts/eval_run_state.py", "scripts/eval_impact.py", "tests/semantic_http_stub.py"),
        "定向离线", ("tests/test_eval_gate.py", "tests/test_eval_timeout_resume.py"),
    ),
    ImpactRule(
        ("scripts/semantic_judge_context.py", "scripts/run_inquiry_grounding_eval.py", "scripts/replay_archived_owner_judge.py"),
        "独立判定", ("tests/test_reply_semantic_judge.py", "tests/test_semantic_judge_reasoned_config.py"),
        ("reply_semantic_calibration", "reply_scope_calibration"),
    ),
    ImpactRule(
        ("scripts/run_reply_scope_calibration.py", "eval/golden/reply_scope*.jsonl"),
        "独立判定", ("tests/test_reply_scope_calibration.py",), ("reply_scope_calibration",),
    ),
    ImpactRule(
        ("scripts/run_empty_pool_calibration.py", "eval/golden/empty_pool_evidence_calibration_20261105.jsonl"),
        "独立判定", ("tests/test_inquiry_live_evidence.py",), ("empty_pool_evidence_calibration",),
    ),
    ImpactRule(
        ("scripts/run_recruitment_obligation_calibration.py", "eval/golden/recruitment_obligation_calibration_20261105.jsonl"),
        "独立判定", ("tests/test_eval_gate.py",), ("recruitment_obligation_calibration",),
    ),
    ImpactRule(
        ("eval/golden/reply_semantic_calibration.jsonl",),
        "独立判定", ("tests/test_reply_semantic_calibration.py",), ("reply_semantic_calibration",),
    ),
    ImpactRule(
        (RUNTIME + "action_attribution.py", RUNTIME + "prompts/agent_runtime_reply_self_review.md"),
        "对客审查", ("tests/test_action_attribution.py", "tests/test_customer_visible_review_contract.py"),
    ),
    ImpactRule(
        (RUNTIME + "customer_visible_*", RUNTIME + "visibility.py", RUNTIME + "review_config.py",
         RUNTIME + "prompts/customer_visible*"),
        "主 Agent 真模型", ("tests/test_customer_visible_contract.py", "tests/test_customer_visible_review_contract.py"),
        ("privacy_isolation_all_cases",),
    ),
    ImpactRule(
        (RUNTIME + "prompts/agent_runtime_system.md", RUNTIME + "domains/context_builders/*",
         RUNTIME + "domains/query_contract.py", RUNTIME + "services/context_service.py", "scripts/run_real_owner_chat_live_eval.py",
         "eval/golden/real_owner_chat*.jsonl"),
        "主 Agent 真模型", ("tests/test_context_transport_projection.py", "tests/test_real_owner_chat_golden_validator.py"),
        ("real_owner_all_scenarios_with_reply_semantics", "inquiry_grounding_six_cases"),
    ),
    ImpactRule(
        (RUNTIME + "summary.py", RUNTIME + "summary_evaluation.py",
         RUNTIME + "task_context.py", RUNTIME + "prompts/*summary*"),
        "定向离线", ("tests/test_context_summary_quality.py", "tests/test_agent_run_recovery.py"),
    ),
    ImpactRule(
        (RUNTIME + "group_chat/*", "scripts/run_real_group_chat_flow_eval.py"),
        "主 Agent 真模型", ("tests/test_group_chat", "tests/test_real_group_chat_dataset.py"),
        ("real_group_all_scenarios",),
    ),
    ImpactRule(
        (RUNTIME + "group_chat/*", "scripts/run_real_group_chat_flow_eval.py"),
        "定向离线", (), ("group_flow_deterministic",),
    ),
    ImpactRule(
        ("scripts/run_group_topic_regression_eval.py",), "主 Agent 真模型",
        ("tests/test_eval_gate_topic_scope.py",), ("group_topic_recovery_20261105",),
    ),
    ImpactRule(
        ("scripts/run_concurrency_eval.py",), "定向离线",
        ("tests/test_concurrency_eval.py",), ("concurrency_deterministic",),
    ),
    ImpactRule(
        ("eval/golden/active_input_revision*.jsonl", "eval/golden/query_reply_state_calibration.jsonl"),
        "主 Agent 真模型", ("tests/test_concurrency_eval.py",), ("real_concurrency_all_scenarios",),
    ),
    ImpactRule(
        ("scripts/run_privacy_isolation_live_eval.py", "eval/adversarial/privacy_isolation.jsonl"),
        "主 Agent 真模型", ("tests/test_privacy_live_evidence.py",), ("privacy_isolation_all_cases",),
    ),
    ImpactRule(
        (RUNTIME + "stores/*", RUNTIME + "store.py", RUNTIME + "coordination.py",
         RUNTIME + "domains/tools/*", RUNTIME + "services/tool_service.py"),
        "定向离线", ("tests/test_production_invariants.py", "tests/test_concurrency_eval.py"),
        ("runtime_deterministic", "concurrency_deterministic"),
    ),
)

SHARED_RISK = (
    ".env*", ".github/workflows/*", "pyproject.toml", "requirements*.txt", "uv.lock",
    RUNTIME + "llm.py", RUNTIME + "responses_api.py", RUNTIME + "env.py", RUNTIME + "runtime.py",
    RUNTIME + "runtime_components.py", RUNTIME + "budget.py", RUNTIME + "tracing.py",
    RUNTIME + "models.py", RUNTIME + "services/action_service.py", RUNTIME + "services/loop*.py",
    RUNTIME + "domains/tools/*", RUNTIME + "stores/*", RUNTIME + "store.py",
    "scripts/eval_product_contracts.py", "scripts/eval_topic_policy.py",
    "scripts/run_inquiry_grounding_eval.py", "scripts/run_real_owner_chat_live_eval.py",
)
DATASET_SCOPES = {
    "conversation_topic_continuity_20261105": "conversation_topic_continuity",
    "group_delegation_acceptance_20261105": "group_delegation_acceptance_20261105",
    "group_topic_recovery_20261105": "group_topic_recovery_20261105",
    "group_delegation_scope_drift_20261105": "group_delegation_scope_drift_20261105",
    "group_query_history_join_20261105": "group_query_history_join_20261105",
}
GROUP_DATASET_PREFIX = "eval/golden/real_group_chat_"


def changed_paths(root: Path, base: str) -> tuple[str, list[str]]:
    """比较索引和工作树，保留未追踪文件及重命名前后的影响。"""

    def git(*args: str) -> bytes:
        """从 Git 取得路径与提交标识，错误由上层给出固定说明。"""
        return subprocess.check_output(("git", *args), cwd=root, stderr=subprocess.DEVNULL)

    try:
        revision = git("rev-parse", "--verify", "--end-of-options", f"{base}^{{commit}}").decode("ascii").strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("无法解析 Git 基线。请在仓库中指定有效的提交、分支或标签。") from exc
    diff_args = ("--no-ext-diff", "--name-only", "--no-renames", "-z", revision, "--")
    try:
        outputs = (
            git("diff", *diff_args),
            git("diff", "--cached", *diff_args),
            git("ls-files", "--others", "--exclude-standard", "-z"),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("无法读取 Git 暂存区、工作树或未追踪文件。") from exc
    return revision, sorted({os.fsdecode(path) for output in outputs for path in output.split(b"\0") if path})


def assess_eval_impact(
    paths: list[str], *, removed_paths: frozenset[str] = frozenset(),
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """依据文件匹配已知入口并提示共有影响，保留无法归类的路径供复核。"""

    files = {layer: set() for layer in LAYERS}
    commands = {layer: set() for layer in LAYERS}
    tests: set[str] = set()
    for path in paths:
        if path in removed_paths:
            # 被删用例和资产仍有覆盖影响，但不能用已不存在的文件或范围执行。
            files["需人工核对"].add(path)
            commands["需人工核对"].add(OFFLINE)
            continue

        known = False
        for rule in RULES:
            if rule.matches(path):
                known = True
                files[rule.layer].add(path)
                if rule.tests:
                    files["定向离线"].add(path)
                    tests.update(rule.tests)
                commands[rule.layer].update(f"{SELECT} {scope}" for scope in rule.scopes)

        entry = PurePosixPath(path)
        scope = DATASET_SCOPES.get(entry.stem) if entry.parent == PurePosixPath("eval/golden") and entry.suffix == ".jsonl" else None
        if scope:
            known = True
            files["主 Agent 真模型"].add(path)
            commands["主 Agent 真模型"].add(f"{SELECT} {scope}")
        if path.startswith(GROUP_DATASET_PREFIX) and entry.suffix == ".jsonl":
            suffix = path.removeprefix(GROUP_DATASET_PREFIX).removesuffix(".jsonl")
            if len(suffix) == 8 and suffix.isascii() and suffix.isdecimal():
                known = True
                files["主 Agent 真模型"].add(path)
                files["定向离线"].add(path)
                commands["主 Agent 真模型"].add(f"{SELECT} real_group_all_scenarios_{suffix}")
                commands["定向离线"].add(f"{SELECT} group_flow_deterministic_{suffix}")
        if path.startswith("tests/") and entry.name.startswith("test_") and entry.suffix == ".py":
            known = True
            files["定向离线"].add(path)
            tests.add(path)
        is_documentation = (
            path.startswith("docs/") or path == "eval/README.md"
            or (entry.parent == PurePosixPath(".") and entry.suffix == ".md")
        )
        if not known and is_documentation:
            known = True
            files["文档校验"].add(path)
            commands["文档校验"].update(("git diff --check", "git diff --cached --check"))
        if not known or any(fnmatchcase(path, pattern) for pattern in SHARED_RISK):
            files["需人工核对"].add(path)
            commands["需人工核对"].add(OFFLINE)

    if tests:
        commands["定向离线"].add(f"{PYTEST} {shlex.join(sorted(tests))}")
    return files, commands


def print_eval_impact(root: Path, base: str) -> None:
    """输出可复制的命令供开发者选择，不运行评测或创建证据文件。"""

    revision, paths = changed_paths(root, base)
    removed_paths = frozenset(path for path in paths if not (root / path).exists())
    files, commands = assess_eval_impact(paths, removed_paths=removed_paths)
    print(f"Git 基线 {revision}，发现 {len(paths)} 条变化路径。")
    for layer in LAYERS:
        if files[layer]:
            print(f"\n{layer}")
            for path in sorted(files[layer]):
                print(f"  - {json.dumps(path, ensure_ascii=False)}")
            for command in sorted(commands[layer]):
                print(f"  $ {command}")
    if files["对客审查"]:
        print("\n对客审查可在核实原待审文本及工具、状态证据后，从归档只复验受影响的审查正反例。见 eval/README.md。")
    if files["独立判定"]:
        print("\n独立判定输出无效且留有完整有效主执行证据时，可使用 scripts/replay_archived_owner_judge.py 定向复验。")
    mixed_runners = {"scripts/run_inquiry_grounding_eval.py", "scripts/run_real_owner_chat_live_eval.py"}
    if mixed_runners.intersection(files["需人工核对"]):
        print("\n老板和询问评测脚本同时含主 Agent 回放与独立判定。先确认实际改动环节再选入口。")
    if files["需人工核对"]:
        print("\n共有实现、配置、已移除或未映射文件需人工确认下游范围。配置或发布变更按仓库验收要求执行。")
    print("\n这是基于文件的验证建议。运行时模型配置、环境和证据归属需另行核对。")
    print("完整通过仍以同一候选版本的显式 --full 为准。")
