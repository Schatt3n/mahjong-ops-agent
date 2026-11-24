"""将统一评测报告投影为可在公开 GitHub Actions 页面展示的摘要。"""

import json
import os
import re
from pathlib import Path

SCOPE_NAME = re.compile(r"[a-z0-9_]+")
SCOPE_STATUSES = {"pending", "running", "passed", "failed", "not_run"}
GATE_STATUSES = {"running", "running_with_failures", "passed", "failed"}


def main() -> None:
    """只展示提交、门禁模式和范围状态，忽略日志路径、错误详情和对话。"""
    summary = ["## Evaluation gate", "", f"Commit: `{os.environ['GITHUB_SHA']}`", ""]
    reports = list(Path("runtime_data/eval_runs").glob("*/gate_report.json"))
    if len(reports) != 1:
        summary.append("A unique gate report was unavailable. See the evaluation step status.")
    else:
        report = json.loads(reports[0].read_text(encoding="utf-8"))
        mode = "Full" if report["full"] else "Offline"
        status = report["status"]
        summary.extend([
            f"{mode} evaluation: **{status if status in GATE_STATUSES else 'unknown'}**",
            "",
            "| Scope | Status |",
            "| --- | --- |",
        ])
        for scope in report["scopes"]:
            name, status = scope["scope"], scope["status"]
            if SCOPE_NAME.fullmatch(name) and status in SCOPE_STATUSES:
                summary.append(f"| `{name}` | {status} |")

    with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as output:
        output.write("\n".join(summary) + "\n")


if __name__ == "__main__":
    main()
