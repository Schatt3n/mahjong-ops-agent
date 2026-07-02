#!/usr/bin/env python3
from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V3_SOURCE_ROOT = ROOT / "src" / "mahjong_agent_v3"
V3_ENTRYPOINTS = (ROOT / "scripts" / "run_agent_v3_app.py",)
FORBIDDEN_PACKAGES = {"mahjong_agent", "mahjong_agent_v2"}
FORBIDDEN_MODULE_NAMES = {
    "action_validator",
    "autonomous_agent",
    "candidate_semantics",
    "controlled_runtime",
    "controlled_workflow",
    "input_gate",
    "parser",
    "reply_guard",
    "semantic_resolver",
    "state_machine",
    "tool_orchestrator",
    "trial_entry",
    "trial_runtime_policy",
    "trial_tool_gateway",
    "workflow",
}


@dataclass(frozen=True, slots=True)
class BoundaryViolation:
    path: Path
    line: int
    message: str

    def format(self) -> str:
        return f"{self.path.relative_to(ROOT)}:{self.line}: {self.message}"


def target_files() -> list[Path]:
    return [*sorted(V3_SOURCE_ROOT.rglob("*.py")), *V3_ENTRYPOINTS]


def verify_file(path: Path) -> list[BoundaryViolation]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[BoundaryViolation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                violations.extend(_violations_for_module(path, node.lineno, alias.name))
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module:
                violations.extend(_violations_for_module(path, node.lineno, module))
    return violations


def _violations_for_module(path: Path, line: int, module: str) -> list[BoundaryViolation]:
    normalized = module.strip()
    if not normalized:
        return []
    violations: list[BoundaryViolation] = []
    for package in FORBIDDEN_PACKAGES:
        if normalized == package or normalized.startswith(f"{package}."):
            violations.append(
                BoundaryViolation(
                    path=path,
                    line=line,
                    message=f"Agent Runtime V3 must not import legacy package {normalized!r}",
                )
            )
    parts = set(normalized.split("."))
    for name in sorted(parts & FORBIDDEN_MODULE_NAMES):
        violations.append(
            BoundaryViolation(
                path=path,
                line=line,
                message=f"Agent Runtime V3 must not import legacy module {name!r}",
            )
        )
    return violations


def main() -> int:
    violations: list[BoundaryViolation] = []
    for path in target_files():
        violations.extend(verify_file(path))
    if violations:
        print("Agent Runtime V3 boundary check failed:", file=sys.stderr)
        for violation in violations:
            print(f"  {violation.format()}", file=sys.stderr)
        return 1
    print("PASS Agent Runtime V3 boundary: no legacy parser/workflow/guard imports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
