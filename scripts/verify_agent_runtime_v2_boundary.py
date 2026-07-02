#!/usr/bin/env python3
from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V2_SOURCE_ROOT = ROOT / "src" / "mahjong_agent_v2"
V2_ENTRYPOINTS = (
    ROOT / "scripts" / "run_agent_v2_app.py",
    ROOT / "scripts" / "run_agent_runtime_v2_eval.py",
)
FORBIDDEN_LEGACY_PACKAGE = "mahjong_agent"
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
}


@dataclass(frozen=True, slots=True)
class BoundaryViolation:
    path: Path
    line: int
    message: str

    def format(self) -> str:
        return f"{self.path.relative_to(ROOT)}:{self.line}: {self.message}"


def target_files() -> list[Path]:
    source_files = sorted(V2_SOURCE_ROOT.rglob("*.py"))
    return [*source_files, *V2_ENTRYPOINTS]


def verify_files(paths: list[Path] | None = None) -> list[BoundaryViolation]:
    violations: list[BoundaryViolation] = []
    for path in paths or target_files():
        violations.extend(verify_file(path))
    return violations


def verify_file(path: Path) -> list[BoundaryViolation]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[BoundaryViolation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                violations.extend(_violations_for_module(path, node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level > 0 and module:
                violations.extend(_violations_for_module(path, node.lineno, module))
            elif module:
                violations.extend(_violations_for_module(path, node.lineno, module))
            for alias in node.names:
                if node.level > 0:
                    violations.extend(_violations_for_module(path, node.lineno, alias.name))
    return violations


def _violations_for_module(path: Path, line: int, module: str) -> list[BoundaryViolation]:
    normalized = module.strip()
    if not normalized:
        return []
    violations: list[BoundaryViolation] = []
    if normalized == FORBIDDEN_LEGACY_PACKAGE or normalized.startswith(f"{FORBIDDEN_LEGACY_PACKAGE}."):
        if not normalized == "mahjong_agent_v2" and not normalized.startswith("mahjong_agent_v2."):
            violations.append(
                BoundaryViolation(
                    path=path,
                    line=line,
                    message=f"Agent Runtime V2 must not import legacy package {normalized!r}",
                )
            )
    module_parts = set(normalized.split("."))
    forbidden_parts = sorted(module_parts & FORBIDDEN_MODULE_NAMES)
    for part in forbidden_parts:
        violations.append(
            BoundaryViolation(
                path=path,
                line=line,
                message=f"Agent Runtime V2 must not import legacy module {part!r}",
            )
        )
    return violations


def main() -> int:
    violations = verify_files()
    if violations:
        print("Agent Runtime V2 boundary check failed:", file=sys.stderr)
        for violation in violations:
            print(f"  {violation.format()}", file=sys.stderr)
        return 1
    print("PASS Agent Runtime V2 boundary: no legacy parser/workflow/guard imports")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
