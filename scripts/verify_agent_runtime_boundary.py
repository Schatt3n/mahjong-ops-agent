#!/usr/bin/env python3
from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE_ROOTS = (ROOT / "src" / "mahjong_agent_runtime",)
RUNTIME_ENTRYPOINTS = (
    ROOT / "scripts" / "run_agent_app.py",
    ROOT / "scripts" / "agent_runtime_app.py",
    ROOT / "scripts" / "run_agent_v3_app.py",
)
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
FORBIDDEN_SEMANTIC_PATCH_TOKENS = {
    "_normalize_pool_query_text": "旧试用台归一化函数不能进入当前主链路",
    "_guard_suggested_reply": "旧试用台业务回复 guard 不能进入当前主链路",
    "ReplyGuard": "当前主链路不允许接入旧回复 guard",
    "re.sub": "当前主链路后端不应用正则替换修麻将语义",
    "re.findall": "当前主链路后端不应用正则抽取修麻将语义",
    "0，5": "当前主链路后端不应硬编码 0.5 口误 badcase",
    "0。5": "当前主链路后端不应硬编码 0.5 口误 badcase",
    "0/5": "当前主链路后端不应硬编码 0.5 口误 badcase",
    "人气开": "当前主链路后端不应硬编码人齐开口误 badcase",
    "asap_when_full": "当前主链路客户可见链路不应泄漏旧内部枚举补丁",
    "先帮你留意下": "当前主链路不应把过早停止话术硬编码为兜底回复",
}
FORBIDDEN_ENTRYPOINT_TOKENS = {
    "/api/analyze": "当前服务入口不应重新暴露旧试用台 analyze 接口",
    "ControlledWorkflowService": "当前服务入口不应接入旧 controlled workflow 服务",
    "AgentResponder": "当前服务入口不应接入旧 responder",
}


@dataclass(frozen=True, slots=True)
class BoundaryViolation:
    path: Path
    line: int
    message: str

    def format(self) -> str:
        return f"{self.path.relative_to(ROOT)}:{self.line}: {self.message}"


def target_files() -> list[Path]:
    source_files: list[Path] = []
    for root in RUNTIME_SOURCE_ROOTS:
        source_files.extend(sorted(root.rglob("*.py")))
    return [*source_files, *RUNTIME_ENTRYPOINTS]


def verify_file(path: Path) -> list[BoundaryViolation]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    violations: list[BoundaryViolation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                violations.extend(_violations_for_module(path, node.lineno, alias.name))
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module:
                violations.extend(_violations_for_module(path, node.lineno, module))
    violations.extend(_semantic_patch_violations(path, text))
    violations.extend(_entrypoint_violations(path, text))
    return violations


def verify_files(paths: list[Path] | None = None) -> list[BoundaryViolation]:
    violations: list[BoundaryViolation] = []
    for path in paths or target_files():
        violations.extend(verify_file(path))
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
                    message=f"Agent Runtime must not import legacy package {normalized!r}",
                )
            )
    parts = set(normalized.split("."))
    for name in sorted(parts & FORBIDDEN_MODULE_NAMES):
        violations.append(
            BoundaryViolation(
                path=path,
                line=line,
                message=f"Agent Runtime must not import legacy module {name!r}",
            )
        )
    return violations


def _semantic_patch_violations(path: Path, text: str) -> list[BoundaryViolation]:
    violations: list[BoundaryViolation] = []
    for token, reason in FORBIDDEN_SEMANTIC_PATCH_TOKENS.items():
        offset = text.find(token)
        if offset < 0:
            continue
        line = text[:offset].count("\n") + 1
        violations.append(
            BoundaryViolation(
                path=path,
                line=line,
                message=f"Agent Runtime semantic boundary violation: {reason} ({token!r})",
            )
        )
    return violations


def _entrypoint_violations(path: Path, text: str) -> list[BoundaryViolation]:
    violations: list[BoundaryViolation] = []
    if path not in RUNTIME_ENTRYPOINTS:
        return violations
    for token, reason in FORBIDDEN_ENTRYPOINT_TOKENS.items():
        offset = text.find(token)
        if offset < 0:
            continue
        line = text[:offset].count("\n") + 1
        violations.append(
            BoundaryViolation(
                path=path,
                line=line,
                message=f"Agent Runtime entrypoint boundary violation: {reason} ({token!r})",
            )
        )
    return violations


def main() -> int:
    violations = verify_files()
    if violations:
        print("Agent Runtime boundary check failed:", file=sys.stderr)
        for violation in violations:
            print(f"  {violation.format()}", file=sys.stderr)
        return 1
    print("PASS Agent Runtime boundary: no legacy imports or semantic patch code")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
