#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys
from typing import Iterable, NamedTuple


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b[a-f0-9]{32}\.[A-Za-z0-9_-]{8,}\b", re.IGNORECASE),
    re.compile(
        r"\b(?:MAHJONG|DEEPSEEK|DASHSCOPE|OPENAI|ZAI)_[A-Z0-9_]*API_KEY\s*=\s*['\"]?"
        r"(?!(?:your|你|test|fake|dummy|example|redacted|configured|none|null|<|\{))[^'\"\s]+",
        re.IGNORECASE,
    ),
)
SECRET_SCAN_TARGETS = (
    "README.md",
    "pyproject.toml",
    "docs",
    "eval",
    "scripts",
    "skills",
    "src",
    "tests",
)
SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "htmlcov",
    "logs",
    "node_modules",
}
TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


class Step(NamedTuple):
    name: str
    command: list[str] | None = None
    env: dict[str, str] | None = None


def iter_scan_files(paths: Iterable[pathlib.Path]) -> Iterable[pathlib.Path]:
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            if should_scan_file(path):
                yield path
            continue
        for child in path.rglob("*"):
            if any(part in SKIP_DIRS for part in child.parts):
                continue
            if child.is_file() and should_scan_file(child):
                yield child


def should_scan_file(path: pathlib.Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def secret_scan(paths: Iterable[pathlib.Path]) -> list[pathlib.Path]:
    hits: list[pathlib.Path] = []
    for path in iter_scan_files(paths):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            hits.append(path)
    return sorted(set(hits))


def default_scan_paths(root: pathlib.Path = ROOT) -> list[pathlib.Path]:
    return [root / target for target in SECRET_SCAN_TARGETS]


def build_steps(*, with_deepseek: bool) -> list[Step]:
    env = {"PYTHONPATH": str(SRC)}
    steps = [
        Step(
            "py_compile",
            [
                sys.executable,
                "-m",
                "py_compile",
                str(ROOT / "scripts" / "run_boss_trial_app.py"),
                str(ROOT / "scripts" / "run_deepseek_integration_test.py"),
                str(ROOT / "scripts" / "run_controlled_agent_acceptance.py"),
                str(SRC / "mahjong_agent" / "llm.py"),
                str(SRC / "mahjong_agent" / "responder.py"),
            ],
        ),
        Step("pytest_offline", [sys.executable, "-m", "pytest", "-q"], env),
        Step("scenario_eval", [sys.executable, str(ROOT / "scripts" / "run_scenario_eval.py")], env),
        Step("custom_test_runner", [sys.executable, str(ROOT / "scripts" / "run_tests.py")], env),
    ]
    if with_deepseek:
        deepseek_env = {**env, "MAHJONG_RUN_DEEPSEEK_INTEGRATION": "1"}
        steps.append(
            Step(
                "deepseek_integration",
                [sys.executable, str(ROOT / "scripts" / "run_deepseek_integration_test.py")],
                deepseek_env,
            )
        )
    return steps


def run_step(step: Step) -> bool:
    if step.command is None:
        return True
    env = dict(os.environ)
    env.update(step.env or {})
    print(f"\n== {step.name} ==")
    print(" ".join(step.command))
    result = subprocess.run(step.command, cwd=str(ROOT), env=env, check=False)
    if result.returncode == 0:
        print(f"PASS {step.name}")
        return True
    print(f"FAIL {step.name} returncode={result.returncode}")
    return False


def run_secret_scan() -> bool:
    print("\n== secret_scan ==")
    hits = secret_scan(default_scan_paths(ROOT))
    if not hits:
        print("PASS secret_scan")
        return True
    print("FAIL secret_scan: potential secrets found in these files:")
    for path in hits:
        print(f"  - {path.relative_to(ROOT)}")
    print("Secret values are intentionally not printed. Remove them or move them to environment variables.")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the controlled-agent acceptance gate for local production readiness."
    )
    parser.add_argument(
        "--with-deepseek",
        action="store_true",
        help="Also run the real DeepSeek integration smoke test. Requires an API key in the environment.",
    )
    args = parser.parse_args()

    ok = run_secret_scan()
    for step in build_steps(with_deepseek=args.with_deepseek):
        ok = run_step(step) and ok

    if not args.with_deepseek:
        print(
            "\nSKIP deepseek_integration: run with --with-deepseek and "
            "MAHJONG_DEEPSEEK_API_KEY to verify the real model call."
        )

    print("\nAcceptance result: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
