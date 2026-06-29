from __future__ import annotations

import argparse
import importlib.util
import inspect
import os
import pathlib
import subprocess
import sys
import traceback


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
sys.path.insert(0, str(SRC))


def load_module(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def has_pytest_mark(fn, mark_name: str) -> bool:
    for mark in getattr(fn, "pytestmark", []) or []:
        if getattr(mark, "name", "") == mark_name:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local tests and optional integration checks.")
    parser.add_argument(
        "--with-deepseek",
        action="store_true",
        help="Run the real DeepSeek integration smoke test after local tests.",
    )
    parser.add_argument(
        "--deepseek-text",
        default="老板，今天下班有人打麻将吗？0.5或者1都行，烟也都可",
        help="Message used by the DeepSeek integration smoke test.",
    )
    args = parser.parse_args()

    passed = 0
    failed = 0
    for path in sorted(TESTS.glob("test_*.py")):
        module = load_module(path)
        for name, fn in sorted(inspect.getmembers(module, inspect.isfunction)):
            if not name.startswith("test_"):
                continue
            if has_pytest_mark(fn, "integration"):
                print(f"SKIP {path.name}::{name} (pytest integration test; use --with-deepseek or pytest -m integration)")
                continue
            try:
                fn()
            except Exception:
                failed += 1
                print(f"FAIL {path.name}::{name}")
                traceback.print_exc()
            else:
                passed += 1
                print(f"PASS {path.name}::{name}")
    if args.with_deepseek:
        print("\nRUN integration::deepseek")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC)
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_deepseek_integration_test.py"),
                "--text",
                args.deepseek_text,
            ],
            cwd=str(ROOT),
            env=env,
            check=False,
        )
        if result.returncode == 0:
            passed += 1
            print("PASS integration::deepseek")
        else:
            failed += 1
            print(f"FAIL integration::deepseek returncode={result.returncode}")
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
