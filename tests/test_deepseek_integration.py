from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_deepseek_integration_test.py"


def load_deepseek_script():
    spec = importlib.util.spec_from_file_location("run_deepseek_integration_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "enabled"}


@pytest.mark.integration
def test_deepseek_real_semantic_smoke() -> None:
    if not enabled(os.getenv("MAHJONG_RUN_DEEPSEEK_INTEGRATION")):
        pytest.skip("set MAHJONG_RUN_DEEPSEEK_INTEGRATION=1 to call the real DeepSeek model")

    module = load_deepseek_script()
    args = argparse.Namespace(
        text=os.getenv("MAHJONG_DEEPSEEK_TEST_TEXT", module.DEFAULT_TEXT),
        model=os.getenv("MAHJONG_DEEPSEEK_MODEL", "deepseek-v4-flash"),
        base_url=os.getenv("MAHJONG_DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        timeout_seconds=float(os.getenv("MAHJONG_DEEPSEEK_TIMEOUT_SECONDS", "30")),
        max_completion_tokens=int(os.getenv("MAHJONG_DEEPSEEK_MAX_COMPLETION_TOKENS", "1024")),
        max_calls=int(os.getenv("MAHJONG_DEEPSEEK_MAX_CALLS", "2")),
        max_tokens=int(os.getenv("MAHJONG_DEEPSEEK_MAX_TOKENS", "12000")),
        max_tokens_per_call=int(os.getenv("MAHJONG_DEEPSEEK_MAX_TOKENS_PER_CALL", "8000")),
        max_cost=float(os.getenv("MAHJONG_DEEPSEEK_MAX_COST", "1.0")),
        input_price_per_1k=float(os.getenv("MAHJONG_DEEPSEEK_INPUT_PRICE_PER_1K", "0")),
        output_price_per_1k=float(os.getenv("MAHJONG_DEEPSEEK_OUTPUT_PRICE_PER_1K", "0")),
        min_confidence=float(os.getenv("MAHJONG_DEEPSEEK_MIN_CONFIDENCE", "0.45")),
    )

    result = module.run_semantic_smoke(args)

    assert result == 0, (
        "DeepSeek integration must make a real deepseek call and return a usable semantic result. "
        "If this failed with return code 2, configure MAHJONG_DEEPSEEK_API_KEY or DEEPSEEK_API_KEY."
    )
