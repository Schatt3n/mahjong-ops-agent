from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "scripts" / "run_agent_v2_app.py"


def load_app_module_without_runtime():
    spec = importlib.util.spec_from_file_location("run_agent_v2_app_for_test", APP_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    module.build_runtime = lambda: None
    spec.loader.exec_module(module)
    return module


def test_agent_v2_console_exposes_observable_panels() -> None:
    module = load_app_module_without_runtime()

    html = module.index_html()

    assert "本轮结果" in html
    assert "模型决策" in html
    assert "工具调用" in html
    assert "状态变化" in html
    assert "Trace" in html
    assert "Badcase" in html
    assert "/api/v2/message" in html
    assert "/api/v2/state" in html
    assert "/api/v2/traces" in html
    assert "/api/v2/badcases" in html
