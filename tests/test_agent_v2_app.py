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
    assert "message_id" in html
    assert "/api/v2/message" in html
    assert "/api/v2/state" in html
    assert "/api/v2/traces" in html
    assert "/api/v2/badcases" in html


def test_agent_v2_app_defaults_to_main_trial_port(monkeypatch) -> None:
    monkeypatch.delenv("MAHJONG_AGENT_V2_PORT", raising=False)

    module = load_app_module_without_runtime()

    assert module.PORT == 8790


def test_agent_v2_budget_is_configured_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("MAHJONG_AGENT_V2_MAX_TOKENS_PER_CALL", "64000")
    monkeypatch.setenv("MAHJONG_AGENT_V2_MAX_CALLS_PER_TURN", "9")

    module = load_app_module_without_runtime()
    budget = module.budget_from_env()

    assert budget.max_tokens_per_call == 64000
    assert budget.max_calls_per_turn == 9


def test_agent_v2_budget_uses_legacy_env_alias_for_transition(monkeypatch) -> None:
    monkeypatch.delenv("MAHJONG_AGENT_V2_MAX_TOKENS_PER_CALL", raising=False)
    monkeypatch.setenv("MAHJONG_LLM_MAX_TOKENS_PER_CALL", "48000")

    module = load_app_module_without_runtime()

    assert module.budget_from_env().max_tokens_per_call == 48000


def test_agent_v2_reply_review_defaults_on_and_can_be_disabled(monkeypatch) -> None:
    monkeypatch.delenv("MAHJONG_AGENT_V2_REPLY_REVIEW_ENABLED", raising=False)
    module = load_app_module_without_runtime()

    assert module.env_bool("MAHJONG_AGENT_V2_REPLY_REVIEW_ENABLED", True) is True

    monkeypatch.setenv("MAHJONG_AGENT_V2_REPLY_REVIEW_ENABLED", "false")
    assert module.env_bool("MAHJONG_AGENT_V2_REPLY_REVIEW_ENABLED", True) is False


def test_agent_v2_entrypoints_do_not_import_legacy_main_chain() -> None:
    files = [
        ROOT / "scripts" / "run_agent_v2_app.py",
        ROOT / "scripts" / "run_agent_runtime_v2_eval.py",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in files)

    forbidden = [
        "from mahjong_agent import",
        "import mahjong_agent.",
        "semantic_resolver",
        "controlled_workflow",
        "reply_guard",
        "backend_fallback",
    ]
    for item in forbidden:
        assert item not in source
