from __future__ import annotations

from pathlib import Path

from mahjong_agent_runtime import AgentRuntime, ContextSummaryManager, ContextSummaryPolicy, ToolGateway


ROOT = Path(__file__).resolve().parents[1]


def test_stable_runtime_package_exposes_versionless_api() -> None:
    import mahjong_agent_runtime

    assert mahjong_agent_runtime.AgentRuntime is AgentRuntime
    assert mahjong_agent_runtime.ContextSummaryManager is ContextSummaryManager
    assert mahjong_agent_runtime.ContextSummaryPolicy is ContextSummaryPolicy
    assert mahjong_agent_runtime.ToolGateway is ToolGateway
    assert not hasattr(mahjong_agent_runtime, "AgentRuntimeV3")
    assert not hasattr(mahjong_agent_runtime, "ToolGatewayV3")


def test_stable_runtime_package_does_not_import_compatibility_package() -> None:
    for path in (ROOT / "src" / "mahjong_agent_runtime").glob("**/*.py"):
        text = path.read_text(encoding="utf-8")
        assert "mahjong_agent_v3" not in text, f"{path} should not import compatibility package"


def test_historical_v3_package_has_been_removed_from_main_repo() -> None:
    assert not (ROOT / "src" / "mahjong_agent_v3").exists()
    assert not (ROOT / "scripts" / "run_agent_v3_app.py").exists()
    assert not (ROOT / "scripts" / "run_agent_runtime_v3_eval.py").exists()
    assert not (ROOT / "scripts" / "verify_agent_runtime_v3_boundary.py").exists()
    assert not (ROOT / "docs" / "agent_runtime_v3.md").exists()
