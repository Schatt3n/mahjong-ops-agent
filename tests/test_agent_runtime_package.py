from __future__ import annotations

from pathlib import Path

from mahjong_agent_runtime import AgentRuntime, ToolGateway


ROOT = Path(__file__).resolve().parents[1]


def test_stable_runtime_package_exposes_versionless_api() -> None:
    import mahjong_agent_runtime

    assert mahjong_agent_runtime.AgentRuntime is AgentRuntime
    assert mahjong_agent_runtime.ToolGateway is ToolGateway
    assert not hasattr(mahjong_agent_runtime, "AgentRuntimeV3")
    assert not hasattr(mahjong_agent_runtime, "ToolGatewayV3")


def test_stable_runtime_package_does_not_import_compatibility_package() -> None:
    for path in (ROOT / "src" / "mahjong_agent_runtime").glob("**/*.py"):
        text = path.read_text(encoding="utf-8")
        assert "mahjong_agent_v3" not in text, f"{path} should not import compatibility package"


def test_historical_v3_package_is_only_a_compatibility_alias() -> None:
    from mahjong_agent_v3 import AgentRuntimeV3, ToolGatewayV3

    assert AgentRuntimeV3 is AgentRuntime
    assert ToolGatewayV3 is ToolGateway
