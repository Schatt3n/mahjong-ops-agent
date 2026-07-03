from __future__ import annotations

from pathlib import Path

from mahjong_agent_runtime import AgentRuntime, AgentRuntimeV3, ToolGateway, ToolGatewayV3
from mahjong_agent_runtime.tracing import validate_trace, validate_trace_v3


ROOT = Path(__file__).resolve().parents[1]


def test_stable_runtime_package_reexports_current_implementation() -> None:
    assert AgentRuntime is AgentRuntimeV3
    assert ToolGateway is ToolGatewayV3
    assert validate_trace is validate_trace_v3


def test_stable_runtime_package_does_not_import_compatibility_package() -> None:
    for path in (ROOT / "src" / "mahjong_agent_runtime").glob("**/*.py"):
        text = path.read_text(encoding="utf-8")
        assert "mahjong_agent_v3" not in text, f"{path} should not import compatibility package"
