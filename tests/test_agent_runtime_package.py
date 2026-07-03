from __future__ import annotations

from mahjong_agent_runtime import AgentRuntime, AgentRuntimeV3, ToolGateway, ToolGatewayV3
from mahjong_agent_runtime.tracing import validate_trace, validate_trace_v3


def test_stable_runtime_package_reexports_current_implementation() -> None:
    assert AgentRuntime is AgentRuntimeV3
    assert ToolGateway is ToolGatewayV3
    assert validate_trace is validate_trace_v3
