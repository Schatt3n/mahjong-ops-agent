"""定义主循环各阶段交换的轻量结果对象，避免服务间传递松散字典。"""

from __future__ import annotations

from dataclasses import dataclass, field

from .budget import TokenBudget
from .models import AgentAction, ToolResult


@dataclass(slots=True)
class TurnBudgets:
    """主Agent、审查和文案写作按每用户消息分配的LLM预算。"""

    agent: TokenBudget
    review: TokenBudget
    text_generation: TokenBudget


@dataclass(slots=True)
class ModelActionStep:
    """一次主模型调用的规范化结果。"""

    action: AgentAction | None
    raw_response: str = ""
    errors: list[str] = field(default_factory=list)
    final_reply: str | None = None
    stop_loop: bool = False


@dataclass(slots=True)
class ActionProcessingResult:
    """处理一个 AgentAction 后返回给 Agent 循环的结果。"""

    action: AgentAction
    tool_results: list[ToolResult] = field(default_factory=list)
    pending_tool_results: list[ToolResult] = field(default_factory=list)
    final_reply: str | None = None
    stop_loop: bool = False
    continue_loop: bool = False


@dataclass(slots=True)
class ProgressHandlingResult:
    """应用一个 ProgressDecision 后的循环级结果。"""

    pending_tool_results: list[ToolResult] = field(default_factory=list)
    guard_result: ToolResult | None = None
    final_reply: str | None = None
    stop_loop: bool = False
    runtime_status: str | None = None
