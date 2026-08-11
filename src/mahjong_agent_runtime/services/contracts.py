"""编排contracts相关的应用服务，不直接承载通道或数据库细节。

"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import AgentAction, StateTransition, ToolResult


@dataclass(slots=True)
class SingleToolExecution:
    """一个工具结果加上调度器控制信号。"""

    result: ToolResult
    blocked_by_consistency: bool = False
    blocked_by_stale_run: bool = False


@dataclass(slots=True)
class LoopStepOutcome:
    """一个模型/工具循环步骤，缩减为外层循环必须保留的状态。"""

    action: AgentAction | None = None
    tool_results: list[ToolResult] = field(default_factory=list)
    evidence_results: list[ToolResult] = field(default_factory=list)
    pending_tool_results: list[ToolResult] = field(default_factory=list)
    summary_transition: StateTransition | None = None
    final_reply: str | None = None
    stop_loop: bool = False
    runtime_status: str | None = None


__all__ = ["LoopStepOutcome", "SingleToolExecution"]
