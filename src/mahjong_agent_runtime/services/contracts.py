from __future__ import annotations

"""Internal contracts shared by runtime services."""

from dataclasses import dataclass, field

from ..models import AgentAction, StateTransition, ToolResult


@dataclass(slots=True)
class SingleToolExecution:
    """One tool outcome plus scheduler control signals."""

    result: ToolResult
    blocked_by_consistency: bool = False
    blocked_by_stale_run: bool = False


@dataclass(slots=True)
class LoopStepOutcome:
    """One model/tool loop step reduced to state the outer loop must retain."""

    action: AgentAction | None = None
    tool_results: list[ToolResult] = field(default_factory=list)
    evidence_results: list[ToolResult] = field(default_factory=list)
    pending_tool_results: list[ToolResult] = field(default_factory=list)
    summary_transition: StateTransition | None = None
    final_reply: str | None = None
    stop_loop: bool = False
    runtime_status: str | None = None


__all__ = ["LoopStepOutcome", "SingleToolExecution"]
