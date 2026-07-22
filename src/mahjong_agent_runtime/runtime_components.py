from __future__ import annotations

from dataclasses import dataclass, field

from .budget import TokenBudget
from .models import AgentAction, ToolResult


@dataclass(slots=True)
class TurnBudgets:
    """Per-user-message LLM budgets for the main agent, review, and copywriting."""

    agent: TokenBudget
    review: TokenBudget
    text_generation: TokenBudget


@dataclass(slots=True)
class ModelActionStep:
    """Normalized result of one main model call."""

    action: AgentAction | None
    raw_response: str = ""
    errors: list[str] = field(default_factory=list)
    final_reply: str | None = None
    stop_loop: bool = False


@dataclass(slots=True)
class ActionProcessingResult:
    """Result returned to the agent loop after processing one AgentAction."""

    action: AgentAction
    tool_results: list[ToolResult] = field(default_factory=list)
    pending_tool_results: list[ToolResult] = field(default_factory=list)
    final_reply: str | None = None
    stop_loop: bool = False
    continue_loop: bool = False


@dataclass(slots=True)
class ProgressHandlingResult:
    """Loop-level outcome after applying one ProgressDecision."""

    pending_tool_results: list[ToolResult] = field(default_factory=list)
    guard_result: ToolResult | None = None
    final_reply: str | None = None
    stop_loop: bool = False
    runtime_status: str | None = None
