from __future__ import annotations

"""Generic loop and no-progress detection for the main Agent runtime.

The monitor deliberately knows nothing about Mahjong semantics. It compares
stable action/result fingerprints and material state transitions within one
user turn. The first detected stall asks the model to replan; a repeated stall
after that recovery attempt aborts the loop.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any

from .models import AgentAction, ToolResult


VOLATILE_KEYS = {
    "trace_id",
    "idempotency_key",
    "created_at",
    "updated_at",
    "occurred_at",
    "captured_at",
    "timestamp",
    "elapsed_ms",
    "run_id",
    "raw_response",
}


def build_progress_hint(actions: list[AgentAction], *, recent_limit: int = 3) -> str:
    """Build a compact, domain-neutral self-diagnosis hint for the next model step."""

    if not actions:
        return ""
    labels = [_action_strategy_label(action) for action in actions]
    recent = labels[-max(1, int(recent_limit)) :]
    repeated = 1
    for label in reversed(labels[:-1]):
        if label != labels[-1]:
            break
        repeated += 1
    return "\n".join(
        [
            f"[执行进度] 已执行 {len(actions)} 步；最近动作: [{', '.join(recent)}]",
            f"连续相同动作: {repeated}次",
            '若重复同一策略且没有新结果，请令 self_assessment.progress="stalled"；确需退出时 should_escalate=true。',
        ]
    )


def _action_strategy_label(action: AgentAction) -> str:
    if action.tool_calls:
        return "tool_call:" + "+".join(call.name or "unknown" for call in action.tool_calls)
    return f"terminal:{action.objective_status or 'unknown'}"


@dataclass(slots=True)
class ProgressDecision:
    """Result of inspecting one non-terminal Agent step."""

    step_index: int
    observation_kind: str
    observation_signature: str
    progress_made: bool
    progress_reasons: list[str] = field(default_factory=list)
    repeated_observation_count: int = 0
    consecutive_no_progress_steps: int = 0
    cycle_period: int | None = None
    detection_reasons: list[str] = field(default_factory=list)
    should_replan: bool = False
    should_abort: bool = False

    @property
    def detected(self) -> bool:
        return bool(self.detection_reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "observation_kind": self.observation_kind,
            "observation_signature": self.observation_signature,
            "progress_made": self.progress_made,
            "progress_reasons": list(self.progress_reasons),
            "repeated_observation_count": self.repeated_observation_count,
            "consecutive_no_progress_steps": self.consecutive_no_progress_steps,
            "cycle_period": self.cycle_period,
            "detection_reasons": list(self.detection_reasons),
            "should_replan": self.should_replan,
            "should_abort": self.should_abort,
        }


@dataclass(slots=True)
class ProgressMonitor:
    """Detect repeated observations, short cycles, and consecutive stalls.

    A new tool result is treated as information progress, while a non-replayed
    state transition is treated as state progress. Repeating the same action and
    same stable result is not progress even if the tool physically ran again.
    """

    repeated_observation_limit: int = 2
    consecutive_no_progress_limit: int = 2
    max_replan_attempts: int = 1
    max_cycle_period: int = 3
    _observation_history: list[str] = field(default_factory=list, init=False, repr=False)
    _seen_result_signatures: set[str] = field(default_factory=set, init=False, repr=False)
    _consecutive_no_progress_steps: int = field(default=0, init=False, repr=False)
    _replan_attempts: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.repeated_observation_limit = max(2, int(self.repeated_observation_limit))
        self.consecutive_no_progress_limit = max(1, int(self.consecutive_no_progress_limit))
        self.max_replan_attempts = max(0, int(self.max_replan_attempts))
        self.max_cycle_period = max(1, int(self.max_cycle_period))

    def observe_action(
        self,
        action: AgentAction,
        tool_results: list[ToolResult],
        *,
        step_index: int,
    ) -> ProgressDecision:
        """Inspect a tool step after execution and classify its progress."""

        material_state_change = any(result.state_transitions and not result.deduplicated for result in tool_results)
        if material_state_change:
            self._reset_epoch()

        result_signatures = [
            stable_fingerprint(
                {
                    "tool_call": (
                        {
                            "name": action.tool_calls[index].name,
                            "arguments": stable_value(action.tool_calls[index].arguments),
                        }
                        if index < len(action.tool_calls)
                        else None
                    ),
                    "tool_result": tool_result_payload(result),
                }
            )
            for index, result in enumerate(tool_results)
        ]
        unseen_results = [item for item in result_signatures if item not in self._seen_result_signatures]
        observation_payload = {
            "kind": "tool_action",
            "tool_calls": [
                {
                    "name": call.name,
                    "arguments": stable_value(call.arguments),
                }
                for call in action.tool_calls
            ],
            "tool_results": result_signatures,
        }
        progress_reasons: list[str] = []
        if material_state_change:
            progress_reasons.append("state_transition")
        if unseen_results:
            progress_reasons.append("new_tool_result")
        return self._observe(
            observation_kind="tool_action",
            observation_payload=observation_payload,
            result_signatures=result_signatures,
            progress_made=bool(progress_reasons),
            progress_reasons=progress_reasons,
            step_index=step_index,
        )

    def observe_runtime_feedback(
        self,
        feedback_kind: str,
        payload: dict[str, Any],
        *,
        step_index: int,
    ) -> ProgressDecision:
        """Inspect a failed runtime step such as an invalid output contract."""

        return self._observe(
            observation_kind=feedback_kind,
            observation_payload={"kind": feedback_kind, "payload": stable_value(payload)},
            result_signatures=[],
            progress_made=False,
            progress_reasons=[],
            step_index=step_index,
        )

    def feedback_result(self, decision: ProgressDecision) -> ToolResult:
        """Create structured feedback that can be appended to model context."""

        if decision.should_abort:
            instruction = (
                "The Agent repeated a stalled execution after a replan attempt. Stop this run; "
                "do not call another tool from the same state."
            )
        else:
            instruction = (
                "Replan from the current objective and previous_tool_results. Choose a materially different action, "
                "different validated arguments, or a legal terminal status. Do not repeat the same tool call with "
                "the same arguments when neither its result nor backend state has changed."
            )
        return ToolResult(
            name="agent_progress_guard",
            called=False,
            allowed=False,
            result={
                "classification": "agent_loop_or_no_progress",
                "instruction": instruction,
                "decision": decision.to_dict(),
            },
            error="agent loop made no material progress",
        )

    def _observe(
        self,
        *,
        observation_kind: str,
        observation_payload: dict[str, Any],
        result_signatures: list[str],
        progress_made: bool,
        progress_reasons: list[str],
        step_index: int,
    ) -> ProgressDecision:
        signature = stable_fingerprint(observation_payload)
        self._observation_history.append(signature)
        repeated_count = consecutive_tail_count(self._observation_history)
        cycle_period = detect_tail_cycle(self._observation_history, self.max_cycle_period)

        if progress_made:
            self._consecutive_no_progress_steps = 0
            self._replan_attempts = 0
        else:
            self._consecutive_no_progress_steps += 1

        self._seen_result_signatures.update(result_signatures)
        detection_reasons: list[str] = []
        if not progress_made and repeated_count >= self.repeated_observation_limit:
            detection_reasons.append("repeated_observation")
        if not progress_made and cycle_period is not None:
            detection_reasons.append("short_cycle")
        if not progress_made and self._consecutive_no_progress_steps >= self.consecutive_no_progress_limit:
            detection_reasons.append("consecutive_no_progress")

        should_replan = False
        should_abort = False
        if detection_reasons:
            if self._replan_attempts < self.max_replan_attempts:
                self._replan_attempts += 1
                should_replan = True
            else:
                should_abort = True

        return ProgressDecision(
            step_index=step_index,
            observation_kind=observation_kind,
            observation_signature=signature,
            progress_made=progress_made,
            progress_reasons=progress_reasons,
            repeated_observation_count=repeated_count,
            consecutive_no_progress_steps=self._consecutive_no_progress_steps,
            cycle_period=cycle_period,
            detection_reasons=detection_reasons,
            should_replan=should_replan,
            should_abort=should_abort,
        )

    def _reset_epoch(self) -> None:
        """Forget fingerprints from the prior backend state after a real write."""

        self._observation_history.clear()
        self._seen_result_signatures.clear()
        self._consecutive_no_progress_steps = 0
        self._replan_attempts = 0


def detect_tail_cycle(history: list[str], max_cycle_period: int) -> int | None:
    """Return the shortest repeated multi-step tail period, if one exists."""

    upper = min(max(0, int(max_cycle_period)), len(history) // 2)
    for period in range(2, upper + 1):
        if history[-period:] == history[-2 * period : -period]:
            return period
    return None


def consecutive_tail_count(history: list[str]) -> int:
    """Count only immediately repeated observations at the end of history."""

    if not history:
        return 0
    count = 1
    for item in reversed(history[:-1]):
        if item != history[-1]:
            break
        count += 1
    return count


def tool_result_payload(result: ToolResult) -> dict[str, Any]:
    """Keep result facts while removing execution-identity noise."""

    return {
        "name": result.name,
        "called": result.called,
        "allowed": result.allowed,
        "result": stable_value(result.result),
        "error": result.error,
        "state_transitions": [
            {
                "entity_type": item.entity_type,
                "entity_id": item.entity_id,
                "from_status": item.from_status,
                "to_status": item.to_status,
                "reason": item.reason,
            }
            for item in result.state_transitions
        ],
    }


def stable_value(value: Any) -> Any:
    """Recursively remove volatile runtime fields before hashing."""

    if isinstance(value, dict):
        return {
            str(key): stable_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in VOLATILE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [stable_value(item) for item in value]
    if isinstance(value, set):
        return sorted((stable_value(item) for item in value), key=lambda item: repr(item))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def stable_fingerprint(payload: Any) -> str:
    """Build a short deterministic fingerprint for trace and comparisons."""

    encoded = json.dumps(stable_value(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
