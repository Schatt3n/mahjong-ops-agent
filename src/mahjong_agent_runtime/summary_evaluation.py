"""Decision-consistency evaluation for compressed conversation checkpoints.

The evaluator deliberately does not execute tools. It presents the same current
message to the same decision model twice: first with full recent history, then
with a newly generated checkpoint and only post-checkpoint turns. A summary is
decision-preserving when both model calls select the same terminal status,
customer reply, and canonical tool calls.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .action_contract import parse_action
from .domains.context_builders.builder import AgentContextBuilder, BuiltContext
from .llm import AgentLLMClient
from .models import AgentAction, ConversationCheckpoint, UserMessage
from .summary import ContextSummaryManager
ReplyComparator = Callable[[str, str], bool]


class TraceRecorder(Protocol):
    def record(
        self,
        trace_id: str,
        step: str,
        content: dict[str, Any],
        *,
        level: str = "INFO",
    ) -> None: ...


@dataclass(slots=True)
class DecisionSnapshot:
    """The decision-bearing subset of one AgentAction."""

    objective_status: str
    reply_to_user: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def tool_names(self) -> list[str]:
        return [str(item["name"]) for item in self.tool_calls]

    @classmethod
    def from_action(cls, action: AgentAction) -> "DecisionSnapshot":
        return cls(
            objective_status=action.objective_status,
            reply_to_user=normalize_reply(action.reply_to_user),
            tool_calls=[
                {
                    "name": call.name,
                    "arguments": canonical_value(call.arguments),
                }
                for call in action.tool_calls
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_status": self.objective_status,
            "reply_to_user": self.reply_to_user,
            "tool_calls": [dict(item) for item in self.tool_calls],
        }


@dataclass(slots=True)
class DecisionConsistencyReport:
    """Auditable before/after result for one compression quality case."""

    before: DecisionSnapshot
    after: DecisionSnapshot | None
    checkpoint: ConversationCheckpoint | None
    objective_status_consistent: bool
    tool_calls_consistent: bool
    reply_consistent: bool
    baseline_context_audit: dict[str, Any]
    compressed_context_audit: dict[str, Any]
    differences: list[str] = field(default_factory=list)
    summary_error: str | None = None

    @property
    def consistent(self) -> bool:
        return (
            self.after is not None
            and self.checkpoint is not None
            and self.summary_error is None
            and self.objective_status_consistent
            and self.tool_calls_consistent
            and self.reply_consistent
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "consistent": self.consistent,
            "objective_status_consistent": self.objective_status_consistent,
            "tool_calls_consistent": self.tool_calls_consistent,
            "reply_consistent": self.reply_consistent,
            "before": self.before.to_dict(),
            "after": self.after.to_dict() if self.after else None,
            "checkpoint": self.checkpoint.to_dict() if self.checkpoint else None,
            "baseline_context_audit": dict(self.baseline_context_audit),
            "compressed_context_audit": dict(self.compressed_context_audit),
            "differences": list(self.differences),
            "summary_error": self.summary_error,
        }


@dataclass(slots=True)
class ContextSummaryQualityEvaluator:
    """Compare the same model decision before and after checkpoint compression."""

    context_builder: AgentContextBuilder
    summary_manager: ContextSummaryManager
    decision_client: AgentLLMClient
    trace_recorder: TraceRecorder | None = None
    timeout_seconds: float = 30.0
    reply_comparator: ReplyComparator = field(default=lambda left, right: left == right)

    def evaluate(self, *, message: UserMessage, trace_id: str) -> DecisionConsistencyReport:
        """Generate a checkpoint and return a decision-consistency report.

        No tool is executed in either branch, so the only intentional mutation
        is the checkpoint written by ``ContextSummaryManager``.
        """

        baseline_context = self.context_builder.build(
            message,
            trace_id=f"{trace_id}:before",
        )
        before = self._decision_snapshot(
            baseline_context,
            trace_id=f"{trace_id}:decision:before",
        )
        summary_result = self.summary_manager.summarize_for_quality_evaluation(
            conversation_id=message.conversation_id,
            trace_id=f"{trace_id}:summary",
        )
        if not summary_result.summarized or summary_result.checkpoint is None:
            report = DecisionConsistencyReport(
                before=before,
                after=None,
                checkpoint=None,
                objective_status_consistent=False,
                tool_calls_consistent=False,
                reply_consistent=False,
                baseline_context_audit=dict(baseline_context.audit),
                compressed_context_audit={},
                differences=["checkpoint was not generated"],
                summary_error=summary_result.reason,
            )
            self._record(trace_id, report)
            return report

        compressed_context = self.context_builder.build(
            message,
            trace_id=f"{trace_id}:after",
        )
        after = self._decision_snapshot(
            compressed_context,
            trace_id=f"{trace_id}:decision:after",
        )
        objective_status_consistent = before.objective_status == after.objective_status
        tool_calls_consistent = before.tool_calls == after.tool_calls
        reply_consistent = self.reply_comparator(before.reply_to_user, after.reply_to_user)
        differences: list[str] = []
        if not objective_status_consistent:
            differences.append(
                f"objective_status changed: {before.objective_status!r} -> {after.objective_status!r}"
            )
        if not tool_calls_consistent:
            differences.append(
                "tool_calls changed: "
                f"{json.dumps(before.tool_calls, ensure_ascii=False, sort_keys=True)} -> "
                f"{json.dumps(after.tool_calls, ensure_ascii=False, sort_keys=True)}"
            )
        if not reply_consistent:
            differences.append(
                f"reply_to_user changed: {before.reply_to_user!r} -> {after.reply_to_user!r}"
            )
        report = DecisionConsistencyReport(
            before=before,
            after=after,
            checkpoint=summary_result.checkpoint,
            objective_status_consistent=objective_status_consistent,
            tool_calls_consistent=tool_calls_consistent,
            reply_consistent=reply_consistent,
            baseline_context_audit=dict(baseline_context.audit),
            compressed_context_audit=dict(compressed_context.audit),
            differences=differences,
        )
        self._record(trace_id, report)
        return report

    def _decision_snapshot(self, context: BuiltContext, *, trace_id: str) -> DecisionSnapshot:
        raw_response = self.decision_client.complete(
            context.messages,
            trace_id=trace_id,
            timeout_seconds=self.timeout_seconds,
        )
        action, errors = parse_action(raw_response)
        if errors:
            raise ValueError("decision model contract invalid: " + "; ".join(errors))
        return DecisionSnapshot.from_action(action)

    def _record(self, trace_id: str, report: DecisionConsistencyReport) -> None:
        if self.trace_recorder is not None:
            self.trace_recorder.record(
                trace_id,
                "context_summary_decision_consistency",
                report.to_dict(),
                level="INFO" if report.consistent else "WARN",
            )


def canonical_value(value: Any) -> Any:
    """Return a stable JSON-compatible value for semantic tool comparison."""

    if isinstance(value, dict):
        return {
            str(key): canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        return [canonical_value(item) for item in value]
    if isinstance(value, tuple):
        return [canonical_value(item) for item in value]
    return value


def normalize_reply(text: str) -> str:
    """Remove formatting-only variation while retaining reply semantics."""

    normalized = unicodedata.normalize("NFC", str(text or "")).strip()
    return " ".join(normalized.split())


__all__ = [
    "ContextSummaryQualityEvaluator",
    "DecisionConsistencyReport",
    "DecisionSnapshot",
    "canonical_value",
    "normalize_reply",
]
