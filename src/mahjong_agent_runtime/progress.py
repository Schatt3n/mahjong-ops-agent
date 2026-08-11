"""检测智能体循环中的重复观察、无进展和周期性动作，防止死循环。

"""

from __future__ import annotations

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
    """构建执行进展hint，只组合调用方允许进入当前作用域的数据。

"""

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
    """检查一个非终止 Agent 步骤的结果。"""

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
        """转换为只包含可序列化字段的字典表示。"""
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
    """跟踪动作和观察结果，识别重复、停滞或短周期循环。

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
        """完成数据类初始化后的依赖组装和不变量校验。"""
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
        """执行后检查工具步骤并分类其进度。"""

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
        """检查失败的运行时步骤，例如无效的输出契约。"""

        return self._observe(
            observation_kind=feedback_kind,
            observation_payload={"kind": feedback_kind, "payload": stable_value(payload)},
            result_signatures=[],
            progress_made=False,
            progress_reasons=[],
            step_index=step_index,
        )

    def feedback_result(self, decision: ProgressDecision) -> ToolResult:
        """创建可附加到模型上下文的结构化反馈。"""

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

    def snapshot(self) -> dict[str, Any]:
        """在工作进程之间仅持久化确定性的循环检测状态。"""

        return {
            "observation_history": list(self._observation_history),
            "seen_result_signatures": sorted(self._seen_result_signatures),
            "consecutive_no_progress_steps": self._consecutive_no_progress_steps,
            "replan_attempts": self._replan_attempts,
        }

    def restore(self, payload: dict[str, Any]) -> None:
        """在继续循环检测时不遗漏崩溃前已出现的重复。"""

        self._observation_history = [
            str(item)
            for item in payload.get("observation_history") or []
        ]
        self._seen_result_signatures = {
            str(item)
            for item in payload.get("seen_result_signatures") or []
        }
        self._consecutive_no_progress_steps = max(
            0,
            int(payload.get("consecutive_no_progress_steps") or 0),
        )
        self._replan_attempts = max(0, int(payload.get("replan_attempts") or 0))

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
        """真实写入后，清除先前后端状态留下的指纹。"""

        self._observation_history.clear()
        self._seen_result_signatures.clear()
        self._consecutive_no_progress_steps = 0
        self._replan_attempts = 0


def detect_tail_cycle(history: list[str], max_cycle_period: int) -> int | None:
    """如果存在，返回最短的重复多步尾部周期。"""

    upper = min(max(0, int(max_cycle_period)), len(history) // 2)
    for period in range(2, upper + 1):
        if history[-period:] == history[-2 * period : -period]:
            return period
    return None


def consecutive_tail_count(history: list[str]) -> int:
    """仅统计历史末尾紧邻重复的观测。"""

    if not history:
        return 0
    count = 1
    for item in reversed(history[:-1]):
        if item != history[-1]:
            break
        count += 1
    return count


def tool_result_payload(result: ToolResult) -> dict[str, Any]:
    """保留结果事实，同时去除执行身份噪声。"""

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
    """哈希前递归移除易变的运行时字段。"""

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
    """为追踪与比较构建简短的确定性指纹。"""

    encoded = json.dumps(stable_value(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
