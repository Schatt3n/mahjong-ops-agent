"""按模型声明的工具依赖调度执行，不直接承载通道或数据库细节。"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

from ..models import AgentAction, ToolCall, ToolResult, UserMessage
from ..domains.tools import ToolGateway
from .contracts import SingleToolExecution


ToolExecutor = Callable[..., SingleToolExecution]


@dataclass(slots=True)
class ToolCallScheduler:
    """按依赖关系分批执行工具，同时保持原始结果顺序。

    导航：由 ``ToolExecutionService`` 构造；``execute_one`` 是调用方注入的回调，而非
    本模块实现。只有注册表同时声明 ``read_only`` 和 ``parallel_safe`` 的工具可并行。
    """

    tool_gateway: ToolGateway
    trace_recorder: Any
    max_parallel_read_tools: int = 4

    def __post_init__(self) -> None:
        """完成数据类初始化后的依赖组装和不变量校验。"""
        self.max_parallel_read_tools = max(1, int(self.max_parallel_read_tools))

    def execute(
        self,
        action: AgentAction,
        *,
        execute_one: ToolExecutor,
        message: UserMessage,
        trace_id: str,
        previous_step_tool_results: list[ToolResult],
        step_index: int,
        run_id: str,
        run_version: int,
        context_payload: dict[str, Any] | None,
    ) -> tuple[dict[int, ToolResult], bool, bool]:
        """有完整 ``call_id/depends_on`` 时走依赖图，否则兼容旧的顺序模式。"""

        arguments = {
            "execute_one": execute_one,
            "message": message,
            "trace_id": trace_id,
            "previous_step_tool_results": previous_step_tool_results,
            "step_index": step_index,
            "run_id": run_id,
            "run_version": run_version,
            "context_payload": context_payload,
        }
        if self._uses_dependency_graph(action):
            return self._execute_dependency_graph(action, **arguments)
        return self._execute_legacy_sequence(action, **arguments)

    @staticmethod
    def _uses_dependency_graph(action: AgentAction) -> bool:
        return bool(action.tool_calls) and all(
            bool(call.call_id) and call.depends_on is not None for call in action.tool_calls
        )

    def _execute_legacy_sequence(
        self,
        action: AgentAction,
        *,
        execute_one: ToolExecutor,
        message: UserMessage,
        trace_id: str,
        previous_step_tool_results: list[ToolResult],
        step_index: int,
        run_id: str,
        run_version: int,
        context_payload: dict[str, Any] | None,
    ) -> tuple[dict[int, ToolResult], bool, bool]:
        """为没有图元数据的响应保持顺序执行。"""

        results: dict[int, ToolResult] = {}
        consistency_blocked = False
        stale_blocked = False
        for call_index, call in enumerate(action.tool_calls, start=1):
            observed = previous_step_tool_results + [results[index] for index in sorted(results)]
            outcome = execute_one(
                call,
                call_index=call_index,
                call_id=call.call_id,
                observed_results=observed,
                message=message,
                trace_id=trace_id,
                step_index=step_index,
                run_id=run_id,
                run_version=run_version,
                context_payload=context_payload,
            )
            results[call_index] = outcome.result
            consistency_blocked = consistency_blocked or outcome.blocked_by_consistency
            stale_blocked = stale_blocked or outcome.blocked_by_stale_run
            if outcome.blocked_by_consistency or outcome.blocked_by_stale_run:
                break
        return results, consistency_blocked, stale_blocked

    def _execute_dependency_graph(
        self,
        action: AgentAction,
        *,
        execute_one: ToolExecutor,
        message: UserMessage,
        trace_id: str,
        previous_step_tool_results: list[ToolResult],
        step_index: int,
        run_id: str,
        run_version: int,
        context_payload: dict[str, Any] | None,
    ) -> tuple[dict[int, ToolResult], bool, bool]:
        """执行依赖就绪的波次，并保留声明的结果顺序。"""

        calls = {index: call for index, call in enumerate(action.tool_calls, start=1)}
        remaining = set(calls)
        completed_ids: set[str] = set()
        succeeded: dict[str, bool] = {}
        results: dict[int, ToolResult] = {}
        consistency_blocked = False
        stale_blocked = False
        wave_index = 0

        while remaining and not stale_blocked:
            blocked = [
                index
                for index in sorted(remaining)
                if any(
                    dependency in completed_ids and not succeeded.get(dependency, False)
                    for dependency in calls[index].depends_on or []
                )
            ]
            for index in blocked:
                call = calls[index]
                failed = [
                    dependency
                    for dependency in call.depends_on or []
                    if dependency in completed_ids and not succeeded.get(dependency, False)
                ]
                result = self._dependency_failure(call, failed)
                results[index] = result
                remaining.remove(index)
                completed_ids.add(str(call.call_id))
                succeeded[str(call.call_id)] = False
                consistency_blocked = True
                self.trace_recorder.record(trace_id, "tool_dependency_blocked", result.to_dict(), level="WARN")

            if not remaining:
                break
            ready = [
                index
                for index in sorted(remaining)
                if set(calls[index].depends_on or []) <= completed_ids
            ]
            if not ready:
                for index in sorted(remaining):
                    result = self._invalid_graph_result(calls[index])
                    results[index] = result
                    self.trace_recorder.record(trace_id, "tool_dependency_invalid", result.to_dict(), level="WARN")
                consistency_blocked = True
                break

            parallel_ready = [index for index in ready if self._is_parallel_safe(calls[index])]
            is_parallel = len(parallel_ready) >= 2 and self.max_parallel_read_tools > 1
            batch = parallel_ready if is_parallel else [ready[0]]
            wave_index += 1
            observed = previous_step_tool_results + [results[index] for index in sorted(results)]
            outcomes = self._execute_wave(
                batch,
                calls=calls,
                execute_one=execute_one,
                execution_mode="parallel_read" if is_parallel else "sequential",
                wave_index=wave_index,
                observed_results=observed,
                message=message,
                trace_id=trace_id,
                step_index=step_index,
                run_id=run_id,
                run_version=run_version,
                context_payload=context_payload,
            )
            for index, outcome in outcomes.items():
                call = calls[index]
                results[index] = outcome.result
                remaining.remove(index)
                completed_ids.add(str(call.call_id))
                succeeded[str(call.call_id)] = self._result_succeeded(outcome.result)
                consistency_blocked = consistency_blocked or outcome.blocked_by_consistency
                stale_blocked = stale_blocked or outcome.blocked_by_stale_run
        return results, consistency_blocked, stale_blocked

    def _execute_wave(
        self,
        call_indices: list[int],
        *,
        calls: dict[int, ToolCall],
        execute_one: ToolExecutor,
        execution_mode: str,
        wave_index: int,
        observed_results: list[ToolResult],
        message: UserMessage,
        trace_id: str,
        step_index: int,
        run_id: str,
        run_version: int,
        context_payload: dict[str, Any] | None,
    ) -> dict[int, SingleToolExecution]:
        """执行单个图波次并记录批次级耗时。"""

        started = time.perf_counter()
        call_ids = [calls[index].call_id for index in call_indices]
        self.trace_recorder.record(
            trace_id,
            "tool_batch_started",
            {
                "step_index": step_index,
                "wave_index": wave_index,
                "execution_mode": execution_mode,
                "call_ids": call_ids,
                "tool_names": [calls[index].name for index in call_indices],
            },
        )
        common = {
            "observed_results": observed_results,
            "message": message,
            "trace_id": trace_id,
            "step_index": step_index,
            "run_id": run_id,
            "run_version": run_version,
            "context_payload": context_payload,
        }
        if execution_mode == "parallel_read":
            outcomes = self._execute_parallel(call_indices, calls=calls, execute_one=execute_one, common=common)
        else:
            index = call_indices[0]
            call = calls[index]
            outcomes = {
                index: execute_one(call, call_index=index, call_id=call.call_id, **common)
            }
        self.trace_recorder.record(
            trace_id,
            "tool_batch_completed",
            {
                "step_index": step_index,
                "wave_index": wave_index,
                "execution_mode": execution_mode,
                "call_ids": call_ids,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "succeeded": {
                    str(calls[index].call_id): self._result_succeeded(outcome.result)
                    for index, outcome in outcomes.items()
                },
            },
        )
        return outcomes

    def _execute_parallel(
        self,
        call_indices: list[int],
        *,
        calls: dict[int, ToolCall],
        execute_one: ToolExecutor,
        common: dict[str, Any],
    ) -> dict[int, SingleToolExecution]:
        outcomes: dict[int, SingleToolExecution] = {}
        with ThreadPoolExecutor(
            max_workers=min(self.max_parallel_read_tools, len(call_indices)),
            thread_name_prefix="agent-read-tool",
        ) as executor:
            futures = {
                executor.submit(
                    execute_one,
                    calls[index],
                    call_index=index,
                    call_id=calls[index].call_id,
                    observed_results=list(common["observed_results"]),
                    **{key: value for key, value in common.items() if key != "observed_results"},
                ): index
                for index in call_indices
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    outcomes[index] = future.result()
                except Exception as exc:
                    call = calls[index]
                    result = ToolResult(
                        name=call.name,
                        called=False,
                        allowed=False,
                        call_id=call.call_id,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    outcomes[index] = SingleToolExecution(result=result, blocked_by_consistency=True)
                    self.trace_recorder.record(
                        str(common["trace_id"]),
                        "parallel_tool_worker_failed",
                        {"call": call.to_dict(), "error": result.error},
                        level="ERROR",
                    )
        return outcomes

    def _is_parallel_safe(self, call: ToolCall) -> bool:
        definition = self.tool_gateway.tools.get(call.name)
        return bool(definition and definition.execution_mode == "read_only" and definition.parallel_safe)

    @staticmethod
    def _result_succeeded(result: ToolResult) -> bool:
        return bool(result.called and result.allowed and not result.error)

    @staticmethod
    def _dependency_failure(call: ToolCall, failed: list[str]) -> ToolResult:
        return ToolResult(
            name=call.name,
            called=False,
            allowed=False,
            call_id=call.call_id,
            result={
                "failed_dependencies": failed,
                "instruction": "A prerequisite tool failed. Replan from its result before retrying this call.",
            },
            error="tool dependency failed: " + ",".join(failed),
        )

    @staticmethod
    def _invalid_graph_result(call: ToolCall) -> ToolResult:
        return ToolResult(
            name=call.name,
            called=False,
            allowed=False,
            call_id=call.call_id,
            result={"declared_dependencies": list(call.depends_on or [])},
            error="tool dependency graph is unresolved or cyclic",
        )


__all__ = ["ToolCallScheduler"]
