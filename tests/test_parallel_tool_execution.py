from __future__ import annotations

import threading
import time
from collections.abc import Callable

from mahjong_agent_runtime.models import AgentAction, ToolCall, ToolResult, UserMessage
from mahjong_agent_runtime.domains.tools import ToolDefinition, ToolGateway
from mahjong_agent_runtime.services import ToolExecutionService
from mahjong_agent_runtime.stores.memory import InMemoryAgentStore
from mahjong_agent_runtime.tracing import InMemoryTraceRecorder


def test_independent_parallel_safe_reads_overlap_and_restore_declared_order() -> None:
    tracker = ConcurrencyTracker()

    def slow_first(*_: object) -> ToolResult:
        with tracker.track():
            time.sleep(0.08)
        return successful_result("read_first", value=1)

    def fast_second(*_: object) -> ToolResult:
        with tracker.track():
            time.sleep(0.02)
        return successful_result("read_second", value=2)

    service, trace = service_with_tools(
        read_first=definition("read_first", slow_first, parallel_safe=True),
        read_second=definition("read_second", fast_second, parallel_safe=True),
    )

    result = execute(
        service,
        [
            ToolCall("read_first", reason="read one", call_id="read_1", depends_on=[]),
            ToolCall("read_second", reason="read two", call_id="read_2", depends_on=[]),
        ],
    )

    assert tracker.max_active == 2
    assert [item.call_id for item in result.tool_results] == ["read_1", "read_2"]
    assert [item.result["value"] for item in result.tool_results] == [1, 2]
    batches = [event for event in trace.events if event.step == "tool_batch_started"]
    assert len(batches) == 1
    assert batches[0].content["execution_mode"] == "parallel_read"


def test_dependent_tool_waits_for_its_prerequisite() -> None:
    prerequisite_finished = threading.Event()

    def prerequisite(*_: object) -> ToolResult:
        prerequisite_finished.set()
        return successful_result("prerequisite")

    def dependent(*_: object) -> ToolResult:
        assert prerequisite_finished.is_set()
        return successful_result("dependent")

    service, trace = service_with_tools(
        prerequisite=definition("prerequisite", prerequisite, parallel_safe=True),
        dependent=definition("dependent", dependent, parallel_safe=True),
    )

    result = execute(
        service,
        [
            ToolCall("prerequisite", reason="first", call_id="first", depends_on=[]),
            ToolCall("dependent", reason="second", call_id="second", depends_on=["first"]),
        ],
    )

    assert [item.call_id for item in result.tool_results] == ["first", "second"]
    batches = [event for event in trace.events if event.step == "tool_batch_started"]
    assert [event.content["call_ids"] for event in batches] == [["first"], ["second"]]


def test_failed_prerequisite_blocks_dependent_write_without_executing_it() -> None:
    write_count = 0

    def failing_read(*_: object) -> ToolResult:
        return ToolResult(name="failing_read", called=False, allowed=False, error="upstream unavailable")

    def write_tool(*_: object) -> ToolResult:
        nonlocal write_count
        write_count += 1
        return successful_result("write_tool")

    service, _ = service_with_tools(
        failing_read=definition("failing_read", failing_read, parallel_safe=True),
        write_tool=definition("write_tool", write_tool, execution_mode="state_write"),
    )

    result = execute(
        service,
        [
            ToolCall("failing_read", reason="read", call_id="read", depends_on=[]),
            ToolCall("write_tool", reason="write", call_id="write", depends_on=["read"]),
        ],
    )

    assert write_count == 0
    assert result.tool_results[1].called is False
    assert result.tool_results[1].error == "tool dependency failed: read"


def test_write_tools_are_serial_even_when_model_declares_them_independent() -> None:
    tracker = ConcurrencyTracker()

    def write_one(*_: object) -> ToolResult:
        with tracker.track():
            time.sleep(0.03)
        return successful_result("write_one")

    def write_two(*_: object) -> ToolResult:
        with tracker.track():
            time.sleep(0.03)
        return successful_result("write_two")

    service, trace = service_with_tools(
        write_one=definition("write_one", write_one, execution_mode="state_write"),
        write_two=definition("write_two", write_two, execution_mode="state_write"),
    )

    result = execute(
        service,
        [
            ToolCall("write_one", reason="write one", call_id="write_1", depends_on=[]),
            ToolCall("write_two", reason="write two", call_id="write_2", depends_on=[]),
        ],
    )

    assert tracker.max_active == 1
    assert len(result.tool_results) == 2
    batches = [event for event in trace.events if event.step == "tool_batch_started"]
    assert [event.content["execution_mode"] for event in batches] == ["sequential", "sequential"]


def test_legacy_calls_without_dependency_metadata_remain_sequential() -> None:
    tracker = ConcurrencyTracker()

    def legacy_one(*_: object) -> ToolResult:
        with tracker.track():
            time.sleep(0.02)
        return successful_result("legacy_one")

    def legacy_two(*_: object) -> ToolResult:
        with tracker.track():
            time.sleep(0.02)
        return successful_result("legacy_two")

    service, trace = service_with_tools(
        legacy_one=definition("legacy_one", legacy_one, parallel_safe=True),
        legacy_two=definition("legacy_two", legacy_two, parallel_safe=True),
    )

    result = execute(
        service,
        [
            ToolCall("legacy_one", reason="legacy one"),
            ToolCall("legacy_two", reason="legacy two"),
        ],
    )

    assert tracker.max_active == 1
    assert len(result.tool_results) == 2
    assert not [event for event in trace.events if event.step == "tool_batch_started"]


def test_parallel_worker_limit_can_disable_parallel_execution() -> None:
    tracker = ConcurrencyTracker()

    def read_one(*_: object) -> ToolResult:
        with tracker.track():
            time.sleep(0.02)
        return successful_result("read_one")

    def read_two(*_: object) -> ToolResult:
        with tracker.track():
            time.sleep(0.02)
        return successful_result("read_two")

    service, _ = service_with_tools(
        max_parallel_read_tools=1,
        read_one=definition("read_one", read_one, parallel_safe=True),
        read_two=definition("read_two", read_two, parallel_safe=True),
    )

    result = execute(
        service,
        [
            ToolCall("read_one", reason="read one", call_id="read_1", depends_on=[]),
            ToolCall("read_two", reason="read two", call_id="read_2", depends_on=[]),
        ],
    )

    assert tracker.max_active == 1
    assert len(result.tool_results) == 2


class ConcurrencyTracker:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def track(self) -> "TrackedExecution":
        return TrackedExecution(self)


class TrackedExecution:
    def __init__(self, tracker: ConcurrencyTracker) -> None:
        self.tracker = tracker

    def __enter__(self) -> None:
        with self.tracker.lock:
            self.tracker.active += 1
            self.tracker.max_active = max(self.tracker.max_active, self.tracker.active)

    def __exit__(self, *_: object) -> None:
        with self.tracker.lock:
            self.tracker.active -= 1


def successful_result(name: str, **payload: object) -> ToolResult:
    return ToolResult(name=name, called=True, allowed=True, result=dict(payload))


def definition(
    name: str,
    handler: Callable[..., ToolResult],
    *,
    execution_mode: str = "read_only",
    parallel_safe: bool = False,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"test tool {name}",
        risk_level="low",
        execution_mode=execution_mode,
        schema={"type": "object", "additionalProperties": False},
        handler=handler,
        parallel_safe=parallel_safe,
    )


def service_with_tools(
    *,
    max_parallel_read_tools: int = 4,
    **tools: ToolDefinition,
) -> tuple[ToolExecutionService, InMemoryTraceRecorder]:
    store = InMemoryAgentStore()
    trace = InMemoryTraceRecorder()
    gateway = ToolGateway(store=store, tools=tools, trace_recorder=trace)
    return (
        ToolExecutionService(
            store=store,
            tool_gateway=gateway,
            trace_recorder=trace,
            max_parallel_read_tools=max_parallel_read_tools,
        ),
        trace,
    )


def execute(service: ToolExecutionService, calls: list[ToolCall]):
    action = AgentAction(
        goal="test tool scheduling",
        objective_status="needs_tool",
        reasoning_summary="test",
        tool_calls=calls,
    )
    return service.execute_tool_calls(
        action,
        message=UserMessage(
            conversation_id="parallel_tools",
            sender_id="customer",
            sender_name="customer",
            text="test",
            message_id="parallel_tool_message",
        ),
        trace_id="trace_parallel_tools",
        previous_step_tool_results=[],
        step_index=1,
        run_id="run_parallel_tools",
        run_version=0,
    )
