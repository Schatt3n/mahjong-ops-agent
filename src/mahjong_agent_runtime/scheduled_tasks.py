"""持久化未来任务并按时重新进入主智能体流程。

"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .models import ScheduledAgentTask, now


class ScheduledTaskDeferred(RuntimeError):
    """业务处理因权限或通道开关暂停，不等同于已执行成功或传输失败。"""

    def __init__(self, reason: str, delay_seconds: int = 60) -> None:
        """携带结构化暂停原因和有限重查间隔，不修改权限、外发开关或业务状态。"""
        super().__init__(reason)
        self.reason = str(reason)
        self.delay_seconds = max(1, int(delay_seconds))


@dataclass(slots=True)
class ScheduledAgentTaskScheduler:
    """按租约领取持久化定时任务，并把业务分发交给应用层回调。

    导航：Application 将 ``handle_due_scheduled_agent_task`` 注入 ``handler``。本类只做
    查询、领取、成功完成或失败重试；具体 ``task_type`` 的含义不在这里判断。
    """
    store: Any
    handler: Callable[[ScheduledAgentTask, str], None]
    trace_recorder: Any
    poll_interval_seconds: float = 1.0
    batch_limit: int = 50
    max_attempts: int = 3
    retry_delay_seconds: int = 60
    lease_seconds: int = 180
    clock: Callable[[], datetime] = now
    _stop_event: threading.Event = field(init=False, repr=False)
    _thread: threading.Thread | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        """完成数据类初始化后的依赖组装和不变量校验。"""
        self.poll_interval_seconds = max(0.05, float(self.poll_interval_seconds))
        self.batch_limit = max(1, int(self.batch_limit))
        self.max_attempts = max(1, int(self.max_attempts))
        self.retry_delay_seconds = max(1, int(self.retry_delay_seconds))
        if type(self.lease_seconds) is not int or self.lease_seconds <= 0:
            raise ValueError("lease_seconds must be a positive integer")
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="scheduled-agent-task-scheduler", daemon=True)
        self._thread.start()

    def stop(self, timeout_seconds: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, float(timeout_seconds)))

    def run_due_once(self, *, at: datetime | None = None) -> int:
        """查询本批到期任务；生产每次领取重读时钟，显式at则保持确定性模拟时间。"""
        stamp = at if at is not None else self.clock()
        due = self.store.due_scheduled_tasks(at=stamp, limit=self.batch_limit)
        return self._run_tasks([candidate.task_id for candidate in due], at)

    def run_task_once(self, task_id: str, *, at: datetime | None = None) -> int:
        """只执行指定且到期的任务；手动沙盒触发复查时不领取其他领域任务。"""
        return self._run_tasks([task_id], at)

    def _run_tasks(self, task_ids: list[str], fixed_at: datetime | None) -> int:
        """逐任务领取并按结算时刻落库，前一任务的审查耗时不能吃掉后一任务租约。

        显式fixed_at用于历史/模拟回放，所有步骤都保持该时间；生产入口传None，领取与
        完成、失败、延期分别读取clock，重试间隔也从真实处理结束时刻起算。
        """
        claimed_count = 0
        for task_id in task_ids:
            claimed = self.store.claim_scheduled_task(
                task_id, at=fixed_at if fixed_at is not None else self.clock(), lease_seconds=self.lease_seconds,
            )
            if claimed is None:
                continue
            claimed_count += 1
            # 在任何回调前捕获不可变领取凭证；handler或其他worker可能已推进同一task_id。
            claim_credentials = {
                "expected_attempts": claimed.attempts,
                "expected_lease_until": claimed.lease_until,
            }
            trace_id = f"trace_scheduled_{uuid.uuid4().hex[:12]}"
            self.trace_recorder.record(
                trace_id,
                "scheduled_agent_task_claimed",
                {"task": claimed.to_dict()},
            )
            try:
                self.handler(claimed, trace_id)
            except ScheduledTaskDeferred as exc:
                task, transition = self.store.defer_scheduled_task(
                    claimed.task_id, trace_id=trace_id, reason=exc.reason,
                    retry_delay_seconds=exc.delay_seconds,
                    at=fixed_at if fixed_at is not None else self.clock(), **claim_credentials,
                )
                self.trace_recorder.record(
                    trace_id,
                    "scheduled_agent_task_deferred" if transition else "scheduled_agent_task_deferral_ignored",
                    {"task": task.to_dict() if task else None,
                     "transition": transition.to_dict() if transition else None, "reason": exc.reason},
                )
                continue
            except Exception as exc:
                task, transition = self.store.fail_scheduled_task(
                    claimed.task_id,
                    trace_id=trace_id,
                    error=f"{type(exc).__name__}: {exc}",
                    max_attempts=self.max_attempts,
                    retry_delay_seconds=self.retry_delay_seconds,
                    at=fixed_at if fixed_at is not None else self.clock(),
                    **claim_credentials,
                )
                self.trace_recorder.record(
                    trace_id,
                    "scheduled_agent_task_failed" if transition else "scheduled_agent_task_failure_ignored",
                    {
                        "task": task.to_dict() if task else None,
                        "transition": transition.to_dict() if transition else None,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    level="ERROR",
                )
                continue
            task, transition = self.store.complete_scheduled_task(
                claimed.task_id,
                trace_id=trace_id,
                at=fixed_at if fixed_at is not None else self.clock(),
                **claim_credentials,
            )
            self.trace_recorder.record(
                trace_id,
                "scheduled_agent_task_completed" if transition else "scheduled_agent_task_completion_ignored",
                {
                    "task": task.to_dict() if task else None,
                    "transition": transition.to_dict() if transition else None,
                },
            )
        return claimed_count

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.run_due_once()
            self._stop_event.wait(self.poll_interval_seconds)
