"""持久化未来任务并按时重新进入主智能体流程。

"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .models import ScheduledAgentTask, now


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
    _stop_event: threading.Event = field(init=False, repr=False)
    _thread: threading.Thread | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        """完成数据类初始化后的依赖组装和不变量校验。"""
        self.poll_interval_seconds = max(0.05, float(self.poll_interval_seconds))
        self.batch_limit = max(1, int(self.batch_limit))
        self.max_attempts = max(1, int(self.max_attempts))
        self.retry_delay_seconds = max(1, int(self.retry_delay_seconds))
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
        """领取到期租约，调用 handler，并落为完成、等待重试或最终失败。"""
        stamp = at or now()
        due = self.store.due_scheduled_tasks(at=stamp, limit=self.batch_limit)
        claimed_count = 0
        for candidate in due:
            claimed = self.store.claim_scheduled_task(candidate.task_id, at=stamp)
            if claimed is None:
                continue
            claimed_count += 1
            trace_id = f"trace_scheduled_{uuid.uuid4().hex[:12]}"
            self.trace_recorder.record(
                trace_id,
                "scheduled_agent_task_claimed",
                {"task": claimed.to_dict()},
            )
            try:
                self.handler(claimed, trace_id)
            except Exception as exc:
                task, transition = self.store.fail_scheduled_task(
                    claimed.task_id,
                    trace_id=trace_id,
                    error=f"{type(exc).__name__}: {exc}",
                    max_attempts=self.max_attempts,
                    retry_delay_seconds=self.retry_delay_seconds,
                    at=stamp,
                )
                self.trace_recorder.record(
                    trace_id,
                    "scheduled_agent_task_failed",
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
                at=stamp,
            )
            self.trace_recorder.record(
                trace_id,
                "scheduled_agent_task_completed",
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
