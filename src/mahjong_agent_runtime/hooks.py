"""定义运行时钩子事件并按注册顺序分发，供旁路能力订阅主链路结果。"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(slots=True)
class HookEvent:
    """运行时钩子收到的不可变事件名称、链路标识与载荷。"""

    name: str
    trace_id: str
    payload: dict[str, Any] = field(default_factory=dict)


HookHandler = Callable[[HookEvent], None]


@dataclass(slots=True)
class HookManager:
    """注册并顺序分发旁路回调；单个钩子失败默认不破坏主链路。

    导航：``AgentRuntime.__post_init__`` 注册匹配与群看板触发器，各 Service 在生命周期
    节点调用 :meth:`emit`。这种间接调用不会完整出现在 IDE 的静态调用层级中。
    """

    fail_fast: bool = False
    handlers: dict[str, list[HookHandler]] = field(default_factory=lambda: defaultdict(list))

    def register(self, event_name: str, handler: HookHandler) -> None:
        self.handlers[str(event_name)].append(handler)

    def emit(self, event_name: str, *, trace_id: str, payload: dict[str, Any] | None = None) -> list[Exception]:
        errors: list[Exception] = []
        event = HookEvent(name=str(event_name), trace_id=trace_id, payload=dict(payload or {}))
        for handler in list(self.handlers.get(event.name, [])):
            try:
                handler(event)
            except Exception as exc:
                if self.fail_fast:
                    raise
                errors.append(exc)
        return errors
