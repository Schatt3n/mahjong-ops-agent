"""声明群聊通道发送能力，由应用层适配具体渠道。"""

from __future__ import annotations

from typing import Any, Protocol


class GroupMessenger(Protocol):
    """群聊 Handler 依赖的结构化发送接口。

    实现：应用层 ``WechatyGroupMessenger`` 通过同名方法满足本 Protocol，没有显式继承。
    在该实现中，发送会按适用场景检查房间策略和桥接开关；返回 ``suppressed:*`` 表示
    没有真实外发。这些是 Wechaty 适配器语义，不是 Protocol 对所有实现的保证。
    """
    def send_group_message(self, room_id: str, text: str, *, metadata: dict[str, Any] | None = None) -> str:
        ...

    def send_private_message(
        self,
        external_user_id: str,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        ...


__all__ = ["GroupMessenger"]
