"""实现群聊场景中的messenger能力。

"""

from __future__ import annotations

from typing import Any, Protocol


class GroupMessenger(Protocol):
    """定义群聊messenger需要遵守的抽象接口契约。"""
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
