"""共享群看板归属与安全状态投影；写入仍由各 Store 的原子边界负责。"""

from __future__ import annotations

from typing import Any

from ..models import ScheduledTaskStatus


def board_room_candidates(game, policies, links, switch=None):
    """按可信会话协议和已配置群策略选范围，绝不让模型指定群或修改授权。

    普通私聊可进入本店已托管且启用看板的群；群原生会话或已有来源关联只回原群。
    群转私聊记录只用于保留来源，不是普通私聊进入看板的前提。
    """
    eligible = {item.room_id: item for item in policies if item.managed and item.board_enabled}
    origin_rooms = {item.room_id for item in links if item.link_type in {"origin", "private_switch_created"}}
    conversation_id = game.conversation_id
    native_group = conversation_id.startswith(("group:", "wechaty:room:"))
    if native_group:
        origin_rooms.update(
            room_id for room_id in eligible
            if conversation_id.startswith(f"group:{room_id}:customer:")
            or conversation_id == f"wechaty:room:{room_id}"
            or conversation_id.startswith(f"wechaty:room:{room_id}:post:")
        )
    if switch is not None and switch.private_conversation_id == conversation_id:
        origin_rooms.add(switch.room_id)
    room_ids = origin_rooms if native_group or origin_rooms else set(eligible)
    return [eligible[room_id] for room_id in sorted(room_ids) if room_id in eligible]


def board_arrangement_projection(game_id, policies, links, tasks, snapshots) -> dict[str, Any]:
    """只向模型投影安排状态和计数，不暴露群标识、成员或传输内部信息。"""
    room_ids = {item.room_id for item in links}
    active_tasks = [item for item in tasks if item.task_type == "publish_group_board"
                    and item.aggregate_type == "group_room" and item.aggregate_id in room_ids
                    and item.status in {ScheduledTaskStatus.PENDING, ScheduledTaskStatus.PROCESSING}]
    publication_sent = any(
        snapshot is not None and any(item.game_id == game_id for item in snapshot.items)
        for snapshot in snapshots
    )
    return {
        "status": "scheduled" if active_tasks else ("registered" if room_ids else "unconfigured"),
        "linked_room_count": len(room_ids),
        "pending_task_count": len(active_tasks),
        "outbound_authorized_room_count": sum(item.room_id in room_ids and item.managed
                                               and item.board_enabled and item.outbound_enabled for item in policies),
        "publication_sent": publication_sent,
        "instruction": ("这是已持久化的看板归属与刷新责任，不等于已发布或已联系其他人。"
                        "publication_sent=true才说明已有成功发布快照；外发仍受群权限和外发开关约束。"
                        "unconfigured表示本店尚无可用看板群，不能宣称已安排发布。"),
    }
