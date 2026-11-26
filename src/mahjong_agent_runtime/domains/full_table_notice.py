"""把私聊成局的持久化状态转换为受托管群约束的单收件人通知任务。"""

from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json

from ..models import Game, GameStatus, ScheduledAgentTask


FULL_TABLE_NOTICE_TASK_TYPE = "group_game_full_notice"
_OCCUPIED_PARTICIPATION = {"joined", "confirmed"}


def full_table_notice_facts(game: Game) -> dict:
    """回告只投影满桌和开场条件，避免连带重述档位、时长、包间或其他顾客。"""
    return {
        "status": game.status.value,
        "remaining_seats": game.remaining_seats(),
        "planned_start_at": game.planned_start_at.isoformat() if game.planned_start_at else None,
        "public_requirement": {
            key: game.requirement[key] for key in ("game_type", "start_time_kind", "start_time") if key in game.requirement
        },
    }


def ready_table_fingerprint(game: Game) -> str:
    """给同一场成局的时间和实际席位取稳定摘要，重开或改期后旧任务不能误发。"""
    snapshot = [
        full_table_notice_facts(game),
        sorted((row.customer_id, row.seat_count) for row in game.participants if row.status in _OCCUPIED_PARTICIPATION),
    ]
    return hashlib.sha256(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def full_table_private_contact(store, game: Game, *, room_id: str, customer_id: str):
    """根据原局关联和同一渠道好友授权核对收件人，群派生通知不跨账号或降级公开发送。"""
    policy = store.get_group_room_policy(room_id)
    if (policy is None or not policy.managed or not policy.board_enabled or not policy.outbound_enabled
            or not any(link.room_id == room_id for link in store.game_conversation_links(game_id=game.game_id))
            or not any(row.customer_id == customer_id and row.status in _OCCUPIED_PARTICIPATION for row in game.participants)):
        return None
    identity = store.get_channel_identity_for_customer(customer_id, channel=policy.channel)
    profile = store.customers.get(customer_id)
    if identity is None or not identity.is_friend or not identity.can_private_message or getattr(profile, "no_contact", False):
        return None
    return identity


def full_table_notice_tasks(store, game: Game, *, filled_by: str, at: datetime) -> list[ScheduledAgentTask]:
    """在满桌写事务内生成已占席好友的通知责任，私聊刚报名者由本轮回执承接。"""
    expires_at = game.planned_start_at or at + timedelta(minutes=30)
    if (game.status != GameStatus.READY or game.remaining_seats() != 0 or game.actual_started_at is not None
            or expires_at <= at):
        return []
    rooms = sorted({link.room_id for link in store.game_conversation_links(game_id=game.game_id)})
    fingerprint = ready_table_fingerprint(game)
    tasks = []
    for participant in game.participants:
        if participant.customer_id == filled_by or participant.status not in _OCCUPIED_PARTICIPATION:
            continue
        for room_id in rooms:
            identity = full_table_private_contact(store, game, room_id=room_id, customer_id=participant.customer_id)
            if identity is None:
                continue
            # 同一人不因同局出现在多个群而收到多条通知。
            key = f"{FULL_TABLE_NOTICE_TASK_TYPE}:{game.game_id}:{identity.customer_id}:{fingerprint}"
            tasks.append(ScheduledAgentTask(
                task_id=key, idempotency_key=key, task_type=FULL_TABLE_NOTICE_TASK_TYPE,
                aggregate_type="game", aggregate_id=game.game_id,
                conversation_id=identity.private_conversation_id, subject_id=identity.customer_id,
                subject_name=identity.public_name, due_at=at, created_at=at, updated_at=at,
                payload={"room_id": room_id, "channel": identity.channel, "fingerprint": fingerprint,
                         "expires_at": expires_at.isoformat()},
            ))
            break
    return tasks
