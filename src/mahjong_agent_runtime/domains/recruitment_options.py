"""校验同需求备选方案及报名联系人权限，不解释客户自然语言。"""

from __future__ import annotations

from typing import Any

from ..models import Game, GameStatus, now
from .game_commitments import game_commitment_window, requirement_commitment_window, requirement_overlaps_game


def creation_option_root(
    games: list[Game], *, conversation_id: str, organizer_id: str,
    requirement: dict[str, Any], alternative_to_game_id: str | None,
) -> str | None:
    """允许显式关联的不同时间待组方案；普通重复建局和已成局冲突仍拒绝。"""
    source = next((game for game in games if game.game_id == alternative_to_game_id), None)
    root = None
    if alternative_to_game_id is not None:
        if source is None:
            raise ValueError("alternative source game not found")
        if source.conversation_id != conversation_id or source.organizer_id != organizer_id:
            raise ValueError("alternative source requires its authenticated organizer and originating conversation")
        if (source.status not in {GameStatus.FORMING, GameStatus.INVITING}
                or (source.expires_at is not None and source.expires_at <= now())):
            raise ValueError("alternative source must still be forming or inviting")
        root = source.option_root_game_id or source.game_id
    for game in games:
        if (game.conversation_id != conversation_id or game.organizer_id != organizer_id
                or game.status not in {GameStatus.FORMING, GameStatus.INVITING, GameStatus.READY}
                or not requirement_overlaps_game(requirement, game)):
            continue
        same_group = root is not None and (game.option_root_game_id or game.game_id) == root
        # 本次能力是不同开场时刻的互斥选项；相同时间重复提交不能伪装成备选。
        different_start = requirement_commitment_window(requirement)[0] != game_commitment_window(game)[0]
        if not same_group or game.status == GameStatus.READY or not different_start:
            raise ValueError(f"active game already exists: {game.game_id}")
    return root


def validate_reported_withdrawal(
    game: Game, *, customer_id: str, reported_by: tuple[str, str], status: str,
) -> None:
    """仅原报名联系人可报告其代登记成员退出；独立报名/接管座位不授予代办权。"""
    organizer_id, conversation_id = reported_by
    participant = next((item for item in game.participants if item.customer_id == customer_id), None)
    if (organizer_id != game.organizer_id or conversation_id != game.conversation_id
            or participant is None or participant.registered_by_customer_id != organizer_id
            or customer_id == organizer_id):
        raise ValueError("registered member withdrawal requires the original registration contact")
    if (status != "declined" or game.status not in {GameStatus.FORMING, GameStatus.INVITING, GameStatus.READY}
            or (game.expires_at is not None and game.expires_at <= now())):
        raise ValueError("reported withdrawal only releases a registered seat in an active pre-play game")
    if participant.status not in {"joined", "confirmed", "declined"}:
        raise ValueError("registered seat is no longer eligible for reported withdrawal")
