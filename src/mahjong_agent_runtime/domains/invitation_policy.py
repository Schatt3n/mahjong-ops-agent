"""封装邀约策略相关的领域规则和不变量。

"""

from __future__ import annotations

from ..models import GameParticipant, InviteStatus

CONFIRMED_CANDIDATE_STATUSES = {"accepted", "confirmed", "arrived"}

UNCONFIRMED_CANDIDATE_STATUSES = {"declined", "negotiating", "no_reply"}


class ClaimedSeatReplyConflict(ValueError):
    """未确认邀约回复与当前已占席位冲突，落库前拒绝且保留原状态。"""


def validate_claimed_seat_reply(participant: GameParticipant | None, status: str) -> None:
    """在参与事务内区分未确认邀约的进度和已占席位的明确退出。"""
    if participant is not None and participant.status in {"joined", "confirmed"} and status in {"negotiating", "no_reply"}:
        raise ClaimedSeatReplyConflict(
            "该顾客已占有席位，协商或未回复不能释放座位。原座位与整桌时间保持；"
            "如只讨论到店或开场时间，按现有安排回复，有待其他参与人同意时如实说明。"
            "仅有顾客明确退出的依据时才记录拒绝。"
        )


def invite_status_from_candidate_status(status: str) -> InviteStatus:
    mapping = {
        "accepted": InviteStatus.CONFIRMED,
        "confirmed": InviteStatus.CONFIRMED,
        "arrived": InviteStatus.CONFIRMED,
        "declined": InviteStatus.DECLINED,
        "negotiating": InviteStatus.NEGOTIATING,
        "no_reply": InviteStatus.NO_REPLY,
    }
    return mapping.get(status, InviteStatus.NEGOTIATING)
