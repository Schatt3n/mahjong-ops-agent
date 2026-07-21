"""Domain rules for invitation policy."""

from __future__ import annotations

from ..models import InviteStatus

CONFIRMED_CANDIDATE_STATUSES = {"accepted", "confirmed", "arrived"}

UNCONFIRMED_CANDIDATE_STATUSES = {"declined", "negotiating", "no_reply"}

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
