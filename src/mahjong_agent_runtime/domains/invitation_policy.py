"""封装邀约策略相关的领域规则和不变量。

"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from math import isfinite
from typing import Any

from ..models import Game, GameParticipant, GameStatus, InviteDraft, InviteStatus, MessageReference, StateTransition, ToolCall
from .game_domain import recruitment_open_time
from .game_participants import participation_table_terms, planned_start_has_passed
from .stake_values import parse_number
from .value_utils import first_present_value

CONFIRMED_CANDIDATE_STATUSES = {"accepted", "confirmed", "arrived"}

UNCONFIRMED_CANDIDATE_STATUSES = {"declined", "negotiating", "no_reply"}

_MANUAL_DELIVERY_CLAIM = "manual_delivery_claim"
_MANUAL_DELIVERY_RECEIPT = "manual_delivery_receipt"


def invite_creation_metadata(metadata: object) -> dict[str, Any]:
    """定向外发的认领和回执只能由 Store 写入，生成草稿的调用者不能预填。"""
    return {key: value for key, value in metadata.items()
            if key not in {_MANUAL_DELIVERY_CLAIM, _MANUAL_DELIVERY_RECEIPT}} if isinstance(metadata, dict) else {}


def invite_duration_terms(game: Game) -> dict[str, float | None]:
    """审稿基线保存明确时长和上限；内部默认排期永远不是对外邀约的条件。"""
    duration = parse_number(first_present_value(game.requirement, "duration_hours", "duration"))
    maximum = parse_number(game.requirement.get("max_duration_hours"))
    return {
        "duration_hours": duration if duration is not None and isfinite(duration) and duration > 0 else None,
        "max_duration_hours": maximum if maximum is not None and isfinite(maximum) and maximum > 0 else None,
    }


def manual_invite_block_reason(
    draft: InviteDraft, game: Game | None, source: MessageReference | None, *, at: datetime,
) -> str | None:
    """检查店家此刻还能否处理该草稿，使用原始审稿桌况，不从文案猜条件。"""
    if has_manual_invite_claim(draft):
        return "invite_delivery_requires_reconciliation"
    if draft.status != InviteStatus.PENDING_APPROVAL:
        return "invite_not_pending_approval"
    if (game is None or game.status not in {GameStatus.FORMING, GameStatus.INVITING}
            or game.actual_started_at is not None or game.remaining_seats() <= 0
            or (game.expires_at is not None and at >= game.expires_at)):
        return "invite_game_not_recruiting"
    if planned_start_has_passed(game, at=at):
        # 普通招募草稿没有本次晚到的参与人授权；独立协商流程核对自己的回执。
        return "invite_planned_start_passed"
    opens_at = recruitment_open_time(game)
    if opens_at is not None and at < opens_at:
        return "invite_recruitment_not_open"
    if any(draft.customer_id in claim["known_member_ids"] or draft.customer_id == claim["contact_id"]
           for claim in game.seat_claims()):
        return "invite_recipient_already_seated"
    if (source is None or source.message_id != draft.draft_id or source.conversation_id != game.conversation_id
            or source.business_ref_type != "invite_draft" or source.business_ref_id != draft.draft_id
            or source.recipient_id != draft.customer_id or source.text != draft.message_text
            or source.metadata.get("original_participation_terms") != participation_table_terms(game)
            or source.metadata.get("original_invite_duration_terms") != invite_duration_terms(game)):
        return "invite_terms_or_source_changed"
    return None


def claim_manual_invite(
    draft: InviteDraft, game: Game | None, source: MessageReference | None, *, trace_id: str, at: datetime,
) -> tuple[InviteDraft, StateTransition]:
    """写事务内同时核对当前桌况和审稿批准，再为这一草稿签发唯一外发权。"""
    if reason := manual_invite_block_reason(draft, game, source, at=at):
        raise ValueError(reason)
    if draft.metadata.get("content_review_approved") is not True:
        raise ValueError("invite draft has no backend-issued content review approval")
    claimed = replace(draft, metadata={**draft.metadata, _MANUAL_DELIVERY_CLAIM: trace_id}, updated_at=at)
    return claimed, StateTransition("invite_draft", draft.draft_id, draft.status.value, draft.status.value,
                                    "human_approved_invite_claimed", trace_id)


def finalize_manual_invite(
    draft: InviteDraft, *, trace_id: str, at: datetime, platform_message_id: str | None,
) -> tuple[InviteDraft, StateTransition]:
    """认领人用已核实回执结算，保留在途期间的真实回复；仅明确未发才释放认领。"""
    if not trace_id or draft.metadata.get(_MANUAL_DELIVERY_CLAIM) != trace_id:
        raise ValueError("invite_delivery_claim_changed")
    metadata = dict(draft.metadata)
    metadata.pop(_MANUAL_DELIVERY_CLAIM)
    if platform_message_id:
        metadata[_MANUAL_DELIVERY_RECEIPT] = platform_message_id
    status = InviteStatus.SENT if platform_message_id and draft.status == InviteStatus.PENDING_APPROVAL else draft.status
    reason = "human_approved_invite_sent" if platform_message_id else "human_approved_invite_not_sent"
    updated = replace(draft, metadata=metadata, status=status, updated_at=at)
    return updated, StateTransition("invite_draft", draft.draft_id, draft.status.value, status.value, reason, trace_id)


def has_manual_invite_claim(draft: InviteDraft) -> bool:
    """已取得外发权的草稿只能由该次可核对交付结算，不能走普通审批状态更新。"""
    return _MANUAL_DELIVERY_CLAIM in draft.metadata


def manual_invite_message_id(draft: InviteDraft) -> str | None:
    """只为经过平台回执结算的历史审批返回可复用的实际消息编号。"""
    value = draft.metadata.get(_MANUAL_DELIVERY_RECEIPT)
    return value if isinstance(value, str) and value else None


def tool_confirms_seat(call: ToolCall) -> bool:
    """只识别实际确认占席的工具；拒绝、取消和协商仍按各自合同处理。"""
    status = call.arguments.get("status")
    return call.name == "join_game" or (
        call.name == "record_candidate_reply" and isinstance(status, str) and status in CONFIRMED_CANDIDATE_STATUSES
    )


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
