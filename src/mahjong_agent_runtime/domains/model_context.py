"""封装模型上下文相关的领域规则和不变量。

"""

from __future__ import annotations

import math
from typing import Any
from ..models import (
    CustomerProfile,
    Game,
    InviteDraft,
    OutboundMessageDraft,
)
from .stake_values import parse_number, parse_stake_range
from .value_utils import first_present_value


_PUBLIC_LOOKUP_TERMS = (
    "requested_game", "game_type", "game_variant", "variant", "ruleset", "stake", "base_stake", "cap_score", "stake_label",
    "smoke_preference", "date", "start_time_kind", "start_time", "planned_start_at", "duration_kind", "duration_hours",
    "duration", "max_duration_hours", "known_player_count", "needed_seats", "seat_format", "recruitment_mode",
    "recruitment_status",
)
_PUBLIC_SEAT_COUNTS = ("seats_total", "claimed_seats", "remaining_seats")

def customer_visible_name(customers: dict[str, CustomerProfile], customer_id: str, fallback: str | None = None) -> str:
    profile = customers.get(str(customer_id or ""))
    if profile:
        return profile.visible_name()
    return str(fallback or customer_id or "")

def game_for_model_context(game: Game, customers: dict[str, CustomerProfile]) -> dict[str, Any]:
    """投影业务局况，并将实际到场名单和订房记录标识保留在运营审计中。"""
    payload = game.to_dict()
    payload.pop("actual_attendee_ids", None)
    payload.pop("play_reservation_id", None)
    payload["organizer_name"] = customer_visible_name(customers, game.organizer_id, game.organizer_name)
    _rewrite_contact_names(payload, customers)
    return payload


def has_agreed_game_duration(requirement: dict[str, Any]) -> bool:
    """只有明确时长能支持对外的计划结束时刻；最大的可打时长及生命周期默认值都不算。"""
    duration = parse_number(first_present_value(requirement, "duration_hours", "duration"))
    return duration is not None and math.isfinite(duration) and duration > 0


def public_game_lookup_projection(game: dict[str, Any], *, retain_operational_fields: bool = False) -> dict[str, Any]:
    """允许跨客户传递的桌况；身份、参与方和来源会话留在 Store 供授权工具查验。"""
    if not isinstance(game, dict):
        return {}
    terms = game.get("requirement")
    terms = terms if isinstance(terms, dict) else {}
    # 条件只接受已知标量及标准档位区间；自由摘要、备注和嵌套参与方不能穿过本边界。
    public = {key: terms[key] for key in _PUBLIC_LOOKUP_TERMS if type(terms.get(key)) in (str, int, float)}
    try:
        bounds = parse_stake_range(terms.get("stake_range"))
    except ValueError:
        bounds = None  # 历史工具里的不合格条件不能让整个对话投影失败或原样透传。
    if bounds:
        public["stake_range"] = {"min": bounds[0], "max": bounds[1]}
    if isinstance(stakes := terms.get("stakes"), list) and all(type(value) in (int, float) and math.isfinite(value) for value in stakes):
        public["stakes"] = list(stakes)
    result = {key: game[key] for key in ("game_id", "status", "planned_start_at", "recruitment_status") if key in game}
    result["requirement"] = public
    if isinstance(seats := game.get("seat_summary"), dict):
        result["seat_summary"] = {key: seats[key] for key in _PUBLIC_SEAT_COUNTS if type(seats.get(key)) is int}
    if type(game.get("remaining_seats")) is int:
        result["remaining_seats"] = game["remaining_seats"]
    # 工具原始回执用于比较版本及日期；模型只能用明确约定的时长解释结束时间。
    if "planned_end_at" in game and (retain_operational_fields or has_agreed_game_duration(public)):
        result["planned_end_at"] = game["planned_end_at"]
    if retain_operational_fields and "updated_at" in game:
        result["updated_at"] = game["updated_at"]
    return result


def game_requirement_for_conversation(requirement: dict[str, Any]) -> dict[str, Any]:
    """保留业务条件，以独立投影隔离房态、有效期和未约定的默认结束时刻。"""
    hidden = {"lifecycle_expires_at", "lifecycle_ttl_hours"}
    if not has_agreed_game_duration(requirement):
        hidden.add("planned_end_at")
    return {key: value for key, value in requirement.items() if key not in hidden}

def invite_draft_for_model_context(draft: InviteDraft, customers: dict[str, CustomerProfile]) -> dict[str, Any]:
    payload = draft.to_dict()
    payload["display_name"] = customer_visible_name(customers, draft.customer_id, draft.display_name)
    payload["metadata"] = visible_draft_metadata(payload.get("metadata"))
    return payload

def outbound_message_draft_for_model_context(
    draft: OutboundMessageDraft,
    customers: dict[str, CustomerProfile],
) -> dict[str, Any]:
    payload = draft.to_dict()
    payload["recipient_name"] = customer_visible_name(customers, draft.recipient_id, draft.recipient_name)
    payload["metadata"] = visible_draft_metadata(payload.get("metadata"))
    return payload

def visible_draft_metadata(metadata: Any) -> dict[str, Any]:
    metadata = dict(metadata or {}) if isinstance(metadata, dict) else {}
    return {
        key: value
        for key, value in metadata.items()
        if key in {"source", "game_id", "purpose", "channel"}
    }

def _rewrite_contact_names(payload: dict[str, Any], customers: dict[str, CustomerProfile]) -> None:
    for item in payload.get("participants") or []:
        if isinstance(item, dict):
            item["display_name"] = customer_visible_name(customers, str(item.get("customer_id") or ""), item.get("display_name"))
    for item in payload.get("parties") or []:
        if isinstance(item, dict):
            item["contact_name"] = customer_visible_name(customers, str(item.get("contact_id") or ""), item.get("contact_name"))
    for item in payload.get("seat_claims") or []:
        if isinstance(item, dict):
            item["contact_name"] = customer_visible_name(customers, str(item.get("contact_id") or ""), item.get("contact_name"))
    requirement = payload.get("requirement")
    if not isinstance(requirement, dict):
        return
    requester = requirement.get("requesting_party")
    if isinstance(requester, dict):
        requester["contact_name"] = customer_visible_name(
            customers,
            str(requester.get("contact_id") or requester.get("customer_id") or ""),
            requester.get("contact_name") or requester.get("display_name"),
        )
    for item in requirement.get("seat_claims") or []:
        if isinstance(item, dict):
            item["contact_name"] = customer_visible_name(
                customers,
                str(item.get("contact_id") or item.get("customer_id") or ""),
                item.get("contact_name") or item.get("display_name"),
            )
