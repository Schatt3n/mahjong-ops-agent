from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..core import AgentCore
from ..models import DEFAULT_TZ, GameRequest
from ..workflow_models import GameRequirement


@dataclass(slots=True)
class CandidateSearchTool:
    core: AgentCore
    max_results: int = 8

    def search(
        self,
        requirement: GameRequirement,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        effective_now = now or datetime.now(DEFAULT_TZ)
        legacy_game = self._to_legacy_game_request(requirement)
        available_customers = [
            customer
            for customer in self.core.store.customers.values()
            if not customer.no_contact
            and self.core.customer_active_lock(customer.id, exclude_game_id=legacy_game.id) is None
        ]
        fatigue_by_customer = {
            customer.id: self.core.customer_fatigue(
                customer.id,
                proposed_start_at=legacy_game.start_at or effective_now,
                now=effective_now,
                exclude_game_id=legacy_game.id,
            )
            for customer in available_customers
        }
        recommendations = self.core.matcher.recommend_customers(
            legacy_game,
            available_customers,
            now=effective_now,
            fatigue_by_customer=fatigue_by_customer,
        )[: self.max_results]
        return {
            "candidates": [
                {
                    "customer_id": item.customer_id,
                    "display_name": item.display_name,
                    "score": item.score,
                    "reasons": list(item.reasons),
                    "warnings": list(item.warnings),
                }
                for item in recommendations
            ],
            "result_count": len(recommendations),
            "query": requirement.to_prompt_dict(),
        }

    def _to_legacy_game_request(self, requirement: GameRequirement) -> GameRequest:
        slots = requirement.slots
        game = GameRequest(
            organizer_id=requirement.organizer_id or "workflow",
            organizer_name=requirement.organizer_name or "workflow",
            channel_id="workflow_context",
            seats_total=requirement.seats_total,
            game_type=str(slots["game_type"].value) if slots.get("game_type") else "hangzhou_mahjong",
            ruleset=str(slots["ruleset"].value) if slots.get("ruleset") else None,
            variant=str(slots["variant"].value) if slots.get("variant") else None,
            level=str(slots["stake"].value) if slots.get("stake") else None,
        )
        if slots.get("base_score"):
            game.base_score = _safe_float(slots["base_score"].value)
        if slots.get("cap_score"):
            game.cap_score = _safe_float(slots["cap_score"].value)
        if slots.get("current_player_count"):
            game.current_player_count = _safe_int(slots["current_player_count"].value)
        if slots.get("missing_count"):
            game.missing_count = _safe_int(slots["missing_count"].value)
        if slots.get("duration_hours"):
            game.duration_hours = _safe_float(slots["duration_hours"].value)
        if slots.get("start_at"):
            parsed_start = _safe_datetime(slots["start_at"].value)
            if parsed_start:
                game.start_at = parsed_start
        smoke = str(slots["smoke"].value) if slots.get("smoke") else ""
        if smoke == "no_smoke":
            game.rules.append("无烟")
        elif smoke == "smoke_ok":
            game.rules.append("可吸烟")
        elif smoke == "any":
            game.rules.append("烟况都可")
        if slots.get("play_options") and isinstance(slots["play_options"].value, list):
            game.play_options.extend(str(item) for item in slots["play_options"].value)
        if slots.get("rules") and isinstance(slots["rules"].value, list):
            game.rules.extend(str(item) for item in slots["rules"].value)
        return game


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=DEFAULT_TZ)
