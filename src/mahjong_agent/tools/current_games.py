from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..workflow_models import ConversationContext, GameRequirement


@dataclass(slots=True)
class CurrentGameSearchTool:
    max_results: int = 8

    def search(
        self,
        context: ConversationContext,
        requirement: GameRequirement,
    ) -> dict[str, Any]:
        matches: list[dict[str, Any]] = []
        for game in context.open_games:
            score, reasons = self._score(requirement, game)
            if score <= 0:
                continue
            matches.append(
                {
                    "score": score,
                    "reasons": reasons,
                    "game_requirement": game.to_prompt_dict(),
                    "summary": self._summary(game),
                }
            )
        matches.sort(key=lambda item: item["score"], reverse=True)
        return {
            "matches": matches[: self.max_results],
            "result_count": len(matches),
            "query": requirement.to_prompt_dict(),
        }

    def _score(self, requirement: GameRequirement, game: GameRequirement) -> tuple[int, list[str]]:
        score = 0
        reasons: list[str] = []
        for slot_name, weight in (("game_type", 35), ("stake", 30), ("smoke", 15)):
            requested = requirement.slot(slot_name)
            offered = game.slot(slot_name)
            if not requested or not requested.usable:
                continue
            if not offered or not offered.usable:
                continue
            if slot_name == "smoke" and requested.value == "any":
                score += weight
                reasons.append("烟况不限，可匹配")
                continue
            if requested.value == offered.value:
                score += weight
                reasons.append(f"{slot_name}匹配")
            else:
                return 0, []

        missing = game.slot("missing_count")
        if missing and missing.usable:
            try:
                if int(missing.value) <= 0:
                    return 0, []
                score += 20
                reasons.append(f"现有局还缺{missing.value}人")
            except (TypeError, ValueError):
                reasons.append("现有局缺口需确认")
        else:
            reasons.append("现有局缺口未知，需后续确认")

        start_mode = requirement.slot("start_time_mode")
        offered_start_mode = game.slot("start_time_mode")
        if start_mode and start_mode.usable and offered_start_mode and offered_start_mode.usable:
            if start_mode.value == offered_start_mode.value or start_mode.value == "people_ready":
                score += 10
                reasons.append("开局时间策略可匹配")
        return score, reasons

    def _summary(self, game: GameRequirement) -> str:
        slots = game.slots
        parts = [
            str(slots.get("game_type").value) if slots.get("game_type") else "麻将",
            str(slots.get("stake").value) if slots.get("stake") else "",
            str(slots.get("start_at").value) if slots.get("start_at") else str(slots.get("start_time_mode").value if slots.get("start_time_mode") else ""),
            f"缺{slots.get('missing_count').value}" if slots.get("missing_count") else "",
            str(slots.get("smoke").value) if slots.get("smoke") else "",
        ]
        return " ".join(part for part in parts if part).strip()
