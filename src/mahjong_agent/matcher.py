from __future__ import annotations

from datetime import datetime

from .models import (
    CandidateRecommendation,
    CustomerFatigue,
    CustomerProfile,
    GameRequest,
    GameStatus,
    MergeSuggestion,
    PlayPreference,
)


class MatchingEngine:
    def recommend_customers(
        self,
        game: GameRequest,
        customers: list[CustomerProfile],
        now: datetime | None = None,
        limit: int = 20,
        fatigue_by_customer: dict[str, CustomerFatigue] | None = None,
    ) -> list[CandidateRecommendation]:
        recommendations: list[CandidateRecommendation] = []
        fatigue_by_customer = fatigue_by_customer or {}
        for customer in customers:
            if customer.no_contact:
                continue
            if customer.id == game.organizer_id:
                continue
            if customer.id in game.participant_ids or customer.id in game.reserved_customer_ids:
                continue

            score = 20.0
            reasons: list[str] = []
            warnings: list[str] = []

            fatigue = fatigue_by_customer.get(customer.id)
            if fatigue and fatigue.hard_block:
                continue

            play_preference = self._matching_play_preference(game, customer.play_preferences)
            if customer.play_preferences:
                if play_preference:
                    play_score, play_reasons, play_warnings = self._score_play_preference(game, play_preference)
                    score += play_score
                    reasons.extend(play_reasons)
                    warnings.extend(play_warnings)
                elif game.game_type != "mahjong":
                    score -= 12
                    warnings.append("画像里没有这个玩法偏好")

            preferred_levels = (
                play_preference.preferred_levels
                if play_preference and play_preference.preferred_levels
                else customer.preferred_levels
            )
            level_score, level_reason = self._score_level(game.level, preferred_levels)
            score += level_score
            if level_reason:
                reasons.append(level_reason)

            if "无烟" in game.rules:
                if customer.smoke_free_preference is True:
                    score += 18
                    reasons.append("偏好无烟局")
                elif customer.smoke_free_preference is False:
                    score -= 28
                    warnings.append("画像显示可能不偏好无烟局")

            matching_tags = sorted(set(game.rules) & set(customer.tags))
            if matching_tags:
                score += min(20, 8 * len(matching_tags))
                reasons.append(f"标签匹配：{', '.join(matching_tags)}")

            if game.start_at:
                if game.start_at.hour in customer.usual_start_hours:
                    score += 12
                    reasons.append(f"常在 {game.start_at.hour}:00 左右开打")
                if game.start_at.weekday() in customer.usual_weekdays:
                    score += 6
                    reasons.append("常在这个星期时段活跃")

            if customer.decline_count_30d >= 3:
                score -= 10
                warnings.append("近 30 天拒绝较多，注意打扰频率")

            if customer.last_invited_at and now:
                hours_since_invite = (now - customer.last_invited_at).total_seconds() / 3600
                if hours_since_invite < customer.invite_cooldown_hours:
                    score -= 25
                    warnings.append(f"{customer.invite_cooldown_hours:g} 小时内已经邀请过")

            if fatigue:
                score += fatigue.score_adjustment
                reasons.extend(fatigue.reasons)
                warnings.extend(fatigue.warnings)

            if score >= 25:
                if not reasons:
                    reasons.append("基础条件可尝试")
                recommendations.append(
                    CandidateRecommendation(
                        customer_id=customer.id,
                        display_name=customer.display_name,
                        score=round(score, 1),
                        reasons=reasons,
                        warnings=warnings,
                    )
                )

        return sorted(recommendations, key=lambda item: item.score, reverse=True)[:limit]

    def suggest_merges(
        self,
        games: list[GameRequest],
        time_tolerance_minutes: int = 45,
    ) -> list[MergeSuggestion]:
        suggestions: list[MergeSuggestion] = []
        open_games = [
            game
            for game in games
            if game.status in {GameStatus.OPEN, GameStatus.NEED_CLARIFICATION, GameStatus.NEGOTIATING}
        ]

        for index, left in enumerate(open_games):
            for right in open_games[index + 1 :]:
                suggestion = self._score_merge(left, right, time_tolerance_minutes)
                if suggestion and suggestion.score >= 45:
                    suggestions.append(suggestion)

        return sorted(suggestions, key=lambda item: item.score, reverse=True)

    def _score_merge(
        self,
        left: GameRequest,
        right: GameRequest,
        time_tolerance_minutes: int,
    ) -> MergeSuggestion | None:
        conflicts: list[str] = []
        reasons: list[str] = []
        score = 100.0

        if left.level and right.level:
            level_delta = self._level_delta(left.level, right.level)
            if level_delta is None:
                score -= 30
                conflicts.append("档位无法比较")
            elif level_delta == 0:
                reasons.append(f"同为 {left.level} 档")
            elif level_delta <= 0.5:
                score -= 15
                reasons.append("档位接近，可协商")
            else:
                score -= 45
                conflicts.append("档位差距较大")
        elif left.level or right.level:
            score -= 12
            conflicts.append("其中一桌档位不明确")

        proposed_start_at = None
        if left.start_at and right.start_at:
            delta_minutes = abs((left.start_at - right.start_at).total_seconds()) / 60
            if delta_minutes <= time_tolerance_minutes:
                proposed_start_at = min(left.start_at, right.start_at) + (abs(left.start_at - right.start_at) / 2)
                score -= delta_minutes * 0.55
                reasons.append(f"开局时间相差 {int(delta_minutes)} 分钟")
            else:
                score -= 55
                conflicts.append(f"开局时间相差 {int(delta_minutes)} 分钟")
        else:
            score -= 20
            conflicts.append("其中一桌时间不明确")

        left_players = left.current_player_count
        right_players = right.current_player_count
        if left_players is not None and right_players is not None:
            total_players = left_players + right_players
            if total_players == left.seats_total:
                reasons.append("两边人数刚好拼成一桌")
                score += 10
            elif total_players < left.seats_total:
                score -= 10
                reasons.append(f"合并后仍缺 {left.seats_total - total_players} 人")
            else:
                score -= 50
                conflicts.append("合并后人数超过一桌")
        else:
            score -= 15
            conflicts.append("其中一桌人数不明确")

        if left.duration_hours and right.duration_hours:
            duration_delta = abs(left.duration_hours - right.duration_hours)
            if duration_delta <= 1:
                reasons.append("时长接近")
            else:
                score -= 12
                conflicts.append("预期时长差距较大")

        common_rules = sorted(set(left.rules) & set(right.rules))
        if common_rules:
            score += min(10, len(common_rules) * 5)
            reasons.append(f"规则偏好一致：{', '.join(common_rules)}")

        return MergeSuggestion(
            game_ids=[left.id, right.id],
            score=round(max(0.0, score), 1),
            proposed_start_at=proposed_start_at,
            reasons=reasons,
            conflicts=conflicts,
        )

    def _score_level(self, game_level: str | None, preferred_levels: list[str]) -> tuple[float, str | None]:
        if not game_level or not preferred_levels:
            return 0, None
        if game_level in preferred_levels:
            return 35, f"常打 {game_level} 档"
        game_value = self._parse_level(game_level)
        game_bounds = self._parse_level_bounds(game_level)
        best_delta: float | None = None
        best_level: str | None = None
        for preferred in preferred_levels:
            if preferred == game_level:
                return 35, f"常打 {preferred} 档"
            preferred_value = self._parse_level(preferred)
            preferred_bounds = self._parse_level_bounds(preferred)
            if game_bounds and preferred_bounds:
                base_delta = abs(game_bounds[0] - preferred_bounds[0])
                cap_delta = abs((game_bounds[1] or 0) - (preferred_bounds[1] or 0))
                delta = base_delta + cap_delta / 32
            elif preferred_value is not None and game_value is not None:
                delta = abs(game_value - preferred_value)
            else:
                continue
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_level = preferred
        if best_delta is None:
            return 0, None
        if best_delta == 0:
            return 35, f"常打 {best_level} 档"
        if best_delta <= 0.5:
            return 16, f"常打 {best_level} 档，和本局接近"
        return -18, f"常打 {best_level} 档，和本局差距较大"

    def _level_delta(self, left: str, right: str) -> float | None:
        left_bounds = self._parse_level_bounds(left)
        right_bounds = self._parse_level_bounds(right)
        if left_bounds and right_bounds:
            base_delta = abs(left_bounds[0] - right_bounds[0])
            cap_delta = abs((left_bounds[1] or 0) - (right_bounds[1] or 0))
            return base_delta + cap_delta / 32
        left_value = self._parse_level(left)
        right_value = self._parse_level(right)
        if left_value is None or right_value is None:
            return None
        return abs(left_value - right_value)

    def _parse_level(self, value: str | None) -> float | None:
        bounds = self._parse_level_bounds(value)
        if bounds:
            return bounds[0]
        if value is None:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def _parse_level_bounds(self, value: str | None) -> tuple[float, float | None] | None:
        if not value:
            return None
        if "-" in value:
            left, _, right = value.partition("-")
            try:
                return float(left), float(right)
            except ValueError:
                return None
        try:
            return float(value), None
        except ValueError:
            return None

    def _matching_play_preference(
        self,
        game: GameRequest,
        preferences: list[PlayPreference],
    ) -> PlayPreference | None:
        best: tuple[int, PlayPreference] | None = None
        for preference in preferences:
            if preference.game_type not in {game.game_type, game.ruleset}:
                continue
            score = 10
            if game.ruleset and game.ruleset in preference.preferred_rulesets:
                score += 6
            if game.variant and game.variant in preference.preferred_variants:
                score += 8
            score += 2 * len(set(game.play_options) & set(preference.preferred_play_options))
            if best is None or score > best[0]:
                best = (score, preference)
        return best[1] if best else None

    def _score_play_preference(
        self,
        game: GameRequest,
        preference: PlayPreference,
    ) -> tuple[float, list[str], list[str]]:
        score = 18.0
        reasons = [f"偏好玩法：{preference.game_type}"]
        warnings: list[str] = []

        if game.ruleset and game.ruleset in preference.preferred_rulesets:
            score += 8
            reasons.append(f"常打规则：{game.ruleset}")
        if game.variant and game.variant in preference.preferred_variants:
            score += 12
            reasons.append(f"常打细分：{game.variant}")

        matching_options = sorted(set(game.play_options) & set(preference.preferred_play_options))
        if matching_options:
            score += min(16, 6 * len(matching_options))
            reasons.append(f"玩法选项匹配：{', '.join(matching_options)}")

        avoided_options = sorted(set(game.play_options) & set(preference.avoid_play_options))
        if avoided_options:
            score -= 30
            warnings.append(f"画像不偏好：{', '.join(avoided_options)}")

        return score, reasons, warnings
