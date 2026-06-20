from __future__ import annotations

from .models import CandidateRecommendation, CustomerProfile, ExtractionResult, GameRequest, RoomAvailability


GAME_TYPE_LABELS = {
    "mahjong": "麻将",
    "sichuan_mahjong": "川麻",
    "chongqing_mahjong": "重庆麻将",
    "hangzhou_mahjong": "杭麻",
    "hongzhong_mahjong": "红中麻将",
    "zhuoji_mahjong": "捉鸡麻将",
    "hunan_mahjong": "湖南麻将",
}

VARIANT_LABELS = {
    "caiqiao": "财敲",
    "yaoji": "幺鸡",
    "suji": "素鸡",
    "yaoji_47": "幺鸡47",
    "shayu": "鲨鱼",
}

GAME_RULE_LABELS = {
    "sichuan_mahjong": "川麻",
    "chongqing_mahjong": "重庆麻将",
    "hangzhou_mahjong": "杭麻",
    "hongzhong_mahjong": "红中",
    "zhuoji_mahjong": "捉鸡",
    "hunan_mahjong": "湖南麻将",
}


class MessageComposer:
    def clarification(self, extraction: ExtractionResult) -> str:
        if not extraction.follow_up_questions:
            return "信息基本够了，我先帮你找人。"
        return "我确认一下：" + " ".join(extraction.follow_up_questions)

    def room_time_conflict(self, game: GameRequest, availability: RoomAvailability) -> str:
        requested = availability.requested_start_at.strftime("%H:%M")
        if availability.suggested_start_at:
            suggested = availability.suggested_start_at.strftime("%H:%M")
            return (
                f"{requested} 这个时间目前满房，最快 {suggested} 有房。"
                f"你看能不能改到 {suggested} 开？可以的话我按 {suggested} 继续帮你组；"
                f"如果还是想 {requested}，我先给你记候补，有房马上同步。"
            )
        return f"{requested} 这个时间目前满房，我这边暂时没看到可用房间。你看能不能换个时间，我再帮你组。"

    def group_post(self, game: GameRequest) -> str:
        parts = ["组局"]
        parts.extend(self._game_labels(game))
        if game.level:
            parts.append(self._stake_label(game))
        if game.start_at:
            parts.append(game.start_at.strftime("%m-%d %H:%M"))
        if game.missing_count:
            parts.append(f"缺{game.missing_count}位")
        play_options = self._visible_play_options(game)
        if play_options:
            parts.append("、".join(play_options))
        if game.duration_hours:
            parts.append(f"预计{self._format_hours(game.duration_hours)}小时")
        rules = self._visible_rules(game)
        if rules:
            parts.append("、".join(rules))
        return "，".join(parts) + "。方便的朋友私我。"

    def private_invite(self, game: GameRequest, customer: CustomerProfile | CandidateRecommendation) -> str:
        name = customer.display_name
        parts = []
        if game.start_at:
            when = f"今晚 {game.start_at.strftime('%H:%M')}" if game.start_at.hour >= 12 else game.start_at.strftime("%H:%M")
            parts.append(f"{when} 有一桌")
        else:
            parts.append("有一桌")
        parts.extend(self._game_labels(game))
        if game.level:
            parts.append(self._stake_label(game))
        if game.missing_count:
            parts.append(f"还差{game.missing_count}位")
        play_options = self._visible_play_options(game)
        if play_options:
            parts.append("、".join(play_options))
        rules = self._visible_rules(game)
        if rules:
            parts.append("、".join(rules))
        if game.duration_hours:
            parts.append(f"预计{self._format_hours(game.duration_hours)}小时")
        return f"{name}，" + "，".join(parts) + "，你方便来吗？"

    def already_full(self, game: GameRequest) -> str:
        when = game.start_at.strftime("%H:%M") if game.start_at else "这桌"
        return f"{when} 这桌刚刚已经组好了，我先给你记个候补，有合适的再问你。"

    def already_committed(self, game: GameRequest) -> str:
        return "你已经在另一桌有效局里了，我先不重复帮你安排。"

    def already_expired(self, game: GameRequest) -> str:
        when = game.start_at.strftime("%H:%M") if game.start_at else "这桌"
        return f"{when} 这桌已经过了，我先不再占这个位置。你要的话我重新帮你看新局。"

    def confirmed(self, game: GameRequest) -> str:
        when = game.start_at.strftime("%m-%d %H:%M") if game.start_at else "约好的时间"
        parts = [when]
        parts.extend(self._game_labels(game))
        parts.append(self._stake_label(game) if game.level else "这桌")
        return f"确认：{'，'.join(parts)}，人数已齐。"

    def candidate_summary(self, candidates: list[CandidateRecommendation], limit: int = 5) -> str:
        if not candidates:
            return "暂时没有高匹配候选人。"
        lines = []
        for item in candidates[:limit]:
            reasons = "；".join(item.reasons)
            warning = f" 注意：{'；'.join(item.warnings)}" if item.warnings else ""
            lines.append(f"{item.display_name}（{item.score}）：{reasons}{warning}")
        return "\n".join(lines)

    def _format_hours(self, hours: float) -> str:
        return str(int(hours)) if hours.is_integer() else str(hours)

    def _game_type_label(self, game: GameRequest) -> str | None:
        if game.game_type == "mahjong":
            return None
        return GAME_TYPE_LABELS.get(game.game_type, game.game_type)

    def _variant_label(self, game: GameRequest) -> str | None:
        if not game.variant:
            return None
        return VARIANT_LABELS.get(game.variant, game.variant)

    def _game_labels(self, game: GameRequest) -> list[str]:
        labels = []
        game_type = self._game_type_label(game)
        variant = self._variant_label(game)
        if game_type:
            labels.append(game_type)
        if variant and variant not in labels:
            labels.append(variant)
        return labels

    def _stake_label(self, game: GameRequest) -> str:
        if game.base_score is not None and game.cap_score is not None:
            return f"{game.level}档(底注{game.base_score:g}/封顶{game.cap_score:g})"
        return f"{game.level}档"

    def _visible_rules(self, game: GameRequest) -> list[str]:
        hidden = set(self._game_labels(game)) | set(game.play_options)
        if game.game_type in GAME_RULE_LABELS:
            hidden.add(GAME_RULE_LABELS[game.game_type])
        return [rule for rule in game.rules if rule not in hidden]

    def _visible_play_options(self, game: GameRequest) -> list[str]:
        variant = self._variant_label(game)
        if not variant:
            return game.play_options
        return [option for option in game.play_options if option != variant]
