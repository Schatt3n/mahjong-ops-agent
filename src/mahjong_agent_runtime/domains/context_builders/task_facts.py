"""提供词法候选供主模型解释语义角色，不把条件或引用自动绑定成当前事实。"""

from __future__ import annotations

from typing import Any

from ...group_chat.parsing import parse_explicit_need
from ..temporal import parse_context_datetime


FACT_FIELD_ORDER = (
    "game_type",
    "game_variant",
    "stake",
    "stake_range",
    "base_stake",
    "cap_score",
    "stake_label",
    "smoke_preference",
    "start_time_kind",
    "start_time",
    "planned_start_at",
    "duration_hours",
    "known_player_count",
    "needed_seats",
    "seat_format",
)


def project_explicit_task_facts(
    recent_conversation: list[dict[str, Any]],
    current_message: dict[str, Any],
    checkpoint: Any,
) -> dict[str, Any]:
    """返回基于用户原话的窄候选投影；值的语义角色仍未确认。

    ``parse_explicit_need`` 只抽取时间、玩法、大小、烟况等稳定字段；它不是完整意图
    解析，也不会写数据库。协商含义和下一步行动仍由主模型判断。
    """

    values: dict[str, Any] = {}
    evidence: dict[str, str] = {}
    checkpoint_facts = _checkpoint_facts(checkpoint)
    _merge_canonical(values, checkpoint_facts)
    for field in FACT_FIELD_ORDER:
        if field in values:
            evidence[field] = "conversation_checkpoint"

    user_turns = [
        (
            str(turn.get("content") or ""),
            parse_context_datetime(turn.get("occurred_at")),
        )
        for turn in recent_conversation
        if str(turn.get("role") or "") == "user" and str(turn.get("content") or "").strip()
    ]
    current_text = str(current_message.get("text") or "").strip()
    if current_text:
        user_turns.append(
            (
                current_text,
                parse_context_datetime(current_message.get("sent_at")),
            )
        )

    for index, (text, anchor) in enumerate(user_turns):
        parsed = parse_explicit_need(text, anchor=anchor)
        if "requested_game" in parsed:
            parsed["game_type"] = parsed.pop("requested_game")
        _merge_canonical(values, parsed)
        source = "current_message" if index == len(user_turns) - 1 and current_text else "recent_user_turn"
        for field in FACT_FIELD_ORDER:
            if field in parsed:
                evidence[field] = source

    # 词法命中不等于用户在陈述当前事实：条件、引用、否定和错字需要模型判定。
    # 这些候选值不直接绑定工具参数；主模型用 fact_assertions 声明语义角色后再校验。
    return {
        "facts": {}, "binding_fields": [],
        "candidates": {field: values[field] for field in FACT_FIELD_ORDER if field in values},
        "evidence": {field: evidence[field] for field in FACT_FIELD_ORDER if field in evidence},
        "contract": "Lexical candidates only, never authoritative current facts. Determine assertion/condition/quoted/uncertain from user context. Put current assertions in objective_state.fact_assertions with source_message_id; keep joining conditions separate from occupied seats.",
    }



def _checkpoint_facts(checkpoint: Any) -> dict[str, Any]:
    if checkpoint is None:
        return {}
    if isinstance(checkpoint, dict):
        payload = checkpoint.get("facts") if isinstance(checkpoint.get("facts"), dict) else checkpoint
        return dict(payload)
    facts = getattr(checkpoint, "facts", None)
    return dict(facts) if isinstance(facts, dict) else {}


def _merge_canonical(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    for field in FACT_FIELD_ORDER:
        value = incoming.get(field)
        if value is not None and value != "":
            target[field] = value


__all__ = ["FACT_FIELD_ORDER", "project_explicit_task_facts"]
