"""构建任务事实上下文，并执行作用域裁剪与隐私隔离。

"""

from __future__ import annotations

from typing import Any

from ...group_chat.parsing import parse_explicit_need
from ..temporal import parse_context_datetime


FACT_FIELD_ORDER = (
    "game_type",
    "game_variant",
    "stake",
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
    """返回基于本任务显式用户文本的、以最新为准的事实。

        该投影有意比语义理解更窄。它保留在工具调用之间不得漂移的稳定领域值，而意图、协商和下一步行动仍由模型决策。
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

    ordered = {field: values[field] for field in FACT_FIELD_ORDER if field in values}
    return {
        "facts": ordered,
        "binding_fields": list(ordered),
        "evidence": {field: evidence[field] for field in ordered if field in evidence},
        "contract": (
            "These values come from explicit user statements in the current task. Preserve every binding field "
            "in search/create/update tool arguments until a later explicit user statement changes it. "
            "Do not reinterpret one WeChat contact as one occupied seat."
        ),
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
