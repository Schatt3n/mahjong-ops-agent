from __future__ import annotations

from typing import Any

from .workflow_models import ActionName


_NO_ARGUMENT_ACTIONS: frozenset[ActionName] = frozenset(
    {
        ActionName.SEARCH_EXISTING_GAMES,
        ActionName.ASK_CREATE_CONFIRMATION,
        ActionName.ASK_CLARIFICATION,
        ActionName.CREATE_GAME,
        ActionName.QUEUE_INVITES,
        ActionName.HUMAN_REVIEW,
        ActionName.IGNORE,
        ActionName.UNKNOWN,
    }
)
_REFERENCE_ACTION_ARGUMENTS: dict[ActionName, frozenset[str]] = {
    ActionName.MATCH_EXISTING_GAME: frozenset({"game_id"}),
    ActionName.JOIN_GAME: frozenset({"game_id", "outbox_id"}),
    ActionName.ACCEPT_SEAT: frozenset({"game_id", "outbox_id"}),
}
_CLOSE_ACTION_ARGUMENTS: frozenset[str] = frozenset({"game_id", "reason_code"})
_CLOSE_REASON_CODES: frozenset[str] = frozenset(
    {
        "user_cancelled",
        "organizer_cancelled",
        "candidate_cancelled",
        "game_full",
        "expired",
        "operator_cancelled",
    }
)


def validate_action_arguments_contract(action: ActionName, arguments: Any) -> list[str]:
    if arguments is None:
        return []
    if not isinstance(arguments, dict):
        return ["action_arguments must be an object when provided"]
    if action in _NO_ARGUMENT_ACTIONS:
        return _reject_unexpected_keys(action, arguments, allowed=frozenset())
    if action in _REFERENCE_ACTION_ARGUMENTS:
        return _validate_reference_arguments(action, arguments, allowed=_REFERENCE_ACTION_ARGUMENTS[action])
    if action in {ActionName.CANCEL_GAME, ActionName.CLOSE_GAME}:
        return _validate_close_arguments(action, arguments)
    return _reject_unexpected_keys(action, arguments, allowed=frozenset())


def _validate_reference_arguments(
    action: ActionName,
    arguments: dict[str, Any],
    *,
    allowed: frozenset[str],
) -> list[str]:
    errors = _reject_unexpected_keys(action, arguments, allowed=allowed)
    for key in allowed:
        if key in arguments and not _is_non_empty_string(arguments[key]):
            errors.append(f"action_arguments.{key} must be a non-empty string")
    return errors


def _validate_close_arguments(action: ActionName, arguments: dict[str, Any]) -> list[str]:
    errors = _reject_unexpected_keys(action, arguments, allowed=_CLOSE_ACTION_ARGUMENTS)
    if "game_id" in arguments and not _is_non_empty_string(arguments["game_id"]):
        errors.append("action_arguments.game_id must be a non-empty string")
    if "reason_code" in arguments:
        reason_code = str(arguments.get("reason_code") or "").strip()
        if reason_code not in _CLOSE_REASON_CODES:
            errors.append(f"action_arguments.reason_code invalid {reason_code!r}")
    return errors


def _reject_unexpected_keys(
    action: ActionName,
    arguments: dict[str, Any],
    *,
    allowed: frozenset[str],
) -> list[str]:
    errors: list[str] = []
    for key in sorted(arguments):
        if key not in allowed:
            errors.append(f"action_arguments.{key} is not allowed for {action.value}")
    return errors


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
