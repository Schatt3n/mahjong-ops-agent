from __future__ import annotations

import os
from typing import Any


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return env_bool_value(raw, default=default)


def env_bool_value(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def legacy_trial_workflow_allowed() -> bool:
    return env_bool("MAHJONG_TRIAL_ALLOW_LEGACY_WORKFLOW", False)


def use_controlled_trial_workflow(payload: dict[str, Any] | None = None) -> bool:
    payload = payload or {}
    explicit = (
        payload.get("use_controlled_workflow")
        if "use_controlled_workflow" in payload
        else payload.get("controlled_workflow")
    )
    if explicit is not None:
        requested_controlled = env_bool_value(explicit, default=True)
        if requested_controlled:
            return True
        return not legacy_trial_workflow_allowed()
    env_requested_controlled = env_bool("MAHJONG_TRIAL_USE_CONTROLLED_WORKFLOW", True)
    if env_requested_controlled:
        return True
    return not legacy_trial_workflow_allowed()
