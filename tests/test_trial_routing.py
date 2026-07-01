from __future__ import annotations

from mahjong_agent.trial_routing import env_bool_value, use_controlled_trial_workflow


def test_env_bool_value_accepts_common_switch_values() -> None:
    assert env_bool_value(True, default=False) is True
    assert env_bool_value(False, default=True) is False
    assert env_bool_value("on", default=False) is True
    assert env_bool_value("0", default=True) is False
    assert env_bool_value("unknown", default=True) is True
    assert env_bool_value("unknown", default=False) is False


def test_trial_routing_defaults_to_controlled_and_locks_legacy_behind_env(monkeypatch) -> None:
    monkeypatch.delenv("MAHJONG_TRIAL_USE_CONTROLLED_WORKFLOW", raising=False)
    monkeypatch.delenv("MAHJONG_TRIAL_ALLOW_LEGACY_WORKFLOW", raising=False)
    assert use_controlled_trial_workflow({}) is True
    assert use_controlled_trial_workflow({"use_controlled_workflow": "false"}) is True

    monkeypatch.setenv("MAHJONG_TRIAL_USE_CONTROLLED_WORKFLOW", "0")
    assert use_controlled_trial_workflow({}) is True

    monkeypatch.setenv("MAHJONG_TRIAL_ALLOW_LEGACY_WORKFLOW", "1")
    assert use_controlled_trial_workflow({}) is False
    assert use_controlled_trial_workflow({"controlled_workflow": "false"}) is False
    assert use_controlled_trial_workflow({"controlled_workflow": "true"}) is True
