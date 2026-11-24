"""验证查询权限回放保持读写边界，且明确委托只落到同一条件的新待组局。"""

from datetime import date
import json

import pytest

from mahjong_agent_runtime import SQLiteAgentStore
from scripts.replay_venue_consent import _seed_store
from scripts.replay_venue_lookup import (
    _CONTROL_MESSAGE_ID, _control_game_created, _control_terms_match, _execute_lookup,
    _load_lookup_archive, _load_lookup_fixture,
)
from scripts.run_venue_simulation import run_simulation

from .sim_adapter import build_runtime
from .venue_channel import SandboxChannel
from .venue_fixture import VenueFixtureAgent


@pytest.fixture(scope="module")
def lookup_archive(tmp_path_factory):
    """产生可引用的混合场景归档，无需旧设备中的数据或模型凭证。"""
    source_dir = tmp_path_factory.mktemp("lookup-source") / "run"
    report = run_simulation(source_dir, mode="mock", seed=42, scenario_date=date(2026, 11, 25), background_events=0,
                            scenario_name="mixed", active_users=3, workers=3, dialogue_turns=2)
    assert report["status"] == "passed", report
    seeker = next(customer for customer, result in report["cohort"]["expected_outcomes"].items() if result == "no_match")
    return source_dir, seeker


@pytest.mark.parametrize("use_portable", [True, False])
@pytest.mark.parametrize("create_too_early", [True, False])
def test_lookup_replay_only_creates_forming_game_after_explicit_delegation(tmp_path, lookup_archive,
                                                                       use_portable, create_too_early):
    """对两种输入核验最后一条只读消息不能写入，明确委托可正式建局。"""
    source = _load_lookup_fixture() if use_portable else _load_lookup_archive(*lookup_archive)
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    fixture = VenueFixtureAgent(seed=source.seed)
    runtime, _ = build_runtime(store, "mock", seed=source.seed, offline_client=fixture, local_coordination=True)
    channel = SandboxChannel(runtime, room_id="venue-group", identities=source.identities,
                             archive_path=tmp_path / "chat.sqlite3")
    game_id = _seed_store(store, channel, source)
    requirement = {
        "game_type": source.expected_game_type, "stake": "1", "start_time_kind": "scheduled",
        "start_time": "20:00", "duration_kind": "fixed", "duration_hours": 2,
        "planned_start_at": source.turns[-1].sent_at.replace(hour=20, minute=0).isoformat(),
    }
    for index, message in enumerate(source.turns):
        fixture.turns[message.message_id] = ("cohort_no_match", {"game_type": source.expected_game_type, "followup": index > 0})
    if create_too_early:
        fixture.turns[source.turns[-1].message_id] = ("need", {"requirement": requirement})
    fixture.turns[_CONTROL_MESSAGE_ID] = ("need", {"requirement": requirement})
    path = tmp_path / "events.jsonl"
    result = _execute_lookup(store, channel, source, game_id, path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["input"]["text"] for row in rows[:len(source.turns)]] == [message.text for message in source.turns]
    if create_too_early:
        assert result["status"] == "failed", result
        assert result["checks"][f"readonly_turn_{len(source.turns)}_no_unauthorized_writes"] is False
        assert result["checks"]["control_created_forming_game"] is None
        assert len(rows) == len(source.turns)
    else:
        assert result["status"] == "passed", result
        assert rows[-1]["phase"] == "authored_explicit_delegation"
        assert result["checks"]["control_created_forming_game"]
        assert result["checks"]["control_only_authorized_organizer_seat"]
        assert result["checks"]["control_no_other_game_room_or_waiting_changes"]


def test_replay_checks_the_persisted_two_hour_interval_when_duration_kind_is_missing():
    """真实创建可以省略时长标签，仍须落出完全符合委托的两小时区间。"""
    control = _load_lookup_fixture().turns[0]
    game_type = "national_standard_mahjong"
    requirement = {
        "game_type": game_type, "base_stake": 1, "duration_hours": 2,
        "planned_start_at": "2026-11-25T20:00:00+08:00", "planned_end_at": "2026-11-25T22:00:00+08:00",
    }
    assert _control_terms_match(requirement, control, game_type)
    assert not _control_terms_match({**requirement, "planned_end_at": "2026-11-25T23:00:00+08:00"}, control, game_type)
    assert not _control_terms_match({**requirement, "duration_kind": "flexible"}, control, game_type)
    assert not _control_terms_match({**requirement, "planned_end_at": None}, control, game_type)
    game = {"game_id": "created", "status": "inviting", "organizer_id": control.sender_id, "requirement": requirement}
    create = {"result": {"game": {"game_id": "created", "status": "forming"}}}
    assert _control_game_created(game, [create], control, game_type)
    assert not _control_game_created({**game, "status": "ready"}, [create], control, game_type)
