"""核验授权重放隔离两条原始入站，并且只在安全的模糊续轮后执行明确报名对照。"""

from dataclasses import replace
from datetime import date
import json

import pytest

from mahjong_agent_runtime import SQLiteAgentStore
from scripts.replay_venue_consent import _CONTROL_MESSAGE_ID, _execute_replay, _load_portable_fixture, _load_source, _seed_store
from scripts.run_venue_simulation import run_simulation

from .sim_adapter import build_runtime
from .venue_channel import SandboxChannel
from .venue_fixture import VenueFixtureAgent


@pytest.fixture(scope="module")
def formation_archive(tmp_path_factory):
    """由真实仿真接线生成临时归档，测试文件不包含既有现场的聊天内容。"""
    source_dir = tmp_path_factory.mktemp("formation-source") / "run"
    report = run_simulation(source_dir, mode="mock", seed=43, scenario_date=date(2026, 11, 25), background_events=0,
                            scenario_name="formation", active_users=1, workers=1, dialogue_turns=2)
    assert report["status"] == "passed", report
    return source_dir


@pytest.mark.parametrize("mutates_before_consent", [True, False])
@pytest.mark.parametrize("use_portable", [True, False])
def test_consent_replay_stops_when_second_turn_writes_and_tests_only_the_safe_path(
    tmp_path, formation_archive, mutates_before_consent, use_portable,
):
    """模糊问句不占席才执行明确报名对照，且对照须走真实工具写入。"""
    source = _load_portable_fixture() if use_portable else _load_source(formation_archive)
    opening, ambiguous = source.turns
    if not use_portable:
        ambiguous = replace(ambiguous, text="我得先问一下，桌子有空吗，等我确定再报名？")
        source = replace(source, turns=(opening, ambiguous))
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    fixture = VenueFixtureAgent(seed=source.seed)
    runtime, _ = build_runtime(store, "mock", seed=source.seed, offline_client=fixture, local_coordination=True)
    channel = SandboxChannel(runtime, room_id="venue-group", identities=source.identities,
                             archive_path=tmp_path / "chat.sqlite3")
    game_id = _seed_store(store, channel, source)
    fixture.turns[opening.message_id] = ("cohort_offer", {"game_id": game_id, "duration": 2})
    fixture.turns[ambiguous.message_id] = (("cohort_join" if mutates_before_consent else "dialogue_open"), {"game_id": game_id})
    fixture.turns[_CONTROL_MESSAGE_ID] = ("cohort_join", {"game_id": game_id})
    path = tmp_path / "events.jsonl"
    result = _execute_replay(store, channel, source, game_id, path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["phase"] for row in rows[:2]] == [
        f"{source.source_kind}_opening", f"{source.source_kind}_ambiguous_continuation",
    ]
    assert [row["input"]["text"] for row in rows[:2]] == [opening.text, ambiguous.text]
    assert [row["input"]["virtual_at"] for row in rows[:2]] == [opening.sent_at.isoformat(), ambiguous.sent_at.isoformat()]
    if mutates_before_consent:
        assert result["status"] == "failed"
        assert result["checks"]["ambiguous_no_participation_write"] is False
        assert result["checks"]["explicit_control_tool_confirmed"] is None
        assert len(rows) == 2
    else:
        assert result["status"] == "passed", result
        assert all(result["checks"].values())
        assert result["checks"]["explicit_control_no_unrequested_room_booking"] is True
        assert rows[2]["phase"] == "authored_explicit_control"
        assert rows[2]["input"]["message_id"] == _CONTROL_MESSAGE_ID
