"""从隔离 SQLite 核对仿真画像证据，并防止候选误计为正式画像。"""

from datetime import date, datetime, timedelta
import json

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.models import CustomerProfile, DEFAULT_TZ, using_clock
from scripts.run_venue_simulation import _profile_evidence, _profile_snapshot, run_simulation


def test_venue_profile_evidence_counts_only_real_profile_field_changes(tmp_path):
    """身份更新、当次记忆、待审候选和正式偏好分别统计，汇总不包含原值。"""
    path = tmp_path / "profile.sqlite3"
    store = SQLiteAgentStore(path)
    store.upsert_customer(CustomerProfile("profile-member", "原昵称"))
    store.upsert_customer(CustomerProfile("nickname-member", "原昵称"))
    before = _profile_snapshot(store)
    store.upsert_customer(CustomerProfile("nickname-member", "换了昵称"))
    store.upsert_customer(CustomerProfile("profile-member", "原昵称", preferred_games=["hangzhou_mahjong"],
                                          smoke_preference="no_smoking"))
    memory = dict(conversation_id="profile-chat", customer_id="nickname-member", memory_type="preference",
                  field="smoke_preference", value="no_smoking", evidence="只打无烟的", trace_id="profile-test")
    store.record_task_memory(**memory)
    store.record_pending_memory_candidate(**memory)

    evidence = _profile_evidence(SQLiteAgentStore(path), before, {"memory_from_visible_message": True})

    assert evidence["formal_profiles"] == {
        "customers_at_start": 2,
        "customers_at_end": 2,
        "customers_with_changed_fields": 1,
        "customers_with_non_default_fields": 1,
        "changed_fields": {"preferred_games": 1, "smoke_preference": 1},
    }
    assert evidence["current_task_memory_records_by_status"] == {"active": 1}
    assert evidence["long_term_candidate_records_by_status"] == {"pending_review": 1}
    assert evidence["attendance_defaults"] == {
        "customers_with_completed_play": 0,
        "customers_with_consistent_completed_play": 0,
        "customers_eligible_for_attendance_defaults": 0,
    }
    assert evidence["visible_source_pending_candidate_check"] is True
    summary = json.dumps(evidence, ensure_ascii=False)
    for hidden in ("profile-member", "nickname-member", "换了昵称", "只打无烟的", "hangzhou_mahjong", "no_smoking"):
        assert hidden not in summary


def test_venue_report_keeps_successful_visible_candidate_distinct_from_profile(tmp_path):
    """仿真原有可见来话证据门禁通过时，报告仍可以显示零正式画像变动。"""
    report = run_simulation(tmp_path / "candidate-only", mode="mock", seed=42,
                            scenario_date=date(2026, 11, 25), background_events=0)

    assert report["status"] == "passed"
    assert report["checks"]["memory_from_visible_message"] is True
    evidence = report["profile_evidence"]
    assert evidence["visible_source_pending_candidate_check"] is True
    assert evidence["long_term_candidate_records_by_status"] == {"pending_review": 1}
    assert evidence["formal_profiles"] == {
        "customers_at_start": 500,
        "customers_at_end": 500,
        "customers_with_changed_fields": 0,
        "customers_with_non_default_fields": 0,
        "changed_fields": {},
    }
    assert evidence["attendance_defaults"]["customers_with_consistent_completed_play"] == 0


def test_venue_counts_attendance_defaults_separately_from_formal_profile_changes(tmp_path):
    """三次真实完赛能留下推导默认值，冲突的正式画像仍优先。"""
    store = SQLiteAgentStore(tmp_path / "played.sqlite3")
    store.configure_rooms(["east"])
    attendees = ("host", "another-host-prefers-sichuan", "north", "south")
    for customer in attendees:
        store.upsert_customer(CustomerProfile(customer, "同名", preferred_games=(
            ["sichuan_mahjong"] if customer == attendees[1] else []
        )))
    store.upsert_customer(CustomerProfile("unplayed", "同名"))
    before = _profile_snapshot(store)
    day_one = datetime(2026, 11, 21, 19, tzinfo=DEFAULT_TZ)
    for day in range(3):
        start = day_one + timedelta(days=day)
        with using_clock(lambda: start - timedelta(hours=1)):
            game, _ = store.create_game(conversation_id=f"daily:{day}", organizer_id=attendees[0], organizer_name="同名",
                requirement={"game_type": "hangzhou_mahjong", "stake": "0.5", "smoke_preference": "no_smoking",
                             "start_time_kind": "scheduled", "planned_start_at": start.isoformat(), "duration_hours": 2},
                known_players=[], trace_id=f"create:{day}")
            for customer in attendees[1:]:
                store.join_game(game_id=game.game_id, customer_id=customer, display_name="同名", trace_id=f"join:{day}")
        store.start_game_play(game.game_id, attendee_ids=attendees, room_id="east", at=start, trace_id=f"play:{day}")
        store.finish_game_play(game.game_id, at=start + timedelta(hours=2), trace_id=f"finish:{day}")

    evidence = _profile_evidence(SQLiteAgentStore(store.path), before, {})
    assert evidence["attendance_defaults"] == {
        "customers_with_completed_play": 4,
        "customers_with_consistent_completed_play": 4,
        "customers_eligible_for_attendance_defaults": 3,
    }
    assert evidence["formal_profiles"]["customers_with_changed_fields"] == 0
    assert evidence["formal_profiles"]["customers_with_non_default_fields"] == 1
    assert evidence["long_term_candidate_records_by_status"] == {}
    assert "another-host-prefers-sichuan" not in json.dumps(evidence)
