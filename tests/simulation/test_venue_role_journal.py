"""核对角色事件的顺序、恢复和并发写入，不访问模型或正式通信渠道。"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys
from threading import Barrier
from zoneinfo import ZoneInfo

import pytest

from .venue_role_journal import VenueRoleJournal


START = datetime(2026, 11, 18, 20, 14, 57, tzinfo=ZoneInfo("Asia/Shanghai"))


def test_duplicate_trigger_reuses_event_and_rejects_payload_collision(tmp_path):
    journal = VenueRoleJournal(tmp_path / "role.sqlite3")
    args = {"actor_id": "customer-a", "kind": "delivered", "origin_id": "group:m04", "caused_by": "group:m04"}
    first = journal.enqueue(**args, due_at=START, details={"weather": "rain", "can_arrive": False})
    repeated = journal.enqueue(**args, due_at=START.astimezone(timezone.utc), details={"can_arrive": False, "weather": "rain"})
    assert first == repeated
    assert first.to_dict()["due_at"] == "2026-11-18T12:14:57+00:00"
    with pytest.raises(ValueError, match="conflicts with an existing event"):
        journal.enqueue(**args, due_at=START + timedelta(seconds=1), details=first.details)
    with pytest.raises(ValueError, match="conflicts with an existing event"):
        journal.enqueue(**args, due_at=START, details={"can_arrive": True})
    with pytest.raises(ValueError, match="include a timezone"):
        journal.enqueue(**args, due_at=START.replace(tzinfo=None))
    assert journal.events() == (first,)


def test_wave_uses_earliest_time_and_claims_unrelated_actors_in_order(tmp_path):
    journal = VenueRoleJournal(tmp_path / "role.sqlite3")
    later = journal.enqueue(actor_id="a", kind="timer", origin_id="late", due_at=START + timedelta(minutes=1))
    first = journal.enqueue(actor_id="a", kind="timer", origin_id="initial", due_at=START)
    second = journal.enqueue(actor_id="a", kind="world", origin_id="weather", due_at=START)
    other = journal.enqueue(actor_id="b", kind="timer", origin_id="initial", due_at=START)
    with journal.run_owner():
        with pytest.raises(RuntimeError, match="already has an active scheduler"):
            with VenueRoleJournal(journal.path).run_owner():
                pytest.fail("a second scheduler must not acquire the same journal")
        first_claim, other_claim = journal.claim_wave(2)
        assert (first_claim.event_id, other_claim.event_id) == (first.event_id, other.event_id)
        assert journal.claim_wave(2) == ()
        with pytest.raises(ValueError, match="saved decision"):
            journal.finish(first_claim, {"status": "wait"})
        with pytest.raises(ValueError, match="before the virtual clock"):
            journal.enqueue(actor_id="c", kind="timer", origin_id="past", due_at=START - timedelta(seconds=1))
        journal.finish(journal.save_decision(first_claim, {"action": "wait"}), {"status": "wait"})
        second_claim, = journal.claim_wave(2)
        assert second_claim.event_id == second.event_id
        journal.finish(journal.save_decision(second_claim, {"action": "wait"}), {"status": "wait"})
        assert journal.claim_wave(2) == ()
        journal.finish(journal.save_decision(other_claim, {"action": "wait"}), {"status": "wait"})
        assert journal.claim_wave(2)[0].event_id == later.event_id
    assert journal.get(later.event_id).status == "pending"
    with pytest.raises(RuntimeError, match="requires the active scheduler"):
        journal.claim_wave(1)


def test_process_crash_preserves_decision_and_invalidates_old_claim(tmp_path):
    path = tmp_path / "role.sqlite3"
    journal = VenueRoleJournal(path)
    event = journal.enqueue(actor_id="a", kind="delivered", origin_id="dm:reply", due_at=START,
                            caused_by="dm:reply", details={"intention": "先向店家确认"})
    child = subprocess.run([
        sys.executable, "-c", """
import os
from pathlib import Path
import sys
from tests.simulation.venue_role_journal import VenueRoleJournal

journal = VenueRoleJournal(Path(sys.argv[1]))
with journal.run_owner():
    claim, = journal.claim_wave(1)
    journal.save_decision(claim, {"action": "send", "text": "还有最后一席吗？"})
    os._exit(23)
""", str(path),
    ], cwd=Path(__file__).resolve().parents[2], check=False, timeout=10)
    assert child.returncode == 23
    stranded = journal.get(event.event_id)
    assert stranded.status == "claimed"
    reopened = VenueRoleJournal(path)
    with reopened.run_owner():
        assert reopened.get(event.event_id).status == "pending"
        recovered, = reopened.claim_wave(1)
        assert recovered.claim_id != stranded.claim_id
        assert recovered.decision == {"action": "send", "text": "还有最后一席吗？"}
        assert (recovered.caused_by, recovered.details) == (event.caused_by, event.details)
        with pytest.raises(ValueError, match="stale role event claim"):
            reopened.finish(stranded, {"status": "delivered"})
        assert reopened.save_decision(recovered, {"text": "还有最后一席吗？", "action": "send"}) == recovered
        reopened.finish(recovered, {"status": "delivered", "message_id": "customer:" + recovered.event_id})
    persisted = VenueRoleJournal(path).get(event.event_id)
    assert persisted.status == "finished"
    assert persisted.result == {"status": "delivered", "message_id": "customer:" + recovered.event_id}
    assert persisted.decision == recovered.decision


def test_concurrent_retries_cannot_overwrite_decision_or_finish_twice(tmp_path):
    journal = VenueRoleJournal(tmp_path / "role.sqlite3")
    event = journal.enqueue(actor_id="a", kind="timer", origin_id="initial", due_at=START)
    start = Barrier(2)
    with journal.run_owner():
        claim, = journal.claim_wave(1)

        def attempt(text):
            start.wait(timeout=5)
            try:
                return journal.save_decision(claim, {"action": "send", "text": text}).decision
            except ValueError as exc:
                assert "conflicts with its saved decision" in str(exc)
                return None

        with ThreadPoolExecutor(max_workers=2) as workers:
            decisions = list(workers.map(attempt, ("先看看", "我报名")))
        saved, = (proposal for proposal in decisions if proposal is not None)
        assert journal.get(event.event_id).decision == saved
        journal.finish(claim, {"status": "delivered"})
        with pytest.raises(ValueError, match="stale role event claim"):
            journal.finish(claim, {"status": "wait"})


def test_500_roles_keep_separate_events_and_results_across_reopen(tmp_path):
    path = tmp_path / "role.sqlite3"
    journal = VenueRoleJournal(path)
    for number in range(500):
        actor_id = f"customer-{number:03d}"
        journal.enqueue(actor_id=actor_id, kind="world", origin_id="opening", due_at=START,
                        details={"free_minutes": number})
    with journal.run_owner():
        while wave := journal.claim_wave(32):
            assert len(wave) == len({item.actor_id for item in wave}) <= 32
            for item in wave:
                minutes = item.details["free_minutes"]
                journal.save_decision(item, {"action": "wait", "free_minutes": minutes})
                journal.finish(item, {"status": "wait"})
    persisted = VenueRoleJournal(path).events()
    assert len(persisted) == len({item.event_id for item in persisted}) == 500
    assert [item.decision["free_minutes"] for item in persisted] == list(range(500))
    assert all(item.status == "finished" and item.result == {"status": "wait"} for item in persisted)
