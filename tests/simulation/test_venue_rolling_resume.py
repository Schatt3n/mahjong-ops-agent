"""重开整场持续角色仿真，核对已进渠道的消息只交付一次。"""

from datetime import date
import json

import pytest

from scripts.run_venue_simulation import resume_rolling_simulation, run_simulation

from .venue_role_journal import RoleSchedulerBusyError, VenueRoleJournal


def test_resume_reuses_original_actors_and_receipt_if_interrupted_after_sending(tmp_path, monkeypatch):
    """渠道已交付而角色未标为完成时，重开仍使用当初的决定与消息 ID。"""
    class Interruption(BaseException):
        pass

    run = tmp_path / "rolling"
    finish = VenueRoleJournal.finish
    interrupted = {}

    def stop_after_send(journal, event, result):
        if result.get("channel_trace_id"):
            interrupted.update(event_id=event.event_id, message_id=result["message_id"], decision=event.decision)
            raise Interruption
        return finish(journal, event, result)

    with monkeypatch.context() as patch:
        patch.setattr(VenueRoleJournal, "finish", stop_after_send)
        with pytest.raises(Interruption):
            run_simulation(run, mode="mock", seed=42, scenario_date=date(2026, 11, 25), background_events=0,
                           scenario_name="rolling-roles", active_users=4, workers=2, dialogue_turns=2)

    original_roster = (run / "personas.json").read_bytes()
    journal = VenueRoleJournal(run / "role_events.sqlite3")
    assert journal.get(interrupted["event_id"]).decision == interrupted["decision"]
    assert not (run / "report.json").exists()
    with journal.run_owner(), pytest.raises(RoleSchedulerBusyError):
        resume_rolling_simulation(run)
    backup = journal.path.with_suffix(".backup")
    journal.path.rename(backup)
    with pytest.raises(ValueError, match="角色事件账本"):
        resume_rolling_simulation(run)
    assert not journal.path.exists()
    journal.path.touch()
    with pytest.raises(ValueError, match="角色事件账本"):
        resume_rolling_simulation(run)
    assert journal.path.stat().st_size == 0
    backup.replace(journal.path)
    assert not (run / "report.json").exists()
    report = resume_rolling_simulation(run)
    assert report["status"] == "passed", report
    assert report["initial_contacts"]["initial_archive_messages"] == 0
    assert (run / "personas.json").read_bytes() == original_roster
    stored = journal.get(interrupted["event_id"])
    assert stored.result["delivery_deduplicated"] and stored.decision == interrupted["decision"]
    rows = [json.loads(row) for row in (run / "chat.jsonl").read_text().splitlines()]
    assert sum(row["message_id"] == interrupted["message_id"] for row in rows) == 1
