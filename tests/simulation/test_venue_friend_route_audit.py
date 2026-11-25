"""从持久账本捕获好友回执错投和跨账号续聊。"""

from contextlib import closing
from datetime import date
import json
import sqlite3

from scripts.run_venue_simulation import run_simulation
from scripts.summarize_venue_run import audit_owner_friend_routes


def test_friend_route_audit_rejects_other_friends_cause_and_receipt(tmp_path):
    """归档计数相同也不能把另一位好友的回执当作本人续聊。"""
    run_dir = tmp_path / "routes"
    report = run_simulation(run_dir, mode="mock", seed=42, scenario_date=date(2026, 11, 25),
                            background_events=0, scenario_name="dialogues", active_users=2, dialogue_turns=2)
    archive = run_dir / "chat.sqlite3"
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    audit = audit_owner_friend_routes(report, archive, events)
    assert audit["status"] == "passed"
    assert (audit["audited_private_routes"], audit["matched_private_replies"], audit["causal_two_round_customers"]) == (2, 4, 2)

    first, other, second, _ = events
    wrong_cause = {**second, "caused_by": other["reply_message_id"]}
    audit = audit_owner_friend_routes(report, archive, [first, other, wrong_cause, events[-1]])
    assert audit["status"] == "failed"
    assert audit["checks"]["all_replies_returned_to_originating_friend"]
    assert not audit["checks"]["all_round_two_events_followed_own_first_reply"]

    with closing(sqlite3.connect(archive)) as db, db:
        db.execute("UPDATE venue_messages SET recipient_id = ? WHERE message_id = ?",
                   (other["actor_id"], first["reply_message_id"]))
    audit = audit_owner_friend_routes(report, archive, events)
    assert audit["status"] == "failed"
    assert not audit["checks"]["all_replies_returned_to_originating_friend"]
    assert audit["matched_private_replies"] == 3
