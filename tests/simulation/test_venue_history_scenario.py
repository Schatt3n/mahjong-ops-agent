"""验证历史群聊进入可见收件箱，并沿正式入口产生新的因果往返。"""

from datetime import date
import json

from scripts.run_venue_simulation import run_simulation


def test_historical_group_runs_both_channels_without_leaking_future_or_private_replies(tmp_path):
    """每个新角色决定只看到已投递前缀，私聊回执不能成为别人的触发。"""
    run = tmp_path / "history"
    report = run_simulation(run, mode="mock", seed=42, scenario_date=date(2026, 11, 18),
                            background_events=0, scenario_name="historical-group", active_users=3,
                            workers=1, dialogue_turns=2)
    assert report["status"] == "passed"
    assert report["historical_continuation"]["sent_channels"] == {"private": 4, "group": 1}
    assert report["historical_continuation"]["causal_followups"] == 2
    chat = [json.loads(line) for line in (run / "chat.jsonl").read_text().splitlines()]
    history = [row for row in chat if row["direction"] == "history"]
    assert [(row["message_id"], row["quoted_message_id"]) for row in history] == [
        ("history:m01", None), ("history:m02", None), ("history:m03", "history:m01"), ("history:m04", "history:m03"),
    ]
    records = {row["message_id"]: row for row in chat}
    turns = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    for turn in turns:
        visible = turn["decision"]["context"]["history"]
        assert [row["message_id"] for row in visible if row["message_id"].startswith("history:")] == [
            row["message_id"] for row in history
        ]
        for row in visible:
            archived = records[row["message_id"]]
            if archived["kind"] == "private":
                assert turn["actor_id"] in {archived["sender_id"], archived["recipient_id"]}
        cause = records[turn["caused_by"]]
        assert turn["decision"]["context"]["delivery"]["message_id"] == cause["message_id"]
        if cause["kind"] == "private":
            assert turn["actor_id"] == cause["recipient_id"]
