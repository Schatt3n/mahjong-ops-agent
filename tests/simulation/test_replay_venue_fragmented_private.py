"""验收分开的真实入站、到期派发及结合全部聊天条件的正式工具查询。"""

import json

import pytest

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.llm import StaticAgentClient

from scripts import agent_runtime_app as app
from scripts.replay_venue_fragmented_private import _CHECKS, _execute, _seed_tables, _source
from .sim_adapter import build_runtime
from .venue_actors import SemanticJudge
from .venue_channel import SandboxChannel
from .venue_fixture import _call, _reply


@pytest.mark.parametrize(("single", "quiet_due", "early", "keep_smoke", "expected"), [
    (False, False, False, True, "passed"),
    (False, True, False, True, "passed"),
    (True, False, False, True, "passed"),
    (False, False, True, True, "passed"),
    (False, False, False, False, "failed"),
])
def test_same_private_request_replays_one_or_four_bubbles_through_production_aggregation(
    tmp_path, monkeypatch, single, quiet_due, early, keep_smoke, expected,
):
    source = _source(single_message=single)
    batches = []
    window_size = 2 if early else len(source.turns)

    def input_gate(message, *, input_batch, quiet_period_elapsed, **_kwargs):
        ids = message.metadata["input_window"]["source_message_ids"]
        batches.append((ids, quiet_period_elapsed))
        ready = len(ids) == window_size and (quiet_period_elapsed or not quiet_due)
        return {"action": "process_business" if ready else "wait_for_more_input", "should_route": ready, "errors": []}

    monkeypatch.setattr(app, "run_wechaty_input_gate", input_gate)
    requirement = {"start_time_range": {"start_at": "2026-11-26T15:00:00+08:00", "end_at": "2026-11-26T17:00:00+08:00"}}
    if keep_smoke:
        requirement["smoke_preference"] = "no_smoking"
    client = StaticAgentClient([
        *([_reply("好，下午我看看。 ")] if early else []),
        _call("search_current_games", {"requirement": requirement, "include_nearby": False}),
        _reply("下午四点有桌无烟杭麻，一块底，还差三个人。"),
    ])
    store = SQLiteAgentStore(tmp_path / "business.sqlite3")
    runtime, _ = build_runtime(store, "mock", offline_client=client)
    channel = SandboxChannel(runtime, room_id="fragmented-private", identities=source.identities,
                             archive_path=tmp_path / "chat.sqlite3", production_private_routing=True,
                             production_private_aggregation=True)
    game = _seed_tables(store, channel, source)
    findings = [{"check": check, "verdict": "pass", "reason": "测试独立评审的证据协议",
                 "evidence_ids": ["bubble-1", "actual-read", "actual-reply"]} for check in _CHECKS]
    judge = SemanticJudge(StaticAgentClient([json.dumps({"verdict": "pass", "findings": findings}, ensure_ascii=False)]))

    report = _execute(store, channel, source, game, tmp_path / "events.jsonl", judge=judge)

    ids = [turn.message_id for turn in source.turns]
    assert report["status"] == expected, report
    windows = [ids[:2], ids[2:]] if early else [ids]
    assert report["input_window_message_ids"] == windows
    expected_batches = [(window[:index], False) for window in windows for index in range(1, len(window) + 1)]
    assert batches == expected_batches + ([(ids, True)] if quiet_due else [])
    assert report["checks"]["final_query_keeps_both_late_constraints"] is keep_smoke
    assert report["checks"]["read_only_at_every_step"] is True
    assert report["fixture_source"]["interval_seconds"] == (None if single else 5)
    assert report["fixture_source"]["source_turns"] == [1, 3]
    store._connection.close()
