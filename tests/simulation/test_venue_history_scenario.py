"""验证历史群聊进入可见收件箱，并沿正式入口产生新的因果往返。"""

from datetime import date
import json

import pytest

from scripts.compare_venue_history_variants import compare_runs
from scripts.run_venue_simulation import run_simulation
from mahjong_agent_runtime import JsonlTraceRecorder


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


def test_historical_private_hydrates_exactly_one_real_conversation(tmp_path):
    """真人店员的旧答复在双方可见档案内，并进入新一轮店家提示。"""
    run = tmp_path / "history-private"
    report = run_simulation(run, mode="mock", seed=42, scenario_date=date(2026, 7, 5),
        background_events=0, scenario_name="historical-private", active_users=1, workers=1, dialogue_turns=2)
    assert report["status"] == "passed"
    assert report["checks"]["historical_private_turns_delivered_and_persisted"]
    chat = [json.loads(line) for line in (run / "chat.jsonl").read_text().splitlines()]
    history = [item for item in chat if item["direction"] == "history"]
    assert len(history) == 2 and history[-1]["sender_id"] == "simulation-owner"
    assert history[0]["conversation_id"] == history[-1]["conversation_id"]
    first, *_ = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    assert first["caused_by"] == history[-1]["message_id"]
    assert [row["message_id"] for row in first["decision"]["context"]["history"]] == [
        row["message_id"] for row in history
    ]
    prompts = [entry.content["messages"] for entry in JsonlTraceRecorder(run / "agent_trace.log").get_trace(first["trace_id"])
               if entry.step == "llm_prompt"]
    assert prompts and all(row["text"] in json.dumps(prompts[0], ensure_ascii=False) for row in history)


def test_private_holdout_source_does_not_supply_role_tendencies(tmp_path):
    """独立老板 B 私聊供验收，合成角色线索仍来自原审核素材。"""
    run = tmp_path / "private-holdout"
    report = run_simulation(run, mode="mock", seed=42, scenario_date=date(2026, 7, 12), background_events=0,
        scenario_name="historical-private", private_history_episode="holdout-friend", role_source="history",
        active_users=1, workers=1, dialogue_turns=2)
    assert report["status"] == "passed"
    assert report["historical_continuation"]["episode"] == "friend_as_separate_participant"
    assert report["historical_continuation"]["held_out_messages"] == 3
    attribution = json.loads((run / "role_traits.json").read_text())
    assert len(attribution) == 500
    assert report["historical_continuation"]["case"] not in {row["source"]["case_id"] for row in attribution}


@pytest.mark.parametrize(("pair", "field"), [
    (("smoke-pending", "smoke-confirmed"), "text"),
    (("arrival-2100", "arrival-2140"), "earliest_arrival"),
    (("question-unquoted", "question-quoted"), "quoted_message_id"),
])
def test_paired_variants_change_one_dimension_in_the_actual_actor_context(tmp_path, pair, field):
    """核对每对新分支实际进入角色输入的差异，原来话和附加场景来源仍可分辨。"""
    contexts, values = [], []
    for variant in pair:
        run = tmp_path / variant
        report = run_simulation(run, mode="mock", seed=42, scenario_date=date(2026, 11, 18),
            background_events=0, scenario_name="historical-group", active_users=1, workers=1,
            dialogue_turns=2, history_variant=variant, role_source="history")
        assert report["status"] == "passed", report
        assert report["role_seed_profiles"]["derived_roles"] == 500
        first = json.loads((run / "events.jsonl").read_text().splitlines()[0])
        ctx = first["decision"]["context"]
        if field == "earliest_arrival":
            assert ctx["delivery"] is None
            values.append(ctx["events"][0]["details"].pop(field))
        else:
            values.append(ctx["delivery"].pop(field, None))
            synthetic = next(row for row in ctx["history"] if row["message_id"].startswith("variant:"))
            assert synthetic.pop(field, None) == values[-1]
            chat = [json.loads(line) for line in (run / "chat.jsonl").read_text().splitlines()]
            assert sum(row["direction"] == "scenario" for row in chat) == 1
        contexts.append(ctx)
    assert values[0] != values[1] and contexts[0] == contexts[1]
    assert compare_runs(tmp_path / pair[0], tmp_path / pair[1])["status"] == "passed"
