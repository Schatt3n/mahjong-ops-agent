from __future__ import annotations

import json

from mahjong_agent.trial_eval_data import (
    BOSS_REPLY_FEW_SHOTS,
    TrialEvalDataPaths,
    TrialEvalDataStore,
    eval_case_tags,
    few_shot_parsed_text,
    read_jsonl_records,
)


def make_paths(tmp_path) -> TrialEvalDataPaths:
    return TrialEvalDataPaths(
        golden=tmp_path / "golden.jsonl",
        boss_trial_golden=tmp_path / "boss_trial_golden.jsonl",
        badcase=tmp_path / "badcases.jsonl",
        few_shot=tmp_path / "few_shot.jsonl",
        skills=tmp_path / "skills.jsonl",
    )


def test_trial_eval_data_store_overview_counts_and_recent_records(tmp_path) -> None:
    paths = make_paths(tmp_path)
    store = TrialEvalDataStore(paths)
    store.append_case("badcase", {"kind": "badcase", "id": "bad_1"})
    store.append_case("golden", {"kind": "golden", "id": "gold_1"})
    store.append_case("few_shot", {"kind": "few_shot", "id": "few_1"})

    overview = store.overview()

    assert overview["paths"]["badcase"] == str(paths.badcase)
    assert overview["counts"]["badcase"] == 1
    assert overview["counts"]["golden"] == 1
    assert overview["counts"]["few_shot"] == 1
    assert overview["recent"]["badcase"] == [{"kind": "badcase", "id": "bad_1"}]
    assert overview["runner"] == "PYTHONPATH=src python scripts/run_scenario_eval.py"


def test_trial_eval_data_store_merges_static_and_dynamic_few_shots(tmp_path) -> None:
    paths = make_paths(tmp_path)
    dynamic_records = [
        {
            "kind": "few_shot",
            "id": f"few_{index}",
            "name": f"老板话术 {index}",
            "customer_message": f"消息 {index}",
            "parsed": "杭麻 0.5",
            "reply_text": f"回复 {index}",
        }
        for index in range(10)
    ]
    paths.few_shot.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in dynamic_records),
        encoding="utf-8",
    )

    examples = TrialEvalDataStore(paths).few_shot_examples()

    assert examples[: len(BOSS_REPLY_FEW_SHOTS)] == BOSS_REPLY_FEW_SHOTS
    assert [item["customer_message"] for item in examples[-2:]] == ["消息 8", "消息 9"]
    assert len(examples) == len(BOSS_REPLY_FEW_SHOTS) + 8


def test_eval_case_tags_are_deduped_and_contextual() -> None:
    tags = eval_case_tags(
        {"tags": "弱意图, 张哥 弱意图"},
        {
            "used_short_memory": True,
            "decision": {"action": "queue_invites"},
            "parsed": {"game_label": "杭麻"},
        },
        "badcase",
    )

    assert tags == ["弱意图", "张哥", "badcase", "queue_invites", "short_memory", "杭麻"]


def test_few_shot_parsed_text_summarizes_slots_and_missing_fields() -> None:
    text = few_shot_parsed_text(
        {
            "game_label": "杭麻",
            "level": "0.5",
            "start_time": "14:00",
            "missing_count": 3,
            "rules": ["无烟", "财敲"],
        },
        {"missing_fields": ["duration"]},
    )

    assert text == "杭麻，0.5，14:00，缺3，无烟，财敲，待确认：duration"


def test_read_jsonl_records_skips_invalid_lines_and_limits_from_tail(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"id":1}\nnot json\n[]\n{"id":2}\n{"id":3}\n', encoding="utf-8")

    assert read_jsonl_records(path, limit=2) == [{"id": 2}, {"id": 3}]
