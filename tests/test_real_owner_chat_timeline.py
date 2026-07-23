from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TIMELINE_PATH = ROOT / "eval" / "golden" / "real_owner_chat_timeline_20260705_20260718.json"

EXPECTED_SOURCE_IMAGES = {
    "codex-clipboard-9726f879-aea8-49b6-a2b8-02817e7a1e5d.png",
    "codex-clipboard-5b1cb5d0-a77b-4aab-a4a4-665d2d11fee6.png",
    "codex-clipboard-5fc06355-c46c-4ba2-be2d-f7a9314d343d.png",
    "codex-clipboard-79912835-33a3-4ff4-b44d-807faa0fd01e.png",
    "codex-clipboard-8ce3d528-9e1f-48d3-86de-577a7cd71d07.png",
    "codex-clipboard-89f95267-8e41-4a9a-9f35-2d6a2c5219eb.png",
    "codex-clipboard-cf450f2a-45b7-47c0-b5f3-6e7d377c8e17.png",
    "codex-clipboard-e3011fdb-2c03-4846-9672-964a2a2c1e7f.png",
    "codex-clipboard-9163f3f5-59a0-41d1-9d1b-0660e1fbc373.png",
    "codex-clipboard-1d059ae8-c9c7-458b-a481-7544db61b6d5.png",
    "codex-clipboard-24717835-db6b-4cff-b878-a69ea85d9a89.png",
    "codex-clipboard-d4a0d34b-ee95-40b3-9a61-454e4a40a39d.png",
}


def _load_timeline() -> dict:
    return json.loads(TIMELINE_PATH.read_text(encoding="utf-8"))


def test_real_owner_chat_timeline_preserves_source_order_and_counts() -> None:
    record = _load_timeline()
    messages = record["messages"]

    assert record["id"] == "owner_chat_multi_episode_timeline_20260705_20260718_001"
    assert len(messages) == 86
    assert [item["turn"] for item in messages] == list(range(1, 87))
    assert sum(item["role"] == "customer" for item in messages) == 35
    assert sum(item["role"] == "boss" for item in messages) == 51
    assert all(item["text"].strip() for item in messages)
    assert all("表情包" not in item["text"] for item in messages)

    source = record["source"]
    assert source["omitted_media_policy"] == "exclude"
    assert set(source["source_image_files"]) == EXPECTED_SOURCE_IMAGES


def test_real_owner_chat_timeline_episode_ranges_partition_messages() -> None:
    record = _load_timeline()
    messages = {item["turn"]: item for item in record["messages"]}
    episodes = record["episodes"]

    covered_turns: list[int] = []
    for episode in episodes:
        episode_turns = list(range(episode["start_turn"], episode["end_turn"] + 1))
        covered_turns.extend(episode_turns)
        observed_times = [
            datetime.fromisoformat(messages[turn]["observed_at"])
            for turn in episode_turns
        ]
        assert observed_times == sorted(observed_times)

    assert len(episodes) == 7
    assert covered_turns == list(range(1, 87))
    assert [episode["business_exchange_rounds"] for episode in episodes] == [2, 4, 1, 1, 3, 8, 4]
    assert [episode["outcome"] for episode in episodes] == [
        "pending_unknown",
        "cancelled",
        "pending_unknown",
        "resolved_customer_unavailable",
        "cancelled",
        "completed",
        "completed",
    ]


def test_real_owner_chat_timeline_metrics_do_not_overclaim_success() -> None:
    metrics = _load_timeline()["timeline_metrics"]

    assert metrics["message_count"] == 86
    assert metrics["customer_message_count"] == 35
    assert metrics["boss_message_count"] == 51
    assert metrics["episode_count"] == 7
    assert metrics["terminal_episode_count"] == 5
    assert metrics["successful_episode_count"] == 2
    assert metrics["pending_or_unknown_episode_count"] == 2
    assert metrics["median_terminal_business_exchange_rounds"] == 4
    assert any("不能据此宣称稳定" in item for item in metrics["interpretation"])


def test_real_owner_chat_timeline_contains_high_value_replay_scenarios() -> None:
    record = _load_timeline()
    scenarios = {item["id"]: item for item in record["replay_scenarios"]}

    assert set(scenarios) == {
        "replay_fragmented_availability",
        "replay_existing_game_duration_exit",
        "replay_reverse_match_customer_already_full",
        "replay_parallel_options_then_cancel",
        "replay_two_people_two_games",
        "replay_couple_smoke_assignment_swap",
    }

    duration_exit = scenarios["replay_existing_game_duration_exit"]["expected"]
    assert duration_exit["final_task_status"] == "cancelled"
    assert duration_exit["must_not_leave_customer_confirmed_in_game"] is True

    parallel_cancel = scenarios["replay_parallel_options_then_cancel"]["expected"]
    assert parallel_cancel["cancel_releases_all_temporary_options"] is True
    assert parallel_cancel["later_match_requires_new_confirmation"] is True

    two_people = scenarios["replay_two_people_two_games"]["expected"]
    assert two_people["distinct_participants"] == ["customer", "girlfriend"]
    assert two_people["girlfriend_removed_from_agent_managed_options_after_turn_69"] is True
    assert two_people["room_reply"] == "03"

    role_swap = scenarios["replay_couple_smoke_assignment_swap"]["expected"]
    assert role_swap["must_not_conflate_participants"] is True
    assert role_swap["customer_final_smoke_assignment"] == "smoking"
    assert role_swap["girlfriend_final_smoke_assignment"] == "non_smoking"


def test_real_owner_chat_timeline_eval_cases_reference_real_turns() -> None:
    record = _load_timeline()
    turn_ids = {item["turn"] for item in record["messages"]}

    for case in record["eval_cases"]:
        assert case["id"]
        assert case["description"]
        assert set(case.get("context_turns", [])).issubset(turn_ids)
        assert case.get("input_turn") in turn_ids
        assert case["expected"]
