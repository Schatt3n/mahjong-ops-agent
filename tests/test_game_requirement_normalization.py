from __future__ import annotations

from mahjong_agent_runtime.domains.game_domain import normalize_requirement


def test_normalize_requirement_derives_four_player_seat_format_from_counts() -> None:
    assert normalize_requirement({"known_player_count": 1, "needed_seats": 3})["seat_format"] == "173"
    assert normalize_requirement({"known_player_count": 2, "needed_seats": 2})["seat_format"] == "272"
    assert normalize_requirement({"known_player_count": 3, "needed_seats": 1})["seat_format"] == "371"


def test_normalize_requirement_repairs_conflicting_seat_format_from_authoritative_counts() -> None:
    normalized = normalize_requirement(
        {
            "known_player_count": 3,
            "needed_seats": 1,
            "seat_format": "272",
        }
    )

    assert normalized["seat_format"] == "371"


def test_normalize_requirement_does_not_invent_seat_format_for_incomplete_or_invalid_counts() -> None:
    assert "seat_format" not in normalize_requirement({"known_player_count": 3})
    assert "seat_format" not in normalize_requirement({"known_player_count": 3, "needed_seats": 2})

