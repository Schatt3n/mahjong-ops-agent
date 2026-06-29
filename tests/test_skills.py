from __future__ import annotations

from mahjong_agent.skills import load_skill_records, select_relevant_skills


def test_skill_library_loads_operation_skills() -> None:
    records = load_skill_records()

    assert records
    assert any(record["id"] == "slot_party_size_confirmation" for record in records)
    assert all(record.get("kind") == "operation_skill" for record in records)


def test_select_relevant_skills_by_stage_and_trigger() -> None:
    skills = select_relevant_skills(
        stage="reply_draft",
        text="下午两点 0.5 无烟杭麻，帮我组一桌",
        limit=4,
    )
    ids = {skill["id"] for skill in skills}

    assert "slot_party_size_confirmation" in ids
    assert "time_ambiguity_guard" in ids
    assert all("instructions" in skill for skill in skills)


def test_select_relevant_skills_for_flexible_start_and_overnight() -> None:
    skills = select_relevant_skills(
        stage="semantic_resolution",
        text="尽快开吧，时间可以再商量，通宵也可以",
        limit=4,
    )
    ids = {skill["id"] for skill in skills}

    assert "flexible_start_and_overnight_strategy" in ids
