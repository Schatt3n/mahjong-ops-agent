"""送达错报率按冻结批次计算，来源、真实送达和待核证据各有口径。"""

import sqlite3

import pytest

from scripts.report_venue_hallucinations import summarize_cohort


def _archive(tmp_path):
    """建立有群看板、顾客跟进及未核对回执的小型账本。"""
    with sqlite3.connect(tmp_path / "chat.sqlite3") as connection:
        connection.executescript("""
            CREATE TABLE venue_messages (
                sequence INTEGER, message_id TEXT, trace_id TEXT, sent_at TEXT, direction TEXT, sender_id TEXT
            );
            INSERT INTO venue_messages VALUES
                (1, 'history', 'old', '2026-11-25T18:59:00+08:00', 'history', 'simulation-agent'),
                (2, 'notice', 'notice1', '2026-11-25T19:00:00+08:00', 'outbound', 'simulation-agent'),
                (3, 'customer:timer1', 'c1', '2026-11-25T19:00:00+08:00', 'inbound', 'customer'),
                (4, 'reply1', 'r1', '2026-11-25T19:00:00+08:00', 'outbound', 'simulation-agent'),
                (5, 'customer:private1', 'c2', '2026-11-25T19:02:00+08:00', 'inbound', 'customer'),
                (6, 'reply2', 'r2', '2026-11-25T19:02:00+08:00', 'outbound', 'simulation-agent'),
                (7, 'reply3', 'r3', '2026-11-25T19:03:00+08:00', 'outbound', 'simulation-agent');
        """)
    with sqlite3.connect(tmp_path / "role_events.sqlite3") as connection:
        connection.executescript("""
            CREATE TABLE venue_role_events (event_id TEXT, kind TEXT, status TEXT, due_at TEXT);
            INSERT INTO venue_role_events VALUES
                ('timer1', 'timer', 'finished', '2026-11-25T11:00:00.000000+00:00'),
                ('timer2', 'timer', 'finished', '2026-11-25T11:00:00.000000+00:00'),
                ('private1', 'private_receipt', 'finished', '2026-11-25T11:02:00.000000+00:00');
        """)


def test_hallucination_report_preserves_origin_and_assessment_coverage(tmp_path):
    _archive(tmp_path)
    report = summarize_cohort({"label": "shop", "role": "agent", "run_directory": tmp_path,
        "through_sequence": 6, "assessments": [
            {"sequence": 2, "source": "deterministic", "verdict": "no_issue_found", "evidence": ["board"]},
            {"sequence": 4, "source": "model", "verdict": "unsupported_claim", "evidence": ["receipt"]},
        ]})

    assert [row["sequence"] for row in report["deliveries"]] == [2, 4, 6]
    assert report["sources"] == {"deterministic": 1, "model": 1, "unknown": 1}
    assert report["all_delivered"]["unreviewed"] == 1
    assert report["model_delivered"]["confirmed_issue_per_delivery"] == 1
    assert report["model_delivered"]["factual_mismatch_per_delivery"] == 0


def test_hallucination_report_uses_actual_timer_receipt_and_sequence_range(tmp_path):
    _archive(tmp_path)
    cohort = {"label": "customers", "role": "customer", "run_directory": tmp_path,
        "through_sequence": 6, "timer_due_at": "2026-11-25T19:00:00+08:00", "assessments": [
            {"sequence": 3, "source": "model", "verdict": "uncertain", "evidence": ["actor event"]},
        ]}
    report = summarize_cohort(cohort)

    assert [row["sequence"] for row in report["deliveries"]] == [3]
    assert report["model_delivered"]["total"] == report["model_delivered"]["uncertain"] == 1
    assert report["model_delivered"]["confirmed_issue_among_resolved"] is None
    with pytest.raises(ValueError, match="frozen cohort"):
        summarize_cohort({**cohort, "from_sequence": 4})
    with sqlite3.connect(tmp_path / "role_events.sqlite3") as connection:
        connection.execute("UPDATE venue_role_events SET status = 'claimed' WHERE event_id = 'timer2'")
    with pytest.raises(ValueError, match="still in progress"):
        summarize_cohort(cohort)
