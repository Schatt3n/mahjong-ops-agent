from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from mahjong_agent.observability import (
    CONTROLLED_TRACE_SCHEMA_VERSION,
    CONTROLLED_WORKFLOW_REQUIRED_TRACE_STEPS,
    InMemoryTraceRecorder,
    JsonlTraceRecorder,
    TraceStep,
    validate_controlled_trace_completeness,
)
from mahjong_agent.workflow_models import ActionName, ActionSource, ProposedAction


TZ = ZoneInfo("Asia/Shanghai")


def test_trace_event_formats_auditable_log_line() -> None:
    recorder = InMemoryTraceRecorder()

    event = recorder.record(
        "trace_001",
        TraceStep.ACTION_PROPOSED,
        {
            "action": ProposedAction(
                name=ActionName.CREATE_GAME,
                source=ActionSource.LLM,
                confidence=0.91,
                reason="用户确认组局",
            )
        },
        occurred_at=datetime(2026, 6, 30, 15, 8, 9, tzinfo=TZ),
    )

    assert event.format_log_line().startswith("trace_001-2026-06-30 15:08:09-INFO: ")
    assert '"name": "create_game"' in event.format_log_line()
    assert recorder.get_trace("trace_001") == [event]


def test_trace_recorder_clear_by_trace_or_all() -> None:
    recorder = InMemoryTraceRecorder()
    recorder.record("trace_a", "custom_step", {"a": 1})
    recorder.record("trace_b", "custom_step", {"b": 2})

    assert recorder.clear("trace_a") == 1
    assert recorder.get_trace("trace_a") == []
    assert recorder.clear() == 1
    assert recorder.get_trace("trace_b") == []


def test_jsonl_trace_recorder_persists_structured_events(tmp_path) -> None:
    path = tmp_path / "traces" / "workflow_trace.jsonl"
    recorder = JsonlTraceRecorder(path)

    recorder.record(
        "trace_jsonl",
        TraceStep.FINAL_OUTPUT,
        {"final_text": "好的，我帮你问问。"},
        occurred_at=datetime(2026, 6, 30, 16, 1, 2, tzinfo=TZ),
    )

    text = path.read_text(encoding="utf-8")
    assert f'"schema_version":"{CONTROLLED_TRACE_SCHEMA_VERSION}"' in text
    assert '"step":"final_output"' in text
    assert "trace_jsonl-2026-06-30 16:01:02-INFO:" in text
    assert "好的，我帮你问问。" in text

    loaded = recorder.get_trace("trace_jsonl")
    assert len(loaded) == 1
    assert loaded[0].step == TraceStep.FINAL_OUTPUT
    assert loaded[0].content["final_text"] == "好的，我帮你问问。"


def test_validate_controlled_trace_completeness_reports_missing_steps() -> None:
    recorder = InMemoryTraceRecorder()
    recorder.record("trace_partial", TraceStep.USER_INPUT, {"text": "老板"})
    recorder.record("trace_partial", TraceStep.FINAL_OUTPUT, {"final_text": "收到"})

    report = validate_controlled_trace_completeness(recorder.get_trace("trace_partial"))

    assert report.schema_version == CONTROLLED_TRACE_SCHEMA_VERSION
    assert report.complete is False
    assert report.required_steps == [step.value for step in CONTROLLED_WORKFLOW_REQUIRED_TRACE_STEPS]
    assert report.present_steps == ["user_input", "final_output"]
    assert "context_built" in report.missing_steps
    assert "reply_guarded" in report.missing_steps
    assert "reply_approval" in report.missing_steps
