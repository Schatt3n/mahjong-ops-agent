"""Trace helpers for the stable Mahjong Agent Runtime import surface."""

from __future__ import annotations

from mahjong_agent_v3.tracing import (
    InMemoryTraceRecorderV3,
    JsonlTraceRecorderV3,
    TraceEventV3,
    validate_trace_v3,
)

InMemoryTraceRecorder = InMemoryTraceRecorderV3
JsonlTraceRecorder = JsonlTraceRecorderV3
TraceEvent = TraceEventV3
validate_trace = validate_trace_v3

__all__ = [
    "InMemoryTraceRecorder",
    "InMemoryTraceRecorderV3",
    "JsonlTraceRecorder",
    "JsonlTraceRecorderV3",
    "TraceEvent",
    "TraceEventV3",
    "validate_trace",
    "validate_trace_v3",
]
