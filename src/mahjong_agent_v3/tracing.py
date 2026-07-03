"""Compatibility trace imports for historical ``mahjong_agent_v3`` users."""

from __future__ import annotations

from mahjong_agent_runtime.tracing import (
    InMemoryTraceRecorder as InMemoryTraceRecorderV3,
    JsonlTraceRecorder as JsonlTraceRecorderV3,
    TraceEvent as TraceEventV3,
    trace_steps,
    validate_trace as validate_trace_v3,
)

__all__ = [
    "InMemoryTraceRecorderV3",
    "JsonlTraceRecorderV3",
    "TraceEventV3",
    "trace_steps",
    "validate_trace_v3",
]
