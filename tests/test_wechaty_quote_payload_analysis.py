from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_wechaty_quote_payloads.py"
    spec = importlib.util.spec_from_file_location("analyze_wechaty_quote_payloads", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_detects_nested_payload_quote_field() -> None:
    module = load_module()

    report = module.analyze_records(
        [
            {
                "conversation_id": "wechaty:contact:alice",
                "source_message_id": "m1",
                "sender_id": "alice",
                "sender_name": "Alice",
                "text": "可以",
                "payload": {
                    "id": "m1",
                    "text": "可以",
                    "quote": {"id": "q1", "text": "14:00，0.5无烟，打吗？"},
                },
            }
        ]
    )

    assert report["total_records"] == 1
    assert report["candidate_record_count"] == 1
    candidate = report["candidate_records"][0]
    assert candidate["source_message_id"] == "m1"
    assert any(item["path"] == "$.payload.quote" for item in candidate["candidates"])
    assert "14:00" in candidate["candidates"][0]["value_preview"]


def test_detects_raw_observation_quote_candidates() -> None:
    module = load_module()

    report = module.analyze_records(
        [
            {
                "conversation_id": "wechaty:room:test",
                "source_message_id": "m2",
                "text": "不来了",
                "raw_observation": {
                    "quote_candidates": [
                        {
                            "path": "$.payload.quoted_message",
                            "value": {"id": "invite1", "text": "今晚 7 点三缺一"},
                        }
                    ]
                },
            }
        ]
    )

    assert report["candidate_record_count"] == 1
    paths = {item["path"] for item in report["candidate_records"][0]["candidates"]}
    assert "$.raw_observation.quote_candidates" in paths
    assert "$.raw_observation.quote_candidates[0].path" in paths
    assert "$.raw_observation.quote_candidates[0].value" in paths


def test_non_quote_record_is_counted_without_candidates() -> None:
    module = load_module()

    report = module.analyze_records(
        [
            {
                "conversation_id": "wechaty:contact:alice",
                "source_message_id": "m3",
                "text": "哈哈",
                "payload": {"id": "m3", "text": "哈哈"},
            }
        ]
    )

    assert report["total_records"] == 1
    assert report["candidate_record_count"] == 0
    assert report["candidate_records"] == []
