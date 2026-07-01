from mahjong_agent import (
    SlotSource,
    SlotValue,
    semantic_slot_confidence,
    semantic_slot_needs_confirmation,
    semantic_slot_source,
    semantic_slot_usable,
    semantic_slot_value,
)


def test_semantic_slot_helpers_read_dict_contract() -> None:
    slot = {
        "value": "0.5",
        "source": "explicit",
        "confidence": 0.91,
        "needs_confirmation": False,
    }

    assert semantic_slot_value(slot) == "0.5"
    assert semantic_slot_source(slot) == "explicit"
    assert semantic_slot_confidence(slot) == 0.91
    assert semantic_slot_needs_confirmation(slot) is False
    assert semantic_slot_usable(slot, min_confidence=0.7) is True


def test_semantic_slot_helpers_read_slot_value_contract() -> None:
    slot = SlotValue(
        name="stake",
        value="1",
        source=SlotSource.CONTEXT,
        confidence=0.8,
        confirmed=True,
        needs_confirmation=False,
    )

    assert semantic_slot_value(slot) == "1"
    assert semantic_slot_source(slot) == "context"
    assert semantic_slot_confidence(slot) == 0.8
    assert semantic_slot_needs_confirmation(slot) is False
    assert semantic_slot_usable(slot, min_confidence=0.75) is True


def test_semantic_slot_usable_rejects_low_confidence_or_confirmation_needed() -> None:
    assert (
        semantic_slot_usable(
            {
                "value": "no_smoke",
                "source": "explicit",
                "confidence": 0.4,
                "needs_confirmation": False,
            },
            min_confidence=0.7,
        )
        is False
    )
    assert (
        semantic_slot_usable(
            {
                "value": "no_smoke",
                "source": "explicit",
                "confidence": 0.9,
                "needs_confirmation": True,
            },
            min_confidence=0.7,
        )
        is False
    )


def test_semantic_slot_usable_rejects_unknown_or_unstructured_slots() -> None:
    assert (
        semantic_slot_usable(
            {
                "value": "0.5",
                "source": "unknown",
                "confidence": 0.9,
                "needs_confirmation": False,
            },
            min_confidence=0.7,
        )
        is False
    )
    assert semantic_slot_value("raw") == "raw"
    assert semantic_slot_confidence("raw") == 0.0
    assert semantic_slot_source("raw") == ""
    assert semantic_slot_needs_confirmation("raw") is True
    assert semantic_slot_usable("raw", min_confidence=0.1) is False
