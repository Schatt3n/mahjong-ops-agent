from __future__ import annotations

from mahjong_agent import normalize_mahjong_text


def test_normalize_decimal_half_stake_variants_with_evidence() -> None:
    samples = [
        "通宵0。5有人吗",
        "通宵0．5有人吗",
        "通宵0，5有人吗",
        "通宵0,5有人吗",
        "通宵0、5有人吗",
        "通宵0 5有人吗",
    ]

    for text in samples:
        result = normalize_mahjong_text(text)
        assert "0.5" in result.text
        assert "0/5" not in result.text
        rule_ids = result.changed_rule_ids()
        assert any(rule_id.startswith("stake.decimal") for rule_id in rule_ids) or "unicode.width" in rule_ids


def test_normalize_renqikai_typo_without_business_slot_inference() -> None:
    result = normalize_mahjong_text("现在有没有0。5无烟人气开的")

    assert result.text == "现在有没有0.5无烟人齐开"
    assert "stake.decimal_half" in result.changed_rule_ids()
    assert "mahjong.aliases" in result.changed_rule_ids()
