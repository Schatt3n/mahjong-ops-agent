from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass(slots=True)
class NormalizationChange:
    rule_id: str
    before: str
    after: str
    reason: str


@dataclass(slots=True)
class TextNormalizationResult:
    raw_text: str
    text: str
    changes: list[NormalizationChange] = field(default_factory=list)

    def changed_rule_ids(self) -> list[str]:
        return [item.rule_id for item in self.changes]


def normalize_mahjong_text(text: str) -> TextNormalizationResult:
    """Normalize deterministic input variants before semantic parsing.

    This layer only does low-risk canonicalization: unicode width, common
    punctuation variants, decimal stake separators, and domain typo aliases.
    It must not infer missing business facts such as people count or smoke
    preference.
    """

    raw = str(text or "")
    changes: list[NormalizationChange] = []

    normalized = _replace_number_emoji(raw)
    if normalized != raw:
        changes.append(
            NormalizationChange(
                rule_id="unicode.number_emoji",
                before=raw,
                after=normalized,
                reason="把数字 emoji 转成普通数字，便于后续解析。",
            )
        )

    normalized = _recording_replace(
        normalized,
        changes,
        rule_id="unicode.width",
        reason="使用 Unicode NFKC 统一全角/半角字符。",
        transform=lambda value: unicodedata.normalize("NFKC", value),
    )
    normalized = _recording_regex_sub(
        normalized,
        changes,
        rule_id="control_chars",
        pattern=r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
        replacement=" ",
        reason="移除不可见控制字符。",
    )
    normalized = _recording_regex_sub(
        normalized,
        changes,
        rule_id="wechat_prefix",
        pattern=r"(?:^|\s)(?:wxid_[a-z0-9]+|[a-z0-9_]{6,32})\s*:\s*",
        replacement=" ",
        reason="去掉导出消息中的微信前缀。",
        flags=re.I,
    )
    normalized = _recording_regex_sub(
        normalized,
        changes,
        rule_id="stake.decimal_half",
        pattern=r"(?<!\d)0\s*[\.,，、。．]\s*5(?!\d)",
        replacement="0.5",
        reason="麻将档位里 0。5/0，5/0,5/0、5 通常是 0.5。",
    )
    normalized = _recording_regex_sub(
        normalized,
        changes,
        rule_id="stake.decimal_space",
        pattern=r"(?<!\d)0\s+5(?!\d)",
        replacement="0.5",
        reason="麻将档位里 0 5 通常是 0.5。",
    )
    normalized = _recording_translate(
        normalized,
        changes,
        rule_id="punctuation.common",
        mapping={
            "，": ",",
            "。": ".",
            "：": ":",
            "～": "-",
            "－": "-",
            "🈵": "满",
            "🈲": "禁",
        },
        reason="统一常见中文标点和麻将聊天符号。",
    )
    normalized = _recording_translate(
        normalized,
        changes,
        rule_id="mahjong.aliases",
        mapping={
            "人气开": "人齐开",
            "人齐开的": "人齐开",
        },
        reason="归一常见输入法/语音转写别字。",
    )
    normalized = normalized.strip().lower()
    return TextNormalizationResult(raw_text=raw, text=normalized, changes=changes)


def _replace_number_emoji(text: str) -> str:
    number_emoji = {
        "0️⃣": "0",
        "1️⃣": "1 ",
        "2️⃣": "2 ",
        "3️⃣": "3 ",
        "4️⃣": "4 ",
        "5️⃣": "5 ",
        "6️⃣": "6 ",
        "7️⃣": "7 ",
        "8️⃣": "8 ",
        "9️⃣": "9 ",
    }
    for emoji, value in number_emoji.items():
        text = text.replace(emoji, value)
    return text


def _recording_replace(
    text: str,
    changes: list[NormalizationChange],
    *,
    rule_id: str,
    reason: str,
    transform,
) -> str:
    replaced = transform(text)
    if replaced != text:
        changes.append(NormalizationChange(rule_id=rule_id, before=text, after=replaced, reason=reason))
    return replaced


def _recording_regex_sub(
    text: str,
    changes: list[NormalizationChange],
    *,
    rule_id: str,
    pattern: str,
    replacement: str,
    reason: str,
    flags: int = 0,
) -> str:
    replaced = re.sub(pattern, replacement, text, flags=flags)
    if replaced != text:
        changes.append(NormalizationChange(rule_id=rule_id, before=text, after=replaced, reason=reason))
    return replaced


def _recording_translate(
    text: str,
    changes: list[NormalizationChange],
    *,
    rule_id: str,
    mapping: dict[str, str],
    reason: str,
) -> str:
    replaced = text
    for before, after in mapping.items():
        replaced = replaced.replace(before, after)
    if replaced != text:
        changes.append(NormalizationChange(rule_id=rule_id, before=text, after=replaced, reason=reason))
    return replaced
