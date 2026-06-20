from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .models import Message


TEXT_METADATA_KEYS = [
    "caption",
    "audio_transcript",
    "asr_text",
    "image_ocr_text",
    "ocr_text",
    "image_description",
    "vision_description",
    "sticker_description",
    "sticker_text",
    "emoji_text",
]


LEAD_PATTERNS = [
    re.compile(r"(打|玩|搓|约).*(麻将|牌|麻|一把|一局)"),
    re.compile(r"(麻将|牌局|麻局).*(有人|有局|约吗|来吗|缺人|三缺一|二缺二|一缺三)"),
    re.compile(r"(有人|有局|有没有人).*(麻将|牌|麻|搓|打|玩)"),
    re.compile(r"(下班|今晚|今天|明天|周末).*(麻将|牌|搓|来一把|约一局)"),
    re.compile(r"(🀄|🀅|🀆|🀇|🀈|🀉|🀊|🀋|🀌|🀍|🀎|🀏|🀐|🀑|🀒|🀓|🀔|🀕|🀖|🀗|🀘|🀙|🀚|🀛|🀜|🀝|🀞|🀟|🀠|🀡|🀢|🀣|🀤|🀥|🀦|🀧|🀨|🀩|🀪|🀫)"),
]


@dataclass(slots=True)
class IntentEvidence:
    combined_text: str
    modalities: list[str] = field(default_factory=list)
    evidence: dict[str, str] = field(default_factory=dict)
    lead_score: float = 0.0
    lead_reasons: list[str] = field(default_factory=list)

    @property
    def is_potential_lead(self) -> bool:
        return self.lead_score >= 0.35


def extract_intent_evidence(message: Message) -> IntentEvidence:
    evidence: dict[str, str] = {}
    modalities: list[str] = []

    if message.text.strip() and not _is_transport_placeholder(message.text):
        evidence["text"] = message.text.strip()
        modalities.append("text")

    for key in TEXT_METADATA_KEYS:
        value = _string_value(message.metadata.get(key))
        if value:
            evidence[key] = value
            modalities.append(_modality_for_key(key))

    message_type = _string_value(message.metadata.get("message_type"))
    if message_type and message_type not in modalities:
        modalities.append(message_type)

    combined_text = " ".join(evidence.values()).strip()
    lead_score, lead_reasons = _score_lead(combined_text, evidence)

    return IntentEvidence(
        combined_text=combined_text,
        modalities=sorted(set(modalities)),
        evidence=evidence,
        lead_score=lead_score,
        lead_reasons=lead_reasons,
    )


def message_for_intent(message: Message, evidence: IntentEvidence | None = None) -> Message:
    evidence = evidence or extract_intent_evidence(message)
    if not evidence.combined_text or evidence.combined_text == message.text:
        return message
    metadata = dict(message.metadata)
    metadata["intent_evidence"] = {
        "modalities": evidence.modalities,
        "evidence": evidence.evidence,
        "lead_score": evidence.lead_score,
        "lead_reasons": evidence.lead_reasons,
    }
    return Message(
        text=evidence.combined_text,
        sender_id=message.sender_id,
        sender_name=message.sender_name,
        channel_id=message.channel_id,
        channel_type=message.channel_type,
        sent_at=message.sent_at,
        id=message.id,
        metadata=metadata,
    )


def has_intent_content(message: Message) -> bool:
    return bool(extract_intent_evidence(message).combined_text.strip())


def _score_lead(text: str, evidence: dict[str, str]) -> tuple[float, list[str]]:
    normalized = _normalize(text)
    reasons: list[str] = []
    score = 0.0

    for pattern in LEAD_PATTERNS:
        if pattern.search(normalized):
            score += 0.35
            reasons.append("命中麻将/牌局意向表达")
            break

    if any(key in evidence for key in ["audio_transcript", "asr_text"]):
        score += 0.08
        reasons.append("来自语音转写")
    if any(key in evidence for key in ["image_ocr_text", "ocr_text", "image_description", "vision_description"]):
        score += 0.08
        reasons.append("来自图片识别")
    if any(key in evidence for key in ["sticker_description", "sticker_text", "emoji_text"]):
        score += 0.08
        reasons.append("来自表情/表情包")
    if re.search(r"(下班|今晚|今天|明天|周末|几点|现在|晚点)", normalized):
        score += 0.12
        reasons.append("包含时间/到店场景")
    if re.search(r"(有人|有局|约吗|来吗|缺人|还缺|三缺一|二缺二|一缺三)", normalized):
        score += 0.16
        reasons.append("包含找人/有局信号")
    if re.search(r"(0\.5|五毛|一块|1块|2块|两块|无烟|包间)", normalized):
        score += 0.12
        reasons.append("包含玩法/档位偏好")

    return round(min(score, 1.0), 2), reasons


def _normalize(text: str) -> str:
    return (
        text.strip()
        .replace("，", ",")
        .replace("。", ".")
        .replace("：", ":")
        .lower()
    )


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def _is_transport_placeholder(text: str) -> bool:
    return text.strip().lower() in {
        "[语音]",
        "[音频]",
        "[图片]",
        "[表情]",
        "[表情包]",
        "[audio]",
        "[image]",
        "[sticker]",
    }


def _modality_for_key(key: str) -> str:
    if key in {"audio_transcript", "asr_text"}:
        return "audio"
    if key in {"image_ocr_text", "ocr_text", "image_description", "vision_description"}:
        return "image"
    if key in {"sticker_description", "sticker_text", "emoji_text"}:
        return "sticker"
    return "text"
