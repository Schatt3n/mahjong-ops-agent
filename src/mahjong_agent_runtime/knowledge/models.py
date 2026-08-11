"""定义麻将馆领域术语及其匹配结果的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DomainTerm:
    """一个经审核的术语及其规范化机器可读含义。"""

    term_id: str
    category: str
    aliases: tuple[str, ...]
    definition: str
    canonical: dict[str, Any]
    examples: tuple[str, ...] = ()
    usage_notes: tuple[str, ...] = ()
    confidence: str = "verified"
    source: str = "domain_review"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DomainTerm":
        """从持久化字典恢复领域对象，并补齐兼容的默认值。"""
        return cls(
            term_id=str(payload["term_id"]),
            category=str(payload["category"]),
            aliases=tuple(str(item) for item in payload.get("aliases") or ()),
            definition=str(payload["definition"]),
            canonical=dict(payload.get("canonical") or {}),
            examples=tuple(str(item) for item in payload.get("examples") or ()),
            usage_notes=tuple(str(item) for item in payload.get("usage_notes") or ()),
            confidence=str(payload.get("confidence") or "verified"),
            source=str(payload.get("source") or "domain_review"),
        )


@dataclass(frozen=True, slots=True)
class MatchedDomainTerm:
    """为当前上下文选出的术语，附带可审计证据。"""

    term: DomainTerm
    matched_aliases: tuple[str, ...] = field(default_factory=tuple)

    def to_context(self) -> dict[str, Any]:
        return {
            "term_id": self.term.term_id,
            "category": self.term.category,
            "matched_aliases": list(self.matched_aliases),
            "definition": self.term.definition,
            "canonical": dict(self.term.canonical),
            "usage_notes": list(self.term.usage_notes),
            "confidence": self.term.confidence,
            "source": self.term.source,
        }


__all__ = ["DomainTerm", "MatchedDomainTerm"]
