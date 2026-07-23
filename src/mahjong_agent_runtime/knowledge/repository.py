"""Load and retrieve reviewed Mahjong terminology without semantic branching."""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from importlib.resources import files
from typing import Iterable

from .models import DomainTerm, MatchedDomainTerm


class DomainTerminologyRepository:
    """Read-only terminology index shared by deterministic and model paths."""

    def __init__(self, terms: Iterable[DomainTerm]) -> None:
        self._terms = tuple(terms)
        self._validate()

    @classmethod
    def from_package_data(cls) -> "DomainTerminologyRepository":
        resource = files("mahjong_agent_runtime.knowledge").joinpath("mahjong_terms.jsonl")
        terms: list[DomainTerm] = []
        with resource.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"terminology line {line_number} must be a JSON object")
                terms.append(DomainTerm.from_dict(payload))
        return cls(terms)

    @property
    def terms(self) -> tuple[DomainTerm, ...]:
        return self._terms

    def relevant_terms(
        self,
        texts: Iterable[str],
        *,
        categories: set[str] | None = None,
        max_terms: int = 20,
    ) -> list[MatchedDomainTerm]:
        """Return only terms evidenced by this task's bounded text."""

        normalized_texts = [_normalize(item) for item in texts if str(item or "").strip()]
        ranked: list[tuple[int, int, MatchedDomainTerm]] = []
        for term_index, term in enumerate(self._terms):
            if categories is not None and term.category not in categories:
                continue
            matched_aliases = {
                alias
                for alias in term.aliases
                if any(_alias_present(text, alias) for text in normalized_texts)
            }
            if not matched_aliases:
                continue
            longest = max(len(_normalize(alias)) for alias in matched_aliases)
            ranked.append(
                (
                    -longest,
                    term_index,
                    MatchedDomainTerm(
                        term=term,
                        matched_aliases=tuple(sorted(matched_aliases, key=lambda item: (-len(item), item))),
                    ),
                )
            )
        ranked.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in ranked[: max(0, max_terms)]]

    def context_for_texts(self, texts: Iterable[str], *, max_terms: int = 20) -> list[dict]:
        return [item.to_context() for item in self.relevant_terms(texts, max_terms=max_terms)]

    def first_match(
        self,
        text: str,
        *,
        categories: set[str] | None = None,
    ) -> MatchedDomainTerm | None:
        matches = self.relevant_terms([text], categories=categories, max_terms=len(self._terms))
        return matches[0] if matches else None

    def _validate(self) -> None:
        term_ids: set[str] = set()
        for term in self._terms:
            if not term.term_id or term.term_id in term_ids:
                raise ValueError(f"duplicate or empty terminology term_id: {term.term_id!r}")
            if not term.aliases or any(not alias.strip() for alias in term.aliases):
                raise ValueError(f"terminology term {term.term_id!r} must have non-empty aliases")
            term_ids.add(term.term_id)


@lru_cache(maxsize=1)
def default_terminology_repository() -> DomainTerminologyRepository:
    """Return one immutable process-wide terminology index."""

    return DomainTerminologyRepository.from_package_data()


def _normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    return (
        text.replace("，", ",")
        .replace("。", ".")
        .replace("．", ".")
        .replace("－", "-")
        .replace("—", "-")
    )


def _alias_present(normalized_text: str, alias: str) -> bool:
    normalized_alias = _normalize(alias)
    if not normalized_alias:
        return False
    if normalized_alias.isdigit():
        return re.search(rf"(?<!\d){re.escape(normalized_alias)}(?!\d)", normalized_text) is not None
    if normalized_alias.isascii() and normalized_alias.isalpha():
        return (
            re.search(
                rf"(?<![a-z]){re.escape(normalized_alias)}(?![a-z])",
                normalized_text,
            )
            is not None
        )
    return normalized_alias in normalized_text


__all__ = ["DomainTerminologyRepository", "default_terminology_repository"]
