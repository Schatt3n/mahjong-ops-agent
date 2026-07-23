"""Versioned Mahjong domain terminology used by parsers and model context."""

from .models import DomainTerm, MatchedDomainTerm
from .repository import DomainTerminologyRepository, default_terminology_repository

__all__ = [
    "DomainTerm",
    "DomainTerminologyRepository",
    "MatchedDomainTerm",
    "default_terminology_repository",
]
