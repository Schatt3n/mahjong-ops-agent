from __future__ import annotations

"""Shared conservative token estimation for context and budget guardrails."""

import json
import math
import unicodedata
from typing import Any


def estimate_tokens(value: Any) -> int:
    """Estimate tokens without coupling the runtime to one model tokenizer.

    Chinese characters are usually much closer to one token per character than
    the old four-characters-per-token heuristic, so they are counted separately.
    """

    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    cjk = 0
    ascii_word = 0
    punctuation = 0
    for char in text:
        codepoint = ord(char)
        if _is_cjk(codepoint) or (codepoint > 127 and unicodedata.category(char).startswith("L")):
            cjk += 1
        elif char.isascii() and (char.isalnum() or char.isspace() or char == "_"):
            ascii_word += 1
        else:
            punctuation += 1
    return max(1, cjk + math.ceil(ascii_word / 4) + math.ceil(punctuation / 2) + 4)


def _is_cjk(codepoint: int) -> bool:
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
        or 0x3040 <= codepoint <= 0x30FF
        or 0xAC00 <= codepoint <= 0xD7AF
    )
