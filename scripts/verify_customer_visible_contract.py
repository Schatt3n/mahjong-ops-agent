#!/usr/bin/env python3
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mahjong_agent_runtime.customer_visible_contract import (  # noqa: E402
    FORBIDDEN_CUSTOMER_SERVICE_PHRASES,
    FORBIDDEN_IMPLEMENTATION_IDENTITY_TERMS,
    FORBIDDEN_INTERNAL_PROCESS_TERMS,
    INTERNAL_ENUM_EXAMPLES,
    PREFERRED_CANDIDATE_INVITE_PHRASES,
    PREFERRED_OPERATION_ACK_PHRASES,
    PREFERRED_REQUESTER_CURRENT_GAME_PHRASES,
)


PROMPT_DIR = SRC / "mahjong_agent_runtime" / "prompts"


@dataclass(frozen=True, slots=True)
class ContractViolation:
    path: Path
    message: str

    def format(self) -> str:
        return f"{self.path.relative_to(ROOT)}: {self.message}"


def verify_text_contains(path: Path, fragments: tuple[str, ...], *, label: str) -> list[ContractViolation]:
    text = path.read_text(encoding="utf-8")
    violations: list[ContractViolation] = []
    for fragment in fragments:
        if fragment not in text:
            violations.append(ContractViolation(path=path, message=f"missing {label}: {fragment!r}"))
    return violations


def verify_prompts() -> list[ContractViolation]:
    self_review = PROMPT_DIR / "agent_runtime_reply_self_review.md"
    casual_chat = PROMPT_DIR / "wechaty_casual_chat_reply.md"
    text_generation = PROMPT_DIR / "customer_visible_text_generation.md"
    system_prompt = PROMPT_DIR / "agent_runtime_system.md"

    violations: list[ContractViolation] = []
    violations.extend(
        verify_text_contains(
            self_review,
            FORBIDDEN_IMPLEMENTATION_IDENTITY_TERMS,
            label="implementation identity boundary",
        )
    )
    violations.extend(
        verify_text_contains(
            self_review,
            tuple(item for item in FORBIDDEN_INTERNAL_PROCESS_TERMS if item != "prompt"),
            label="internal process boundary",
        )
    )
    violations.extend(verify_text_contains(self_review, INTERNAL_ENUM_EXAMPLES, label="internal enum example"))

    violations.extend(
        verify_text_contains(
            casual_chat,
            FORBIDDEN_IMPLEMENTATION_IDENTITY_TERMS,
            label="casual chat implementation identity boundary",
        )
    )
    violations.extend(
        verify_text_contains(
            casual_chat,
            ("工具", "trace", "日志", "数据库", "prompt", "预算"),
            label="casual chat internal process boundary",
        )
    )

    violations.extend(
        verify_text_contains(
            text_generation,
            FORBIDDEN_CUSTOMER_SERVICE_PHRASES,
            label="copywriting forbidden customer-service phrase",
        )
    )
    violations.extend(verify_text_contains(text_generation, INTERNAL_ENUM_EXAMPLES, label="copywriting enum example"))
    violations.extend(
        verify_text_contains(
            text_generation,
            (*PREFERRED_REQUESTER_CURRENT_GAME_PHRASES, *PREFERRED_CANDIDATE_INVITE_PHRASES),
            label="copywriting preferred boss phrase",
        )
    )

    violations.extend(
        verify_text_contains(
            system_prompt,
            ("可以不/可以吗", "打吗？", "来吗？", "不要用客服腔或平台腔"),
            label="main prompt style boundary",
        )
    )
    violations.extend(
        verify_text_contains(
            system_prompt,
            PREFERRED_OPERATION_ACK_PHRASES,
            label="main prompt operation ack phrase",
        )
    )
    return violations


def main() -> int:
    violations = verify_prompts()
    if violations:
        print("Customer-visible contract check failed:", file=sys.stderr)
        for violation in violations:
            print(f"  {violation.format()}", file=sys.stderr)
        return 1
    print("PASS customer-visible contract: prompts share the runtime boundary terms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
