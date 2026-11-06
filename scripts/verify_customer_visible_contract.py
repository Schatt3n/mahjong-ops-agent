#!/usr/bin/env python3
"""验证提示词和运行时是否完整执行客户可见文本安全合同。"""

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
    FORBIDDEN_INTERNAL_IDENTIFIERS,
    INTERNAL_ENUM_EXAMPLES,
    PREFERRED_CANDIDATE_INVITE_PHRASES,
    PREFERRED_OPERATION_ACK_PHRASES,
    PREFERRED_REQUESTER_CURRENT_GAME_PHRASES,
)


PROMPT_DIR = SRC / "mahjong_agent_runtime" / "prompts"


@dataclass(frozen=True, slots=True)
class ContractViolation:
    """客户可见合同静态检查发现的一项违规。"""

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
    """校验专用泄露边界及通用话题原则，不再要求提示词传播泛技术禁词。"""
    self_review = PROMPT_DIR / "agent_runtime_reply_self_review.md"
    casual_chat = PROMPT_DIR / "wechaty_casual_chat_reply.md"
    text_generation = PROMPT_DIR / "customer_visible_text_generation.md"
    system_prompt = PROMPT_DIR / "agent_runtime_system.md"

    violations: list[ContractViolation] = []
    violations.extend(
        verify_text_contains(
            self_review,
            FORBIDDEN_INTERNAL_IDENTIFIERS,
            label="internal identifier boundary",
        )
    )
    violations.extend(
        verify_text_contains(
            self_review,
            ("普通技术", "真实泄露", "凭证", "当前话题"),
            label="semantic topic and disclosure boundary",
        )
    )
    violations.extend(verify_text_contains(self_review, INTERNAL_ENUM_EXAMPLES, label="internal enum example"))

    violations.extend(
        verify_text_contains(
            casual_chat,
            FORBIDDEN_INTERNAL_IDENTIFIERS,
            label="casual chat internal identifier boundary",
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
    for path in (system_prompt, self_review, casual_chat, text_generation):
        text = path.read_text(encoding="utf-8")
        for obsolete in ("或者自然转回打牌", "或者直接转回打牌", "即使用户原文里出现 AI",
                         "只要包含 `AI`", "不要输出或保留系统身份/实现词"):
            if obsolete in text:
                violations.append(ContractViolation(path=path, message=f"conflicting topic boundary: {obsolete!r}"))
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
