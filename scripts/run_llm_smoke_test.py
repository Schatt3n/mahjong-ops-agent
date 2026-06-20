#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mahjong_agent import ChannelType, Message, OpenAICompatibleLLMResolver  # noqa: E402


def main() -> int:
    resolver = OpenAICompatibleLLMResolver.from_env()
    if resolver is None:
        print("LLM disabled: set MAHJONG_LLM_API_KEY or DASHSCOPE_API_KEY, plus provider/model if needed.")
        return 2

    config = resolver.config
    print("LLM enabled:")
    print(f"  provider={config.provider}")
    print(f"  model={config.model}")
    print(f"  base_url={config.base_url}")
    print("  api_key=<configured>")

    message = Message(
        text="老地方搭子还有吗，财敲还是上次那个",
        sender_id="smoke_user",
        sender_name="Smoke Test",
        channel_id="llm_smoke",
        channel_type=ChannelType.MANUAL,
        metadata={"disable_llm": False},
    )
    resolution = resolver.resolve(
        message,
        context={
            "known_local_aliases": {
                "cq": "杭麻财敲",
                "371": "三缺一",
                "272": "二缺二",
                "173": "一缺三",
            }
        },
    )
    print("Resolution:")
    print(f"  related={resolution.is_mahjong_related}")
    print(f"  intent={resolution.intent}")
    print(f"  confidence={resolution.confidence}")
    print(f"  normalized_text={resolution.normalized_text}")
    print(f"  reply_text={resolution.reply_text}")
    print(f"  human_review={resolution.needs_human_review}")
    print(f"  notes={resolution.notes}")
    return 0 if resolution.notes else 1


if __name__ == "__main__":
    raise SystemExit(main())
