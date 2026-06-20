#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys
from datetime import datetime
from zoneinfo import ZoneInfo


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mahjong_agent import AgentResponder, ChannelType, Message, OpenAICompatibleLLMResolver  # noqa: E402


TZ = ZoneInfo("Asia/Shanghai")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one agent-level LLM smoke test.")
    parser.add_argument("--text", default="老地方搭子还有吗，财敲还是上次那个")
    parser.add_argument("--channel-type", default=ChannelType.MANUAL.value)
    args = parser.parse_args()

    resolver = OpenAICompatibleLLMResolver.from_env()
    if resolver is None:
        print("LLM disabled: env config was not loaded.")
        return 2

    print("LLM config:")
    print(f"  provider={resolver.config.provider}")
    print(f"  model={resolver.config.model}")
    print(f"  base_url={resolver.config.base_url}")
    print("  api_key=<configured>")

    responder = AgentResponder(llm_resolver=resolver)
    message = Message(
        text=args.text,
        sender_id="llm_user",
        sender_name="LLM用户",
        channel_id="llm_console",
        channel_type=ChannelType(args.channel_type),
    )
    decision = responder.respond(
        message,
        now=datetime(2026, 6, 19, 14, 0, tzinfo=TZ),
    )
    print("Decision:")
    print(f"  action={decision.action.value}")
    print(f"  confidence={decision.confidence}")
    print(f"  should_reply={decision.should_reply}")
    print(f"  reply={decision.reply_text}")
    print(f"  game_id={decision.game_id}")
    print(f"  notes={decision.notes}")
    if decision.game_id:
        game = responder.core.store.games[decision.game_id]
        print("Game:")
        print(f"  type={game.game_type}")
        print(f"  variant={game.variant}")
        print(f"  level={game.level}")
        print(f"  current={game.current_player_count}")
        print(f"  missing={game.missing_count}")
        print(f"  start_at={game.start_at}")
        print(f"  rules={game.rules}")
        print(f"  options={game.play_options}")
        print(f"  status={game.status.value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
