#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pathlib
import sys
from datetime import datetime
from zoneinfo import ZoneInfo


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mahjong_agent import (  # noqa: E402
    AgentResponder,
    AgentRuntime,
    ChannelAddress,
    ChannelType,
    ConsoleInboundSource,
    ConsoleOutboundAdapter,
    CustomerProfile,
    DurableAgentProcessor,
    OpenAICompatibleLLMResolver,
    OutputRouter,
    PlayPreference,
    RuntimeConfig,
    SQLiteDurableStore,
    WeChatTestOutboundAdapter,
    dispatch_pending_outbox,
)


TZ = ZoneInfo("Asia/Shanghai")


def default_agent_timeout_seconds() -> float:
    explicit = os.getenv("MAHJONG_AGENT_TIMEOUT_SECONDS")
    if explicit:
        return float(explicit)
    llm_timeout = os.getenv("MAHJONG_LLM_TIMEOUT_SECONDS")
    if llm_timeout:
        return float(llm_timeout) + 5.0
    return 3.0


def build_responder() -> AgentResponder:
    responder = AgentResponder(
        invite_limit=5,
        llm_resolver=OpenAICompatibleLLMResolver.from_env(),
    )
    for customer in [
        CustomerProfile(
            id="zhang",
            display_name="张哥",
            play_preferences=[
                PlayPreference(
                    game_type="hangzhou_mahjong",
                    preferred_levels=["0.5"],
                    preferred_rulesets=["hangzhou_mahjong"],
                    preferred_variants=["caiqiao"],
                    preferred_play_options=["财敲"],
                ),
                PlayPreference(
                    game_type="sichuan_mahjong",
                    preferred_levels=["1-32"],
                    preferred_rulesets=["sichuan_mahjong"],
                    preferred_play_options=["换三张"],
                ),
            ],
            tags=["杭麻", "川麻", "换三张"],
            smoke_free_preference=True,
            usual_start_hours=[19, 20],
        ),
        CustomerProfile(
            id="amy",
            display_name="Amy",
            preferred_levels=["0.5"],
            tags=["无烟"],
            smoke_free_preference=True,
            usual_start_hours=[17, 18, 19],
        ),
        CustomerProfile(
            id="chen",
            display_name="陈姐",
            preferred_levels=["0.5", "1"],
            tags=["无烟", "熟人局"],
            smoke_free_preference=True,
            usual_start_hours=[17, 18],
        ),
        CustomerProfile(
            id="lin",
            display_name="林姐",
            preferred_levels=["0.5"],
            tags=["财敲", "无烟"],
            smoke_free_preference=True,
            usual_start_hours=[19, 20],
        ),
        CustomerProfile(
            id="ben",
            display_name="Ben",
            preferred_levels=["2"],
            tags=["可吸烟"],
            smoke_free_preference=False,
            usual_start_hours=[20, 21],
            max_games_per_day=2,
            min_hours_between_games=4,
            invite_cooldown_hours=4,
            fatigue_sensitivity=0.8,
        ),
    ]:
        responder.core.upsert_customer(customer)
    return responder


def build_router(args: argparse.Namespace) -> OutputRouter:
    adapters = {
        "console": ConsoleOutboundAdapter(),
        "manual": ConsoleOutboundAdapter(),
        "web_console": ConsoleOutboundAdapter(),
    }
    if args.dispatch == "wechat-test":
        adapters["wechat"] = WeChatTestOutboundAdapter(
            test_recipient_id=args.test_recipient,
            command=[
                sys.executable,
                str(ROOT / "scripts" / "send_wechat_mac.py"),
                "--send",
                "--app-name",
                args.wechat_app_name,
            ]
            if args.wechat_live_send
            else None,
            dry_run=not args.wechat_live_send,
        )
        return OutputRouter(
            adapters=adapters,
            default_channel="console",
            test_redirect=ChannelAddress("wechat", "private", args.test_recipient),
        )
    return OutputRouter(adapters=adapters, default_channel="console")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Console inbound runner with decoupled outbox dispatch.")
    parser.add_argument("--db", default=str(ROOT / "data" / "console_agent.sqlite3"))
    parser.add_argument("--tenant-id", default="default")
    parser.add_argument("--channel-id", default="console_main")
    parser.add_argument("--channel-type", default=ChannelType.MANUAL.value, choices=[item.value for item in ChannelType])
    parser.add_argument("--sender-id", default="operator_console")
    parser.add_argument("--sender-name", default="控制台输入")
    parser.add_argument("--output-channel", default="console", help="Logical output channel stored in outbox.")
    parser.add_argument("--dispatch", choices=["none", "console", "wechat-test"], default="console")
    parser.add_argument("--test-recipient", default="radon_1")
    parser.add_argument("--wechat-live-send", action="store_true", help="Actually send to the test recipient through Mac WeChat UI.")
    parser.add_argument("--wechat-app-name", default="WeChat")
    parser.add_argument("--agent-timeout-seconds", type=float, default=default_agent_timeout_seconds())
    parser.add_argument("--reset", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    store = SQLiteDurableStore(args.db)
    if args.reset:
        store.reset_all()

    processor = DurableAgentProcessor(
        AgentRuntime(
            build_responder(),
            RuntimeConfig(
                log_path=ROOT / "logs" / "console_agent_events.jsonl",
                timeout_seconds=args.agent_timeout_seconds,
            ),
        ),
        store,
    )
    source = ConsoleInboundSource(
        tenant_id=args.tenant_id,
        channel_id=args.channel_id,
        channel_type=ChannelType(args.channel_type),
        sender_id=args.sender_id,
        sender_name=args.sender_name,
    )
    router = build_router(args)

    print("Console Agent started.")
    print("Type a message and press Enter. Commands: :quit, :state, :dispatch, :reset")
    print(f"Inbound: {args.tenant_id}:{args.channel_type}:{args.channel_id}")
    print(f"Dispatch: {args.dispatch}; test_recipient={args.test_recipient}")
    if args.wechat_live_send:
        print("WARNING: live WeChat UI sending is enabled.")

    try:
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                break
            if not text:
                continue
            if text in {":quit", ":q", "exit"}:
                break
            if text == ":reset":
                store.reset_all()
                print("State reset.")
                continue
            if text == ":state":
                print_state(processor)
                continue
            if text == ":dispatch":
                dispatch(store, router)
                continue

            envelope = source.envelope_for_text(
                text,
                metadata={
                    "input_adapter": "console",
                    "output_channel": args.output_channel,
                    "logical_input_channel": args.channel_type,
                },
            )
            result = processor.process(envelope, now=datetime.now(TZ))
            decision = result.runtime_result.decision if result.runtime_result else None
            if decision:
                print(f"action={decision.action.value} confidence={decision.confidence} outbox_created={result.outbox_created}")
                print(decision.reply_text or "<silent>")
            else:
                print("queued: waiting for previous sequence")

            if args.dispatch != "none":
                dispatch(store, router)
    finally:
        processor.shutdown()
    return 0


def dispatch(store: SQLiteDurableStore, router: OutputRouter) -> None:
    dispatched = dispatch_pending_outbox(store, router)
    if not dispatched:
        print("No pending outbox.")
        return
    for message, result in dispatched:
        status = "sent" if result.ok else "failed"
        print(f"outbox {status}: {message.id} via {result.adapter} -> {result.external_id or result.error}")


def print_state(processor: DurableAgentProcessor) -> None:
    snapshot = processor.snapshot()
    durable = snapshot["durable"]
    print("metrics:", snapshot["metrics"])
    print("durable_counts:", durable["counts"])
    print("message_statuses:", durable["message_statuses"])
    print("recent_outbox:")
    for item in durable["outbox"][:10]:
        print(
            f"  {item['id']} {item['status']} {item.get('output_channel')} "
            f"{item['target_type']}:{item['target_id']} {item['message_text'][:80]}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
