from __future__ import annotations

import json
import os
import pathlib
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from zoneinfo import ZoneInfo


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mahjong_agent import (
    AgentResponder,
    AgentRuntime,
    ChannelType,
    CustomerProfile,
    DurableAgentProcessor,
    IncomingEnvelope,
    Message,
    OpenAICompatibleLLMResolver,
    PlayPreference,
    RuntimeConfig,
    SQLiteDurableStore,
)


TZ = ZoneInfo("Asia/Shanghai")


def runtime_timeout_seconds() -> float:
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
            max_games_per_day=1,
            min_hours_between_games=6,
            invite_cooldown_hours=6,
        ),
        CustomerProfile(
            id="chen",
            display_name="陈姐",
            preferred_levels=["0.5", "1"],
            tags=["无烟", "熟人局"],
            smoke_free_preference=True,
            usual_start_hours=[17, 18],
            max_games_per_day=1,
            min_hours_between_games=6,
            invite_cooldown_hours=6,
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


PROCESSOR = DurableAgentProcessor(
    AgentRuntime(
        build_responder(),
        RuntimeConfig(
            log_path=ROOT / "logs" / "agent_events.jsonl",
            timeout_seconds=runtime_timeout_seconds(),
        ),
    ),
    SQLiteDurableStore(ROOT / "data" / "agent_server.sqlite3"),
)


class AgentHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            snapshot = PROCESSOR.snapshot()
            self._json({"ok": True, "runtime": snapshot["metrics"], "durable": snapshot["durable"], "llm": llm_status()})
            return
        self._json({"error": "not_found"}, status=404)

    def do_POST(self) -> None:
        if self.path != "/respond":
            self._json({"error": "not_found"}, status=404)
            return

        try:
            body = self._read_json()
            message = Message(
                text=str(body["text"]),
                sender_id=str(body.get("sender_id", "unknown")),
                sender_name=str(body.get("sender_name", body.get("sender_id", "unknown"))),
                channel_id=str(body.get("channel_id", "manual")),
                channel_type=ChannelType(str(body.get("channel_type", ChannelType.MANUAL.value))),
                metadata=dict(body.get("metadata") or {}),
            )
            now = self._parse_now(body.get("now"))
            envelope = IncomingEnvelope(
                message=message,
                tenant_id=str(body.get("tenant_id", "default")),
                source_message_id=str(body["source_message_id"]) if body.get("source_message_id") else message.id,
                sequence=int(body["sequence"]) if body.get("sequence") is not None else None,
            )
            result = PROCESSOR.process(envelope, now=now)
        except Exception as exc:
            self._json({"error": type(exc).__name__, "message": str(exc)}, status=400)
            return

        self._json(result.to_dict())

    def log_message(self, format: str, *args) -> None:
        print(format % args)

    def _read_json(self) -> dict:
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _parse_now(self, value: object) -> datetime | None:
        if not value:
            return None
        if not isinstance(value, str):
            raise ValueError("now must be an ISO datetime string")
        return datetime.fromisoformat(value).astimezone(TZ)

    def _json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def llm_status() -> dict:
    resolver = PROCESSOR.runtime.responder.llm_resolver
    config = getattr(resolver, "config", None)
    budget_manager = getattr(resolver, "budget_manager", None)
    return {
        "enabled": resolver is not None,
        "model": getattr(config, "model", None),
        "base_url": getattr(config, "base_url", None),
        "budget": budget_manager.snapshot() if budget_manager else None,
    }


def main() -> None:
    server = HTTPServer(("127.0.0.1", 8787), AgentHandler)
    print("Mahjong agent server listening on http://127.0.0.1:8787")
    print("POST /respond with JSON: {\"text\":\"今晚5点 0.5 三缺一 无烟\"}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nMahjong agent server stopped.")
    finally:
        PROCESSOR.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
