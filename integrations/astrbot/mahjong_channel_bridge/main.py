from __future__ import annotations

import asyncio
import json
import os
import traceback
from datetime import datetime
from typing import Any
from urllib import request

from astrbot.api import logger, star
from astrbot.api.event import AstrMessageEvent, filter


DEFAULT_ENDPOINT = "http://127.0.0.1:8790/api/channels/astrbot/raw"


def _primitive(value: Any) -> Any:
    raw = getattr(value, "value", value)
    if raw is None or isinstance(raw, (str, int, float, bool)):
        return raw
    return value


def _jsonable(value: Any, depth: int = 0) -> Any:
    if depth > 5:
        return repr(value)
    value = _primitive(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item, depth + 1) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item, depth + 1) for key, item in value.items()}
    data: dict[str, Any] = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(value, name)
        except Exception:
            continue
        if callable(attr):
            continue
        if isinstance(attr, (str, int, float, bool, type(None), dict, list, tuple, set)):
            data[name] = _jsonable(attr, depth + 1)
    return data or repr(value)


def _message_payload(event: AstrMessageEvent) -> dict[str, Any]:
    message_obj = getattr(event, "message_obj", None)
    raw_message = getattr(message_obj, "raw_message", None)
    message_id = str(getattr(message_obj, "message_id", "") or "")
    return {
        "captured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "channel": "astrbot",
        "platform_name": str(event.get_platform_name() or ""),
        "platform_id": str(event.get_platform_id() or ""),
        "message_type": str(_primitive(event.get_message_type()) or ""),
        "conversation_id": str(event.unified_msg_origin or event.get_session_id() or ""),
        "session_id": str(event.get_session_id() or ""),
        "group_id": str(event.get_group_id() or ""),
        "sender_id": str(event.get_sender_id() or ""),
        "sender_name": str(event.get_sender_name() or ""),
        "self_id": str(event.get_self_id() or ""),
        "message_id": message_id,
        "source_message_id": message_id,
        "text": str(event.get_message_str() or ""),
        "outline": str(event.get_message_outline() or ""),
        "message_chain": _jsonable(getattr(message_obj, "message", None)),
        "raw_message": _jsonable(raw_message),
        "message_obj": _jsonable(message_obj),
    }


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return {"raw_response": body}
        return parsed if isinstance(parsed, dict) else {"response": parsed}


@star.register(
    "mahjong_channel_bridge",
    "wangjie",
    "Forward AstrBot private/group Weixin messages to Mahjong Agent Runtime.",
    "0.1.0",
)
class MahjongChannelBridge(star.Star):
    def __init__(self, context: star.Context) -> None:
        super().__init__(context)
        self.endpoint = os.getenv("MAHJONG_ASTRBOT_RAW_ENDPOINT", DEFAULT_ENDPOINT)
        self.timeout = float(os.getenv("MAHJONG_ASTRBOT_BRIDGE_TIMEOUT_SECONDS", "3"))
        self.stop_event = os.getenv("MAHJONG_ASTRBOT_BRIDGE_STOP_EVENT", "1").strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        logger.info(f"mahjong_channel_bridge loaded, endpoint={self.endpoint}, stop_event={self.stop_event}")

    @filter.event_message_type(filter.EventMessageType.ALL, priority=10_000_000)
    async def forward_weixin_message(self, event: AstrMessageEvent) -> None:
        payload = _message_payload(event)
        try:
            result = await asyncio.to_thread(_post_json, self.endpoint, payload, self.timeout)
            logger.info(
                f"mahjong_channel_bridge forwarded message_id={payload.get('message_id')} "
                f"trace_id={result.get('trace_id')}"
            )
        except Exception as exc:
            logger.error(f"mahjong_channel_bridge forward failed: {exc!r}\n{traceback.format_exc()}")
        finally:
            if self.stop_event:
                event.stop_event()

    async def terminate(self) -> None:
        logger.info("mahjong_channel_bridge terminated")
