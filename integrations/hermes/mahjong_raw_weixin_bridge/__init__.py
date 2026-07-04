"""Hermes plugin: forward raw Weixin gateway messages to Mahjong Agent Runtime."""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime
from urllib import request


DEFAULT_ENDPOINT = "http://127.0.0.1:8790/api/channels/hermes/raw"


def _primitive_value(value):
    raw = getattr(value, "value", value)
    if isinstance(raw, (str, int, float, bool)) or raw is None:
        return raw
    return value


def _platform_name(value) -> str:
    raw = _primitive_value(value)
    return str(raw or "").strip()


def _jsonable(value, depth: int = 0):
    if depth > 5:
        return repr(value)
    value = _primitive_value(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item, depth + 1) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item, depth + 1) for key, item in value.items()}

    data = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(value, name)
        except Exception:
            continue
        if callable(attr):
            continue
        if name in {"source", "text", "message_id", "internal", "attachments", "raw"} or isinstance(
            attr,
            (str, int, float, bool, type(None), dict, list, tuple, set),
        ):
            data[name] = _jsonable(attr, depth + 1)
    return data or repr(value)


def _event_payload(event) -> dict:
    source = getattr(event, "source", None)
    platform = _platform_name(getattr(source, "platform", None) or getattr(event, "platform", None))
    message_id = getattr(event, "message_id", None) or getattr(event, "id", None)
    return {
        "captured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform,
        "text": getattr(event, "text", None),
        "message_id": message_id,
        "source_message_id": message_id,
        "internal": getattr(event, "internal", None),
        "source": _jsonable(source),
        "event": _jsonable(event),
    }


def _post_json(url: str, payload: dict, timeout: float = 3.0) -> dict:
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
            return json.loads(body)
        except json.JSONDecodeError:
            return {"raw_response": body}


def _platform_allowed(platform: str) -> bool:
    allowed = os.getenv("MAHJONG_HERMES_BRIDGE_PLATFORMS", "weixin,wechat")
    wanted = {item.strip().lower() for item in allowed.split(",") if item.strip()}
    if not wanted:
        return True
    return not platform or platform.lower() in wanted


def forward_raw_weixin(event, **kwargs):
    del kwargs
    payload = _event_payload(event)
    if not _platform_allowed(str(payload.get("platform") or "")):
        return None

    endpoint = os.getenv("MAHJONG_HERMES_RAW_ENDPOINT", DEFAULT_ENDPOINT)
    fail_closed = os.getenv("MAHJONG_HERMES_BRIDGE_FAIL_CLOSED", "1").lower() not in {"0", "false", "no"}
    skip_after_forward = os.getenv("MAHJONG_HERMES_BRIDGE_SKIP", "1").lower() not in {"0", "false", "no"}
    try:
        result = _post_json(endpoint, payload)
        print(f"[mahjong-raw-weixin-bridge] forwarded raw message to {endpoint}: {result}")
        if skip_after_forward:
            return {"action": "skip", "reason": "mahjong-raw-weixin-bridge-forwarded"}
        return None
    except Exception as exc:
        print("[mahjong-raw-weixin-bridge] failed to forward raw message:", repr(exc))
        traceback.print_exc()
        if fail_closed:
            return {"action": "skip", "reason": "mahjong-raw-weixin-bridge-forward-failed"}
        return None


def register(ctx) -> None:
    ctx.register_hook("pre_gateway_dispatch", forward_raw_weixin)
    print("[mahjong-raw-weixin-bridge] registered pre_gateway_dispatch hook")
