"""Exercise the owner workspace and the same-chat visual observation boundary."""

from __future__ import annotations

import json
from datetime import datetime
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from mahjong_agent_runtime import ToolCall, UserMessage
from mahjong_agent_runtime.group_chat.tool_scope import tool_execution_scope
from mahjong_agent_runtime.models import DEFAULT_TZ
from tests.test_agent_app import local_agent_http, local_wechat_bridge


ACCOUNT = "fixture-media-account"
THREAD = "wxid_fixture_media"
CONVERSATION = f"wechaty:contact:{THREAD}"
STICKER = "fixture-sticker"
VISUAL = {
    "kind": "sticker", "source_message_id": STICKER,
    "description": "一只白猫", "visible_text": "周末见。",
}


def sticker_payload(thread=THREAD, message_id=STICKER):
    """Build the same trusted envelope the native adapter uses for private stickers."""
    return {
        "channel": "agent-wechat", "conversation_id": f"wechaty:contact:{thread}",
        "message_id": message_id, "source_message_id": message_id, "sender_id": thread,
        "sender_name": "合成联系人", "self_message": False, "is_room": False, "message_type": 5,
        "text": "<img>raw-protocol-must-not-enter-context</img>",
        "payload": {"id": message_id, "threadId": thread, "account_id": ACCOUNT,
                    "direction": "incoming", "native_type": 47},
    }


def test_owner_workspace_opens_only_exact_original_and_projects_visual_text(
    local_agent_http, local_wechat_bridge, monkeypatch,
):
    context = local_agent_http
    app = context.app
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_ACCOUNT", ACCOUNT)
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_CONTROL_TOKEN", "test-control-token")
    data = b"GIF89a-test-bytes"
    attempts = []

    def fetch(ref, variant, *, token, max_bytes):
        attempts.append((ref, variant, token))
        assert max_bytes == app.DISPLAY_MAX_BYTES
        return WeChatImage(data, "image/gif", "original", "original")

    from agent_wechat_media import WeChatImage
    monkeypatch.setattr(app, "fetch_wechat_image", fetch)
    api = f"{context.base}/api/wechat/media"
    params = {"thread_id": THREAD, "kind": "direct", "message_id": STICKER, "variant": "original"}
    url = f"{api}?{urlencode(params)}"
    for request, status in (
        (Request(url), 401),
        (Request(url + "&account_id=spoof", headers=context.headers), 400),
        (Request(url + "&kind=direct", headers=context.headers), 400),
        (Request(url.replace("direct", "group"), headers=context.headers), 400),
    ):
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=5)
        assert error.value.code == status
        error.value.close()
    assert attempts == []
    with urlopen(Request(url, headers=context.headers), timeout=5) as response:
        assert response.read() == data
        assert response.headers["Content-Type"] == "image/gif"
        assert response.headers["Cache-Control"] == "private, no-store"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Mahjong-Media-Variant"] == "original"
    ref, variant, token = attempts.pop()
    assert (ref.account_id, ref.thread_id, ref.message_id, variant, token) == (
        ACCOUNT, THREAD, STICKER, "original", "test-control-token",
    )

    with urlopen(f"{context.base}/", timeout=5) as response:
        page = response.read().decode()
    assert 'id="wechatMediaViewer"' in page
    assert "查看原件" in page and "/api/wechat/media?" in page

    stored = {
        "channel": "agent-wechat", "source_message_id": STICKER,
        "conversation_id": CONVERSATION, "payload": {
            "raw_payload": sticker_payload(),
            "route_result": {"audit": {"media_vision": {"status": "recognized", **VISUAL}}},
        },
    }
    context.runtime.store.upsert_channel_observation(stored)
    history = f"{context.base}/api/wechat/conversation?{urlencode({'thread_id': THREAD, 'kind': 'direct'})}"
    native_row = {"id": STICKER, "at": "2026-11-19T15:00:00+08:00", "direction": "inbound",
                  "author": "contact", "speaker": None, "text": "[表情]",
                  "media": {"kind": "sticker", "url": "https://private.invalid/sticker"}}
    local_wechat_bridge.conversation = (200, {
        "ok": True, "id": THREAD, "kind": "direct", "account_id": ACCOUNT,
        "items": [native_row], "nextCursor": None,
    })
    with urlopen(Request(history, headers=context.headers), timeout=5) as response:
        visible = json.load(response)
    assert visible["items"][0]["media"] == {
        "kind": "sticker", "vision": {"description": VISUAL["description"], "visible_text": VISUAL["visible_text"]},
    }
    assert "private.invalid" not in json.dumps(visible)
    assert "account_id" not in visible
    context.runtime.store.upsert_channel_observation({**stored, "conversation_id": "wechaty:contact:other"})
    with urlopen(Request(history, headers=context.headers), timeout=5) as response:
        assert json.load(response)["items"][0]["media"] == {"kind": "sticker"}


def test_private_sticker_reaches_only_same_chat_with_real_owner_turn(local_agent_http, monkeypatch, tmp_path):
    context = local_agent_http
    app = context.app
    policy = tmp_path / "media-policy.json"
    policy.write_text(json.dumps({
        "account_id": ACCOUNT, "managed_direct": True,
        "recipients": [{"thread_id": THREAD, "label": "合成联系人"}],
    }))
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_ACCOUNT", ACCOUNT)
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_OUTBOUND_ALLOWLIST_PATH", str(policy))
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_INPUT_AGGREGATION_ENABLED", "true")
    monkeypatch.setenv("MAHJONG_WECHATY_INPUT_GATE_ENABLED", "true")
    app.persist_wechat_owner_reply(
        context.runtime, channel="agent-wechat", thread_id=THREAD, source_id="owner-message",
        text="发一个表情包吧", occurred_at=datetime.now(DEFAULT_TZ), trace_id="owner-context",
    )
    app.persist_wechat_owner_reply(
        context.runtime, channel="agent-wechat", thread_id="wxid_other", source_id="other-owner-message",
        text="private unrelated owner text", occurred_at=datetime.now(DEFAULT_TZ), trace_id="other-owner-context",
    )
    seen = []

    def recognize(ref, *, trace_id, token):
        seen.append(ref.thread_id)
        return {"status": "recognized", **VISUAL}

    monkeypatch.setattr(app, "recognize_wechat_image", recognize)
    context.client.outputs.append(json.dumps({
        "action": "ignore", "should_route": False, "category": "non_mahjong",
        "confidence": 1, "input_completeness": "complete", "expects_more_fragments": False,
        "missing_information": [], "reasoning_summary": "表情包回应了老板请求", "evidence": [],
    }))

    def post(payload):
        url = f"{context.base}/api/channels/agent-wechat/raw"
        with urlopen(Request(url, data=json.dumps(payload).encode(), headers=context.headers), timeout=5) as response:
            return json.load(response)["route_result"]

    rejected = post(sticker_payload("wxid_other", "other-sticker"))
    assert rejected["observe_only"] is True
    assert seen == [] and context.client.calls == []
    routed = post(sticker_payload())
    assert seen == [THREAD]
    assert routed["input_status"] == "ignored"
    assert routed["audit"]["media_vision"]["description"] == "一只白猫"
    assert routed["agent_result"] is None
    prompt = json.loads(context.client.calls[0]["messages"][-1]["content"])
    assert prompt["current_message"]["metadata"]["visual_observations"] == [VISUAL]
    assert "raw-protocol" not in prompt["current_message"]["text"]
    assert "private unrelated owner text" not in json.dumps(prompt)
    assert prompt["recent_conversation"][-1]["metadata"]["reply_author"] == "owner"
    assert prompt["recent_conversation"][-1]["content"] == "发一个表情包吧"
    assert context.runtime.store.recent_turns(CONVERSATION)[-1].metadata["visual_observations"] == [VISUAL]
    assert context.runtime.store.get_channel_observation("agent-wechat", STICKER)["route_status"] == "analyzed"
    assert post(sticker_payload())["input_status"] == "duplicate"
    assert len(context.runtime.store.recent_turns(CONVERSATION)) == 2
    assert len(context.client.calls) == 1


def test_pure_visual_context_cannot_authorize_booking(local_agent_http):
    context = local_agent_http
    runtime = context.runtime
    message = UserMessage(
        conversation_id=CONVERSATION, sender_id=THREAD, sender_name="合成联系人", text="[客户发了一个表情包]",
        metadata={"_trusted_visual_observation_only": True, "visual_observations": [VISUAL]},
    )
    scope = tool_execution_scope(runtime.store, message)
    assert scope == "visual_read_only"
    call = ToolCall(name="create_game", arguments={
        "organizer_id": THREAD, "organizer_name": "合成联系人",
        "requirement": {"requested_game": "hangzhou_mahjong", "stake": "0.5", "known_player_count": 2},
        "known_players": [],
    })
    result = runtime.tool_gateway.execute(
        call, trace_id="visual-business-scope", conversation_id=CONVERSATION,
        sender_id=THREAD, sender_name="合成联系人", step_index=1, execution_scope=scope,
    )
    assert not result.called and not result.allowed and not runtime.store.games
    assert "请让客户用文字确认" in result.error
    query = runtime.tool_gateway.execute(
        ToolCall(name="search_current_games", arguments={"requirement": {}}),
        trace_id="visual-business-scope", conversation_id=CONVERSATION,
        sender_id=THREAD, sender_name="合成联系人", step_index=2, execution_scope=scope,
    )
    assert query.called and query.allowed
    message_from_api, _ = context.app.build_api_user_message({
        "conversation_id": CONVERSATION, "sender_id": THREAD, "sender_name": "合成联系人",
        "text": "我自己说的话", "metadata": message.metadata,
    })
    assert message_from_api.metadata == {}


def test_media_observation_survives_later_typed_fragment_without_granting_authority():
    from mahjong_agent_runtime import PendingInputBatch, aggregate_pending_input_batch

    media = UserMessage(
        conversation_id=CONVERSATION, sender_id=THREAD, sender_name="合成联系人",
        text="[客户发了一个表情包]", message_id=STICKER,
        metadata={"_trusted_visual_observation_only": True, "visual_observations": [VISUAL]},
    ).to_dict()
    typed = UserMessage(
        conversation_id=CONVERSATION, sender_id=THREAD, sender_name="合成联系人",
        text="我是问图里的时间。", message_id="typed-follow-up",
        metadata={"visual_observations": None},
    ).to_dict()
    batch = PendingInputBatch(batch_id="visual-window", conversation_id=CONVERSATION,
                              sender_id=THREAD, sender_name="合成联系人", fragments=[media])
    pure = aggregate_pending_input_batch(batch, quiet_period_elapsed=True, trigger="fixture")
    assert pure.metadata["_trusted_visual_observation_only"] is True
    batch.fragments.append(typed)
    mixed = aggregate_pending_input_batch(batch, quiet_period_elapsed=True, trigger="fixture")
    assert mixed.metadata["_trusted_visual_observation_only"] is False
    assert mixed.metadata["visual_observations"] == [VISUAL]
    assert mixed.text.endswith("我是问图里的时间。")

    batch.fragments = [{**media, "message_id": "spoofed-identity"}]
    spoofed = aggregate_pending_input_batch(batch, quiet_period_elapsed=True, trigger="fixture")
    assert "visual_observations" not in spoofed.metadata
    assert "_trusted_visual_observation_only" not in spoofed.metadata
