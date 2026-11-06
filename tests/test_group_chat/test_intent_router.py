from __future__ import annotations

from mahjong_agent_runtime import InMemoryAgentStore
from mahjong_agent_runtime.group_chat import ChannelIdentity, GroupMessage, L2IntentRouter
from tests.test_group_chat.topic_test_helpers import FixedTopicClassifier


def _message(text: str, *, sender: str = "external-a") -> GroupMessage:
    return GroupMessage(
        room_id="room-1",
        conversation_id="wechaty:room:room-1",
        sender_external_id=sender,
        sender_name="用户A",
        text=text,
        message_id=f"message:{text}",
    )


def _store(*, friend: bool) -> InMemoryAgentStore:
    store = InMemoryAgentStore()
    store.upsert_channel_identity(
        ChannelIdentity(
            channel="wechaty",
            external_user_id="external-a",
            customer_id="customer-a",
            public_name="用户A",
            private_conversation_id="wechaty:contact:external-a",
            can_private_message=friend,
            is_friend=friend,
        )
    )
    return store


def test_simple_query_uses_agent_with_short_public_reply_contract() -> None:
    decision = L2IntentRouter(_store(friend=True), classifier=FixedTopicClassifier("query")).route(_message("3号齐了吗"))

    assert decision.action == "agent_loop"
    assert decision.channel == "group"
    assert decision.reply_constraints.max_length == 20
    assert decision.reply_constraints.no_private_info is True


def test_complex_request_from_friend_switches_to_private() -> None:
    decision = L2IntentRouter(_store(friend=True), classifier=FixedTopicClassifier("delegate"), ack_picker=lambda _: "私聊回你").route(
        _message("帮我约个今晚0.5无烟的")
    )

    assert decision.action == "private_switch"
    assert decision.group_ack == "私聊回你"
    assert decision.switch_context is not None
    assert decision.switch_context.private_conversation_id == "wechaty:contact:external-a"
    assert decision.switch_context.user_original_text == "帮我约个今晚0.5无烟的"


def test_complex_request_from_non_friend_stays_in_group_with_privacy_contract() -> None:
    decision = L2IntentRouter(_store(friend=False), classifier=FixedTopicClassifier("create")).route(_message("帮我组个局"))

    assert decision.action == "agent_loop"
    assert decision.channel == "group"
    assert decision.reply_constraints.max_length == 50
    assert decision.reply_constraints.no_private_info is True


def test_cancel_is_routed_to_agent_with_very_short_reply() -> None:
    decision = L2IntentRouter(_store(friend=True), classifier=FixedTopicClassifier("cancel")).route(_message("我不来了，取消吧"))

    assert decision.action == "agent_loop"
    assert decision.reply_constraints.max_length == 10


def test_unrelated_group_chat_is_ignored() -> None:
    assert L2IntentRouter(_store(friend=True), classifier=FixedTopicClassifier("chitchat")).route(_message("今天下雨了哈哈")).action == "ignore"
