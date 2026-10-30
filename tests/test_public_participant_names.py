"""公开昵称名单只在已关联的私聊局中投影，不能扩大身份或通道权限。"""

from __future__ import annotations

import json

import pytest

from mahjong_agent_runtime import (
    AgentAction,
    AgentContextBuilder,
    AgentRuntime,
    CustomerProfile,
    InMemoryAgentStore,
    InMemoryTraceRecorder,
    SQLiteAgentStore,
    ToolGateway,
    UserMessage,
)
from mahjong_agent_runtime.customer_visible_review import build_reply_self_review_payload
from mahjong_agent_runtime.domains.context_builders.game_context import build_game_context
from mahjong_agent_runtime.group_chat.projections import public_group_game_summary


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    """两种事实源使用独立运行库，测试不修改已有对话或线上观测数据。"""
    value = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "public_names.sqlite3")
    yield value
    if request.param == "sqlite":
        value._connection.close()


def _profile(store, customer_id, public_name):
    """把公开登记与私有显示名故意分开，防止通过显示名回退偷渡备注。"""
    store.upsert_customer(CustomerProfile(
        customer_id=customer_id,
        display_name=f"PRIVATE_DISPLAY_{customer_id}",
        public_name=public_name,
        private_remark=f"PRIVATE_REMARK_{customer_id}",
        notes=f"PRIVATE_NOTES_{customer_id}",
    ))


def _game(store, *, conversation_id="private-current", organizer_id="summer", member_ids=("smile", "kexing")):
    """准备真实占座者，未把角色或确认状态写入客户可见摘要。"""
    game, _ = store.create_game(
        conversation_id=conversation_id,
        organizer_id=organizer_id,
        organizer_name=f"PRIVATE_DISPLAY_{organizer_id}",
        requirement={
            "game_type": "hangzhou_mahjong", "stake": "0.5",
            "start_time_kind": "asap_when_full", "user_visible_summary": "七点三缺一",
        },
        known_players=[
            {"customer_id": customer_id, "display_name": f"PRIVATE_DISPLAY_{customer_id}"}
            for customer_id in (organizer_id, *member_ids)
        ],
        trace_id=f"seed-{organizer_id}",
    )
    return game


def _seed_names(store):
    """昵称来自独立公开字段，不依赖联系人显示名或其他客户的私聊。"""
    for customer_id, public_name in (("summer", "夏日"), ("smile", "笑脸"), ("kexing", "可星")):
        _profile(store, customer_id, public_name)


def _context(store, *, conversation_id="private-current", sender_id="requester", metadata=None, text="哪些人啊"):
    """走实际上下文构建器，由可信通道信封确定名单投影，不解析提问关键词。"""
    message = UserMessage(
        conversation_id=conversation_id, sender_id=sender_id, sender_name="客户",
        text=text, message_id="public-name-question", metadata=metadata or {},
    )
    return message, AgentContextBuilder(store, ToolGateway(store)).build(message, trace_id="public-name-context")


@pytest.mark.parametrize("text", ["哪些人啊", "刚才那句话我打错了"])
def test_related_private_game_projects_registered_names_without_semantic_routing(store, text):
    _seed_names(store)
    game = _game(store)
    message, built = _context(store, text=text)
    summary = built.payload["active_game_visible_summaries"][0]
    assert summary["game_id"] == game.game_id
    assert summary["public_participant_names"] == ["夏日", "笑脸", "可星"]
    assert all(type(name) is str for name in summary["public_participant_names"])
    assert not any(token in json.dumps(summary, ensure_ascii=False) for token in ("PRIVATE_DISPLAY", "PRIVATE_REMARK", "PRIVATE_NOTES"))

    action = AgentAction(goal="回答所问公开名单", objective_status="completed", reasoning_summary="", reply_to_user="夏日、笑脸、可星")
    review = build_reply_self_review_payload(
        message=message, action=action,
        review_items=[{"item_id": "reply_to_user", "recipient_id": message.sender_id, "text": action.reply_to_user}],
        context_payload=built.payload, review_scope="reply_to_user",
    )
    assert review["active_game_visible_summaries"][0]["public_participant_names"] == ["夏日", "笑脸", "可星"]


@pytest.mark.parametrize("public_name", [None, "", "   "])
def test_missing_registered_name_never_falls_back_to_display_remark_or_identifier(store, public_name):
    _profile(store, "summer", public_name)
    _profile(store, "smile", public_name)
    _game(store)  # 可星没有客户登记；即使 Game 中有显示名也不能回退。
    _, built = _context(store)
    assert built.payload["active_game_visible_summaries"][0]["public_participant_names"] == []


@pytest.mark.parametrize("metadata", [
    {"source": "group", "room_id": "room-1", "reply_constraints": {"max_length": 20, "no_private_info": True}},
    {"source": "group"},
    {"room_id": "room-1"},
    {"is_room": True},
    {"reply_constraints": {}},
])
def test_group_or_room_envelope_never_receives_private_roster_permission(store, metadata):
    _seed_names(store)
    game = _game(store)
    _, built = _context(store, metadata=metadata)
    assert len(built.payload["active_game_visible_summaries"]) == 1
    assert "public_participant_names" not in built.payload["active_game_visible_summaries"][0]
    assert "public_participant_names" not in public_group_game_summary(game)


@pytest.mark.parametrize("sender_id", ["summer", "smile"])
def test_organizer_or_effective_seat_holder_can_read_related_roster_in_private(store, sender_id):
    _seed_names(store)
    _game(store, conversation_id="another-private-conversation")
    _, built = _context(store, sender_id=sender_id)
    assert built.payload["active_game_visible_summaries"][0]["public_participant_names"] == ["夏日", "笑脸", "可星"]


def test_unrelated_customer_cannot_read_roster_just_because_profiles_have_public_names(store):
    _seed_names(store)
    _game(store, conversation_id="another-private-conversation")
    _, built = _context(store)
    assert built.payload["active_game_visible_summaries"] == []


@pytest.mark.parametrize("sent", [False, True])
def test_candidate_invitation_is_internal_relatedness_without_new_roster_permission(store, sent):
    _seed_names(store)
    game = _game(store, conversation_id="another-private-conversation")
    drafts, _ = store.create_invite_drafts(
        game_id=game.game_id,
        invitations=[{"customer_id": "requester", "display_name": "客户", "message_text": "七点打吗"}],
        trace_id="candidate-invitation",
    )
    if sent:
        store.update_invite_delivery_status(draft_id=drafts[0].draft_id, status="sent", trace_id="sent-invitation", reason="test")
    _, built = _context(store)
    assert len(built.payload["active_game_visible_summaries"]) == 1
    assert "public_participant_names" not in built.payload["active_game_visible_summaries"][0]


def test_declined_member_is_not_projected_or_granted_roster_permission(store):
    _seed_names(store)
    game = _game(store, conversation_id="another-private-conversation")
    store.record_candidate_reply(
        game_id=game.game_id, customer_id="smile", display_name="笑脸", status="declined", trace_id="decline",
    )
    _, owner_built = _context(store, sender_id="summer")
    assert owner_built.payload["active_game_visible_summaries"][0]["public_participant_names"] == ["夏日", "可星"]
    _, declined_built = _context(store, sender_id="smile")
    assert len(declined_built.payload["active_game_visible_summaries"]) == 1
    assert "public_participant_names" not in declined_built.payload["active_game_visible_summaries"][0]


def test_game_context_without_private_channel_authorization_keeps_roster_absent(store):
    _seed_names(store)
    _game(store)
    context = build_game_context(store, conversation_id="private-current", sender_id="requester")
    assert "public_participant_names" not in context.visible_summaries[0]


def test_main_loop_and_review_receive_the_same_scoped_public_name_evidence(store):
    _seed_names(store)
    _game(store)
    observed = []

    class MainClient:
        def complete(self, messages, **kwargs):
            payload = json.loads(messages[-1]["content"])
            names = payload["active_game_visible_summaries"][0]["public_participant_names"]
            observed.append(("main", names))
            return json.dumps(AgentAction(
                goal="回答名单", objective_status="completed", reasoning_summary="使用受控公开名单",
                reply_to_user="、".join(names),
                stop_reason={"can_stop": True, "why": "已回答所问", "pending_work": [], "depends_on_tool_results": False},
            ).to_dict(), ensure_ascii=False)

    class ReviewClient:
        def complete(self, messages, **kwargs):
            payload = json.loads(messages[-1]["content"])
            names = payload["active_game_visible_summaries"][0]["public_participant_names"]
            observed.append(("review", names))
            return json.dumps({
                "approved": True, "needs_human": False, "reasoning_summary": "仅输出当前相关局公开昵称", "violations": [],
                "item_reviews": [{"item_id": item["item_id"], "approved": True, "suggested_safe_text": item["text"],
                    "reasoning_summary": "已提供受控公开名单", "violations": []} for item in payload["review_items"]],
            }, ensure_ascii=False)

    runtime = AgentRuntime(
        store=store, llm_client=MainClient(), trace_recorder=InMemoryTraceRecorder(),
        reply_self_review_enabled=True, reply_self_review_client=ReviewClient(),
    )
    result = runtime.handle_user_message(UserMessage(
        conversation_id="private-current", sender_id="requester", sender_name="客户",
        text="哪些人啊", message_id="public-name-main-loop",
    ))
    assert result.status == "completed" and result.final_reply == "夏日、笑脸、可星"
    assert observed == [("main", ["夏日", "笑脸", "可星"]), ("review", ["夏日", "笑脸", "可星"])]
