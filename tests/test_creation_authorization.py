"""验证建局前使用已送达的本任务授权以及拒绝后无副作用的工具边界。"""

import json
from datetime import timedelta
from pathlib import Path

import pytest

from mahjong_agent_runtime import (
    AgentRuntime, CustomerProfile, InMemoryAgentStore, InMemoryTraceRecorder, SQLiteAgentStore, ToolCall, UserMessage,
)
from mahjong_agent_runtime.domains.creation_authorization import _evidence, _parse_decision, check_creation_authorization
from mahjong_agent_runtime.models import now, using_clock


_FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "eval" / "golden" /
                       "venue_creation_authorization_replay.json").read_text(encoding="utf-8"))


class Client:
    """按次记录模型请求，给运行时返回明确的测试响应。"""
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    def complete(self, messages, **_):
        self.calls.append(messages)
        return json.dumps(next(self.answers), ensure_ascii=False)


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryAgentStore()
    else:
        db = SQLiteAgentStore(tmp_path / "creation-authorization.sqlite3")
        yield db
        db._connection.close()


def _action(*, tool=None, reply="好", status="completed"):
    return {
        "goal": "按客户已确认的范围处理",
        "objective_status": status,
        "reasoning_summary": "核对本轮实际授权",
        "reply_to_user": reply if tool is None else "",
        "tool_calls": [] if tool is None else [tool],
        "needs_human": False,
        "stop_reason": {"can_stop": tool is None, "depends_on_tool_results": tool is not None,
                        "pending_work": [] if tool is None else ["核对工具结果"], "why": "按工具结果决定"},
    }


def _create():
    start = now() + timedelta(hours=2)
    return {
        "name": "create_game",
        "arguments": {
            "organizer_id": "guest", "organizer_name": "牌友甲",
            "requirement": {"game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "no_smoking",
                            "start_time": start.strftime("%H:%M"), "planned_start_at": start.isoformat(),
                            "duration_hours": 4, "start_time_kind": "scheduled", "duration_kind": "fixed"},
            "known_players": [{"customer_id": "guest", "display_name": "牌友甲"}],
        },
        "reason": "主模型提出建局",
    }


def test_interest_only_model_proposal_is_blocked_before_any_business_write(store):
    tool = _create()
    proposed = _action(tool=tool, status="needs_tool")
    planner = Client([proposed, proposed, _action(reply="那目前先不安排。")])
    authorizer = Client([{"scope": "inquire", "customer_ref": 2, "shop_offer_ref": 1}])
    runtime = AgentRuntime(llm_client=planner, creation_authorization_client=authorizer, store=store)
    message = UserMessage(
        conversation_id="dm", sender_id="guest", sender_name="牌友甲", text=_FIXTURE["customer_inquiry"],
        message_id="interest",
        metadata={"game_creation_authorization": "create"},
    )
    task = runtime.task_context_manager.prepare(message, trace_id="before-inquiry")
    with using_clock(lambda: message.sent_at - timedelta(seconds=1)):
        store.append_assistant_turn("dm", _FIXTURE["delivered_shop_offer"], "interest-offer",
                                    metadata={"delivery_status": "delivered", "task_context_id": task.context.task_context_id})
    result = runtime.handle_user_message(message)
    assert len(authorizer.calls) == 1
    evidence = json.loads(authorizer.calls[0][-1]["content"])["conversation"]
    assert [item["text"] for item in evidence] == [_FIXTURE["delivered_shop_offer"], _FIXTURE["customer_inquiry"]]
    attempts = [item for item in result.tool_results if item.name == "create_game"]
    assert len(attempts) == 2
    assert all(not item.called and not item.allowed and not item.state_transitions for item in attempts)
    assert not store.games and not store.invite_drafts and not store.room_reservations
    assert "create_game_authorization_required" in planner.calls[1][-1]["content"]


def test_explicit_org_consent_still_creates_the_proposed_table(store):
    tool = _create()
    candidate_query = {"name": "search_customers", "reason": "核对组局候选",
                       "arguments": {"game_id": "dynamic", "limit": 6}}
    authorizer = Client([{"scope": "create", "customer_ref": 1, "shop_offer_ref": None}])

    class Planner(Client):
        def complete(self, messages, **kwargs):
            if not self.calls:
                return super().complete(messages, **kwargs)
            context = json.loads(messages[-1]["content"])
            if len(self.calls) == 1:
                created, = [row for row in context["turn_tool_evidence"] if row["name"] == "create_game"]
                candidate_query["arguments"]["game_id"] = created["result"]["game"]["game_id"]
            return super().complete(messages, **kwargs)

    planner = Planner([_action(tool=tool, status="needs_tool"), _action(tool=candidate_query, status="needs_tool"), _action()])
    result = AgentRuntime(llm_client=planner, creation_authorization_client=authorizer, store=store).handle_user_message(
        UserMessage(conversation_id="dm", sender_id="guest", sender_name="牌友甲",
                    text=_FIXTURE["explicit_creation_control"], message_id="direct-create"),
    )
    writes = [item for item in result.tool_results if item.name == "create_game"]
    assert len(writes) == 1 and writes[0].allowed and writes[0].called and writes[0].error is None
    game, = store.games.values()
    assert game.organizer_id == "guest" and game.remaining_seats() == 3
    assert game.planned_end_at - game.planned_start_at == timedelta(hours=4)


def test_evidence_only_contains_same_customer_and_delivered_earlier_shop_offer(store):
    original = "要我帮你开一桌凑人吗？"
    store.append_assistant_turn("dm", original, "sent", metadata={"delivery_status": "delivered"})
    store.append_assistant_turn("dm", "私下草稿", "pending", metadata={"delivery_status": "pending_operator_send"})
    store.append_assistant_turn("other", "另一位顾客的回复", "unrelated", metadata={"delivery_status": "delivered"})
    store.append_user_turn(UserMessage(conversation_id="dm", sender_id="other", sender_name="他人", text="非本人"), "foreign")
    current = UserMessage(conversation_id="dm", sender_id="guest", sender_name="牌友甲", text="行，按这桌组", message_id="accept")
    evidence = _evidence(store, current)
    assert evidence == [{"speaker": "shop", "text": original, "ref": 1},
                        {"speaker": "customer", "text": current.text, "current": True, "ref": 2}]
    assert _parse_decision('{"scope":"create","customer_ref":2,"shop_offer_ref":1}', evidence)["scope"] == "create"
    for bad in (
        '{"scope":"create","customer_ref":null,"shop_offer_ref":null}',
        '{"scope":"create","customer_ref":1,"shop_offer_ref":null}',
        '{"scope":"create","customer_ref":2,"shop_offer_ref":3}',
        '{"scope":"inquire","scope":"create","customer_ref":2,"shop_offer_ref":1}',
    ):
        with pytest.raises(ValueError):
            _parse_decision(bad, evidence)


def test_creation_citation_repairs_visible_reversed_offer_once_without_serving_stale_permission(store):
    at = now()
    current = UserMessage(conversation_id="dm", sender_id="guest", sender_name="牌友甲",
                          text="就我一个先，七点无烟的你帮我组着，我能打满四小时", sent_at=at)
    earlier = UserMessage(conversation_id="dm", sender_id="guest", sender_name="牌友甲",
                          text="七点有无烟吗？", sent_at=at - timedelta(minutes=2))
    store.append_user_turn(earlier, "first")
    with using_clock(lambda: at - timedelta(minutes=1)):
        store.append_assistant_turn("dm", "我帮你组一桌，就你一个人来打吗？", "effective-offer",
                                    metadata={"delivery_status": "delivered"})
    initial = {"scope": "create", "customer_ref": 1, "shop_offer_ref": 2}
    corrected = {"scope": "create", "customer_ref": 3, "shop_offer_ref": 2}
    client = Client([initial, corrected])
    trace = InMemoryTraceRecorder()
    tool = ToolCall(**_create())
    options = {"trace_id": "repair", "step_index": 101, "timeout_seconds": 45, "trace_recorder": trace}
    assert check_creation_authorization(store, client, tool, current, **options) == "create"
    assert len(client.calls) == 2 and client.calls[0][-1] == client.calls[1][-1]
    assert "current=true" in client.calls[1][0]["content"]
    assert [event.step for event in trace.events] == [
        "creation_authorization_prompt", "creation_authorization_invalid",
        "creation_authorization_prompt", "creation_authorization_result",
    ]
    assert not store.games
    assert check_creation_authorization(store, client, tool, current, **options) == "create"
    assert len(client.calls) == 2


@pytest.mark.parametrize("explicit_smoke,visible", [(None, True), ("smoking", False)])
def test_creation_authorizer_only_receives_compatible_same_customer_attendance_as_terms(
    store, monkeypatch, explicit_smoke, visible,
):
    """历史补的是实际操作条件，独立审核仍须引用当次明确组局来话。"""
    history = {"game_type": "hangzhou_mahjong", "stake": "0.5", "smoke_preference": "no_smoking", "completed_games": 3}
    store.upsert_customer(CustomerProfile("guest", "牌友甲", smoke_preference=explicit_smoke, private_remark="不应发送的备注"))
    requested = []

    def observed(_store, customer_id):
        requested.append(customer_id)
        return history

    monkeypatch.setattr(type(store), "completed_game_preference", observed)
    current = UserMessage(conversation_id="dm", sender_id="guest", sender_name="牌友甲", message_id="returning",
                          text="帮我组个19:00的局呗", metadata={"attendance_preference": {"stake": "99"}})
    fixture = _create()
    fixture["arguments"]["requirement"]["stake"] = "0.5"
    tool = ToolCall(**fixture)
    client = Client([{"scope": "create", "customer_ref": None, "shop_offer_ref": None}])
    status = check_creation_authorization(store, client, tool, current, trace_id="learned",
        step_index=1, timeout_seconds=1, trace_recorder=InMemoryTraceRecorder())
    assert requested == ["guest"]
    assert status == "unconfirmed" and not store.games
    sent = client.calls[0][-1]["content"]
    payload = json.loads(sent)
    assert payload.get("attendance_preference") == (history if visible else None)
    assert payload["conversation"] == [{"speaker": "customer", "current": True, "ref": 1, "text": current.text}]
    assert "不应发送的备注" not in sent and '"99"' not in sent


@pytest.mark.parametrize("text,seats", [("帮我组个19:00的局呗", 1), ("帮我组个19:00的局呗，我和搭子两个人。", 2)])
def test_returning_personal_organization_uses_default_or_stated_party(store, monkeypatch, text, seats):
    """直接委托组个人局沿用通常条件和一席，顾客主动说出同行时以实际人数为准。"""
    store.upsert_customer(CustomerProfile("guest", "牌友甲"))
    monkeypatch.setattr(type(store), "completed_game_preference", lambda *_: {
        "game_type": "hangzhou_mahjong", "stake": "0.5", "smoke_preference": "no_smoking", "completed_games": 3,
    })
    call = _create()
    call["arguments"]["requirement"]["stake"] = "0.5"
    call["arguments"]["known_players"] = []
    if seats > 1:
        call["arguments"]["requesting_party"] = {"contact_id": "guest", "contact_name": "牌友甲", "seat_count": seats}
    proposal = _action(tool=call, status="needs_tool")
    candidate_query = {"name": "search_customers", "reason": "按已建的桌查牌友", "arguments": {"game_id": "dynamic"}}

    class Planner(Client):
        def complete(self, messages, **kwargs):
            if len(self.calls) == 1:
                context = json.loads(messages[-1]["content"])
                created, = [row for row in context["turn_tool_evidence"] if row["name"] == "create_game"]
                candidate_query["arguments"]["game_id"] = created["result"]["game"]["game_id"]
            return super().complete(messages, **kwargs)

    planner = Planner([proposal, _action(tool=candidate_query, status="needs_tool"),
                       _action(reply="好，这桌先给你组着。")])
    authorizer = Client([{"scope": "create", "customer_ref": 1, "shop_offer_ref": None}])
    runtime = AgentRuntime(llm_client=planner, creation_authorization_client=authorizer, store=store)
    result = runtime.handle_user_message(UserMessage("dm", "guest", "牌友甲", text, "opening"))
    confirmed, search = result.tool_results
    assert confirmed.called and confirmed.allowed
    assert search.name == "search_customers" and search.called and search.allowed
    game, = store.games.values()
    assert game.remaining_seats() == 4 - seats
    assert game.participants[0].customer_id == "guest" and game.participants[0].seat_count == seats
    assert result.final_reply == "好，这桌先给你组着。"
