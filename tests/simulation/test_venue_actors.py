"""模拟角色隔离、合同、模型原文和判定证据。"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from mahjong_agent_runtime.llm import AgentLLMConfig, OpenAICompatibleAgentClient

from . import venue_actors
from .venue_actors import ActorContext, FamiliarShop, MockActor, ModelActor, SemanticJudge, generate_personas, read_personas, write_personas


NOW = "2026-11-25T18:00:00+08:00"
OWNER = "simulation-agent"
ROOM = "test-room"


class _Response:
    status = 200
    headers = {"x-request-id": "local-unit-request"}

    def __init__(self, raw: str) -> None:
        self.raw = raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def read(self):
        return json.dumps({"id": "unit_response", "model": "reply", "choices": [
            {"message": {"content": self.raw}, "finish_reason": "stop"},
        ]}).encode()


def _client(raw: str | tuple[str, ...], captured: list, *, model: str = "actor-test") -> OpenAICompatibleAgentClient:
    responses = iter(raw) if not isinstance(raw, str) else None

    def urlopen(request, **_):
        captured.append(json.loads(request.data))
        return _Response(raw if isinstance(raw, str) else next(responses))

    return OpenAICompatibleAgentClient(
        config=AgentLLMConfig(api_key="test-only", model=model, base_url="https://example.invalid/v1", retry_attempts=1),
        urlopen=urlopen,
    )


def _event(kind: str, event_id: str | None = None, **details) -> dict:
    return {"kind": kind, "event_id": event_id or kind, "at": NOW, "details": details}


def test_500_personas_are_reproducible_with_separate_identities_and_directed_edges(tmp_path: Path):
    cast = generate_personas(73)
    other_cast = generate_personas(74)
    assert cast == generate_personas(73)
    assert cast != other_cast
    assert [person.persona_id for person in cast] == [person.persona_id for person in other_cast]
    assert len(cast) == 500
    assert len({person.name for person in cast}) == 500
    assert len({identity for person in cast for identity in (
        person.persona_id, person.sender_id, person.customer_id, person.conversation_id,
    )}) == 2000
    assert {person.activity for person in cast} == {"frequent", "occasional", "quiet"}
    people = {person.persona_id: person for person in cast}
    for person in cast:
        assert {edge.kind for edge in person.relationships} == {"regular_partner", "avoid_same_table", "neutral"}
        assert all(edge.target_id != person.persona_id and edge.target_name == people[edge.target_id].name
                   for edge in person.relationships)
        avoid = next(edge for edge in person.relationships if edge.kind == "avoid_same_table")
        assert not any(edge.target_id == person.persona_id and edge.kind == "avoid_same_table"
                       for edge in people[avoid.target_id].relationships)

    store = tmp_path / "agent.sqlite3"
    store.write_bytes(b"existing simulator state")
    path = tmp_path / "personas.json"
    write_personas(path, cast)
    assert read_personas(path) == cast
    saved = json.loads(path.read_text())
    assert saved["schema_version"] == 1
    assert len(saved["personas"]) == len(cast)
    assert saved["personas"][0]["relationships"][0]["target_id"] == cast[0].relationships[0].target_id
    assert store.read_bytes() == b"existing simulator state"
    assert set(tmp_path.iterdir()) == {path, store}


def test_single_person_and_unusable_count():
    assert generate_personas(19, count=1)[0].relationships == ()
    with pytest.raises(ValueError, match="positive integer"):
        generate_personas(19, count=0)


def test_cast_styles_and_behaviors_are_distinct_and_affect_actual_outbound_text():
    cast = generate_personas(42)
    assert len({person.speech_profile for person in cast}) == 500
    assert len({person.speaking_style for person in cast}) == 500
    assert len({person.temperament for person in cast}) == 500
    mock = MockActor(42)
    terms = _event("opportunity", start_time="19:00", game_type="杭州麻将", stake=1, duration_hours=4, smoke="non_smoking")
    messages = []
    for person in cast:
        compatible = replace(person, preferred_game="杭州麻将", stake_min=0.5, stake_max=2, smoking="non_smoking")
        context = ActorContext(compatible, NOW, events=(terms,))
        action = mock.decide(context, trace_id="same-conditions").action
        assert action and action.action == "send" and "有符合条件的直接算我一个" in action.text
        assert person.name not in action.text and person.persona_id not in action.text
        messages.append(action.text)
    assert len(set(messages)) == 500


def test_opening_disclosure_varies_by_seed_and_is_an_actor_only_tendency():
    cast = generate_personas(42)
    tendencies = tuple(person.temperament.opening_disclosure for person in cast)
    assert set(tendencies) == {"main_need_first", "key_conditions_when_useful", "share_context_upfront"}
    assert tendencies == tuple(person.temperament.opening_disclosure for person in generate_personas(42))
    assert tendencies != tuple(person.temperament.opening_disclosure for person in generate_personas(43))
    counts = Counter(tendencies)
    assert counts["main_need_first"] > counts["key_conditions_when_useful"] + counts["share_context_upfront"]

    person = cast[0]
    event = _event("session_open", case_goal="try_join", start_time="19:00", game_type="四川麻将",
                   stake=1, duration_hours=3, party_size=2, smoke="non_smoking")
    responses = (
        ("main_need_first", {"action": "send", "text": "七点还有位吗？"}),
        ("key_conditions_when_useful", {"action": "send", "text": "七点有川麻吗？我俩"}),
        ("share_context_upfront", {"action": "send", "text": "七点我俩想打无烟川麻，1块底有位不"}),
        ("main_need_first", {"action": "send", "text": "川麻一块底的，无烟，七点俩人能上吗"}),
    )
    raw = tuple(json.dumps(action, ensure_ascii=False) for _, action in responses)
    captured = []
    model = ModelActor(_client(raw, captured))
    for (tendency, action), original in zip(responses, raw, strict=True):
        actor = replace(person, temperament=replace(person.temperament, opening_disclosure=tendency))
        answer = model.decide(ActorContext(actor, NOW, events=(event,)), trace_id=f"opening:{tendency}")
        assert answer.context["persona"]["temperament"]["opening_disclosure"] == tendency
        assert answer.context["events"][0]["details"] == event["details"]
        assert answer.raw == original and answer.response_source["http_response_complete"]
        assert answer.action.to_dict() == {**action, "kind": "private", "recipient_id": OWNER}
    assert all("没有固定逐轮补条件的流程" in request["messages"][0]["content"] for request in captured)
    system = [request["messages"][0]["content"] for request in captured]
    assert system[0].startswith("你刚要开启这段聊天")
    assert system[1].startswith("你刚要开口。先决定最想")
    assert system[2].startswith("你刚要开口。你有时会把想清楚")


@pytest.mark.parametrize(("text", "kind", "recipient"), [
    ("老板，群里1号那桌是川麻吧？我还想问下今晚七点后有没一块的红中。", "private", OWNER),
    ("这桌还有位置嘛", "group", ROOM),
])
def test_actor_can_anchor_cross_channel_ask_or_keep_a_shared_reference_short(text, kind, recipient):
    """群发可带已见引用，私聊用正文辨认原桌，角色原话和渠道由模型决定。"""
    person = generate_personas(42, count=1)[0]
    board = {"message_id": "board", "kind": "group", "sender_id": OWNER, "recipient_id": ROOM,
             "sent_at": NOW, "text": "当前缺人局：\n1、人齐开 川麻 2～3元 无烟 173"}
    hidden = {**board, "message_id": "hidden", "recipient_id": "another-room", "text": "别群的同号桌"}
    event = {**_event("delivered"), "caused_by": "board"}
    action = {"action": "send", "text": text, "kind": kind, "recipient_id": recipient}
    if kind == "group":
        action["quote_message_id"] = "board"
    captured = []
    result = ModelActor(_client(json.dumps(action, ensure_ascii=False), captured)).decide(
        ActorContext(person, NOW, history=(board, hidden), events=(event,), group_targets=(ROOM,)), trace_id="board-message",
    )
    assert result.action.to_dict() == action
    system, request = captured[0]["messages"]
    assert "收件人刚才看到了什么" in system["content"] and "两件事讲清楚" in system["content"]
    projected = json.loads(request["content"])
    assert projected["delivery"]["message_id"] == "board"
    assert [row["message_id"] for row in projected["history"]] == ["board"]


@pytest.mark.parametrize(("kind", "condition", "new_event", "text"), [
    ("delivered", "ordinary_day", False, "那麻烦帮我问问呗"),
    ("delivered", "en_route_with_traffic_delay", False, "能留位的话告诉我一声"),
    ("traffic", "en_route_with_traffic_delay", True, "我堵路上了，还能等十来分钟不"),
])
def test_recent_arrival_uses_visible_own_message_and_actual_world_event(kind, condition, new_event, text):
    """当前可达下限和自己刚说过的时间同时可见，普通回复不会被标成行程变化。"""
    person = generate_personas(42, count=1)[0]
    known = {"message_id": "my-arrival", "kind": "private", "sender_id": person.sender_id,
             "recipient_id": OWNER, "sent_at": "2026-11-25T10:00:00+00:00", "text": "我八点三刻左右到。"}
    shop = {"message_id": "shop-reply", "kind": "private", "sender_id": OWNER, "recipient_id": person.sender_id,
            "sent_at": "2026-11-25T18:00:30+08:00", "text": "这会儿没座，要我帮你问问吗？"}
    public = {**known, "message_id": "group-post", "kind": "group", "recipient_id": ROOM,
              "sent_at": "2026-11-25T18:01:00+08:00", "text": "群里有人吗？"}
    later = "2026-11-25T18:02:00+08:00"
    event = _event(kind, earliest_arrival="2026-11-25T19:43:00+08:00", can_arrive=True, event_condition=condition)
    if kind == "delivered":
        event["caused_by"] = "shop-reply"
    raw = json.dumps({"action": "send", "text": text}, ensure_ascii=False)
    captured = []
    action = ModelActor(_client(raw, captured)).decide(ActorContext(person, later, history=(known, shop, public),
                                                events=(event,), group_targets=(ROOM,)), trace_id="travel-continuation")
    context = action.context
    expected = {"new_travel_event": new_event, "previous_shop_message_id": "my-arrival",
                "minutes_since_previous_message": 2}
    details = dict(event["details"])
    prefix = "你是麻将馆"
    if not new_event:
        expected["earliest_possible_if_leaving_now"] = details.pop("earliest_arrival")
        prefix = "这轮是接店家刚刚发来的私聊"
    assert context["travel_update"] == expected
    assert context["events"][0]["details"] == details
    assert captured[0]["messages"][0]["content"].startswith(prefix)
    assert action.raw == raw and action.action.text == text


def test_returning_actor_can_use_personal_private_shop_receipt_without_repeating_its_contents():
    person = generate_personas(42, count=1)[0]
    person = replace(person, temperament=replace(person.temperament, opening_disclosure="main_need_first"))
    receipt = {"message_id": "past-checkout", "kind": "private", "sender_id": OWNER,
               "recipient_id": person.sender_id, "sent_at": "2026-11-24T21:00:00+08:00",
               "text": "今天一块底的无烟杭麻收桌啦，下回见。"}
    terms = {"game_type": "hangzhou_mahjong", "stake": 1, "smoke_preference": "no_smoking"}
    context = ActorContext(person, NOW, history=(receipt,),
        events=(_event("session_open", case_goal="organize_new", start_time="19:00", knows_shopkeeper=True),),
        familiar_shop=FamiliarShop("past-checkout", terms | {"raw_customer_profile": "PRIVATE_PROFILE"}))
    captured = []
    raw = '{"action":"send","text":"帮我组个19:00的局呗"}'
    decision = ModelActor(_client(raw, captured)).decide(context, trace_id="returning-customer")
    role_request = json.loads(captured[0]["messages"][-1]["content"])
    assert captured[0]["messages"][0]["content"].startswith("这次是你主动开启一段新私聊")
    assert context.to_dict()["familiar_shop"]["usual_preferences"] == terms
    assert role_request["familiar_shop"] == {
        "acknowledgement_message_id": "past-checkout", "shared_condition_fields": sorted(terms),
    }
    assert role_request["history"][0]["message_id"] == "past-checkout" and "text" not in role_request["history"][0]
    assert "preferred_game" not in role_request["persona"] and "stake_min" not in role_request["persona"]
    assert decision.context == role_request
    assert "PRIVATE_PROFILE" not in str(role_request)
    assert role_request["delivery"] is None
    assert decision.raw == raw and decision.response_source["http_response_complete"]
    assert decision.action.to_dict() == {"action": "send", "text": "帮我组个19:00的局呗", "kind": "private", "recipient_id": OWNER}


@pytest.mark.parametrize("changed", [
    {"sender_id": "other-player"}, {"kind": "group"}, {"recipient_id": "someone-else"},
    {"sent_at": "2026-11-26T21:00:00+08:00"}, {"visible_to": ["someone-else"]},
    {"message_id": "different-receipt"}, {"sent_at": "2026-11-24T21:00:00"},
])
def test_shop_friendship_alone_or_inaccessible_receipt_does_not_supply_usual_preferences(changed):
    person = generate_personas(42, count=1)[0]
    receipt = {"message_id": "past-checkout", "kind": "private", "sender_id": OWNER,
               "recipient_id": person.sender_id, "sent_at": "2026-11-24T21:00:00+08:00", "text": "今天结账了。"}
    context = ActorContext(person, NOW, history=(receipt | changed,),
        events=(_event("session_open", case_goal="organize_new", knows_shopkeeper=True),),
        familiar_shop=FamiliarShop("past-checkout", {"game_type": "hangzhou_mahjong"}))
    assert "familiar_shop" not in context.to_dict()


def test_background_chat_reacts_to_personal_intent_and_the_current_time():
    person = generate_personas(42, count=1)[0]
    person = replace(person, activity="frequent", available_periods=("午后",),
                     temperament=replace(person.temperament, purpose="bring_friends"))
    event = _event("activity", case_goal="group")
    action = MockActor().decide(ActorContext(person, NOW, events=(event,), group_targets=(ROOM,)),
                                trace_id="social").action
    assert action.kind == "group" and action.recipient_id == ROOM
    assert "明天" in action.text and "朋友" in action.text and "俩人" in action.text


def test_actor_model_receives_only_the_actors_own_visible_projection_and_verifiable_output():
    person, peer = generate_personas(73, count=2)
    captured = []
    raw = '{"action":"send","text":"今晚还有位吗？"}'
    model = ModelActor(_client(raw, captured))
    context = ActorContext(person, NOW, group_targets=(ROOM,), history=(
        {"kind": "private", "sender_id": OWNER, "recipient_id": peer.sender_id, "text": "PEER_DM"},
        {"kind": "private", "sender_id": OWNER, "recipient_id": person.sender_id, "text": "只给我的"},
        {"kind": "group", "sender_id": peer.sender_id, "recipient_id": ROOM, "text": "群里能看到"},
        {"kind": "group", "sender_id": person.sender_id, "recipient_id": ROOM, "text": "我此前发过的"},
        {"kind": "group", "sender_id": peer.sender_id, "recipient_id": "another-room", "text": "OTHER_ROOM"},
        {"kind": "group", "sender_id": person.sender_id, "recipient_id": "another-room", "text": "OTHER_ROOM_OWN"},
    ), events=(
        {**_event("need"), "visible_to": [peer.persona_id], "text": "PEER_NEED"},
        {**_event("opportunity", game_type=person.preferred_game, attendees=["UNSEEN_NAME"], internal="INTERNAL_NOTE",
                  known_attendees=[{"name": peer.name, "persona_id": peer.persona_id, "internal": "UNSEEN_ATTRIBUTE"}]),
         "internal": "INTERNAL_STATE"},
    ))
    result = model.decide(context, trace_id="actor-unit")
    payload = json.loads(captured[0]["messages"][-1]["content"])
    assert payload == result.context
    assert "PEER_DM" not in str(payload) and "PEER_NEED" not in str(payload) and "OTHER_ROOM" not in str(payload)
    assert "UNSEEN_NAME" not in str(payload) and "UNSEEN_ATTRIBUTE" not in str(payload)
    assert "INTERNAL_NOTE" not in str(payload) and "INTERNAL_STATE" not in str(payload)
    assert person.customer_id not in str(payload) and person.conversation_id not in str(payload)
    assert [(row["text"], row["from_self"]) for row in payload["history"]] == [
        ("只给我的", False), ("群里能看到", False), ("我此前发过的", True),
    ]
    assert payload["events"][0]["details"]["known_attendees"] == [{"name": peer.name, "persona_id": peer.persona_id}]
    assert result.action.to_dict() == {"action": "send", "text": "今晚还有位吗？", "kind": "private", "recipient_id": OWNER}
    assert result.status == "valid" and result.raw == raw
    assert result.response_source["trace_id"] == "actor-unit"
    assert result.response_source["http_response_complete"] is True
    assert result.response_source["raw_response_sha256"] == sha256(raw.encode()).hexdigest()


@pytest.mark.parametrize("raw", [
    "not JSON", '{"action":"send","text":" "}', '{"action":"send","text":"你好","recipient_id":"external"}',
    '{"action":"send","text":"你好","kind":"group"}', '{"action":"arrive","recipient_id":"peer"}',
    '{"action":"arrive","action":"wait"}', '{"action":["wait"]}',
    json.dumps({"action": "send", "text": "老板\x00你好"}),
])
def test_invalid_role_actions_are_retained_and_never_executable(raw):
    result = ModelActor(_client(raw, [])).decide(ActorContext(generate_personas(5, count=1)[0], NOW), trace_id="bad-action")
    assert result.status == "invalid"
    assert result.to_dict()["action"] is None
    assert result.raw == raw and result.error
    assert result.response_source["http_response_complete"] is True


def test_role_can_separate_two_questions_with_a_line_break_in_one_message():
    text = "老板，群里那桌财敲还有位吗？\n我七点后有空，有没有一块底的红中？"
    raw = json.dumps({"action": "send", "text": text}, ensure_ascii=False)
    context = ActorContext(generate_personas(5, count=1)[0], NOW)
    result = ModelActor(_client(raw, [])).decide(context, trace_id="role-multiline-message")
    assert result.status == "valid" and result.action.text == text
    assert result.raw == raw


@pytest.fixture
def group_quote_context():
    """同群、跨群和私聊都存在于输入周围，只有同群已见消息可引用。"""
    person, peer = generate_personas(43, count=2)
    row = {"kind": "group", "sender_id": peer.sender_id, "recipient_id": ROOM, "text": "九点那桌缺一", "sent_at": NOW}
    return ActorContext(person, NOW, group_targets=(ROOM, "other-group"), history=(
        {**row, "message_id": "same-room"},
        {**row, "message_id": "second-same-room", "quoted_message_id": "missing-original"},
        {**row, "message_id": "filtered", "visible_to": [peer.persona_id]},
        {**row, "message_id": "other-room", "recipient_id": "other-group"},
        {**row, "message_id": "visible-private", "kind": "private", "recipient_id": person.sender_id},
        {**row, "message_id": "future", "sent_at": "2026-11-25T18:01:00+08:00"},
    ), events=({**row, "message_id": "event-only"},))


def test_live_actor_accepts_visible_quote_for_same_target_group(group_quote_context):
    raw = json.dumps({"action": "send", "text": "底注多少？", "kind": "group", "recipient_id": ROOM,
                      "quote_message_id": "same-room"})
    captured = []
    result = ModelActor(_client(raw, captured)).decide(group_quote_context, trace_id="same-room-quote")
    assert result.status == "valid" and result.action.to_dict() == json.loads(raw)
    assert "quote_message_id" in captured[0]["messages"][0]["content"]


@pytest.mark.parametrize(("quote", "kind", "recipient"), [
    ("filtered", "group", ROOM), ("other-room", "group", ROOM), ("visible-private", "group", ROOM),
    ("event-only", "group", ROOM), ("missing-original", "group", ROOM), ("future", "group", ROOM),
    ("same-room", "private", OWNER), (None, "group", ROOM), ("", "group", ROOM),
])
def test_live_actor_rejects_undelivered_and_out_of_scope_quotes(group_quote_context, quote, kind, recipient):
    raw = json.dumps({"action": "send", "text": "底注多少？", "kind": kind, "recipient_id": recipient,
                      "quote_message_id": quote})
    captured = []
    result = ModelActor(_client(raw, captured), repair_schema=True).decide(group_quote_context, trace_id="bad-quote")
    assert result.status == "invalid" and result.action is None and len(captured) == 1
    assert result.raw == raw and "visible in the target group" in result.error


@pytest.mark.parametrize(("first_quote", "repaired_quote", "status"), [
    ("same-room", "same-room", "valid"), ("same-room", "second-same-room", "invalid"),
    ("same-room", None, "invalid"), (None, "same-room", "invalid"),
])
def test_live_actor_schema_repair_preserves_quote(group_quote_context, first_quote, repaired_quote, status):
    message = {"action": "send", "text": "底注多少？", "kind": "group", "recipient_id": ROOM}
    first = {"send": message | ({"quote_message_id": first_quote} if first_quote else {})}
    repaired = message | ({"quote_message_id": repaired_quote} if repaired_quote else {})
    captured = []
    result = ModelActor(_client((json.dumps(first), json.dumps(repaired)), captured), repair_schema=True).decide(
        group_quote_context, trace_id="quote-repair",
    )
    assert result.status == status and len(captured) == 2
    assert len(result.prior_attempts) == 1
    if status == "valid":
        assert result.action.to_dict() == repaired
    else:
        assert result.action is None and "changed the original actor decision" in result.error


def test_live_actor_repairs_wrapped_send_using_same_intent_and_remaining_budget(monkeypatch):
    """修正必须来自第二次模型响应，保留原轮正文、收件人和两次可核验证据。"""
    captured = []
    first = '{"send":{"action":"send_message","text":"无烟的还有位吗？","kind":"private","recipient_id":"simulation-agent"}}'
    corrected = '{"action":"send","text":"无烟的还有位吗？","kind":"private","recipient_id":"simulation-agent"}'
    clock = iter((1.0, 6.0))
    monkeypatch.setattr(venue_actors, "monotonic", lambda: next(clock))
    actor = ModelActor(_client((first, corrected), captured), timeout_seconds=10, repair_schema=True)
    context = ActorContext(generate_personas(43, count=1)[0], NOW, events=(_event("session_open"),))
    result = actor.decide(context, trace_id="shape-retry")

    assert result.status == "valid" and result.raw == corrected
    assert result.action.text == "无烟的还有位吗？"
    assert result.response_source["step_index"] == 2
    assert len(captured) == 2
    assert captured[1]["messages"][:2] == captured[0]["messages"]
    assert json.loads(captured[1]["messages"][-1]["content"])["previous_output"] == first
    prior, = result.to_dict()["prior_attempts"]
    assert prior["status"] == "invalid" and prior["raw"] == first
    assert prior["response_source"]["step_index"] == 1 and prior["response_source"]["http_response_complete"]
    assert prior["diagnostics"]["attempts"][0]["timeout_seconds"] == pytest.approx(10, abs=0.1)
    assert result.diagnostics["attempts"][0]["timeout_seconds"] == pytest.approx(5, abs=0.1)


@pytest.mark.parametrize(("first", "second", "calls"), [
    ('{"send":{"text":"无烟的还有位吗？","recipient_id":"external"}}', "", 1),
    ('{"send":{"text":"无烟的还有位吗？"}}', '{"action":"send","text":"直接帮我订房"}', 2),
    ('{"send":{"text":"无烟的还有位吗？"}}', '{"action":"wait"}', 2),
    ('{"send":{"text":"无烟的还有位吗？"}}', '{"send":{"text":"无烟的还有位吗？"}}', 2),
])
def test_live_actor_repair_never_changes_audience_text_or_intent_and_cannot_repeat(first, second, calls):
    """结构修正最多执行一次，外部收件人和意思变更始终无法执行。"""
    captured = []
    actor = ModelActor(_client((first, second), captured), repair_schema=True)
    result = actor.decide(ActorContext(generate_personas(43, count=1)[0], NOW), trace_id="rejected-repair")
    assert result.status == "invalid" and result.action is None and len(captured) == calls
    assert len(result.prior_attempts) == calls - 1


def test_live_actor_cannot_repair_after_its_model_budget(monkeypatch):
    """首次请求用完角色整轮预算后不发起修正。"""
    captured = []
    clock = iter((1.0, 36.0))
    monkeypatch.setattr(venue_actors, "monotonic", lambda: next(clock))
    actor = ModelActor(_client('{"send":{"text":"无烟的还有位吗？"}}', captured), repair_schema=True)
    result = actor.decide(ActorContext(generate_personas(43, count=1)[0], NOW), trace_id="expired-repair")
    assert result.status == "invalid" and result.action is None and len(captured) == 1


@pytest.mark.parametrize(("draft", "claim", "can_arrive", "earliest"), [
    ("今晚七点半左右到", {"kind": "arrival", "quote": "七点半左右到", "local_at": "2026-11-25T19:30:00+08:00"},
     False, None),
    ("我大概能打到十一点", {"kind": "play_until", "quote": "能打到十一点", "local_at": "2026-11-25T23:00:00+08:00"},
     True, "2026-11-25T19:31:00+08:00"),
    ("晚上红中有合适的吗，我打四五个钟头", {"kind": "play_duration", "quote": "我打四五个钟头", "minimum_hours": 4},
     True, "2026-11-25T19:12:00+08:00"),
])
def test_live_actor_independently_checks_and_replans_contradictory_owned_schedule_claims(draft, claim, can_arrive, earliest):
    """确认工作、离场和时长三类冲突在真实发言前被发现，只使用角色二次决定的话。"""
    context = ActorContext(generate_personas(43, count=1)[0], NOW, events=(_event("session_open",
        can_arrive=can_arrive, earliest_arrival=earliest, free_from="2026-11-25T09:00:00+08:00",
        free_until="2026-11-25T22:00:00+08:00" if can_arrive else "2026-11-25T14:30:00+08:00"),))
    proposal = json.dumps({"action": "send", "text": draft}, ensure_ascii=False)
    extraction = json.dumps({"claims": [{"local_date": "2026-11-25", **claim}]}, ensure_ascii=False)
    corrected = '{"action":"send","text":"今天不来了，明天再问你。"}'
    captured = []
    result = ModelActor(_client((proposal, extraction, corrected, '{"claims":[]}'), captured)).decide(
        context, trace_id="actor-world-claim")

    assert len(captured) == 4
    assert result.action.text == "今天不来了，明天再问你。" and result.raw == corrected
    assert result.action.to_dict().keys() == {"action", "text", "kind", "recipient_id"}
    refused, = result.prior_attempts
    assert refused["status"] == "invalid" and refused["raw"] == proposal
    for step, (review, expected_raw) in enumerate(zip(result.availability_reviews, (extraction, '{"claims":[]}')), 1):
        assert review["raw"] == expected_raw and review["response_source"]["http_response_complete"]
        assert review["step_index"] == step * 2
    assert result.availability_reviews[0]["conflict"] and not result.availability_reviews[1]["conflict"]
    extraction_payload = json.loads(captured[1]["messages"][-1]["content"])
    assert extraction_payload == {"draft": draft, "local_now": NOW, "received_text": None}
    assert result.response_source["http_response_complete"] and result.response_source["step_index"] == 3


@pytest.mark.parametrize("raw", [
    '{}',
    '{"claims":[{"kind":"visit","quote":"未出现的文字","local_date":"2026-11-25"}]}',
    '{"claims":[{"kind":"arrival","quote":"七点半左右到","local_date":"2026-11-26","local_at":"2026-11-25T19:30:00+08:00"}]}',
    '{"claims":[{"kind":"venue_start","quote":"七点半左右到","local_date":"2026-11-25"}]}',
    '{"claims":[{"kind":"visit","quote":"七点半左右到","local_date":"2026-11-25","approximate":true}]}',
    '{"claims":[{"kind":"play_duration","quote":"七点半左右到","local_date":"2026-11-25",'
    '"minimum_hours":4,"approximate":"true"}]}',
    '{"claims":[{"kind":"play_duration","quote":"七点半左右到","local_date":"2026-11-25","approximate":true}]}',
])
def test_live_actor_withholds_risky_draft_when_independent_claim_evidence_is_invalid(raw):
    """缺少提取结果、捏造引文、改日期或误用角色不能当作通过。"""
    captured = []
    context = ActorContext(generate_personas(43, count=1)[0], NOW, events=(_event("session_open",
        can_arrive=False, free_from="2026-11-25T09:00:00+08:00", free_until="2026-11-25T14:30:00+08:00"),))
    result = ModelActor(_client(('{"action":"send","text":"我今晚七点半左右到"}', raw, raw), captured)).decide(
        context, trace_id="bad-availability-extraction")

    assert len(captured) == 3 and result.status == "invalid" and result.action is None
    assert [review["error"] for review in result.availability_reviews] == [result.error, result.error]


@pytest.mark.parametrize("free_until, fits", [("23:00", False), ("23:30", True)])
def test_archived_approximate_duration_marker_keeps_minimum_and_world_check(free_until, fits):
    """真实返回中的约数标记不触发格式重试，也不会给四小时安排增加可用时间。"""
    now = "2026-11-25T19:00:00+08:00"
    context = ActorContext(generate_personas(43, count=1)[0], now, events=(_event("session_open",
        can_arrive=True, earliest_arrival="2026-11-25T19:09:00+08:00", free_from="2026-11-25T09:30:00+08:00",
        free_until=f"2026-11-25T{free_until}:00+08:00"),))
    action = ('{"action":"send","text":"老板晚上好，现在这个点还有无烟的四川麻将桌能搭一桌吗，'
              '2块或3块的底都行，我大概能打四个小时，能搭一桌吗？","kind":"private","recipient_id":"simulation-agent"}')
    claims = ('{"claims":[{"kind":"play_duration","quote":"我大概能打四个小时","local_date":"2026-11-25",'
              '"minimum_hours":4,"approximate":true}]}')
    captured = []

    outcome = ModelActor(_client((action, claims, '{"action":"wait"}'), captured)).decide(
        context, trace_id="archived-rolling-0103-approximate-duration")

    review, = outcome.availability_reviews
    assert review["raw"] == claims and "error" not in review
    assert review["claims"][0]["minimum_hours"] == 4 and review["response_source"]["http_response_complete"]
    if fits:
        assert len(captured) == 2 and review["conflict"] is None
        assert outcome.action.text == json.loads(action)["text"]
    else:
        assert len(captured) == 3 and "最少打牌时长" in review["conflict"]
        assert outcome.action.action == "wait" and outcome.prior_attempts[0]["raw"] == action


@pytest.mark.parametrize("can_arrive", [True, False])
def test_live_actor_rechecks_same_draft_after_extractor_mistakes_duration_cap_for_minimum(can_arrive):
    """归档中的时长上限不能阻断有效来话，也不能掩盖同一来话的实际可达性冲突。"""
    context = ActorContext(generate_personas(43, count=1)[0], NOW, events=(_event("session_open",
        can_arrive=can_arrive, free_from="2026-11-25T09:00:00+08:00", free_until="2026-11-25T23:00:00+08:00"),))
    draft = "老板好 七点半那桌 371 0.5 的还能搭一桌吗 我到那边得八点左右 最多打三小时"
    proposal = json.dumps({"action": "send", "text": draft}, ensure_ascii=False)
    arrival = {"kind": "arrival", "quote": "我到那边得八点左右", "local_date": "2026-11-25",
               "local_at": "2026-11-25T20:00:00+08:00", "approximate": True}
    malformed = json.dumps({"claims": [arrival, {"kind": "play_duration", "quote": "最多打三小时",
        "minimum_hours": None, "local_at": "2026-11-25T20:00:00+08:00"}]}, ensure_ascii=False)
    corrected = json.dumps({"claims": [arrival]}, ensure_ascii=False)
    responses = (proposal, malformed, corrected)
    if not can_arrive:
        responses += ('{"action":"send","text":"今天来不了，明天再问。"}', '{"claims":[]}')
    captured = []

    result = ModelActor(_client(responses, captured)).decide(context, trace_id="archived-duration-ceiling")

    first, repaired, *remaining = result.availability_reviews
    assert first["raw"] == malformed and "local_date" in first["error"]
    assert repaired["claims"] == [arrival] and repaired["step_index"] == 3
    assert json.loads(captured[2]["messages"][1]["content"])["draft"] == draft
    if can_arrive:
        assert result.action.text == draft and not result.prior_attempts and not remaining
    else:
        assert repaired["conflict"] and result.prior_attempts[0]["raw"] == proposal
        assert result.action.text == "今天来不了，明天再问。" and result.response_source["step_index"] == 4
        assert remaining[0]["step_index"] == 5


@pytest.mark.parametrize(("draft", "extracted"), [
    ("晚点你们还有桌吗？", {"claims": []}),
    ("明天我七点到", {"claims": [{"kind": "arrival", "quote": "明天我七点到", "local_date": "2026-11-26",
                                      "local_at": "2026-11-26T19:00:00+08:00"}]}),
])
def test_unavailable_actor_can_still_inquire_and_decide_about_a_later_day(draft, extracted):
    captured = []
    context = ActorContext(generate_personas(43, count=1)[0], NOW, events=(_event("session_open",
        can_arrive=False, free_from="2026-11-25T09:00:00+08:00", free_until="2026-11-25T14:30:00+08:00"),))
    response = json.dumps({"action": "send", "text": draft}, ensure_ascii=False)
    result = ModelActor(_client((response, json.dumps(extracted, ensure_ascii=False)), captured)).decide(
        context, trace_id="valid-future-inquiry")
    assert len(captured) == 2 and result.action.text == draft and not result.prior_attempts


@pytest.mark.parametrize("reproduce_archived_extraction", [True, False])
def test_archived_0353_inventory_inquiry_depends_on_separate_fact_evidence(reproduce_archived_extraction):
    """同一原稿与晚班日程中，定位旧误判来自独立提取，询桌和偏好可按原话放行。"""
    now = "2026-11-25T19:00:00+08:00"
    context = ActorContext(generate_personas(43, count=1)[0], now, events=(_event("activity",
        activity="open_dialogue", can_arrive=False, work_status="evening_shift", location="near_store",
        free_from="2026-11-25T11:00:00+08:00", free_until="2026-11-25T14:30:00+08:00"),))
    draft = "老板晚上好 还有四川麻将的位吗 1块档 我只想打两三个钟"
    proposal = json.dumps({"action": "send", "text": draft, "kind": "private", "recipient_id": OWNER}, ensure_ascii=False)
    archived = ('{"claims":[{"kind":"visit","quote":"还有四川麻将的位吗 1块档","local_date":"2026-11-25"},'
                '{"kind":"play_duration","quote":"我只想打两三个钟","local_date":"2026-11-25","minimum_hours":2}]}')
    responses = (proposal, archived, '{"action":"wait"}') if reproduce_archived_extraction else (proposal, '{"claims":[]}')
    captured = []

    decision = ModelActor(_client(responses, captured)).decide(context, trace_id="archived-0353-inquiry")

    assert json.loads(captured[1]["messages"][1]["content"]) == {"draft": draft, "local_now": now, "received_text": None}
    review, = decision.availability_reviews
    assert review["raw"] == responses[1] and review["response_source"]["http_response_complete"]
    if reproduce_archived_extraction:
        assert review["conflict"] and decision.action.action == "wait" and len(captured) == 3
        assert decision.prior_attempts[0]["raw"] == proposal
    else:
        assert review["claims"] == [] and review["conflict"] is None and len(captured) == 2
        assert decision.action.text == draft and decision.action.recipient_id == OWNER
        assert not decision.prior_attempts


@pytest.mark.parametrize("location, work", [("near_store", "evening_shift"), ("outside_city", "rest_day")])
def test_asking_about_stock_does_not_hide_an_actual_attendance_commitment(location, work):
    """问句之后的赴约或可打时长仍必须符合晚班和不在本地事实。"""
    context = ActorContext(generate_personas(43, count=1)[0], NOW, events=(_event("activity",
        activity="open_dialogue", can_arrive=False, work_status=work, location=location,
        free_from="2026-11-25T11:00:00+08:00", free_until="2026-11-25T14:30:00+08:00"),))
    draft = "晚上还有川麻吗？有位我就过去，我能打两个小时。"
    proposal = json.dumps({"action": "send", "text": draft}, ensure_ascii=False)
    claims = json.dumps({"claims": [
        {"kind": "visit", "quote": "有位我就过去", "local_date": "2026-11-25"},
        {"kind": "play_duration", "quote": "我能打两个小时", "local_date": "2026-11-25", "minimum_hours": 2},
    ]}, ensure_ascii=False)
    captured = []

    decision = ModelActor(_client((proposal, claims, '{"action":"wait"}'), captured)).decide(
        context, trace_id="inventory-inquiry-with-attendance-commitment")

    assert len(captured) == 3 and decision.action.action == "wait"
    assert decision.prior_attempts[0]["raw"] == proposal
    review, = decision.availability_reviews
    assert review["claims"] == json.loads(claims)["claims"] and "今天不能到店" in review["conflict"]


def test_actor_play_duration_uses_their_actual_proposed_arrival_and_not_the_world_lower_bound():
    """世界最早到店能容纳时长，原稿另选更晚的到店时刻仍需一起核对。"""
    captured = []
    context = ActorContext(generate_personas(43, count=1)[0], NOW, events=(_event("session_open",
        can_arrive=True, earliest_arrival="2026-11-25T19:00:00+08:00", free_from="2026-11-25T17:00:00+08:00",
        free_until="2026-11-25T23:00:00+08:00"),))
    action = '{"action":"send","text":"我晚八点到，想打四小时。"}'
    claims = json.dumps({"claims": [
        {"local_date": "2026-11-25", "kind": "arrival", "quote": "晚八点到", "local_at": "2026-11-25T20:00:00+08:00"},
        {"local_date": "2026-11-25", "kind": "play_duration", "quote": "想打四小时", "minimum_hours": 4},
    ]}, ensure_ascii=False)
    result = ModelActor(_client((action, claims, '{"action":"wait"}'), captured)).decide(context, trace_id="late-start-cap")
    assert result.action.action == "wait" and len(captured) == 3
    assert result.prior_attempts[0]["raw"] == action and result.availability_reviews[0]["conflict"]


def test_actor_may_communicate_a_rounded_estimate_consistent_with_new_world_event():
    """实际出行改变时允许自然改约，口语约数不必逐字念计算时刻。"""
    captured = []
    context = ActorContext(generate_personas(43, count=1)[0], NOW, events=(_event("traffic",
        can_arrive=True, earliest_arrival="2026-11-25T19:32:00+08:00", event_condition="en_route_with_traffic_delay",
        free_from="2026-11-25T17:00:00+08:00", free_until="2026-11-25T23:00:00+08:00"),))
    text = "堵车了，七点半左右到。"
    draft = json.dumps({"action": "send", "text": text}, ensure_ascii=False)
    claims = json.dumps({"claims": [{"local_date": "2026-11-25", "kind": "arrival",
        "quote": "七点半左右到", "local_at": "2026-11-25T19:30:00+08:00", "approximate": True}]}, ensure_ascii=False)
    result = ModelActor(_client((draft, claims), captured)).decide(context, trace_id="traffic-approximate-arrival")
    assert result.action.text == text and len(captured) == 2
    assert result.availability_reviews[0]["conflict"] is None


@pytest.mark.parametrize("board_id", ["outbound:board:actual", "ordinary-group-message"])
def test_recent_shop_confirmation_and_public_board_stay_in_the_same_actor_context(board_id):
    """刚获组局回执后的公开发布可供识别新桌，不把时钟重算或每条群信当成新的委托。"""
    person = generate_personas(43, count=1)[0]
    history = (
        {"message_id": "my-commission", "kind": "private", "sender_id": person.sender_id, "recipient_id": OWNER,
         "sent_at": "2026-11-25T17:55:00+08:00", "text": "麻烦帮我组一桌吧，我六点四十一能到。"},
        {"message_id": "my-shop-receipt", "kind": "private", "sender_id": OWNER, "recipient_id": person.sender_id,
         "sent_at": "2026-11-25T17:56:00+08:00", "text": "给你记上了，还差三个人。"},
        {"message_id": board_id, "kind": "group", "sender_id": OWNER, "recipient_id": ROOM,
         "sent_at": "2026-11-25T17:58:00+08:00", "text": "当前缺人局：人齐开 杭麻 173"},
    )
    event = _event("delivered", can_arrive=True, earliest_arrival="2026-11-25T18:43:00+08:00",
                   free_from="2026-11-25T17:00:00+08:00", free_until="2026-11-25T23:00:00+08:00") | {"caused_by": board_id}
    captured = []
    context = ActorContext(person, NOW, history=history, events=(event,), group_targets=(ROOM,))
    result = ModelActor(_client('{"action":"wait"}', captured)).decide(context, trace_id="own-commission-board")
    messages = captured[0]["messages"]
    projected = json.loads(messages[1]["content"])
    assert result.action.action == "wait" and projected["delivery"]["message_id"] == board_id
    assert [row["message_id"] for row in projected["history"]] == [row["message_id"] for row in history]
    assert projected["travel_update"]["previous_shop_message_id"] == "my-commission"
    if board_id.startswith("outbound:board:"):
        assert messages[0]["content"].startswith("刚收到的是群里看板更新")
        assert projected["travel_update"]["earliest_possible_if_leaving_now"] == "2026-11-25T18:43:00+08:00"
        assert "earliest_arrival" not in projected["events"][0]["details"]
    else:
        assert messages[0]["content"].startswith("你是麻将馆的一位模拟顾客")
        assert projected["events"][0]["details"]["earliest_arrival"] == "2026-11-25T18:43:00+08:00"


def test_availability_replan_keeps_infeasible_rounding_out_of_the_delivered_action():
    """重现开场下界 12:06，拒绝向下取整的估计，独立核验回应模型重新选择的说法。"""
    now = "2026-11-26T11:35:00+08:00"
    context = ActorContext(generate_personas(43, count=1)[0], now, events=(_event("activity",
        can_arrive=True, earliest_arrival="2026-11-26T12:06:00+08:00",
        free_from="2026-11-26T11:00:00+08:00", free_until="2026-11-26T21:00:00+08:00"),))

    def proposal(raw_text, clock):
        action = json.dumps({"action": "send", "text": raw_text}, ensure_ascii=False)
        claim = json.dumps({"claims": [{"local_date": "2026-11-26", "kind": "arrival", "quote": raw_text,
            "local_at": f"2026-11-26T{clock}:00+08:00", "approximate": True}]}, ensure_ascii=False)
        return action, claim

    first = proposal("我最早十二点出头能到", "12:00")
    corrected = proposal("我十二点十分左右才能到", "12:10")
    captured = []
    result = ModelActor(_client(first + corrected, captured)).decide(context, trace_id="archived-early-rounded-floor")
    assert result.status == "valid" and result.action.text == "我十二点十分左右才能到"
    assert result.prior_attempts[0]["raw"] == first[0] and "12:06" in result.prior_attempts[0]["error"]
    assert [review["raw"] for review in result.availability_reviews] == [first[1], corrected[1]]
    assert "别把下界向下取整" in captured[2]["messages"][-2]["content"]


def test_archived_evening_shift_duration_claim_repair_requires_its_own_date():
    """复现 0102 续跑中第二条陈述漏日期，格式修复后仍须拒绝不能兑现的晚班赴约。"""
    now = "2026-11-25T19:02:00+08:00"
    context = ActorContext(generate_personas(43, count=1)[0], now, events=(_event("activity",
        can_arrive=False, work_status="evening_shift", free_from="2026-11-25T11:00:00+08:00",
        free_until="2026-11-25T15:30:00+08:00"),))
    draft = json.dumps({"action": "send", "text": "帮我组一桌吧，我晚上八点到，打两个小时左右。"}, ensure_ascii=False)
    arrival = {"kind": "arrival", "quote": "我晚上八点到", "local_date": "2026-11-25", "local_at": "2026-11-25T20:00:00+08:00"}
    duration = {"kind": "play_duration", "quote": "打两个小时左右", "minimum_hours": 2, "approximate": True}
    malformed = json.dumps({"claims": [arrival, duration]}, ensure_ascii=False)
    repaired = json.dumps({"claims": [arrival, {**duration, "local_date": "2026-11-25"}]}, ensure_ascii=False)
    decline = '{"action":"send","text":"今天我上班，先不用组了哈。"}'
    captured = []
    result = ModelActor(_client((draft, malformed, repaired, decline, '{"claims":[]}'), captured)).decide(
        context, trace_id="archived-0102-evening-shift-date")

    assert result.status == "valid" and result.action.text == "今天我上班，先不用组了哈。"
    assert [review["step_index"] for review in result.availability_reviews] == [2, 3, 5]
    attempted, checked, final = result.availability_reviews
    assert attempted["raw"] == malformed and "local_date" in attempted["error"]
    assert checked["raw"] == repaired and "今天不能到店" in checked["conflict"]
    assert final["claims"] == [] and final["conflict"] is None
    repair_request = captured[2]["messages"]
    assert "每个 claims 对象单独填写" in repair_request[-2]["content"]
    assert "play_duration 必填 local_date" in repair_request[-2]["content"]
    assert json.loads(repair_request[1]["content"]) == {
        "draft": "帮我组一桌吧，我晚上八点到，打两个小时左右。", "received_text": None, "local_now": now,
    }


@pytest.mark.parametrize("hard_four_hours", [False, True])
def test_archived_duration_replan_gets_a_feasible_private_budget_without_relaxing_it(hard_four_hours):
    """复现 0103 的新可开场下界，可提议接近四小时，仍须拦下实际承诺打足四小时。"""
    now = "2026-11-25T19:02:00+08:00"
    context = ActorContext(generate_personas(43, count=1)[0], now, events=(_event("activity", can_arrive=True,
        earliest_arrival="2026-11-25T19:11:00+08:00", free_from="2026-11-25T09:30:00+08:00",
        free_until="2026-11-25T23:00:00+08:00"),))
    first = '{"action":"send","text":"我19点09以后能到，凑齐人就开，我打四个小时左右。"}'
    arrival = {"kind": "arrival", "quote": "19点09以后能到", "local_date": "2026-11-25", "local_at": "2026-11-25T19:09:00+08:00"}
    duration = {"kind": "play_duration", "quote": "我打四个小时左右", "local_date": "2026-11-25",
                "minimum_hours": 4, "approximate": True}
    commitment = "我得打足四个小时" if hard_four_hours else "想找个差不多四个小时的局，我最晚十一点得走"
    replanned = json.dumps({"action": "send", "text": f"我19点11以后能到，{commitment}。"}, ensure_ascii=False)
    arrival_after = {**arrival, "quote": "19点11以后能到", "local_at": "2026-11-25T19:11:00+08:00"}
    duration_after = ({"kind": "play_duration", "quote": commitment, "local_date": "2026-11-25", "minimum_hours": 4}
        if hard_four_hours else {"kind": "play_until", "quote": "最晚十一点得走", "local_date": "2026-11-25",
                                 "local_at": "2026-11-25T23:00:00+08:00"})
    captured = []
    result = ModelActor(_client((first, json.dumps({"claims": [arrival, duration]}, ensure_ascii=False), replanned,
        json.dumps({"claims": [arrival_after, duration_after]}, ensure_ascii=False)), captured)).decide(
            context, trace_id="archived-0103-duration-budget")

    correction = captured[2]["messages"]
    assert "本人实际能打多久要在可用时间内" in correction[-2]["content"]
    correction_context = json.loads(correction[-1]["content"])
    assert correction_context["earliest_play_at"] == "2026-11-25T19:11:00+08:00"
    assert correction_context["latest_play_until"] == "2026-11-25T23:00:00+08:00"
    assert correction_context["available_play_minutes_if_earliest"] == 229
    assert result.prior_attempts[0]["raw"] == first
    if hard_four_hours:
        assert result.status == "invalid" and result.action is None and "最少打牌时长" in result.error
    else:
        assert result.status == "valid" and commitment in result.action.text
        assert result.availability_reviews[-1]["conflict"] is None


@pytest.mark.parametrize(("daypart", "clock", "expected"), [
    ("上午", "10:00", "valid"), ("晚上", "22:00", "invalid"),
])
def test_archived_daypart_replan_keeps_the_actual_qualified_clock(daypart, clock, expected):
    """重现 0038 的粗时段被错读成当前时刻，重选的具体时点仍按原可达窗口校验。"""
    now = "2026-11-26T09:35:00+08:00"
    context = ActorContext(generate_personas(43, count=1)[0], now, events=(_event("activity", can_arrive=True,
        earliest_arrival="2026-11-26T09:48:00+08:00", free_from="2026-11-26T09:30:00+08:00",
        free_until="2026-11-26T16:00:00+08:00"),))
    first = '{"action":"send","text":"老板，上午我可以过去，有一底杭麻吗？"}'
    ungrounded = ('{"claims":[{"kind":"arrival","quote":"上午我可以过去","local_date":"2026-11-26",'
                  '"local_at":"2026-11-26T09:35:00+08:00","approximate":true}]}')
    actual_text = f"我今天{daypart}十点以后能到，一底杭麻有位吗？"
    corrected = json.dumps({"action": "send", "text": actual_text}, ensure_ascii=False)
    extracted = json.dumps({"claims": [{"kind": "arrival", "quote": f"我今天{daypart}十点以后能到",
        "local_date": "2026-11-26", "local_at": f"2026-11-26T{clock}:00+08:00"}]}, ensure_ascii=False)
    captured = []
    result = ModelActor(_client((first, ungrounded, corrected, extracted), captured)).decide(
        context, trace_id="archived-0038-actual-clock")
    assert result.status == expected and result.availability_reviews[0]["conflict"]
    assert result.availability_reviews[-1]["claims"][0]["local_at"] == f"2026-11-26T{clock}:00+08:00"
    assert "local_now 供日期和原话时段消歧，不是这个人声称的到店时刻" in captured[1]["messages"][0]["content"]
    assert "说清上午、下午或晚上" in captured[2]["messages"][-2]["content"]
    if expected == "valid":
        assert result.action.text == actual_text
    else:
        assert result.action is None and "超出当前可达时段" in result.error


@pytest.mark.parametrize("can_arrive", [True, False])
def test_broad_daypart_visit_has_no_invented_clock_and_still_needs_a_possible_visit(can_arrive):
    """模糊说上午到没有钟点，今天实际到不了时仍拦下这个赴约陈述。"""
    now = "2026-11-26T09:35:00+08:00"
    context = ActorContext(generate_personas(43, count=1)[0], now, events=(_event("activity", can_arrive=can_arrive,
        earliest_arrival="2026-11-26T09:48:00+08:00", free_from="2026-11-26T09:30:00+08:00",
        free_until="2026-11-26T16:00:00+08:00"),))
    draft = '{"action":"send","text":"我上午可以过去，一底杭麻有位吗？"}'
    claims = '{"claims":[{"kind":"visit","quote":"我上午可以过去","local_date":"2026-11-26"}]}'
    captured = []
    result = ModelActor(_client((draft, claims, '{"action":"wait"}'), captured)).decide(
        context, trace_id="broad-morning-visit")
    assert result.status == "valid"
    assert result.action.action == ("send" if can_arrive else "wait")
    assert (result.availability_reviews[0]["conflict"] is None) is can_arrive
    assert "local_at" not in result.availability_reviews[0]["claims"][0]


def test_physical_arrival_requires_current_world_and_uses_one_bounded_correction():
    captured = []
    context = ActorContext(generate_personas(43, count=1)[0], NOW, events=(_event("arrival",
        can_arrive=True, event_condition="en_route_with_traffic_delay", earliest_arrival="2026-11-25T18:12:00+08:00",
        free_from="2026-11-25T09:00:00+08:00", free_until="2026-11-25T22:00:00+08:00"),))
    result = ModelActor(_client(('{"action":"arrive"}', '{"action":"arrive"}'), captured)).decide(
        context, trace_id="too-early-to-check-in")
    assert len(captured) == 2 and result.status == "invalid" and result.action is None
    assert len(result.availability_reviews) == 2 and all(row["conflict"] for row in result.availability_reviews)


def test_mock_actor_initiates_authorizes_and_uses_only_observed_directional_relations():
    cast = generate_personas(17, count=6)
    person = cast[0]
    avoid = next(edge for edge in person.relationships if edge.kind == "avoid_same_table")
    compat = {"stake": person.stake_min, "game_type": person.preferred_game, "start_time": "19:00", "duration_hours": 4}
    actor = MockActor(seed=17)

    def act(*events):
        return actor.decide(ActorContext(person, NOW, events=events, group_targets=(ROOM,)), trace_id="offline").action

    request = act(_event("need", **compat))
    assert request.recipient_id == OWNER and "建个" in request.text and "招人" in request.text
    assert "直接算我一个" in act(_event("opportunity", **compat, attendees=[avoid.target_id])).text
    assert "确认上这桌" in act(_event("opportunity", **compat), _event("delivered", offer=True)).text
    follow_up = _event("delivered") | {"text": "有一桌时间对得上，要不要留个位置？"}
    assert "确认上这桌" in act(_event("opportunity", **compat), follow_up).text
    refusal = act(_event("opportunity", **compat, known_attendees=[{"sender_id": avoid.target_sender_id}]))
    assert "先不去" in refusal.text and avoid.target_name not in refusal.text
    assert "这个档我先不打" in act(_event("opportunity", stake=10)).text
    assert "取消" in act(_event("need", activity="cancel")).text
    assert act(_event("arrival")).action == "arrive" and act(_event("departure")).action == "leave"
    group = act(_event("share_joined", **compat))
    assert (group.kind, group.recipient_id) == ("group", ROOM) and "已经报上" in group.text
    assert "不用订" in act(_event("slot_query", **compat)).text
    assert act(_event("delivered")).action == "wait"
    assert act(_event("activity")) == act(_event("activity"))


def _real_delivery(person, message_id: str, text: str) -> dict:
    return {"message_id": message_id, "kind": "private", "sender_id": OWNER,
            "recipient_id": person.sender_id, "text": text, "sent_at": NOW}


def test_free_role_session_answers_the_specific_delivered_question_then_respects_lookup_scope():
    person, other = generate_personas(17, count=2)
    terms = {"case_goal": "lookup", "start_time": "19:00", "game_type": "杭州麻将", "stake": 1, "duration_hours": 4}
    actor = MockActor(17)
    initial = actor.decide(ActorContext(person, NOW, events=(_event("session_open", **terms),)), trace_id="open")
    assert initial.action.kind == "private"
    assert "先问问" in initial.action.text and "直接算我" not in initial.action.text
    assert sum(term in initial.action.text for term in ("19:00", "晚上7点", "19点", "杭州麻将", "底注1", "4小时")) == 1

    history = (_real_delivery(person, "question", "你想几点开始，能打多久？"),
               _real_delivery(other, "private-foreign", "你有几个朋友一起来？"),
               {"message_id": "group-chatter", "kind": "group", "sender_id": other.sender_id,
                "recipient_id": ROOM, "text": "群里有人吗？"})
    follow_up = _event("delivered", activity="open_dialogue", **terms) | {"caused_by": "question"}
    reply = actor.decide(ActorContext(person, NOW, history=history, events=(follow_up,), group_targets=(ROOM,)), trace_id="answer")
    assert reply.context["delivery"]["message_id"] == "question"
    assert reply.action.text == "19:00可以，打4小时。"
    answer = _real_delivery(person, "result", "19:00正好有一桌还有位，你要不要留位？")
    result = actor.decide(ActorContext(person, NOW, history=(*history, answer),
        events=(follow_up | {"caused_by": "result"},)), trace_id="result")
    assert result.action.text == "好，我先了解一下，不用留位。"


def test_causal_delivery_missing_or_for_another_recipient_never_triggers_a_mock_or_model_reply():
    person, peer = generate_personas(23, count=2)
    captured = []
    event = _event("delivered", activity="open_dialogue", case_goal="try_join") | {"caused_by": "missing"}
    history = (_real_delivery(person, "older", "你想几点打？"), _real_delivery(peer, "missing", "别人收到了"))
    context = ActorContext(person, NOW, history=history, events=(event,))
    assert MockActor().decide(context, trace_id="offline").action.action == "wait"
    model = ModelActor(_client('{"action":"send","text":"19点可以"}', captured))
    result = model.decide(context, trace_id="blocked")
    assert result.source == "no_delivery" and result.action.action == "wait"
    assert captured == [] and result.context["delivery"] is None


def test_compatible_join_negotiation_wait_and_cancellation_are_based_on_actual_prior_messages():
    profile = generate_personas(17, count=1)[0]
    temperament = replace(profile.temperament, commitment="ready_to_join", negotiation="adjust_time", fallback="follow_up")
    person = replace(profile, temperament=temperament, stake_min=0.5, stake_max=2, smoking="non_smoking")
    terms = {"case_goal": "try_join", "start_time": "19:00", "game_type": "杭州麻将", "stake": 1}
    mock = MockActor()

    def reply_to(text, role=person, history=()):
        event = _event("delivered", activity="open_dialogue", **terms) | {"caused_by": "answer"}
        visible = (*history, _real_delivery(role, "answer", text))
        return mock.decide(ActorContext(role, NOW, history=visible, events=(event,)), trace_id="route").action

    assert "帮我确认上这桌" in reply_to("19点有一桌，无烟，底注1，你要不要来？").text
    assert reply_to("19点有一桌，有人抽烟，你要不要来？").text == "这桌我就不去了，有无烟的再叫我。"
    assert "这档我不打" in reply_to("有一桌底注5，你要不要来？").text
    assert "20:30也行" in reply_to("19点房间满了，最早20:30，你方便吗？").text
    assert reply_to("有消息再叫你。").text == "行，有合适的再叫我。"
    assert "确认上这桌" not in reply_to("没有空位。").text
    assert reply_to("位置留好了。", role=person).action == "wait"
    steady = replace(person, temperament=replace(temperament, negotiation="hold_terms"))
    assert "取消" not in reply_to("没有合适的，你再问问？", role=steady).text
    own = {"message_id": "own-1", "kind": "private", "sender_id": person.sender_id,
           "recipient_id": OWNER, "text": "有符合条件的帮我报名"}
    assert "取消" in reply_to("没有合适的，你再问问？", role=steady, history=(own,)).text
    silent = replace(person, temperament=replace(temperament, fallback="leave_unanswered"))
    assert reply_to("没有合适的。", role=silent).action == "wait"


def test_peer_invite_is_a_private_question_with_one_way_recipient_permission():
    cast = generate_personas(17, count=5)
    organizer = cast[0]
    relationship = next(edge for edge in organizer.relationships if edge.kind == "regular_partner")
    partner = next(person for person in cast if person.sender_id == relationship.target_sender_id)
    terms = {"game_type": "杭州麻将", "stake": 1, "start_time": "19:00", "duration_hours": 4}
    invite = _event("peer_invite", peer_id=partner.sender_id, **terms)
    actor = MockActor(17)
    allowed = ActorContext(organizer, NOW, events=(invite,), private_targets=(OWNER, partner.sender_id))
    decision = actor.decide(allowed, trace_id="peer-allowed")
    message = decision.action
    assert decision.context["events"][0]["details"]["peer_id"] == partner.sender_id
    assert (message.kind, message.recipient_id) == ("private", partner.sender_id)
    assert all(term in message.text for term in ("19:00", "杭州麻将", "4小时", "想来不"))
    assert "订" not in message.text

    reverse = _event("peer_invite", peer_id=organizer.sender_id, **terms)
    denied = ActorContext(partner, NOW, events=(reverse,))
    mocked = actor.decide(denied, trace_id="peer-denied-mock")
    assert mocked.action.action == "wait"
    assert "peer_id" not in mocked.context["events"][0]["details"]
    raw = json.dumps({"action": "send", "text": "今晚打不打？", "kind": "private", "recipient_id": organizer.sender_id})
    rejected = ModelActor(_client(raw, [])).decide(denied, trace_id="peer-denied-model")
    assert rejected.status == "invalid" and rejected.action is None
    assert rejected.raw == raw


@pytest.mark.parametrize(("smoking", "expected"), [
    ("non_smoking", "只打无烟"), ("smoking_ok", "有人抽烟的也没问题"), ("flexible", "主要打四川麻将"),
])
def test_preference_discloses_the_roles_persistent_style_not_current_game_terms(smoking, expected):
    person = replace(generate_personas(17, count=1)[0], smoking=smoking, preferred_game="四川麻将")
    event = _event("preference", game_type="杭州麻将", smoke="smoking_ok")
    context = ActorContext(person, NOW, events=(event,), group_targets=(ROOM,))
    decision = MockActor(17).decide(context, trace_id="preference")
    assert decision.status == "valid"
    assert (decision.action.kind, decision.action.recipient_id) == ("private", OWNER)
    assert expected in decision.action.text and "以后" in decision.action.text
    assert "杭州麻将" not in decision.action.text


def _judge_raw(verdict, check, cited, *, overall=None):
    return json.dumps({"verdict": overall or verdict, "findings": [
        {"check": check, "verdict": verdict, "evidence_ids": cited, "reason": "收到店家确认后再进群说明已报名。"},
    ]}, ensure_ascii=False)


def test_independent_judge_references_frozen_evidence_and_retains_source():
    captured = []
    evidence = [{"message_id": "confirmed-1", "text": "留好了", "details": {"confirmed": True}}]
    raw = _judge_raw("pass", "真实确认", ["confirmed-1"])
    judge = SemanticJudge(_client(raw, captured, model="judge-test"))
    result = judge.judge("case", evidence, trace_id="judge-trace", checks=("真实确认",))
    evidence[0]["details"]["confirmed"] = False
    assert result.status == "valid" and result.verdict == "pass"
    assert result.raw == raw and result.findings[0].evidence_ids == ("confirmed-1",)
    assert result.evidence[0]["details"]["confirmed"] is True
    assert result.response_source["http_response_complete"] and result.response_source["trace_id"] == "judge-trace"
    assert captured[0]["model"] == "judge-test"


@pytest.mark.parametrize("returned_case_id, expected_status", [("case", "valid"), ("other", "invalid")])
def test_judge_can_echo_only_the_case_under_review(returned_case_id, expected_status):
    """兼容判定器回显同一案例，同时拒绝其他案例的判决。"""
    output = {"case_id": returned_case_id, **json.loads(_judge_raw("fail", "真实确认", ["e1"]))}
    raw = json.dumps(output, ensure_ascii=False)
    result = SemanticJudge(_client(raw, [])).judge(
        "case", [{"event_id": "e1"}], trace_id="judge-case-id", checks=("真实确认",),
    )
    assert result.status == expected_status and result.raw == raw
    assert result.verdict == ("fail" if expected_status == "valid" else "inconclusive")


@pytest.mark.parametrize("raw", [
    _judge_raw("pass", "真实确认", ["fabricated"]), _judge_raw("pass", "真实确认", []),
    _judge_raw("fail", "真实确认", ["e1"], overall="pass"), _judge_raw("pass", "忽略检查", ["e1"]),
    "not JSON",
])
def test_judge_invalid_verdict_does_not_masquerade_as_a_business_pass(raw):
    judge = SemanticJudge(_client(raw, []))
    result = judge.judge("case", [{"event_id": "e1"}], trace_id="bad-verdict", checks=("真实确认",))
    assert (result.status, result.verdict, result.raw) == ("invalid", "inconclusive", raw)
    assert result.error and result.findings == ()


def test_transport_error_keeps_both_tasks_non_executable_without_leaking_exception_text():
    class FailingClient:
        def complete(self, messages, *, trace_id, timeout_seconds):
            raise TimeoutError("private remote exception text")

    client = FailingClient()
    person = generate_personas(0, count=1)[0]
    role = ModelActor(client).decide(ActorContext(person, NOW), trace_id="role-error")
    judge = SemanticJudge(client).judge("case", [{"event_id": "e1"}], trace_id="judge-error")
    assert role.status == judge.status == "transport_error"
    assert role.action is None and role.raw is None
    assert judge.verdict == "inconclusive" and judge.raw is None
    assert role.error == judge.error == "TimeoutError"
