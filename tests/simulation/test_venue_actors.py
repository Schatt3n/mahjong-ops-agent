"""模拟角色隔离、合同、模型原文和判定证据。"""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from mahjong_agent_runtime.llm import AgentLLMConfig, OpenAICompatibleAgentClient

from .venue_actors import ActorContext, MockActor, ModelActor, SemanticJudge, generate_personas, write_personas


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


def _client(raw: str, captured: list, *, model: str = "actor-test") -> OpenAICompatibleAgentClient:
    def urlopen(request, **_):
        captured.append(json.loads(request.data))
        return _Response(raw)

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
    assert [row["text"] for row in payload["history"]] == ["只给我的", "群里能看到"]
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
])
def test_invalid_role_actions_are_retained_and_never_executable(raw):
    result = ModelActor(_client(raw, [])).decide(ActorContext(generate_personas(5, count=1)[0], NOW), trace_id="bad-action")
    assert result.status == "invalid"
    assert result.to_dict()["action"] is None
    assert result.raw == raw and result.error
    assert result.response_source["http_response_complete"] is True


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
