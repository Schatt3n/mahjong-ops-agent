"""验证晚到征询跨私聊的实际送达和多方答复，不代桌改期或占席。"""

from datetime import datetime, timedelta
import json

import pytest

from mahjong_agent_runtime import AgentAction, InMemoryAgentStore, InMemoryTraceRecorder, SQLiteAgentStore, ToolCall, ToolGateway, UserMessage
from mahjong_agent_runtime.domains.context_builders.tool_results import tool_result_for_context
from mahjong_agent_runtime.domains.game_participants import LateParticipationRequiresConfirmation
from mahjong_agent_runtime.group_chat.models import ChannelIdentity, GroupRoomPolicy
from mahjong_agent_runtime.late_arrival import (
    LATE_ARRIVAL_QUESTION_TASK_TYPE, LATE_ARRIVAL_UPDATE_TASK_TYPE, late_arrival_context,
    late_arrival_delivery_context, late_arrival_seat_status, record_late_arrival_decision, request_late_arrival,
)
from mahjong_agent_runtime.models import MessageReference, using_clock
from mahjong_agent_runtime.services import ToolExecutionService


TOUCH = datetime.fromisoformat("2026-11-26T11:39:30+08:00")
START = TOUCH.replace(hour=17, minute=30, second=0)
ETA = START + timedelta(hours=1)
ROOM = "room"
CHANNEL = "wechaty"


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryAgentStore()
        return
    db = SQLiteAgentStore(tmp_path / "consultation.sqlite")
    try:
        yield db
    finally:
        db._connection.close()


def _private(account):
    return "private:" + account


def _seed(store, *, peer=False, reachable=True, duration_hours=4, start=START):
    accounts = ["host", "guest"] + (["peer"] if peer else [])
    store.upsert_group_room_policy(GroupRoomPolicy(room_id=ROOM, channel=CHANNEL))
    for account in accounts:
        store.upsert_channel_identity(ChannelIdentity(channel=CHANNEL, external_user_id=account,
            customer_id=account, public_name=account, private_conversation_id=_private(account),
            can_private_message=account != "peer" or reachable))
    game, _ = store.create_game(conversation_id=_private("host"), organizer_id="host", organizer_name="桌主",
        requirement={"game_type": "hangzhou_mahjong", "stake": "2", "start_time_kind": "scheduled",
                     "planned_start_at": start.isoformat(),
                     **({"duration_hours": duration_hours} if duration_hours is not None else {})},
        known_players=[{"customer_id": account, "display_name": account, "seat_count": 1}
                       for account in accounts if account != "guest"], trace_id="setup-table")
    store.ensure_game_board_arrangement(game.game_id, trace_id="setup-board")
    return game


def _input(store, account, text, message_id, trace_id, at, *, metadata=None):
    store.append_user_turn(UserMessage(_private(account), account, account, text, message_id, at,
                                       metadata=metadata or {}), trace_id)


def _query(store, game, trace_id, *, other_match=False, truncated=False):
    matches = [{"match_kind": "exact", "game": game.to_dict()}]
    if other_match:
        matches.append({"match_kind": "exact", "game": {"game_id": "other"}})
    store.append_tool_turn(_private("guest"), json.dumps([{"name": "search_current_games", "called": True,
        "allowed": True, "result": {"matches": matches, "selection_truncated": truncated}}]), trace_id)


def _request(store, game, trace_id="inquiry", eta=ETA):
    call = ToolCall(name="request_late_arrival", arguments={"game_id": game.game_id, "arrival_at": eta.isoformat()})
    return request_late_arrival(store, call, trace_id, _private("guest"), "guest", "客人")


def _receipt(store, task, *, recipient=None):
    reference = MessageReference(message_id=task.idempotency_key, conversation_id=task.conversation_id,
        business_ref_type=task.task_type, business_ref_id=task.aggregate_id, text="这桌到这个点来行吗？",
        channel=CHANNEL, recipient_id=recipient or task.subject_id,
        metadata={"delivery_id": f"sent:{task.task_id}", "negotiation_id": task.payload["negotiation_id"]})
    store.register_message_reference(reference)
    return reference


def _answer(store, account, task, decision="accepted"):
    trace = f"answer:{account}"
    source = f"answer-msg:{account}"
    call = ToolCall(name="record_late_arrival_decision", arguments={
        "decision_task_id": task.task_id, "decision": decision, "evidence_message_id": source,
    })
    return record_late_arrival_decision(store, call, trace, _private(account), account, account)


def _tasks(store, task_type):
    return [task for task in store.scheduled_tasks.values() if task.task_type == task_type]


@pytest.mark.parametrize(("text", "asked", "expected"), [
    ("老板好 今晚七点半那桌杭麻一块的底 还有位吗 我到七点半稍晚一点 能先留一个吗", None, None),
    ("我19:45才下班 可能晚点到 这桌能等吗？", None, None),
    ("我朋友19:45能到，我还不确定。", None, None),
    ("我19:45到不了，帮我问下。", None, None),
    ("我大概19:45能到，能不能等一下？", None, True),
    ("大概19:45", "same_table", True),
    ("大概19:45", "another_table", None),
    ("大概19:45", None, None),
])
def test_late_arrival_requires_actual_customer_time_and_preserves_an_estimate(store, text, asked, expected):
    """原话不含精确到店或同桌问话时，不能将模型指定的19:45写成待派发时点。"""
    at = datetime.fromisoformat("2026-11-26T19:05:00+08:00")
    arrival = at.replace(minute=45)
    with using_clock(lambda: at):
        game = _seed(store, duration_hours=None, start=at.replace(minute=30))
        if asked:
            lookup = {"game_type": "hangzhou_mahjong", "stake": "2", "base_stake": 2.0, "stake_label": "2",
                      "start_time_kind": "scheduled"}
            store.append_assistant_turn(_private("guest"), "七点半那桌还有位。你大概几点能到？", "prior",
                metadata={"delivery_status": "delivered", "delivered_game_offer": {
                    "recipient_id": "guest", "game_id": game.game_id if asked == "same_table" else "different",
                    "terms": {**lookup, "planned_start_at": game.planned_start_at.isoformat()},
                    "game_reference_only": True,
                }})
        _input(store, "guest", text, "v8-0235", "inquiry", at)
        _query(store, game, "inquiry")
        before = store.require_game(game.game_id).to_dict()
        request = _request(store, game, eta=arrival)
        assert store.require_game(game.game_id).to_dict() == before
        questions = _tasks(store, LATE_ARRIVAL_QUESTION_TASK_TYPE)
        if expected is None:
            assert not request.called and not request.allowed and not questions
            assert request.result["validation_error"]["code"] == "late_arrival_customer_clock_required"
        else:
            question, = questions
            assert request.called and request.result["contact_started"] is False
            assert request.result["late_arrival_request"]["arrival_is_estimate"] is expected
            context = tool_result_for_context(request)["result"]["late_arrival_request"]
            assert context["arrival_is_estimate"] is expected
            assert question.payload["arrival_at"] == arrival.isoformat()
            assert question.payload["arrival_is_estimate"] is expected


def test_late_arrival_hides_unagreed_duration_and_rejects_incomplete_lookup(store):
    """截断搜索不能证明唯一，同桌的内部占用预估也不能冒充约定散场。"""
    with using_clock(lambda: TOUCH):
        game = _seed(store, duration_hours=None)
        _input(store, "guest", "我六点半到，能搭就算我。", "ask-msg", "inquiry", TOUCH)
        _query(store, game, "inquiry", truncated=True)
        refused = _request(store, game)
        assert not refused.called and not refused.allowed
        assert not _tasks(store, LATE_ARRIVAL_QUESTION_TASK_TYPE)
        assert tool_result_for_context(refused)["result"]["late_arrival_request"]["status"] == "unavailable"

        _query(store, game, "inquiry")
        result = _request(store, game)
        assert result.called and result.allowed and game.planned_end_at is not None
        assert "planned_end_at" not in result.result["late_arrival_request"]
        assert "remaining_play_minutes" not in result.result["late_arrival_request"]


@pytest.mark.parametrize("reachable", [True, False])
def test_v4_0022_missing_current_lookup_can_retry_same_key_without_double_arranging(store, reachable):
    """按同一来话补查原桌后可重领尚无副作用的调用，仍由后端核对联系人边界。"""
    at = datetime.fromisoformat("2026-11-25T19:37:00+08:00")
    start, arrival = at.replace(minute=30), at.replace(minute=52)
    message = UserMessage(_private("guest"), "guest", "客人", "要的 麻烦帮我问下这桌能不能等到我19:52到 谢谢",
                          message_id="late-arrival-input", sent_at=at)
    trace = InMemoryTraceRecorder()
    gateway = ToolGateway(store, trace_recorder=trace)
    service = ToolExecutionService(store, gateway, trace)

    def submit(call, step):
        goal = AgentAction("核实顾客晚到", "needs_tool", "按同桌现状核对", tool_calls=[call])
        result, = service.execute_tool_calls(goal, message=message, trace_id="v4-late-arrival",
            previous_step_tool_results=[], step_index=step, run_id="v4-late-run",
            run_version=store.conversation_version(message.conversation_id)).tool_results
        return result

    with using_clock(lambda: at):
        game = _seed(store, peer=not reachable, reachable=reachable, duration_hours=None, start=start)
        store.update_game_requirement(game_id=game.game_id, requirement_patch={"max_duration_hours": 2.5},
            reason="原桌约定最迟可用时间", trace_id="table-cap")
        game_before = store.require_game(game.game_id)
        assert game_before.planned_end_at == at.replace(hour=22, minute=0)
        store.append_user_turn(message, "v4-late-arrival")

        def request(step):
            return submit(ToolCall("request_late_arrival", {"game_id": game.game_id, "arrival_at": arrival.isoformat()},
                                   call_id=f"late-{step}"), step)

        first = request(1)
        assert not first.called and not first.allowed and not first.deduplicated
        assert first.result["retryable_no_effect"] is True
        assert first.result["validation_error"]["code"] == "late_arrival_current_table_lookup_required"
        assert "无需再问" in tool_result_for_context(first)["result"]["instruction"]
        assert not _tasks(store, LATE_ARRIVAL_QUESTION_TASK_TYPE)

        wrong = submit(ToolCall("search_current_games", {"lookup_scope": "specific_table",
            "requirement": {"game_type": "sichuan_mahjong"}}, call_id="not-this-table"), 2)
        assert wrong.called and wrong.allowed and not wrong.result["matches"]
        absent = request(3)
        assert not absent.called and not absent.allowed and not absent.deduplicated
        assert absent.result["retryable_no_effect"] is True and not _tasks(store, LATE_ARRIVAL_QUESTION_TASK_TYPE)

        read = submit(ToolCall("search_current_games", {"lookup_scope": "specific_table", "requirement": {
            "game_type": "hangzhou_mahjong", "planned_start_at": start.isoformat(), "start_time_kind": "scheduled",
        }}, call_id="original-table"), 4)
        assert read.called and read.allowed
        only, = read.result["matches"]
        assert only["match_kind"] == "exact" and only["game"]["game_id"] == game.game_id
        retried = request(5)
        replay = request(6)

        assert retried.called is retried.allowed is reachable and not retried.deduplicated
        assert "retryable_no_effect" not in retried.result
        assert replay.called is replay.allowed is reachable and replay.deduplicated
        assert first.idempotency_key == absent.idempotency_key == retried.idempotency_key == replay.idempotency_key
        assert store.require_game(game.game_id).to_dict() == game_before.to_dict()
        if reachable:
            assert retried.result["late_arrival_request"]["status"] == "queued"
            assert retried.result["contact_started"] is False
            question, = _tasks(store, LATE_ARRIVAL_QUESTION_TASK_TYPE)
            assert question.subject_id == "host" and question.payload["source_message_id"] == message.message_id
            assert question.payload["arrival_at"] == arrival.isoformat()
        else:
            assert "联系人授权不足" in retried.error and not _tasks(store, LATE_ARRIVAL_QUESTION_TASK_TYPE)


@pytest.mark.parametrize("same_instant_duplicate", [False, True])
def test_v8_0022_dated_lookup_disambiguates_late_arrival_without_changing_seats(store, same_instant_duplicate):
    """同钟点的隔日桌不能混入确切日期查局，同一瞬时仍有两桌时继续拒绝联系。"""
    at = datetime.fromisoformat("2026-11-26T19:35:00+08:00")
    start = "2026-11-26T19:30:00+08:00"
    next_start = "2026-11-27T19:30:00+08:00"
    arrival = "2026-11-26T20:03:00+08:00"
    trace_id = "dated-late-arrival"
    message = UserMessage(_private("guest"), "guest", "客人",
        "老板好 七点半那桌杭麻1到3块无烟的我20点03分能到 晚一点还能搭吗",
        message_id="dated-inquiry", sent_at=at)
    trace = InMemoryTraceRecorder()
    service = ToolExecutionService(store, ToolGateway(store, trace_recorder=trace), trace)
    terms = {"game_type": "hangzhou_mahjong", "stake_range": {"min": 1, "max": 3},
             "smoke_preference": "no_smoking", "start_time_kind": "scheduled", "start_time": "19:30"}

    def submit(name, arguments, step):
        action = AgentAction("核实晚到", "needs_tool", "沿用当前原桌条件", tool_calls=[
            ToolCall(name, arguments, call_id=f"step-{step}"),
        ])
        return service.execute_tool_calls(action, message=message, trace_id=trace_id,
            previous_step_tool_results=[], step_index=step, run_id="dated-inquiry-run",
            run_version=store.conversation_version(message.conversation_id)).tool_results[0]

    def lookup(requirement, step):
        report = submit("search_current_games", {"lookup_scope": "specific_table", "requirement": requirement}, step)
        assert report.called and report.allowed
        return report.result, {row["game"]["game_id"] for row in report.result["matches"]}

    with using_clock(lambda: at):
        original = _seed(store, duration_hours=None, start=datetime.fromisoformat(start))
        store.update_game_requirement(game_id=original.game_id, requirement_patch={**terms, "max_duration_hours": 2.5},
            reason="写入原桌条件", trace_id="table-terms")
        tomorrow, _ = store.create_game(conversation_id=_private("tomorrow"), organizer_id="tomorrow", organizer_name="另一桌",
            requirement={**terms, "stake_range": {"min": 2, "max": 3}, "planned_start_at": next_start},
            known_players=[], trace_id="tomorrow-table")
        store.append_user_turn(message, trace_id)
        request = {"game_id": original.game_id, "arrival_at": arrival}
        _, undated = lookup(terms, 1)
        assert undated == {original.game_id, tomorrow.game_id}
        denied = submit("request_late_arrival", request, 2)
        assert denied.result["validation_error"]["code"] == "late_arrival_current_table_lookup_required"
        assert not _tasks(store, LATE_ARRIVAL_QUESTION_TASK_TYPE)
        _, other_day = lookup({**terms, "planned_start_at": next_start}, 3)
        assert other_day == {tomorrow.game_id}
        duplicate_ids = set()
        if same_instant_duplicate:
            duplicate, _ = store.create_game(conversation_id=_private("duplicate"), organizer_id="duplicate", organizer_name="同一晚另一桌",
                requirement={**terms, "planned_start_at": start}, known_players=[], trace_id="same-opening-table")
            duplicate_ids.add(duplicate.game_id)
        before = {key: game.to_dict() for key, game in store.games.items()}
        precise, exact = lookup({**terms, "planned_start_at": start}, 4)
        assert exact == {original.game_id} | duplicate_ids
        assert "planned_start_at" in precise["customer_reply_contract"]["evidence_scope"]["applied_filter_fields"]
        _, equivalent = lookup({**terms, "planned_start_at": "2026-11-26T11:30:00+00:00"}, 5)
        assert equivalent == exact
        result = submit("request_late_arrival", request, 6)
        assert result.called is result.allowed is not same_instant_duplicate
        assert result.result["contact_started"] is False
        assert {key: game.to_dict() for key, game in store.games.items()} == before
        if same_instant_duplicate:
            assert result.result["validation_error"]["code"] == "late_arrival_current_table_lookup_required"
            assert not _tasks(store, LATE_ARRIVAL_QUESTION_TASK_TYPE)
        else:
            assert result.result["late_arrival_request"]["status"] == "queued"
            question, = _tasks(store, LATE_ARRIVAL_QUESTION_TASK_TYPE)
            assert question.aggregate_id == original.game_id and question.subject_id == "host"
            assert question.payload["source_message_id"] == message.message_id
            assert question.payload["arrival_at"] == arrival


def test_late_arrival_requires_real_receipts_and_all_confirmed_contacts(store):
    with using_clock(lambda: TOUCH):
        game = _seed(store, peer=True)
        before = store.require_game(game.game_id).to_dict()
        _input(store, "guest", "我六点半到，能帮我问问这桌能否补吗？", "ask-msg", "inquiry", TOUCH)
        _query(store, game, "inquiry")
        first = _request(store, game)
        assert first.result["late_arrival_request"]["status"] == "queued"
        assert first.result["contact_started"] is False
        own, = late_arrival_context(store, conversation_id=_private("guest"), sender_id="guest",
                                           at=TOUCH)["current_late_arrival_requests"]
        assert own == first.result["late_arrival_request"]
        assert own["planned_end_at"] == (START + timedelta(hours=4)).isoformat()
        assert own["remaining_play_minutes"] == 180
        compact = tool_result_for_context(first)
        assert compact["result"]["late_arrival_request"] == own
        assert compact["result"]["contact_started"] is False
        assert all(item["entity_id"] is None for item in compact["state_transitions"])
        assert len(first.state_transitions) == 2
        assert not _request(store, game).state_transitions
        questions = {task.subject_id: task for task in _tasks(store, LATE_ARRIVAL_QUESTION_TASK_TYPE)}
        assert set(questions) == {"peer", "host"}
        for role, task in questions.items():
            assert task.conversation_id == _private(role)
            assert late_arrival_delivery_context(store, task, at=TOUCH)["recipient"].customer_id == role
        assert late_arrival_context(store, conversation_id=_private("host"), sender_id="host", at=TOUCH) == {}
        assert late_arrival_seat_status(store, game=game, customer_id="guest", at=TOUCH) == "queued"
        with pytest.raises(LateParticipationRequiresConfirmation) as pending_seat:
            store.join_game(game_id=game.game_id, customer_id="guest", display_name="客人", trace_id="before-receipts")
        assert pending_seat.value.status == "queued"
    delivered_at = TOUCH + timedelta(hours=3)
    with using_clock(lambda: delivered_at):
        for account, question in questions.items():
            _receipt(store, question)
            own = late_arrival_context(store, conversation_id=_private(account), sender_id=account, at=delivered_at)
            pending, = own["pending_late_arrival_decisions"]
            assert pending["question_message_id"] == question.idempotency_key
            assert "客人" not in json.dumps(own, ensure_ascii=False)
        assert _request(store, game).result["contact_started"] is True
        assert not _answer(store, "host", questions["host"]).allowed
        assert not late_arrival_context(store, conversation_id=_private("host"), sender_id="guest", at=delivered_at)
    reply_at = delivered_at + timedelta(minutes=1)
    with using_clock(lambda: reply_at):
        _input(store, "host", "可以", "answer-msg:host", "answer:host", reply_at)
        host = _answer(store, "host", questions["host"])
        assert host.result["late_arrival_decision"]["status"] == "pending"
        assert not _tasks(store, LATE_ARRIVAL_UPDATE_TASK_TYPE)
        _input(store, "peer", "我同意", "answer-msg:peer", "answer:peer", reply_at)
        assert not _answer(store, "peer", questions["host"]).allowed
        peer = _answer(store, "peer", questions["peer"])
        assert peer.result["late_arrival_decision"]["status"] == "accepted"
        assert not _answer(store, "peer", questions["peer"]).state_transitions
        update, = _tasks(store, LATE_ARRIVAL_UPDATE_TASK_TYPE)
        assert update.subject_id == "guest"
        assert update.payload["decision"] == "accepted"
        assert late_arrival_seat_status(store, game=game, customer_id="guest", at=reply_at) == "pending"
        _receipt(store, update)
        assert late_arrival_seat_status(store, game=game, customer_id="guest", at=reply_at) == "accepted"
        assert late_arrival_seat_status(store, game=game, customer_id="other", at=reply_at) is None
        assert late_arrival_seat_status(store, game=game, customer_id="guest", at=ETA) == "stale"
        expired_context = late_arrival_context(store, conversation_id=_private("guest"), sender_id="guest", at=ETA)
        old_request, = expired_context["current_late_arrival_requests"]
        assert old_request["game_id"] == game.game_id and old_request["status"] == "arrival_elapsed"
        assert store.require_game(game.game_id).to_dict() == before

    confirmed_at = START + timedelta(minutes=20)
    with using_clock(lambda: confirmed_at):
        with pytest.raises(LateParticipationRequiresConfirmation) as wrong_candidate:
            store.join_game(game_id=game.game_id, customer_id="other", display_name="另一位客户", trace_id="foreign")
        assert wrong_candidate.value.status == "unverified"
        with pytest.raises(LateParticipationRequiresConfirmation) as two_seats:
            store.join_game(game_id=game.game_id, customer_id="guest", display_name="客人", seat_count=2, trace_id="two")
        assert two_seats.value.status == "seat_count_not_approved"
        assert store.require_game(game.game_id).to_dict() == before

        _input(store, "guest", "那我六点半到，给我留一个。", "guest-confirmed", "candidate-join", confirmed_at)
        joined = ToolGateway(store).execute(ToolCall("join_game", {
            "game_id": game.game_id, "customer_id": "guest", "display_name": "客人", "seat_count": 1,
        }), trace_id="candidate-join", conversation_id=_private("guest"), sender_id="guest", sender_name="客人",
            source_message_id="guest-confirmed", step_index=1)
        assert joined.called and joined.allowed and joined.error is None
        persisted = store.require_game(game.game_id)
        assert persisted.planned_start_at == START and persisted.actual_started_at is None
        assert persisted.remaining_seats() == 1
        assert [person.status for person in persisted.participants if person.customer_id == "guest"] == ["confirmed"]


@pytest.mark.parametrize("condition", ["no_search", "ambiguous", "other_trace", "no_channel", "past_eta", "group_source"])
def test_late_arrival_denies_untrusted_source_or_unworkable_request(store, condition):
    with using_clock(lambda: TOUCH):
        game = _seed(store, peer=condition == "no_channel", reachable=condition != "no_channel")
        _input(store, "guest", "我六点半到，能问问吗？", "ask-msg", "inquiry", TOUCH,
               metadata={"source": "group", "room_id": ROOM} if condition == "group_source" else None)
        if condition != "no_search":
            _query(store, game, "unrelated" if condition == "other_trace" else "inquiry",
                   other_match=condition == "ambiguous")
        result = _request(store, game, eta=TOUCH if condition == "past_eta" else ETA)
        assert not result.allowed and not result.called and not result.state_transitions
        assert result.result["contact_started"] is False
        assert not _tasks(store, LATE_ARRIVAL_QUESTION_TASK_TYPE)


def test_wrong_recipient_receipt_cannot_enable_answer(store):
    with using_clock(lambda: TOUCH):
        game = _seed(store)
        _input(store, "guest", "我六点半到，问问这桌吧。", "ask-msg", "inquiry", TOUCH)
        _query(store, game, "inquiry")
        assert _request(store, game).allowed
        question, = _tasks(store, LATE_ARRIVAL_QUESTION_TASK_TYPE)
        _receipt(store, question, recipient="other")
        _input(store, "host", "可以", "answer-msg:host", "answer:host", TOUCH)
        assert not _answer(store, "host", question).allowed
        assert not _tasks(store, LATE_ARRIVAL_UPDATE_TASK_TYPE)


def test_new_arrival_proposal_supersedes_old_questions_at_same_clock(store):
    with using_clock(lambda: TOUCH):
        game = _seed(store)
        _input(store, "guest", "六点半我能到，问问吧", "old-ask", "inquiry", TOUCH)
        _query(store, game, "inquiry")
        assert _request(store, game).allowed
        old, = _tasks(store, LATE_ARRIVAL_QUESTION_TASK_TYPE)
        _receipt(store, old)
        new_eta = ETA + timedelta(minutes=10)
        _input(store, "guest", "来不及，问下六点四十行么", "new-ask", "revised-inquiry", TOUCH)
        _query(store, game, "revised-inquiry")
        assert _request(store, game, trace_id="revised-inquiry", eta=new_eta).allowed
        current = next(task for task in _tasks(store, LATE_ARRIVAL_QUESTION_TASK_TYPE) if task.task_id != old.task_id)
        assert late_arrival_delivery_context(store, old, at=TOUCH) is None
        assert late_arrival_delivery_context(store, current, at=TOUCH)["arrival_at"] == new_eta
        own = late_arrival_context(store, conversation_id=_private("guest"), sender_id="guest", at=TOUCH)
        assert [row["arrival_at"] for row in own["current_late_arrival_requests"]] == [new_eta.isoformat()]
        _input(store, "host", "可以", "answer-msg:host", "answer:host", TOUCH)
        assert not _answer(store, "host", old).allowed
        assert not _tasks(store, LATE_ARRIVAL_UPDATE_TASK_TYPE)


def test_revoked_host_consent_cannot_support_seat_after_result_delivery(store):
    with using_clock(lambda: TOUCH):
        game = _seed(store)
        _input(store, "guest", "六点半能到，问下还能补不", "ask-msg", "inquiry", TOUCH)
        _query(store, game, "inquiry")
        assert _request(store, game).allowed
        question, = _tasks(store, LATE_ARRIVAL_QUESTION_TASK_TYPE)
        _receipt(store, question)
    at = TOUCH + timedelta(minutes=1)
    with using_clock(lambda: at):
        _input(store, "host", "六点半可以", "answer-msg:host", "answer:host", at)
        assert _answer(store, "host", question).allowed
        update, = _tasks(store, LATE_ARRIVAL_UPDATE_TASK_TYPE)
        _receipt(store, update)
        assert late_arrival_seat_status(store, game=game, customer_id="guest", at=at) == "accepted"
        assert store.invalidate_message_context(_private("host"), "host", "answer-msg:host", trace_id="recall")
        assert late_arrival_seat_status(store, game=game, customer_id="guest", at=at) != "accepted"
