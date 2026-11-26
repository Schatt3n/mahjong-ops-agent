"""验证迟回报名只依赖本人真实收到、独立核验且仍有效的一桌。"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from mahjong_agent_runtime import AgentRuntime, InMemoryAgentStore, SQLiteAgentStore, StaticAgentClient, UserMessage
from mahjong_agent_runtime.domains.delivered_game_offer import DELIVERED_GAME_OFFER, delivered_offer_matches_game
from mahjong_agent_runtime.models import using_clock

from tests.test_agent_runtime import action_json


FIRST_REPLY = "有的，17:30开，2-3元，这桌打到21:30。你18:30到、打到21:30没问题吧？行的话给你算上。"
LATE_ARRIVAL_REPLY = "17:30那桌还缺人，打到21:30。你18:30才到的话，得先问那桌能不能等。"
ON_TIME_OFFER = "19:30那桌还缺人，2到3底，晚上能来吗？"


def _at(clock):
    return datetime.fromisoformat(f"2026-11-26T{clock}:00+08:00")


def _message(text, event, clock, *, sender_id="guest", conversation_id="guest-private", metadata=None):
    return UserMessage(conversation_id=conversation_id, sender_id=sender_id, sender_name="客户", text=text,
                       message_id=event, sent_at=_at(clock), metadata=metadata or {})


def _runtime(store, main_outputs, reviews):
    reviewer = StaticAgentClient(reviews)
    model = StaticAgentClient(main_outputs)
    return AgentRuntime(llm_client=model, store=store, reply_self_review_enabled=True,
                        reply_self_review_client=reviewer), model, reviewer


def _review(text, *, game_id=None, reference_id=None):
    item = {"item_id": "reply_to_user", "approved": True, "suggested_safe_text": text,
            "follow_up_commitments": [], "reasoning_summary": "当前顾客可见", "violations": []}
    if game_id:
        item["reviewed_game_offer_id"] = game_id
    if reference_id:
        item["reviewed_game_reference_id"] = reference_id
    return json.dumps({"approved": True, "needs_human": False, "reasoning_summary": "核对可见内容",
                       "violations": [], "item_reviews": [item]}, ensure_ascii=False)


def _lookup(start="17:30"):
    return action_json(objective_status="needs_tool", tool_calls=[{
        "name": "search_current_games", "arguments": {"requirement": {
            "game_type": "hangzhou_mahjong", "stake_range": {"min": 2, "max": 3}, "start_time": start,
            "start_time_kind": "scheduled", "planned_start_at": _at(start).isoformat(),
        }}, "reason": "核对所指桌的当前局况",
    }])


def _join_tool(game_id):
    return action_json(objective_status="needs_tool", tool_calls=[{
        "name": "join_game", "arguments": {"game_id": game_id, "customer_id": "guest", "display_name": "客户"},
        "reason": "客户明确同意并已查到同一桌，登记本人一席",
    }])


def _offer(store, *, review_id="actual", deliver=True, metadata=None, add_second_match=False, include_proposal_id=True,
           discuss_only=False, start="17:30"):
    with using_clock(lambda: _at("11:39")):
        requirement = {"game_type": "hangzhou_mahjong", "stake_range": {"min": 2, "max": 3},
                       "duration_hours": 4, "planned_start_at": _at(start).isoformat()}
        game, _ = store.create_game(conversation_id="table-host-private", organizer_id="host",
            organizer_name="不能披露的店员备注", requirement=requirement,
            known_players=[{"customer_id": "host", "display_name": "不能披露的店员备注"}], trace_id="seed-table")
        if add_second_match:
            store.create_game(conversation_id="other-table-private", organizer_id="other-host", organizer_name="其他客户",
                              requirement=requirement, known_players=[], trace_id="seed-other-table")
        first_reply = {"17:30": FIRST_REPLY, "19:30": ON_TIME_OFFER}[start]
        if discuss_only:
            first_reply = LATE_ARRIVAL_REPLY
        status = "completed" if discuss_only else "waiting_user"
        reviewed_game_id = game.game_id if review_id == "actual" else review_id
        runtime, _, reviewer = _runtime(store, [
            _lookup(start),
            action_json(objective_status=status, reply_to_user=first_reply,
                        objective_state={"current_phase": "completed" if discuss_only else "wait_user",
                                         "active_game_id": game.game_id if include_proposal_id else None}),
        ], [_review(first_reply, **({"reference_id": reviewed_game_id} if discuss_only else {"game_id": reviewed_game_id}))])
        request = _message(f"{start}那桌2到3底的还凑人吗？我18:30能到。", "initial", "11:39", metadata=metadata)
        result = runtime.handle_user_message(request, trace_id="offer")
        assert result.final_reply == first_reply
        assert result.status == status
        assert game.remaining_seats() == 3
        if deliver:
            assert store.claim_assistant_reply_delivery(request.conversation_id, trace_id="offer")
            assert store.mark_assistant_reply_delivered(request.conversation_id, trace_id="offer")
        reviewer_input = json.loads(reviewer.calls[0]["messages"][1]["content"])
    return game.game_id, reviewer_input["review_items"][0]


@pytest.mark.parametrize("backend", ["memory", "reopened_sqlite"])
def test_late_acceptance_reuses_delivered_table_after_task_rotation(tmp_path, backend):
    """相隔近七小时，重新查实同一桌再经正式网关入席，旧对话不进入新任务。"""
    path = tmp_path / "accept.sqlite3"
    store = InMemoryAgentStore() if backend == "memory" else SQLiteAgentStore(path)
    game_id, review_item = _offer(store, include_proposal_id=backend == "memory", start="19:30")
    assert review_item["game_offer_candidate"]["game_id"] == game_id
    if backend == "reopened_sqlite":
        store._connection.close()
        store = SQLiteAgentStore(path)
    initial_tasks = set(store.scheduled_tasks)
    stamp = _at("18:35")
    with using_clock(lambda: stamp):
        runtime, model, reviewer = _runtime(store, [
            _lookup("19:30"), _join_tool(game_id), action_json(objective_status="completed", reply_to_user="好，给你记上了。"),
        ], [_review("好，给你记上了。")])
        reply = _message("晚上能来，这桌算我一个吧", "accepted", "18:35")
        result = runtime.handle_user_message(reply, trace_id="late-acceptance")
    prompt = json.loads(model.calls[0]["messages"][1]["content"])
    offer = prompt[DELIVERED_GAME_OFFER]
    assert offer["game_id"] == game_id
    assert offer["terms"]["planned_start_at"] == _at("19:30").isoformat()
    assert offer["terms"]["planned_end_at"] == _at("23:30").isoformat()
    assert ON_TIME_OFFER not in json.dumps(prompt, ensure_ascii=False)
    assert "不能披露的店员备注" not in json.dumps(prompt, ensure_ascii=False)
    assert prompt["recovered_task_contexts"] == []
    assert store.current_task_context(reply.conversation_id, reply.sender_id).reset_reason == "idle_task_timeout"
    assert [item.name for item in result.tool_results] == ["search_current_games", "join_game", "customer_visible_content_review"]
    assert result.final_reply == "好，给你记上了。"
    assert any(item.customer_id == reply.sender_id and item.status == "confirmed"
               for item in store.require_game(game_id).participants)
    assert len(store.games) == 1 and set(store.scheduled_tasks) == initial_tasks
    assert not store.invite_drafts and not store.outbound_message_drafts
    assert json.loads(reviewer.calls[0]["messages"][1]["content"])[DELIVERED_GAME_OFFER]["game_id"] == game_id
    if isinstance(store, SQLiteAgentStore):
        store._connection.close()


@pytest.mark.parametrize("backend", ["memory", "reopened_sqlite"])
def test_delayed_reply_rechecks_the_discussed_table_without_treating_it_as_an_offer(tmp_path, backend):
    """此前只确认是哪桌时，轮转后仍能重查；核对当前到店时间前不写席位。"""
    path = tmp_path / "discussed.sqlite3"
    store = InMemoryAgentStore() if backend == "memory" else SQLiteAgentStore(path)
    game_id, review_item = _offer(store, discuss_only=True, include_proposal_id=False)
    assert review_item["game_offer_candidate"]["game_id"] == game_id
    if backend == "reopened_sqlite":
        store._connection.close()
        store = SQLiteAgentStore(path)
    delivered = next(turn for turn in store.recent_turns("guest-private", 5)
                     if turn.metadata.get("delivery_status") == "delivered")
    reference = delivered.metadata[DELIVERED_GAME_OFFER]
    assert reference["game_reference_only"] is True
    assert not delivered_offer_matches_game(reference, game=store.require_game(game_id), recipient_id="guest")

    final_reply = "你现在到店了吗？这桌还在凑人。"
    with using_clock(lambda: _at("18:35")):
        runtime, model, _ = _runtime(store, [
            _lookup(), action_json(objective_status="waiting_user", reply_to_user=final_reply),
        ], [_review(final_reply)])
        message = _message("18:30到、打到21:30没问题，这桌算我一个吧", "late", "18:35")
        result = runtime.handle_user_message(message, trace_id="late-discussion")

    model_input = json.loads(model.calls[0]["messages"][1]["content"])
    context_reference = model_input[DELIVERED_GAME_OFFER]
    assert context_reference["game_reference_only"] is True
    assert context_reference["game_id"] == game_id
    assert LATE_ARRIVAL_REPLY not in json.dumps(model_input, ensure_ascii=False)
    assert store.current_task_context(message.conversation_id, message.sender_id).reset_reason == "idle_task_timeout"
    search, review = result.tool_results
    assert search.name == "search_current_games" and search.called and search.allowed
    assert [match["game"]["game_id"] for match in search.result["matches"]] == [game_id]
    assert review.name == "customer_visible_content_review"
    assert result.final_reply == final_reply
    assert store.require_game(game_id).remaining_seats() == 3
    if isinstance(store, SQLiteAgentStore):
        store._connection.close()


@pytest.mark.parametrize("condition", [
    "not_delivered", "not_verified", "unverified_id", "two_matches", "group_handoff",
    "other_conversation", "other_guest", "terms_changed", "expired", "filled",
])
def test_late_offer_is_absent_on_invalid_delivery_scope_and_live_state(condition):
    """丢弃未送达、不唯一、不属于本人或已变更的推荐，不把旧文本带进新任务。"""
    store = InMemoryAgentStore()
    review_id = {"not_verified": None, "unverified_id": "unrelated-table"}.get(condition, "actual")
    group_metadata = {"source": "group", "room_id": "another-room"} if condition == "group_handoff" else None
    game_id, item = _offer(store, review_id=review_id, deliver=condition != "not_delivered",
                           metadata=group_metadata, add_second_match=condition == "two_matches")
    if condition in {"group_handoff", "two_matches"}:
        assert "game_offer_candidate" not in item
    clock = "22:00" if condition == "expired" else "18:35"
    with using_clock(lambda: _at(clock)):
        if condition == "terms_changed":
            store.update_game_requirement(game_id=game_id, requirement_patch={"duration_hours": 3},
                                          reason="时长调整", trace_id="edit-table")
        if condition == "filled":
            with using_clock(lambda: _at("16:35")):
                store.join_game(game_id=game_id, customer_id="party", display_name="三位", seat_count=3, trace_id="fill")
        runtime, model, _ = _runtime(store, [action_json(objective_status="waiting_user", reply_to_user="你说哪桌？")],
                                    [_review("你说哪桌？")])
        reply = _message("这桌算我一个吧", "late", clock,
                         sender_id="stranger" if condition == "other_guest" else "guest",
                         conversation_id="other-private" if condition == "other_conversation" else "guest-private")
        result = runtime.handle_user_message(reply, trace_id="late")
        prompt = json.loads(model.calls[0]["messages"][1]["content"])
    assert DELIVERED_GAME_OFFER not in prompt
    assert all(item.name != "join_game" for item in result.tool_results)
    assert all(item.customer_id not in {"guest", "stranger"} for item in store.require_game(game_id).participants)


def test_delivered_offer_does_not_substitute_for_fresh_participation_access():
    """直接用历史推荐写席位仍由网关拒绝，换新任务后须先成功查询。"""
    store = InMemoryAgentStore()
    game_id, _ = _offer(store)
    with using_clock(lambda: _at("18:35")):
        runtime, model, _ = _runtime(store, [
            _join_tool(game_id), action_json(objective_status="waiting_user", reply_to_user="我先核实一下这桌。"),
        ], [_review("我先核实一下这桌。")])
        result = runtime.handle_user_message(_message("这桌算我一个吧", "late", "18:35"), trace_id="late")
    assert json.loads(model.calls[0]["messages"][1]["content"])[DELIVERED_GAME_OFFER]["game_id"] == game_id
    join = next(item for item in result.tool_results if item.name == "join_game")
    assert not join.called and not join.allowed and "participation access required" in join.error
    assert store.require_game(game_id).remaining_seats() == 3
