"""复现已确认顾客商量晚到时错误释放席位，并验证模型工具的事务边界。"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import json
from pathlib import Path
from threading import Barrier

import pytest

from mahjong_agent_runtime import AgentRuntime, InMemoryAgentStore, SQLiteAgentStore, ToolCall, ToolGateway, UserMessage
from mahjong_agent_runtime.models import using_clock


FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "eval" / "golden" /
                      "venue_confirmed_seat_eta_replay.json").read_text(encoding="utf-8"))
VENUE_TIME = datetime.fromisoformat("2026-11-22T18:58:30+08:00")


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryAgentStore()
    else:
        db = SQLiteAgentStore(tmp_path / "seat-negotiation.sqlite3")
        yield db
        db._connection.close()


def _seed_game(store, *, confirmed):
    expected = FIXTURE["expected"]
    start_at = VENUE_TIME.replace(hour=19, minute=0, second=0)
    game, _ = store.create_game(
        conversation_id="private:host", organizer_id="host", organizer_name="店友甲",
        requirement={"game_type": expected["game_type"], "stake": "1", "smoke_preference": "no_smoking",
                     "start_time": expected["start_time"], "planned_start_at": start_at.isoformat(),
                     "start_time_kind": "scheduled", "duration_hours": expected["duration_hours"], "duration_kind": "fixed"},
        known_players=[{"customer_id": "host", "display_name": "店友甲"}], trace_id="setup",
    )
    invites, _ = store.create_invite_drafts(game_id=game.game_id,
        invitations=[{"customer_id": "guest", "display_name": "牌友", "message": FIXTURE["delivered_shop_message"]}],
        trace_id="setup_invitation")
    for peer in ("player_a", "player_b"):
        store.join_game(game_id=game.game_id, customer_id=peer, display_name=peer, trace_id=f"setup:{peer}")
    if confirmed:
        store.join_game(game_id=game.game_id, customer_id="guest", display_name="牌友", trace_id="setup:guest")
    return game.game_id, invites[0].draft_id


def _reply_arguments(game_id, status):
    return {"game_id": game_id, "customer_id": "guest", "display_name": "牌友", "status": status, "seat_count": 1}


def _action(call=None):
    return {
        "goal": "回答原桌是否能延迟开场", "objective_status": "completed" if call is None else "needs_tool",
        "reasoning_summary": "以实际席位和开场为准", "reply_to_user": "你这席还在，七点半开得先问其他人。" if call is None else "",
        "tool_calls": [] if call is None else [call], "needs_human": False,
        "stop_reason": {"can_stop": call is None, "depends_on_tool_results": call is not None,
                        "pending_work": [] if call is None else ["核对参与状态"], "why": "按工具结果决定"},
    }


class _Planner:
    """先重现源档的错误工具动作，再检查新事实回喂给模型。"""

    def __init__(self, game_id):
        self.game_id = game_id
        self.contexts = []

    def complete(self, messages, **_):
        context = json.loads(messages[-1]["content"])
        self.contexts.append(context)
        call = None
        if len(self.contexts) == 1:
            call = {"name": "record_candidate_reply", "reason": "模拟源档对晚到的错误分类",
                    "arguments": _reply_arguments(self.game_id, FIXTURE["expected"]["rejected_status"])}
        return json.dumps(_action(call), ensure_ascii=False)


def _deliver(store, status, game_id, *, source="direct"):
    return ToolGateway(store).execute(ToolCall(name="record_candidate_reply", arguments=_reply_arguments(game_id, status)),
        trace_id=f"trace:{source}", conversation_id="private:guest", sender_id="guest", sender_name="牌友",
        step_index=1, source_message_id=source)


def test_frozen_eta_tool_attempt_does_not_reopen_a_ready_table(store):
    with using_clock(lambda: VENUE_TIME):
        game_id, invite_id = _seed_game(store, confirmed=True)
        planner = _Planner(game_id)
        runtime = AgentRuntime(store=store, llm_client=planner)
        message = UserMessage(conversation_id="private:guest", sender_id="guest", sender_name="牌友",
                              text=FIXTURE["customer_eta_message"], message_id="frozen-eta", sent_at=VENUE_TIME)
        context = runtime.task_context_manager.prepare(message, trace_id="before-eta")
        with using_clock(lambda: VENUE_TIME - timedelta(seconds=1)):
            store.append_assistant_turn("private:guest", FIXTURE["delivered_shop_message"], "shop-receipt",
                                        metadata={"delivery_status": "delivered", "task_context_id": context.context.task_context_id})
        before = store.require_game(game_id).to_dict()
        invite_before = store.invite_drafts[invite_id].to_dict()
        result = runtime.handle_user_message(message)
        blocked, = [tool for tool in result.tool_results if tool.name == "record_candidate_reply"]
        assert not blocked.called and not blocked.allowed and not blocked.state_transitions
        assert blocked.result["validation_error"]["code"] == "claimed_seat_reply_conflict"
        assert "claimed_seat_reply_conflict" in json.dumps(planner.contexts[1]["turn_tool_evidence"], ensure_ascii=False)
        assert result.final_reply == "你这席还在，七点半开得先问其他人。"
        assert store.require_game(game_id).to_dict() == before
        assert store.invite_drafts[invite_id].to_dict() == invite_before
        expected = FIXTURE["expected"]
        current = store.require_game(game_id)
        assert current.status.value == "ready" and current.remaining_seats() == expected["remaining_seats"]
        assert next(item.status for item in current.participants if item.customer_id == "guest") == expected["customer_status"]
        if isinstance(store, SQLiteAgentStore):
            reopened = SQLiteAgentStore(store.path)
            try:
                assert reopened.require_game(game_id).to_dict() == before
                assert reopened.invite_drafts[invite_id].to_dict() == invite_before
            finally:
                reopened._connection.close()


def test_no_reply_cannot_release_a_seat_and_explicit_withdrawal_still_can(store):
    with using_clock(lambda: VENUE_TIME):
        game_id, _ = _seed_game(store, confirmed=True)
        no_reply = _deliver(store, "no_reply", game_id, source="unanswered-follow-up")
        assert not no_reply.called and not no_reply.allowed
        assert store.require_game(game_id).remaining_seats() == 0
        leave = _deliver(store, "declined", game_id, source="explicit-leave")
        assert leave.called and leave.allowed and leave.result["recorded_status"] == "declined"
        assert store.require_game(game_id).remaining_seats() == 1


def test_unconfirmed_invite_still_accepts_negotiation(store):
    with using_clock(lambda: VENUE_TIME):
        game_id, invite_id = _seed_game(store, confirmed=False)
        result = _deliver(store, "negotiating", game_id)
        assert result.called and result.allowed and result.result["recorded_status"] == "negotiating"
        assert store.invite_drafts[invite_id].status.value == "negotiating"
        assert store.require_game(game_id).remaining_seats() == 1


def test_model_negotiation_cannot_overwrite_concurrent_confirmation(store):
    with using_clock(lambda: VENUE_TIME):
        game_id, _ = _seed_game(store, confirmed=False)
    other = SQLiteAgentStore(store.path) if isinstance(store, SQLiteAgentStore) else store
    barrier = Barrier(2)

    def reply(target, status):
        with using_clock(lambda: VENUE_TIME):
            barrier.wait(timeout=5)
            return _deliver(target, status, game_id, source=status)

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(reply, store, "confirmed")
            second = executor.submit(reply, other, "negotiating")
            confirmation, negotiation = first.result(), second.result()
        assert confirmation.called and confirmation.allowed
        if not negotiation.called:
            assert negotiation.result["validation_error"]["code"] == "claimed_seat_reply_conflict"
        with using_clock(lambda: VENUE_TIME):
            current = store.require_game(game_id)
            assert current.status.value == "ready" and current.remaining_seats() == 0
            assert next(item.status for item in current.participants if item.customer_id == "guest") == "confirmed"
    finally:
        if other is not store:
            other._connection.close()
