"""离线核对凌晨私聊约早场、群发招募与主动私邀的时间边界。"""

from dataclasses import replace
from datetime import datetime, timedelta
import json

import pytest

from mahjong_agent_runtime import AgentRuntime, SQLiteAgentStore
from mahjong_agent_runtime.domains.customer_play_preference import completed_play_attendees
from mahjong_agent_runtime.models import RecruitmentStatus, ScheduledTaskStatus, using_clock

from .sim_adapter import StaticAgentLLMClient
from .venue_actors import ActorAction
from .venue_channel import AGENT_ID
from .venue_fixture import _call, _reply
from .venue_overnight_morning import OVERNIGHT_MORNING_CASES, VenueOvernightMorning


_TERMS = {"game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "no_smoking"}


class _OvernightShop(StaticAgentLLMClient):
    """固定决策合约测试接线，熟客分支只能从真正送入规划器的画像取得条件。"""

    def __init__(self, case) -> None:
        super().__init__()
        self.case = case
        self.seen_profiles = []

    def complete(self, messages, *, trace_id, timeout_seconds):
        context = self._context_payload(messages)
        message = context.get("current_message") or {}
        if message.get("message_id") != f"customer:{self.case.case_id}":
            return super().complete(messages, trace_id=trace_id, timeout_seconds=timeout_seconds)
        previous = context.get("previous_tool_results") or []
        profile = context.get("sender_profile") or {}
        if self.case.returning:
            learned = profile["attendance_preference"]
            terms = {key: str(learned[key]) for key in _TERMS}
        else:
            terms = _TERMS
        window_start, window_end = self.case.opening_window
        if not previous:
            self.seen_profiles.append(profile)
            return _call("search_current_games", {"lookup_scope": "personal", "requirement": {
                **terms, "start_time_range": {"start_at": window_start.isoformat(), "end_at": window_end.isoformat()},
            }})
        last = previous[-1]
        if "result_reference" in last:
            last = context["turn_tool_evidence"][last["result_reference"]["index"]]
        if last["name"] == "search_current_games" and last["called"] and last["allowed"] and not last["error"]:
            assert last["result"]["matches"] == []
            opened = window_start + timedelta(minutes=30)
            return _call("create_game", {
                "organizer_id": message["sender_id"], "organizer_name": message["sender_name"],
                "requirement": {**terms, "start_time_kind": "scheduled", "planned_start_at": opened.isoformat()},
            })
        if last["name"] == "create_game" and last["called"] and last["allowed"] and not last["error"]:
            date = "明天" if self.case.days_ahead else "今天"
            return _reply(f"{date}早上九点半先给你组上了，还差三位。")
        return _reply("我再确认一下。")


class _OvernightReview:
    """为真实的只读时间审核及独立建局审核提供可核对的受控判断。"""

    def __init__(self, case, *, delegate_choice: bool = True) -> None:
        self.case = case
        self.delegate_choice = delegate_choice

    def complete(self, messages, *, trace_id, timeout_seconds):
        evidence = json.loads(messages[-1]["content"])
        ref, = (row["ref"] for row in evidence["conversation"] if row.get("current"))
        if "proposed_table" in evidence:
            return json.dumps({"scope": "create", "customer_ref": ref, "shop_offer_ref": None})
        start, end = self.case.opening_window
        candidates = evidence["current_clock_observation"]["candidates"]
        assert [item["clock_text"] for item in candidates] == ["九点", "十点"]
        return json.dumps({
            "clock_role": "other", "clock_roles": [{"clock_ref": item["clock_ref"], "role": "opening"}
                                                  for item in candidates],
            "lookup_scope": "personal", "opening_window": {"start_at": start.isoformat(), "end_at": end.isoformat()},
            "opening_dayparts": {"date": start.date().isoformat(), "parts": ["morning"], "source_ref": ref},
            "accepted_stake_range": None, "known_cap_score": None,
            "has_fixed_opening_authority": self.delegate_choice,
            "opening_customer_ref": ref if self.delegate_choice else None,
            "current_ref": ref, "supporting_customer_refs": [ref],
        })


@pytest.mark.parametrize("case", OVERNIGHT_MORNING_CASES, ids=lambda case: case.case_id)
def test_overnight_morning_private_request_publishes_public_board_before_private_window(case, tmp_path):
    """九至十点授权内落一桌，仅本人先收回执，群友在凌晨看到正式招募。"""
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    shop = _OvernightShop(case)
    review = _OvernightReview(case)
    runtime = AgentRuntime(llm_client=shop, store=store, table_availability_client=review,
                           creation_authorization_client=review)
    scenario = VenueOvernightMorning(case, runtime, tmp_path)
    customer_id = scenario.customer.sender_id
    observer_id = scenario.same_name_friend.sender_id
    initial_context = scenario.opening_context().to_dict()
    assert initial_context["delivery"] is None
    assert ("familiar_shop" in initial_context) is case.returning
    assert not store.customers[customer_id].profile_facts
    assert not store.completed_game_preference(observer_id)
    assert not scenario.channel.events_for(observer_id)
    if case.returning:
        assert len(scenario.history_game_ids) == 3
        assert all(customer_id in completed_play_attendees(store.require_game(game_id))
                   for game_id in scenario.history_game_ids)
        assert all(word not in case.deterministic_message for word in ("杭麻", "一块", "无烟"))
    else:
        assert not store.completed_game_preference(customer_id)
        assert not scenario.channel.events()
    assert scenario.send_opening(ActorAction("wait")) is None

    result = scenario.send_opening(ActorAction("send", case.deterministic_message, "private", AGENT_ID))
    search, creation = (tool for tool in result["tool_results"] if tool["called"] and tool["allowed"] and not tool["error"])
    assert (search["name"], creation["name"]) == ("search_current_games", "create_game")
    assert search["result"]["matches"] == []
    start, end = case.opening_window
    assert search["result"]["requirement"]["start_time_range"] == {
        "start_at": start.isoformat(), "end_at": end.isoformat(),
    }
    profile, = shop.seen_profiles
    assert ("attendance_preference" in profile) is case.returning
    if case.returning:
        assert profile["attendance_preference"] == {**_TERMS, "completed_games": 3}

    game_id = creation["result"]["game"]["game_id"]
    persisted = SQLiteAgentStore(store.path)
    try:
        game = persisted.require_game(game_id)
        assert set(persisted.games) - set(scenario.history_game_ids) == {game_id}
    finally:
        persisted._connection.close()
    assert game.conversation_id == scenario.channel.private_conversation_id(customer_id)
    assert game.organizer_id == customer_id and game.remaining_seats() == 3
    assert {key: game.requirement[key] for key in _TERMS} == _TERMS
    assert game.planned_start_at == start + timedelta(minutes=30)
    assert game.planned_start_at.date() == (case.request_at.date() + timedelta(days=case.days_ahead))
    assert game.recruitment_status == RecruitmentStatus.SCHEDULED

    receipt, = result["archive_outputs"]
    assert (receipt["kind"], receipt["recipient_id"], receipt["source_message_id"]) == (
        "private", customer_id, f"customer:{case.case_id}",
    )
    assert receipt["text"] and "九点半" in receipt["text"]
    assert not scenario.channel.events_for(observer_id)
    assert not store.invite_drafts and not store.outbound_message_drafts
    assert not store.room_reservations or all(res.game_id != game_id for res in store.room_reservations.values())

    due = scenario.board.next_due()
    assert due is not None and due.date() == case.request_at.date() and 1 <= due.hour < 6
    assert scenario.board.publish_due(due - timedelta(seconds=1))["status"] == "no_due_task"
    board_result = scenario.board.publish_due(due)
    board, = board_result["archive_outputs"]
    snapshot, = board_result["snapshots"]
    item, = snapshot["items"]
    assert board_result["status"] == "completed"
    assert board["message_id"] == snapshot["external_message_id"] and item["game_id"] == game_id
    assert (board["kind"], board["recipient_id"], board["conversation_id"]) == (
        "group", scenario.channel.room_id, scenario.channel.group_conversation_id,
    )
    assert datetime.fromisoformat(board["sent_at"]) == due
    assert game.planned_start_at.strftime("%Y-%m-%d %H:%M") in item["rendered_text"]
    assert "夜班" not in board["text"]
    assert "帮我组" not in board["text"] and "老样子" not in board["text"]
    assert scenario.channel.events_for(observer_id) == [board]

    activation = store.scheduled_task_for_game(game_id)
    assert activation is not None and activation.status == ScheduledTaskStatus.PENDING
    assert due < activation.due_at == game.planned_start_at - timedelta(hours=2)
    last_pre_window = activation.due_at - timedelta(seconds=1)
    with using_clock(lambda: last_pre_window), pytest.raises(ValueError, match="private candidate outreach is not open yet"):
        store.create_invite_drafts(
            game_id=game_id, trace_id="overnight-premature-outreach",
            invitations=[{"customer_id": observer_id, "display_name": scenario.same_name_friend.name,
                          "message_text": "九点半有桌杭麻，一块底无烟，你要来吗？"}],
        )
    assert not store.invite_drafts and not store.outbound_message_drafts


def test_overnight_morning_range_without_delegation_does_not_commit_a_fixed_start(tmp_path):
    """顾客能打的范围可用于查桌，等客户选好开场再写固定预约。"""
    case = replace(
        OVERNIGHT_MORNING_CASES[0],
        deterministic_message="想请你帮我组桌杭麻，今天早上九点到十点我都有空。"
                              "一块底，无烟。先别定几点开，有合适的告诉我。",
    )
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    review = _OvernightReview(case, delegate_choice=False)
    runtime = AgentRuntime(llm_client=_OvernightShop(case), store=store, table_availability_client=review,
                           creation_authorization_client=review)
    scenario = VenueOvernightMorning(case, runtime, tmp_path)

    result = scenario.send_opening(ActorAction("send", case.deterministic_message, "private", AGENT_ID))
    search, creation = result["tool_results"]
    assert search["name"] == "search_current_games" and search["called"] and search["allowed"]
    assert creation["name"] == "create_game" and not creation["called"] and not creation["allowed"]
    assert creation["result"]["validation_error"]["code"] == "verified_table_availability_conflict"
    assert not store.games and scenario.board.next_due() is None
    assert not store.invite_drafts and not store.outbound_message_drafts
