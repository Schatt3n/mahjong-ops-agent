"""验收固定原话重放是否能区分已发邀约的真实关闭与未来通知。"""

import json

import pytest

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.models import InviteStatus
from scripts.replay_venue_consent import _seed_store
from scripts.replay_venue_invite_decline import (
    _CUSTOMER, _INVITE_ID, _NEIGHBOR, _execute_replay, _load_fixture, _source,
)

from .sim_adapter import build_runtime
from .venue_actors import SemanticJudge
from .venue_channel import SandboxChannel
from .venue_fixture import VenueFixtureAgent, _call, _reply


@pytest.mark.parametrize("decline, schedule, cite_reply, decline_only, expected", [
    (True, False, True, False, "passed"),
    (True, True, True, False, "passed"),
    (False, True, True, False, "failed"),
    (False, False, True, False, "failed"),
    (True, False, False, False, "inconclusive"),
    (True, False, True, True, "passed"),
    (True, True, True, True, "failed"),
])
def test_invite_decline_replay_checks_original_draft_independent_of_future_notice(
    tmp_path, decline, schedule, cite_reply, decline_only, expected,
):
    """即使判定通过或未来已有安排，漏掉当前邀约也仍是硬失败。"""
    data = _load_fixture()
    source = _source(data, decline_only=decline_only)
    turn, = source.turns
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    game_id = ""

    class ReplayAgent(VenueFixtureAgent):
        def complete(self, messages, *, trace_id, timeout_seconds):
            """桩只选择动作，权限、交付和持久化仍通过正式运行链路。"""
            context = self._context_payload(messages)
            previous = context.get("turn_tool_evidence") or context.get("previous_tool_results", ())
            calls = {tool.get("name") for tool in previous}
            if decline and "record_candidate_reply" not in calls:
                return _call("record_candidate_reply", {
                    "game_id": game_id, "customer_id": _CUSTOMER, "display_name": source.identities[_CUSTOMER],
                    "status": "declined",
                })
            if schedule and "schedule_follow_up" not in calls:
                return _call("schedule_follow_up", {
                    "goal": "复查下次杭麻2底的牌局", "waiting_for": "players",
                    "due_at": "2026-11-25T19:30:00+08:00", "expires_at": "2026-11-26T12:00:00+08:00",
                })
            if decline_only:
                return _reply("好，那这桌就不算你啦。")
            return _reply("好，这桌不算你。下次想打的话，通常哪天方便？", status="waiting_user")

    class Judgment:
        def complete(self, messages, **_options):
            """模拟结构化判定，验证它看到的是实际交付和重开的数据库。"""
            evidence = json.loads(messages[-1]["content"])
            persisted, invitation, customer, reply = evidence["evidence"]
            assert persisted["invitation_status"] == ("declined" if decline else "sent")
            notices = persisted["future_arrangements"]
            assert bool(notices) == schedule
            if schedule:
                assert notices[0]["kind"] == "conversation_follow_up"
                assert notices[0]["waiting_for"] == "players"
            assert customer["text"] == data["adjacent_reply" if decline_only else "customer_reply"]
            assert invitation["text"] == data["invite"]
            assert reply["speaker"] == "shop_reply"
            citations = [persisted["evidence_id"], customer["evidence_id"]]
            if cite_reply:
                citations.append(reply["evidence_id"])
            return json.dumps({"verdict": "pass", "findings": [{
                "check": evidence["checks"][0], "verdict": "pass", "evidence_ids": citations,
                "reason": "离线合同桩。",
            }]})

    runtime, _ = build_runtime(store, "mock", seed=source.seed, offline_client=ReplayAgent(seed=source.seed),
                               local_coordination=True)
    channel = SandboxChannel(runtime, room_id="venue-group", identities=source.identities,
                             archive_path=tmp_path / "chat.sqlite3")
    game_id = _seed_store(store, channel, source)
    result = _execute_replay(store, channel, source, game_id, tmp_path / "events.jsonl",
                             fixture=data, judge=SemanticJudge(Judgment()), decline_only=decline_only)
    assert result["status"] == expected, result
    assert result["checks"]["decline_persisted"] is decline
    assert result["checks"]["decline_tool_recorded_on_original_invite"] is decline
    assert result["checks"]["table_and_seats_preserved"]
    assert result["checks"]["unrelated_invites_and_rooms_preserved"]
    sibling, = (draft for draft in store.invite_drafts.values() if draft.customer_id == _NEIGHBOR)
    assert sibling.status == InviteStatus.SENT
    assert [row["message_id"] for row in channel.events_for(_CUSTOMER)][:2] == [_INVITE_ID, turn.message_id]

    before_duplicate = store.invite_drafts
    duplicate = channel.send(turn)
    assert duplicate["trace_id"] == result["trace_id"]
    assert store.invite_drafts == before_duplicate
    assert len(channel.events()) == 4
