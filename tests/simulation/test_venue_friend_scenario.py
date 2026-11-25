"""验收昵称相同也可分别私聊开局，并持久化各自的入站授权和组局。"""

from dataclasses import replace
from datetime import datetime
import json

from mahjong_agent_runtime import JsonlTraceRecorder, SQLiteAgentStore
from mahjong_agent_runtime.models import DEFAULT_TZ

from .sim_adapter import build_runtime
from .venue_actors import MockActor, generate_personas
from .venue_channel import AGENT_ID, SandboxChannel
from .venue_fixture import VenueFixtureAgent
from .venue_friend_scenario import VenueFriendScenario
from .venue_write_guard import VenueWriteGuard


class _OfflineOrganizerAuthorization:
    """仅用于验收真实写入网关与已归档原话的接线。"""

    def complete(self, messages, *, trace_id, timeout_seconds):
        """同一条合成私聊明确提出新局时，保留它的原文引用。"""
        text = json.loads(messages[-1]["content"])["current"]["text"]
        return json.dumps({"permissions": [{
            "scope": "organize", "evidence": text, "offer_message_id": None, "offer_evidence": None,
        }] if "新" in text else []}, ensure_ascii=False)


def test_existing_friends_with_same_nickname_create_separate_games_via_their_private_chats(tmp_path):
    """没有前置群消息或现成牌局，好友先私聊再答条件，正式局写回各自会话。"""
    people = tuple(replace(person, name="周末来打牌") for person in generate_personas(42, count=2))
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    guard = VenueWriteGuard(_OfflineOrganizerAuthorization(), tmp_path / "write_authorization.jsonl")
    fixture = VenueFixtureAgent(seed=42)
    runtime, _ = build_runtime(store, "mock", offline_client=fixture, local_coordination=True,
                               trace_recorder=JsonlTraceRecorder(tmp_path / "agent_trace.log"),
                               tool_gateway=guard.guarded_gateway(store))
    channel = SandboxChannel(runtime, room_id="venue-group", identities={person.sender_id: person.name for person in people},
                             archive_path=tmp_path / "chat.sqlite3", write_guard=guard)
    scenario = VenueFriendScenario(
        people=people, actor=MockActor(42), judge=None, channel=channel, store=store, run_dir=tmp_path,
        start_at=datetime(2026, 11, 25, 19, tzinfo=DEFAULT_TZ), seed=42, background_events=0, fixture_agent=fixture,
        active_users=2, workers=2, dialogue_turns=3, judge_users=2,
    )
    assert not store.games
    assert all(not item.preferred_games and not item.profile_facts for item in store.customers.values())
    report = scenario.run()

    assert all(report["checks"].values()), report
    assert report["private_initiation"] == {
        "owner_friends": 2, "private_openers": 2, "verified_organizers": 2, "persisted_new_games": 2,
        "forming_games": 2, "recruiting_games": 0, "scoped_creation_evidence_required": True,
    }
    persisted = SQLiteAgentStore(store.path).games
    assert {game.organizer_id for game in persisted.values()} == {person.sender_id for person in people}
    assert {game.organizer_name for game in persisted.values()} == {"周末来打牌"}
    assert {game.conversation_id for game in persisted.values()} == {channel.private_conversation_id(p.sender_id) for p in people}
    assert {game.planned_start_at.strftime("%H:%M") for game in persisted.values()} == {"19:00", "19:30"}
    archive = channel.events()
    assert all(row["kind"] == "private" for row in archive)
    assert sum(row["recipient_id"] == AGENT_ID for row in archive) == 4
    for person in people:
        visible = channel.events_for(person.sender_id)
        assert len(visible) == 4
        assert {row["conversation_id"] for row in visible} == {channel.private_conversation_id(person.sender_id)}
