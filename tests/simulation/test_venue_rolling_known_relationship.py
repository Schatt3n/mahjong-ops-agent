"""验证顾客实际私聊告知的避桌约束只影响本人的当前查局。"""

from datetime import datetime, timedelta
import json

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.models import DEFAULT_TZ, using_clock

from .sim_adapter import build_runtime
from .venue_channel import AGENT_ID, OWNER_ID, MessageEnvelope, SandboxChannel
from .venue_fixture import VenueFixtureAgent, _call, _reply
from .venue_rolling_board import VenueRollingBoard


_MATCH_TERMS = {"game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "no_smoking"}
_SEARCH_MESSAGES = {"before-disclosure", "after-disclosure", "independent-search"}
_DISCLOSURE = "这次别安排我跟阿川同桌"
_PRIVATE_REASON = "家里刚有点事，我还没有告诉其他人"


class _RelationshipFixture(VenueFixtureAgent):
    """提出实际来话授权的工具动作，匹配与记忆交给正式 Runtime。"""

    def __init__(self):
        """保留三次查局时的模型输入，用于核对披露与跨会话边界。"""
        super().__init__(seed=42)
        self.search_contexts = {}

    def complete(self, messages, *, trace_id, timeout_seconds):
        """从实际工具返回读取匹配数，只给来问的顾客回应。"""
        context = self._context_payload(messages)
        message = context.get("current_message") or {}
        source = message.get("message_id")
        if source != "disclose-relationship" and source not in _SEARCH_MESSAGES:
            return super().complete(messages, trace_id=trace_id, timeout_seconds=timeout_seconds)
        previous = context.get("previous_tool_results") or []
        evidence = context.get("turn_tool_evidence") or []
        if source == "disclose-relationship":
            if not previous:
                return _call("record_user_memory", {"task_memories": [{
                    "customer_id": message["sender_id"], "memory_type": "relationship", "field": "avoid_playing",
                    "target_customer_id": "owner-a", "value": True, "evidence": _DISCLOSURE,
                    "confidence": 1, "scope": "current_task",
                }]})
            if previous[-1].get("allowed") and not previous[-1].get("error"):
                return _reply("好，这次我留意一下。")
        elif not previous:
            self.search_contexts[source] = context
            return _call("search_current_games", {"requirement": _MATCH_TERMS, "limit": 5})
        else:
            result = previous[-1]
            if "result_reference" in result:
                result = evidence[result["result_reference"]["index"]]
            if result.get("allowed") and not result.get("error"):
                matches = result["result"]["matches"]
                return _reply(f"我查到有{len(matches)}桌符合你这次的条件。")
        return _reply("这会儿我还没确认好。")


def _private(channel, *, source, sender, text, at, quote=None):
    """以相同的营业时间提交实际消息，回执必须归属于原顾客。"""
    with using_clock(lambda: at):
        result = channel.send(MessageEnvelope(source, sender, text, at, "private", OWNER_ID, quoted_message_id=quote))
    reply, = result["archive_outputs"]
    assert (reply["kind"], reply["sender_id"], reply["recipient_id"], reply["conversation_id"],
            reply["source_message_id"]) == ("private", AGENT_ID, sender, channel.private_conversation_id(sender), source)
    return result, reply


def _successful_tool(response, name):
    """业务证据来自正式 Tool Gateway 的成功返回。"""
    tool, = [item for item in response["tool_results"] if item["name"] == name]
    assert tool["called"] and tool["allowed"] and not tool.get("error"), tool
    return tool["result"]


def test_only_disclosed_current_task_avoidance_filters_exact_compatible_tables(tmp_path):
    """同样条件先查到两桌，私聊告知后只排除指定桌，另一顾客仍查到两桌。"""
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    fixture = _RelationshipFixture()
    runtime, _ = build_runtime(store, "mock", offline_client=fixture, local_coordination=True)
    members = {"owner-a": "阿川", "owner-b": "阿荷", "seeker": "小叶", "observer": "小方"}
    channel = SandboxChannel(runtime, room_id="venue-group", identities=members, archive_path=tmp_path / "chat.sqlite3")
    opened = datetime(2026, 11, 25, 18, 30, tzinfo=DEFAULT_TZ)
    start = opened.replace(hour=19, minute=0)
    game_ids = {}
    for offset, owner in enumerate(("owner-a", "owner-b")):
        message_id = f"create:{owner}"
        fixture.turns[message_id] = ("need", {"requirement": {
            **_MATCH_TERMS, "start_time_kind": "scheduled", "start_time": "19:00",
            "planned_start_at": start.isoformat(), "duration_kind": "fixed", "duration_hours": 2,
        }})
        created, _ = _private(channel, source=message_id, sender=owner, at=opened + timedelta(seconds=offset),
                              text="老板帮我组今晚七点的一块底无烟杭麻，打两小时，先算我一个。")
        game = _successful_tool(created, "create_game")["game"]
        assert game["organizer_id"] == owner
        game_ids[owner] = game["game_id"]
    assert len(set(game_ids.values())) == 2
    before_games = {key: game.to_dict() for key, game in store.games.items()}
    board = VenueRollingBoard(channel)
    at = board.next_due()
    assert at is not None
    publication = board.publish_due(at)
    snapshot, = publication["snapshots"]
    assert publication["status"] == "completed"
    assert {item["game_id"] for item in snapshot["items"]} == set(game_ids.values())

    first, first_reply = _private(channel, source="before-disclosure", sender="seeker", at=at + timedelta(seconds=1),
                                 text="老板，今晚一块底无烟杭麻还有座吗？")
    before = _successful_tool(first, "search_current_games")
    assert {item["game"]["game_id"] for item in before["matches"]} == set(game_ids.values())
    assert store.task_memories == {} and store.customer_relationships == {}

    disclosed, disclosed_reply = _private(channel, source="disclose-relationship", sender="seeker",
        at=at + timedelta(seconds=2), text=f"{_DISCLOSURE}，{_PRIVATE_REASON}。", quote=first_reply["message_id"])
    written, = _successful_tool(disclosed, "record_user_memory")["task_memories"]
    assert written["target_customer_id"] == "owner-a" and written["evidence"] == _DISCLOSURE

    follow, follow_reply = _private(channel, source="after-disclosure", sender="seeker",
        at=at + timedelta(seconds=3), text="按刚才那个条件，再帮我看看还有没有能打的桌。", quote=disclosed_reply["message_id"])
    control, control_reply = _private(channel, source="independent-search", sender="observer", at=at + timedelta(seconds=4),
                                      text="老板，今晚一块底无烟杭麻还有座吗？")
    scoped_search = _successful_tool(follow, "search_current_games")
    independent_search = _successful_tool(control, "search_current_games")
    assert before["requirement"] == scoped_search["requirement"] == independent_search["requirement"]
    assert [item["game"]["game_id"] for item in scoped_search["matches"]] == [game_ids["owner-b"]]
    assert {item["game"]["game_id"] for item in independent_search["matches"]} == set(game_ids.values())

    reopened = SQLiteAgentStore(store.path)
    memory, = reopened.task_memories.values()
    assert (memory.customer_id, memory.target_customer_id, memory.scope, memory.status, memory.source_trace_id) == (
        "seeker", "owner-a", "current_task", "active", disclosed["trace_id"],
    )
    assert memory.conversation_id == channel.private_conversation_id("seeker")
    current_task = reopened.current_task_context(memory.conversation_id, "seeker")
    assert current_task is not None and memory.metadata["task_context_id"] == current_task.task_context_id
    assert reopened.customer_relationships == {} and reopened.pending_memory_candidates == {}
    assert {key: game.to_dict() for key, game in reopened.games.items()} == before_games
    assert not fixture.search_contexts["before-disclosure"].get("task_memories")
    assert [item["memory_id"] for item in fixture.search_contexts["after-disclosure"]["task_memories"]] == [
        memory.memory_id,
    ]
    control_context = fixture.search_contexts["independent-search"]
    assert not control_context.get("task_memories")
    serialized_control = json.dumps(control_context, ensure_ascii=False)
    assert all(hidden not in serialized_control for hidden in (_DISCLOSURE, _PRIVATE_REASON))

    records = {row["message_id"]: row for row in channel.events()}
    assert records["after-disclosure"]["quoted_message_id"] == disclosed_reply["message_id"]
    assert "1桌" in follow_reply["text"] and "2桌" in control_reply["text"]
    assert all(_PRIVATE_REASON not in row["text"] for row in records.values() if row["direction"] == "outbound")
    for other in ("owner-a", "owner-b", "observer"):
        visible = channel.events_for(other)
        assert all(row["message_id"] not in {"disclose-relationship", disclosed_reply["message_id"]} and
                   _DISCLOSURE not in row["text"] and _PRIVATE_REASON not in row["text"] for row in visible)
