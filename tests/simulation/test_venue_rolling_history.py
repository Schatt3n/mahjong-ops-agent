"""验证滚动演练复用真人前缀时的时间、引用、公私边界与恢复。"""

from datetime import datetime, timedelta

import pytest

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.models import using_clock

from .sim_adapter import build_runtime
from .venue_channel import SandboxChannel
from .venue_rolling_history import deliver_rolling_group_history


_START = datetime.fromisoformat("2026-11-26T19:00:00+08:00")
_ROSTER = ("active-a", "active-b", "history-a", "history-b")


def _channel(tmp_path):
    """重新打开同一模拟群和聊天账本。"""
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    runtime, _ = build_runtime(store, "mock")
    return SandboxChannel(runtime, room_id="rolling-room", archive_path=tmp_path / "chat.sqlite3",
                          identities={member: member for member in _ROSTER})


def test_reanchored_history_keeps_quotes_and_only_writes_public_chat(tmp_path, monkeypatch):
    """归档保留原有间隔和引用，模型入口、参与状态和留出消息都不受影响。"""
    channel = _channel(tmp_path)

    def reject_ingress(*args, **kwargs):
        pytest.fail("历史消息不能进入现场群聊处理器")

    monkeypatch.setattr(channel, "_handle_group", reject_ingress)
    store = channel.runtime.store
    with using_clock(lambda: _START):
        game, _ = store.create_game(
            conversation_id=channel.private_conversation_id(_ROSTER[0]), organizer_id=_ROSTER[0],
            organizer_name=_ROSTER[0], trace_id="existing-game",
            requirement={"game_type": "hangzhou_mahjong", "start_time": "21:00", "duration_hours": 2},
            known_players=[{"customer_id": _ROSTER[0], "display_name": _ROSTER[0]}],
        )
    original = game.to_dict()
    evidence = deliver_rolling_group_history(channel, roster_ids=iter(_ROSTER),
                                             active_actor_ids=_ROSTER[:2], start_at=_START)
    messages = channel.events_for(_ROSTER[0])
    assert [message["direction"] for message in messages] == ["history", "history"]
    assert [message["sender_id"] for message in messages] == list(_ROSTER[2:])
    assert all(message["kind"] == "group" and message["recipient_id"] == channel.room_id for message in messages)
    assert messages[0]["message_id"].endswith(":m01") and messages[1]["message_id"].endswith(":m02")
    assert messages[1]["quoted_message_id"] == messages[0]["message_id"]
    assert messages[1]["message_id"] == evidence.last_message_id and evidence.last_author_id == _ROSTER[-1]
    first, last = (datetime.fromisoformat(message["sent_at"]) for message in messages)
    assert last == evidence.last_message_at == _START - timedelta(minutes=2)
    assert last - first == timedelta(seconds=2269)
    assert evidence.audit()["prefix_messages"] == 2 and evidence.audit()["held_out_messages"] == 1
    assert evidence.audit()["time_basis"] == "reanchored" and evidence.audit()["active_archive_authors"] == 0
    assert set(store.games) == {game.game_id} and store.games[game.game_id].to_dict() == original
    assert not store.waiting_demands


def test_resume_recovers_partial_prefix_without_duplicate_chat(tmp_path, monkeypatch):
    """重开时即使第一条历史行尚无完成回执，也只补余下原话。"""
    channel = _channel(tmp_path)

    def interrupt(*args, **kwargs):
        raise InterruptedError("after history row")

    with monkeypatch.context() as patch:
        patch.setattr(channel, "_finish", interrupt)
        with pytest.raises(InterruptedError):
            deliver_rolling_group_history(channel, roster_ids=_ROSTER, active_actor_ids=_ROSTER[:2], start_at=_START)
    first, = channel.events()
    reopened = _channel(tmp_path)
    result = deliver_rolling_group_history(reopened, roster_ids=_ROSTER, active_actor_ids=_ROSTER[:2], start_at=_START)
    assert reopened.events()[0] == first
    assert len(reopened.events()) == 2
    assert deliver_rolling_group_history(reopened, roster_ids=_ROSTER, active_actor_ids=_ROSTER[:2], start_at=_START) == result
    assert len(reopened.events()) == 2 and not reopened.runtime.store.games


def test_full_roster_uses_last_two_existing_members_and_rejects_drift(tmp_path):
    """全员唤醒时不增加成员，重开时也不能换作者或改写场景时间。"""
    channel = _channel(tmp_path)
    result = deliver_rolling_group_history(channel, roster_ids=_ROSTER, active_actor_ids=_ROSTER, start_at=_START)
    assert result.active_archive_authors == 2 and result.last_author_id == _ROSTER[-1]
    assert set(channel.owner_friends) == set(_ROSTER)
    with pytest.raises(ValueError, match="已审核前缀不一致"):
        deliver_rolling_group_history(channel, roster_ids=_ROSTER, active_actor_ids=_ROSTER,
                                      start_at=_START + timedelta(minutes=1))
    assert len(channel.events()) == 2
