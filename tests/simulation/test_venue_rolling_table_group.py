"""用实际四席业务状态和桌群回执验证建设、交流和中断恢复。"""

from collections import Counter
from datetime import datetime, timedelta

import pytest

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.models import GameStatus, using_clock

from .sim_adapter import build_runtime
from .venue_actors import ActorAction, MockActor, generate_personas
from .venue_channel import AGENT_ID, MessageEnvelope, SandboxChannel
from .venue_rolling_roles import VenueRollingRoles
from .venue_table_group import VenueTableGroup


_AT = datetime.fromisoformat("2026-11-27T18:05:00+08:00")
_PEOPLE = generate_personas(42, count=5)
_HOST, _B, _C, _D, _BYSTANDER = (person.sender_id for person in _PEOPLE)
_ASK = "各自几点能到？时间先在这儿商量。"
_REPLY = "我七点以后方便，你们呢？"
_ACK = "收到，等其他几位也说一下。"


class _TableActor(MockActor):
    """收到真实桌群开场消息后，由发起人答一次自己的时间。"""

    def _choose(self, context, projected, event, rng):
        delivery = projected["delivery"] or {}
        if delivery.get("text") == _ASK and context.persona.sender_id == _HOST:
            room = delivery["recipient_id"]
            assert context.group_targets == (room,)
            return ActorAction("send", _REPLY, "group", room, delivery["message_id"])
        return ActorAction("wait")


class _TableShop:
    """仅验证专用店家入口，店家协商政策由另一层提供。"""

    def __init__(self):
        self.handled = []

    def __call__(self, group, reply_to, at):
        self.handled.append((group.game_id, reply_to))
        source_id = reply_to.message.message_id if reply_to else group.game_id
        text = _ACK if reply_to else _ASK
        return MessageEnvelope(f"owner-table:{source_id}", AGENT_ID, text, at, "group", group.room_id)


def _open(tmp_path, shop, *, enabled=True):
    """重开独立渠道和调度账本，成员和业务记录留在原 SQLite 中。"""
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    runtime, _ = build_runtime(store, "mock")
    channel = SandboxChannel(runtime, room_id="notice-room", archive_path=tmp_path / "chat.sqlite3",
                             identities={person.sender_id: person.name for person in _PEOPLE})
    return VenueRollingRoles(
        people=_PEOPLE, actor=_TableActor(seed=42), judge=None, channel=channel, run_dir=tmp_path,
        start_at=_AT, seed=42, active_users=5, workers=4, dialogue_turns=5,
        judge_users=2, table_groups=enabled, table_shop=shop,
    )


def _game(scenario, *, ready):
    """由实际业务预约接口形成一张人齐开的三席或四席牌局。"""
    store = scenario.channel.runtime.store
    with using_clock(lambda: _AT - timedelta(minutes=1)):
        game, _ = store.create_game(
            conversation_id=scenario.channel.private_conversation_id(_HOST), organizer_id=_HOST,
            organizer_name=_PEOPLE[0].name, known_players=[],
            requirement={"game_type": "hangzhou_mahjong", "stake": "1", "start_time_kind": "asap_when_full"},
            trace_id="table-create",
        )
        for person in _PEOPLE[1:(4 if ready else 3)]:
            game, _ = store.join_game(game_id=game.game_id, customer_id=person.sender_id,
                display_name=person.name, trace_id=f"table-confirm:{person.sender_id}")
    assert (game.status == GameStatus.READY) is ready
    return game


@pytest.mark.parametrize(("enabled", "ready"), [(False, True), (True, False)])
def test_rolling_table_group_requires_opt_in_and_four_confirmed_seats(tmp_path, enabled, ready):
    """未启用或尚差一席时，桌群和店家控制事件都不发生。"""
    shop = _TableShop() if enabled else None
    scenario = _open(tmp_path, shop, enabled=enabled)
    game = _game(scenario, ready=ready)
    report = scenario.run()
    assert all(report["checks"].values())
    assert not scenario.channel.table_group(VenueTableGroup.room_id_for_game(game.game_id))
    assert not any(item.kind == "table_group_open" for item in scenario.journal.events())
    if enabled:
        assert report["rolling_table_groups"]["opened"] == 0
        assert not shop.handled


@pytest.mark.parametrize("interrupt", [False, True])
def test_rolling_table_group_uses_one_shop_ingress_and_exact_recipients(tmp_path, monkeypatch, interrupt):
    """小群开场与原话走正式回执，崩溃恢复复用已送出的店家回复。"""
    class Interruption(BaseException):
        """模拟店家已送达但控制事件尚未办结时退出。"""

    shop = _TableShop()
    scenario = _open(tmp_path, shop)
    game = _game(scenario, ready=True)
    scenario.channel.publish_owner(kind="group", recipient_id=scenario.channel.room_id, text="公开营业公告。",
                                   sent_at=_AT - timedelta(minutes=2), message_id="public-venue-board")
    if interrupt:
        original = scenario._wake_from_deliveries

        def interrupt_after_ack(parent, receipts):
            if receipts and receipts[0].get("text") == _ACK:
                raise Interruption
            return original(parent, receipts)

        with monkeypatch.context() as patch:
            patch.setattr(scenario, "_wake_from_deliveries", interrupt_after_ack)
            with pytest.raises(Interruption):
                scenario.run()
        assert len(shop.handled) == 2
        assert len([row for row in scenario.channel.events() if row["text"] == _ACK]) == 1
        scenario = _open(tmp_path, shop)

    report = scenario.run()
    assert all(report["checks"].values())
    assert report["rolling_table_groups"] == {"opened": 1, "durable_group_messages": 3, "role_receipts": 11, "shop_receipts": 1}
    assert len(shop.handled) == 2
    assert shop.handled[0] == (game.game_id, None)
    room_id = VenueTableGroup.room_id_for_game(game.game_id)
    group = scenario.channel.table_group(room_id)
    assert group.member_ids() == {AGENT_ID, _HOST, _B, _C, _D}
    opened, player, owner = [row for row in scenario.channel.events() if row.get("table_game_id") == game.game_id]
    assert player["text"] == _REPLY and owner["text"] == _ACK
    assert (opened["recipient_id"], player["recipient_id"], owner["recipient_id"]) == (room_id,) * 3
    assert player["quoted_message_id"] == opened["message_id"]
    assert owner["source_message_id"] == player["message_id"]
    assert shop.handled[1][1].message.text == _REPLY
    assert set(player["visible_to"]) == {AGENT_ID, _HOST, _B, _C, _D}
    assert [row["message_id"] for row in scenario.channel.events_for(_BYSTANDER)] == ["public-venue-board"]
    counts = Counter(event.caused_by for event in scenario.journal.events() if event.kind == "table_receipt")
    assert counts == {opened["message_id"]: 4, player["message_id"]: 3, owner["message_id"]: 4}
    with pytest.raises(ValueError, match="sender is not admitted"):
        scenario.channel.send(MessageEnvelope("unauthorized", _BYSTANDER, "你们几个人？", _AT, "group", room_id))
    another_public_room = SandboxChannel(scenario.channel.runtime, room_id="different-notice-room",
        archive_path=scenario.channel.archive_path, identities={person.sender_id: person.name for person in _PEOPLE})
    assert another_public_room.table_group(room_id) is None
    assert not another_public_room.events_for(_HOST)
    assert scenario.channel.runtime.store.require_game(game.game_id).status == GameStatus.READY
    assert scenario.run()["rolling_table_groups"] == report["rolling_table_groups"]
    assert len(shop.handled) == 2
