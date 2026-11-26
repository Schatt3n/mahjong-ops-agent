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
from .venue_table_party import VenueTableParty
from .venue_table_shop import VenueTableShop


_AT = datetime.fromisoformat("2026-11-27T18:05:00+08:00")
_V5_AT = datetime.fromisoformat("2026-11-25T19:37:00+08:00")
_V5_PLANNED = datetime.fromisoformat("2026-11-25T20:00:00+08:00")
_PEOPLE = generate_personas(42, count=5)
_HOST, _B, _C, _D, _BYSTANDER = (person.sender_id for person in _PEOPLE)
_ASK = "各自几点能到？时间先在这儿商量。"
_REPLY = "我七点以后方便，你们呢？"
_ACK = "收到，等其他几位也说一下。"
_DEPARTED_FOLLOWUP = "我之前收到的是不是原时间？"
_COMPANION_REQUEST = "带朋友来的，麻烦把还没进群的同行拉进来哈。"


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


def _open(tmp_path, shop, *, enabled=True, at=_AT):
    """重开独立渠道和调度账本，成员和业务记录留在原 SQLite 中。"""
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    runtime, _ = build_runtime(store, "mock")
    channel = SandboxChannel(runtime, room_id="notice-room", archive_path=tmp_path / "chat.sqlite3",
                             identities={person.sender_id: person.name for person in _PEOPLE})
    return VenueRollingRoles(
        people=_PEOPLE, actor=_TableActor(seed=42), judge=None, channel=channel, run_dir=tmp_path,
        start_at=at, seed=42, active_users=5, workers=4, dialogue_turns=5,
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


def _two_companion_game(scenario, *, planned_start):
    """复现 V5 由发起人登记三席、第四位顾客到 19:37 才独立确认的落库顺序。"""
    store = scenario.channel.runtime.store
    terms = {"game_type": "hangzhou_mahjong", "stake_range": {"min": 1, "max": 3},
             "start_time_kind": "asap_when_full"}
    if planned_start is not None:
        terms.update(start_time_kind="scheduled", planned_start_at=planned_start.isoformat())
    with using_clock(lambda: _V5_AT - timedelta(minutes=35)):
        first, _ = store.create_game(conversation_id=scenario.channel.private_conversation_id(_HOST),
            organizer_id=_HOST, organizer_name=_PEOPLE[0].name, requirement=terms, trace_id="v5-create-three",
            known_players=[{"customer_id": _HOST, "seat_count": 3, "anonymous_seat_count": 2}])
    assert first.status == GameStatus.FORMING and first.remaining_seats() == 1
    with using_clock(lambda: _V5_AT):
        ready, _ = store.join_game(game_id=first.game_id, customer_id=_B, display_name=_PEOPLE[1].name,
                                   trace_id="v5-confirm-fourth")
    assert ready.status == GameStatus.READY and ready.remaining_seats() == 0
    assert ready.seat_summary()["anonymous_seat_count"] == 2
    return ready


def test_rolling_v5_fixed_table_reminder_records_only_current_members_after_resume(tmp_path, monkeypatch):
    """原始格局的店家提醒走真实桌群收件，送达后中断能补唤醒并保持固定开场。"""
    class Interruption(BaseException):
        """在新提醒已写渠道账本、尚未办结唤醒时中断。"""

    shop = VenueTableShop(None, tmp_path / "shop.jsonl")
    scenario = _open(tmp_path, shop, at=_V5_AT)
    game = _two_companion_game(scenario, planned_start=_V5_PLANNED)
    original = scenario._wake_from_deliveries

    def stop_after_reminder(parent, receipts):
        if any(row["text"] == _COMPANION_REQUEST for row in receipts):
            raise Interruption
        return original(parent, receipts)

    with monkeypatch.context() as patch:
        patch.setattr(scenario, "_wake_from_deliveries", stop_after_reminder)
        with pytest.raises(Interruption):
            scenario.run()

    room = VenueTableGroup.room_id_for_game(game.game_id)
    first, = scenario.channel.table_group(room).delivered_receipts()
    assert first.message.sender_id == AGENT_ID and first.message.sent_at == _V5_AT
    assert first.message.text == _COMPANION_REQUEST and set(first.recipient_ids) == {_HOST, _B}

    resumed = _open(tmp_path, shop, at=_V5_AT)
    report = resumed.run()
    assert all(report["checks"].values())
    group = resumed.channel.table_group(room)
    assert group.delivered_receipts() == (first,)
    assert shop(group, None, _V5_AT + timedelta(seconds=1)) is None
    assert group.member_ids() == {AGENT_ID, _HOST, _B}
    assert all(not resumed.channel.events_for(person) for person in (_C, _D, _BYSTANDER))
    opened, = [event for event in resumed.journal.events() if event.kind == "table_group_open"]
    wakes = [event for event in resumed.journal.events() if event.kind == "table_receipt"]
    assert opened.result["opened"] and opened.result["message_id"] == first.message.message_id
    assert set(opened.result["scheduled_wakes"]) == {event.event_id for event in wakes}
    assert {(event.actor_id, event.caused_by) for event in wakes} == {
        (_HOST, first.message.message_id), (_B, first.message.message_id),
    }
    assert resumed.channel.runtime.store.require_game(game.game_id).to_dict() == game.to_dict()
    assert not shop.trace_path.exists()


@pytest.mark.parametrize("scheduled", [False, True])
def test_rolling_table_invite_reminder_waits_for_actual_companions_before_flexible_inquiry(tmp_path, monkeypatch, scheduled):
    """两个同行各自同意并占原席才加入协商，已有固定开场的桌继续沿用原时间。"""
    shop = VenueTableShop(None, tmp_path / "shop.jsonl")
    scenario = _open(tmp_path, shop, at=_V5_AT)
    game = _two_companion_game(scenario, planned_start=_V5_PLANNED if scheduled else None)
    invited_at = _V5_AT + timedelta(seconds=10)
    accepted_at = invited_at + timedelta(seconds=10)
    bound_at = accepted_at + timedelta(seconds=10)
    original = scenario._wake_from_deliveries

    def bind_only_after_reminder(parent, receipts):
        if any(row["text"] == _COMPANION_REQUEST for row in receipts):
            group = scenario.channel.table_group(VenueTableGroup.room_id_for_game(game.game_id))
            assert group.member_ids() == {AGENT_ID, _HOST, _B}
            for actor_id in (_C, _D):
                invitation = group.invite_companion(f"from-contact:{actor_id}", inviter_id=_HOST,
                                                     recipient_id=actor_id, at=invited_at)
                assert group.member_ids(at=invited_at) == {AGENT_ID, _HOST, _B}
                group.respond_companion(invitation.invitation_id, actor_id=actor_id, at=accepted_at, accept=True)
            for actor_id in (_C, _D):
                VenueTableParty(group).bind_companion(f"from-contact:{actor_id}", actor_id=actor_id, at=bound_at)
        return original(parent, receipts)

    monkeypatch.setattr(scenario, "_wake_from_deliveries", bind_only_after_reminder)
    report = scenario.run()
    assert all(report["checks"].values())
    group = scenario.channel.table_group(VenueTableGroup.room_id_for_game(game.game_id))
    reminder, *later = group.delivered_receipts()
    assert reminder.message.text == _COMPANION_REQUEST and set(reminder.recipient_ids) == {_HOST, _B}
    assert group.member_ids() == {AGENT_ID, _HOST, _B, _C, _D}
    if scheduled:
        assert not later
    else:
        opening_question, = later
        assert opening_question.message.sent_at == bound_at
        assert "各自几点能到" in opening_question.message.text
        assert set(opening_question.recipient_ids) == {_HOST, _B, _C, _D}
    assert all(reminder.message.message_id not in {row["message_id"] for row in scenario.channel.events_for(person)}
               for person in (_C, _D, _BYSTANDER))
    updated = scenario.channel.runtime.store.require_game(game.game_id)
    assert updated.seat_summary()["claimed_seats"] == 4 and updated.seat_summary()["anonymous_seat_count"] == 0
    assert updated.planned_start_at == game.planned_start_at


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


def test_rolling_pauses_shop_and_actor_without_rewriting_a_delivered_table_message(tmp_path, monkeypatch):
    """有人退席且尚无正式补位时暂停发言，原来确实收到的回执保持可见。"""
    class _FollowupActor(_TableActor):
        """已退出的一位尝试接住退出前确实收过的最后一条消息。"""

        def _choose(self, context, projected, event, rng):
            delivery = projected["delivery"] or {}
            if context.persona.sender_id == _D and delivery.get("text") == _REPLY:
                return ActorAction("send", _DEPARTED_FOLLOWUP, "group", delivery["recipient_id"], delivery["message_id"])
            return super()._choose(context, projected, event, rng)

    shop = _TableShop()
    scenario = _open(tmp_path, shop)
    scenario.actor = _FollowupActor(seed=42)
    game = _game(scenario, ready=True)
    store = scenario.channel.runtime.store
    original = scenario._wake_from_deliveries

    def change_after_receipt(parent, receipts):
        if any(row.get("text") == _REPLY for row in receipts):
            with using_clock(lambda: parent.due_at):
                store.record_candidate_reply(game_id=game.game_id, customer_id=_D, display_name=_D,
                                             status="declined", trace_id="old-seat-withdrawn")
        return original(parent, receipts)

    monkeypatch.setattr(scenario, "_wake_from_deliveries", change_after_receipt)
    report = scenario.run()
    assert report["rolling_table_groups"]["roster_pauses"] == 2
    assert report["rolling_table_groups"]["shop_receipts"] == 0
    assert report["checks"]["table_group_roster_no_paused_deliveries"] is False
    assert len(shop.handled) == 1
    paused = [event for event in scenario.journal.events() if event.result.get("table_roster_paused")]
    assert {(event.kind, event.actor_id) for event in paused} == {("table_shop_receipt", AGENT_ID), ("table_receipt", _D)}
    assert {row["text"] for row in scenario.channel.events_for(_D)} == {_ASK, _REPLY}
    assert not scenario.channel.events_for(_BYSTANDER)
    assert [_ASK, _REPLY] == [row["text"] for row in scenario.channel.events()]


def test_binding_the_fourth_contact_wakes_existing_table_shop_once(tmp_path):
    """匿名同行绑定后在原桌群开启四人协商，进程重开不多建群或重复问时间。"""
    shop = _TableShop()
    scenario = _open(tmp_path, shop)
    store = scenario.channel.runtime.store
    with using_clock(lambda: _AT - timedelta(minutes=1)):
        game, _ = store.create_game(
            conversation_id=scenario.channel.private_conversation_id(_HOST), organizer_id=_HOST,
            organizer_name=_PEOPLE[0].name,
            known_players=[{"customer_id": _HOST, "seat_count": 2, "anonymous_seat_count": 1}],
            requirement={"game_type": "hangzhou_mahjong", "stake": "1", "start_time_kind": "asap_when_full"},
            trace_id="companion-table",
        )
        for person in _PEOPLE[1:3]:
            store.join_game(game_id=game.game_id, customer_id=person.sender_id, display_name=person.name,
                            trace_id=f"confirmed:{person.sender_id}")
    group = scenario.channel.open_table_group(game.game_id, at=_AT)
    group.invite_companion("companion", inviter_id=_HOST, recipient_id=_D, at=_AT + timedelta(seconds=10))
    group.respond_companion("companion", actor_id=_D, at=_AT + timedelta(seconds=20), accept=True)
    bound_at = _AT + timedelta(seconds=30)
    VenueTableParty(group).bind_companion("companion", actor_id=_D, at=bound_at)

    report = scenario.run()
    assert all(report["checks"].values())
    roster_wake, = [item for item in scenario.journal.events() if item.kind == "table_roster_ready"]
    assert roster_wake.due_at == bound_at
    assert roster_wake.result["opened"] is False
    assert report["rolling_table_groups"]["opened"] == 0
    question, = [receipt for receipt in group.delivered_receipts() if receipt.message.text == _ASK]
    assert set(question.recipient_ids) == {_HOST, _B, _C, _D}
    assert shop.handled.count((game.game_id, None)) == 1
    reopened = _open(tmp_path, shop)
    reopened.run()
    assert shop.handled.count((game.game_id, None)) == 1
