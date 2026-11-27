"""用真实桌群收件回放冲突钟点、改提议及四份新票。"""

from datetime import datetime, timedelta
import json

import pytest

from mahjong_agent_runtime import SQLiteAgentStore

from scripts.run_venue_table_opening import _CompanionProbe, _prepare_companion
from .sim_adapter import build_runtime
from .test_venue_table_opening import _OPEN, _propose, _table, _vote
from .venue_actors import ActorAction, MockActor
from .venue_channel import AGENT_ID, SandboxChannel
from .venue_table_group import VenueTableGroup
from .venue_table_opening import TABLE_OPENING_OFFER_PREFIX, VenueTableOpening
from .venue_table_shop import VenueTableShop


_ABANDONED = datetime.fromisoformat("2026-11-27T18:12:00+08:00")
_REVISED = _ABANDONED.replace(minute=20)
_ORIGINAL = "老板好，18:12 我这边最早 18:14 才能到，晚两分钟能搭一桌吗？"
_PEER_TIME = "老板好，我从城另一头过去，最早18:10到，能搭一桌吗"
_ASK_FOR_PROPOSAL = "我这边六点前能到，不用等我太早，老板定开就成。"


class _ReplayActor(MockActor):
    """用留档发言重放角色路径，每人只针对本人的实际收件作答。"""

    def _choose(self, context, projected, event, rng):
        if companion := projected.get("companion_invite"):
            return ActorAction("invite_companion" if companion["stage"] == "offer" else "accept_companion")
        delivery = projected["delivery"] or {}
        message_id = delivery.get("message_id", "")
        actor_id = context.persona.sender_id
        if message_id.startswith("table-opening-inquiry:") and actor_id == "venue-sender-0015":
            text = _PEER_TIME
        elif delivery.get("text") == _PEER_TIME and actor_id == "venue-sender-0013":
            text = _ASK_FOR_PROPOSAL
        elif message_id.startswith(f"{TABLE_OPENING_OFFER_PREFIX}{_ABANDONED.isoformat()}|"):
            text = _ORIGINAL if actor_id == "venue-sender-0015" else "18:12 我能到，就定这个时间吧。"
        elif message_id.startswith(f"{TABLE_OPENING_OFFER_PREFIX}{_REVISED.isoformat()}|"):
            text = "18:20 才开可以。"
        else:
            return ActorAction("wait")
        return ActorAction("send", text, "group", delivery["recipient_id"], message_id)


class _FrozenReviewer:
    """同一条原话继续重放曾误判的 other，其他确定票义人工核准。"""

    def complete(self, messages, **options):
        payload = json.loads(messages[-1]["content"])
        text = payload["current"]["text"]
        if text == _ASK_FOR_PROPOSAL:
            return json.dumps({"vote": "other", "proposed_start": _ABANDONED.isoformat()})
        if text == _ORIGINAL:
            assert payload["current"]["verified_arrival_not_before_at"] == _ABANDONED.replace(minute=14).isoformat()
        return json.dumps({"vote": "accept" if text in {"18:12 我能到，就定这个时间吧。", "18:20 才开可以。"} else "other",
                           "proposed_start": None})


def test_archived_late_arrival_after_peer_round_revises_and_wakes_all_four_after_resume(tmp_path, monkeypatch):
    class Interrupted(BaseException):
        """新版已送达但投票唤醒尚未持久化时模拟进程中断。"""

    at, seed, people, channel, game_id, _, checks = _prepare_companion(tmp_path, unavailable=False)
    assert all(checks.values())
    shop = VenueTableShop(_FrozenReviewer(), tmp_path / "shop.jsonl")
    def _scenario(channel):
        return _CompanionProbe(people=people, actor=_ReplayActor(seed), judge=None, channel=channel,
            run_dir=tmp_path, start_at=at, seed=seed, active_users=len(people), workers=4, dialogue_turns=2,
            judge_users=1, table_groups=True, table_shop=shop)

    scenario = _scenario(channel)
    wake = scenario._wake_from_deliveries
    def interrupt(parent, rows):
        if any(row["message_id"].startswith(f"{TABLE_OPENING_OFFER_PREFIX}{_REVISED.isoformat()}|") for row in rows):
            assert parent.details["depth"] == 4
            raise Interrupted
        return wake(parent, rows)

    monkeypatch.setattr(scenario, "_wake_from_deliveries", interrupt)
    with pytest.raises(Interrupted):
        scenario.run()
    store = SQLiteAgentStore(channel.runtime.store.path)
    assert store.require_game(game_id).planned_start_at is None
    room_id = VenueTableGroup.room_id_for_game(game_id)
    group = channel.table_group(room_id)
    first, revised = [row for row in group.delivered_receipts() if row.message.message_id.startswith(TABLE_OPENING_OFFER_PREFIX)]
    assert revised.message.sent_at == at + timedelta(minutes=10)
    assert VenueTableOpening(group).current()["declined_count"] == 1
    assert any(row.message.text == _ORIGINAL for row in group.delivered_receipts())

    runtime, _ = build_runtime(store, "mock")
    restored = SandboxChannel(runtime, room_id=channel.room_id, archive_path=channel.archive_path,
                              identities={person.sender_id: person.name for person in people})
    rerun = _scenario(restored)
    checks = rerun.run()["checks"]
    assert {key for key, value in checks.items() if not value} == {"all_target_roles_reached_two_calendar_days"}
    group = restored.table_group(room_id)
    expected = {person.sender_id for person in people[:-1]}
    assert set(first.recipient_ids) == set(revised.recipient_ids) == expected
    wakes = [event for event in rerun.journal.events() if event.kind == "table_receipt"
             and event.caused_by == revised.message.message_id]
    assert len(wakes) == 4 and {event.actor_id for event in wakes} == expected
    assert {row.message.sender_id for row in group.delivered_receipts() if row.message.quoted_message_id == revised.message.message_id} == expected
    assert VenueTableOpening(group).current() == {"proposal_id": revised.message.message_id, "start_at": _REVISED.isoformat(),
                                                  "accepted_count": 4, "declined_count": 0, "committed": True}
    assert store.require_game(game_id).planned_start_at == _REVISED
    assert sum(row.reason == "venue_table_opening_confirmed" for row in store.transitions) == 1
    assert not group.events_for(people[-1].sender_id)
    guards = [json.loads(line)["vote_guard"] for line in shop.trace_path.read_text().splitlines() if '"vote_guard"' in line]
    assert guards == [{"reason": "opening_before_explicit_arrival", "arrival_not_before": "2026-11-27T18:14:00+08:00",
                       "model_vote": "other"}]


@pytest.mark.parametrize("text,model_vote,declines", [
    ("能晚一点开吗？我这个时间其实也可以。", "other", 0),
    ("我可能最早 19:10 才能到，能晚点吗？", "other", 0),
    ("我最早 19.10 才能到，晚点吗？", "other", 0),
    ("我最早 19:10 才能到？", "other", 0),
    ("我最早 20:10 才能到，能晚一点吗？", "accept", 1),
    ("我最早 19:10 才能到，最多打三个小时。", "other", 1),
    ("我最早 19:10 才能到，19:15 就得走。", "other", 1),
])
def test_only_real_compatible_options_get_reproposed(tmp_path, text, model_vote, declines):
    class _Client:
        def complete(self, messages, **options):
            return json.dumps({"vote": model_vote, "proposed_start": None})

    store, game_id, group, opening = _table(tmp_path)
    start = _OPEN.replace(hour=19, minute=0)
    first = _propose(opening, group, f"{TABLE_OPENING_OFFER_PREFIX}{start.isoformat()}|first", 1, start)
    receipt = _vote(group, "d", "possible-conflict", 3, text, first.message.message_id)
    shop = VenueTableShop(_Client(), tmp_path / "review.jsonl")
    assert shop(group, receipt, receipt.message.sent_at) is None
    latest = opening.current()
    assert latest["accepted_count"] == 0 and latest["declined_count"] == declines
    assert store.require_game(game_id).planned_start_at is None


def test_restored_decline_does_not_offer_a_time_before_the_earliest_arrival(tmp_path):
    class _Client:
        def complete(self, messages, **options):
            payload = json.loads(messages[-1]["content"])
            start = "2026-11-27T19:05:00+08:00" if payload["active_proposal"]["declined_count"] else None
            return json.dumps({"vote": "other", "proposed_start": start})

    store, game_id, group, opening = _table(tmp_path)
    start = _OPEN.replace(hour=19, minute=0)
    first = _propose(opening, group, f"{TABLE_OPENING_OFFER_PREFIX}{start.isoformat()}|first", 1, start)
    earlier = _vote(group, "d", "fixed-arrival", 3, "我最早 19:10 才能到，19:15 能不能开？", first.message.message_id)
    shop = VenueTableShop(_Client(), tmp_path / "reviews.jsonl")
    # 顾客同时提了不同钟点，保留明确冲突并等待具有来源的候选。
    assert shop(group, earlier, earlier.message.sent_at) is None
    store = SQLiteAgentStore(store.path)
    group = VenueTableGroup(store, group.archive_path, game_id, actor_ids=("a", "b", "c", "d", "outsider"), opened_at=_OPEN)
    later = _vote(group, "b", "too-early", 4, "要不 19:05 开？")
    assert shop(group, later, later.message.sent_at) is None
    assert VenueTableOpening(group).current()["declined_count"] == 1
    assert store.require_game(game_id).planned_start_at is None
