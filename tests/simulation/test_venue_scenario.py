"""用离线模型合同跑通 500 人门店验收场景并核对证据文件。"""

from dataclasses import replace
from datetime import date, datetime, timedelta
import json
import sqlite3
from types import SimpleNamespace

import pytest

from mahjong_agent_runtime import SQLiteAgentStore
from scripts.run_venue_simulation import run_simulation
from scripts import run_venue_simulation as runner
from tests.simulation.venue_actors import ActorAction, ActorDecision, MockActor, SemanticJudge, generate_personas
from tests.simulation.venue_channel import AGENT_ID, OWNER_ID, SandboxChannel
from tests.simulation.venue_scenario import VenueScenario, _judge_evidence, add_social_refusal_facts, select_roles


@pytest.mark.parametrize("seed", [42, 69])
def test_venue_full_game_keeps_role_truth_chat_and_actual_play_separate(tmp_path, seed):
    """真实群和私聊入口处理模拟发言，并以办结事务和可复订证明散场。"""
    run_dir = tmp_path / "isolated-run"
    report = run_simulation(run_dir, mode="mock", seed=seed, scenario_date=date(2026, 11, 25), background_events=4)

    assert report["status"] == "passed", report
    assert report["people"] == 500
    assert report["independent_judge"] is None
    assert report["main_model"] == "offline fixture"
    assert report["room_id"] == "东二包间"
    assert all(report["checks"].values())
    assert report["checks"]["memory_from_visible_message"]
    assert report["checks"]["peer_delivery_isolation"]
    assert report["checks"]["board_delivered"]
    assert report["checks"]["social_refusal_delivery"]
    assert report["social_refusal"]["status"] == "not_assessed"
    personas = json.loads((run_dir / "personas.json").read_text(encoding="utf-8"))["personas"]
    assert len(personas) == len({person["sender_id"] for person in personas}) == 500
    assert "venue-persona-" not in (run_dir / "agent_trace.log").read_text(encoding="utf-8")

    archive = [json.loads(line) for line in (run_dir / "chat.jsonl").read_text(encoding="utf-8").splitlines()]
    with sqlite3.connect(run_dir / "chat.sqlite3") as database:
        assert database.execute("SELECT COUNT(*) FROM venue_messages").fetchone()[0] == len(archive)
    assert {"inbound", "peer", "outbound"} <= {row["direction"] for row in archive}

    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    by_message = {row["message_id"]: row for row in archive}
    boards = [event for event in events if event["kind"] in {"announce", "board_refresh"}]
    assert [event["board_table_state"]["remaining_seats"] for event in boards] == [3, 2, 1, 0]
    assert report["checks"]["board_schedule_drained"] is True
    snapshots = [snapshot for event in boards for snapshot in event["publication"]["snapshots"]]
    assert [snapshot["items"][0]["rendered_text"].split()[-1] for snapshot in snapshots] == ["173", "272", "371"]
    published = [row for row in archive if row["message_id"].startswith("outbound:board:")]
    assert [row["message_id"] for row in published] == [snapshot["external_message_id"] for snapshot in snapshots]
    for receipt, snapshot in zip(published, snapshots, strict=True):
        assert (receipt["direction"], receipt["kind"], receipt["sender_id"], receipt["recipient_id"]) == (
            "outbound", "group", AGENT_ID, snapshot["room_id"],
        )
        assert receipt["conversation_id"] == snapshot["conversation_id"] and receipt["text"] == snapshot["rendered_text"]
        assert datetime.fromisoformat(receipt["sent_at"]) == datetime.fromisoformat(snapshot["created_at"])
    assert boards[-1]["board_table_state"]["status"] == "ready" and not boards[-1]["board_table_state"]["eligible"]
    full, = boards[-1]["publication"]["tasks"]
    assert full["status"] == "completed" and full["snapshot"] is None and not boards[-1]["publication"]["archive_outputs"]
    refreshed = [event for event in events if event.get("decision") and event["kind"] == "opportunity"
                 and event["caused_by"] in {row["message_id"] for row in published}]
    assert [event["caused_by"] for event in refreshed] == [row["message_id"] for row in published]
    for event in refreshed:
        assert sum(row["message_id"] == event["caused_by"] for row in event["decision"]["context"]["history"]) == 1
    invitation = next(row for row in events if row["kind"] == "peer_invite" and row.get("decision"))
    opening = next(row for row in events if row["kind"] == "need" and row.get("decision"))
    opening_context = opening["decision"]["context"]
    assert len(invitation["decision"]["context"]["private_targets"]) == 1
    assert AGENT_ID not in invitation["decision"]["context"]["private_targets"]
    assert opening_context["private_targets"] == [AGENT_ID] and not opening_context["group_targets"]
    assert not any(row["message_id"] in by_message and by_message[row["message_id"]]["direction"] == "peer"
                   for row in opening_context["history"])
    assert opening_context["events"][-1]["details"]["case_goal"] == "organize_new"
    assert opening_context["events"][-1]["details"]["duration_hours"] == 4

    invitation_event = next(row for row in events if row["kind"] == "social_invite")
    shop_invitation = invitation_event["shop_invitation"]
    receipt = shop_invitation["receipt"]
    assert receipt == by_message[receipt["message_id"]]
    assert (receipt["sender_id"], receipt["direction"], receipt["kind"]) == (OWNER_ID, "owner", "private")
    assert by_message[invitation_event["caused_by"]]["kind"] == "group" and receipt["source_message_id"] is None
    assert shop_invitation["game_id"] == report["game_id"] and shop_invitation["status"] == "sent"
    reaction = next(row for row in events if row.get("decision") and row["caused_by"] == receipt["message_id"])
    role_delivery = reaction["decision"]["context"]["delivery"]
    assert role_delivery["message_id"] == receipt["message_id"] and role_delivery["sender_id"] == AGENT_ID
    assert role_delivery["recipient_id"] == reaction["actor_id"] == receipt["recipient_id"]
    assert datetime.fromisoformat(receipt["sent_at"]) < datetime.fromisoformat(reaction["virtual_at"])

    arrivals = [row for row in events if row["kind"] == "arrival" and row.get("decision")]
    assert len(arrivals) == 4 and all(row["decision"]["action"] == {"action": "arrive"} for row in arrivals)
    for row in arrivals:
        delivered = row["decision"]["context"]["delivery"]
        assert delivered is not None and delivered["message_id"] == row["caused_by"]
        assert by_message[row["caused_by"]]["direction"] == "outbound"
        assert by_message[row["caused_by"]]["recipient_id"] == row["actor_id"]

    started = next(row for row in events if row["kind"] == "start")
    finished = next(row for row in events if row["kind"] == "finish")
    assert all(datetime.fromisoformat(row["sent_at"]) < datetime.fromisoformat(started["virtual_at"]) for row in published)
    assert datetime.fromisoformat(started["game"]["actual_started_at"]) - datetime.fromisoformat(
        started["game"]["planned_start_at"]
    ) == timedelta(minutes=3)
    assert datetime.fromisoformat(finished["game"]["actual_ended_at"]) - datetime.fromisoformat(
        started["game"]["actual_started_at"]
    ) == timedelta(hours=4)
    assert len(set(started["game"]["actual_attendee_ids"])) == 4
    assert finished["released"]["status"] == "released"
    rebooked = next(row for row in events if row["kind"] == "rebook")["next_booking"]
    persisted = SQLiteAgentStore(run_dir / "agent.sqlite3").room_reservations
    assert persisted[rebooked["reservation_id"]].room_id == report["room_id"]
    assert persisted[finished["released"]["reservation_id"]].status == "released"
    assert report["physical_play"] == {
        "status": "passed", "actual_started_at": started["game"]["actual_started_at"],
        "actual_ended_at": finished["game"]["actual_ended_at"],
        "evidence_ids": {row["kind"]: row["event_id"] for row in events
                         if row["kind"] in {"start", "occupied", "finish", "rebook"}},
    }


def test_venue_distinguishes_persisted_game_from_missing_business_reply(tmp_path, monkeypatch):
    """已正确建桌后若回复流程失败，不能报成组局时间或时长错误。"""
    send = SandboxChannel.send

    def send_with_failed_reply(channel, message):
        result = send(channel, message)
        if any(tool["name"] == "create_game" and tool.get("called") and tool.get("allowed")
               for tool in result.get("tool_results", ())):
            return {**result, "status": "needs_human"}
        return result

    monkeypatch.setattr(SandboxChannel, "send", send_with_failed_reply)
    run_dir = tmp_path / "persisted-with-reply-failure"
    report = run_simulation(run_dir, mode="mock", seed=42, scenario_date=date(2026, 11, 25), background_events=0)

    assert report["status"] == "failed"
    assert report["checks"]["create_via_agent"] is False
    assert any(row["reason"] == "组局已按请求落库，但店家未送达有效业务回执" for row in report["failed_events"])
    game = SQLiteAgentStore(run_dir / "agent.sqlite3").require_game(report["game_id"])
    assert game.planned_end_at - game.planned_start_at == timedelta(hours=4)


def test_venue_publishes_forming_tables_at_the_planned_start_boundary_without_customer_input(tmp_path, monkeypatch):
    """未满的桌过原定开场后，正常排期更新群提示而不代开局或保留直接认领。"""
    _, _, _, _, last_player, _ = select_roles(generate_personas(42))

    class LastPlayerWaits(MockActor):
        def decide(self, context, *, trace_id):
            result = super().decide(context, trace_id=trace_id)
            if context.persona == last_player and context.events[-1]["kind"] == "opportunity":
                return replace(result, action=ActorAction("wait"), raw='{"action":"wait"}')
            return result

    monkeypatch.setattr(runner, "MockActor", LastPlayerWaits)
    run_dir = tmp_path / "still-forming"
    report = run_simulation(run_dir, mode="mock", seed=42, scenario_date=date(2026, 11, 25), background_events=0)

    assert report["status"] == "failed" and not report["checks"]["four_real_confirmations"]
    assert report["checks"]["board_schedule_drained"] is True
    store = SQLiteAgentStore(run_dir / "agent.sqlite3")
    game = store.require_game(report["game_id"])
    assert game.actual_started_at is None and game.remaining_seats() == 1
    archived = [json.loads(line) for line in (run_dir / "chat.jsonl").read_text(encoding="utf-8").splitlines()]
    late_boards = [row for row in archived if row["message_id"].startswith("outbound:board:")
                   and datetime.fromisoformat(row["sent_at"]) >= game.planned_start_at]
    receipt, = late_boards
    assert datetime.fromisoformat(receipt["sent_at"]) == game.planned_start_at + timedelta(microseconds=1)
    assert not any(row["direction"] == "inbound" and row["sent_at"] == receipt["sent_at"] for row in archived)
    assert receipt["kind"] == "group" and receipt["recipient_id"] == receipt["room_id"]
    assert "仍在凑人，晚到需先确认" in receipt["text"] and "即可认领" not in receipt["text"]
    assert store.get_board_snapshot_by_message_id(receipt["room_id"], receipt["message_id"]).rendered_text == receipt["text"]


def test_venue_judge_receives_actual_native_claim_and_participation_scope(tmp_path, monkeypatch):
    """真实群认领、满桌通知收件人的既有席位和自行谢绝都能回查。"""
    class LastGroupClaim(MockActor):
        opportunities = 0

        def decide(self, context, *, trace_id):
            result = super().decide(context, trace_id=trace_id)
            event = context.events[-1]
            if event["kind"] == "opportunity" and event["details"].get("case_goal") == "try_join":
                self.opportunities += 1
                if self.opportunities == 3:
                    return replace(result, action=ActorAction("send", "1来", "group", context.group_targets[0],
                                                            quote_message_id=event["caused_by"]))
            if event["kind"] == "share_joined":
                board = next(row for row in context.history if row["message_id"].startswith("outbound:board:"))
                return replace(result, action=replace(result.action, quote_message_id=board["message_id"]))
            return result

    class RecordingJudgeClient:
        paraphrase = False

        def complete(self, messages, *, trace_id, timeout_seconds):
            self.instructions = messages[0]["content"]
            self.request = json.loads(messages[1]["content"])
            checks = self.request["checks"].copy()
            if self.paraphrase:
                checks[1] = "拒邀及组局确认与可见对话一致"
            rows = self.request["evidence"]
            cited = rows[0]["evidence_id"]
            refusal = next(row["evidence_id"] for row in rows if row.get("social_refusal_candidate"))
            return json.dumps({"verdict": "pass", "findings": [
                {"check": check, "verdict": "pass", "evidence_ids": [refusal if check == "social_refusal" else cited],
                 "reason": "离线检查判定结构"}
                for check in checks
            ]}, ensure_ascii=False)

    client = RecordingJudgeClient()
    judge = SemanticJudge(client)
    scenario = runner.VenueScenario
    send = SandboxChannel.send
    internal_tools = {"customer_visible_content_review", "customer_visible_text_generation"}

    def send_with_internal_tools(channel, message):
        response = send(channel, message)
        if message.text == "1来":
            # 离线 Agent 不启用真实话术审查，用受控工具回执覆盖同样的投影边界。
            receipts = [{"name": name, "called": True, "allowed": True} for name in sorted(internal_tools)]
            response = response | {"tool_results": response["tool_results"] + receipts}
        return response

    monkeypatch.setattr(runner, "MockActor", LastGroupClaim)
    monkeypatch.setattr(runner, "VenueScenario", lambda **kwargs: scenario(**(kwargs | {"judge": judge})))
    monkeypatch.setattr(SandboxChannel, "send", send_with_internal_tools)
    run_dir = tmp_path / "judge-projection"
    report = run_simulation(run_dir, mode="mock", seed=42, scenario_date=date(2026, 11, 25), background_events=0)

    assert report["status"] == "passed", report
    assert client.request["checks"] == ["role_behavior", "participation", "business_facts", "social_refusal"]
    assert all(f"{check}：" in client.instructions for check in client.request["checks"])
    assert report["independent_judge"]["status"] == "valid"
    evidence = report["independent_judge"]["evidence"]
    assert [row["evidence_alias"] for row in evidence] == [row["evidence_id"] for row in client.request["evidence"]]
    native, = [row for row in evidence if row.get("group_claim")]
    assert native["customer_message"]["text"] == "1来"
    assert not any(item["name"] == "join_game" for item in native["tool_results"])
    claim = native["group_claim"]
    assert claim["routing"]["status"] == "claimed"
    assert claim["game_status"] == "ready" and claim["remaining_seats"] == 0
    assert claim["persisted_claim"]["source_message_id"] == native["message_id"]
    assert claim["persisted_claim"]["game_id"] == report["game_id"]
    assert claim["persisted_participant"] == {"customer_id": native["actor_id"], "status": "confirmed", "seat_count": 1}
    assert any(row["trace_id"] == native["trace_id"] and row["entity_type"] == "game_participant"
               and row["reason"] == "record_candidate_reply" and row["to_status"] == "confirmed"
               for row in claim["persisted_transitions"])
    assert any(row["recipient_id"] == native["actor_id"] for row in native["outputs"])
    public_share, = [row for row in evidence if row.get("kind") == "share_joined" and "customer_message" in row]
    group_message = public_share["customer_message"]
    assert group_message["quoted_message_id"] != native["customer_message"]["quoted_message_id"]
    chat = [json.loads(raw) for raw in (run_dir / "chat.jsonl").read_text(encoding="utf-8").splitlines()]
    notified = [row for row in chat if row["source_message_id"] == native["message_id"]
                and row["direction"] == "outbound" and row["sender_id"] == AGENT_ID and row["kind"] == "private"]
    assert len({row["recipient_id"] for row in notified}) == 4
    organizer = SQLiteAgentStore(run_dir / "agent.sqlite3").require_game(report["game_id"]).organizer_id
    assert claim["private_recipient_memberships"] == [
        {"message_id": row["message_id"], "recipient_id": row["recipient_id"], "seat_count": 1,
         "status": "joined" if row["recipient_id"] == organizer else "confirmed"}
        for row in notified
    ]
    older_private_confirmations = {
        row["actor_id"] for row in evidence
        if datetime.fromisoformat(row["virtual_at"]) < datetime.fromisoformat(native["virtual_at"])
        and row.get("customer_message", {}).get("kind") == "private"
        and any(tool["name"] == "join_game" and tool.get("allowed") is True
                and tool.get("result", {}).get("recorded_status") == "confirmed"
                and tool["result"]["game"]["game_id"] == report["game_id"] for tool in row.get("tool_results", ()))
    }
    assert len(older_private_confirmations) == 2
    assert older_private_confirmations <= {row["recipient_id"] for row in claim["private_recipient_memberships"]}
    assert "venue-event 编号表示入队顺序，不表示实际发生顺序" in client.instructions
    shared, = [row for row in chat if row["message_id"] == public_share["message_id"]]
    assert shared["quoted_message_id"] == group_message["quoted_message_id"]
    store = SQLiteAgentStore(run_dir / "agent.sqlite3")
    for message_id in (shared["quoted_message_id"], native["customer_message"]["quoted_message_id"]):
        snapshot = store.get_board_snapshot_by_message_id(shared["room_id"], message_id)
        assert [item.game_id for item in snapshot.items] == [report["game_id"]]

    refusal, = [row for row in evidence if "participation_before" in row]
    assert refusal["social_refusal_candidate"] is True
    assert report["social_refusal"] == {"status": "verified", "delivery_evidence_id": refusal["evidence_id"]}
    shop_invitation, = [row["shop_invitation"] for row in evidence if row.get("shop_invitation")]
    receipt = shop_invitation["receipt"]
    assert refusal["caused_by"] == receipt["message_id"]
    assert refusal["public_attendee_message_id"] == public_share["message_id"]
    assert refusal["customer_message"]["kind"] == "private" and refusal["outputs"]
    assert refusal["participation_before"] == refusal["participation_after"] == {
        "game_id": report["game_id"], "participant": None, "invitation_statuses": ["sent"],
    }
    assert refusal["earlier_shop_private_messages"] == [{key: receipt[key] for key in ("message_id", "text", "sent_at")}]
    reference = store.resolve_message_reference(conversation_id=receipt["conversation_id"], message_id=receipt["message_id"])
    assert reference.business_ref_id == shop_invitation["draft_id"] and reference.recipient_id == refusal["actor_id"]
    assert store.invite_drafts[reference.business_ref_id].game_id == report["game_id"]
    role = next(row for row in evidence if row["event_id"] == refusal["event_id"].removeprefix("delivery:"))
    personas = json.loads((run_dir / "personas.json").read_text(encoding="utf-8"))["personas"]
    public_name = next(person["name"] for person in personas if person["sender_id"] == public_share["actor_id"])
    assert role["decision"]["context"]["events"][-1]["details"]["note"] == (
        f"这条已收到的群消息显示的发送者昵称是「{public_name}」。")

    probes = [row for row in evidence if row.get("room_query")]
    assert len(probes) == 2
    for row in probes:
        assert "不用订" in row["customer_message"]["text"]
        assert row["room_query"] == {"intended_scope": "availability_only", "reservations_unchanged": True}
        query, = [tool["result"] for tool in row["tool_results"] if tool["name"] == "check_room_availability"]
        assert datetime.fromisoformat(query["start_at"]) == datetime.fromisoformat(row["virtual_at"])
        assert row["room_inventory_reads"] == [{
            "source": "agent_query", "observed_at": row["virtual_at"], "window_start_at": query["start_at"],
            "window_end_at": query["end_at"], "available_room_ids": query["available_room_ids"],
            "occupied_room_ids": query["occupied_room_ids"],
        }]
    world_events = {row["kind"]: row for row in evidence if row["kind"] in {"start", "occupied", "finish", "rebook"}}
    start = world_events["start"]["room_inventory_write"]
    finish = world_events["finish"]["room_inventory_write"]
    timing = world_events["finish"]["play_timing"]
    assert timing == {
        "game_id": report["game_id"], "planned_start_at": "2026-11-25T19:00:00+08:00",
        "planned_end_at": "2026-11-25T23:00:00+08:00", "actual_started_at": "2026-11-25T19:03:00+08:00",
        "actual_ended_at": "2026-11-25T23:03:00+08:00", "actual_duration_minutes": 240,
    }
    assert "reason 要指出哪条实际陈述与哪条证据冲突" in client.instructions
    rebook = world_events["rebook"]["room_inventory_write"]
    assert start["operation"] == "play_start" and start["status"] == "occupied"
    assert finish["operation"] == "play_end" and finish["status"] == "released"
    assert rebook["operation"] == "fixture_rebook" and rebook["status"] == "held"
    assert start["reservation_id"] == finish["reservation_id"] != rebook["reservation_id"]
    assert start["room_id"] == finish["room_id"] == rebook["room_id"] == report["room_id"]
    assert start["start_at"] == start["occurred_at"] and start["end_at"] == finish["occurred_at"]
    assert rebook["game_id"] is None and rebook["occurred_at"] == rebook["start_at"]
    overlap, = world_events["occupied"]["room_inventory_reads"]
    assert overlap["source"] == "overlap_probe" and start["room_id"] in overlap["occupied_room_ids"]
    before, after = [probe["room_inventory_reads"][0] for probe in probes]
    assert start["room_id"] in before["occupied_room_ids"] and start["room_id"] in after["available_room_ids"]
    assert (datetime.fromisoformat(finish["occurred_at"]) < datetime.fromisoformat(after["observed_at"])
            < datetime.fromisoformat(rebook["occurred_at"]) < datetime.fromisoformat(after["window_end_at"]))
    assert "不能倒推先前查询不实" in client.instructions

    raw = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    raw_roles = {row["event_id"]: row["decision"] for row in raw if "decision" in row}
    raw_deliveries = {row["delivery"]["event_id"]: row["delivery"] for row in raw if "delivery" in row}
    assert internal_tools <= {tool["name"] for tool in raw_deliveries[native["event_id"]]["tool_results"]}
    for row in evidence:
        if "decision" in row:
            recorded = raw_roles[row["event_id"]]
            assert {"raw", "diagnostics", "response_source", "prior_attempts"} <= recorded.keys()
            assert {"raw", "diagnostics", "response_source", "prior_attempts"}.isdisjoint(row["decision"])
            assert row["decision"]["context"] == recorded["context"] and row["decision"]["action"] == recorded["action"]
        if "tool_results" in row:
            recorded = raw_deliveries[row["event_id"]]
            assert row["tool_results"] == [tool for tool in recorded["tool_results"] if tool["name"] not in internal_tools]
            assert row["outputs"] == recorded["outputs"]

    client.paraphrase = True
    invalid = judge.judge(client.request["case_id"], evidence, trace_id="judge-replay-test", checks=client.request["checks"])
    assert invalid.status == "invalid" and invalid.verdict == "inconclusive"
    assert invalid.error == "findings must cover every requested check exactly once"


def test_venue_judge_room_snapshot_keeps_query_timestamp_distinct_from_the_future_slot():
    """预查未来用房的快照保留读取时刻，拒绝的写入和查询不能充当库存或留房证据。"""
    window = {"start_at": "2026-11-25T19:00:00+08:00", "end_at": "2026-11-25T23:00:00+08:00",
              "available_room_ids": ["东二包间", "西侧包间"], "occupied_room_ids": ["东一包间"]}
    read = {"name": "check_room_availability", "called": True, "allowed": True, "result": window}
    originals = [{"kind": "need", "event_id": "preflight", "virtual_at": "2026-11-25T17:55:00+08:00",
                  "tool_results": [read, read | {"allowed": False}, read | {"name": "reserve_room", "allowed": False}]}]
    evidence, = _judge_evidence(originals)
    assert evidence["tool_results"] == originals[0]["tool_results"]
    assert evidence["room_inventory_reads"] == [{
        "source": "agent_query", "observed_at": originals[0]["virtual_at"], "window_start_at": window["start_at"],
        "window_end_at": window["end_at"], "available_room_ids": window["available_room_ids"],
        "occupied_room_ids": window["occupied_room_ids"],
    }]
    assert "room_inventory_write" not in evidence
    assert "room_inventory_reads" not in originals[0]


@pytest.mark.parametrize(("reaction", "verdict", "delivery_checked", "status"), [
    ("inquiry", "fail", True, "not_observed"),
    ("wait", "inconclusive", False, "not_observed"),
    ("wrong_citation", "pass", True, "inconclusive"),
])
def test_venue_judge_requires_delivered_refusal_not_an_inquiry_or_wait(tmp_path, monkeypatch,
                                                               reaction, verdict, delivery_checked, status):
    """仅有因果投递不证明拒绝，静默和错引都不能计入拒绝覆盖。"""
    class RefusalControl(MockActor):
        def decide(self, context, *, trace_id):
            result = super().decide(context, trace_id=trace_id)
            event = context.events[-1]
            if event["kind"] == "opportunity" and event["details"].get("known_attendees"):
                if reaction == "inquiry":
                    return replace(result, action=ActorAction("send", "这桌还有位置吗？", "private", AGENT_ID))
                if reaction == "wait":
                    return replace(result, action=ActorAction("wait"))
            return result

    class RefusalJudgeClient:
        def complete(self, messages, *, trace_id, timeout_seconds):
            self.request = json.loads(messages[-1]["content"])
            context_id = self.request["evidence"][0]["evidence_id"]
            candidates = [row["evidence_id"] for row in self.request["evidence"] if row.get("social_refusal_candidate")]
            assert bool(candidates) is delivery_checked
            common = [{"check": check, "verdict": "pass", "evidence_ids": [context_id], "reason": "离线通过"}
                      for check in self.request["checks"] if check != "social_refusal"]
            refusal = {"check": "social_refusal", "verdict": verdict,
                       "evidence_ids": candidates if reaction == "inquiry" else [context_id],
                       "reason": "保留实际来话含义"}
            return json.dumps({"verdict": verdict, "findings": [*common, refusal]}, ensure_ascii=False)

    client = RefusalJudgeClient()
    scenario = runner.VenueScenario
    monkeypatch.setattr(runner, "MockActor", RefusalControl)
    monkeypatch.setattr(runner, "VenueScenario", lambda **kwargs: scenario(**(kwargs | {"judge": SemanticJudge(client)})))
    report = run_simulation(tmp_path / reaction, mode="mock", seed=42, scenario_date=date(2026, 11, 25), background_events=0)

    assert report["checks"]["social_refusal_delivery"] is delivery_checked
    assert report["social_refusal"]["status"] == status
    assert bool(report["social_refusal"]["delivery_evidence_id"]) is delivery_checked
    assert report["physical_play"]["status"] == "passed"
    assert report["independent_judge"]["status"] == ("invalid" if reaction == "wrong_citation" else "valid")
    if reaction == "inquiry":
        inquiry, = [row for row in client.request["evidence"] if row.get("social_refusal_candidate")]
        assert inquiry["customer_message"]["text"] == "这桌还有位置吗？"


def test_venue_does_not_wake_avoider_when_owner_invitation_is_not_delivered(tmp_path, monkeypatch):
    """店家草稿不算收到邀约，无真实送达则仍可打完原桌而缺少拒绝覆盖。"""
    monkeypatch.setattr(SandboxChannel, "publish_owner", lambda channel, **kwargs: {"status": "failed"})
    run_dir = tmp_path / "missing-social-invite"
    report = run_simulation(run_dir, mode="mock", seed=42, scenario_date=date(2026, 11, 25), background_events=0)

    assert report["status"] == "failed"
    assert report["physical_play"]["status"] == "passed"
    assert report["social_refusal"] == {"status": "not_observed", "delivery_evidence_id": None}
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    social, = [row for row in events if row["kind"] == "social_invite"]
    assert social["skipped"] == "店员私聊没有实际送达"
    assert not any(row.get("caused_by") == f"owner:{social['event_id']}" for row in events)


@pytest.mark.parametrize("case", ["exclusion", "text_reoffer", "recorded_reoffer", "missing_public_share"])
def test_social_refusal_evidence_separates_confirmed_decline_actual_reoffer_and_other_table_query(case):
    """只让实际收件与落库证明拒桌，文本提及原桌继续由评审判断其作用。"""
    def receipt(message_id, sequence, sender, recipient, text, *, direction="outbound", kind="private", source=None):
        return {"message_id": message_id, "sequence": sequence, "sender_id": sender, "recipient_id": recipient,
                "text": text, "direction": direction, "kind": kind, "source_message_id": source,
                "room_id": "room", "conversation_id": "dm" if kind == "private" else "group",
                "sent_at": f"2026-11-25T18:0{sequence}:00+08:00", "trace_id": "shop-trace"}

    invited = receipt("invite", 1, OWNER_ID, "a", "这桌你来吗？", direction="owner")
    shared = receipt("share", 2, "b", "room", "七点这桌我去。", direction="inbound", kind="group")
    inbound = receipt("input", 3, "a", AGENT_ID, "老板，今晚七点这桌我就不去了，帮我留意下别的局吧。", direction="inbound")
    exclusion = "今晚除了七点那桌，没查到别的现成局了。你想打哪天、什么玩法和档位？"
    reply_text = "七点这桌还给你留着，回来打吗？" if case == "text_reoffer" else exclusion
    reply = receipt("reply", 4, AGENT_ID, "a", reply_text, source="input")
    rows = [
        {"shop_invitation": {"game_id": "same-table", "status": "sent", "receipt": invited}},
        {"kind": "share_joined", "message_id": "share"},
        {"event_id": "delivery:refusal", "message_id": "input", "trace_id": "shop-trace", "actor_id": "a",
         "customer_message": {"text": inbound["text"]}, "social_refusal_candidate": True,
         "caused_by": "invite", "public_attendee_message_id": "share",
         "participation_before": {"game_id": "same-table", "invitation_statuses": ["sent"]},
         "participation_after": {"game_id": "same-table", "invitation_statuses": ["declined"], "participant": None},
         "tool_results": [{"name": "record_candidate_reply", "called": True, "allowed": True,
                           "result": {"game": {"game_id": "same-table"}, "recorded_status": "declined"}},
                          {"name": "search_current_games", "called": True, "allowed": True}]},
    ]
    chat = [invited, inbound, reply] if case == "missing_public_share" else [invited, shared, inbound, reply]
    if case == "recorded_reoffer":
        next_invite = receipt("invite-again", 5, OWNER_ID, "a", "再问一次，七点这桌来不来？", direction="owner")
        chat.append(next_invite)
        rows.append({"shop_invitation": {"game_id": "same-table", "status": "sent", "receipt": next_invite}})

    facts = add_social_refusal_facts(rows, chat)[2]["social_refusal_facts"]
    assert facts["source_verified"] is facts["declined_same_game_in_store"] is (case != "missing_public_share")
    assert facts["executed_lookup_tools"] == ["search_current_games"] and not facts["recorded_followup_tools"]
    assert facts["recorded_same_game_reoffers"] == (["invite-again"] if case == "recorded_reoffer" else [])
    if facts["source_verified"]:
        assert facts["actual_private_shop_replies"][0]["text"] == reply_text
    else:
        assert not facts["actual_private_shop_replies"]
    scenario = SimpleNamespace(social_refusal_delivery_id="delivery:refusal", checks={"social_refusal_delivery": True})
    verdict = "fail" if case == "text_reoffer" else "pass"
    judgment = {"status": "valid", "verdict": verdict, "findings": [
        {"check": "social_refusal", "verdict": verdict, "evidence_ids": ["delivery:refusal"]},
    ]}
    result = VenueScenario._social_refusal_report(scenario, judgment, facts)
    if case == "exclusion":
        assert result["status"] == "verified"
    elif case == "text_reoffer":
        assert result["status"] == "failed" and judgment["verdict"] == "fail"
    else:
        assert result["status"] == "inconclusive" and judgment["status"] == "invalid"


@pytest.mark.parametrize(("other", "followup", "reoffer", "include_reply", "expected"), [
    ("pass", "pass", False, True, "verified"),
    ("pass", "fail", False, True, "failed"),
    ("pass", "pass", False, False, "inconclusive"),
    ("pass", "pass", True, True, "inconclusive"),
    ("fail", "pass", False, True, "inconclusive"),
])
def test_only_isolated_refusal_wording_is_rechecked_with_both_channel_receipts(other, followup, reoffer, include_reply, expected):
    """保留原始判决，补判不能覆盖其他失败、重发邀约或缺少发送回执。"""
    requests = []

    class JudgeClient:
        def complete(self, messages, **_):
            payload = json.loads(messages[-1]["content"])
            requests.append(payload)
            assert payload["checks"] == ["wording"]
            assert [item["evidence_id"] for item in payload["evidence"]] == ["E01", "E02", "E03"]
            assert payload["evidence"][2]["original_invite_and_refusal_same_game_verified"] is True
            return json.dumps({"verdict": followup, "findings": [
                {"check": "wording", "verdict": followup, "evidence_ids": ["E01", "E02"] if include_reply else ["E01"],
                 "reason": "逐字核对拒绝和店家回复。"},
            ]}, ensure_ascii=False)

    scenario = SimpleNamespace(judge=SemanticJudge(JudgeClient()), _retry_timed_out_model=lambda x, *_args, **_kw: x)
    first = {"status": "valid", "verdict": "inconclusive", "raw": "保留原始模型结论", "findings": [
        {"check": "role_behavior", "verdict": "pass"}, {"check": "participation", "verdict": "pass"},
        {"check": "business_facts", "verdict": other}, {"check": "social_refusal", "verdict": "inconclusive"},
    ]}
    row = {"event_id": "delivery:refusal", "customer_message": {"text": "七点那桌我不去，帮我看看其他。"},
           "social_refusal_facts": {"source_verified": True, "declined_same_game_in_store": True,
             "recorded_same_game_reoffers": ["original-table-invitation"] if reoffer else [],
             "invitation_message_id": "original-invite", "recorded_followup_tools": [],
             "actual_private_shop_replies": [{"message_id": "actual-reply", "text": "今晚暂时没别的桌。"}]}}
    social = {"status": "inconclusive", "delivery_evidence_id": row["event_id"]}

    supplemental = VenueScenario._refine_inconclusive_refusal(scenario, first, row, social)

    assert social["status"] == expected and first["verdict"] == "inconclusive" and first["raw"] == "保留原始模型结论"
    assert bool(requests) is (not reoffer and other == "pass")
    assert bool(supplemental) == bool(requests)


def test_venue_read_only_probe_rejects_reservation_even_without_tool_receipt(tmp_path, monkeypatch):
    """房态查询仍正确时，也不能漏判正式工具回执之外发生的预约写入。"""
    send = SandboxChannel.send
    inserted = []

    def send_with_unreported_reservation(channel, message):
        response = send(channel, message)
        if not inserted and "不用订" in message.text:
            query, = [row["result"] for row in response["tool_results"] if row["name"] == "check_room_availability"]
            booking, _ = channel.runtime.store.reserve_room(conversation_id=channel.private_conversation_id(message.sender_id),
                game_id=None, start_at=query["start_at"], end_at=query["end_at"], trace_id=response["trace_id"],
                room_id=query["available_room_ids"][0])
            inserted.append(booking.reservation_id)
        return response

    monkeypatch.setattr(SandboxChannel, "send", send_with_unreported_reservation)
    run_dir = tmp_path / "unexpected-reservation"
    report = run_simulation(run_dir, mode="mock", seed=42, scenario_date=date(2026, 11, 25), background_events=0)

    assert report["status"] == "failed", report
    assert report["checks"]["agent_pre_end_read"] and report["checks"]["agent_post_end_read"]
    assert not report["checks"]["agent_pre_end_read_only"] and report["checks"]["agent_post_end_read_only"]
    booking_id, = inserted
    assert booking_id in SQLiteAgentStore(run_dir / "agent.sqlite3").room_reservations


def test_venue_does_not_rebook_if_an_attendee_has_not_left(tmp_path, monkeypatch):
    """虚拟时钟到了也必须等到实际离场，原房间仍占用。"""

    class OneAttendeeWaits(MockActor):
        def decide(self, context, *, trace_id):
            result = super().decide(context, trace_id=trace_id)
            if context.events[-1]["kind"] == "departure" and not getattr(self, "_waited", False):
                self._waited = True
                return replace(result, action=ActorAction("wait"), raw='{"action":"wait"}')
            return result

    monkeypatch.setattr(runner, "MockActor", OneAttendeeWaits)
    run_dir = tmp_path / "still-playing"
    report = run_simulation(run_dir, mode="mock", seed=42, scenario_date=date(2026, 11, 25), background_events=0)
    assert report["status"] == "failed"
    assert report["checks"]["actual_start"] and not report["checks"]["four_departures"]
    assert report["checks"]["actual_four_hour_finish"] is None
    assert report["checks"]["same_room_rebookable"] is None
    assert report["physical_play"]["status"] == "failed"
    assert set(report["physical_play"]["evidence_ids"]) == {"start", "occupied"}
    assert report["physical_play"]["actual_ended_at"] is None
    store = SQLiteAgentStore(run_dir / "agent.sqlite3")
    game = store.require_game(report["game_id"])
    assert game.is_playing
    assert store.room_reservations[game.play_reservation_id].status == "occupied"
    assert all(item.conversation_id != "venue-next-booking" for item in store.room_reservations.values())


def test_venue_reports_unavailable_model_without_inventing_traffic(tmp_path, monkeypatch):
    """角色模型传输失败应保留原始原因及报告，且不能补发固定话术。"""

    class BlockedRoleModel:
        def decide(self, context, *, trace_id):
            return ActorDecision(trace_id, "model", "transport_error", None, None, context.to_dict(),
                                 {"attempts": [{"http_status": 402}]}, error="RuntimeError")

    class UnreachableJudge:
        def judge(self, *args, **kwargs):
            raise AssertionError("没有双向证据时不应调用裁判")

    build = runner.build_runtime
    monkeypatch.setattr(runner, "load_dotenv_defaults", lambda path: None)
    monkeypatch.setattr(runner, "_real_roles", lambda: (BlockedRoleModel(), UnreachableJudge()))
    monkeypatch.setattr(runner, "build_runtime", lambda store, mode, **kwargs: build(store, "mock", **kwargs))
    run_dir = tmp_path / "model-blocked"
    report = run_simulation(run_dir, mode="real", seed=42, scenario_date=date(2026, 11, 25),
                            background_events=0, model_source="project")

    assert report["status"] == "inconclusive", report
    assert report["chat_events"] == 0
    assert report["game_id"] is None
    assert report["independent_judge"]["status"] == "not_run"
    assert all(row["http_statuses"] == [402] for row in report["model_transport_errors"])
    assert (run_dir / "events.jsonl").stat().st_size > 0
