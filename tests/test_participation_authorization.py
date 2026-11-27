"""回归真正的运行时与网关占席授权，时长未知时使用独立证据检查。"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from mahjong_agent_runtime import AgentRuntime, InMemoryAgentStore, SQLiteAgentStore, StaticAgentClient, UserMessage
from mahjong_agent_runtime.domains.delivered_game_offer import DELIVERED_GAME_OFFER, _offer_terms
from mahjong_agent_runtime.domains.participation_authorization import customer_participation_deadline
from mahjong_agent_runtime.domains.table_availability_authorization import explicit_customer_play_deadline
from mahjong_agent_runtime.llm import AgentLLMConfig, LLMCompletionText, OpenAICompatibleAgentClient
from mahjong_agent_runtime.models import ConversationRole, ConversationTurn, InviteStatus, QuotedMessageRef, using_clock
from mahjong_agent_runtime.task_context import TaskContextManager

from tests.test_agent_runtime import action_json


AT = datetime.fromisoformat("2026-11-25T19:37:00+08:00")


@pytest.mark.parametrize(("mode", "expected"), [
    ("conditional", False), ("direct", True), ("known_duration", True), ("transport_failure", False),
    ("changed_while_reviewing", False), ("visible_offer", True), ("pending_offer", False),
    ("other_game_offer", False), ("other_guest_offer", False), ("quoted_sent_invite", True),
    ("quoted_pending_invite", False), ("quoted_other_conversation", False), ("quoted_changed_invite", False),
    ("newer_other_game_offer", False), ("gate_to_store_join", False), ("gate_to_store_reply", False),
])
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_unknown_table_duration_preflight_protects_real_seat_write(tmp_path, monkeypatch, mode, expected, backend):
    """未知条件、错误来源和桌况变化无副作用，合法本人占席仍经原网关和 Store。"""
    store = InMemoryAgentStore() if backend == "memory" else SQLiteAgentStore(tmp_path / "seats.sqlite3")
    with using_clock(lambda: AT):
        requirement = {"game_type": "hangzhou_mahjong", "stake": "1", "planned_start_at": "2026-11-25T20:45:00+08:00"}
        if mode == "known_duration":
            requirement["duration_hours"] = 2
        game, _ = store.create_game(conversation_id="guest-thread", organizer_id="host", organizer_name="发起人",
            requirement=requirement, known_players=[{"customer_id": "host", "display_name": "发起人"}], trace_id="seed")
        TaskContextManager(store).prepare(UserMessage(conversation_id="guest-thread", sender_id="guest",
            sender_name="顾客", text="晚上有杭麻吗？", message_id="prior", sent_at=AT), trace_id="prepare")
        reference = None
        accepting = mode.endswith("_offer") or mode.startswith("quoted_")
        if mode.endswith("_offer"):
            store.append_assistant_turn("guest-thread", "20:45的杭麻还有位，现在给你留一个吗？", "offer", metadata={
                "delivery_status": "pending_operator_send" if mode == "pending_offer" else "delivered",
                DELIVERED_GAME_OFFER: {"game_id": "other-game" if mode == "other_game_offer" else game.game_id,
                                       "recipient_id": "another-guest" if mode == "other_guest_offer" else "guest",
                                       "terms": _offer_terms(game.to_dict())},
            })
            if mode == "newer_other_game_offer":
                store.append_assistant_turn("guest-thread", "隔壁21:30这桌也缺人，给你留吗？", "another-offer", metadata={
                    "delivery_status": "delivered", DELIVERED_GAME_OFFER: {"game_id": "different-table",
                        "recipient_id": "guest", "terms": _offer_terms(game.to_dict())},
                })
        elif mode.startswith("quoted_"):
            drafts, _ = store.create_invite_drafts(game_id=game.game_id, invitations=[{
                "customer_id": "guest", "display_name": "顾客", "message_text": "20:45杭麻还有位，你要来吗？",
            }], trace_id="draft-invite")
            draft = drafts[0]
            if mode != "quoted_pending_invite":
                store.update_invite_delivery_status(draft_id=draft.draft_id, status=InviteStatus.SENT,
                                                    reason="平台送达", trace_id="invite-delivered")
            invite_conversation = "other-thread" if mode == "quoted_other_conversation" else "guest-thread"
            message_id = "platform-invite"
            store.link_message_reference(conversation_id=invite_conversation, message_id=message_id,
                source_message_id=draft.draft_id, metadata={"platform_message_id": message_id})
            reference = QuotedMessageRef(message_id=message_id, conversation_id=invite_conversation)
            if mode == "quoted_other_conversation":
                store.append_assistant_turn("guest-thread", "20:45这桌也还有位，给你留吗？", "own-offer", metadata={
                    "delivery_status": "delivered", DELIVERED_GAME_OFFER: {"game_id": game.game_id,
                        "recipient_id": "guest", "terms": _offer_terms(game.to_dict())},
                })
            if mode == "quoted_changed_invite":
                store.update_game_requirement(game_id=game.game_id,
                    requirement_patch={"planned_start_at": "2026-11-25T21:00:00+08:00"},
                    reason="邀约发出后改期", trace_id="invite-changed")
        else:
            store.append_assistant_turn("guest-thread", "20:45这桌几点结束还没定。你能打到几点？", "offer",
                                       metadata={"delivery_status": "delivered"})

        class IndependentReview:
            """只模拟审查返回结构，正常运行时仍自行准备证据和验证数据库。"""

            def __init__(self):
                self.calls = []

            def complete(self, messages, *, trace_id, timeout_seconds):
                payload = json.loads(messages[-1]["content"])
                self.calls.append((payload, timeout_seconds))
                assert payload["agreed_duration_hours"] is None
                assert "00:45" not in messages[-1]["content"]
                if mode == "transport_failure":
                    raise TimeoutError("simulated reviewer deadline")
                if mode == "changed_while_reviewing":
                    store.update_game_requirement(game_id=game.game_id,
                        requirement_patch={"planned_start_at": "2026-11-25T21:00:00+08:00"},
                        reason="测试中改期开场", trace_id="change-during-review")
                if mode == "conditional":
                    return '{"scope":"unconfirmed","customer_ref":null,"shop_offer_ref":null}'
                customer_ref = next(row["ref"] for row in payload["conversation"] if row.get("current"))
                if accepting:
                    latest = next((row for row in reversed(payload["conversation"]) if row["speaker"] == "shop"), {})
                    assert bool(latest.get("verified_same_table_offer")) is expected
                    return json.dumps({"scope": "accept_delivered_offer", "customer_ref": customer_ref, "shop_offer_ref": 1})
                return json.dumps({"scope": "direct_now", "customer_ref": customer_ref, "shop_offer_ref": None})

        body = ("行。" if accepting else
                "我只能待到十一点，时间合适才给我留一位。" if mode in ("conditional", "known_duration") else
                "先给我留一位吧，顺便问一下几点散？")
        name = "record_candidate_reply" if mode in ("conditional", "gate_to_store_reply") else "join_game"
        if mode.startswith("gate_to_store_"):
            prior_claim = type(store).claim_idempotent_result

            def reschedule_on_claim(target_store, key, pending, **kwargs):
                if target_store is store and pending.name == name:
                    store.update_game_requirement(game_id=game.game_id,
                        requirement_patch={"planned_start_at": "2026-11-25T21:00:00+08:00"},
                        reason="网关认可旧桌况后并发改期", trace_id="changed-after-gateway")
                return prior_claim(target_store, key, pending, **kwargs)

            monkeypatch.setattr(type(store), "claim_idempotent_result", reschedule_on_claim)
        arguments = {"game_id": game.game_id, "customer_id": "guest", "display_name": "顾客"}
        if name == "record_candidate_reply":
            arguments["status"] = "accepted"
        reviewer = IndependentReview()
        planner = StaticAgentClient([action_json(objective_status="needs_tool", tool_calls=[{
            "name": name, "arguments": arguments, "reason": "处理本人留位",
        }]), action_json(objective_status="completed" if expected else "waiting_user",
                         reply_to_user="位给你留了。" if expected else "先确认一下你这次要不要留位。")])
        result = AgentRuntime(llm_client=planner, store=store, participation_authorization_client=reviewer).handle_user_message(
            UserMessage(conversation_id="guest-thread", sender_id="guest", sender_name="顾客", text=body,
                        message_id="current-seat-request", sent_at=AT, quoted_message=reference), trace_id="seat-runtime")
        tool = next(row for row in result.tool_results if row.name == name)
        assert (tool.allowed, tool.called) == (expected, expected), tool.error
        assert any(p.customer_id == "guest" and p.status == "confirmed"
                   for p in store.require_game(game.game_id).participants) == expected
        if mode == "known_duration":
            assert not reviewer.calls
        else:
            assert len(reviewer.calls) == 1
            assert reviewer.calls[0][1] <= 20
        if not expected:
            assert "未占席" in tool.error
        if mode.startswith("gate_to_store_"):
            assert tool.result["validation_error"]["code"] == "participation_terms_changed"
            assert store.require_game(game.game_id).planned_start_at.isoformat() == "2026-11-25T21:00:00+08:00"


@pytest.mark.parametrize("reply_review_enabled", [False, True])
def test_http_runtime_keeps_separate_write_preflight_independent_of_reply_review(reply_review_enabled):
    """切换对客话术审查不会撤掉真正模型链路的写授权检查。"""
    main = OpenAICompatibleAgentClient(AgentLLMConfig(api_key="local-test", model="test", base_url="http://localhost"))
    runtime = AgentRuntime(llm_client=main, reply_self_review_enabled=reply_review_enabled)
    actual = runtime.tool_execution_service.participation_authorization_client
    assert isinstance(actual, OpenAICompatibleAgentClient)
    assert actual is not main and actual is not runtime.creation_authorization_client
    assert actual.config.temperature == 0
    assert actual.config.max_tokens == 1200


@pytest.mark.parametrize("scope", ["direct_now", "accept_delivered_offer"])
def test_v8_direct_booking_after_tentative_reply_keeps_exact_authorization_evidence(tmp_path, scope):
    """本人当前直接报名可独立核验，未获认可的上轮方案不可替代授权。"""
    store = SQLiteAgentStore(tmp_path / "v8-participation.sqlite3")
    sent_at = datetime.fromisoformat("2026-11-26T18:35:00+08:00")
    prior = UserMessage("guest-thread", "guest", "顾客", "老板 七点半那桌杭麻无烟的 我七点能到 打到十点半 帮我留一个",
                        message_id="prior-0151", sent_at=sent_at)
    tentative = "七点半那桌杭麻无烟的找到了，1～3元，现在一位，还缺三位。这桌现在还没约定打多久，可能到不了十点半。你要不要先按这桌来？"
    table_conditions = {"game_type": "hangzhou_mahjong", "stake_range": {"min": 1, "max": 3}, "smoke_preference": "no_smoking",
                        "planned_start_at": "2026-11-26T19:30:00+08:00"}
    with using_clock(lambda: sent_at):
        game, _ = store.create_game(conversation_id="host-thread", organizer_id="host", organizer_name="发起人",
            requirement={**table_conditions, "max_duration_hours": 2.5},
            known_players=[{"customer_id": "host", "display_name": "发起人"}], trace_id="seed")
        lookup = StaticAgentClient([action_json(objective_status="needs_tool", tool_calls=[{
            "name": "search_current_games", "arguments": {"lookup_scope": "specific_table", "requirement": table_conditions},
            "reason": "先查顾客所指的原桌",
        }]), action_json(objective_status="waiting_user", reply_to_user=tentative)])
        previous = AgentRuntime(llm_client=lookup, store=store).handle_user_message(prior, trace_id="prior")
        assert any(tool.name == "search_current_games" and tool.called and tool.allowed for tool in previous.tool_results)
        store.append_assistant_turn("guest-thread", tentative, "shop", metadata={"delivery_status": "delivered"})

    raw = json.dumps({"scope": scope, "customer_ref": 3, "shop_offer_ref": 2 if scope == "accept_delivered_offer" else None})

    class Review:
        """固定审查结果，仅核验真实网关和响应原件追踪。"""

        def complete(self, messages, **kwargs):
            history = json.loads(messages[-1]["content"])["conversation"]
            assert history[0]["text"] == prior.text
            assert history[1]["text"] == tentative and history[1]["verified_same_table_offer"] is False
            assert history[2]["current"] is True
            return LLMCompletionText(raw, {"attempts": [{"http_status": 200, "body_complete": True}]})

    allowed = scope == "direct_now"
    planner = StaticAgentClient([action_json(objective_status="needs_tool", tool_calls=[{
        "name": "join_game", "arguments": {"game_id": game.game_id, "customer_id": "guest", "display_name": "顾客"},
        "reason": "处理本人报名",
    }]), action_json(objective_status="completed" if allowed else "waiting_user",
                     reply_to_user="好，给你报上了。" if allowed else "再确认一下你现在要留位吗？")])
    runtime = AgentRuntime(llm_client=planner, store=store, participation_authorization_client=Review())
    at = sent_at.replace(minute=37)
    with using_clock(lambda: at):
        result = runtime.handle_user_message(UserMessage("guest-thread", "guest", "顾客", "行 就按这桌来 帮我留一个",
            message_id="current-0151", sent_at=at), trace_id="v8-0151")
    tool = next(entry for entry in result.tool_results if entry.name == "join_game")
    assert tool.called is allowed and tool.allowed is allowed, tool.error
    assert store.require_game(game.game_id).remaining_seats() == (2 if allowed else 3)
    step = "participation_authorization_result" if allowed else "participation_authorization_response_rejected"
    event = next(entry.content for entry in runtime.trace_recorder.get_trace("v8-0151") if entry.step == step)
    assert event["raw"] == raw and event["response_source"]["trace_id"].startswith("v8-0151:participation:")
    if not allowed:
        assert event["reason"] == "participation authorization lacks current customer or delivered offer evidence"
        assert event["diagnostics"]["attempts"][0]["body_complete"] is True
    store._connection.close()


@pytest.mark.parametrize(("mode", "extra_requirement", "allowed", "agreed_end"), [
    ("repeats", {}, False, None),
    ("repeats", {"max_duration_hours": 2}, False, None),
    ("repeats", {"duration_hours": 4}, False, "23:30"),
    ("repeats", {"duration_hours": 2.5}, True, None),
    ("agrees_only", {}, False, None),
    ("agrees_to_changed_old_offer", {"duration_hours": 2.5}, False, "23:30"),
    ("explicitly_plays_full_table", {}, True, None),
    ("explicitly_changes_limit", {"duration_hours": 4}, True, None),
    ("changed_after_gateway", {"duration_hours": 2.5}, False, "23:30"),
])
def test_real_0103_hard_play_cutoff_requires_matching_agreed_table_end(
    tmp_path, monkeypatch, mode, extra_requirement, allowed, agreed_end,
):
    """即使占席审核认可同桌提议，仍用真实散场约定保护本人硬离店条件。"""
    prior_at = datetime.fromisoformat("2026-11-26T18:35:00+08:00")
    current_at = prior_at.replace(minute=37)
    first = "老板好，看到群里那桌19点半的川麻无烟，底注0.5到1块，还有位吗，我18点50左右能到，最多打到22点，能搭一桌吗？"
    tentative_offer = ("那桌19点半川麻无烟、0.5到1块，现在两个还差两个，18点50到正好赶得上开场，可以搭。"
                       "就是这桌还没约定打到几点，你说的22点走一般没问题，但我不给你打包票。要给你报这一席吗？")
    typical = "老板好，麻烦帮我报这一席吧，18点50左右到，最多打到22点，能搭一桌就行。"
    body = {
        "agrees_only": "好，那帮我报这一席吧。",
        "agrees_to_changed_old_offer": "好，那帮我报这一席吧。",
        "explicitly_plays_full_table": "我可以打到散场，给我报这一席吧。",
        "explicitly_changes_limit": "麻烦帮我报这一席吧，最多打到23:30。",
    }.get(mode, typical)
    store = SQLiteAgentStore(tmp_path / "v3-participation.sqlite3")
    terms = {"game_type": "sichuan_mahjong", "smoke_preference": "no_smoking",
             "stake_range": {"min": 0.5, "max": 1}, "planned_start_at": "2026-11-26T19:30:00+08:00"}
    with using_clock(lambda: prior_at):
        game, _ = store.create_game(conversation_id="host-thread", organizer_id="host", organizer_name="发起人",
            requirement=terms | extra_requirement, known_players=[{"customer_id": "other", "display_name": "牌友"}],
            trace_id="seed")
        if "max_duration_hours" not in extra_requirement and "duration_hours" not in extra_requirement:
            assert game.planned_end_at.isoformat() == "2026-11-26T23:30:00+08:00"
        lookup_planner = StaticAgentClient([action_json(objective_status="needs_tool", tool_calls=[{
            "name": "search_current_games", "arguments": {"lookup_scope": "specific_table", "requirement": terms},
            "reason": "按原桌线索查询",
        }]), action_json(objective_status="waiting_user", reply_to_user=tentative_offer)])
        prior_message = UserMessage("guest-thread", "guest", "顾客", first, message_id="exact-0103-prior", sent_at=prior_at)
        prior = AgentRuntime(llm_client=lookup_planner, store=store).handle_user_message(prior_message, trace_id="prior")
        queried = next(row for row in prior.tool_results if row.name == "search_current_games")
        assert queried.called and queried.allowed and queried.result["matches"][0]["game"]["game_id"] == game.game_id
        store.append_assistant_turn("guest-thread", tentative_offer, "offer-delivered", metadata={
            "delivery_status": "delivered", DELIVERED_GAME_OFFER: {
                "game_id": game.game_id, "recipient_id": "guest", "terms": _offer_terms(game.to_dict()),
            },
        })
        if mode == "agrees_to_changed_old_offer":
            store.update_game_requirement(game_id=game.game_id, requirement_patch={"duration_hours": 4},
                reason="同一桌在回复顾客报名之前改了时长", trace_id="change-after-offer")

    class AcceptsTentativeOffer:
        """重放 V3 独立模型误认可送达提议的判定。"""

        calls = 0

        def complete(self, messages, **kwargs):
            self.calls += 1
            history = json.loads(messages[-1]["content"])["conversation"]
            offer = next(entry for entry in reversed(history) if entry["speaker"] == "shop")
            assert offer["verified_same_table_offer"] and offer["text"] == tentative_offer
            current = next(row for row in history if row.get("current"))
            return json.dumps({"scope": "accept_delivered_offer", "customer_ref": current["ref"],
                               "shop_offer_ref": offer["ref"]})

    if mode == "changed_after_gateway":
        prior_claim = type(store).claim_idempotent_result

        def lengthen_during_claim(target, key, pending, **kwargs):
            if target is store and pending.name == "join_game":
                store.update_game_requirement(game_id=game.game_id, requirement_patch={"duration_hours": 4},
                    reason="写入前其它会话改时长", trace_id="table-change")
            return prior_claim(target, key, pending, **kwargs)

        monkeypatch.setattr(type(store), "claim_idempotent_result", lengthen_during_claim)
    planner = StaticAgentClient([action_json(objective_status="needs_tool", tool_calls=[{
        "name": "join_game", "arguments": {"game_id": game.game_id, "customer_id": "guest", "display_name": "顾客"},
        "reason": "处理本人报名",
    }]), action_json(objective_status="completed" if allowed else "waiting_user",
                     reply_to_user="好，给你报上了。" if allowed else "这桌还得先确认能不能22点离开，位还没报。")])
    reviewer = AcceptsTentativeOffer()
    runtime = AgentRuntime(llm_client=planner, store=store, participation_authorization_client=reviewer)
    with using_clock(lambda: current_at):
        res = runtime.handle_user_message(UserMessage("guest-thread", "guest", "顾客", body,
            message_id="exact-0103-enrollment", sent_at=current_at), trace_id="join")
    tool = next(row for row in res.tool_results if row.name == "join_game")
    assert (tool.called, tool.allowed) == (allowed, allowed), tool.error
    assert reviewer.calls == (0 if "duration_hours" in extra_requirement else 1)
    assert any(p.customer_id == "guest" and p.status == "confirmed"
               for p in store.require_game(game.game_id).participants) is allowed
    if not allowed:
        error = tool.result["validation_error"]
        assert error["code"] == "participation_departure_requires_confirmation"
        assert error["customer_deadline"]["must_end_by"] == "2026-11-26T22:00:00+08:00"
        assert error["customer_deadline"]["source_message_id"] == (
            "exact-0103-prior" if mode.startswith("agrees_") else "exact-0103-enrollment"
        )
        assert error["agreed_end_at"] == (f"2026-11-26T{agreed_end}:00+08:00" if agreed_end else None)
    store._connection.close()


@pytest.mark.parametrize(("text", "deadline"), [
    ("我最多打到晚上十点", "2026-11-25T22:00:00+08:00"),
    ("我今晚最多打到凌晨一点", "2026-11-26T01:00:00+08:00"),
    ("我最多待到明天凌晨一点", "2026-11-26T01:00:00+08:00"),
    ("22点必须走", "2026-11-25T22:00:00+08:00"),
    ("我想打到22点", None),
    ("我最晚22点能到", None),
    ("这桌22点结束吗？", None),
    ("他最多打到22点", None),
])
def test_participation_deadline_identifies_only_own_hard_departure(text, deadline):
    """在当前来源锚定明确的本人时限，并区分柔性想法、到店、桌况和他人的时间。"""
    evidence = explicit_customer_play_deadline(text, sent_at=AT)
    assert (evidence["must_end_by"] if evidence else None) == deadline
    assert bool(evidence) == bool(deadline)


def test_unresolved_hard_departure_does_not_turn_into_unconditional_participation():
    """未能核实的具体离店钟点仍然是报名待决条件。"""
    assert explicit_customer_play_deadline("22点左右必须走", sent_at=AT) == {
        "must_end_by": None, "clock_text": "22点",
    }


def test_legacy_customer_deadline_uses_original_trace_without_inventing_a_message_id():
    """旧会话缺少平台消息编号时，仍可凭原始轮次来源保留顾客条件。"""
    store = InMemoryAgentStore()
    with using_clock(lambda: AT):
        game, _ = store.create_game(conversation_id="host-thread", organizer_id="host", organizer_name="店家",
            requirement={"game_type": "sichuan_mahjong", "planned_start_at": "2026-11-25T20:45:00+08:00"},
            known_players=[], trace_id="seed")
        store.append_turn("guest-thread", ConversationTurn(
            ConversationRole.USER, "我最多打到22点", "legacy-trace", sender_id="guest",
        ))
        store.append_assistant_turn("guest-thread", "散场还没定，要报吗？", "table-offer", metadata={
            "delivery_status": "delivered", DELIVERED_GAME_OFFER: {
                "game_id": game.game_id, "recipient_id": "guest", "terms": _offer_terms(game.to_dict()),
            },
        })
        reply = UserMessage("guest-thread", "guest", "顾客", "好，给我报一下。")
        assert customer_participation_deadline(store, reply, game) == {
            "must_end_by": "2026-11-25T22:00:00+08:00", "clock_text": "22点",
            "source_message_id": None, "source_trace_id": "legacy-trace",
        }
