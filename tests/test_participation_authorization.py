"""回归真正的运行时与网关占席授权，时长未知时使用独立证据检查。"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from mahjong_agent_runtime import AgentRuntime, InMemoryAgentStore, SQLiteAgentStore, StaticAgentClient, UserMessage
from mahjong_agent_runtime.domains.delivered_game_offer import DELIVERED_GAME_OFFER, _offer_terms
from mahjong_agent_runtime.llm import AgentLLMConfig, OpenAICompatibleAgentClient
from mahjong_agent_runtime.models import InviteStatus, QuotedMessageRef, using_clock
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
