from __future__ import annotations

from .venue_acceptance import AGENT_ID, evaluate_venue_acceptance


def inbound(key, customer, *, kind="private", quoted=None):
    return {
        "message_id": key, "conversation_id": f"dm-{customer}" if kind == "private" else "the-group",
        "kind": kind, "direction": "inbound", "sender_id": customer,
        "recipient_id": AGENT_ID if kind == "private" else "the-group", "quoted_message_id": quoted,
        "text": "客户表达本轮的目标或补充条件",
    }


def reply(key, customer, *, source, kind="private"):
    return {
        "message_id": key, "conversation_id": f"dm-{customer}" if kind == "private" else "the-group",
        "kind": kind, "direction": "outbound", "sender_id": AGENT_ID,
        "recipient_id": customer if kind == "private" else "the-group", "source_message_id": source,
        "text": "老板实际答复",
    }


def tool(name, **result):
    return {"name": name, "called": True, "allowed": True, "result": result}


def test_counts_interleaved_personal_dialogues_separately_from_chatter_and_unlinked_queries():
    group = inbound("group", "a", kind="group")
    peer = {**inbound("peer", "a"), "direction": "peer", "recipient_id": "b"}
    messages = [
        group, reply("group-answer", "a", source="group", kind="group"), peer,
        inbound("a1", "a"), reply("ar1", "a", source="a1"),
        inbound("b1", "b"), reply("br1", "b", source="b1"),
        inbound("a2", "a", quoted="ar1"), reply("ar2", "a", source="a2"),
        inbound("b2", "b"), reply("br2", "b", source="b2"),
        inbound("c1", "c"), reply("cr1", "c", source="c1"),
        inbound("c2", "c", quoted="br2"), reply("cr2", "c", source="c2"),
        inbound("c3", "c"), reply("wrong-recipient", "b", source="c3"),
    ]
    turns = [{"message_id": "b1", "actor_id": "b", "event_id": "delivery-b1"},
             {"message_id": "b2", "actor_id": "b", "caused_by": "delivery-b1"}]
    result = evaluate_venue_acceptance(messages=messages, turns=turns, outcomes=[])

    assert result["concurrent_private_conversations"] == 2
    assert result["multi_round_private_customers"] == 2
    assert result["conversations"]["a"] == {
        "group_messages": 1, "group_rounds": 1, "group_to_private_rounds": 0,
        "private_messages": 2, "private_rounds": 2, "causal_private_followups": 1, "longest_private_chain": 2,
    }
    assert result["conversations"]["b"]["longest_private_chain"] == 2
    assert result["conversations"]["c"]["private_messages"] == 3
    assert result["conversations"]["c"]["private_rounds"] == 2
    assert result["conversations"]["c"]["longest_private_chain"] == 1


def test_accepts_state_backed_formation_and_business_non_formation():
    identifiers = ("completed", "formed", "waiting", "no_match", "declined", "cancelled", "expired_unfilled")
    messages = [item for key in identifiers for item in (inbound(key, key), reply(f"r-{key}", key, source=key))]
    waiting = {"demand_id": "d1", "sender_id": "waiting", "status": "active"}
    game = {
        "game_id": "g1", "status": "finished", "organizer_id": "completed", "remaining_seats": 0, "seats_total": 4,
        "actual_attendee_ids": ["completed", "formed", "x", "y"], "play_reservation_id": "room-r1",
        "actual_started_at": "2031-04-01T18:00:00+08:00", "actual_ended_at": "2031-04-01T22:00:00+08:00",
        "participants": [{"customer_id": name, "status": "confirmed"} for name in ("completed", "formed", "x", "y")],
    }
    released_room = {"reservation_id": "room-r1", "game_id": "g1", "status": "released"}
    cancelled_game = {"game_id": "g2", "status": "cancelled", "organizer_id": "cancelled"}
    expired_game = {"game_id": "g3", "status": "cancelled", "closed_reason": "expired_without_full_table",
                    "organizer_id": "expired_unfilled"}
    turns = [
        {"message_id": "completed", "tool_results": [tool("create_game", game=game)]},
        {"message_id": "formed", "state_transitions": [{"entity_type": "game_participant", "entity_id": "g1:formed"}]},
        {"message_id": "waiting", "tool_results": [tool("register_waiting_demand", waiting_demand=waiting)]},
        {"message_id": "no_match", "tool_results": [tool("search_current_games", matches=[], alternatives=[])]},
        {"message_id": "cancelled", "tool_results": [tool("update_game_status", game=cancelled_game)]},
    ]
    evidence = [
        {"game": game, "reservation": released_room, "duration_hours": 4}, {"game": game}, {"waiting_demand": waiting},
        {}, {"explicit_decline": True, "game": game}, {"game": cancelled_game}, {"game": expired_game},
    ]
    outcomes = [{"customer_id": key, "kind": key, "source_message_ids": [key], **facts}
                for key, facts in zip(identifiers, evidence, strict=True)]

    result = evaluate_venue_acceptance(messages=messages, turns=turns, outcomes=outcomes)

    assert result["status"] == "passed"
    assert result["outcome_counts"] == dict.fromkeys(identifiers, 1)
    assert result["valid_non_formation_count"] == 5


def test_silent_or_failed_query_cannot_be_used_as_a_valid_non_formation_outcome():
    messages = [inbound("a1", "a"), inbound("b1", "b"), reply("br1", "b", source="b1")]
    turns = [
        {"message_id": "a1", "actor_id": "a", "trace_id": "trace-a", "runtime_status": "failed", "error_type": "TimeoutError"},
        {"message_id": "b1", "actor_id": "b", "tool_results": [tool("search_current_games", matches=[{"game_id": "g"}], alternatives=[])]},
    ]
    outcomes = [{"customer_id": name, "kind": "no_match", "source_message_ids": [name + "1"]} for name in ("a", "b")]
    result = evaluate_venue_acceptance(messages=messages, turns=turns, outcomes=outcomes)

    assert [item["status"] for item in result["outcomes"]] == ["inconclusive", "failed"]
    assert result["valid_non_formation_count"] == 0
    assert result["technical_failures"] == [{"customer_id": "a", "message_id": "a1", "trace_id": "trace-a", "kind": "timeout"}]


def test_options_backed_by_a_search_can_be_offered_without_claiming_formation():
    messages = [inbound("a1", "a"), reply("ar1", "a", source="a1")]
    turns = [{"message_id": "a1", "actor_id": "a", "runtime_status": "needs_human",
              "tool_results": [tool("search_current_games", matches=[], alternatives=[{"game": {"game_id": "g1"}}])]}]
    options = {"customer_id": "a", "kind": "options_found", "source_message_ids": ["a1"]}
    outcomes = [{**options, "game_id": "g1"}, {**options, "game_id": "g2"}]
    result = evaluate_venue_acceptance(messages=messages, turns=turns, outcomes=outcomes)
    assert [item["status"] for item in result["outcomes"]] == ["passed", "failed"]
    assert result["outcome_counts"] == {"options_found": 1}
    assert result["valid_non_formation_count"] == 0
    assert result["technical_failures"] == []


def test_checks_known_reply_constraints_without_using_hidden_or_other_customers_facts():
    messages = [inbound("b1", "b"), inbound("a1", "a"), reply("ar1", "a", source="a1"),
                inbound("a2", "a"), reply("ar2", "a", source="a2")]
    known = {"smoke": {"value": "no_smoking", "source_message_id": "a1"}}
    constraints = [
        {"message_id": "a2", "known_facts": known, "reply_facts": {"smoke": "no_smoking"}, "asked_fields": ["time"]},
        {"message_id": "a2", "known_facts": known, "asked_fields": ["smoke"], "tool_facts": {"smoke": "smoking"}},
        {"message_id": "a2", "known_facts": {"smoke": {"value": "smoking", "source_message_id": "b1"}}, "reply_facts": {}},
        {"message_id": "a1", "known_facts": {"smoke": {"value": "smoking", "source_message_id": "a2"}}, "reply_facts": {}},
        {"message_id": "a2", "known_facts": known, "asked_fields": ["smoke"], "allowed_reconfirm_fields": ["smoke"]},
    ]
    result = evaluate_venue_acceptance(messages=messages, turns=[], outcomes=[], constraints=constraints)
    assert [row["status"] for row in result["turn_constraints"]] == ["passed", "failed", "inconclusive", "inconclusive", "passed"]
    assert result["turn_constraints"][1]["violations"] == ["conflicting_tool_facts:smoke", "reasked:smoke"]
