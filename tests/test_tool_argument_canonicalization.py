from __future__ import annotations

from mahjong_agent_runtime import InMemoryAgentStore, ToolCall, ToolGateway
from mahjong_agent_runtime.domains.tools.search_tools import canonical_search_customers_arguments


def test_equivalent_game_search_arguments_share_one_idempotency_key() -> None:
    gateway = ToolGateway(InMemoryAgentStore())
    first = ToolCall(
        name="search_current_games",
        arguments={
            "requirement": {
                "game_type": "hangzhou_mahjong",
                "stake": "1",
                "smoke_preference": "无烟",
            },
            "limit": 5,
        },
        reason="先查当前局",
    )
    repeated = ToolCall(
        name="search_current_games",
        arguments={
            "requirement": {
                "game_type": "hangzhou_mahjong",
                "stake": "1",
                "base_stake": 1.0,
                "stake_label": "1",
                "smoke_preference": "无烟",
            },
            "limit": 5,
        },
        reason="重复确认当前局",
    )

    first_result = gateway.execute(
        first,
        trace_id="trace-first",
        conversation_id="conversation-1",
        sender_id="customer-1",
        sender_name="客户",
        step_index=1,
        source_message_id="message-1",
    )
    repeated_result = gateway.execute(
        repeated,
        trace_id="trace-repeated",
        conversation_id="conversation-1",
        sender_id="customer-1",
        sender_name="客户",
        step_index=2,
        source_message_id="message-1",
    )

    assert first_result.called is True
    assert first_result.deduplicated is False
    assert repeated_result.called is True
    assert repeated_result.deduplicated is True
    assert repeated_result.idempotency_key == first_result.idempotency_key


def test_candidate_search_idempotency_identity_includes_game_id() -> None:
    first = canonical_search_customers_arguments(
        {"game_id": "game-a", "requirement": {"stake": "1"}}
    )
    second = canonical_search_customers_arguments(
        {"game_id": "game-b", "requirement": {"stake": "1"}}
    )

    assert first["game_id"] == "game-a"
    assert second["game_id"] == "game-b"
    assert first != second


def test_same_trace_without_source_message_deduplicates_equivalent_tool_proposals() -> None:
    gateway = ToolGateway(InMemoryAgentStore())
    call = ToolCall(
        name="search_current_games",
        arguments={"requirement": {"game_type": "hangzhou_mahjong", "stake": "1"}},
        idempotency_key="model-generated-key-is-not-authoritative",
    )

    first = gateway.execute(
        call,
        trace_id="trace-without-source-message",
        conversation_id="conversation-without-source-message",
        sender_id="customer-1",
        sender_name="客户",
        step_index=1,
    )
    repeated = gateway.execute(
        call,
        trace_id="trace-without-source-message",
        conversation_id="conversation-without-source-message",
        sender_id="customer-1",
        sender_name="客户",
        step_index=4,
    )

    assert first.called is True
    assert first.deduplicated is False
    assert repeated.called is True
    assert repeated.deduplicated is True
    assert repeated.idempotency_key == first.idempotency_key
    assert (first.idempotency_key or "").startswith(
        "trace:trace-without-source-message:conversation:conversation-without-source-message:"
    )
