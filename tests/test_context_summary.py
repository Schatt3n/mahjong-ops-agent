from __future__ import annotations

import json

from mahjong_agent_runtime import (
    AgentRuntime,
    ContextSummaryManager,
    ContextSummaryPolicy,
    CustomerProfile,
    InMemoryAgentStore,
    InMemoryTraceRecorder,
    StaticAgentClient,
    ToolGateway,
    UserMessage,
)
from mahjong_agent_runtime.context import AgentContextBuilder
from mahjong_agent_runtime.tracing import trace_steps


def test_context_summary_runs_after_turn_and_writes_checkpoint() -> None:
    store = InMemoryAgentStore()
    trace = InMemoryTraceRecorder()
    main_client = StaticAgentClient(
        [
            agent_action(
                objective_status="completed",
                reasoning_summary="本轮直接回复用户。",
                reply_to_user="好，我帮你问问。",
            )
        ]
    )
    summary_client = StaticAgentClient(
        [
            json.dumps(
                {
                    "summary": "张哥想组杭麻0.5，人齐开，目前系统准备帮他问人。",
                    "facts": {
                        "intent": "find_players",
                        "game_type": "hangzhou_mahjong",
                        "stake": "0.5",
                        "start_time_kind": "asap_when_full",
                    },
                    "open_questions": [],
                    "confidence": 0.91,
                },
                ensure_ascii=False,
            )
        ]
    )
    summary_manager = ContextSummaryManager(
        store=store,
        llm_client=summary_client,
        trace_recorder=trace,
        policy=ContextSummaryPolicy(
            min_turns_before_summary=2,
            min_turns_since_last_summary=1,
            max_recent_tokens_before_summary=1,
        ),
    )
    runtime = AgentRuntime(
        llm_client=main_client,
        store=store,
        trace_recorder=trace,
        context_summary_manager=summary_manager,
    )

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="summary_runtime",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我组个0.5",
            message_id="summary_msg_001",
        ),
        trace_id="trace_summary_runtime",
    )

    checkpoint = store.get_conversation_checkpoint("summary_runtime")
    assert result.final_reply == "好，我帮你问问。"
    assert checkpoint is not None
    assert checkpoint.summary == "张哥想组杭麻0.5，人齐开，目前系统准备帮他问人。"
    assert checkpoint.facts["stake"] == "0.5"
    assert result.state_transitions[-1].entity_type == "conversation_checkpoint"
    steps = trace_steps(trace.get_trace("trace_summary_runtime"))
    assert "context_summary_checked" in steps
    assert "context_summary_prompt" in steps
    assert "context_summary_response" in steps
    assert "context_summary_saved" in steps


def test_context_summary_failure_does_not_change_final_reply_or_checkpoint() -> None:
    store = InMemoryAgentStore()
    trace = InMemoryTraceRecorder()
    main_client = StaticAgentClient(
        [
            agent_action(
                objective_status="completed",
                reasoning_summary="本轮直接回复用户。",
                reply_to_user="我先看看。",
            )
        ]
    )
    summary_client = StaticAgentClient(["not json"])
    runtime = AgentRuntime(
        llm_client=main_client,
        store=store,
        trace_recorder=trace,
        context_summary_manager=ContextSummaryManager(
            store=store,
            llm_client=summary_client,
            trace_recorder=trace,
            policy=ContextSummaryPolicy(
                min_turns_before_summary=2,
                min_turns_since_last_summary=1,
                max_recent_tokens_before_summary=1,
            ),
        ),
    )

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="summary_invalid",
            sender_id="zhang",
            sender_name="张哥",
            text="有人吗",
            message_id="summary_invalid_msg_001",
        ),
        trace_id="trace_summary_invalid",
    )

    assert result.final_reply == "我先看看。"
    assert store.get_conversation_checkpoint("summary_invalid") is None
    steps = trace_steps(trace.get_trace("trace_summary_invalid"))
    assert "context_summary_contract_error" in steps
    assert "final_output" in steps


def test_context_builder_reads_checkpoint_created_by_summary() -> None:
    store = InMemoryAgentStore()
    trace = InMemoryTraceRecorder()
    runtime = AgentRuntime(
        llm_client=StaticAgentClient(
            [
                agent_action(
                    objective_status="completed",
                    reasoning_summary="本轮直接回复用户。",
                    reply_to_user="好。",
                )
            ]
        ),
        store=store,
        trace_recorder=trace,
        context_summary_manager=ContextSummaryManager(
            store=store,
            llm_client=StaticAgentClient(
                [
                    json.dumps(
                        {
                            "summary": "张哥上一轮确认要组杭麻1块，人齐开。",
                            "facts": {"intent": "find_players", "stake": "1", "start_time_kind": "asap_when_full"},
                            "open_questions": ["还需要确认烟况"],
                            "confidence": 0.8,
                        },
                        ensure_ascii=False,
                    )
                ]
            ),
            trace_recorder=trace,
            policy=ContextSummaryPolicy(
                min_turns_before_summary=2,
                min_turns_since_last_summary=1,
                max_recent_tokens_before_summary=1,
            ),
        ),
    )
    runtime.handle_user_message(
        UserMessage(
            conversation_id="summary_context_builder",
            sender_id="zhang",
            sender_name="张哥",
            text="组1块",
            message_id="summary_context_msg_001",
        ),
        trace_id="trace_summary_context",
    )

    built = AgentContextBuilder(store, ToolGateway(store)).build(
        UserMessage(
            conversation_id="summary_context_builder",
            sender_id="zhang",
            sender_name="张哥",
            text="烟都行",
            message_id="summary_context_msg_002",
        ),
        trace_id="trace_summary_context_2",
    )

    assert built.payload["conversation_checkpoint"]["summary"] == "张哥上一轮确认要组杭麻1块，人齐开。"
    assert built.payload["conversation_checkpoint"]["facts"]["stake"] == "1"
    assert built.payload["conversation_checkpoint"]["open_questions"] == ["还需要确认烟况"]
    assert built.payload["context_budget"]["conversation_checkpoint_present"] is True


def test_context_summary_respects_confidence_threshold() -> None:
    store = InMemoryAgentStore()
    trace = InMemoryTraceRecorder()
    store.append_user_turn(
        UserMessage(
            conversation_id="summary_confidence",
            sender_id="zhang",
            sender_name="张哥",
            text="随便看看",
            message_id="summary_confidence_seed",
        ),
        "trace_seed_summary_confidence",
    )
    store.append_assistant_turn("summary_confidence", "我先看看。", "trace_seed_summary_confidence")
    manager = ContextSummaryManager(
        store=store,
        llm_client=StaticAgentClient(
            [
                json.dumps(
                    {
                        "summary": "信息不足。",
                        "facts": {},
                        "open_questions": [],
                        "confidence": 0.2,
                    },
                    ensure_ascii=False,
                )
            ]
        ),
        trace_recorder=trace,
        policy=ContextSummaryPolicy(
            min_turns_before_summary=2,
            min_turns_since_last_summary=1,
            max_recent_tokens_before_summary=1,
            min_confidence=0.6,
        ),
    )

    result = manager.maybe_summarize_after_turn(conversation_id="summary_confidence", trace_id="trace_summary_confidence")

    assert result.summarized is False
    assert result.reason == "confidence below threshold"
    assert store.get_conversation_checkpoint("summary_confidence") is None
    steps = trace_steps(trace.get_trace("trace_summary_confidence"))
    assert "context_summary_rejected" in steps


def test_context_summary_payload_uses_public_names_and_sanitized_draft_metadata() -> None:
    store = InMemoryAgentStore()
    store.upsert_customer(
        CustomerProfile(
            customer_id="liu",
            display_name="刘峻甫-21M-高分子-宜宾",
            public_name="刘峻甫",
            private_remark="老板备注：测试白名单",
            notes="内部备注：好哥们儿",
        )
    )
    game, _ = store.create_game(
        conversation_id="summary_public_boundary",
        organizer_id="liu",
        organizer_name="刘峻甫-21M-高分子-宜宾",
        requirement={"game_type": "hangzhou_mahjong", "stake": "0.5", "needed_seats": 3},
        known_players=[{"customer_id": "liu", "display_name": "刘峻甫-21M-高分子-宜宾"}],
        trace_id="trace_summary_public_boundary_seed",
    )
    store.create_invite_drafts(
        game_id=game.game_id,
        invitations=[
            {
                "customer_id": "liu",
                "display_name": "刘峻甫-21M-高分子-宜宾",
                "message_text": "七点三缺一，打吗？",
                "metadata": {
                    "channel": "wechaty",
                    "platform_message_id": "wechat_msg_private",
                    "private_note": "老板备注：只给自己看",
                },
            }
        ],
        trace_id="trace_summary_public_boundary_invite",
    )
    store.create_outbound_message_drafts(
        conversation_id="summary_public_boundary",
        drafts=[
            {
                "recipient_id": "liu",
                "recipient_name": "刘峻甫-21M-高分子-宜宾",
                "channel": "wechaty",
                "message_text": "七点三缺一，打吗？",
                "purpose": "offer_existing_game",
                "metadata": {
                    "source": "wechaty",
                    "platform_message_id": "wechat_msg_private",
                    "private_reason": "响应率高，老板备注测试",
                },
            }
        ],
        trace_id="trace_summary_public_boundary_outbound",
    )
    manager = ContextSummaryManager(
        store=store,
        llm_client=StaticAgentClient([]),
        trace_recorder=InMemoryTraceRecorder(),
    )

    payload = manager._build_summary_payload("summary_public_boundary")
    exposed = json.dumps(payload, ensure_ascii=False)

    assert "刘峻甫" in exposed
    assert "高分子" not in exposed
    assert "宜宾" not in exposed
    assert "老板备注" not in exposed
    assert "好哥们儿" not in exposed
    assert "wechat_msg_private" not in exposed
    assert "private_note" not in exposed
    assert payload["active_games"][0]["organizer_name"] == "刘峻甫"
    assert payload["active_games"][0]["participants"][0]["display_name"] == "刘峻甫"
    assert payload["invite_drafts"][0]["display_name"] == "刘峻甫"
    assert payload["invite_drafts"][0]["metadata"] == {"channel": "wechaty"}
    assert payload["outbound_message_drafts"][0]["recipient_name"] == "刘峻甫"
    assert payload["outbound_message_drafts"][0]["metadata"] == {"source": "wechaty"}


def agent_action(*, objective_status: str, reasoning_summary: str, reply_to_user: str) -> str:
    return json.dumps(
        {
            "goal": "测试摘要系统",
            "objective_status": objective_status,
            "reasoning_summary": reasoning_summary,
            "reply_to_user": reply_to_user,
            "tool_calls": [],
            "needs_human": objective_status == "needs_human",
            "stop_reason": {
                "can_stop": True,
                "why": "本轮已经可以回复用户。",
                "pending_work": [],
                "depends_on_tool_results": False,
            },
            "badcase": None,
        },
        ensure_ascii=False,
    )
