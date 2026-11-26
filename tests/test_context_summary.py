from __future__ import annotations

from datetime import datetime, timedelta
import json

import pytest

from mahjong_agent_runtime import (
    AgentRuntime,
    ContextSummaryManager,
    ContextSummaryPolicy,
    CustomerProfile,
    InMemoryAgentStore,
    InMemoryTraceRecorder,
    SQLiteAgentStore,
    StaticAgentClient,
    TokenBudget,
    ToolGateway,
    ToolResult,
    UserMessage,
)
from mahjong_agent_runtime.domains.context_builders import AgentContextBuilder, BuiltContext
from mahjong_agent_runtime.domains.context_builders.builder import (
    main_agent_system_message,
    main_agent_user_message,
    read_main_agent_system_contract,
)
from mahjong_agent_runtime.models import ConversationRole, ConversationTurn, now, using_clock
from mahjong_agent_runtime.services.context_service import ContextLifecycleManager
from mahjong_agent_runtime.task_context import TaskContextManager
from mahjong_agent_runtime.tracing import trace_steps
from mahjong_agent_runtime.token_estimation import estimate_tokens


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
    store.upsert_customer(CustomerProfile(customer_id="candidate", display_name="候选人内部标记", public_name="阿友"))
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
            },
            {"customer_id": "candidate", "display_name": "候选人内部标记", "message_text": "这桌还缺一位，你来吗？"},
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
    store.append_user_turn(UserMessage(
        conversation_id="summary_public_boundary", sender_id="liu", sender_name="刘峻甫",
        text="这桌帮我继续找人", message_id="summary_owner_message",
    ), "trace_summary_owner_message")
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
    assert "候选人内部标记" not in exposed
    assert payload["active_games"][0]["organizer_name"] == "刘峻甫"
    assert payload["active_games"][0]["participants"][0]["display_name"] == "刘峻甫"
    assert payload["invite_drafts"][0]["display_name"] == "刘峻甫"
    assert payload["invite_drafts"][0]["metadata"] == {"channel": "wechaty"}
    assert [(item["customer_id"], item["display_name"]) for item in payload["invite_drafts"]] == [
        ("liu", "刘峻甫"), ("candidate", "阿友"),
    ]
    assert payload["outbound_message_drafts"][0]["recipient_name"] == "刘峻甫"
    assert payload["outbound_message_drafts"][0]["metadata"] == {"source": "wechaty"}


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_summary_provider_does_not_receive_other_game_identities_or_drafts(tmp_path, backend):
    """摘要模型在真实调用参数里只读外局桌况，旧原始工具审计不改写。"""
    store = InMemoryAgentStore() if backend == "memory" else SQLiteAgentStore(tmp_path / "summary_scope.sqlite3")
    message = UserMessage(conversation_id="summary-current", sender_id="viewer", sender_name="本人",
                          text="我想看看现在还有什么牌局", message_id="current-input")
    task = TaskContextManager(store).prepare(message, trace_id="current-task").context
    store.append_user_turn(message, "current-task")
    visible_game, _ = store.create_game(
        conversation_id=message.conversation_id, organizer_id="other-host-key", organizer_name="另一桌主私下名字",
        requirement={"game_type": "sichuan_mahjong", "stake": "2", "duration_hours": 3,
                     "user_visible_summary": "不准从自填摘要泄露的备注"},
        known_players=[{"customer_id": "other-member-key", "display_name": "另一桌牌友私下名字"}],
        trace_id="seed-visible-game",
    )
    history_game, _ = store.create_game(
        conversation_id="another-private-thread", organizer_id="history-host-key", organizer_name="历史牌局私下名字",
        requirement={"game_type": "red_center_mahjong", "stake": "1"}, known_players=[], trace_id="seed-foreign-history",
    )
    drafts, _ = store.create_invite_drafts(game_id=visible_game.game_id, trace_id="seed-invites", invitations=[
        {"customer_id": "other-invite-key", "display_name": "第三人姓名", "message_text": "第三人待发邀约的原文"},
        {"customer_id": message.sender_id, "display_name": message.sender_name, "message_text": "本人收到的邀约"},
    ])
    store.update_invite_delivery_status(draft_id=drafts[1].draft_id, status="sent", trace_id="own-receipt",
                                        reason="test confirmed delivery")
    store.create_outbound_message_drafts(conversation_id=message.conversation_id, trace_id="seed-outbound", drafts=[
        {"recipient_id": "other-outbound-key", "recipient_name": "另一收件人", "message_text": "第三人外发正文",
         "channel": "wechaty", "purpose": "offer_existing_game", "metadata": {"game_id": visible_game.game_id}},
        {"recipient_id": message.sender_id, "recipient_name": message.sender_name, "message_text": "给本人的草稿",
         "channel": "wechaty", "purpose": "offer_existing_game"},
    ])
    # 兼容防线要覆盖已经落库的工具收据，不能只依赖新的查询结果从源头最小化。
    raw_tool_history = json.dumps([{"name": "search_current_games", "called": True, "allowed": True,
                                    "result": {"matches": [{"game": history_game.to_dict()}]}}], ensure_ascii=False)
    store.append_tool_turn(message.conversation_id, raw_tool_history, "historical-search")
    client = StaticAgentClient([json.dumps({"summary": "已查询现成局", "facts": {}, "open_questions": [],
                                            "confidence": 0.9}, ensure_ascii=False)])
    summary = ContextSummaryManager(store=store, llm_client=client)

    result = summary.summarize_for_quality_evaluation(conversation_id=message.conversation_id, trace_id="summary-call",
                                                       task_context_id=task.task_context_id)

    assert result.summarized
    request = client.calls[0]["messages"][-1]["content"]
    payload = json.loads(request)
    assert [(row["game_id"], row["seat_summary"]["remaining_seats"]) for row in payload["active_games"]] == [
        (visible_game.game_id, 2),
    ]
    assert [row["message_text"] for row in payload["invite_drafts"]] == ["本人收到的邀约"]
    assert [row["message_text"] for row in payload["outbound_message_drafts"]] == ["给本人的草稿"]
    old_result, = [row for row in payload["recent_conversation"] if row["role"] == "tool"]
    matched = json.loads(old_result["content"])[0]["result"]["matches"][0]["game"]
    assert matched["game_id"] == history_game.game_id and matched["requirement"]["game_type"] == "red_center_mahjong"
    for forbidden in ("other-host-key", "另一桌主私下名字", "other-member-key", "另一桌牌友私下名字",
                      "不准从自填摘要泄露的备注", "another-private-thread", "history-host-key", "历史牌局私下名字",
                      "other-invite-key", "第三人待发邀约的原文", "other-outbound-key", "第三人外发正文"):
        assert forbidden not in request
    assert [turn.content for turn in store.recent_turns(message.conversation_id) if turn.role == ConversationRole.TOOL] == [
        raw_tool_history,
    ]


def test_private_game_context_only_marks_delivered_invites_for_the_sender() -> None:
    """只给私聊里的本人提供已送达、未处理邀约的桌号证据。"""
    from mahjong_agent_runtime.domains.context_builders import build_game_context

    store = InMemoryAgentStore()
    games = {}
    for label, recipient, sent in (("own", "guest", True), ("unsent", "guest", False), ("another", "other", True)):
        game, _ = store.create_game(conversation_id=f"host-{label}", organizer_id=f"host-{label}",
            organizer_name="牌友", requirement={"game_type": "hangzhou_mahjong", "stake": "1"},
            known_players=[], trace_id=f"setup-{label}")
        games[label] = game.game_id
        drafts, _ = store.create_invite_drafts(game_id=game.game_id, trace_id=f"invite-{label}", invitations=[
            {"customer_id": recipient, "display_name": recipient, "message_text": f"邀请-{label}"},
        ])
        if sent:
            store.update_invite_delivery_status(draft_id=drafts[0].draft_id, status="sent",
                                               trace_id=f"delivery-{label}", reason="test delivered")

    def offered_to_guest(allowed):
        context = build_game_context(store, conversation_id="guest-private", sender_id="guest",
                                     allow_private_public_roster=allowed)
        return context.current_sender_invites

    assert offered_to_guest(True) == [{"game_id": games["own"], "text": "邀请-own"}]
    assert not offered_to_guest(False)
    store.record_candidate_reply(game_id=games["own"], customer_id="guest", display_name="客人",
                                 status="declined", trace_id="guest-refused")
    assert not offered_to_guest(True)


def test_context_summary_is_scoped_to_explicit_task_context() -> None:
    store = InMemoryAgentStore()
    conversation_id = "summary_task_scope"
    manager = TaskContextManager(store)
    future_message = UserMessage(
        conversation_id=conversation_id,
        sender_id="A",
        sender_name="A",
        text="明天晚上七点0.5无烟，帮我约一桌",
        message_id="summary_future_source",
    )
    future_task = manager.prepare(future_message, trace_id="trace_summary_future")
    store.append_user_turn(future_message, "trace_summary_future")
    store.append_assistant_turn(conversation_id, "好，明天开局前我再问人。", "trace_summary_future")

    current_task, _ = store.activate_task_context(
        conversation_id=conversation_id,
        customer_id="A",
        trace_id="trace_summary_current",
        activity_at=now(),
        started_at=now(),
        reason="test_new_task",
        force_new=True,
        archive_previous=False,
    )
    current_message = UserMessage(
        conversation_id=conversation_id,
        sender_id="A",
        sender_name="A",
        text="今天下午再帮我组一场1块的",
        message_id="summary_current_source",
    )
    store.append_user_turn(current_message, "trace_summary_current")
    store.append_assistant_turn(conversation_id, "好，我先看今天的。", "trace_summary_current")

    summary_client = StaticAgentClient(
        [
            json.dumps(
                {
                    "summary": "明天晚上七点0.5无烟预约，尚未开始邀约。",
                    "facts": {"stake": "0.5", "planned_start": "明天19:00"},
                    "open_questions": [],
                    "confidence": 0.95,
                },
                ensure_ascii=False,
            )
        ]
    )
    summary_manager = ContextSummaryManager(
        store=store,
        llm_client=summary_client,
        trace_recorder=InMemoryTraceRecorder(),
    )

    result = summary_manager.summarize_for_quality_evaluation(
        conversation_id=conversation_id,
        trace_id="trace_summary_explicit_task",
        task_context_id=future_task.context.task_context_id,
    )

    prompt_payload = json.loads(summary_client.calls[0]["messages"][1]["content"])
    prompt_text = json.dumps(prompt_payload, ensure_ascii=False)
    assert result.summarized is True
    assert prompt_payload["task_context_id"] == future_task.context.task_context_id
    assert "明天晚上七点0.5无烟" in prompt_text
    assert "今天下午再帮我组一场1块的" not in prompt_text
    assert store.get_task_context_checkpoint(future_task.context.task_context_id) is not None
    assert store.get_task_context_checkpoint(current_task.task_context_id) is None


def test_scheduled_system_event_summarizes_original_future_task_not_latest_task() -> None:
    store = InMemoryAgentStore()
    conversation_id = "summary_scheduled_scope"
    manager = TaskContextManager(store)
    appointment = UserMessage(
        conversation_id=conversation_id,
        sender_id="A",
        sender_name="A",
        text="明天晚上七点0.5无烟，帮我约一桌",
        message_id="summary_scheduled_source",
    )
    original = manager.prepare(appointment, trace_id="trace_scheduled_summary_source")
    store.append_user_turn(appointment, "trace_scheduled_summary_source")
    latest, _ = store.activate_task_context(
        conversation_id=conversation_id,
        customer_id="A",
        trace_id="trace_scheduled_summary_latest",
        activity_at=now(),
        started_at=now(),
        reason="test_newer_task",
        force_new=True,
        archive_previous=False,
    )
    runtime = AgentRuntime(
        llm_client=StaticAgentClient(
            [
                agent_action(
                    objective_status="completed",
                    reasoning_summary="预约招募任务本轮完成。",
                    reply_to_user="已检查预约任务。",
                )
            ]
        ),
        store=store,
        trace_recorder=InMemoryTraceRecorder(),
        context_summary_manager=ContextSummaryManager(
            store=store,
            llm_client=StaticAgentClient(
                [
                    json.dumps(
                        {
                            "summary": "明天晚上七点0.5无烟预约进入招募窗口。",
                            "facts": {"stake": "0.5", "planned_start": "明天19:00"},
                            "open_questions": [],
                            "confidence": 0.95,
                        },
                        ensure_ascii=False,
                    )
                ]
            ),
            policy=ContextSummaryPolicy(
                min_turns_before_summary=2,
                min_turns_since_last_summary=1,
                max_recent_tokens_before_summary=1,
            ),
        ),
    )

    runtime.handle_system_event(
        UserMessage(
            conversation_id=conversation_id,
            sender_id="A",
            sender_name="A",
            text="预约局进入招募窗口",
            message_id="summary_scheduled_wakeup",
            metadata={"task_context_id": original.context.task_context_id},
        ),
        trace_id="trace_scheduled_summary_wakeup",
    )

    original_checkpoint = store.get_task_context_checkpoint(original.context.task_context_id)
    assert original_checkpoint is not None
    assert original_checkpoint.summary == "明天晚上七点0.5无烟预约进入招募窗口。"
    assert store.get_task_context_checkpoint(latest.task_context_id) is None


def test_runtime_summarizes_before_llm_when_context_nears_budget() -> None:
    store = InMemoryAgentStore()
    trace = InMemoryTraceRecorder()
    long_text = "张哥之前一直在补充组局条件，倾向杭麻，0.5或1块，人齐开，烟都可。" * 45
    for index in range(10):
        store.append_user_turn(
            UserMessage(
                conversation_id="summary_before_budget",
                sender_id="zhang",
                sender_name="张哥",
                text=f"{index}-{long_text}",
                message_id=f"summary_before_budget_seed_user_{index}",
            ),
            f"trace_seed_user_{index}",
        )
        store.append_assistant_turn(
            "summary_before_budget",
            f"我先记一下 {index}。" + long_text,
            f"trace_seed_assistant_{index}",
        )

    main_client = StaticAgentClient(
        [
            agent_action(
                objective_status="completed",
                reasoning_summary="预算前置摘要后，主模型正常回复。",
                reply_to_user="好，我按刚才的信息继续看。",
            )
        ]
    )
    summary_client = StaticAgentClient(
        [
            json.dumps(
                {
                    "summary": "张哥多轮表达想组杭麻，0.5或1块，人齐开，烟都可。",
                    "facts": {
                        "intent": "find_players",
                        "game_type": "hangzhou_mahjong",
                        "stake_options": ["0.5", "1"],
                        "start_time_kind": "asap_when_full",
                        "smoke_preference": "any",
                    },
                    "open_questions": ["还要确认当前人数"],
                    "confidence": 0.93,
                },
                ensure_ascii=False,
            )
        ]
    )
    current_message = UserMessage(
        conversation_id="summary_before_budget", sender_id="zhang", sender_name="张哥",
        text="那现在继续帮我看下。", message_id="summary_before_budget_current",
    )
    # 以当前生产合同的空历史大小确定预算：静态合同本身不触发摘要，加入长历史后
    # 才跨过 85% 阈值。这样测试的是压缩效果，而不是某版提示词恰好小于 32k。
    empty_store = InMemoryAgentStore()
    empty_context = AgentContextBuilder(empty_store, ToolGateway(empty_store)).build(current_message, trace_id="empty-budget")
    baseline_tokens = sum(estimate_tokens(item["content"]) for item in empty_context.messages)
    test_budget = int((baseline_tokens + 2000) / 0.85)
    assert baseline_tokens < int(test_budget * 0.85)
    runtime = AgentRuntime(
        llm_client=main_client,
        store=store,
        trace_recorder=trace,
        token_budget=TokenBudget(max_tokens_per_call=test_budget, max_calls_per_turn=4),
        context_summary_manager=ContextSummaryManager(
            store=store,
            llm_client=summary_client,
            trace_recorder=trace,
            policy=ContextSummaryPolicy(
                min_turns_before_summary=99,
                min_turns_since_last_summary=99,
                max_recent_tokens_before_summary=999_999,
                max_summary_input_tokens=20_000,
            ),
        ),
    )

    result = runtime.handle_user_message(
        current_message,
        trace_id="trace_summary_before_budget",
    )

    prompt_payload = json.loads(main_client.calls[0]["messages"][1]["content"])
    events = trace.get_trace("trace_summary_before_budget")
    steps = trace_steps(events)
    rebuilt = next(item for item in events if item.step == "context_rebuilt_after_summary")

    assert result.final_reply == "好，我按刚才的信息继续看。"
    assert len(summary_client.calls) == 1
    assert "context_summary_budget_triggered" in steps
    assert "context_rebuilt_after_summary" in steps
    assert rebuilt.content["previous_estimated_tokens"] >= int(test_budget * 0.85)
    assert rebuilt.content["rebuilt_estimated_tokens"] < test_budget
    assert prompt_payload["conversation_checkpoint"]["summary"] == "张哥多轮表达想组杭麻，0.5或1块，人齐开，烟都可。"
    assert prompt_payload["context_budget"]["checkpoint_covered_turn_count"] >= 20
    assert prompt_payload["context_budget"]["included_turn_count"] == 0
    assert any(item.entity_type == "conversation_checkpoint" for item in result.state_transitions)


def test_budget_precheck_reuses_exact_tool_contracts_before_calling_summary() -> None:
    """静态工具合同使两步越过软阈值时，发送副本可压缩而不用重复总结顾客原话。"""
    store, trace, summary = InMemoryAgentStore(), InMemoryTraceRecorder(), StaticAgentClient([])
    manager = ContextLifecycleManager(AgentContextBuilder(store, ToolGateway(store)), trace,
        context_summary_manager=ContextSummaryManager(store=store, llm_client=summary, trace_recorder=trace))
    text = '我19点11才能到，最晚打到22点。'
    tool_schema = {"type": "object", "properties": {"condition": {
        "type": "string", "description": "保留顾客现有条件、原局和已回执的时间区间。" * 35,
    }}}
    tools = [{"name": f"lookup_{index}", "schema": tool_schema} for index in range(5)]
    payload = {"current_message": {"text": text}, "available_tools": tools, "previous_tool_results": [],
               "output_contract": {}, "planning_contract": {},
               "turn_tool_evidence": [], "recent_conversation": [{"role": "user", "content": text}]}
    system = main_agent_system_message("按真实证据处理。", payload)
    user = main_agent_user_message(payload)
    baseline = estimate_tokens(system["content"]) + estimate_tokens(main_agent_user_message(payload, dense=True)["content"])
    budget = TokenBudget(max_tokens_per_call=int((baseline - 25) / 0.85))
    trigger = int(budget.max_tokens_per_call * 0.85)
    assert trigger < baseline <= sum(estimate_tokens(item["content"]) for item in (system, user)) < budget.max_tokens_per_call
    message = UserMessage(conversation_id="compact-budget", sender_id="guest", sender_name="顾客", text=text)

    for step in (2, 3):
        context = BuiltContext([system, user], payload, {})
        packed, transition = manager.summarize_and_rebuild_context_if_needed(
            message, built=context, trace_id="static-budget", pending_tool_results=[], turn_tool_evidence=[],
            run_id="run-static-budget", run_version=0, step_index=step, budget=budget)
        assert transition is None and sum(estimate_tokens(item["content"]) for item in packed.messages) < trigger
        request = json.loads(packed.messages[-1]["content"])
        packed_tools = read_main_agent_system_contract(packed.messages[0])["available_tools"]
        assert request["current_message"]["text"] == request["recent_conversation"][0]["content"] == text
        assert [item["name"] for item in packed_tools] == [item["name"] for item in tools]
        assert "available_tools" not in request
        for value in packed_tools:
            assert request["shared_tool_schemas"][value["schema"]["$ref"].rsplit("/", 1)[-1]] == tool_schema

    assert "context_summary_budget_triggered" not in trace_steps(trace.get_trace("static-budget"))
    assert not summary.calls and read_main_agent_system_contract(system)["available_tools"] == tools


def test_budget_summary_skips_futile_preemptive_reclamation_but_still_handles_hard_limit() -> None:
    """静态合同跨软阈值时不浪费摘要调用，超过硬上限时仍进入摘要保护。"""
    store, trace, summary = InMemoryAgentStore(), InMemoryTraceRecorder(), StaticAgentClient([])
    manager = ContextLifecycleManager(AgentContextBuilder(store, ToolGateway(store)), trace,
        context_summary_manager=ContextSummaryManager(store=store, llm_client=summary, trace_recorder=trace))
    text = "我等位，能赶上就来。"
    payload = {"current_message": {"text": text}, "recent_conversation": [{"role": "user", "content": text}]}
    messages = [{"role": "system", "content": "请依据实际工具来源和顾客的授权。" * 220},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)}]
    baseline = sum(estimate_tokens(value["content"]) for value in messages)
    message = UserMessage(conversation_id="static-pressure", sender_id="guest", sender_name="顾客", text=text)
    for name, limit in (("soft", baseline + 100), ("hard", baseline - 100)):
        manager.summarize_and_rebuild_context_if_needed(message, built=BuiltContext(messages, payload, {}),
            trace_id=name, pending_tool_results=[], turn_tool_evidence=[], run_id=name, run_version=0, step_index=3,
            budget=TokenBudget(max_tokens_per_call=limit))

    assert "context_summary_budget_not_reclaimable" in trace_steps(trace.get_trace("soft"))
    assert "context_summary_budget_triggered" not in trace_steps(trace.get_trace("soft"))
    assert "context_summary_budget_triggered" in trace_steps(trace.get_trace("hard"))


def test_budget_summary_keeps_bounded_exact_customer_sources_from_the_current_task() -> None:
    """摘要把到店误作开场时，模型的真实压缩请求中仍须收到本人同任务时间原话。"""
    store = InMemoryAgentStore()
    trace = InMemoryTraceRecorder()
    sent_at = datetime.fromisoformat("2026-11-26T09:07:00+08:00")
    conversation_id = "budget-summary-customer-sources"
    customer_id = "customer"
    current = UserMessage(conversation_id=conversation_id, sender_id=customer_id, sender_name="顾客",
        text="行 帮我凑一桌 红中2块3块的 我九点半前能到 能打到晚上",
        message_id="current-arrival", sent_at=sent_at)
    earlier = "老板今天有合适的桌吗 红中2块3块的 我时长能调 上午下午都行"
    stale_at = sent_at - timedelta(minutes=15)
    task_at = sent_at - timedelta(minutes=3)
    with using_clock(lambda: stale_at):
        previous = UserMessage(conversation_id=conversation_id, sender_id=customer_id, sender_name="顾客",
            text="旧任务只打晚上", message_id="old-task", sent_at=stale_at)
        TaskContextManager(store).prepare(previous, trace_id="old-task")
        store.append_user_turn(previous, "old-task")
    with using_clock(lambda: task_at):
        active, _ = store.activate_task_context(conversation_id=conversation_id, customer_id=customer_id,
            trace_id="active-task", activity_at=task_at, started_at=task_at, reason="new_goal",
            force_new=True, archive_previous=True)
        inputs = [
            ("scope-excess", "先看下平时的桌"),
            ("scope-first", "我这次不带人"),
            ("scope-second", "就留在这个店"),
            ("scope-availability", earlier),
        ]
        for offset, (source_id, content) in enumerate(inputs):
            store.append_user_turn(UserMessage(conversation_id=conversation_id, sender_id=customer_id, sender_name="顾客",
                text=content, message_id=source_id, sent_at=task_at + timedelta(seconds=offset)), source_id)
        for offset, (source_id, sender, metadata) in enumerate((
            ("another-customer", "other", {}),
            ("system-inbound", customer_id, {"internal_event": True}),
            ("invalidated-input", customer_id, {"context_invalidated": True}),
        ), start=5):
            store.append_turn(conversation_id, ConversationTurn(role=ConversationRole.USER, sender_id=sender,
                content=source_id, trace_id=source_id, occurred_at=task_at + timedelta(seconds=offset),
                metadata={"task_context_id": active.task_context_id, "source_message_id": source_id, **metadata}))
        for index in range(8):
            store.append_assistant_turn(conversation_id, "旧检索过程的细节。" * 150, f"long-owner-turn-{index}")

    main_client = StaticAgentClient([agent_action(objective_status="completed",
        reasoning_summary="正常返回", reply_to_user="好，我看看。")])
    flawed_summary = {"summary": "九点半是固定开场。", "facts": {"start_time": "09:30", "start_time_kind": "scheduled"},
                      "open_questions": [], "confidence": 0.9}
    summary_client = StaticAgentClient([json.dumps(flawed_summary, ensure_ascii=False)])
    empty = InMemoryAgentStore()
    baseline = AgentContextBuilder(empty, ToolGateway(empty)).build(current, trace_id="budget-baseline")
    baseline_tokens = sum(estimate_tokens(item["content"]) for item in baseline.messages)
    budget = int((baseline_tokens + 1800) / 0.85)
    runtime = AgentRuntime(llm_client=main_client, store=store, trace_recorder=trace,
        token_budget=TokenBudget(max_tokens_per_call=budget, max_calls_per_turn=4),
        context_summary_manager=ContextSummaryManager(store=store, llm_client=summary_client, trace_recorder=trace,
            policy=ContextSummaryPolicy(min_turns_before_summary=99, min_turns_since_last_summary=99,
                max_recent_tokens_before_summary=999_999, max_summary_input_tokens=20_000)))

    with using_clock(lambda: sent_at):
        response = runtime.handle_user_message(current, trace_id="compact-with-original-sources")

    assert response.final_reply == "好，我看看。"
    assert len(summary_client.calls) == len(main_client.calls) == 1
    assert "context_rebuilt_after_summary" in trace_steps(trace.get_trace("compact-with-original-sources"))
    sent_messages = main_client.calls[0]["messages"]
    assert sum(estimate_tokens(item["content"]) for item in sent_messages) < budget
    payload = json.loads(sent_messages[-1]["content"])
    assert payload["conversation_checkpoint"]["summary"] == flawed_summary["summary"]
    assert payload["current_message"]["message_id"] == current.message_id
    assert payload["context_budget"]["checkpoint_customer_sources_retained"] == 3
    assert [(turn["metadata"]["source_message_id"], turn["content"]) for turn in payload["recent_conversation"]] == inputs[-3:]


@pytest.mark.parametrize("gameplay_source", ["none", "retained_customer", "profile"])
def test_v5_0405_checkpoint_does_not_promote_guessed_gameplay(tmp_path, gameplay_source):
    """同轮模型摘要与查局筛项都不是顾客的玩法证据，本人原话和画像仍有自己的来源。"""
    at = datetime.fromisoformat("2026-11-25T19:00:00+08:00")
    store = SQLiteAgentStore(tmp_path / "checkpoint.sqlite3")
    customer = "customer-0405"
    conversation = "0405-private"
    old = UserMessage(conversation_id=conversation, sender_id=customer, sender_name="客人",
        text="我上次打的杭麻。", message_id="different-task", sent_at=at - timedelta(minutes=20))
    earlier_text = "老板晚上好 想问下今晚还有局能搭一桌吗 无烟的 档位2到3 我大概八点前能到"
    earlier = UserMessage(conversation_id=conversation, sender_id=customer, sender_name="客人",
        text=earlier_text + (" 想打红中" if gameplay_source == "retained_customer" else ""),
        message_id="earlier-customer", sent_at=at)
    current = UserMessage(conversation_id=conversation, sender_id=customer, sender_name="客人",
        text="老板我这边现在最早也要七点四十一才能到 要是组局时间能往七点五十左右挪一下的话 无烟2到3档我就能打 大概能打三个小时 你看能行吗",
        message_id="current-customer", sent_at=at)
    saved_facts = {"game_type": "hangzhou_mahjong", "requested_game": "hangzhou_mahjong",
                   "game_variant": "variant-from-summary", "smoke_preference": "no_smoke"}
    with using_clock(lambda: old.sent_at):
        TaskContextManager(store).prepare(old, trace_id="different-task")
        store.append_user_turn(old, "different-task")
    with using_clock(lambda: at):
        task, _ = store.activate_task_context(conversation_id=conversation, customer_id=customer,
            trace_id="new-task", activity_at=at, started_at=at, reason="new_goal", force_new=True, archive_previous=True)
        store.append_user_turn(earlier, "earlier-trace")
        if gameplay_source == "profile":
            store.upsert_customer(CustomerProfile(customer_id=customer, display_name="客人",
                                                 preferred_games=["sichuan_mahjong"]))
        store.upsert_conversation_checkpoint(conversation_id=conversation, task_context_id=task.task_context_id,
            summary="客人想打无烟杭州麻将。", facts=saved_facts, open_questions=[], trace_id="synthetic-summary")
        search = ToolResult(name="search_current_games", called=True, allowed=True, call_id="search_5",
            result={"requirement": {"game_type": "hangzhou_mahjong"}, "matches": [], "lookup_scope": "personal"})
        built = AgentContextBuilder(store, ToolGateway(store)).build(current, trace_id="current-trace",
                                                                     turn_tool_evidence=[search])
    payload = json.loads(built.messages[-1]["content"])
    assert payload["conversation_checkpoint"]["facts"] == {"smoke_preference": "no_smoke"}
    assert payload["conversation_checkpoint"]["summary"] == "客人想打无烟杭州麻将。"
    assert store.get_task_context_checkpoint(task.task_context_id).facts == saved_facts
    assert payload["turn_tool_evidence"][0]["result"]["requirement"]["game_type"] == "hangzhou_mahjong"
    assert [turn["metadata"]["source_message_id"] for turn in payload["recent_conversation"]] == ["earlier-customer"]
    facts = payload["explicit_task_facts"]
    assert facts["binding_fields"] == []
    if gameplay_source == "retained_customer":
        assert facts["candidates"]["game_type"] == "red_center_mahjong"
        assert facts["evidence"]["game_type"] == "recent_user_turn"
    else:
        assert "game_type" not in facts["candidates"] and "game_type" not in facts["evidence"]
    if gameplay_source == "profile":
        assert payload["sender_profile"]["preferred_games"] == ["sichuan_mahjong"]


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
