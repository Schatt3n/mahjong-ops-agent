from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mahjong_agent_v3 import (
    AgentRuntimeV3,
    CustomerProfileV3,
    InMemoryAgentStoreV3,
    InMemoryTraceRecorderV3,
    JsonlTraceRecorderV3,
    SQLiteAgentStoreV3,
    StaticAgentClientV3,
    UserMessageV3,
)
from mahjong_agent_v3.tracing import trace_steps, validate_trace_v3


ROOT = Path(__file__).resolve().parents[1]


def test_v3_main_chain_does_not_import_legacy_parser_workflow_or_guard() -> None:
    forbidden = [
        "mahjong_agent_v2",
        "from mahjong_agent.",
        "import mahjong_agent.",
        "reply_guard",
        "workflow",
        "semantic_resolver",
        "trial_",
        "responder",
    ]
    for path in (ROOT / "src" / "mahjong_agent_v3").glob("**/*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path} contains forbidden legacy token {token!r}"


def test_v3_runtime_lets_model_drive_tool_sequence_until_final_reply() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorderV3()
    client = PlanningClient(store)
    runtime = AgentRuntimeV3(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessageV3(
            conversation_id="v3_test",
            sender_id="zhang",
            sender_name="张哥",
            text="通宵1块有人吗？没有就帮我组一个",
            message_id="msg_v3_drive_001",
        ),
        trace_id="trace_v3_drive_001",
    )

    assert result.final_reply == "好的，我帮你问问，有消息跟你说。"
    assert [call.name for action in result.actions for call in action.tool_calls] == [
        "search_current_games",
        "create_game",
        "search_customers",
        "create_invite_drafts",
    ]
    assert len(store.games) == 1
    assert len(store.invite_drafts) == 2
    assert [draft.message_text for draft in store.invite_drafts.values()] == [
        "冉姐，1块通宵，打吗？",
        "何哥，1块通宵，打吗？",
    ]
    events = trace.get_trace("trace_v3_drive_001")
    assert validate_trace_v3(events)["complete"] is True
    steps = trace_steps(events)
    assert steps.count("llm_prompt") == 5
    assert steps.count("tool_called") == 4
    assert "state_transition" in steps
    prompts = [event.content for event in events if event.step == "llm_prompt"]
    last_payload = json.loads(prompts[-1]["messages"][1]["content"])
    assert last_payload["previous_tool_results"][0]["name"] == "create_invite_drafts"


def test_v3_backend_does_not_interpret_short_confirmation_as_create_game() -> None:
    store = seeded_store()
    client = StaticAgentClientV3(
        [
            action_json(
                objective_status="completed",
                reasoning_summary="模型选择只回复，不调用工具。",
                reply_to_user="好的。",
            )
        ]
    )
    runtime = AgentRuntimeV3(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorderV3())

    result = runtime.handle_user_message(
        UserMessageV3(
            conversation_id="v3_no_backend_semantic",
            sender_id="zhang",
            sender_name="张哥",
            text="组",
            message_id="msg_v3_no_backend_semantic",
        ),
        trace_id="trace_v3_no_backend_semantic",
    )

    assert result.final_reply == "好的。"
    assert result.tool_results == []
    assert store.games == {}
    assert len(client.calls) == 1


def test_v3_tool_schema_error_is_fed_back_to_model_not_repaired_by_backend() -> None:
    store = seeded_store()
    client = StaticAgentClientV3(
        [
            action_json(
                objective_status="needs_tool",
                tool_calls=[
                    {
                        "name": "create_invite_drafts",
                        "arguments": {"invitations": []},
                        "reason": "故意缺 game_id，验证工具错误回喂模型。",
                    }
                ],
            ),
            action_json(
                objective_status="waiting_user",
                reasoning_summary="上一步工具返回 schema 错误，模型决定等待人工补充。",
                reply_to_user="我先确认一下。",
            ),
        ]
    )
    runtime = AgentRuntimeV3(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorderV3())

    result = runtime.handle_user_message(
        UserMessageV3(
            conversation_id="v3_schema_error",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我问问",
            message_id="msg_v3_schema_error",
        ),
        trace_id="trace_v3_schema_error",
    )

    assert result.tool_results[0].error == "missing required argument: game_id"
    second_prompt = json.loads(client.calls[1]["messages"][1]["content"])
    assert second_prompt["previous_tool_results"][0]["error"] == "missing required argument: game_id"
    assert store.games == {}
    assert result.final_reply == "我先确认一下。"


def test_v3_jsonl_trace_is_structured_and_replayable(tmp_path) -> None:
    store = seeded_store()
    trace = JsonlTraceRecorderV3(tmp_path / "agent_v3_trace.log")
    client = PlanningClient(store)
    runtime = AgentRuntimeV3(llm_client=client, store=store, trace_recorder=trace)

    runtime.handle_user_message(
        UserMessageV3(
            conversation_id="v3_jsonl_trace",
            sender_id="zhang",
            sender_name="张哥",
            text="通宵1块有人吗？没有就帮我组一个",
            message_id="msg_v3_jsonl_trace",
        ),
        trace_id="trace_v3_jsonl_trace",
    )

    events = trace.get_trace("trace_v3_jsonl_trace")
    steps = trace_steps(events)
    assert validate_trace_v3(events)["complete"] is True
    assert "raw_log_line" not in steps
    assert "llm_prompt" in steps
    assert "llm_response" in steps
    assert "tool_called" in steps
    assert "tool_result" in steps
    assert "state_transition" in steps
    prompt_payload = json.loads(next(event for event in events if event.step == "llm_prompt").content["messages"][1]["content"])
    assert prompt_payload["runtime"] == "mahjong_agent_v3"


def test_v3_sqlite_store_persists_runtime_state_and_idempotency(tmp_path) -> None:
    db_path = tmp_path / "agent_v3.sqlite3"
    store = seeded_store(SQLiteAgentStoreV3(db_path))
    trace = InMemoryTraceRecorderV3()
    client = PlanningClient(store)
    runtime = AgentRuntimeV3(llm_client=client, store=store, trace_recorder=trace)
    message = UserMessageV3(
        conversation_id="v3_sqlite",
        sender_id="zhang",
        sender_name="张哥",
        text="通宵1块有人吗？没有就帮我组一个",
        message_id="msg_v3_sqlite_persist",
    )

    result = runtime.handle_user_message(message, trace_id="trace_v3_sqlite_1")

    assert result.final_reply == "好的，我帮你问问，有消息跟你说。"
    assert len(store.games) == 1
    assert len(store.invite_drafts) == 2
    assert result.tool_results[0].idempotency_key

    reopened = SQLiteAgentStoreV3(db_path)
    assert len(reopened.customers) == 3
    assert len(reopened.games) == 1
    assert len(reopened.invite_drafts) == 2
    assert len(reopened.transitions) >= 3
    assert len(reopened.recent_turns("v3_sqlite")) >= 3
    assert reopened.idempotent_result(result.tool_results[0].idempotency_key) is not None

    cached_client = StaticAgentClientV3([])
    runtime_after_restart = AgentRuntimeV3(
        llm_client=cached_client,
        store=reopened,
        trace_recorder=InMemoryTraceRecorderV3(),
    )
    cached = runtime_after_restart.handle_user_message(message, trace_id="trace_v3_sqlite_2")

    assert cached.final_reply == result.final_reply
    assert cached_client.calls == []
    assert len(reopened.games) == 1
    assert reopened.idempotent_message_result("msg_v3_sqlite_persist") is not None


def test_v3_sqlite_store_persists_badcases_from_tool(tmp_path) -> None:
    store = seeded_store(SQLiteAgentStoreV3(tmp_path / "agent_v3_badcase.sqlite3"))
    client = StaticAgentClientV3(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="模型主动归档 badcase。",
                tool_calls=[
                    {
                        "name": "record_badcase",
                        "arguments": {
                            "reason": "测试回复不合适",
                            "input": {"text": "组"},
                            "actual": {"reply": "留意"},
                            "expected": {"reply": "应该继续规划"},
                        },
                        "reason": "记录评测样本。",
                    }
                ],
            ),
            action_json(
                objective_status="completed",
                reasoning_summary="badcase 已记录。",
                reply_to_user="我记下来了。",
            ),
        ]
    )
    runtime = AgentRuntimeV3(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorderV3())

    runtime.handle_user_message(
        UserMessageV3(
            conversation_id="v3_badcase",
            sender_id="zhang",
            sender_name="张哥",
            text="组",
            message_id="msg_v3_badcase",
        ),
        trace_id="trace_v3_badcase",
    )

    reopened = SQLiteAgentStoreV3(tmp_path / "agent_v3_badcase.sqlite3")
    assert len(reopened.badcases) == 1
    assert reopened.badcases[0]["reason"] == "测试回复不合适"


def seeded_store(store=None):
    store = store or InMemoryAgentStoreV3()
    store.upsert_customer(
        CustomerProfileV3(
            customer_id="zhang",
            display_name="张哥",
            gender="男",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["0.5", "1"],
            smoke_preference="any",
            response_score=0.9,
        )
    )
    store.upsert_customer(
        CustomerProfileV3(
            customer_id="ran",
            display_name="冉姐",
            gender="女",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["1"],
            smoke_preference="any",
            response_score=0.9,
        )
    )
    store.upsert_customer(
        CustomerProfileV3(
            customer_id="he",
            display_name="何哥",
            gender="男",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["1"],
            smoke_preference="any",
            response_score=0.8,
        )
    )
    return store


class PlanningClient:
    def __init__(self, store: InMemoryAgentStoreV3) -> None:
        self.store = store
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: list[dict[str, str]], *, trace_id: str, timeout_seconds: float) -> str:
        self.calls.append({"messages": messages, "trace_id": trace_id, "timeout_seconds": timeout_seconds})
        call_index = len(self.calls)
        if call_index == 1:
            return action_json(
                objective_status="needs_tool",
                reasoning_summary="需要先查询当前局池。",
                tool_calls=[
                    {
                        "name": "search_current_games",
                        "arguments": {"requirement": {"game_type": "hangzhou_mahjong", "stake": "1", "duration_kind": "overnight"}, "limit": 5},
                        "reason": "回答现有局前先查状态。",
                    }
                ],
            )
        if call_index == 2:
            return action_json(
                objective_status="needs_tool",
                reasoning_summary="没有现成局，用户也要求帮忙组。",
                tool_calls=[
                    {
                        "name": "create_game",
                        "arguments": {
                            "requirement": {
                                "game_type": "hangzhou_mahjong",
                                "stake": "1",
                                "duration_kind": "overnight",
                                "user_visible_summary": "杭麻 1块 通宵",
                            },
                            "known_players": [{"customer_id": "zhang", "display_name": "张哥"}],
                        },
                        "reason": "创建待组局记录。",
                    }
                ],
            )
        if call_index == 3:
            return action_json(
                objective_status="needs_tool",
                reasoning_summary="已有待组局，继续找候选人。",
                tool_calls=[
                    {
                        "name": "search_customers",
                        "arguments": {
                            "requirement": {"game_type": "hangzhou_mahjong", "stake": "1", "duration_kind": "overnight"},
                            "exclude_customer_ids": ["zhang"],
                            "limit": 2,
                        },
                        "reason": "搜索匹配候选人。",
                    }
                ],
            )
        if call_index == 4:
            game_id = next(iter(self.store.games.values())).game_id
            return action_json(
                objective_status="needs_tool",
                reasoning_summary="候选人已返回，生成待审批邀约草稿。",
                tool_calls=[
                    {
                        "name": "create_invite_drafts",
                        "arguments": {
                            "game_id": game_id,
                            "invitations": [
                                {"customer_id": "ran", "display_name": "冉姐", "message_text": "冉姐，1块通宵，打吗？"},
                                {"customer_id": "he", "display_name": "何哥", "message_text": "何哥，1块通宵，打吗？"},
                            ],
                        },
                        "reason": "只创建待审批草稿，不直接发送。",
                    }
                ],
            )
        return action_json(
            objective_status="completed",
            reasoning_summary="待审批草稿已创建，向发起人自然确认。",
            reply_to_user="好的，我帮你问问，有消息跟你说。",
        )


def action_json(
    *,
    objective_status: str,
    reasoning_summary: str = "test",
    reply_to_user: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    needs_human: bool = False,
) -> str:
    return json.dumps(
        {
            "goal": "测试 V3 agent 主链路",
            "objective_status": objective_status,
            "reasoning_summary": reasoning_summary,
            "reply_to_user": reply_to_user,
            "tool_calls": tool_calls or [],
            "needs_human": needs_human,
            "badcase": None,
        },
        ensure_ascii=False,
    )
