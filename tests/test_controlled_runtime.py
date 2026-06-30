from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from mahjong_agent.controlled_runtime import ControlledRuntimeConfig, build_controlled_runtime
from mahjong_agent.core import AgentCore
from mahjong_agent.customer_repository import SQLiteCustomerProfileRepository
from mahjong_agent.input_gate import SQLiteInputGate
from mahjong_agent.memory import SQLiteShortTermMemoryStore
from mahjong_agent.models import ChannelType, CustomerProfile, Message, PlayPreference
from mahjong_agent.state_machine import InMemoryWorkflowStateStore, SQLiteWorkflowStateStore
from mahjong_agent.tool_orchestrator import SQLiteToolExecutionLedger
from mahjong_agent.tools import OUTBOX_APPROVED, SQLitePendingOutboxStore
from mahjong_agent.workflow_models import ActionName, GameWorkflowStatus, ToolName


TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 6, 30, 17, 0, tzinfo=TZ)


def test_controlled_runtime_config_reads_approval_enabled_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MAHJONG_TRACE_JSONL_PATH", str(tmp_path / "trace.jsonl"))
    monkeypatch.setenv("MAHJONG_APPROVAL_ENABLED", "false")
    monkeypatch.setenv("MAHJONG_INPUT_GATE_SQLITE_PATH", str(tmp_path / "input_gate.sqlite3"))
    monkeypatch.setenv("MAHJONG_SHORT_MEMORY_SQLITE_PATH", str(tmp_path / "short_memory.sqlite3"))
    monkeypatch.setenv("MAHJONG_CUSTOMER_PROFILE_SQLITE_PATH", str(tmp_path / "customers.sqlite3"))

    config = ControlledRuntimeConfig.from_env()

    assert config.approval_enabled is False
    assert config.input_gate_sqlite_path == tmp_path / "input_gate.sqlite3"
    assert config.short_memory_sqlite_path == tmp_path / "short_memory.sqlite3"
    assert config.customer_profile_sqlite_path == tmp_path / "customers.sqlite3"


def test_controlled_runtime_fails_closed_without_llm_and_writes_jsonl_trace(tmp_path, monkeypatch) -> None:
    for key in ("MAHJONG_LLM_API_KEY", "MAHJONG_LLM_MODEL", "MAHJONG_LLM_PROVIDER", "OPENAI_API_KEY", "DASHSCOPE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    trace_path = tmp_path / "logs" / "controlled_trace.jsonl"
    runtime = build_controlled_runtime(
        config=ControlledRuntimeConfig(
            trace_jsonl_path=trace_path,
            fail_closed_without_llm=True,
        )
    )
    message = Message(
        text="老板，今天有人打吗",
        sender_id="zhang",
        sender_name="张哥",
        channel_id="boss_trial",
        channel_type=ChannelType.WEB_CONSOLE,
        sent_at=NOW,
        id="msg_runtime",
        metadata={"conversation_id": "boss_trial"},
    )

    result = runtime.service.handle_message(message, now=NOW, trace_id="trace_runtime")

    assert result.run.semantic_resolution is not None
    assert result.run.semantic_resolution.proposed_action.name == ActionName.HUMAN_REVIEW
    assert result.final_text == "这个我先转人工确认一下。"
    assert trace_path.exists()
    trace_text = trace_path.read_text(encoding="utf-8")
    assert "trace_runtime-2026-06-30 17:00:00-INFO:" in trace_text
    assert "LLM 未配置" in trace_text
    assert runtime.memory_store.load("boss_trial", "zhang", now=NOW)[0].system_reply == "这个我先转人工确认一下。"


def test_controlled_runtime_can_be_built_with_injected_llm_client(tmp_path) -> None:
    class FakeLLMClient:
        def complete(self, messages, *, trace_id, timeout_seconds):
            return {
                "intent": "inquire_existing_game",
                "proposed_action": "search_existing_games",
                "confidence": 0.88,
                "reasoning_summary": "用户只是咨询现有局。",
                "slots": {
                    "stake": {
                        "value": "0.5",
                        "source": "explicit",
                        "confidence": 0.9,
                        "confirmed": True,
                        "needs_confirmation": False,
                    }
                },
            }

    runtime = build_controlled_runtime(
        llm_client=FakeLLMClient(),
        config=ControlledRuntimeConfig(trace_jsonl_path=tmp_path / "trace.jsonl"),
    )
    message = Message(
        text="现在有0.5的吗",
        sender_id="zhang",
        sender_name="张哥",
        channel_id="boss_trial",
        channel_type=ChannelType.WEB_CONSOLE,
        sent_at=NOW,
        id="msg_runtime_fake",
        metadata={"conversation_id": "boss_trial"},
    )

    result = runtime.service.handle_message(message, now=NOW, trace_id="trace_runtime_fake")

    assert result.run.validated_action is not None
    assert result.run.validated_action.effective_action == ActionName.ASK_CREATE_CONFIRMATION
    assert result.final_text == "现在没有合适的，要组一个吗？"


def test_controlled_runtime_exposes_state_store_for_applied_transitions(tmp_path) -> None:
    class CreateGameLLMClient:
        def complete(self, messages, *, trace_id, timeout_seconds):
            return {
                "intent": "find_players",
                "proposed_action": "create_game",
                "confidence": 0.92,
                "reasoning_summary": "用户明确要组局，信息齐全。",
                "slots": {
                    "game_type": {
                        "value": "hangzhou_mahjong",
                        "source": "explicit",
                        "confidence": 0.9,
                        "confirmed": True,
                        "needs_confirmation": False,
                    },
                    "stake": {
                        "value": "0.5",
                        "source": "explicit",
                        "confidence": 0.9,
                        "confirmed": True,
                        "needs_confirmation": False,
                    },
                    "start_time_mode": {
                        "value": "people_ready",
                        "source": "explicit",
                        "confidence": 0.9,
                        "confirmed": True,
                        "needs_confirmation": False,
                    },
                    "missing_count": {
                        "value": 3,
                        "source": "explicit",
                        "confidence": 0.9,
                        "confirmed": True,
                        "needs_confirmation": False,
                    },
                    "smoke": {
                        "value": "any",
                        "source": "explicit",
                        "confidence": 0.9,
                        "confirmed": True,
                        "needs_confirmation": False,
                    },
                    "duration_mode": {
                        "value": "overnight",
                        "source": "explicit",
                        "confidence": 0.9,
                        "confirmed": True,
                        "needs_confirmation": False,
                    },
                },
            }

    state_store = InMemoryWorkflowStateStore()
    runtime = build_controlled_runtime(
        llm_client=CreateGameLLMClient(),
        state_store=state_store,
        config=ControlledRuntimeConfig(trace_jsonl_path=tmp_path / "trace_state.jsonl"),
    )
    message = Message(
        text="通宵0.5人齐开，173，烟都可",
        sender_id="zhang",
        sender_name="张哥",
        channel_id="boss_trial",
        channel_type=ChannelType.WEB_CONSOLE,
        sent_at=NOW,
        id="msg_runtime_state",
        metadata={"conversation_id": "boss_trial"},
    )

    result = runtime.service.handle_message(message, now=NOW, trace_id="trace_runtime_state")

    assert runtime.state_store is state_store
    assert result.run.state_transitions
    game_id = result.run.state_transitions[-1].entity_id
    assert runtime.state_store.current_status("game", game_id) == GameWorkflowStatus.OPEN.value
    assert runtime.state_store.transition_history(entity_type="game", entity_id=game_id)[0].metadata["store_applied"] is True
    assert runtime.tool_ledger.history(tool_name=ToolName.SEARCH_CANDIDATE_CUSTOMERS)


def test_controlled_runtime_created_game_is_visible_to_next_current_game_search(tmp_path) -> None:
    class SequencedLLMClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages, *, trace_id, timeout_seconds):
            self.calls += 1
            if self.calls == 1:
                return {
                    "intent": "find_players",
                    "proposed_action": "create_game",
                    "confidence": 0.92,
                    "reasoning_summary": "用户明确要组局，信息齐全。",
                    "slots": {
                        "game_type": {"value": "hangzhou_mahjong", "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                        "stake": {"value": "0.5", "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                        "start_time_mode": {"value": "people_ready", "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                        "missing_count": {"value": 3, "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                        "smoke": {"value": "any", "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                        "duration_mode": {"value": "overnight", "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                    },
                }
            return {
                "intent": "inquire_existing_game",
                "proposed_action": "search_existing_games",
                "confidence": 0.9,
                "reasoning_summary": "用户咨询是否有 0.5 的现有局。",
                "slots": {
                    "game_type": {"value": "hangzhou_mahjong", "source": "context", "confidence": 0.8, "confirmed": True, "needs_confirmation": False},
                    "stake": {"value": "0.5", "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                    "smoke": {"value": "any", "source": "context", "confidence": 0.75, "confirmed": True, "needs_confirmation": False},
                    "start_time_mode": {"value": "people_ready", "source": "context", "confidence": 0.75, "confirmed": True, "needs_confirmation": False},
                },
            }

    runtime = build_controlled_runtime(
        llm_client=SequencedLLMClient(),
        config=ControlledRuntimeConfig(trace_jsonl_path=tmp_path / "trace_created_game_search.jsonl"),
    )
    first_message = Message(
        text="通宵0.5人齐开，173，烟都可",
        sender_id="zhang",
        sender_name="张哥",
        channel_id="boss_trial",
        channel_type=ChannelType.WEB_CONSOLE,
        sent_at=NOW,
        id="msg_created_game_first",
        metadata={"conversation_id": "boss_trial", "source_message_id": "wechat_created_game_1", "sequence": 1},
    )
    second_message = Message(
        text="现在有0.5的人齐开局吗",
        sender_id="wang",
        sender_name="王姐",
        channel_id="boss_trial",
        channel_type=ChannelType.WEB_CONSOLE,
        sent_at=NOW,
        id="msg_created_game_second",
        metadata={"conversation_id": "boss_trial", "source_message_id": "wechat_created_game_2", "sequence": 2},
    )

    first = runtime.service.handle_message(first_message, now=NOW, trace_id="trace_created_game_first")
    second = runtime.service.handle_message(second_message, now=NOW, trace_id="trace_created_game_second")

    assert first.run.state_transitions
    assert second.context_build.context.open_games
    assert second.context_build.context.open_games[0].slot("stake").value == "0.5"
    assert second.run.validated_action is not None
    assert second.run.validated_action.effective_action == ActionName.MATCH_EXISTING_GAME
    search_result = second.tool_orchestration.result_for(ToolName.SEARCH_CURRENT_OPEN_GAMES)
    assert search_result is not None
    assert search_result.result["result_count"] == 1


def test_controlled_runtime_created_game_is_visible_from_sqlite_state_after_restart(tmp_path) -> None:
    class CreateGameLLMClient:
        def complete(self, messages, *, trace_id, timeout_seconds):
            return {
                "intent": "find_players",
                "proposed_action": "create_game",
                "confidence": 0.92,
                "reasoning_summary": "用户明确要组局，信息齐全。",
                "slots": {
                    "game_type": {"value": "hangzhou_mahjong", "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                    "stake": {"value": "0.5", "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                    "start_time_mode": {"value": "people_ready", "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                    "missing_count": {"value": 3, "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                    "smoke": {"value": "any", "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                    "duration_mode": {"value": "overnight", "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                },
            }

    class SearchGameLLMClient:
        def complete(self, messages, *, trace_id, timeout_seconds):
            return {
                "intent": "inquire_existing_game",
                "proposed_action": "search_existing_games",
                "confidence": 0.9,
                "reasoning_summary": "用户咨询是否有 0.5 的现有局。",
                "slots": {
                    "game_type": {"value": "hangzhou_mahjong", "source": "context", "confidence": 0.8, "confirmed": True, "needs_confirmation": False},
                    "stake": {"value": "0.5", "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                    "smoke": {"value": "any", "source": "context", "confidence": 0.75, "confirmed": True, "needs_confirmation": False},
                    "start_time_mode": {"value": "people_ready", "source": "context", "confidence": 0.75, "confirmed": True, "needs_confirmation": False},
                },
            }

    state_path = tmp_path / "state" / "workflow_state.sqlite3"
    gate_path = tmp_path / "state" / "input_gate.sqlite3"
    first_runtime = build_controlled_runtime(
        llm_client=CreateGameLLMClient(),
        config=ControlledRuntimeConfig(
            trace_jsonl_path=tmp_path / "trace_sqlite_created_game_first.jsonl",
            state_sqlite_path=state_path,
            input_gate_sqlite_path=gate_path,
        ),
    )
    first_message = Message(
        text="通宵0.5人齐开，173，烟都可",
        sender_id="zhang",
        sender_name="张哥",
        channel_id="boss_trial",
        channel_type=ChannelType.WEB_CONSOLE,
        sent_at=NOW,
        id="msg_sqlite_created_game_first",
        metadata={"conversation_id": "boss_trial", "source_message_id": "wechat_sqlite_created_game_1", "sequence": 1},
    )

    first_runtime.service.handle_message(first_message, now=NOW, trace_id="trace_sqlite_created_game_first")

    restarted_runtime = build_controlled_runtime(
        llm_client=SearchGameLLMClient(),
        config=ControlledRuntimeConfig(
            trace_jsonl_path=tmp_path / "trace_sqlite_created_game_second.jsonl",
            state_sqlite_path=state_path,
            input_gate_sqlite_path=gate_path,
        ),
    )
    second_message = Message(
        text="现在有0.5的人齐开局吗",
        sender_id="wang",
        sender_name="王姐",
        channel_id="boss_trial",
        channel_type=ChannelType.WEB_CONSOLE,
        sent_at=NOW,
        id="msg_sqlite_created_game_second",
        metadata={"conversation_id": "boss_trial", "source_message_id": "wechat_sqlite_created_game_2", "sequence": 2},
    )

    second = restarted_runtime.service.handle_message(second_message, now=NOW, trace_id="trace_sqlite_created_game_second")

    assert isinstance(restarted_runtime.state_store, SQLiteWorkflowStateStore)
    assert second.context_build.context.open_games
    assert second.run.validated_action is not None
    assert second.run.validated_action.effective_action == ActionName.MATCH_EXISTING_GAME
    search_result = second.tool_orchestration.result_for(ToolName.SEARCH_CURRENT_OPEN_GAMES)
    assert search_result is not None
    assert search_result.result["result_count"] == 1


def test_controlled_runtime_can_use_sqlite_state_store(tmp_path) -> None:
    class CreateGameLLMClient:
        def complete(self, messages, *, trace_id, timeout_seconds):
            return {
                "intent": "find_players",
                "proposed_action": "create_game",
                "confidence": 0.92,
                "reasoning_summary": "用户明确要组局，信息齐全。",
                "slots": {
                    "game_type": {"value": "hangzhou_mahjong", "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                    "stake": {"value": "0.5", "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                    "start_time_mode": {"value": "people_ready", "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                    "missing_count": {"value": 3, "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                    "smoke": {"value": "any", "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                    "duration_mode": {"value": "overnight", "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                },
            }

    state_path = tmp_path / "state" / "workflow_state.sqlite3"
    runtime = build_controlled_runtime(
        llm_client=CreateGameLLMClient(),
        config=ControlledRuntimeConfig(
            trace_jsonl_path=tmp_path / "trace_sqlite_state.jsonl",
            state_sqlite_path=state_path,
        ),
    )
    message = Message(
        text="通宵0.5人齐开，173，烟都可",
        sender_id="zhang",
        sender_name="张哥",
        channel_id="boss_trial",
        channel_type=ChannelType.WEB_CONSOLE,
        sent_at=NOW,
        id="msg_runtime_sqlite_state",
        metadata={"conversation_id": "boss_trial"},
    )

    result = runtime.service.handle_message(message, now=NOW, trace_id="trace_runtime_sqlite_state")

    assert isinstance(runtime.state_store, SQLiteWorkflowStateStore)
    game_id = result.run.state_transitions[-1].entity_id
    reloaded = SQLiteWorkflowStateStore(state_path)
    assert reloaded.current_status("game", game_id) == GameWorkflowStatus.OPEN.value
    assert reloaded.transition_history(entity_type="game", entity_id=game_id)[0].metadata["store_backend"] == "sqlite"


def test_controlled_runtime_can_use_sqlite_tool_ledger(tmp_path) -> None:
    class SearchGameLLMClient:
        def complete(self, messages, *, trace_id, timeout_seconds):
            return {
                "intent": "inquire_existing_game",
                "proposed_action": "search_existing_games",
                "confidence": 0.88,
                "reasoning_summary": "用户咨询当前有没有合适局。",
                "slots": {
                    "stake": {"value": "0.5", "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                    "smoke": {"value": "no_smoke", "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                },
            }

    ledger_path = tmp_path / "ledger" / "tool_ledger.sqlite3"
    runtime = build_controlled_runtime(
        llm_client=SearchGameLLMClient(),
        config=ControlledRuntimeConfig(
            trace_jsonl_path=tmp_path / "trace_sqlite_tool_ledger.jsonl",
            tool_ledger_sqlite_path=ledger_path,
        ),
    )
    message = Message(
        text="现在有0.5无烟的吗",
        sender_id="zhang",
        sender_name="张哥",
        channel_id="boss_trial",
        channel_type=ChannelType.WEB_CONSOLE,
        sent_at=NOW,
        id="msg_runtime_sqlite_tool_ledger",
        metadata={"conversation_id": "boss_trial"},
    )

    runtime.service.handle_message(message, now=NOW, trace_id="trace_runtime_sqlite_tool_ledger")

    assert isinstance(runtime.tool_ledger, SQLiteToolExecutionLedger)
    history = SQLiteToolExecutionLedger(ledger_path).history(tool_name=ToolName.SEARCH_CURRENT_OPEN_GAMES)
    assert len(history) == 1
    assert history[0].called is True
    assert history[0].allowed is True


def test_controlled_runtime_can_use_sqlite_input_gate_across_restarts(tmp_path) -> None:
    class SearchGameLLMClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages, *, trace_id, timeout_seconds):
            self.calls += 1
            return {
                "intent": "inquire_existing_game",
                "proposed_action": "search_existing_games",
                "confidence": 0.88,
                "reasoning_summary": "用户咨询当前有没有合适局。",
                "slots": {
                    "stake": {
                        "value": "0.5",
                        "source": "explicit",
                        "confidence": 0.9,
                        "confirmed": True,
                        "needs_confirmation": False,
                    }
                },
            }

    class ExplodingLLMClient:
        def complete(self, messages, *, trace_id, timeout_seconds):
            raise AssertionError("duplicate message should not call LLM after restart")

    gate_path = tmp_path / "input_gate" / "gate.sqlite3"
    first_llm = SearchGameLLMClient()
    runtime = build_controlled_runtime(
        llm_client=first_llm,
        config=ControlledRuntimeConfig(
            trace_jsonl_path=tmp_path / "trace_sqlite_input_gate_first.jsonl",
            input_gate_sqlite_path=gate_path,
        ),
    )
    first_message = Message(
        text="现在有0.5的吗",
        sender_id="zhang",
        sender_name="张哥",
        channel_id="boss_trial",
        channel_type=ChannelType.WEB_CONSOLE,
        sent_at=NOW,
        id="msg_runtime_sqlite_gate_first",
        metadata={"conversation_id": "boss_trial", "source_message_id": "wechat_gate_001", "sequence": 1},
    )

    first = runtime.service.handle_message(first_message, now=NOW, trace_id="trace_runtime_sqlite_gate_first")

    restarted_runtime = build_controlled_runtime(
        llm_client=ExplodingLLMClient(),
        config=ControlledRuntimeConfig(
            trace_jsonl_path=tmp_path / "trace_sqlite_input_gate_second.jsonl",
            input_gate_sqlite_path=gate_path,
        ),
    )
    duplicate_message = Message(
        text="现在有0.5的吗",
        sender_id="zhang",
        sender_name="张哥",
        channel_id="boss_trial",
        channel_type=ChannelType.WEB_CONSOLE,
        sent_at=NOW,
        id="msg_runtime_sqlite_gate_duplicate",
        metadata={"conversation_id": "boss_trial", "source_message_id": "wechat_gate_001", "sequence": 1},
    )

    duplicate = restarted_runtime.service.handle_message(
        duplicate_message,
        now=NOW,
        trace_id="trace_runtime_sqlite_gate_duplicate",
    )

    assert isinstance(runtime.input_gate, SQLiteInputGate)
    assert isinstance(restarted_runtime.input_gate, SQLiteInputGate)
    assert first_llm.calls == 1
    assert first.final_text == "现在没有合适的，要组一个吗？"
    assert duplicate.final_text == first.final_text
    assert duplicate.run.validated_action is not None
    assert duplicate.run.validated_action.code == "input_gate_duplicate"


def test_controlled_runtime_can_use_sqlite_short_memory_across_restarts(tmp_path) -> None:
    class SearchGameLLMClient:
        def complete(self, messages, *, trace_id, timeout_seconds):
            return {
                "intent": "inquire_existing_game",
                "proposed_action": "search_existing_games",
                "confidence": 0.88,
                "reasoning_summary": "用户咨询当前有没有合适局。",
                "slots": {
                    "stake": {
                        "value": "0.5",
                        "source": "explicit",
                        "confidence": 0.9,
                        "confirmed": True,
                        "needs_confirmation": False,
                    }
                },
            }

    class CapturingLLMClient:
        def __init__(self) -> None:
            self.calls = []

        def complete(self, messages, *, trace_id, timeout_seconds):
            self.calls.append({"messages": messages, "trace_id": trace_id})
            return {
                "intent": "find_players",
                "proposed_action": "ask_clarification",
                "confidence": 0.82,
                "reasoning_summary": "当前消息是在回答上一轮是否组局，但仍缺少关键信息。",
                "slots": {
                    "stake": {
                        "value": "0.5",
                        "source": "context",
                        "confidence": 0.84,
                        "confirmed": True,
                        "needs_confirmation": False,
                    }
                },
            }

    memory_path = tmp_path / "memory" / "short_memory.sqlite3"
    gate_path = tmp_path / "memory" / "input_gate.sqlite3"
    first_runtime = build_controlled_runtime(
        llm_client=SearchGameLLMClient(),
        config=ControlledRuntimeConfig(
            trace_jsonl_path=tmp_path / "trace_sqlite_memory_first.jsonl",
            input_gate_sqlite_path=gate_path,
            short_memory_sqlite_path=memory_path,
        ),
    )
    first_message = Message(
        text="现在有0.5的吗",
        sender_id="zhang",
        sender_name="张哥",
        channel_id="boss_trial",
        channel_type=ChannelType.WEB_CONSOLE,
        sent_at=NOW,
        id="msg_runtime_sqlite_memory_first",
        metadata={"conversation_id": "boss_trial", "source_message_id": "wechat_memory_001", "sequence": 1},
    )

    first_runtime.service.handle_message(first_message, now=NOW, trace_id="trace_runtime_sqlite_memory_first")

    capturing_llm = CapturingLLMClient()
    restarted_runtime = build_controlled_runtime(
        llm_client=capturing_llm,
        config=ControlledRuntimeConfig(
            trace_jsonl_path=tmp_path / "trace_sqlite_memory_second.jsonl",
            input_gate_sqlite_path=gate_path,
            short_memory_sqlite_path=memory_path,
        ),
    )
    second_message = Message(
        text="可以",
        sender_id="zhang",
        sender_name="张哥",
        channel_id="boss_trial",
        channel_type=ChannelType.WEB_CONSOLE,
        sent_at=NOW,
        id="msg_runtime_sqlite_memory_second",
        metadata={"conversation_id": "boss_trial", "source_message_id": "wechat_memory_002", "sequence": 2},
    )

    result = restarted_runtime.service.handle_message(second_message, now=NOW, trace_id="trace_runtime_sqlite_memory_second")

    assert isinstance(first_runtime.memory_store, SQLiteShortTermMemoryStore)
    assert isinstance(restarted_runtime.memory_store, SQLiteShortTermMemoryStore)
    assert result.context_build.used_short_memory is True
    assert result.context_build.followup_context["previous_system_reply"] == "现在没有合适的，要组一个吗？"
    prompt_text = capturing_llm.calls[0]["messages"][1]["content"]
    assert '"previous_system_reply": "现在没有合适的，要组一个吗？"' in prompt_text
    assert '"value": "0.5"' in prompt_text


def test_controlled_runtime_can_use_sqlite_pending_outbox_store(tmp_path) -> None:
    class CreateGameLLMClient:
        def complete(self, messages, *, trace_id, timeout_seconds):
            return {
                "intent": "find_players",
                "proposed_action": "create_game",
                "confidence": 0.92,
                "reasoning_summary": "用户明确要组局，信息齐全。",
                "slots": {
                    "game_type": {"value": "hangzhou_mahjong", "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                    "stake": {"value": "0.5", "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                    "start_time_mode": {"value": "people_ready", "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                    "missing_count": {"value": 3, "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                    "smoke": {"value": "no_smoke", "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                    "duration_hours": {"value": 4, "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                },
            }

    core = AgentCore()
    core.upsert_customer(
        CustomerProfile(
            id="ran",
            display_name="冉姐",
            preferred_levels=["0.5"],
            smoke_free_preference=True,
            play_preferences=[PlayPreference(game_type="hangzhou_mahjong", preferred_levels=["0.5"])],
            usual_start_hours=[16, 17],
        )
    )
    outbox_path = tmp_path / "outbox" / "pending_outbox.sqlite3"
    runtime = build_controlled_runtime(
        core=core,
        llm_client=CreateGameLLMClient(),
        config=ControlledRuntimeConfig(
            trace_jsonl_path=tmp_path / "trace_sqlite_outbox.jsonl",
            outbox_sqlite_path=outbox_path,
        ),
    )
    message = Message(
        text="0.5无烟人齐开，173，4h",
        sender_id="zhang",
        sender_name="张哥",
        channel_id="boss_trial",
        channel_type=ChannelType.WEB_CONSOLE,
        sent_at=NOW,
        id="msg_runtime_sqlite_outbox",
        metadata={"conversation_id": "boss_trial"},
    )

    result = runtime.service.handle_message(message, now=NOW, trace_id="trace_runtime_sqlite_outbox")

    outbox_result = result.tool_orchestration.result_for(ToolName.CREATE_PENDING_OUTBOX)
    assert outbox_result is not None
    assert outbox_result.called is True
    assert outbox_result.result["stored_count"] == 1
    pending = SQLitePendingOutboxStore(outbox_path).list_pending(conversation_id="boss_trial")
    assert len(pending) == 1
    assert pending[0]["target_customer_id"] == "ran"
    assert pending[0]["message_text"].endswith("打吗？")
    assert runtime.outbox_store is not None
    assert runtime.approval_service is not None

    approved = runtime.approval_service.decide(
        outbox_id=pending[0]["id"],
        decision="approved",
        reviewer_id="boss",
        reason="运行时审批测试",
        trace_id="trace_runtime_approval",
        idempotency_key="runtime_approval_once",
    )
    persisted = SQLitePendingOutboxStore(outbox_path).get(pending[0]["id"])

    assert approved["ok"] is True
    assert persisted["status"] == OUTBOX_APPROVED
    assert persisted["metadata"]["decision_trace_id"] == "trace_runtime_approval"
    assert runtime.tool_ledger.history(tool_name=ToolName.RECORD_APPROVAL_DECISION)


def test_controlled_runtime_loads_customer_profiles_from_sqlite_after_restart(tmp_path) -> None:
    class CreateGameLLMClient:
        def complete(self, messages, *, trace_id, timeout_seconds):
            return {
                "intent": "find_players",
                "proposed_action": "create_game",
                "confidence": 0.92,
                "reasoning_summary": "用户明确要组局，信息齐全。",
                "slots": {
                    "game_type": {"value": "hangzhou_mahjong", "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                    "stake": {"value": "0.5", "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                    "start_time_mode": {"value": "people_ready", "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                    "missing_count": {"value": 3, "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                    "smoke": {"value": "no_smoke", "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                    "duration_hours": {"value": 4, "source": "explicit", "confidence": 0.9, "confirmed": True, "needs_confirmation": False},
                },
            }

    customer_path = tmp_path / "customers" / "profiles.sqlite3"
    seeded_core = AgentCore()
    seeded_core.upsert_customer(
        CustomerProfile(
            id="ran",
            display_name="冉姐",
            preferred_levels=["0.5"],
            smoke_free_preference=True,
            play_preferences=[PlayPreference(game_type="hangzhou_mahjong", preferred_levels=["0.5"])],
            usual_start_hours=[16, 17],
        )
    )
    build_controlled_runtime(
        core=seeded_core,
        llm_client=CreateGameLLMClient(),
        config=ControlledRuntimeConfig(
            trace_jsonl_path=tmp_path / "trace_customer_seed.jsonl",
            customer_profile_sqlite_path=customer_path,
        ),
    )

    restarted_runtime = build_controlled_runtime(
        llm_client=CreateGameLLMClient(),
        config=ControlledRuntimeConfig(
            trace_jsonl_path=tmp_path / "trace_customer_reload.jsonl",
            customer_profile_sqlite_path=customer_path,
        ),
    )
    message = Message(
        text="0.5无烟人齐开，173，4h",
        sender_id="zhang",
        sender_name="张哥",
        channel_id="boss_trial",
        channel_type=ChannelType.WEB_CONSOLE,
        sent_at=NOW,
        id="msg_runtime_customer_reload",
        metadata={"conversation_id": "boss_trial"},
    )

    result = restarted_runtime.service.handle_message(message, now=NOW, trace_id="trace_runtime_customer_reload")

    assert isinstance(restarted_runtime.customer_repository, SQLiteCustomerProfileRepository)
    assert "ran" in restarted_runtime.core.store.customers
    candidate_result = result.tool_orchestration.result_for(ToolName.SEARCH_CANDIDATE_CUSTOMERS)
    assert candidate_result is not None
    assert candidate_result.result["candidates"][0]["customer_id"] == "ran"


def test_controlled_profile_update_persists_observations_to_sqlite(tmp_path) -> None:
    class ProfileObservationLLMClient:
        def complete(self, messages, *, trace_id, timeout_seconds):
            return {
                "intent": "find_players",
                "proposed_action": "ask_clarification",
                "confidence": 0.82,
                "reasoning_summary": "用户说烟都可，先追问缺失信息，同时沉淀画像观察。",
                "slots": {
                    "smoke": {
                        "value": "any",
                        "source": "explicit",
                        "confidence": 0.9,
                        "confirmed": True,
                        "needs_confirmation": False,
                    }
                },
                "profile_observations": [
                    {
                        "field": "smoke_preference",
                        "value": "any",
                        "confidence": 0.82,
                        "source": "current_message",
                        "evidence": "用户说烟都可",
                        "risk": "low",
                    }
                ],
            }

    customer_path = tmp_path / "customers" / "profiles.sqlite3"
    runtime = build_controlled_runtime(
        llm_client=ProfileObservationLLMClient(),
        config=ControlledRuntimeConfig(
            trace_jsonl_path=tmp_path / "trace_profile_observation.jsonl",
            customer_profile_sqlite_path=customer_path,
        ),
    )
    message = Message(
        text="老板，今天有人打吗，烟都可",
        sender_id="zhang",
        sender_name="张哥",
        channel_id="boss_trial",
        channel_type=ChannelType.WEB_CONSOLE,
        sent_at=NOW,
        id="msg_runtime_profile_observation",
        metadata={"conversation_id": "boss_trial"},
    )

    result = runtime.service.handle_message(message, now=NOW, trace_id="trace_runtime_profile_observation")

    profile_update = result.tool_orchestration.result_for(ToolName.PROFILE_UPDATE)
    assert profile_update is not None
    assert profile_update.result["applied_count"] == 1
    loaded = SQLiteCustomerProfileRepository(customer_path).load_all()
    assert loaded[0].id == "zhang"
    observations = loaded[0].metadata["controlled_profile_observations"]
    assert observations[0]["field"] == "smoke_preference"
    assert observations[0]["evidence"] == "用户说烟都可"
