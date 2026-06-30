from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .action_validator import ActionValidator
from .approval import PendingOutboxApprovalConfig, PendingOutboxApprovalService
from .context_builder import WorkflowContextBuilder, WorkflowContextBuilderConfig
from .controlled_workflow import ControlledWorkflowConfig, ControlledWorkflowService
from .core import AgentCore
from .input_gate import InMemoryInputGate, InputGate
from .llm_client import OpenAICompatibleSemanticLLMClient
from .memory import InMemoryShortTermMemoryStore, ShortTermMemoryStore
from .observability import JsonlTraceRecorder, TraceRecorder
from .reply_guard import ReplyGuard
from .reply_policy import ReplyDraftLLMClient, ReplyPolicy
from .semantic_resolver import SemanticLLMClient, SemanticResolver, SemanticResolverConfig
from .state_machine import InMemoryWorkflowStateStore, SQLiteWorkflowStateStore, StateMachine, WorkflowStateStore
from .tool_orchestrator import (
    InMemoryToolExecutionLedger,
    SQLiteToolExecutionLedger,
    ToolExecutionLedger,
    ToolOrchestrator,
    ToolOrchestratorConfig,
)
from .tools import PendingOutboxStore, PendingOutboxTool, SQLitePendingOutboxStore


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACE_PATH = ROOT / "logs" / "controlled_workflow_trace.jsonl"


@dataclass(slots=True)
class ControlledRuntimeConfig:
    trace_jsonl_path: Path = DEFAULT_TRACE_PATH
    state_sqlite_path: Path | None = None
    tool_ledger_sqlite_path: Path | None = None
    outbox_sqlite_path: Path | None = None
    short_memory_ttl_seconds: int = 30 * 60
    short_memory_max_records: int = 20
    llm_timeout_seconds: float | None = None
    fail_closed_without_llm: bool = True
    approval_enabled: bool = True

    @classmethod
    def from_env(cls) -> "ControlledRuntimeConfig":
        return cls(
            trace_jsonl_path=Path(os.getenv("MAHJONG_TRACE_JSONL_PATH", str(DEFAULT_TRACE_PATH))),
            state_sqlite_path=Path(os.environ["MAHJONG_STATE_SQLITE_PATH"])
            if os.getenv("MAHJONG_STATE_SQLITE_PATH")
            else None,
            tool_ledger_sqlite_path=Path(os.environ["MAHJONG_TOOL_LEDGER_SQLITE_PATH"])
            if os.getenv("MAHJONG_TOOL_LEDGER_SQLITE_PATH")
            else None,
            outbox_sqlite_path=Path(os.environ["MAHJONG_OUTBOX_SQLITE_PATH"])
            if os.getenv("MAHJONG_OUTBOX_SQLITE_PATH")
            else None,
            short_memory_ttl_seconds=int(os.getenv("MAHJONG_SHORT_MEMORY_TTL_SECONDS", str(30 * 60))),
            short_memory_max_records=int(os.getenv("MAHJONG_SHORT_MEMORY_MAX_RECORDS", "20")),
            llm_timeout_seconds=_env_float("MAHJONG_LLM_TIMEOUT_SECONDS"),
            fail_closed_without_llm=_env_bool("MAHJONG_FAIL_CLOSED_WITHOUT_LLM", True),
            approval_enabled=_env_bool("MAHJONG_APPROVAL_ENABLED", True),
        )


@dataclass(slots=True)
class ControlledRuntime:
    service: ControlledWorkflowService
    core: AgentCore
    memory_store: ShortTermMemoryStore
    state_store: WorkflowStateStore
    tool_ledger: ToolExecutionLedger
    input_gate: InputGate
    outbox_store: PendingOutboxStore | None
    approval_service: PendingOutboxApprovalService | None
    trace_recorder: TraceRecorder
    config: ControlledRuntimeConfig


class FailClosedSemanticLLMClient:
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        trace_id: str,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        return {
            "intent": "unknown",
            "proposed_action": "human_review",
            "confidence": 0.0,
            "needs_human_review": True,
            "reasoning_summary": "LLM 未配置，受控工作流按失败关闭策略转人工。",
            "slots": {},
        }


def build_controlled_runtime(
    *,
    core: AgentCore | None = None,
    llm_client: SemanticLLMClient | None = None,
    reply_llm_client: ReplyDraftLLMClient | None = None,
    memory_store: ShortTermMemoryStore | None = None,
    state_store: WorkflowStateStore | None = None,
    tool_ledger: ToolExecutionLedger | None = None,
    trace_recorder: TraceRecorder | None = None,
    config: ControlledRuntimeConfig | None = None,
) -> ControlledRuntime:
    runtime_config = config or ControlledRuntimeConfig.from_env()
    runtime_core = core or AgentCore()
    recorder = trace_recorder or JsonlTraceRecorder(runtime_config.trace_jsonl_path)
    memory = memory_store or InMemoryShortTermMemoryStore(
        ttl_seconds=runtime_config.short_memory_ttl_seconds,
        max_records_per_scope=runtime_config.short_memory_max_records,
    )
    workflow_state_store = state_store or _state_store_from_config(runtime_config)
    workflow_tool_ledger = tool_ledger or _tool_ledger_from_config(runtime_config)
    input_gate = InMemoryInputGate()
    outbox_store = _outbox_store_from_config(runtime_config)
    pending_outbox_tool = PendingOutboxTool(store=outbox_store) if outbox_store is not None else None
    approval_service = (
        PendingOutboxApprovalService(
            outbox_store,
            execution_ledger=workflow_tool_ledger,
            config=PendingOutboxApprovalConfig(approval_enabled=runtime_config.approval_enabled),
        )
        if outbox_store is not None
        else None
    )
    semantic_client = llm_client or _llm_client_from_env(recorder, runtime_config, stage_name="semantic")
    reply_client = reply_llm_client
    if reply_client is None and llm_client is None:
        reply_client = _optional_llm_client_from_env(recorder, stage_name="reply")
    context_builder = WorkflowContextBuilder(
        runtime_core,
        memory,
        WorkflowContextBuilderConfig(
            max_memory_records=min(8, runtime_config.short_memory_max_records),
        ),
    )
    semantic_resolver_config = SemanticResolverConfig()
    if runtime_config.llm_timeout_seconds is not None:
        semantic_resolver_config.timeout_seconds = runtime_config.llm_timeout_seconds
    service = ControlledWorkflowService(
        core=runtime_core,
        context_builder=context_builder,
        semantic_resolver=SemanticResolver(semantic_client, semantic_resolver_config),
        action_validator=ActionValidator(),
        tool_orchestrator=ToolOrchestrator(
            runtime_core,
            ToolOrchestratorConfig(allow_state_write=True),
            outbox_tool=pending_outbox_tool,
            execution_ledger=workflow_tool_ledger,
        ),
        state_machine=StateMachine(),
        state_store=workflow_state_store,
        reply_policy=ReplyPolicy(reply_client),
        reply_guard=ReplyGuard(),
        memory_store=memory,
        input_gate=input_gate,
        trace_recorder=recorder,
        config=ControlledWorkflowConfig(persist_short_memory=True),
    )
    return ControlledRuntime(
        service=service,
        core=runtime_core,
        memory_store=memory,
        state_store=workflow_state_store,
        tool_ledger=workflow_tool_ledger,
        input_gate=input_gate,
        outbox_store=outbox_store,
        approval_service=approval_service,
        trace_recorder=recorder,
        config=runtime_config,
    )


def _llm_client_from_env(
    trace_recorder: TraceRecorder,
    config: ControlledRuntimeConfig,
    *,
    stage_name: str,
) -> SemanticLLMClient:
    client = OpenAICompatibleSemanticLLMClient.from_env(
        audit_logger=lambda trace_id, event, payload: trace_recorder.record(
            trace_id,
            f"llm_client.{event}",
            payload,
        ),
        stage_name=stage_name,
    )
    if client is not None:
        return client
    if config.fail_closed_without_llm:
        return FailClosedSemanticLLMClient()
    raise RuntimeError("LLM is not configured. Set MAHJONG_LLM_API_KEY and MAHJONG_LLM_MODEL.")


def _optional_llm_client_from_env(
    trace_recorder: TraceRecorder,
    *,
    stage_name: str,
) -> OpenAICompatibleSemanticLLMClient | None:
    return OpenAICompatibleSemanticLLMClient.from_env(
        audit_logger=lambda trace_id, event, payload: trace_recorder.record(
            trace_id,
            f"llm_client.{event}",
            payload,
        ),
        stage_name=stage_name,
    )


def _state_store_from_config(config: ControlledRuntimeConfig) -> WorkflowStateStore:
    if config.state_sqlite_path is not None:
        return SQLiteWorkflowStateStore(config.state_sqlite_path)
    return InMemoryWorkflowStateStore()


def _tool_ledger_from_config(config: ControlledRuntimeConfig) -> ToolExecutionLedger:
    if config.tool_ledger_sqlite_path is not None:
        return SQLiteToolExecutionLedger(config.tool_ledger_sqlite_path)
    return InMemoryToolExecutionLedger()


def _outbox_store_from_config(config: ControlledRuntimeConfig) -> PendingOutboxStore | None:
    if config.outbox_sqlite_path is None:
        return None
    return SQLitePendingOutboxStore(config.outbox_sqlite_path)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return float(raw)
