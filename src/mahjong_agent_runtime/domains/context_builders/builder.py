"""为每轮模型调用构建最小、隔离且可审计的受控上下文。

"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...models import ToolResult, UserMessage
from ...stores import AgentStore
from ...group_chat.projections import public_group_game_summary
from ...knowledge import DomainTerminologyRepository, default_terminology_repository
from ..tools.gateway import ToolGateway
from .contracts import output_contract, planning_contract
from .conversation_context import (
    ContextPackingPolicy,
    build_conversation_context,
)
from .customer_context import build_customer_context
from .game_context import build_game_context
from .message_context import (
    build_message_reference_contract,
    resolve_quoted_message_context,
)
from .relationship_context import build_relationship_context
from .sanitization import sanitize_current_message_for_context
from .tool_results import tool_result_for_context
from .task_recovery import recover_referenced_task_contexts
from .task_facts import project_explicit_task_facts


DEFAULT_PROMPT_PATH = Path(__file__).resolve().parents[2].joinpath("prompts", "agent_runtime_system.md")


@dataclass(slots=True)
class BuiltContext:
    """保存一次模型调用的消息、结构化载荷和可观测审计信息。"""
    messages: list[dict[str, str]]
    payload: dict[str, Any]
    audit: dict[str, Any]


@dataclass(slots=True)
class AgentContextBuilder:
    """按会话、任务和客户可见性边界组装每轮模型真正需要的上下文。

    上游：``ContextLifecycleManager`` 在每个主模型步骤调用。下游：各领域上下文投影
    和 ``AgentStore`` 读接口。边界：这里不判断完整意图、不写业务状态；返回值中
    ``messages`` 才是真正发送给模型的内容，``payload`` 是结构化来源，``audit`` 仅审计。
    """

    store: AgentStore
    tool_gateway: ToolGateway
    prompt_path: Path = DEFAULT_PROMPT_PATH
    packing_policy: ContextPackingPolicy = field(default_factory=ContextPackingPolicy)
    terminology: DomainTerminologyRepository = field(default_factory=default_terminology_repository)

    def build(
        self,
        message: UserMessage,
        *,
        trace_id: str,
        previous_tool_results: list[ToolResult] | None = None,
        turn_tool_evidence: list[ToolResult] | None = None,
        run_id: str | None = None,
        run_version: int | None = None,
    ) -> BuiltContext:
        """只读 Store 并组合一轮受控模型请求；不把原始数据库整体暴露给模型。"""

        prompt = self.prompt_path.read_text(encoding="utf-8")
        # latest_tool_results 只表示上一模型步刚返回的结果；turn_tool_evidence
        # 则保留本轮任务已经取得的全部证据，防止多步调用时模型遗忘早期结果。
        latest_tool_results = list(previous_tool_results or [])
        current_turn_evidence = list(
            turn_tool_evidence if turn_tool_evidence is not None else latest_tool_results
        )
        # 会话记录先按 task_context 切成当前业务片段，再做轮数和 token 预算打包。
        # 因此同一客户上午和下午的两次组局不会默认混在一个上下文中。
        conversation = build_conversation_context(
            self.store,
            message,
            trace_id=trace_id,
            previous_tool_results=current_turn_evidence,
            packing_policy=self.packing_policy,
        )
        task_context = conversation.task_context
        checkpoint = conversation.checkpoint
        current_version = self.store.conversation_version(message.conversation_id)

        # 各子构建器只负责投影自己的领域数据，主构建器负责组合而不承担业务决策。
        games = build_game_context(
            self.store,
            conversation_id=message.conversation_id,
            sender_id=message.sender_id,
        )
        customer = build_customer_context(
            self.store,
            conversation_id=message.conversation_id,
            sender_id=message.sender_id,
            task_context=task_context,
        )
        relationship = build_relationship_context(
            self.store,
            sender_id=message.sender_id,
            active_games=games.games,
        )

        # 原始消息先去除不应交给模型的通道内部字段；明确事实另外投影成结构化槽位，
        # 避免模型每一步都从自然语言历史中重新猜测人数、时间和烟况。
        current_message = sanitize_current_message_for_context(message.to_dict())
        explicit_task_facts = project_explicit_task_facts(
            conversation.recent_conversation,
            current_message,
            checkpoint,
        )
        system_trigger = _system_trigger_context(message)
        # 引用消息优先按 source_message_id 回查持久化原文及业务引用。
        # 它是恢复旧任务的显式入口，不会因为一段相似文本就把上午任务误带进下午任务。
        quoted_message_context = resolve_quoted_message_context(self.store, message, current_message)
        message_reference_contract, quoted_reference_status = build_message_reference_contract(
            message,
            quoted_message_context,
        )
        quoted_message = message.quoted_message
        recovered_tasks = recover_referenced_task_contexts(
            self.store,
            message,
            current_message,
            current_task_context_id=task_context.task_context_id if task_context else None,
            packing_policy=self.packing_policy,
        )
        # 回复长度、群 ID 等约束只接受通道适配层写入的可信 metadata，
        # 不能让用户通过普通文本伪造系统约束或切换到其他会话。
        reply_constraints = _trusted_reply_constraints(message)
        group_room_id = _trusted_group_room_id(message)
        room_board_games = (
            [
                public_group_game_summary(game)
                for game in self.store.get_board_eligible_games(group_room_id)
            ]
            if group_room_id
            else []
        )
        # 术语库按当前任务文本按需召回，而不是把所有地方玩法说明塞满上下文。
        domain_terminology = self.terminology.context_for_texts(
            _terminology_context_texts(
                current_message=current_message,
                recent_conversation=conversation.recent_conversation,
                quoted_message_context=quoted_message_context,
                system_trigger=system_trigger,
            )
        )

        # audit 不参与模型决策，只记录本次组装使用了哪些来源，便于追溯串话、漏召回和预算问题。
        audit = {
            **conversation.audit,
            "sender_relationship_count": len(relationship.relationships),
            "task_memory_count": len(customer.task_memories),
            "pending_memory_candidate_count": len(customer.pending_memory_candidates),
            "active_game_visible_summary_count": len(games.visible_summaries),
            "quoted_message_present": quoted_message is not None,
            "quoted_message_id": quoted_message.message_id if quoted_message else None,
            "quoted_message_reference_resolved": quoted_message_context is not None,
            "quoted_message_reference_status": quoted_reference_status,
            "quoted_message_business_ref_type": (
                quoted_message_context.get("business_ref_type")
                if quoted_message_context
                else None
            ),
            "conversation_version": current_version,
            "run_version": run_version,
            "run_current": run_version is None or int(run_version) == current_version,
            "task_context_id": task_context.task_context_id if task_context else None,
            "task_context_started_at": task_context.started_at.isoformat() if task_context else None,
            "system_trigger_present": system_trigger is not None,
            "system_trigger_type": system_trigger.get("trigger_type") if system_trigger else None,
            "reply_constraints_present": reply_constraints is not None,
            "group_room_board_game_count": len(room_board_games),
            "explicit_task_fact_count": len(explicit_task_facts["facts"]),
            "latest_tool_result_count": len(latest_tool_results),
            "turn_tool_evidence_count": len(current_turn_evidence),
            "turn_tool_evidence_complete": True,
            "domain_terminology_count": len(domain_terminology),
            "domain_terminology_term_ids": [item["term_id"] for item in domain_terminology],
            **recovered_tasks.audit,
        }

        # 模型看到的是有字段语义的 JSON，而不是由字符串拼接形成的一大段聊天记录。
        # 这让工具合同、任务事实、可见性边界和当前状态都能独立校验与演进。
        payload = {
            "runtime": "mahjong_agent_runtime",
            "trace_id": trace_id,
            "customer_visibility_contract": relationship.visibility_contract,
            "conversation_state": self._conversation_state(
                message,
                task_context_id=task_context.task_context_id if task_context else None,
                current_version=current_version,
                run_id=run_id,
                run_version=run_version,
            ),
            "task_context_window": self._task_context_window(task_context),
            "current_message": current_message,
            "domain_terminology": domain_terminology,
            "explicit_task_facts": explicit_task_facts,
            "reply_constraints": reply_constraints,
            "group_room_board_games": room_board_games,
            "system_trigger": system_trigger,
            "message_reference_contract": message_reference_contract,
            "quoted_message_context": quoted_message_context,
            "recent_conversation": conversation.recent_conversation,
            "conversation_checkpoint": checkpoint.to_dict() if checkpoint else None,
            "recovered_task_contexts": recovered_tasks.items,
            "context_budget": audit,
            "sender_profile": customer.profile,
            "sender_relationships": relationship.relationships,
            "task_memories": customer.task_memories,
            "pending_memory_candidates": customer.pending_memory_candidates,
            "active_games": games.model_contexts,
            "active_game_visible_summaries": games.visible_summaries,
            "sender_active_game_memberships": games.sender_memberships,
            "active_parties": games.active_parties,
            "outbound_message_drafts": customer.outbound_message_drafts,
            "available_tools": self.tool_gateway.tool_specs_for_prompt(),
            "previous_tool_results": [
                tool_result_for_context(item)
                for item in latest_tool_results
            ],
            "turn_tool_evidence": [
                tool_result_for_context(item)
                for item in current_turn_evidence
            ],
            "planning_contract": planning_contract(),
            "output_contract": output_contract(),
        }
        # 稳定行为准则放 system；每轮动态状态放 user JSON。群聊临时约束使用额外的
        # system 消息，但其值仍由后端可信信封产生，用户无法自行注入。
        messages = [{"role": "system", "content": prompt}]
        if reply_constraints is not None:
            privacy = "且不得提及任何其他客户的身份、关系或私聊信息" if reply_constraints["no_private_info"] else ""
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "【本轮客户可见回复约束】只输出当前用户真正需要的最短回复；"
                        f"回复不得超过 {reply_constraints['max_length']} 个中文字符{privacy}。"
                    ),
                }
            )
        messages.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)})
        return BuiltContext(
            messages=messages,
            payload=payload,
            audit=audit,
        )

    def _resolve_quoted_message_context(
        self,
        message: UserMessage,
        current_message: dict[str, Any],
    ) -> dict[str, Any] | None:
        """兼容旧调用方，委托统一的引用解析器恢复被引用消息。

"""

        return resolve_quoted_message_context(self.store, message, current_message)

    @staticmethod
    def _conversation_state(
        message: UserMessage,
        *,
        task_context_id: str | None,
        current_version: int,
        run_id: str | None,
        run_version: int | None,
    ) -> dict[str, Any]:
        """向模型暴露当前运行版本，阻止旧运行覆盖较新的用户输入。"""
        return {
            "conversation_id": message.conversation_id,
            "task_context_id": task_context_id,
            "current_version": current_version,
            "run_id": run_id,
            "run_version": run_version,
            "run_current": run_version is None or int(run_version) == current_version,
            "version_contract": (
                "每条新用户消息都会推进 conversation version；旧版本未发送的回复、邀约草稿和外发草稿会被标记为 superseded。"
                "如果工具结果提示 stale_run，必须停止旧动作并基于当前消息重新判断。"
            ),
        }

    @staticmethod
    def _task_context_window(task_context: Any) -> dict[str, Any]:
        """声明本轮业务片段的边界以及可跨任务复用的数据范围。"""
        return {
            "task_context_id": task_context.task_context_id if task_context else None,
            "started_at": task_context.started_at.isoformat() if task_context else None,
            "reset_reason": task_context.reset_reason if task_context else None,
            "scope_contract": (
                "recent_conversation, conversation_checkpoint, task_memories and pending task facts only belong "
                "to this business episode. Stable sender_profile and approved customer relationships may cross episodes."
            ),
        }


def _system_trigger_context(message: UserMessage) -> dict[str, Any] | None:
    """仅提取后端创建的系统触发器，拒绝把普通用户文本当成内部事件。

"""

    metadata = message.metadata if isinstance(message.metadata, dict) else {}
    if metadata.get("input_source") != "system_trigger":
        return None
    trigger = metadata.get("system_trigger")
    return dict(trigger) if isinstance(trigger, dict) else None


def _trusted_reply_constraints(message: UserMessage) -> dict[str, Any] | None:
    """读取群聊入口写入的回复限制，并把长度收敛到安全范围。

"""

    metadata = message.metadata if isinstance(message.metadata, dict) else {}
    if metadata.get("source") != "group":
        return None
    raw = metadata.get("reply_constraints")
    if not isinstance(raw, dict):
        return None
    try:
        max_length = max(1, min(100, int(raw.get("max_length") or 0)))
    except (TypeError, ValueError):
        return None
    return {
        "max_length": max_length,
        "no_private_info": bool(raw.get("no_private_info", True)),
    }


def _trusted_group_room_id(message: UserMessage) -> str | None:
    """只从后端群聊信封读取房间标识，防止用户文本伪造目标群。

"""

    metadata = message.metadata if isinstance(message.metadata, dict) else {}
    if metadata.get("source") != "group":
        return None
    room_id = str(metadata.get("room_id") or "").strip()
    return room_id or None


def _terminology_context_texts(
    *,
    current_message: dict[str, Any],
    recent_conversation: list[dict[str, Any]],
    quoted_message_context: dict[str, Any] | None,
    system_trigger: dict[str, Any] | None,
) -> list[str]:
    """选择当前任务的少量文本，用于按需召回麻将领域术语。

"""

    texts = [str(current_message.get("text") or "")]
    for turn in recent_conversation[-8:]:
        texts.append(str(turn.get("content") or turn.get("text") or ""))
    if quoted_message_context is not None:
        texts.append(str(quoted_message_context.get("text") or quoted_message_context.get("content") or ""))
    if system_trigger is not None:
        texts.append(json.dumps(system_trigger, ensure_ascii=False, sort_keys=True))
    return [item for item in texts if item.strip()]


__all__ = [
    "AgentContextBuilder",
    "BuiltContext",
    "ContextPackingPolicy",
    "DEFAULT_PROMPT_PATH",
]
