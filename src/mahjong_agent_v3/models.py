"""Compatibility model imports for historical ``mahjong_agent_v3`` users."""

from __future__ import annotations

from mahjong_agent_runtime.models import (
    DEFAULT_TZ as DEFAULT_TZ_V3,
    AgentAction as AgentActionV3,
    AgentRuntimeResult as AgentRuntimeResultV3,
    ConversationCheckpoint as ConversationCheckpointV3,
    ConversationRole as ConversationRoleV3,
    ConversationTurn as ConversationTurnV3,
    CustomerProfile as CustomerProfileV3,
    Game as GameV3,
    GameParticipant as GameParticipantV3,
    GameStatus as GameStatusV3,
    InviteDraft as InviteDraftV3,
    InviteStatus as InviteStatusV3,
    OutboundDraftStatus as OutboundDraftStatusV3,
    OutboundMessageDraft as OutboundMessageDraftV3,
    StateTransition as StateTransitionV3,
    ToolCall as ToolCallV3,
    ToolResult as ToolResultV3,
    UserMessage as UserMessageV3,
    new_id,
    now as now_v3,
)

__all__ = [
    "DEFAULT_TZ_V3",
    "AgentActionV3",
    "AgentRuntimeResultV3",
    "ConversationCheckpointV3",
    "ConversationRoleV3",
    "ConversationTurnV3",
    "CustomerProfileV3",
    "GameParticipantV3",
    "GameStatusV3",
    "GameV3",
    "InviteDraftV3",
    "InviteStatusV3",
    "OutboundDraftStatusV3",
    "OutboundMessageDraftV3",
    "StateTransitionV3",
    "ToolCallV3",
    "ToolResultV3",
    "UserMessageV3",
    "new_id",
    "now_v3",
]
