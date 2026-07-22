"""Built-in tool definitions and registry assembly."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

from ...models import ToolCall, ToolResult
from ...stores import AgentStore
from .audit_tools import record_badcase, record_user_memory, update_context_checkpoint
from .draft_tools import create_invite_drafts, create_outbound_message_drafts
from .mutation_tools import (
    create_game,
    join_game,
    record_candidate_reply,
    reserve_room,
    update_game_requirement,
    update_game_status,
)
from .schemas import (
    badcase_schema,
    checkpoint_schema,
    invitation_schema,
    known_player_schema,
    memory_schema,
    non_empty_string,
    outbound_message_draft_schema,
    requesting_party_schema,
    requirement_schema,
)
from .search_tools import check_room_availability, search_current_games, search_customers
from .shared import CANDIDATE_REPLY_STATUSES, GAME_STATUSES
from .waiting_tools import cancel_waiting_demand, register_waiting_demand


ToolHandler = Callable[[ToolCall, str, str, str, str], ToolResult]


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    risk_level: str
    execution_mode: str
    schema: dict[str, Any]
    handler: ToolHandler | None = None
    parallel_safe: bool = False

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "risk_level": self.risk_level,
            "execution_mode": self.execution_mode,
            "parallel_safe": self.parallel_safe,
            "schema": self.schema,
        }


def default_tool_definitions(store: AgentStore) -> dict[str, ToolDefinition]:
    return {
        "check_room_availability": ToolDefinition(
            "check_room_availability",
            "只读查询指定起止时间内的真实房间库存。只要用户询问明确时段的局，推荐或创建前先查询；未配置库存时不能声称有房。",
            "low",
            "read_only",
            {
                "type": "object",
                "required": ["start_at", "end_at"],
                "additionalProperties": False,
                "properties": {
                    "start_at": non_empty_string,
                    "end_at": non_empty_string,
                },
            },
            partial(check_room_availability, store),
            parallel_safe=True,
        ),
        "reserve_room": ToolDefinition(
            "reserve_room",
            "在已确认时间区间内原子占用一个可用房间。必须先查询库存；成功才表示已暂占，不能凭模型文字承诺。",
            "medium",
            "state_write",
            {
                "type": "object",
                "required": ["start_at", "end_at"],
                "additionalProperties": False,
                "properties": {
                    "game_id": {"type": "string"},
                    "room_id": {"type": "string"},
                    "start_at": non_empty_string,
                    "end_at": non_empty_string,
                },
            },
            partial(reserve_room, store),
        ),
        "search_current_games": ToolDefinition(
            "search_current_games",
            "只读查询当前局池。模型提供结构化 requirement；工具只按字段匹配，不理解自然语言。",
            "low",
            "read_only",
            {"type": "object", "required": ["requirement"], "properties": {"requirement": requirement_schema, "limit": {"type": "integer", "minimum": 1, "maximum": 20}}},
            partial(search_current_games, store),
            parallel_safe=True,
        ),
        "search_customers": ToolDefinition(
            "search_customers",
            "只读查询候选客户。模型负责给出筛选条件；工具只做确定性排序，并会参考关系画像避开不愿同桌的人。若已知当前局内人员，应在 requirement 里提供 existing_player_ids 或 organizer_id。",
            "low",
            "read_only",
            {"type": "object", "required": ["requirement"], "properties": {"requirement": requirement_schema, "exclude_customer_ids": {"type": "array", "items": {"type": "string"}}, "limit": {"type": "integer", "minimum": 1, "maximum": 20}}},
            partial(search_customers, store),
            parallel_safe=True,
        ),
        "register_waiting_demand": ToolDefinition(
            "register_waiting_demand",
            "当前没有匹配局且客户明确愿意等待时，将需求登记到等待列表；有合适局时系统会主动通知并再次征求客户确认。登记不代表客户已加入任何局。",
            "medium",
            "state_write",
            {
                "type": "object",
                "required": ["stake", "smoke_preference", "time_preference"],
                "additionalProperties": False,
                "properties": {
                    "stake": non_empty_string,
                    "smoke_preference": {
                        "type": "string",
                        "enum": ["烟", "无烟", "不限"],
                    },
                    "time_preference": non_empty_string,
                    "extra_constraints": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "expires_at": {"type": "string"},
                },
            },
            partial(register_waiting_demand, store),
        ),
        "cancel_waiting_demand": ToolDefinition(
            "cancel_waiting_demand",
            "客户明确表示算了、不打了或不再等待时，取消其本人当前会话中的等待需求。后端按鉴权身份限制范围。",
            "medium",
            "state_write",
            {
                "type": "object",
                "required": ["reason"],
                "additionalProperties": False,
                "properties": {
                    "demand_id": {"type": "string"},
                    "reason": non_empty_string,
                },
            },
            partial(cancel_waiting_demand, store),
        ),
        "create_game": ToolDefinition(
            "create_game",
            "创建待组局记录。只落库，不发消息、不确认房间。固定时间且距离开局超过招募提前量时，后端会持久化定时任务；局立即进入列表，但暂不私聊候选人。模型必须显式提供 organizer_id 和 organizer_name，后端不从当前消息脑补组织者。",
            "medium",
            "state_write",
            {
                "type": "object",
                "required": ["requirement", "organizer_id", "organizer_name"],
                "additionalProperties": False,
                "properties": {
                    "requirement": requirement_schema,
                    "organizer_id": non_empty_string,
                    "organizer_name": non_empty_string,
                    "known_players": {"type": "array", "items": known_player_schema},
                    "requesting_party": requesting_party_schema,
                },
            },
            partial(create_game, store),
        ),
        "join_game": ToolDefinition(
            "join_game",
            "把当前已鉴权客户加入指定局。仅用于客户明确接受/确认参加；后端原子校验容量、跨局冲突和状态机，并写入独立参与者表。拒绝、协商、未回复仍使用 record_candidate_reply。",
            "medium",
            "state_write",
            {
                "type": "object",
                "required": ["game_id", "customer_id", "display_name"],
                "additionalProperties": False,
                "properties": {
                    "game_id": non_empty_string,
                    "customer_id": non_empty_string,
                    "display_name": non_empty_string,
                    "seat_count": {"type": "integer", "minimum": 1, "maximum": 4},
                },
            },
            partial(join_game, store),
        ),
        "create_invite_drafts": ToolDefinition(
            "create_invite_drafts",
            "创建待审批邀约草稿。只生成草稿，不代表已发送。未来局在 recruitment_opens_at 之前会被统一时间策略拒绝；不要通过改写话术绕过。",
            "medium",
            "draft_write",
            {
                "type": "object",
                "required": ["game_id", "invitations"],
                "additionalProperties": False,
                "properties": {
                    "game_id": non_empty_string,
                    "invitations": {"type": "array", "items": invitation_schema, "minItems": 1},
                },
            },
            partial(create_invite_drafts, store),
        ),
        "update_game_requirement": ToolDefinition(
            "update_game_requirement",
            "更新尚未成局的组局条件。仅用于客户明确补充或协商确认后的时间、时长、玩法、档位、烟况等条件；不能修改参与者、座位快照或生命周期计算字段。",
            "medium",
            "state_write",
            {
                "type": "object",
                "required": ["game_id", "requirement_patch", "reason"],
                "additionalProperties": False,
                "properties": {
                    "game_id": non_empty_string,
                    "requirement_patch": requirement_schema,
                    "reason": non_empty_string,
                },
            },
            partial(update_game_requirement, store),
        ),
        "create_outbound_message_drafts": ToolDefinition(
            "create_outbound_message_drafts",
            "创建通道无关的待审批外发消息草稿。只落库，不代表已发送，可用于当前用户回复、群消息或其他渠道输出。",
            "medium",
            "draft_write",
            {
                "type": "object",
                "required": ["drafts"],
                "additionalProperties": False,
                "properties": {
                    "drafts": {"type": "array", "items": outbound_message_draft_schema, "minItems": 1},
                },
            },
            partial(create_outbound_message_drafts, store),
        ),
        "record_candidate_reply": ToolDefinition(
            "record_candidate_reply",
            "记录某个局里客户/候选人本轮发生的参与状态或代表座位数变化，并推进受控状态。适用于已邀约候选人，也适用于当前已在局内的客户。status 可表示 accepted/confirmed/arrived/declined/negotiating/no_reply；客户拒绝、退出、不打了或条件不接受时也要调用，通常用 declined。记忆写入不能代替本工具。若 active_games 中该客户已经是相同状态且座位数没有变化，不要重复调用。若客户表示“我这边两个人/我们3个”，模型必须把代表座位数写入 seat_count。",
            "medium",
            "state_write",
            {
                "type": "object",
                "required": ["game_id", "customer_id", "display_name", "status"],
                "additionalProperties": False,
                "properties": {
                    "game_id": non_empty_string,
                    "customer_id": non_empty_string,
                    "display_name": non_empty_string,
                    "status": {"type": "string", "enum": CANDIDATE_REPLY_STATUSES},
                    "seat_count": {"type": "integer", "minimum": 1, "maximum": 4},
                },
            },
            partial(record_candidate_reply, store),
        ),
        "update_game_status": ToolDefinition(
            "update_game_status",
            "只按状态机更新局的生命周期状态。非法状态迁移由后端拒绝；本工具不能修改时长、烟况、档位、时间或人数等 requirement，不能为了记录用户约束而调用。",
            "medium",
            "state_write",
            {
                "type": "object",
                "required": ["game_id", "status", "reason"],
                "additionalProperties": False,
                "properties": {
                    "game_id": non_empty_string,
                    "status": {"type": "string", "enum": GAME_STATUSES},
                    "reason": non_empty_string,
                },
            },
            partial(update_game_status, store),
        ),
        "record_badcase": ToolDefinition(
            "record_badcase",
            "记录 badcase/eval 候选样本，不改变业务状态。",
            "low",
            "audit_write",
            badcase_schema,
            partial(record_badcase, store),
        ),
        "record_user_memory": ToolDefinition(
            "record_user_memory",
            "记录用户表达的当前任务约束和待确认长期画像候选。当前任务约束会立即影响查现有局和找候选人；长期画像候选只进入待审核队列，不直接改客户画像，也不代替当前局状态写入。",
            "medium",
            "state_write",
            memory_schema,
            partial(record_user_memory, store),
        ),
        "update_context_checkpoint": ToolDefinition(
            "update_context_checkpoint",
            "更新当前会话的长期上下文 checkpoint。模型负责总结需要跨窗口保留的事实、待确认问题和当前任务状态；工具只校验并存储。",
            "medium",
            "state_write",
            checkpoint_schema,
            partial(update_context_checkpoint, store),
        ),
    }
