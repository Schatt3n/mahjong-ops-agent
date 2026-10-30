"""声明工具参数的结构协议；类型校验不替模型选择客户玩法或查询范围。"""

from __future__ import annotations

stake_range_schema = {
    "type": "object",
    "required": ["min", "max"],
    "additionalProperties": False,
    "properties": {"min": {"type": "number", "exclusiveMinimum": 0}, "max": {"type": "number", "exclusiveMinimum": 0}},
    "description": "底注闭区间，0<min<=max；不表示封顶。",
}
requirement_schema = {
    "type": "object", "additionalProperties": True,
    "properties": {
        "game_type": {
            "type": "string",
            "description": "开放玩法标识。个人匹配传可靠默认；盘点按显式范围。",
        },
        "stake_range": stake_range_schema,
    },
}
non_empty_string = {"type": "string", "minLength": 1}
known_player_schema = {
    "type": "object",
    "required": ["customer_id", "display_name"],
    "additionalProperties": True,
    "properties": {
        "customer_id": non_empty_string,
        "display_name": non_empty_string,
        "status": {"type": "string", "enum": ["joined", "confirmed"]},
        "source": {"type": "string"},
        "seat_count": {"type": "integer", "minimum": 1, "maximum": 4},
        "known_member_ids": {"type": "array", "items": {"type": "string"}},
        "anonymous_seat_count": {"type": "integer", "minimum": 0, "maximum": 4},
    },
}
requesting_party_schema = {
    "type": "object",
    "required": ["contact_id", "contact_name", "seat_count"],
    "additionalProperties": True,
    "properties": {
        "contact_id": non_empty_string,
        "contact_name": non_empty_string,
        "seat_count": {"type": "integer", "minimum": 1, "maximum": 4},
        "known_member_ids": {"type": "array", "items": {"type": "string"}},
        "anonymous_seat_count": {"type": "integer", "minimum": 0, "maximum": 4},
        "source": {"type": "string"},
    },
}
# 查询仅表达容量约束，不要求取得建局所需的联系人资料；字段层级仍必须明确，
# 避免模型提供的人数被查询处理器静默忽略。未知人数可以省略，不能由桌况猜测。
query_requirement_schema = {
    **requirement_schema,
    "description": "本次检索实际采用的条件；已知且仍适用的条件须逐项写入参数，只写在目标或 known_facts 中不构成过滤。未知条件可省略，盘点不自动补个人条件。",
    "properties": {
        **requirement_schema["properties"],
        "stake": {
            "description": "固定底注，传数字或数值字符串；明确固定值使用 stake，不改写为上下界相同的 stake_range。",
        },
        # 显式声明检索已支持的 canonical 字段，避免模型只能在附加字段中猜参数。
        # 不收紧既有规范化器接受的标量/多值形式，也不把排序条件宣传为硬过滤。
        "smoke_preference": {
            "description": "烟况适配条件，首选 no_smoking（无烟）、smoking（有烟）、any（明确不限）；也保留既有烟况表达及多值。已知无烟或有烟须实际传入，省略不会自动应用画像；冲突桌只能按真实画像支持作为待确认备选，不能当严格匹配。",
        },
        "cap_score": {
            "description": "独立封顶，传数字或数值字符串；与 stake 或 stake_range 分开。已知封顶冲突会排除，目标封顶未知不能当作已证实相符；不把封顶数字当底注区间上界。",
        },
        "duration_kind": {
            "description": "时长类别，如 fixed 或 overnight，也保留既有多值表达；现有检索以此排序，不保证排除不同类别或验证精确小时数。未知或不限不补值。",
        },
        "start_time_kind": {
            "type": "string",
            "description": "开场方式；客户选定固定时刻才用 scheduled，明确人齐开才用 asap_when_full；宽时段、现在或时间不限不自动选定开场方式。",
        },
        "start_time": {
            "type": "string", "minLength": 1,
            "description": "客户选定开场的日内钟点优先用 HH:MM，不能从消息时间或宽时段起点编造。来源已确认日期/时区时须同时提交 planned_start_at，单独钟点不能保留日历约束。",
        },
        "planned_start_at": {
            "type": "string", "minLength": 1,
            "description": "已确认固定时刻的完整带时区ISO日期时间；从同来源的当前事实保留日期、钟点和时区并实际传入查询，不只写在模型known_facts中。日期未知不猜，盘点或时间不限可省略。",
        },
        "requesting_party": {
            "type": "object",
            "additionalProperties": False,
            "properties": dict(requesting_party_schema["properties"]),
            "description": "个人匹配的己方人数放在 requirement.requesting_party.seat_count；未知可省略，不是目标桌已有人数。",
        },
    },
}
invitation_schema = {
    "type": "object",
    "required": ["customer_id", "display_name", "message_text"],
    "additionalProperties": True,
    "properties": {
        "customer_id": non_empty_string,
        "display_name": non_empty_string,
        "message_text": non_empty_string,
        "metadata": {"type": "object", "additionalProperties": True},
    },
}
outbound_message_draft_schema = {
    "type": "object",
    "required": ["recipient_id", "recipient_name", "channel", "message_text", "purpose"],
    "additionalProperties": False,
    "properties": {
        "recipient_id": non_empty_string,
        "recipient_name": non_empty_string,
        "channel": non_empty_string,
        "message_text": non_empty_string,
        "purpose": non_empty_string,
        "metadata": {"type": "object", "additionalProperties": True},
    },
}
checkpoint_schema = {
    "type": "object",
    "required": ["summary"],
    "additionalProperties": False,
    "properties": {
        "summary": non_empty_string,
        "facts": {"type": "object", "additionalProperties": True},
        "open_questions": {"type": "array", "items": non_empty_string},
    },
}
badcase_schema = {
    "type": "object",
    "required": ["reason", "input", "actual", "expected"],
    "additionalProperties": True,
    "properties": {
        "reason": non_empty_string,
        "input": {"type": "object", "additionalProperties": True},
        "actual": {"type": "object", "additionalProperties": True},
        "expected": {"type": "object", "additionalProperties": True},
        "tags": {"type": "array", "items": non_empty_string},
        "metadata": {"type": "object", "additionalProperties": True},
    },
}
memory_item_schema = {
    "type": "object",
    "required": ["customer_id", "memory_type", "field", "value", "evidence", "confidence"],
    "additionalProperties": True,
    "properties": {
        "customer_id": non_empty_string,
        "memory_type": non_empty_string,
        "field": {
            **non_empty_string,
            "description": (
                "稳定的结构化字段名。时长上限使用 max_duration_hours，明确约定时长使用 duration_hours；"
                "避免同一语义在不同轮次使用不同字段名。"
            ),
        },
        "value": {},
        "target_customer_id": {"type": "string"},
        "target_customer_name": {"type": "string"},
        "operation": {"type": "string"},
        "evidence": non_empty_string,
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "scope": {"type": "string", "enum": ["current_task", "session", "today", "long_term"]},
        "metadata": {"type": "object", "additionalProperties": True},
    },
}
memory_schema = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "task_memories": {"type": "array", "items": memory_item_schema},
        "pending_long_term_memories": {"type": "array", "items": memory_item_schema},
    },
}
