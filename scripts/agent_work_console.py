"""把现有入站观察和执行 Trace 投影为只读工作记录。"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from mahjong_agent_runtime.group_chat.projections import public_game_start_display, public_group_game_summary
from mahjong_agent_runtime.tracing import TraceEvent


TRACE_LIMIT = 4 * 1024 * 1024
RECORD_LIMIT = 50
MODEL_STAGES = {
    "llm": "主模型", "wechaty_input_gate": "入口判断", "wechaty_casual_chat": "闲聊回复",
    "customer_visible_text_generation": "文案生成", "customer_visible_content_review": "回复审查",
    "group_semantic_routing": "群消息分流", "context_summary": "上下文摘要",
}
SECRET_FIELD = re.compile(r"(?i)^(?:authorization|cookie|api[_-]?key|(?:access[_-]?)?token|password|secret)$")
SENSITIVE = re.compile(
    r"(?i)\b(?:sk-[a-z0-9_-]{16,}|bearer\s+[a-z0-9._~-]{16,}|"
    r"(?:api[_ -]?key|access[_ -]?token)\s*[:=]\s*\S{8,})"
)
SIMPLE = {
    "wechaty_raw_message_received": ("收到微信消息", "working"),
    "agent_wechat_ingress_processing_started": ("后台开始处理微信消息", "working"),
    "console_message_received": ("收到控制台消息", "working"),
    "wechaty_input_gate_prompt": ("入口正在判断消息", "working"),
    "wechaty_raw_message_routed_to_agent": ("进入主 Agent", "working"),
    "input_batch_routed_to_agent": ("输入批次进入主 Agent", "working"),
    "wechaty_raw_message_routed_to_casual_chat": ("转入闲聊回复", "working"),
    "input_batch_routed_to_casual_chat": ("转入闲聊回复", "working"),
    "input_batch_waiting": ("等待更多输入", "working"),
    "input_batch_background_queued": ("已进入后台队列", "working"),
    "input_batch_claimed": ("后台已领取输入批次", "working"),
    "wechaty_casual_chat_prompt": ("闲聊模型正在生成回复", "working"),
    "group_semantic_routing_started": ("开始群消息分流", "working"),
    "group_semantic_routing_completed": ("群消息分流完成", "neutral"),
    "budget_checked": ("检查模型调用预算", "working"),
}
SKIP_STEPS = {
    "wechaty_raw_message_not_routed", "wechaty_managed_group_message_ignored",
    "input_batch_ignored", "wechaty_observe_only_message_analyzed",
    "agent_wechat_group_permission_observed", "agent_wechat_group_topic_observation_only",
    "agent_wechat_private_contact_observed", "agent_wechat_self_message_observed",
}
DELIVERY_REASONS = {
    "wechaty_outbound_switch_disabled": "发送总闸或自动外发已关闭",
    "send_channel_disabled": "微信发送总闸已关闭",
    "auto_send_disabled": "自动外发已关闭",
    "managed_room_outbound_disabled": "该群未允许自动外发",
    "origin_managed_room_outbound_disabled": "来源群未允许自动外发",
    "managed_room_board_disabled": "该群未允许发送看板",
    "wechaty_outbound_request_failed": "桥接请求没有返回可确认的结果",
    "delayed_room_send_not_supported_by_current_bridge": "当前桥接未启用此延迟群回复",
    "no_customer_visible_reply": "本次没有对外回复",
    "duplicate_of_immediate_ack": "与即时确认重复，已跳过",
}
NATIVE_DELIVERY = {
    "accepted": ("confirmed", "微信通道已确认"),
    "already_delivered": ("confirmed", "此前已确认发送"),
    "needs_review": ("unconfirmed", "发送结果待核对"),
    "failed": ("blocked", "微信发送失败"),
    "withheld": ("blocked", "外发已拦截"),
    "cancelled": ("blocked", "发送已取消"),
    "owner_already_replied": ("blocked", "老板已回复，跳过自动外发"),
}


def _text(value, limit=300):
    """限制可见文本长度并遮盖常见认证字段。"""
    return SENSITIVE.sub("[已隐藏凭证]", value.strip())[:limit] if isinstance(value, str) else ""


def _mapping(value):
    """将旧 Trace 中可缺省的对象约束为字典。"""
    return value if isinstance(value, dict) else {}


def _model_body(value):
    """只为鉴权后的选中详情脱敏模型正文，保留完整消息与 JSON 结构。"""
    if isinstance(value, dict):
        return {key: "[已隐藏凭证]" if SECRET_FIELD.fullmatch(key) else _model_body(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_model_body(item) for item in value]
    if isinstance(value, str):
        try:
            structured = json.loads(value)
        except (ValueError, TypeError):
            return SENSITIVE.sub("[已隐藏凭证]", value)
        if isinstance(structured, (dict, list)):
            return json.dumps(_model_body(structured), ensure_ascii=False, indent=2)
        return SENSITIVE.sub("[已隐藏凭证]", value)
    return value


def _model_io(event):
    """仅提取已记录的提示词和返回正文，不把其他 Trace 字段当成模型内容。"""
    prefix, _, suffix = event.step.rpartition("_")
    if prefix not in MODEL_STAGES or suffix not in {"prompt", "response"}:
        return None
    data = event.content
    value = data.get("messages") if suffix == "prompt" else data.get("content", data.get("raw_response"))
    if value is None:
        return None
    body = _model_body(value)
    return {
        "kind": "request" if suffix == "prompt" else "response",
        "stage": MODEL_STAGES[prefix],
        "body": body if isinstance(body, str) else json.dumps(body, ensure_ascii=False, indent=2),
        "step_index": data.get("step_index"), "elapsed_ms": data.get("elapsed_ms"),
        "call_id": _text(_mapping(data.get("completion_diagnostics")).get("call_id"), 100),
    }


def _recent_events(recorder):
    """只读取日志末尾，跳过半行与正在写入的末行，轮询成本不随历史增长。"""
    path = getattr(recorder, "path", None)
    if path is None:
        return list(recorder.events[-4000:])
    try:
        path = Path(path)
        current = path.stat()
        return _read_trace_tail(path, current.st_size, current.st_mtime_ns, current.st_ino)
    except FileNotFoundError:
        return []


@lru_cache(maxsize=3)
def _read_trace_tail(path, size, _modified, _identity):
    """等待同一次模型调用时允许重用日志尾部，追加或轮换后重新解析。"""
    with path.open("rb") as handle:
        handle.seek(max(0, size - TRACE_LIMIT))
        data = handle.read(min(size, TRACE_LIMIT))
    lines = data.split(b"\n")
    return [
        event for raw in lines[int(size > TRACE_LIMIT):-1]
        if (event := TraceEvent.from_log_line(raw.rstrip(b"\r").decode("utf-8", errors="replace"))) is not None
    ]


def _trace_index(events):
    """按精确批次和来源 Trace 连接异步运行，避免按昵称、时间猜测。"""
    traces, batch_runs, linked = defaultdict(list), defaultdict(set), defaultdict(set)
    for sequence, event in enumerate(events):
        traces[event.trace_id].append((sequence, event))
        if event.step in {"input_background_dispatch_due", "input_quiet_period_elapsed", "input_batch_claimed"}:
            data = event.content
            batch_id = data.get("batch_id") or _mapping(data.get("batch")).get("batch_id")
            if isinstance(batch_id, str) and batch_id:
                batch_runs[batch_id].add(event.trace_id)
            origin = data.get("origin_trace_id")
            if isinstance(origin, str) and origin:
                linked[origin].add(event.trace_id)
    return traces, batch_runs, linked


def _related_events(record, traces, batch_runs, linked):
    """从入站记录或本条 Trace 中获取真实批次标识，再按事件写入顺序合并。"""
    trace_id = record["trace_id"]
    batch = _mapping(_mapping(_mapping(record.get("payload")).get("route_result")).get("input_batch"))
    batch_id = batch.get("batch_id")
    for _, event in traces[trace_id]:
        if event.step in {"input_batch_waiting", "input_batch_background_queued", "input_batch_claimed"}:
            batch_id = (_mapping(event.content.get("batch") or event.content.get("input_batch")).get("batch_id")
                        or batch_id)
    related_ids = {trace_id} | linked[trace_id] | batch_runs.get(batch_id, set())
    merged = sorted(entry for identity in related_ids for entry in traces[identity])
    return [event for _, event in merged], batch_id


def delivery_observation(payload):
    """桥接仅可报告受限的发送阶段和回执 ID，禁止写入任意 Trace 内容。"""
    if not isinstance(payload, dict):
        raise ValueError("invalid delivery observation")
    outcome = payload.get("outcome")
    kind = payload.get("kind")
    trace_id = payload.get("trace_id")
    delivery_id = payload.get("delivery_id")
    if (not isinstance(trace_id, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", trace_id)
            or not isinstance(delivery_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", delivery_id)
            or outcome not in ("sending", "confirmed", "blocked", "unconfirmed")
            or kind not in ("ack", "reply")):
        raise ValueError("invalid delivery observation")
    message_id = _text(payload.get("message_id"), 160)
    if outcome == "confirmed" and not message_id:
        raise ValueError("missing platform receipt")
    reason = payload.get("reason")
    return trace_id, {
        "outcome": outcome, "kind": kind, "delivery_id": delivery_id, "message_id": message_id,
        "reason": reason if isinstance(reason, str) and reason in DELIVERY_REASONS else "",
    }


def _delivery(step, data, source_message_id):
    """按业务角色区分本条回复和其他通知，保留准确的发送尝试标识。"""
    kind = "reply"
    metadata = _mapping(data.get("metadata"))
    if step.startswith("managed_group_outbound_"):
        message_type = metadata.get("message_type")
        historical_reply = not data.get("delivery_id") and not message_type and metadata.get("source_message_id") == source_message_id
        if message_type != "agent_reply" and not historical_reply:
            kind = "notification"
    outcome, reason, message_id = "", data.get("reason"), ""
    if step == "wechaty_bridge_delivery":
        kind, outcome, message_id = data["kind"], data["outcome"], data["message_id"]
    elif step in {"agent_wechat_ingress_reply_started", "agent_wechat_ingress_reply_retry_scheduled",
                  "agent_wechat_ingress_reply_completed"}:
        if step == "agent_wechat_ingress_reply_completed":
            outcome, label = NATIVE_DELIVERY.get(data.get("delivery_state"), ("", ""))
        else:
            outcome, label = "sending", ("等待重试交付" if step.endswith("_scheduled") else "正提交微信发送")
        if not outcome:
            return None
        return {
            "kind": "reply", "label": label, "tone": {"confirmed": "success", "sending": "working"}.get(outcome, "warning"),
            "status": outcome, "delivery_id": "native-ingress-reply", "target_type": "", "text": _text(data.get("text"), 1500),
            "message_id": _text(data.get("message_id"), 160), "detail": "",
        }
    elif step in {"delayed_reply_delivery_started", "managed_group_outbound_started"}:
        outcome = "sending"
    elif step in {"delayed_reply_delivered", "managed_group_outbound_delivered", "recovered_reply_delivered"}:
        message_id = data.get("platform_message_id", data.get("message_id") or _mapping(data.get("response")).get("message_id"))
        fallback_ref = metadata.get("source_message_id")
        if not isinstance(message_id, str) or message_id.startswith("wechaty_out_") or message_id == fallback_ref:
            message_id = ""
        outcome = "confirmed" if message_id else "unconfirmed"
    elif step == "api_reply_delivered":
        outcome = "local"
    elif step in {"delayed_reply_delivery_failed", "managed_group_outbound_suppressed"}:
        outcome = "unconfirmed" if step.endswith("_failed") or reason == "wechaty_outbound_request_failed" else "blocked"
    elif step in {"delayed_reply_delivery_skipped", "recovered_reply_delivery_skipped"}:
        if reason in {"no_customer_visible_reply", "duplicate_of_immediate_ack", "source_channel_has_no_delayed_sender"}:
            return None
        outcome = "blocked"
    if not outcome:
        return None
    label, tone = {
        "sending": ("正提交微信发送", "working"), "confirmed": ("微信通道已确认", "success"),
        "blocked": ("外发已拦截", "warning"), "unconfirmed": ("发送结果未确认", "warning"),
        "local": ("已返回控制台", "success"),
    }[outcome]
    prefix = {"ack": "即时确认 · ", "notification": "其他外发 · "}.get(kind, "")
    return {
        "kind": kind, "label": prefix + label, "tone": tone, "status": outcome,
        "delivery_id": _text(data.get("delivery_id"), 80),
        "target_type": {"room": "群聊", "contact": "私聊"}.get(data.get("target_type"), ""),
        "text": _text(data.get("text"), 500), "message_id": _text(message_id, 160),
        "detail": DELIVERY_REASONS.get(reason, ""),
    }


def _model_attempt(step, data, stage=""):
    """展示实际 HTTP 尝试和退避阶段，不读取请求或响应正文。"""
    attempt = data.get("attempt")
    detail = " · ".join(filter(None, [
        _text(data.get("model"), 80), f"第 {attempt} 次尝试" if type(attempt) is int else "",
    ]))
    if step == "llm_http_retry_scheduled":
        error = _text(data.get("error_type"), 80)
        status = data.get("http_status")
        label = "模型请求失败，等待重试"
        if stage:
            label = f"{stage} · {label}"
        return label, "working", " · ".join(filter(None, [
            detail, error, f"HTTP {status}" if type(status) is int else "",
        ]))
    label = "等待模型 API 返回"
    if stage:
        label = f"{stage} · {label}"
    return label, "working", detail


def _action_details(data):
    """保留结构化提案和简短依据，对完整提示词和模型原始输出保持隔离。"""
    goal = _text(data.get("goal"))
    summary = _text(data.get("reasoning_summary"))
    calls = data.get("tool_calls")
    tools = [_text(_mapping(item).get("name"), 65) for item in calls[:8]] if isinstance(calls, list) else []
    return " · ".join(filter(None, [goal, summary, "拟调用 " + "、".join(filter(None, tools)) if any(tools) else ""]))


def _tool_details(data):
    """将工具证据缩减成允许列表字段，不返回其他客户记录和原始参数。"""
    result = _mapping(data.get("result"))
    fields = []
    for key in ("count", "total", "remaining_seats", "available_seats", "current_players", "target_players"):
        value = result.get(key)
        if type(value) is int:
            fields.append(f"{key}={value}")
    for key in ("games", "items", "matches", "alternatives"):
        if isinstance(result.get(key), list):
            fields.append(f"{key} {len(result[key])} 项")
    return " · ".join(fields)


def _timeline_event(event, model_stage=""):
    """投影白名单内的事件，模型开始、返回与工具执行各自保留证据。"""
    step, data = event.step, event.content
    if step.startswith("agent-wechat_"):
        equivalent = "wechaty_" + step.removeprefix("agent-wechat_")
        if equivalent in SIMPLE or equivalent in SKIP_STEPS:
            step = equivalent
    if step in SIMPLE:
        label, tone = SIMPLE[step]
        return label, tone, ""
    if step in SKIP_STEPS:
        return ("仅观察" if step.endswith(("_observation_only", "_observed", "_message_analyzed")) else "入口已跳过",
                "neutral", _text(data.get("reason"), 110))
    if step == "llm_request_started":
        return model_stage or "主模型正在处理", "working", ""
    if step in {"llm_http_attempt_started", "llm_http_retry_scheduled"}:
        return _model_attempt(step, data, model_stage)
    if step == "llm_response":
        elapsed = data.get("elapsed_ms")
        return "模型已返回", "working", f"{elapsed} 毫秒" if type(elapsed) is int else ""
    if step in {"llm_error", "wechaty_casual_chat_error", "group_semantic_routing_error", "input_batch_dispatch_failed", "input_quiet_period_handler_error"}:
        return "模型调用失败" if "batch" not in step and "handler" not in step else "后台处理失败", "error", _text(data.get("error_type"), 80)
    if step == "action_proposed":
        return "模型提案", "working", _action_details(data)
    if step == "tool_called":
        name = _text(_mapping(data.get("call")).get("name"), 90)
        return ("开始查局" if name == "search_current_games" else "正在调用工具"), "working", name
    if step in {"customer_visible_text_generation_prompt", "customer_visible_content_review_prompt"}:
        label = "生成候选回复" if "generation" in step else "正在审查对外回复"
        return label, "working", ""
    if step in {"customer_visible_text_generation_response", "customer_visible_content_review_response"}:
        label = "文案模型已返回" if "generation" in step else "审查模型已返回"
        return label, "working", ""
    if step == "tool_result":
        name = _text(data.get("name"), 90)
        failed = bool(data.get("error")) or data.get("allowed") is False
        evidence = _tool_details(data)
        if name == "search_current_games" and not failed:
            result = _mapping(data.get("result"))
            parts = [f"匹配 {len(result[key])} 局" if key == "matches" else f"备选 {len(result[key])} 局"
                     for key in ("matches", "alternatives") if isinstance(result.get(key), list)]
            return "查局完成", "working", " · ".join(parts) or evidence
        return ("工具未通过" if failed else "工具返回", "warning" if failed else "working",
                " · ".join(filter(None, [name, evidence])))
    if step == "customer_visible_immediate_ack":
        return "即时确认已生成", "working", ""
    if step == "final_output":
        text = _text(data.get("reply"))
        return ("业务回复已生成", "neutral", "") if text else ("未生成业务回复", "neutral", _text(data.get("reason"), 100))
    return None


_MODEL_STAGE_START = {
    "group_semantic_routing_started": "群消息分流",
    "wechaty_input_gate_prompt": "入口判断",
    "wechaty_casual_chat_prompt": "闲聊回复",
    "customer_visible_text_generation_prompt": "生成候选回复",
    "customer_visible_content_review_prompt": "回复审查",
}
_MODEL_STAGE_END = {
    "group_semantic_routing_completed", "group_semantic_routing_error", "llm_response", "llm_error",
    "customer_visible_text_generation_response", "customer_visible_content_review_response",
    "wechaty_input_gate_response", "wechaty_input_gate_error", "wechaty_casual_chat_response", "wechaty_casual_chat_error",
}


def _managed_group_outcome(result):
    """群入口可以直接完成业务，任何已完成的路由都应关闭此前的模型等待。"""
    action = result.get("action")
    game_id = _text(result.get("game_id"), 100)
    if action == "register_self_recruitment":
        if game_id:
            return "已静默登记", "success", "已登记到牌局看板。本次没有群回复。"
        return "未登记牌局", "neutral", "未形成可登记的牌局，不会发群回复。"
    if action == "ignore":
        return "群聊入口已跳过", "neutral", ""
    if action == "private_switch":
        return "群聊处理完成 · 转私聊", "neutral", ""
    return "群聊处理完成", "neutral", ""


def _record_snapshot(record, events, batch_id, *, details, include_model_io=False):
    """汇总当前阶段和真实回复，历史缺少发送证据时保持未确认。"""
    title = record.get("room_topic") if record.get("is_room") else record.get("sender_name")
    is_group = bool(record.get("is_room"))
    phase, tone = ("收到消息", "working")
    fallbacks = {
        "route_failed": ("入口处理失败", "error"), "queued": ("后台排队", "working"),
        "analyzed": ("入口已处理", "neutral"), "processed": ("处理完成", "neutral"),
    }
    if record.get("route_mode") == "observe_only":
        fallbacks["analyzed"] = ("仅观察", "neutral")
    if record.get("route_mode") == "ingress_queue":
        phase = "后台排队"
    timeline, deliveries, attempts = [], {}, {}
    reply, draft, ack = "", "", ""
    game_id = ""
    group_outcome = None
    call_started_at = None
    had_error = False
    system_runs = defaultdict(int)
    # 老的托管群日志没有分流开始事件。主 Agent 开始前的模型尝试属于群分流，避免重试次数看起来被重置。
    model_stages = {record["trace_id"]: "群消息分流"} if record.get("route_mode") == "managed_group" else {}
    queried_traces = set()
    for sequence, event in enumerate(events):
        data, step = event.content, event.step
        if step in {"system_event_received", "system_trigger_received"}:
            system_runs[event.trace_id] += 1
        elif step in {"system_event_completed", "system_trigger_completed"}:
            system_runs[event.trace_id] = max(system_runs[event.trace_id] - 1, 0)
        delivery = _delivery(step, data, record["source_message_id"])
        if system_runs[event.trace_id] and delivery is None:
            # 群满员通知等子运行可共用 Trace，但其输出不能替代入站消息的正文或进度。
            continue
        if step in _MODEL_STAGE_START:
            model_stages[event.trace_id] = _MODEL_STAGE_START[step]
        elif step == "llm_request_started":
            model_stages[event.trace_id] = "处理查局结果" if event.trace_id in queried_traces else "主模型决策"
        elif step == "tool_result" and data.get("name") == "search_current_games" and not data.get("error") and data.get("allowed") is not False:
            queried_traces.add(event.trace_id)
        projected = _timeline_event(event, model_stages.get(event.trace_id, ""))
        model_io = _model_io(event) if details and include_model_io else None
        if model_io:
            label = model_io["stage"] + (" · 模型输入" if model_io["kind"] == "request" else " · 模型返回")
            projected = (label, "working", f"{model_io['elapsed_ms']} 毫秒" if type(model_io["elapsed_ms"]) is int else "")
        if step in _MODEL_STAGE_END:
            model_stages.pop(event.trace_id, None)
        if step == "customer_visible_immediate_ack":
            ack = _text(data.get("reply"), 500)
        if step == "action_proposed" and data.get("reply_to_user"):
            draft = _text(data["reply_to_user"], 1500)
        if step == "final_output":
            reply = _text(data.get("reply"), 1500)
        if step in {"wechaty_managed_group_message_processed", "agent-wechat_managed_group_message_processed"}:
            group_outcome = _mapping(data.get("result"))
            game_id = _text(group_outcome.get("game_id"), 100)
            if not reply:
                reply = _text(group_outcome.get("reply"), 1500)
            projected = _managed_group_outcome(group_outcome)
        if delivery:
            identifier = delivery["delivery_id"]
            # 老日志没有尝试 ID，单独列为历史事件，避免按发送内容或相邻顺序猜测配对。
            attempts[(event.trace_id, identifier or f"history-{sequence}")] = delivery
            if delivery["kind"] != "notification":
                deliveries[delivery["kind"]] = delivery
            if delivery["kind"] == "reply" and delivery["text"] and not reply:
                reply = delivery["text"]
            projected = delivery["label"], delivery["tone"], delivery["detail"] or delivery["message_id"]
        if not projected:
            continue
        label, event_tone, detail = projected
        entry = {"label": label, "tone": event_tone, "detail": detail, "time": event.occurred_at}
        if details:
            entry.update(id=f"{event.trace_id}:{sequence}", step=step, trace_id=event.trace_id)
            if model_io:
                entry["model_io"] = model_io
        timeline.append(entry)
        if delivery and delivery["kind"] != "reply":
            continue
        if event_tone == "error":
            had_error = True
        if step == "final_output" and had_error:
            continue
        phase, tone = label, event_tone
        call_started_at = event.occurred_at if step in {
            "llm_request_started", "llm_http_attempt_started", "wechaty_input_gate_prompt", "wechaty_casual_chat_prompt",
            "customer_visible_text_generation_prompt", "customer_visible_content_review_prompt", "group_semantic_routing_started",
        } else None
    if record["channel"] == "agent-wechat":
        payload = _mapping(record.get("payload"))
        route = _mapping(payload.get("route_result"))
        saved = _mapping(payload.get("deferred_reply"))
        outgoing_text = (_mapping(route.get("scoped_group_reply")).get("text") if is_group
                         else _mapping(route.get("agent_result")).get("final_reply"))
        if not reply:
            reply = _text(outgoing_text, 1500)
        if saved:
            reply = _text(saved.get("text") or outgoing_text, 1500)
        delivery = _delivery("agent_wechat_ingress_reply_completed", {**saved, "text": reply},
                             record["source_message_id"])
        if delivery:
            attempts[(record["trace_id"], delivery["delivery_id"])] = delivery
            deliveries["reply"] = delivery
        elif record.get("route_mode") == "ingress_reply_pending" and not deliveries.get("reply"):
            phase, tone = "等待交付微信回复", "working"
            call_started_at = None
    terminal = deliveries.get("reply")
    if group_outcome is None and record.get("route_mode") == "managed_group" and record.get("route_status") == "processed":
        # 处理完的群消息已有持久结果，即使对应日志超出时间线窗口，也不能继续显示等待。
        group_outcome = _mapping(_mapping(_mapping(record.get("payload")).get("route_result")).get("group_result"))
        game_id = _text(group_outcome.get("game_id"), 100)
        phase, tone, _ = _managed_group_outcome(group_outcome)
        call_started_at = None
    if terminal and terminal["status"] in {"confirmed", "blocked", "unconfirmed", "local"}:
        phase, tone = terminal["label"], terminal["tone"]
        call_started_at = None
    elif record.get("route_status") == "route_failed" and not had_error:
        phase, tone = fallbacks["route_failed"]
        call_started_at = None
    elif record.get("route_mode") == "ingress_queue" and phase == "收到微信消息":
        phase, tone = fallbacks["queued"]
    elif not timeline or (phase == "收到微信消息" and record.get("route_status") in fallbacks):
        phase, tone = fallbacks.get(record.get("route_status"), (phase, tone))
    result = {
        "key": record["channel"] + ":" + (record["trace_id"] if record["channel"] == "api" else record["source_message_id"]),
        "source_message_id": record["source_message_id"], "trace_id": record["trace_id"],
        "channel": record["channel"], "kind": "群聊" if is_group else "控制台" if record["channel"] == "api" else "微信私聊",
        "title": _text(title, 100) or ("微信群" if is_group else "未知发送者"),
        "sender": _text(record.get("sender_name"), 100), "time": record.get("received_at"),
        "text": _text(record.get("text"), 1500 if details else 110),
        "phase": phase, "tone": tone, "call_started_at": call_started_at,
    }
    if details:
        result.update({
            "batch_id": batch_id, "timeline": timeline[-100:], "reply": reply, "draft": draft,
            "ack": ack, "deliveries": deliveries, "delivery_attempts": list(attempts.values())[-100:],
            "route_mode": record.get("route_mode"), "game_id": game_id,
            "trace_ids": list(dict.fromkeys([record["trace_id"], *(event.trace_id for event in events)])),
        })
    return result


def _roster_name(customers, channel_name, customer_id, registered_name=""):
    """展示公开昵称或已登记称呼，缺少信息时不把微信标识和私有备注当作名字。"""
    profile = customers.get(customer_id)
    hidden_names = {_text(customer_id, 100), _text(getattr(profile, "private_remark", ""), 100)}
    for candidate in (getattr(profile, "public_name", ""), registered_name, getattr(profile, "display_name", "")):
        name = _text(candidate, 100)
        if name and name not in hidden_names and not name.casefold().startswith(("wxid_", "wechaty:")):
            return name
    name = _text(channel_name(customer_id), 100)
    return name if name not in hidden_names and not name.casefold().startswith(("wxid_", "wechaty:")) else ""


def _game_roster(game, customers, channel_name):
    """以参与登记为事实源，分别投影发起人、已占位的小组和已离局登记。"""
    active_statuses = {"joined", "confirmed"}
    organizer_is_seated = False
    roster = []
    for participant in game.participants:
        member_ids = list(dict.fromkeys(participant.known_member_ids or [participant.customer_id]))
        is_seated = participant.status in active_statuses
        seats = max(1, participant.seat_count) if is_seated else 0
        if seats and game.organizer_id in member_ids:
            organizer_is_seated = True
        roster.append({
            "name": _roster_name(customers, channel_name, participant.customer_id, participant.display_name),
            "organizer": participant.customer_id == game.organizer_id,
            "status": participant.status,
            "seats": seats,
            "registered_at": participant.joined_at.isoformat(),
            "registered_by_other": bool(participant.registered_by_customer_id
                                        and participant.registered_by_customer_id != participant.customer_id),
            "named_companions": [
                {"name": _roster_name(customers, channel_name, member_id)}
                for member_id in member_ids if member_id != participant.customer_id
            ] if is_seated else [],
            "anonymous_seats": max(0, seats - len(member_ids)),
        })
    roster.sort(key=lambda row: (not row["seats"], not row["organizer"], row["registered_at"]))
    return {
        "organizer": {
            "name": _roster_name(customers, channel_name, game.organizer_id, game.organizer_name),
            "is_seated": organizer_is_seated,
        },
        "participants": roster,
    }


def _game_board(store, selected_game):
    """只读取已持久化的当前牌局，轮询不会触发过期写入或群看板发布。"""
    active = {"forming", "inviting", "ready"}
    games = list(store.games.values())
    games.sort(key=lambda game: (
        game.game_id != selected_game, game.status.value not in active,
        (game.planned_start_at or game.created_at).timestamp()
        if game.status.value in active else -game.updated_at.timestamp(),
    ))
    customers = store.customers if games else {}

    @lru_cache(maxsize=None)
    def channel_name(customer_id):
        """没有已登记称呼时，仅查同一客户的公开渠道身份；一次刷新内复用查找。"""
        identity = store.get_channel_identity_for_customer(customer_id) if customer_id else None
        return identity.public_name if identity else ""

    board = []
    for game in games[:RECORD_LIMIT]:
        item = public_group_game_summary(game)
        item["start_display"] = public_game_start_display(game)
        item["source"] = "群内自招" if game.requirement.get("recruitment_mode") == "board_only" else ""
        item["recruitment_status"] = game.recruitment_status.value
        item["recruitment_opens_at"] = game.recruitment_opens_at.isoformat() if game.recruitment_opens_at else None
        item["roster"] = _game_roster(game, customers, channel_name)
        board.append(item)
    return board


def agent_work_snapshot(runtime, *, selected="", include_model_io=False):
    """提供近 50 条收件与选中详情，复用已有 SQLite 观察表及追踪文件。"""
    events = _recent_events(runtime.trace_recorder)
    traces, batch_runs, linked = _trace_index(events)
    records = [
        record for channel in ("wechaty", "agent-wechat")
        for record in runtime.store.list_channel_observations(channel=channel, limit=RECORD_LIMIT)
    ]
    console_records = {}
    for event in reversed(events):
        if event.step != "console_message_received":
            continue
        message = event.content
        key = event.trace_id
        console_records.setdefault(key, {
            **message, "channel": "api", "trace_id": event.trace_id, "received_at": event.occurred_at,
        })
        if len(console_records) >= RECORD_LIMIT:
            break
    records = sorted([*records, *console_records.values()],
                     key=lambda row: (row.get("received_at") or "").replace(" ", "T", 1), reverse=True)[:RECORD_LIMIT]
    result, detail = [], None
    for record in records:
        related, batch_id = _related_events(record, traces, batch_runs, linked)
        summary = _record_snapshot(record, related, batch_id, details=False)
        result.append(summary)
        if summary["key"] == selected or (not selected and detail is None):
            detail = _record_snapshot(record, related, batch_id, details=True, include_model_io=include_model_io)
    return {"records": result, "detail": detail,
            "games": _game_board(runtime.store, (detail or {}).get("game_id")),
            "server_time": datetime.now().isoformat(timespec="seconds")}
