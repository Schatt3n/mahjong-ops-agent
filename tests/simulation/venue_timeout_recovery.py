"""根据正式运行轨迹与沙盒送达账本，核实同一次来话的超时恢复。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from mahjong_agent_runtime.tracing import TraceEvent

from .venue_channel import AGENT_ID


_ATTENTION = "agent_runtime_attention"
_RESOLVED = "agent_runtime_attention_resolved"
_FINAL = "final_output"
_INGRESS_QUEUED = "input_gate_retry_queued"
_INPUT_BATCH_FINISHED = "input_batch_finished"
_GROUP_TOPIC_QUEUED = "group_topic_retry_queued"
_GROUP_TOPIC_FINISHED = "group_topic_retry_finished"
_RECOVERABLE_REASONS = {"planner_timeout", "reply_review_timeout"}


def verified_runtime_timeout_recoveries(trace_path: Path, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """同一运行、来话及实际交付均需吻合，等待回执不算业务回复。

    私聊独立校验原收件人。托管群必须匹配群来源及原群送达，或匹配真实群转私聊的
    单人隔离会话及原发言人。等待提示、静默路由和只有模型终稿都不算业务回复送达。
    """
    if not trace_path.is_file():
        return []
    inbound = {row["message_id"]: row for row in messages
               if row["direction"] == "inbound" and row["kind"] == "private"}
    inbound_group = {row["message_id"]: row for row in messages
                     if row["direction"] == "inbound" and row["kind"] == "group"}
    delivered = defaultdict(list)
    group_delivered = defaultdict(list)
    for row in messages:
        if row["direction"] != "outbound" or row["sender_id"] != AGENT_ID:
            continue
        proof = (row["trace_id"], row["source_message_id"], row["conversation_id"], row["recipient_id"], row["text"])
        if row["kind"] == "private":
            delivered[proof].append(row)
        if row["source_message_id"] in inbound_group:
            group_delivered[(*proof, row["kind"])].append(row)
    open_timeout: dict[tuple, int] = {}
    resolved: dict[tuple, tuple[str, dict]] = {}
    final_outputs: dict[str, tuple[int, dict]] = {}
    ingress_queued: dict[tuple, tuple[int, int]] = {}
    ingress_resolved: dict[tuple, tuple[str, dict]] = {}
    group_queued: dict[tuple, int] = {}
    group_resolved: dict[tuple, tuple[dict, dict]] = {}
    step_markers = tuple(f'"step": "{step}"' for step in (
        _ATTENTION, _RESOLVED, _FINAL, _INGRESS_QUEUED, _INPUT_BATCH_FINISHED,
        _GROUP_TOPIC_QUEUED, _GROUP_TOPIC_FINISHED,
    ))
    with trace_path.open(encoding="utf-8") as handle:
        for position, line in enumerate(handle):
            if not any(marker in line for marker in step_markers):
                continue
            event = TraceEvent.from_log_line(line.rstrip("\n"))
            if event is None:
                continue
            payload = event.content
            if event.step == _FINAL:
                final_outputs[event.trace_id] = (position, payload)
                continue
            if event.step in {_GROUP_TOPIC_QUEUED, _GROUP_TOPIC_FINISHED}:
                message_id, room_id, batch_id, version = (payload.get(field) for field in (
                    "source_message_id", "room_id", "batch_id", "batch_version"))
                source = inbound_group.get(message_id)
                if (not source or not isinstance(room_id, str) or not room_id
                        or source.get("room_id") != room_id or source.get("trace_id") != event.trace_id
                        or not isinstance(batch_id, str) or not batch_id or type(version) is not int):
                    continue
                key = (event.trace_id, message_id, batch_id, version)
                if event.step == _GROUP_TOPIC_QUEUED and payload.get("technical_failure") == "read_timeout":
                    group_queued[key] = position
                if event.step == _GROUP_TOPIC_FINISHED and key in group_queued:
                    final_position, final = final_outputs.get(event.trace_id, (-1, {}))
                    if (group_queued[key] < final_position < position and payload.get("status") == "completed"
                            and payload.get("reason") == "group_topic_retry_routed"
                            and final.get("objective_status") in {"completed", "waiting_user"}):
                        group_resolved[key] = (payload, final)
                continue
            if event.step == _INGRESS_QUEUED:
                batch_id, version, message_id = (payload.get(name) for name in (
                    "batch_id", "batch_version", "source_message_id",
                ))
                if (isinstance(batch_id, str) and batch_id and type(version) is int and isinstance(message_id, str)
                        and message_id in inbound and payload.get("origin_trace_id") == event.trace_id
                        and payload.get("attempt") == 1 and payload.get("open") is True
                        and type(payload.get("conversation_version")) is int):
                    ingress_queued[(event.trace_id, message_id, batch_id, version)] = (
                        position, payload["conversation_version"],
                    )
                continue
            if event.step == _INPUT_BATCH_FINISHED:
                batch = payload.get("batch") or {}
                decision = batch.get("decision") or {}
                gate = decision.get("gate_recovery") or {}
                result = decision.get("agent_result") or {}
                fragments = batch.get("fragments") or []
                message_id = fragments[-1].get("message_id") if fragments else None
                identity = event.trace_id, message_id, batch.get("batch_id"), batch.get("version")
                queued_at, version = ingress_queued.get(identity, (-1, None))
                final_position, final = final_outputs.get(event.trace_id, (-1, {}))
                source = inbound.get(message_id) or {}
                status = result.get("status")
                if (queued_at >= 0 and queued_at < final_position < position and status in {"completed", "waiting_user"}
                        and batch.get("status") == "completed" and gate.get("open") is False and gate.get("attempt") == 2
                        and gate.get("origin_trace_id") == event.trace_id and gate.get("conversation_version") == version
                        and decision.get("completed_trace_id") == result.get("trace_id") == event.trace_id
                        and batch.get("conversation_id") == result.get("conversation_id") == source.get("conversation_id")
                        and batch.get("sender_id") == source.get("sender_id") and result.get("final_reply") == final.get("reply")):
                    ingress_resolved[identity] = (status, final)
                continue
            if event.step not in {_ATTENTION, _RESOLVED}:
                continue
            message_id, run_id, run_version, reason = (payload.get(name) for name in (
                "message_id", "run_id", "run_version", "reason",
            ))
            if (not isinstance(message_id, str) or message_id not in inbound or not isinstance(run_id, str)
                    or not run_id or type(run_version) is not int or not isinstance(reason, str)
                    or reason not in _RECOVERABLE_REASONS):
                continue
            identity = (event.trace_id, message_id, run_id, run_version, reason)
            if event.step == _ATTENTION and payload.get("retrying") is True:
                open_timeout[identity] = position
            elif event.step == _RESOLVED and identity in open_timeout:
                status = payload.get("runtime_status")
                final_position, final = final_outputs.get(event.trace_id, (-1, {}))
                if status in ("completed", "waiting_user") and final_position > open_timeout[identity]:
                    resolved[identity] = (status, final)
    recovered = []
    proof_kinds = (
        (resolved, "agent", ("run_id", "run_version")),
        (ingress_resolved, "input_gate", ("batch_id", "batch_version")),
    )
    for proofs, stage, identity_fields in proof_kinds:
        for identity, (status, final) in proofs.items():
            trace_id, message_id, entity_id, entity_version, *reasons = identity
            source = inbound[message_id]
            reply = final.get("reply")
            if (source["trace_id"] != trace_id or final.get("objective_status") != status
                    or not isinstance(reply, str) or not reply.strip()
                    or reply.strip().rstrip("。！？.!?") == "稍等一下"):
                continue
            proof = (trace_id, message_id, source["conversation_id"], source["sender_id"], reply)
            final_messages = delivered.get(proof, ())
            if len(final_messages) == 1 and final_messages[0]["sequence"] > source["sequence"]:
                recovered.append({"model_stage": stage, "reason": reasons[0] if reasons else "input_gate_timeout",
                                  "trace_id": trace_id, "message_id": message_id,
                                  identity_fields[0]: entity_id, identity_fields[1]: entity_version,
                                  "reply_message_id": final_messages[0]["message_id"], "transport_recovered": True})
    for (trace_id, message_id, batch_id, version), (route, final) in group_resolved.items():
        source = inbound_group[message_id]
        reply = final.get("reply")
        if not isinstance(reply, str) or not reply.strip() or reply.strip().rstrip("。！？.!?") == "稍等一下":
            continue
        if route.get("route_action") == "agent_loop":
            target = (source["conversation_id"], source["room_id"], "group")
        elif route.get("route_action") == "private_switch":
            private = route.get("private_conversation_id")
            if not isinstance(private, str) or not private or private == source["conversation_id"]:
                continue
            target = (private, source["sender_id"], "private")
        else:
            continue
        conversation, recipient, kind = target
        receipts = group_delivered[(trace_id, message_id, conversation, recipient, reply, kind)]
        if (len(receipts) == 1 and receipts[0].get("room_id") == source["room_id"]
                and receipts[0]["sequence"] > source["sequence"]):
            recovered.append({"model_stage": "group_topic", "reason": "group_topic_read_timeout",
                "trace_id": trace_id, "message_id": message_id, "batch_id": batch_id, "batch_version": version,
                "reply_message_id": receipts[0]["message_id"], "transport_recovered": True})
    return recovered
