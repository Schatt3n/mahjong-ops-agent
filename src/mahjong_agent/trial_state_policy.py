from __future__ import annotations

from typing import Any


STATE_MACHINE_VERSION = "state_machine.v1"

APPROVAL_STATUS_LABELS = {
    "pending": "待审批",
    "approved": "已审批",
    "rejected": "审批拒绝",
}

GAME_STATUS_TRANSITIONS: dict[str, set[str]] = {
    "待补充": {"待补充", "待组局", "邀约中", "已满", "已成局", "已取消"},
    "待组局": {"待补充", "待组局", "邀约中", "已满", "已成局", "已取消"},
    "邀约中": {"待组局", "邀约中", "已满", "已成局", "已取消"},
    "已满": {"邀约中", "已满", "已成局", "已取消"},
    "已成局": {"已成局"},
    "已取消": {"已取消"},
}

OUTBOX_STATUS_TRANSITIONS: dict[str, set[str]] = {
    "待审批": {"待审批", "已审批", "审批拒绝", "已复制", "已发送", "未回复", "待确认", "待协商", "已确认", "已到店", "拒绝", "下次再问", "别再打扰", "局取消"},
    "已审批": {"已审批", "已复制", "已发送", "未回复", "待确认", "待协商", "已确认", "已到店", "拒绝", "下次再问", "别再打扰", "局取消"},
    "审批拒绝": {"审批拒绝"},
    "已复制": {"已复制", "已发送", "未回复", "待确认", "待协商", "已确认", "已到店", "拒绝", "下次再问", "别再打扰", "局取消"},
    "已发送": {"已发送", "未回复", "待确认", "待协商", "已确认", "已到店", "拒绝", "下次再问", "别再打扰", "局取消"},
    "未回复": {"未回复", "待确认", "待协商", "已确认", "已到店", "拒绝", "下次再问", "别再打扰", "局取消"},
    "待确认": {"待确认", "待协商", "已确认", "已到店", "拒绝", "未回复", "下次再问", "别再打扰", "局取消"},
    "待协商": {"待协商", "待确认", "已确认", "已到店", "拒绝", "未回复", "下次再问", "别再打扰", "局取消"},
    "已确认": {"已确认", "已到店", "局取消"},
    "已到店": {"已到店"},
    "拒绝": {"拒绝"},
    "下次再问": {"下次再问"},
    "别再打扰": {"别再打扰"},
    "局取消": {"局取消"},
}

FOLLOWUP_STATUS_TRANSITIONS: dict[str, set[str]] = {
    "待审批": {"待审批", "已审批", "审批拒绝", "局取消"},
    "已审批": {"已审批", "局取消"},
    "审批拒绝": {"审批拒绝"},
    "局取消": {"局取消"},
}


def approval_status_label(status: str | None) -> str:
    normalized = str(status or "").strip().lower()
    return APPROVAL_STATUS_LABELS.get(normalized, str(status or "") or "待审批")


def state_transition_verdict(
    *,
    entity_type: str,
    current_status: str | None,
    next_status: str,
    event: str,
) -> dict[str, Any]:
    registries = {
        "game": GAME_STATUS_TRANSITIONS,
        "outbox": OUTBOX_STATUS_TRANSITIONS,
        "followup": FOLLOWUP_STATUS_TRANSITIONS,
    }
    registry = registries.get(entity_type)
    if registry is None:
        return {
            "allowed": False,
            "code": "unknown_entity_type",
            "reason": f"未知状态机实体：{entity_type}",
            "state_machine_version": STATE_MACHINE_VERSION,
        }
    current = str(current_status or "").strip()
    target = str(next_status or "").strip()
    if not target:
        return {
            "allowed": False,
            "code": "missing_next_status",
            "reason": "缺少目标状态。",
            "state_machine_version": STATE_MACHINE_VERSION,
        }
    if not current:
        allowed = target in registry
    else:
        allowed = target in registry.get(current, set())
    code = "state_transition_allowed" if allowed else "state_transition_rejected"
    reason = (
        f"{entity_type} 状态允许从 {current or '<new>'} -> {target}，事件：{event}。"
        if allowed
        else f"{entity_type} 状态不允许从 {current or '<new>'} -> {target}，事件：{event}。"
    )
    return {
        "allowed": allowed,
        "code": code,
        "reason": reason,
        "entity_type": entity_type,
        "from_status": current or None,
        "to_status": target,
        "event": event,
        "state_machine_version": STATE_MACHINE_VERSION,
    }


def require_state_transition(
    *,
    entity_type: str,
    current_status: str | None,
    next_status: str,
    event: str,
) -> dict[str, Any]:
    verdict = state_transition_verdict(
        entity_type=entity_type,
        current_status=current_status,
        next_status=next_status,
        event=event,
    )
    if not verdict["allowed"]:
        raise ValueError(verdict["reason"])
    return verdict
