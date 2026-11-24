"""可写业务的离线合同桩，只验证同一套工具、消息和存储接线。"""

from __future__ import annotations

import json
from typing import Any

from .sim_adapter import StaticAgentLLMClient


class VenueFixtureAgent(StaticAgentLLMClient):
    """按已登记的测试轮次返回工具合同，工具结果仍由真实 Gateway 生成。"""

    def __init__(self, *, seed: int) -> None:
        super().__init__(seed=seed)
        self.semantic_intent = "member_discussion"
        self.turns: dict[str, tuple[str, dict[str, Any]]] = {}

    def complete(self, messages: list[dict[str, str]], *, trace_id: str, timeout_seconds: float) -> str:
        """按测试消息选择合同；维护真正模型没有的离线专用期望表。"""
        context = self._context_payload(messages)
        message = context.get("current_message") or {}
        spec = self.turns.get(str(message.get("message_id") or ""))
        if spec is None:
            return super().complete(messages, trace_id=trace_id, timeout_seconds=timeout_seconds)
        phase, details = spec
        previous = list(context.get("previous_tool_results") or [])
        evidence = context.get("turn_tool_evidence") or []
        previous = [
            evidence[item["result_reference"]["index"]] if "result_reference" in item else item
            for item in previous
        ]
        if phase == "dialogue_open":
            return _reply("你想几点开始，能打多久？", status="waiting_user")
        if phase == "dialogue_followup":
            if not previous:
                return _call("search_current_games", {"requirement": {}, "limit": 5})
            if previous[-1].get("allowed") and not previous[-1].get("error"):
                return _reply("我查了一下，这会儿暂时没找到合适的局。")
        if phase == "cohort_offer":
            if not previous:
                return _call("search_current_games", {"requirement": {}, "limit": 5})
            matches = previous[-1].get("result", {}).get("matches", [])
            if any(item["game"]["game_id"] == details["game_id"] for item in matches):
                return _reply(f"今晚七点有一桌无烟杭州麻将，一块底，打{details['duration']}小时，还差一人，要不要帮你留位？",
                              status="waiting_user")
        if phase == "cohort_join":
            if not previous:
                return _call("join_game", {"game_id": details["game_id"], "customer_id": message["sender_id"],
                                          "display_name": message["sender_name"], "seat_count": 1})
            if previous[-1].get("allowed") and not previous[-1].get("error"):
                return _reply("好，给你留上一个位置了。")
        if phase == "need" and not previous:
            return _call("create_game", {
                "requirement": details["requirement"],
                "organizer_id": message["sender_id"],
                "organizer_name": message["sender_name"],
                "requesting_party": {
                    "contact_id": message["sender_id"], "contact_name": message["sender_name"], "seat_count": 1,
                },
            })
        if phase == "need" and previous[-1]["name"] == "create_game":
            game = (previous[-1].get("result") or {}).get("game") or {}
            if game.get("game_id"):
                return _call("search_customers", {"game_id": game["game_id"], "limit": 3})
        if phase == "need" and previous[-1]["name"] == "search_customers":
            search = previous[-1].get("result") or {}
            if search.get("candidates") and not (search.get("continuation") or {}).get("can_stop"):
                return _call("create_invite_drafts", {"game_id": search["bound_game_id"], "invitations": [
                    {"customer_id": item["customer_id"], "display_name": item["display_name"],
                     "message_text": "今晚七点有无烟杭州麻将，一块底，打四小时，你想来吗？"}
                    for item in search["candidates"]
                ]})
        if phase in {"opportunity", "delivered"} and not previous:
            return _call("search_current_games", {"requirement": {}, "limit": 5})
        if phase in {"opportunity", "delivered"} and previous[-1]["name"] == "search_current_games":
            matches = (previous[-1].get("result") or {}).get("matches") or []
            if matches:
                return _call("join_game", {
                    "game_id": matches[0]["game"]["game_id"],
                    "customer_id": message["sender_id"],
                    "display_name": message["sender_name"],
                    "seat_count": 1,
                })
        if phase == "preference" and not previous:
            return _call("record_user_memory", {"pending_long_term_memories": [{
                "customer_id": message["sender_id"], "memory_type": "preference", "field": "smoke_preference",
                "value": "no_smoking", "evidence": message["text"], "confidence": 1, "scope": "long_term",
            }]})
        if phase == "preference" and previous[-1].get("allowed"):
            return _reply("好，记下你习惯无烟的。")
        if phase == "slot_query" and not previous:
            return _call("check_room_availability", {"start_at": details["start_at"], "end_at": details["end_at"]})
        if phase == "slot_query" and previous:
            result = previous[-1].get("result") or {}
            if result.get("configured"):
                return _reply("这会儿有空包间。" if result.get("available_room_ids") else "这会儿包间都占着。")
        replies = {
            "need": ("search_customers", "好，给你登记了今晚七点的局。"),
            "opportunity": ("join_game", "好，给你留上一个位置了。"),
            "delivered": ("join_game", "好，给你留上一个位置了。"),
        }
        if phase in replies and previous:
            name, reply = replies[phase]
            last = previous[-1]
            expected = {name, "create_invite_drafts"} if phase == "need" else {name}
            if last["name"] in expected and last.get("allowed") and not last.get("error"):
                return _reply(reply)
        return _reply("行，知道了。")


def _call(name: str, arguments: dict[str, Any]) -> str:
    """使用现有离线桩的动作结构，替换为当前轮的可信工具调用。"""
    body = StaticAgentLLMClient._tool_action()
    body["goal"] = "执行本轮客户明确请求"
    body["objective_state"]["current_phase"] = name
    body["objective_plan"] = [{
        "step_id": name, "title": "执行本轮业务工具", "status": "in_progress", "tool": name, "depends_on": [],
    }]
    body["tool_calls"] = [{
        "call_id": name, "depends_on": [], "name": name, "arguments": arguments, "reason": "按本轮客户请求执行",
    }]
    body["stop_reason"]["pending_work"] = [name]
    return json.dumps(body, ensure_ascii=False)


def _reply(text: str, *, status: str = "completed") -> str:
    """结束当前离线路由，保留真实工具返回后才确认业务写入的约束。"""
    return json.dumps({
        "goal": "结束本轮客户交互", "objective_status": status,
        "reasoning_summary": "当前轮无需再调用工具",
        "objective_state": {"current_phase": "answer_user", "known_facts": {}, "missing_facts": [], "blockers": []},
        "objective_plan": [], "plan_revision_reason": "已得到本轮结果",
        "reply_to_user": text, "tool_calls": [], "needs_human": False,
        "stop_reason": {"can_stop": True, "why": "可以回复客户", "pending_work": [], "depends_on_tool_results": False},
        "badcase": None,
    }, ensure_ascii=False)
