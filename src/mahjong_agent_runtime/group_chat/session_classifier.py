"""One-shot semantic classifier for one isolated group-chat session."""

from __future__ import annotations

import json
from typing import Any

from ..llm import AgentLLMClient
from .models import BoardState, ChatSession, GroupMessage, SessionClassification


_SYSTEM_PROMPT = """你是麻将群运营助手。你只做当前群聊 Session 的意图识别和通道路由，不执行工具。

输入只包含当前看板、当前 Session 的历史和一条新的聚合消息。不得猜测或引用其他 Session、其他私聊或内部系统信息。

判断规则：
- claim：用户提到看板上某局的特征并表达加入意愿。
- new_demand：用户想组局，但当前看板没有唯一匹配项。
- query：用户询问局的状态或公开信息。
- thread_update：同一组局讨论中的条件补充或确认。
- chitchat：与麻将组局无关的闲聊。

通道动作：
- 认领或新需求优先 private_switch；歧义认领也切私聊确认。
- 当前 Session 正在询问或讨论组局时，其他人回复“+1/我也来/行/ok”属于 thread_update，不要误判为独立认领。
- 匹配前先枚举所有满足用户已明确条件的看板项。用户没提到的字段是通配条件，不能据此排除固定时间或其他字段不同的局。
- 只有恰好一个看板项满足全部已明确条件时才能填写 matched_board_no；多个看板项都满足时必须为 null，confidence 必须低于 0.7。
- 只有简单公开查询可以 group_reply。
- 中间更新和闲聊使用 ignore。

严格输出一个 JSON 对象，不得输出 Markdown、解释前缀或额外字段。"""


def session_classification_contract() -> dict[str, Any]:
    return {
        "required": [
            "intent",
            "extracted_features",
            "matched_board_no",
            "confidence",
            "reasoning",
            "response",
            "channel_action",
        ],
        "intent": ["claim", "new_demand", "query", "thread_update", "chitchat"],
        "channel_action": ["private_switch", "group_reply", "ignore"],
        "extracted_features": {
            "game_type": "string|null",
            "stakes": "string|null",
            "time": "string|null",
            "table_id": "string|null",
            "smoking": "string|null",
        },
        "matched_board_no": "integer|null; must exist in board_state when non-null",
        "confidence": "number in [0,1]",
        "reasoning": "short string beginning with [T<number>]",
        "response": "string|null; only public query needs a group reply",
    }


class GroupSessionClassifier:
    """Call a model once and validate the semantic result as a strict contract."""

    def __init__(self, llm: AgentLLMClient, *, timeout_seconds: float = 12) -> None:
        self.llm = llm
        self.timeout_seconds = timeout_seconds

    def classify(
        self,
        *,
        board_state: BoardState | None,
        session: ChatSession,
        new_message: GroupMessage,
        trace_id: str,
    ) -> SessionClassification:
        payload = {
            "board_state": board_state.to_dict() if board_state else {"room_id": new_message.room_id, "items": []},
            "current_session": session.to_context(),
            "new_message": new_message.to_dict(),
            "output_contract": session_classification_contract(),
        }
        content = self.llm.complete(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
            ],
            trace_id=trace_id,
            timeout_seconds=self.timeout_seconds,
        )
        return self._parse(content, board_state=board_state)

    @staticmethod
    def _parse(content: str, *, board_state: BoardState | None) -> SessionClassification:
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]).strip()
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("group session classifier must return one JSON object")
        required = set(session_classification_contract()["required"])
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"group session classifier missing fields: {', '.join(missing)}")
        intent = str(payload["intent"])
        action = str(payload["channel_action"])
        if intent not in session_classification_contract()["intent"]:
            raise ValueError(f"invalid group session intent: {intent}")
        if action not in session_classification_contract()["channel_action"]:
            raise ValueError(f"invalid group session channel_action: {action}")
        features = payload["extracted_features"]
        if not isinstance(features, dict):
            raise ValueError("extracted_features must be an object")
        confidence = float(payload["confidence"])
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be in [0,1]")
        matched = payload["matched_board_no"]
        if matched is not None:
            if isinstance(matched, bool) or not isinstance(matched, int):
                raise ValueError("matched_board_no must be an integer or null")
            valid_numbers = {item.display_no for item in board_state.items} if board_state else set()
            if matched not in valid_numbers:
                raise ValueError("matched_board_no does not exist in current board")
        response = payload["response"]
        if response is not None and not isinstance(response, str):
            raise ValueError("response must be a string or null")
        classification = SessionClassification(
            intent=intent,
            extracted_features={
                key: features.get(key) for key in ("game_type", "stakes", "time", "table_id", "smoking")
            },
            matched_board_no=matched,
            confidence=confidence,
            reasoning=str(payload["reasoning"]),
            response=response,
            channel_action=action,
        )
        return GroupSessionClassifier._enforce_unique_board_match(classification, board_state=board_state)

    @staticmethod
    def _enforce_unique_board_match(
        classification: SessionClassification,
        *,
        board_state: BoardState | None,
    ) -> SessionClassification:
        """Prevent a claim from selecting one item when stated facts match several."""

        if classification.intent != "claim" or classification.matched_board_no is None or board_state is None:
            return classification
        field_map = {
            "game_type": "game_type",
            "stakes": "stakes",
            "time": "time",
            "table_id": "table_id",
            "smoking": "smoking",
        }
        stated = {
            feature: str(value).strip().lower()
            for feature, value in classification.extracted_features.items()
            if feature in field_map and value not in (None, "", [])
        }
        if not stated:
            return classification
        matching = [
            item
            for item in board_state.items
            if all(str(getattr(item, field_map[key]) or "").strip().lower() == value for key, value in stated.items())
        ]
        if len(matching) <= 1:
            return classification
        return SessionClassification(
            intent=classification.intent,
            extracted_features=classification.extracted_features,
            matched_board_no=None,
            confidence=min(classification.confidence, 0.69),
            reasoning=f"{classification.reasoning} [合同校验：已明确条件同时匹配多个看板项，需私聊确认。]",
            response=classification.response,
            channel_action="private_switch",
        )


__all__ = ["GroupSessionClassifier", "session_classification_contract"]
