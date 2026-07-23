"""One-shot semantic classifier for one isolated group-chat session."""

from __future__ import annotations

import json
from typing import Any

from ..knowledge import DomainTerminologyRepository, default_terminology_repository
from ..llm import AgentLLMClient
from .models import BoardState, ChatSession, GroupMessage, SessionClassification


_SYSTEM_PROMPT = """你是麻将群运营助手。你只做当前群聊 Session 的意图识别和通道路由，不执行工具。

输入只包含当前看板、当前 Session 的历史和一条新的聚合消息。不得猜测或引用其他 Session、其他私聊或内部系统信息。

判断规则（先按用户本轮言语行为分类，再参考看板；看板为空不能改变言语行为）：
- query：用户在询问当前有没有局、还有没有位置、是否齐人或其他公开状态。例如“1块杭麻有人吗”“0.5还有嘛”即使看板为空也仍是 query，后续主 Agent 会负责查询。
- claim：用户明确表达要加入某个现有看板局，例如“4来”“红中568我打”。没有加入意愿，或只是发布“三缺一/在线等”，都不是 claim。
- new_demand：用户发布自己的缺人需求，或明确要求老板新建/组一个局。例如“三缺一在线等”“帮我组个0.5无烟局”。
- thread_update：同一组局讨论中的条件补充或确认。
- chitchat：与麻将组局无关的闲聊。

通道动作：
- 认领或新需求优先 private_switch；歧义认领也切私聊确认。
- 当前 Session 正在询问或讨论组局时，其他人回复“+1/我也来/行/ok”属于 thread_update，不要误判为独立认领。
- 匹配前先枚举所有满足用户已明确条件的看板项。用户没提到的字段是通配条件，不能据此排除固定时间或其他字段不同的局。
- 只有恰好一个看板项满足全部已明确条件时才能填写 matched_board_no；多个看板项都满足时必须为 null，confidence 必须低于 0.7。
- 只有简单公开查询可以 group_reply。
- 中间更新和闲聊使用 ignore。

领域语义：
- 只采用输入里的 domain_terminology 作为本轮黑话解释依据，不得凭通用常识覆盖场馆定义。
- game_type 只写主类型，ruleset/game_variant 单独写子玩法；人数短码写入 participant_code。
- confidence="opaque" 的规则码只原样提取为 rule_code，不得解释成底注、封顶或番数。
- “一块也行、0.5 也可以”表示多个可接受档位，按出现顺序写入 accepted_stakes。
- “三缺一/真三缺一”必须提取 current_players=3、missing_players=1；未出现的玩法、时间、烟况不得补造。
- “在线等/急”等明确紧急表达统一写 urgency="high"；普通需求写 "normal" 或 null，不得创造其他枚举值。

严格输出一个 JSON 对象，不得输出 Markdown、解释前缀或额外字段。
intent 和 channel_action 必须分别是一个字符串标量，禁止输出数组。"""


class SessionClassificationContractError(ValueError):
    """Raised after bounded model repair still cannot satisfy the contract."""

    def __init__(self, message: str, *, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


def _canonical_feature(value: object, *, field_name: str) -> str:
    """Compare semantic values without changing the board's display representation."""

    normalized = str(value or "").strip().lower()
    if field_name == "stakes":
        return normalized.removesuffix("块").removesuffix("元")
    if field_name == "game_type":
        matched = default_terminology_repository().first_match(
            normalized,
            categories={"game_type", "game_variant"},
        )
        if matched is not None:
            return str(matched.term.canonical.get("game_type") or normalized).lower()
    return normalized


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
        "scalar_fields": ["intent", "channel_action", "confidence", "reasoning"],
        "extracted_features": {
            "game_type": "string|null",
            "stakes": "string|null",
            "time": "string|null",
            "participant_code": "173|272|371|null; participant count shorthand",
            "smoking": "string|null",
            "accepted_stakes": "array<string>|null",
            "current_players": "integer|null",
            "missing_players": "integer|null",
            "urgency": "high|normal|null",
            "duration_hours": "number|null",
            "ruleset": "string|null",
            "rule_code": "string|null",
        },
        "matched_board_no": "integer|null; must exist in board_state when non-null",
        "confidence": "number in [0,1]",
        "reasoning": "short string beginning with [T<number>]",
        "response": "string|null; only public query needs a group reply",
    }


class GroupSessionClassifier:
    """Run one semantic task with bounded contract repair, never schema coercion."""

    def __init__(
        self,
        llm: AgentLLMClient,
        *,
        timeout_seconds: float = 12,
        max_contract_retries: int = 1,
        terminology: DomainTerminologyRepository | None = None,
    ) -> None:
        self.llm = llm
        self.timeout_seconds = timeout_seconds
        self.max_contract_retries = max(0, max_contract_retries)
        self.terminology = terminology or default_terminology_repository()

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
            "domain_terminology": self.terminology.context_for_texts(
                [
                    *(item.text for item in session.messages[-12:]),
                    new_message.text,
                    *(
                        " ".join(
                            str(value or "")
                            for value in (
                                item.game_type,
                                item.ruleset,
                                item.participant_code,
                                item.rule_code,
                                item.stakes,
                            )
                        )
                        for item in (board_state.items if board_state else [])
                    ),
                ]
            ),
            "output_contract": session_classification_contract(),
        }
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ]
        attempts = self.max_contract_retries + 1
        last_error: ValueError | TypeError | json.JSONDecodeError | None = None
        for attempt_index in range(attempts):
            content = self.llm.complete(
                messages,
                trace_id=trace_id if attempt_index == 0 else f"{trace_id}_contract_retry_{attempt_index}",
                timeout_seconds=self.timeout_seconds,
            )
            try:
                return self._parse(content, board_state=board_state)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt_index + 1 >= attempts:
                    break
                messages.extend(
                    [
                        {"role": "assistant", "content": content},
                        {
                            "role": "user",
                            "content": (
                                "上一个输出未通过合同校验："
                                f"{type(exc).__name__}: {exc}。请重新输出完整 JSON 对象；"
                                "intent 与 channel_action 必须是单个字符串标量，禁止数组；"
                                "不得解释、不得输出 Markdown、不得省略字段。"
                            ),
                        },
                    ]
                )
        assert last_error is not None
        raise SessionClassificationContractError(
            f"group session classification contract failed after {attempts} attempts: {last_error}",
            attempts=attempts,
        ) from last_error

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
        intent = payload["intent"]
        action = payload["channel_action"]
        if not isinstance(intent, str):
            raise ValueError("intent must be one string scalar")
        if not isinstance(action, str):
            raise ValueError("channel_action must be one string scalar")
        if intent not in session_classification_contract()["intent"]:
            raise ValueError(f"invalid group session intent: {intent}")
        if action not in session_classification_contract()["channel_action"]:
            raise ValueError(f"invalid group session channel_action: {action}")
        features = payload["extracted_features"]
        if not isinstance(features, dict):
            raise ValueError("extracted_features must be an object")
        raw_confidence = payload["confidence"]
        if isinstance(raw_confidence, bool) or not isinstance(raw_confidence, (int, float)):
            raise ValueError("confidence must be a number")
        confidence = float(raw_confidence)
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
        reasoning = payload["reasoning"]
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError("reasoning must be a non-empty string")
        allowed_feature_keys = (
            "game_type",
            "stakes",
            "time",
            "participant_code",
            "smoking",
            "accepted_stakes",
            "current_players",
            "missing_players",
            "urgency",
            "duration_hours",
            "ruleset",
            "rule_code",
        )
        classification = SessionClassification(
            intent=intent,
            extracted_features={key: features.get(key) for key in allowed_feature_keys},
            matched_board_no=matched,
            confidence=confidence,
            reasoning=reasoning,
            response=response,
            channel_action=action,
        )
        urgency = classification.extracted_features.get("urgency")
        if urgency not in (None, "high", "normal"):
            raise ValueError(f"invalid urgency: {urgency}")
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
            "participant_code": "participant_code",
            "smoking": "smoking",
            "ruleset": "ruleset",
            "rule_code": "rule_code",
        }
        stated = {
            feature: _canonical_feature(value, field_name=feature)
            for feature, value in classification.extracted_features.items()
            if feature in field_map and value not in (None, "", [])
        }
        if not stated:
            return classification
        matching = [
            item
            for item in board_state.items
            if all(
                _canonical_feature(getattr(item, field_map[key]), field_name=key) == value
                for key, value in stated.items()
            )
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


__all__ = [
    "GroupSessionClassifier",
    "SessionClassificationContractError",
    "session_classification_contract",
]
