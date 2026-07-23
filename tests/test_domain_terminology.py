from __future__ import annotations

import json

from mahjong_agent_runtime import AgentContextBuilder, InMemoryAgentStore, ToolGateway, UserMessage
from mahjong_agent_runtime.group_chat import ChatSession, GroupMessage, GroupSessionClassifier
from mahjong_agent_runtime.group_chat.parsing import parse_explicit_need
from mahjong_agent_runtime.knowledge import default_terminology_repository


class _RecordingLLM:
    def __init__(self, output: dict) -> None:
        self.output = json.dumps(output, ensure_ascii=False)
        self.messages: list[dict[str, str]] = []

    def complete(self, messages, *, trace_id: str, timeout_seconds: float) -> str:
        self.messages = list(messages)
        return self.output


def _classification_output() -> dict:
    return {
        "intent": "claim",
        "extracted_features": {
            "game_type": "杭麻",
            "stakes": "1块",
            "time": None,
            "participant_code": "371",
            "smoking": "无烟",
            "accepted_stakes": None,
            "current_players": 3,
            "missing_players": 1,
            "ruleset": "财敲",
            "rule_code": None,
        },
        "matched_board_no": None,
        "confidence": 0.96,
        "reasoning": "[T1] cq371 是财敲三缺一。",
        "response": None,
        "channel_action": "private_switch",
    }


def test_reviewed_terminology_resolves_compact_group_dialect() -> None:
    repository = default_terminology_repository()

    matches = repository.context_for_texts(["cq371 人齐开 无烟 1块"])
    terms = {item["term_id"]: item for item in matches}

    assert terms["game_variant_caiqiao"]["canonical"]["requested_game"] == "hangzhou_mahjong"
    assert "不是重庆麻将" in terms["game_variant_caiqiao"]["definition"]
    assert terms["participant_code_371"]["canonical"] == {
        "participant_code": "371",
        "seat_format": "371",
        "known_player_count": 3,
        "needed_seats": 1,
    }
    assert terms["start_asap_when_full"]["canonical"]["start_time_kind"] == "asap_when_full"


def test_opaque_venue_code_is_retrieved_without_invented_semantics() -> None:
    repository = default_terminology_repository()

    match = repository.first_match("红中272 19.00 无烟 568", categories={"opaque_rule_code"})

    assert match is not None
    assert match.term.term_id == "venue_rule_code_568"
    assert match.term.confidence == "opaque"
    assert match.term.canonical == {"rule_code": "568"}
    assert "底注" not in match.term.canonical
    assert "封顶" not in match.term.canonical


def test_private_parser_and_model_context_share_the_same_terminology() -> None:
    parsed = parse_explicit_need("cq371 人齐开 无烟 1块")
    assert parsed["requested_game"] == "hangzhou_mahjong"
    assert parsed["game_variant"] == "caiqiao"
    assert parsed["known_player_count"] == 3
    assert parsed["needed_seats"] == 1

    store = InMemoryAgentStore()
    built = AgentContextBuilder(store, ToolGateway(store)).build(
        UserMessage(
            conversation_id="terminology_private",
            sender_id="customer-a",
            sender_name="A",
            text="cq371 人齐开 无烟 1块",
            message_id="terminology_private_001",
        ),
        trace_id="trace_terminology_private",
    )

    term_ids = {item["term_id"] for item in built.payload["domain_terminology"]}
    assert {"game_variant_caiqiao", "participant_code_371", "start_asap_when_full"} <= term_ids
    assert built.audit["domain_terminology_term_ids"] == [
        item["term_id"] for item in built.payload["domain_terminology"]
    ]


def test_group_classifier_receives_only_relevant_reviewed_terms() -> None:
    llm = _RecordingLLM(_classification_output())
    classifier = GroupSessionClassifier(llm, max_contract_retries=0)
    message = GroupMessage(
        room_id="room-1",
        conversation_id="wechaty:room:room-1",
        sender_external_id="customer-a",
        sender_name="A",
        text="cq371 人齐开 无烟 1块",
        message_id="group-term-001",
    )
    session = ChatSession(
        id="session-1",
        room_id="room-1",
        messages=[message],
        participants={"customer-a"},
    )

    result = classifier.classify(
        board_state=None,
        session=session,
        new_message=message,
        trace_id="trace_group_terminology",
    )
    payload = json.loads(llm.messages[-1]["content"])
    term_ids = {item["term_id"] for item in payload["domain_terminology"]}

    assert result.extracted_features["ruleset"] == "财敲"
    assert {"game_variant_caiqiao", "participant_code_371", "start_asap_when_full"} <= term_ids
    assert "game_sichuan_mahjong" not in term_ids
