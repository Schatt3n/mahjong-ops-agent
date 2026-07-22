from __future__ import annotations

import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from mahjong_agent_runtime import InMemoryAgentStore, SQLiteAgentStore
from mahjong_agent_runtime.group_chat import (
    GroupMessage,
    GroupSessionClassifier,
    GroupSessionPipeline,
    MessageAccumulator,
    OwnerMessageParser,
    QuickFilter,
    SessionCrystallizer,
    SessionMerger,
    SessionRouter,
)


TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 7, 22, 17, 0, tzinfo=TZ)
OWNER_BOARD = """cq371 人齐开 1块无烟
cq272 18.30 1块无烟

红中173 18.00 无烟3爆
红中272 19.00 无烟 568
红中173 22.30 368"""


class RecordingLLM:
    def __init__(self, *outputs: dict) -> None:
        self.outputs = [json.dumps(item, ensure_ascii=False) for item in outputs]
        self.calls: list[dict] = []

    def complete(self, messages, *, trace_id: str, timeout_seconds: float) -> str:
        self.calls.append(
            {
                "messages": messages,
                "trace_id": trace_id,
                "timeout_seconds": timeout_seconds,
            }
        )
        if not self.outputs:
            raise AssertionError("unexpected model call")
        return self.outputs.pop(0)


def message(
    text: str,
    *,
    sender: str,
    message_id: str,
    sent_at: datetime = NOW,
    room_id: str = "room-1",
    quote: str | None = None,
) -> GroupMessage:
    return GroupMessage(
        room_id=room_id,
        conversation_id=f"wechaty:room:{room_id}",
        sender_external_id=sender,
        sender_name=sender,
        text=text,
        message_id=message_id,
        sent_at=sent_at,
        quoted_message_id=quote,
    )


def classification(
    *,
    intent: str,
    matched_board_no: int | None,
    confidence: float,
    channel_action: str,
    features: dict | None = None,
    response: str | None = None,
) -> dict:
    return {
        "intent": intent,
        "extracted_features": {
            "game_type": None,
            "stakes": None,
            "time": None,
            "table_id": None,
            **(features or {}),
        },
        "matched_board_no": matched_board_no,
        "confidence": confidence,
        "reasoning": "[T1] 根据当前看板和本 Session 判断。",
        "response": response,
        "channel_action": channel_action,
    }


def pipeline(store, llm: RecordingLLM, *, on_crystallized=None) -> GroupSessionPipeline:
    router = SessionRouter(clock=lambda: NOW)
    return GroupSessionPipeline(
        store=store,
        owner_parser=OwnerMessageParser(owner_external_ids={"owner"}),
        quick_filter=QuickFilter(),
        accumulator=MessageAccumulator(quiet_seconds=5, continuation_seconds=120),
        session_router=router,
        session_merger=SessionMerger(router),
        crystallizer=SessionCrystallizer(on_crystallized=on_crystallized),
        classifier=GroupSessionClassifier(llm, timeout_seconds=5),
    )


def load_board(target: GroupSessionPipeline) -> None:
    result = target.accept(message(OWNER_BOARD, sender="owner", message_id="board"), trace_id="trace-board")
    assert result.action == "board_replaced"


def test_case_1_owner_multiline_board_replaces_persisted_board_without_model() -> None:
    store = InMemoryAgentStore()
    llm = RecordingLLM()
    target = pipeline(store, llm)

    result = target.accept(message(OWNER_BOARD, sender="owner", message_id="board"), trace_id="trace-board")

    assert result.action == "board_replaced"
    board = store.get_group_board_state("room-1")
    assert board is not None
    assert len(board.items) == 5
    assert [item.display_no for item in board.items] == [1, 2, 3, 4, 5]
    assert board.items[0].to_dict() | {} == {
        "id": board.items[0].id,
        "display_no": 1,
        "game_type": "cq",
        "table_id": "371",
        "time": "人齐开",
        "smoking": "无烟",
        "stakes": "1块",
        "special_rules": None,
        "status": "waiting",
        "slots_total": 4,
        "slots_filled": 3,
        "participants": [],
    }
    assert board.items[2].special_rules == "3爆"
    assert board.items[2].stakes == ""
    assert board.items[3].stakes == "568"
    assert board.items[4].stakes == "368"
    assert llm.calls == []


def test_case_1_board_state_survives_sqlite_restart(tmp_path) -> None:
    path = tmp_path / "group-session.sqlite3"
    llm = RecordingLLM()
    target = pipeline(SQLiteAgentStore(path), llm)
    load_board(target)

    restored = SQLiteAgentStore(path).get_group_board_state("room-1")

    assert restored is not None
    assert restored.source_message_id == "board"
    assert [(item.game_type, item.table_id, item.stakes) for item in restored.items] == [
        ("cq", "371", "1块"),
        ("cq", "272", "1块"),
        ("红中", "173", ""),
        ("红中", "272", "568"),
        ("红中", "173", "368"),
    ]


def test_case_2_owner_single_line_updates_unique_board_item_without_invalid_status() -> None:
    store = InMemoryAgentStore()
    llm = RecordingLLM()
    target = pipeline(store, llm)
    load_board(target)

    result = target.accept(
        message("371人齐开", sender="owner", message_id="update", sent_at=NOW + timedelta(minutes=1)),
        trace_id="trace-update",
    )

    assert result.action == "board_updated"
    board = store.get_group_board_state("room-1")
    assert board is not None
    item = next(item for item in board.items if item.table_id == "371")
    assert item.time == "人齐开"
    assert item.status == "waiting"
    assert board.source_message_id == "update"
    assert llm.calls == []


def test_owner_single_full_board_line_updates_one_item_without_erasing_board() -> None:
    store = InMemoryAgentStore()
    llm = RecordingLLM()
    target = pipeline(store, llm)
    load_board(target)

    result = target.accept(
        message("cq272 18.45 1块无烟", sender="owner", message_id="single-full", sent_at=NOW + timedelta(minutes=1)),
        trace_id="trace-single-full",
    )

    assert result.action == "board_updated"
    board = store.get_group_board_state("room-1")
    assert board is not None
    assert len(board.items) == 5
    item = next(item for item in board.items if item.game_type == "cq" and item.table_id == "272")
    assert item.time == "18:45"
    assert item.stakes == "1块"
    assert [entry.display_no for entry in board.items] == [1, 2, 3, 4, 5]
    assert llm.calls == []


def test_owner_single_new_board_line_appends_without_erasing_existing_items() -> None:
    store = InMemoryAgentStore()
    llm = RecordingLLM()
    target = pipeline(store, llm)
    load_board(target)

    result = target.accept(
        message("川麻371 20.00 2块有烟", sender="owner", message_id="single-new", sent_at=NOW + timedelta(minutes=1)),
        trace_id="trace-single-new",
    )

    assert result.action == "board_updated"
    board = store.get_group_board_state("room-1")
    assert board is not None
    assert len(board.items) == 6
    assert board.items[-1].game_type == "川麻"
    assert board.items[-1].display_no == 6
    assert llm.calls == []


def test_case_3_fragmented_unique_claim_is_one_model_call_with_only_session_and_board_context() -> None:
    store = InMemoryAgentStore()
    llm = RecordingLLM(
        classification(
            intent="claim",
            matched_board_no=4,
            confidence=0.96,
            channel_action="private_switch",
            features={"game_type": "红中", "stakes": "568", "time": "19:00", "table_id": "272"},
        )
    )
    target = pipeline(store, llm)
    load_board(target)

    first = target.accept(
        message("红中 568 我打", sender="德不孤", message_id="claim-1"),
        trace_id="trace-claim-1",
    )
    second = target.accept(
        message("7点", sender="德不孤", message_id="claim-2", sent_at=NOW + timedelta(seconds=2)),
        trace_id="trace-claim-2",
    )
    outcomes = target.flush_due(at=NOW + timedelta(seconds=8))

    assert first.action == second.action == "buffered"
    assert len(outcomes) == 1
    assert outcomes[0].action == "private_switch"
    assert outcomes[0].classification is not None
    assert outcomes[0].classification.intent == "claim"
    assert outcomes[0].classification.matched_board_no == 4
    assert len(llm.calls) == 1
    payload = json.loads(llm.calls[0]["messages"][-1]["content"])
    assert payload["current_session"]["participants"] == ["德不孤"]
    assert [item["display_no"] for item in payload["board_state"]["items"]] == [1, 2, 3, 4, 5]
    assert "红中 568 我打" in payload["new_message"]["text"]
    assert "7点" in payload["new_message"]["text"]
    assert "群里其他人的原文" not in llm.calls[0]["messages"][-1]["content"]


def test_case_4_ambiguous_claim_stays_unmatched_and_switches_private() -> None:
    store = InMemoryAgentStore()
    llm = RecordingLLM(
        classification(
            intent="claim",
            matched_board_no=None,
            confidence=0.62,
            channel_action="private_switch",
            features={"stakes": "1块"},
        )
    )
    target = pipeline(store, llm)
    load_board(target)
    target.accept(message("1块无烟的我来", sender="someone", message_id="ambiguous"), trace_id="trace-a")

    outcomes = target.flush_due(at=NOW + timedelta(seconds=6))

    assert len(outcomes) == 1
    assert outcomes[0].action == "private_switch"
    assert outcomes[0].classification.matched_board_no is None
    assert outcomes[0].classification.confidence < 0.7
    assert len(llm.calls) == 1


def test_claim_contract_downgrades_model_match_when_stated_facts_match_multiple_items() -> None:
    store = InMemoryAgentStore()
    llm = RecordingLLM(
        classification(
            intent="claim",
            matched_board_no=1,
            confidence=0.95,
            channel_action="private_switch",
            features={"stakes": "1块", "smoking": "无烟"},
        )
    )
    target = pipeline(store, llm)
    load_board(target)
    target.accept(message("1块无烟的我来", sender="someone", message_id="wrong-unique"), trace_id="trace-wrong")

    outcome = target.flush_due(at=NOW + timedelta(seconds=6))[0]

    assert outcome.classification is not None
    assert outcome.classification.matched_board_no is None
    assert outcome.classification.confidence < 0.7
    assert "合同校验" in outcome.classification.reasoning


def test_query_session_accepts_followup_confirmation_and_becomes_formation() -> None:
    llm = RecordingLLM(
        classification(
            intent="query",
            matched_board_no=None,
            confidence=0.9,
            channel_action="group_reply",
            response="暂时没有。",
        ),
        classification(
            intent="thread_update",
            matched_board_no=None,
            confidence=0.9,
            channel_action="ignore",
        ),
    )
    target = pipeline(InMemoryAgentStore(), llm)
    target.accept(message("有人今晚打吗", sender="A", message_id="query"), trace_id="trace-query")
    first = target.flush_due(at=NOW + timedelta(seconds=6))[0]
    target.accept(
        message("+1", sender="B", message_id="followup", sent_at=NOW + timedelta(minutes=1)),
        trace_id="trace-followup",
    )
    second = target.flush_due(at=NOW + timedelta(minutes=1, seconds=6))[0]

    assert first.session_id == second.session_id
    session = target.session_router.list_sessions("room-1")[0]
    assert session.status == "active"
    assert session.topic_type == "formation"
    assert session.participants == {"A", "B"}


def test_case_5_chitchat_is_classified_then_ignored() -> None:
    llm = RecordingLLM(
        classification(
            intent="chitchat",
            matched_board_no=None,
            confidence=0.98,
            channel_action="ignore",
        )
    )
    target = pipeline(InMemoryAgentStore(), llm)
    target.accept(
        message("我以为是群主搞的高科技", sender="ninet", message_id="chat"),
        trace_id="trace-chat",
    )

    outcomes = target.flush_due(at=NOW + timedelta(seconds=6))

    assert len(outcomes) == 1
    assert outcomes[0].action == "ignore"
    assert outcomes[0].classification.intent == "chitchat"
    assert len(llm.calls) == 1


def test_case_6_repeated_laughter_is_filtered_before_accumulator_and_model() -> None:
    llm = RecordingLLM()
    target = pipeline(InMemoryAgentStore(), llm)

    result = target.accept(
        message("哈哈哈哈哈哈", sender="ninet", message_id="noise"),
        trace_id="trace-noise",
    )

    assert result.action == "filtered"
    assert target.flush_due(at=NOW + timedelta(seconds=30)) == []
    assert target.session_router.list_sessions("room-1") == []
    assert llm.calls == []


def test_case_7_unmatched_explicit_need_becomes_new_demand() -> None:
    llm = RecordingLLM(
        classification(
            intent="new_demand",
            matched_board_no=None,
            confidence=0.94,
            channel_action="private_switch",
            features={"game_type": "川麻换三", "stakes": "132", "time": "18:30"},
        )
    )
    target = pipeline(InMemoryAgentStore(), llm)
    target.accept(
        message("川麻换三 6.30 132 三缺一", sender="元宝", message_id="demand"),
        trace_id="trace-demand",
    )

    outcomes = target.flush_due(at=NOW + timedelta(seconds=6))

    assert outcomes[0].classification.intent == "new_demand"
    assert outcomes[0].action == "private_switch"


def test_case_8_related_sessions_merge_after_same_initiator_returns() -> None:
    llm = RecordingLLM(
        classification(
            intent="new_demand",
            matched_board_no=None,
            confidence=0.8,
            channel_action="private_switch",
            features={"time": "今晚"},
        ),
        classification(
            intent="thread_update",
            matched_board_no=None,
            confidence=0.9,
            channel_action="ignore",
        ),
        classification(
            intent="thread_update",
            matched_board_no=None,
            confidence=0.96,
            channel_action="ignore",
            features={"stakes": "1块", "time": "19:00", "smoking": "无烟"},
        ),
    )
    target = pipeline(InMemoryAgentStore(), llm)

    target.accept(message("有人今晚打吗", sender="A", message_id="m1"), trace_id="trace-m1")
    target.flush_due(at=NOW + timedelta(seconds=6))
    target.accept(
        message("+1", sender="B", message_id="m2", sent_at=NOW + timedelta(minutes=1)),
        trace_id="trace-m2",
    )
    target.flush_due(at=NOW + timedelta(minutes=1, seconds=6))
    target.accept(
        message("7点1块无烟", sender="A", message_id="m3", sent_at=NOW + timedelta(minutes=5)),
        trace_id="trace-m3",
    )
    target.flush_due(at=NOW + timedelta(minutes=5, seconds=6))

    sessions = target.session_router.list_sessions("room-1")
    active = [item for item in sessions if item.status == "active"]
    merged = [item for item in sessions if item.status == "merged"]
    assert len(active) == 1
    assert len(merged) == 1
    assert [item.message_id for item in active[0].messages] == ["m1", "m2", "m3"]
    assert active[0].participants == {"A", "B"}
    assert {key: active[0].extracted_state.get(key) for key in ("time", "stakes", "smoking")} == {
        "time": "19:00",
        "stakes": "1块",
        "smoking": "无烟",
    }


def test_case_9_stable_three_person_formation_crystallizes_once() -> None:
    published: list[str] = []
    outputs = [
        classification(
            intent="new_demand" if index == 0 else "thread_update",
            matched_board_no=None,
            confidence=0.9,
            channel_action="private_switch" if index == 0 else "ignore",
            features={"time": "19:00", "stakes": "1块", "smoking": "无烟"} if index == 3 else None,
        )
        for index in range(6)
    ]
    target = pipeline(
        InMemoryAgentStore(),
        RecordingLLM(*outputs),
        on_crystallized=lambda session: published.append(session.id),
    )
    inputs = [
        ("A", "有人打吗"),
        ("B", "+1"),
        ("C", "我也来"),
        ("A", "7点1块无烟"),
        ("B", "行"),
        ("C", "ok"),
    ]
    actions = []
    for index, (sender, text) in enumerate(inputs):
        sent_at = NOW + timedelta(minutes=index)
        target.accept(message(text, sender=sender, message_id=f"c{index}", sent_at=sent_at), trace_id=f"trace-c{index}")
        actions.extend(target.flush_due(at=sent_at + timedelta(seconds=6)))

    assert published and len(set(published)) == 1
    assert sum(item.action == "board_update" for item in actions) == 1
    session = next(item for item in target.session_router.list_sessions("room-1") if item.id == published[0])
    assert session.extracted_state["crystallized"] is True


def test_case_10_concurrent_topics_remain_isolated() -> None:
    llm = RecordingLLM(
        classification(
            intent="claim",
            matched_board_no=4,
            confidence=0.95,
            channel_action="private_switch",
            features={"game_type": "红中", "stakes": "568"},
        ),
        classification(
            intent="query",
            matched_board_no=None,
            confidence=0.9,
            channel_action="group_reply",
            features={"game_type": "cq"},
            response="cq还有两个局。",
        ),
    )
    target = pipeline(InMemoryAgentStore(), llm)
    load_board(target)
    target.accept(message("红中568我打", sender="德不孤", message_id="parallel-a"), trace_id="trace-pa")
    target.accept(message("cq还有吗", sender="另一个人", message_id="parallel-b"), trace_id="trace-pb")

    outcomes = target.flush_due(at=NOW + timedelta(seconds=6))

    assert {item.classification.intent for item in outcomes} == {"claim", "query"}
    assert len({item.session_id for item in outcomes}) == 2
    sessions = [item for item in target.session_router.list_sessions("room-1") if item.messages]
    assert {tuple(sorted(item.participants)) for item in sessions} == {("德不孤",), ("另一个人",)}
    first_payload = json.loads(llm.calls[0]["messages"][-1]["content"])
    second_payload = json.loads(llm.calls[1]["messages"][-1]["content"])
    assert first_payload["current_session"]["participants"] == ["德不孤"]
    assert second_payload["current_session"]["participants"] == ["另一个人"]
    assert "cq还有吗" not in llm.calls[0]["messages"][-1]["content"]
    assert "红中568我打" not in llm.calls[1]["messages"][-1]["content"]
