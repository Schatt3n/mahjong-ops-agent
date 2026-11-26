"""复核看板混合查询评测的正式读写、收件和独立判定接线。"""

import json

import pytest

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.llm import StaticAgentClient

from scripts.replay_venue_mixed_board import (
    _CHECKS, _CLEAR_REFERENCE_CHECKS, _RED_CENTER_OFFER, _RED_CENTER_RESULT,
    _evening_red_center, _execute, _seed_board_game, _source,
)
from .sim_adapter import build_runtime
from .venue_actors import SemanticJudge
from .venue_channel import SandboxChannel
from .venue_fixture import _call, _reply


def _query(scope, requirement):
    """给两个独立搜索写不同调用编号，照常交由正式工具处理。"""
    payload = json.loads(_call("search_current_games", {
        "lookup_scope": scope, "requirement": requirement, "include_nearby": False,
    }))
    payload["tool_calls"][0]["call_id"] = scope
    return json.dumps(payload)


@pytest.mark.parametrize(("stake", "accepted"), [
    ({"stake": "1"}, True),
    ({"stake_range": {"min": 1, "max": 1}}, True),
    ({"stake_range": {"min": 1, "max": 2}}, False),
    ({"stake": "1-16"}, False),
])
def test_personal_query_accepts_equivalent_fixed_yuan_only(stake, accepted):
    """同一固定底注的表达可等价，宽档或自添封顶不算原条件。"""
    result = {"requirement": {"game_type": "red_center_mahjong", **stake, "start_time_range": {
        "start_at": "2026-11-26T19:00:00+08:00", "end_at": "2026-11-27T06:00:00+08:00",
    }}, "matches": [], "alternatives": []}
    assert _evening_red_center(result, _source().turns[0]) is accepted


@pytest.mark.parametrize(("end_at", "accepted"), [
    ("2026-11-26T23:59:00+08:00", False),
    ("2026-11-27T02:00:00+08:00", False),
    ("2026-11-27T06:00:00+08:00", True),
    ("2026-11-27T12:00:00+08:00", False),
])
def test_empty_evening_search_requires_the_complete_overnight_window(end_at, accepted):
    result = {"requirement": {"game_type": "red_center_mahjong", "stake": "1", "start_time_range": {
        "start_at": "2026-11-26T19:00:00+08:00", "end_at": end_at,
    }}, "matches": []}
    assert _evening_red_center(result, _source().turns[0]) is accepted


def test_numbered_board_inquiry_is_default_and_original_ambiguous_text_is_preserved():
    clear = _source()
    ambiguous = _source(ambiguous_reference=True)
    assert clear.initial_game == ambiguous.initial_game
    first, = clear.turns
    original, = ambiguous.turns
    assert first.kind == original.kind == "private"
    assert first.quoted_message_id is original.quoted_message_id is None
    assert "群里1号" in first.text and "人齐开" in first.text and "一缺" not in first.text
    assert original.message_id == "customer:synthetic-mixed-board"
    assert original.text == "刚才群里说一缺的那桌，是川麻吗？另外今晚七点以后有一块底的红中吗？"


@pytest.mark.parametrize(("ambiguous", "named", "start", "semantic", "expected"), [
    (False, True, "19:00", "pass", "passed"),
    (False, True, "19:00", "wrong_reference", "failed"),
    (False, True, "19:00", "missing_offer", "failed"),
    (False, True, "19:00", "missing_scope_citation", "inconclusive"),
    (False, True, "18:00", "pass", "failed"),
    (False, False, "19:00", "pass", "failed"),
    (True, True, "19:00", "pass", "passed"),
    (True, True, "19:00", "wrong_reference", "failed"),
    (True, False, "19:00", "pass", "passed"),
    (True, False, "19:00", "missing_result", "failed"),
])
def test_mixed_board_replay_keeps_durable_scope_and_semantic_evidence(tmp_path, ambiguous, named, start, semantic, expected):
    """同一正式库存对照两种输入，各自判定实际送达的指代处理。"""
    source = _source(ambiguous_reference=ambiguous)
    checks = _CHECKS if ambiguous else _CLEAR_REFERENCE_CHECKS
    reference_check = next(check for check in checks if check not in {_RED_CENTER_RESULT, _RED_CENTER_OFFER})
    failed_check = {"wrong_reference": reference_check, "missing_offer": _RED_CENTER_OFFER,
                    "missing_result": _RED_CENTER_RESULT}.get(semantic)
    direct = "1号是川麻，还差三位。"
    clarify = "你问的是群里人齐开的1号川麻桌吗？"
    if ambiguous:
        board_reply = "你说的一缺就是那桌川麻，还差三位。" if semantic == "wrong_reference" else clarify
    else:
        board_reply = clarify if semantic == "wrong_reference" else direct
    reply = board_reply
    if semantic != "missing_result":
        reply += "今晚七点后一块底的红中暂时没有。"
        if not ambiguous and semantic != "missing_offer":
            reply += "要不我帮你组一桌？"
    action = StaticAgentClient([
        *([_query("specific_table", {"game_type": "sichuan_mahjong"})] if named else []),
        _query("personal", {"game_type": "red_center_mahjong", "stake": "1", "start_time_range": {
            "start_at": f"2026-11-26T{start}:00+08:00", "end_at": "2026-11-27T06:00:00+08:00",
        }}),
        _reply(reply),
    ])
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    runtime, _ = build_runtime(store, "mock", offline_client=action, local_coordination=True)
    channel = SandboxChannel(runtime, room_id="venue-group", identities=source.identities,
                             archive_path=tmp_path / "chat.sqlite3")
    game = _seed_board_game(store, channel, source)
    findings = [{"check": check, "verdict": "fail" if check == failed_check else "pass",
                 "reason": "测试独立结果",
                 "evidence_ids": sorted(ids if semantic != "missing_scope_citation" else {"reply"})}
                for check, ids in checks.items()]
    evaluator = StaticAgentClient([json.dumps({
        "verdict": "fail" if failed_check else "pass", "findings": findings,
    }, ensure_ascii=False)])
    result = _execute(store, channel, source, game, tmp_path / "events.jsonl", judge=SemanticJudge(evaluator))

    assert result["status"] == expected, result
    hard_pass = start == "19:00" and (named or ambiguous)
    assert result["hard_checks_passed"] is hard_pass, result
    assert result["fixture_source"]["scenario"] == ("ambiguous_reference_stress" if ambiguous else "clear_reference")
    assert result["checks"]["game_room_waiting_and_invite_state_unchanged"]
    assert result["checks"].get("same_table_found_without_personal_window") is (None if ambiguous and not named else named)
    if hard_pass:
        evidence = json.loads(evaluator.calls[0]["messages"][-1]["content"])["evidence"]
        facts = {row["message_id"]: row for row in evidence}
        assert facts["inbound"]["text"] == source.turns[0].text
        assert "人齐开" in facts["board"]["text"] and "173" in facts["board"]["text"]
        if named:
            assert facts["table-read"]["remaining_seats"] == 3
        else:
            assert "table-read" not in facts
        assert facts["personal-read"]["match_count"] == 0
        assert facts["reply"]["text"] == reply
        if failed_check:
            assert result["semantic_judgment"]["findings"][failed_check] == "fail"
    else:
        assert evaluator.calls == []
