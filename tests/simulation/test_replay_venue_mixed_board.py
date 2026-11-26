"""复核看板混合查询评测的正式读写、收件和独立判定接线。"""

import json

import pytest

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.llm import StaticAgentClient

from scripts.replay_venue_mixed_board import _CHECKS, _evening_red_center, _execute, _seed_board_game, _source
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
        "start_at": "2026-11-26T19:00:00+08:00", "end_at": "2026-11-26T23:59:00+08:00",
    }}, "matches": [], "alternatives": []}
    assert _evening_red_center(result, _source().turns[0]) is accepted


@pytest.mark.parametrize(("start", "semantic", "expected"), [
    ("19:00", "pass", "passed"),
    ("19:00", "fail", "failed"),
    ("19:00", "missing_scope_citation", "inconclusive"),
    ("18:00", "pass", "failed"),
])
def test_mixed_board_replay_keeps_durable_scope_and_semantic_evidence(tmp_path, start, semantic, expected):
    """两类真实读取都正确才判实际话术，含糊指代和失去引用须分别处理。"""
    source = _source()
    reply = ("你问人齐开的1号川麻桌吗？那桌还缺三位。今晚七点后的一块红中暂时没有。"
             if semantic != "fail" else "你说的一缺就是那桌川麻，还差三位。今晚七点后的一块红中暂时没有。")
    action = StaticAgentClient([
        _query("specific_table", {"game_type": "sichuan_mahjong"}),
        _query("personal", {"game_type": "red_center_mahjong", "stake": "1", "start_time_range": {
            "start_at": f"2026-11-26T{start}:00+08:00", "end_at": "2026-11-26T23:59:00+08:00",
        }}),
        _reply(reply),
    ])
    store = SQLiteAgentStore(tmp_path / "agent.sqlite3")
    runtime, _ = build_runtime(store, "mock", offline_client=action, local_coordination=True)
    channel = SandboxChannel(runtime, room_id="venue-group", identities=source.identities,
                             archive_path=tmp_path / "chat.sqlite3")
    game = _seed_board_game(store, channel, source)
    findings = [{"check": check, "verdict": "fail" if semantic == "fail" and index == 2 else "pass",
                 "reason": "测试独立结果",
                 "evidence_ids": sorted(ids if semantic != "missing_scope_citation" else {"reply"})}
                for index, (check, ids) in enumerate(_CHECKS.items())]
    evaluator = StaticAgentClient([json.dumps({
        "verdict": "fail" if semantic == "fail" else "pass", "findings": findings,
    }, ensure_ascii=False)])
    result = _execute(store, channel, source, game, tmp_path / "events.jsonl", judge=SemanticJudge(evaluator))

    assert result["status"] == expected, result
    assert result["hard_checks_passed"] is (start == "19:00"), result
    assert result["checks"]["game_room_waiting_and_invite_state_unchanged"]
    assert result["checks"]["same_table_found_without_personal_window"]
    if start == "19:00":
        evidence = json.loads(evaluator.calls[0]["messages"][-1]["content"])["evidence"]
        facts = {row["message_id"]: row for row in evidence}
        assert "人齐开" in facts["board"]["text"] and "173" in facts["board"]["text"]
        assert facts["table-read"]["remaining_seats"] == 3
        assert facts["personal-read"]["match_count"] == 0
        assert facts["reply"]["text"] == reply
    else:
        assert evaluator.calls == []
