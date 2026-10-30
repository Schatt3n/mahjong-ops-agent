"""合成回放按因果时间装配，固定未来锚与并发共享案例不被Runtime接收过程篡改。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import datetime as dt
import importlib.util
import json
from pathlib import Path
import sys
from threading import Lock
from types import SimpleNamespace

import pytest

from mahjong_agent_runtime import InMemoryAgentStore, QuotedMessageRef, SQLiteAgentStore, StaticAgentClient, UserMessage
from mahjong_agent_runtime.models import ConversationRole, ConversationTurn


@pytest.fixture
def owner_eval():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_real_owner_chat_live_eval.py"
    spec = importlib.util.spec_from_file_location("fixture_chronology_owner_eval", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    value = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "fixture-time.sqlite3")
    yield value
    if isinstance(value, SQLiteAgentStore):
        value._connection.close()


def test_explicit_authorization_history_precedes_assistant_and_cloned_current(owner_eval, store):
    scene = next(case for case in owner_eval.live_eval_scenarios() if case.scenario_id == "explicit_stake_conflict_must_not_loop")
    original = deepcopy(scene.message.to_dict())
    scene.setup(store)
    history = store.recent_turns(scene.message.conversation_id)
    assert [turn.role for turn in history] == [ConversationRole.USER, ConversationRole.ASSISTANT]
    cloned = owner_eval.scenario_after_seeded_history(scene, store)
    assert history[0].occurred_at < history[1].occurred_at < cloned.message.sent_at
    assert history[0].content == "三缺一，帮我约个无烟局"
    assert cloned.message.text == original["text"] and cloned.message.message_id == original["message_id"]
    assert scene.message.to_dict() == original and cloned is not scene and cloned.message is not scene.message


@pytest.mark.parametrize("case_id", ["flexible_time_followup_matches_evening_game", "flexible_time_empty_pool_offers_help"])
def test_fixed_future_timeline_keeps_original_anchor_and_prior_turn_order(owner_eval, store, case_id):
    scene = next(case for case in owner_eval.live_eval_scenarios() if case.scenario_id == case_id)
    original = deepcopy(scene.message.to_dict())
    scene.setup(store)
    history = store.recent_turns(scene.message.conversation_id)
    assert len(history) == 2
    assert history[0].occurred_at < history[1].occurred_at < scene.message.sent_at
    assert history[0].occurred_at.date() == history[1].occurred_at.date() == scene.message.sent_at.date()
    cloned = owner_eval.scenario_after_seeded_history(scene, store, received_at=scene.message.sent_at-dt.timedelta(hours=1))
    assert cloned.message.sent_at == scene.message.sent_at == owner_eval.FLEXIBLE_TIME_FIXTURE_ANCHOR
    assert scene.message.to_dict() == original
    if case_id == "flexible_time_followup_matches_evening_game":
        game = next(iter(store.games.values()))
        assert game.planned_start_at.date() == scene.message.sent_at.date()
        assert game.planned_start_at.hour == 19


def test_scenario_clone_preserves_text_id_and_deeply_isolates_nested_input(owner_eval, store):
    at = dt.datetime.now().astimezone()
    message = UserMessage(conversation_id="isolation", sender_id="customer", sender_name="客户", text="原始输入", message_id="original-id",
        sent_at=at, metadata={"nested": {"value": "original"}},
        quoted_message=QuotedMessageRef(message_id="quote", text="引用原文", metadata={"nested": {"value": "quote"}}))
    scene = owner_eval.LiveEvalScenario(scenario_id="isolated", name="隔离", message=message, setup=lambda store: None)
    original = deepcopy(message.to_dict())
    cloned = owner_eval.scenario_after_seeded_history(scene, store, received_at=at+dt.timedelta(seconds=1))
    cloned.message.metadata["nested"]["value"] = "changed"
    cloned.message.quoted_message.metadata["nested"]["value"] = "changed"
    assert scene.message.to_dict() == original
    assert cloned.message.text == message.text and cloned.message.message_id == message.message_id


def test_run_scenario_parallel_repeats_do_not_mutate_shared_scenario(owner_eval, tmp_path, monkeypatch):
    """真实Runtime仅用明确标记的协议替身，测试时间装配与隔离，不声明真实模型语义通过。"""
    stamp = dt.datetime.now().astimezone()-dt.timedelta(minutes=2)
    def setup(current_store):
        current_store.append_user_turn(UserMessage(conversation_id="shared", sender_id="customer", sender_name="客户",
            text="前轮委托", message_id="prior", sent_at=stamp), "prior")
        current_store.append_turn("shared", ConversationTurn(role=ConversationRole.ASSISTANT, content="补充条件？",
            trace_id="prior-assistant", occurred_at=stamp+dt.timedelta(minutes=1)))
    scene = owner_eval.LiveEvalScenario(scenario_id="shared-timing", name="共享回放输入隔离", setup=setup,
        message=UserMessage(conversation_id="shared", sender_id="customer", sender_name="客户", text="本轮补充", message_id="current",
            sent_at=stamp, metadata={"nested": {"value": "original"}}))
    original = deepcopy(scene.message.to_dict())
    monkeypatch.setattr(owner_eval, "reply_semantic_expectation", lambda scenario: {
        "required_meaning": [{"id": "unit", "description": "只测试执行机制"}],
        "product_contract_revision_ids": [],
    })
    monkeypatch.setattr(owner_eval, "assess_reply_semantics", lambda *args, **kwargs: {"passed": True, "error": None})
    action = json.dumps({"goal": "离线机制测试", "objective_status": "completed", "reasoning_summary": "确定性协议替身",
        "reply_to_user": "收到。", "tool_calls": [], "needs_human": False,
        "stop_reason": {"can_stop": True, "why": "本轮结束", "pending_work": [], "depends_on_tool_results": False}})
    client = StaticAgentClient(outputs=[action, action])
    clones, lock = [], Lock()
    real_clone = owner_eval.scenario_after_seeded_history
    def capture(*args, **kwargs):
        cloned = real_clone(*args, **kwargs)
        with lock:
            clones.append(cloned)
        return cloned
    monkeypatch.setattr(owner_eval, "scenario_after_seeded_history", capture)
    def replay(index):
        args = SimpleNamespace(db_path=tmp_path / f"run-{index}" / "owner.sqlite3", max_tokens_per_call=64000,
            max_calls_per_turn=4, max_steps=3, timeout_seconds=1, skip_text_generation=True, skip_review=True)
        return owner_eval.run_scenario(client, args, scene, iteration=index)
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports = list(pool.map(replay, [1, 2]))
    assert scene.message.to_dict() == original
    assert len(clones) == 2 and clones[0].message is not clones[1].message
    assert clones[0].message.metadata["nested"] is not clones[1].message.metadata["nested"]
    for report in reports:
        assert report["status"] == "passed", report["checks"]
        assert report["input"]["text"] == original["text"] and report["input"]["message_id"] == original["message_id"]
        assert dt.datetime.fromisoformat(report["input"]["sent_at"]) > stamp+dt.timedelta(minutes=1)
        raw = json.loads(Path(report["raw_trace_path"]).read_text())
        context = next(event["content"] for event in raw["events"] if event["step"] == "context_built")
        assert [turn["content"] for turn in context["recent_conversation"] if turn["role"] in {"user", "assistant"}] == [
            "前轮委托", "补充条件？", "本轮补充"]
