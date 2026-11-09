"""验证候选人协商可跨对话轮恢复，桌主答复与回告责任原子持久化。"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from threading import Barrier

import pytest

from mahjong_agent_runtime import InMemoryAgentStore, ScheduledAgentTask, SQLiteAgentStore
from mahjong_agent_runtime.models import MessageReference, ScheduledTaskStatus, UserMessage, now
from mahjong_agent_runtime.stores.task_store import (
    BOARD_CANDIDATE_DECISION_TASK_TYPE,
    BOARD_CANDIDATE_UPDATE_TASK_TYPE,
)


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryAgentStore()
    else:
        result = SQLiteAgentStore(tmp_path / "candidate-decisions.db")
        try:
            yield result
        finally:
            result._connection.close()


def register(store, *, candidate="B", game=None, question_id="host-question"):
    if game is None:
        game, _ = store.create_game(conversation_id="group:room:A", organizer_id="A", organizer_name="甲",
            requirement={"game_type": "hangzhou_mahjong", "recruitment_mode": "board_only", "stake": "1"},
            known_players=[], trace_id="game-created")
    task_id = f"decision:{game.game_id}:{candidate}"
    task, _ = store.ensure_board_candidate_decision_task(ScheduledAgentTask(
        task_id=task_id, task_type=BOARD_CANDIDATE_DECISION_TASK_TYPE, aggregate_type="game",
        aggregate_id=game.game_id, conversation_id="private:A", subject_id="A", subject_name="甲",
        due_at=now(), idempotency_key=task_id, payload={"room_id": "room",
            "expires_at": (now() + timedelta(hours=1)).isoformat(), "candidate_id": candidate,
            "candidate_conversation_id": f"private:{candidate}", "inquiry_trace_id": f"inquiry:{candidate}",
            "question_message_id": question_id}), trace_id=f"queued:{candidate}")
    return game, task


def deliver_question(store, task, *, conversation_id="private:A", candidate_task_id=None):
    store.register_message_reference(MessageReference(message_id=task.payload["question_message_id"],
        conversation_id=conversation_id, recipient_id="A", business_ref_type="group_board_stake_confirmation",
        business_ref_id=task.aggregate_id, text="有个男生想打，你们可以吗？",
        metadata={"delivery_id": "delivery-1", "candidate_decision_task_id": candidate_task_id or task.task_id}))


def answer(store, *, text="男生的话也行吧", message_id="host-answer", sender="A", trace_id="earlier-trace",
           metadata=None):
    store.append_user_turn(UserMessage(conversation_id="private:A", sender_id=sender, sender_name="甲",
        text=text, message_id=message_id, metadata=metadata or {}), trace_id=trace_id)


def proposed_update(task, **changes):
    recipient = task.payload["candidate_id"]
    task_id = f"update:{task.aggregate_id}:{recipient}"
    payload = {key: task.payload[key] for key in ("room_id", "expires_at", "inquiry_trace_id")}
    payload["decision_task_id"] = task.task_id
    return replace(ScheduledAgentTask(task_id=task_id, task_type=BOARD_CANDIDATE_UPDATE_TASK_TYPE,
        aggregate_type="game", aggregate_id=task.aggregate_id, conversation_id=f"private:{recipient}",
        subject_id=recipient, subject_name=recipient, due_at=task.due_at, idempotency_key=task_id,
        payload=payload), **changes)


def resolve(store, task, *, evidence="host-answer", decision="accepted", update=None, at=None):
    return store.resolve_board_candidate_decision(task.task_id, host_id="A", conversation_id="private:A",
        decision=decision, evidence_message_id=evidence, update_task=update, trace_id="later-trace", at=at)


def test_earlier_answer_completes_its_own_candidate_and_wakes_followup(store):
    game, task = register(store)
    _, other_task = register(store, game=game, candidate="C", question_id="other-question")
    original_game = game.to_dict()
    assert store.claim_scheduled_task(task.task_id, at=now()) is None
    deliver_question(store, task)
    answer(store)
    answer(store, text="他这人干净吗", message_id="side-question", trace_id="later-trace")
    update = proposed_update(task)
    saved, changes = resolve(store, task, update=update)
    assert saved.status == ScheduledTaskStatus.COMPLETED
    assert saved.payload["evidence_trace_id"] == "earlier-trace"
    assert saved.payload["evidence_message_id"] == "host-answer"
    assert saved.payload["question_delivery_id"] == "delivery-1"
    assert [change.reason for change in changes] == ["board_candidate_accepted", "board_candidate_update_ready"]
    assert store.scheduled_tasks[other_task.task_id].status == ScheduledTaskStatus.PENDING
    assert [ready.task_id for ready in store.due_scheduled_tasks(at=now())] == [update.task_id]
    assert store.scheduled_tasks[update.task_id].payload["decision"] == "accepted"
    assert store.require_game(game.game_id).to_dict() == original_game
    assert resolve(store, task, update=update)[1] == []


@pytest.mark.parametrize("invalid", ["before_receipt", "recalled", "wrong_receipt_session", "other_candidate",
                                     "wrong_update_session", "wrong_sender", "group_event", "system_event"])
def test_invalid_evidence_or_update_does_not_consume_pending_decision(store, invalid):
    _, task = register(store)
    if invalid == "before_receipt":
        answer(store)
    deliver_question(store, task, conversation_id="private:C" if invalid == "wrong_receipt_session" else "private:A",
        candidate_task_id="different-candidate" if invalid == "other_candidate" else None)
    if invalid != "before_receipt":
        metadata = {"group_event": {"source": "group"}, "system_event": {"input_source": "system_trigger"}}
        answer(store, sender="C" if invalid == "wrong_sender" else "A", metadata=metadata.get(invalid))
    if invalid == "recalled":
        assert store.invalidate_message_context("private:A", "A", "host-answer", trace_id="recall")
    update = proposed_update(task, conversation_id="private:C") if invalid == "wrong_update_session" else proposed_update(task)
    with pytest.raises(ValueError):
        resolve(store, task, update=update)
    assert store.scheduled_tasks[task.task_id].status == ScheduledTaskStatus.PENDING
    assert update.task_id not in store.scheduled_tasks


def test_decline_and_duplicate_registration_never_schedule_candidate_update(store):
    _, task = register(store)
    deliver_question(store, task)
    answer(store, text="这次只要女生")
    saved, changes = resolve(store, task, decision="declined")
    assert saved.payload["decision"] == "declined"
    assert [change.reason for change in changes] == ["board_candidate_declined"]
    duplicate, changes = store.ensure_board_candidate_decision_task(replace(task, task_id="another-id",
        idempotency_key="another-id", payload={**task.payload, "question_message_id": "another-question"}), trace_id="retry")
    assert duplicate.to_dict() == saved.to_dict() and changes == []
    assert not any(item.task_type == BOARD_CANDIDATE_UPDATE_TASK_TYPE for item in store.scheduled_tasks.values())


def test_expired_candidate_decision_cannot_enqueue_notification(store):
    _, task = register(store)
    deliver_question(store, task)
    answer(store)
    update = proposed_update(task)
    with pytest.raises(ValueError, match="no longer pending"):
        resolve(store, task, update=update, at=task.due_at)
    assert update.task_id not in store.scheduled_tasks


def test_sqlite_rolls_back_candidate_decision_and_update_together(tmp_path, monkeypatch):
    path = tmp_path / "rollback-candidate.db"
    store = SQLiteAgentStore(path)
    try:
        _, task = register(store)
        deliver_question(store, task)
        answer(store)
        update = proposed_update(task)
        with monkeypatch.context() as patch:
            append_transition = type(store)._append_transition

            def fail_on_update(instance, transition):
                if transition.reason == "board_candidate_update_ready":
                    raise RuntimeError("controlled transition failure")
                append_transition(instance, transition)

            patch.setattr(type(store), "_append_transition", fail_on_update)
            with pytest.raises(RuntimeError, match="controlled transition failure"):
                resolve(store, task, update=update)
        assert store.scheduled_tasks[task.task_id].status == ScheduledTaskStatus.PENDING
        assert update.task_id not in store.scheduled_tasks
        assert resolve(store, task, update=update)[0].status == ScheduledTaskStatus.COMPLETED
    finally:
        store._connection.close()

    restarted = SQLiteAgentStore(path)
    try:
        assert restarted.scheduled_tasks[task.task_id].payload["evidence_message_id"] == "host-answer"
        assert restarted.scheduled_tasks[update.task_id].status == ScheduledTaskStatus.PENDING
        assert resolve(restarted, task, update=update)[1] == []
    finally:
        restarted._connection.close()


def test_sqlite_two_connections_only_one_resolves_same_candidate(tmp_path):
    path = tmp_path / "race-candidate.db"
    stores = [SQLiteAgentStore(path), SQLiteAgentStore(path)]
    try:
        _, task = register(stores[0])
        deliver_question(stores[0], task)
        answer(stores[0])
        update = proposed_update(task)
        ready = Barrier(2)

        def complete(store):
            ready.wait(timeout=5)
            return resolve(store, task, update=update)[1]

        with ThreadPoolExecutor(max_workers=2) as pool:
            changes = list(pool.map(complete, stores))
        assert sorted(map(len, changes)) == [0, 2]
        assert [item.task_id for item in stores[0].due_scheduled_tasks(at=now())] == [update.task_id]
    finally:
        for store in stores:
            store._connection.close()
