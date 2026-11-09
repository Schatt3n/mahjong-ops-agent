"""验证档位确认责任的持久化与幂等；是否应询问以及问谁由业务工具另行验证。"""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta
from threading import Barrier
from zoneinfo import ZoneInfo

import pytest

from mahjong_agent_runtime import InMemoryAgentStore, ScheduledAgentTask, SQLiteAgentStore
from mahjong_agent_runtime.models import ScheduledTaskStatus, now


STAMP = datetime(2026, 11, 9, 17, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    result = InMemoryAgentStore() if request.param == "memory" else SQLiteAgentStore(tmp_path / "confirmation.db")
    yield result
    if request.param == "sqlite":
        result._connection.close()


def task(*, game_id="game-a", task_id=None, requester="customer-b", due_at=STAMP):
    task_id = task_id or f"board-stake:{game_id}"
    return ScheduledAgentTask(
        task_id=task_id, task_type="group_board_stake_confirmation", aggregate_type="game",
        aggregate_id=game_id, conversation_id="organizer-conversation", subject_id="organizer-a",
        subject_name="发局人A", due_at=due_at, idempotency_key=task_id,
        payload={"game_id": game_id, "requester_id": requester, "question": {"field": "stake"}},
        created_at=STAMP, updated_at=STAMP,
    )


def test_same_confirmation_task_retry_preserves_original_due_payload_and_transition(store):
    first = task()
    saved, changes = store.ensure_board_stake_confirmation_task(first, trace_id="first")
    assert saved.to_dict() == first.to_dict() and len(changes) == 1
    assert changes[0].entity_type == "scheduled_agent_task"
    assert changes[0].entity_id == first.task_id and changes[0].from_status is None
    assert changes[0].to_status == "pending" and changes[0].reason == "board_stake_confirmation_scheduled"
    original = deepcopy(saved.to_dict())
    retry = task(requester="customer-c", due_at=STAMP + timedelta(hours=1))
    repeated, retry_changes = store.ensure_board_stake_confirmation_task(retry, trace_id="retry")
    assert repeated.to_dict() == original and retry_changes == []
    assert len(store.scheduled_tasks) == 1 and len(store.transitions) == 1


def test_another_requester_cannot_duplicate_confirmation_even_with_another_task_id(store):
    first, _ = store.ensure_board_stake_confirmation_task(task(), trace_id="first")
    another, changes = store.ensure_board_stake_confirmation_task(
        task(task_id="accidental-per-requester-id", requester="customer-c"), trace_id="second",
    )
    assert another.to_dict() == first.to_dict() and changes == []
    assert list(store.scheduled_tasks) == [first.task_id]


@pytest.mark.parametrize("state", ["processing", "completed", "failed"])
def test_confirmation_retry_never_resets_lease_attempts_or_terminal_state(store, state):
    saved, _ = store.ensure_board_stake_confirmation_task(task(), trace_id="first")
    claim = store.claim_scheduled_task(saved.task_id, at=STAMP)
    assert claim is not None
    if state == "completed":
        store.complete_scheduled_task(saved.task_id, trace_id="done", expected_attempts=claim.attempts,
            expected_lease_until=claim.lease_until, at=STAMP + timedelta(seconds=1))
    elif state == "failed":
        store.fail_scheduled_task(saved.task_id, trace_id="failed", error="controlled failure",
            expected_attempts=claim.attempts, expected_lease_until=claim.lease_until,
            max_attempts=1, at=STAMP + timedelta(seconds=1))
    before = deepcopy(store.scheduled_tasks[saved.task_id].to_dict())
    transition_count = len(store.transitions)
    retried, changes = store.ensure_board_stake_confirmation_task(
        task(requester="customer-c", due_at=STAMP + timedelta(hours=2)), trace_id="late-retry",
    )
    assert retried.to_dict() == before and retried.status.value == state
    assert changes == [] and len(store.transitions) == transition_count


def test_confirmation_input_and_returned_objects_do_not_mutate_authoritative_task(store):
    supplied = task()
    saved, _ = store.ensure_board_stake_confirmation_task(supplied, trace_id="first")
    original = deepcopy(saved.to_dict())
    supplied.payload["question"]["field"] = "different-source"
    supplied.due_at += timedelta(hours=3)
    saved.payload["question"]["field"] = "different-return"
    assert store.scheduled_tasks[saved.task_id].to_dict() == original


@pytest.mark.parametrize("changes", [
    {"task_type": "conversation_follow_up"},
    {"aggregate_type": "customer"},
    {"aggregate_id": ""},
    {"task_id": ""},
    {"status": ScheduledTaskStatus.PROCESSING},
])
def test_confirmation_registration_rejects_wrong_scope_or_new_nonpending_task(store, changes):
    with pytest.raises(ValueError):
        store.ensure_board_stake_confirmation_task(replace(task(), **changes), trace_id="invalid")
    assert not store.scheduled_tasks and not store.transitions


@pytest.mark.parametrize("collision", ["task_id", "idempotency_key"])
def test_confirmation_identity_collision_cannot_overwrite_another_game(store, collision):
    first, _ = store.ensure_board_stake_confirmation_task(task(), trace_id="first")
    conflicting = replace(task(game_id="game-b"), **{collision: getattr(first, collision)})
    with pytest.raises(ValueError, match="identity conflict"):
        store.ensure_board_stake_confirmation_task(conflicting, trace_id="conflict")
    assert len(store.scheduled_tasks) == 1 and len(store.transitions) == 1
    assert store.scheduled_tasks[first.task_id].to_dict() == first.to_dict()


def test_separate_games_have_independent_confirmation_responsibilities(store):
    first, _ = store.ensure_board_stake_confirmation_task(task(), trace_id="first")
    second, changes = store.ensure_board_stake_confirmation_task(task(game_id="game-b"), trace_id="second")
    assert first.task_id != second.task_id and len(changes) == 1
    assert len(store.scheduled_tasks) == 2 and len(store.transitions) == 2


def test_sqlite_two_connections_cannot_create_duplicate_confirmation_for_different_requesters(tmp_path):
    path = tmp_path / "concurrent-confirmation.db"
    stores = [SQLiteAgentStore(path), SQLiteAgentStore(path)]
    barrier = Barrier(2)

    def register(index):
        barrier.wait(timeout=5)
        return stores[index].ensure_board_stake_confirmation_task(
            task(task_id=f"candidate-task-{index}", requester=f"requester-{index}"), trace_id=f"concurrent-{index}",
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(register, (0, 1)))
        assert results[0][0].task_id == results[1][0].task_id
        assert sum(len(changes) for _, changes in results) == 1
        assert len(stores[0].scheduled_tasks) == 1 and len(stores[1].transitions) == 1
        restarted = SQLiteAgentStore(path)
        try:
            replay, changes = restarted.ensure_board_stake_confirmation_task(task(), trace_id="restart")
            assert replay.task_id == results[0][0].task_id and changes == []
            assert len(restarted.scheduled_tasks) == 1 and len(restarted.transitions) == 1
        finally:
            restarted._connection.close()
    finally:
        for item in stores:
            item._connection.close()


def test_sqlite_confirmation_task_and_transition_roll_back_together(tmp_path, monkeypatch):
    store = SQLiteAgentStore(tmp_path / "rollback-confirmation.db")

    def fail_transition(self, transition):
        raise RuntimeError("controlled transition failure")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(type(store), "_append_transition", fail_transition)
            with pytest.raises(RuntimeError, match="controlled transition failure"):
                store.ensure_board_stake_confirmation_task(task(), trace_id="failed-insert")
        assert not store.scheduled_tasks and not store.transitions
        saved, changes = store.ensure_board_stake_confirmation_task(task(), trace_id="retry")
        assert saved.task_id == task().task_id and len(changes) == 1
    finally:
        store._connection.close()


def update_task(*, game_id="game-a", recipient="customer-b", expires_at=None, task_id=None):
    expires_at = expires_at or now() + timedelta(hours=2)
    task_id = task_id or f"board-stake-update:{game_id}:{recipient}"
    return ScheduledAgentTask(
        task_id=task_id, task_type="group_board_stake_update", aggregate_type="game", aggregate_id=game_id,
        conversation_id=f"private:{recipient}", subject_id=recipient, subject_name=recipient,
        due_at=expires_at, idempotency_key=task_id,
        payload={"expires_at": expires_at.isoformat(), "room_id": "room-1", "source_message_id": f"inquiry:{recipient}"},
    )


def board_game(store, *, stake=None, board_only=True):
    requirement = {"game_type": "hangzhou_mahjong"}
    if board_only:
        requirement["recruitment_mode"] = "board_only"
    if stake is not None:
        requirement["stake"] = stake
    return store.create_game(conversation_id="group:room-1:customer:organizer-a", organizer_id="organizer-a",
        organizer_name="发局人A", requirement=requirement, known_players=[], trace_id="board-fixture")[0]


def test_question_and_requester_update_are_registered_together_without_refreshing(store):
    proposed_update = update_task()
    saved, transitions = store.ensure_board_stake_confirmation_task(task(), update_task=proposed_update, trace_id="both")
    assert saved.task_id == task().task_id and len(transitions) == 2
    assert store.scheduled_tasks[proposed_update.task_id].to_dict() == proposed_update.to_dict()
    before = {key: deepcopy(value.to_dict()) for key, value in store.scheduled_tasks.items()}
    changed_update = update_task(task_id="new-accidental-id", expires_at=now() + timedelta(hours=3))
    _, changes = store.ensure_board_stake_confirmation_task(
        task(due_at=STAMP + timedelta(hours=1)), update_task=changed_update, trace_id="both-retry",
    )
    assert changes == [] and {key: value.to_dict() for key, value in store.scheduled_tasks.items()} == before


def test_requesters_share_one_host_question_but_keep_independent_update_tasks(store):
    first = update_task(recipient="customer-b")
    second = update_task(recipient="customer-c")
    owner, _ = store.ensure_board_stake_confirmation_task(task(), update_task=first, trace_id="b")
    repeated_owner, transitions = store.ensure_board_stake_confirmation_task(task(), update_task=second, trace_id="c")
    assert owner.to_dict() == repeated_owner.to_dict() and len(transitions) == 1
    assert len(store.scheduled_tasks) == 3
    assert store.scheduled_tasks[first.task_id].conversation_id == "private:customer-b"
    assert store.scheduled_tasks[second.task_id].conversation_id == "private:customer-c"
    assert "customer-c" not in str(store.scheduled_tasks[first.task_id].payload)
    assert "customer-b" not in str(store.scheduled_tasks[second.task_id].payload)


@pytest.mark.parametrize("invalid", ["wrong_game", "wrong_type", "missing_expiry", "naive_expiry", "same_id", "not_pending"])
def test_invalid_update_registration_cannot_leave_an_orphan_question(store, invalid):
    proposed = update_task()
    if invalid == "wrong_game":
        proposed.aggregate_id = "another-game"
    elif invalid == "wrong_type":
        proposed.task_type = "conversation_follow_up"
    elif invalid == "missing_expiry":
        del proposed.payload["expires_at"]
    elif invalid == "naive_expiry":
        proposed.payload["expires_at"] = "2026-11-09T20:00:00"
    elif invalid == "same_id":
        proposed.task_id = task().task_id
    else:
        proposed.status = ScheduledTaskStatus.COMPLETED
    with pytest.raises(ValueError):
        store.ensure_board_stake_confirmation_task(task(), update_task=proposed, trace_id="invalid-update")
    assert not store.scheduled_tasks and not store.transitions


def test_already_known_stake_registers_ready_update_without_new_host_question(store):
    game = board_game(store, stake="1")
    proposed = update_task(game_id=game.game_id)
    before = now()
    question, transitions = store.ensure_board_stake_confirmation_task(task(game_id=game.game_id),
        update_task=proposed, trace_id="price-raced-ahead")
    saved = store.scheduled_tasks[proposed.task_id]
    assert question.status == ScheduledTaskStatus.COMPLETED and question.completed_at is not None
    assert saved.status == ScheduledTaskStatus.PENDING and before <= saved.due_at <= now()
    assert saved.due_at < proposed.due_at and proposed.due_at.isoformat() == proposed.payload["expires_at"]
    assert any(item.reason == "board_stake_confirmation_already_known" for item in transitions)


def fit_question(game_id, *, candidate="customer-b", task_id=None):
    stamp = now()
    return replace(task(game_id=game_id, task_id=task_id or f"board-fit:{game_id}:{candidate}", due_at=stamp),
        payload={"question_kind": "candidate_fit_only", "candidate_id": candidate,
            "candidate_conversation_id": f"private:{candidate}", "candidate_gender": "男",
            "inquiry_trace_id": f"inquiry:{candidate}", "inquiry_source_message_id": f"message:{candidate}",
            "expires_at": (stamp + timedelta(hours=2)).isoformat(), "room_id": "room-1"})


@pytest.mark.parametrize("fit_state", ["pending", "processing", "completed"])
def test_completed_price_question_does_not_block_separate_fit_question_or_refresh_updates(store, fit_state):
    game = board_game(store)
    proposed_update = update_task(game_id=game.game_id)
    question, _ = store.ensure_board_stake_confirmation_task(task(game_id=game.game_id),
        update_task=proposed_update, trace_id="original-price-question")
    claim = store.claim_scheduled_task(question.task_id, at=STAMP)
    store.complete_scheduled_task(question.task_id, trace_id="price-question-delivered",
        expected_attempts=claim.attempts, expected_lease_until=claim.lease_until, at=STAMP + timedelta(seconds=1))
    store.update_game_requirement(game_id=game.game_id, requirement_patch={"stake": "1"},
        reason="桌主已确认价格", trace_id="price-confirmed")
    original = {key: deepcopy(value.to_dict()) for key, value in store.scheduled_tasks.items()}

    proposed_fit = fit_question(game.game_id)
    fit, transitions = store.ensure_board_stake_confirmation_task(proposed_fit, trace_id="fit-question")
    assert fit.status == ScheduledTaskStatus.PENDING and fit.completed_at is None
    assert fit.task_id != question.task_id and fit.payload["question_kind"] == "candidate_fit_only"
    assert len(transitions) == 1 and transitions[0].entity_id == fit.task_id
    assert {key: store.scheduled_tasks[key].to_dict() for key in original} == original

    if fit_state != "pending":
        fit_claim = store.claim_scheduled_task(fit.task_id, at=fit.due_at)
        if fit_state == "completed":
            store.complete_scheduled_task(fit.task_id, trace_id="fit-question-delivered",
                expected_attempts=fit_claim.attempts, expected_lease_until=fit_claim.lease_until,
                at=fit.due_at + timedelta(seconds=1))
    before_retry = {key: deepcopy(value.to_dict()) for key, value in store.scheduled_tasks.items()}
    retry = replace(proposed_fit, task_id="accidental-second-fit", idempotency_key="accidental-second-fit",
        due_at=proposed_fit.due_at + timedelta(hours=1), payload={**proposed_fit.payload,
            "inquiry_trace_id": "new-inquiry", "inquiry_source_message_id": "new-input"})
    repeated, changes = store.ensure_board_stake_confirmation_task(retry, trace_id="fit-retry")
    assert repeated.task_id == fit.task_id and repeated.status.value == fit_state
    assert changes == [] and {key: value.to_dict() for key, value in store.scheduled_tasks.items()} == before_retry


def test_question_kind_cannot_reuse_another_kind_identity(store):
    question, _ = store.ensure_board_stake_confirmation_task(task(), trace_id="price")
    fit = fit_question(question.aggregate_id, task_id=question.task_id)
    with pytest.raises(ValueError, match="identity conflict"):
        store.ensure_board_stake_confirmation_task(fit, trace_id="same-identity-different-purpose")
    assert len(store.scheduled_tasks) == 1 and store.scheduled_tasks[question.task_id].to_dict() == question.to_dict()


def test_same_game_fit_questions_are_independent_per_candidate_and_do_not_enqueue_price_updates(store):
    game = board_game(store, stake="1")
    one, _ = store.ensure_board_stake_confirmation_task(fit_question(game.game_id), trace_id="first")
    two, _ = store.ensure_board_stake_confirmation_task(fit_question(game.game_id, candidate="customer-c"), trace_id="second")
    assert one.task_id != two.task_id and one.status == two.status == ScheduledTaskStatus.PENDING
    assert one.payload["candidate_id"] == "customer-b" and two.payload["candidate_id"] == "customer-c"
    assert set(store.scheduled_tasks) == {one.task_id, two.task_id}
    wrong_binding = replace(one, payload={**one.payload, "candidate_conversation_id": two.payload["candidate_conversation_id"]})
    with pytest.raises(ValueError, match="identity conflict"):
        store.ensure_board_stake_confirmation_task(wrong_binding, trace_id="mismatched-retry")
    assert set(store.scheduled_tasks) == {one.task_id, two.task_id}


@pytest.mark.parametrize("invalid", ["with_price_update", "missing_source", "invalid_gender", "no_price", "wrong_host", "expired"])
def test_invalid_fit_question_leaves_no_task(store, invalid):
    game = board_game(store, stake=None if invalid == "no_price" else "1")
    fit = fit_question(game.game_id)
    if invalid == "missing_source":
        del fit.payload["inquiry_source_message_id"]
    elif invalid == "invalid_gender":
        fit.payload["candidate_gender"] = "未知"
    elif invalid == "wrong_host":
        fit.subject_id = "different-host"
    elif invalid == "expired":
        fit.payload["expires_at"] = (now() - timedelta(minutes=1)).isoformat()
    extra = update_task(game_id=game.game_id) if invalid == "with_price_update" else None
    with pytest.raises(ValueError):
        store.ensure_board_stake_confirmation_task(fit, update_task=extra, trace_id="invalid-fit")
    assert not store.scheduled_tasks


@pytest.mark.parametrize("patch", [{"stake": "1"}, {"stake_range": {"min": 1, "max": 3}}])
def test_first_known_board_stake_atomically_activates_each_requester_once(store, patch):
    game = board_game(store)
    updates = [update_task(game_id=game.game_id, recipient=recipient) for recipient in ("customer-b", "customer-c")]
    for proposed in updates:
        store.ensure_board_stake_confirmation_task(task(game_id=game.game_id), update_task=proposed, trace_id="register")
    question_before = store.scheduled_tasks[f"board-stake:{game.game_id}"].to_dict()
    changed, _ = store.update_game_requirement(game_id=game.game_id, requirement_patch=patch, reason="桌主确认档位", trace_id="price")
    snapshots = {}
    for proposed in updates:
        current = store.scheduled_tasks[proposed.task_id]
        assert current.due_at == changed.updated_at and current.due_at < proposed.due_at
        assert current.payload == proposed.payload and current.idempotency_key == proposed.idempotency_key
        assert current.attempts == 0 and current.lease_until is None
        snapshots[current.task_id] = deepcopy(current.to_dict())
    assert store.scheduled_tasks[f"board-stake:{game.game_id}"].to_dict() == question_before
    assert len([item for item in store.transitions if item.reason == "board_stake_update_ready"]) == 2
    store.update_game_requirement(game_id=game.game_id, requirement_patch={"stake": "2"}, reason="后来改价", trace_id="later-price")
    assert {key: store.scheduled_tasks[key].to_dict() for key in snapshots} == snapshots
    assert len([item for item in store.transitions if item.reason == "board_stake_update_ready"]) == 2


@pytest.mark.parametrize("state", ["expired", "processing", "completed"])
def test_price_confirmation_does_not_modify_expired_claimed_or_completed_updates(store, state):
    game = board_game(store)
    proposed = update_task(game_id=game.game_id, expires_at=now() - timedelta(seconds=1) if state == "expired" else None)
    store.ensure_board_stake_confirmation_task(task(game_id=game.game_id), update_task=proposed, trace_id="register")
    if state != "expired":
        claim = store.claim_scheduled_task(proposed.task_id, at=proposed.due_at)
        if state == "completed":
            store.complete_scheduled_task(proposed.task_id, trace_id="done", expected_attempts=claim.attempts,
                expected_lease_until=claim.lease_until, at=proposed.due_at + timedelta(seconds=1))
    original = deepcopy(store.scheduled_tasks[proposed.task_id].to_dict())
    store.update_game_requirement(game_id=game.game_id, requirement_patch={"stake": "1"}, reason="桌主确认", trace_id="price")
    assert store.scheduled_tasks[proposed.task_id].to_dict() == original
    assert not any(item.reason == "board_stake_update_ready" for item in store.transitions)


def test_non_board_game_update_does_not_activate_board_price_notifications(store):
    game = board_game(store, board_only=False)
    proposed = update_task(game_id=game.game_id)
    store.ensure_board_stake_confirmation_task(task(game_id=game.game_id), update_task=proposed, trace_id="register")
    store.update_game_requirement(game_id=game.game_id, requirement_patch={"stake": "1"}, reason="普通局改价", trace_id="price")
    assert store.scheduled_tasks[proposed.task_id].due_at == proposed.due_at


def test_sqlite_question_and_update_registration_roll_back_together(tmp_path, monkeypatch):
    store = SQLiteAgentStore(tmp_path / "registration-pair.db")
    proposed = update_task()
    original_append = type(store)._append_transition

    def fail_update_transition(self, transition):
        if transition.entity_id == proposed.task_id:
            raise RuntimeError("controlled update registration failure")
        return original_append(self, transition)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(type(store), "_append_transition", fail_update_transition)
            with pytest.raises(RuntimeError, match="controlled update registration failure"):
                store.ensure_board_stake_confirmation_task(task(), update_task=proposed, trace_id="rollback")
        assert not store.scheduled_tasks and not store.transitions
    finally:
        store._connection.close()


def test_sqlite_price_and_requester_activation_roll_back_together(tmp_path, monkeypatch):
    store = SQLiteAgentStore(tmp_path / "price-activation.db")
    game = board_game(store)
    proposed = update_task(game_id=game.game_id)
    store.ensure_board_stake_confirmation_task(task(game_id=game.game_id), update_task=proposed, trace_id="register")
    original_game = store.require_game(game.game_id).to_dict()
    original_tasks = {key: value.to_dict() for key, value in store.scheduled_tasks.items()}
    original_append = type(store)._append_transition

    def fail_activation(self, transition):
        if transition.reason == "board_stake_update_ready":
            raise RuntimeError("controlled activation failure")
        return original_append(self, transition)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(type(store), "_append_transition", fail_activation)
            with pytest.raises(RuntimeError, match="controlled activation failure"):
                store.update_game_requirement(game_id=game.game_id, requirement_patch={"stake": "1"}, reason="确认", trace_id="rollback")
        assert store.require_game(game.game_id).to_dict() == original_game
        assert {key: value.to_dict() for key, value in store.scheduled_tasks.items()} == original_tasks
        assert not any(item.reason == "board_stake_update_ready" for item in store.transitions)
    finally:
        store._connection.close()


def test_sqlite_price_confirmation_racing_subscription_never_loses_ready_update(tmp_path):
    path = tmp_path / "price-registration-race.db"
    stores = [SQLiteAgentStore(path), SQLiteAgentStore(path)]
    game = board_game(stores[0])
    proposed = update_task(game_id=game.game_id)
    barrier = Barrier(2)

    def act(index):
        barrier.wait(timeout=5)
        if index == 0:
            return stores[index].ensure_board_stake_confirmation_task(task(game_id=game.game_id), update_task=proposed, trace_id="register")
        return stores[index].update_game_requirement(game_id=game.game_id, requirement_patch={"stake": "1"}, reason="桌主确认", trace_id="price")

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(act, (0, 1)))
        saved = stores[0].scheduled_tasks[proposed.task_id]
        assert saved.status == ScheduledTaskStatus.PENDING and saved.due_at <= now()
        assert saved.due_at < proposed.due_at and stores[1].require_game(game.game_id).requirement["stake"] == "1"
        assert len(stores[0].scheduled_tasks) == 2
    finally:
        for item in stores:
            item._connection.close()
