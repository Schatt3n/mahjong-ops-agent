from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from .workflow_models import EntityType, GameWorkflowStatus, StateTransition


STATE_MACHINE_VERSION = "controlled_state_machine.v1"


GAME_ALLOWED_TRANSITIONS: dict[GameWorkflowStatus, set[GameWorkflowStatus]] = {
    GameWorkflowStatus.NEED_CLARIFICATION: {
        GameWorkflowStatus.OPEN,
        GameWorkflowStatus.CANCELLED,
        GameWorkflowStatus.EXPIRED,
    },
    GameWorkflowStatus.OPEN: {
        GameWorkflowStatus.NEGOTIATING,
        GameWorkflowStatus.HOLDING,
        GameWorkflowStatus.CONFIRMED,
        GameWorkflowStatus.CANCELLED,
        GameWorkflowStatus.EXPIRED,
    },
    GameWorkflowStatus.NEGOTIATING: {
        GameWorkflowStatus.HOLDING,
        GameWorkflowStatus.CONFIRMED,
        GameWorkflowStatus.CANCELLED,
        GameWorkflowStatus.EXPIRED,
    },
    GameWorkflowStatus.HOLDING: {
        GameWorkflowStatus.CONFIRMED,
        GameWorkflowStatus.COMPLETED,
        GameWorkflowStatus.CANCELLED,
        GameWorkflowStatus.EXPIRED,
    },
    GameWorkflowStatus.CONFIRMED: {
        GameWorkflowStatus.COMPLETED,
        GameWorkflowStatus.CANCELLED,
    },
    GameWorkflowStatus.COMPLETED: set(),
    GameWorkflowStatus.CANCELLED: set(),
    GameWorkflowStatus.EXPIRED: set(),
}


@dataclass(slots=True)
class StateMachine:
    version: str = STATE_MACHINE_VERSION

    def can_transition_game(
        self,
        from_status: GameWorkflowStatus | str | None,
        to_status: GameWorkflowStatus | str,
    ) -> bool:
        target = self._coerce_game_status(to_status)
        if target is None:
            return False
        if from_status is None:
            return target in {GameWorkflowStatus.NEED_CLARIFICATION, GameWorkflowStatus.OPEN}
        source = self._coerce_game_status(from_status)
        if source is None:
            return False
        if source == target:
            return True
        return target in GAME_ALLOWED_TRANSITIONS.get(source, set())

    def validate_game_transition(
        self,
        *,
        entity_id: str,
        from_status: GameWorkflowStatus | str | None,
        to_status: GameWorkflowStatus | str,
        reason: str,
    ) -> StateTransition:
        target = self._coerce_game_status(to_status)
        allowed = target is not None and self.can_transition_game(from_status, target)
        return StateTransition(
            entity_type=EntityType.GAME.value,
            entity_id=entity_id,
            from_status=str(from_status) if from_status is not None else None,
            to_status=target.value if target else str(to_status),
            reason=reason,
            allowed=allowed,
            metadata={"state_machine_version": self.version},
        )

    def _coerce_game_status(self, status: GameWorkflowStatus | str | None) -> GameWorkflowStatus | None:
        if isinstance(status, GameWorkflowStatus):
            return status
        if status is None:
            return None
        try:
            return GameWorkflowStatus(str(status))
        except ValueError:
            return None


class WorkflowStateStore(Protocol):
    def current_status(self, entity_type: str, entity_id: str) -> str | None:
        ...

    def apply_transition(self, transition: StateTransition) -> StateTransition:
        ...

    def transition_history(
        self,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
    ) -> list[StateTransition]:
        ...


class InMemoryWorkflowStateStore:
    """Small state ledger for the controlled workflow.

    The state machine decides whether a transition is legal. The store applies
    legal transitions, rejects stale transitions, and keeps an auditable history.
    A SQLite/Redis implementation can replace this class behind the same
    protocol when the local trial moves from in-memory to durable deployment.
    """

    def __init__(self) -> None:
        self._statuses: dict[tuple[str, str], str] = {}
        self._history: list[StateTransition] = []

    def current_status(self, entity_type: str, entity_id: str) -> str | None:
        return self._statuses.get((str(entity_type), str(entity_id)))

    def apply_transition(self, transition: StateTransition) -> StateTransition:
        key = (str(transition.entity_type), str(transition.entity_id))
        if not transition.allowed:
            rejected = _transition_with_metadata(
                transition,
                allowed=False,
                store_applied=False,
                store_rejected_reason="transition_not_allowed",
                store_previous_status=self._statuses.get(key),
            )
            self._history.append(rejected)
            return rejected

        current_status = self._statuses.get(key)
        if current_status != transition.from_status:
            rejected = _transition_with_metadata(
                transition,
                allowed=False,
                store_applied=False,
                store_rejected_reason="state_store_status_mismatch",
                store_previous_status=current_status,
                expected_from_status=transition.from_status,
            )
            self._history.append(rejected)
            return rejected

        applied = _transition_with_metadata(
            transition,
            store_applied=True,
            store_previous_status=current_status,
            store_new_status=transition.to_status,
        )
        self._statuses[key] = transition.to_status
        self._history.append(applied)
        return applied

    def transition_history(
        self,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
    ) -> list[StateTransition]:
        history = list(self._history)
        if entity_type is not None:
            history = [item for item in history if item.entity_type == str(entity_type)]
        if entity_id is not None:
            history = [item for item in history if item.entity_id == str(entity_id)]
        return history


def _transition_with_metadata(
    transition: StateTransition,
    *,
    allowed: bool | None = None,
    **metadata: object,
) -> StateTransition:
    return replace(
        transition,
        allowed=transition.allowed if allowed is None else allowed,
        metadata={**transition.metadata, **metadata},
    )
