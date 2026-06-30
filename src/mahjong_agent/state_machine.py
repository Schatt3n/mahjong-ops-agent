from __future__ import annotations

from dataclasses import dataclass

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
