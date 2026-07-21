"""SQLite drafts store operations."""

from __future__ import annotations

from typing import Any
from datetime import datetime
from ...models import (
    DEFAULT_TZ,
    Game,
    GameParticipant,
    GameStatus,
    InviteDraft,
    InviteStatus,
    MessageReference,
    OPEN_INVITE_STATUSES,
    OutboundMessageDraft,
    RecruitmentStatus,
    StateTransition,
    new_id,
    now,
)
from ...store import (
    CONFIRMED_CANDIDATE_STATUSES,
    UNCONFIRMED_CANDIDATE_STATUSES,
    apply_game_recruitment_policy,
    invite_status_from_candidate_status,
    normalize_game_parties,
    ready_commitment_conflicts,
    refresh_requirement_seat_snapshot,
    resolve_full_game_commitments,
)

class SQLiteDraftsStoreMixin:
    """Backend-specific operations extracted from the compatibility store."""

    __slots__ = ()

    def create_invite_drafts(
        self,
        *,
        game_id: str,
        invitations: list[dict[str, Any]],
        trace_id: str,
    ) -> tuple[list[InviteDraft], list[StateTransition]]:
        self._expire_stale_games(trace_id=trace_id)
        # The open-invite check and draft inserts form one invariant.  A
        # deferred SQLite transaction lets multiple processes all observe
        # "no open invite" before any of them writes, so reserve the writer
        # slot before reading mutable state.
        with self._write_transaction():
            from ...models import new_id, now

            game = self.require_game(game_id)
            if game.status not in {GameStatus.FORMING, GameStatus.INVITING}:
                raise ValueError(f"game does not accept invitations in status={game.status.value}: {game_id}")
            apply_game_recruitment_policy(game)
            if game.recruitment_status == RecruitmentStatus.SCHEDULED:
                opens_at = game.recruitment_opens_at.isoformat() if game.recruitment_opens_at else "unknown"
                raise ValueError(
                    "private candidate outreach is not open yet: "
                    f"recruitment_opens_at={opens_at}; keep the future game listed and wait for the scheduled task"
                )
            requested_customer_ids = [
                str(item.get("customer_id") or "").strip()
                for item in invitations
                if isinstance(item, dict)
            ]
            if any(not customer_id for customer_id in requested_customer_ids):
                raise ValueError("every invitation requires customer_id")
            if len(requested_customer_ids) != len(set(requested_customer_ids)):
                raise ValueError("duplicate customer_id in invitation request")
            open_customer_ids = {
                draft.customer_id
                for draft in self.invite_drafts.values()
                if draft.game_id == game_id and draft.status in OPEN_INVITE_STATUSES
            }
            duplicated = sorted(open_customer_ids.intersection(requested_customer_ids))
            if duplicated:
                raise ValueError(f"customer already has an open invitation for this game: {','.join(duplicated)}")
            transitions: list[StateTransition] = []
            if game.status == GameStatus.FORMING:
                old = game.status.value
                game.status = GameStatus.INVITING
                game.updated_at = now()
                transitions.append(StateTransition("game", game.game_id, old, game.status.value, "create_invite_drafts", trace_id))
            if game.recruitment_status != RecruitmentStatus.ACTIVE:
                old_recruitment = game.recruitment_status.value
                game.recruitment_status = RecruitmentStatus.ACTIVE
                game.updated_at = now()
                transitions.append(
                    StateTransition(
                        "game_recruitment",
                        game.game_id,
                        old_recruitment,
                        game.recruitment_status.value,
                        "create_invite_drafts",
                        trace_id,
                    )
                )
            self._save_game(game)
            drafts: list[InviteDraft] = []
            for raw in invitations:
                if not isinstance(raw, dict):
                    continue
                draft = InviteDraft(
                    draft_id=new_id("draft"),
                    game_id=game_id,
                    customer_id=str(raw.get("customer_id") or ""),
                    display_name=str(raw.get("display_name") or raw.get("customer_id") or ""),
                    message_text=str(raw.get("message_text") or ""),
                    metadata=dict(raw.get("metadata") or {}) if isinstance(raw.get("metadata"), dict) else {},
                )
                drafts.append(draft)
                self._save_message_reference(
                    MessageReference(
                        message_id=draft.draft_id,
                        conversation_id=game.conversation_id,
                        business_ref_type="invite_draft",
                        business_ref_id=draft.draft_id,
                        text=draft.message_text,
                        channel=str(draft.metadata.get("channel") or "internal"),
                        recipient_id=draft.customer_id,
                        recipient_name=draft.display_name,
                        metadata={"source": "create_invite_drafts", "game_id": game_id},
                    )
                )
                transitions.append(StateTransition("invite_draft", draft.draft_id, None, draft.status.value, "create_invite_drafts", trace_id))
                self._save_invite(draft)
            for transition in transitions:
                self._append_transition(transition)
            return drafts, transitions

    def create_outbound_message_drafts(
        self,
        *,
        conversation_id: str,
        drafts: list[dict[str, Any]],
        trace_id: str,
    ) -> tuple[list[OutboundMessageDraft], list[StateTransition]]:
        with self._lock, self._connection:
            from ...models import new_id

            created: list[OutboundMessageDraft] = []
            transitions: list[StateTransition] = []
            for raw in drafts:
                if not isinstance(raw, dict):
                    continue
                draft = OutboundMessageDraft(
                    draft_id=new_id("outbound"),
                    conversation_id=conversation_id,
                    recipient_id=str(raw.get("recipient_id") or ""),
                    recipient_name=str(raw.get("recipient_name") or raw.get("recipient_id") or ""),
                    channel=str(raw.get("channel") or ""),
                    message_text=str(raw.get("message_text") or ""),
                    purpose=str(raw.get("purpose") or ""),
                    metadata=dict(raw.get("metadata") or {}) if isinstance(raw.get("metadata"), dict) else {},
                )
                created.append(draft)
                self._save_message_reference(
                    MessageReference(
                        message_id=draft.draft_id,
                        conversation_id=draft.conversation_id,
                        business_ref_type="outbound_message_draft",
                        business_ref_id=draft.draft_id,
                        text=draft.message_text,
                        channel=draft.channel,
                        recipient_id=draft.recipient_id,
                        recipient_name=draft.recipient_name,
                        metadata={"source": "create_outbound_message_drafts", "purpose": draft.purpose},
                    )
                )
                transitions.append(
                    StateTransition(
                        "outbound_message_draft",
                        draft.draft_id,
                        None,
                        draft.status.value,
                        "create_outbound_message_drafts",
                        trace_id,
                    )
                )
                self._save_outbound_message_draft(draft)
            for transition in transitions:
                self._append_transition(transition)
            return created, transitions

    def update_invite_delivery_status(
        self,
        *,
        draft_id: str,
        status: InviteStatus | str,
        trace_id: str,
        reason: str,
    ) -> tuple[InviteDraft, StateTransition]:
        """Persist an approved send outcome in the same SQLite transaction."""

        with self._write_transaction():
            draft = self.invite_drafts.get(draft_id)
            if draft is None:
                raise ValueError(f"invite draft not found: {draft_id}")
            target = status if isinstance(status, InviteStatus) else InviteStatus(str(status))
            allowed = {
                InviteStatus.PENDING_APPROVAL: {InviteStatus.SENT, InviteStatus.SUPERSEDED},
                InviteStatus.SENT: {InviteStatus.SENT},
            }
            if target not in allowed.get(draft.status, set()):
                raise ValueError(f"invalid invite delivery transition: {draft.status.value}->{target.value}")
            old = draft.status.value
            draft.status = target
            draft.updated_at = now()
            transition = StateTransition("invite_draft", draft_id, old, target.value, reason, trace_id)
            self._save_invite(draft)
            self._append_transition(transition)
            return draft, transition

    def record_candidate_reply(
        self,
        *,
        game_id: str,
        customer_id: str,
        display_name: str,
        status: str,
        seat_count: int = 1,
        trace_id: str,
    ) -> tuple[Game, list[StateTransition]]:
        with self._write_transaction():
            game = self.require_game(game_id)
            transitions: list[StateTransition] = []
            normalized_status = status.strip()
            normalized_seat_count = max(1, min(4, int(seat_count or 1)))
            existing_participant = next((item for item in game.participants if item.customer_id == customer_id), None)
            if normalized_status in CONFIRMED_CANDIDATE_STATUSES:
                conflicts = ready_commitment_conflicts(
                    game,
                    {customer_id},
                    list(self.games.values()),
                )
                if conflicts:
                    raise ValueError(
                        "customer already committed to overlapping ready game: "
                        + ",".join(item.game_id for item in conflicts)
                    )
                claimed_by_others = sum(
                    max(1, int(item.seat_count))
                    for item in game.participants
                    if item.customer_id != customer_id and item.status in {"joined", "confirmed"}
                )
                available_seats = max(0, game.seats_total - claimed_by_others)
                if normalized_seat_count > available_seats:
                    raise ValueError(
                        f"seat capacity exceeded for game {game_id}: requested={normalized_seat_count}, "
                        f"available={available_seats}"
                    )
            for draft in self.invite_drafts.values():
                if draft.game_id == game_id and draft.customer_id == customer_id:
                    old = draft.status.value
                    draft.status = invite_status_from_candidate_status(normalized_status)
                    draft.updated_at = datetime.now(DEFAULT_TZ)
                    transitions.append(StateTransition("invite_draft", draft.draft_id, old, draft.status.value, "record_candidate_reply", trace_id))
                    self._save_invite(draft)
            if normalized_status in CONFIRMED_CANDIDATE_STATUSES:
                if existing_participant is None:
                    game.participants.append(
                        GameParticipant(
                            customer_id=customer_id,
                            display_name=display_name or customer_id,
                            status="confirmed",
                            source="candidate_reply",
                            seat_count=normalized_seat_count,
                        )
                    )
                    transitions.append(
                        StateTransition(
                            "game_participant",
                            f"{game.game_id}:{customer_id}",
                            None,
                            "confirmed",
                            "record_candidate_reply",
                            trace_id,
                        )
                    )
                else:
                    old_status = existing_participant.status
                    old_seat_count = max(1, int(existing_participant.seat_count))
                    existing_participant.status = "confirmed"
                    existing_participant.seat_count = normalized_seat_count
                    if old_status != existing_participant.status or old_seat_count != normalized_seat_count:
                        transitions.append(
                            StateTransition(
                                "game_participant",
                                f"{game.game_id}:{customer_id}",
                                f"{old_status}:seats={old_seat_count}",
                                f"{existing_participant.status}:seats={normalized_seat_count}",
                                "record_candidate_reply",
                                trace_id,
                            )
                        )
            elif normalized_status in UNCONFIRMED_CANDIDATE_STATUSES and existing_participant is not None:
                old_status = existing_participant.status
                old_seat_count = max(1, int(existing_participant.seat_count))
                existing_participant.status = normalized_status
                existing_participant.seat_count = normalized_seat_count
                if old_status != existing_participant.status or old_seat_count != normalized_seat_count:
                    transitions.append(
                        StateTransition(
                            "game_participant",
                            f"{game.game_id}:{customer_id}",
                            f"{old_status}:seats={old_seat_count}",
                            f"{existing_participant.status}:seats={normalized_seat_count}",
                            "record_candidate_reply",
                            trace_id,
                        )
                    )
            game.parties = normalize_game_parties(game.participants)
            game.requirement = refresh_requirement_seat_snapshot(game.requirement, game.parties, game.remaining_seats())
            if game.remaining_seats() == 0 and game.status != GameStatus.READY:
                resolution = resolve_full_game_commitments(
                    game,
                    games=list(self.games.values()),
                    invite_drafts=list(self.invite_drafts.values()),
                    trace_id=trace_id,
                )
                transitions.extend(resolution.transitions)
                for changed_game in resolution.changed_games:
                    self._save_game(changed_game)
                for changed_invite in resolution.changed_invites:
                    self._save_invite(changed_invite)
            elif game.remaining_seats() > 0 and game.status == GameStatus.READY:
                old = game.status.value
                game.status = (
                    GameStatus.INVITING
                    if any(draft.game_id == game.game_id for draft in self.invite_drafts.values())
                    else GameStatus.FORMING
                )
                transitions.append(StateTransition("game", game.game_id, old, game.status.value, "seats_reopened", trace_id))
            game.updated_at = datetime.now(DEFAULT_TZ)
            self._save_game(game)
            for transition in transitions:
                self._append_transition(transition)
            return game, transitions
