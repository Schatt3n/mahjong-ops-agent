from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .matcher import MatchingEngine
from .messages import MessageComposer
from .models import (
    CandidateRecommendation,
    CustomerFatigue,
    CustomerProfile,
    DEFAULT_TZ,
    ExtractionResult,
    GameRequest,
    GameStatus,
    Invitation,
    InvitationStatus,
    Message,
    RoomAvailability,
    RoomHold,
    RoomHoldStatus,
)
from .parser import MahjongMessageParser


@dataclass(slots=True)
class InMemoryStore:
    messages: dict[str, Message] = field(default_factory=dict)
    games: dict[str, GameRequest] = field(default_factory=dict)
    customers: dict[str, CustomerProfile] = field(default_factory=dict)
    invitations: dict[str, Invitation] = field(default_factory=dict)
    room_capacity: int | None = None
    room_holds: dict[str, RoomHold] = field(default_factory=dict)


@dataclass(slots=True)
class IngestOutcome:
    extraction: ExtractionResult
    candidates: list[CandidateRecommendation] = field(default_factory=list)
    draft_group_post: str | None = None
    clarification_text: str | None = None
    room_availability: RoomAvailability | None = None
    room_conflict_text: str | None = None


@dataclass(slots=True)
class AcceptOutcome:
    accepted: bool
    invitation: Invitation
    game: GameRequest
    message_to_customer: str
    conflict_game_id: str | None = None
    cancelled_invitations: list[Invitation] = field(default_factory=list)


ACTIVE_GAME_STATUSES = {
    GameStatus.OPEN,
    GameStatus.NEGOTIATING,
    GameStatus.HOLDING,
    GameStatus.CONFIRMED,
}
ACTIVE_INVITATION_STATUSES = {
    InvitationStatus.QUEUED,
    InvitationStatus.SENT,
    InvitationStatus.ACCEPTED,
}
PENDING_INVITATION_STATUSES = {
    InvitationStatus.QUEUED,
    InvitationStatus.SENT,
}
DEFAULT_GAME_DURATION_HOURS = 4.0
COMPLETED_GAME_RELEASE_GRACE_MINUTES = 30
UNCONFIRMED_GAME_EXPIRE_GRACE_MINUTES = 30
PARTY_SIZE_PROFILE_CONFIDENCE_THRESHOLD = 0.75
ROOM_SEARCH_STEP_MINUTES = 15
ROOM_SEARCH_HORIZON_HOURS = 12


class AgentCore:
    def __init__(
        self,
        parser: MahjongMessageParser | None = None,
        matcher: MatchingEngine | None = None,
        composer: MessageComposer | None = None,
        store: InMemoryStore | None = None,
    ) -> None:
        self.parser = parser or MahjongMessageParser()
        self.matcher = matcher or MatchingEngine()
        self.composer = composer or MessageComposer()
        self.store = store or InMemoryStore()

    def upsert_customer(self, profile: CustomerProfile) -> None:
        self.store.customers[profile.id] = profile

    def ingest_message(self, message: Message, now: datetime | None = None) -> IngestOutcome:
        effective_now = now or datetime.now(DEFAULT_TZ)
        self.advance_game_lifecycle(effective_now)
        self.store.messages[message.id] = message
        extraction = self.parser.parse(message, now=effective_now)
        self._apply_customer_party_size(extraction, message)

        candidates: list[CandidateRecommendation] = []
        draft_group_post: str | None = None
        clarification_text: str | None = None
        room_availability: RoomAvailability | None = None
        room_conflict_text: str | None = None

        if extraction.game:
            game = extraction.game
            room_availability = self.check_room_availability(
                game.start_at,
                game.duration_hours or DEFAULT_GAME_DURATION_HOURS,
            )
            if room_availability and not room_availability.available:
                self._apply_room_conflict(extraction, room_availability)
                room_conflict_text = self.composer.room_time_conflict(game, room_availability)
            else:
                room_conflict_text = None
            self.store.games[game.id] = game
            if room_conflict_text:
                clarification_text = room_conflict_text
            elif game.status == GameStatus.OPEN:
                available_customers = [
                    customer
                    for customer in self.store.customers.values()
                    if self.customer_active_lock(customer.id, exclude_game_id=game.id) is None
                ]
                fatigue_by_customer = {
                    customer.id: self.customer_fatigue(
                        customer.id,
                        proposed_start_at=game.start_at or effective_now,
                        now=effective_now,
                        exclude_game_id=game.id,
                    )
                    for customer in available_customers
                }
                candidates = self.matcher.recommend_customers(
                    game,
                    available_customers,
                    now=effective_now,
                    fatigue_by_customer=fatigue_by_customer,
                )
                draft_group_post = self.composer.group_post(game)
            else:
                clarification_text = self.composer.clarification(extraction)

        return IngestOutcome(
            extraction=extraction,
            candidates=candidates,
            draft_group_post=draft_group_post,
            clarification_text=clarification_text,
            room_availability=room_availability,
            room_conflict_text=room_conflict_text,
        )

    def configure_room_capacity(self, capacity: int | None) -> None:
        if capacity is not None and capacity < 1:
            raise ValueError("room capacity must be positive")
        self.store.room_capacity = capacity

    def add_room_hold(
        self,
        start_at: datetime,
        end_at: datetime,
        room_id: str | None = None,
        source: str = "manual",
        game_id: str | None = None,
        notes: list[str] | None = None,
    ) -> RoomHold:
        if end_at <= start_at:
            raise ValueError("room hold end_at must be after start_at")
        hold = RoomHold(
            start_at=start_at,
            end_at=end_at,
            room_id=room_id,
            source=source,
            game_id=game_id,
            notes=notes or [],
        )
        self.store.room_holds[hold.id] = hold
        return hold

    def check_room_availability(
        self,
        start_at: datetime | None,
        duration_hours: float,
    ) -> RoomAvailability | None:
        if start_at is None or self.store.room_capacity is None:
            return None
        duration = timedelta(hours=duration_hours)
        requested_end_at = start_at + duration
        occupied = self._occupied_room_count(start_at, requested_end_at)
        if occupied < self.store.room_capacity:
            return RoomAvailability(
                requested_start_at=start_at,
                requested_end_at=requested_end_at,
                duration_hours=duration_hours,
                available=True,
                occupied_rooms=occupied,
                capacity=self.store.room_capacity,
            )

        step = timedelta(minutes=ROOM_SEARCH_STEP_MINUTES)
        search_until = start_at + timedelta(hours=ROOM_SEARCH_HORIZON_HOURS)
        candidate = start_at + step
        while candidate <= search_until:
            candidate_end = candidate + duration
            candidate_occupied = self._occupied_room_count(candidate, candidate_end)
            if candidate_occupied < self.store.room_capacity:
                return RoomAvailability(
                    requested_start_at=start_at,
                    requested_end_at=requested_end_at,
                    duration_hours=duration_hours,
                    available=False,
                    suggested_start_at=candidate,
                    suggested_end_at=candidate_end,
                    occupied_rooms=occupied,
                    capacity=self.store.room_capacity,
                    reason="requested_time_full",
                )
            candidate += step

        return RoomAvailability(
            requested_start_at=start_at,
            requested_end_at=requested_end_at,
            duration_hours=duration_hours,
            available=False,
            occupied_rooms=occupied,
            capacity=self.store.room_capacity,
            reason="no_room_found_in_search_horizon",
        )

    def _apply_room_conflict(self, extraction: ExtractionResult, availability: RoomAvailability) -> None:
        game = extraction.game
        if game is None:
            return
        game.status = GameStatus.NEED_CLARIFICATION
        requested = availability.requested_start_at.strftime("%H:%M")
        if availability.suggested_start_at:
            suggested = availability.suggested_start_at.strftime("%H:%M")
            conflict = f"{requested} 目前满房，最快 {suggested} 有房"
            question = f"{requested} 目前满房，是否可以改到 {suggested} 开局？"
        else:
            conflict = f"{requested} 目前满房，暂未找到可用房间"
            question = f"{requested} 目前满房，是否可以换一个时间？"
        if conflict not in game.notes:
            game.notes.append(conflict)
        if conflict not in game.ambiguities:
            game.ambiguities.append(conflict)
        if question not in extraction.follow_up_questions:
            extraction.follow_up_questions.insert(0, question)
        extraction.raw["room_availability"] = {
            "available": availability.available,
            "requested_start_at": availability.requested_start_at.isoformat(),
            "requested_end_at": availability.requested_end_at.isoformat(),
            "suggested_start_at": availability.suggested_start_at.isoformat() if availability.suggested_start_at else None,
            "suggested_end_at": availability.suggested_end_at.isoformat() if availability.suggested_end_at else None,
            "occupied_rooms": availability.occupied_rooms,
            "capacity": availability.capacity,
            "reason": availability.reason,
        }

    def _occupied_room_count(self, start_at: datetime, end_at: datetime) -> int:
        occupied_room_ids: set[str] = set()
        anonymous_holds = 0
        for hold in self.store.room_holds.values():
            if hold.status != RoomHoldStatus.ACTIVE:
                continue
            if not self._time_ranges_overlap(start_at, end_at, hold.start_at, hold.end_at):
                continue
            if hold.room_id:
                occupied_room_ids.add(hold.room_id)
            else:
                anonymous_holds += 1
        return len(occupied_room_ids) + anonymous_holds

    def _time_ranges_overlap(
        self,
        left_start: datetime,
        left_end: datetime,
        right_start: datetime,
        right_end: datetime,
    ) -> bool:
        return left_start < right_end and right_start < left_end

    def queue_invitations(
        self,
        game_id: str,
        candidates: list[CandidateRecommendation],
        limit: int | None = None,
        now: datetime | None = None,
    ) -> list[Invitation]:
        if now is not None:
            self.advance_game_lifecycle(now)
        game = self.store.games[game_id]
        if game.status not in {GameStatus.OPEN, GameStatus.NEGOTIATING}:
            return []
        queued: list[Invitation] = []
        selected = candidates[:limit] if limit is not None else candidates
        for candidate in selected:
            if game.open_slots == 0:
                break
            if self._existing_active_invitation(game_id, candidate.customer_id):
                continue
            if self.customer_active_lock(candidate.customer_id, exclude_game_id=game_id):
                continue
            fatigue = self.customer_fatigue(
                candidate.customer_id,
                proposed_start_at=game.start_at or now or datetime.now(DEFAULT_TZ),
                now=now,
                exclude_game_id=game_id,
            )
            if fatigue.hard_block:
                continue
            invitation = Invitation(
                game_id=game_id,
                customer_id=candidate.customer_id,
                customer_name=candidate.display_name,
                status=InvitationStatus.QUEUED,
                message_text=self.composer.private_invite(game, candidate),
            )
            self.store.invitations[invitation.id] = invitation
            queued.append(invitation)
        if queued and game.status == GameStatus.OPEN:
            game.status = GameStatus.NEGOTIATING
            game.touch()
        return queued

    def mark_invitation_sent(self, invitation_id: str, sent_at: datetime | None = None) -> Invitation:
        invitation = self.store.invitations[invitation_id]
        invitation.set_status(InvitationStatus.SENT)
        customer = self.store.customers.get(invitation.customer_id)
        if customer:
            customer.last_invited_at = sent_at
        return invitation

    def accept_invitation(self, invitation_id: str, now: datetime | None = None) -> AcceptOutcome:
        if now is not None:
            self.advance_game_lifecycle(now)
        invitation = self.store.invitations[invitation_id]
        game = self.store.games[invitation.game_id]

        if game.status not in ACTIVE_GAME_STATUSES:
            invitation.set_status(InvitationStatus.SUPERSEDED)
            return AcceptOutcome(
                accepted=False,
                invitation=invitation,
                game=game,
                message_to_customer=self.composer.already_expired(game),
            )

        hard_lock = self.customer_active_lock(
            invitation.customer_id,
            exclude_game_id=game.id,
            hard_only=True,
        )
        if hard_lock:
            invitation.set_status(InvitationStatus.SUPERSEDED)
            return AcceptOutcome(
                accepted=False,
                invitation=invitation,
                game=game,
                conflict_game_id=hard_lock[0],
                message_to_customer=self.composer.already_committed(game),
            )

        if game.open_slots == 0 and invitation.customer_id not in game.reserved_customer_ids:
            invitation.set_status(InvitationStatus.SUPERSEDED)
            return AcceptOutcome(
                accepted=False,
                invitation=invitation,
                game=game,
                message_to_customer=self.composer.already_full(game),
            )

        invitation.set_status(InvitationStatus.ACCEPTED)
        if invitation.customer_id not in game.reserved_customer_ids:
            game.reserved_customer_ids.append(invitation.customer_id)
        game.status = GameStatus.CONFIRMED if game.open_slots == 0 else GameStatus.HOLDING
        game.touch()

        cancelled = self._cancel_pending_for_customer(invitation.customer_id, exclude_game_id=game.id)
        if game.is_full:
            cancelled.extend(self._cancel_pending_for_full_game(game))

        return AcceptOutcome(
            accepted=True,
            invitation=invitation,
            game=game,
            message_to_customer=self.composer.confirmed(game) if game.is_full else "已帮你先占位，我继续确认剩余人数。",
            cancelled_invitations=cancelled,
        )

    def decline_invitation(self, invitation_id: str) -> Invitation:
        invitation = self.store.invitations[invitation_id]
        invitation.set_status(InvitationStatus.DECLINED)
        customer = self.store.customers.get(invitation.customer_id)
        if customer:
            customer.decline_count_30d += 1
        return invitation

    def set_game_status(self, game_id: str, status: GameStatus) -> list[Invitation]:
        game = self.store.games[game_id]
        game.status = status
        game.touch()
        if status in {GameStatus.CONFIRMED, GameStatus.CANCELLED, GameStatus.EXPIRED}:
            return self._cancel_pending_for_full_game(game)
        return []

    def suggest_merges(self) -> list:
        return self.matcher.suggest_merges(list(self.store.games.values()))

    def advance_game_lifecycle(self, now: datetime | None = None) -> list[GameRequest]:
        now = now or datetime.now(DEFAULT_TZ)
        changed: list[GameRequest] = []
        for game in self.store.games.values():
            if game.start_at is None:
                continue
            if game.status == GameStatus.CONFIRMED:
                release_at = self._estimated_end_at(game) + timedelta(
                    minutes=COMPLETED_GAME_RELEASE_GRACE_MINUTES
                )
                if now >= release_at:
                    game.status = GameStatus.COMPLETED
                    game.touch()
                    self._cancel_pending_for_full_game(game)
                    changed.append(game)
                continue

            if game.status in {GameStatus.OPEN, GameStatus.NEGOTIATING, GameStatus.HOLDING, GameStatus.NEED_CLARIFICATION}:
                expire_at = game.start_at + timedelta(minutes=UNCONFIRMED_GAME_EXPIRE_GRACE_MINUTES)
                if now >= expire_at:
                    game.status = GameStatus.EXPIRED
                    game.touch()
                    self._cancel_pending_for_full_game(game)
                    changed.append(game)
        return changed

    def customer_active_lock(
        self,
        customer_id: str,
        exclude_game_id: str | None = None,
        hard_only: bool = False,
    ) -> tuple[str, str] | None:
        """Return the active game currently holding this customer, if any."""
        for game in self.store.games.values():
            if game.id == exclude_game_id or game.status not in ACTIVE_GAME_STATUSES:
                continue
            if customer_id == game.organizer_id:
                return game.id, "organizer"
            if customer_id in game.participant_ids:
                return game.id, "participant"
            if customer_id in game.reserved_customer_ids:
                return game.id, "reserved"

        active_statuses = {InvitationStatus.ACCEPTED} if hard_only else ACTIVE_INVITATION_STATUSES
        for invitation in self.store.invitations.values():
            if invitation.game_id == exclude_game_id:
                continue
            if invitation.customer_id != customer_id or invitation.status not in active_statuses:
                continue
            game = self.store.games.get(invitation.game_id)
            if game is None or game.status not in ACTIVE_GAME_STATUSES:
                continue
            return game.id, f"invitation:{invitation.status.value}"
        return None

    def _apply_customer_party_size(self, extraction: ExtractionResult, message: Message) -> None:
        game = extraction.game
        if game is None:
            return
        if game.current_player_count is not None or game.missing_count is not None:
            return

        customer = self.store.customers.get(message.sender_id)
        if customer is None:
            return

        party_size = customer.usual_party_size
        confidence = customer.usual_party_size_confidence
        if party_size is None:
            raw_party_size = customer.metadata.get("usual_party_size") or customer.metadata.get("default_party_size")
            if raw_party_size is None:
                return
            try:
                party_size = int(raw_party_size)
            except (TypeError, ValueError):
                return
            raw_confidence = customer.metadata.get("usual_party_size_confidence") or customer.metadata.get(
                "party_size_confidence"
            )
            try:
                confidence = float(raw_confidence) if raw_confidence is not None else confidence
            except (TypeError, ValueError):
                confidence = 0.0

        if confidence < PARTY_SIZE_PROFILE_CONFIDENCE_THRESHOLD:
            return
        if party_size < 1 or party_size > game.seats_total:
            return

        game.current_player_count = party_size
        game.missing_count = max(0, game.seats_total - party_size)
        note = f"人数根据客户画像推断：{party_size}人，置信度{confidence:.2f}"
        if note not in game.notes:
            game.notes.append(note)
        extraction.raw["profile_party_size"] = party_size
        extraction.raw["profile_party_size_confidence"] = confidence

        extraction.follow_up_questions = [
            question
            for question in extraction.follow_up_questions
            if "几缺几" not in question and "几个人" not in question
        ]
        game.status = GameStatus.OPEN if not extraction.follow_up_questions else GameStatus.NEED_CLARIFICATION

    def customer_fatigue(
        self,
        customer_id: str,
        proposed_start_at: datetime | None = None,
        now: datetime | None = None,
        exclude_game_id: str | None = None,
    ) -> CustomerFatigue:
        customer = self.store.customers.get(customer_id)
        proposed_start_at = proposed_start_at or now or datetime.now(DEFAULT_TZ)
        now = now or datetime.now(DEFAULT_TZ)
        if customer is None:
            return CustomerFatigue(customer_id=customer_id)

        target_date = proposed_start_at.astimezone(DEFAULT_TZ).date()
        played_starts: list[datetime] = []
        counted_game_ids: set[str] = set()
        for game in self.store.games.values():
            if game.id == exclude_game_id or game.start_at is None:
                continue
            if game.start_at.astimezone(DEFAULT_TZ).date() != target_date:
                continue
            if game.status not in {GameStatus.HOLDING, GameStatus.CONFIRMED, GameStatus.COMPLETED}:
                continue
            if not self._customer_belongs_to_game(customer_id, game):
                continue
            counted_game_ids.add(game.id)
            played_starts.append(game.start_at)

        invitations_on_day = 0
        for invitation in self.store.invitations.values():
            if invitation.game_id == exclude_game_id:
                continue
            if invitation.customer_id != customer_id or invitation.status not in PENDING_INVITATION_STATUSES:
                continue
            game = self.store.games.get(invitation.game_id)
            if game is None or game.start_at is None:
                continue
            if game.start_at.astimezone(DEFAULT_TZ).date() != target_date:
                continue
            if game.status not in ACTIVE_GAME_STATUSES:
                continue
            invitations_on_day += 1

        hours_since_last_game = None
        if played_starts:
            last_start = max(played_starts)
            hours_since_last_game = (proposed_start_at - last_start).total_seconds() / 3600

        reasons: list[str] = []
        warnings: list[str] = []
        hard_block = False
        sensitivity = max(0.0, customer.fatigue_sensitivity)
        games_on_day = len(counted_game_ids)

        if games_on_day >= customer.max_games_per_day:
            hard_block = True
            warnings.append(f"今日已打 {games_on_day} 场，达到画像上限 {customer.max_games_per_day} 场")
        elif games_on_day == 0:
            reasons.append("今日未记录已打局，疲劳低")
        else:
            warnings.append(f"今日已打 {games_on_day} 场，未超过画像上限 {customer.max_games_per_day} 场")

        if invitations_on_day >= customer.daily_invite_limit:
            hard_block = True
            warnings.append(f"今日待确认邀约 {invitations_on_day} 次，达到邀约上限 {customer.daily_invite_limit} 次")
        elif invitations_on_day:
            warnings.append(f"今日已有 {invitations_on_day} 个待确认邀约")

        if hours_since_last_game is not None and hours_since_last_game < customer.min_hours_between_games:
            hard_block = True
            warnings.append(f"距上一场约 {hours_since_last_game:.1f} 小时，低于画像间隔 {customer.min_hours_between_games:g} 小时")

        score_adjustment = 0.0
        if not hard_block:
            if games_on_day == 0:
                score_adjustment += 6
            else:
                score_adjustment -= 12 * games_on_day * sensitivity
            if invitations_on_day:
                score_adjustment -= 6 * invitations_on_day * sensitivity
            if hours_since_last_game is not None:
                if hours_since_last_game >= customer.min_hours_between_games * 2:
                    score_adjustment += 4
                else:
                    score_adjustment -= 4 * sensitivity

        return CustomerFatigue(
            customer_id=customer_id,
            games_on_day=games_on_day,
            invitations_on_day=invitations_on_day,
            max_games_per_day=customer.max_games_per_day,
            daily_invite_limit=customer.daily_invite_limit,
            hours_since_last_game=round(hours_since_last_game, 2) if hours_since_last_game is not None else None,
            score_adjustment=round(score_adjustment, 1),
            hard_block=hard_block,
            reasons=reasons,
            warnings=warnings,
        )

    def reserve_customer_for_game(
        self,
        game_id: str,
        customer_id: str,
        now: datetime | None = None,
    ) -> tuple[bool, list[Invitation], str | None]:
        if now is not None:
            self.advance_game_lifecycle(now)
        game = self.store.games[game_id]
        if game.status not in ACTIVE_GAME_STATUSES:
            return False, [], None
        if game.open_slots == 0 and customer_id not in game.reserved_customer_ids:
            return False, [], None

        hard_lock = self.customer_active_lock(customer_id, exclude_game_id=game_id, hard_only=True)
        if hard_lock:
            return False, [], hard_lock[0]

        if customer_id not in game.reserved_customer_ids:
            game.reserved_customer_ids.append(customer_id)
        game.status = GameStatus.CONFIRMED if game.open_slots == 0 else GameStatus.HOLDING
        game.touch()

        cancelled = self._cancel_pending_for_customer(customer_id, exclude_game_id=game_id)
        if game.is_full:
            cancelled.extend(self._cancel_pending_for_full_game(game))
        return True, cancelled, None

    def _estimated_end_at(self, game: GameRequest) -> datetime:
        duration = game.duration_hours or DEFAULT_GAME_DURATION_HOURS
        return game.start_at + timedelta(hours=duration)

    def _customer_belongs_to_game(self, customer_id: str, game: GameRequest) -> bool:
        if customer_id == game.organizer_id:
            return True
        if customer_id in game.participant_ids or customer_id in game.reserved_customer_ids:
            return True
        return any(
            invitation.game_id == game.id
            and invitation.customer_id == customer_id
            and invitation.status == InvitationStatus.ACCEPTED
            for invitation in self.store.invitations.values()
        )

    def _existing_active_invitation(self, game_id: str, customer_id: str) -> bool:
        return any(
            invitation.game_id == game_id
            and invitation.customer_id == customer_id
            and invitation.status in ACTIVE_INVITATION_STATUSES
            for invitation in self.store.invitations.values()
        )

    def _cancel_pending_for_full_game(self, game: GameRequest) -> list[Invitation]:
        cancelled: list[Invitation] = []
        for invitation in self.store.invitations.values():
            if invitation.game_id != game.id:
                continue
            if invitation.status in {InvitationStatus.QUEUED, InvitationStatus.SENT}:
                invitation.set_status(InvitationStatus.SUPERSEDED)
                cancelled.append(invitation)
        return cancelled

    def _cancel_pending_for_customer(
        self,
        customer_id: str,
        exclude_game_id: str | None = None,
    ) -> list[Invitation]:
        cancelled: list[Invitation] = []
        for invitation in self.store.invitations.values():
            if invitation.customer_id != customer_id:
                continue
            if invitation.game_id == exclude_game_id:
                continue
            if invitation.status in PENDING_INVITATION_STATUSES:
                invitation.set_status(InvitationStatus.SUPERSEDED)
                cancelled.append(invitation)
        return cancelled
