"""由公开招募素材派生凌晨私聊约早场的角色事件。"""

from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from pathlib import Path

from mahjong_agent_runtime.models import DEFAULT_TZ, using_clock

from .venue_actors import ActorAction, ActorContext, FamiliarShop, generate_personas
from .venue_channel import AGENT_ID, OWNER_ID, MessageEnvelope, SandboxChannel
from .venue_rolling_board import VenueRollingBoard


_SEED = 72
_HISTORY_ROOM = "历史演练室"
_HISTORY_TERMS = {"game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "no_smoking"}


@dataclass(frozen=True, slots=True)
class OvernightMorningCase:
    """虚拟日期与顾客窗口可重放，店家提供的具体开场仍须经过工具。"""

    case_id: str
    days_ahead: int
    returning: bool
    deterministic_message: str
    request_date: date = date(2026, 11, 25)

    @property
    def request_at(self) -> datetime:
        return datetime.combine(self.request_date, time(1, 12), tzinfo=DEFAULT_TZ)

    @property
    def opening_window(self) -> tuple[datetime, datetime]:
        morning = self.request_at + timedelta(days=self.days_ahead)
        return morning.replace(hour=9, minute=0), morning.replace(hour=10, minute=0)

    def opening_context(
        self, customer, channel, *, familiar_shop: FamiliarShop | None = None,
    ) -> ActorContext:
        """由本人的计划唤醒，熟客背景只来自本人沙盒已经收到的营业回执。"""
        start, end = self.opening_window
        intent = (
            "你想请这位老板组一桌。九点到十点都可以开，由老板在窗口内挑时间。"
            + ("你和老板熟悉，可按平常习惯省略已知条件。"
               if self.returning else "你第一次找老板，宜说清玩法、底注和无烟要求。")
        )
        history = [{**row, "sender_id": AGENT_ID} if row["sender_id"] == OWNER_ID else row
                   for row in channel.events_for(customer.sender_id)]
        return ActorContext(
            customer, self.request_at.isoformat(), history=history,
            events=[{"event_id": self.case_id, "kind": "session_open", "at": self.request_at.isoformat(),
                     "details": {"case_goal": "organize_new", "case_intent": intent,
                                 "earliest_start_time": start.isoformat(), "latest_start_time": end.isoformat()}}],
            familiar_shop=familiar_shop,
        )


# 公开素材只启发「凌晨公开招早场」。原素材的「明早」可能按作息说，
# 可执行样本分别使用「今天早上」与「明天早上」，两种按上海当地日分别计算。
OVERNIGHT_MORNING_CASES = (
    OvernightMorningCase(
        "overnight-new-guest", 0, False,
        "老板你还没睡啊。帮我组桌杭麻吧，今天早上九点到十点开都行，你挑个时间。"
        "一块底，无烟。我刚下夜班，夜班的事别在群里提哈。",
    ),
    OvernightMorningCase(
        "overnight-returning-guest", 1, True,
        "老板，帮我组一桌老样子的呗，明天早上九点到十点开都行，你挑个时间。",
    ),
)


class VenueOvernightMorning:
    """同一可注入的沙盒支撑受控回放和模型顾客，业务状态由传进来的 Runtime 持有。"""

    def __init__(self, case: OvernightMorningCase, runtime, run_dir: Path) -> None:
        self.case = case
        self.store = runtime.store
        roster = generate_personas(_SEED)
        self.customer = next(person for person in roster
                             if person.preferred_game == "杭州麻将" and person.smoking == "non_smoking"
                             and person.stake_min <= 1 <= person.stake_max
                             and person.temperament.opening_disclosure == "main_need_first")
        *peers, observer = tuple(person for person in roster if person != self.customer)[:4]
        self.peers = tuple(peers)
        self.same_name_friend = replace(observer, name=self.customer.name)
        self.history_game_ids: tuple[str, ...] = ()
        self.familiar_shop: FamiliarShop | None = None
        with using_clock(lambda: case.request_at - timedelta(days=8)):
            self.store.configure_rooms([_HISTORY_ROOM])
        # 这些已散场的私域演练先于当前群启用，不留下旧的待发看板。
        if case.returning:
            self.history_game_ids, checkout_at = self._seed_completed_visits()
        with using_clock(lambda: case.request_at):
            self.channel = SandboxChannel(
                runtime, room_id="overnight-morning-venue",
                identities={person.sender_id: person.name for person in (
                    self.customer, *self.peers, self.same_name_friend,
                )},
                archive_path=run_dir / "chat.sqlite3",
            )
        self.board = VenueRollingBoard(self.channel)
        if case.returning:
            self._deliver_checkout(checkout_at)

    def _seed_completed_visits(self) -> tuple[tuple[str, ...], datetime]:
        """用模拟签到及真实领域的开打、散场接口，供店家推导本人的常打条件。"""
        customer = self.customer
        attendance = [person.sender_id for person in (customer, *self.peers)]
        game_ids = []
        for age in (7, 5, 2):
            start = (self.case.request_at - timedelta(days=age)).replace(hour=9, minute=30)
            finish = start + timedelta(hours=2)
            with using_clock(lambda: start - timedelta(hours=1)):
                game, _ = self.store.create_game(
                    conversation_id=f"overnight-history:{customer.sender_id}",
                    organizer_id=customer.sender_id, organizer_name=customer.name, known_players=[],
                    requirement={**_HISTORY_TERMS, "start_time_kind": "scheduled",
                                 "planned_start_at": start.isoformat(), "duration_hours": 2},
                    trace_id=f"overnight-history:create:{age}",
                )
                for peer in self.peers:
                    self.store.join_game(game_id=game.game_id, customer_id=peer.sender_id,
                                         display_name=peer.name, trace_id=f"overnight-history:join:{age}")
            with using_clock(lambda: start):
                self.store.start_game_play(game.game_id, attendee_ids=attendance, room_id=_HISTORY_ROOM,
                                           at=start, trace_id=f"overnight-history:start:{age}")
            with using_clock(lambda: finish):
                self.store.finish_game_play(game.game_id, at=finish, trace_id=f"overnight-history:finish:{age}")
            game_ids.append(game.game_id)
        return tuple(game_ids), finish

    def _deliver_checkout(self, at: datetime) -> None:
        """只生成本人历史营业回执，供熟客角色判断已知条件。"""
        checkout_id = "overnight-history-checkout"
        self.channel.publish_owner(
            kind="private", recipient_id=self.customer.sender_id, sent_at=at,
            message_id=checkout_id, text="这桌散了哈，下回老样子我还记着。",
        )
        if preferences := self.store.completed_game_preference(self.customer.sender_id):
            self.familiar_shop = FamiliarShop(checkout_id, preferences)

    def opening_context(self) -> ActorContext:
        """供调用方让实际角色决定是否开口、以及如何表达。"""
        return self.case.opening_context(self.customer, self.channel, familiar_shop=self.familiar_shop)

    def send_opening(self, action: ActorAction | None) -> dict | None:
        """不替未开口的顾客补写话术，只将本轮决定投到真实沙盒私聊入口。"""
        if action is None or action.action != "send":
            return None
        if action.kind != "private" or action.recipient_id != AGENT_ID:
            raise ValueError("overnight morning opening must be sent to the invited shop")
        at = self.case.request_at
        with using_clock(lambda: at):
            return self.channel.send(MessageEnvelope(
                f"customer:{self.case.case_id}", self.customer.sender_id,
                action.text, at, action.kind, action.recipient_id, action.quote_message_id,
            ))
