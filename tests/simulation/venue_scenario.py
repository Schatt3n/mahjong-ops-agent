"""推进一座沙盒麻将馆的事件，保存模拟角色与真实业务的独立证据。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from heapq import heappop, heappush
import json
from pathlib import Path
import random
from typing import Any

from mahjong_agent_runtime import SQLiteAgentStore
from mahjong_agent_runtime.domains.full_table_notice import FULL_TABLE_NOTICE_TASK_TYPE
from mahjong_agent_runtime.group_chat import BOARD_TASK_TYPE
from mahjong_agent_runtime.models import GameStatus, InviteStatus, ScheduledTaskStatus, using_clock

from .venue_actors import ActorContext, JudgeResult, ModelActor, Persona, SemanticJudge
from .venue_channel import AGENT_ID, OWNER_ID, MessageEnvelope
from .venue_dialogues import _live_actor_action
from .venue_judging import judge_with_short_evidence_ids
from .venue_rolling_board import VenueRollingBoard


@dataclass(frozen=True, slots=True, order=True)
class VenueEvent:
    """同一时刻按入队顺序处理，可追溯到稳定的调度事件。"""

    at: datetime
    sequence: int
    kind: str = field(compare=False)
    actor_id: str | None = field(default=None, compare=False)
    caused_by: str | None = field(default=None, compare=False)

    @property
    def event_id(self) -> str:
        return f"venue-event-{self.sequence:04d}"


VENUE = {
    "name": "晚风棋牌室（模拟）",
    "opens": "12:00",
    "closes": "23:50",
    "rooms": [
        {"id": "东一包间", "seats": 4, "smoking": False, "hourly_price": 32, "equipment": ["自动麻将桌", "排风"]},
        {"id": "东二包间", "seats": 4, "smoking": False, "hourly_price": 36, "equipment": ["自动麻将桌", "空气净化器"]},
        {"id": "西侧包间", "seats": 4, "smoking": True, "hourly_price": 28, "equipment": ["自动麻将桌", "排风"]},
    ],
    "initial_blocks": [{"room_id": "东一包间", "starts": "18:30", "ends": "20:00", "reason": "设备检修"}],
}
PLAY_DURATION = timedelta(hours=4)
ACTUAL_START_DELAY = timedelta(minutes=3)
PROBE_DURATION = timedelta(minutes=30)
REBOOK_DELAY = timedelta(minutes=14)

_JUDGE_CHECKS = ("role_behavior", "participation", "business_facts", "social_refusal")
_PHYSICAL_CHECKS = (
    "four_real_confirmations", "four_arrivals", "actual_start", "occupied_before_end", "overlap_denied",
    "four_departures", "actual_four_hour_finish", "release_persisted", "same_room_rebookable",
)
_JUDGE_INSTRUCTIONS = (
    "findings.check 只填 checks 中的固定编号，分别用 reason 说明判断。\n"
    "以 virtual_at 和交付 sent_at 判断先后，注意时区。venue-event 编号表示入队顺序，不表示实际发生顺序。\n"
    "role_behavior：核对角色发言是否符合本人的目标、偏好和可见信息。\n"
    "participation：核对拒绝和组局确认是否符合可见对话与实际执行。"
    "群看板认领由群入口原子写入。核对 group_claim 的请求来源、落库认领、本人席位、领域迁移和实际回执，"
    "允许该路径没有 join_game 工具，但缺少写入证据时不能确认占座或满桌。"
    "判断拒邀、退席或只是说这桌不参加时，核对本人已收到的店家私聊、原席位、邀约及回执；"
    "如果没有本人已收到的邀约或原席位，店家仍应明白本人不参加，但无需写拒邀或退席。\n"
    "business_facts：核对 Agent 的业务承诺是否符合工具结果、业务状态和实际投递。"
    "play_timing 将原计划开场和真实开散场分开，已按原时间及其时区计算实际分钟数。"
    "牌局公开说几点开是计划；正常晚几分钟开且实际打满约定时长，与当时介绍计划并不冲突。"
    "只有确有落库或交付事实反证时才判该项 fail，reason 要指出哪条实际陈述与哪条证据冲突；"
    "若核对后认为没有可指明的差错，不能给 fail。拒绝某桌又被推回的问题在 social_refusal 单独判。"
    "招募进度区分已获授权并落库的待组责任，与实际发布或已经联系到牌友。"
    "已有带看板刷新任务的待组局，可以说开始凑人，不能说已经外发、通知到某人或有人报名；"
    "待审批草稿与计划任务均不能证明送达。"
    "核对群认领后的满桌私聊时，用输出 message_id 对应 group_claim.private_recipient_memberships。"
    "它只记录当轮实际私聊收件人在同一局的落库状态，joined 和 confirmed 都是已占席。用该处状态和余位判断通知是否合法，"
    "不能因为收件人没有在认领者的这一轮发言就判定他未入席。"
    "私聊补满后的回告使用 private_full_table_notices，只有同一局落库收件人、余位为零和已交付引用才支持满桌陈述。"
    "房态用 room_inventory_reads 的 observed_at 核对查询当时的库存，"
    "window_start_at 和 window_end_at 只是查询的使用区间。"
    "判断房态回复错误时，指出店家实际送达的原话、它所说的时间和房间、以及与之不符的真实回执或操作证据。"
    "查询有空只描述当时可用，不能当成已为顾客留房。需要同时核对是否实际并获授权写入。"
    "room_inventory_write 的 occurred_at 才是状态变化时刻。空房后来被同一局占用、散场释放或被不同预约复订是顺序上的不同状态。"
    "fixture_rebook 是单独的受控复订，不是店家对前一条查房来话的回执，也不能倒推先前查询不实。\n"
    "social_refusal：只核对避桌顾客实际送达的拒绝。通过须引用带 social_refusal_candidate 的交付行，"
    "其 customer_message 应清楚表明这桌不参加，例如明确拒绝这桌或条件已满足时表示这桌算了。"
    "用 caused_by 对应同一局送达的 shop_invitation，并从 public_attendee_message_id 对应的群里原话核对公开信息。"
    "social_refusal_facts 分别给出拒哪一桌、是否落库拒邀、本人实际收到的店家文字、以及有没有再次实际送达的同桌邀约。"
    "核对拒桌和对后续别桌的请求是两件事。‘除了这桌没查到别的’明确把那桌排除，提到原桌用于划范围不是邀请回原桌。"
    "顾客同时要留意别桌时，相关查询或询问别桌条件可以通过，无需先另发一句确认拒绝。"
    "询问顾客未来哪场合适、征求要不要继续找、描述可选后续，均不能说已经盯着或已经发通知。"
    "不要将选择等待、没有占席、询问这桌有无位置或表达有意参加当成拒绝。"
    "明确不参加公开桌即使原本没占席也属于本检查的拒绝。若店家明确邀请回同桌，仍判 fail；"
    "以同桌再邀请为理由判 fail 时，指出真实收件中的完整邀请分句或再次送达的同局邀约凭证。"
    "缺少应写的邀约或席位时，不要求编造拒邀或退席写入。"
    "候选消息只是询问或想参加时判 fail；没有实际送达的候选来话时判 inconclusive。"
    "无需顾客解释私人关系；其他判项仍依据各自的实际证据评估。"
)


def _room_inventory_read(result: dict[str, Any], at: str, *, source: str) -> dict[str, Any]:
    """区分库存快照观察时刻与未来用房区间，两者不能混用。"""
    return {
        "source": source, "observed_at": at, "window_start_at": result.get("start_at"),
        "window_end_at": result.get("end_at"), "available_room_ids": result.get("available_room_ids"),
        "occupied_room_ids": result.get("occupied_room_ids"),
    }


def _judge_evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """保留独立评审所需的来话、业务工具和持久化事实，调用细节仍在原始归档。"""
    evidence = []
    for original in rows:
        row = dict(original)
        if "decision" in row:
            row["decision"] = {key: value for key, value in row["decision"].items()
                               if key not in {"raw", "diagnostics", "response_source", "prior_attempts", "transport_timeout"}}
        if "tool_results" in row:
            row["tool_results"] = [tool for tool in row["tool_results"]
                                   if tool["name"] not in {"customer_visible_content_review", "customer_visible_text_generation"}]
        reads = [
            _room_inventory_read(tool["result"], row["virtual_at"], source="agent_query")
            for tool in row.get("tool_results", ())
            if tool["name"] == "check_room_availability" and tool.get("called") is True and tool.get("allowed") is True
            and not tool.get("error") and isinstance(tool.get("result"), dict)
        ]
        if row["kind"] == "occupied":
            reads.append(_room_inventory_read(row["availability"], row["virtual_at"], source="overlap_probe"))
        if reads:
            row["room_inventory_reads"] = reads
        for field, operation in (("occupied_reservation", "play_start"), ("released", "play_end"),
                                 ("next_booking", "fixture_rebook")):
            if reservation := row.get(field):
                row["room_inventory_write"] = {"operation": operation, "occurred_at": row["virtual_at"],
                    **{key: reservation[key] for key in ("reservation_id", "room_id", "game_id", "start_at", "end_at", "status")}}
                break
        if row["kind"] == "finish":
            game = row.get("game") or {}
            started = game.get("actual_started_at")
            ended = game.get("actual_ended_at")
            if started and ended:
                # 评审区分计划与实打，时长直接由同一局持久化的开散场计算。
                actual = datetime.fromisoformat(ended) - datetime.fromisoformat(started)
                row["play_timing"] = {
                    "game_id": game["game_id"], "planned_start_at": game.get("planned_start_at"),
                    "planned_end_at": game.get("planned_end_at"), "actual_started_at": started,
                    "actual_ended_at": ended, "actual_duration_minutes": actual.total_seconds() / 60,
                }
        evidence.append(row)
    return evidence


def add_social_refusal_facts(rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """只从事件及渠道账本验证拒桌来源、持久拒邀、本人实际收到的后续，语义交给独立评审。"""
    chat = {row["message_id"]: row for row in records}
    invitations = [row["shop_invitation"] for row in rows if row.get("shop_invitation")]
    scoped = []
    for row in rows:
        if not row.get("social_refusal_candidate"):
            scoped.append(row)
            continue
        before, after = row.get("participation_before", {}), row.get("participation_after", {})
        game_id = before.get("game_id")
        inbound = chat.get(row.get("message_id")) or {}
        invite = chat.get(row.get("caused_by")) or {}
        public_share = chat.get(row.get("public_attendee_message_id")) or {}
        recipient = row["actor_id"]
        source_valid = (
            bool(game_id) and (inbound.get("direction"), inbound.get("kind"), inbound.get("sender_id"), inbound.get("recipient_id"))
                == ("inbound", "private", recipient, AGENT_ID)
            and inbound.get("text") == row["customer_message"]["text"] and inbound.get("trace_id") == row.get("trace_id")
            and (invite.get("kind"), invite.get("recipient_id"), invite.get("conversation_id"))
                == ("private", recipient, inbound.get("conversation_id"))
            and any(item["game_id"] == game_id and item["status"] == "sent" and item["receipt"] == invite
                    for item in invitations)
            and (public_share.get("direction"), public_share.get("kind"), public_share.get("recipient_id"))
                == ("inbound", "group", inbound.get("room_id"))
            and any(item.get("kind") == "share_joined" and item.get("message_id") == public_share["message_id"]
                    for item in rows)
            and max(invite["sequence"], public_share["sequence"]) < inbound["sequence"]
        )
        called = [tool for tool in row["tool_results"] if tool.get("called") is True and tool.get("allowed") is True
                  and not tool.get("error")]
        decline_recorded = (source_valid and after.get("game_id") == game_id and "sent" in before.get("invitation_statuses", ())
                            and "declined" in after.get("invitation_statuses", ()) and any(
            tool["name"] == "record_candidate_reply" and (result := tool.get("result") or {}).get("recorded_status") == "declined"
            and (result.get("game") or {}).get("game_id") == game_id for tool in called))
        later = [receipt for receipt in records if source_valid and receipt["sequence"] > inbound["sequence"]
                 and receipt["recipient_id"] == recipient and receipt["conversation_id"] == inbound["conversation_id"]
                 and receipt["kind"] == "private" and receipt["direction"] in {"outbound", "owner"}
                 and receipt["sender_id"] in {AGENT_ID, OWNER_ID}]
        scoped.append(row | {"social_refusal_facts": {
            "source_verified": source_valid, "game_id": game_id,
            "invitation_message_id": invite.get("message_id"), "public_share_message_id": public_share.get("message_id"),
            "declined_same_game_in_store": decline_recorded, "seat_after": after.get("participant"),
            "actual_private_shop_replies": [{key: receipt[key] for key in ("message_id", "text", "source_message_id", "sent_at")}
                                            for receipt in later],
            "recorded_same_game_reoffers": [item["receipt"]["message_id"] for item in invitations
                                           if item["game_id"] == game_id and item["status"] == "sent" and item["receipt"] in later],
            "executed_lookup_tools": [tool["name"] for tool in called if tool["name"] in {"search_current_games", "search_customers"}],
            "recorded_followup_tools": [tool["name"] for tool in called if tool["name"] in {"register_waiting_demand", "schedule_follow_up"}],
        }})
    return scoped


def judge_verified_refusal_wording(judge: SemanticJudge, row: dict[str, Any], *, trace_id: str) -> JudgeResult:
    """邀约、送达和拒邀已核实时，只让独立模型核对当事人实际说了什么。"""
    facts = row["social_refusal_facts"]
    assert facts["source_verified"] and facts["declined_same_game_in_store"] and not facts["recorded_same_game_reoffers"]
    replies = facts["actual_private_shop_replies"]
    evidence = [{"event_id": row["event_id"], "text": row["customer_message"]["text"]},
                *({"message_id": reply["message_id"], "text": reply["text"]} for reply in replies),
                {"event_id": "verified-same-game-source", "original_delivered_invite_id": facts["invitation_message_id"],
                 "original_invite_and_refusal_same_game_verified": True, "decline_persisted": True,
                 "subsequent_typed_same_game_outreach_count": 0,
                 "subsequent_monitoring_records": facts["recorded_followup_tools"]}]
    return judge_with_short_evidence_ids(judge, "social-refusal-wording", evidence, trace_id=trace_id,
        checks=("wording",), instructions=(
            "已核验前一条送达邀约、拒绝及落库针对同一张桌，只评实际聊天。"
            "客户明确拒桌，店家遵从且没有承诺未登记的长期通知可通过。"
            "店家明确叫顾客回同一桌，或称已安排无记录的提醒，应失败。"
            "用原桌描述查询排除范围不算再次邀约，询问别桌条件可以。"
            "引用顾客原话和实际回复，用完整分句区分邀请原桌与找别桌。"
        ))


def select_roles(people: tuple[Persona, ...]) -> tuple[Persona, Persona, Persona, Persona, Persona, Persona]:
    """从种子产生的真实有向关系中选搭子、避桌者、补位者和新一局询问者。"""
    by_id = {person.persona_id: person for person in people}
    compatible = [person for person in people if person.preferred_game == "杭州麻将"
                  and person.stake_min <= 1 <= person.stake_max
                  and "晚上" in person.available_periods]
    compatible_ids = {person.persona_id for person in compatible}
    avoiders = {edge.target_id: person for person in compatible for edge in person.relationships
                if edge.kind == "avoid_same_table"}
    def role_fit(person):
        return (abs(person.temperament.preferred_duration_hours - 4),
                person.temperament.commitment == "ask_only")

    pairs = [(organizer, by_id[edge.target_id], avoiders[edge.target_id])
             for organizer in compatible for edge in organizer.relationships
             if edge.kind == "regular_partner" and edge.target_id in compatible_ids and edge.target_id in avoiders]
    pairs.sort(key=lambda pair: (role_fit(pair[0])[0] + role_fit(pair[1])[0],
                                 role_fit(pair[1]), role_fit(pair[0])))
    for organizer, partner, avoider in pairs:
        taken = {organizer.persona_id, partner.persona_id, avoider.persona_id}
        if len(taken) != 3:
            continue
        extras = [person for person in compatible if person.persona_id not in taken
                  and not any(e.kind == "avoid_same_table" and e.target_id in taken for e in person.relationships)]
        if len(extras) >= 3:
            first, second, *others = sorted(extras, key=role_fit)
            probe = min(others, key=lambda person: person.temperament.purpose != "ask_venue")
            return organizer, partner, avoider, first, second, probe
    raise ValueError("该种子无法生成有搭子、避桌和补位的兼容角色，请检查角色人口")


class VenueScenario:
    """被测 Agent 只收到普通消息，门店受控事件通过正式 Store 执行。"""

    def __init__(self, *, people, actor, judge, channel, store, run_dir: Path, start_at: datetime,
                 seed: int, background_events: int, fixture_agent=None) -> None:
        self.actor = actor
        self.judge = judge
        self.channel = channel
        self.store = store
        self.board = VenueRollingBoard(channel)
        self.run_dir = run_dir
        self.people = {person.sender_id: person for person in people}
        self.organizer, self.partner, self.avoider, self.extra_a, self.extra_b, self.probe = select_roles(people)
        self.players = (self.organizer, self.partner, self.extra_a, self.extra_b)
        preference_people = [person for person in people if person.smoking == "non_smoking"
                             and person not in (*self.players, self.avoider, self.probe)]
        self.memory_person = min(preference_people, key=lambda person: person.temperament.purpose != "socialize")
        self.start_at = start_at
        self.expected_play_end_at: datetime | None = None
        self.clock = start_at - timedelta(minutes=70)
        self.terms = {"game_type": "杭州麻将", "stake": 1, "smoke": "non_smoking",
                      "duration_hours": 4, "start_time": start_at.strftime("%H:%M")}
        self.requirement = {"game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "no_smoking",
                            "duration_kind": "fixed", "duration_hours": 4, "start_time_kind": "scheduled",
                            "start_time": self.terms["start_time"], "planned_start_at": start_at.isoformat()}
        self.fixture_agent = fixture_agent
        self.events: list[VenueEvent] = []
        self.sequence = self.opening_turns = 0
        self.game_id: str | None = None
        self.booking_id: str | None = None
        self.room_id: str | None = None
        self.social_message_id: str | None = None
        self.social_invitation_id: str | None = None
        self.social_reaction_scheduled = False
        self.announcement_message_id: str | None = None
        self.current_board_message_id: str | None = None
        self.board_poll_blocked = False
        self.board_cutoff = start_at + ACTUAL_START_DELAY + PLAY_DURATION + REBOOK_DELAY
        self.social_refusal_delivery_id: str | None = None
        self.opportunity_turns: dict[str, int] = {}
        self.confirmation_messages: dict[str, str] = {}
        self.real_customer_actions = 0
        self.controlled_actions_from_model = True
        self.arrivals: set[str] = set()
        self.departures: set[str] = set()
        self.checks: dict[str, bool | None] = dict.fromkeys((
            "create_via_agent", "board_delivered", "board_schedule_drained", "social_refusal_delivery",
            "peer_delivery_isolation", "memory_from_visible_message",
            "four_real_confirmations", "four_arrivals", "actual_start",
            "occupied_before_end", "overlap_denied", "agent_pre_end_read", "agent_post_end_read",
            "agent_pre_end_read_only", "agent_post_end_read_only",
            "four_departures", "actual_four_hour_finish", "release_persisted", "same_room_rebookable",
        ))
        self.evidence: list[dict[str, Any]] = []
        self.failures: list[dict[str, str]] = []
        self.model_transport_errors: list[dict[str, Any]] = []
        self.model_timeout_retries: list[dict[str, Any]] = []
        self.timeline = run_dir / "events.jsonl"

        self._enqueue("need", start_at - timedelta(minutes=65), self.organizer)
        self._enqueue("peer_invite", start_at - timedelta(minutes=66), self.organizer)
        self._enqueue("preference", start_at - timedelta(minutes=68), self.memory_person)
        actual_start = start_at + ACTUAL_START_DELAY
        self._enqueue("start", actual_start)
        focus = {person.sender_id for person in (*self.players, self.avoider, self.probe, self.memory_person)}
        background = [person for person in people if person.sender_id not in focus]
        weights = [{"quiet": 1, "occasional": 3, "frequent": 7}[person.activity] for person in background]
        rng = random.Random(seed)
        # 活跃度只调整被唤醒概率；角色模型仍有等待及不发言的选择。
        spread = (start_at + timedelta(hours=4) - self.clock) / max(1, background_events)
        for index in range(background_events):
            person = rng.choices(background, weights=weights)[0]
            self._enqueue("activity", self.clock + spread * index, person)

    def _enqueue(self, kind: str, at: datetime, person: Persona | None = None, *, caused_by: str | None = None) -> None:
        self.sequence += 1
        heappush(self.events, VenueEvent(at, self.sequence, kind, person.sender_id if person else None, caused_by))

    def _record(self, event: VenueEvent, **payload: Any) -> dict[str, Any]:
        row = {"event_id": event.event_id, "virtual_at": event.at.isoformat(), "kind": event.kind,
               "caused_by": event.caused_by, **payload}
        with self.timeline.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def _retry_timed_out_model(self, initial, retry, *, stage: str, event_id: str | None = None):
        """重试一次尚未产生输出的请求，先把原始证据写入归档。

        调用方提供同一个角色上下文或已冻结的评审证据。重试在投递或语义判决之前，
        不重放来话、不重新执行业务写入，也不替换已经有效的失败判决。
        """
        if not initial.transport_timeout or initial.status != "transport_error" or initial.raw is not None:
            return initial
        retry_trace_id = f"{initial.trace_id}:timeout-retry"
        identity = {"model_stage": stage, "event_id": event_id, "original_trace_id": initial.trace_id,
                    "retry_trace_id": retry_trace_id}
        archive = self.run_dir / "model_timeout_attempts.jsonl"
        with archive.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({**identity, "phase": "original", "result": initial.to_dict()}, ensure_ascii=False) + "\n")
        result = retry(retry_trace_id)
        with archive.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({**identity, "phase": "retry", "result": result.to_dict()}, ensure_ascii=False) + "\n")
        self.model_timeout_retries.append({**identity, "transport_recovered": result.status == "valid"})
        return result

    def run(self) -> dict[str, Any]:
        """每个事件使用同一仿真时间；失败也留下可独立复盘的事件。"""
        with using_clock(lambda: self.clock):
            while (event := self._next_event()) is not None:
                self.clock = event.at
                try:
                    if event.actor_id:
                        self._act(event, self.people[event.actor_id])
                    else:
                        self._world(event)
                except (KeyError, ValueError, RuntimeError) as exc:
                    if event.kind in {"announce", "board_refresh"}:
                        due = self.board.next_due()
                        # 真正的调度失败会改下次到期时间。未提交任何进度时不能在同一时刻重复处理。
                        self.board_poll_blocked = due is not None and due <= event.at
                    failure = {"event_id": event.event_id, "kind": event.kind, "reason": str(exc)}
                    self.failures.append(failure)
                    self._record(event, failure=failure)
        return self._report()

    def _next_event(self) -> VenueEvent | None:
        """只领取正式调度中已存在的看板工作，不根据模拟开散场额外安排发布。"""
        due = self.board.next_due() if self.game_id and not self.board_poll_blocked else None
        if due is not None:
            at = max(due, self.clock)
            if at <= self.board_cutoff and (not self.events or at <= self.events[0].at):
                self.sequence += 1
                kind = "announce" if self.announcement_message_id is None else "board_refresh"
                return VenueEvent(at, self.sequence, kind)
        return heappop(self.events) if self.events else None

    def _actor_context(self, event: VenueEvent, person: Persona, visible: list[dict]) -> ActorContext:
        """给每次来话绑定实际受众和已见来源；这桌的时长计划只属于这一晚。"""
        kind = event.kind
        details = dict(self.terms) if kind in {"need", "peer_invite", "opportunity", "delivered", "share_joined"} else {}
        private_targets = (AGENT_ID,)
        groups = (self.channel.room_id,)
        group_targets = groups if kind in {"opportunity", "activity", "share_joined"} else ()
        if kind == "peer_invite":
            private_targets = (self.partner.sender_id,)
            details.update(peer_id=self.partner.sender_id, case_intent=(
                "你打算请店家组织今晚这桌，自己已留出完整四小时，想先问常搭档愿不愿一起。"
                "店家还没登记，不替他报名。"
            ))
        elif kind == "need":
            visible = [row for row in visible if row["kind"] == "private"
                       and AGENT_ID in (row["sender_id"], row["recipient_id"])]
            details.update(case_goal="organize_new", party_size=1, case_intent=(
                "今晚特意空出四小时和常搭档玩，开场和打满四小时都已列入你的个人安排。"
                "你这边只有自己确定参加，主动请店家新组这场并把自己算一位。搭档会自己报名。"
                "店家还未登记时可以先自然说来意，按实际追问补条件。这次只组局找人。"
            ))
        elif kind == "preference":
            details.update(case_intent=(
                "你一直不打有烟的桌。今天另有安排，晚上不会去店里。这一条只是顺便告诉店家"
                "以后推荐给你的局都要无烟，不需要对方查今晚的牌局。"
            ))
        elif kind in {"opportunity", "delivered"}:
            if person == self.avoider:
                details.update(case_intent=(
                    "店家私聊问你要不要参加今晚这桌，你还没回。群里也刚有一位认识的人发消息，看看对方怎么说。"
                    "结合你实际收到的两条消息和自己的意愿决定怎么接邀约；如果不去，你倾向于回店家一声，不解释私人原因。"
                ))
                group_targets = ()
                source = next((row for row in visible if row["message_id"] == self.social_message_id
                               and row["kind"] == "group" and row["recipient_id"] == self.channel.room_id), None)
                # 微信群昵称来自当事人实际收到的群消息，不向角色补齐任何未见的席位事实。
                if source and (nickname := self.channel.owner_friends.get(source["sender_id"])):
                    details["note"] = f"这条已收到的群消息显示的发送者昵称是「{nickname}」。"
                    details["known_attendees"] = [{"sender_id": source["sender_id"], "name": nickname}]
                # 店员和 Agent 都使用店家的对客会话；完整账本继续保留手工邀约来源。
                visible = [{**row, "sender_id": AGENT_ID} if row["message_id"] == self.social_invitation_id
                           and row["sender_id"] == OWNER_ID else row for row in visible]
            else:
                details.update(case_goal="try_join", party_size=1, case_intent=(
                    "今晚这一段你安排了完整四小时，想自己加入一桌。你看到群里有个合适的机会，"
                    "自己决定按公开看板报名还是私聊店家确认，已经看见的条件不必重问。"
                ))
        elif kind == "share_joined":
            private_targets = ()
            details.update(case_intent=(
                "刚拿到本人报名确认。你愿意在群里告诉同场熟人你会参加哪桌，只提自己的公开到场计划。"
            ))
        elif kind in {"arrival", "departure"}:
            private_targets = ()
            details["case_intent"] = (
                "你按已确认的约定来到店里，当前就在门口准备签到。" if kind == "arrival"
                else "你自己已经打完今天约好的四小时，现在收拾好准备离店。"
            )
        elif kind == "slot_query":
            situation = (
                "你和另外三个人在附近，有半小时空。" if self.clock < self.expected_play_end_at
                else "刚才还在等的人已经到齐，你们现在有半小时空。"
            )
            details.update(start_time=self.clock.strftime("%H:%M"), duration_hours=0.5, party_size=4,
                           case_intent=situation + (
                               "用你平常的说法问店家现在起半小时有没有空包间，消息里说明只是先问、暂时不用留。"
                               "别问能不能约或订一个包间，这会让店家以为你已经决定要订。"
                           ))
        return ActorContext(person, self.clock.isoformat(), history=visible,
            events=[{"event_id": event.event_id, "kind": kind, "at": self.clock.isoformat(),
                     "caused_by": event.caused_by, "details": details}],
            private_targets=private_targets, group_targets=group_targets, visible_groups=groups)

    def _act(self, event: VenueEvent, person: Persona) -> None:
        if event.kind in {"opportunity", "delivered", "share_joined", "arrival", "departure", "slot_query"} and not self.game_id:
            self._record(event, skipped="尚无已登记牌局")
            return
        if event.kind == "share_joined" and person.sender_id not in self._confirmed():
            self._record(event, skipped="本人尚未获得实际的入局确认")
            return
        if event.kind == "opportunity" and event.caused_by == self.announcement_message_id:
            # 顾客醒来时见到的是最新送达的公开看板。已摘下的桌不能继续触发参报。
            if self.current_board_message_id is None:
                self._record(event, skipped="当前正式看板没有继续招募这桌")
                return
            event = replace(event, caused_by=self.current_board_message_id)
        visible = self.channel.events_for(person.sender_id)[-24:]
        if event.kind == "arrival":
            # 到店前实际收到的成局更新优先于最初的留位回执，角色仍可自主选择不来。
            prefix = f"outbound:{FULL_TABLE_NOTICE_TASK_TYPE}:{self.game_id}:"
            full_notice = next(
                (row["message_id"] for row in reversed(visible)
                 if row["kind"] == "private" and row["recipient_id"] == person.sender_id
                 and row["message_id"].startswith(prefix)), None,
            )
            if full_notice:
                event = replace(event, caused_by=full_notice)
        if event.caused_by and not any(row["message_id"] == event.caused_by for row in visible):
            self._record(event, skipped="本角色未收到触发这一轮的消息")
            return
        if person == self.avoider and event.kind == "opportunity" and not any(
            row["message_id"] == self.social_message_id and row["kind"] == "group" for row in visible
        ):
            self._record(event, skipped="本角色未收到同桌者公开分享")
            return
        context = self._actor_context(event, person, visible)
        decision = self.actor.decide(context, trace_id=f"role:{event.event_id}")
        decision = self._retry_timed_out_model(
            decision, lambda trace: self.actor.decide(context, trace_id=trace),
            stage="role", event_id=event.event_id,
        )
        row = self._record(event, actor_id=person.sender_id, decision=decision.to_dict())
        verified = _live_actor_action(row)
        self.real_customer_actions += int(verified)
        if event.kind != "activity":
            self.evidence.append(row)
            self.controlled_actions_from_model &= verified
        action = decision.action
        if action is None:
            self.failures.append({"event_id": event.event_id, "kind": event.kind, "reason": "角色生成失败或动作合同不合格"})
            if decision.status == "transport_error":
                self.model_transport_errors.append({
                    "event_id": event.event_id, "model_stage": "role",
                    "http_statuses": [item["http_status"] for item in decision.diagnostics.get("attempts", [])
                                      if item.get("http_status") is not None],
                })
            return
        if action.action == "arrive":
            if event.kind != "arrival" or person.sender_id not in self._confirmed():
                raise ValueError("签到缺少到店事件或本人确认")
            self.arrivals.add(person.sender_id)
        if action.action == "leave":
            if event.kind != "departure" or person.sender_id not in self.arrivals:
                raise ValueError("离场缺少到期事件或本人签到")
            self.departures.add(person.sender_id)
        if action.action == "send":
            self._message(event, person, action)

    def _message(self, event: VenueEvent, person: Persona, action) -> None:
        """投递角色来话，按该回合核对业务前后状态，再把实际证据交给评审。"""
        kind = action.kind or "private"
        recipient = action.recipient_id or (self.channel.room_id if kind == "group" else AGENT_ID)
        message_id = f"customer:{event.event_id}"
        social_refusal = (person == self.avoider and event.kind == "opportunity"
                          and event.caused_by == self.social_invitation_id)
        if social_refusal:
            before = self._attendance_state(person.sender_id)
            visible_before = self.channel.events_for(person.sender_id)
        if event.kind == "slot_query":
            reservations_before = self._room_snapshot()
        if self.fixture_agent:
            inputs = {"requirement": self.requirement, "start_at": self.clock.isoformat(),
                      "end_at": (self.clock + timedelta(minutes=30)).isoformat()}
            phase = "decline" if social_refusal else event.kind
            self.fixture_agent.turns[message_id] = (phase, inputs)
        # 模型选择的公开引用已经按本人的可见范围校验，投递时沿用该引用。
        quote = action.quote_message_id
        response = self.channel.send(MessageEnvelope(
            message_id, person.sender_id, action.text, self.clock, kind, recipient, quoted_message_id=quote,
        ))
        evidence = {
            "event_id": f"delivery:{event.event_id}", "message_id": message_id, "actor_id": person.sender_id,
            "virtual_at": event.at.isoformat(), "kind": event.kind, "caused_by": event.caused_by,
            "customer_message": {"kind": kind, "text": action.text, "recipient_id": recipient, "quoted_message_id": quote},
            "trace_id": response.get("trace_id"), "tool_results": response.get("tool_results", []),
            "outputs": response.get("archive_outputs", []), "runtime_status": response.get("status"),
        }
        if claim := self._group_claim_evidence(response, message_id, person.sender_id):
            evidence["group_claim"] = claim
        private_full_table_notices = []
        prefix = f"outbound:{FULL_TABLE_NOTICE_TASK_TYPE}:"
        for output in evidence["outputs"]:
            if output["kind"] != "private" or not output["message_id"].startswith(prefix):
                continue
            reference = self.store.resolve_message_reference(
                conversation_id=output["conversation_id"], message_id=output["message_id"].removeprefix("outbound:"),
            )
            if reference is None or reference.business_ref_type != FULL_TABLE_NOTICE_TASK_TYPE:
                continue
            game = self.store.require_game(reference.business_ref_id)
            recipient = next((row for row in game.participants if row.customer_id == reference.recipient_id), None)
            private_full_table_notices.append({
                "message_id": output["message_id"], "game_id": game.game_id, "recipient_id": reference.recipient_id,
                "remaining_seats": game.remaining_seats(), "participant_status": recipient.status if recipient else None,
            })
        if private_full_table_notices:
            evidence["private_full_table_notices"] = private_full_table_notices
        if social_refusal:
            evidence["participation_before"] = before
            evidence["participation_after"] = self._attendance_state(person.sender_id)
            evidence["earlier_shop_private_messages"] = [
                {key: row[key] for key in ("message_id", "text", "sent_at")}
                for row in visible_before if row["kind"] == "private" and row["direction"] in {"outbound", "owner"}
                and row["recipient_id"] == person.sender_id
            ]
            if kind == "private" and recipient == AGENT_ID:
                # 先记录真实交付锚点。是否表达了拒绝由独立评审判断。
                evidence["social_refusal_candidate"] = True
                evidence["public_attendee_message_id"] = self.social_message_id
                self.social_refusal_delivery_id = evidence["event_id"]
        if event.kind == "slot_query":
            unchanged = reservations_before == self._room_snapshot()
            evidence["room_query"] = {"intended_scope": "availability_only", "reservations_unchanged": unchanged}
            key = "agent_pre_end_read" if self.clock < self.expected_play_end_at else "agent_post_end_read"
            self.checks[f"{key}_only"] = unchanged and not any(
                tool["name"] == "reserve_room" and tool.get("called") and tool.get("allowed") and not tool.get("error")
                for tool in evidence["tool_results"]
            )
            relevant = "occupied_room_ids" if key == "agent_pre_end_read" else "available_room_ids"
            self.checks[key] = any(
                tool["name"] == "check_room_availability" and tool.get("called") and tool.get("allowed")
                and not tool.get("error") and self.room_id in tool["result"].get(relevant, [])
                and datetime.fromisoformat(tool["result"]["start_at"]) == self.clock
                and datetime.fromisoformat(tool["result"]["end_at"]) == self.clock + PROBE_DURATION
                for tool in evidence["tool_results"]
            ) and bool(evidence["outputs"])
        self._record(event, delivery=evidence)
        if event.kind != "activity":
            self.evidence.append(evidence)
        if event.kind == "need":
            self.opening_turns += 1
            self._after_need(event, person, evidence)
        if event.kind == "preference":
            self.checks["memory_from_visible_message"] = any(
                candidate.customer_id == person.sender_id and candidate.source_trace_id == evidence["trace_id"]
                and candidate.field == "smoke_preference" and candidate.status == "pending_review"
                and candidate.value in ("no_smoking", "no_smoke", "无烟") and candidate.evidence
                and candidate.evidence in action.text
                and candidate.metadata.get("source_message_id") == evidence["message_id"]
                for candidate in self.store.pending_memory_candidates.values()
            )
        if event.kind == "peer_invite" and recipient == self.partner.sender_id and kind == "private":
            self.checks["peer_delivery_isolation"] = (
                any(row["message_id"] == message_id for row in self.channel.events_for(self.partner.sender_id))
                and all(not any(row["message_id"] == message_id for row in self.channel.events_for(user))
                        for user in (self.avoider.sender_id, AGENT_ID))
            )
        if event.kind in {"opportunity", "delivered"} and person in self.players and person != self.organizer:
            self._candidate_follow_up(event, person, evidence)
            self._schedule_social_reaction(event)
        if event.kind == "share_joined" and kind == "group":
            self.social_message_id = message_id
            self._schedule_social_reaction(event)

    def _schedule_social_reaction(self, event: VenueEvent) -> None:
        """配齐原桌四席再唤醒避桌顾客，即便他改变主意也不能抢走实打席位。"""
        at = event.at + timedelta(minutes=1)
        if (self.social_reaction_scheduled or not self.social_invitation_id or not self.social_message_id
                or at >= self.start_at or self._confirmed() != {person.sender_id for person in self.players}):
            return
        self.social_reaction_scheduled = True
        self._enqueue("opportunity", at, self.avoider, caused_by=self.social_invitation_id)

    def _attendance_state(self, customer_id: str) -> dict[str, Any]:
        """只查看当前顾客在目标桌的席位和邀约状态，不能用其他顾客的状态解释拒绝。"""
        game = self.store.require_game(self.game_id)
        participant = next((item for item in game.participants if item.customer_id == customer_id), None)
        return {
            "game_id": game.game_id,
            "participant": {"status": participant.status, "seat_count": participant.seat_count} if participant else None,
            "invitation_statuses": sorted(draft.status.value for draft in self.store.invite_drafts.values()
                                          if draft.game_id == game.game_id and draft.customer_id == customer_id),
        }

    def _group_claim_evidence(self, response: dict, message_id: str, customer_id: str) -> dict | None:
        """看板认领由群入口原子执行，从持久化认领和同 trace 领域迁移取证。"""
        routing = (response.get("group_processing") or {}).get("routing") or {}
        if routing.get("action") != "claim":
            return None
        claim = self.store.get_game_claim_by_source(self.channel.group_conversation_id, message_id)
        game = self.store.games.get(routing.get("game_id"))
        transitions = [item.to_dict() for item in self.store.transitions
                       if item.trace_id == response["trace_id"] and game is not None
                       and item.entity_id in {game.game_id, f"{game.game_id}:{customer_id}"}]
        participants = {item.customer_id: item for item in game.participants} if game else {}
        participant = participants.get(customer_id)
        private_recipient_memberships = []
        for output in response.get("archive_outputs", ()):
            if (output.get("source_message_id") != message_id or output.get("direction") != "outbound"
                    or output.get("sender_id") != AGENT_ID or output.get("kind") != "private"):
                continue
            recipient_id = output["recipient_id"]
            membership = participants.get(recipient_id)
            private_recipient_memberships.append({
                "message_id": output["message_id"], "recipient_id": recipient_id,
                "status": membership.status if membership else None,
                "seat_count": membership.seat_count if membership else None,
            })
        return {
            "routing": {"game_id": routing.get("game_id"), **(routing.get("detail") or {})},
            "persisted_claim": claim.to_dict() if claim else None,
            "persisted_participant": {"customer_id": customer_id, "status": participant.status,
                                      "seat_count": participant.seat_count} if participant else None,
            "game_status": game.status.value if game else None,
            "remaining_seats": game.remaining_seats() if game else None,
            "persisted_transitions": transitions,
            "private_recipient_memberships": private_recipient_memberships,
        }

    def _room_snapshot(self) -> dict[str, Any]:
        """对比查询前后完整房态预约记录，防止漏算没有工具回执的业务写入。"""
        return {key: reservation.to_dict() for key, reservation in self.store.room_reservations.items()}

    def _after_need(self, event: VenueEvent, person: Persona, evidence: dict[str, Any]) -> None:
        created = [tool["result"]["game"]["game_id"] for tool in evidence["tool_results"]
                   if tool.get("name") == "create_game" and tool.get("called")
                   and tool.get("allowed") and not tool.get("error")]
        if not created:
            delivered = self._visible_reply(person, evidence)
            if delivered and self.opening_turns < 4:
                self._enqueue("need", event.at + timedelta(minutes=1), person, caused_by=delivered)
            return

        self.game_id = created[0]
        game = self.store.require_game(self.game_id)
        terms_match = (game.organizer_id == self.organizer.sender_id and len(created) == 1
            and game.planned_start_at == self.start_at and game.planned_end_at == self.start_at + timedelta(hours=4))
        reply_delivered = evidence["runtime_status"] in {"completed", "waiting_user"} and bool(evidence["outputs"])
        self.checks["create_via_agent"] = terms_match and reply_delivered
        if not terms_match:
            raise ValueError("正式建局结果的组织者、时间或时长与本轮请求不符")
        if not reply_delivered:
            raise ValueError("组局已按请求落库，但店家未送达有效业务回执")
        self._schedule_arrival(person, evidence)
        if self.board.next_due() is None:
            raise ValueError("待组局缺少正式看板发布任务")

    @staticmethod
    def _visible_reply(person: Persona, evidence: dict[str, Any]) -> str | None:
        return next((message["message_id"] for message in reversed(evidence["outputs"])
                     if message.get("recipient_id") == person.sender_id or message.get("kind") == "group"), None)

    def _candidate_follow_up(self, event: VenueEvent, person: Persona, evidence: dict[str, Any]) -> None:
        count = self.opportunity_turns.get(person.sender_id, 0) + 1
        self.opportunity_turns[person.sender_id] = count
        cause = self._visible_reply(person, evidence)
        if person.sender_id in self.confirmation_messages:
            return
        if self._schedule_arrival(person, evidence):
            if person == self.partner:
                self._enqueue("share_joined", event.at + timedelta(minutes=1), person, caused_by=cause)
        elif cause and count < 3:
            self._enqueue("delivered", event.at + timedelta(minutes=1), person, caused_by=cause)

    def _schedule_arrival(self, person: Persona, evidence: dict) -> bool:
        """只在本人已确认且收到回复后，安排其有真实对话来源的到店事件。"""
        cause = self._visible_reply(person, evidence)
        if (not cause or evidence["runtime_status"] not in {"completed", "waiting_user"}
                or person.sender_id in self.confirmation_messages or person.sender_id not in self._confirmed()):
            return False
        self.confirmation_messages[person.sender_id] = cause
        arrival_time = self.start_at - timedelta(minutes=5) + timedelta(seconds=self.players.index(person))
        if arrival_time >= self.clock:
            self._enqueue("arrival", arrival_time, person, caused_by=cause)
        return True

    def _confirmed(self) -> set[str]:
        return {person.customer_id for person in self.store.require_game(self.game_id).participants
                if person.status in ("joined", "confirmed")}

    def _world(self, event: VenueEvent) -> None:
        if event.kind in {"announce", "board_refresh"}:
            self._publish_boards(event)
        elif event.kind == "social_invite":
            self._send_social_invitation(event)
        elif event.kind == "start":
            self._start(event)
        elif event.kind == "occupied":
            self._occupancy(event)
        elif event.kind == "finish":
            self._finish(event)
        elif event.kind == "rebook":
            self._rebook(event)

    def _publish_boards(self, event: VenueEvent) -> None:
        """通过正式 BoardEngine 和沙盒渠道留存发布、实际收件与发布时桌况。"""
        result = self.channel.publish_board(trace_id=event.event_id, sent_at=event.at)
        game = self.store.require_game(self.game_id)
        eligible = {row.game_id for row in self.store.get_board_eligible_games(self.channel.room_id)}
        self.evidence.append(self._record(event, publication=result, board_table_state={
            "game_id": game.game_id, "status": game.status.value, "remaining_seats": game.remaining_seats(),
            "eligible": game.game_id in eligible,
        }))
        if not result["tasks"]:
            raise ValueError("到期的正式看板任务未能取得领取凭证")
        archived = self.channel.events()
        for task in result["tasks"]:
            if task["status"] != "completed":
                continue
            snapshot = task["snapshot"]
            if snapshot is None:
                if eligible:
                    raise ValueError("仍有符合资格的局，但正式看板任务完成时没有快照")
                self.current_board_message_id = None
                continue
            receipts = [row for row in archived if row["message_id"] == snapshot["external_message_id"]]
            if (len(receipts) != 1 or (receipt := receipts[0])["kind"] != "group"
                    or receipt["direction"] != "outbound" or receipt["sender_id"] != AGENT_ID
                    or receipt["recipient_id"] != self.channel.room_id
                    or snapshot["room_id"] != self.channel.room_id
                    or receipt["conversation_id"] != self.channel.group_conversation_id
                    or snapshot["conversation_id"] != receipt["conversation_id"]
                    or receipt["text"] != snapshot["rendered_text"]
                    or datetime.fromisoformat(receipt["sent_at"]) != datetime.fromisoformat(snapshot["created_at"])):
                raise ValueError("正式看板快照缺少同群、同时且唯一的渠道收据")
            if {item["game_id"] for item in snapshot["items"]} != eligible:
                raise ValueError("正式看板快照与当前可发布的牌局不符")
            self.current_board_message_id = (receipt["message_id"] if self.game_id in eligible else None)
        if self.announcement_message_id is not None or self.current_board_message_id is None:
            return
        self.announcement_message_id = self.current_board_message_id
        self.checks["board_delivered"] = True
        self._enqueue("social_invite", event.at + timedelta(seconds=1), caused_by=self.announcement_message_id)
        for delay, target in ((2, self.partner), (8, self.extra_a), (11, self.extra_b)):
            self._enqueue("opportunity", event.at + timedelta(minutes=delay), target, caused_by=self.announcement_message_id)

    def _send_social_invitation(self, event: VenueEvent) -> None:
        """桌有空时由场景店员发本人邀约，送达后才更新邀约与消息引用。"""
        game = self.store.require_game(self.game_id)
        if game.remaining_seats() == 0:
            self._record(event, skipped="原桌已经满员，没有发出额外邀约")
            return
        recipient = self.avoider.sender_id
        message_id = f"owner:{event.event_id}"
        text = f"群看板上今晚{game.planned_start_at:%H:%M}那桌还在找人，你有空来打吗？"
        drafts, _ = self.store.create_invite_drafts(game_id=game.game_id, invitations=[{
            "customer_id": recipient, "display_name": self.avoider.name, "message_text": text,
        }], trace_id=event.event_id)
        draft, = drafts
        # 公告是场景因果证据，跨会话不能把群消息登记为私聊引用。
        delivery = self.channel.publish_owner(kind="private", recipient_id=recipient, text=text, sent_at=event.at,
                                              message_id=message_id)
        receipt = next((row for row in self.channel.events_for(recipient) if row["message_id"] == message_id
                        and row["kind"] == "private" and row["sender_id"] == OWNER_ID and row["text"] == text
                        and row["recipient_id"] == recipient), None)
        if delivery["status"] != "delivered" or receipt is None:
            self._record(event, skipped="店员私聊没有实际送达")
            return
        self.store.update_invite_delivery_status(draft_id=draft.draft_id, status=InviteStatus.SENT, trace_id=event.event_id,
                                                 reason="沙盒店员的私聊邀约已入账")
        reference = self.store.link_message_reference(
            conversation_id=receipt["conversation_id"], message_id=message_id, source_message_id=draft.draft_id,
            business_ref_type="invite_draft", business_ref_id=draft.draft_id, metadata={"platform_message_id": message_id},
        )
        self.social_invitation_id = message_id
        self.evidence.append(self._record(event, shop_invitation={
            "source": "fixture_owner", "game_id": game.game_id, "draft_id": reference.business_ref_id,
            "status": self.store.invite_drafts[draft.draft_id].status.value, "receipt": receipt,
        }))

    def _start(self, event: VenueEvent) -> None:
        if not self.game_id:
            raise ValueError("没有形成待验证的牌局")
        expected = {person.sender_id for person in self.players}
        self.checks["social_refusal_delivery"] = bool(self.social_refusal_delivery_id and self.game_id
            and self.avoider.sender_id not in self._confirmed())
        self.checks["four_real_confirmations"] = bool(self.game_id and self._confirmed() == expected)
        self.checks["four_arrivals"] = self.arrivals == expected
        if not (self.checks["four_real_confirmations"] and self.checks["four_arrivals"]):
            raise ValueError("到点未凑齐四名实际确认且签到的客户")

        end = event.at + PLAY_DURATION
        opening = datetime.strptime(VENUE["opens"], "%H:%M").time()
        closing = datetime.strptime(VENUE["closes"], "%H:%M").time()
        candidates = [room["id"] for room in VENUE["rooms"] if room["seats"] >= 4 and not room["smoking"]
                      and event.at.date() == end.date() and opening <= event.at.time() < end.time() <= closing]
        available = self.store.search_room_availability(start_at=event.at, end_at=end)["available_room_ids"]
        booked = [item.room_id for item in self.store.room_reservations.values() if item.game_id == self.game_id
                  and item.status in {"held", "confirmed"} and item.start_at <= event.at < item.end_at]
        self.room_id = next((name for name in candidates if name in booked), None) or next(
            (name for name in candidates if name in available), None)
        if self.room_id is None:
            raise ValueError("营业时间内没有可承接本局的四人无烟包间")
        game, transitions = self.store.start_game_play(
            self.game_id, attendee_ids=sorted(self.arrivals), room_id=self.room_id, at=event.at, trace_id=event.event_id)
        self.booking_id = game.play_reservation_id
        self.expected_play_end_at = game.expected_play_end_at
        self.checks["actual_start"] = (game.is_playing and game.actual_started_at == event.at
                                       and game.actual_started_at != game.planned_start_at
                                       and game.expected_play_end_at == end)
        reservation = self.store.room_reservations[self.booking_id]
        self.evidence.append(self._record(event, game=game.to_dict(), occupied_reservation=reservation.to_dict(),
                                          transitions=[item.to_dict() for item in transitions]))
        self._enqueue("occupied", end - timedelta(minutes=1))
        self._enqueue("slot_query", end - timedelta(minutes=18), self.probe)
        for person in self.players:
            self._enqueue("departure", end, person)
        self._enqueue("finish", end)

    def _occupancy(self, event: VenueEvent) -> None:
        availability = self.store.search_room_availability(start_at=event.at, end_at=event.at + timedelta(minutes=2))
        self.checks["occupied_before_end"] = self.room_id in availability["occupied_room_ids"]
        before = set(self.store.room_reservations)
        try:
            self.store.reserve_room(conversation_id="venue-overlap-probe", game_id=None, start_at=event.at,
                end_at=event.at + timedelta(minutes=2), room_id=self.room_id, trace_id=event.event_id)
        except ValueError as exc:
            self.checks["overlap_denied"] = "room is unavailable" in str(exc) and set(self.store.room_reservations) == before
        else:
            self.checks["overlap_denied"] = False
        self.evidence.append(self._record(event, availability=availability, overlap_denied=self.checks["overlap_denied"]))

    def _finish(self, event: VenueEvent) -> None:
        self.checks["four_departures"] = self.departures == self.arrivals
        if not self.checks["four_departures"]:
            raise ValueError("缺少四位已到场客户的离店事件")
        game, changes = self.store.finish_game_play(self.game_id, at=event.at, trace_id=event.event_id)
        reopen = SQLiteAgentStore(self.run_dir / "agent.sqlite3")
        persisted = reopen.require_game(game.game_id)
        reservation = reopen.room_reservations[self.booking_id]
        self.checks["actual_four_hour_finish"] = bool(
            persisted.status == GameStatus.FINISHED and persisted.actual_ended_at and persisted.actual_started_at
            and persisted.actual_ended_at - persisted.actual_started_at == PLAY_DURATION)
        self.checks["release_persisted"] = reservation.status == "released" and any(
            step.entity_id == self.booking_id and step.to_status == "released" for step in changes)
        self.evidence.append(self._record(event, game=persisted.to_dict(), released=reservation.to_dict(),
            transitions=[step.to_dict() for step in changes]))
        if self.checks["actual_four_hour_finish"] and self.checks["release_persisted"]:
            self._enqueue("slot_query", event.at + timedelta(minutes=12), self.probe)
            self._enqueue("rebook", event.at + REBOOK_DELAY)

    def _rebook(self, event: VenueEvent) -> None:
        next_booking, _ = self.store.reserve_room(conversation_id="venue-next-booking", game_id=None, start_at=event.at,
            end_at=event.at + PROBE_DURATION, room_id=self.room_id, trace_id=event.event_id)
        persisted = SQLiteAgentStore(self.run_dir / "agent.sqlite3").room_reservations
        booking = persisted[next_booking.reservation_id]
        self.checks["same_room_rebookable"] = (
            booking.room_id == self.room_id and booking.status in {"held", "confirmed"}
            and booking.start_at == event.at and booking.end_at == event.at + PROBE_DURATION
            and persisted[self.booking_id].status == "released"
        )
        self.evidence.append(self._record(event, next_booking=next_booking.to_dict()))

    def _report(self) -> dict[str, Any]:
        records = self.channel.events()
        (self.run_dir / "chat.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
        self.checks["population"] = len(self.people) == 500 == len(set(self.people))
        if self.game_id is not None:
            due = self.board.next_due()
            self.checks["board_schedule_drained"] = (
                not self.board_poll_blocked and (due is None or due > self.board_cutoff)
                and not any(task.task_type == BOARD_TASK_TYPE and task.aggregate_id == self.channel.room_id
                            and task.status == ScheduledTaskStatus.FAILED for task in self.store.scheduled_tasks.values())
            )
        self.checks["conversation_evidence"] = any(row["direction"] == "inbound" for row in records) and any(
            row["direction"] == "outbound" for row in records)
        trace = self.run_dir / "agent_trace.log"
        if trace.exists():
            with trace.open(encoding="utf-8") as handle:
                self.checks["role_truth_not_in_agent_trace"] = bool(trace.stat().st_size) and not any(
                    "venue-persona-" in line for line in handle)
        else:
            self.checks["role_truth_not_in_agent_trace"] = None
        judge_input_exists = self.checks["conversation_evidence"] and self.evidence
        judged = None
        evidence = add_social_refusal_facts(_judge_evidence(self.evidence), records)
        refusal_row = next((row for row in evidence if row["event_id"] == self.social_refusal_delivery_id), None)
        refusal_facts = refusal_row.get("social_refusal_facts") if refusal_row else None
        if self.judge is not None and judge_input_exists:
            def judge(trace_id: str):
                return judge_with_short_evidence_ids(self.judge, "venue-four-hour-evening", evidence, trace_id=trace_id,
                                                     checks=_JUDGE_CHECKS, instructions=_JUDGE_INSTRUCTIONS)

            judged = self._retry_timed_out_model(judge("venue-independent-judge"), judge, stage="judge").to_dict()
        if self.judge is not None and judged is None:
            judged = {"status": "not_run", "verdict": "inconclusive", "reason": "没有有效的双向聊天证据"}
        social_refusal = self._social_refusal_report(judged, refusal_facts)
        wording = self._refine_inconclusive_refusal(judged, refusal_row, social_refusal)
        wording_evidence = {}
        if wording is not None:
            wording_evidence["independent_social_wording"] = wording
            if social_refusal["status"] in {"verified", "failed"}:
                wording_evidence["effective_independent_judge"] = {
                    "status": "valid", "verdict": "pass" if social_refusal["status"] == "verified" else "fail",
                }
        if isinstance(self.actor, ModelActor):
            self.checks["customer_decisions_from_model"] = bool(self.real_customer_actions and self.controlled_actions_from_model)
        physical_events = {row["kind"]: row for row in self.evidence if row["kind"] in {"start", "occupied", "finish", "rebook"}}
        physical_checks = [self.checks[key] for key in _PHYSICAL_CHECKS]
        physical_play = {
            "status": "passed" if all(physical_checks) else "failed" if False in physical_checks else "not_run",
            "actual_started_at": physical_events.get("start", {}).get("game", {}).get("actual_started_at"),
            "actual_ended_at": physical_events.get("finish", {}).get("game", {}).get("actual_ended_at"),
            "evidence_ids": {kind: row["event_id"] for kind, row in physical_events.items()},
        }
        return {"checks": self.checks, "failed_events": self.failures, "independent_judge": judged,
                **wording_evidence,
                "model_transport_errors": self.model_transport_errors,
                "model_timeout_retries": self.model_timeout_retries,
                "physical_play": physical_play,
                "social_refusal": social_refusal,
                "social_refusal_facts": refusal_facts,
                "game_id": self.game_id, "room_id": self.room_id, "chat_events": len(records),
                "people": len(self.people), "virtual_end": self.clock.isoformat(),
                "real_customer_actions": self.real_customer_actions}

    def _refine_inconclusive_refusal(
        self, judged: dict[str, Any] | None, row: dict[str, Any] | None, social_refusal: dict[str, Any],
    ) -> dict[str, Any] | None:
        """只有完整证据下单独未定的拒桌文字可以补判，其他失败或缺证仍保留。"""
        facts = (row or {}).get("social_refusal_facts") or {}
        primary = (judged or {}).get("findings") or ()
        if (self.judge is None or not judged or judged["status"] != "valid" or social_refusal["status"] != "inconclusive"
                or {item["check"]: item["verdict"] for item in primary}
                    != {check: "inconclusive" if check == "social_refusal" else "pass" for check in _JUDGE_CHECKS}
                or not facts.get("source_verified") or not facts.get("declined_same_game_in_store")
                or facts.get("recorded_same_game_reoffers") or not facts.get("actual_private_shop_replies")):
            return None

        def judge(trace_id):
            return judge_verified_refusal_wording(self.judge, row, trace_id=trace_id)

        measured = self._retry_timed_out_model(judge("venue-independent-social-wording"), judge, stage="judge")
        expected = {row["event_id"], *(reply["message_id"] for reply in facts["actual_private_shop_replies"])}
        if (measured.status == "valid" and measured.verdict in {"pass", "fail"}
                and expected <= set(measured.findings[0].evidence_ids)):
            social_refusal["status"] = "verified" if measured.verdict == "pass" else "failed"
        return measured.to_dict()

    def _social_refusal_report(self, judged: dict[str, Any] | None, facts: dict[str, Any] | None) -> dict[str, Any]:
        """拒绝覆盖须同时具备原始交付、未占席和锚定该来话的独立判断。"""
        delivery = self.social_refusal_delivery_id
        finding = next((item for item in judged["findings"] if item["check"] == "social_refusal"), None) if (
            judged and judged["status"] == "valid") else None
        if finding and finding["verdict"] in {"pass", "fail"} and (
            not delivery or delivery not in finding["evidence_ids"] or not facts or not facts["source_verified"]
        ):
            judged.update(status="invalid", verdict="inconclusive", findings=(),
                          error="social refusal judgment lacks its actual causal delivery")
            finding = None
        if finding and finding["verdict"] == "pass" and facts["recorded_same_game_reoffers"]:
            judged.update(status="invalid", verdict="inconclusive", findings=(),
                          error="social refusal pass conflicts with a delivered same-game reoffer")
            finding = None
        status = "not_observed"
        if self.checks["social_refusal_delivery"]:
            if judged is None:
                status = "not_assessed"
            elif finding is None or finding["verdict"] == "inconclusive":
                status = "inconclusive"
            elif finding["verdict"] == "pass":
                status = "verified"
            elif facts and facts["declined_same_game_in_store"]:
                status = "failed"
        return {"status": status, "delivery_evidence_id": delivery}
