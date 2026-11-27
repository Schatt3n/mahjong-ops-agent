"""用实际席位和收件回执驱动模拟桌群的开场沟通。"""

from datetime import datetime, timedelta
import json
import re
import unicodedata

from mahjong_agent_runtime.domains.game_commitments import active_game_participant_ids
from mahjong_agent_runtime.domains.game_domain import START_KIND_ASAP_WHEN_FULL, flexible_opening_not_before
from mahjong_agent_runtime.domains.invitation_policy import invite_duration_terms
from mahjong_agent_runtime.domains.stake_values import format_number, parse_number
from mahjong_agent_runtime.group_chat.parsing import observe_clock_candidates
from mahjong_agent_runtime.llm import completion_diagnostics, model_response_source
from mahjong_agent_runtime.models import GameStatus

from .venue_channel import AGENT_ID, MessageEnvelope
from .venue_role_journal import ROLE_REPLY_DELAY
from .venue_table_opening import TABLE_OPENING_OFFER_LIMIT, TABLE_OPENING_OFFER_PREFIX, VenueTableOpening


_CONFIRM_PREFIX = "table-opening-confirm|"
_COMPANION_INQUIRY_PREFIX = "table-companion-inquiry:"
_HAN_HOURS = {digit: hours for hours, digit in enumerate("零一二三四五六七八九")} | {"两": 2, "半": 0.5}
_HOURS_VALUE = r"(?:[0-9]+(?:\.[0-9]+)?|[一二两三四五六七八九半])"
_HARD_PLAY_LIMIT = re.compile(
    r"(?:^|[、,。;!?\n]\s*)(?:我(?:今晚|今天|明天|这次|这边)?\s*)?"
    r"(?:最多|至多|顶多|只能|只可以|不能超过|不超过)\s*(?:能|可以|只能)?\s*(?:打|玩)?\s*"
    rf"(?:(?P<range>[一二两三四五六七八九]{{2}})|(?P<low>{_HOURS_VALUE})(?:\s*(?:到|至|[-~～–—])\s*(?P<high>{_HOURS_VALUE}))?)"
    r"\s*个?\s*(?P<half>半)?\s*(?:小时|钟头|[hH](?![a-zA-Z]))(?P<after_half>半)?"
)
_UNSETTLED_LIMIT_SUFFIX = re.compile(r"\s*(?:左右|上下|前后|吗|嘛|么)")
_ARRIVAL_PREFIX = re.compile(r"(?:^|\s)我(?:这边|这里|这儿)?\s*最早\s*(?:今晚|晚上|傍晚|下午|上午)?\s*$")
_ARRIVAL_SUFFIX = re.compile(r"[ \t]*(?:才(?:能|可以)?|能|可以)?[ \t]*(?:到店|赶到|到|过来)(?=[ \t]*(?:[,.!;，。！；\n]|$))")
_ARRIVAL_CLAUSE_SEPARATOR = re.compile(r"[，,。；;！!？?\n]")
_UNCERTAIN_ARRIVAL = re.compile(r"如果|假如|假设|要是|可能|大概|估计|不是|不一定|[“”「」\"]")
_OTHER_ARRIVAL_DATE = re.compile(r"昨天|昨晚|明天|明早|明晚|后天|上周|下周")
_PROMPT = """你在模拟麻将馆的四人桌群负责约开场。只用本群已送达的话，没收到的同意不能补写。
vote 仅判断 current 这一条顾客消息对 active_proposal 的态度。明确同意该版确切开场用 accept。明确不行或反提其他时间用 decline。比如「我最早 18:14 才能到，晚两分钟能参加吗」对 18:12 已明确赶不上，即使句尾有问号也用 decline。只有问能否方便晚一点、报不确定估计、条件式接受、对旧版回复或无开场提议用 other。引用其他提议不能当成当前的同意。
table_terms.agreed_duration_hours 是当前桌已明确约定的时长，null 表示未定。顾客说只打更短时间、或说符合某个条件才来时，不能当作无条件接受。按本群原话理解同一人的限制，直到本人明确撤回或修改。未定时长不能猜成已约定的时长，近似偏好也不能推成硬上限。
proposed_start 仅在本群顾客已说出可用时间或给出足够协商信息，且没有提议、当前来话明确拒绝或 active_proposal.declined_count 大于零时填写。一个人可以先说赶不及，随后自己或另一位才说明能接受的时间。用当地带时区 ISO 分钟，选一个未来开场，考虑已经说出的到店、离店和可打时长。对赶不上的人重提议时给问答及出发留几分钟，仍请四人重新表态。若当前时间有人同意且没有尚未撤回的拒绝，等其他人。无合适时间时为 null，不从身份或画像猜。
只输出 JSON {"vote":"accept|decline|other","proposed_start":null}。"""


def _explicit_play_limit(text: str) -> float | None:
    """只识别本人明确上限的常见词形，口语范围使用其上沿。"""
    text = unicodedata.normalize("NFKC", text)
    matches = list(_HARD_PLAY_LIMIT.finditer(text))
    if len(matches) != 1:
        return None
    match = matches[0]
    if _UNSETTLED_LIMIT_SUFFIX.match(text, match.end()):
        return None
    if match["range"]:
        low, hours = (_HAN_HOURS[digit] for digit in match["range"])
        if hours - low != 1:
            return None
    else:
        value = match["high"] or match["low"]
        hours = _HAN_HOURS[value] if value in _HAN_HOURS else parse_number(value)
    if hours is None or hours <= 0:
        return None
    return hours + 0.5 if match["half"] or match["after_half"] else hours


def companion_inquiry_id(game_id: str, revision_id: str | None) -> str:
    """用同一局和名册版本标记店家实际发出的同行提醒。"""
    message_id = f"{_COMPANION_INQUIRY_PREFIX}{game_id}"
    return f"{message_id}|{revision_id}" if revision_id is not None else message_id


def _own_arrival_floor(message: MessageEnvelope) -> tuple[datetime | None, set[datetime]]:
    """只核对本人明确陈述、能绑定到原话钟点的到店下界。其他问法交给语义核对。"""
    text = unicodedata.normalize("NFKC", message.text)
    if _OTHER_ARRIVAL_DATE.search(text):
        return None, set()
    clocks = observe_clock_candidates(text, anchor=message.sent_at, include_source_spans=True)
    lower = []
    for clock in clocks:
        begin, end = clock["source_span"]
        prefix = _ARRIVAL_CLAUSE_SEPARATOR.split(text[:begin])[-1]
        if (clock["resolution"] in {"explicit_day_or_24h", "explicit_daypart"} and not clock["qualifier"]
                and "." not in clock["clock_text"] and _ARRIVAL_PREFIX.search(prefix) and not _UNCERTAIN_ARRIVAL.search(prefix)
                and _ARRIVAL_SUFFIX.match(text, end)):
            value = datetime.fromisoformat(clock["local_candidate_at"])
            if value > message.sent_at:
                lower.append(value)
    return (lower[0] if len(lower) == 1 else None,
            {datetime.fromisoformat(clock["local_candidate_at"]) for clock in clocks})


def _late_arrival_option(earliest: datetime) -> datetime:
    """在已报最早到店后留一轮回复时间，再上取到便于沟通的五分钟刻度。"""
    ready = earliest + ROLE_REPLY_DELAY
    return ready + timedelta(minutes=(-ready.minute) % 5)


class VenueTableShop:
    """给滚动沙盒挂接独立的桌群处理器，同时保留原始判断和真实通道回执。"""

    def __init__(self, client, trace_path) -> None:
        self.client = client
        self.trace_path = trace_path

    def __call__(self, group, reply_to, at):
        game = group.store.require_game(group.game_id)
        if game.status != GameStatus.READY or game.actual_started_at:
            return None
        seat_ids = active_game_participant_ids(game)
        if group.member_ids(at=at) != {AGENT_ID, *seat_ids}:
            return None
        revision_id = group.roster_revision(at=at)
        if len(seat_ids) != 4:
            seats = game.seat_summary()
            # 席位已确认，仍须由实际报名联络人把同行邀请进群。
            if (len(seat_ids) < 4 and reply_to is None and seats["seats_total"] == seats["claimed_seats"] == 4
                    and seats["anonymous_seat_count"]
                    and (game.planned_start_at is None or at < game.planned_start_at)):
                message_id = companion_inquiry_id(group.game_id, revision_id)
                if not any(receipt.message.message_id == message_id for receipt in group.delivered_receipts()):
                    return self._message(group, message_id, at, "带朋友来的，麻烦把还没进群的同行拉进来哈。")
            return None
        opening = VenueTableOpening(group)
        if reply_to is not None and reply_to.roster_revision != revision_id:
            return None
        records = tuple(row for row in group.delivered_receipts()
                        if row.message.sent_at <= at and row.roster_revision == revision_id)
        proposed = [row for row in records if row.message.sender_id == AGENT_ID
                    and row.message.message_id.startswith(TABLE_OPENING_OFFER_PREFIX)]
        for receipt in proposed:
            start = datetime.fromisoformat(receipt.message.message_id.split("|", 2)[1])
            opening.record_proposal(receipt, start_at=start)
        active = opening.current()
        positions = {row.message.message_id: index for index, row in enumerate(records)}
        if active and active["proposal_id"] not in positions:
            active = None
        if active and active["committed"]:
            return self._confirmation(group, at, active, records)
        if game.planned_start_at is not None or game.requirement.get("start_time_kind") != START_KIND_ASAP_WHEN_FULL:
            return None
        agreed_hours = invite_duration_terms(game)["duration_hours"]
        floor = flexible_opening_not_before(game.requirement)
        if reply_to is None:
            message_id = f"table-opening-inquiry:{group.game_id}"
            if revision_id is not None:
                message_id = f"{message_id}|{revision_id}"
            hours = f"这桌打 {format_number(agreed_hours)} 小时。" if agreed_hours is not None else ""
            opening_time = f"最早 {floor:%H:%M} 能开。" if floor else ""
            return self._message(group, message_id, at, f"人齐了，{hours}{opening_time}大家各自几点能到？咱们在这儿定个开场。")

        source = reply_to.message
        observed = active if (active and positions[active["proposal_id"]] < positions[source.message_id]
                              and source.quoted_message_id in {None, active["proposal_id"]}) else None
        arrival_floor, other_times = _own_arrival_floor(source) if observed else (None, set())
        opening_at = datetime.fromisoformat(observed["start_at"]) if observed else None
        arrival_conflict = (arrival_floor if arrival_floor and arrival_floor.date() == opening_at.date()
                            and arrival_floor > opening_at else None)
        vote, start = self._decide(group, reply_to, at, records, observed, agreed_hours=agreed_hours, floor=floor,
                                   arrival_conflict=arrival_conflict, extra_clocks=other_times - {opening_at, arrival_floor})
        result = None
        if observed and vote in {"accept", "decline"}:
            result = opening.record_vote(reply_to, proposal_id=observed["proposal_id"], accepts=vote == "accept")
        if result == "committed":
            return self._confirmation(group, at, opening.current(), records)
        if (start is None or start <= at or start.second or start.microsecond or len(proposed) >= TABLE_OPENING_OFFER_LIMIT):
            return None
        if active and (observed is None or start.isoformat() == active["start_at"]
                       or opening.current()["declined_count"] == 0):
            return None
        known_late = {}
        if observed:
            # 严格下界按本群实际收件顺序重建。后来本人给了另一个明确下界才替换该记录。
            for receipt in records[positions[observed["proposal_id"]] + 1:positions[source.message_id] + 1]:
                if receipt.message.sender_id != AGENT_ID:
                    bound, _ = _own_arrival_floor(receipt.message)
                    if bound is not None and bound.date() == opening_at.date():
                        known_late[receipt.message.sender_id] = bound
        late = max((bound for bound in known_late.values() if bound > opening_at), default=None)
        if ((late is not None and start < late + ROLE_REPLY_DELAY)
                or not opening.accepts_bounded_proposal(start, same_day_as=opening_at if late is not None else None)):
            return None
        message_id = f"{TABLE_OPENING_OFFER_PREFIX}{start.isoformat()}|{source.message_id}"
        return opening.proposal_message(message_id, at=at, start_at=start)

    def _decide(self, group, reply, at, records, active, *, agreed_hours, floor, arrival_conflict, extra_clocks):
        """单轮只解释当前发言和同桌公开历史，调用失败继续保留待协商状态。"""
        body = {
            "now": at.isoformat(),
            "current": reply.history_row(),
            "active_proposal": active,
            "table_terms": {"agreed_duration_hours": agreed_hours},
            "history": [
                {"speaker": "shop" if row.message.sender_id == AGENT_ID else row.message.sender_id,
                 "message_id": row.message.message_id, "text": row.message.text,
                 "sent_at": row.message.sent_at.isoformat(), "quoted_message_id": row.message.quoted_message_id}
                for row in records[-24:]
            ],
        }
        # 只上传桌群现有事实；真实顾客的私人出行安排始终由各自角色保留。
        body["current"] = {key: body["current"][key] for key in
                           ("message_id", "sender_id", "text", "sent_at", "quoted_message_id")}
        if floor is not None:
            body["table_terms"]["opening_not_before_at"] = floor.isoformat()
        if arrival_conflict is not None:
            body["current"]["verified_arrival_not_before_at"] = arrival_conflict.isoformat()
        trace_id = f"table-shop:{group.game_id}:{reply.message.message_id}"
        evidence = {"trace_id": trace_id, "input": body}
        vote, start = "other", None
        try:
            raw = self.client.complete(
                [{"role": "system", "content": _PROMPT},
                 {"role": "user", "content": json.dumps(body, ensure_ascii=False)}],
                trace_id=trace_id, timeout_seconds=25,
            )
            evidence.update(raw=str(raw), response_source=model_response_source(raw, trace_id=trace_id, step_index=1))
            result = json.loads(str(raw))
            if set(result) != {"vote", "proposed_start"} or result["vote"] not in ("accept", "decline", "other"):
                raise ValueError("invalid table shop decision")
            if result["proposed_start"] is not None:
                start = datetime.fromisoformat(result["proposed_start"])
                if start.utcoffset() is None:
                    raise ValueError("table shop proposed opening has no timezone")
                start = start.astimezone(at.tzinfo)
            vote = result["vote"]
        except Exception as exc:
            evidence.update(error=type(exc).__name__, diagnostics=completion_diagnostics(exc))
            vote, start = "other", None
        if agreed_hours is not None and (limit := _explicit_play_limit(reply.message.text)) is not None and limit < agreed_hours:
            evidence["vote_guard"] = {"reason": "agreed_duration_exceeds_customer_limit", "customer_max_hours": limit,
                                      "model_vote": vote}
            vote, start = "decline", None
        elif arrival_conflict is not None:
            evidence["vote_guard"] = {"reason": "opening_before_explicit_arrival", "arrival_not_before": arrival_conflict.isoformat(),
                                      "model_vote": vote}
            vote = "decline"
            if (start is None or start < arrival_conflict + ROLE_REPLY_DELAY) and not active["declined_count"] and not extra_clocks:
                start = _late_arrival_option(arrival_conflict)
                evidence["tentative_late_arrival_proposal"] = start.isoformat()
        with self.trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(evidence, ensure_ascii=False) + chr(10))
        return vote, start

    def _confirmation(self, group, at, active, records):
        """已落相同牌局时才报定时，重跑复用原群消息。"""
        message_id = f"{_CONFIRM_PREFIX}{active['proposal_id']}"
        if any(row.message.message_id == message_id for row in records):
            return None
        start = datetime.fromisoformat(active["start_at"])
        when = start.strftime("%H:%M" if start.date() == at.date() else "%m月%d日 %H:%M")
        return self._message(group, message_id, at, f"那就 {when} 开，大家按这个时间到哈。")

    @staticmethod
    def _message(group, message_id, at, text):
        return MessageEnvelope(message_id, AGENT_ID, text, at, "group", group.room_id)
