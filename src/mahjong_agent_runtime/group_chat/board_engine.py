"""维护群聊缺人局看板的实时投影、编号快照和发布节流。

"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
import hashlib
import json
from typing import Any

from ..domains.game_domain import START_KIND_ASAP_WHEN_FULL
from ..knowledge import default_terminology_repository
from ..models import Game, GameStatus, MessageReference, ToolCall, new_id, now
from ..stores import AgentStore
from ..domains.tools.gateway import backend_tool_idempotency_key
from .messenger import GroupMessenger
from .models import BoardSnapshot, BoardSnapshotItem, GameConversationLink, GroupMessage
from .parsing import parse_game_post
from .projections import public_game_start_display
from .topic_models import PUBLIC_TOPIC_FIELDS, validate_public_topic_summary


BOARD_TASK_TYPE = "publish_group_board"
DEFAULT_MERGE_WINDOW_SECONDS = 30


class BoardPublicationError(RuntimeError):
    """看板没有取得真实交付回执；调度器须保留重试责任，不能记为发布完成。"""


class BoardEngine:
    """将活动牌局投影为带版本、可解析引用的房间公告板。

    牌局聚合仍是事实来源。公告板快照仅冻结一条对外消息所使用的公开编号，因此即使当前公告板被重新排序，被引用的声明仍可精确解析。
    """

    CRITICAL_EVENTS = frozenset({"seat_claimed", "seat_released"})

    def __init__(
        self,
        *,
        store: AgentStore,
        messenger: GroupMessenger,
        clock: Callable[[], datetime] = now,
    ) -> None:
        self.store = store
        self.messenger = messenger
        self.clock = clock

    def on_game_event(
        self,
        room_id: str,
        event_type: str,
        *,
        trace_id: str,
    ):
        """合并常规变更，并将座位变更移到队首。"""

        policy = self.store.get_group_room_policy(room_id)
        merge_seconds = (
            max(0, int(policy.merge_window_seconds))
            if policy is not None
            else DEFAULT_MERGE_WINDOW_SECONDS
        )
        urgent = event_type in self.CRITICAL_EVENTS
        due_at = self.clock() if urgent else self.clock() + timedelta(seconds=merge_seconds)
        return self.store.ensure_group_board_publish_task(
            room_id=room_id,
            due_at=due_at,
            trace_id=trace_id,
            urgent=urgent,
        )

    def import_game_from_post(self, msg: GroupMessage, *, trace_id: str) -> Game:
        """为已获授权的内部调用方导入结构化帖子。普通群消息入口不调用此方法。"""

        parsed = parse_game_post(msg.text, anchor=msg.sent_at)
        if parsed is None:
            raise ValueError("message is not an explicit game post")
        identity = self.store.get_channel_identity(msg.channel, msg.sender_external_id)
        customer_id = identity.customer_id if identity is not None else f"{msg.channel}:{msg.sender_external_id}"
        public_name = (
            identity.public_name
            if identity is not None and identity.public_name
            else msg.sender_name or msg.sender_external_id
        )
        game_conversation_id = f"{msg.conversation_id}:post:{msg.message_id}"
        game, _ = self.store.create_game(
            conversation_id=game_conversation_id,
            organizer_id=customer_id,
            organizer_name=public_name,
            requirement=parsed,
            known_players=[],
            trace_id=trace_id,
        )
        self.store.link_game_conversation(
            GameConversationLink(
                link_id=new_id("game_conversation_link"),
                game_id=game.game_id,
                conversation_id=msg.conversation_id,
                room_id=msg.room_id,
                customer_id=customer_id,
                link_type="origin",
            )
        )
        self.on_game_event(msg.room_id, "game_created", trace_id=trace_id)
        return game

    def register_self_recruitment(self, msg: GroupMessage, receipt: dict, *, trace_id: str,
                                  tool_gateway) -> Game | None:
        """把已核验的自招来源静默登记到本群看板，不接手邀约或确认任何新成员。

        完整发局按本人明示条件创建；明确续接本人原局时，单独补充的同桌偏好写回该局。
        工具网关负责写入与幂等，本入口不发 Hook，避免把静默登记变成委托匹配或外发。
        """
        policy = self.store.get_group_room_policy(msg.room_id)
        identity = self.store.get_channel_identity(msg.channel, msg.sender_external_id)
        if (policy is None or not policy.managed or not policy.board_enabled
                or policy.channel != msg.channel or identity is None):
            return None
        recorded = self.store.get_group_topic_message(
            msg.channel, msg.room_id, identity.customer_id, msg.message_id,
        )
        if (recorded is None or receipt.get("message") != recorded
                or recorded["text"] != msg.text
                or datetime.fromisoformat(recorded["sent_at"]) != msg.sent_at):
            return None
        decision = recorded["classification"]
        if (recorded["intent"] != "self_recruitment" or decision.get("intent") != "self_recruitment"
                or decision.get("goal_relation") != "none" or decision.get("task_context_id") is not None
                or decision.get("topic_action") not in {"new", "existing"}
                or decision.get("source_message_id") != msg.message_id):
            return None
        constraints = decision.get("constraints")
        if not isinstance(constraints, dict):
            return None
        if constraints.get("source_message_id", msg.message_id) != msg.message_id:
            return None
        if tool_gateway.store is not self.store:
            raise ValueError("self recruitment gateway must use the board store")
        conversation_id = f"group:{msg.room_id}:customer:{identity.customer_id}"
        name = identity.public_name or msg.sender_name
        source_message_id = f"group-self-recruitment:{recorded['source_key']}"
        gender_fields = {"preferred_gender", "required_gender"}
        if (decision["topic_action"] == "existing" and gender_fields & constraints.keys()
                and set(constraints) <= gender_fields | {"source_message_id"}):
            patch = {key: value for key, value in constraints.items() if key in gender_fields}
            return self._supplement_self_recruitment_preference(
                msg, recorded, patch, customer_id=identity.customer_id, conversation_id=conversation_id,
                name=name, source_message_id=source_message_id, trace_id=trace_id, tool_gateway=tool_gateway,
            )
        party_size = constraints.get("party_size")
        # 房间默认值只参与建局参数，不能被记录成客户本人的原话条件。
        if constraints.get("game_type") is None and policy.self_recruitment_default_game_type:
            constraints = {**constraints, "game_type": policy.self_recruitment_default_game_type}
        missing = [field for field in ("party_size", "game_type") if constraints.get(field) is None]
        if missing:
            recorder = tool_gateway.trace_recorder
            if recorder is not None:
                recorder.record(trace_id, "group_self_recruitment_not_registered", {
                    "reason": "missing_required_fields", "fields": missing, "source_message_id": msg.message_id,
                })
            return None
        if type(party_size) is not int or not 1 <= party_size <= 3:
            return None
        requirement = self._self_recruitment_requirement(constraints)
        if requirement is None:
            return None
        call = ToolCall(name="create_game", reason="按已登记自招来源静默保存本群待组局", arguments={
            "organizer_id": identity.customer_id, "organizer_name": name,
            "requirement": requirement,
            "requesting_party": {
                "contact_id": identity.customer_id, "contact_name": name, "seat_count": party_size,
                "known_member_ids": [identity.customer_id], "anonymous_seat_count": party_size - 1,
            },
        })
        key = backend_tool_idempotency_key(call, conversation_id=conversation_id,
            sender_id=identity.customer_id, source_message_id=source_message_id)
        previous = self.store.idempotent_result(key)
        # 新来源不能自行覆盖或再开本人已存在的方案；同来源交给网关复用原结果。
        # 已关闭局的旧结果也只能读取当前状态，不能重建或重新安排发布。
        if previous is None and any(
            game.conversation_id == conversation_id and game.organizer_id == identity.customer_id
            and game.status in {GameStatus.FORMING, GameStatus.INVITING, GameStatus.READY}
            for game in self.store.games.values()
        ):
            return None
        result = tool_gateway.execute(call, trace_id=trace_id, conversation_id=conversation_id,
            sender_id=identity.customer_id, sender_name=name, step_index=0,
            source_message_id=source_message_id)
        if not result.called or not result.allowed or result.error:
            return None
        game_id = result.result.get("game", {}).get("game_id")
        game = self.store.games.get(game_id)
        if (game is None or game.status not in {GameStatus.FORMING, GameStatus.INVITING}
                or game.expires_at is not None and game.expires_at <= self.clock()):
            return None
        return game

    def _supplement_self_recruitment_preference(self, msg, recorded, patch, *, customer_id, conversation_id,
                                               name, source_message_id, trace_id, tool_gateway):
        """只写入当前话题唯一绑定、仍缺人的本人原局；旧来源不能覆盖较新的补充。"""
        topic_id = recorded["topic_id"]
        if not topic_id or recorded["classification"].get("topic_id") != topic_id:
            return None
        bindings = self.store.list_group_topic_customer_bindings(topic_id, customer_id)
        linked = {link.game_id for link in self.store.game_conversation_links(room_id=msg.room_id)
                  if link.link_type == "origin" and link.customer_id == customer_id
                  and link.conversation_id == conversation_id}
        bound = {game_id for binding in bindings if binding["conversation_id"] == conversation_id
                 and binding.get("task_context_id") is None for game_id in binding["game_ids"]}
        games = self.store.games
        stamp = self.clock()
        matches = []
        for game_id in bound & linked:
            game = games.get(game_id)
            if (game and game.organizer_id == customer_id and game.conversation_id == conversation_id
                    and game.requirement.get("recruitment_mode") == "board_only"
                    and game.status in {GameStatus.FORMING, GameStatus.INVITING} and game.remaining_seats() > 0
                    and (game.expires_at is None or game.expires_at > stamp)):
                matches.append(game_id)
        if len(matches) != 1:
            return None
        game_id = matches[0]
        if any(game_id in binding["game_ids"] and datetime.fromisoformat(binding["updated_at"]) > msg.sent_at
               for binding in bindings):
            return None
        call = ToolCall(name="update_game_requirement", reason="本人补充原自招局的同桌偏好",
                        arguments={"game_id": game_id, "requirement_patch": patch,
                                   "reason": "group_self_recruitment_preference_supplement"})
        result = tool_gateway.execute(call, trace_id=trace_id, conversation_id=conversation_id, sender_id=customer_id,
                                      sender_name=name, step_index=0, source_message_id=source_message_id)
        if not result.called or not result.allowed or result.error:
            return None
        return self.store.require_game(game_id)

    @staticmethod
    def _self_recruitment_requirement(constraints: dict) -> dict | None:
        """投影已分类字段，不从原句猜档位、日期、开场方式或未提供的成员身份。"""
        try:
            public = validate_public_topic_summary({
                key: value for key, value in constraints.items() if key in PUBLIC_TOPIC_FIELDS
            })
            if not public.get("game_type"):
                return None
            requirement = {"game_type": public["game_type"], "recruitment_mode": "board_only"}
            if "ruleset" in public:
                variant = next((term for term in default_terminology_repository().terms
                    if term.category == "game_variant" and term.canonical.get("ruleset") == public["ruleset"]
                    and term.canonical.get("requested_game") == public["game_type"]), None)
                if variant is None:
                    return None
                requirement.update(ruleset=public["ruleset"], game_variant=variant.canonical["game_variant"])
            if public.get("stakes"):
                if len(public["stakes"]) == 1:
                    requirement["stake"] = public["stakes"][0]
                else:
                    requirement.update(stakes=public["stakes"], stake_label="/".join(public["stakes"]))
            if "smoking" in public:
                requirement["smoke_preference"] = public["smoking"]
            for field in ("preferred_gender", "required_gender"):
                if field in constraints:
                    if constraints[field] not in ("男", "女"):
                        return None
                    requirement[field] = constraints[field]
            if (requirement.get("preferred_gender") and requirement.get("required_gender")
                    and requirement["preferred_gender"] != requirement["required_gender"]):
                return None
            if public.get("date"):
                requirement["date"] = public["date"]
            window = public.get("time_window") or {}
            if window:
                requirement["time_window"] = window
                if window.get("start"):
                    # 仅钟点或宽窗口不等于已确认日期的固定开场；禁止自动套创建当天。
                    requirement["start_time"] = window["start"] if not window.get("end") else (
                        f"{window['start']}–{window['end']}"
                    )
            for field in ("planned_start_at", "latest_arrival_at", "latest_wait_at", "end_at"):
                if field not in constraints:
                    continue
                value = datetime.fromisoformat(constraints[field])
                if value.tzinfo is None:
                    return None
                requirement[field] = value.isoformat()
            if "planned_start_at" in requirement:
                value = datetime.fromisoformat(requirement["planned_start_at"])
                requirement.update(start_time_kind="scheduled", start_time=value.strftime("%H:%M"))
                if "end_at" in requirement:
                    # 同来源明确的起止时刻应进入规范时长，否则生命周期会用默认四小时
                    # 覆盖已说明的五小时。只有完整且有效的区间才推导，单边时间仍未知。
                    hours = (datetime.fromisoformat(requirement["end_at"]) - value).total_seconds() / 3600
                    if not 0 < hours <= 24:
                        return None
                    requirement.update(duration_hours=hours, duration_kind="fixed")
            if constraints.get("start_when_full") is True:
                if "planned_start_at" in requirement or window.get("start") or window.get("end"):
                    return None
                requirement.update(start_time_kind=START_KIND_ASAP_WHEN_FULL, start_time="人齐开")
            return requirement
        except (TypeError, ValueError):
            return None

    def publish(
        self, room_id: str, *, trace_id: str, publication_id: str | None = None,
    ) -> BoardSnapshot | None:
        """按最新有效局生成看板；同一任务及相同内容的重试复用交付标识。

        空板属于无需发布，允许完成任务；空回执或被阻断的外发不能与空板混同。
        publication_id来自持久刷新任务，不能使用每次重试都会改变的trace_id代替。
        内容变化会生成新版编号快照，旧锚点保留，避免把新内容冒充旧版消息。
        """
        if publication_id is not None and (not isinstance(publication_id, str) or not publication_id.strip()):
            raise ValueError("publication_id must be a nonempty string")

        games = self.store.get_board_eligible_games(room_id)
        if not games:
            return None
        items: list[BoardSnapshotItem] = []
        lines = ["当前缺人局："]
        for item_no, game in enumerate(games, start=1):
            rendered = self._render_game(item_no, game)
            items.append(BoardSnapshotItem(item_no=item_no, game_id=game.game_id, rendered_text=rendered))
            lines.append(rendered)
        # 自招登记不是店方代办入局，混合看板的认领示例只能指向普通托管局。
        ordinary_numbers = [index for index, game in enumerate(games, start=1)
                            if game.requirement.get("recruitment_mode") != "board_only"]
        if len(ordinary_numbers) == len(games):
            lines.extend(("", "回复编号即可认领，如\"2来\""))
        elif ordinary_numbers:
            lines.extend(("", f'普通组局可回复编号认领，如"{ordinary_numbers[0]}来"；群友自招请先联系发局人。'))
        else:
            lines.extend(("", "群友自招请先联系发局人。"))
        board_text = "\n".join(lines)
        snapshot_id = (
            "board_snapshot_" + hashlib.sha256(
                json.dumps({"room_id": room_id, "publication_id": publication_id, "text": board_text,
                    # 同文案也可能来自不同牌局；编号->game_id身份必须进入不可变版本。
                    "items": [(item.item_no, item.game_id, item.rendered_text) for item in items]},
                    ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:32]
            if publication_id is not None else new_id("board_snapshot")
        )
        conversation_id = f"wechaty:room:{room_id}"
        source_message_id = f"group_board:{snapshot_id}"
        previous = self.store.get_board_snapshot(snapshot_id)
        if previous is not None:
            # 发送成功后任务完成前崩溃时，已有快照足以证明交付；不重复发送。
            self._link_published_reference(previous, source_message_id)
            return previous
        # 先登记业务锚点再调用通道；部分微信 Puppet 只有在后续
        # 自消息回显中才会暴露平台消息标识。
        self.store.register_message_reference(
            MessageReference(
                message_id=source_message_id,
                conversation_id=conversation_id,
                business_ref_type="group_board_snapshot",
                business_ref_id=snapshot_id,
                text=board_text,
                channel="wechaty",
                recipient_id=room_id,
                metadata={
                    "room_id": room_id,
                    "item_game_ids": [item.game_id for item in items],
                    "trace_id": trace_id,
                    "reference_role": "transport_source_anchor",
                },
            )
        )
        external_message_id = self.messenger.send_group_message(
            room_id,
            board_text,
            metadata={
                "trace_id": trace_id,
                "message_type": "game_board",
                "source_message_id": source_message_id,
                "business_ref_type": "group_board_snapshot",
                "business_ref_id": snapshot_id,
            },
        )
        if not external_message_id or external_message_id.startswith("suppressed:"):
            raise BoardPublicationError("group board delivery has no successful receipt")
        snapshot = BoardSnapshot(
            snapshot_id=snapshot_id,
            room_id=room_id,
            conversation_id=conversation_id,
            external_message_id=external_message_id,
            rendered_text=board_text,
            items=items,
            created_at=self.clock(),
        )
        self.store.save_board_snapshot(snapshot)
        self._link_published_reference(snapshot, source_message_id)
        return snapshot

    def _link_published_reference(self, snapshot: BoardSnapshot, source_message_id: str) -> None:
        """幂等关联已发送消息与编号快照；快照落库后的重试也能补齐引用。"""
        if snapshot.external_message_id != source_message_id:
            self.store.link_message_reference(
                conversation_id=snapshot.conversation_id,
                message_id=snapshot.external_message_id,
                source_message_id=source_message_id,
                channel="wechaty",
                text=snapshot.rendered_text,
                metadata={"reference_role": "transport_response"},
            )

    def resolve_item_no(
        self,
        room_id: str,
        item_no: int,
        quoted_message_id: str | None = None,
    ) -> Game | None:
        """解析并确定itemno，返回消除歧义后的权威结果。

"""

        snapshot = None
        if quoted_message_id:
            snapshot = self.store.get_board_snapshot_by_message_id(room_id, quoted_message_id)
            if snapshot is None:
                reference = self.store.resolve_message_reference(
                    conversation_id=f"wechaty:room:{room_id}",
                    message_id=quoted_message_id,
                )
                if reference is not None and reference.business_ref_type == "group_board_snapshot":
                    candidate = self.store.get_board_snapshot(reference.business_ref_id)
                    if candidate is not None and candidate.room_id == room_id:
                        snapshot = candidate
        else:
            snapshot = self.store.get_latest_board_snapshot(room_id)
        if snapshot is None:
            return None
        board_item = next((item for item in snapshot.items if item.item_no == int(item_no)), None)
        if board_item is None:
            return None
        try:
            game = self.store.require_game(board_item.game_id)
        except (KeyError, ValueError):
            return None
        if game.status not in {GameStatus.FORMING, GameStatus.INVITING} or game.remaining_seats() <= 0:
            return None
        return game

    @staticmethod
    def _render_game(item_no: int, game: Game) -> str:
        """仅渲染玩法、档位、烟况、时间和权威座位数，不带参与者身份。"""
        requirement = game.requirement
        start = public_game_start_display(game)
        stake = str(
            requirement.get("stake_label")
            or requirement.get("stake")
            or requirement.get("base_stake")
            or "档位待定"
        )
        smoke = {
            "no_smoking": "无烟",
            "smoking": "有烟",
            "any": "烟都可",
        }.get(str(requirement.get("smoke_preference") or ""), "烟况待定")
        remaining = game.remaining_seats()
        claimed = max(0, game.seats_total - remaining)
        seat_code = f"{claimed}7{remaining}"
        game_name = BoardEngine._display_game_name(requirement)
        fields = [f"{item_no}、{start}"]
        if game_name:
            fields.append(game_name)
        fields.extend((stake, smoke, seat_code))
        if requirement.get("recruitment_mode") == "board_only":
            fields.append("群友自招（先联系发局人）")
        return " ".join(fields)

    @staticmethod
    def _display_game_name(requirement: dict[str, Any]) -> str:
        """按公开玩法允许列表显示；未知标识不原样外露，也不猜成已知玩法。"""
        # 私聊建局使用game_type；requested_game仍是当前群报局解析器的输出，二者冲突以主类型为准。
        game_type = str(requirement.get("game_type") or requirement.get("requested_game") or "")
        variant = str(requirement.get("game_variant") or "")
        if game_type == "hangzhou_mahjong" and variant == "caiqiao":
            return "cq"
        return {
            "hangzhou_mahjong": "杭麻",
            "sichuan_mahjong": "川麻",
            "red_center_mahjong": "红中",
            "national_standard_mahjong": "国标",
        }.get(game_type, "玩法待确认")


__all__ = ["BOARD_TASK_TYPE", "BoardEngine"]
