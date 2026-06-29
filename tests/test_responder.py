from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from mahjong_agent import (
    AgentResponder,
    ChannelType,
    CustomerProfile,
    GameStatus,
    LLMResolution,
    Message,
    ReplyAction,
)


TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 6, 16, 12, 0, tzinfo=TZ)


class FakeLLMResolver:
    def __init__(self, resolution: LLMResolution) -> None:
        self.resolution = resolution
        self.calls: list[Message] = []
        self.contexts: list[dict | None] = []

    def resolve(self, message: Message, context: dict | None = None) -> LLMResolution:
        self.calls.append(message)
        self.contexts.append(context)
        return self.resolution


def seed_responder() -> AgentResponder:
    responder = AgentResponder(invite_limit=3)
    for customer in [
        CustomerProfile(
            id="amy",
            display_name="Amy",
            preferred_levels=["0.5"],
            tags=["无烟"],
            smoke_free_preference=True,
            usual_start_hours=[17, 18],
        ),
        CustomerProfile(
            id="chen",
            display_name="陈姐",
            preferred_levels=["0.5", "1"],
            tags=["无烟", "熟人局"],
            smoke_free_preference=True,
            usual_start_hours=[17],
        ),
        CustomerProfile(
            id="ben",
            display_name="Ben",
            preferred_levels=["2"],
            tags=["可吸烟"],
            smoke_free_preference=False,
            usual_start_hours=[21],
        ),
    ]:
        responder.core.upsert_customer(customer)
    return responder


def seed_responder_with_llm(resolution: LLMResolution) -> tuple[AgentResponder, FakeLLMResolver]:
    fake = FakeLLMResolver(resolution)
    responder = seed_responder()
    responder.llm_resolver = fake
    return responder, fake


def msg(text: str, sender_id: str = "host", channel_id: str = "group") -> Message:
    return Message(
        text=text,
        sender_id=sender_id,
        sender_name=sender_id,
        channel_id=channel_id,
        channel_type=ChannelType.WECHAT_GROUP,
    )


def timed_msg(
    text: str,
    seconds: int,
    sender_id: str = "host",
    channel_id: str = "group",
) -> Message:
    return Message(
        text=text,
        sender_id=sender_id,
        sender_name=sender_id,
        channel_id=channel_id,
        channel_type=ChannelType.WECHAT_GROUP,
        sent_at=NOW + timedelta(seconds=seconds),
    )


def test_clear_game_request_queues_invitation_drafts() -> None:
    responder = seed_responder()

    decision = responder.respond(msg("今晚5点 0.5 三缺一 无烟 打四小时"), now=NOW)

    assert decision.action == ReplyAction.QUEUE_INVITES
    assert decision.game_id is not None
    assert decision.needs_human_review is True
    assert "建议先私聊" in decision.reply_text
    assert decision.draft_group_post is not None
    assert len(decision.invitation_drafts) == 2


def test_full_room_suggests_next_available_time_without_invites() -> None:
    responder = seed_responder()
    now = datetime(2026, 6, 16, 16, 0, tzinfo=TZ)
    responder.core.configure_room_capacity(1)
    responder.core.add_room_hold(
        start_at=datetime(2026, 6, 16, 16, 0, tzinfo=TZ),
        end_at=datetime(2026, 6, 16, 18, 0, tzinfo=TZ),
        room_id="room-1",
        source="room_schedule",
    )

    decision = responder.respond(
        msg("5点 0.5 三缺一 无烟", sender_id="host", channel_id="group"),
        now=now,
    )

    assert decision.action == ReplyAction.ASK_CLARIFICATION
    assert decision.game_id is not None
    assert decision.invitation_drafts == []
    assert decision.draft_group_post is None
    assert "17:00" in decision.reply_text
    assert "18:00" in decision.reply_text
    assert "满房" in decision.reply_text
    assert "改到 18:00" in decision.reply_text
    assert any("暂停邀约" in note for note in decision.notes)


def test_ambiguous_time_asks_clarification() -> None:
    responder = seed_responder()

    decision = responder.respond(msg("0.5 5点开 371 无烟"), now=NOW)

    assert decision.action == ReplyAction.ASK_CLARIFICATION
    assert "上午还是下午" in decision.reply_text


def test_clear_sichuan_partial_game_enters_pending_queue() -> None:
    responder = seed_responder()

    decision = responder.respond(msg("川麻216三等一"), now=NOW)

    assert decision.action == ReplyAction.CREATE_PENDING_GAME
    assert decision.game_id is not None
    assert decision.invitation_drafts == []
    assert decision.draft_group_post is None
    assert "待组局队列" in decision.reply_text
    assert "川麻" in decision.reply_text
    assert "2-16档" in decision.reply_text
    assert "3缺1" in decision.reply_text
    assert "希望几点开局" in decision.reply_text
    game = responder.core.store.games[decision.game_id]
    assert game.status == GameStatus.NEED_CLARIFICATION


def test_clear_hongzhong_partial_game_enters_pending_queue_before_soft_lead_reply() -> None:
    responder = seed_responder()

    decision = responder.respond(msg("红中🀄368鲨鱼，371，无烟"), now=NOW)

    assert decision.action == ReplyAction.CREATE_PENDING_GAME
    assert decision.game_id is not None
    assert "待组局队列" in decision.reply_text
    assert "红中麻将" in decision.reply_text
    assert "鲨鱼" in decision.reply_text
    assert "368档" in decision.reply_text
    assert "3缺1" in decision.reply_text
    assert "希望几点开局" in decision.reply_text


def test_invited_customer_accepts_and_locks_seat() -> None:
    responder = seed_responder()
    first = responder.respond(msg("今晚5点 0.5 三缺一 无烟"), now=NOW)
    assert first.invitation_drafts

    decision = responder.respond(msg("我来", sender_id=first.invitation_drafts[0].customer_id), now=NOW)

    assert decision.action == ReplyAction.ACCEPT_SEAT
    assert decision.game_id == first.game_id
    assert "人数已齐" in decision.reply_text
    game = responder.core.store.games[first.game_id]
    assert game.status == GameStatus.CONFIRMED


def test_locked_customer_is_not_invited_to_second_active_game() -> None:
    responder = seed_responder()
    first = responder.respond(msg("今晚5点 0.5 三缺一 无烟", sender_id="host1", channel_id="group1"), now=NOW)
    assert first.invitation_drafts

    second = responder.respond(msg("今晚6点 0.5 三缺一 无烟", sender_id="host2", channel_id="group2"), now=NOW)

    assert second.action == ReplyAction.CREATE_GAME
    assert second.invitation_drafts == []
    assert "没有高匹配候选人" in second.reply_text


def test_accepted_customer_cannot_join_another_active_game() -> None:
    responder = seed_responder()
    first = responder.respond(msg("今晚5点 0.5 三缺一 无烟", sender_id="host1", channel_id="group1"), now=NOW)
    amy_invite = next(invitation for invitation in first.invitation_drafts if invitation.customer_id == "amy")

    accepted = responder.respond(msg("我来", sender_id=amy_invite.customer_id, channel_id="group1"), now=NOW)
    assert accepted.action == ReplyAction.ACCEPT_SEAT

    second = responder.respond(msg("今晚6点 0.5 三缺一 无烟", sender_id="host2", channel_id="group2"), now=NOW)
    assert all(invitation.customer_id != "amy" for invitation in second.invitation_drafts)

    duplicate_join = responder.respond(msg("我来", sender_id="amy", channel_id="group2"), now=NOW)

    assert duplicate_join.action == ReplyAction.DECLINE_INVITE
    assert "不重复帮你安排" in duplicate_join.reply_text


def test_completed_morning_game_does_not_block_evening_game() -> None:
    responder = seed_responder()
    responder.core.store.customers["amy"].max_games_per_day = 2
    morning_now = datetime(2026, 6, 16, 7, 0, tzinfo=TZ)
    morning = responder.respond(
        msg("早上9点 0.5 三缺一 无烟", sender_id="host1", channel_id="morning_group"),
        now=morning_now,
    )
    assert morning.action == ReplyAction.QUEUE_INVITES
    amy_invite = next(invitation for invitation in morning.invitation_drafts if invitation.customer_id == "amy")

    accepted = responder.respond(
        msg("我来", sender_id=amy_invite.customer_id, channel_id="morning_group"),
        now=morning_now,
    )
    assert accepted.action == ReplyAction.ACCEPT_SEAT
    assert responder.core.store.games[morning.game_id].status == GameStatus.CONFIRMED

    evening = responder.respond(
        msg("今晚7点 0.5 三缺一 无烟", sender_id="host1", channel_id="evening_group"),
        now=datetime(2026, 6, 16, 15, 0, tzinfo=TZ),
    )

    assert responder.core.store.games[morning.game_id].status == GameStatus.COMPLETED
    assert evening.action == ReplyAction.QUEUE_INVITES
    assert {invitation.customer_id for invitation in evening.invitation_drafts} == {"amy", "chen"}


def test_default_fatigue_keeps_completed_player_out_of_evening_invites() -> None:
    responder = seed_responder()
    morning_now = datetime(2026, 6, 16, 7, 0, tzinfo=TZ)
    morning = responder.respond(
        msg("早上9点 0.5 三缺一 无烟", sender_id="host1", channel_id="morning_group"),
        now=morning_now,
    )
    amy_invite = next(invitation for invitation in morning.invitation_drafts if invitation.customer_id == "amy")
    responder.respond(
        msg("我来", sender_id=amy_invite.customer_id, channel_id="morning_group"),
        now=morning_now,
    )

    evening = responder.respond(
        msg("今晚7点 0.5 三缺一 无烟", sender_id="host2", channel_id="evening_group"),
        now=datetime(2026, 6, 16, 15, 0, tzinfo=TZ),
    )

    assert evening.action == ReplyAction.QUEUE_INVITES
    assert all(invitation.customer_id != "amy" for invitation in evening.invitation_drafts)
    assert any(invitation.customer_id == "chen" for invitation in evening.invitation_drafts)


def test_group_irrelevant_message_is_silent() -> None:
    responder = seed_responder()

    decision = responder.respond(msg("今天天气不错"), now=NOW)

    assert decision.action == ReplyAction.IGNORE
    assert decision.should_reply is False
    assert decision.reply_text == ""


def test_soft_mahjong_inquiry_asks_clarification_instead_of_silent() -> None:
    responder = seed_responder()

    decision = responder.respond(msg("今天下班有人打麻将吗"), now=NOW)

    assert decision.action == ReplyAction.ASK_CLARIFICATION
    assert decision.should_reply is True
    assert "几点到" in decision.reply_text
    assert "几个人" in decision.reply_text
    assert "帮你看看能不能拼一桌" in decision.reply_text
    assert "潜在客户" in responder.core.store.customers["host"].tags
    assert "组局意向" in responder.core.store.customers["host"].tags
    assert "下班后活跃" in responder.core.store.customers["host"].tags


def test_known_single_customer_inquiry_uses_profile_party_size() -> None:
    responder = seed_responder()
    responder.core.upsert_customer(
        CustomerProfile(
            id="zhang",
            display_name="张哥",
            usual_party_size=1,
            usual_party_size_confidence=0.9,
        )
    )

    decision = responder.respond(msg("今天下班有人打麻将吗", sender_id="zhang"), now=NOW)

    assert decision.action == ReplyAction.ASK_CLARIFICATION
    assert "你一个人" in decision.reply_text
    assert "几个人" not in decision.reply_text
    assert "几点到" in decision.reply_text
    assert "人数不一样" in decision.reply_text


def test_fragmented_messages_are_combined_for_clarification() -> None:
    responder = seed_responder()

    first = responder.respond(timed_msg("老板", 0), now=NOW)
    second = responder.respond(timed_msg("今天下午", 2), now=NOW + timedelta(seconds=2))
    third = responder.respond(timed_msg("有没有打麻将的", 4), now=NOW + timedelta(seconds=4))
    fourth = responder.respond(timed_msg("0.5或者1都行", 6), now=NOW + timedelta(seconds=6))
    fifth = responder.respond(timed_msg("烟也都可", 8), now=NOW + timedelta(seconds=8))

    assert first.action == ReplyAction.IGNORE
    assert second.action == ReplyAction.IGNORE
    assert third.action == ReplyAction.ASK_CLARIFICATION
    assert fourth.action == ReplyAction.ASK_CLARIFICATION
    assert fifth.action == ReplyAction.ASK_CLARIFICATION
    assert "今天下午" in fifth.reply_text
    assert "0.5或1都可以" in fifth.reply_text
    assert "烟况不限" in fifth.reply_text
    assert "几个人" in fifth.reply_text
    assert "几点能到" in fifth.reply_text
    assert any("碎片消息" in note for note in fifth.notes)


def test_fragmented_known_single_customer_does_not_ask_people_count() -> None:
    responder = seed_responder()
    responder.core.upsert_customer(
        CustomerProfile(
            id="zhang",
            display_name="张哥",
            usual_party_size=1,
            usual_party_size_confidence=0.9,
        )
    )

    for index, text in enumerate(["老板", "今天下午", "有没有打麻将的", "0.5或者1都行", "烟也都可"]):
        decision = responder.respond(
            timed_msg(text, index * 2, sender_id="zhang"),
            now=NOW + timedelta(seconds=index * 2),
        )

    assert decision.action == ReplyAction.ASK_CLARIFICATION
    assert "你一个人" in decision.reply_text
    assert "几个人" not in decision.reply_text
    assert "几点能到" in decision.reply_text
    assert "人数不一样" in decision.reply_text


def test_fragmented_messages_do_not_cross_senders() -> None:
    responder = seed_responder()

    responder.respond(timed_msg("今天下午", 0, sender_id="host-a"), now=NOW)
    decision = responder.respond(
        timed_msg("有没有打麻将的", 2, sender_id="host-b"),
        now=NOW + timedelta(seconds=2),
    )

    assert decision.action == ReplyAction.ASK_CLARIFICATION
    assert "今天下午" not in decision.reply_text


def test_llm_normalized_text_can_flow_back_into_core_parser() -> None:
    responder, fake = seed_responder_with_llm(
        LLMResolution(
            is_mahjong_related=True,
            intent="find_players",
            confidence=0.86,
            normalized_text="今晚7点 0.5 三缺一 无烟",
            notes=["fake llm"],
        )
    )

    decision = responder.respond(msg("老地方搭子 三缺一"), now=NOW)

    assert fake.calls
    assert decision.action == ReplyAction.QUEUE_INVITES
    assert decision.game_id is not None
    assert "06-16 19:00" in decision.reply_text
    assert any("LLM intent=find_players" in note for note in decision.notes)
    assert fake.contexts
    assert fake.contexts[0]["schema_version"] == "mahjong_context.v1"
    assert fake.contexts[0]["current_message"]["text"] == "老地方搭子 三缺一"
    assert fake.contexts[0]["tool_policy"]["tool_calling_enabled"] is False
    assert any("ContextBuilder" in note for note in decision.notes)
    assert decision.llm_context_digest == fake.contexts[0]["audit"]["context_digest"]
    assert decision.llm_context_snapshot == fake.contexts[0]


def test_llm_slots_can_flow_back_into_core_parser_when_text_is_messy() -> None:
    responder, fake = seed_responder_with_llm(
        LLMResolution(
            is_mahjong_related=True,
            intent="find_players",
            confidence=0.9,
            slots={
                "query_mode": {"value": "create_new", "confidence": 0.9, "source": "explicit"},
                "game_type": {"value": "hangzhou_mahjong", "confidence": 0.82, "source": "region_default"},
                "level": {"value": "0.5", "confidence": 0.92, "source": "explicit", "evidence": "0，5"},
                "start_time": {"value": "19:00", "confidence": 0.9, "source": "explicit", "evidence": "7p"},
                "missing_count": {"value": 1, "confidence": 0.95, "source": "explicit", "evidence": "三等一"},
                "smoke": {"value": "no_smoke", "confidence": 0.9, "source": "explicit", "evidence": "不抽"},
            },
            notes=["fake llm"],
        )
    )

    decision = responder.respond(msg("麻将老地方 7p 0，5 三等一 不抽"), now=NOW)

    assert fake.calls
    assert decision.action == ReplyAction.QUEUE_INVITES
    assert decision.game_id is not None
    assert "06-16 19:00" in decision.reply_text
    assert "0.5档" in decision.reply_text
    assert any("LLM slots=" in note for note in decision.notes)


def test_llm_slots_can_express_people_ready_and_overnight_strategy() -> None:
    responder = seed_responder()
    notes: list[str] = []

    normalized = responder._normalized_text_from_llm_slots(
        {
            "query_mode": {"value": "create_new", "confidence": 0.9, "source": "explicit"},
            "game_type": {"value": "hangzhou_mahjong", "confidence": 0.82, "source": "region_default"},
            "level": {"value": "0.5", "confidence": 0.92, "source": "explicit", "evidence": "五毛"},
            "start_time": {"value": None, "confidence": 0.0, "source": "explicit", "needs_confirmation": False},
            "start_time_mode": {
                "value": "people_ready",
                "confidence": 0.9,
                "source": "explicit",
                "evidence": "尽快开，时间可以商量",
            },
            "duration_mode": {"value": "overnight", "confidence": 0.9, "source": "explicit", "evidence": "通宵"},
            "known_players": {"value": 1, "confidence": 0.95, "source": "explicit", "evidence": "173"},
            "smoke": {"value": "any", "confidence": 0.9, "source": "explicit", "evidence": "烟随便"},
        },
        original_text="老板，五毛，173，通宵，尽快开吧，时间可以商量，烟随便",
        notes=notes,
    )

    assert normalized is not None
    assert "人齐开" in normalized
    assert "通宵" in normalized
    assert "0.5" in normalized
    assert "烟都可" in normalized
    assert "LLM start_time 槽位需要确认" not in " ".join(notes)


def test_llm_profile_party_size_slot_is_not_committed_as_fact() -> None:
    responder, fake = seed_responder_with_llm(
        LLMResolution(
            is_mahjong_related=True,
            intent="find_players",
            confidence=0.88,
            slots={
                "query_mode": {"value": "create_new", "confidence": 0.9, "source": "explicit"},
                "game_type": {"value": "hangzhou_mahjong", "confidence": 0.9, "source": "explicit"},
                "level": {"value": "0.5", "confidence": 0.9, "source": "explicit"},
                "start_time": {"value": "14:00", "confidence": 0.9, "source": "explicit"},
                "known_players": {"value": 1, "confidence": 0.9, "source": "profile"},
                "smoke": {"value": "no_smoke", "confidence": 0.9, "source": "explicit"},
            },
            reply_text="可以，我先帮你看。你一个人吗？",
            notes=["fake llm"],
        )
    )

    decision = responder.respond(msg("下午两点 0.5 无烟杭麻，帮我组一桌"), now=NOW)

    assert fake.calls
    assert decision.action == ReplyAction.ASK_CLARIFICATION
    assert decision.game_id is not None
    game = responder.core.store.games[decision.game_id]
    assert game.current_player_count is None
    assert game.missing_count is None
    assert (
        "一个人" in decision.reply_text
        or "几个人" in decision.reply_text
        or "几缺几" in decision.reply_text
    )
    assert any("人数槽位不是原文明确" in note for note in decision.notes)


def test_llm_related_but_incomplete_message_asks_followup() -> None:
    responder, fake = seed_responder_with_llm(
        LLMResolution(
            is_mahjong_related=True,
            intent="find_players",
            confidence=0.66,
            reply_text="你想什么时候打、几个人、打多大的？",
            notes=["fake llm"],
        )
    )

    decision = responder.respond(msg("有搭子吗"), now=NOW)

    assert fake.calls
    assert decision.action == ReplyAction.ASK_CLARIFICATION
    assert "什么时候打" in decision.reply_text
    assert "潜在客户" in responder.core.store.customers["host"].tags


def test_audio_transcript_can_create_potential_customer() -> None:
    responder = seed_responder()

    decision = responder.respond(
        msg("[语音]", sender_id="voice_user"),
        now=NOW,
    )
    assert decision.action == ReplyAction.IGNORE

    decision = responder.respond(
        Message(
            text="[语音]",
            sender_id="voice_user",
            sender_name="语音客",
            channel_id="group",
            channel_type=ChannelType.WECHAT_GROUP,
            metadata={"message_type": "audio", "audio_transcript": "下班想搓一把，有局吗"},
        ),
        now=NOW,
    )

    assert decision.action == ReplyAction.ASK_CLARIFICATION
    assert "帮你看看能不能拼一桌" in decision.reply_text
    customer = responder.core.store.customers["voice_user"]
    assert "潜在客户" in customer.tags
    assert "audio" in customer.metadata["last_lead_modalities"]


def test_image_ocr_can_create_clear_game_request() -> None:
    responder = seed_responder()

    decision = responder.respond(
        Message(
            text="[图片]",
            sender_id="image_user",
            sender_name="图片客",
            channel_id="group",
            channel_type=ChannelType.WECHAT_GROUP,
            metadata={
                "message_type": "image",
                "image_ocr_text": "群截图：今晚7点 0.5 三缺一 无烟",
            },
        ),
        now=NOW,
    )

    assert decision.action == ReplyAction.QUEUE_INVITES
    assert decision.draft_group_post is not None
    assert "0.5档" in decision.reply_text


def test_sticker_description_can_be_potential_customer() -> None:
    responder = seed_responder()

    decision = responder.respond(
        Message(
            text="[表情包]",
            sender_id="sticker_user",
            sender_name="表情客",
            channel_id="group",
            channel_type=ChannelType.WECHAT_GROUP,
            metadata={
                "message_type": "sticker",
                "sticker_description": "麻将表情包：🀄 约吗",
            },
        ),
        now=NOW,
    )

    assert decision.action == ReplyAction.ASK_CLARIFICATION
    assert "帮你看看能不能拼一桌" in decision.reply_text
    customer = responder.core.store.customers["sticker_user"]
    assert "潜在客户" in customer.tags
    assert "sticker" in customer.metadata["last_lead_modalities"]


def test_sensitive_message_goes_to_human_review() -> None:
    responder = seed_responder()

    decision = responder.respond(msg("这桌输赢结算你帮我代收一下"), now=NOW)

    assert decision.action == ReplyAction.HUMAN_REVIEW
    assert decision.needs_human_review is True
    assert "转人工" in decision.reply_text
