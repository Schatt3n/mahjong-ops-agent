from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from mahjong_agent import AgentCore, CustomerProfile, GameStatus, InvitationStatus, Message, PlayPreference
from mahjong_agent.parser import MahjongMessageParser


TZ = ZoneInfo("Asia/Shanghai")


def test_parse_371_and_ambiguous_time() -> None:
    parser = MahjongMessageParser()
    message = Message(
        text="0.5 5点开 371 无烟 打四小时",
        sender_id="host",
        sender_name="张哥",
        channel_id="group",
    )

    result = parser.parse(message, now=datetime(2026, 6, 16, 12, 0, tzinfo=TZ))

    assert result.game is not None
    assert result.game.level == "0.5"
    assert result.game.current_player_count == 3
    assert result.game.missing_count == 1
    assert result.game.start_at == datetime(2026, 6, 16, 17, 0, tzinfo=TZ)
    assert result.game.duration_hours == 4
    assert "无烟" in result.game.rules
    assert result.game.status == GameStatus.NEED_CLARIFICATION
    assert any("上午还是下午" in question for question in result.follow_up_questions)


def test_parse_explicit_evening_time_is_open() -> None:
    parser = MahjongMessageParser()
    message = Message(
        text="今晚5点半 0.5 三缺一 无烟",
        sender_id="host",
        sender_name="张哥",
        channel_id="group",
    )

    result = parser.parse(message, now=datetime(2026, 6, 16, 12, 0, tzinfo=TZ))

    assert result.game is not None
    assert result.game.start_at == datetime(2026, 6, 16, 17, 30, tzinfo=TZ)
    assert result.game.status == GameStatus.OPEN


def test_default_region_hangzhou_fills_ambiguous_mahjong() -> None:
    parser = MahjongMessageParser(default_region="hangzhou")
    message = Message(
        text="下午两点 0.5 无烟 371",
        sender_id="host",
        sender_name="张哥",
        channel_id="group",
    )

    result = parser.parse(message, now=datetime(2026, 6, 16, 12, 0, tzinfo=TZ))

    assert result.game is not None
    assert result.game.game_type == "hangzhou_mahjong"
    assert result.game.ruleset == "hangzhou_mahjong"
    assert result.game.variant is None
    assert "杭麻" in result.game.rules
    assert "按当前地区默认玩法：杭麻" in result.game.notes


def test_default_region_sichuan_fills_ambiguous_mahjong() -> None:
    parser = MahjongMessageParser(default_region="sichuan")
    message = Message(
        text="下午两点 1块 371",
        sender_id="host",
        sender_name="张哥",
        channel_id="group",
    )

    result = parser.parse(message, now=datetime(2026, 6, 16, 12, 0, tzinfo=TZ))

    assert result.game is not None
    assert result.game.game_type == "sichuan_mahjong"
    assert result.game.ruleset == "sichuan_mahjong"
    assert "川麻" in result.game.rules
    assert "按当前地区默认玩法：川麻" in result.game.notes


def test_parse_sichuan_compact_pending_game() -> None:
    parser = MahjongMessageParser()
    message = Message(
        text="川麻216三等一",
        sender_id="host",
        sender_name="张哥",
        channel_id="group",
    )

    result = parser.parse(message, now=datetime(2026, 6, 16, 18, 0, tzinfo=TZ))

    assert result.game is not None
    assert result.game.game_type == "sichuan_mahjong"
    assert result.game.ruleset == "sichuan_mahjong"
    assert result.game.level == "2-16"
    assert result.game.base_score == 2
    assert result.game.cap_score == 16
    assert result.game.current_player_count == 3
    assert result.game.missing_count == 1
    assert "川麻" in result.game.rules
    assert result.game.status == GameStatus.NEED_CLARIFICATION
    assert result.follow_up_questions == ["希望几点开局？"]


def test_parse_hangzhou_caiqiao_dot_time_group_post() -> None:
    parser = MahjongMessageParser()
    message = Message(
        text="cq371 0.5 19.30 无烟",
        sender_id="host",
        sender_name="张哥",
        channel_id="group",
    )

    result = parser.parse(message, now=datetime(2026, 6, 16, 18, 0, tzinfo=TZ))

    assert result.game is not None
    assert result.game.game_type == "hangzhou_mahjong"
    assert result.game.ruleset == "hangzhou_mahjong"
    assert result.game.variant == "caiqiao"
    assert result.game.level == "0.5"
    assert result.game.base_score == 0.5
    assert result.game.cap_score is None
    assert result.game.current_player_count == 3
    assert result.game.missing_count == 1
    assert result.game.start_at == datetime(2026, 6, 16, 19, 30, tzinfo=TZ)
    assert {"杭麻", "无烟"}.issubset(set(result.game.rules))
    assert "财敲" in result.game.play_options
    assert result.game.status == GameStatus.OPEN


def test_parse_llm_normalized_smoke_ok_expression() -> None:
    parser = MahjongMessageParser()
    message = Message(
        text="今晚下班后打麻将，档位0.5或1，烟局可接受，六点",
        sender_id="host",
        sender_name="张哥",
        channel_id="group",
    )

    result = parser.parse(message, now=datetime(2026, 6, 27, 14, 30, tzinfo=TZ))

    assert result.game is not None
    assert result.game.start_at is not None
    assert result.game.start_at.hour == 18
    assert result.game.level == "0.5"
    assert "烟况都可" in result.game.rules


def test_parse_hangzhou_template_with_cap_and_smoke_field() -> None:
    parser = MahjongMessageParser()
    message = Message(
        text=(
            "帮忙摇下人可以不，✨杭麻✨ 财敲\n"
            "吃两摊，碰无限，十风，3财翻，4财翻，跳碰亮白，硬跟，封顶256\n"
            "人数:272\n"
            "开始：人齐开\n"
            "大小：1\n"
            "🚬：无"
        ),
        sender_id="host",
        sender_name="张哥",
        channel_id="group",
    )

    result = parser.parse(message, now=datetime(2026, 6, 16, 18, 0, tzinfo=TZ))

    assert result.game is not None
    assert result.game.game_type == "hangzhou_mahjong"
    assert result.game.variant == "caiqiao"
    assert result.game.level == "1"
    assert result.game.base_score == 1
    assert result.game.cap_score == 256
    assert result.game.current_player_count == 2
    assert result.game.missing_count == 2
    assert {"无烟", "人齐开"}.issubset(set(result.game.rules))
    assert {"财敲", "3财翻", "4财翻", "硬跟", "吃两摊", "碰无限", "十风", "跳碰亮白"}.issubset(
        set(result.game.play_options)
    )
    assert result.game.status == GameStatus.OPEN


def test_parse_people_ready_start_does_not_require_time() -> None:
    parser = MahjongMessageParser()
    message = Message(
        text="cq272 有烟0.5人齐开",
        sender_id="host",
        sender_name="张哥",
        channel_id="group",
    )

    result = parser.parse(message, now=datetime(2026, 6, 16, 18, 0, tzinfo=TZ))

    assert result.game is not None
    assert result.game.game_type == "hangzhou_mahjong"
    assert result.game.variant == "caiqiao"
    assert result.game.level == "0.5"
    assert result.game.current_player_count == 2
    assert result.game.missing_count == 2
    assert result.game.start_at is None
    assert {"杭麻", "可吸烟", "人齐开"}.issubset(set(result.game.rules))
    assert "财敲" in result.game.play_options
    assert result.game.status == GameStatus.OPEN


def test_parse_flexible_start_phrases_as_people_ready_start() -> None:
    parser = MahjongMessageParser()
    message = Message(
        text="0.5 173 烟都行 通宵 尽快开吧，时间可以再商量",
        sender_id="host",
        sender_name="张哥",
        channel_id="group",
    )

    result = parser.parse(message, now=datetime(2026, 6, 28, 22, 0, tzinfo=TZ))

    assert result.game is not None
    assert result.game.level == "0.5"
    assert result.game.current_player_count == 1
    assert result.game.missing_count == 3
    assert result.game.start_at is None
    assert {"人齐开", "烟况都可", "通宵"}.issubset(set(result.game.rules))
    assert not any("希望几点开局" in question for question in result.follow_up_questions)
    assert result.game.status == GameStatus.OPEN


def test_parse_current_party_size_phrase() -> None:
    parser = MahjongMessageParser()
    message = Message(
        text="下午两点 0.5 无烟杭麻，帮我组一桌\n我这边两个人",
        sender_id="host",
        sender_name="张哥",
        channel_id="group",
    )

    result = parser.parse(message, now=datetime(2026, 6, 16, 10, 0, tzinfo=TZ))

    assert result.game is not None
    assert result.game.current_player_count == 2
    assert result.game.missing_count == 2
    assert result.raw["missing_raw"] == "我这边两个人"


def test_parse_sichuan_huan_san_zhang_with_cap() -> None:
    parser = MahjongMessageParser()
    message = Message(
        text="今晚7点 川麻1-32换三张定缺 371",
        sender_id="host",
        sender_name="张哥",
        channel_id="group",
    )

    result = parser.parse(message, now=datetime(2026, 6, 16, 12, 0, tzinfo=TZ))

    assert result.game is not None
    assert result.game.game_type == "sichuan_mahjong"
    assert result.game.level == "1-32"
    assert result.game.base_score == 1
    assert result.game.cap_score == 32
    assert {"换三张", "定缺"}.issubset(set(result.game.play_options))
    assert result.game.status == GameStatus.OPEN


def test_parse_real_chat_smoke_and_plain_people_ready() -> None:
    parser = MahjongMessageParser()
    message = Message(
        text="0.5少烟 371 人齐",
        sender_id="host",
        sender_name="张哥",
        channel_id="group",
    )

    result = parser.parse(message, now=datetime(2026, 6, 16, 18, 0, tzinfo=TZ))

    assert result.game is not None
    assert result.game.level == "0.5"
    assert result.game.current_player_count == 3
    assert result.game.missing_count == 1
    assert {"少烟", "人齐开"}.issubset(set(result.game.rules))
    assert result.game.status == GameStatus.OPEN


def test_parse_wxid_prefixed_sichuan_huan_3_zhang() -> None:
    parser = MahjongMessageParser()
    message = Message(
        text="wxid_j329bt6i1wdu22:\n下午川麻有嘛  173  换3张  1/2都可",
        sender_id="host",
        sender_name="张哥",
        channel_id="group",
    )

    result = parser.parse(message, now=datetime(2026, 6, 16, 12, 0, tzinfo=TZ))

    assert result.game is not None
    assert result.game.game_type == "sichuan_mahjong"
    assert result.game.level == "1-2"
    assert result.game.current_player_count == 1
    assert result.game.missing_count == 3
    assert "换三张" in result.game.play_options


def test_parse_hongzhong_368_shayu() -> None:
    parser = MahjongMessageParser()
    message = Message(
        text="红中🀄368鲨鱼，371，无烟",
        sender_id="host",
        sender_name="张哥",
        channel_id="group",
    )

    result = parser.parse(message, now=datetime(2026, 6, 16, 18, 0, tzinfo=TZ))

    assert result.game is not None
    assert result.game.game_type == "hongzhong_mahjong"
    assert result.game.variant == "shayu"
    assert result.game.level == "368"
    assert result.game.current_player_count == 3
    assert result.game.missing_count == 1
    assert {"红中", "无烟"}.issubset(set(result.game.rules))
    assert "鲨鱼" in result.game.play_options
    assert result.follow_up_questions == ["希望几点开局？"]


def test_time_range_does_not_override_explicit_level() -> None:
    parser = MahjongMessageParser()
    message = Message(
        text="0.5元/128,18.30-19.00开始，371，无烟",
        sender_id="host",
        sender_name="张哥",
        channel_id="group",
    )

    result = parser.parse(message, now=datetime(2026, 6, 16, 12, 0, tzinfo=TZ))

    assert result.game is not None
    assert result.game.level == "0.5-128"
    assert result.game.start_at == datetime(2026, 6, 16, 18, 30, tzinfo=TZ)
    assert result.game.current_player_count == 3
    assert result.game.missing_count == 1


def test_parse_hongzhong_568_boom_variant_level() -> None:
    parser = MahjongMessageParser()
    message = Message(
        text="红中🀄568/3块爆炸 173 人齐开",
        sender_id="host",
        sender_name="张哥",
        channel_id="group",
    )

    result = parser.parse(message, now=datetime(2026, 6, 16, 18, 0, tzinfo=TZ))

    assert result.game is not None
    assert result.game.game_type == "hongzhong_mahjong"
    assert result.game.level == "568/3"
    assert result.game.current_player_count == 1
    assert result.game.missing_count == 3
    assert {"人齐开", "红中"}.issubset(set(result.game.rules))
    assert "爆炸" in result.game.play_options


def test_parse_extra_mahjong_families() -> None:
    parser = MahjongMessageParser()

    red = parser.parse(
        Message(text="今晚8点 红中麻将 0.5 371", sender_id="a", sender_name="A", channel_id="g"),
        now=datetime(2026, 6, 16, 12, 0, tzinfo=TZ),
    )
    zhuoji = parser.parse(
        Message(text="今晚8点 捉鸡麻将 1 371", sender_id="b", sender_name="B", channel_id="g"),
        now=datetime(2026, 6, 16, 12, 0, tzinfo=TZ),
    )
    yaoji = parser.parse(
        Message(text="今晚8点 幺鸡47 1-32 371", sender_id="c", sender_name="C", channel_id="g"),
        now=datetime(2026, 6, 16, 12, 0, tzinfo=TZ),
    )

    assert red.game is not None
    assert red.game.game_type == "hongzhong_mahjong"
    assert zhuoji.game is not None
    assert zhuoji.game.game_type == "zhuoji_mahjong"
    assert yaoji.game is not None
    assert yaoji.game.game_type == "sichuan_mahjong"
    assert yaoji.game.ruleset == "yaoji_mahjong"
    assert yaoji.game.variant == "yaoji_47"
    assert "幺鸡47" in yaoji.game.play_options


def test_customer_recommendation_and_invitation_locking() -> None:
    core = AgentCore()
    core.upsert_customer(
        CustomerProfile(
            id="amy",
            display_name="Amy",
            preferred_levels=["0.5"],
            tags=["无烟"],
            smoke_free_preference=True,
            usual_start_hours=[17],
        )
    )
    core.upsert_customer(
        CustomerProfile(
            id="ben",
            display_name="Ben",
            preferred_levels=["0.5"],
            tags=["无烟"],
            smoke_free_preference=True,
            usual_start_hours=[17],
        )
    )
    outcome = core.ingest_message(
        Message(
            text="今晚5点 0.5 三缺一 无烟",
            sender_id="host",
            sender_name="张哥",
            channel_id="group",
        ),
        now=datetime(2026, 6, 16, 12, 0, tzinfo=TZ),
    )

    assert outcome.extraction.game is not None
    assert len(outcome.candidates) == 2
    invitations = core.queue_invitations(outcome.extraction.game.id, outcome.candidates)
    assert len(invitations) == 2

    accepted = core.accept_invitation(invitations[0].id)

    assert accepted.accepted is True
    assert accepted.game.status == GameStatus.CONFIRMED
    assert accepted.invitation.status == InvitationStatus.ACCEPTED
    assert len(accepted.cancelled_invitations) == 1
    assert accepted.cancelled_invitations[0].status == InvitationStatus.SUPERSEDED


def test_room_conflict_keeps_game_in_clarification_and_skips_candidates() -> None:
    core = AgentCore()
    core.configure_room_capacity(1)
    core.add_room_hold(
        start_at=datetime(2026, 6, 16, 16, 0, tzinfo=TZ),
        end_at=datetime(2026, 6, 16, 18, 0, tzinfo=TZ),
        room_id="room-1",
        source="room_schedule",
    )
    core.upsert_customer(
        CustomerProfile(
            id="amy",
            display_name="Amy",
            preferred_levels=["0.5"],
            tags=["无烟"],
            smoke_free_preference=True,
            usual_start_hours=[17],
        )
    )

    outcome = core.ingest_message(
        Message(
            text="今晚5点 0.5 三缺一 无烟",
            sender_id="host",
            sender_name="张哥",
            channel_id="group",
        ),
        now=datetime(2026, 6, 16, 16, 0, tzinfo=TZ),
    )

    assert outcome.extraction.game is not None
    assert outcome.extraction.game.status == GameStatus.NEED_CLARIFICATION
    assert outcome.candidates == []
    assert outcome.draft_group_post is None
    assert outcome.room_availability is not None
    assert outcome.room_availability.available is False
    assert outcome.room_availability.suggested_start_at == datetime(2026, 6, 16, 18, 0, tzinfo=TZ)
    assert outcome.room_conflict_text is not None
    assert "满房" in outcome.room_conflict_text


def test_known_customer_party_size_fills_missing_count() -> None:
    core = AgentCore()
    core.upsert_customer(
        CustomerProfile(
            id="zhang",
            display_name="张哥",
            usual_party_size=1,
            usual_party_size_confidence=0.9,
        )
    )

    outcome = core.ingest_message(
        Message(
            text="今晚7点 0.5 有人打麻将吗",
            sender_id="zhang",
            sender_name="张哥",
            channel_id="group",
        ),
        now=datetime(2026, 6, 16, 12, 0, tzinfo=TZ),
    )

    assert outcome.extraction.game is not None
    game = outcome.extraction.game
    assert game.current_player_count == 1
    assert game.missing_count == 3
    assert game.status == GameStatus.OPEN
    assert not any("几缺几" in question for question in outcome.extraction.follow_up_questions)
    assert outcome.clarification_text is None
    assert "profile_party_size" in outcome.extraction.raw


def test_low_confidence_party_size_does_not_fill_missing_count() -> None:
    core = AgentCore()
    core.upsert_customer(
        CustomerProfile(
            id="zhang",
            display_name="张哥",
            usual_party_size=1,
            usual_party_size_confidence=0.4,
        )
    )

    outcome = core.ingest_message(
        Message(
            text="今晚7点 0.5 有人打麻将吗",
            sender_id="zhang",
            sender_name="张哥",
            channel_id="group",
        ),
        now=datetime(2026, 6, 16, 12, 0, tzinfo=TZ),
    )

    assert outcome.extraction.game is not None
    game = outcome.extraction.game
    assert game.current_player_count is None
    assert game.missing_count is None
    assert game.status == GameStatus.NEED_CLARIFICATION
    assert any("几缺几" in question for question in outcome.extraction.follow_up_questions)


def test_single_profile_play_preference_overrides_regional_default() -> None:
    core = AgentCore()
    core.upsert_customer(
        CustomerProfile(
            id="lin",
            display_name="林姐",
            play_preferences=[
                PlayPreference(
                    game_type="hongzhong_mahjong",
                    preferred_levels=["1"],
                    preferred_rulesets=["hongzhong_mahjong"],
                )
            ],
            tags=["红中"],
        )
    )

    outcome = core.ingest_message(
        Message(
            text="今晚7点 1块 371",
            sender_id="lin",
            sender_name="林姐",
            channel_id="private",
        ),
        now=datetime(2026, 6, 16, 12, 0, tzinfo=TZ),
    )

    assert outcome.extraction.game is not None
    game = outcome.extraction.game
    assert game.game_type == "hongzhong_mahjong"
    assert game.ruleset == "hongzhong_mahjong"
    assert game.variant is None
    assert "红中" in game.rules
    assert "杭麻" not in game.rules
    assert "玩法根据客户画像推断：红中麻将" in game.notes
    assert not any("按当前地区默认玩法" in note for note in game.notes)
    assert outcome.extraction.raw["profile_play_source"] == "customer_profile"
    assert outcome.extraction.raw["profile_previous_game_type"] == "hangzhou_mahjong"


def test_multiple_profile_play_preferences_do_not_override_regional_default() -> None:
    core = AgentCore()
    core.upsert_customer(
        CustomerProfile(
            id="feng",
            display_name="冯哥",
            play_preferences=[
                PlayPreference(
                    game_type="hangzhou_mahjong",
                    preferred_levels=["0.5"],
                    preferred_rulesets=["hangzhou_mahjong"],
                ),
                PlayPreference(
                    game_type="hongzhong_mahjong",
                    preferred_levels=["1"],
                    preferred_rulesets=["hongzhong_mahjong"],
                ),
            ],
            tags=["杭麻", "红中"],
        )
    )

    outcome = core.ingest_message(
        Message(
            text="今晚7点 1块 371",
            sender_id="feng",
            sender_name="冯哥",
            channel_id="private",
        ),
        now=datetime(2026, 6, 16, 12, 0, tzinfo=TZ),
    )

    assert outcome.extraction.game is not None
    game = outcome.extraction.game
    assert game.game_type == "hangzhou_mahjong"
    assert game.ruleset == "hangzhou_mahjong"
    assert "杭麻" in game.rules
    assert "profile_game_type" not in outcome.extraction.raw


def test_structured_play_preferences_rank_matching_customer() -> None:
    core = AgentCore()
    core.upsert_customer(
        CustomerProfile(
            id="zhang",
            display_name="张哥",
            play_preferences=[
                PlayPreference(
                    game_type="hangzhou_mahjong",
                    preferred_levels=["0.5"],
                    preferred_rulesets=["hangzhou_mahjong"],
                    preferred_variants=["caiqiao"],
                    preferred_play_options=["财敲"],
                ),
                PlayPreference(
                    game_type="sichuan_mahjong",
                    preferred_levels=["1-32"],
                    preferred_rulesets=["sichuan_mahjong"],
                    preferred_play_options=["换三张"],
                ),
            ],
            tags=["杭麻", "川麻"],
            usual_start_hours=[19],
        )
    )
    core.upsert_customer(
        CustomerProfile(
            id="generic",
            display_name="泛麻将客",
            preferred_levels=["0.5"],
            usual_start_hours=[19],
        )
    )

    hangzhou = core.ingest_message(
        Message(
            text="今晚7点 cq371 0.5",
            sender_id="host",
            sender_name="老板",
            channel_id="hangzhou_group",
        ),
        now=datetime(2026, 6, 16, 12, 0, tzinfo=TZ),
    )

    assert hangzhou.extraction.game is not None
    assert hangzhou.candidates[0].customer_id == "zhang"
    assert "财敲" in " ".join(hangzhou.candidates[0].reasons)

    sichuan = core.ingest_message(
        Message(
            text="今晚7点 川麻1-32换三张 371",
            sender_id="host",
            sender_name="老板",
            channel_id="sichuan_group",
        ),
        now=datetime(2026, 6, 16, 12, 0, tzinfo=TZ),
    )

    assert sichuan.extraction.game is not None
    assert sichuan.candidates[0].customer_id == "zhang"
    assert "换三张" in " ".join(sichuan.candidates[0].reasons)


def test_customer_active_invitation_locks_other_games_until_declined() -> None:
    core = AgentCore()
    for customer_id, name in [("amy", "Amy"), ("chen", "陈姐")]:
        core.upsert_customer(
            CustomerProfile(
                id=customer_id,
                display_name=name,
                preferred_levels=["0.5"],
                tags=["无烟"],
                smoke_free_preference=True,
                usual_start_hours=[17],
            )
        )

    first = core.ingest_message(
        Message(
            text="今晚5点 0.5 三缺一 无烟",
            sender_id="host1",
            sender_name="张哥",
            channel_id="group1",
        ),
        now=datetime(2026, 6, 16, 12, 0, tzinfo=TZ),
    )
    assert first.extraction.game is not None
    first_invites = core.queue_invitations(first.extraction.game.id, first.candidates)
    assert {item.customer_id for item in first_invites} == {"amy", "chen"}

    second = core.ingest_message(
        Message(
            text="今晚5点半 0.5 三缺一 无烟",
            sender_id="host2",
            sender_name="李哥",
            channel_id="group2",
        ),
        now=datetime(2026, 6, 16, 12, 0, tzinfo=TZ),
    )

    assert second.extraction.game is not None
    assert second.candidates == []
    assert core.queue_invitations(second.extraction.game.id, second.candidates) == []

    core.decline_invitation(first_invites[0].id)
    third = core.ingest_message(
        Message(
            text="今晚6点 0.5 三缺一 无烟",
            sender_id="host3",
            sender_name="王姐",
            channel_id="group3",
        ),
        now=datetime(2026, 6, 16, 12, 0, tzinfo=TZ),
    )

    assert third.extraction.game is not None
    assert [candidate.customer_id for candidate in third.candidates] == [first_invites[0].customer_id]


def test_accepted_customer_stays_locked_to_confirmed_game() -> None:
    core = AgentCore()
    for customer_id, name in [("amy", "Amy"), ("chen", "陈姐")]:
        core.upsert_customer(
            CustomerProfile(
                id=customer_id,
                display_name=name,
                preferred_levels=["0.5"],
                tags=["无烟"],
                smoke_free_preference=True,
                usual_start_hours=[17],
            )
        )

    first = core.ingest_message(
        Message(
            text="今晚5点 0.5 三缺一 无烟",
            sender_id="host1",
            sender_name="张哥",
            channel_id="group1",
        ),
        now=datetime(2026, 6, 16, 12, 0, tzinfo=TZ),
    )
    assert first.extraction.game is not None
    first_invites = core.queue_invitations(first.extraction.game.id, first.candidates)
    accepted = core.accept_invitation(first_invites[0].id)

    assert accepted.accepted is True
    assert accepted.game.status == GameStatus.CONFIRMED

    second = core.ingest_message(
        Message(
            text="今晚6点 0.5 三缺一 无烟",
            sender_id="host2",
            sender_name="李哥",
            channel_id="group2",
        ),
        now=datetime(2026, 6, 16, 12, 0, tzinfo=TZ),
    )

    assert second.extraction.game is not None
    candidate_ids = [candidate.customer_id for candidate in second.candidates]
    assert accepted.invitation.customer_id not in candidate_ids
    assert set(candidate_ids) == {"chen"}


def test_confirmed_game_releases_customer_locks_after_estimated_end() -> None:
    core = AgentCore()
    for customer_id, name in [("amy", "Amy"), ("chen", "陈姐")]:
        core.upsert_customer(
            CustomerProfile(
                id=customer_id,
                display_name=name,
                preferred_levels=["0.5"],
                tags=["无烟"],
                smoke_free_preference=True,
                usual_start_hours=[19],
                max_games_per_day=2,
            )
        )

    morning = core.ingest_message(
        Message(
            text="早上9点 0.5 三缺一 无烟",
            sender_id="host1",
            sender_name="张哥",
            channel_id="morning_group",
        ),
        now=datetime(2026, 6, 16, 7, 0, tzinfo=TZ),
    )
    assert morning.extraction.game is not None
    morning_invites = core.queue_invitations(morning.extraction.game.id, morning.candidates)
    accepted = core.accept_invitation(morning_invites[0].id)
    assert accepted.game.status == GameStatus.CONFIRMED

    afternoon_now = datetime(2026, 6, 16, 15, 0, tzinfo=TZ)
    changed = core.advance_game_lifecycle(afternoon_now)

    assert accepted.game in changed
    assert accepted.game.status == GameStatus.COMPLETED

    evening = core.ingest_message(
        Message(
            text="今晚7点 0.5 三缺一 无烟",
            sender_id="host1",
            sender_name="张哥",
            channel_id="evening_group",
        ),
        now=afternoon_now,
    )

    assert evening.extraction.game is not None
    assert {candidate.customer_id for candidate in evening.candidates} == {"amy", "chen"}


def test_default_daily_fatigue_skips_customer_after_one_completed_game() -> None:
    core = AgentCore()
    for customer_id, name in [("amy", "Amy"), ("chen", "陈姐")]:
        core.upsert_customer(
            CustomerProfile(
                id=customer_id,
                display_name=name,
                preferred_levels=["0.5"],
                tags=["无烟"],
                smoke_free_preference=True,
                usual_start_hours=[19],
            )
        )

    morning = core.ingest_message(
        Message(
            text="早上9点 0.5 三缺一 无烟",
            sender_id="host1",
            sender_name="张哥",
            channel_id="morning_group",
        ),
        now=datetime(2026, 6, 16, 7, 0, tzinfo=TZ),
    )
    assert morning.extraction.game is not None
    morning_invites = core.queue_invitations(morning.extraction.game.id, morning.candidates)
    amy_invite = next(invitation for invitation in morning_invites if invitation.customer_id == "amy")
    core.accept_invitation(amy_invite.id)
    core.advance_game_lifecycle(datetime(2026, 6, 16, 15, 0, tzinfo=TZ))

    evening = core.ingest_message(
        Message(
            text="今晚7点 0.5 三缺一 无烟",
            sender_id="host2",
            sender_name="李哥",
            channel_id="evening_group",
        ),
        now=datetime(2026, 6, 16, 15, 0, tzinfo=TZ),
    )

    assert evening.extraction.game is not None
    assert [candidate.customer_id for candidate in evening.candidates] == ["chen"]
    fatigue = core.customer_fatigue(
        "amy",
        proposed_start_at=evening.extraction.game.start_at,
        now=datetime(2026, 6, 16, 15, 0, tzinfo=TZ),
    )
    assert fatigue.hard_block is True
    assert "达到画像上限" in " ".join(fatigue.warnings)


def test_frequent_player_can_be_recommended_again_with_fatigue_penalty() -> None:
    core = AgentCore()
    core.upsert_customer(
        CustomerProfile(
            id="frequent",
            display_name="连场客",
            preferred_levels=["0.5"],
            tags=["无烟"],
            smoke_free_preference=True,
            usual_start_hours=[9, 19],
            max_games_per_day=3,
            min_hours_between_games=2,
            invite_cooldown_hours=1,
            fatigue_sensitivity=0.35,
        )
    )
    core.upsert_customer(
        CustomerProfile(
            id="fresh",
            display_name="新鲜客",
            preferred_levels=["0.5"],
            tags=["无烟"],
            smoke_free_preference=True,
            usual_start_hours=[19],
        )
    )

    morning = core.ingest_message(
        Message(
            text="早上9点 0.5 三缺一 无烟",
            sender_id="host1",
            sender_name="张哥",
            channel_id="morning_group",
        ),
        now=datetime(2026, 6, 16, 7, 0, tzinfo=TZ),
    )
    assert morning.extraction.game is not None
    morning_invites = core.queue_invitations(morning.extraction.game.id, morning.candidates)
    frequent_invite = next(invitation for invitation in morning_invites if invitation.customer_id == "frequent")
    core.accept_invitation(frequent_invite.id)
    core.advance_game_lifecycle(datetime(2026, 6, 16, 15, 0, tzinfo=TZ))

    evening = core.ingest_message(
        Message(
            text="今晚7点 0.5 三缺一 无烟",
            sender_id="host2",
            sender_name="李哥",
            channel_id="evening_group",
        ),
        now=datetime(2026, 6, 16, 15, 0, tzinfo=TZ),
    )

    assert evening.extraction.game is not None
    candidate_ids = [candidate.customer_id for candidate in evening.candidates]
    assert "frequent" in candidate_ids
    assert candidate_ids[0] == "fresh"
    frequent = next(candidate for candidate in evening.candidates if candidate.customer_id == "frequent")
    assert "今日已打 1 场" in " ".join(frequent.warnings)


def test_unconfirmed_game_expires_after_start_grace_and_releases_invites() -> None:
    core = AgentCore()
    core.upsert_customer(
        CustomerProfile(
            id="amy",
            display_name="Amy",
            preferred_levels=["0.5"],
            tags=["无烟"],
            smoke_free_preference=True,
            usual_start_hours=[10],
        )
    )

    opening = core.ingest_message(
        Message(
            text="早上9点 0.5 三缺一 无烟",
            sender_id="host1",
            sender_name="张哥",
            channel_id="group1",
        ),
        now=datetime(2026, 6, 16, 7, 0, tzinfo=TZ),
    )
    assert opening.extraction.game is not None
    invites = core.queue_invitations(opening.extraction.game.id, opening.candidates)
    assert invites[0].status == InvitationStatus.QUEUED

    core.advance_game_lifecycle(datetime(2026, 6, 16, 9, 31, tzinfo=TZ))

    assert opening.extraction.game.status == GameStatus.EXPIRED
    assert invites[0].status == InvitationStatus.SUPERSEDED

    later = core.ingest_message(
        Message(
            text="上午10点 0.5 三缺一 无烟",
            sender_id="host2",
            sender_name="李哥",
            channel_id="group2",
        ),
        now=datetime(2026, 6, 16, 9, 40, tzinfo=TZ),
    )

    assert later.extraction.game is not None
    assert [candidate.customer_id for candidate in later.candidates] == ["amy"]


def test_merge_suggestion_for_371_and_173() -> None:
    core = AgentCore()
    first = core.ingest_message(
        Message(
            text="今晚5点半 0.5 371",
            sender_id="a",
            sender_name="A",
            channel_id="g1",
        ),
        now=datetime(2026, 6, 16, 12, 0, tzinfo=TZ),
    )
    second = core.ingest_message(
        Message(
            text="今晚5点 0.5 173",
            sender_id="b",
            sender_name="B",
            channel_id="g2",
        ),
        now=datetime(2026, 6, 16, 12, 0, tzinfo=TZ),
    )

    assert first.extraction.game is not None
    assert second.extraction.game is not None
    suggestions = core.suggest_merges()

    assert suggestions
    assert suggestions[0].score >= 80
    assert set(suggestions[0].game_ids) == {first.extraction.game.id, second.extraction.game.id}
