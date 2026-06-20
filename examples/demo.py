from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from mahjong_agent import AgentCore, ChannelType, CustomerProfile, Message


def main() -> None:
    tz = ZoneInfo("Asia/Shanghai")
    core = AgentCore()
    for customer in [
        CustomerProfile(
            id="u_amy",
            display_name="Amy",
            preferred_levels=["0.5"],
            tags=["无烟"],
            smoke_free_preference=True,
            usual_start_hours=[17, 18, 19],
            usual_weekdays=[1, 2, 3, 4, 5],
        ),
        CustomerProfile(
            id="u_ben",
            display_name="Ben",
            preferred_levels=["1"],
            tags=["可吸烟"],
            smoke_free_preference=False,
            usual_start_hours=[20, 21],
        ),
        CustomerProfile(
            id="u_chen",
            display_name="陈姐",
            preferred_levels=["0.5", "1"],
            tags=["无烟", "熟人局"],
            smoke_free_preference=True,
            usual_start_hours=[17, 18],
        ),
    ]:
        core.upsert_customer(customer)

    message = Message(
        text="今晚5点开 0.5 371 无烟 打四小时，帮忙找一位",
        sender_id="u_host",
        sender_name="张哥",
        channel_id="g_mahjong",
        channel_type=ChannelType.WECHAT_GROUP,
    )
    outcome = core.ingest_message(message, now=datetime(2026, 6, 16, 12, 0, tzinfo=tz))

    print("识别置信度:", outcome.extraction.confidence)
    print("追问:", outcome.extraction.follow_up_questions)
    if outcome.extraction.game:
        game = outcome.extraction.game
        print("局:", game)
        print("群发草稿:", outcome.draft_group_post)
        print("候选人:")
        print(core.composer.candidate_summary(outcome.candidates))
        invitations = core.queue_invitations(game.id, outcome.candidates, limit=2)
        print("邀约草稿:")
        for invitation in invitations:
            print("-", invitation.message_text)


if __name__ == "__main__":
    main()
