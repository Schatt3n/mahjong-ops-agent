from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mahjong_agent import AgentResponder, ChannelType, CustomerProfile, Message, ReplyAction


TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 6, 16, 12, 0, tzinfo=TZ)


@dataclass(slots=True)
class Scenario:
    name: str
    text: str
    sender_id: str
    expected_action: ReplyAction
    contains: str | None = None
    should_reply: bool | None = None
    metadata: dict = field(default_factory=dict)


def seed() -> AgentResponder:
    responder = AgentResponder(invite_limit=3)
    customers = [
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
    ]
    for customer in customers:
        responder.core.upsert_customer(customer)
    return responder


def make_message(text: str, sender_id: str, name: str | None = None) -> Message:
    return Message(
        text=text,
        sender_id=sender_id,
        sender_name=name or sender_id,
        channel_id="group_main",
        channel_type=ChannelType.WECHAT_GROUP,
    )


def main() -> int:
    scenarios = [
        Scenario(
            name="清晰组局",
            text="今晚5点 0.5 三缺一 无烟 打四小时",
            sender_id="host",
            expected_action=ReplyAction.QUEUE_INVITES,
            contains="建议先私聊",
        ),
        Scenario(
            name="模糊时间追问",
            text="0.5 5点开 371 无烟",
            sender_id="host2",
            expected_action=ReplyAction.ASK_CLARIFICATION,
            contains="上午还是下午",
        ),
        Scenario(
            name="川麻明确缺时间入待组局",
            text="川麻216三等一",
            sender_id="host_sichuan",
            expected_action=ReplyAction.CREATE_PENDING_GAME,
            contains="待组局队列",
        ),
        Scenario(
            name="杭麻财敲清晰组局",
            text="cq371 0.5 19.30 无烟",
            sender_id="host_caiqiao",
            expected_action=ReplyAction.QUEUE_INVITES,
            contains="杭麻，财敲",
        ),
        Scenario(
            name="无关群消息静默",
            text="今天路上有点堵",
            sender_id="passerby",
            expected_action=ReplyAction.IGNORE,
            should_reply=False,
        ),
        Scenario(
            name="弱组局咨询追问",
            text="今天下班有人打麻将吗",
            sender_id="passerby",
            expected_action=ReplyAction.ASK_CLARIFICATION,
            contains="帮你看看能不能拼一桌",
        ),
        Scenario(
            name="语音意向追问",
            text="[语音]",
            sender_id="voice_user",
            expected_action=ReplyAction.ASK_CLARIFICATION,
            contains="帮你看看能不能拼一桌",
            metadata={"message_type": "audio", "audio_transcript": "下班想搓一把，有局吗"},
        ),
        Scenario(
            name="图片 OCR 清晰组局",
            text="[图片]",
            sender_id="image_user",
            expected_action=ReplyAction.QUEUE_INVITES,
            contains="建议先私聊",
            metadata={"message_type": "image", "image_ocr_text": "群截图：今晚7点 0.5 三缺一 无烟"},
        ),
        Scenario(
            name="表情包意向追问",
            text="[表情包]",
            sender_id="sticker_user",
            expected_action=ReplyAction.ASK_CLARIFICATION,
            contains="帮你看看能不能拼一桌",
            metadata={"message_type": "sticker", "sticker_description": "麻将表情包：🀄 约吗"},
        ),
        Scenario(
            name="敏感资金内容转人工",
            text="这桌输赢结算你帮我代收一下",
            sender_id="host3",
            expected_action=ReplyAction.HUMAN_REVIEW,
            contains="转人工",
        ),
    ]

    failed = 0
    for scenario in scenarios:
        responder = seed()
        message = make_message(scenario.text, scenario.sender_id)
        message.metadata.update(scenario.metadata)
        decision = responder.respond(message, now=NOW)
        errors = []
        if decision.action != scenario.expected_action:
            errors.append(f"action={decision.action.value}, expected={scenario.expected_action.value}")
        if scenario.contains and scenario.contains not in decision.reply_text:
            errors.append(f"reply does not contain {scenario.contains!r}: {decision.reply_text!r}")
        if scenario.should_reply is not None and decision.should_reply != scenario.should_reply:
            errors.append(f"should_reply={decision.should_reply}, expected={scenario.should_reply}")

        if errors:
            failed += 1
            print(f"FAIL {scenario.name}: " + "; ".join(errors))
        else:
            print(f"PASS {scenario.name}: {decision.action.value} -> {decision.reply_text or '<silent>'}")

    responder = seed()
    responder.respond(make_message("今晚7点 0.5 三缺一 无烟", "host_accept"), now=NOW)
    first_invitation = next(iter(responder.core.store.invitations.values()))
    accept_decision = responder.respond(make_message("我来", first_invitation.customer_id), now=NOW)
    if accept_decision.action == ReplyAction.ACCEPT_SEAT and "人数已齐" in accept_decision.reply_text:
        print(f"PASS 被邀请用户接受: {accept_decision.reply_text}")
    else:
        failed += 1
        print(f"FAIL 被邀请用户接受: {accept_decision.to_dict()}")

    print(f"\n{len(scenarios) + 1 - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
