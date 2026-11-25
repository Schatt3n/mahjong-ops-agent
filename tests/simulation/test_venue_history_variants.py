"""检验真人前缀不变且每组反事实只改变约定的一项。"""

from dataclasses import replace
from datetime import time

import pytest

from .venue_history_seed import load_historical_group_seed
from .venue_history_variants import load_group_history_variant


def test_smoking_pair_only_changes_recruiters_public_confirmation():
    """在同一切点由同一人引用同一招募帖，只切换公开的烟况确认。"""
    pending = load_group_history_variant("smoke-pending")
    confirmed = load_group_history_variant("smoke-confirmed")
    assert pending.seed == confirmed.seed == load_historical_group_seed(cutoff="m04")
    assert pending.seed.held_out_count == 22
    assert pending.message.actor_alias == pending.seed.prefix[0].actor_alias
    assert pending.message.quoted_message_alias == "m01"
    assert pending.message.message_alias == "simulated-variant"
    assert pending.message.offset_seconds == pending.seed.prefix[-1].offset_seconds + 1
    assert pending.message.text != confirmed.message.text
    assert replace(pending.message, text=confirmed.message.text) == confirmed.message
    assert pending.focus_actor_alias is confirmed.focus_actor_alias is None


def test_arrival_pair_only_changes_the_same_guests_private_earliest_time():
    """到场时间只给同一位顾客，群里不会多收到一条补写消息。"""
    early = load_group_history_variant("arrival-2100")
    late = load_group_history_variant("arrival-2140")
    assert early.seed == late.seed == load_historical_group_seed(cutoff="m18")
    assert early.message is late.message is None
    assert early.focus_actor_alias == late.focus_actor_alias == "customer_5"
    assert early.focus_actor_alias in early.seed.actor_aliases
    assert early.earliest_arrival == time(21, 0)
    assert late.earliest_arrival == time(21, 40)


def test_question_pair_keeps_paraphrase_sender_and_time_identical():
    """两次群里追问的正文相同，引用有无是唯一改变的公开事实。"""
    plain = load_group_history_variant("question-unquoted")
    linked = load_group_history_variant("question-quoted")
    assert plain.seed == linked.seed == load_historical_group_seed(cutoff="m25")
    assert plain.seed.held_out_count == 1
    assert plain.message.text == linked.message.text == "这桌几点开？"
    assert plain.message.actor_alias == "simulated_inquirer"
    assert plain.message.actor_alias not in plain.seed.actor_aliases
    assert plain.message.quoted_message_alias is None
    assert linked.message.quoted_message_alias == "m25"
    assert replace(plain.message, content_type="quote_reply", quoted_message_alias="m25") == linked.message


def test_unknown_variant_is_not_silently_substituted():
    """错误的对照名称提前失败，避免真实跑测使用另一条件。"""
    with pytest.raises(ValueError, match="未知的历史群聊对照"):
        load_group_history_variant("other")
