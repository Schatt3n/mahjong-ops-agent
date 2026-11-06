"""约束话题公开卡片与来源记录；话题只是检索投影，不代表建局或参加授权。"""

from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timedelta
import hashlib
import json
import re

from ..knowledge import default_terminology_repository
from ..models import DEFAULT_TZ

PUBLIC_TOPIC_FIELDS = frozenset({'game_type', 'stakes', 'date', 'time_window', 'smoking', 'ruleset'})
TOPIC_INTENTS = frozenset({'query', 'delegate', 'create', 'claim', 'update', 'cancel', 'chitchat', 'uncertain'})


def validate_public_topic_summary(value: dict | None) -> dict:
    """只允许规范领域值进入跨客户候选卡片，拒绝姓名、备注和任意自由摘要。"""
    if value is None:
        return {}
    if not isinstance(value, dict) or set(value) - PUBLIC_TOPIC_FIELDS:
        raise ValueError('invalid public topic summary fields')
    terms = default_terminology_repository().terms
    games = {t.canonical.get('requested_game') for t in terms if t.category in {'game_type', 'game_variant'}} - {None}
    rules = {t.canonical.get('ruleset') for t in terms if t.category == 'game_variant'} - {None}
    result = {}
    for key, item in value.items():
        if item is None:
            continue
        if key == 'game_type' and (not isinstance(item, str) or item not in games):
            raise ValueError('invalid topic game_type')
        if key == 'ruleset' and (not isinstance(item, str) or item not in rules):
            raise ValueError('invalid topic ruleset')
        if key == 'smoking' and (not isinstance(item, str) or item not in {'no_smoking', 'smoking', 'any'}):
            raise ValueError('invalid topic smoking')
        if key == 'stakes':
            if not isinstance(item, list) or len(item) > 10 or any(not isinstance(x, str) or re.fullmatch(r'\d{1,6}(?:\.\d{1,3})?', x) is None for x in item):
                raise ValueError('invalid topic stakes')
        if key == 'date':
            if not isinstance(item, str) or re.fullmatch(r'\d{4}-\d{2}-\d{2}', item) is None:
                raise ValueError('invalid topic date')
            date.fromisoformat(item)
        if key == 'time_window':
            if not isinstance(item, dict) or set(item) - {'start', 'end'}:
                raise ValueError('invalid topic time_window')
            if any(x is not None and (not isinstance(x, str) or re.fullmatch(r'(?:[01]\d|2[0-3]):[0-5]\d', x) is None) for x in item.values()):
                raise ValueError('invalid topic time_window')
        result[key] = deepcopy(item)
    return result


def topic_source_key(channel: str, room_id: str, customer_id: str, message_id: str) -> str:
    """用完整来源作用域生成稳定标识，跨群或跨客户的同编号不会互相去重。"""
    values = (channel, room_id, customer_id, message_id)
    if any(not isinstance(item, str) or not item.strip() or len(item) > 1000 for item in values):
        raise ValueError('invalid topic message source')
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode()).hexdigest()


def prepare_topic_record(*, existing_topic, existing_message, channel, room_id, customer_id,
                         message_id, text, sent_at, intent, topic_id=None, public_summary=None,
                         classification=None, expected_version=None, trace_id='', idle_seconds=1800,
                         create_topic=True) -> dict:
    """在 Store 锁或事务内校验幂等和版本，生成纯数据变更；不能在这里执行业务工具。"""
    source_key = topic_source_key(channel, room_id, customer_id, message_id)
    if not isinstance(text, str) or not text.strip() or len(text) > 16000:
        raise ValueError('invalid topic message text')
    if not isinstance(sent_at, datetime) or sent_at.tzinfo is None:
        raise ValueError('topic sent_at requires timezone')
    sent_at = sent_at.astimezone(DEFAULT_TZ)
    source = {'channel': channel, 'room_id': room_id, 'customer_id': customer_id,
              'message_id': message_id, 'text': text, 'sent_at': sent_at.isoformat()}
    if existing_message is not None:
        if any(existing_message[key] != value for key, value in source.items()):
            raise ValueError('topic message id conflict')
        return {'topic': deepcopy(existing_topic), 'message': deepcopy(existing_message), 'deduplicated': True}
    if not isinstance(intent, str) or intent not in TOPIC_INTENTS:
        raise ValueError('invalid topic intent')
    if not isinstance(classification, (dict, type(None))):
        raise ValueError('classification must be an object')
    json.dumps(classification or {}, ensure_ascii=False, allow_nan=False)
    if isinstance(idle_seconds, bool) or not isinstance(idle_seconds, int) or not 1 <= idle_seconds <= 604800:
        raise ValueError('invalid topic idle_seconds')
    summary = validate_public_topic_summary(public_summary)
    topic = None
    if topic_id is not None:
        if existing_topic is None:
            raise ValueError('topic does not exist')
        if existing_topic['room_id'] != room_id:
            raise ValueError('topic room mismatch')
        if type(expected_version) is not int or expected_version != existing_topic['version']:
            raise ValueError('topic version conflict')
        topic = deepcopy(existing_topic)
        topic['version'] += 1
        # 共同话题卡片只填空缺，不把另一人的新时间/烟况覆盖成全体的承诺。
        for key, value in summary.items():
            if key not in topic['public_summary']:
                topic['public_summary'][key] = deepcopy(value)
        latest = max(datetime.fromisoformat(topic['last_activity_at']), sent_at)
        topic.update(last_activity_at=latest.isoformat(), active_until=(latest + timedelta(seconds=idle_seconds)).isoformat())
    elif create_topic:
        topic = {'topic_id': 'topic_' + source_key[:24], 'room_id': room_id, 'status': 'active',
                 'version': 1, 'public_summary': summary, 'created_at': sent_at.isoformat(),
                 'last_activity_at': sent_at.isoformat(), 'active_until': (sent_at+timedelta(seconds=idle_seconds)).isoformat()}
    message = {**source, 'source_key': source_key, 'topic_id': topic['topic_id'] if topic else None,
               'topic_version': topic['version'] if topic else None, 'intent': intent,
               'public_summary': summary, 'classification': deepcopy(classification or {}), 'trace_id': trace_id}
    return {'topic': topic, 'message': message, 'deduplicated': False}


def topic_is_hot(topic: dict, at: datetime) -> bool:
    """判断是否进入当前活跃候选集合，冷却不删除来源，也不改变任务或牌局状态。"""
    return topic['status'] == 'active' and datetime.fromisoformat(topic['active_until']) > at


def prepare_topic_binding(store, *, topic, previous, topic_id, customer_id, conversation_id,
                          task_context_id=None, game_ids=(), source_message_id=None,
                          allow_conversation_change=False, at=None) -> dict:
    """校验本人目标关联；换目标保留另一条绑定，跨会话变更必须由调用方显式声明。"""
    if topic is None:
        raise ValueError('topic does not exist')
    if not customer_id or not conversation_id:
        raise ValueError('topic binding requires customer and conversation')
    if previous and previous['conversation_id'] != conversation_id and not allow_conversation_change:
        raise ValueError('topic binding conversation conflict')
    if task_context_id:
        context = store.get_task_context(task_context_id)
        if context is None or context.customer_id != customer_id:
            raise ValueError('topic task owner mismatch')
        if context.conversation_id != conversation_id:
            raise ValueError('topic task conversation mismatch')
    if not isinstance(game_ids, (list, tuple)) or any(not isinstance(value, str) or value not in store.games for value in game_ids):
        raise ValueError('topic game reference does not exist')
    from ..models import now
    stamp = at or now()
    if not isinstance(stamp, datetime) or stamp.tzinfo is None:
        raise ValueError('topic binding timestamp requires timezone')
    stamp = stamp.astimezone(DEFAULT_TZ)
    binding_id = hashlib.sha256(json.dumps([topic_id, customer_id, conversation_id, task_context_id, source_message_id], ensure_ascii=False).encode()).hexdigest()
    return {'binding_id': binding_id, 'topic_id': topic_id, 'customer_id': customer_id,
            'conversation_id': conversation_id, 'task_context_id': task_context_id,
            'game_ids': list(dict.fromkeys(game_ids)), 'source_message_id': source_message_id,
            'updated_at': stamp.isoformat()}
