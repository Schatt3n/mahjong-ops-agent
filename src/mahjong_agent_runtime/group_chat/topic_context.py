"""核验群输入的持久来源，向本人任务投影必要条件；话题不授予业务写权限。"""

from __future__ import annotations

from datetime import datetime

from .topic_models import PUBLIC_TOPIC_FIELDS, validate_public_topic_summary


def verified_topic_message(store, message):
    """忽略无持久凭据的外部 metadata，拒绝跨客户、跨群或改写原文的任务恢复。"""
    source = (message.metadata or {}).get('group_topic_source')
    if not isinstance(source, dict):
        return None
    channel, room = source.get('channel'), source.get('room_id')
    external = source.get('external_user_id')
    if not all(isinstance(x, str) and x for x in (channel, room, external)) or not message.message_id:
        return None
    identity = store.get_channel_identity(channel, external)
    customer = identity.customer_id if identity else f'{channel}:{external}'
    if customer != message.sender_id:
        return None
    record = store.get_group_topic_message(channel, room, customer, message.message_id)
    if not record or record['text'] != message.text or datetime.fromisoformat(record['sent_at']) != message.sent_at:
        return None
    group_conversation = f'group:{room}:customer:{customer}'
    if message.conversation_id == group_conversation:
        return record
    if not identity or not identity.can_private_message or message.conversation_id != identity.private_conversation_id:
        return None
    # 身份允许私聊不等于本条询局应转私聊；验证持久分类及既有个人绑定，防止 metadata 擅改通道。
    classification = record['classification']
    if classification.get('intent') in {'delegate', 'create', 'claim'}:
        return record
    if record.get('topic_id') and classification.get('goal_relation') == 'continue':
        bindings = store.list_group_topic_customer_bindings(record['topic_id'], customer)
        if any(b['conversation_id'] == message.conversation_id and b.get('task_context_id')
               and b['task_context_id'] == classification.get('task_context_id') for b in bindings):
            return record
    return None


def selected_topic_binding(store, record, conversation_id):
    """只接受本人已绑定的目标；重复接收同一消息时恢复第一次选择的任务。"""
    if not record or not record.get('topic_id'):
        return None
    bindings = store.list_group_topic_customer_bindings(record['topic_id'], record['customer_id'])
    exact = [b for b in bindings if b['conversation_id'] == conversation_id
             and b.get('source_message_id') == record['message_id'] and b.get('task_context_id')]
    if exact:
        return exact[0]
    selected = record['classification'].get('task_context_id')
    if record['classification'].get('goal_relation') != 'continue' or not selected:
        return None
    return next((b for b in bindings if b.get('task_context_id') == selected
                 and b['conversation_id'] == conversation_id), None)


def _own_constraints(row):
    """只投影强类型个人条件，原文、自由备注与身份字段不会借分类记录混入上下文。"""
    value = row['classification'].get('constraints', {})
    if not isinstance(value, dict):
        return {}
    result = {}
    for key in PUBLIC_TOPIC_FIELDS:
        if key in value:
            try:
                result.update(validate_public_topic_summary({key: value[key]}))
            except (ValueError, TypeError):
                continue
    party_size = value.get('party_size')
    if type(party_size) is int and 1 <= party_size <= 4:
        result['party_size'] = party_size
    for key in ('planned_start_at', 'latest_arrival_at', 'latest_wait_at', 'end_at'):
        stamp = value.get(key)
        if isinstance(stamp, str):
            try:
                parsed = datetime.fromisoformat(stamp)
                if parsed.tzinfo is not None:
                    result[key] = stamp
            except ValueError:
                continue
    if value.get('source_message_id') == row['message_id']:
        result['source_message_id'] = row['message_id']
    return result


def _own_goal_sources(store, record, selected, at):
    """按持久来源绑定收集本人目标，不按同 topic 或时间邻近推定属于同一需求。

    群转私聊新建上下文时，委托消息仍精确引用原群任务；只沿该明确关系继承结构条件。
    递归范围有界，旧于窗口的事实仍由原任务 checkpoint 保存，不拷贝群原文到私聊历史。
    """
    topic_id, customer_id = record['topic_id'], record['customer_id']
    bindings = store.list_group_topic_customer_bindings(topic_id, customer_id)
    owned = {}
    for binding in bindings:
        task_id = binding.get('task_context_id')
        context = store.get_task_context(task_id) if task_id else None
        if context and context.customer_id == customer_id and context.conversation_id == binding['conversation_id']:
            owned.setdefault(task_id, set()).add(binding.get('source_message_id'))
    if selected not in owned:
        return [record], []
    candidates = store.list_group_topic_customer_messages(topic_id, customer_id, limit=30)
    candidates = [row for row in candidates if datetime.fromisoformat(row['sent_at']) <= at]
    target_ids, visited = {selected}, set()
    rows = {record['source_key']: record}
    while target_ids - visited:
        task_id = next(iter(target_ids - visited))
        visited.add(task_id)
        for row in candidates:
            if row['message_id'] not in owned[task_id]:
                continue
            rows[row['source_key']] = row
            prior = row['classification'].get('task_context_id')
            if row['classification'].get('goal_relation') == 'continue' and prior in owned:
                target_ids.add(prior)
    games = list(dict.fromkeys(g for b in bindings if b.get('task_context_id') == selected for g in b['game_ids']))
    ordered = sorted(rows.values(), key=lambda row: (datetime.fromisoformat(row['sent_at']), row['message_id']))
    return ordered[-6:], games


def topic_context_projection(store, message):
    """只输出本人条件和来源引用；公共卡片不能覆盖个人事实，也不能代表建局授权。"""
    record = verified_topic_message(store, message)
    if record is None:
        return None
    classification = record['classification']
    rows, game_ids = [record], []
    if record.get('topic_id') and classification.get('goal_relation') == 'continue':
        rows, game_ids = _own_goal_sources(store, record, classification.get('task_context_id'), message.sent_at)
    if record.get('topic_id') and classification.get('intent') in {'delegate', 'create'}:
        bindings = store.list_group_topic_customer_bindings(record['topic_id'], record['customer_id'])
        games = store.games
        registered_ids = [game_id for b in bindings
            if b['conversation_id'] == message.conversation_id for game_id in b['game_ids']
            if game_id in games and games[game_id].organizer_id == record['customer_id']]
        game_ids = list(dict.fromkeys(game_ids + registered_ids))
    facts = [{'source_message_id': row['message_id'], 'sent_at': row['sent_at'], 'constraints': _own_constraints(row)}
             for row in rows]
    return {'topic_id': record.get('topic_id'), 'intent': classification['intent'],
            'goal_relation': classification.get('goal_relation'), 'related_game_ids': game_ids,
            'own_condition_sources': facts,
            'contract': '分类与条件是当前用户原话的解释建议，不是执行授权或已完成状态。询局不占座、不自动建局。明确委托后，若本人已登记的related_game_ids可定位原局，应按原局查找补位、复用看板，不另建一桌。board_only自招局只安排群看板，画像排序不证明有符合条件的人。条件冲突以本轮原话和数据库事实核对，无法确定目标时先澄清。没有具体可参加选项的全新群转私聊委托，完成实际工作后使用silent；群内承接可简短反馈或静默。面向个人的参加征询优先私聊。只有明确能补齐的具体选项且无法私聊，才考虑在原群用公开昵称简短征询本人，或请本人主动联系店方。只提公开局况，勿公开其他人的身份或通道偏好。'}
