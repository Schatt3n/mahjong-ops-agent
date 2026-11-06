"""将经持久来源验证的只读询局边界转成工具权限，不参与自然语言识别。"""

from .topic_context import verified_topic_message


def unresolved_topic_target(store, record):
    """判断持久分类是否缺少可验证的本人目标；不解释原话，也不按最近任务补空引用。"""
    classification = record['classification']
    if classification.get('intent')=='uncertain' or classification.get('goal_relation')=='uncertain' or classification.get('topic_action')=='uncertain':
        return True
    if classification.get('intent') not in {'update','cancel'}:
        return False
    selected = classification.get('task_context_id')
    topic_id = record.get('topic_id')
    if classification.get('goal_relation')!='continue' or not topic_id or not selected:
        return True
    context = store.get_task_context(selected)
    if context is None or context.customer_id != record['customer_id']:
        return True
    return not any(binding.get('task_context_id')==selected and binding['conversation_id']==context.conversation_id
                   for binding in store.list_group_topic_customer_bindings(topic_id,record['customer_id']))


def tool_execution_scope(store, message):
    """查询或目标不确定时允许读取与记录本人上下文，拒绝占座、建局及外发副作用。"""
    record = verified_topic_message(store, message)
    if record and (record['classification'].get('intent')=='query' or unresolved_topic_target(store,record)):
        return 'query_only'
    return None
