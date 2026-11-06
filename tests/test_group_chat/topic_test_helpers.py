"""固定结构化语义建议，仅用于检验后端执行边界，不假装验证模型理解能力。"""

class FixedTopicClassifier:
    max_candidates = 5

    def __init__(self, intent='query', by_message_id=None):
        self.intent = intent
        self.by_message_id = by_message_id or {}
        self.calls = []

    def classify(self, *, message, candidates, own_context=None, board=None, trace_id):
        self.calls.append(message)
        intent = self.by_message_id.get(message.message_id, self.intent)
        candidate = next((x for x in candidates if x.get('own_task_context_id')), None)
        # 用确定的测试引用模拟模型 continue，不按自然语言关键词作分类。
        if candidate is None and candidates:
            candidate = candidates[0]
        tid = candidate.get('own_task_context_id') if candidate else None
        return {'intent':intent, 'topic_action':'none' if intent=='chitchat' else 'existing' if candidate else 'new',
            'topic_id':candidate['topic_id'] if candidate and intent!='chitchat' else None,
            'goal_relation':'none' if intent=='chitchat' else 'continue' if tid else 'new',
            'task_context_id':tid, 'public_summary':{}, 'constraints':{},
            'source_message_id':message.message_id, 'confidence':1, 'reasoning':'固定测试语义建议'}
