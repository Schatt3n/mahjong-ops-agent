"""内存话题账本与本人任务关联，使用同一把 Store 锁保持来源和版本原子性。"""

from copy import deepcopy
from ...group_chat.topic_models import prepare_topic_record, prepare_topic_binding, topic_source_key, topic_is_hot
from ...models import now


class InMemoryGroupTopicStoreMixin:
    """提供与 SQLite 相同的话题合同；不借话题归属修改人数、任务或用户授权。"""

    def get_group_topic(self, topic_id):
        """返回独立快照，调用方修改不能绕过事务覆盖话题。"""
        with self._lock:
            return deepcopy(self.group_topics.get(topic_id))

    def get_group_topic_message(self, channel, room_id, customer_id, message_id):
        """按完整消息来源读取已经保存的分类证据。"""
        with self._lock:
            return deepcopy(self.group_topic_messages.get(topic_source_key(channel,room_id,customer_id,message_id)))

    def get_group_topic_for_message_reference(self, channel, room_id, message_id):
        """引用仅唯一定位公开话题；跨客户编号歧义返回空，绝不返回被引用者原文。"""
        with self._lock:
            ids = {message['topic_id'] for message in self.group_topic_messages.values()
                   if message['channel'] == channel and message['room_id'] == room_id
                   and message['message_id'] == message_id and message['topic_id'] is not None}
            return self.get_group_topic(next(iter(ids))) if len(ids) == 1 else None

    def record_group_topic_message(self, **kwargs):
        """一并保存原文、分类、话题版本；重放不会给话题续期。"""
        with self._lock:
            key=topic_source_key(kwargs['channel'],kwargs['room_id'],kwargs['customer_id'],kwargs['message_id'])
            old=self.group_topic_messages.get(key)
            topic_id=old['topic_id'] if old else kwargs.get('topic_id')
            result=prepare_topic_record(existing_topic=self.group_topics.get(topic_id),existing_message=old,**kwargs)
            if not result['deduplicated']:
                if result['topic']:self.group_topics[result['topic']['topic_id']]=deepcopy(result['topic'])
                self.group_topic_messages[key]=deepcopy(result['message'])
            return deepcopy(result)

    def list_group_topics(self, *, room_id, customer_id=None, at=None, limit=5, include_cold=False):
        """按活动时间读取有界候选；指定客户时严格只返回其消息或绑定关联的话题。"""
        stamp=at or now()
        with self._lock:
            own={m['topic_id'] for m in self.group_topic_messages.values() if m['customer_id']==customer_id}
            own.update(b['topic_id'] for b in self.group_topic_bindings.values() if b['customer_id']==customer_id)
            rows=[t for t in self.group_topics.values() if t['room_id']==room_id
                  and (customer_id is None or t['topic_id'] in own) and (include_cold or topic_is_hot(t,stamp))]
            return deepcopy(sorted(rows,key=lambda t:(t['last_activity_at'],t['topic_id']),reverse=True)[:max(0,min(limit,100))])

    def list_group_topic_customer_messages(self, topic_id, customer_id, *, limit=6):
        """只返回本人的原始输入，公共话题查询不能获得另一位客户原文。"""
        with self._lock:
            rows=[m for m in self.group_topic_messages.values() if m['topic_id']==topic_id and m['customer_id']==customer_id]
            return deepcopy(sorted(rows,key=lambda m:(m['sent_at'],m['source_key']),reverse=True)[:max(0,min(limit,100))])

    def get_group_topic_customer(self, topic_id, customer_id):
        """读取最近一次本人目标关联；历史目标仍通过绑定列表可检索。"""
        rows=self.list_group_topic_customer_bindings(topic_id,customer_id)
        return rows[0] if rows else None

    def list_group_topic_customer_bindings(self, topic_id, customer_id):
        """保留同一话题中的多次个人目标，不把旧任务覆盖成新任务。"""
        with self._lock:
            rows=[b for b in self.group_topic_bindings.values() if b['topic_id']==topic_id and b['customer_id']==customer_id]
            return deepcopy(sorted(rows,key=lambda b:(b['updated_at'],b['binding_id']),reverse=True))

    def bind_group_topic_customer(self, **kwargs):
        """在同一锁内校验本人任务与会话，再保存稳定关联；不接收模型直接授权。"""
        with self._lock:
            topic=self.group_topics.get(kwargs['topic_id'])
            previous=self.get_group_topic_customer(kwargs['topic_id'],kwargs['customer_id'])
            binding=prepare_topic_binding(self,topic=topic,previous=previous,**kwargs)
            self.group_topic_bindings[binding['binding_id']]=deepcopy(binding)
            return deepcopy(binding)
