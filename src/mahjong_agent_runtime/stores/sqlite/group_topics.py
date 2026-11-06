"""持久保存群话题、不可变来源与个人目标关联；Redis 不参与事实提交事务。"""

from ...group_chat.topic_models import prepare_topic_record, prepare_topic_binding, topic_source_key, topic_is_hot
from ...models import DEFAULT_TZ, now
from .serialization import _dumps, _loads


class SQLiteGroupTopicStoreMixin:
    """用立即事务串行化来源去重与话题 CAS，保证跨连接不会覆盖新的分类结果。"""

    __slots__ = ()

    def get_group_topic(self, topic_id):
        """按稳定 ID 读取话题；冷却只影响检索，不删除该记录。"""
        with self._lock:
            row=self._connection.execute('SELECT payload FROM runtime_group_topics WHERE topic_id=?',(topic_id,)).fetchone()
            return _loads(row['payload']) if row else None

    def get_group_topic_message(self, channel, room_id, customer_id, message_id):
        """按完整来源作用域读取不可变原文和分类证据。"""
        with self._lock:
            key=topic_source_key(channel,room_id,customer_id,message_id)
            row=self._connection.execute('SELECT payload FROM runtime_group_topic_messages WHERE source_key=?',(key,)).fetchone()
            return _loads(row['payload']) if row else None

    def get_group_topic_for_message_reference(self, channel, room_id, message_id):
        """引用仅唯一定位公开话题；跨客户编号歧义返回空，绝不返回被引用者原文。"""
        with self._lock:
            rows = self._connection.execute(
                'SELECT DISTINCT topic_id FROM runtime_group_topic_messages WHERE channel=? AND room_id=? AND message_id=? AND topic_id IS NOT NULL',
                (channel, room_id, message_id),
            ).fetchall()
            ids = {row['topic_id'] for row in rows}
            return self.get_group_topic(next(iter(ids))) if len(ids) == 1 else None

    def record_group_topic_message(self, **kwargs):
        """先在事务内检查原文和版本，再原子写入话题及消息账本；失败全部回滚。"""
        with self._write_transaction():
            old=self.get_group_topic_message(kwargs['channel'],kwargs['room_id'],kwargs['customer_id'],kwargs['message_id'])
            topic_id=old['topic_id'] if old else kwargs.get('topic_id')
            result=prepare_topic_record(existing_topic=self.get_group_topic(topic_id),existing_message=old,**kwargs)
            if result['deduplicated']:return result
            topic=result['topic'];message=result['message']
            if topic:
                self._connection.execute('''INSERT INTO runtime_group_topics(topic_id,room_id,status,version,last_activity_at,active_until,payload)
                    VALUES(?,?,?,?,?,?,?) ON CONFLICT(topic_id) DO UPDATE SET status=excluded.status,version=excluded.version,
                    last_activity_at=excluded.last_activity_at,active_until=excluded.active_until,payload=excluded.payload''',
                    (topic['topic_id'],topic['room_id'],topic['status'],topic['version'],topic['last_activity_at'],topic['active_until'],_dumps(topic)))
            self._connection.execute('''INSERT INTO runtime_group_topic_messages(source_key,channel,room_id,customer_id,message_id,topic_id,sent_at,payload)
                VALUES(?,?,?,?,?,?,?,?)''',(message['source_key'],message['channel'],message['room_id'],message['customer_id'],message['message_id'],
                                          message['topic_id'],message['sent_at'],_dumps(message)))
            return result

    def list_group_topics(self, *, room_id, customer_id=None, at=None, limit=5, include_cold=False):
        """数据库负责候选全集；本人查询在 SQL 中隔离，Redis 缺失时仍能恢复冷目标。"""
        stamp=(at or now()).astimezone(DEFAULT_TZ);clauses=['room_id=?'];params=[room_id]
        if customer_id is not None:
            clauses.append('''topic_id IN (SELECT topic_id FROM runtime_group_topic_messages WHERE customer_id=?
                UNION SELECT topic_id FROM runtime_group_topic_bindings WHERE customer_id=?)''')
            params.extend([customer_id,customer_id])
        if not include_cold:
            clauses.extend(["status='active'",'active_until>?']);params.append(stamp.isoformat())
        params.append(max(0,min(limit,100)))
        with self._lock:
            rows=self._connection.execute('SELECT payload FROM runtime_group_topics WHERE '+' AND '.join(clauses)+' ORDER BY last_activity_at DESC,topic_id DESC LIMIT ?',params).fetchall()
            return [_loads(row['payload']) for row in rows]

    def list_group_topic_customer_messages(self, topic_id, customer_id, *, limit=6):
        """只读当前客户在指定话题的原文，公共候选卡片没有消息正文。"""
        with self._lock:
            rows=self._connection.execute('''SELECT payload FROM runtime_group_topic_messages WHERE topic_id=? AND customer_id=?
                ORDER BY sent_at DESC,source_key DESC LIMIT ?''',(topic_id,customer_id,max(0,min(limit,100)))).fetchall()
            return [_loads(row['payload']) for row in rows]

    def get_group_topic_customer(self, topic_id, customer_id):
        """获取最近个人绑定；旧目标关联不会因新目标建立而消失。"""
        rows=self.list_group_topic_customer_bindings(topic_id,customer_id)
        return rows[0] if rows else None

    def list_group_topic_customer_bindings(self, topic_id, customer_id):
        """按更新顺序返回该客户的目标关联历史，不包含其他客户任务。"""
        with self._lock:
            rows=self._connection.execute('''SELECT payload FROM runtime_group_topic_bindings WHERE topic_id=? AND customer_id=?
                ORDER BY updated_at DESC,binding_id DESC''',(topic_id,customer_id)).fetchall()
            return [_loads(row['payload']) for row in rows]

    def bind_group_topic_customer(self, **kwargs):
        """目标与来源关联经过 Store 事务校验，不改变任何牌局或授权状态。"""
        with self._write_transaction():
            topic=self.get_group_topic(kwargs['topic_id'])
            previous=self.get_group_topic_customer(kwargs['topic_id'],kwargs['customer_id'])
            binding=prepare_topic_binding(self,topic=topic,previous=previous,**kwargs)
            self._connection.execute('''INSERT INTO runtime_group_topic_bindings(binding_id,topic_id,customer_id,conversation_id,task_context_id,updated_at,payload)
                VALUES(?,?,?,?,?,?,?) ON CONFLICT(binding_id) DO UPDATE SET updated_at=excluded.updated_at,payload=excluded.payload''',
                (binding['binding_id'],binding['topic_id'],binding['customer_id'],binding['conversation_id'],binding['task_context_id'],binding['updated_at'],_dumps(binding)))
            return binding
