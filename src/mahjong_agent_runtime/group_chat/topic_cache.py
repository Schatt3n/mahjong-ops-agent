"""缓存可重建的公开话题卡片；Redis 丢失、乱序或不可用均不得改变 SQLite 事实。"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math

from ..models import now
from .topic_models import topic_is_hot, validate_public_topic_summary


# 同版本重放不刷新 TTL；版本检查和索引更新必须一起完成，避免并发旧结果覆盖新卡片。
# 每群索引最多保留 200 个热提示，防止持续活跃群留下无限过期成员；全集仍由 Store 核验。
_PUT_SCRIPT = '''
local old = redis.call('GET', KEYS[1])
if old then
  local ok, value = pcall(cjson.decode, old)
  if ok and tonumber(value.version) >= tonumber(ARGV[2]) then return 0 end
end
local clock = redis.call('TIME')
local now_ms = tonumber(clock[1]) * 1000 + math.floor(tonumber(clock[2]) / 1000)
local expires = math.min(tonumber(ARGV[3]), now_ms + tonumber(ARGV[5]))
if expires <= now_ms then return 0 end
redis.call('SET', KEYS[1], ARGV[1], 'PXAT', expires)
redis.call('ZADD', KEYS[2], ARGV[4], ARGV[6])
redis.call('ZREMRANGEBYRANK', KEYS[2], 0, -201)
local index_ttl = redis.call('PTTL', KEYS[2])
if index_ttl < expires - now_ms then redis.call('PEXPIREAT', KEYS[2], expires) end
return 1
'''


def public_topic_card(topic: dict) -> dict:
    """投影固定的公开卡片字段；原文、分类原因、客户 ID 和目标绑定不进入 Redis。"""
    required = ('topic_id', 'room_id', 'status', 'version', 'created_at', 'last_activity_at', 'active_until')
    card = {key: topic[key] for key in required}
    if type(card['version']) is not int or card['version'] < 1:
        raise ValueError('invalid cached topic version')
    if card['status'] not in {'active', 'closed'}:
        raise ValueError('invalid cached topic status')
    for key in ('topic_id', 'room_id'):
        if not isinstance(card[key], str) or not card[key]:
            raise ValueError('invalid cached topic identity')
    for key in ('created_at', 'last_activity_at', 'active_until'):
        stamp = datetime.fromisoformat(card[key])
        if stamp.tzinfo is None:
            raise ValueError('cached topic timestamp requires timezone')
    card['public_summary'] = validate_public_topic_summary(topic.get('public_summary'))
    return card


class RedisGroupTopicCache:
    """按群索引活跃卡片，绝不存用户画像、原文或私人任务；错误只触发数据库回源。"""

    def __init__(self, client, *, namespace: str = 'mahjong:group_topics', max_ttl_seconds: int = 1800):
        """接收独立 Redis 客户端；命名空间避免测试或多实例清理影响协调锁。"""
        if not namespace or max_ttl_seconds <= 0:
            raise ValueError('invalid topic cache configuration')
        self.client = client
        self.namespace = namespace
        self.max_ttl_seconds = max_ttl_seconds
        self.last_error: str | None = None

    @classmethod
    def from_url(cls, url: str, *, namespace: str = 'mahjong:group_topics', max_ttl_seconds: int = 1800):
        """可选加载 Redis 依赖；连接超时受限，入口可按 MAHJONG_TOPIC_REDIS_URL 单独装配。"""
        import redis
        client = redis.Redis.from_url(url, decode_responses=True, socket_connect_timeout=0.2, socket_timeout=0.2)
        return cls(client, namespace=namespace, max_ttl_seconds=max_ttl_seconds)

    def _topic_key(self, topic_id: str) -> str:
        """散列外部标识，防止分隔符或群名称污染键结构。"""
        return self.namespace + ':topic:' + hashlib.sha256(topic_id.encode()).hexdigest()

    def _room_key(self, room_id: str) -> str:
        """每群独立索引；不把个人绑定索引放入共享 Redis。"""
        return self.namespace + ':room:' + hashlib.sha256(room_id.encode()).hexdigest()

    def put(self, topic: dict) -> bool:
        """仅保存更新版本并使用事实中的绝对过期时间，读取和重复处理均不会续活跃。"""
        card = public_topic_card(topic)
        if card['status'] != 'active':
            return False
        try:
            accepted = self.client.eval(
                _PUT_SCRIPT, 2, self._topic_key(card['topic_id']), self._room_key(card['room_id']),
                json.dumps(card, ensure_ascii=False), card['version'],
                math.floor(datetime.fromisoformat(card['active_until']).timestamp()*1000),
                datetime.fromisoformat(card['last_activity_at']).timestamp(),
                self.max_ttl_seconds*1000, card['topic_id'],
            )
            self.last_error = None
            return bool(accepted)
        except Exception as exc:
            # 缓存不是提交条件；记录错误类型供 trace 使用，不记录可能含凭证的异常文本。
            self.last_error = type(exc).__name__
            return False

    def get(self, topic_id: str) -> dict | None:
        """读取严格公开投影，损坏或未命中视为缓存缺失，由调用方回源版本核验。"""
        try:
            payload = self.client.get(self._topic_key(topic_id))
            result = public_topic_card(json.loads(payload)) if payload else None
            if result is not None and result['topic_id'] != topic_id:
                raise ValueError('cached topic identity mismatch')
            self.last_error = None
            return result
        except Exception as exc:
            self.last_error = type(exc).__name__
            return None

    def list_candidates(self, room_id: str, *, at: datetime | None = None, limit: int = 5) -> list[dict]:
        """从热索引读取有限候选提示；逐卡检查活动期，索引本身不证明需求存在。"""
        stamp = at or now()
        count = max(0, min(limit, 100))
        if count == 0:
            return []
        try:
            ids = self.client.zrevrangebyscore(self._room_key(room_id), '+inf', '-inf', start=0, num=count)
            self.last_error = None
        except Exception as exc:
            self.last_error = type(exc).__name__
            return []
        cards = []
        for topic_id in ids:
            card = self.get(topic_id.decode() if isinstance(topic_id, bytes) else topic_id)
            if card is not None and card['room_id'] == room_id and topic_is_hot(card, stamp):
                cards.append(card)
        return cards


def load_group_topic_candidates(store, cache: RedisGroupTopicCache | None = None, *, room_id: str,
                                customer_id: str | None = None, at: datetime | None = None,
                                limit: int = 5, include_cold: bool = False) -> list[dict]:
    """用事实索引补齐部分缓存并核验版本；本人冷目标直接回源，不泄露其他人的绑定。

    SQLite 候选查询始终执行：一部分 Redis 写入失败后，单纯命中热索引不能证明候选全集。
    目前优先保持完整性，缓存只复用公开卡片；后续优化需增加可证明完整的房间修订号。
    """
    stamp = at or now()
    rows = store.list_group_topics(room_id=room_id, customer_id=customer_id, at=stamp,
                                   limit=limit, include_cold=include_cold)
    if cache is None:
        return rows
    cached = {item['topic_id']: item for item in cache.list_candidates(room_id, at=stamp, limit=limit)}
    result = []
    for row in rows:
        card = public_topic_card(row)
        hit = cached.get(row['topic_id'])
        if hit == card:
            result.append(hit)
        else:
            result.append(row)
            if topic_is_hot(row, stamp):
                cache.put(row)
    return result
