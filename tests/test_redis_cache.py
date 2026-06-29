from __future__ import annotations

from io import BytesIO

from mahjong_agent.redis_cache import RedisCache, RedisCacheError


def test_redis_cache_config_from_url() -> None:
    cache = RedisCache.from_url("redis://user:secret@localhost:6380/2", timeout_seconds=1.2)

    assert cache.config.host == "localhost"
    assert cache.config.port == 6380
    assert cache.config.db == 2
    assert cache.config.username == "user"
    assert cache.config.password == "secret"
    assert cache.config.timeout_seconds == 1.2


def test_redis_resp_parser_handles_core_types() -> None:
    cache = RedisCache()

    assert cache._read_response(BytesIO(b"+PONG\r\n")) == "PONG"
    assert cache._read_response(BytesIO(b":3\r\n")) == 3
    assert cache._read_response(BytesIO(b"$5\r\nhello\r\n")) == b"hello"
    assert cache._read_response(BytesIO(b"$-1\r\n")) is None
    assert cache._read_response(BytesIO(b"*2\r\n+OK\r\n:1\r\n")) == ["OK", 1]


def test_redis_resp_parser_raises_on_error_response() -> None:
    cache = RedisCache()

    try:
        cache._read_response(BytesIO(b"-ERR bad command\r\n"))
    except RedisCacheError as exc:
        assert "ERR bad command" in str(exc)
    else:
        raise AssertionError("expected RedisCacheError")
