from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any, BinaryIO
from urllib.parse import unquote, urlparse


class RedisCacheError(RuntimeError):
    pass


@dataclass(frozen=True)
class RedisCacheConfig:
    host: str = "127.0.0.1"
    port: int = 6379
    db: int = 0
    username: str | None = None
    password: str | None = None
    timeout_seconds: float = 0.3


class RedisCache:
    def __init__(self, config: RedisCacheConfig | None = None) -> None:
        self.config = config or RedisCacheConfig()

    @classmethod
    def from_url(cls, url: str, *, timeout_seconds: float = 0.3) -> "RedisCache":
        parsed = urlparse(url)
        if parsed.scheme not in {"redis", ""}:
            raise RedisCacheError(f"unsupported redis url scheme: {parsed.scheme}")
        db_text = parsed.path.lstrip("/") if parsed.path else "0"
        try:
            db = int(db_text or "0")
        except ValueError as exc:
            raise RedisCacheError(f"invalid redis db: {db_text}") from exc
        return cls(
            RedisCacheConfig(
                host=parsed.hostname or "127.0.0.1",
                port=parsed.port or 6379,
                db=db,
                username=unquote(parsed.username) if parsed.username else None,
                password=unquote(parsed.password) if parsed.password else None,
                timeout_seconds=timeout_seconds,
            )
        )

    def ping(self) -> bool:
        return self.execute("PING") == "PONG"

    def set_json(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> None:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if ttl_seconds is None:
            self.execute("SET", key, payload)
            return
        self.execute("SET", key, payload, "EX", str(max(1, int(ttl_seconds))))

    def get_json(self, key: str, default: Any = None) -> Any:
        raw = self.execute("GET", key)
        if raw is None:
            return default
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            return json.loads(str(raw))
        except json.JSONDecodeError:
            return default

    def delete(self, *keys: str) -> int:
        if not keys:
            return 0
        result = self.execute("DEL", *keys)
        return int(result or 0)

    def ttl(self, key: str) -> int:
        return int(self.execute("TTL", key))

    def execute(self, *args: str | bytes) -> Any:
        try:
            with socket.create_connection(
                (self.config.host, self.config.port),
                timeout=self.config.timeout_seconds,
            ) as sock:
                sock.settimeout(self.config.timeout_seconds)
                stream = sock.makefile("rb")
                if self.config.password:
                    if self.config.username:
                        self._execute_connected(sock, stream, "AUTH", self.config.username, self.config.password)
                    else:
                        self._execute_connected(sock, stream, "AUTH", self.config.password)
                if self.config.db:
                    self._execute_connected(sock, stream, "SELECT", str(self.config.db))
                return self._execute_connected(sock, stream, *args)
        except OSError as exc:
            raise RedisCacheError(str(exc)) from exc

    def _execute_connected(self, sock: socket.socket, stream: BinaryIO, *args: str | bytes) -> Any:
        self._write_command(sock, *args)
        return self._read_response(stream)

    def _write_command(self, sock: socket.socket, *args: str | bytes) -> None:
        chunks = [f"*{len(args)}\r\n".encode("utf-8")]
        for arg in args:
            data = arg if isinstance(arg, bytes) else str(arg).encode("utf-8")
            chunks.append(f"${len(data)}\r\n".encode("utf-8"))
            chunks.append(data)
            chunks.append(b"\r\n")
        sock.sendall(b"".join(chunks))

    def _read_response(self, stream: BinaryIO) -> Any:
        prefix = stream.read(1)
        if not prefix:
            raise RedisCacheError("empty redis response")
        line = stream.readline()
        if not line.endswith(b"\r\n"):
            raise RedisCacheError("malformed redis response")
        value = line[:-2]
        if prefix == b"+":
            return value.decode("utf-8")
        if prefix == b"-":
            raise RedisCacheError(value.decode("utf-8", errors="replace"))
        if prefix == b":":
            return int(value)
        if prefix == b"$":
            length = int(value)
            if length == -1:
                return None
            data = stream.read(length)
            trailer = stream.read(2)
            if trailer != b"\r\n":
                raise RedisCacheError("malformed redis bulk string")
            return data
        if prefix == b"*":
            length = int(value)
            if length == -1:
                return None
            return [self._read_response(stream) for _ in range(length)]
        raise RedisCacheError(f"unsupported redis response type: {prefix!r}")
