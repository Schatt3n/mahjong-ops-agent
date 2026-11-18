"""在启动后台检查 agent-wechat 的稳定版本，仅在本地终端提示。"""

from __future__ import annotations

from http.client import HTTPException
import json
from pathlib import Path
import re
import stat
import tempfile
import time
from urllib.request import Request, urlopen


RELEASE_API = "https://api.github.com/repos/thisnick/agent-wechat/releases/latest"
RELEASE_PAGE = "https://github.com/thisnick/agent-wechat/releases/tag/"
CACHE_SECONDS = 24 * 60 * 60
MAX_RESPONSE_BYTES = 128 * 1024
VERSION = re.compile(r"v?(0|[1-9][0-9]{0,8})\.(0|[1-9][0-9]{0,8})\.(0|[1-9][0-9]{0,8})")


def _version(value: object) -> tuple[int, int, int] | None:
    """只比较稳定版的三段数字，预发布标签不会被当作可用更新。"""
    match = VERSION.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        return None
    return int(match[1]), int(match[2]), int(match[3])


def _cached_tag(path: Path) -> str | None:
    """只读取一天以内的有效标签，缓存丢失或损坏时允许重新检查。"""
    try:
        metadata = path.stat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > 32
            or not 0 <= time.time() - metadata.st_mtime < CACHE_SECONDS
        ):
            return None
        with path.open(encoding="ascii") as file:
            tag = file.read(33).strip()
        return tag if _version(tag) is not None else None
    except (OSError, ValueError):
        return None


def _latest_tag() -> str | None:
    """以无凭证请求查询上游稳定版，限制响应体，网络故障由启动流程忽略。"""
    request = Request(RELEASE_API, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "Mahjong-Ops-Agent",
    })
    try:
        with urlopen(request, timeout=4) as response:
            if response.geturl() != RELEASE_API:
                return None
            body = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            return None
        release = json.loads(body)
    except (OSError, HTTPException, ValueError, RecursionError):
        return None
    if not isinstance(release, dict) or release.get("draft") is not False or release.get("prerelease") is not False:
        return None
    tag = release.get("tag_name")
    return tag if _version(tag) is not None else None


def _cache_tag(path: Path, tag: str) -> None:
    """用权限为 0600 的临时文件原子更新缓存，写入失败不影响提示。"""
    temp_path = None
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="ascii", dir=path.parent, delete=False) as file:
            temp_path = Path(file.name)
            file.write(f"{tag}\n")
        temp_path.replace(path)
    except OSError:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def check_for_update(current_version: str, cache_path: Path) -> None:
    """在后台比较正在运行的版本与上游稳定版，发现更新后打印发布说明链接。"""
    current = _version(current_version)
    if current is None:
        return
    tag = _cached_tag(cache_path)
    if tag is None:
        tag = _latest_tag()
        if tag is None:
            return
        _cache_tag(cache_path, tag)
    latest = _version(tag)
    if latest is not None and latest > current:
        print(
            f"agent-wechat 有新版本 v{tag.removeprefix('v')}（当前 v{current_version.removeprefix('v')}）。"
            f"发布说明 {RELEASE_PAGE}{tag}",
            flush=True,
        )
