"""在启动后台检查 agent-wechat 的稳定版本，仅在本地终端提示。"""

from __future__ import annotations

from http.client import HTTPException
import json
import re
from urllib.request import Request, urlopen


RELEASE_API = "https://api.github.com/repos/thisnick/agent-wechat/releases/latest"
RELEASE_PAGE = "https://github.com/thisnick/agent-wechat/releases/tag/"
MAX_RESPONSE_BYTES = 128 * 1024
VERSION = re.compile(r"v?(0|[1-9][0-9]{0,8})\.(0|[1-9][0-9]{0,8})\.(0|[1-9][0-9]{0,8})")


def _version(value: object) -> tuple[int, int, int] | None:
    """只比较稳定版的三段数字，预发布标签不会被当作可用更新。"""
    match = VERSION.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        return None
    return int(match[1]), int(match[2]), int(match[3])


def _latest_tag() -> str | None:
    """以无凭证请求查询上游稳定版，限制响应体，网络故障由启动流程忽略。"""
    request = Request(RELEASE_API, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "Mahjong-Ops-Agent",
        "Cache-Control": "no-cache",
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


def check_for_update(current_version: str) -> None:
    """每次启动重新查询正式发布，发现比运行版本更新的发布时提示手动升级。"""
    current = _version(current_version)
    if current is None:
        return
    tag = _latest_tag()
    if tag is None:
        return
    latest = _version(tag)
    if latest is not None and latest > current:
        print(
            f"agent-wechat 有新版本 v{tag.removeprefix('v')}（当前 v{current_version.removeprefix('v')}）。"
            f"请查看发布说明并手动更新：{RELEASE_PAGE}{tag}",
            flush=True,
        )
