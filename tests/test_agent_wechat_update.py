from __future__ import annotations

import io
import json
from urllib.error import URLError

import pytest

from scripts import check_agent_wechat_update as update


class ReleaseResponse(io.BytesIO):
    def geturl(self):
        return update.RELEASE_API


def release_response(tag, *, draft=False, prerelease=False):
    return ReleaseResponse(json.dumps({
        "tag_name": tag,
        "draft": draft,
        "prerelease": prerelease,
        "html_url": "https://invalid.example/ignore-me",
    }).encode())


def test_each_start_fetches_latest_release_and_reports_newer_version(monkeypatch, capsys):
    requests = []
    published_tags = iter(["v2.9.0", "v2.10.0", "v2.10.0"])

    def fake_request(request, *, timeout):
        requests.append(request)
        assert request.full_url == update.RELEASE_API
        assert not request.has_header("Authorization")
        assert request.get_header("Cache-control") == "no-cache"
        assert timeout == 4
        return release_response(next(published_tags))

    monkeypatch.setattr(update, "urlopen", fake_request)
    update.check_for_update("2.9.0")
    assert capsys.readouterr().out == ""

    # 上一次刚检查过，本次也应立即看到其后发布的新版本。
    update.check_for_update("v2.9.0")
    output = capsys.readouterr().out
    assert "有新版本 v2.10.0（当前 v2.9.0）" in output
    assert "请查看发布说明并手动更新" in output
    assert "https://github.com/thisnick/agent-wechat/releases/tag/v2.10.0" in output
    assert "invalid.example" not in output
    update.check_for_update("2.11.0")
    assert capsys.readouterr().out == ""
    assert len(requests) == 3


@pytest.mark.parametrize(
    ("tag", "draft", "prerelease"),
    [("v2.6.0-rc.1", False, False), ("v02.6.0", False, False), ("v2.6.0", True, False), ("v2.6.0", False, True)],
)
def test_ignores_invalid_release(monkeypatch, capsys, tag, draft, prerelease):
    monkeypatch.setattr(update, "urlopen", lambda *args, **kwargs: release_response(tag, draft=draft, prerelease=prerelease))

    update.check_for_update("2.5.0")

    assert not capsys.readouterr().out


def test_unavailable_network_is_silent(monkeypatch, capsys):
    def unavailable(*args, **kwargs):
        raise URLError("offline")

    monkeypatch.setattr(update, "urlopen", unavailable)

    update.check_for_update("2.5.0")

    assert capsys.readouterr() == ("", "")


def test_unidentified_local_version_skips_check(monkeypatch):
    monkeypatch.setattr(update, "urlopen", lambda *args, **kwargs: pytest.fail("Unknown version should skip network"))

    update.check_for_update("2.5.0-dev")
