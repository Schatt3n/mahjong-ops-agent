from __future__ import annotations

import io
import json
import os
import time
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


def test_newer_release_is_reported_and_read_from_private_cache(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "state" / "latest-release"
    requests = []

    def fake_request(request, *, timeout):
        requests.append(request)
        assert request.full_url == update.RELEASE_API
        assert not request.has_header("Authorization")
        assert timeout == 4
        return release_response("v2.10.0")

    monkeypatch.setattr(update, "urlopen", fake_request)
    update.check_for_update("2.9.0", cache)

    output = capsys.readouterr().out
    assert "有新版本 v2.10.0（当前 v2.9.0）" in output
    assert "https://github.com/thisnick/agent-wechat/releases/tag/v2.10.0" in output
    assert "invalid.example" not in output
    assert cache.read_text(encoding="ascii") == "v2.10.0\n"
    assert cache.stat().st_mode & 0o777 == 0o600

    update.check_for_update("v2.9.0", cache)
    assert capsys.readouterr().out == output
    update.check_for_update("2.10.0", cache)
    assert capsys.readouterr().out == ""
    assert len(requests) == 1


def test_expired_cache_is_refreshed(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "latest-release"
    cache.write_text("v2.5.0\n", encoding="ascii")
    expired = time.time() - update.CACHE_SECONDS - 60
    os.utime(cache, (expired, expired))
    monkeypatch.setattr(update, "urlopen", lambda *args, **kwargs: release_response("v2.6.0"))

    update.check_for_update("2.5.0", cache)

    assert "有新版本 v2.6.0" in capsys.readouterr().out
    assert cache.read_text(encoding="ascii") == "v2.6.0\n"


@pytest.mark.parametrize(
    ("tag", "draft", "prerelease"),
    [("v2.6.0-rc.1", False, False), ("v02.6.0", False, False), ("v2.6.0", True, False), ("v2.6.0", False, True)],
)
def test_ignores_invalid_release(tmp_path, monkeypatch, capsys, tag, draft, prerelease):
    cache = tmp_path / "latest-release"
    monkeypatch.setattr(update, "urlopen", lambda *args, **kwargs: release_response(tag, draft=draft, prerelease=prerelease))

    update.check_for_update("2.5.0", cache)

    assert not capsys.readouterr().out
    assert not cache.exists()


def test_unavailable_network_is_silent(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "latest-release"

    def unavailable(*args, **kwargs):
        raise URLError("offline")

    monkeypatch.setattr(update, "urlopen", unavailable)

    update.check_for_update("2.5.0", cache)

    assert capsys.readouterr() == ("", "")
    assert not cache.exists()


def test_unidentified_local_version_skips_check(tmp_path, monkeypatch):
    monkeypatch.setattr(update, "urlopen", lambda *args, **kwargs: pytest.fail("Unknown version should skip network"))

    update.check_for_update("2.5.0-dev", tmp_path / "latest-release")
