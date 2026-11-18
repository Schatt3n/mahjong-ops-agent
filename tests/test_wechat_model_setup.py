from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from urllib.parse import urlencode, urlsplit

import pytest

from scripts import wechat_model_setup as setup


def request(url, *, fields=None, headers=None, body=None):
    target = urlsplit(url)
    connection = HTTPConnection(target.hostname, target.port, timeout=1)
    try:
        if fields is None and body is None:
            connection.request("GET", target.path, headers=headers or {})
        else:
            defaults = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Origin": f"http://{target.netloc}",
            }
            connection.request("POST", target.path, body=urlencode(fields) if body is None else body,
                               headers={**defaults, **(headers or {})})
        with connection.getresponse() as response:
            return response.status, dict(response.getheaders()), response.read().decode()
    finally:
        connection.close()


def run_setup(monkeypatch, browser, validate, initial=None):
    monkeypatch.setattr(setup, "SETUP_TIMEOUT_SECONDS", 3)
    with ThreadPoolExecutor(max_workers=1) as pool:
        browser_run = None

        def open_browser(url):
            nonlocal browser_run
            browser_run = pool.submit(browser, url)
            return True

        monkeypatch.setattr(setup, "_open_browser", open_browser)
        try:
            return setup.configure_model_in_browser(initial=initial or {}, validate=validate)
        finally:
            browser_run.result(timeout=1)


def model_fields(**replacements):
    return {
        "provider": "openai_compatible",
        "model": "test-original-model",
        "base_url": "http://127.0.0.1:9000/v1",
        "api_protocol": "chat_completions",
        "response_format": "",
        "api_key": "test-form-secret",
        "action": "save",
        **replacements,
    }


def test_opens_private_local_form_and_returns_all_settings(monkeypatch, capsys):
    initial = {"MAHJONG_LLM_API_KEY": "test-existing-secret", "MAHJONG_LLM_MODEL": "<model>"}
    fields = model_fields()
    called_with = []

    def browser(url):
        target = urlsplit(url)
        assert target.hostname == "127.0.0.1"
        assert target.port is not None
        assert target.path.startswith("/setup/") and len(target.path.split("/")[-1]) >= 32
        assert initial["MAHJONG_LLM_API_KEY"] not in url
        status, headers, page = request(url)
        assert status == 200
        assert headers["Cache-Control"] == "no-store"
        assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]
        assert 'type="password" name="api_key"' in page
        assert '<option value="chat_completions" selected>' in page
        assert 'value="&lt;model&gt;"' in page
        assert initial["MAHJONG_LLM_API_KEY"] not in page
        status, _, confirmation = request(url, fields=fields)
        assert status == 200
        assert "已提交模型配置" in confirmation
        assert fields["api_key"] not in confirmation

    result = run_setup(monkeypatch, browser, called_with.append, initial)

    assert result == {key: fields[name] for name, key in setup.FIELDS.items()}
    assert called_with == [result]
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("configured_protocol", [None, "chat_completions"])
def test_luna_defaults_to_responses_and_preserves_an_explicit_choice(monkeypatch, configured_protocol):
    initial = {"MAHJONG_LLM_PROVIDER": "opencode_go", "MAHJONG_LLM_MODEL": "gpt-6-luna"}
    if configured_protocol is not None:
        initial["MAHJONG_LLM_API_PROTOCOL"] = configured_protocol
    selected = configured_protocol or "responses"

    def browser(url):
        status, _, page = request(url)
        assert status == 200
        assert f'<option value="{selected}" selected>' in page
        assert "GPT‑6 Luna 请选择 Responses" in page
        fields = model_fields(provider="opencode_go", model="gpt-6-luna", base_url="", api_protocol="responses")
        assert request(url, fields=fields)[0] == 200

    result = run_setup(monkeypatch, browser, lambda _: None, initial)

    assert result["MAHJONG_LLM_API_PROTOCOL"] == "responses"


def test_rejects_bad_host_session_and_origin_before_validation(monkeypatch):
    fields = model_fields()
    called_with = []

    def browser(url):
        assert request(url, headers={"Host": "external.example"})[0] == 404
        assert request(url + "wrong")[0] == 404
        assert request(url + "wrong", fields=fields)[0] == 403
        assert request(url, fields=fields, headers={"Origin": "https://external.example"})[0] == 403
        assert request(url, fields=fields, headers={"Host": "external.example"})[0] == 403
        assert request(url, fields=fields)[0] == 200

    assert run_setup(monkeypatch, browser, called_with.append) is not None
    assert len(called_with) == 1


def test_bad_validation_keeps_form_without_reflecting_key(monkeypatch, capsys):
    fields = model_fields()
    requested = []

    def validate(settings):
        requested.append(settings["MAHJONG_LLM_MODEL"])
        if settings["MAHJONG_LLM_MODEL"] == fields["model"]:
            raise ValueError("<错误> " + settings["MAHJONG_LLM_API_KEY"])

    def browser(url):
        status, headers, retry = request(url, fields=fields)
        assert status == 422
        assert headers["Cache-Control"] == "no-store"
        assert "&lt;错误&gt; [已隐藏]" in retry
        assert fields["api_key"] not in retry
        assert request(url, fields={**fields, "model": "retry-model"})[0] == 200

    result = run_setup(monkeypatch, browser, validate)

    assert result["MAHJONG_LLM_MODEL"] == "retry-model"
    assert requested == [fields["model"], "retry-model"]
    assert capsys.readouterr() == ("", "")


def test_rejects_malformed_payload_and_can_cancel(monkeypatch):
    requested = []

    def browser(url):
        assert request(url, body="a" * (setup.MAX_POST_BYTES + 1))[0] == 400
        assert request(url, body=urlencode(model_fields()) + "&model=duplicate")[0] == 400
        status, _, page = request(url, fields=model_fields(api_key="", model="", action="cancel"))
        assert status == 200
        assert "已取消模型配置" in page

    with pytest.raises(setup.ModelSetupCancelled, match="已取消"):
        run_setup(monkeypatch, browser, requested.append)
    assert requested == []


def test_no_desktop_returns_to_terminal_but_wait_expiration_cancels(monkeypatch):
    requested = []
    monkeypatch.setattr(setup, "_open_browser", lambda _: False)
    assert setup.configure_model_in_browser(initial={}, validate=requested.append) is None

    monkeypatch.setattr(setup, "_open_browser", lambda _: True)
    monkeypatch.setattr(setup, "SETUP_TIMEOUT_SECONDS", 0)
    with pytest.raises(setup.ModelSetupCancelled, match="等待超时"):
        setup.configure_model_in_browser(initial={}, validate=requested.append)
    assert requested == []
