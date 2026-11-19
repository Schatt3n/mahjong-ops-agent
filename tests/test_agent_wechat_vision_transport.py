"""Exercise the real vision client against an isolated transport fixture."""

from __future__ import annotations

import io
import json
import re
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from mahjong_agent_runtime.llm import (
    OPENCODE_GO_BASE_URL,
    AgentLLMConfig,
    OpenAICompatibleAgentClient,
    llm_conversation_scope,
)
from scripts import agent_wechat_media as media


CONFIG = AgentLLMConfig(
    api_key="synthetic-vision-key",
    provider="opencode_go",
    model="deepseek-v4.1-flash",
    base_url=OPENCODE_GO_BASE_URL,
)
IMAGE_REF = media.WeChatImageRef("private-account", "private-thread", "first-image", "sticker", True)


def test_open_code_vision_uses_the_same_opaque_session_as_text_gate(monkeypatch):
    monkeypatch.setattr(AgentLLMConfig, "from_env", classmethod(lambda cls: CONFIG))
    downloads = []
    requests = []

    def download(ref, representation, *, token, max_bytes):
        downloads.append((ref.message_id, representation, token, max_bytes))
        return media.WeChatImage(b"GIF89a", "image/gif", "original", "original")

    def open_model(request, *, timeout):
        requests.append(request)
        answer = json.dumps({"description": "A cat", "visible_text": "OK"})
        response = io.BytesIO(json.dumps({"choices": [{"message": {"content": answer}}]}).encode())
        response.status = 200
        return response

    monkeypatch.setattr(media, "fetch_wechat_image", download)
    monkeypatch.setattr(media, "_open_vision", open_model)
    result = media.recognize_wechat_image(IMAGE_REF, trace_id="trace-first", token="adapter-token")
    other = replace(IMAGE_REF, thread_id="another-thread", message_id="another-image")
    media.recognize_wechat_image(other, trace_id="trace-second", token="adapter-token")
    with llm_conversation_scope("wechaty:contact:private-thread"):
        OpenAICompatibleAgentClient(CONFIG, urlopen=open_model).complete(
            [{"role": "user", "content": "Consider the image observation with the conversation."}],
            trace_id="trace-classifier", timeout_seconds=2,
        )

    first, second, gate = requests
    first_session = first.get_header("X-opencode-session")
    assert re.fullmatch(r"ses_[0-9a-f]{32}", first_session)
    assert first_session == gate.get_header("X-opencode-session")
    assert first_session != second.get_header("X-opencode-session")
    assert first.get_header("User-agent") == "mahjong-ops-agent/1.0"
    assert first.full_url == f"{OPENCODE_GO_BASE_URL}/chat/completions"
    payload = json.loads(first.data)
    assert payload["model"] == CONFIG.model
    assert payload["messages"][1]["content"][1] == {
        "type": "image_url", "image_url": {"url": "data:image/gif;base64,R0lGODlh"},
    }
    assert downloads == [
        ("first-image", "preview", "adapter-token", media.VISION_MAX_BYTES),
        ("another-image", "preview", "adapter-token", media.VISION_MAX_BYTES),
    ]
    assert result["status"] == "recognized"
    assert (result["description"], result["visible_text"]) == ("A cat", "OK")
    assert result["completion"]["attempts"][0]["http_status"] == 200
    encoded = json.dumps(result)
    assert "R0lGODlh" not in encoded and CONFIG.api_key not in encoded and first_session not in encoded


@pytest.mark.parametrize(
    "ref,config,reason",
    [
        (replace(IMAGE_REF, incoming=False), CONFIG, "vision_outside_private_ingress"),
        (replace(IMAGE_REF, thread_id="example@chatroom"), CONFIG, "vision_outside_private_ingress"),
        (IMAGE_REF, replace(CONFIG, model="gpt-6-luna"), "vision_model_unconfigured"),
        (IMAGE_REF, replace(CONFIG, api_protocol="responses"), "vision_model_unconfigured"),
    ],
)
def test_unsupported_model_or_conversation_cannot_download_media(monkeypatch, ref, config, reason):
    monkeypatch.setattr(AgentLLMConfig, "from_env", classmethod(lambda cls: config))

    def unexpected_download(*args, **kwargs):
        pytest.fail("Vision must reject this request before media download.")

    monkeypatch.setattr(media, "fetch_wechat_image", unexpected_download)
    assert media.recognize_wechat_image(ref, trace_id="trace-rejected", token="adapter-token") == {
        "status": "skipped", "reason": reason,
    }


def test_authenticated_media_request_never_follows_redirect(monkeypatch):
    requests = []

    class RedirectingAdapter(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            if self.path.startswith("/media?"):
                self.send_response(302)
                self.send_header("Location", "/unexpected")
            else:
                self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), RedirectingAdapter)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_CONTROL_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    try:
        with pytest.raises(media.WeChatMediaError) as exc:
            media.fetch_wechat_image(IMAGE_REF, "original", token="adapter-token", max_bytes=1024)
        assert exc.value.status == 503
        (path,) = requests
        assert path.startswith("/media?")
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
