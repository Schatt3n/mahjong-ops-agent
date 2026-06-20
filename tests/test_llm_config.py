from __future__ import annotations

import os

from mahjong_agent import ChannelType, Message
from mahjong_agent.llm import LLMConfig, OpenAICompatibleLLMResolver


def test_qwen_provider_defaults_to_dashscope_compatible_endpoint() -> None:
    with patched_env(
        MAHJONG_LLM_PROVIDER="qwen",
        MAHJONG_LLM_API_KEY="test-key",
        MAHJONG_LLM_MODEL=None,
        MAHJONG_LLM_BASE_URL=None,
        DASHSCOPE_API_KEY=None,
        OPENAI_API_KEY=None,
    ):
        config = LLMConfig.from_env()

    assert config is not None
    assert config.provider == "qwen"
    assert config.model == "qwen-plus"
    assert config.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_dashscope_api_key_implies_qwen_provider() -> None:
    with patched_env(
        DASHSCOPE_API_KEY="test-key",
        MAHJONG_LLM_PROVIDER=None,
        MAHJONG_LLM_API_KEY=None,
        MAHJONG_LLM_MODEL=None,
        MAHJONG_LLM_BASE_URL=None,
        OPENAI_API_KEY=None,
    ):
        config = LLMConfig.from_env()

    assert config is not None
    assert config.provider == "qwen"
    assert config.model == "qwen-plus"


def test_llm_payload_uses_context_message_view_instead_of_raw_metadata() -> None:
    resolver = OpenAICompatibleLLMResolver(
        LLMConfig(api_key="test-key", model="test-model")
    )
    message = Message(
        text="原始消息 13812345678",
        sender_id="wxid_raw_secret",
        sender_name="张哥",
        channel_id="group",
        channel_type=ChannelType.WECHAT_GROUP,
        metadata={"raw_private_token": "should-not-be-sent"},
    )
    context = {
        "current_message": {
            "text": "脱敏消息 [手机号]",
            "sender_ref": "customer_hash",
            "sender_display_name": "张哥",
            "channel_type": "wechat_group",
            "modalities": ["text"],
            "source": {"message_ref": "message_hash"},
        }
    }

    payload = resolver._message_payload(message, context)

    assert payload["text"] == "脱敏消息 [手机号]"
    assert payload["sender_ref"] == "customer_hash"
    assert "13812345678" not in str(payload)
    assert "wxid_raw_secret" not in str(payload)
    assert "raw_private_token" not in str(payload)


class patched_env:
    def __init__(self, **values: str | None) -> None:
        self.values = values
        self.original: dict[str, str | None] = {}

    def __enter__(self) -> None:
        for key, value in self.values.items():
            self.original[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def __exit__(self, exc_type, exc, tb) -> None:
        for key, value in self.original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
