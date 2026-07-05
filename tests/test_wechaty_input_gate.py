from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from mahjong_agent_runtime import (
    AgentContextBuilder,
    AgentRuntime,
    InMemoryAgentStore,
    StaticAgentClient,
    ToolGateway,
    UserMessage,
)


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "scripts" / "agent_runtime_app.py"
spec = importlib.util.spec_from_file_location("agent_runtime_app_for_test", APP_PATH)
assert spec is not None and spec.loader is not None
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


def test_parse_wechaty_input_gate_response_routes_operational_message() -> None:
    decision, errors = app.parse_wechaty_input_gate_response(
        json.dumps(
            {
                "should_route": True,
                "category": "operational",
                "confidence": 0.92,
                "reasoning_summary": "用户在问 0.5 的局。",
                "evidence": ["有没有 0.5"],
            },
            ensure_ascii=False,
        )
    )

    assert errors == []
    assert decision["should_route"] is True
    assert decision["category"] == "operational"


def test_parse_wechaty_input_gate_response_blocks_casual_chat() -> None:
    decision, errors = app.parse_wechaty_input_gate_response(
        json.dumps(
            {
                "should_route": False,
                "category": "casual_chat",
                "confidence": 0.88,
                "reasoning_summary": "朋友日常闲聊，不涉及麻将馆运营。",
                "evidence": ["晚上吃什么"],
            },
            ensure_ascii=False,
        )
    )

    assert errors == []
    assert decision["should_route"] is False
    assert decision["category"] == "casual_chat"


def test_run_wechaty_input_gate_fails_closed_on_invalid_contract(monkeypatch) -> None:
    monkeypatch.delenv("MAHJONG_WECHATY_INPUT_GATE_LLM_MODEL", raising=False)
    monkeypatch.setenv("MAHJONG_WECHATY_INPUT_GATE_ENABLED", "true")
    monkeypatch.setenv("MAHJONG_WECHATY_INPUT_GATE_FAIL_OPEN", "false")
    runtime = AgentRuntime(llm_client=StaticAgentClient(outputs=["不是 JSON"]))
    message = UserMessage(
        conversation_id="wechaty:contact:test",
        sender_id="friend",
        sender_name="朋友",
        text="晚上吃啥",
        message_id="msg_gate_invalid",
    )

    decision = app.run_wechaty_input_gate(message, trace_id="trace_gate_invalid", runtime=runtime)

    assert decision["enabled"] is True
    assert decision["should_route"] is False
    assert decision["errors"]


def test_build_wechaty_user_message_preserves_quoted_message(monkeypatch) -> None:
    monkeypatch.setenv("MAHJONG_WECHATY_ROUTE_SCOPE", "all")
    message, audit = app.build_wechaty_user_message(
        {
            "conversation_id": "wechaty:contact:friend",
            "sender_id": "friend",
            "sender_name": "朋友",
            "message_id": "msg_reply",
            "text": "可以",
            "self_message": False,
            "quoted_message": {
                "message_id": "msg_invite",
                "sender_id": "boss",
                "sender_name": "老板",
                "text": "14:00，0.5无烟，打吗？",
                "business_ref_type": "outbound_message_draft",
                "business_ref_id": "draft_001",
                "metadata": {"source": "wechaty"},
            },
        }
    )

    assert message is not None
    assert message.quoted_message is not None
    assert message.quoted_message.message_id == "msg_invite"
    assert message.quoted_message.text == "14:00，0.5无烟，打吗？"
    assert message.quoted_message.business_ref_type == "outbound_message_draft"
    assert message.quoted_message.business_ref_id == "draft_001"
    assert audit["quoted_message"]["message_id"] == "msg_invite"


def test_build_wechaty_user_message_sanitizes_quoted_message_metadata(monkeypatch) -> None:
    monkeypatch.setenv("MAHJONG_WECHATY_ROUTE_SCOPE", "all")
    message, audit = app.build_wechaty_user_message(
        {
            "conversation_id": "wechaty:contact:friend",
            "sender_id": "friend",
            "sender_name": "朋友",
            "message_id": "msg_reply_quote_metadata",
            "text": "可以",
            "self_message": False,
            "quoted_message": {
                "message_id": "msg_invite_with_private_metadata",
                "sender_id": "boss",
                "sender_name": "老板",
                "text": "14:00，0.5无烟，打吗？",
                "metadata": {
                    "source": "wechaty_quote",
                    "raw_chatusr": "friend_wechat_id",
                    "raw_payload": {"secret": "not-for-model"},
                    "private_note": "老板备注不该进入模型上下文",
                    "relationship_hint": "内部关系判断不该透给模型",
                },
            },
        }
    )

    assert message is not None
    assert message.quoted_message is not None
    assert message.quoted_message.metadata == {
        "source": "wechaty_quote",
        "raw_chatusr": "friend_wechat_id",
    }
    assert audit["quoted_message"]["metadata"] == {
        "source": "wechaty_quote",
        "raw_chatusr": "friend_wechat_id",
    }


def test_build_wechaty_user_message_routes_transcribed_voice_with_metadata(monkeypatch) -> None:
    monkeypatch.setenv("MAHJONG_WECHATY_ROUTE_SCOPE", "all")
    message, audit = app.build_wechaty_user_message(
        {
            "conversation_id": "wechaty:contact:voice_user",
            "sender_id": "voice_user",
            "sender_name": "语音客",
            "message_id": "voice_msg_001",
            "message_type": 2,
            "self_message": False,
            "metadata": {"audio_transcript": "晚上十点杭麻财敲有人吗，我一个人，打五毛"},
            "raw_observation": {
                "media_candidates": [
                    {"path": "$.payload.fileBox", "kind": "audio", "value": "voice_msg_001.silk"}
                ]
            },
        }
    )

    assert message is not None
    assert message.text == "晚上十点杭麻财敲有人吗，我一个人，打五毛"
    assert message.metadata["text_source"] == "audio_transcript"
    assert "voice" in message.metadata["modalities"]
    assert message.metadata["media_requires_transcription"] is False
    assert audit["text_source"] == "audio_transcript"
    assert audit["metadata"]["media_candidates"][0]["kind"] == "voice"


def test_build_wechaty_user_message_sanitizes_metadata_before_context(monkeypatch) -> None:
    monkeypatch.setenv("MAHJONG_WECHATY_ROUTE_SCOPE", "all")
    message, audit = app.build_wechaty_user_message(
        {
            "conversation_id": "wechaty:contact:friend",
            "sender_id": "friend",
            "sender_name": "朋友",
            "message_id": "msg_metadata_boundary",
            "message_type": 7,
            "text": "现在有人打牌吗",
            "self_message": False,
            "metadata": {
                "source": "wechaty_bridge",
                "raw_provider_payload": {"secret": "do-not-send-to-model"},
                "audio_transcript": "这个字段已经用于 text，不再保留到 metadata",
                "private_note": "老板备注不该进入模型上下文",
                "modalities": ["text", "voice"],
                "raw_observation_summary": {"quote_candidate_count": "2", "ignored": "x"},
            },
        }
    )

    assert message is not None
    assert message.metadata["source"] == "wechaty_bridge"
    assert message.metadata["channel"] == "wechaty"
    assert message.metadata["modalities"] == ["text"]
    assert message.metadata["raw_observation_summary"] == {"quote_candidate_count": 0, "media_candidate_count": 0}
    assert "raw_provider_payload" not in message.metadata
    assert "private_note" not in message.metadata
    assert "audio_transcript" not in message.metadata
    assert "raw_provider_payload" not in audit["metadata"]


def test_build_api_user_message_sanitizes_metadata() -> None:
    message, missing = app.build_api_user_message(
        {
            "conversation_id": "api_conv",
            "sender_id": "api_user",
            "sender_name": "接口用户",
            "text": "今晚有局吗",
            "metadata": {
                "channel": "manual_console",
                "source_message_id": "src_001",
                "raw_payload": {"secret": "not-for-model"},
                "private_note": "不要进上下文",
                "modalities": ["text", "image"],
                "media_candidates": [
                    {
                        "path": "$.raw_observation.media_candidates[0]",
                        "kind": "image",
                        "value_type": "str",
                        "value": "large-raw-value",
                        "text_preview": "截图 OCR 摘要",
                    }
                ],
            },
        }
    )

    assert missing == []
    assert message is not None
    assert message.metadata["channel"] == "manual_console"
    assert message.metadata["source_message_id"] == "src_001"
    assert message.metadata["modalities"] == ["text", "image"]
    assert message.metadata["media_candidates"] == [
        {
            "path": "$.raw_observation.media_candidates[0]",
            "kind": "image",
            "value_type": "str",
            "text_preview": "截图 OCR 摘要",
        }
    ]
    assert "raw_payload" not in message.metadata
    assert "private_note" not in message.metadata


def test_build_wechaty_user_message_blocks_untranscribed_media_with_audit(monkeypatch) -> None:
    monkeypatch.setenv("MAHJONG_WECHATY_ROUTE_SCOPE", "all")
    message, audit = app.build_wechaty_user_message(
        {
            "conversation_id": "wechaty:contact:voice_user",
            "sender_id": "voice_user",
            "sender_name": "语音客",
            "message_id": "voice_msg_002",
            "message_type": 2,
            "self_message": False,
            "raw_observation": {
                "media_candidates": [
                    {"path": "$.payload.fileBox", "kind": "audio", "value": "voice_msg_002.silk"}
                ]
            },
        }
    )

    assert message is None
    assert audit["reason"] == "non_text_without_transcript_or_ocr"
    assert "voice" in audit["modalities"]
    assert audit["media_requires_transcription"] is True
    assert audit["metadata"]["media_candidates"][0]["value_type"] == "str"


def test_build_wechaty_user_message_uses_raw_observation_quote_candidate(monkeypatch) -> None:
    monkeypatch.setenv("MAHJONG_WECHATY_ROUTE_SCOPE", "all")
    message, audit = app.build_wechaty_user_message(
        {
            "conversation_id": "wechaty:contact:friend",
            "sender_id": "friend",
            "sender_name": "朋友",
            "message_id": "msg_reply_from_observation",
            "text": "可以",
            "self_message": False,
            "raw_observation": {
                "quote_candidates": [
                    {
                        "path": "payload.referMsg",
                        "value": {
                            "msgId": "wechat_invite_msg_001",
                            "content": "14:00，0.5无烟，打吗？",
                            "senderId": "boss",
                            "senderName": "老板",
                        },
                    }
                ],
                "media_candidates": [],
            },
        }
    )

    assert message is not None
    assert message.quoted_message is not None
    assert message.quoted_message.message_id == "wechat_invite_msg_001"
    assert message.quoted_message.text == "14:00，0.5无烟，打吗？"
    assert message.quoted_message.sender_id == "boss"
    assert message.quoted_message.sender_name == "老板"
    assert audit["quoted_message"]["message_id"] == "wechat_invite_msg_001"


def test_build_wechaty_user_message_extracts_display_quote_text(monkeypatch) -> None:
    monkeypatch.setenv("MAHJONG_WECHATY_ROUTE_SCOPE", "all")
    message, audit = app.build_wechaty_user_message(
        {
            "conversation_id": "wechaty:room:test",
            "sender_id": "friend",
            "sender_name": "朋友",
            "message_id": "msg_display_quote_reply",
            "text": "「超大牌西斗门店（10点-23点）：财敲1    371   八点半开」\n- - - - - - - - - - - - - - -\n人齐",
            "self_message": False,
            "payload": {
                "id": "msg_display_quote_reply",
                "type": 7,
            },
        }
    )

    assert message is not None
    assert message.text == "人齐"
    assert message.quoted_message is not None
    assert message.quoted_message.message_id.startswith("display_quote_")
    assert message.quoted_message.text == "超大牌西斗门店（10点-23点）：财敲1    371   八点半开"
    assert message.quoted_message.metadata == {"source": "wechat_display_quote"}
    assert audit["text"] == "人齐"
    assert audit["quoted_message"]["metadata"] == {"source": "wechat_display_quote"}


def test_build_wechaty_user_message_does_not_extract_display_quote_without_separator(monkeypatch) -> None:
    monkeypatch.setenv("MAHJONG_WECHATY_ROUTE_SCOPE", "all")
    raw_text = "「不是引用，只是普通引号」\n继续说一句"

    message, audit = app.build_wechaty_user_message(
        {
            "conversation_id": "wechaty:room:test",
            "sender_id": "friend",
            "sender_name": "朋友",
            "message_id": "msg_not_display_quote",
            "text": raw_text,
            "self_message": False,
        }
    )

    assert message is not None
    assert message.text == raw_text
    assert message.quoted_message is None
    assert audit["quoted_message"] is None


def test_build_wechaty_user_message_extracts_refermsg_xml_quote(monkeypatch) -> None:
    monkeypatch.setenv("MAHJONG_WECHATY_ROUTE_SCOPE", "all")
    xml = """
    <msg>
      <appmsg>
        <type>57</type>
        <title>可以</title>
        <refermsg>
          <type>1</type>
          <svrid>wechat_invite_msg_xml_001</svrid>
          <fromusr>boss_wechat_id</fromusr>
          <chatusr>friend_wechat_id</chatusr>
          <displayname>老板</displayname>
          <content>14:00，0.5无烟，打吗？</content>
        </refermsg>
      </appmsg>
    </msg>
    """

    message, audit = app.build_wechaty_user_message(
        {
            "conversation_id": "wechaty:contact:friend",
            "sender_id": "friend",
            "sender_name": "朋友",
            "message_id": "msg_reply_with_xml_quote",
            "text": "可以",
            "self_message": False,
            "payload": {
                "id": "msg_reply_with_xml_quote",
                "type": 7,
                "text": xml,
            },
        }
    )

    assert message is not None
    assert message.quoted_message is not None
    assert message.quoted_message.message_id == "wechat_invite_msg_xml_001"
    assert message.quoted_message.text == "14:00，0.5无烟，打吗？"
    assert message.quoted_message.sender_id == "boss_wechat_id"
    assert message.quoted_message.sender_name == "老板"
    assert message.quoted_message.conversation_id is None
    assert message.quoted_message.metadata == {
        "source": "wechat_refermsg_xml",
        "raw_chatusr": "friend_wechat_id",
    }
    assert audit["quoted_message"]["message_id"] == "wechat_invite_msg_xml_001"


def test_build_wechaty_user_message_extracts_refermsg_xml_quote_without_chat_user(monkeypatch) -> None:
    monkeypatch.setenv("MAHJONG_WECHATY_ROUTE_SCOPE", "all")
    xml = """
    <msg>
      <appmsg>
        <type>57</type>
        <refermsg>
          <svrid>wechat_invite_msg_xml_no_chat_user</svrid>
          <fromusr>boss_wechat_id</fromusr>
          <displayname>老板</displayname>
          <content>14:00，0.5无烟，打吗？</content>
        </refermsg>
      </appmsg>
    </msg>
    """

    message, audit = app.build_wechaty_user_message(
        {
            "conversation_id": "wechaty:contact:friend",
            "sender_id": "friend",
            "sender_name": "朋友",
            "message_id": "msg_reply_with_xml_quote_no_chat_user",
            "text": "可以",
            "self_message": False,
            "payload": {
                "id": "msg_reply_with_xml_quote_no_chat_user",
                "type": 7,
                "text": xml,
            },
        }
    )

    assert message is not None
    assert message.quoted_message is not None
    assert message.quoted_message.message_id == "wechat_invite_msg_xml_no_chat_user"
    assert message.quoted_message.text == "14:00，0.5无烟，打吗？"
    assert message.quoted_message.sender_id == "boss_wechat_id"
    assert message.quoted_message.sender_name == "老板"
    assert message.quoted_message.conversation_id is None
    assert message.quoted_message.metadata == {"source": "wechat_refermsg_xml"}
    assert audit["quoted_message"]["message_id"] == "wechat_invite_msg_xml_no_chat_user"


def test_build_wechaty_user_message_keeps_runtime_conversation_id_from_refermsg_xml(monkeypatch) -> None:
    monkeypatch.setenv("MAHJONG_WECHATY_ROUTE_SCOPE", "all")
    xml = """
    <msg>
      <appmsg>
        <refermsg>
          <svrid>wechat_invite_msg_xml_002</svrid>
          <chatusr>wechaty:contact:friend</chatusr>
          <content>14:00，0.5无烟，打吗？</content>
        </refermsg>
      </appmsg>
    </msg>
    """

    message, _ = app.build_wechaty_user_message(
        {
            "conversation_id": "wechaty:contact:friend",
            "sender_id": "friend",
            "sender_name": "朋友",
            "message_id": "msg_reply_with_normalized_xml_quote",
            "text": "可以",
            "self_message": False,
            "payload": {"id": "msg_reply_with_normalized_xml_quote", "text": xml},
        }
    )

    assert message is not None
    assert message.quoted_message is not None
    assert message.quoted_message.conversation_id == "wechaty:contact:friend"
    assert message.quoted_message.metadata == {
        "source": "wechat_refermsg_xml",
        "raw_chatusr": "wechaty:contact:friend",
    }


def test_build_wechaty_user_message_uses_raw_observation_refermsg_xml_candidate(monkeypatch) -> None:
    monkeypatch.setenv("MAHJONG_WECHATY_ROUTE_SCOPE", "all")
    xml = """
    <msg>
      <appmsg>
        <refermsg>
          <svrid>wechat_invite_msg_xml_003</svrid>
          <fromusr>boss_wechat_id</fromusr>
          <chatusr>friend_wechat_id</chatusr>
          <displayname>老板</displayname>
          <content>15:00，1块有烟，打吗？</content>
        </refermsg>
      </appmsg>
    </msg>
    """

    message, audit = app.build_wechaty_user_message(
        {
            "conversation_id": "wechaty:contact:friend",
            "sender_id": "friend",
            "sender_name": "朋友",
            "message_id": "msg_reply_with_observed_xml_quote",
            "text": "可以",
            "self_message": False,
            "raw_observation": {
                "quote_candidates": [
                    {
                        "path": "$.payload.text",
                        "value": xml,
                    }
                ]
            },
        }
    )

    assert message is not None
    assert message.quoted_message is not None
    assert message.quoted_message.message_id == "wechat_invite_msg_xml_003"
    assert message.quoted_message.text == "15:00，1块有烟，打吗？"
    assert message.quoted_message.conversation_id is None
    assert message.quoted_message.metadata["raw_chatusr"] == "friend_wechat_id"
    assert audit["quoted_message"]["message_id"] == "wechat_invite_msg_xml_003"


def test_refermsg_raw_chatusr_quote_resolves_in_current_runtime_conversation(monkeypatch) -> None:
    monkeypatch.setenv("MAHJONG_WECHATY_ROUTE_SCOPE", "all")
    store = InMemoryAgentStore()
    drafts, _ = store.create_outbound_message_drafts(
        conversation_id="wechaty:contact:friend",
        trace_id="trace_wechat_refermsg_resolution_seed",
        drafts=[
            {
                "recipient_id": "friend",
                "recipient_name": "朋友",
                "channel": "wechaty",
                "message_text": "14:00，0.5无烟，打吗？",
                "purpose": "invite_candidate",
            }
        ],
    )
    store.link_message_reference(
        conversation_id="wechaty:contact:friend",
        message_id="wechat_invite_msg_xml_004",
        source_message_id=drafts[0].draft_id,
        channel="wechaty",
        text=drafts[0].message_text,
        metadata={"source": "wechaty_outbound_echo"},
    )
    xml = """
    <msg>
      <appmsg>
        <refermsg>
          <svrid>wechat_invite_msg_xml_004</svrid>
          <chatusr>friend_wechat_id</chatusr>
          <content>14:00，0.5无烟，打吗？</content>
        </refermsg>
      </appmsg>
    </msg>
    """

    message, _ = app.build_wechaty_user_message(
        {
            "conversation_id": "wechaty:contact:friend",
            "sender_id": "friend",
            "sender_name": "朋友",
            "message_id": "msg_reply_with_xml_quote_resolution",
            "text": "可以",
            "self_message": False,
            "payload": {"id": "msg_reply_with_xml_quote_resolution", "text": xml},
        }
    )
    assert message is not None
    assert message.quoted_message is not None
    assert message.quoted_message.conversation_id is None

    built = AgentContextBuilder(store=store, tool_gateway=ToolGateway(store)).build(
        message,
        trace_id="trace_wechat_refermsg_resolution",
    )

    assert built.payload["quoted_message_context"]["business_ref_type"] == "outbound_message_draft"
    assert built.payload["quoted_message_context"]["business_ref_id"] == drafts[0].draft_id
    assert built.payload["current_message"]["quoted_message"]["metadata"]["raw_chatusr"] == "friend_wechat_id"
    assert built.payload["context_budget"]["quoted_message_reference_resolved"] is True


def test_build_wechaty_user_message_does_not_treat_generic_reply_candidate_as_quote(monkeypatch) -> None:
    monkeypatch.setenv("MAHJONG_WECHATY_ROUTE_SCOPE", "all")
    message, audit = app.build_wechaty_user_message(
        {
            "conversation_id": "wechaty:contact:friend",
            "sender_id": "friend",
            "sender_name": "朋友",
            "message_id": "msg_generic_reply_candidate",
            "text": "可以",
            "self_message": False,
            "raw_observation": {
                "quote_candidates": [
                    {
                        "path": "payload.replyMeta",
                        "value": {
                            "id": "not_a_quoted_message",
                            "content": "这是普通回复元数据，不是引用消息。",
                        },
                    }
                ]
            },
        }
    )

    assert message is not None
    assert message.quoted_message is None
    assert audit["quoted_message"] is None


def test_link_delivered_message_reference_maps_platform_message_to_business_anchor() -> None:
    store = InMemoryAgentStore()
    drafts, _ = store.create_outbound_message_drafts(
        conversation_id="owner_conversation",
        trace_id="trace_link_delivered_seed",
        drafts=[
            {
                "recipient_id": "wang",
                "recipient_name": "王哥",
                "channel": "wechaty",
                "message_text": "七点三缺一，打吗？",
                "purpose": "offer_existing_game",
            }
        ],
    )
    runtime = AgentRuntime(llm_client=StaticAgentClient(outputs=[]), store=store)

    result = app.link_delivered_message_reference(
        runtime,
        {
            "conversation_id": "wechaty:contact:wang",
            "platform_message_id": "wechat_platform_msg_001",
            "source_message_id": drafts[0].draft_id,
            "channel": "wechaty",
            "text": "七点三缺一，打吗？",
        },
    )

    assert result["ok"] is True
    reference = store.resolve_message_reference(
        conversation_id="wechaty:contact:wang",
        message_id="wechat_platform_msg_001",
    )
    assert reference is not None
    assert reference.business_ref_type == "outbound_message_draft"
    assert reference.business_ref_id == drafts[0].draft_id
    assert reference.metadata["platform_message_id"] == "wechat_platform_msg_001"


def test_run_wechaty_input_gate_uses_recent_context_for_short_answer(monkeypatch) -> None:
    monkeypatch.delenv("MAHJONG_WECHATY_INPUT_GATE_LLM_MODEL", raising=False)
    monkeypatch.setenv("MAHJONG_WECHATY_INPUT_GATE_ENABLED", "true")
    runtime = AgentRuntime(
        llm_client=StaticAgentClient(
            outputs=[
                json.dumps(
                    {
                        "should_route": True,
                        "category": "followup_answer",
                        "confidence": 0.91,
                        "reasoning_summary": "上一轮在问是否要组局，用户短答可以。",
                        "evidence": ["recent_conversation 中有麻将运营追问", "当前消息是可以"],
                    },
                    ensure_ascii=False,
                )
            ]
        )
    )
    runtime.store.append_assistant_turn(
        "wechaty:contact:test",
        "现在没有，要组一个吗？",
        "trace_prev",
        metadata={"delivery_status": "sent"},
    )
    message = UserMessage(
        conversation_id="wechaty:contact:test",
        sender_id="friend",
        sender_name="朋友",
        text="可以",
        message_id="msg_gate_followup",
    )

    decision = app.run_wechaty_input_gate(message, trace_id="trace_gate_followup", runtime=runtime)

    assert decision["should_route"] is True
    assert decision["category"] == "followup_answer"


def test_wechaty_casual_chat_prompt_forbids_repeating_system_identity_terms() -> None:
    prompt = (ROOT / "src" / "mahjong_agent_runtime" / "prompts" / "wechaty_casual_chat_reply.md").read_text(
        encoding="utf-8"
    )

    assert "即使用户原文里出现 AI" in prompt
    assert "输出前逐字检查 `reply_to_user`" in prompt
    assert "如果包含，必须重写" in prompt
    assert "用“这个”“这种事”“这类事”带过" in prompt
    assert "用“这个”“这种事”“工具”这类模糊说法带过" not in prompt
    assert "不能解释实现方式、消息通道、工具、trace、日志、数据库、prompt、审查、预算" in prompt
    assert "不要回：“要是真有AI能帮我组局就好了”" in prompt
    assert "闲聊回复不能顺手提当前局、可选局或任何组局进展" in prompt
    assert "不要回：“打牌直接说就行。七点三缺一，打吗？”" in prompt
    assert "不要回：“这个先不聊，打牌你直接说就行。”" in prompt
    assert "哈哈，组局确实挺费脑子的，条件太多了。" in prompt


def test_wechaty_input_gate_prompt_defines_multimodal_boundary() -> None:
    prompt = (ROOT / "src" / "mahjong_agent_runtime" / "prompts" / "wechaty_input_gate.md").read_text(
        encoding="utf-8"
    )

    assert "`current_message.metadata.modalities`" in prompt
    assert "`text_source`" in prompt
    assert "只有存在可读文本或可信转写/OCR" in prompt
    assert "不要猜里面说了什么" in prompt


def test_handle_wechaty_casual_chat_reviews_reply_before_return(monkeypatch) -> None:
    monkeypatch.setenv("MAHJONG_WECHATY_CASUAL_CHAT_REPLY_ENABLED", "true")
    client = StaticAgentClient(
        outputs=[
            json.dumps(
                {
                    "should_reply": True,
                    "reply_to_user": "收到，我说人话点。",
                    "reasoning_summary": "用户在反馈话术太像机器，短句承接即可。",
                    "needs_human": False,
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "approved": True,
                    "needs_human": False,
                    "reasoning_summary": "没有泄露系统或其他用户信息。",
                    "violations": [],
                    "item_reviews": [
                        {
                            "item_id": "casual_chat.reply_to_user",
                            "approved": True,
                            "suggested_safe_text": "收到，我说人话点。",
                            "reasoning_summary": "安全。",
                            "violations": [],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, reply_self_review_enabled=True)
    message = UserMessage(
        conversation_id="wechaty:contact:friend",
        sender_id="friend",
        sender_name="朋友",
        text="你说话人性化点",
        message_id="msg_casual_001",
    )

    result = app.handle_wechaty_casual_chat(
        message,
        trace_id="trace_casual_001",
        runtime=runtime,
        gate_decision={
            "should_route": False,
            "category": "casual_chat",
            "confidence": 0.93,
            "reasoning_summary": "闲聊反馈，不进入麻将主流程。",
            "evidence": ["人性化点"],
        },
    )

    assert result.final_reply == "收到，我说人话点。"
    assert len(client.calls) == 2
    assert result.tool_results[0].name == "customer_visible_content_review"
    steps = [event.step for event in runtime.trace_recorder.get_trace("trace_casual_001")]
    assert "wechaty_casual_chat_prompt" in steps
    assert "customer_visible_content_review_prompt" in steps
    assert "final_output" in steps
    turns = runtime.store.recent_turns("wechaty:contact:friend", 5)
    assert turns[0].content == "你说话人性化点"
    assert turns[-1].content == "收到，我说人话点。"


def test_wechaty_casual_chat_uses_review_safe_rewrite(monkeypatch) -> None:
    monkeypatch.setenv("MAHJONG_WECHATY_CASUAL_CHAT_REPLY_ENABLED", "true")
    client = StaticAgentClient(
        outputs=[
            json.dumps(
                {
                    "should_reply": True,
                    "reply_to_user": "我是智能助手，不能透露系统信息。",
                    "reasoning_summary": "模型错误暴露身份。",
                    "needs_human": False,
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "approved": False,
                    "needs_human": False,
                    "reasoning_summary": "原文暴露智能助手身份，已改写。",
                    "violations": ["leaks_agent_identity"],
                    "item_reviews": [
                        {
                            "item_id": "casual_chat.reply_to_user",
                            "approved": False,
                            "suggested_safe_text": "想打啥直接说就行。",
                            "reasoning_summary": "删除身份信息。",
                            "violations": ["leaks_agent_identity"],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, reply_self_review_enabled=True)
    message = UserMessage(
        conversation_id="wechaty:contact:friend",
        sender_id="friend",
        sender_name="朋友",
        text="你是机器人吗",
        message_id="msg_casual_002",
    )

    result = app.handle_wechaty_casual_chat(
        message,
        trace_id="trace_casual_002",
        runtime=runtime,
        gate_decision={"should_route": False, "category": "non_mahjong", "confidence": 0.9},
    )

    assert result.final_reply == "想打啥直接说就行。"
    assert result.tool_results[0].result["approved"] is False
    steps = [event.step for event in runtime.trace_recorder.get_trace("trace_casual_002")]
    assert steps[-1] == "final_output"


def test_route_wechaty_casual_chat_returns_reviewed_agent_result(monkeypatch) -> None:
    monkeypatch.setenv("MAHJONG_WECHATY_CASUAL_CHAT_REPLY_ENABLED", "true")
    monkeypatch.setenv("MAHJONG_WECHATY_INPUT_GATE_ENABLED", "true")
    monkeypatch.setenv("MAHJONG_WECHATY_INPUT_GATE_FAIL_OPEN", "false")
    client = StaticAgentClient(
        outputs=[
            json.dumps(
                {
                    "should_route": False,
                    "category": "casual_chat",
                    "confidence": 0.95,
                    "reasoning_summary": "日常闲聊。",
                    "evidence": ["晚上吃啥"],
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "should_reply": True,
                    "reply_to_user": "还没想好，晚点看。",
                    "reasoning_summary": "日常闲聊可以短句回复。",
                    "needs_human": False,
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "approved": True,
                    "needs_human": False,
                    "reasoning_summary": "没有泄露。",
                    "violations": [],
                    "item_reviews": [
                        {
                            "item_id": "casual_chat.reply_to_user",
                            "approved": True,
                            "suggested_safe_text": "还没想好，晚点看。",
                            "reasoning_summary": "安全。",
                            "violations": [],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, reply_self_review_enabled=True)
    previous_runtime = app.RUNTIME
    app.RUNTIME = runtime
    try:
        routed = app.route_wechaty_raw_to_agent(
            {
                "conversation_id": "wechaty:contact:friend",
                "sender_id": "friend",
                "sender_name": "朋友",
                "message_id": "msg_route_casual",
                "text": "晚上吃啥",
                "self_message": True,
            },
            trace_id="trace_route_casual",
        )
    finally:
        app.RUNTIME = previous_runtime

    assert routed["routed_to_agent"] is False
    assert routed["audit"]["reason"] == "wechaty_input_gate_routed_to_casual_chat"
    assert routed["agent_result"]["final_reply"] == "还没想好，晚点看。"
    assert routed["casual_chat_result"]["final_reply"] == "还没想好，晚点看。"


def test_quoted_casual_chat_reply_does_not_enter_operational_runtime(monkeypatch) -> None:
    monkeypatch.setenv("MAHJONG_WECHATY_CASUAL_CHAT_REPLY_ENABLED", "true")
    monkeypatch.setenv("MAHJONG_WECHATY_INPUT_GATE_ENABLED", "true")
    monkeypatch.setenv("MAHJONG_WECHATY_INPUT_GATE_FAIL_OPEN", "false")
    client = StaticAgentClient(
        outputs=[
            json.dumps(
                {
                    "should_route": False,
                    "category": "casual_chat",
                    "confidence": 0.94,
                    "reasoning_summary": "用户引用的是前面的闲聊反馈并回复哈哈，不是麻将组局动作。",
                    "evidence": ["quoted_message.text=真人回复跟AI回复还是有区别的", "current_message.text=哈哈哈"],
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "should_reply": True,
                    "reply_to_user": "哈哈哈，懂你意思。",
                    "reasoning_summary": "闲聊承接，不改变局状态。",
                    "needs_human": False,
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "approved": True,
                    "needs_human": False,
                    "reasoning_summary": "没有泄露系统或其他用户信息。",
                    "violations": [],
                    "item_reviews": [
                        {
                            "item_id": "casual_chat.reply_to_user",
                            "approved": True,
                            "suggested_safe_text": "哈哈哈，懂你意思。",
                            "reasoning_summary": "安全。",
                            "violations": [],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, reply_self_review_enabled=True)
    runtime.store.create_game(
        conversation_id="wechaty:contact:friend",
        organizer_id="friend",
        organizer_name="朋友",
        requirement={
            "game_type": "hangzhou_mahjong",
            "stake": "0.5",
            "smoke_preference": "no_smoke",
            "start_time_kind": "scheduled",
            "start_time": "19:00",
            "needed_seats": 2,
        },
        known_players=[{"customer_id": "friend", "display_name": "朋友"}],
        trace_id="trace_quote_casual_seed",
    )
    previous_runtime = app.RUNTIME
    app.RUNTIME = runtime
    try:
        routed = app.route_wechaty_raw_to_agent(
            {
                "conversation_id": "wechaty:contact:friend",
                "sender_id": "friend",
                "sender_name": "朋友",
                "message_id": "msg_quote_casual_reply",
                "text": "哈哈哈",
                "self_message": True,
                "quoted_message": {
                    "message_id": "msg_ai_chitchat",
                    "sender_id": "friend",
                    "sender_name": "朋友",
                    "text": "真人回复跟AI回复还是有区别的",
                },
            },
            trace_id="trace_quote_casual_reply",
        )
    finally:
        app.RUNTIME = previous_runtime

    assert routed["routed_to_agent"] is False
    assert routed["audit"]["reason"] == "wechaty_input_gate_routed_to_casual_chat"
    assert routed["audit"]["input_gate"]["category"] == "casual_chat"
    assert routed["agent_result"]["final_reply"] == "哈哈哈，懂你意思。"
    assert len(runtime.store.active_games("wechaty:contact:friend")) == 1
    steps = [event.step for event in runtime.trace_recorder.get_trace("trace_quote_casual_reply")]
    assert "wechaty_raw_message_routed_to_agent" not in steps
    assert "tool_called" not in steps
    assert "wechaty_raw_message_routed_to_casual_chat" in steps
