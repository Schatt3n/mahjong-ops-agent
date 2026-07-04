from __future__ import annotations

import json
import importlib.util
import sys
import threading
import time
from pathlib import Path
from typing import Any

from mahjong_agent_runtime import (
    AgentRuntime,
    AgentContextBuilder,
    CustomerProfile,
    CustomerRelationship,
    InMemoryAgentStore,
    InMemoryTraceRecorder,
    JsonlTraceRecorder,
    QuotedMessageRef,
    SQLiteAgentStore,
    StaticAgentClient,
    ToolCall,
    ToolGateway,
    ToolResult,
    TokenBudget,
    UserMessage,
)
from mahjong_agent_runtime.runtime import message_idempotency_key
from mahjong_agent_runtime.store import normalize_requirement
from mahjong_agent_runtime.tracing import trace_steps, validate_trace


ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_SCRIPT = ROOT / "scripts" / "verify_agent_runtime_boundary.py"


def load_boundary_module():
    spec = importlib.util.spec_from_file_location("verify_agent_runtime_boundary_for_test", BOUNDARY_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runtime_main_chain_does_not_import_legacy_parser_workflow_or_guard() -> None:
    forbidden = [
        "mahjong_agent_v2",
        "from mahjong_agent.",
        "import mahjong_agent.",
        "reply_guard",
        "workflow",
        "semantic_resolver",
        "trial_",
        "responder",
    ]
    for path in (ROOT / "src" / "mahjong_agent_runtime").glob("**/*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path} contains forbidden legacy token {token!r}"


def test_runtime_boundary_script_rejects_semantic_patch_code(tmp_path) -> None:
    module = load_boundary_module()
    bad_file = tmp_path / "bad_runtime_semantic_patch.py"
    bad_file.write_text(
        "def patch(text):\n"
        "    return re.sub('0，5', '0.5', text)\n",
        encoding="utf-8",
    )

    violations = module.verify_files([bad_file])

    messages = "\n".join(violation.message for violation in violations)
    assert "semantic boundary violation" in messages
    assert "正则替换修麻将语义" in messages
    assert "0.5 口误 badcase" in messages


def test_runtime_system_prompt_requires_customer_visible_reply_self_check() -> None:
    prompt = (ROOT / "src" / "mahjong_agent_runtime" / "prompts" / "agent_runtime_system.md").read_text(encoding="utf-8")

    assert "客户可见内容自检" in prompt
    assert "麻将馆主流程准则" in prompt
    assert "每次准备输出 `reply_to_user` 或工具参数里的 `message_text` 前" in prompt
    assert "泄露系统信息" in prompt
    assert "泄露其他用户信息" in prompt
    assert "如果当前消息会改变局内事实" in prompt
    assert "必须先调用相应写工具记录事实" in prompt
    assert "`current_message.quoted_message` 表示用户本轮引用/回复的上一条消息" in prompt
    assert "`quoted_message_context` 是后端根据 messageId 解析出的业务锚点" in prompt
    assert "先把当前短句解释为对引用消息的回应" in prompt
    assert "business_ref_type/business_ref_id" in prompt
    assert "引用消息只是上下文锚点" in prompt
    assert "候选人名单" in prompt
    assert "待审批" in prompt
    assert "草稿" in prompt
    assert "如果自检不通过，必须在同一次输出中重写客户可见文本" in prompt
    assert "customer_visible_content_review" in prompt
    assert "才使用 `objective_status=needs_human`" in prompt
    assert "用户只是问“有没有局/现在有人吗/通宵有人吗/0.5有人吗/人齐开有没有”" in prompt
    assert "必须先调用 `search_current_games`" in prompt
    assert "三位数字中间是 `7`" in prompt
    assert "`173=1缺3`" in prompt
    assert "`272=2缺2`" in prompt
    assert "`371=3缺1`" in prompt
    assert "`216` 在川麻语境里按 `2-16` 归一化" in prompt
    assert "`232` 按 `2-32` 归一化" in prompt
    assert "`1-32` 表示 1 元底" in prompt
    assert "`10-32` 表示 10 元底" in prompt
    assert "结构化槽位里 `stake` 只表示底注" in prompt
    assert "`cap_score` 表示封顶" in prompt
    assert "`stake_label` 表示客户习惯说法" in prompt
    assert "人数结构短码本身已经回答了当前人数" in prompt
    assert "给发起客户设置 `seat_count=known_player_count`" in prompt
    assert "`0.5`、`0，5`、`0、5`、`0 5`" in prompt
    assert "默认地区是杭州" in prompt
    assert "时间 + 档位 + 人数短码/缺口 + 烟况" in prompt
    assert "如果没有匹配局，继续按组局目标调用 `create_game`" in prompt
    assert "如果有一个或多个匹配局，回复可选现成局" in prompt
    assert "最终只用“好/好的/我帮你看看/我帮你问问”这类短句承接" in prompt
    assert "高置信默认值" in prompt
    assert "95% 打 0.5" in prompt
    assert "不要机械追问“打多大/几个人”" in prompt
    assert "帮我约个 6.30 无烟的" in prompt
    assert "七点三缺一，可以不" in prompt
    assert "有明确时间词" in prompt
    assert "必须用 `start_time_kind=scheduled`" in prompt
    assert "不要只回复“留意/看看/帮你问问”就停止" in prompt
    assert "必须继续调用 `create_game`、`search_customers`" in prompt
    assert "必须简短，建议 30 字以内" in prompt
    assert "然后用候选人结果调用 `create_invite_drafts`" in prompt
    assert "只读工具结果里的 `result.requirement` 是刚刚实际执行的查询条件" in prompt
    assert "`search_current_games` 的每个匹配结果会带 `join_projection`" in prompt
    assert "不要上一轮按固定时间查询，下一轮建局时改成人齐开" in prompt
    assert "后端会做跨工具参数一致性校验" in prompt
    assert "一个联系人可能代表多个座位" in prompt
    assert "给候选人的 `message_text` 只写候选人需要知道的公共条件" in prompt
    assert "不要写 `asap_when_full`" in prompt
    assert "`duration_kind=flexible` 表示“时长还没定/打多久还不确定”" in prompt
    assert "烟都可以，打多久还不确定，你想打多久呢" in prompt
    assert "不要用“时长灵活、烟不限、你看行不”这类系统化总结代替运营对话" in prompt
    assert "不要用客服腔或平台腔" in prompt
    assert "要加入吗/是否加入/要不要加入/要一起吗" in prompt
    assert "现成局或邀约优先说“打吗？”" in prompt
    assert "不要直接说“他是组这个局的人/发起人”" in prompt
    assert "公开可见的微信昵称或对方本来能看到的群昵称" in prompt
    assert "不能给老板自己的私有微信备注" in prompt
    assert "5小时不行/我不打了/退群了" in prompt
    assert "第一步必须输出 `objective_status=needs_tool`" in prompt
    assert "调用 `record_candidate_reply` 记录该客户对当前局的 `declined`" in prompt
    assert "客户可见回复不要再带问号" in prompt
    assert "不要继续问可接受的时长、时间或其他想法" in prompt
    assert "`search_current_games` 返回的 `game.requirement.user_visible_summary`" in prompt
    assert "不要把搜索条件、画像默认槽位或工具里的结构化字段展开" in prompt
    assert "找老板帮忙组局的发起客户/首位玩家" in prompt
    assert "发起客户找老板组局时，默认他本人要打" in prompt
    assert "优先传 `requesting_party.seat_count`" in prompt
    assert "后端会合并成统一 party/seat_claim" in prompt
    assert "算的，加上你两个，还差两个。" in prompt
    assert "existing_player_ids" in prompt


def test_runtime_context_includes_quoted_message_anchor() -> None:
    store = InMemoryAgentStore()
    gateway = ToolGateway(store=store)
    builder = AgentContextBuilder(store=store, tool_gateway=gateway)
    message = UserMessage(
        conversation_id="quote_case",
        sender_id="wang",
        sender_name="王哥",
        text="可以",
        message_id="msg_reply",
        quoted_message=QuotedMessageRef(
            message_id="msg_invite",
            sender_id="boss",
            sender_name="老板",
            text="14:00，0.5无烟，打吗？",
            conversation_id="quote_case",
            business_ref_type="outbound_message_draft",
            business_ref_id="draft_001",
            metadata={"channel": "wechaty"},
        ),
    )

    built = builder.build(message, trace_id="trace_quote_case")

    quoted = built.payload["current_message"]["quoted_message"]
    assert quoted == {
        "message_id": "msg_invite",
        "sender_id": "boss",
        "sender_name": "老板",
        "text": "14:00，0.5无烟，打吗？",
        "conversation_id": "quote_case",
        "business_ref_type": "outbound_message_draft",
        "business_ref_id": "draft_001",
        "metadata": {"channel": "wechaty"},
    }
    prompt_payload = json.loads(built.messages[1]["content"])
    assert prompt_payload["current_message"]["quoted_message"]["business_ref_id"] == "draft_001"
    assert prompt_payload["current_message"]["quoted_message"]["text"] == "14:00，0.5无烟，打吗？"


def test_runtime_context_resolves_quoted_message_business_reference() -> None:
    store = InMemoryAgentStore()
    gateway = ToolGateway(store=store)
    builder = AgentContextBuilder(store=store, tool_gateway=gateway)
    drafts, _ = store.create_outbound_message_drafts(
        conversation_id="quote_resolve",
        trace_id="trace_quote_resolve_seed",
        drafts=[
            {
                "recipient_id": "wang",
                "recipient_name": "王哥",
                "channel": "wechat",
                "message_text": "14:00，0.5无烟，打吗？",
                "purpose": "invite_candidate",
            }
        ],
    )

    built = builder.build(
        UserMessage(
            conversation_id="quote_resolve",
            sender_id="wang",
            sender_name="王哥",
            text="可以",
            message_id="msg_quote_resolve_reply",
            quoted_message=QuotedMessageRef(message_id=drafts[0].draft_id, text=""),
        ),
        trace_id="trace_quote_resolve",
    )

    quoted = built.payload["current_message"]["quoted_message"]
    assert quoted["message_id"] == drafts[0].draft_id
    assert quoted["business_ref_type"] == "outbound_message_draft"
    assert quoted["business_ref_id"] == drafts[0].draft_id
    assert quoted["text"] == "14:00，0.5无烟，打吗？"
    assert quoted["metadata"]["resolved_message_reference"]["recipient_id"] == "wang"
    assert built.payload["quoted_message_context"]["business_ref_type"] == "outbound_message_draft"
    assert built.payload["quoted_message_context"]["business_ref_id"] == drafts[0].draft_id
    assert built.payload["context_budget"]["quoted_message_reference_resolved"] is True


def test_sqlite_store_persists_message_references(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime.sqlite3"
    store = SQLiteAgentStore(db_path)
    drafts, _ = store.create_outbound_message_drafts(
        conversation_id="quote_persist",
        trace_id="trace_quote_persist_seed",
        drafts=[
            {
                "recipient_id": "wang",
                "recipient_name": "王哥",
                "channel": "wechat",
                "message_text": "七点三缺一，打吗？",
                "purpose": "offer_existing_game",
            }
        ],
    )

    reopened = SQLiteAgentStore(db_path)
    reference = reopened.resolve_message_reference(
        conversation_id="quote_persist",
        message_id=drafts[0].draft_id,
    )

    assert reference is not None
    assert reference.business_ref_type == "outbound_message_draft"
    assert reference.business_ref_id == drafts[0].draft_id
    assert reference.text == "七点三缺一，打吗？"
    assert reference.recipient_id == "wang"


def test_runtime_review_prompt_rejects_internal_enum_and_backend_workflow_leakage() -> None:
    prompt = (ROOT / "src" / "mahjong_agent_runtime" / "prompts" / "agent_runtime_reply_self_review.md").read_text(
        encoding="utf-8"
    )

    assert "`asap_when_full`" in prompt
    assert "`pending_approval`" in prompt
    assert "`hangzhou_mahjong`" in prompt
    assert "客户可见文本应改成自然中文" in prompt
    assert "时间或人齐开" in prompt
    assert "不要透露发起人是谁" in prompt
    assert "还缺几人" in prompt
    assert "微信昵称或群昵称" in prompt
    assert "老板私有微信备注" in prompt
    assert "用户问“某某是谁”不等于授权暴露这个人在当前局里的角色" in prompt
    assert "某某算不算人/他不打吗" in prompt
    assert "算的，加上你两个，还差两个" in prompt
    assert "leaks_participant_role" in prompt


def test_runtime_customer_visible_text_generation_prompt_defines_boss_tone_and_visibility_layers() -> None:
    prompt = (ROOT / "src" / "mahjong_agent_runtime" / "prompts" / "customer_visible_text_generation.md").read_text(
        encoding="utf-8"
    )

    assert "客户可见话术生成器" in prompt
    assert "语义保真改写器" in prompt
    assert "不做业务决策" in prompt
    assert "唯一可信事实来源是本轮输入里的 `items[].text`" in prompt
    assert "不补槽位，不查局，不查人，不判断谁确认" in prompt
    assert "不得新增或修改：人数、缺口、时间、档位、烟况、时长、玩法" in prompt
    assert "semantic_preserved" in prompt
    assert "不要把“有个1块有烟人齐开的局”改成“有个173”" in prompt
    assert "`stake=1`、`1`、`1.0` 在明显表示档位时说成“1块”" in prompt
    assert "把1改成1块" in prompt or "把 1 改成 1块" in prompt
    assert "默认不要在回复开头带客户姓名或微信备注" in prompt
    assert "公开微信昵称或群昵称" in prompt
    assert "老板私有备注" in prompt
    assert "候选邀约可以短到：“人齐开，1块，烟都可以，打吗？”" in prompt
    assert "现成局询问也用麻将馆口吻" in prompt
    assert "不要写“要加入吗/是否加入/要一起吗”" in prompt
    assert "只列原文里已经出现的事实" in prompt


def test_runtime_context_includes_sender_relationships_for_active_game() -> None:
    store = InMemoryAgentStore()
    store.upsert_customer(CustomerProfile(customer_id="zhang", display_name="张哥"))
    store.upsert_customer(CustomerProfile(customer_id="wang01", display_name="王哥"))
    store.upsert_customer_relationship(
        CustomerRelationship(
            customer_a_id="zhang",
            customer_b_id="wang01",
            played_together_count=0,
            notes="暂无共同打牌记录。",
        )
    )
    store.create_game(
        conversation_id="runtime_relationship_context",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"game_type": "hangzhou_mahjong", "stake": "0.5"},
        known_players=[{"customer_id": "zhang", "display_name": "张哥", "source": "organizer"}],
        trace_id="trace_relationship_context_seed",
    )

    built = AgentContextBuilder(store, ToolGateway(store)).build(
        UserMessage(
            conversation_id="runtime_relationship_context",
            sender_id="wang01",
            sender_name="王哥",
            text="张哥是谁",
            message_id="msg_relationship_context",
        ),
        trace_id="trace_relationship_context",
    )

    assert built.payload["sender_relationships"] == [
        {
            "customer_id": "zhang",
            "display_name": "张哥",
            "played_together_count": 0,
            "avoid_playing": False,
            "relationship_label": "no_prior_play_record",
            "notes": "暂无共同打牌记录。",
        }
    ]


def test_runtime_search_customers_avoids_known_pair_conflicts() -> None:
    store = seeded_store()
    store.upsert_customer_relationship(
        CustomerRelationship(
            customer_a_id="zhang",
            customer_b_id="ran",
            avoid_playing=True,
            notes="张哥不和冉姐同桌。",
        )
    )

    candidates = store.search_customers(
        {
            "game_type": "hangzhou_mahjong",
            "stake": "1",
            "smoke_preference": "any",
            "organizer_id": "zhang",
        },
        exclude_customer_ids=["zhang"],
        limit=10,
    )

    assert [item["customer"]["customer_id"] for item in candidates] == ["he"]


def test_runtime_sqlite_search_customers_avoids_known_pair_conflicts(tmp_path) -> None:
    store = seeded_store(SQLiteAgentStore(tmp_path / "agent_runtime_relationships.sqlite3"))
    store.upsert_customer_relationship(
        CustomerRelationship(
            customer_a_id="zhang",
            customer_b_id="ran",
            avoid_playing=True,
            notes="张哥不和冉姐同桌。",
        )
    )

    candidates = store.search_customers(
        {
            "game_type": "hangzhou_mahjong",
            "stake": "1",
            "smoke_preference": "any",
            "organizer_id": "zhang",
        },
        exclude_customer_ids=["zhang"],
        limit=10,
    )

    assert [item["customer"]["customer_id"] for item in candidates] == ["he"]


def test_runtime_boundary_script_rejects_legacy_analyze_endpoint_in_entrypoint(tmp_path, monkeypatch) -> None:
    module = load_boundary_module()
    bad_entrypoint = tmp_path / "run_agent_runtime_app.py"
    bad_entrypoint.write_text(
        "def route(parsed):\n"
        "    if parsed.path == '/api/analyze':\n"
        "        return 'legacy analyze'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "RUNTIME_ENTRYPOINTS", (bad_entrypoint,))

    violations = module.verify_files([bad_entrypoint])

    messages = "\n".join(violation.message for violation in violations)
    assert "entrypoint boundary violation" in messages
    assert "旧试用台 analyze 接口" in messages


def test_runtime_boundary_script_passes_current_main_chain() -> None:
    module = load_boundary_module()

    assert module.verify_files() == []


def test_runtime_default_eval_runner_only_targets_current_main_chain() -> None:
    runner = (ROOT / "scripts" / "run_evals.py").read_text(encoding="utf-8")
    assert "verify_agent_runtime_boundary.py" in runner
    assert "run_agent_runtime_eval.py" in runner
    assert "tests/test_agent_runtime.py" in runner
    assert "tests/test_real_owner_chat_golden.py" in runner
    assert "tests/test_agent_runtime_v3.py" not in runner
    assert "tests/test_agent_v3_app.py" not in runner
    assert "run_agent_runtime_v3_eval.py" not in runner
    assert "verify_agent_runtime_v3_boundary.py" not in runner
    assert "verify_agent_runtime_v2_boundary.py" not in runner
    assert "run_agent_runtime_v2_eval.py" not in runner
    assert "run_controlled_workflow_eval.py" not in runner
    assert "run_scenario_eval.py" not in runner


def test_runtime_lets_model_drive_tool_sequence_until_final_reply() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = PlanningClient(store)
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_test",
            sender_id="zhang",
            sender_name="张哥",
            text="通宵1块有人吗？没有就帮我组一个",
            message_id="msg_runtime_drive_001",
        ),
        trace_id="trace_drive_001",
    )

    assert result.final_reply == "好的，我帮你问问，有消息跟你说。"
    assert [call.name for action in result.actions for call in action.tool_calls] == [
        "search_current_games",
        "create_game",
        "search_customers",
        "create_invite_drafts",
    ]
    assert len(store.games) == 1
    assert len(store.invite_drafts) == 2
    assert [draft.message_text for draft in store.invite_drafts.values()] == [
        "冉姐，1块通宵，打吗？",
        "何哥，1块通宵，打吗？",
    ]
    events = trace.get_trace("trace_drive_001")
    assert validate_trace(events)["complete"] is True
    steps = trace_steps(events)
    assert steps.count("llm_prompt") == 5
    assert steps.count("tool_called") == 4
    assert "state_transition" in steps
    prompts = [event.content for event in events if event.step == "llm_prompt"]
    last_payload = json.loads(prompts[-1]["messages"][1]["content"])
    assert last_payload["previous_tool_results"][0]["name"] == "create_invite_drafts"


def test_runtime_shorthand_current_players_sets_requester_seat_count() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="先查是否可拼。",
                tool_calls=[
                    {
                        "name": "search_current_games",
                        "arguments": {
                            "requirement": {
                                "game_type": "sichuan_mahjong",
                                "stake": "232",
                                "smoke_preference": "no_smoke",
                                "known_player_count": 2,
                                "needed_seats": 2,
                            },
                            "limit": 10,
                        },
                        "reason": "查川麻2-32无烟可拼局。",
                    }
                ],
            ),
            action_json(
                objective_status="needs_tool",
                reasoning_summary="无匹配，继续建局找人。",
                tool_calls=[
                    {
                        "name": "create_game",
                        "arguments": {
                            "requirement": {
                                "game_type": "sichuan_mahjong",
                                "stake": "232",
                                "smoke_preference": "no_smoke",
                                "known_player_count": 2,
                                "needed_seats": 2,
                                "user_visible_summary": "川麻 2-32 无烟 272",
                            },
                            "organizer_id": "zhang",
                            "organizer_name": "张哥",
                            "known_players": [
                                {
                                    "customer_id": "zhang",
                                    "display_name": "张哥",
                                    "source": "organizer",
                                    "seat_count": 2,
                                }
                            ],
                        },
                        "reason": "创建2缺2川麻局。",
                    },
                    {
                        "name": "search_customers",
                        "arguments": {
                            "requirement": {
                                "game_type": "sichuan_mahjong",
                                "stake": "232",
                                "smoke_preference": "no_smoke",
                                "organizer_id": "zhang",
                                "existing_player_ids": ["zhang"],
                            },
                            "exclude_customer_ids": ["zhang"],
                            "limit": 2,
                        },
                        "reason": "找2-32川麻候选人。",
                    },
                ],
            ),
            action_json(
                objective_status="completed",
                reasoning_summary="已开始组局。",
                reply_to_user="好，我帮你问问。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_test_shorthand_players",
            sender_id="zhang",
            sender_name="张哥",
            text="川麻无烟232，272",
            message_id="msg_runtime_shorthand_players_001",
        ),
        trace_id="trace_runtime_shorthand_players",
    )

    assert result.final_reply == "好，我帮你问问。"
    assert [tool.name for tool in result.tool_results] == ["search_current_games", "create_game", "search_customers"]
    game = next(iter(store.games.values()))
    assert game.requirement["stake"] == "2"
    assert game.requirement["base_stake"] == 2.0
    assert game.requirement["cap_score"] == 32.0
    assert game.requirement["stake_label"] == "2-32"
    assert game.participants[0].customer_id == "zhang"
    assert game.participants[0].seat_count == 2
    assert game.participants[0].anonymous_seat_count == 1
    assert game.parties[0].contact_id == "zhang"
    assert game.parties[0].seat_count == 2
    assert game.parties[0].anonymous_seat_count == 1
    assert game.to_dict()["seat_summary"]["claimed_seats"] == 2
    assert game.to_dict()["seat_claims"][0]["contact_id"] == "zhang"
    assert game.remaining_seats() == 2


def test_runtime_create_game_derives_requester_party_from_requirement_count() -> None:
    store = seeded_store()

    game, _ = store.create_game(
        conversation_id="runtime_party_contract",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={
            "game_type": "sichuan_mahjong",
            "stake": "232",
            "smoke_preference": "no_smoke",
            "known_player_count": 2,
            "needed_seats": 2,
        },
        known_players=[{"customer_id": "zhang", "display_name": "张哥"}],
        trace_id="trace_party_contract",
    )

    assert game.requirement["stake"] == "2"
    assert game.requirement["cap_score"] == 32.0
    assert game.requirement["requesting_party"]["contact_id"] == "zhang"
    assert game.requirement["requesting_party"]["seat_count"] == 2
    assert game.parties[0].to_dict() == {
        "party_id": "party_zhang",
        "contact_id": "zhang",
        "contact_name": "张哥",
        "seat_count": 2,
        "known_member_ids": ["zhang"],
        "anonymous_seat_count": 1,
        "status": "joined",
        "source": "requester",
    }
    assert game.seat_summary()["claimed_seats"] == 2
    assert game.remaining_seats() == 2


def test_runtime_tool_gateway_accepts_requesting_party_contract() -> None:
    store = seeded_store()
    gateway = ToolGateway(store)

    result = gateway.execute(
        ToolCall(
            name="create_game",
            arguments={
                "organizer_id": "zhang",
                "organizer_name": "张哥",
                "requirement": {
                    "game_type": "sichuan_mahjong",
                    "stake": "232",
                    "smoke_preference": "no_smoke",
                    "known_player_count": 2,
                    "needed_seats": 2,
                },
                "requesting_party": {
                    "contact_id": "zhang",
                    "contact_name": "张哥",
                    "seat_count": 2,
                    "known_member_ids": ["zhang"],
                    "anonymous_seat_count": 1,
                },
            },
            reason="验证party契约。",
        ),
        trace_id="trace_party_gateway",
        conversation_id="runtime_party_gateway",
        sender_id="zhang",
        sender_name="张哥",
        step_index=0,
        source_message_id="msg_party_gateway",
    )

    assert result.allowed is True
    assert result.called is True
    game_payload = result.result["game"]
    assert game_payload["parties"][0]["contact_id"] == "zhang"
    assert game_payload["parties"][0]["seat_count"] == 2
    assert game_payload["seat_summary"]["claimed_seats"] == 2
    assert game_payload["remaining_seats"] == 2


def test_runtime_reply_self_review_rewrites_leaking_customer_reply() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="completed",
                reasoning_summary="模型错误地把后台执行细节写进客户回复。",
                reply_to_user="张哥，邀约草稿已发给冉姐和何哥，等老板审批后就发送邀请。",
            ),
            json.dumps(
                {
                    "approved": False,
                    "needs_human": False,
                    "reasoning_summary": "原回复泄露候选人和审批流程，已改成客户可见进展。",
                    "violations": ["leaks_internal_workflow", "leaks_candidate_names"],
                    "item_reviews": [
                        {
                            "item_id": "reply_to_user",
                            "approved": False,
                            "suggested_safe_text": "张哥，这桌我帮你安排着，有消息跟你说。",
                            "reasoning_summary": "原回复泄露候选人和审批流程。",
                            "violations": ["leaks_internal_workflow", "leaks_candidate_names"],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            action_json(
                objective_status="completed",
                reasoning_summary="根据 customer_visible_content_review 工具结果重写客户可见回复。",
                reply_to_user="张哥，我帮你问问，有消息跟你说。",
            ),
            json.dumps(
                {
                    "approved": True,
                    "needs_human": False,
                    "reasoning_summary": "重写后的回复没有泄露后台流程或候选人。",
                    "violations": [],
                    "item_reviews": [
                        {
                            "item_id": "reply_to_user",
                            "approved": True,
                            "suggested_safe_text": "张哥，我帮你问问，有消息跟你说。",
                            "reasoning_summary": "重写后的回复安全。",
                            "violations": [],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
        ]
    )
    runtime = AgentRuntime(
        llm_client=client,
        store=store,
        trace_recorder=trace,
        reply_self_review_enabled=True,
    )

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_reply_self_review",
            sender_id="zhang",
            sender_name="张哥",
            text="有局吗",
            message_id="msg_runtime_reply_self_review",
        ),
        trace_id="trace_reply_self_review",
    )

    assert result.final_reply == "张哥，我帮你问问，有消息跟你说。"
    assert len(client.calls) == 4
    review_payload = json.loads(client.calls[1]["messages"][1]["content"])
    assert review_payload["review_items"][0]["text"] == "张哥，邀约草稿已发给冉姐和何哥，等老板审批后就发送邀请。"
    assert review_payload["review_contract"]["available_tools"] == []
    assert "不负责润色文风" in review_payload["review_goal"]
    retry_payload = json.loads(client.calls[2]["messages"][1]["content"])
    assert retry_payload["previous_tool_results"][0]["name"] == "customer_visible_content_review"
    assert retry_payload["previous_tool_results"][0]["result"]["approved"] is False
    assert retry_payload["previous_tool_results"][0]["result"]["item_reviews"][0]["suggested_safe_text"] == "张哥，这桌我帮你安排着，有消息跟你说。"
    steps = trace_steps(trace.get_trace("trace_reply_self_review"))
    assert steps.count("customer_visible_content_review_prompt") == 2
    assert steps.count("customer_visible_content_review_response") == 2
    assert steps.count("customer_visible_content_review_result") == 2


def test_runtime_customer_visible_text_generation_rewrites_reply_before_review() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    original_reply = "现在有一个1有烟、人齐开、4小时的局，要加入吗？"
    main_client = StaticAgentClient(
        [
            action_json(
                objective_status="waiting_user",
                reasoning_summary="查到一个现成局，但主模型话术字段味太重。",
                reply_to_user=original_reply,
            )
        ]
    )
    text_client = StaticAgentClient(
        [
            json.dumps(
                {
                    "reasoning_summary": "只保留原文已有公共局条件，去掉客服腔，未新增人数。",
                    "item_rewrites": [
                        {
                            "item_id": "reply_to_user",
                            "final_text": "有个1块有烟、人齐开、4小时左右的局，打吗？",
                            "semantic_preserved": True,
                            "used_facts": ["1块", "有烟", "人齐开", "4小时"],
                            "withheld_facts": ["发起人身份", "后台流程"],
                            "style_checks": ["短句", "老板口吻", "未新增事实"],
                            "change_summary": "把1有烟改成1块有烟，压缩选择过载。",
                        }
                    ],
                },
                ensure_ascii=False,
            )
        ]
    )
    review_client = StaticAgentClient(
        [
            json.dumps(
                {
                    "approved": True,
                    "needs_human": False,
                    "reasoning_summary": "改写后的回复只包含公共条件。",
                    "violations": [],
                    "item_reviews": [
                        {
                            "item_id": "reply_to_user",
                            "approved": True,
                            "suggested_safe_text": "有个1块有烟、人齐开、4小时左右的局，打吗？",
                            "reasoning_summary": "安全。",
                            "violations": [],
                        }
                    ],
                },
                ensure_ascii=False,
            )
        ]
    )
    runtime = AgentRuntime(
        llm_client=main_client,
        store=store,
        trace_recorder=trace,
        customer_visible_text_generation_enabled=True,
        customer_visible_text_generation_client=text_client,
        reply_self_review_enabled=True,
        reply_self_review_client=review_client,
    )

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_copywriting_reply",
            sender_id="wang02",
            sender_name="王哥",
            text="现在有人打牌吗",
            message_id="msg_copywriting_reply",
        ),
        trace_id="trace_copywriting_reply",
    )

    assert result.final_reply == "有个1块有烟、人齐开、4小时左右的局，打吗？"
    assert "要加入吗" not in result.final_reply
    assert [item.name for item in result.tool_results] == ["customer_visible_text_generation", "customer_visible_content_review"]
    generation_payload = json.loads(text_client.calls[0]["messages"][1]["content"])
    assert generation_payload["items"][0]["text"] == original_reply
    assert set(generation_payload["items"][0]) == {"item_id", "source", "text"}
    assert "context" not in generation_payload
    assert "current_message" not in generation_payload
    assert generation_payload["action_boundary"] == {
        "objective_status": "waiting_user",
        "needs_human": False,
        "tool_call_names": [],
    }
    assert generation_payload["output_contract"]["available_tools"] == []
    assert generation_payload["generation_scope"] == "reply_to_user"
    review_payload = json.loads(review_client.calls[0]["messages"][1]["content"])
    assert review_payload["review_items"][0]["text"] == "有个1块有烟、人齐开、4小时左右的局，打吗？"
    assert result.actions[-1].reply_to_user == "有个1块有烟、人齐开、4小时左右的局，打吗？"
    steps = trace_steps(trace.get_trace("trace_copywriting_reply"))
    assert "customer_visible_text_generation_prompt" in steps
    assert "customer_visible_text_generation_result" in steps
    assert "action_after_customer_visible_text_generation" in steps
    assert "customer_visible_content_review_result" in steps


def test_runtime_customer_visible_text_generation_rewrites_invite_text_before_review_and_draft() -> None:
    store = seeded_store()
    game, _ = store.create_game(
        conversation_id="runtime_copywriting_invite",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "any", "start_time_kind": "asap_when_full"},
        known_players=[{"customer_id": "zhang", "display_name": "张哥", "status": "joined"}],
        trace_id="trace_copywriting_invite_seed",
    )
    trace = InMemoryTraceRecorder()
    main_client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="主模型生成了带内部枚举味道的候选人邀约。",
                tool_calls=[
                    {
                        "name": "create_invite_drafts",
                        "arguments": {
                            "game_id": game.game_id,
                            "invitations": [
                                {
                                    "customer_id": "ran",
                                    "display_name": "冉姐",
                                    "message_text": "冉姐，asap_when_full，1，烟都可，打吗？",
                                }
                            ],
                        },
                        "reason": "创建候选人待审批邀约草稿。",
                    }
                ],
            ),
            action_json(
                objective_status="completed",
                reasoning_summary="候选人邀约草稿已创建，回复发起人。",
                reply_to_user="好，我帮你问问，有消息跟你说。",
            ),
        ]
    )
    text_client = StaticAgentClient(
        [
            json.dumps(
                {
                    "reasoning_summary": "把内部枚举和金额字段改成候选人能看懂的话。",
                    "item_rewrites": [
                        {
                            "item_id": "tool_calls[1].arguments.invitations[1].message_text",
                            "final_text": "人齐开，1块，烟都可以，打吗？",
                            "semantic_preserved": True,
                            "used_facts": ["人齐开", "1块", "烟都可以"],
                            "withheld_facts": ["后台草稿状态"],
                            "style_checks": ["短句", "未暴露内部流程", "未新增事实"],
                            "change_summary": "翻译内部时间枚举和金额，去掉开头称呼。",
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "reasoning_summary": "发起人回复已经自然，保持不变。",
                    "item_rewrites": [
                        {
                            "item_id": "reply_to_user",
                            "final_text": "好，我帮你问问，有消息跟你说。",
                            "semantic_preserved": True,
                            "used_facts": ["开始问人"],
                            "withheld_facts": ["候选人名单", "草稿状态"],
                            "style_checks": ["短句", "老板口吻"],
                            "change_summary": "保持原文。",
                        }
                    ],
                },
                ensure_ascii=False,
            ),
        ]
    )
    review_client = StaticAgentClient(
        [
            json.dumps(
                {
                    "approved": True,
                    "needs_human": False,
                    "reasoning_summary": "候选人文案安全。",
                    "violations": [],
                    "item_reviews": [
                        {
                            "item_id": "tool_calls[1].arguments.invitations[1].message_text",
                            "approved": True,
                            "suggested_safe_text": "人齐开，1块，烟都可以，打吗？",
                            "reasoning_summary": "安全。",
                            "violations": [],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            json.dumps(
                {
                    "approved": True,
                    "needs_human": False,
                    "reasoning_summary": "发起人回复安全。",
                    "violations": [],
                    "item_reviews": [
                        {
                            "item_id": "reply_to_user",
                            "approved": True,
                            "suggested_safe_text": "好，我帮你问问，有消息跟你说。",
                            "reasoning_summary": "安全。",
                            "violations": [],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
        ]
    )
    runtime = AgentRuntime(
        llm_client=main_client,
        store=store,
        trace_recorder=trace,
        customer_visible_text_generation_enabled=True,
        customer_visible_text_generation_client=text_client,
        reply_self_review_enabled=True,
        reply_self_review_client=review_client,
    )

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_copywriting_invite",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我问冉姐",
            message_id="msg_copywriting_invite",
        ),
        trace_id="trace_copywriting_invite",
    )

    assert result.final_reply == "好，我帮你问问，有消息跟你说。"
    assert [draft.message_text for draft in store.invite_drafts.values()] == ["人齐开，1块，烟都可以，打吗？"]
    assert [item.name for item in result.tool_results] == [
        "customer_visible_text_generation",
        "customer_visible_content_review",
        "create_invite_drafts",
        "customer_visible_text_generation",
        "customer_visible_content_review",
    ]
    first_review_payload = json.loads(review_client.calls[0]["messages"][1]["content"])
    assert first_review_payload["review_items"][0]["text"] == "人齐开，1块，烟都可以，打吗？"
    steps = trace_steps(trace.get_trace("trace_copywriting_invite"))
    assert steps.count("customer_visible_text_generation_result") == 2
    assert steps.count("customer_visible_content_review_result") == 2
    assert "tool_called" in steps


def test_runtime_customer_visible_text_generation_rejects_non_semantic_rewrite() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    original_reply = "现在有一个1有烟、人齐开、4小时的局，要帮你问问能不能加进去，还是你自己组一个？"
    main_client = StaticAgentClient(
        [
            action_json(
                objective_status="waiting_user",
                reasoning_summary="主模型给出一个字段味回复。",
                reply_to_user=original_reply,
            )
        ]
    )
    text_client = StaticAgentClient(
        [
            json.dumps(
                {
                    "reasoning_summary": "试图补充原文没有的人数。",
                    "item_rewrites": [
                        {
                            "item_id": "reply_to_user",
                            "final_text": "有个173，1块有烟，人齐开，打吗？",
                            "semantic_preserved": False,
                            "used_facts": ["173", "1块", "有烟", "人齐开"],
                            "withheld_facts": [],
                            "style_checks": ["新增了原文没有的人数"],
                            "change_summary": "新增173，不能保真。",
                        }
                    ],
                },
                ensure_ascii=False,
            )
        ]
    )
    review_client = StaticAgentClient(
        [
            json.dumps(
                {
                    "approved": True,
                    "needs_human": False,
                    "reasoning_summary": "话术生成器失败后，审查原始主模型回复。",
                    "violations": [],
                    "item_reviews": [
                        {
                            "item_id": "reply_to_user",
                            "approved": True,
                            "suggested_safe_text": original_reply,
                            "reasoning_summary": "安全。",
                            "violations": [],
                        }
                    ],
                },
                ensure_ascii=False,
            )
        ]
    )
    runtime = AgentRuntime(
        llm_client=main_client,
        store=store,
        trace_recorder=trace,
        customer_visible_text_generation_enabled=True,
        customer_visible_text_generation_client=text_client,
        reply_self_review_enabled=True,
        reply_self_review_client=review_client,
    )

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_copywriting_reject_non_semantic",
            sender_id="wang02",
            sender_name="王哥",
            text="现在有人打牌吗",
            message_id="msg_copywriting_reject_non_semantic",
        ),
        trace_id="trace_copywriting_reject_non_semantic",
    )

    assert result.final_reply == original_reply
    assert [item.name for item in result.tool_results] == ["customer_visible_content_review"]
    review_payload = json.loads(review_client.calls[0]["messages"][1]["content"])
    assert review_payload["review_items"][0]["text"] == original_reply
    steps = trace_steps(trace.get_trace("trace_copywriting_reject_non_semantic"))
    assert "customer_visible_text_generation_contract_error" in steps
    assert "customer_visible_text_generation_result" not in steps


def test_runtime_reply_self_review_can_escalate_to_human() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="completed",
                reasoning_summary="模型准备回复。",
                reply_to_user="张哥，我已经问了几个人。",
            ),
            json.dumps(
                {
                    "approved": False,
                    "needs_human": True,
                    "reasoning_summary": "无法确认是否已经真实外发，交给人工。",
                    "violations": ["unverified_external_action"],
                    "item_reviews": [
                        {
                            "item_id": "reply_to_user",
                            "approved": False,
                            "suggested_safe_text": "这个我先确认一下。",
                            "reasoning_summary": "无法确认是否已经真实外发。",
                            "violations": ["unverified_external_action"],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            action_json(
                objective_status="needs_human",
                reasoning_summary="审查工具要求人工确认。",
                reply_to_user="这个我先转人工确认一下。",
                needs_human=True,
            ),
            json.dumps(
                {
                    "approved": True,
                    "needs_human": False,
                    "reasoning_summary": "人工兜底回复安全。",
                    "violations": [],
                    "item_reviews": [
                        {
                            "item_id": "reply_to_user",
                            "approved": True,
                            "suggested_safe_text": "这个我先转人工确认一下。",
                            "reasoning_summary": "人工兜底回复安全。",
                            "violations": [],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
        ]
    )
    runtime = AgentRuntime(
        llm_client=client,
        store=store,
        trace_recorder=trace,
        reply_self_review_enabled=True,
    )

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_reply_self_review_human",
            sender_id="zhang",
            sender_name="张哥",
            text="有局吗",
            message_id="msg_runtime_reply_self_review_human",
        ),
        trace_id="trace_reply_self_review_human",
    )

    assert result.final_reply == "这个我先转人工确认一下。"
    review_event = next(event for event in trace.get_trace("trace_reply_self_review_human") if event.step == "customer_visible_content_review_result")
    assert review_event.content["needs_human"] is True
    retry_payload = json.loads(client.calls[2]["messages"][1]["content"])
    assert retry_payload["previous_tool_results"][0]["name"] == "customer_visible_content_review"
    assert retry_payload["previous_tool_results"][0]["result"]["needs_human"] is True


def test_runtime_reviews_candidate_visible_invite_text_before_creating_draft() -> None:
    store = seeded_store()
    game, _ = store.create_game(
        conversation_id="runtime_candidate_text_review",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "any"},
        known_players=[{"customer_id": "zhang", "display_name": "张哥", "status": "joined"}],
        trace_id="trace_seed_game",
    )
    trace = InMemoryTraceRecorder()
    unsafe_text = "冉姐，张哥这桌已经建好，草稿等老板审批，何哥也被邀请了，1块有烟，打吗？"
    safe_text = "冉姐，1块有烟，打吗？"
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="先生成候选人邀约草稿，但文案泄露了后台流程。",
                tool_calls=[
                    {
                        "name": "create_invite_drafts",
                        "arguments": {
                            "game_id": game.game_id,
                            "invitations": [
                                {
                                    "customer_id": "ran",
                                    "display_name": "冉姐",
                                    "message_text": unsafe_text,
                                }
                            ],
                        },
                        "reason": "为当前局创建候选人邀约草稿。",
                    }
                ],
            ),
            json.dumps(
                {
                    "approved": False,
                    "needs_human": False,
                    "reasoning_summary": "候选人可见文本泄露发起人、其他候选人和审批流程。",
                    "violations": ["leaks_internal_process", "leaks_other_customer_info"],
                    "item_reviews": [
                        {
                            "item_id": "tool_calls[1].arguments.invitations[1].message_text",
                            "approved": False,
                            "suggested_safe_text": safe_text,
                            "reasoning_summary": "只保留候选人需要知道的公共条件。",
                            "violations": ["leaks_internal_process", "leaks_other_customer_info"],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            action_json(
                objective_status="needs_tool",
                reasoning_summary="根据审查结果重写候选人邀约文本。",
                tool_calls=[
                    {
                        "name": "create_invite_drafts",
                        "arguments": {
                            "game_id": game.game_id,
                            "invitations": [
                                {
                                    "customer_id": "ran",
                                    "display_name": "冉姐",
                                    "message_text": safe_text,
                                }
                            ],
                        },
                        "reason": "用审查通过的客户可见文本创建候选人邀约草稿。",
                    }
                ],
            ),
            json.dumps(
                {
                    "approved": True,
                    "needs_human": False,
                    "reasoning_summary": "候选人邀约文本安全。",
                    "violations": [],
                    "item_reviews": [
                        {
                            "item_id": "tool_calls[1].arguments.invitations[1].message_text",
                            "approved": True,
                            "suggested_safe_text": safe_text,
                            "reasoning_summary": "未泄露后台流程或其他客户。",
                            "violations": [],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            action_json(
                objective_status="completed",
                reasoning_summary="安全邀约草稿已创建。",
                reply_to_user="好的，我帮你问问，有消息跟你说。",
            ),
            json.dumps(
                {
                    "approved": True,
                    "needs_human": False,
                    "reasoning_summary": "发起人回复安全。",
                    "violations": [],
                    "item_reviews": [
                        {
                            "item_id": "reply_to_user",
                            "approved": True,
                            "suggested_safe_text": "好的，我帮你问问，有消息跟你说。",
                            "reasoning_summary": "未泄露候选人或后台流程。",
                            "violations": [],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
        ]
    )
    runtime = AgentRuntime(
        llm_client=client,
        store=store,
        trace_recorder=trace,
        reply_self_review_enabled=True,
    )

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_candidate_text_review",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我问冉姐",
            message_id="msg_candidate_text_review",
        ),
        trace_id="trace_candidate_text_review",
    )

    assert result.final_reply == "好的，我帮你问问，有消息跟你说。"
    assert [draft.message_text for draft in store.invite_drafts.values()] == [safe_text]
    assert unsafe_text not in json.dumps([draft.to_dict() for draft in store.invite_drafts.values()], ensure_ascii=False)
    first_review_payload = json.loads(client.calls[1]["messages"][1]["content"])
    assert first_review_payload["review_scope"] == "tool_calls"
    assert first_review_payload["review_items"][0]["source"] == "create_invite_drafts"
    assert first_review_payload["review_items"][0]["text"] == unsafe_text
    retry_payload = json.loads(client.calls[2]["messages"][1]["content"])
    assert retry_payload["previous_tool_results"][0]["name"] == "customer_visible_content_review"
    assert retry_payload["previous_tool_results"][0]["result"]["item_reviews"][0]["suggested_safe_text"] == safe_text
    events = trace.get_trace("trace_candidate_text_review")
    steps = trace_steps(events)
    assert steps.count("tool_called") == 1
    assert steps.count("customer_visible_content_review_result") == 3


def test_runtime_review_budget_does_not_consume_main_agent_loop_budget() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="completed",
                reasoning_summary="主模型第一版回复泄露了候选人。",
                reply_to_user="张哥，我正在邀请冉姐和何哥，等他们回复。",
            ),
            json.dumps(
                {
                    "approved": False,
                    "needs_human": False,
                    "reasoning_summary": "回复泄露候选人姓名。",
                    "violations": ["leaks_candidate_names"],
                    "item_reviews": [
                        {
                            "item_id": "reply_to_user",
                            "approved": False,
                            "suggested_safe_text": "张哥，我帮你问问，有消息跟你说。",
                            "reasoning_summary": "删除候选人姓名。",
                            "violations": ["leaks_candidate_names"],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            action_json(
                objective_status="completed",
                reasoning_summary="主模型根据审查结果重写回复。",
                reply_to_user="张哥，我帮你问问，有消息跟你说。",
            ),
            json.dumps(
                {
                    "approved": True,
                    "needs_human": False,
                    "reasoning_summary": "重写后的回复安全。",
                    "violations": [],
                    "item_reviews": [
                        {
                            "item_id": "reply_to_user",
                            "approved": True,
                            "suggested_safe_text": "张哥，我帮你问问，有消息跟你说。",
                            "reasoning_summary": "安全。",
                            "violations": [],
                        }
                    ],
                },
                ensure_ascii=False,
            ),
        ]
    )
    runtime = AgentRuntime(
        llm_client=client,
        store=store,
        trace_recorder=trace,
        token_budget=TokenBudget(max_tokens_per_call=24_000, max_calls_per_turn=2),
        review_token_budget=TokenBudget(max_tokens_per_call=24_000, max_calls_per_turn=2),
        reply_self_review_enabled=True,
    )

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_split_review_budget",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我问问",
            message_id="msg_split_review_budget",
        ),
        trace_id="trace_split_review_budget",
    )

    assert result.final_reply == "张哥，我帮你问问，有消息跟你说。"
    events = trace.get_trace("trace_split_review_budget")
    main_budget_events = [event for event in events if event.step == "budget_checked"]
    review_budget_events = [event for event in events if event.step == "customer_visible_content_review_budget_checked"]
    assert [event.content["allowed"] for event in main_budget_events] == [True, True]
    assert [event.content["allowed"] for event in review_budget_events] == [True, True]
    assert all("turn llm call limit exceeded" not in str(event.content) for event in events)


def test_runtime_backend_does_not_interpret_short_confirmation_as_create_game() -> None:
    store = seeded_store()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="completed",
                reasoning_summary="模型选择只回复，不调用工具。",
                reply_to_user="好的。",
            )
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorder())

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_no_backend_semantic",
            sender_id="zhang",
            sender_name="张哥",
            text="组",
            message_id="msg_runtime_no_backend_semantic",
        ),
        trace_id="trace_no_backend_semantic",
    )

    assert result.final_reply == "好的。"
    assert result.tool_results == []
    assert store.games == {}
    assert len(client.calls) == 1


def test_runtime_tool_schema_error_is_fed_back_to_model_not_repaired_by_backend() -> None:
    store = seeded_store()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                tool_calls=[
                    {
                        "name": "create_invite_drafts",
                        "arguments": {"invitations": []},
                        "reason": "故意缺 game_id，验证工具错误回喂模型。",
                    }
                ],
            ),
            action_json(
                objective_status="waiting_user",
                reasoning_summary="上一步工具返回 schema 错误，模型决定等待人工补充。",
                reply_to_user="我先确认一下。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorder())

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_schema_error",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我问问",
            message_id="msg_runtime_schema_error",
        ),
        trace_id="trace_schema_error",
    )

    assert result.tool_results[0].error == "missing required argument: game_id"
    second_prompt = json.loads(client.calls[1]["messages"][1]["content"])
    assert second_prompt["previous_tool_results"][0]["error"] == "missing required argument: game_id"
    assert store.games == {}
    assert result.final_reply == "我先确认一下。"


def test_runtime_action_contract_error_is_fed_back_to_model_for_repair() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            json.dumps(
                {
                    "goal": "查询当前局池，确认是否有适合张哥的局",
                    "objective_status": "needs_tool",
                    "reasoning_summary": "局池为空，需要询问用户偏好，但错误地声明 needs_tool。",
                    "reply_to_user": "",
                    "tool_calls": [],
                    "needs_human": False,
                    "stop_reason": {
                        "can_stop": True,
                        "why": "当前局池为空，需要用户补充偏好信息。",
                        "pending_work": [],
                        "depends_on_tool_results": True,
                    },
                    "badcase": None,
                },
                ensure_ascii=False,
            ),
            action_json(
                objective_status="waiting_user",
                reasoning_summary="上一轮 AgentAction 合同错误，修正为等待用户补充。",
                reply_to_user="现在没有现成的局，要不要帮你组一个？",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_contract_repair",
            sender_id="zhang",
            sender_name="张哥",
            text="有局吗",
            message_id="msg_runtime_contract_repair",
        ),
        trace_id="trace_contract_repair",
    )

    assert result.final_reply == "现在没有现成的局，要不要帮你组一个？"
    assert result.tool_results == []
    assert store.games == {}
    assert len(client.calls) == 2
    second_prompt = json.loads(client.calls[1]["messages"][1]["content"])
    feedback = second_prompt["previous_tool_results"][0]
    assert feedback["name"] == "agent_action_contract"
    assert "needs_tool requires at least one tool_call" in feedback["error"]
    steps = trace_steps(trace.get_trace("trace_contract_repair"))
    assert "action_contract_error" in steps
    assert "contract_error_feedback" in steps
    assert steps[-1] == "final_output"


def test_runtime_schema_rejects_empty_invite_draft_list_without_state_change() -> None:
    store = seeded_store()
    game, _ = store.create_game(
        conversation_id="runtime_empty_invites",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"game_type": "hangzhou_mahjong", "stake": "1"},
        known_players=[{"customer_id": "zhang", "display_name": "张哥"}],
        trace_id="setup_empty_invites",
    )
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="模型错误地请求创建空邀约草稿。",
                tool_calls=[
                    {
                        "name": "create_invite_drafts",
                        "arguments": {"game_id": game.game_id, "invitations": []},
                        "reason": "验证空数组不会产生空副作用。",
                    }
                ],
            ),
            action_json(
                objective_status="waiting_user",
                reasoning_summary="工具 schema 拒绝空 invitations，模型需要重新规划。",
                reply_to_user="我先重新确认一下要问谁。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorder())

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_empty_invites",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我问问",
            message_id="msg_runtime_empty_invites",
        ),
        trace_id="trace_empty_invites",
    )

    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert result.tool_results[0].error == "invitations must contain at least 1 item(s)"
    assert store.games[game.game_id].status.value == "forming"
    assert store.invite_drafts == {}
    second_prompt = json.loads(client.calls[1]["messages"][1]["content"])
    assert second_prompt["previous_tool_results"][0]["error"] == "invitations must contain at least 1 item(s)"
    assert result.final_reply == "我先重新确认一下要问谁。"


def test_runtime_schema_rejects_empty_outbound_message_draft_list() -> None:
    store = seeded_store()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="模型错误地请求创建空外发草稿。",
                tool_calls=[
                    {
                        "name": "create_outbound_message_drafts",
                        "arguments": {"drafts": []},
                        "reason": "验证空数组不会让模型假装已经生成草稿。",
                    }
                ],
            ),
            action_json(
                objective_status="waiting_user",
                reasoning_summary="工具 schema 拒绝空 drafts，模型需要重新规划。",
                reply_to_user="我先重新生成一版草稿。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorder())

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_empty_outbound",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我回一句",
            message_id="msg_runtime_empty_outbound",
        ),
        trace_id="trace_empty_outbound",
    )

    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert result.tool_results[0].error == "drafts must contain at least 1 item(s)"
    assert store.outbound_message_drafts == {}
    second_prompt = json.loads(client.calls[1]["messages"][1]["content"])
    assert second_prompt["previous_tool_results"][0]["error"] == "drafts must contain at least 1 item(s)"
    assert result.final_reply == "我先重新生成一版草稿。"


def test_runtime_create_game_counts_requesting_customer_as_player() -> None:
    store = seeded_store()

    game, _ = store.create_game(
        conversation_id="runtime_requester_counts",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"game_type": "hangzhou_mahjong", "stake": "1"},
        known_players=[],
        trace_id="trace_requester_counts",
    )

    assert [(item.customer_id, item.display_name, item.status, item.source) for item in game.participants] == [
        ("zhang", "张哥", "joined", "requester")
    ]
    assert game.remaining_seats() == 3
    assert game.to_dict()["remaining_seats"] == 3


def test_runtime_sqlite_create_game_counts_requester_and_repairs_legacy_payload(tmp_path) -> None:
    store = seeded_store(SQLiteAgentStore(tmp_path / "runtime_requester_counts.sqlite3"))

    game, _ = store.create_game(
        conversation_id="runtime_sqlite_requester_counts",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"game_type": "hangzhou_mahjong", "stake": "1"},
        known_players=[{"customer_id": "zhang", "display_name": "张哥", "source": "organizer"}],
        trace_id="trace_sqlite_requester_counts",
    )

    assert [(item.customer_id, item.status, item.source) for item in game.participants] == [
        ("zhang", "joined", "requester")
    ]
    assert game.remaining_seats() == 3

    with store._lock, store._connection:
        payload = game.to_dict()
        payload["participants"] = []
        store._connection.execute(
            "UPDATE runtime_games SET payload = ? WHERE game_id = ?",
            (json.dumps(payload, ensure_ascii=False), game.game_id),
        )

    repaired = store.games[game.game_id]
    assert [(item.customer_id, item.status, item.source) for item in repaired.participants] == [
        ("zhang", "joined", "requester")
    ]
    assert repaired.remaining_seats() == 3


def test_runtime_game_participants_can_represent_multiple_seats() -> None:
    store = seeded_store()

    game, _ = store.create_game(
        conversation_id="runtime_party_size",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "no_smoke"},
        known_players=[{"customer_id": "zhang", "display_name": "张哥", "seat_count": 1}],
        trace_id="setup_party_size",
    )
    game, _ = store.record_candidate_reply(
        game_id=game.game_id,
        customer_id="lin01",
        display_name="林01",
        status="accepted",
        seat_count=2,
        trace_id="trace_party_size_lin",
    )

    lin = next(item for item in game.participants if item.customer_id == "lin01")
    assert lin.seat_count == 2
    assert game.remaining_seats() == 1

    matches = store.search_current_games(
        {"game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "no_smoke", "seat_count": 1},
        sender_id="k01",
        limit=5,
    )

    assert len(matches) == 1
    assert matches[0]["game"]["remaining_seats"] == 1
    assert matches[0]["join_projection"] == {
        "sender_id": "k01",
        "sender_already_joined": False,
        "requested_seats": 1,
        "remaining_seats_before_join": 1,
        "remaining_seats_after_join": 0,
        "would_fill_game": True,
        "would_overfill_game": False,
    }


def test_runtime_search_current_games_projects_remaining_seats_after_sender_join() -> None:
    store = seeded_store()

    game, _ = store.create_game(
        conversation_id="runtime_join_projection",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "no_smoke"},
        known_players=[{"customer_id": "zhang", "display_name": "张哥"}],
        trace_id="setup_join_projection",
    )
    store.record_candidate_reply(
        game_id=game.game_id,
        customer_id="lin01",
        display_name="林01",
        status="accepted",
        trace_id="trace_join_projection_lin",
    )

    matches = store.search_current_games(
        {"game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "no_smoke"},
        sender_id="k01",
        limit=5,
    )

    assert matches[0]["game"]["remaining_seats"] == 2
    assert matches[0]["join_projection"]["remaining_seats_before_join"] == 2
    assert matches[0]["join_projection"]["remaining_seats_after_join"] == 1
    assert matches[0]["join_projection"]["would_fill_game"] is False


def test_runtime_sqlite_preserves_party_size_across_reload(tmp_path) -> None:
    db_path = tmp_path / "runtime_party_size.sqlite3"
    store = seeded_store(SQLiteAgentStore(db_path))

    game, _ = store.create_game(
        conversation_id="runtime_sqlite_party_size",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"game_type": "hangzhou_mahjong", "stake": "1", "smoke_preference": "no_smoke"},
        known_players=[{"customer_id": "zhang", "display_name": "张哥", "seat_count": 2}],
        trace_id="setup_sqlite_party_size",
    )

    assert next(item for item in game.participants if item.customer_id == "zhang").seat_count == 2
    assert game.parties[0].seat_count == 2
    assert game.remaining_seats() == 2

    reopened = SQLiteAgentStore(db_path)
    persisted = reopened.games[game.game_id]

    assert next(item for item in persisted.participants if item.customer_id == "zhang").seat_count == 2
    assert persisted.parties[0].contact_id == "zhang"
    assert persisted.parties[0].seat_count == 2
    assert persisted.seat_summary()["claimed_seats"] == 2
    assert persisted.remaining_seats() == 2


def test_runtime_create_game_requires_explicit_organizer_identity() -> None:
    store = seeded_store()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="模型尝试建局，但没有显式提供组织者身份。",
                tool_calls=[
                    {
                        "name": "create_game",
                        "arguments": {
                            "requirement": {"game_type": "hangzhou_mahjong", "stake": "1"},
                            "known_players": [{"customer_id": "zhang", "display_name": "张哥"}],
                        },
                        "reason": "验证后端不会用 sender_id 脑补 organizer。",
                    }
                ],
            ),
            action_json(
                objective_status="waiting_user",
                reasoning_summary="previous_tool_results 返回 organizer_id 缺失，模型需要修正工具参数或转人工。",
                reply_to_user="我先确认一下。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorder())

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_create_game_identity",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我组一个",
            message_id="msg_runtime_create_game_identity",
        ),
        trace_id="trace_create_game_identity",
    )

    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert result.tool_results[0].error == "missing required argument: organizer_id"
    assert store.games == {}
    second_prompt = json.loads(client.calls[1]["messages"][1]["content"])
    assert second_prompt["previous_tool_results"][0]["error"] == "missing required argument: organizer_id"
    assert result.final_reply == "我先确认一下。"


def test_runtime_create_game_must_preserve_previous_search_requirement() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="先按用户明确条件查询现有局。",
                tool_calls=[
                    {
                        "name": "search_current_games",
                        "arguments": {
                            "requirement": {
                                "game_type": "hangzhou_mahjong",
                                "stake": "0.5",
                                "smoke_preference": "no_smoke",
                                "start_time_kind": "scheduled",
                                "start_time": "16:00",
                                "known_player_count": 3,
                                "needed_seats": 1,
                            }
                        },
                        "reason": "用户说四点0.5财敲173无烟，先查有没有匹配局。",
                    }
                ],
            ),
            action_json(
                objective_status="needs_tool",
                reasoning_summary="无匹配局后创建新局，但错误丢失了固定时间和人数。",
                tool_calls=[
                    {
                        "name": "create_game",
                        "arguments": {
                            "requirement": {
                                "game_type": "hangzhou_mahjong",
                                "stake": "0.5",
                                "smoke_preference": "no_smoke",
                                "start_time_kind": "asap_when_full",
                                "known_player_count": 1,
                                "needed_seats": 3,
                                "user_visible_summary": "四点0.5财敲173无烟",
                            },
                            "organizer_id": "zhang",
                            "organizer_name": "张哥",
                            "known_players": [{"customer_id": "zhang", "display_name": "张哥"}],
                        },
                        "reason": "创建待组局记录。",
                    }
                ],
            ),
            action_json(
                objective_status="needs_tool",
                reasoning_summary="后端返回一致性错误后，保留上一轮查询的明确条件重新创建。",
                tool_calls=[
                    {
                        "name": "create_game",
                        "arguments": {
                            "requirement": {
                                "game_type": "hangzhou_mahjong",
                                "stake": "0.5",
                                "smoke_preference": "no_smoke",
                                "start_time_kind": "scheduled",
                                "start_time": "16:00",
                                "known_player_count": 3,
                                "needed_seats": 1,
                                "user_visible_summary": "四点0.5财敲173无烟",
                            },
                            "organizer_id": "zhang",
                            "organizer_name": "张哥",
                            "known_players": [{"customer_id": "zhang", "display_name": "张哥"}],
                        },
                        "reason": "修正工具参数，保留四点和173。",
                    }
                ],
            ),
            action_json(
                objective_status="completed",
                reasoning_summary="已按修正后的条件创建。",
                reply_to_user="好，我帮你问问，有消息跟你说。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_requirement_consistency",
            sender_id="zhang",
            sender_name="张哥",
            text="四点0.5财敲173无烟",
            message_id="msg_runtime_requirement_consistency",
        ),
        trace_id="trace_requirement_consistency",
    )

    assert result.final_reply == "好，我帮你问问，有消息跟你说。"
    assert [item.name for item in result.tool_results] == ["search_current_games", "create_game", "create_game"]
    assert result.tool_results[0].result["requirement"]["start_time"] == "16:00"
    assert result.tool_results[1].called is False
    assert result.tool_results[1].allowed is False
    assert "tool argument consistency violation" in (result.tool_results[1].error or "")
    assert result.tool_results[1].result["reference_requirement"]["start_time_kind"] == "scheduled"
    assert result.tool_results[2].called is True
    game = next(iter(store.games.values()))
    assert game.requirement["start_time_kind"] == "scheduled"
    assert game.requirement["start_time"] == "16:00"
    assert game.requirement["known_player_count"] == 3
    assert game.requirement["needed_seats"] == 1
    third_prompt = json.loads(client.calls[2]["messages"][1]["content"])
    assert third_prompt["previous_tool_results"][0]["result"]["reference_requirement"]["start_time"] == "16:00"
    steps = trace_steps(trace.get_trace("trace_requirement_consistency"))
    assert "tool_argument_consistency_error" in steps


def test_runtime_tool_schema_rejects_empty_critical_strings_without_backend_defaults() -> None:
    store = seeded_store()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="模型提供了空 organizer_id，后端不能替换成 sender_id。",
                tool_calls=[
                    {
                        "name": "create_game",
                        "arguments": {
                            "requirement": {"game_type": "hangzhou_mahjong", "stake": "1"},
                            "organizer_id": "",
                            "organizer_name": "张哥",
                            "known_players": [{"customer_id": "zhang", "display_name": "张哥"}],
                        },
                        "reason": "验证空关键字段会被 schema 拒绝。",
                    }
                ],
            ),
            action_json(
                objective_status="waiting_user",
                reasoning_summary="previous_tool_results 返回 organizer_id 为空，模型需要重新给出合法参数。",
                reply_to_user="我先确认一下。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorder())

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_empty_organizer_identity",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我组一个",
            message_id="msg_runtime_empty_organizer_identity",
        ),
        trace_id="trace_empty_organizer_identity",
    )

    assert result.tool_results[0].called is False
    assert result.tool_results[0].error == "organizer_id must have length >= 1"
    assert store.games == {}
    assert result.final_reply == "我先确认一下。"


def test_runtime_action_contract_rejects_invalid_top_level_types_before_tools() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            json.dumps(
                {
                    "goal": "测试非法顶层类型",
                    "objective_status": "needs_tool",
                    "reasoning_summary": "needs_human 不是布尔值，badcase 不是对象。",
                    "reply_to_user": "",
                    "tool_calls": [
                        {
                            "name": "create_game",
                            "arguments": {
                                "requirement": {"game_type": "hangzhou_mahjong", "stake": "1"},
                                "organizer_id": "zhang",
                                "organizer_name": "张哥",
                                "known_players": [{"customer_id": "zhang", "display_name": "张哥"}],
                            },
                            "reason": "如果合同失败，这个工具不应执行。",
                        }
                    ],
                    "needs_human": "false",
                    "badcase": "badcase should be object",
                },
                ensure_ascii=False,
            )
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_action_contract",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我组一个",
            message_id="msg_runtime_action_contract",
        ),
        trace_id="trace_action_contract",
    )

    assert result.final_reply == "这个我先转人工确认一下。"
    assert result.tool_results == []
    assert store.games == {}
    contract_event = next(event for event in trace.get_trace("trace_action_contract") if event.step == "action_contract_error")
    assert "needs_human must be boolean" in contract_event.content["errors"]
    assert "badcase side-channel is not allowed; call record_badcase tool instead" in contract_event.content["errors"]


def test_runtime_action_contract_rejects_badcase_side_channel_before_audit_write() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="completed",
                reasoning_summary="模型错误地把 badcase 放在旁路字段。",
                reply_to_user="我记下来了。",
                badcase={
                    "reason": "旁路 badcase 不应该被 runtime 自动落库",
                    "input": {"text": "组"},
                    "actual": {"reply": "留意"},
                    "expected": {"behavior": "显式调用 record_badcase 工具"},
                },
            )
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_badcase_side_channel",
            sender_id="zhang",
            sender_name="张哥",
            text="组",
            message_id="msg_runtime_badcase_side_channel",
        ),
        trace_id="trace_badcase_side_channel",
    )

    assert result.final_reply == "这个我先转人工确认一下。"
    assert result.tool_results == []
    assert store.badcases == []
    contract_event = next(event for event in trace.get_trace("trace_badcase_side_channel") if event.step == "action_contract_error")
    assert "badcase side-channel is not allowed; call record_badcase tool instead" in contract_event.content["errors"]


def test_runtime_action_contract_requires_human_status_to_set_human_flag() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_human",
                reasoning_summary="模型说需要人工，但忘了设置 needs_human=true。",
                reply_to_user="我先确认一下。",
                needs_human=False,
            )
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_human_contract",
            sender_id="zhang",
            sender_name="张哥",
            text="这个需要人工吧",
            message_id="msg_runtime_human_contract",
        ),
        trace_id="trace_human_contract",
    )

    assert result.final_reply == "这个我先转人工确认一下。"
    contract_event = next(event for event in trace.get_trace("trace_human_contract") if event.step == "action_contract_error")
    assert "needs_human objective_status requires needs_human=true" in contract_event.content["errors"]


def test_runtime_action_contract_rejects_terminal_status_without_customer_reply() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="completed",
                reasoning_summary="模型声称完成，但没有给客户可见回复。",
                reply_to_user="   ",
            )
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_empty_terminal_reply",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我看看",
            message_id="msg_runtime_empty_terminal_reply",
        ),
        trace_id="trace_empty_terminal_reply",
    )

    assert result.final_reply == "这个我先转人工确认一下。"
    contract_event = next(event for event in trace.get_trace("trace_empty_terminal_reply") if event.step == "action_contract_error")
    assert "completed requires non-empty reply_to_user" in contract_event.content["errors"]


def test_runtime_action_contract_requires_auditable_stop_reason() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            json.dumps(
                {
                    "goal": "测试提前停止",
                    "objective_status": "completed",
                    "reasoning_summary": "模型直接说完成，但没有解释为什么可以停。",
                    "reply_to_user": "好的，我先看看。",
                    "tool_calls": [],
                    "needs_human": False,
                    "badcase": None,
                },
                ensure_ascii=False,
            )
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_stop_reason_contract",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我组一个",
            message_id="msg_runtime_stop_reason_contract",
        ),
        trace_id="trace_stop_reason_contract",
    )

    assert result.final_reply == "这个我先转人工确认一下。"
    assert result.tool_results == []
    assert store.games == {}
    contract_event = next(event for event in trace.get_trace("trace_stop_reason_contract") if event.step == "action_contract_error")
    assert "missing required key: stop_reason" in contract_event.content["errors"]
    assert "stop_reason must be object" in contract_event.content["errors"]


def test_runtime_action_contract_rejects_unknown_tool_calls_and_human_flag_conflict() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="unknown",
                reasoning_summary="模型状态不自洽：unknown 还想执行工具，并且 needs_human 标志和状态冲突。",
                reply_to_user="我没看懂，先确认一下。",
                tool_calls=[
                    {
                        "name": "create_game",
                        "arguments": {"requirement": {"game_type": "hangzhou_mahjong"}},
                        "reason": "状态 unknown 时不应该执行这个副作用工具。",
                    }
                ],
                needs_human=True,
            )
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_unknown_contract",
            sender_id="zhang",
            sender_name="张哥",
            text="随便看看",
            message_id="msg_runtime_unknown_contract",
        ),
        trace_id="trace_unknown_contract",
    )

    assert result.final_reply == "这个我先转人工确认一下。"
    assert result.tool_results == []
    assert store.games == {}
    contract_event = next(event for event in trace.get_trace("trace_unknown_contract") if event.step == "action_contract_error")
    assert "unknown must not include tool_calls" in contract_event.content["errors"]
    assert "needs_human=true requires objective_status=needs_human" in contract_event.content["errors"]


def test_runtime_action_contract_rejects_untraceable_tool_call_fields() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            json.dumps(
                {
                    "goal": "测试工具调用合同",
                    "objective_status": "needs_tool",
                    "reasoning_summary": "模型给出的工具调用缺少可审计字段。",
                    "reply_to_user": "我先看看。",
                    "tool_calls": [
                        {
                            "name": "search_current_games",
                            "arguments": {"requirement": {}},
                            "reason": "",
                        },
                        {
                            "name": "search_customers",
                            "reason": "缺少 arguments 时不能默认为空对象。",
                        },
                        {
                            "name": "create_game",
                            "arguments": {"requirement": {"game_type": "hangzhou_mahjong"}},
                            "reason": "验证 idempotency_key 类型。",
                            "idempotency_key": 123,
                        },
                    ],
                    "needs_human": False,
                    "badcase": None,
                },
                ensure_ascii=False,
            )
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_untraceable_tool_call",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我看看",
            message_id="msg_runtime_untraceable_tool_call",
        ),
        trace_id="trace_untraceable_tool_call",
    )

    assert result.final_reply == "这个我先转人工确认一下。"
    assert result.tool_results == []
    assert store.games == {}
    contract_event = next(event for event in trace.get_trace("trace_untraceable_tool_call") if event.step == "action_contract_error")
    assert "needs_tool requires empty reply_to_user" in contract_event.content["errors"]
    assert "tool_calls[1].reason is required" in contract_event.content["errors"]
    assert "tool_calls[2].arguments is required" in contract_event.content["errors"]
    assert "tool_calls[3].idempotency_key must be string or null" in contract_event.content["errors"]


def test_runtime_invalid_candidate_status_is_rejected_by_tool_schema() -> None:
    store = seeded_store()
    game, _ = store.create_game(
        conversation_id="runtime_candidate_schema",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"game_type": "hangzhou_mahjong", "stake": "1"},
        known_players=[{"customer_id": "zhang", "display_name": "张哥"}],
        trace_id="setup_candidate_schema",
    )
    drafts, _ = store.create_invite_drafts(
        game_id=game.game_id,
        invitations=[{"customer_id": "ran", "display_name": "冉姐", "message_text": "冉姐，1块，打吗？"}],
        trace_id="setup_candidate_schema",
    )
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="模型给了非法候选人状态。",
                tool_calls=[
                    {
                        "name": "record_candidate_reply",
                        "arguments": {
                            "game_id": game.game_id,
                            "customer_id": "ran",
                            "display_name": "冉姐",
                            "status": "maybe",
                        },
                        "reason": "验证非法状态不会被后端脑补。",
                    }
                ],
            ),
            action_json(
                objective_status="waiting_user",
                reasoning_summary="工具 schema 拒绝了非法 status，等待模型下一步修正或追问。",
                reply_to_user="我确认一下她到底来不来。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorder())

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_candidate_schema",
            sender_id="ran",
            sender_name="冉姐",
            text="看情况吧",
            message_id="msg_runtime_candidate_schema",
        ),
        trace_id="trace_candidate_schema",
    )

    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert "status must be one of" in (result.tool_results[0].error or "")
    assert [item.customer_id for item in store.games[game.game_id].participants] == ["zhang"]
    assert store.invite_drafts[drafts[0].draft_id].status.value == "pending_approval"
    second_prompt = json.loads(client.calls[1]["messages"][1]["content"])
    assert "status must be one of" in second_prompt["previous_tool_results"][0]["error"]
    assert result.final_reply == "我确认一下她到底来不来。"


def test_runtime_candidate_join_is_traced_and_persisted_as_state_transition(tmp_path) -> None:
    db_path = tmp_path / "agent_runtime_candidate_join.sqlite3"
    store = seeded_store(SQLiteAgentStore(db_path))
    game, _ = store.create_game(
        conversation_id="runtime_candidate_join",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"game_type": "hangzhou_mahjong", "stake": "1"},
        known_players=[{"customer_id": "zhang", "display_name": "张哥"}],
        trace_id="setup_candidate_join",
    )
    store.create_invite_drafts(
        game_id=game.game_id,
        invitations=[{"customer_id": "ran", "display_name": "冉姐", "message_text": "冉姐，1块，打吗？"}],
        trace_id="setup_candidate_join",
    )
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="候选人明确确认，模型记录候选人加入。",
                tool_calls=[
                    {
                        "name": "record_candidate_reply",
                        "arguments": {
                            "game_id": game.game_id,
                            "customer_id": "ran",
                            "display_name": "冉姐",
                            "status": "accepted",
                            "seat_count": 2,
                        },
                        "reason": "候选人确认参加，记录状态变化。",
                    }
                ],
            ),
            action_json(
                objective_status="completed",
                reasoning_summary="候选人已加入局。",
                reply_to_user="好的，加你进来了。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_candidate_join",
            sender_id="ran",
            sender_name="冉姐",
            text="可以",
            message_id="msg_runtime_candidate_join",
        ),
        trace_id="trace_candidate_join",
    )

    assert any(item.customer_id == "ran" for item in store.games[game.game_id].participants)
    assert next(item for item in store.games[game.game_id].participants if item.customer_id == "ran").seat_count == 2
    assert store.games[game.game_id].remaining_seats() == 1
    participant_transition = next(
        transition
        for transition in result.state_transitions
        if transition.entity_type == "game_participant" and transition.entity_id == f"{game.game_id}:ran"
    )
    assert participant_transition.from_status is None
    assert participant_transition.to_status == "confirmed"
    trace_transition = next(
        event
        for event in trace.get_trace("trace_candidate_join")
        if event.step == "state_transition" and event.content["entity_type"] == "game_participant"
    )
    assert trace_transition.content["entity_id"] == f"{game.game_id}:ran"
    reopened = SQLiteAgentStore(db_path)
    persisted = [
        transition
        for transition in reopened.transitions
        if transition.entity_type == "game_participant" and transition.entity_id == f"{game.game_id}:ran"
    ]
    assert len(persisted) == 1
    assert persisted[0].to_status == "confirmed"
    persisted_game = reopened.games[game.game_id]
    assert next(item for item in persisted_game.participants if item.customer_id == "ran").seat_count == 2
    assert persisted_game.remaining_seats() == 1
    assert result.final_reply == "好的，加你进来了。"


def test_runtime_candidate_decline_releases_existing_seat_and_reopens_game(tmp_path) -> None:
    db_path = tmp_path / "agent_runtime_candidate_decline.sqlite3"
    store = seeded_store(SQLiteAgentStore(db_path))
    game, _ = store.create_game(
        conversation_id="runtime_candidate_decline",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"game_type": "hangzhou_mahjong", "stake": "1"},
        known_players=[
            {"customer_id": "ran", "display_name": "冉姐", "status": "confirmed"},
            {"customer_id": "liu", "display_name": "刘姐", "status": "confirmed"},
            {"customer_id": "an", "display_name": "安姐", "status": "confirmed"},
        ],
        trace_id="setup_candidate_decline",
    )
    store.update_game_status(game_id=game.game_id, status="ready", reason="setup_full_table", trace_id="setup_candidate_decline")
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="局内人明确不打，模型记录退出并释放座位。",
                tool_calls=[
                    {
                        "name": "record_candidate_reply",
                        "arguments": {
                            "game_id": game.game_id,
                            "customer_id": "ran",
                            "display_name": "冉姐",
                            "status": "declined",
                        },
                        "reason": "局内人拒绝参加，释放座位。",
                    }
                ],
            ),
            action_json(
                objective_status="completed",
                reasoning_summary="已记录退出。",
                reply_to_user="好的，那这桌先不算你。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_candidate_decline",
            sender_id="ran",
            sender_name="冉姐",
            text="不来了",
            message_id="msg_runtime_candidate_decline",
        ),
        trace_id="trace_candidate_decline",
    )

    updated = store.games[game.game_id]
    declined = next(item for item in updated.participants if item.customer_id == "ran")
    assert declined.status == "declined"
    assert updated.remaining_seats() == 1
    assert updated.status.value == "forming"
    assert updated.requirement["known_player_count"] == 3
    assert updated.requirement["needed_seats"] == 1
    participant_transition = next(
        transition
        for transition in result.state_transitions
        if transition.entity_type == "game_participant" and transition.entity_id == f"{game.game_id}:ran"
    )
    assert participant_transition.from_status == "confirmed:seats=1"
    assert participant_transition.to_status == "declined:seats=1"
    game_transition = next(
        transition
        for transition in result.state_transitions
        if transition.entity_type == "game" and transition.reason == "seats_reopened"
    )
    assert game_transition.from_status == "ready"
    assert game_transition.to_status == "forming"
    reopened = SQLiteAgentStore(db_path)
    persisted = reopened.games[game.game_id]
    assert next(item for item in persisted.participants if item.customer_id == "ran").status == "declined"
    assert persisted.remaining_seats() == 1
    assert persisted.status.value == "forming"
    assert result.final_reply == "好的，那这桌先不算你。"


def test_runtime_outbound_message_draft_is_tool_driven_and_persisted(tmp_path) -> None:
    db_path = tmp_path / "agent_runtime_outbound_draft.sqlite3"
    store = seeded_store(SQLiteAgentStore(db_path))
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="模型决定先生成待审批外发草稿，而不是声称已经发送。",
                tool_calls=[
                    {
                        "name": "create_outbound_message_drafts",
                        "arguments": {
                            "drafts": [
                                {
                                    "recipient_id": "zhang",
                                    "recipient_name": "张哥",
                                    "channel": "console",
                                    "message_text": "好的，我先帮你问问，有消息跟你说。",
                                    "purpose": "reply_to_organizer",
                                    "metadata": {"game_context": "runtime_outbound_draft"},
                                }
                            ]
                        },
                        "reason": "把客户可见回复作为待审批草稿落库，便于后续人工审批和多通道发送。",
                    }
                ],
            ),
            action_json(
                objective_status="completed",
                reasoning_summary="待审批外发草稿已创建。",
                reply_to_user="已生成待审批回复草稿。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_outbound_draft",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我组一个",
            message_id="msg_runtime_outbound_draft",
        ),
        trace_id="trace_outbound_draft",
    )

    assert result.final_reply == "已生成待审批回复草稿。"
    assert [tool.name for tool in result.tool_results] == ["create_outbound_message_drafts"]
    assert len(store.outbound_message_drafts) == 1
    draft = next(iter(store.outbound_message_drafts.values()))
    assert draft.recipient_id == "zhang"
    assert draft.channel == "console"
    assert draft.status.value == "pending_approval"
    assert draft.message_text == "好的，我先帮你问问，有消息跟你说。"
    transition = next(
        item
        for item in result.state_transitions
        if item.entity_type == "outbound_message_draft" and item.entity_id == draft.draft_id
    )
    assert transition.to_status == "pending_approval"
    trace_transition = next(
        event
        for event in trace.get_trace("trace_outbound_draft")
        if event.step == "state_transition" and event.content["entity_type"] == "outbound_message_draft"
    )
    assert trace_transition.content["entity_id"] == draft.draft_id
    reopened = SQLiteAgentStore(db_path)
    assert len(reopened.outbound_message_drafts) == 1
    persisted = next(iter(reopened.outbound_message_drafts.values()))
    assert persisted.message_text == draft.message_text
    assert persisted.metadata["game_context"] == "runtime_outbound_draft"


def test_runtime_illegal_game_status_transition_is_rejected_by_state_machine() -> None:
    store = seeded_store()
    game, _ = store.create_game(
        conversation_id="runtime_state_machine",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"game_type": "hangzhou_mahjong", "stake": "1"},
        known_players=[{"customer_id": "zhang", "display_name": "张哥"}],
        trace_id="setup_state_machine",
    )
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="模型请求非法状态迁移。",
                tool_calls=[
                    {
                        "name": "update_game_status",
                        "arguments": {"game_id": game.game_id, "status": "finished", "reason": "model requested impossible close"},
                        "reason": "验证状态机拒绝非法迁移。",
                    }
                ],
            ),
            action_json(
                objective_status="needs_human",
                reasoning_summary="状态机拒绝非法迁移，交给人工确认。",
                reply_to_user="这个我先确认一下。",
                needs_human=True,
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorder())

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_state_machine",
            sender_id="zhang",
            sender_name="张哥",
            text="结束掉吧",
            message_id="msg_runtime_state_machine",
        ),
        trace_id="trace_state_machine",
    )

    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert "illegal game status transition: forming->finished" in (result.tool_results[0].error or "")
    assert store.games[game.game_id].status.value == "forming"
    second_prompt = json.loads(client.calls[1]["messages"][1]["content"])
    assert "illegal game status transition" in second_prompt["previous_tool_results"][0]["error"]
    assert result.final_reply == "这个我先确认一下。"


def test_runtime_tool_permission_denial_is_fed_back_to_model_without_side_effect() -> None:
    store = seeded_store()
    game, _ = store.create_game(
        conversation_id="runtime_permission",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"game_type": "hangzhou_mahjong", "stake": "1"},
        known_players=[{"customer_id": "zhang", "display_name": "张哥"}],
        trace_id="setup_permission",
    )
    trace = InMemoryTraceRecorder()
    gateway = ToolGateway(
        store=store,
        trace_recorder=trace,
        allowed_execution_modes={"read_only", "state_write", "audit_write"},
    )
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="模型请求创建邀约草稿，但当前权限禁止 draft_write。",
                tool_calls=[
                    {
                        "name": "create_invite_drafts",
                        "arguments": {
                            "game_id": game.game_id,
                            "invitations": [{"customer_id": "ran", "display_name": "冉姐", "message_text": "冉姐，1块，打吗？"}],
                        },
                        "reason": "验证权限拦截。",
                    }
                ],
            ),
            action_json(
                objective_status="needs_human",
                reasoning_summary="工具权限拒绝创建草稿，交给人工或等待配置恢复。",
                reply_to_user="我先确认一下能不能发邀约。",
                needs_human=True,
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, tool_gateway=gateway, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_permission",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我问冉姐",
            message_id="msg_runtime_permission",
        ),
        trace_id="trace_permission",
    )

    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert result.tool_results[0].error == "tool execution_mode not allowed: draft_write"
    assert store.invite_drafts == {}
    second_prompt = json.loads(client.calls[1]["messages"][1]["content"])
    assert second_prompt["previous_tool_results"][0]["error"] == "tool execution_mode not allowed: draft_write"
    permission_events = [event for event in trace.get_trace("trace_permission") if event.step == "tool_permission_checked"]
    assert permission_events[0].level == "WARN"
    assert permission_events[0].content["allowed"] is False
    assert validate_trace(trace.get_trace("trace_permission"))["complete"] is True
    assert result.final_reply == "我先确认一下能不能发邀约。"


def test_runtime_tool_handler_exception_is_traced_and_fed_back_without_side_effect() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    gateway = ToolGateway(store=store, trace_recorder=trace)

    def failing_handler(call_arg, trace_id: str, conversation_id: str, sender_id: str, sender_name: str):
        raise RuntimeError("database temporarily unavailable")

    gateway.tools["create_game"].handler = failing_handler
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="模型决定建局。",
                tool_calls=[
                    {
                        "name": "create_game",
                        "arguments": {
                            "requirement": {"game_type": "hangzhou_mahjong", "stake": "1"},
                            "organizer_id": "zhang",
                            "organizer_name": "张哥",
                            "known_players": [{"customer_id": "zhang", "display_name": "张哥"}],
                        },
                        "reason": "验证工具内部异常会进入 trace 和下一轮上下文。",
                    }
                ],
            ),
            action_json(
                objective_status="needs_human",
                reasoning_summary="工具返回 RuntimeError，模型不能声称已建局。",
                reply_to_user="这个我先确认一下。",
                needs_human=True,
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, tool_gateway=gateway, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_tool_exception",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我组一个",
            message_id="msg_runtime_tool_exception",
        ),
        trace_id="trace_tool_exception",
    )

    assert result.final_reply == "这个我先确认一下。"
    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert result.tool_results[0].error == "RuntimeError: database temporarily unavailable"
    assert store.games == {}
    exception_event = next(event for event in trace.get_trace("trace_tool_exception") if event.step == "tool_exception")
    assert exception_event.level == "ERROR"
    assert exception_event.content["error_type"] == "RuntimeError"
    second_prompt = json.loads(client.calls[1]["messages"][1]["content"])
    assert second_prompt["previous_tool_results"][0]["error"] == "RuntimeError: database temporarily unavailable"
    assert validate_trace(trace.get_trace("trace_tool_exception"))["complete"] is True


def test_runtime_budget_denial_happens_before_llm_call_and_has_complete_trace() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient([])
    runtime = AgentRuntime(
        llm_client=client,
        store=store,
        trace_recorder=trace,
        token_budget=TokenBudget(max_tokens_per_call=1, max_calls_per_turn=8),
    )

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_budget",
            sender_id="zhang",
            sender_name="张哥",
            text="通宵1块有人吗？没有就帮我组一个",
            message_id="msg_runtime_budget",
        ),
        trace_id="trace_budget",
    )

    assert result.final_reply == "这个我先转人工确认一下。"
    assert result.actions == []
    assert result.tool_results == []
    assert client.calls == []
    events = trace.get_trace("trace_budget")
    steps = trace_steps(events)
    assert "llm_prompt" in steps
    assert "budget_checked" in steps
    assert "llm_response" not in steps
    budget_event = next(event for event in events if event.step == "budget_checked")
    assert budget_event.content["allowed"] is False
    assert "single call token estimate exceeded" in budget_event.content["reason"]
    assert validate_trace(events)["complete"] is True


def test_runtime_llm_error_has_complete_trace_and_no_tool_side_effect() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    runtime = AgentRuntime(
        llm_client=FailingAgentClient(RuntimeError("llm timeout")),
        store=store,
        trace_recorder=trace,
    )

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_llm_error",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我组一个",
            message_id="msg_runtime_llm_error",
        ),
        trace_id="trace_llm_error",
    )

    events = trace.get_trace("trace_llm_error")
    steps = trace_steps(events)
    assert result.final_reply == "这个我先转人工确认一下。"
    assert result.actions == []
    assert result.tool_results == []
    assert store.games == {}
    assert "llm_prompt" in steps
    assert "llm_error" in steps
    assert "llm_response" not in steps
    assert "tool_called" not in steps
    assert validate_trace(events)["complete"] is True


def test_runtime_trace_completeness_requires_context_packing_audit() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="completed",
                reasoning_summary="模型直接回复。",
                reply_to_user="好的。",
            )
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_context_pack_trace",
            sender_id="zhang",
            sender_name="张哥",
            text="好的",
            message_id="msg_runtime_context_pack_trace",
        ),
        trace_id="trace_context_pack_required",
    )

    events = trace.get_trace("trace_context_pack_required")
    assert validate_trace(events)["complete"] is True
    without_context_packed = [event for event in events if event.step != "context_packed"]
    completeness = validate_trace(without_context_packed)
    assert completeness["complete"] is False
    assert "context_packed" in completeness["missing"]


def test_runtime_duplicate_message_id_returns_cached_result_without_reexecuting_side_effects() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = PlanningClient(store)
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)
    message = UserMessage(
        conversation_id="runtime_message_idempotency",
        sender_id="zhang",
        sender_name="张哥",
        text="通宵1块有人吗？没有就帮我组一个",
        message_id="msg_runtime_message_idempotency",
    )

    first = runtime.handle_user_message(message, trace_id="trace_message_idempotency_1")
    second = runtime.handle_user_message(message, trace_id="trace_message_idempotency_2")

    assert first.final_reply == "好的，我帮你问问，有消息跟你说。"
    assert second.final_reply == first.final_reply
    assert second.trace_id == first.trace_id
    assert len(client.calls) == 5
    assert len(store.games) == 1
    assert len(store.invite_drafts) == 2
    dedupe_events = trace.get_trace("trace_message_idempotency_2")
    dedupe_steps = trace_steps(dedupe_events)
    assert dedupe_steps == ["user_input", "message_deduplicated", "final_output"]
    assert validate_trace(dedupe_events)["complete"] is True
    dedupe_event = next(event for event in dedupe_events if event.step == "message_deduplicated")
    assert dedupe_event.content["original_trace_id"] == "trace_message_idempotency_1"
    assert dedupe_event.content["message_idempotency_key"] == message_idempotency_key(message)


def test_runtime_trace_completeness_rejects_deduplicated_trace_with_execution_steps() -> None:
    trace = InMemoryTraceRecorder()
    trace_id = "trace_bad_deduplicated_execution"
    trace.record(trace_id, "user_input", {"message": {"message_id": "msg_duplicate"}})
    trace.record(trace_id, "message_deduplicated", {"message_id": "msg_duplicate", "original_trace_id": "trace_original"})
    trace.record(trace_id, "llm_prompt", {"messages": []})
    trace.record(trace_id, "final_output", {"reply": "cached"})

    completeness = validate_trace(trace.get_trace(trace_id))

    assert completeness["complete"] is False
    assert "deduplicated_trace_must_not_execute_llm_or_tools" in completeness["missing"]


def test_runtime_message_idempotency_is_scoped_by_conversation_and_sender() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="completed",
                reasoning_summary="第一个会话正常回复。",
                reply_to_user="第一个回复。",
            ),
            action_json(
                objective_status="completed",
                reasoning_summary="第二个会话即使 message_id 相同也必须独立处理。",
                reply_to_user="第二个回复。",
            ),
            action_json(
                objective_status="completed",
                reasoning_summary="同会话不同发送者即使 message_id 相同也必须独立处理。",
                reply_to_user="第三个回复。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)
    shared_message_id = "upstream_collision_001"

    first = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_collision_a",
            sender_id="zhang",
            sender_name="张哥",
            text="第一条",
            message_id=shared_message_id,
        ),
        trace_id="trace_collision_a",
    )
    second = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_collision_b",
            sender_id="zhang",
            sender_name="张哥",
            text="第二条",
            message_id=shared_message_id,
        ),
        trace_id="trace_collision_b",
    )
    third = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_collision_a",
            sender_id="ran",
            sender_name="冉姐",
            text="第三条",
            message_id=shared_message_id,
        ),
        trace_id="trace_collision_sender",
    )

    assert first.final_reply == "第一个回复。"
    assert second.final_reply == "第二个回复。"
    assert third.final_reply == "第三个回复。"
    assert len(client.calls) == 3
    assert "message_deduplicated" not in trace_steps(trace.get_trace("trace_collision_b"))
    assert "message_deduplicated" not in trace_steps(trace.get_trace("trace_collision_sender"))


def test_runtime_new_user_message_supersedes_pending_outputs() -> None:
    store = seeded_store()
    game, _ = store.create_game(
        conversation_id="runtime_supersede",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"game_type": "hangzhou_mahjong", "stake": "1"},
        known_players=[],
        trace_id="trace_seed_game",
    )
    invite_drafts, _ = store.create_invite_drafts(
        game_id=game.game_id,
        invitations=[{"customer_id": "ran", "display_name": "冉姐", "message_text": "1块，打吗？"}],
        trace_id="trace_seed_invite",
    )
    outbound_drafts, _ = store.create_outbound_message_drafts(
        conversation_id="runtime_supersede",
        drafts=[
            {
                "recipient_id": "group",
                "recipient_name": "群",
                "channel": "console",
                "message_text": "1块有人吗？",
                "purpose": "group_broadcast",
            }
        ],
        trace_id="trace_seed_outbound",
    )
    store.append_assistant_turn(
        "runtime_supersede",
        "旧回复，老板还没发送。",
        "trace_old_reply",
        metadata={"delivery_status": "pending_operator_send", "conversation_version": 0},
    )
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="completed",
                reasoning_summary="用户补充信息后重新回复。",
                reply_to_user="好的，我重新看一下。",
            )
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_supersede",
            sender_id="zhang",
            sender_name="张哥",
            text="烟都可以，0.5 或 1 都行",
            message_id="msg_runtime_supersede",
        ),
        trace_id="trace_runtime_supersede",
    )

    assert result.final_reply == "好的，我重新看一下。"
    assert store.conversation_version("runtime_supersede") == 1
    assert store.invite_drafts[invite_drafts[0].draft_id].status.value == "superseded"
    assert store.outbound_message_drafts[outbound_drafts[0].draft_id].status.value == "superseded"
    old_reply = next(turn for turn in store.recent_turns("runtime_supersede", 10) if turn.trace_id == "trace_old_reply")
    assert old_reply.metadata["delivery_status"] == "superseded"
    new_reply = next(turn for turn in store.recent_turns("runtime_supersede", 10) if turn.trace_id == "trace_runtime_supersede" and turn.role.value == "assistant")
    assert new_reply.metadata["delivery_status"] == "pending_operator_send"
    steps = trace_steps(trace.get_trace("trace_runtime_supersede"))
    assert "conversation_version_advanced" in steps
    assert "pending_outputs_superseded" in steps
    supersede_event = next(event for event in trace.get_trace("trace_runtime_supersede") if event.step == "pending_outputs_superseded")
    assert supersede_event.content["counts"] == {
        "invite_drafts": 1,
        "outbound_message_drafts": 1,
        "assistant_replies": 1,
    }
    transition_types = [item.entity_type for item in result.state_transitions]
    assert "conversation_version" in transition_types
    assert "invite_draft" in transition_types
    assert "outbound_message_draft" in transition_types
    assert "assistant_reply" in transition_types


def test_runtime_stale_run_blocks_state_writing_tools() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()

    class StaleBeforeWriteClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: list[dict[str, str]], *, trace_id: str, timeout_seconds: float) -> str:
            self.calls += 1
            if self.calls == 1:
                store.advance_conversation_version(
                    "runtime_stale_run",
                    trace_id="trace_external_supplement",
                    reason="simulated_new_user_message_during_running_agent",
                )
                return action_json(
                    objective_status="needs_tool",
                    reasoning_summary="旧 run 试图创建局。",
                    tool_calls=[
                        {
                            "name": "create_game",
                            "arguments": {
                                "requirement": {"game_type": "hangzhou_mahjong", "stake": "1"},
                                "organizer_id": "zhang",
                                "organizer_name": "张哥",
                                "known_players": [],
                            },
                            "reason": "创建待组局记录。",
                        }
                    ],
                )
            return action_json(
                objective_status="completed",
                reasoning_summary="不应到达第二轮。",
                reply_to_user="不应该回复。",
            )

    runtime = AgentRuntime(llm_client=StaleBeforeWriteClient(), store=store, trace_recorder=trace)

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_stale_run",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我组一个",
            message_id="msg_runtime_stale_run",
        ),
        trace_id="trace_runtime_stale_run",
    )

    assert result.final_reply == ""
    assert store.conversation_version("runtime_stale_run") == 2
    assert store.games == {}
    assert result.tool_results[0].name == "create_game"
    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert "stale run" in str(result.tool_results[0].error)
    steps = trace_steps(trace.get_trace("trace_runtime_stale_run"))
    assert "conversation_run_stale" in steps
    final_event = next(event for event in trace.get_trace("trace_runtime_stale_run") if event.step == "final_output")
    assert final_event.content["reason"] == "conversation_run_stale"


def test_runtime_concurrent_duplicate_message_id_serializes_and_deduplicates_side_effects() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = PlanningClient(store)
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)
    message = UserMessage(
        conversation_id="runtime_concurrent_message",
        sender_id="zhang",
        sender_name="张哥",
        text="通宵1块有人吗？没有就帮我组一个",
        message_id="msg_runtime_concurrent_message",
    )
    start = threading.Barrier(3)
    results: dict[str, Any] = {}
    errors: list[BaseException] = []

    def worker(trace_id: str) -> None:
        try:
            start.wait()
            results[trace_id] = runtime.handle_user_message(message, trace_id=trace_id)
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=("trace_concurrent_message_1",)),
        threading.Thread(target=worker, args=("trace_concurrent_message_2",)),
    ]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(results) == 2
    assert len({result.trace_id for result in results.values()}) == 1
    assert len(client.calls) == 5
    assert len(store.games) == 1
    assert len(store.invite_drafts) == 2
    duplicate_traces = [
        trace_id
        for trace_id in results
        if "message_deduplicated" in trace_steps(trace.get_trace(trace_id))
    ]
    assert len(duplicate_traces) == 1
    duplicate_events = trace.get_trace(duplicate_traces[0])
    assert trace_steps(duplicate_events) == ["user_input", "message_deduplicated", "final_output"]
    assert validate_trace(duplicate_events)["complete"] is True


def test_runtime_token_budget_is_isolated_per_concurrent_message_turn() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = TwoStepBarrierClient(expected_first_calls=2)
    runtime = AgentRuntime(
        llm_client=client,
        store=store,
        trace_recorder=trace,
        token_budget=TokenBudget(max_tokens_per_call=24_000, max_calls_per_turn=2),
    )
    start = threading.Barrier(3)
    results: dict[str, Any] = {}
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            start.wait()
            results[f"trace_budget_isolation_{index}"] = runtime.handle_user_message(
                UserMessage(
                    conversation_id=f"runtime_budget_isolation_{index}",
                    sender_id=f"user_{index}",
                    sender_name=f"用户{index}",
                    text="先查一下，没有就回复我",
                    message_id=f"msg_runtime_budget_isolation_{index}",
                ),
                trace_id=f"trace_budget_isolation_{index}",
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in (1, 2)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert errors == []
    assert sorted(results) == ["trace_budget_isolation_1", "trace_budget_isolation_2"]
    assert all(result.final_reply == "查过了，先这样回复。" for result in results.values())
    assert all(len(result.actions) == 2 for result in results.values())
    for trace_id in sorted(results):
        events = trace.get_trace(trace_id)
        assert validate_trace(events)["complete"] is True
        budget_events = [event for event in events if event.step == "budget_checked"]
        assert len(budget_events) == 2
        assert all(event.content["allowed"] is True for event in budget_events)
    assert runtime.token_budget.calls_this_turn == 0


def test_runtime_tool_gateway_serializes_concurrent_same_backend_idempotency_key() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    gateway = ToolGateway(store=store, trace_recorder=trace)
    call = ToolCall(
        name="create_game",
        arguments={
            "requirement": {"game_type": "hangzhou_mahjong", "stake": "1", "user_visible_summary": "杭麻 1块"},
            "organizer_id": "zhang",
            "organizer_name": "张哥",
            "known_players": [{"customer_id": "zhang", "display_name": "张哥"}],
        },
        idempotency_key="model-key-should-not-win",
    )
    original_handler = gateway.tools["create_game"].handler
    execution_count = 0
    execution_count_lock = threading.Lock()

    def slow_handler(call_arg, trace_id: str, conversation_id: str, sender_id: str, sender_name: str):
        nonlocal execution_count
        with execution_count_lock:
            execution_count += 1
        time.sleep(0.05)
        return original_handler(call_arg, trace_id, conversation_id, sender_id, sender_name)

    gateway.tools["create_game"].handler = slow_handler
    start = threading.Barrier(3)
    results = []
    results_lock = threading.Lock()

    def worker(trace_id: str) -> None:
        start.wait()
        result = gateway.execute(
            call,
            trace_id=trace_id,
            conversation_id="runtime_concurrent_tool",
            sender_id="zhang",
            sender_name="张哥",
            step_index=101,
            source_message_id="msg_runtime_concurrent_tool",
        )
        with results_lock:
            results.append(result)

    threads = [
        threading.Thread(target=worker, args=("trace_concurrent_tool_1",)),
        threading.Thread(target=worker, args=("trace_concurrent_tool_2",)),
    ]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join()

    assert execution_count == 1
    assert len(results) == 2
    assert len(store.games) == 1
    assert sorted(result.deduplicated for result in results) == [False, True]
    assert len({result.idempotency_key for result in results}) == 1
    assert all(
        result.idempotency_key.startswith(
            "conversation:runtime_concurrent_tool:sender:zhang:message:msg_runtime_concurrent_tool:tool:create_game:args:"
        )
        for result in results
    )
    hit_values = []
    for trace_id in ("trace_concurrent_tool_1", "trace_concurrent_tool_2"):
        events = trace.get_trace(trace_id)
        hit_values.extend(event.content["hit"] for event in events if event.step == "tool_idempotency_checked")
    assert sorted(hit_values) == [False, True]


def test_runtime_tool_idempotency_is_scoped_by_conversation_and_sender() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    gateway = ToolGateway(store=store, trace_recorder=trace)
    call = ToolCall(
        name="create_game",
        arguments={
            "requirement": {"game_type": "hangzhou_mahjong", "stake": "1"},
            "organizer_id": "zhang",
            "organizer_name": "张哥",
            "known_players": [{"customer_id": "zhang", "display_name": "张哥"}],
        },
        idempotency_key="model-key-cannot-collapse-conversations",
    )
    first = gateway.execute(
        call,
        trace_id="trace_tool_scope_a",
        conversation_id="runtime_tool_scope_a",
        sender_id="zhang",
        sender_name="张哥",
        step_index=101,
        source_message_id="same_upstream_tool_message",
    )
    second = gateway.execute(
        call,
        trace_id="trace_tool_scope_b",
        conversation_id="runtime_tool_scope_b",
        sender_id="zhang",
        sender_name="张哥",
        step_index=101,
        source_message_id="same_upstream_tool_message",
    )
    third = gateway.execute(
        call,
        trace_id="trace_tool_scope_sender",
        conversation_id="runtime_tool_scope_a",
        sender_id="ran",
        sender_name="冉姐",
        step_index=101,
        source_message_id="same_upstream_tool_message",
    )

    assert first.called is True
    assert second.called is True
    assert third.called is True
    assert first.deduplicated is False
    assert second.deduplicated is False
    assert third.deduplicated is False
    assert len(store.games) == 3
    assert len({first.idempotency_key, second.idempotency_key, third.idempotency_key}) == 3
    assert "conversation:runtime_tool_scope_a:sender:zhang:" in (first.idempotency_key or "")
    assert "conversation:runtime_tool_scope_b:sender:zhang:" in (second.idempotency_key or "")
    assert "conversation:runtime_tool_scope_a:sender:ran:" in (third.idempotency_key or "")


def test_runtime_tool_gateway_claims_idempotency_before_executing_side_effect(tmp_path) -> None:
    store = seeded_store(SQLiteAgentStore(tmp_path / "agent_runtime_tool_claim.sqlite3"))
    trace = InMemoryTraceRecorder()
    gateway = ToolGateway(store=store, trace_recorder=trace)
    call = ToolCall(
        name="create_game",
        arguments={
            "requirement": {"game_type": "hangzhou_mahjong", "stake": "1"},
            "organizer_id": "zhang",
            "organizer_name": "张哥",
            "known_players": [{"customer_id": "zhang", "display_name": "张哥"}],
        },
        idempotency_key="model-key-ignored-by-backend-when-message-id-exists",
    )

    result = gateway.execute(
        call,
        trace_id="trace_tool_claim",
        conversation_id="runtime_tool_claim",
        sender_id="zhang",
        sender_name="张哥",
        step_index=101,
        source_message_id="msg_runtime_tool_claim",
    )

    assert result.called is True
    assert result.allowed is True
    assert len(store.games) == 1
    assert result.idempotency_key
    persisted = store.idempotent_result(result.idempotency_key)
    assert persisted is not None
    assert persisted.called is True
    assert persisted.result["game"]["organizer_id"] == "zhang"
    claim_event = next(event for event in trace.get_trace("trace_tool_claim") if event.step == "tool_idempotency_claimed")
    assert claim_event.content["claimed"] is True
    assert claim_event.content["idempotency_key"] == result.idempotency_key
    steps = trace_steps(trace.get_trace("trace_tool_claim"))
    assert steps == [
        "tool_gateway_received",
        "tool_idempotency_checked",
        "tool_definition_checked",
        "tool_schema_checked",
        "tool_permission_checked",
        "tool_idempotency_claimed",
        "tool_gateway_completed",
    ]


def test_runtime_sqlite_idempotency_claim_is_atomic_across_store_instances(tmp_path) -> None:
    db_path = tmp_path / "agent_runtime_claim_atomic.sqlite3"
    first_store = SQLiteAgentStore(db_path)
    second_store = SQLiteAgentStore(db_path)
    key = "message:shared:tool:create_game:args:same"
    first_claim = ToolResult(
        name="create_game",
        called=False,
        allowed=True,
        result={"idempotency_status": "claimed", "claimed_by_trace_id": "trace_first"},
        error="tool execution is already in progress for this idempotency key",
        idempotency_key=key,
    )
    second_claim = ToolResult(
        name="create_game",
        called=False,
        allowed=True,
        result={"idempotency_status": "claimed", "claimed_by_trace_id": "trace_second"},
        error="tool execution is already in progress for this idempotency key",
        idempotency_key=key,
    )

    acquired, existing = first_store.claim_idempotent_result(key, first_claim)
    duplicate_acquired, duplicate_existing = second_store.claim_idempotent_result(key, second_claim)

    assert acquired is True
    assert existing is None
    assert duplicate_acquired is False
    assert duplicate_existing is not None
    assert duplicate_existing.result["claimed_by_trace_id"] == "trace_first"

    final = ToolResult(
        name="create_game",
        called=True,
        allowed=True,
        result={"game": {"game_id": "game_claimed"}},
        idempotency_key=key,
    )
    first_store.remember_result(key, final)
    persisted = second_store.idempotent_result(key)
    assert persisted is not None
    assert persisted.called is True
    assert persisted.result["game"]["game_id"] == "game_claimed"


def test_runtime_jsonl_trace_is_structured_and_replayable(tmp_path) -> None:
    store = seeded_store()
    trace = JsonlTraceRecorder(tmp_path / "agent_runtime_trace.log")
    client = PlanningClient(store)
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_jsonl_trace",
            sender_id="zhang",
            sender_name="张哥",
            text="通宵1块有人吗？没有就帮我组一个",
            message_id="msg_runtime_jsonl_trace",
        ),
        trace_id="trace_jsonl_trace",
    )

    events = trace.get_trace("trace_jsonl_trace")
    steps = trace_steps(events)
    assert validate_trace(events)["complete"] is True
    assert "raw_log_line" not in steps
    assert "llm_prompt" in steps
    assert "llm_response" in steps
    assert "tool_called" in steps
    assert "tool_result" in steps
    assert "state_transition" in steps
    prompt_payload = json.loads(next(event for event in events if event.step == "llm_prompt").content["messages"][1]["content"])
    assert prompt_payload["runtime"] == "mahjong_agent_runtime"


def test_runtime_trace_completeness_requires_action_proposed_after_llm_success() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="completed",
                reasoning_summary="模型直接回复。",
                reply_to_user="好的。",
            )
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_trace_action",
            sender_id="zhang",
            sender_name="张哥",
            text="好的",
            message_id="msg_runtime_trace_action",
        ),
        trace_id="trace_action_required",
    )

    events = trace.get_trace("trace_action_required")
    assert validate_trace(events)["complete"] is True
    without_action = [event for event in events if event.step != "action_proposed"]
    completeness = validate_trace(without_action)
    assert completeness["complete"] is False
    assert "action_proposed" in completeness["missing"]


def test_runtime_trace_completeness_requires_state_transition_for_tool_side_effects() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="模型决定建局。",
                tool_calls=[
                    {
                        "name": "create_game",
                        "arguments": {
                            "requirement": {"game_type": "hangzhou_mahjong", "stake": "1"},
                            "organizer_id": "zhang",
                            "organizer_name": "张哥",
                            "known_players": [{"customer_id": "zhang", "display_name": "张哥"}],
                        },
                        "reason": "创建待组局记录。",
                    }
                ],
            ),
            action_json(
                objective_status="completed",
                reasoning_summary="建局完成。",
                reply_to_user="好的，我帮你问问。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_trace_transition",
            sender_id="zhang",
            sender_name="张哥",
            text="帮我组一个",
            message_id="msg_runtime_trace_transition",
        ),
        trace_id="trace_transition_required",
    )

    events = trace.get_trace("trace_transition_required")
    assert validate_trace(events)["complete"] is True
    without_transition = [event for event in events if event.step not in {"state_transition", "state_transition_replayed"}]
    completeness = validate_trace(without_transition)
    assert completeness["complete"] is False
    assert "state_transition" in completeness["missing"]


def test_runtime_sqlite_store_persists_runtime_state_and_idempotency(tmp_path) -> None:
    db_path = tmp_path / "agent_runtime.sqlite3"
    store = seeded_store(SQLiteAgentStore(db_path))
    trace = InMemoryTraceRecorder()
    client = PlanningClient(store)
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)
    message = UserMessage(
        conversation_id="runtime_sqlite",
        sender_id="zhang",
        sender_name="张哥",
        text="通宵1块有人吗？没有就帮我组一个",
        message_id="msg_runtime_sqlite_persist",
    )

    result = runtime.handle_user_message(message, trace_id="trace_sqlite_1")

    assert result.final_reply == "好的，我帮你问问，有消息跟你说。"
    assert len(store.games) == 1
    assert len(store.invite_drafts) == 2
    assert result.tool_results[0].idempotency_key

    reopened = SQLiteAgentStore(db_path)
    assert len(reopened.customers) == 3
    assert len(reopened.games) == 1
    assert len(reopened.invite_drafts) == 2
    assert len(reopened.transitions) >= 3
    assert len(reopened.recent_turns("runtime_sqlite")) >= 3
    assert reopened.idempotent_result(result.tool_results[0].idempotency_key) is not None

    cached_client = StaticAgentClient([])
    runtime_after_restart = AgentRuntime(
        llm_client=cached_client,
        store=reopened,
        trace_recorder=InMemoryTraceRecorder(),
    )
    cached = runtime_after_restart.handle_user_message(message, trace_id="trace_sqlite_2")

    assert cached.final_reply == result.final_reply
    assert cached_client.calls == []
    assert len(reopened.games) == 1
    assert reopened.idempotent_message_result(message_idempotency_key(message)) is not None


def test_runtime_sqlite_store_persists_badcases_from_tool(tmp_path) -> None:
    store = seeded_store(SQLiteAgentStore(tmp_path / "agent_runtime_badcase.sqlite3"))
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="模型主动归档 badcase。",
                tool_calls=[
                    {
                        "name": "record_badcase",
                        "arguments": {
                            "reason": "测试回复不合适",
                            "input": {"text": "组"},
                            "actual": {"reply": "留意"},
                            "expected": {"reply": "应该继续规划"},
                        },
                        "reason": "记录评测样本。",
                    }
                ],
            ),
            action_json(
                objective_status="completed",
                reasoning_summary="badcase 已记录。",
                reply_to_user="我记下来了。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorder())

    runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_badcase",
            sender_id="zhang",
            sender_name="张哥",
            text="组",
            message_id="msg_runtime_badcase",
        ),
        trace_id="trace_badcase",
    )

    reopened = SQLiteAgentStore(tmp_path / "agent_runtime_badcase.sqlite3")
    assert len(reopened.badcases) == 1
    assert reopened.badcases[0]["reason"] == "测试回复不合适"


def test_runtime_record_badcase_requires_eval_contract_before_persisting() -> None:
    store = seeded_store()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="模型错误地提交空 badcase。",
                tool_calls=[
                    {
                        "name": "record_badcase",
                        "arguments": {"reason": "只有原因，没有输入实际期望"},
                        "reason": "验证 badcase 工具不会记录不可评测样本。",
                    }
                ],
            ),
            action_json(
                objective_status="needs_tool",
                reasoning_summary="previous_tool_results 返回缺 input，模型修正 badcase 参数。",
                tool_calls=[
                    {
                        "name": "record_badcase",
                        "arguments": {
                            "reason": "回复停在留意没有继续规划",
                            "input": {"text": "组"},
                            "actual": {"reply": "好的，我先留意下。"},
                            "expected": {"behavior": "继续结合上下文规划或追问关键缺口"},
                        },
                        "reason": "补齐可评测 badcase 字段后再次记录。",
                    }
                ],
            ),
            action_json(
                objective_status="completed",
                reasoning_summary="badcase 已记录。",
                reply_to_user="我记下来了。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=InMemoryTraceRecorder())

    result = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_badcase_contract",
            sender_id="zhang",
            sender_name="张哥",
            text="组",
            message_id="msg_runtime_badcase_contract",
        ),
        trace_id="trace_badcase_contract",
    )

    assert result.tool_results[0].called is False
    assert result.tool_results[0].allowed is False
    assert result.tool_results[0].error == "missing required argument: input"
    second_prompt = json.loads(client.calls[1]["messages"][1]["content"])
    assert second_prompt["previous_tool_results"][0]["error"] == "missing required argument: input"
    assert len(store.badcases) == 1
    assert store.badcases[0]["expected"]["behavior"] == "继续结合上下文规划或追问关键缺口"


def test_runtime_context_checkpoint_is_tool_driven_and_survives_context_packing() -> None:
    store = seeded_store()
    trace = InMemoryTraceRecorder()
    client = StaticAgentClient(
        [
            action_json(
                objective_status="needs_tool",
                reasoning_summary="用户补充了长期任务事实，模型显式写入 checkpoint。",
                tool_calls=[
                    {
                        "name": "update_context_checkpoint",
                        "arguments": {
                            "summary": "张哥正在让老板帮忙组一桌杭麻，倾向人齐开，烟况都可。",
                            "facts": {
                                "organizer_id": "zhang",
                                "game_type": "hangzhou_mahjong",
                                "start_time_kind": "asap_when_full",
                                "smoke_preference": "any",
                            },
                            "open_questions": ["还需要确认档位和当前人数"],
                        },
                        "reason": "这些事实需要跨多轮保留，避免上下文窗口裁剪后丢失。",
                    }
                ],
            ),
            action_json(
                objective_status="waiting_user",
                reasoning_summary="checkpoint 已更新，继续追问缺失信息。",
                reply_to_user="行，那你这边几个人，打多大？",
            ),
            action_json(
                objective_status="completed",
                reasoning_summary="下一轮能看到 checkpoint，模型无需后端补语义。",
                reply_to_user="收到。",
            ),
        ]
    )
    runtime = AgentRuntime(llm_client=client, store=store, trace_recorder=trace)

    first = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_checkpoint",
            sender_id="zhang",
            sender_name="张哥",
            text="人齐开吧，有烟无烟都行",
            message_id="msg_runtime_checkpoint_1",
        ),
        trace_id="trace_checkpoint_1",
    )

    checkpoint = store.get_conversation_checkpoint("runtime_checkpoint")
    assert checkpoint is not None
    assert checkpoint.summary == "张哥正在让老板帮忙组一桌杭麻，倾向人齐开，烟况都可。"
    assert checkpoint.facts["start_time_kind"] == "asap_when_full"
    assert first.tool_results[0].name == "update_context_checkpoint"
    assert any(
        event.step == "state_transition" and event.content["entity_type"] == "conversation_checkpoint"
        for event in trace.get_trace("trace_checkpoint_1")
    )

    runtime.context_builder.packing_policy.max_recent_conversation_tokens = 1
    second = runtime.handle_user_message(
        UserMessage(
            conversation_id="runtime_checkpoint",
            sender_id="zhang",
            sender_name="张哥",
            text="一个人，1块的",
            message_id="msg_runtime_checkpoint_2",
        ),
        trace_id="trace_checkpoint_2",
    )

    second_prompt = json.loads(client.calls[2]["messages"][1]["content"])
    assert second.final_reply == "收到。"
    assert second_prompt["conversation_checkpoint"]["summary"] == checkpoint.summary
    assert second_prompt["conversation_checkpoint"]["facts"]["smoke_preference"] == "any"
    assert second_prompt["context_budget"]["conversation_checkpoint_present"] is True
    assert second_prompt["context_budget"]["omitted_turn_count"] >= 1
    assert len(second_prompt["recent_conversation"]) == 1


def test_runtime_sqlite_store_persists_context_checkpoint(tmp_path) -> None:
    db_path = tmp_path / "agent_runtime_checkpoint.sqlite3"
    store = SQLiteAgentStore(db_path)

    checkpoint, transition = store.upsert_conversation_checkpoint(
        conversation_id="runtime_checkpoint_sqlite",
        summary="张哥当前在组 1 块杭麻，人齐开。",
        facts={"organizer_id": "zhang", "stake": "1", "start_time_kind": "asap_when_full"},
        open_questions=["烟况是否都可"],
        trace_id="trace_checkpoint_sqlite",
    )

    assert checkpoint.source_trace_id == "trace_checkpoint_sqlite"
    assert transition.entity_type == "conversation_checkpoint"
    reopened = SQLiteAgentStore(db_path)
    persisted = reopened.get_conversation_checkpoint("runtime_checkpoint_sqlite")
    assert persisted is not None
    assert persisted.summary == "张哥当前在组 1 块杭麻，人齐开。"
    assert persisted.facts["stake"] == "1"
    assert persisted.open_questions == ["烟况是否都可"]
    assert any(item.entity_type == "conversation_checkpoint" for item in reopened.transitions)


def test_runtime_store_search_customers_accepts_list_alias_requirement_fields() -> None:
    store = seeded_store()

    candidates = store.search_customers(
        {
            "preferred_games": ["hangzhou_mahjong"],
            "preferred_stakes": ["1"],
            "smoke_preference": "any",
        },
        exclude_customer_ids=["zhang"],
        limit=10,
    )

    assert [item["customer"]["customer_id"] for item in candidates] == ["ran", "he"]
    assert "game_type_matched" in candidates[0]["reasons"]
    assert "stake_matched" in candidates[0]["reasons"]


def test_runtime_sqlite_search_customers_accepts_list_alias_requirement_fields(tmp_path) -> None:
    store = seeded_store(SQLiteAgentStore(tmp_path / "agent_runtime_aliases.sqlite3"))

    candidates = store.search_customers(
        {
            "preferred_games": ["hangzhou_mahjong"],
            "preferred_stakes": ["1"],
            "smoke_preference": "any",
        },
        exclude_customer_ids=["zhang"],
        limit=10,
    )

    assert [item["customer"]["customer_id"] for item in candidates] == ["ran", "he"]
    assert "game_type_matched" in candidates[0]["reasons"]
    assert "stake_matched" in candidates[0]["reasons"]


def test_runtime_requirement_splits_stake_base_and_cap_score() -> None:
    normalized = normalize_requirement({"game_type": "sichuan_mahjong", "stake": "2-32"})

    assert normalized["stake"] == "2"
    assert normalized["base_stake"] == 2.0
    assert normalized["cap_score"] == 32.0
    assert normalized["stake_label"] == "2-32"
    assert normalized["level"] == "2-32"

    shorthand = normalize_requirement({"game_type": "sichuan_mahjong", "stake": "216"})
    assert shorthand["stake"] == "2"
    assert shorthand["base_stake"] == 2.0
    assert shorthand["cap_score"] == 16.0
    assert shorthand["stake_label"] == "2-16"

    compact_cap = normalize_requirement({"game_type": "sichuan_mahjong", "stake": "232"})
    assert compact_cap["stake"] == "2"
    assert compact_cap["base_stake"] == 2.0
    assert compact_cap["cap_score"] == 32.0
    assert compact_cap["stake_label"] == "2-32"

    ten_cap = normalize_requirement({"game_type": "sichuan_mahjong", "stake": "1032"})
    assert ten_cap["stake"] == "10"
    assert ten_cap["base_stake"] == 10.0
    assert ten_cap["cap_score"] == 32.0
    assert ten_cap["stake_label"] == "10-32"


def test_runtime_store_search_current_games_respects_split_cap_score() -> None:
    store = seeded_store()
    game, _ = store.create_game(
        conversation_id="cap_eval",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"game_type": "sichuan_mahjong", "stake": "2-32", "smoke_preference": "any"},
        known_players=[{"customer_id": "zhang", "display_name": "张哥", "seat_count": 1}],
        trace_id="trace_cap",
    )

    assert game.requirement["stake"] == "2"
    assert game.requirement["base_stake"] == 2.0
    assert game.requirement["cap_score"] == 32.0
    assert game.requirement["stake_label"] == "2-32"

    exact_matches = store.search_current_games({"game_type": "sichuan_mahjong", "stake": "2-32"}, limit=5)
    assert [item["game"]["game_id"] for item in exact_matches] == [game.game_id]
    assert "stake_matched" in exact_matches[0]["reasons"]
    assert "cap_score_matched" in exact_matches[0]["reasons"]

    base_only_matches = store.search_current_games({"game_type": "sichuan_mahjong", "stake": "2"}, limit=5)
    assert [item["game"]["game_id"] for item in base_only_matches] == [game.game_id]

    cap_mismatch_matches = store.search_current_games({"game_type": "sichuan_mahjong", "stake": "2-16"}, limit=5)
    assert cap_mismatch_matches == []


def test_runtime_sqlite_store_persists_split_stake_base_and_cap_score(tmp_path) -> None:
    store = seeded_store(SQLiteAgentStore(tmp_path / "agent_runtime_stake_cap.sqlite3"))
    game, _ = store.create_game(
        conversation_id="cap_eval_sqlite",
        organizer_id="zhang",
        organizer_name="张哥",
        requirement={"game_type": "sichuan_mahjong", "stake": "216"},
        known_players=[{"customer_id": "zhang", "display_name": "张哥", "seat_count": 1}],
        trace_id="trace_cap_sqlite",
    )

    persisted = store.require_game(game.game_id)
    assert persisted.requirement["stake"] == "2"
    assert persisted.requirement["base_stake"] == 2.0
    assert persisted.requirement["cap_score"] == 16.0
    assert persisted.requirement["stake_label"] == "2-16"


def seeded_store(store=None):
    store = store or InMemoryAgentStore()
    store.upsert_customer(
        CustomerProfile(
            customer_id="zhang",
            display_name="张哥",
            gender="男",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["0.5", "1"],
            smoke_preference="any",
            response_score=0.9,
        )
    )
    store.upsert_customer(
        CustomerProfile(
            customer_id="ran",
            display_name="冉姐",
            gender="女",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["1"],
            smoke_preference="any",
            response_score=0.9,
        )
    )
    store.upsert_customer(
        CustomerProfile(
            customer_id="he",
            display_name="何哥",
            gender="男",
            preferred_games=["hangzhou_mahjong"],
            preferred_stakes=["1"],
            smoke_preference="any",
            response_score=0.8,
        )
    )
    return store


class PlanningClient:
    def __init__(self, store: InMemoryAgentStore) -> None:
        self.store = store
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: list[dict[str, str]], *, trace_id: str, timeout_seconds: float) -> str:
        self.calls.append({"messages": messages, "trace_id": trace_id, "timeout_seconds": timeout_seconds})
        call_index = len(self.calls)
        if call_index == 1:
            return action_json(
                objective_status="needs_tool",
                reasoning_summary="需要先查询当前局池。",
                tool_calls=[
                    {
                        "name": "search_current_games",
                        "arguments": {"requirement": {"game_type": "hangzhou_mahjong", "stake": "1", "duration_kind": "overnight"}, "limit": 5},
                        "reason": "回答现有局前先查状态。",
                    }
                ],
            )
        if call_index == 2:
            return action_json(
                objective_status="needs_tool",
                reasoning_summary="没有现成局，用户也要求帮忙组。",
                tool_calls=[
                    {
                        "name": "create_game",
                        "arguments": {
                            "requirement": {
                                "game_type": "hangzhou_mahjong",
                                "stake": "1",
                                "duration_kind": "overnight",
                                "user_visible_summary": "杭麻 1块 通宵",
                            },
                            "organizer_id": "zhang",
                            "organizer_name": "张哥",
                            "known_players": [{"customer_id": "zhang", "display_name": "张哥"}],
                        },
                        "reason": "创建待组局记录。",
                    }
                ],
            )
        if call_index == 3:
            return action_json(
                objective_status="needs_tool",
                reasoning_summary="已有待组局，继续找候选人。",
                tool_calls=[
                    {
                        "name": "search_customers",
                        "arguments": {
                            "requirement": {"game_type": "hangzhou_mahjong", "stake": "1", "duration_kind": "overnight"},
                            "exclude_customer_ids": ["zhang"],
                            "limit": 2,
                        },
                        "reason": "搜索匹配候选人。",
                    }
                ],
            )
        if call_index == 4:
            game_id = next(iter(self.store.games.values())).game_id
            return action_json(
                objective_status="needs_tool",
                reasoning_summary="候选人已返回，生成待审批邀约草稿。",
                tool_calls=[
                    {
                        "name": "create_invite_drafts",
                        "arguments": {
                            "game_id": game_id,
                            "invitations": [
                                {"customer_id": "ran", "display_name": "冉姐", "message_text": "冉姐，1块通宵，打吗？"},
                                {"customer_id": "he", "display_name": "何哥", "message_text": "何哥，1块通宵，打吗？"},
                            ],
                        },
                        "reason": "只创建待审批草稿，不直接发送。",
                    }
                ],
            )
        return action_json(
            objective_status="completed",
            reasoning_summary="待审批草稿已创建，向发起人自然确认。",
            reply_to_user="好的，我帮你问问，有消息跟你说。",
        )


class FailingAgentClient:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages: list[dict[str, str]], *, trace_id: str, timeout_seconds: float) -> str:
        self.calls.append({"messages": messages, "trace_id": trace_id, "timeout_seconds": timeout_seconds})
        raise self.exc


class TwoStepBarrierClient:
    def __init__(self, *, expected_first_calls: int) -> None:
        self.first_call_barrier = threading.Barrier(expected_first_calls)
        self.calls_by_trace: dict[str, int] = {}
        self.calls: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def complete(self, messages: list[dict[str, str]], *, trace_id: str, timeout_seconds: float) -> str:
        with self.lock:
            call_number = self.calls_by_trace.get(trace_id, 0) + 1
            self.calls_by_trace[trace_id] = call_number
            self.calls.append(
                {
                    "messages": messages,
                    "trace_id": trace_id,
                    "timeout_seconds": timeout_seconds,
                    "call_number": call_number,
                }
            )
        if call_number == 1:
            self.first_call_barrier.wait(timeout=5)
            return action_json(
                objective_status="needs_tool",
                reasoning_summary="先查当前局池。",
                tool_calls=[
                    {
                        "name": "search_current_games",
                        "arguments": {"requirement": {"game_type": "hangzhou_mahjong"}, "limit": 1},
                        "reason": "验证并发预算隔离时先执行一个只读工具。",
                    }
                ],
            )
        return action_json(
            objective_status="completed",
            reasoning_summary="第二轮模型调用可以正常完成。",
            reply_to_user="查过了，先这样回复。",
        )


def action_json(
    *,
    objective_status: str,
    reasoning_summary: str = "test",
    reply_to_user: str = "",
    tool_calls: list[dict[str, Any]] | None = None,
    needs_human: bool = False,
    stop_reason: dict[str, Any] | None = None,
    badcase: dict[str, Any] | None = None,
) -> str:
    if stop_reason is None:
        if objective_status == "needs_tool":
            stop_reason = {
                "can_stop": False,
                "why": "还需要执行工具才能继续。",
                "pending_work": [call.get("name", "tool") for call in tool_calls or []],
                "depends_on_tool_results": False,
            }
        else:
            stop_reason = {
                "can_stop": True,
                "why": "本轮已经可以停止并回复用户。",
                "pending_work": [],
                "depends_on_tool_results": False,
            }
    return json.dumps(
        {
            "goal": "测试 Agent Runtime 主链路",
            "objective_status": objective_status,
            "reasoning_summary": reasoning_summary,
            "reply_to_user": reply_to_user,
            "tool_calls": tool_calls or [],
            "needs_human": needs_human,
            "stop_reason": stop_reason,
            "badcase": badcase,
        },
        ensure_ascii=False,
    )
