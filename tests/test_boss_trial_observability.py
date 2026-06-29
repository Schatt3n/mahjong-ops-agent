from __future__ import annotations

import importlib.util
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_boss_trial_app.py"
LLM_ENV_KEYS = [
    "MAHJONG_LLM_API_KEY",
    "MAHJONG_LLM_PROVIDER",
    "MAHJONG_LLM_MODEL",
    "MAHJONG_LLM_BASE_URL",
    "MAHJONG_CONTROLLED_AGENT_MODE",
    "MAHJONG_LLM_REQUIRED_FOR_SIDE_EFFECT_TOOLS",
    "MAHJONG_LLM_REQUIRED_FOR_STATE_WRITES",
    "MAHJONG_LLM_MAX_COMPLETION_TOKENS",
    "OPENAI_API_KEY",
    "DASHSCOPE_API_KEY",
]


def load_boss_trial_module():
    spec = importlib.util.spec_from_file_location("run_boss_trial_app", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def without_llm_env():
    saved = {key: os.environ.get(key) for key in LLM_ENV_KEYS}
    for key in LLM_ENV_KEYS:
        os.environ.pop(key, None)
    return saved


def restore_env(saved) -> None:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_io_log_line_uses_required_trace_time_level_format() -> None:
    module = load_boss_trial_module()
    at = datetime(2026, 6, 27, 10, 30, 5, tzinfo=ZoneInfo("Asia/Shanghai"))

    line = module.format_io_log_line("trace_abc", "info", "direction=input", at=at)

    assert line == "trace_abc-2026-06-27 10:30:05-INFO: direction=input"


def test_trace_events_are_persisted_and_queryable() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    original_db_path = module.DB_PATH
    original_log_path = module.LOG_PATH
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            module.DB_PATH = temp_path / "trial.db"
            module.LOG_PATH = temp_path / "io.log"
            store = module.TrialStore(module.DB_PATH)
            service = module.BossTrialService(store)

            module.write_io_log(
                "trace_replay",
                "INFO",
                module.json_dumps(
                    {
                        "direction": "llm",
                        "event": "llm_request",
                        "stage": "reply_draft",
                        "payload": {
                            "messages": [
                                {"role": "system", "content": "system prompt"},
                                {"role": "user", "content": "{\"text\":\"下午两点 0.5\"}"},
                            ]
                        },
                    }
                ),
            )
            module.write_io_log(
                "trace_replay",
                "INFO",
                module.json_dumps(
                    {
                        "direction": "llm",
                        "event": "llm_response",
                        "stage": "reply_draft",
                        "content": "{\"reply_text\":\"好的，我帮你问问。\"}",
                    }
                ),
            )

            view = service.trace_view("trace_replay")
            overview = service.trace_view()

            assert view["schema_version"] == module.TRACE_EVENT_SCHEMA_VERSION
            assert view["event_count"] == 2
            assert [event["event"] for event in view["events"]] == ["llm_request", "llm_response"]
            assert view["events"][0]["payload"]["payload"]["messages"][0]["content"] == "system prompt"
            assert view["events"][1]["payload"]["content"] == "{\"reply_text\":\"好的，我帮你问问。\"}"
            assert any(item["trace_id"] == "trace_replay" for item in overview["traces"])
    finally:
        module.DB_PATH = original_db_path
        module.LOG_PATH = original_log_path
        restore_env(saved_env)


def test_clear_short_memory_only_resets_current_sender_scope() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            now = datetime(2026, 6, 28, 22, 30, tzinfo=ZoneInfo("Asia/Shanghai"))

            service._remember_sender(
                sender_id="zhang",
                sender_name="张哥",
                conversation_id="boss_trial",
                text="通宵有人吗",
                effective_text="通宵有人吗",
                parsed={"summary": "通宵局咨询"},
                missing_fields=[],
                decision={"action": "ignore"},
                game_id=None,
                trace_id="trace_memory_1",
                now=now,
            )
            service._remember_sender(
                sender_id="zhang",
                sender_name="张哥",
                conversation_id="boss_trial",
                text="可以，帮我组一个吧",
                effective_text="通宵有人吗\n可以，帮我组一个吧",
                parsed={"summary": "确认组局"},
                missing_fields=["stake", "known_players"],
                decision={"action": "ask_clarification"},
                game_id=None,
                trace_id="trace_memory_2",
                now=now,
            )
            service._remember_sender(
                sender_id="ran",
                sender_name="冉姐",
                conversation_id="boss_trial",
                text="可以",
                effective_text="可以",
                parsed={"summary": "确认邀约"},
                missing_fields=[],
                decision={"action": "accept_seat"},
                game_id="game_x",
                trace_id="trace_memory_3",
                now=now,
            )

            assert len(service._sender_memory("boss_trial", "zhang", now)) == 2
            assert len(service._sender_memory("boss_trial", "ran", now)) == 1

            result = service.clear_short_memory(
                {
                    "conversation_id": "boss_trial",
                    "sender_id": "zhang",
                    "reason": "测试清空",
                }
            )

            assert result["ok"] is True
            assert result["conversation_id"] == "boss_trial"
            assert result["sender_id"] == "zhang"
            assert result["cleared_count"] == 2
            assert service._sender_memory("boss_trial", "zhang", now) == []
            assert len(service._sender_memory("boss_trial", "ran", now)) == 1
    finally:
        restore_env(saved_env)


def test_state_transition_backfill_records_existing_entities() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "trial.db"
            conn = module.sqlite3.connect(db_path)
            conn.executescript(
                """
                CREATE TABLE trial_games (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    organizer_id TEXT NOT NULL,
                    organizer_name TEXT NOT NULL,
                    source_text TEXT NOT NULL,
                    parsed_json TEXT NOT NULL,
                    reply_text TEXT NOT NULL DEFAULT '',
                    missing_fields TEXT NOT NULL DEFAULT '[]',
                    notes TEXT NOT NULL DEFAULT '[]'
                );
                INSERT INTO trial_games (
                    id, created_at, updated_at, status, organizer_id, organizer_name,
                    source_text, parsed_json, reply_text, missing_fields, notes
                )
                VALUES (
                    'legacy_game',
                    '2026-06-28T12:00:00+08:00',
                    '2026-06-28T12:00:00+08:00',
                    '邀约中',
                    'zhang',
                    '张哥',
                    '历史局',
                    '{}',
                    '',
                    '[]',
                    '[]'
                );
                """
            )
            conn.commit()
            conn.close()

            store = module.TrialStore(db_path)
            events = store.state_transition_events(entity_type="game", entity_id="legacy_game")

            assert len(events) == 1
            assert events[0]["event"] == "migration_backfill"
            assert events[0]["from_status"] is None
            assert events[0]["to_status"] == "邀约中"
            assert events[0]["metadata"]["backfilled"] is True
    finally:
        restore_env(saved_env)


def test_conversation_id_scopes_short_memory() -> None:
    module = load_boss_trial_module()

    class FakeCache:
        def __init__(self) -> None:
            self.data = {}

        def get_json(self, key, default):
            return self.data.get(key, default)

        def set_json(self, key, value, ttl_seconds):
            self.data[key] = value

    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = FakeCache()
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store, cache=cache)
            base_payload = {
                "sender_name": "张哥",
                "sender_id": "same_sender",
                "now": "2026-06-27T15:00:00+08:00",
            }

            service.analyze({**base_payload, "conversationId": "group_a", "text": "老板"})
            merged = service.analyze({**base_payload, "conversation_id": "group_a", "text": "今天下午"})
            isolated = service.analyze({**base_payload, "conversation_id": "group_b", "text": "0.5或者1都行"})

            assert merged["conversation_id"] == "group_a"
            assert merged["used_short_memory"] is True
            assert merged["effective_text"] == "老板\n今天下午"
            assert isolated["conversation_id"] == "group_b"
            assert isolated["used_short_memory"] is False
            assert isolated["effective_text"] == "0.5或者1都行"
            assert "mahjong:trial:conversation:group_a:sender:same_sender:memory" in cache.data
            assert "mahjong:trial:conversation:group_b:sender:same_sender:memory" in cache.data
    finally:
        restore_env(saved_env)


def test_multiturn_followup_fills_pending_game_slots_without_duplicate_memory() -> None:
    module = load_boss_trial_module()

    class FakeCache:
        def __init__(self) -> None:
            self.data = {}

        def get_json(self, key, default):
            return self.data.get(key, default)

        def set_json(self, key, value, ttl_seconds):
            self.data[key] = value

    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = FakeCache()
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store, cache=cache)
            service.save_customer(
                {
                    "id": "zhang",
                    "display_name": "张哥",
                    "preferred_games": ["杭麻", "财敲"],
                    "preferred_levels": ["0.5", "1"],
                    "usual_start_hours": [18, 19, 20],
                    "smoke_preference": "any",
                    "response_speed": "fast",
                    "response_rate": 0.8,
                    "usual_party_size": 1,
                    "usual_party_size_confidence": 0.7,
                    "notes": "男性；常一个人来；杭麻/财敲常打0.5或1块；本人抽烟，但也可以打无烟局。",
                }
            )
            base_payload = {
                "sender_name": "张哥",
                "sender_id": "zhang",
                "conversation_id": "group_a",
            }

            first = service.analyze(
                {
                    **base_payload,
                    "text": "老板，今天下班有人打麻将吗？0.5或者1都行，烟也都可",
                    "now": "2026-06-27T14:30:00+08:00",
                }
            )
            second = service.analyze(
                {
                    **base_payload,
                    "text": "六点吧",
                    "now": "2026-06-27T14:38:00+08:00",
                }
            )

            assert first["missing_fields"] == []
            assert first["parsed"]["user_intent"] == "咨询现有局"
            assert second["used_short_memory"] is True
            assert (
                second["effective_text"]
                == "老板，今天下班有人打麻将吗？0.5或者1都行，烟也都可\n六点吧"
            )
            assert second["parsed"]["start_time"] == "18:00"
            assert second["parsed"]["current_player_count"] == 1
            assert second["parsed"]["missing_count"] == 3
            assert "烟况都可" in second["parsed"]["rules"]
            assert second["missing_fields"] == []
            assert second["parsed"]["intent_action"] == "inquire_existing_game"
            assert second["outbox"] == []
            assert second["state"]["games"] == []
            assert "要组一个吗" in second["suggested_reply"]["text"]
    finally:
        restore_env(saved_env)


def test_multiturn_party_size_update_overrides_profile_default() -> None:
    module = load_boss_trial_module()

    class FakeCache:
        def __init__(self) -> None:
            self.data = {}

        def get_json(self, key, default):
            return self.data.get(key, default)

        def set_json(self, key, value, ttl_seconds):
            self.data[key] = value

    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = FakeCache()
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store, cache=cache)
            service.save_customer(
                {
                    "id": "zhang",
                    "display_name": "张哥",
                    "preferred_games": ["杭麻", "财敲"],
                    "preferred_levels": ["0.5", "1"],
                    "usual_start_hours": [14, 15, 19, 20],
                    "smoke_preference": "any",
                    "response_speed": "fast",
                    "response_rate": 0.8,
                    "usual_party_size": 1,
                    "usual_party_size_confidence": 0.7,
                    "notes": "男性；常一个人来；杭麻/财敲常打0.5或1块；本人抽烟，但也可以打无烟局。",
                }
            )
            base_payload = {
                "sender_name": "张哥",
                "sender_id": "zhang",
                "conversation_id": "group_party_update",
                "now": "2026-06-27T10:00:00+08:00",
            }

            first = service.analyze({**base_payload, "text": "下午两点 0.5 无烟杭麻，打4小时，帮我组一桌"})
            second = service.analyze({**base_payload, "text": "我这边两个人"})

            assert first["parsed"]["current_player_count"] is None
            assert first["parsed"]["missing_count"] is None
            assert first["missing_fields"] == ["known_players"]
            assert second["used_short_memory"] is True
            assert second["effective_text"] == "下午两点 0.5 无烟杭麻，打4小时，帮我组一桌\n我这边两个人"
            assert second["parsed"]["current_player_count"] == 2
            assert second["parsed"]["missing_count"] == 2
            assert second["missing_fields"] == []
            assert second["parsed"]["intent_action"] == "queue_invites"
            assert second["tool_results"]["search_candidate_customers"]["query"]["missing_count"] == 2
            assert second["outbox"]
            assert not any("客户画像推断 1 人" in note for note in second["parsed"]["notes"])
    finally:
        restore_env(saved_env)


def test_multiturn_affirmative_answer_confirms_profile_party_size_and_continues() -> None:
    module = load_boss_trial_module()

    class FakeCache:
        def __init__(self) -> None:
            self.data = {}

        def get_json(self, key, default):
            return self.data.get(key, default)

        def set_json(self, key, value, ttl_seconds):
            self.data[key] = value

    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = FakeCache()
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store, cache=cache)
            service.save_customer(
                {
                    "id": "zhang",
                    "display_name": "张哥",
                    "preferred_games": ["杭麻", "财敲"],
                    "preferred_levels": ["0.5", "1"],
                    "usual_start_hours": [14, 15, 19, 20],
                    "smoke_preference": "any",
                    "response_speed": "fast",
                    "response_rate": 0.8,
                    "usual_party_size": 1,
                    "usual_party_size_confidence": 0.7,
                    "notes": "男性；常一个人来；杭麻/财敲常打0.5或1块；本人抽烟，但也可以打无烟局。",
                }
            )
            base_payload = {
                "sender_name": "张哥",
                "sender_id": "zhang",
                "conversation_id": "group_party_yes",
                "now": "2026-06-28T12:10:00+08:00",
            }

            first = service.analyze({**base_payload, "text": "下午两点 0.5 无烟杭麻，打4小时，帮我组一桌"})
            second = service.analyze({**base_payload, "text": "是的"})

            assert first["missing_fields"] == ["known_players"]
            assert "你一个人吗" in first["suggested_reply"]["text"]
            assert second["used_short_memory"] is True
            assert second["effective_text"] == "下午两点 0.5 无烟杭麻，打4小时，帮我组一桌\n是的"
            assert second["parsed"]["current_player_count"] == 1
            assert second["parsed"]["missing_count"] == 3
            assert second["missing_fields"] == []
            assert second["parsed"]["intent_action"] == "queue_invites"
            assert second["tool_results"]["search_candidate_customers"]["called"] is True
            assert second["tool_results"]["search_candidate_customers"]["query"]["missing_count"] == 3
            assert second["outbox"]
            assert second["suggested_reply"]["text"] == "好的，我帮你问问。"
            assert any("短答确认" in note for note in second["parsed"]["notes"])
    finally:
        restore_env(saved_env)


def test_past_start_time_requires_confirmation_before_invites() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)

            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "group_past_time",
                    "text": "下午两点 0.5 无烟杭麻，打4小时，一缺三，帮我组一桌",
                    "now": "2026-06-27T16:00:00+08:00",
                }
            )

            assert result["missing_fields"] == ["start_time"]
            assert result["parsed"]["intent_action"] == "ask_clarification"
            assert any("已经过了" in item for item in result["parsed"]["ambiguities"])
            assert "两点" in result["suggested_reply"]["text"]
            assert "明天" in result["suggested_reply"]["text"]
            assert "先帮你看" not in result["suggested_reply"]["text"]
            assert "帮你问" not in result["suggested_reply"]["text"]
            assert result["outbox"] == []
            assert result["tool_results"]["search_candidate_customers"]["called"] is False
    finally:
        restore_env(saved_env)


def test_afternoon_numeric_time_resolves_to_same_day_without_hardcoded_badcase() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)

            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "group_afternoon_time",
                    "text": "4点无烟0.5，173，5h",
                    "now": "2026-06-28T13:59:00+08:00",
                }
            )

            assert result["missing_fields"] == []
            assert result["parsed"]["start_time"] == "16:00"
            assert result["parsed"]["start_time_confidence"] >= module.TIME_RESOLUTION_CONFIDENCE_THRESHOLD
            assert result["parsed"]["ambiguities"] == []
            assert result["parsed"]["intent_action"] == "queue_invites"
            assert result["suggested_reply"]["text"] == "好的，我帮你问问。"
            assert "上午还是下午" not in result["suggested_reply"]["text"]
            assert "明天" not in result["suggested_reply"]["text"]
            assert result["tool_results"]["search_candidate_customers"]["called"] is True
            assert result["outbox"]
            assert all("16:00" in item["message_text"] for item in result["outbox"])
    finally:
        restore_env(saved_env)


def test_candidate_recommendation_respects_soft_gender_composition_preference() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)

            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "group_gender_preference",
                    "text": "4点无烟0.5，4h，272，最好再来一男一女",
                    "now": "2026-06-28T13:59:00+08:00",
                }
            )

            preference = result["parsed"]["candidate_composition_preference"]
            assert preference["preferred_candidate_genders"] == ["male", "female"]
            assert result["parsed"]["current_player_count"] == 2
            assert result["parsed"]["missing_count"] == 2
            assert result["missing_fields"] == []
            assert result["tool_results"]["search_candidate_customers"]["called"] is True
            assert result["tool_results"]["search_candidate_customers"]["query"]["candidate_composition_preference"][
                "preferred_candidate_genders"
            ] == ["male", "female"]
            assert len(result["outbox"]) >= 2
            assert [item["gender"] for item in result["outbox"][:2]] == ["male", "female"]
            assert any("符合候选组合偏好：男" in reason for reason in result["outbox"][0]["reasons"])
            assert all("一男一女" not in item["message_text"] for item in result["outbox"])
            assert all("男" not in item["message_text"] and "女" not in item["message_text"] for item in result["outbox"])
    finally:
        restore_env(saved_env)


def test_trial_inferences_preserve_explicit_272_and_keep_gender_preference_out_of_rules() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            game = module.GameRequest(
                organizer_id="zhang",
                organizer_name="张哥",
                channel_id="boss_trial",
                game_type="hangzhou_mahjong",
                current_player_count=1,
                missing_count=3,
                level="0.5",
                rules=["杭麻", "无烟", "女玩家相关"],
            )

            service._apply_trial_inferences(
                game,
                "4点无烟0.5，4h，272，最好再来一男一女",
                "zhang",
                now=datetime.fromisoformat("2026-06-28T13:59:00+08:00"),
                source_text="4点无烟0.5，4h，272，最好再来一男一女",
                sender_memory=[],
            )

            assert game.current_player_count == 2
            assert game.missing_count == 2
            assert "女玩家相关" not in game.rules
            assert service._candidate_composition_preference_from_game(game)["preferred_candidate_genders"] == [
                "male",
                "female",
            ]
            assert any("272" in note for note in game.notes)
    finally:
        restore_env(saved_env)


def test_candidate_acceptance_word_da_updates_progress_after_prior_confirmation_and_guards_llm() -> None:
    saved_env = without_llm_env()
    os.environ.update(
        {
            "MAHJONG_LLM_API_KEY": "test-key",
            "MAHJONG_LLM_PROVIDER": "openai",
            "MAHJONG_LLM_MODEL": "test-model",
            "MAHJONG_LLM_BASE_URL": "https://example.invalid/v1",
        }
    )
    module = load_boss_trial_module()
    captured = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            content = {
                "semantic_type": "accepted",
                "proposed_action": "mark_candidate_confirmed",
                "confidence": 0.96,
                "reply_text": "好的，加你272了。",
                "risk_level": "low",
                "reasoning_summary": "测试模型输出了过期进度。",
                "extracted_facts": {},
                "notes": ["测试"],
            }
            return json.dumps(
                {
                    "choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                },
                ensure_ascii=False,
            ).encode("utf-8")

    original_urlopen = module.urllib.request.urlopen

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        captured.append(payload)
        return FakeResponse(payload)

    module.urllib.request.urlopen = fake_urlopen
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            created = service.manual_create_game(
                {
                    "organizer_id": "zhang",
                    "organizer_name": "张哥",
                    "game_type": "hangzhou_mahjong",
                    "level": "0.5",
                    "start_at": "2026-06-28T16:00:00+08:00",
                    "duration_hours": 4,
                    "current_player_count": 2,
                    "missing_count": 2,
                    "smoke": "no_smoke",
                    "now": "2026-06-28T14:00:00+08:00",
                }
            )
            game_id = created["game"]["id"]
            liu_outbox_id = store.create_outbox(
                game_id=game_id,
                customer_id="liu",
                customer_name="刘姐",
                message_text="刘姐，16:00，0.5无烟，约4小时，打吗？",
                score=100,
                reasons=["测试候选人"],
                warnings=[],
            )
            amy_outbox_id = store.create_outbox(
                game_id=game_id,
                customer_id="amy",
                customer_name="Amy",
                message_text="Amy，16:00，0.5无烟，约4小时，打吗？",
                score=90,
                reasons=["测试候选人"],
                warnings=[],
            )
            store.record_feedback(
                {
                    "game_id": game_id,
                    "outbox_id": liu_outbox_id,
                    "customer_id": "liu",
                    "feedback_type": "accepted",
                    "notes": "测试：刘姐先确认",
                    "now": "2026-06-28T14:02:00+08:00",
                }
            )

            reply = service.candidate_message(
                {
                    "outbox_id": amy_outbox_id,
                    "text": "打！",
                    "trace_id": "trace_candidate_da_accept",
                    "now": "2026-06-28T14:03:00+08:00",
                }
            )

            prompt = json.loads(captured[0]["messages"][1]["content"])
            assert "语义解析器和动作提案器" in captured[0]["messages"][0]["content"]
            assert prompt["candidate"]["reply_text"] == "打！"
            assert prompt["game_state"]["confirmed_before"] == 1
            assert prompt["state_preview"]["if_confirmed"]["confirmed_after"] == 2
            assert prompt["state_preview"]["if_confirmed"]["progress_label_after"] == "人齐"
            assert prompt["state_preview"]["if_confirmed"]["fallback_reply"] == "好的，人齐了。"
            assert reply["candidate_message"]["semantic_type"] == "accepted"
            assert reply["candidate_message"]["proposed_action"] == "mark_candidate_confirmed"
            assert reply["candidate_message"]["feedback_type"] == "accepted"
            assert reply["candidate_message"]["suggested_boss_reply"] == "好的，人齐了。"
            assert reply["outbox_item"]["status"] == "已确认"
    finally:
        module.urllib.request.urlopen = original_urlopen
        restore_env(saved_env)


def test_llm_bad_past_time_reply_is_guarded_before_display() -> None:
    saved_env = without_llm_env()
    os.environ.update(
        {
            "MAHJONG_LLM_API_KEY": "test-key",
            "MAHJONG_LLM_PROVIDER": "openai",
            "MAHJONG_LLM_MODEL": "test-model",
            "MAHJONG_LLM_BASE_URL": "https://example.invalid/v1",
        }
    )
    module = load_boss_trial_module()

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            system_prompt = self.payload["messages"][0]["content"]
            if "语义解析器" in system_prompt:
                content = {
                    "is_mahjong_related": True,
                    "intent": "find_players",
                    "confidence": 0.9,
                    "normalized_text": "下午两点 0.5 无烟杭麻，帮忙组一桌",
                    "reply_text": "可以，我先帮你看。",
                    "reasoning_summary": "测试",
                    "needs_human_review": False,
                    "facts": {"reason": "测试"},
                }
            elif "工具规划器" in system_prompt:
                content = {
                    "tool_calls": [],
                    "reasoning_summary": "时间已过且人数未知，先不调用工具。",
                }
            else:
                content = {
                    "reply_text": "可以，我先帮你看。你说的两点是明天吗，还是改其他时间？ 你一个人吗？",
                    "risk_level": "low",
                    "reasoning_summary": "测试坏话术。",
                    "notes": ["测试"],
                }
            return json.dumps(
                {
                    "choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                },
                ensure_ascii=False,
            ).encode("utf-8")

    original_urlopen = module.urllib.request.urlopen

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        return FakeResponse(payload)

    module.urllib.request.urlopen = fake_urlopen
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)

            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "group_past_time_llm_guard",
                    "text": "下午两点 0.5 无烟杭麻，打4小时，帮我组一桌",
                    "now": "2026-06-27T16:00:00+08:00",
                }
            )

            reply = result["suggested_reply"]["text"]
            assert "两点" in reply
            assert "明天" in reply
            assert "先帮你看" not in reply
            assert "帮你问" not in reply
            assert result["tool_results"]["search_candidate_customers"]["called"] is False
            assert result["outbox"] == []
    finally:
        module.urllib.request.urlopen = original_urlopen
        restore_env(saved_env)


def test_weak_inquiry_searches_existing_pool_before_asking_time_or_inviting_candidates() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            service.analyze(
                {
                    "sender_name": "李姐",
                    "sender_id": "li",
                    "conversation_id": "group_pool",
                    "text": "今晚六点 0.5 无烟杭麻 打4小时 三缺一",
                    "now": "2026-06-27T10:00:00+08:00",
                }
            )
            before_games = service.state(now=module.parse_dt("2026-06-27T14:00:00+08:00"))["games"]

            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "private_zhang",
                    "text": "老板，今天下班有人打麻将吗？0.5或者1都行，烟也都可",
                    "now": "2026-06-27T14:30:00+08:00",
                }
            )

            reply = result["suggested_reply"]["text"]
            assert result["pool_matches"]
            tool_result = result["tool_results"]["search_current_open_games"]
            assert tool_result["called"] is True
            assert tool_result["tool_name"] == "search_current_open_games"
            assert tool_result["result_count"] >= 1
            assert result["parsed"]["intent_action"] == "match_existing_game"
            assert result["parsed"]["user_intent"] == "匹配已有局/可拼局"
            assert result["parsed"]["level_options"] == ["0.5", "1"]
            assert result["parsed"]["smoke_options"] == ["无烟", "可吸烟"]
            assert "18:00" in reply
            assert "0.5" in reply
            assert "大概几点" not in reply
            assert "几点能到" not in reply
            assert result["outbox"] == []
            assert len(result["state"]["games"]) == len(before_games)
    finally:
        restore_env(saved_env)


def test_manual_create_game_feeds_near_start_pool_search_with_profile_fallback() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            created = service.manual_create_game(
                {
                    "organizer_name": "老板手动创建",
                    "game_type": "hangzhou_mahjong",
                    "variant": "caiqiao",
                    "start_time": "18:00",
                    "level": "1",
                    "current_player_count": 3,
                    "missing_count": 1,
                    "duration_hours": 4,
                    "smoke": "smoke_ok",
                    "status": "待组局",
                    "source_text": "电话：六点有烟1块，三缺一",
                    "now": "2026-06-27T17:00:00+08:00",
                }
            )
            assert created["ok"] is True
            assert created["state"]["games"]
            assert created["agent_actions"][0]["protocol"] == "controlled_agent.v1"
            manual_action = created["agent_actions"][0]["validated_actions"][0]
            assert manual_action["tool_name"] == "create_game"
            assert manual_action["proposed_by"] == "boss_manual"
            assert manual_action["approval_required"] is True
            assert manual_action["allowed"] is True
            assert manual_action["idempotency_key"]
            assert manual_action["ledger_status"] == "executed"

            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "t1",
                    "text": "现在有没有人齐开0.5的啊",
                    "now": "2026-06-27T17:00:00+08:00",
                }
            )

            reply = result["suggested_reply"]["text"]
            assert result["tool_results"]["search_current_open_games"]["called"] is True
            assert result["pool_matches"]
            assert result["pool_matches"][0]["level"] == "1"
            assert result["pool_matches"][0]["level_match_type"] == "profile_fallback"
            assert result["parsed"]["intent_action"] == "match_existing_game"
            assert result["parsed"]["level_options"] == ["0.5", "1"]
            assert "0.5" in reply
            assert "暂时没有" in reply
            assert "18:00" in reply
            assert "1块" in reply
            assert "有烟" in reply
            assert "三缺一" in reply
            assert "大概几点" not in reply
            assert result["outbox"] == []
    finally:
        restore_env(saved_env)


def test_near_start_pool_search_normalizes_decimal_separator_and_renqikai_typo() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            service.manual_create_game(
                {
                    "organizer_name": "老板手动创建",
                    "game_type": "hangzhou_mahjong",
                    "variant": "caiqiao",
                    "start_time": "18:00",
                    "level": "0.5",
                    "current_player_count": 3,
                    "missing_count": 1,
                    "duration_hours": 4,
                    "smoke": "no_smoke",
                    "status": "待组局",
                    "now": "2026-06-27T17:20:00+08:00",
                }
            )

            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "t1",
                    "text": "现在有没有0，5无烟人气开的",
                    "now": "2026-06-27T17:20:00+08:00",
                }
            )

            reply = result["suggested_reply"]["text"]
            tool = result["tool_results"]["search_current_open_games"]
            assert tool["called"] is True
            assert tool["query"]["level_options"] == ["0.5", "1"]
            assert tool["query"]["time_window"] == [
                "2026-06-27T17:20:00+08:00",
                "2026-06-27T18:50:00+08:00",
            ]
            assert result["pool_matches"]
            assert result["pool_matches"][0]["level"] == "0.5"
            assert result["pool_matches"][0]["missing_count"] == 1
            assert result["parsed"]["intent_action"] == "match_existing_game"
            assert "18:00" in reply
            assert "0.5" in reply
            assert "无烟" in reply
            assert "三缺一" in reply
            assert "现在没有" not in reply
            assert "大概几点" not in reply
            assert result["outbox"] == []

            full_stop_result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "t1_full_stop",
                    "text": "现在有没有0。5无烟人气开的",
                    "now": "2026-06-27T17:20:00+08:00",
                }
            )

            full_stop_reply = full_stop_result["suggested_reply"]["text"]
            full_stop_tool = full_stop_result["tool_results"]["search_current_open_games"]
            assert full_stop_tool["query"]["level_options"] == ["0.5", "1"]
            assert full_stop_result["pool_matches"]
            assert full_stop_result["pool_matches"][0]["level"] == "0.5"
            assert "0.5" in full_stop_reply
            assert "0/5" not in full_stop_reply
    finally:
        restore_env(saved_env)


def test_pool_no_match_reply_normalizes_full_stop_decimal_level() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "decimal_full_stop",
                    "text": "通宵0。5有人吗",
                    "now": "2026-06-28T22:40:00+08:00",
                }
            )

            tool = result["tool_results"]["search_current_open_games"]
            reply = result["suggested_reply"]["text"]
            assert tool["query"]["level_options"] == ["0.5"]
            assert "0.5" in reply
            assert "0/5" not in reply
            assert "要组一个吗" in reply
    finally:
        restore_env(saved_env)


def test_near_start_pool_search_no_match_asks_whether_to_create_new_game() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "t1",
                    "text": "现在有没有人齐开0.5的啊",
                    "now": "2026-06-27T17:00:00+08:00",
                }
            )

            reply = result["suggested_reply"]["text"]
            assert result["tool_results"]["search_current_open_games"]["called"] is True
            assert result["pool_matches"] == []
            assert "0.5" in reply
            assert "暂时没有" in reply
            assert "要组一个吗" in reply
            assert "大概几点" not in reply
            assert "打多大" not in reply
            assert result["outbox"] == []
    finally:
        restore_env(saved_env)


def test_pool_search_tool_result_is_fed_to_llm_reply_selection() -> None:
    saved_env = without_llm_env()
    os.environ.update(
        {
            "MAHJONG_LLM_API_KEY": "test-key",
            "MAHJONG_LLM_PROVIDER": "openai",
            "MAHJONG_LLM_MODEL": "test-model",
            "MAHJONG_LLM_BASE_URL": "https://example.invalid/v1",
        }
    )
    module = load_boss_trial_module()
    captured = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            system_prompt = self.payload["messages"][0]["content"]
            if "语义解析器" in system_prompt:
                content = {
                    "is_mahjong_related": True,
                    "intent": "join_game",
                    "confidence": 0.9,
                    "normalized_text": "张哥下班后想找0.5或1档、烟都可的现成局",
                    "reply_text": "我帮你看看。",
                    "needs_human_review": False,
                    "facts": {"reason": "测试"},
                }
            elif "工具规划器" in system_prompt:
                content = {
                    "tool_calls": [
                        {
                            "tool_name": "search_current_open_games",
                            "arguments": {},
                            "reason": "用户在问当前是否有合适牌局。",
                        }
                    ],
                    "reasoning_summary": "先查当前局池。",
                }
            else:
                user_payload = json.loads(self.payload["messages"][1]["content"])
                captured.append(user_payload)
                game_id = user_payload["tool_results"]["search_current_open_games"]["matches"][0]["game_id"]
                content = {
                    "reply_text": "有的，18:00 0.5无烟这个比较合适，还缺 1 个。要我帮你问下吗？",
                    "selected_pool_game_id": game_id,
                    "risk_level": "low",
                    "reasoning_summary": "工具返回了匹配局，选择最合适的一条。",
                    "notes": ["测试"],
                }
            return json.dumps(
                {
                    "choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                },
                ensure_ascii=False,
            ).encode("utf-8")

    original_urlopen = module.urllib.request.urlopen

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        return FakeResponse(payload)

    module.urllib.request.urlopen = fake_urlopen
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            store.create_game(
                game_id="game_pool_llm",
                status="待组局",
                organizer_id="li",
                organizer_name="李姐",
                source_text="今晚六点 0.5 无烟杭麻 打4小时 三缺一",
                parsed={
                    "id": "game_pool_llm",
                    "status": "open",
                    "game_type": "hangzhou_mahjong",
                    "game_label": "杭麻",
                    "ruleset": "hangzhou_mahjong",
                    "variant": None,
                    "variant_label": None,
                    "level": "0.5",
                    "base_score": 0.5,
                    "cap_score": None,
                    "start_at": "2026-06-27T18:00:00+08:00",
                    "start_time": "18:00",
                    "duration_hours": 4,
                    "current_player_count": 3,
                    "missing_count": 1,
                    "rules": ["杭麻", "无烟"],
                    "play_options": [],
                    "notes": [],
                    "summary": "杭麻 0.5档 18:00 缺1 无烟",
                },
                reply_text="",
                missing_fields=[],
                notes=[],
            )
            service = module.BossTrialService(store)
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "private_zhang",
                    "text": "老板，今天下班有人打麻将吗？0.5或者1都行，烟也都可",
                    "now": "2026-06-27T14:30:00+08:00",
                }
            )

            assert captured
            prompt = captured[-1]
            assert prompt["tool_results"]["search_current_open_games"]["called"] is True
            assert prompt["tool_results"]["search_current_open_games"]["matches"][0]["game_id"] == "game_pool_llm"
            assert result["suggested_reply"]["source"] == "llm"
            assert result["suggested_reply"]["selected_pool_game_id"] == "game_pool_llm"
            assert result["parsed"]["intent_action"] == "match_existing_game"
    finally:
        module.urllib.request.urlopen = original_urlopen
        restore_env(saved_env)


def test_explicit_party_size_creates_approval_reply_and_invites() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "group_a",
                    "text": "下午两点 0.5 无烟杭麻，打4小时，一缺三，帮我组一桌",
                    "now": "2026-06-27T10:00:00+08:00",
                }
            )

            assert result["suggested_reply"]["status"] == "待审批"
            assert result["suggested_reply"]["source"] == "rules"
            assert result["suggested_reply"]["text"] == "好的，我帮你问问。"
            assert "杭麻" not in result["suggested_reply"]["text"]
            assert "0.5" not in result["suggested_reply"]["text"]
            candidate_tool = result["tool_results"]["search_candidate_customers"]
            send_tool = result["tool_results"]["send_message"]
            assert candidate_tool["called"] is True
            assert candidate_tool["tool_name"] == "search_candidate_customers"
            assert candidate_tool["result_count"] >= len(result["outbox"])
            assert send_tool["called"] is True
            assert send_tool["tool_name"] == "send_message"
            assert send_tool["risk_level"] == "high"
            assert send_tool["approval_required"] is True
            assert send_tool["direct_send_allowed"] is False
            assert send_tool["direct_send_executed"] is False
            assert send_tool["execution_mode"] == "create_pending_outbox"
            assert result["agent_actions"]
            assert all(item["protocol"] == "controlled_agent.v1" for item in result["agent_actions"])
            assert any(
                action["tool_name"] == "search_candidate_customers" and action["allowed"] is True
                and action["ledger_status"] == "executed"
                for plan in result["agent_actions"]
                for action in plan["validated_actions"]
            )
            assert any(
                action["tool_name"] == "send_message"
                and action["allowed"] is True
                and action["approval_required"] is True
                and action["idempotency_key"]
                and action["ledger_status"] == "executed"
                for plan in result["agent_actions"]
                for action in plan["validated_actions"]
            )
            assert any(
                plan["stage"] == "create_game"
                and action["tool_name"] == "create_game"
                and action["allowed"] is True
                and action["idempotency_key"]
                and action["ledger_status"] == "executed"
                for plan in result["agent_actions"]
                for action in plan["validated_actions"]
            )
            assert any(
                item["stage"] == "create_game"
                and item["tool_name"] == "create_game"
                and item["status"] == "executed"
                for item in store.controlled_actions()
            )
            assert any(
                item["tool_name"] == "search_candidate_customers"
                and item["status"] == "executed"
                and item["result"].get("result_count") == candidate_tool["result_count"]
                for item in store.controlled_actions()
            )
            assert any(
                item["tool_name"] == "send_message"
                and item["status"] == "executed"
                and item["result"].get("result_count") == len(result["outbox"])
                for item in store.controlled_actions()
            )
            assert result["outbox"]
            assert all(item["approval_required"] is True for item in result["outbox"])
            assert all(item["direct_send_executed"] is False for item in result["outbox"])
            assert all(item["approval_status"] == "待审批" for item in result["outbox"])
            assert all(item["draft_source"] == "rules" for item in result["outbox"])
            approvals = store.recent_approvals()
            assert len(approvals) == len(result["outbox"])
            for item in result["outbox"]:
                approval = item["approval"]
                assert approval["target_type"] == "outbox"
                assert approval["target_id"] == item["id"]
                assert approval["status"] == "pending"
                assert approval["original_message_text"] == item["message_text"]
                assert approval["final_message_text"] == item["message_text"]
                assert approval["risk_level"] == "high"
                assert approval["metadata"]["game_id"] == item["game_id"]
                assert approval["metadata"]["customer_id"] == item["customer_id"]
                draft = item["message_text"]
                assert item["customer_name"] in draft
                assert "14:00" in draft
                assert "0.5无烟" in draft
                assert "4小时" in draft
                assert "打吗" in draft
                assert "缺" not in draft
                assert "方便来吗" not in draft
                assert "有一桌" not in draft
                assert "杭麻" not in draft
                assert "财敲" not in draft
    finally:
        restore_env(saved_env)


def test_approval_decision_updates_draft_without_sending_and_is_idempotent() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "approval_flow",
                    "text": "下午两点 0.5 无烟杭麻，打4小时，一缺三，帮我组一桌",
                    "now": "2026-06-27T10:00:00+08:00",
                }
            )

            assert len(result["outbox"]) >= 2
            first = result["outbox"][0]
            approval_id = first["approval"]["id"]
            decision_payload = {
                "approval_id": approval_id,
                "decision": "approved",
                "reviewer_id": "boss",
                "reviewer_name": "老板",
                "reason": "测试通过",
                "trace_id": "trace_approval_decision",
                "now": "2026-06-27T10:01:00+08:00",
            }

            approved = service.approval_decision(decision_payload)
            retry = service.approval_decision(decision_payload)
            outbox_item = store.outbox_item(first["id"])

            assert approved["ok"] is True
            assert approved["approval"]["status"] == "approved"
            assert approved["approval"]["reviewer_id"] == "boss"
            assert approved["approval"]["decision_reason"] == "测试通过"
            assert outbox_item["status"] == "已审批"
            assert outbox_item["approval_status"] == "已审批"
            assert outbox_item["approval"]["status"] == "approved"
            assert outbox_item["status"] != "已发送"
            assert retry["deduplicated"] is True
            assert retry["approval"]["status"] == "approved"
            action = approved["agent_actions"][0]["validated_actions"][0]
            assert action["tool_name"] == "record_approval_decision"
            assert action["approval_required"] is True
            assert action["ledger_status"] == "executed"

            sent = service.send_outbox(
                {
                    "outbox_id": first["id"],
                    "channel": "manual",
                    "trace_id": "trace_send_outbox",
                    "now": "2026-06-27T10:01:30+08:00",
                }
            )
            retry_sent = service.send_outbox(
                {
                    "outbox_id": first["id"],
                    "channel": "manual",
                    "trace_id": "trace_send_outbox_retry",
                    "now": "2026-06-27T10:01:40+08:00",
                }
            )
            sent_item = store.outbox_item(first["id"])
            deliveries = store.delivery_attempts_for_outbox(first["id"])

            assert sent["ok"] is True
            assert sent["delivery"]["status"] == "sent"
            assert sent["outbox_item"]["status"] == "已发送"
            assert sent_item["status"] == "已发送"
            assert len(deliveries) == 1
            assert retry_sent["deduplicated"] is True
            assert retry_sent["delivery"]["id"] == sent["delivery"]["id"]
            send_action = sent["agent_actions"][0]["validated_actions"][0]
            assert send_action["tool_name"] == "execute_outbox_delivery"
            assert send_action["risk_level"] == "high"
            assert send_action["ledger_status"] == "executed"

            bypass = service.feedback(
                {
                    "game_id": first["game_id"],
                    "outbox_id": first["id"],
                    "customer_id": first["customer_id"],
                    "feedback_type": "sent",
                    "trace_id": "trace_send_bypass",
                    "now": "2026-06-27T10:01:50+08:00",
                }
            )
            assert bypass["ok"] is False
            assert bypass["rejected"] is True
            bypass_action = bypass["agent_actions"][0]["rejected_actions"][0]
            assert bypass_action["code"] == "send_requires_delivery_gateway"

            second = result["outbox"][1]
            rejected = service.approval_decision(
                {
                    "approval_id": second["approval"]["id"],
                    "decision": "rejected",
                    "reviewer_id": "boss",
                    "reason": "测试拒绝",
                    "trace_id": "trace_approval_reject",
                    "now": "2026-06-27T10:02:00+08:00",
                }
            )
            rejected_item = store.outbox_item(second["id"])

            assert rejected["ok"] is True
            assert rejected["approval"]["status"] == "rejected"
            assert rejected_item["status"] == "审批拒绝"
            assert rejected_item["approval_status"] == "审批拒绝"
            assert rejected_item["approval"]["status"] == "rejected"
    finally:
        restore_env(saved_env)


def test_runtime_policy_blocks_delivery_and_read_only_writes() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "runtime_policy",
                    "text": "下午两点 0.5 无烟杭麻，打4小时，一缺三，帮我组一桌",
                    "now": "2026-06-27T10:00:00+08:00",
                }
            )
            first = result["outbox"][0]
            service.approval_decision(
                {
                    "approval_id": first["approval"]["id"],
                    "decision": "approved",
                    "trace_id": "trace_policy_approval",
                    "now": "2026-06-27T10:01:00+08:00",
                }
            )

            policy_update = service.update_runtime_policy(
                {
                    "delivery_enabled": False,
                    "reason": "测试临时关闭发送",
                    "trace_id": "trace_policy_delivery_off",
                    "now": "2026-06-27T10:02:00+08:00",
                }
            )
            blocked_send = service.send_outbox(
                {
                    "outbox_id": first["id"],
                    "channel": "manual",
                    "trace_id": "trace_policy_blocked_send",
                    "now": "2026-06-27T10:03:00+08:00",
                }
            )

            assert policy_update["policy"]["delivery_enabled"] is False
            assert blocked_send["ok"] is False
            assert blocked_send["rejected"] is True
            send_rejection = blocked_send["agent_actions"][0]["rejected_actions"][0]
            assert send_rejection["code"] == "runtime_policy_delivery_disabled"
            assert store.outbox_item(first["id"])["status"] == "已审批"
            assert store.delivery_attempts_for_outbox(first["id"]) == []

            read_only = service.update_runtime_policy(
                {
                    "read_only_mode": True,
                    "delivery_enabled": True,
                    "reason": "测试只读模式",
                    "trace_id": "trace_policy_read_only",
                    "now": "2026-06-27T10:04:00+08:00",
                }
            )
            blocked_manual = service.manual_create_game(
                {
                    "game_id": "manual_policy_blocked",
                    "organizer_id": "zhang",
                    "organizer_name": "张哥",
                    "level": "0.5",
                    "start_time": "16:00",
                    "current_player_count": 1,
                    "missing_count": 3,
                    "duration_hours": 4,
                    "smoke": "无烟",
                    "trace_id": "trace_policy_blocked_manual",
                    "now": "2026-06-27T10:05:00+08:00",
                }
            )

            assert read_only["policy"]["read_only_mode"] is True
            assert blocked_manual["ok"] is False
            manual_rejection = blocked_manual["agent_actions"][0]["rejected_actions"][0]
            assert manual_rejection["code"] == "runtime_policy_read_only"
            assert not any(item.get("id") == "manual_policy_blocked" for item in store.games(include_final=True))

            eval_off = service.update_runtime_policy(
                {
                    "read_only_mode": False,
                    "eval_writes_enabled": False,
                    "reason": "测试关闭评测写入",
                    "trace_id": "trace_policy_eval_off",
                    "now": "2026-06-27T10:06:00+08:00",
                }
            )
            blocked_eval = service.record_eval_case(
                {
                    "case_type": "badcase",
                    "text": "老板，测试 badcase",
                    "trace_id": "trace_policy_blocked_eval",
                    "now": "2026-06-27T10:07:00+08:00",
                }
            )

            assert eval_off["policy"]["eval_writes_enabled"] is False
            assert blocked_eval["ok"] is False
            assert blocked_eval["rejected"] is True
            eval_rejection = blocked_eval["agent_actions"][0]["rejected_actions"][0]
            assert eval_rejection["code"] == "runtime_policy_eval_writes_disabled"

            restored = service.update_runtime_policy(
                {
                    "read_only_mode": False,
                    "delivery_enabled": True,
                    "eval_writes_enabled": True,
                    "reason": "恢复测试策略",
                    "trace_id": "trace_policy_restore",
                    "now": "2026-06-27T10:08:00+08:00",
                }
            )
            assert restored["policy"]["read_only_mode"] is False
            assert restored["policy"]["delivery_enabled"] is True
            assert restored["policy"]["eval_writes_enabled"] is True
    finally:
        restore_env(saved_env)


def test_runtime_policy_blocks_side_effect_tools_but_allows_read_only_tools() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            now = module.parse_dt("2026-06-28T14:00:00+08:00")
            game = module.GameRequest(
                organizer_id="zhang",
                organizer_name="张哥",
                channel_id="boss_trial",
                game_type="hangzhou_mahjong",
                current_player_count=1,
                missing_count=3,
                level="0.5",
                rules=["杭麻", "无烟"],
                start_at=module.parse_dt("2026-06-28T16:00:00+08:00"),
                duration_hours=4,
            )

            service.update_runtime_policy(
                {
                    "read_only_mode": True,
                    "reason": "测试只读工具门禁",
                    "trace_id": "trace_policy_tool_read_only",
                    "now": "2026-06-28T14:00:00+08:00",
                }
            )
            read_plan = service._validate_tool_plan(
                trace_id="trace_policy_tool_read",
                plan={
                    "source": "llm",
                    "stage": "before_open_game_search",
                    "tool_calls": [
                        {
                            "tool_name": "search_current_open_games",
                            "arguments": {},
                            "reason": "先查当前有没有能拼的局。",
                        }
                    ],
                },
                stage="before_open_game_search",
                game=game,
                missing_fields=[],
                tool_results={},
                now=now,
            )
            write_plan = service._validate_tool_plan(
                trace_id="trace_policy_tool_write",
                plan={
                    "source": "llm",
                    "stage": "after_open_game_search",
                    "tool_calls": [
                        {
                            "tool_name": "search_candidate_customers",
                            "arguments": {},
                            "reason": "只读搜索候选人。",
                        },
                        {
                            "tool_name": "send_message",
                            "arguments": {"execution_mode": "create_pending_outbox"},
                            "reason": "创建待审批邀约。",
                        },
                    ],
                },
                stage="after_open_game_search",
                game=game,
                missing_fields=[],
                tool_results={},
                now=now,
            )

            assert read_plan["validated_actions"][0]["tool_name"] == "search_current_open_games"
            assert read_plan["validated_actions"][0]["validation"]["allowed"] is True
            assert [item["tool_name"] for item in write_plan["tool_calls"]] == ["search_candidate_customers"]
            assert write_plan["validated_actions"][0]["tool_name"] == "search_candidate_customers"
            assert write_plan["validated_actions"][0]["validation"]["allowed"] is True
            rejected_send = write_plan["rejected_actions"][0]
            assert rejected_send["tool_name"] == "send_message"
            assert rejected_send["validation"]["code"] == "runtime_policy_read_only"

            service.update_runtime_policy(
                {
                    "read_only_mode": False,
                    "state_writes_enabled": False,
                    "reason": "测试禁止状态写入",
                    "trace_id": "trace_policy_tool_state_writes_off",
                    "now": "2026-06-28T14:01:00+08:00",
                }
            )
            write_off_plan = service._validate_tool_plan(
                trace_id="trace_policy_tool_state_writes",
                plan={
                    "source": "llm",
                    "stage": "after_open_game_search",
                    "tool_calls": [
                        {
                            "tool_name": "send_message",
                            "arguments": {"execution_mode": "create_pending_outbox"},
                            "reason": "创建待审批邀约。",
                        }
                    ],
                },
                stage="after_open_game_search",
                game=game,
                missing_fields=[],
                tool_results={},
                now=now,
            )
            assert write_off_plan["tool_calls"] == []
            assert write_off_plan["rejected_actions"][0]["validation"]["code"] == (
                "runtime_policy_state_writes_disabled"
            )
    finally:
        restore_env(saved_env)


def test_runtime_policy_blocks_candidate_feedback_state_writes() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "candidate_policy",
                    "text": "下午两点 0.5 无烟杭麻，打4小时，三缺一，帮我组一桌",
                    "now": "2026-06-28T10:00:00+08:00",
                }
            )
            outbox_item = result["outbox"][0]

            service.update_runtime_policy(
                {
                    "read_only_mode": True,
                    "reason": "测试只读模式拦截候选人反馈",
                    "trace_id": "trace_candidate_policy_read_only",
                    "now": "2026-06-28T10:01:00+08:00",
                }
            )
            blocked = service.candidate_message(
                {
                    "outbox_id": outbox_item["id"],
                    "text": "可以",
                    "trace_id": "trace_candidate_policy_blocked",
                    "now": "2026-06-28T10:02:00+08:00",
                }
            )

            assert blocked["ok"] is False
            assert blocked["rejected"] is True
            feedback_action = blocked["agent_actions"][0]["rejected_actions"][0]
            assert feedback_action["tool_name"] == "record_candidate_feedback"
            assert feedback_action["code"] == "runtime_policy_read_only"
            assert store.outbox_item(outbox_item["id"])["status"] == "待审批"
            assert not [
                item
                for item in store.controlled_actions()
                if item["stage"] == "candidate_feedback" and item["status"] == "executed"
            ]
    finally:
        restore_env(saved_env)


def test_runtime_policy_can_require_llm_proposal_for_side_effect_tools() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            now = module.parse_dt("2026-06-28T14:00:00+08:00")
            game = module.GameRequest(
                organizer_id="zhang",
                organizer_name="张哥",
                channel_id="boss_trial",
                game_type="hangzhou_mahjong",
                current_player_count=1,
                missing_count=3,
                level="0.5",
                rules=["杭麻", "无烟"],
                start_at=module.parse_dt("2026-06-28T16:00:00+08:00"),
                duration_hours=4,
            )

            policy_update = service.update_runtime_policy(
                {
                    "llm_required_for_side_effect_tools": True,
                    "reason": "生产模式：副作用工具必须由 LLM 或人工提案",
                    "trace_id": "trace_policy_require_llm_for_side_effects",
                    "now": "2026-06-28T14:00:00+08:00",
                }
            )
            assert policy_update["policy"]["llm_required_for_side_effect_tools"] is True

            backend_plan = service._validate_tool_plan(
                trace_id="trace_backend_side_effect_rejected",
                plan={
                    "source": "backend_fallback",
                    "stage": "after_open_game_search",
                    "fallback_used": True,
                    "tool_calls": [
                        {
                            "tool_name": "search_candidate_customers",
                            "arguments": {},
                            "reason": "后端兜底只读候选人搜索。",
                            "requested_by": "backend_fallback",
                        },
                        {
                            "tool_name": "send_message",
                            "arguments": {"execution_mode": "create_pending_outbox"},
                            "reason": "后端兜底想创建待审批邀约。",
                            "requested_by": "backend_fallback",
                        },
                    ],
                },
                stage="after_open_game_search",
                game=game,
                missing_fields=[],
                tool_results={},
                now=now,
            )

            assert [item["tool_name"] for item in backend_plan["tool_calls"]] == [
                "search_candidate_customers"
            ]
            rejected_send = backend_plan["rejected_actions"][0]
            assert rejected_send["tool_name"] == "send_message"
            assert rejected_send["validation"]["code"] == (
                "runtime_policy_llm_required_for_side_effect_tool"
            )

            llm_plan = service._validate_tool_plan(
                trace_id="trace_llm_side_effect_allowed",
                plan={
                    "source": "llm",
                    "stage": "after_open_game_search",
                    "fallback_used": False,
                    "tool_calls": [
                        {
                            "tool_name": "send_message",
                            "arguments": {"execution_mode": "direct_send"},
                            "reason": "LLM 提议创建邀约，后端应降级为待审批 outbox。",
                            "requested_by": "llm",
                        }
                    ],
                },
                stage="after_open_game_search",
                game=game,
                missing_fields=[],
                tool_results={},
                now=now,
            )

            assert llm_plan["rejected_actions"] == []
            allowed_send = llm_plan["validated_actions"][0]
            assert allowed_send["tool_name"] == "send_message"
            assert allowed_send["validation"]["allowed"] is True
            assert llm_plan["tool_calls"][0]["arguments"]["execution_mode"] == "create_pending_outbox"
            assert any("direct_send" in note for note in allowed_send["validation"]["notes"])
    finally:
        restore_env(saved_env)


def test_runtime_policy_production_mode_enables_strict_llm_requirements() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)

            result = service.update_runtime_policy(
                {
                    "controlled_agent_mode": "production",
                    "reason": "切到生产受控模式",
                    "trace_id": "trace_policy_production_mode",
                    "now": "2026-06-28T14:00:00+08:00",
                }
            )

            assert result["policy"]["controlled_agent_mode"] == "production"
            assert result["policy"]["llm_required_for_side_effect_tools"] is True
            assert result["policy"]["llm_required_for_state_writes"] is True
    finally:
        restore_env(saved_env)


def test_production_controlled_mode_requires_llm_or_human_for_state_writes() -> None:
    saved_env = without_llm_env()
    os.environ["MAHJONG_CONTROLLED_AGENT_MODE"] = "production"
    module = load_boss_trial_module()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)

            policy = store.runtime_policy()
            assert policy["controlled_agent_mode"] == "production"
            assert policy["llm_required_for_side_effect_tools"] is True
            assert policy["llm_required_for_state_writes"] is True

            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "prod_strict_no_llm",
                    "text": "下午两点 0.5 无烟杭麻，173，四小时，帮我组一桌",
                    "now": "2026-06-28T10:00:00+08:00",
                }
            )

            semantic_plan = next(
                item for item in result["agent_actions"] if item["stage"] == "user_semantic_action"
            )
            rejected_semantic = semantic_plan["rejected_actions"][0]
            assert rejected_semantic["code"] == "runtime_policy_llm_required_for_state_write"
            assert result["parsed"]["semantic_action"]["effective_action"] == "human_review"
            assert result["tool_results"]["search_candidate_customers"]["called"] is False
            assert result["tool_results"]["send_message"]["called"] is False
            assert result["outbox"] == []
            assert result["state"]["games"] == []
    finally:
        restore_env(saved_env)


def test_production_controlled_mode_rejects_fallback_candidate_state_write() -> None:
    saved_env = without_llm_env()
    os.environ["MAHJONG_CONTROLLED_AGENT_MODE"] = "production"
    module = load_boss_trial_module()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            created = service.manual_create_game(
                {
                    "organizer_id": "zhang",
                    "organizer_name": "张哥",
                    "game_type": "hangzhou_mahjong",
                    "variant": "caiqiao",
                    "level": "0.5",
                    "start_at": "2026-06-28T14:00:00+08:00",
                    "duration_hours": 4,
                    "current_player_count": 1,
                    "missing_count": 3,
                    "smoke": "no_smoke",
                    "now": "2026-06-28T10:00:00+08:00",
                }
            )
            game_id = created["game"]["id"]
            outbox_id = store.create_outbox(
                game_id=game_id,
                customer_id="ran",
                customer_name="冉姐",
                message_text="冉姐，14:00，0.5无烟，约4小时，打吗？",
                score=100,
                reasons=["测试候选人"],
                warnings=[],
            )
            original_status = store.outbox_item(outbox_id)["status"]

            reply = service.candidate_message(
                {
                    "outbox_id": outbox_id,
                    "text": "可以",
                    "trace_id": "trace_prod_candidate_requires_llm",
                    "now": "2026-06-28T10:03:00+08:00",
                }
            )

            assert reply["ok"] is False
            assert reply["rejected"] is True
            assert reply["outbox_item"]["status"] == original_status
            candidate_plan = reply["agent_actions"][0]
            rejected_action = candidate_plan["rejected_actions"][0]
            assert rejected_action["tool_name"] == "record_candidate_feedback"
            assert rejected_action["code"] == "runtime_policy_llm_required_for_state_write"
    finally:
        restore_env(saved_env)


def test_user_semantic_action_records_llm_create_game_proposal() -> None:
    saved_env = without_llm_env()
    os.environ.update(
        {
            "MAHJONG_LLM_API_KEY": "test-key",
            "MAHJONG_LLM_PROVIDER": "openai",
            "MAHJONG_LLM_MODEL": "test-model",
            "MAHJONG_LLM_BASE_URL": "https://example.invalid/v1",
        }
    )
    module = load_boss_trial_module()

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            system_prompt = self.payload["messages"][0]["content"]
            if "语义解析器" in system_prompt:
                content = {
                    "is_mahjong_related": True,
                    "intent": "find_players",
                    "proposed_action": "create_game",
                    "confidence": 0.92,
                    "normalized_text": "下午两点 0.5 无烟杭麻，173，四小时，帮我组一桌",
                    "reply_text": "好的，我帮你问问。",
                    "needs_human_review": False,
                    "facts": {"reasoning_summary": "用户明确要求老板按完整条件组局。"},
                }
            elif "工具规划器" in system_prompt:
                content = {"tool_calls": [], "reasoning_summary": "本测试只验证语义动作提案。"}
            else:
                content = {
                    "reply_text": "好的，我帮你问问。",
                    "risk_level": "low",
                    "reasoning_summary": "信息完整，极简确认。",
                    "notes": ["测试"],
                }
            return json.dumps(
                {
                    "choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                },
                ensure_ascii=False,
            ).encode("utf-8")

    original_urlopen = module.urllib.request.urlopen

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        return FakeResponse(payload)

    module.urllib.request.urlopen = fake_urlopen
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "group_a",
                    "text": "下午两点 0.5 无烟杭麻，173，四小时，帮我组一桌",
                    "now": "2026-06-27T10:00:00+08:00",
                }
            )

            semantic_plan = next(
                plan for plan in result["agent_actions"] if plan["stage"] == "user_semantic_action"
            )
            semantic_action = semantic_plan["validated_actions"][0]
            assert semantic_plan["source"] == "llm"
            assert semantic_action["tool_name"] == "propose_user_action"
            assert semantic_action["allowed"] is True
            assert result["parsed"]["semantic_action"]["source"] == "llm"
            assert result["parsed"]["semantic_action"]["proposed_action"] == "create_game"
            assert result["parsed"]["semantic_action"]["effective_action"] == "create_game"

            create_plan = next(plan for plan in result["agent_actions"] if plan["stage"] == "create_game")
            create_action = create_plan["validated_actions"][0]
            assert create_plan["source"] == "llm"
            assert create_action["proposed_by"] == "llm"
            assert create_action["allowed"] is True
            assert result["state"]["games"]
    finally:
        module.urllib.request.urlopen = original_urlopen
        restore_env(saved_env)


def test_group_request_without_party_size_asks_confirmation_and_skips_invites() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "group_a",
                    "text": "下午两点 0.5 无烟杭麻，打4小时，帮我组一桌",
                    "now": "2026-06-27T10:00:00+08:00",
                }
            )

            assert result["parsed"]["current_player_count"] is None
            assert result["parsed"]["missing_count"] is None
            assert result["missing_fields"] == ["known_players"]
            assert "你一个人吗" in result["suggested_reply"]["text"]
            assert result["tool_results"]["search_candidate_customers"]["called"] is False
            assert result["tool_results"]["send_message"]["called"] is False
            assert result["outbox"] == []
            assert result["state"]["games"] == []
    finally:
        restore_env(saved_env)


def test_pool_inquiry_without_match_does_not_create_board_game_or_invites() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "group_a",
                    "text": "通常局有人吗",
                    "now": "2026-06-27T17:20:00+08:00",
                }
            )

            assert result["parsed"]["user_intent"] == "咨询现有局"
            assert result["tool_results"]["search_current_open_games"]["called"] is True
            assert result["pool_matches"] == []
            assert result["outbox"] == []
            assert result["state"]["games"] == []
            assert "要组一个吗" in result["suggested_reply"]["text"]
            assert "明天" not in result["suggested_reply"]["text"]
            assert "几个人" not in result["suggested_reply"]["text"]
    finally:
        restore_env(saved_env)


def test_pool_inquiry_uses_semantic_llm_with_profile_and_normalization_context() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    original_urlopen = module.urllib.request.urlopen
    captured_semantic_payloads: list[dict] = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            system_prompt = self.payload["messages"][0]["content"]
            if "语义解析器" in system_prompt:
                captured_semantic_payloads.append(self.payload)
                content = {
                    "is_mahjong_related": True,
                    "intent": "find_players",
                    "proposed_action": "search_existing_games",
                    "confidence": 0.9,
                    "normalized_text": "通宵 0.5 有人吗",
                    "reply_text": "我帮你看看有没有合适的。",
                    "reasoning_summary": "结合画像和麻将语境，将 0。5 理解为 0.5 档并先查现有局。",
                    "needs_human_review": False,
                    "slots": {
                        "query_mode": {"value": "search_existing", "confidence": 0.9, "evidence": "有人吗"},
                        "game_type": {
                            "value": "hangzhou_mahjong",
                            "confidence": 0.8,
                            "source": "region_default",
                            "evidence": "杭州默认",
                        },
                        "level": {
                            "value": "0.5",
                            "confidence": 0.92,
                            "source": "inferred",
                            "evidence": "0。5",
                        },
                        "duration_mode": {
                            "value": "overnight",
                            "confidence": 0.9,
                            "source": "explicit",
                            "evidence": "通宵",
                        },
                    },
                    "facts": {"reason": "测试"},
                }
            else:
                content = {"reply_text": "暂时没有，要不要帮你组一个？", "reasoning_summary": "测试兜底"}
            return json.dumps(
                {
                    "choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                },
                ensure_ascii=False,
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        return FakeResponse(payload)

    try:
        os.environ.update(
            {
                "MAHJONG_LLM_PROVIDER": "deepseek",
                "MAHJONG_LLM_API_KEY": "test-key",
                "MAHJONG_LLM_MODEL": "deepseek-v4-flash",
                "MAHJONG_LLM_BASE_URL": "https://example.invalid",
            }
        )
        module.urllib.request.urlopen = fake_urlopen
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "test02",
                    "text": "通宵0。5有人吗",
                    "now": "2026-06-28T21:49:00+08:00",
                }
            )

            assert captured_semantic_payloads
            semantic_user = json.loads(captured_semantic_payloads[0]["messages"][1]["content"])
            context = semantic_user["context"]
            assert context["text_normalization"]["normalized_text"] == "通宵0.5有人吗"
            assert "stake.decimal_half" in context["text_normalization"]["changed_rule_ids"]
            assert context["customer_profile_summary"]["display_name"] == "张哥"
            assert result["parsed"]["intent_action"] == "inquire_existing_game"
            assert result["tool_results"]["search_current_open_games"]["called"] is True
            assert result["tool_results"]["search_current_open_games"]["result_count"] == 0
            assert result["suggested_reply"]["source"] == "rules"
            assert "要组一个吗" in result["suggested_reply"]["text"]
            assert any(item["source"] == "llm" for item in result["agent_actions"])
    finally:
        module.urllib.request.urlopen = original_urlopen
        restore_env(saved_env)


def test_llm_pool_no_match_explicit_grouping_continues_to_invites() -> None:
    saved_env = without_llm_env()
    os.environ.update(
        {
            "MAHJONG_LLM_API_KEY": "test-key",
            "MAHJONG_LLM_PROVIDER": "openai",
            "MAHJONG_LLM_MODEL": "test-model",
            "MAHJONG_LLM_BASE_URL": "https://example.invalid/v1",
        }
    )
    module = load_boss_trial_module()
    captured_tool_stages: list[str] = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            system_prompt = self.payload["messages"][0]["content"]
            if "工具规划器" in system_prompt:
                prompt = json.loads(self.payload["messages"][1]["content"])
                stage = prompt["stage"]
                captured_tool_stages.append(stage)
                if stage == "before_open_game_search":
                    content = {
                        "tool_calls": [
                            {
                                "tool_name": "search_current_open_games",
                                "arguments": {},
                                "reason": "先查有没有可拼的现成局。",
                            }
                        ],
                        "reasoning_summary": "先查当前局池。",
                    }
                elif stage == "after_open_game_search":
                    content = {
                        "tool_calls": [
                            {
                                "tool_name": "search_candidate_customers",
                                "arguments": {},
                                "reason": "没有现成局，继续搜索候选人。",
                            },
                            {
                                "tool_name": "send_message",
                                "arguments": {"execution_mode": "create_pending_outbox"},
                                "reason": "生成待审批邀约。",
                            },
                        ],
                        "reasoning_summary": "现有局无匹配，进入组局邀约。",
                    }
                else:
                    content = {"tool_calls": [], "reasoning_summary": "测试未覆盖阶段。"}
            else:
                content = {
                    "reply_text": "好的，我帮你问问。",
                    "risk_level": "low",
                    "reasoning_summary": "候选人搜索和待审批邀约已生成，给客户极简确认。",
                    "notes": ["测试"],
                }
            return json.dumps(
                {
                    "choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                },
                ensure_ascii=False,
            ).encode("utf-8")

    original_urlopen = module.urllib.request.urlopen

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        return FakeResponse(payload)

    module.urllib.request.urlopen = fake_urlopen
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            service.responder.llm_resolver = None
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "tool_chain",
                    "text": "通宵有人吗，帮我组一个呗，烟都行，我一个人，一块或者五毛也都可以，人齐开",
                    "now": "2026-06-28T22:21:00+08:00",
                }
            )

            assert captured_tool_stages == ["before_open_game_search", "after_open_game_search"]
            assert result["tool_results"]["search_current_open_games"]["called"] is True
            assert result["tool_results"]["search_current_open_games"]["result_count"] == 0
            assert result["parsed"]["semantic_action"]["effective_action"] == "create_game"
            assert result["parsed"]["intent_action"] == "queue_invites"
            assert result["tool_results"]["search_candidate_customers"]["called"] is True
            assert result["tool_results"]["send_message"]["called"] is True
            assert result["outbox"]
            assert result["suggested_reply"]["text"] == "好的，我帮你问问。"
    finally:
        module.urllib.request.urlopen = original_urlopen
        restore_env(saved_env)


def test_short_followup_uses_previous_agent_reply_in_llm_context() -> None:
    saved_env = without_llm_env()
    os.environ.update(
        {
            "MAHJONG_LLM_API_KEY": "test-key",
            "MAHJONG_LLM_PROVIDER": "openai",
            "MAHJONG_LLM_MODEL": "test-model",
            "MAHJONG_LLM_BASE_URL": "https://example.invalid/v1",
        }
    )
    module = load_boss_trial_module()
    captured_semantic_contexts: list[dict] = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            system_prompt = self.payload["messages"][0]["content"]
            if "语义解析器" in system_prompt:
                user_payload = json.loads(self.payload["messages"][1]["content"])
                context = user_payload["context"]
                captured_semantic_contexts.append(context)
                followup = context.get("workflow_followup_context") or {}
                if followup:
                    content = {
                        "is_mahjong_related": True,
                        "intent": "find_players",
                        "proposed_action": "create_game",
                        "confidence": 0.9,
                        "normalized_text": "通宵 0.5 烟都可 一缺三 人齐开 帮我组一桌",
                        "reply_text": "好的，我帮你问问。",
                        "reasoning_summary": "用户是在确认上一轮“要组一个吗”。",
                        "needs_human_review": False,
                        "slots": {
                            "query_mode": {"value": "create_new", "confidence": 0.9, "evidence": "上一轮问要组一个吗，本轮可以"},
                            "game_type": {"value": "hangzhou_mahjong", "confidence": 0.8, "source": "region_default", "evidence": "杭州默认"},
                            "level": {"value": "0.5", "confidence": 0.92, "source": "inferred", "evidence": "上一轮0。5"},
                            "start_time_mode": {"value": "people_ready", "confidence": 0.82, "source": "inferred", "evidence": "通宵现成局无匹配后确认新组"},
                            "duration_mode": {"value": "overnight", "confidence": 0.9, "source": "explicit", "evidence": "通宵"},
                            "known_players": {"value": 1, "confidence": 0.9, "source": "profile", "evidence": "张哥画像常一个人来"},
                            "missing_count": {"value": 3, "confidence": 0.82, "source": "inferred", "evidence": "画像+确认新组"},
                            "smoke": {"value": "any", "confidence": 0.8, "source": "inferred", "evidence": "张哥可烟可无烟"},
                        },
                        "facts": {"reason": "测试"},
                    }
                else:
                    content = {
                        "is_mahjong_related": True,
                        "intent": "find_players",
                        "proposed_action": "search_existing_games",
                        "confidence": 0.88,
                        "normalized_text": "通宵 0.5 有人吗",
                        "reply_text": "我帮你看看。",
                        "reasoning_summary": "用户先问有没有现成局。",
                        "needs_human_review": False,
                        "slots": {
                            "query_mode": {"value": "search_existing", "confidence": 0.9, "evidence": "有人吗"},
                            "game_type": {"value": "hangzhou_mahjong", "confidence": 0.8, "source": "region_default", "evidence": "杭州默认"},
                            "level": {"value": "0.5", "confidence": 0.92, "source": "inferred", "evidence": "0。5"},
                            "duration_mode": {"value": "overnight", "confidence": 0.9, "source": "explicit", "evidence": "通宵"},
                        },
                        "facts": {"reason": "测试"},
                    }
            elif "工具规划器" in system_prompt:
                user_payload = json.loads(self.payload["messages"][1]["content"])
                if user_payload["stage"] == "after_open_game_search":
                    assert user_payload["workflow_followup_context"]["previous_system_suggested_reply"]
                    content = {
                        "tool_calls": [
                            {"tool_name": "search_candidate_customers", "arguments": {}, "reason": "确认新组后搜索候选人。"},
                            {
                                "tool_name": "send_message",
                                "arguments": {"execution_mode": "create_pending_outbox"},
                                "reason": "创建待审批邀约。",
                            },
                        ],
                        "reasoning_summary": "上一轮确认新组，进入候选人和 outbox。",
                    }
                else:
                    content = {"tool_calls": [], "reasoning_summary": "本阶段无需工具。"}
            elif "私聊邀约起草助手" in system_prompt:
                user_payload = json.loads(self.payload["messages"][1]["content"])
                content = {
                    "drafts": [
                        {
                            "customer_id": item["customer_id"],
                            "message_text": f"{item['display_name']}，通宵0.5，打吗？",
                            "reasoning_summary": "极简邀约。",
                        }
                        for item in user_payload["candidates"]
                    ]
                }
            else:
                content = {
                    "reply_text": "好的，我帮你问问。",
                    "risk_level": "low",
                    "reasoning_summary": "用户确认新组局，候选人草稿已创建。",
                    "notes": ["测试"],
                }
            return json.dumps(
                {
                    "choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                },
                ensure_ascii=False,
            ).encode("utf-8")

    original_urlopen = module.urllib.request.urlopen

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        return FakeResponse(payload)

    module.urllib.request.urlopen = fake_urlopen
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            base = {
                "sender_name": "张哥",
                "sender_id": "zhang",
                "conversation_id": "followup_ctx",
                "now": "2026-06-28T22:55:00+08:00",
            }

            first = service.analyze({**base, "text": "通宵0。5有人吗"})
            second = service.analyze({**base, "text": "可以", "now": "2026-06-28T22:55:20+08:00"})

            assert "要组一个吗" in first["suggested_reply"]["text"]
            followup_contexts = [
                item.get("workflow_followup_context") or {}
                for item in captured_semantic_contexts
                if item.get("workflow_followup_context")
            ]
            assert followup_contexts
            assert followup_contexts[0]["previous_system_suggested_reply"] == first["suggested_reply"]["text"]
            assert followup_contexts[0]["current_user_text"] == "可以"
            assert second["parsed"]["semantic_action"]["source"] == "llm"
            assert second["parsed"]["semantic_action"]["effective_action"] == "create_game"
            assert second["tool_results"]["search_candidate_customers"]["called"] is True
            assert second["tool_results"]["send_message"]["called"] is True
            assert second["outbox"]
            assert second["suggested_reply"]["text"] == "好的，我帮你问问。"
    finally:
        module.urllib.request.urlopen = original_urlopen
        restore_env(saved_env)


def test_truncated_followup_semantic_action_still_reaches_candidate_tools() -> None:
    saved_env = without_llm_env()
    os.environ.update(
        {
            "MAHJONG_LLM_API_KEY": "test-key",
            "MAHJONG_LLM_PROVIDER": "openai",
            "MAHJONG_LLM_MODEL": "test-model",
            "MAHJONG_LLM_BASE_URL": "https://example.invalid/v1",
            "MAHJONG_LLM_MAX_COMPLETION_TOKENS": "512",
        }
    )
    module = load_boss_trial_module()

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            system_prompt = self.payload["messages"][0]["content"]
            if "语义解析器" in system_prompt:
                user_payload = json.loads(self.payload["messages"][1]["content"])
                followup = user_payload["context"].get("workflow_followup_context") or {}
                if followup:
                    content = (
                        '{\n'
                        '  "is_mahjong_related": true,\n'
                        '  "intent": "find_players",\n'
                        '  "proposed_action": "create_game",\n'
                        '  "confidence": 0.85,\n'
                        '  "normalized_text": "通宵 0.5 烟都可 一缺三 人齐开 帮我组一桌",\n'
                        '  "reply_text": "好的，我帮你问问。",\n'
                        '  "reasoning_summary": "用户是在确认上一轮要组一个吗，应该进入找候选人流程",\n'
                        '  "needs_human_review": false,\n'
                        '  "slots": {"duration_mode": {"value": "overnight"'
                    )
                    finish_reason = "length"
                else:
                    content = json.dumps(
                        {
                            "is_mahjong_related": True,
                            "intent": "find_players",
                            "proposed_action": "search_existing_games",
                            "confidence": 0.88,
                            "normalized_text": "通宵 0.5 有人吗",
                            "reply_text": "我帮你看看。",
                            "reasoning_summary": "用户先问有没有现成局。",
                            "needs_human_review": False,
                            "facts": {"reason": "测试"},
                        },
                        ensure_ascii=False,
                    )
                    finish_reason = "stop"
            elif "工具规划器" in system_prompt:
                user_payload = json.loads(self.payload["messages"][1]["content"])
                if user_payload["stage"] == "after_open_game_search":
                    content = json.dumps(
                        {
                            "tool_calls": [
                                {"tool_name": "search_candidate_customers", "arguments": {}, "reason": "确认新组后搜索候选人。"},
                                {"tool_name": "send_message", "arguments": {"execution_mode": "create_pending_outbox"}, "reason": "创建待审批邀约。"},
                            ],
                            "reasoning_summary": "确认新组，进入候选人和 outbox。",
                        },
                        ensure_ascii=False,
                    )
                else:
                    content = json.dumps({"tool_calls": [], "reasoning_summary": "本阶段无需工具。"}, ensure_ascii=False)
                finish_reason = "stop"
            elif "私聊邀约起草助手" in system_prompt:
                user_payload = json.loads(self.payload["messages"][1]["content"])
                content = json.dumps(
                    {
                        "drafts": [
                            {
                                "customer_id": item["customer_id"],
                                "message_text": f"{item['display_name']}，通宵0.5，打吗？",
                                "reasoning_summary": "极简邀约。",
                            }
                            for item in user_payload["candidates"]
                        ]
                    },
                    ensure_ascii=False,
                )
                finish_reason = "stop"
            else:
                content = json.dumps(
                    {
                        "reply_text": "好的，我帮你问问。",
                        "risk_level": "low",
                        "reasoning_summary": "用户确认新组局，候选人草稿已创建。",
                        "notes": ["测试"],
                    },
                    ensure_ascii=False,
                )
                finish_reason = "stop"
            return json.dumps(
                {
                    "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                },
                ensure_ascii=False,
            ).encode("utf-8")

    original_urlopen = module.urllib.request.urlopen

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        return FakeResponse(payload)

    module.urllib.request.urlopen = fake_urlopen
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            base = {
                "sender_name": "张哥",
                "sender_id": "zhang",
                "conversation_id": "truncated_followup_ctx",
                "now": "2026-06-28T22:55:00+08:00",
            }

            first = service.analyze({**base, "text": "通宵0。5有人吗"})
            second = service.analyze({**base, "text": "可以", "now": "2026-06-28T22:55:20+08:00"})

            assert "要组一个吗" in first["suggested_reply"]["text"]
            assert second["parsed"]["semantic_action"]["source"] == "llm"
            assert second["parsed"]["semantic_action"]["effective_action"] == "create_game"
            assert second["tool_results"]["search_candidate_customers"]["called"] is True
            assert second["tool_results"]["send_message"]["called"] is True
            assert second["outbox"]
    finally:
        module.urllib.request.urlopen = original_urlopen
        restore_env(saved_env)


def test_suggested_reply_guard_downgrades_unbacked_invite_promise() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = module.BossTrialService(module.TrialStore(Path(temp_dir) / "trial.db"))
            game = module.GameRequest(
                organizer_id="zhang",
                organizer_name="张哥",
                channel_id="boss_trial",
                game_type="hangzhou_mahjong",
                ruleset="hangzhou_mahjong",
                level="0.5",
                current_player_count=1,
                missing_count=3,
                rules=["杭麻", "无烟", "人齐开", "通宵"],
            )

            guarded = service._guard_suggested_reply(
                "好的，我帮你问问。",
                source_text="可以，帮我组一个吧",
                effective_text="通宵有人吗\n可以，帮我组一个吧",
                sender_id="zhang",
                game=game,
                missing_fields=[],
                pool_matches=[],
                tool_results={
                    "search_current_open_games": {"called": True, "result_count": 0},
                    "search_candidate_customers": {"called": False, "result_count": 0},
                    "send_message": {"called": False, "result_count": 0, "outbox": []},
                },
            )

            assert guarded == "好的，我先帮你留意下。"
    finally:
        restore_env(saved_env)


def test_contextual_group_one_without_game_context_asks_clarification_not_waiting() -> None:
    saved_env = without_llm_env()
    os.environ.update(
        {
            "MAHJONG_LLM_API_KEY": "test-key",
            "MAHJONG_LLM_PROVIDER": "openai",
            "MAHJONG_LLM_MODEL": "test-model",
            "MAHJONG_LLM_BASE_URL": "https://example.invalid/v1",
        }
    )
    module = load_boss_trial_module()

    class FakeCache:
        def __init__(self) -> None:
            self.data = {}

        def get_json(self, key, default):
            return self.data.get(key, default)

        def set_json(self, key, value, ttl_seconds):
            self.data[key] = value

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            system_prompt = self.payload["messages"][0]["content"]
            if "语义解析器" in system_prompt:
                user_payload = json.loads(self.payload["messages"][1]["content"])
                followup = user_payload["context"].get("workflow_followup_context") or {}
                if followup:
                    content = {
                        "is_mahjong_related": True,
                        "intent": "find_players",
                        "proposed_action": "create_game",
                        "confidence": 0.85,
                        "normalized_text": "通宵有人吗 组一个",
                        "reply_text": "好的，我帮你问问。",
                        "needs_human_review": False,
                        "reasoning_summary": "用户确认上一轮要组一个，进入组局意图。",
                        "slots": {"duration_mode": "overnight", "start_time_mode": "people_ready"},
                    }
                else:
                    content = {
                        "is_mahjong_related": True,
                        "intent": "find_players",
                        "proposed_action": "search_existing_games",
                        "confidence": 0.86,
                        "normalized_text": "通宵有人吗",
                        "reply_text": "我帮你看看。",
                        "needs_human_review": False,
                        "reasoning_summary": "用户先问有没有通宵局。",
                    }
            elif "工具规划器" in system_prompt:
                content = {"tool_calls": [], "reasoning_summary": "信息不足，不调用候选人或发送工具。"}
            else:
                content = {
                    "reply_text": "好的，我帮你问问。",
                    "risk_level": "low",
                    "reasoning_summary": "用户已确认组局，但还缺可落库条件。",
                    "notes": ["测试"],
                }
            return json.dumps(
                {
                    "choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                },
                ensure_ascii=False,
            ).encode("utf-8")

    original_urlopen = module.urllib.request.urlopen

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        return FakeResponse(payload)

    module.urllib.request.urlopen = fake_urlopen
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = FakeCache()
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store, cache=cache)
            service.save_customer(
                {
                    "id": "zhang",
                    "display_name": "张哥",
                    "preferred_games": ["杭麻", "财敲"],
                    "preferred_levels": ["0.5", "1"],
                    "usual_party_size": 1,
                    "usual_party_size_confidence": 0.7,
                    "smoke_preference": "any",
                    "notes": "常一个人来，杭麻财敲常打0.5或1块，也能接受烟况灵活。",
                }
            )
            base_payload = {
                "sender_name": "张哥",
                "sender_id": "zhang",
                "conversation_id": "test01_contextual_group_one",
                "now": "2026-06-29T23:00:00+08:00",
            }

            first = service.analyze({**base_payload, "text": "通宵有人吗"})
            second = service.analyze({**base_payload, "text": "组一个", "now": "2026-06-29T23:00:20+08:00"})

            assert "要组一个吗" in first["suggested_reply"]["text"]
            assert second["parsed"]["semantic_action"]["source"] == "llm"
            assert second["parsed"]["semantic_action"]["proposed_action"] == "create_game"
            assert second["parsed"]["semantic_action"]["effective_action"] == "ask_clarification"
            assert second["parsed"]["semantic_action"]["validation"]["code"] == "missing_game_context"
            assert second["tool_results"]["search_candidate_customers"]["called"] is False
            assert second["tool_results"]["send_message"]["called"] is False
            assert second["suggested_reply"]["text"] != "好的，我先帮你留意下。"
            assert "你一个人吗" in second["suggested_reply"]["text"]
    finally:
        module.urllib.request.urlopen = original_urlopen
        restore_env(saved_env)


def test_multiturn_flexible_start_and_overnight_continue_to_candidate_search() -> None:
    module = load_boss_trial_module()

    class FakeCache:
        def __init__(self) -> None:
            self.data = {}

        def get_json(self, key, default):
            return self.data.get(key, default)

        def set_json(self, key, value, ttl_seconds):
            self.data[key] = value

    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = FakeCache()
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store, cache=cache)
            base_payload = {
                "sender_name": "张哥",
                "sender_id": "zhang",
                "conversation_id": "flexible_overnight",
            }

            first = service.analyze(
                {
                    **base_payload,
                    "text": "通宵有人吗",
                    "now": "2026-06-28T21:49:00+08:00",
                }
            )
            second = service.analyze(
                {
                    **base_payload,
                    "text": "帮我组一个呗，烟都行，我一个人，一块或者五毛也都可以",
                    "now": "2026-06-28T21:50:00+08:00",
                }
            )
            third = service.analyze(
                {
                    **base_payload,
                    "text": "尽快开吧，时间可以再商量",
                    "now": "2026-06-28T22:04:00+08:00",
                }
            )

            assert first["parsed"]["intent_action"] == "inquire_existing_game"
            assert "start_time" in second["missing_fields"]
            assert third["used_short_memory"] is True
            assert "通宵有人吗" in third["effective_text"]
            assert "尽快开吧" in third["effective_text"]
            assert third["missing_fields"] == []
            assert third["parsed"]["start_time"] == "人齐开"
            assert third["parsed"]["start_time_mode"] == "people_ready"
            assert third["parsed"]["duration_mode"] == "overnight"
            assert third["parsed"]["duration_text"] == "通宵"
            assert {"人齐开", "通宵", "烟况都可"}.issubset(set(third["parsed"]["rules"]))
            assert third["parsed"]["current_player_count"] == 1
            assert third["parsed"]["missing_count"] == 3
            assert third["parsed"]["intent_action"] == "queue_invites"
            assert third["tool_results"]["search_candidate_customers"]["called"] is True
            assert third["tool_results"]["send_message"]["called"] is True
            assert third["outbox"]
            assert all("人齐开" in item["message_text"] for item in third["outbox"])
            assert all("通宵" in item["message_text"] for item in third["outbox"])
            assert "几点" not in third["suggested_reply"]["text"]
            assert "几个小时" not in third["suggested_reply"]["text"]
    finally:
        restore_env(saved_env)


def test_new_pool_inquiry_does_not_merge_previous_group_request_memory() -> None:
    module = load_boss_trial_module()

    class FakeCache:
        def __init__(self) -> None:
            self.data = {}

        def get_json(self, key, default):
            return self.data.get(key, default)

        def set_json(self, key, value, ttl_seconds):
            self.data[key] = value

    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = FakeCache()
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store, cache=cache)
            base_payload = {
                "sender_name": "张哥",
                "sender_id": "zhang",
                "conversation_id": "group_a",
            }
            first = service.analyze(
                {
                    **base_payload,
                    "text": "下午两点 0.5 无烟杭麻，打4小时，帮我组一桌",
                    "now": "2026-06-27T17:10:00+08:00",
                }
            )
            second = service.analyze(
                {
                    **base_payload,
                    "text": "通常局有人吗",
                    "now": "2026-06-27T17:12:00+08:00",
                }
            )

            assert first["state"]["games"] == []
            assert second["used_short_memory"] is False
            assert second["effective_text"] == "通常局有人吗"
            assert second["parsed"]["user_intent"] == "咨询现有局"
            assert second["state"]["games"] == []
            assert "明天" not in second["suggested_reply"]["text"]
            assert "两点" not in second["suggested_reply"]["text"]
    finally:
        restore_env(saved_env)


def test_llm_cannot_invent_party_size_for_group_request() -> None:
    saved_env = without_llm_env()
    os.environ.update(
        {
            "MAHJONG_LLM_API_KEY": "test-key",
            "MAHJONG_LLM_PROVIDER": "openai",
            "MAHJONG_LLM_MODEL": "test-model",
            "MAHJONG_LLM_BASE_URL": "https://example.invalid/v1",
        }
    )
    module = load_boss_trial_module()

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            system_prompt = self.payload["messages"][0]["content"]
            if "语义解析器" in system_prompt:
                content = {
                    "is_mahjong_related": True,
                    "intent": "find_players",
                    "confidence": 0.95,
                    "normalized_text": "下午两点 0.5 无烟杭麻，打4小时，三缺一，帮我组一桌",
                    "reply_text": "我帮你问问。",
                    "needs_human_review": False,
                    "facts": {"reason": "测试模型幻觉人数"},
                }
            elif "工具规划器" in system_prompt:
                content = {"tool_calls": [], "reasoning_summary": "人数不明确，不调用工具。"}
            else:
                content = {
                    "reply_text": "好的，我帮你问问。",
                    "risk_level": "low",
                    "reasoning_summary": "测试故意给出错误回复。",
                    "notes": ["测试"],
                }
            return json.dumps(
                {
                    "choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                },
                ensure_ascii=False,
            ).encode("utf-8")

    original_urlopen = module.urllib.request.urlopen

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        return FakeResponse(payload)

    module.urllib.request.urlopen = fake_urlopen
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "a001",
                    "text": "下午两点 0.5 无烟杭麻，打4小时，帮我组一桌",
                    "now": "2026-06-27T10:00:00+08:00",
                }
            )

            assert result["parsed"]["current_player_count"] is None
            assert result["parsed"]["missing_count"] is None
            assert result["missing_fields"] == ["known_players"]
            assert "你一个人吗" in result["suggested_reply"]["text"]
            assert result["outbox"] == []
            assert any("移除 LLM" in note for note in result["decision"]["notes"])
    finally:
        module.urllib.request.urlopen = original_urlopen
        restore_env(saved_env)


def test_analysis_exposes_user_intent_slot_and_board_messages() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "group_a",
                    "text": "下午两点 0.5 无烟杭麻，打4小时，一缺三，帮我组一桌",
                    "now": "2026-06-27T10:00:00+08:00",
                }
            )

            assert result["parsed"]["intent_action"] == "queue_invites"
            assert result["parsed"]["user_intent"] == "找人组局"
            game = result["state"]["games"][0]
            assert game["source_text"] == "下午两点 0.5 无烟杭麻，打4小时，一缺三，帮我组一桌"
            assert game["reply_text"] == result["suggested_reply"]["text"]
            assert game["parsed"]["user_intent"] == "找人组局"
    finally:
        restore_env(saved_env)


def test_explicit_group_request_without_duration_does_not_create_game() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "missing_duration",
                    "text": "下午两点 0.5 无烟杭麻，一缺三，帮我组一桌",
                    "now": "2026-06-27T10:00:00+08:00",
                }
            )

            assert result["parsed"]["user_intent"] == "想打/想组局，信息待确认"
            assert result["missing_fields"] == ["duration"]
            assert "几个小时" in result["suggested_reply"]["text"]
            assert result["outbox"] == []
            assert result["state"]["games"] == []
    finally:
        restore_env(saved_env)


def test_clear_board_cancels_active_games_but_keeps_history() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "group_a",
                    "text": "下午两点 0.5 无烟杭麻，打4小时，一缺三，帮我组一桌",
                    "now": "2026-06-27T10:00:00+08:00",
                }
            )
            game_id = result["parsed"]["id"]

            assert result["state"]["games"]
            cleared = service.clear_board({"reason": "测试清空当前局看板"})

            assert cleared["ok"] is True
            assert cleared["cleared_count"] == 1
            assert cleared["cleared_game_ids"] == [game_id]
            assert cleared["agent_actions"][0]["protocol"] == "controlled_agent.v1"
            clear_action = cleared["agent_actions"][0]["validated_actions"][0]
            assert clear_action["tool_name"] == "archive_current_games"
            assert clear_action["risk_level"] == "high"
            assert clear_action["approval_required"] is True
            assert clear_action["allowed"] is True
            assert clear_action["idempotency_key"]
            assert clear_action["ledger_status"] == "executed"
            assert cleared["state"]["games"] == []
            history = store.games(include_final=True)
            assert history[0]["id"] == game_id
            assert history[0]["status"] == "已取消"
            assert history[0]["final_reason"] == "测试清空当前局看板"
            assert {item["status"] for item in store.outbox_for_game(game_id)} == {"局取消"}
    finally:
        restore_env(saved_env)


def test_controlled_action_ledger_deduplicates_manual_state_writes() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            payload = {
                "trace_id": "trace_retry_manual_create",
                "organizer_name": "老板手动创建",
                "game_type": "hangzhou_mahjong",
                "start_time": "18:00",
                "level": "0.5",
                "current_player_count": 3,
                "missing_count": 1,
                "duration_hours": 4,
                "smoke": "no_smoke",
                "source_text": "电话：六点0.5无烟三缺一",
                "now": "2026-06-27T17:00:00+08:00",
            }

            first = service.manual_create_game(payload)
            second = service.manual_create_game(payload)

            assert first["ok"] is True
            assert second["ok"] is True
            assert second["deduplicated"] is True
            assert len(store.games(include_final=True)) == 1
            create_actions = [
                item for item in store.controlled_actions()
                if item["stage"] == "manual_create_game"
            ]
            assert len(create_actions) == 1
            assert create_actions[0]["status"] == "executed"

            clear_payload = {
                "trace_id": "trace_retry_clear_board",
                "reason": "测试重复清空",
                "now": "2026-06-27T17:10:00+08:00",
            }
            cleared_first = service.clear_board(clear_payload)
            cleared_second = service.clear_board(clear_payload)

            assert cleared_first["cleared_count"] == 1
            assert cleared_second["deduplicated"] is True
            assert cleared_second["cleared_count"] == 1
            assert store.conn.execute(
                "SELECT COUNT(*) count FROM feedback WHERE feedback_type = 'board_cleared'"
            ).fetchone()["count"] == 1
            clear_actions = [
                item for item in store.controlled_actions()
                if item["stage"] == "clear_board"
            ]
            assert len(clear_actions) == 1
            assert clear_actions[0]["status"] == "executed"
    finally:
        restore_env(saved_env)


def test_controlled_action_ledger_deduplicates_retried_analyze_outbox_creation() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            payload = {
                "trace_id": "trace_retry_analyze",
                "sender_name": "张哥",
                "sender_id": "zhang",
                "conversation_id": "group_a",
                "text": "下午两点 0.5 无烟杭麻，打4小时，一缺三，帮我组一桌",
                "now": "2026-06-27T10:00:00+08:00",
            }

            first = service.analyze(payload)
            second = service.analyze(payload)

            assert first["outbox"]
            assert second["outbox"]
            assert len(store.games(include_final=True)) == 1
            assert len(store.outbox_for_game(first["parsed"]["id"])) == len(first["outbox"])
            send_actions = [
                item for item in store.controlled_actions()
                if item["tool_name"] == "send_message"
            ]
            assert len(send_actions) == 1
            assert send_actions[0]["status"] == "executed"
            assert send_actions[0]["result"]["result_count"] == len(first["outbox"])
            assert any(
                action["tool_name"] == "send_message"
                and action["deduplicated"] is True
                and action["ledger_status"] == "executed"
                for plan in second["agent_actions"]
                for action in plan["validated_actions"]
            )
    finally:
        restore_env(saved_env)


def test_game_timeout_archives_unfilled_game_with_failure_reason() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            created = service.manual_create_game(
                {
                    "organizer_name": "电话李姐",
                    "game_type": "hangzhou_mahjong",
                    "variant": "caiqiao",
                    "level": "0.5",
                    "start_time": "14:00",
                    "current_player_count": 3,
                    "missing_count": 1,
                    "duration_hours": 4,
                    "smoke": "no_smoke",
                    "now": "2026-06-27T10:00:00+08:00",
                }
            )
            game_id = created["game"]["id"]
            state = service.state(now=module.parse_dt("2026-06-27T16:00:00+08:00"))

            assert state["games"] == []
            archived = state["recent_archived_games"][0]
            assert archived["id"] == game_id
            assert archived["status"] == "已取消"
            assert "超过开局时间" in archived["final_reason"]
            assert "仍未补齐" in archived["final_reason"]
    finally:
        restore_env(saved_env)


def test_all_declined_invites_archive_game_and_keep_reason() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "decline_game",
                    "text": "下午两点 0.5 无烟杭麻，打4小时，一缺三，帮我组一桌",
                    "now": "2026-06-27T10:00:00+08:00",
                }
            )
            game_id = result["parsed"]["id"]
            assert result["outbox"]

            for item in result["outbox"]:
                service.feedback(
                    {
                        "game_id": game_id,
                        "outbox_id": item["id"],
                        "customer_id": item["customer_id"],
                        "feedback_type": "declined",
                        "notes": "今天不来",
                        "now": "2026-06-27T10:20:00+08:00",
                    }
                )

            state = service.state(now=module.parse_dt("2026-06-27T10:21:00+08:00"))
            assert state["games"] == []
            archived = state["recent_archived_games"][0]
            assert archived["id"] == game_id
            assert "均拒绝" in archived["final_reason"]
            assert any("今天不来" in customer["notes"] for customer in state["customers"])
    finally:
        restore_env(saved_env)


def test_rule_fallback_followup_does_not_expose_slots() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "group_a",
                    "text": "下午有人打麻将吗？",
                    "now": "2026-06-27T15:00:00+08:00",
                }
            )

            reply = result["suggested_reply"]["text"]
            assert result["suggested_reply"]["source"] == "rules"
            assert "还差" not in reply
            assert "缺失字段" not in reply
            assert "槽位" not in reply
            assert result["parsed"]["user_intent"] == "咨询现有局"
            assert result["state"]["games"] == []
            assert "要组一个吗" in reply
    finally:
        restore_env(saved_env)


def test_complete_game_llm_reply_is_guarded_to_concise_ack() -> None:
    saved_env = without_llm_env()
    os.environ.update(
        {
            "MAHJONG_LLM_API_KEY": "test-key",
            "MAHJONG_LLM_PROVIDER": "openai",
            "MAHJONG_LLM_MODEL": "test-model",
            "MAHJONG_LLM_BASE_URL": "https://example.invalid/v1",
        }
    )
    module = load_boss_trial_module()
    captured = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            system_prompt = self.payload["messages"][0]["content"]
            if "语义解析器" in system_prompt:
                content = {
                    "is_mahjong_related": True,
                    "intent": "find_players",
                    "confidence": 0.9,
                    "normalized_text": "下午两点 0.5 无烟杭麻，打4小时，一缺三，帮我组一桌",
                    "reply_text": "可以，我先帮你看。",
                    "needs_human_review": False,
                    "facts": {"reason": "测试"},
                }
            elif "工具规划器" in system_prompt:
                prompt = json.loads(self.payload["messages"][1]["content"])
                if prompt["stage"] == "after_open_game_search":
                    content = {
                        "tool_calls": [
                            {
                                "tool_name": "search_candidate_customers",
                                "arguments": {},
                                "reason": "信息齐全，搜索候选人。",
                            },
                            {
                                "tool_name": "send_message",
                                "arguments": {"execution_mode": "create_pending_outbox"},
                                "reason": "候选人搜索后创建待审批邀约。",
                            },
                        ],
                        "reasoning_summary": "信息齐全，可以进入候选人搜索和待审批邀约。",
                    }
                else:
                    content = {"tool_calls": [], "reasoning_summary": "本阶段无需工具。"}
            else:
                content = {
                    "reply_text": "好的张哥，我按杭麻财敲0.5、两点、无烟帮你问人，有合适的先跟你确认。",
                    "risk_level": "low",
                    "reasoning_summary": "信息齐全，直接告知将问人。",
                    "notes": ["测试"],
                }
            return json.dumps(
                {
                    "choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                },
                ensure_ascii=False,
            ).encode("utf-8")

    original_urlopen = module.urllib.request.urlopen

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        captured.append(payload)
        return FakeResponse(payload)

    module.urllib.request.urlopen = fake_urlopen
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "group_a",
                    "text": "下午两点 0.5 无烟杭麻，打4小时，一缺三，帮我组一桌",
                    "now": "2026-06-27T10:00:00+08:00",
                }
            )

            reply_prompt = next(
                json.loads(payload["messages"][1]["content"])
                for payload in captured
                if "微信回复起草助手" in payload["messages"][0]["content"]
            )
            assert reply_prompt["reply_style_hint"]["mode"] == "brief_ack"
            assert reply_prompt["tool_results"]["search_candidate_customers"]["called"] is True
            assert reply_prompt["tool_results"]["send_message"]["approval_required"] is True
            assert reply_prompt["tool_results"]["send_message"]["direct_send_executed"] is False
            assert result["suggested_reply"]["source"] == "llm"
            assert result["suggested_reply"]["text"] == "好的，我帮你问问。"
            assert result["parsed"]["user_intent"] == "找人组局"
    finally:
        module.urllib.request.urlopen = original_urlopen
        restore_env(saved_env)


def test_candidate_invites_are_llm_generated_and_minimal() -> None:
    saved_env = without_llm_env()
    os.environ.update(
        {
            "MAHJONG_LLM_API_KEY": "test-key",
            "MAHJONG_LLM_PROVIDER": "openai",
            "MAHJONG_LLM_MODEL": "test-model",
            "MAHJONG_LLM_BASE_URL": "https://example.invalid/v1",
        }
    )
    module = load_boss_trial_module()
    captured = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            system_prompt = self.payload["messages"][0]["content"]
            if "语义解析器" in system_prompt:
                content = {
                    "is_mahjong_related": True,
                    "intent": "find_players",
                    "confidence": 0.9,
                    "normalized_text": "下午两点 0.5 无烟杭麻，打4小时，一缺三，帮我组一桌",
                    "reply_text": "可以，我先帮你看。",
                    "needs_human_review": False,
                    "facts": {"reason": "测试"},
                }
            elif "私聊邀约起草助手" in system_prompt:
                user_payload = json.loads(self.payload["messages"][1]["content"])
                duration = user_payload["public_invite_terms"]["duration"]
                content = {
                    "drafts": [
                        {
                            "customer_id": item["customer_id"],
                            "message_text": f"{item['display_name']}，14:00，0.5无烟，{duration}，打吗？",
                            "reasoning_summary": "极简邀约，不透露缺口和玩法细分。",
                        }
                        for item in user_payload["candidates"]
                    ]
                }
            elif "工具规划器" in system_prompt:
                user_payload = json.loads(self.payload["messages"][1]["content"])
                stage = user_payload["stage"]
                if stage == "after_open_game_search":
                    content = {
                        "tool_calls": [
                            {
                                "tool_name": "search_candidate_customers",
                                "arguments": {},
                                "reason": "信息完整，先搜索候选人。",
                            },
                            {
                                "tool_name": "send_message",
                                "arguments": {"execution_mode": "create_pending_outbox"},
                                "reason": "候选人可用时创建待审批草稿。",
                            },
                        ],
                        "reasoning_summary": "信息完整，需要找候选并创建待审批草稿。",
                    }
                else:
                    content = {"tool_calls": [], "reasoning_summary": "本阶段不需要工具。"}
            else:
                content = {
                    "reply_text": "好的，我帮你问问。",
                    "risk_level": "low",
                    "reasoning_summary": "信息完整，极简确认。",
                    "notes": ["测试"],
                }
            return json.dumps(
                {
                    "choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                },
                ensure_ascii=False,
            ).encode("utf-8")

    original_urlopen = module.urllib.request.urlopen

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        captured.append(payload)
        return FakeResponse(payload)

    module.urllib.request.urlopen = fake_urlopen
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "group_a",
                    "text": "下午两点 0.5 无烟杭麻，打4小时，一缺三，帮我组一桌",
                    "now": "2026-06-27T10:00:00+08:00",
                }
            )

            invite_prompt = next(
                payload for payload in captured if "私聊邀约起草助手" in payload["messages"][0]["content"]
            )
            assert "禁止出现" in invite_prompt["messages"][0]["content"]
            assert result["outbox"]
            assert all(item["draft_source"] == "llm" for item in result["outbox"])
            for item in result["outbox"]:
                draft = item["message_text"]
                assert draft == f"{item['customer_name']}，14:00，0.5无烟，约4小时，打吗？"
                assert "缺" not in draft
                assert "方便来吗" not in draft
                assert "有一桌" not in draft
                assert "杭麻" not in draft
                assert "财敲" not in draft
    finally:
        module.urllib.request.urlopen = original_urlopen
        restore_env(saved_env)


def test_tool_loop_continues_to_send_message_after_candidate_search_only_plan() -> None:
    saved_env = without_llm_env()
    os.environ.update(
        {
            "MAHJONG_LLM_API_KEY": "test-key",
            "MAHJONG_LLM_PROVIDER": "openai",
            "MAHJONG_LLM_MODEL": "test-model",
            "MAHJONG_LLM_BASE_URL": "https://example.invalid/v1",
        }
    )
    module = load_boss_trial_module()
    captured = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            system_prompt = self.payload["messages"][0]["content"]
            user_payload = json.loads(self.payload["messages"][1]["content"])
            if "语义解析器" in system_prompt:
                content = {
                    "is_mahjong_related": True,
                    "intent": "find_players",
                    "confidence": 0.9,
                    "normalized_text": "下午两点 0.5 无烟杭麻，173，四小时",
                    "reply_text": "好的，我帮你问问。",
                    "needs_human_review": False,
                    "facts": {"reason": "测试"},
                }
            elif "工具规划器" in system_prompt:
                stage = user_payload["stage"]
                if stage == "after_open_game_search":
                    content = {
                        "tool_calls": [
                            {
                                "tool_name": "search_candidate_customers",
                                "arguments": {},
                                "reason": "先搜索候选人。",
                            }
                        ],
                        "reasoning_summary": "先找候选人。",
                    }
                elif stage == "after_candidate_search":
                    content = {
                        "tool_calls": [
                            {
                                "tool_name": "send_message",
                                "arguments": {"execution_mode": "direct_send"},
                                "reason": "候选人已有结果，模型误请求直接发送。",
                            }
                        ],
                        "reasoning_summary": "候选人已找到，但后端应拦截直接发送。",
                    }
                else:
                    content = {"tool_calls": [], "reasoning_summary": "本阶段不需要工具。"}
            elif "私聊邀约起草助手" in system_prompt:
                duration = user_payload["public_invite_terms"]["duration"]
                content = {
                    "drafts": [
                        {
                            "customer_id": item["customer_id"],
                            "message_text": f"{item['display_name']}，14:00，0.5无烟，{duration}，打吗？",
                            "reasoning_summary": "带上明确时长，不透露缺口。",
                        }
                        for item in user_payload["candidates"]
                    ]
                }
            else:
                content = {
                    "reply_text": "好的，我帮你问问。",
                    "risk_level": "low",
                    "reasoning_summary": "信息完整，极简确认。",
                    "notes": ["测试"],
                }
            return json.dumps(
                {
                    "choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                },
                ensure_ascii=False,
            ).encode("utf-8")

    original_urlopen = module.urllib.request.urlopen

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        captured.append(payload)
        return FakeResponse(payload)

    module.urllib.request.urlopen = fake_urlopen
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "group_a",
                    "text": "下午两点 0.5 无烟杭麻，173，四小时",
                    "now": "2026-06-27T10:00:00+08:00",
                }
            )

            tool_stage_payloads = [
                json.loads(payload["messages"][1]["content"])
                for payload in captured
                if "工具规划器" in payload["messages"][0]["content"]
            ]
            assert any(item["stage"] == "after_candidate_search" for item in tool_stage_payloads)
            assert result["tool_results"]["search_candidate_customers"]["called"] is True
            assert result["tool_results"]["send_message"]["called"] is True
            assert result["tool_results"]["send_message"]["tool_plan_source"] == "llm"
            assert result["tool_results"]["send_message"]["execution_mode"] == "create_pending_outbox"
            assert result["tool_results"]["send_message"]["direct_send_executed"] is False
            send_plan = next(
                plan for plan in result["agent_actions"] if plan["stage"] == "after_candidate_search"
            )
            send_action = next(
                action for action in send_plan["validated_actions"] if action["tool_name"] == "send_message"
            )
            assert send_action["allowed"] is True
            assert send_action["approval_required"] is True
            assert any("direct_send" in note for note in send_action["notes"])
            assert result["outbox"]
            for item in result["outbox"]:
                assert "约4小时" in item["message_text"]
                assert "缺" not in item["message_text"]
    finally:
        module.urllib.request.urlopen = original_urlopen
        restore_env(saved_env)


def test_llm_tool_planner_failure_filters_side_effect_fallback_tools() -> None:
    saved_env = without_llm_env()
    os.environ.update(
        {
            "MAHJONG_LLM_API_KEY": "test-key",
            "MAHJONG_LLM_PROVIDER": "openai",
            "MAHJONG_LLM_MODEL": "test-model",
            "MAHJONG_LLM_BASE_URL": "https://example.invalid/v1",
        }
    )
    module = load_boss_trial_module()

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [{"message": {"content": "{}"}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
                },
                ensure_ascii=False,
            ).encode("utf-8")

    original_urlopen = module.urllib.request.urlopen

    def fake_urlopen(request, timeout):
        return FakeResponse()

    module.urllib.request.urlopen = fake_urlopen
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            game = module.GameRequest(
                organizer_id="zhang",
                organizer_name="张哥",
                channel_id="boss_trial",
                game_type="hangzhou_mahjong",
                current_player_count=1,
                missing_count=3,
                level="0.5",
                rules=["杭麻", "无烟"],
                start_at=module.parse_dt("2026-06-28T16:00:00+08:00"),
                duration_hours=4,
            )
            plan = service._llm_tool_plan(
                trace_id="trace_llm_tool_plan_fail_closed",
                stage="after_open_game_search",
                sender_id="zhang",
                sender_name="张哥",
                source_text="下午四点 0.5 无烟杭麻 173 4h",
                effective_text="下午四点 0.5 无烟杭麻 173 4h",
                workflow_followup_context={},
                game=game,
                missing_fields=[],
                decision_action="queue_invites",
                tool_results={},
                now=module.parse_dt("2026-06-28T14:00:00+08:00"),
            )

            assert plan["fallback_used"] is True
            assert plan["llm_source"] == "invalid_or_empty_tool_plan"
            assert [item["tool_name"] for item in plan["tool_calls"]] == ["search_candidate_customers"]
            assert all(action["tool_name"] != "send_message" for action in plan["validated_actions"])
    finally:
        module.urllib.request.urlopen = original_urlopen
        restore_env(saved_env)


def test_tool_registry_scopes_tools_and_send_execution_modes_by_stage() -> None:
    module = load_boss_trial_module()

    before_tools = module.tool_specs_for_stage("before_open_game_search")
    after_open_tools = module.tool_specs_for_stage("after_open_game_search")
    after_candidate_tools = module.tool_specs_for_stage("after_candidate_search")
    followup_tools = module.tool_specs_for_stage("organizer_followup_draft")

    assert module.TOOL_REGISTRY_VERSION == "tool_registry.v1"
    assert [item["name"] for item in before_tools] == ["search_current_open_games"]
    assert [item["name"] for item in after_open_tools] == ["search_candidate_customers", "send_message"]
    assert [item["name"] for item in after_candidate_tools] == ["send_message"]
    assert [item["name"] for item in followup_tools] == ["send_message"]
    assert after_candidate_tools[0]["allowed_execution_modes"] == ["create_pending_outbox"]
    assert after_candidate_tools[0]["arguments_schema"]["properties"]["execution_mode"]["enum"] == [
        "create_pending_outbox"
    ]
    assert followup_tools[0]["allowed_execution_modes"] == ["create_pending_followup"]
    assert followup_tools[0]["arguments_schema"]["properties"]["execution_mode"]["enum"] == [
        "create_pending_followup"
    ]


def test_controlled_agent_architecture_contract_versions_ledgers_and_policies() -> None:
    module = load_boss_trial_module()

    assert module.CONTROLLED_AGENT_PROTOCOL_VERSION == "controlled_agent.v1"
    assert module.TOOL_REGISTRY_VERSION == "tool_registry.v1"
    assert module.STATE_MACHINE_VERSION == "state_machine.v1"
    assert module.RUNTIME_POLICY_VERSION == "runtime_policy.v1"
    assert "candidate_feedback" in module.STATE_WRITE_STAGES
    assert "message_delivery" in module.STATE_WRITE_STAGES
    assert "clear_board" in module.STATE_WRITE_STAGES
    assert "controlled_agent_mode" in module.DEFAULT_RUNTIME_POLICY
    assert "llm_required_for_side_effect_tools" in module.DEFAULT_RUNTIME_POLICY
    assert "llm_required_for_state_writes" in module.DEFAULT_RUNTIME_POLICY

    for stage in [
        "before_open_game_search",
        "after_open_game_search",
        "after_candidate_search",
        "organizer_followup_draft",
    ]:
        specs = module.tool_specs_for_stage(stage)
        assert specs, stage
        for spec in specs:
            assert spec["registry_version"] == module.TOOL_REGISTRY_VERSION
            assert "arguments_schema" in spec
            assert "risk_level" in spec
            assert "side_effect" in spec

    with tempfile.TemporaryDirectory() as temp_dir:
        store = module.TrialStore(Path(temp_dir) / "trial.db")
        table_columns = {
            table: {
                str(row["name"])
                for row in store.conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for table in [
                "controlled_actions",
                "approval_requests",
                "message_delivery_attempts",
                "runtime_policies",
                "state_transition_events",
            ]
        }

    assert {
        "action_id",
        "idempotency_key",
        "trace_id",
        "stage",
        "tool_name",
        "proposed_by",
        "source",
        "risk_level",
        "side_effect",
        "approval_required",
        "status",
        "arguments_json",
        "validation_json",
        "result_json",
    } <= table_columns["controlled_actions"]
    assert {
        "target_type",
        "target_id",
        "action_id",
        "idempotency_key",
        "status",
        "original_message_text",
        "final_message_text",
        "metadata_json",
    } <= table_columns["approval_requests"]
    assert {
        "outbox_id",
        "approval_id",
        "channel",
        "message_text",
        "idempotency_key",
        "action_id",
        "trace_id",
        "status",
    } <= table_columns["message_delivery_attempts"]
    assert {"policy_json", "updated_at", "updated_by", "reason"} <= table_columns["runtime_policies"]
    assert {
        "entity_type",
        "entity_id",
        "from_status",
        "to_status",
        "event",
        "allowed",
        "state_machine_version",
        "schema_version",
        "metadata_json",
    } <= table_columns["state_transition_events"]


def test_tool_gateway_strips_unregistered_args_and_downgrades_send_mode() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            game = module.GameRequest(
                organizer_id="zhang",
                organizer_name="张哥",
                channel_id="boss_trial",
                game_type="hangzhou_mahjong",
                current_player_count=3,
                missing_count=1,
                level="0.5",
                rules=["杭麻", "无烟"],
            )

            rejected = service._validate_tool_plan(
                trace_id="trace_tool_registry_reject",
                plan={
                    "source": "llm",
                    "stage": "before_open_game_search",
                    "tool_calls": [
                        {
                            "tool_name": "send_message",
                            "arguments": {"execution_mode": "direct_send"},
                            "reason": "模型错误地想提前发消息。",
                        }
                    ],
                },
                stage="before_open_game_search",
                game=game,
                missing_fields=[],
                tool_results={},
                now=module.parse_dt("2026-06-28T14:00:00+08:00"),
            )
            assert rejected["tool_calls"] == []
            assert rejected["rejected_actions"][0]["validation"]["code"] == "tool_not_available_for_stage"

            plan = service._validate_tool_plan(
                trace_id="trace_tool_registry_followup",
                plan={
                    "source": "llm",
                    "stage": "organizer_followup_draft",
                    "tool_calls": [
                        {
                            "tool_name": "send_message",
                            "arguments": {
                                "execution_mode": "direct_send",
                                "foo": "bar",
                                "recipient_id": "zhang",
                            },
                            "reason": "模型请求直接问发起人。",
                        }
                    ],
                },
                stage="organizer_followup_draft",
                game=game,
                missing_fields=[],
                tool_results={},
                now=module.parse_dt("2026-06-28T14:00:00+08:00"),
            )

            assert plan["tool_calls"][0]["arguments"] == {"execution_mode": "create_pending_followup"}
            action = plan["validated_actions"][0]
            validation = action["validation"]
            assert validation["allowed"] is True
            assert validation["effective_arguments"] == {"execution_mode": "create_pending_followup"}
            assert any("direct_send" in note for note in validation["notes"])
            assert any("foo" in note and "recipient_id" in note for note in validation["notes"])
            assert action["risk_level"] == "high"
            assert action["approval_required"] is True
    finally:
        restore_env(saved_env)


def test_state_machine_blocks_final_game_reopen_and_outbox_regression() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            assert module.STATE_MACHINE_VERSION == "state_machine.v1"
            rejected = module.state_transition_verdict(
                entity_type="game",
                current_status="已取消",
                next_status="待组局",
                event="retry_reopen",
            )
            assert rejected["allowed"] is False
            assert rejected["code"] == "state_transition_rejected"

            parsed = {
                "id": "game_state_machine",
                "status": "open",
                "game_type": "hangzhou_mahjong",
                "game_label": "杭麻",
                "level": "0.5",
                "start_at": "2026-06-28T16:00:00+08:00",
                "start_time": "16:00",
                "duration_hours": 4,
                "current_player_count": 3,
                "missing_count": 1,
                "rules": ["杭麻", "无烟"],
                "summary": "杭麻 0.5档 16:00 缺1 无烟",
            }
            store.create_game(
                game_id="game_state_machine",
                status="待组局",
                organizer_id="zhang",
                organizer_name="张哥",
                source_text="测试局",
                parsed=parsed,
                reply_text="",
                missing_fields=[],
                notes=[],
            )
            cancelled = store.record_feedback(
                {
                    "game_id": "game_state_machine",
                    "feedback_type": "game_cancelled",
                    "notes": "测试取消",
                    "now": "2026-06-28T14:00:00+08:00",
                }
            )
            assert cancelled["state_transitions"][0]["from_status"] == "待组局"
            assert cancelled["state_transitions"][0]["to_status"] == "已取消"

            with pytest.raises(ValueError, match="不允许"):
                store.create_game(
                    game_id="game_state_machine",
                    status="待组局",
                    organizer_id="zhang",
                    organizer_name="张哥",
                    source_text="测试重开",
                    parsed=parsed,
                    reply_text="",
                    missing_fields=[],
                    notes=[],
                )

            store.create_game(
                game_id="game_outbox_state_machine",
                status="待组局",
                organizer_id="zhang",
                organizer_name="张哥",
                source_text="测试邀约",
                parsed={**parsed, "id": "game_outbox_state_machine"},
                reply_text="",
                missing_fields=[],
                notes=[],
            )
            outbox_id = store.create_outbox(
                game_id="game_outbox_state_machine",
                customer_id="amy",
                customer_name="Amy",
                message_text="Amy，16:00，0.5无烟，打吗？",
                score=90,
                reasons=["测试"],
                warnings=[],
            )
            accepted = store.record_feedback(
                {
                    "game_id": "game_outbox_state_machine",
                    "outbox_id": outbox_id,
                    "customer_id": "amy",
                    "feedback_type": "accepted",
                    "now": "2026-06-28T14:01:00+08:00",
                }
            )
            outbox_transition = accepted["state_transitions"][0]
            assert outbox_transition["entity_type"] == "outbox"
            assert outbox_transition["from_status"] == "待审批"
            assert outbox_transition["to_status"] == "已确认"

            game_events = store.state_transition_events(entity_type="game", entity_id="game_state_machine")
            assert [event["to_status"] for event in reversed(game_events)] == ["待组局", "已取消"]
            assert all(event["schema_version"] == module.STATE_TRANSITION_EVENT_SCHEMA_VERSION for event in game_events)

            outbox_events = store.state_transition_events(entity_type="outbox", entity_id=outbox_id)
            assert [event["to_status"] for event in reversed(outbox_events)] == ["待审批", "已确认"]
            assert outbox_events[0]["metadata"]["feedback_type"] == "accepted"

            with pytest.raises(ValueError, match="不允许"):
                store.record_feedback(
                    {
                        "game_id": "game_outbox_state_machine",
                        "outbox_id": outbox_id,
                        "customer_id": "amy",
                        "feedback_type": "no_reply",
                        "now": "2026-06-28T14:02:00+08:00",
                    }
                )
    finally:
        restore_env(saved_env)


def test_boss_trial_wires_llm_resolver_from_env() -> None:
    saved_env = without_llm_env()
    os.environ.update(
        {
            "MAHJONG_LLM_API_KEY": "test-key",
            "MAHJONG_LLM_PROVIDER": "deepseek",
            "MAHJONG_LLM_MODEL": "deepseek-v4-flash",
        }
    )
    try:
        module = load_boss_trial_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)

            assert service.llm_config is not None
            assert service.llm_config.provider == "deepseek"
            assert service.responder.llm_resolver is not None
    finally:
        restore_env(saved_env)


def test_candidate_message_accepts_invite_and_auto_marks_game_success() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "candidate_accept",
                    "text": "下午两点 0.5 无烟杭麻，打4小时，三缺一，帮我组一桌",
                    "now": "2026-06-28T10:00:00+08:00",
                }
            )
            invite = result["outbox"][0]

            reply = service.candidate_message(
                {
                    "outbox_id": invite["id"],
                    "text": "可以我来",
                    "now": "2026-06-28T10:03:00+08:00",
                }
            )

            assert reply["candidate_message"]["feedback_type"] == "accepted"
            assert reply["candidate_message"]["suggested_boss_reply"] == "好的，人齐了。"
            assert "留" not in reply["candidate_message"]["suggested_boss_reply"]
            assert reply["outbox_item"]["status"] == "已确认"
            assert reply["auto_success"]["status"] == "已成局"
            assert reply["state"]["games"] == []
            archived = reply["state"]["recent_archived_games"][0]
            assert archived["status"] == "已成局"
            assert "缺口已补齐" in archived["final_reason"]
    finally:
        restore_env(saved_env)


def test_candidate_message_accepts_invite_reports_272_after_first_join() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "candidate_accept_272",
                    "text": "下午两点 0.5 无烟杭麻，打4小时，一缺三，帮我组一桌",
                    "now": "2026-06-28T10:00:00+08:00",
                }
            )
            invite = result["outbox"][0]

            reply = service.candidate_message(
                {
                    "outbox_id": invite["id"],
                    "text": "可以",
                    "now": "2026-06-28T10:03:00+08:00",
                }
            )

            assert reply["candidate_message"]["feedback_type"] == "accepted"
            assert reply["candidate_message"]["suggested_boss_reply"] == "好的，加你272了。"
            assert "留" not in reply["candidate_message"]["suggested_boss_reply"]
            assert reply["outbox_item"]["status"] == "已确认"
            assert reply["auto_success"] is None
            assert reply["state"]["games"]
            live_game = reply["state"]["games"][0]
            assert live_game["confirmed_count"] == 1
            assert live_game["remaining_missing_count"] == 2
            assert live_game["active_player_count"] == 2
            assert "缺2" in live_game["live_summary"]
            assert "缺3" not in live_game["live_summary"]
    finally:
        restore_env(saved_env)


def test_candidate_message_with_changed_duration_stays_negotiating_and_keeps_thread() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "candidate_duration_negotiation",
                    "text": "下午三点 0.5 无烟杭麻，打5小时，一缺三，帮我组一桌",
                    "now": "2026-06-28T10:00:00+08:00",
                }
            )
            invite = next(item for item in result["outbox"] if item["customer_name"] == "潘姐")

            reply = service.candidate_message(
                {
                    "outbox_id": invite["id"],
                    "text": "可以，不过我想打六个小时",
                    "now": "2026-06-28T10:03:00+08:00",
                }
            )

            candidate = reply["candidate_message"]
            assert candidate["feedback_type"] == "candidate_negotiation"
            assert candidate["status"] == "待协商"
            assert candidate["requested_duration_hours"] == 6.0
            assert candidate["current_duration_hours"] == 5.0
            assert reply["outbox_item"]["status"] == "待协商"
            assert reply["candidate_message"]["suggested_boss_reply"] == "可以，我问下这桌其他人能不能打6小时。"
            assert "加你" not in reply["candidate_message"]["suggested_boss_reply"]
            assert reply["organizer_followup"]["recipient_name"] == "张哥"
            assert "潘姐" in reply["organizer_followup"]["message_text"]
            assert "6小时" in reply["organizer_followup"]["message_text"]
            assert reply["organizer_followup"]["status"] == "待审批"
            assert reply["organizer_followup"]["approval_status"] == "待审批"
            assert reply["organizer_followup"]["approval"]["target_type"] == "followup"
            assert reply["organizer_followup"]["approval"]["status"] == "pending"
            assert reply["organizer_followup"]["approval"]["original_message_text"] == reply["organizer_followup"]["message_text"]
            assert reply["auto_success"] is None
            live_game = reply["state"]["games"][0]
            assert live_game["confirmed_count"] == 0
            assert live_game["remaining_missing_count"] == 3
            assert live_game["followups"]
            assert live_game["followups"][0]["message_text"] == reply["organizer_followup"]["message_text"]
            updated_invite = next(item for item in live_game["outbox"] if item["id"] == invite["id"])
            assert updated_invite["conversation"][-1]["candidate_text"] == "可以，不过我想打六个小时"
            assert updated_invite["conversation"][-1]["boss_reply"] == "可以，我问下这桌其他人能不能打6小时。"
    finally:
        restore_env(saved_env)


def test_candidate_message_with_changed_start_time_creates_organizer_followup() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "candidate_start_time_negotiation",
                    "text": "4点无烟0.5，173，5h",
                    "now": "2026-06-28T13:59:00+08:00",
                }
            )
            invite = next(item for item in result["outbox"] if item["customer_name"] == "Amy")

            reply = service.candidate_message(
                {
                    "outbox_id": invite["id"],
                    "text": "可以倒是可以，但是我最快要四点半",
                    "now": "2026-06-28T14:14:00+08:00",
                }
            )

            candidate = reply["candidate_message"]
            assert candidate["feedback_type"] == "candidate_negotiation"
            assert candidate["requested_start_time"] == "16:30"
            assert candidate["requested_start_time_label"] == "四点半"
            assert "四点半" in candidate["suggested_boss_reply"]
            assert reply["outbox_item"]["status"] == "待协商"
            followup = reply["organizer_followup"]
            assert followup["recipient_id"] == "zhang"
            assert followup["recipient_name"] == "张哥"
            assert followup["status"] == "待审批"
            assert "Amy" in followup["message_text"]
            assert "四点半" in followup["message_text"]
            assert "可以吗" in followup["message_text"]
            assert followup["direct_send_executed"] is False
            assert followup["approval_status"] == "待审批"
            assert followup["approval"]["target_type"] == "followup"
            assert followup["approval"]["status"] == "pending"
            live_game = reply["state"]["games"][0]
            assert live_game["followups"][0]["message_text"] == followup["message_text"]
    finally:
        restore_env(saved_env)


def test_candidate_negotiation_uses_llm_to_draft_organizer_followup() -> None:
    saved_env = without_llm_env()
    os.environ.update(
        {
            "MAHJONG_LLM_API_KEY": "test-key",
            "MAHJONG_LLM_PROVIDER": "openai",
            "MAHJONG_LLM_MODEL": "test-model",
            "MAHJONG_LLM_BASE_URL": "https://example.invalid/v1",
        }
    )
    module = load_boss_trial_module()
    captured = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            system_prompt = self.payload["messages"][0]["content"]
            if "协商消息起草助手" in system_prompt:
                content = {
                    "should_create_message": True,
                    "message_text": "张哥，Amy最快四点半到，你们四点半开可以吗？",
                    "risk_level": "low",
                    "reasoning_summary": "候选人改到店时间，需要发起人确认。",
                    "notes": [],
                }
            else:
                content = {
                    "semantic_type": "candidate_negotiation",
                    "proposed_action": "start_negotiation",
                    "confidence": 0.94,
                    "reply_text": "我先问下这桌其他人，看大家能不能对上。",
                    "risk_level": "low",
                    "reasoning_summary": "候选人提出时间变更。",
                    "extracted_facts": {"requested_start_time": "16:30"},
                    "notes": [],
                }
            return json.dumps(
                {
                    "choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                },
                ensure_ascii=False,
            ).encode("utf-8")

    original_urlopen = module.urllib.request.urlopen

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        captured.append(payload)
        return FakeResponse(payload)

    module.urllib.request.urlopen = fake_urlopen
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            created = service.manual_create_game(
                {
                    "organizer_id": "zhang",
                    "organizer_name": "张哥",
                    "game_type": "hangzhou_mahjong",
                    "level": "0.5",
                    "start_at": "2026-06-28T16:00:00+08:00",
                    "duration_hours": 5,
                    "current_player_count": 3,
                    "missing_count": 1,
                    "smoke": "no_smoke",
                    "now": "2026-06-28T14:00:00+08:00",
                }
            )
            outbox_id = store.create_outbox(
                game_id=created["game"]["id"],
                customer_id="amy",
                customer_name="Amy",
                message_text="Amy，16:00，0.5无烟，约5小时，打吗？",
                score=90,
                reasons=["测试候选人"],
                warnings=[],
            )

            reply = service.candidate_message(
                {
                    "outbox_id": outbox_id,
                    "text": "可以倒是可以，但是我最快要四点半",
                    "trace_id": "trace_candidate_negotiation_tool",
                    "now": "2026-06-28T14:14:00+08:00",
                }
            )

            organizer_payload = next(
                payload for payload in captured if "协商消息起草助手" in payload["messages"][0]["content"]
            )
            organizer_prompt = json.loads(organizer_payload["messages"][1]["content"])
            assert organizer_prompt["organizer"]["customer_name"] == "张哥"
            assert organizer_prompt["candidate"]["customer_name"] == "Amy"
            assert organizer_prompt["backend_classification"]["requested_start_time"] == "16:30"
            assert reply["organizer_followup"]["source"] == "llm"
            assert reply["organizer_followup"]["model"] == "test-model"
            assert reply["organizer_followup"]["message_text"] == "张哥，Amy最快四点半到，你们四点半开可以吗？"
            assert reply["organizer_followup"]["direct_send_executed"] is False
            assert reply["organizer_followup"]["approval_status"] == "待审批"
            assert reply["organizer_followup"]["approval"]["target_type"] == "followup"
            assert reply["organizer_followup"]["approval"]["status"] == "pending"
            assert [item["stage"] for item in reply["agent_actions"]] == [
                "candidate_feedback",
                "organizer_followup_draft",
            ]
            followup_action = reply["organizer_followup"]["agent_actions"][0]["validated_actions"][0]
            assert followup_action["tool_name"] == "send_message"
            assert followup_action["allowed"] is True
            assert followup_action["approval_required"] is True
            assert followup_action["idempotency_key"]
            assert any("followup" in note for note in followup_action["notes"])
    finally:
        module.urllib.request.urlopen = original_urlopen
        restore_env(saved_env)


def test_candidate_message_uses_llm_prompt_and_guards_bad_acceptance_phrase() -> None:
    saved_env = without_llm_env()
    os.environ.update(
        {
            "MAHJONG_LLM_API_KEY": "test-key",
            "MAHJONG_LLM_PROVIDER": "openai",
            "MAHJONG_LLM_MODEL": "test-model",
            "MAHJONG_LLM_BASE_URL": "https://example.invalid/v1",
        }
    )
    module = load_boss_trial_module()
    captured = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            content = {
                "semantic_type": "accepted",
                "proposed_action": "mark_candidate_confirmed",
                "confidence": 0.95,
                "reply_text": "冉姐，好，我给你留着。",
                "risk_level": "low",
                "reasoning_summary": "候选人明确说可以。",
                "extracted_facts": {},
                "notes": ["测试模型输出了不合适的话术"],
            }
            return json.dumps(
                {
                    "choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                },
                ensure_ascii=False,
            ).encode("utf-8")

    original_urlopen = module.urllib.request.urlopen

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        captured.append(payload)
        return FakeResponse(payload)

    module.urllib.request.urlopen = fake_urlopen
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            created = service.manual_create_game(
                {
                    "organizer_id": "zhang",
                    "organizer_name": "张哥",
                    "game_type": "hangzhou_mahjong",
                    "variant": "caiqiao",
                    "level": "0.5",
                    "start_at": "2026-06-28T14:00:00+08:00",
                    "duration_hours": 4,
                    "current_player_count": 1,
                    "missing_count": 3,
                    "smoke": "no_smoke",
                    "now": "2026-06-28T10:00:00+08:00",
                }
            )
            game_id = created["game"]["id"]
            outbox_id = store.create_outbox(
                game_id=game_id,
                customer_id="ran",
                customer_name="冉姐",
                message_text="冉姐，14:00，0.5无烟，打吗？",
                score=100,
                reasons=["测试候选人"],
                warnings=[],
            )

            reply = service.candidate_message(
                {
                    "outbox_id": outbox_id,
                    "text": "可以",
                    "trace_id": "trace_candidate_accept",
                    "now": "2026-06-28T10:03:00+08:00",
                }
            )

            assert captured
            assert "语义解析器和动作提案器" in captured[0]["messages"][0]["content"]
            assert "不要说“给你留着/留位/留座”" in captured[0]["messages"][0]["content"]
            prompt = json.loads(captured[0]["messages"][1]["content"])
            assert prompt["candidate"]["reply_text"] == "可以"
            assert prompt["game_state"]["confirmed_before"] == 0
            assert prompt["state_preview"]["if_confirmed"]["confirmed_after"] == 1
            assert prompt["state_preview"]["if_confirmed"]["progress_label_after"] == "272"
            assert prompt["state_preview"]["if_confirmed"]["fallback_reply"] == "好的，加你272了。"
            assert reply["candidate_message"]["reply_source"] == "llm"
            assert reply["candidate_message"]["model"] == "test-model"
            assert reply["candidate_message"]["semantic_type"] == "accepted"
            assert reply["candidate_message"]["proposed_action"] == "mark_candidate_confirmed"
            assert reply["candidate_message"]["suggested_boss_reply"] == "好的，加你272了。"
            assert "留" not in reply["candidate_message"]["suggested_boss_reply"]
            assert reply["agent_actions"][0]["stage"] == "candidate_feedback"
            candidate_action = reply["agent_actions"][0]["validated_actions"][0]
            assert candidate_action["tool_name"] == "record_candidate_feedback"
            assert candidate_action["allowed"] is True
            assert candidate_action["code"] == "allowed"
            assert candidate_action["idempotency_key"]
    finally:
        module.urllib.request.urlopen = original_urlopen
        restore_env(saved_env)


def test_candidate_message_question_stays_pending_and_suggests_boss_reply() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "candidate_question",
                    "text": "下午两点 0.5 无烟杭麻，打4小时，三缺一，帮我组一桌",
                    "now": "2026-06-28T10:00:00+08:00",
                }
            )
            invite = result["outbox"][0]

            reply = service.candidate_message(
                {
                    "outbox_id": invite["id"],
                    "text": "几点啊",
                    "now": "2026-06-28T10:03:00+08:00",
                }
            )

            assert reply["candidate_message"]["feedback_type"] == "candidate_question"
            assert "14:00" in reply["candidate_message"]["suggested_boss_reply"]
            assert reply["outbox_item"]["status"] == "待确认"
            assert reply["auto_success"] is None
            assert reply["state"]["games"]
    finally:
        restore_env(saved_env)


def test_llm_reply_payload_includes_few_shot_examples() -> None:
    saved_env = without_llm_env()
    os.environ.update(
        {
            "MAHJONG_LLM_API_KEY": "test-key",
            "MAHJONG_LLM_PROVIDER": "openai",
            "MAHJONG_LLM_MODEL": "test-model",
            "MAHJONG_LLM_BASE_URL": "https://example.invalid/v1",
        }
    )
    module = load_boss_trial_module()
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "reply_text": "可以，我先帮你问人。有合适的我再跟你确认。",
                                        "risk_level": "low",
                                        "notes": ["测试"],
                                    },
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                },
                ensure_ascii=False,
            ).encode("utf-8")

    original_urlopen = module.urllib.request.urlopen

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    module.urllib.request.urlopen = fake_urlopen
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            module.FEW_SHOT_EXAMPLES_PATH = Path(temp_dir) / "few_shot_examples.jsonl"
            module.FEW_SHOT_EXAMPLES_PATH.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "few_shot",
                        "id": "trial_good_reply_001",
                        "customer_message": "下午两点 0.5 无烟杭麻，打4小时，帮我组一桌",
                        "parsed": "人数未知，不能推断三缺一",
                        "reply_text": "可以，我先帮你看。你一个人吗？",
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "group_a",
                    "text": "下午两点 0.5 无烟杭麻，打4小时，一缺三，帮我组一桌",
                    "now": "2026-06-27T10:00:00+08:00",
                }
            )

            user_payload = json.loads(captured["payload"]["messages"][1]["content"])
            assert result["suggested_reply"]["source"] == "llm"
            assert user_payload["few_shot_examples"]
            assert user_payload["active_skills"]
            assert any("272财敲0.5" in item["customer_message"] for item in user_payload["few_shot_examples"])
            assert any("你一个人吗" in item["reply_text"] for item in user_payload["few_shot_examples"])
            assert any(item["id"] == "slot_party_size_confirmation" for item in user_payload["active_skills"])
            assert "参考 few_shot_examples" in captured["payload"]["messages"][0]["content"]
    finally:
        module.urllib.request.urlopen = original_urlopen
        restore_env(saved_env)


def test_boss_trial_can_archive_eval_cases_from_analysis() -> None:
    module = load_boss_trial_module()
    saved_env = without_llm_env()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            module.GOLDEN_DATASET_PATH = temp_path / "golden_dataset.jsonl"
            module.BADCASE_PATH = temp_path / "badcases.jsonl"
            module.FEW_SHOT_EXAMPLES_PATH = temp_path / "few_shot_examples.jsonl"

            store = module.TrialStore(temp_path / "trial.db")
            service = module.BossTrialService(store)
            analysis = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "group_a",
                    "text": "下午两点 0.5 无烟杭麻，打4小时，一缺三，帮我组一桌",
                    "now": "2026-06-27T10:00:00+08:00",
                }
            )

            badcase = service.record_eval_case(
                {
                    "case_type": "badcase",
                    "analysis": analysis,
                    "note": "测试 badcase 归档",
                    "expected": {"action": "queue_invites"},
                }
            )
            golden = service.record_eval_case(
                {
                    "case_type": "golden",
                    "analysis": analysis,
                    "note": "老板确认当前动作正确",
                }
            )
            few_shot = service.record_eval_case(
                {
                    "case_type": "few_shot",
                    "analysis": analysis,
                    "note": "老板认可这句回复",
                }
            )

            assert badcase["ok"] is True
            assert golden["ok"] is True
            assert few_shot["ok"] is True

            badcase_records = module.read_jsonl_records(module.BADCASE_PATH)
            golden_records = module.read_jsonl_records(module.GOLDEN_DATASET_PATH)
            few_shot_records = module.read_jsonl_records(module.FEW_SHOT_EXAMPLES_PATH)

            assert badcase_records[0]["kind"] == "badcase"
            assert badcase_records[0]["trace_id"] == analysis["trace_id"]
            assert badcase_records[0]["actual"]["suggested_reply"]["status"] == "待审批"
            assert golden_records[0]["kind"] == "golden"
            assert golden_records[0]["expected"]["action"] == analysis["decision"]["action"]
            assert few_shot_records[0]["kind"] == "few_shot"
            assert few_shot_records[0]["customer_message"] == analysis["source_text"]
            assert few_shot_records[0]["reply_text"]
            counts = service.eval_overview()["counts"]
            assert counts["golden"] == 1
            assert counts["badcase"] == 1
            assert counts["few_shot"] == 1
            assert "boss_trial_golden" in counts
    finally:
        restore_env(saved_env)


def test_llm_reply_guard_does_not_offer_sichuan_when_hangzhou_default_applies() -> None:
    saved_env = without_llm_env()
    os.environ.update(
        {
            "MAHJONG_LLM_API_KEY": "test-key",
            "MAHJONG_LLM_PROVIDER": "openai",
            "MAHJONG_LLM_MODEL": "test-model",
            "MAHJONG_LLM_BASE_URL": "https://example.invalid/v1",
        }
    )
    module = load_boss_trial_module()
    captured = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            system_prompt = self.payload["messages"][0]["content"]
            if "语义解析器" in system_prompt:
                content = {
                    "is_mahjong_related": True,
                    "intent": "find_players",
                    "confidence": 0.86,
                    "normalized_text": "下午 杭麻财敲 0.5 有人打麻将吗",
                    "reply_text": "可以，我先帮你看。",
                    "needs_human_review": False,
                    "facts": {"reason": "测试"},
                }
            else:
                content = {
                    "reply_text": "张哥，下午想打杭麻财敲还是川麻？大概几点能到，现在几个人？",
                    "risk_level": "low",
                    "notes": ["测试"],
                }
            return json.dumps(
                {
                    "choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                },
                ensure_ascii=False,
            ).encode("utf-8")

    original_urlopen = module.urllib.request.urlopen

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        captured.append(payload)
        return FakeResponse(payload)

    module.urllib.request.urlopen = fake_urlopen
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = module.TrialStore(Path(temp_dir) / "trial.db")
            service = module.BossTrialService(store)
            result = service.analyze(
                {
                    "sender_name": "张哥",
                    "sender_id": "zhang",
                    "conversation_id": "group_a",
                    "text": "下午帮我找人打麻将，0.5都行",
                    "now": "2026-06-27T15:00:00+08:00",
                }
            )

            reply = result["suggested_reply"]["text"]
            reply_prompt = next(payload["messages"][0]["content"] for payload in captured if "微信回复起草助手" in payload["messages"][0]["content"])
            assert result["suggested_reply"]["source"] == "llm"
            assert "还是川麻" not in reply
            assert "不要问“杭麻还是川麻”" in reply_prompt
    finally:
        module.urllib.request.urlopen = original_urlopen
        restore_env(saved_env)
