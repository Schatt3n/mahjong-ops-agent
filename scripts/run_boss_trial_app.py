from __future__ import annotations

import html
import hashlib
import json
import os
import pathlib
import re
import sqlite3
import sys
import threading
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mahjong_agent import (  # noqa: E402
    AgentResponder,
    CandidateFeedbackActionService,
    CandidateReplyDraftService,
    CandidateReplyFactService,
    CandidateSemanticProposalAdapter,
    CandidateSemanticResolverService,
    CandidateRecommendation,
    CandidateActionProposalValidator,
    ChannelType,
    ControlledRuntimeConfig,
    CustomerProfile,
    GameRequest,
    GameStatus,
    LLMConfig,
    LLMBudgetManager,
    Message,
    MessageComposer,
    OpenAICompatibleLLMResolver,
    OrganizerFollowupDraftService,
    PlayPreference,
    RedisCache,
    RedisCacheError,
    TrialApprovalDecisionAdapter,
    TrialCandidateMessageAdapter,
    TrialControlledEntryAdapter,
    TrialControlledPersistenceAdapter,
    TrialControlledRequestBuilder,
    TrialControlledResponseAdapter,
    TrialManualGameAdapter,
    TrialOutboxDeliveryAdapter,
    TrialOrganizerFollowupAdapter,
    TrialReplyDraftAdapter,
    TrialReplyDraftCallbacks,
    TrialReplyDraftInput,
    TrialReplyRulePolicy,
    TrialReplyRulePolicyCallbacks,
    TrialReplyRulePolicyInput,
    TrialCreateGameStateInput,
    TrialGameStateCreationAdapter,
    TrialGameStateCreationCallbacks,
    RUNTIME_POLICY_VERSION,
    DEFAULT_RUNTIME_POLICY,
    STATE_WRITE_STAGES,
    TOOL_REGISTRY,
    TOOL_REGISTRY_VERSION,
    TOOL_STAGE_POLICY,
    STATE_MACHINE_VERSION,
    approval_status_label,
    default_runtime_policy,
    env_bool,
    require_state_transition,
    state_transition_verdict,
    tool_spec_for_stage,
    tool_specs_for_stage,
    TRACE_EVENT_SCHEMA_VERSION,
    TrialTraceLogger,
    ensure_trace_events_table,
    format_io_log_line,
    trace_payload_from_content,
    trusted_action_proposer,
    use_controlled_trial_workflow,
    TrialShortMemoryTextMerger,
    TrialToolGateway,
    TrialToolOrchestrationCallbacks,
    TrialToolOrchestrationInput,
    TrialToolOrchestrationService,
    TrialToolActionProposalFactory,
    TrialToolActionValidator,
    TrialToolCallNormalizer,
    TrialToolPlanPromptBuilder,
    TrialToolPlanPromptInput,
    TrialToolRequestFactory,
    TrialWorkflowFollowupContextBuilder,
    build_controlled_runtime,
)
from mahjong_agent.budget import usage_from_response  # noqa: E402
from mahjong_agent.normalization import normalize_mahjong_text  # noqa: E402
from mahjong_agent.skills import DEFAULT_SKILL_LIBRARY_PATH, select_relevant_skills  # noqa: E402


TZ = ZoneInfo("Asia/Shanghai")
DB_PATH = ROOT / "data" / "boss_trial.sqlite3"
LOG_PATH = ROOT / "logs" / "boss_trial_io.log"
EVAL_DIR = ROOT / "eval"
GOLDEN_DATASET_PATH = EVAL_DIR / "golden" / "scenario_golden.jsonl"
BOSS_TRIAL_GOLDEN_PATH = EVAL_DIR / "golden" / "boss_trial_golden.jsonl"
BADCASE_PATH = EVAL_DIR / "badcases" / "badcases.jsonl"
FEW_SHOT_EXAMPLES_PATH = EVAL_DIR / "few_shot_examples.jsonl"
SKILL_LIBRARY_PATH = DEFAULT_SKILL_LIBRARY_PATH
DEFAULT_REDIS_URL = "redis://127.0.0.1:6379/0"
CACHE_PREFIX = os.environ.get("MAHJONG_CACHE_PREFIX", "mahjong:trial").strip(":")
STATE_CACHE_TTL_SECONDS = 5 * 60
GAME_CACHE_TTL_SECONDS = 24 * 60 * 60
SHORT_MEMORY_TTL_SECONDS = 2 * 60 * 60
SHORT_MEMORY_MERGE_WINDOW_SECONDS = 10 * 60
TRIAL_PROFILE_PARTY_SIZE_CONFIDENCE_THRESHOLD = 0.65
GAME_EXPIRE_GRACE_MINUTES = int(os.environ.get("MAHJONG_GAME_EXPIRE_GRACE_MINUTES", "90"))
TIME_RESOLUTION_CONFIDENCE_THRESHOLD = 0.75
LOCAL_AFTERNOON_CONTEXT_HOUR = 13
LOCAL_TIME_MAX_LOOKAHEAD_HOURS = 8
EVAL_LOCK = threading.Lock()
TRIAL_TRACE_LOGGER = TrialTraceLogger(db_path=DB_PATH, log_path=LOG_PATH)
GENDER_LABELS = {"male": "男", "female": "女", "unknown": "未知"}
GENDER_NOTE_PREFIX = "候选人组合偏好："


GAME_TYPE_LABELS = {
    "mahjong": "麻将",
    "hangzhou_mahjong": "杭麻",
    "sichuan_mahjong": "川麻",
    "hongzhong_mahjong": "红中",
    "zhuoji_mahjong": "捉鸡",
    "hunan_mahjong": "湖南麻将",
    "chongqing_mahjong": "重庆麻将",
}

VARIANT_LABELS = {
    "caiqiao": "财敲",
    "yaoji": "幺鸡",
    "suji": "素鸡",
    "yaoji_47": "幺鸡47",
    "shayu": "鲨鱼",
}

CRITICAL_FIELDS = {"known_players", "start_time", "stake", "smoke", "duration"}
FINAL_GAME_STATUSES = {"已成局", "已取消"}
ACTIVE_GAME_STATUSES = {"待补充", "待组局", "邀约中", "已满"}
DECLINED_OUTBOX_STATUSES = {"拒绝", "别再打扰", "下次再问"}
CONTROLLED_AGENT_PROTOCOL_VERSION = "controlled_agent.v1"
STATE_TRANSITION_EVENT_SCHEMA_VERSION = "state_transition_events.v1"
BOSS_REPLY_FEW_SHOTS = [
    {
        "name": "明确组局，信息基本够",
        "source": "真实聊天脱敏改写：客户给出财敲、人数、档位、烟况和开局时间。",
        "customer_message": "可以帮忙摇下人吗，272财敲0.5，无烟，7点开4小时",
        "parsed": "财敲，0.5，二缺二，19:00，4小时，无烟",
        "reply_text": "可以，我先按财敲0.5、7点开、二缺二、无烟帮你问人。有合适的我先给你确认。",
    },
    {
        "name": "组一桌但人数未知",
        "source": "老板反馈：组一桌不等于三缺一，人数要确认。",
        "customer_message": "下午两点 0.5 无烟杭麻，帮我组一桌",
        "parsed": "时间、玩法、档位、烟况明确，但当前人数未知",
        "conditions": "仅适用时间未过期；时间已过先确认，不能说“先帮你看”。",
        "reply_text": "可以，我先帮你看。你一个人吗？",
    },
    {
        "name": "缺关键字段，先追问",
        "source": "真实聊天脱敏改写：群里有人只表达川麻意向。",
        "customer_message": "川麻132 晚上有人吗",
        "parsed": "川麻，1-32，晚上，人数和烟况不明确",
        "reply_text": "可以，我帮你看看。你大概几点能到、现在几个人、有烟无烟有要求吗？",
    },
    {
        "name": "客户可接受备选玩法",
        "source": "真实聊天脱敏改写：客户先要川麻，后续补充杭麻也可以。",
        "customer_message": "川麻没人的话，杭麻0.5无烟的也行",
        "parsed": "优先川麻，备选杭麻0.5无烟",
        "reply_text": "收到，我先按川麻帮你看；如果川麻不好凑，我再帮你看杭麻0.5无烟的局。",
    },
    {
        "name": "已有相近局，先协商时间",
        "source": "真实聊天脱敏改写：客户能接受相近开局时间，需要跟另一桌确认。",
        "customer_message": "七点半0.5帮我问下吧",
        "parsed": "0.5，19:30，人数未完全明确",
        "reply_text": "好，我先帮你问下7点半左右0.5的局。如果那边时间能对上，我再跟你确认。",
    },
    {
        "name": "群内弱意图，问清信息",
        "source": "真实聊天脱敏改写：群里有人问现在有没有三缺一。",
        "customer_message": "有没有三缺一的局啊现在",
        "parsed": "想找局，时间为现在，玩法、档位、烟况不明确",
        "reply_text": "我帮你看下。你想打杭麻还是川麻，0.5还是1，有烟无烟有要求吗？",
    },
]


SEED_CUSTOMERS = [
    {
        "id": "zhang",
        "display_name": "张哥",
        "contact": "微信备注：张哥",
        "preferred_games": ["杭麻", "川麻"],
        "preferred_levels": ["0.5", "1", "1-32", "2", "2-64"],
        "usual_start_hours": [14, 15, 19, 20],
        "smoke_preference": "any",
        "response_speed": "fast",
        "response_rate": 0.86,
        "notes": "男性；常一个人来；经常打杭麻和川麻；杭麻/财敲常打0.5或1块；川麻常打1块或2块五番封顶，可按1-32/2-64理解；本人抽烟，但也可以打无烟局；财敲响应快。",
        "usual_party_size": 1,
        "usual_party_size_confidence": 0.9,
    },
    {
        "id": "wangjie",
        "display_name": "王姐",
        "contact": "微信备注：王姐无烟",
        "preferred_games": ["杭麻", "红中"],
        "preferred_levels": ["0.5", "1"],
        "usual_start_hours": [13, 14, 15, 18],
        "smoke_preference": "no_smoke",
        "response_speed": "fast",
        "response_rate": 0.78,
        "notes": "无烟优先，下午常来。",
        "usual_party_size": 1,
        "usual_party_size_confidence": 0.8,
    },
    {
        "id": "chen",
        "display_name": "陈姐",
        "contact": "微信备注：陈姐",
        "preferred_games": ["杭麻", "财敲"],
        "preferred_levels": ["0.5"],
        "usual_start_hours": [14, 15, 16, 19],
        "smoke_preference": "no_smoke",
        "response_speed": "medium",
        "response_rate": 0.66,
        "notes": "熟人局更愿意来。",
    },
    {
        "id": "li",
        "display_name": "李总",
        "contact": "微信备注：李总",
        "preferred_games": ["川麻", "幺鸡"],
        "preferred_levels": ["1-32", "2-16"],
        "usual_start_hours": [19, 20, 21],
        "smoke_preference": "smoke_ok",
        "response_speed": "medium",
        "response_rate": 0.58,
        "notes": "最近别问太频繁。",
    },
    {
        "id": "zhao",
        "display_name": "赵哥",
        "contact": "微信备注：赵哥",
        "preferred_games": ["川麻", "换三张"],
        "preferred_levels": ["1-32"],
        "usual_start_hours": [18, 19, 20],
        "smoke_preference": "any",
        "response_speed": "slow",
        "response_rate": 0.42,
        "notes": "有空会回，但慢。",
    },
    {
        "id": "amy",
        "display_name": "Amy",
        "contact": "微信备注：Amy",
        "preferred_games": ["杭麻", "财敲"],
        "preferred_levels": ["0.5"],
        "usual_start_hours": [17, 18, 19],
        "smoke_preference": "no_smoke",
        "response_speed": "fast",
        "response_rate": 0.82,
        "notes": "无烟，晚饭后响应好。",
    },
    {
        "id": "ben",
        "display_name": "Ben",
        "contact": "微信备注：Ben",
        "preferred_games": ["红中", "湖南麻将"],
        "preferred_levels": ["1", "2"],
        "usual_start_hours": [20, 21, 22],
        "smoke_preference": "smoke_ok",
        "response_speed": "medium",
        "response_rate": 0.55,
        "notes": "能接受有烟。",
    },
    {
        "id": "sun",
        "display_name": "孙姐",
        "contact": "微信备注：孙姐",
        "preferred_games": ["杭麻"],
        "preferred_levels": ["0.5"],
        "usual_start_hours": [13, 14, 15],
        "smoke_preference": "no_smoke",
        "response_speed": "medium",
        "response_rate": 0.61,
        "notes": "下午局优先。",
    },
    {
        "id": "xu",
        "display_name": "徐哥",
        "contact": "微信备注：徐哥",
        "preferred_games": ["捉鸡", "川麻"],
        "preferred_levels": ["1", "2-16"],
        "usual_start_hours": [19, 20],
        "smoke_preference": "any",
        "response_speed": "fast",
        "response_rate": 0.74,
        "notes": "川麻也可。",
    },
    {
        "id": "zhou",
        "display_name": "周姐",
        "contact": "微信备注：周姐",
        "preferred_games": ["杭麻", "财敲"],
        "preferred_levels": ["0.5", "1"],
        "usual_start_hours": [18, 19],
        "smoke_preference": "no_smoke",
        "response_speed": "fast",
        "response_rate": 0.8,
        "notes": "不喜欢烟味。",
    },
    {
        "id": "huang",
        "display_name": "黄哥",
        "contact": "微信备注：黄哥",
        "preferred_games": ["川麻", "定缺"],
        "preferred_levels": ["2-16"],
        "usual_start_hours": [14, 15, 20],
        "smoke_preference": "smoke_ok",
        "response_speed": "medium",
        "response_rate": 0.52,
        "notes": "川麻 216 可问。",
    },
    {
        "id": "lin",
        "display_name": "林姐",
        "contact": "微信备注：林姐",
        "preferred_games": ["红中"],
        "preferred_levels": ["368", "0.5"],
        "usual_start_hours": [13, 14, 18],
        "smoke_preference": "no_smoke",
        "response_speed": "slow",
        "response_rate": 0.48,
        "notes": "红中老客户。",
    },
    {
        "id": "gao",
        "display_name": "高哥",
        "contact": "微信备注：高哥",
        "preferred_games": ["杭麻", "财敲"],
        "preferred_levels": ["1"],
        "usual_start_hours": [20, 21],
        "smoke_preference": "any",
        "response_speed": "medium",
        "response_rate": 0.57,
        "notes": "晚上 1 档多。",
    },
    {
        "id": "liu",
        "display_name": "刘姐",
        "contact": "微信备注：刘姐",
        "preferred_games": ["杭麻", "财敲"],
        "preferred_levels": ["0.5"],
        "usual_start_hours": [14, 15, 16],
        "smoke_preference": "no_smoke",
        "response_speed": "fast",
        "response_rate": 0.72,
        "notes": "下午可重点问。",
    },
    {
        "id": "ma",
        "display_name": "马哥",
        "contact": "微信备注：马哥",
        "preferred_games": ["幺鸡", "素鸡", "川麻"],
        "preferred_levels": ["1-32", "2-16"],
        "usual_start_hours": [18, 19, 20],
        "smoke_preference": "smoke_ok",
        "response_speed": "medium",
        "response_rate": 0.6,
        "notes": "幺鸡可问。",
    },
    {
        "id": "qian",
        "display_name": "钱姐",
        "contact": "微信备注：钱姐",
        "preferred_games": ["杭麻"],
        "preferred_levels": ["0.5"],
        "usual_start_hours": [19, 20],
        "smoke_preference": "no_smoke",
        "response_speed": "medium",
        "response_rate": 0.63,
        "notes": "不喜欢太吵。",
    },
    {
        "id": "feng",
        "display_name": "冯哥",
        "contact": "微信备注：冯哥",
        "preferred_games": ["湖南麻将", "红中"],
        "preferred_levels": ["1", "2"],
        "usual_start_hours": [20, 21],
        "smoke_preference": "any",
        "response_speed": "fast",
        "response_rate": 0.68,
        "notes": "晚上活跃。",
    },
    {
        "id": "tang",
        "display_name": "唐姐",
        "contact": "微信备注：唐姐",
        "preferred_games": ["杭麻", "财敲"],
        "preferred_levels": ["0.5", "1"],
        "usual_start_hours": [13, 14, 19],
        "smoke_preference": "no_smoke",
        "response_speed": "medium",
        "response_rate": 0.59,
        "notes": "熟人局优先。",
    },
    {
        "id": "wu",
        "display_name": "吴哥",
        "contact": "微信备注：吴哥",
        "preferred_games": ["川麻", "换三张"],
        "preferred_levels": ["1-32"],
        "usual_start_hours": [19, 20, 21],
        "smoke_preference": "any",
        "response_speed": "slow",
        "response_rate": 0.41,
        "notes": "不急时可问。",
    },
    {
        "id": "xie",
        "display_name": "谢姐",
        "contact": "微信备注：谢姐",
        "preferred_games": ["杭麻", "红中"],
        "preferred_levels": ["0.5"],
        "usual_start_hours": [14, 15, 18],
        "smoke_preference": "no_smoke",
        "response_speed": "fast",
        "response_rate": 0.77,
        "notes": "无烟、0.5 响应好。",
    },
    {
        "id": "mei",
        "display_name": "梅姐",
        "contact": "微信备注：梅姐",
        "preferred_games": ["杭麻", "财敲"],
        "preferred_levels": ["0.5"],
        "usual_start_hours": [12, 13, 14],
        "smoke_preference": "no_smoke",
        "response_speed": "fast",
        "response_rate": 0.81,
        "notes": "午后局响应好，偏无烟。",
    },
    {
        "id": "guo",
        "display_name": "郭哥",
        "contact": "微信备注：郭哥",
        "preferred_games": ["川麻", "换三张"],
        "preferred_levels": ["2-16", "1-32"],
        "usual_start_hours": [15, 16, 20],
        "smoke_preference": "any",
        "response_speed": "medium",
        "response_rate": 0.62,
        "notes": "川麻可约，下午偶尔来。",
    },
    {
        "id": "du",
        "display_name": "杜姐",
        "contact": "微信备注：杜姐",
        "preferred_games": ["红中", "鲨鱼"],
        "preferred_levels": ["368", "568"],
        "usual_start_hours": [18, 19, 20],
        "smoke_preference": "no_smoke",
        "response_speed": "medium",
        "response_rate": 0.64,
        "notes": "红中鲨鱼可问，偏无烟。",
    },
    {
        "id": "jiang",
        "display_name": "蒋哥",
        "contact": "微信备注：蒋哥",
        "preferred_games": ["杭麻"],
        "preferred_levels": ["1"],
        "usual_start_hours": [19, 20, 21],
        "smoke_preference": "smoke_ok",
        "response_speed": "fast",
        "response_rate": 0.73,
        "notes": "晚上 1 档局响应快。",
    },
    {
        "id": "pan",
        "display_name": "潘姐",
        "contact": "微信备注：潘姐",
        "preferred_games": ["杭麻", "财敲"],
        "preferred_levels": ["0.5", "1"],
        "usual_start_hours": [14, 15, 18],
        "smoke_preference": "no_smoke",
        "response_speed": "fast",
        "response_rate": 0.84,
        "notes": "熟人无烟局优先。",
        "usual_party_size": 1,
        "usual_party_size_confidence": 0.82,
    },
    {
        "id": "lu",
        "display_name": "陆哥",
        "contact": "微信备注：陆哥",
        "preferred_games": ["川麻", "定缺"],
        "preferred_levels": ["1-32"],
        "usual_start_hours": [20, 21, 22],
        "smoke_preference": "smoke_ok",
        "response_speed": "slow",
        "response_rate": 0.39,
        "notes": "晚上可问，回复偏慢。",
    },
    {
        "id": "han",
        "display_name": "韩姐",
        "contact": "微信备注：韩姐",
        "preferred_games": ["杭麻", "红中"],
        "preferred_levels": ["0.5"],
        "usual_start_hours": [13, 14, 19],
        "smoke_preference": "no_smoke",
        "response_speed": "medium",
        "response_rate": 0.67,
        "notes": "0.5 无烟常客。",
    },
    {
        "id": "yu",
        "display_name": "余哥",
        "contact": "微信备注：余哥",
        "preferred_games": ["捉鸡", "川麻"],
        "preferred_levels": ["1", "2"],
        "usual_start_hours": [18, 19],
        "smoke_preference": "any",
        "response_speed": "medium",
        "response_rate": 0.56,
        "notes": "玩法接受度高。",
    },
    {
        "id": "bai",
        "display_name": "白姐",
        "contact": "微信备注：白姐",
        "preferred_games": ["杭麻", "财敲"],
        "preferred_levels": ["0.5"],
        "usual_start_hours": [12, 13, 14, 15],
        "smoke_preference": "no_smoke",
        "response_speed": "fast",
        "response_rate": 0.79,
        "notes": "下午可优先问。",
    },
    {
        "id": "shi",
        "display_name": "石哥",
        "contact": "微信备注：石哥",
        "preferred_games": ["湖南麻将", "红中"],
        "preferred_levels": ["1", "2"],
        "usual_start_hours": [20, 21, 22],
        "smoke_preference": "any",
        "response_speed": "medium",
        "response_rate": 0.53,
        "notes": "晚场可尝试。",
    },
    {
        "id": "song",
        "display_name": "宋姐",
        "contact": "微信备注：宋姐",
        "preferred_games": ["杭麻"],
        "preferred_levels": ["0.5", "1"],
        "usual_start_hours": [15, 16, 19],
        "smoke_preference": "no_smoke",
        "response_speed": "medium",
        "response_rate": 0.65,
        "notes": "不喜欢烟，下午晚些可问。",
    },
    {
        "id": "luo",
        "display_name": "罗哥",
        "contact": "微信备注：罗哥",
        "preferred_games": ["幺鸡", "幺鸡47", "川麻"],
        "preferred_levels": ["1-32", "2-16"],
        "usual_start_hours": [18, 19, 20],
        "smoke_preference": "smoke_ok",
        "response_speed": "fast",
        "response_rate": 0.7,
        "notes": "幺鸡47 可优先问。",
    },
    {
        "id": "deng",
        "display_name": "邓姐",
        "contact": "微信备注：邓姐",
        "preferred_games": ["红中"],
        "preferred_levels": ["368"],
        "usual_start_hours": [14, 15, 18],
        "smoke_preference": "no_smoke",
        "response_speed": "slow",
        "response_rate": 0.45,
        "notes": "红中 368 可问，别催。",
    },
    {
        "id": "he",
        "display_name": "何哥",
        "contact": "微信备注：何哥",
        "preferred_games": ["杭麻", "财敲"],
        "preferred_levels": ["1"],
        "usual_start_hours": [19, 20],
        "smoke_preference": "any",
        "response_speed": "medium",
        "response_rate": 0.58,
        "notes": "1 档财敲可问。",
    },
    {
        "id": "cai",
        "display_name": "蔡姐",
        "contact": "微信备注：蔡姐",
        "preferred_games": ["杭麻", "财敲"],
        "preferred_levels": ["0.5"],
        "usual_start_hours": [13, 14, 18],
        "smoke_preference": "no_smoke",
        "response_speed": "fast",
        "response_rate": 0.76,
        "notes": "无烟，熟人局响应更好。",
    },
    {
        "id": "yao",
        "display_name": "姚哥",
        "contact": "微信备注：姚哥",
        "preferred_games": ["川麻", "换三张", "定缺"],
        "preferred_levels": ["2-16"],
        "usual_start_hours": [20, 21],
        "smoke_preference": "smoke_ok",
        "response_speed": "medium",
        "response_rate": 0.51,
        "notes": "川麻 216 老玩家。",
    },
    {
        "id": "kang",
        "display_name": "康姐",
        "contact": "微信备注：康姐",
        "preferred_games": ["杭麻"],
        "preferred_levels": ["0.5", "1"],
        "usual_start_hours": [16, 17, 18],
        "smoke_preference": "no_smoke",
        "response_speed": "medium",
        "response_rate": 0.6,
        "notes": "傍晚前后可问。",
    },
    {
        "id": "qi",
        "display_name": "齐哥",
        "contact": "微信备注：齐哥",
        "preferred_games": ["红中", "湖南麻将"],
        "preferred_levels": ["1"],
        "usual_start_hours": [19, 20, 21],
        "smoke_preference": "any",
        "response_speed": "fast",
        "response_rate": 0.69,
        "notes": "红中、湖南都可以。",
    },
    {
        "id": "ran",
        "display_name": "冉姐",
        "contact": "微信备注：冉姐",
        "preferred_games": ["杭麻", "财敲"],
        "preferred_levels": ["0.5"],
        "usual_start_hours": [12, 13, 14],
        "smoke_preference": "no_smoke",
        "response_speed": "fast",
        "response_rate": 0.83,
        "notes": "中午后响应快。",
    },
    {
        "id": "pei",
        "display_name": "裴哥",
        "contact": "微信备注：裴哥",
        "preferred_games": ["川麻", "幺鸡"],
        "preferred_levels": ["1-32"],
        "usual_start_hours": [18, 19, 20],
        "smoke_preference": "any",
        "response_speed": "medium",
        "response_rate": 0.55,
        "notes": "幺鸡普通局可问。",
    },
    {
        "id": "niu",
        "display_name": "牛姐",
        "contact": "微信备注：牛姐",
        "preferred_games": ["杭麻"],
        "preferred_levels": ["0.5"],
        "usual_start_hours": [15, 16],
        "smoke_preference": "no_smoke",
        "response_speed": "slow",
        "response_rate": 0.44,
        "notes": "下午偶尔来，不要频繁催。",
    },
    {
        "id": "tao",
        "display_name": "陶哥",
        "contact": "微信备注：陶哥",
        "preferred_games": ["捉鸡", "川麻"],
        "preferred_levels": ["2", "2-16"],
        "usual_start_hours": [20, 21],
        "smoke_preference": "smoke_ok",
        "response_speed": "medium",
        "response_rate": 0.57,
        "notes": "大一点的局可问。",
    },
    {
        "id": "xiong",
        "display_name": "熊姐",
        "contact": "微信备注：熊姐",
        "preferred_games": ["红中", "鲨鱼"],
        "preferred_levels": ["368", "568"],
        "usual_start_hours": [18, 19],
        "smoke_preference": "no_smoke",
        "response_speed": "medium",
        "response_rate": 0.62,
        "notes": "红中熟人局优先。",
    },
    {
        "id": "long",
        "display_name": "龙哥",
        "contact": "微信备注：龙哥",
        "preferred_games": ["川麻", "定缺"],
        "preferred_levels": ["1-32", "2-16"],
        "usual_start_hours": [19, 20, 21],
        "smoke_preference": "any",
        "response_speed": "fast",
        "response_rate": 0.71,
        "notes": "川麻缺人可优先问。",
    },
    {
        "id": "yan",
        "display_name": "严姐",
        "contact": "微信备注：严姐",
        "preferred_games": ["杭麻", "财敲"],
        "preferred_levels": ["1"],
        "usual_start_hours": [18, 19],
        "smoke_preference": "no_smoke",
        "response_speed": "medium",
        "response_rate": 0.61,
        "notes": "1 档无烟财敲可问。",
    },
    {
        "id": "qiu",
        "display_name": "邱哥",
        "contact": "微信备注：邱哥",
        "preferred_games": ["湖南麻将", "捉鸡"],
        "preferred_levels": ["1", "2"],
        "usual_start_hours": [20, 21, 22],
        "smoke_preference": "smoke_ok",
        "response_speed": "slow",
        "response_rate": 0.4,
        "notes": "晚场备选。",
    },
    {
        "id": "an",
        "display_name": "安姐",
        "contact": "微信备注：安姐",
        "preferred_games": ["杭麻"],
        "preferred_levels": ["0.5"],
        "usual_start_hours": [13, 14, 15, 19],
        "smoke_preference": "no_smoke",
        "response_speed": "fast",
        "response_rate": 0.75,
        "notes": "0.5 无烟稳定。",
    },
    {
        "id": "mo",
        "display_name": "莫哥",
        "contact": "微信备注：莫哥",
        "preferred_games": ["川麻", "换三张"],
        "preferred_levels": ["1-32"],
        "usual_start_hours": [18, 19, 20],
        "smoke_preference": "any",
        "response_speed": "medium",
        "response_rate": 0.59,
        "notes": "川麻换三张可问。",
    },
    {
        "id": "shao",
        "display_name": "邵姐",
        "contact": "微信备注：邵姐",
        "preferred_games": ["杭麻", "财敲"],
        "preferred_levels": ["0.5", "1"],
        "usual_start_hours": [14, 15, 16],
        "smoke_preference": "no_smoke",
        "response_speed": "medium",
        "response_rate": 0.64,
        "notes": "下午财敲可问。",
    },
    {
        "id": "min",
        "display_name": "闵哥",
        "contact": "微信备注：闵哥",
        "preferred_games": ["红中", "川麻"],
        "preferred_levels": ["1", "2-16"],
        "usual_start_hours": [19, 20, 21],
        "smoke_preference": "any",
        "response_speed": "fast",
        "response_rate": 0.72,
        "notes": "晚上红中、川麻都能问。",
    },
]


def now_tz() -> datetime:
    return datetime.now(TZ)


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).astimezone(TZ)
    except ValueError:
        return None


def dt_text(value: str | None, fallback: str = "-") -> str:
    dt = parse_dt(value)
    if not dt:
        return fallback
    return dt.strftime("%m-%d %H:%M")


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def json_loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def normalize_gender(value: Any) -> str:
    text = str(value or "").strip().lower()
    mapping = {
        "male": "male",
        "m": "male",
        "man": "male",
        "男": "male",
        "男性": "male",
        "男生": "male",
        "男士": "male",
        "female": "female",
        "f": "female",
        "woman": "female",
        "女": "female",
        "女性": "female",
        "女生": "female",
        "女士": "female",
        "unknown": "unknown",
        "未知": "unknown",
        "不确定": "unknown",
        "": "unknown",
    }
    return mapping.get(text, "unknown")


def infer_gender_from_customer_text(display_name: str, notes: str = "") -> str:
    name = display_name.strip()
    normalized_name = name.lower()
    note_text = notes.strip()
    if re.search(r"(^|[；;，,\s])男(性|生|士)?([；;，,\s]|$)", note_text):
        return "male"
    if re.search(r"(^|[；;，,\s])女(性|生|士)?([；;，,\s]|$)", note_text):
        return "female"
    if name.endswith("哥"):
        return "male"
    if name.endswith("姐"):
        return "female"
    if normalized_name in {"ben"}:
        return "male"
    if normalized_name in {"amy"}:
        return "female"
    return "unknown"


def read_jsonl_records(path: pathlib.Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    if limit is None:
        return records
    return records[-limit:]


def append_jsonl_record(path: pathlib.Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def count_jsonl_records(path: pathlib.Path) -> int:
    return len(read_jsonl_records(path))


def make_trace_id() -> str:
    stamp = now_tz().strftime("%Y%m%d%H%M%S")
    return f"trace_{stamp}_{uuid.uuid4().hex[:8]}"


def _trial_trace_logger() -> TrialTraceLogger:
    global TRIAL_TRACE_LOGGER
    if TRIAL_TRACE_LOGGER.db_path != DB_PATH or TRIAL_TRACE_LOGGER.log_path != LOG_PATH:
        TRIAL_TRACE_LOGGER = TrialTraceLogger(db_path=DB_PATH, log_path=LOG_PATH)
    return TRIAL_TRACE_LOGGER


def write_trace_event(trace_id: str, level: str, content: str) -> None:
    _trial_trace_logger().write_trace_event(trace_id, level, content)


def write_io_log(trace_id: str, level: str, content: str) -> None:
    _trial_trace_logger().write_io_log(trace_id, level, content)


def write_llm_audit_log(trace_id: str, event: str, payload: dict[str, Any]) -> None:
    _trial_trace_logger().write_llm_audit_log(trace_id, event, payload)


def write_tool_audit_log(trace_id: str, event: str, payload: dict[str, Any]) -> None:
    _trial_trace_logger().write_tool_audit_log(trace_id, event, payload)


def log_input_content(path: str, body: dict[str, Any]) -> str:
    if path == "/api/analyze":
        return json_dumps(
            {
                "direction": "input",
                "path": path,
                "conversation_id": body.get("conversation_id") or body.get("conversationId"),
                "sender_id": body.get("sender_id"),
                "sender_name": body.get("sender_name"),
                "text": truncate_text(str(body.get("text") or ""), 240),
            }
        )
    if path == "/api/feedback":
        return json_dumps(
            {
                "direction": "input",
                "path": path,
                "game_id": body.get("game_id"),
                "outbox_id": body.get("outbox_id"),
                "customer_id": body.get("customer_id"),
                "feedback_type": body.get("feedback_type"),
            }
        )
    if path == "/api/approval-decision":
        return json_dumps(
            {
                "direction": "input",
                "path": path,
                "approval_id": body.get("approval_id"),
                "target_type": body.get("target_type"),
                "target_id": body.get("target_id"),
                "decision": body.get("decision") or body.get("status"),
            }
        )
    if path == "/api/send-outbox":
        return json_dumps(
            {
                "direction": "input",
                "path": path,
                "outbox_id": body.get("outbox_id"),
                "channel": body.get("channel") or "manual",
            }
        )
    if path == "/api/runtime-policy":
        return json_dumps(
            {
                "direction": "input",
                "path": path,
                "controlled_agent_mode": body.get("controlled_agent_mode"),
                "read_only_mode": body.get("read_only_mode"),
                "state_writes_enabled": body.get("state_writes_enabled"),
                "delivery_enabled": body.get("delivery_enabled"),
                "approval_enabled": body.get("approval_enabled"),
                "eval_writes_enabled": body.get("eval_writes_enabled"),
                "llm_required_for_side_effect_tools": body.get("llm_required_for_side_effect_tools"),
                "llm_required_for_state_writes": body.get("llm_required_for_state_writes"),
                "reason": truncate_text(str(body.get("reason") or ""), 160),
            }
        )
    if path == "/api/candidate-message":
        return json_dumps(
            {
                "direction": "input",
                "path": path,
                "game_id": body.get("game_id"),
                "outbox_id": body.get("outbox_id"),
                "source_trace_id": body.get("source_trace_id"),
                "sender_id": body.get("sender_id"),
                "text": truncate_text(str(body.get("text") or ""), 240),
            }
        )
    if path == "/api/clear-board":
        return json_dumps(
            {
                "direction": "input",
                "path": path,
                "reason": truncate_text(str(body.get("reason") or ""), 160),
            }
        )
    if path == "/api/clear-short-memory":
        return json_dumps(
            {
                "direction": "input",
                "path": path,
                "conversation_id": body.get("conversation_id") or body.get("conversationId"),
                "sender_id": body.get("sender_id") or body.get("senderId"),
                "reason": truncate_text(str(body.get("reason") or ""), 160),
            }
        )
    if path == "/api/manual-create-game":
        return json_dumps(
            {
                "direction": "input",
                "path": path,
                "organizer_id": body.get("organizer_id"),
                "organizer_name": body.get("organizer_name"),
                "game_type": body.get("game_type"),
                "level": body.get("level"),
                "start_time": body.get("start_time"),
                "current_player_count": body.get("current_player_count"),
                "missing_count": body.get("missing_count"),
                "smoke": body.get("smoke"),
                "status": body.get("status"),
            }
        )
    if path == "/api/customers":
        return json_dumps(
            {
                "direction": "input",
                "path": path,
                "customer_id": body.get("id"),
                "display_name": body.get("display_name"),
                "gender": body.get("gender"),
                "preferred_games": body.get("preferred_games"),
                "preferred_levels": body.get("preferred_levels"),
            }
        )
    if path == "/api/eval-cases":
        analysis = body.get("analysis") if isinstance(body.get("analysis"), dict) else {}
        return json_dumps(
            {
                "direction": "input",
                "path": path,
                "case_type": body.get("case_type") or body.get("kind"),
                "source_trace_id": body.get("source_trace_id") or analysis.get("trace_id"),
                "sender_id": body.get("sender_id") or analysis.get("sender_id"),
                "text": truncate_text(
                    str(body.get("text") or analysis.get("source_text") or analysis.get("effective_text") or ""),
                    240,
                ),
                "note": truncate_text(str(body.get("note") or body.get("notes") or ""), 160),
            }
        )
    return json_dumps({"direction": "input", "path": path})


def log_output_content(path: str, payload: dict[str, Any]) -> str:
    if path == "/api/analyze":
        decision = payload.get("decision") or {}
        parsed = payload.get("parsed") if isinstance(payload.get("parsed"), dict) else {}
        return json_dumps(
            {
                "direction": "output",
                "path": path,
                "action": parsed.get("intent_action") or decision.get("action"),
                "raw_action": decision.get("action"),
                "intent_action": parsed.get("intent_action"),
                "user_intent": parsed.get("user_intent"),
                "reply_text": truncate_text(str(decision.get("reply_text") or ""), 240),
                "suggested_reply": truncate_text(str((payload.get("suggested_reply") or {}).get("text") or ""), 240),
                "suggested_reasoning": truncate_text(
                    str((payload.get("suggested_reply") or {}).get("reasoning_summary") or ""),
                    240,
                ),
                "missing_fields": payload.get("missing_fields") or [],
                "group_draft": truncate_text(str(payload.get("group_draft") or ""), 240),
                "candidate_count": len(payload.get("candidates") or []),
                "outbox_count": len(payload.get("outbox") or []),
                "pool_match_count": len(payload.get("pool_matches") or []),
                "used_short_memory": bool(payload.get("used_short_memory")),
                "conversation_id": payload.get("conversation_id"),
                "game_id": (payload.get("parsed") or {}).get("id"),
            }
        )
    if path == "/api/state":
        return json_dumps(
            {
                "direction": "output",
                "path": path,
                "customer_count": len(payload.get("customers") or []),
                "game_count": len(payload.get("games") or []),
                "recent_outbox_count": len(payload.get("recent_outbox") or []),
                "redis_enabled": (payload.get("cache") or {}).get("redis_enabled"),
            }
        )
    if path == "/api/feedback":
        state = payload.get("state") or {}
        return json_dumps(
            {
                "direction": "output",
                "path": path,
                "ok": payload.get("ok"),
                "game_count": len(state.get("games") or []),
                "recent_outbox_count": len(state.get("recent_outbox") or []),
            }
        )
    if path == "/api/approval-decision":
        state = payload.get("state") or {}
        approval = payload.get("approval") or {}
        return json_dumps(
            {
                "direction": "output",
                "path": path,
                "ok": payload.get("ok"),
                "approval_id": approval.get("id"),
                "approval_status": approval.get("status"),
                "target_type": approval.get("target_type"),
                "target_id": approval.get("target_id"),
                "recent_approval_count": len(state.get("recent_approvals") or []),
            }
        )
    if path == "/api/send-outbox":
        state = payload.get("state") or {}
        delivery = payload.get("delivery") or {}
        outbox_item = payload.get("outbox_item") or {}
        return json_dumps(
            {
                "direction": "output",
                "path": path,
                "ok": payload.get("ok"),
                "deduplicated": payload.get("deduplicated"),
                "delivery_id": delivery.get("id"),
                "outbox_id": delivery.get("outbox_id") or outbox_item.get("id"),
                "outbox_status": outbox_item.get("status"),
                "recent_delivery_count": len(state.get("recent_delivery_attempts") or []),
            }
        )
    if path == "/api/runtime-policy":
        policy = payload.get("policy") or {}
        return json_dumps(
            {
                "direction": "output",
                "path": path,
                "ok": payload.get("ok"),
                "controlled_agent_mode": policy.get("controlled_agent_mode"),
                "read_only_mode": policy.get("read_only_mode"),
                "state_writes_enabled": policy.get("state_writes_enabled"),
                "delivery_enabled": policy.get("delivery_enabled"),
                "approval_enabled": policy.get("approval_enabled"),
                "eval_writes_enabled": policy.get("eval_writes_enabled"),
                "llm_required_for_side_effect_tools": policy.get("llm_required_for_side_effect_tools"),
                "llm_required_for_state_writes": policy.get("llm_required_for_state_writes"),
            }
        )
    if path == "/api/candidate-message":
        state = payload.get("state") or {}
        candidate = payload.get("candidate_message") or {}
        return json_dumps(
            {
                "direction": "output",
                "path": path,
                "ok": payload.get("ok"),
                "intent": candidate.get("intent"),
                "feedback_type": candidate.get("feedback_type"),
                "suggested_boss_reply": truncate_text(str(candidate.get("suggested_boss_reply") or ""), 240),
                "game_count": len(state.get("games") or []),
                "recent_outbox_count": len(state.get("recent_outbox") or []),
            }
        )
    if path == "/api/manual-create-game":
        state = payload.get("state") or {}
        game = payload.get("game") or {}
        return json_dumps(
            {
                "direction": "output",
                "path": path,
                "ok": payload.get("ok"),
                "game_id": game.get("id"),
                "game_count": len(state.get("games") or []),
            }
        )
    if path == "/api/clear-board":
        state = payload.get("state") or {}
        return json_dumps(
            {
                "direction": "output",
                "path": path,
                "ok": payload.get("ok"),
                "cleared_count": payload.get("cleared_count"),
                "cleared_game_ids": payload.get("cleared_game_ids") or [],
                "game_count": len(state.get("games") or []),
            }
        )
    if path == "/api/clear-short-memory":
        return json_dumps(
            {
                "direction": "output",
                "path": path,
                "ok": payload.get("ok"),
                "conversation_id": payload.get("conversation_id"),
                "sender_id": payload.get("sender_id"),
                "cleared_count": payload.get("cleared_count"),
                "cache_key": payload.get("cache_key"),
            }
        )
    if path == "/api/customers":
        return json_dumps(
            {
                "direction": "output",
                "path": path,
                "customer_id": payload.get("id"),
                "display_name": payload.get("display_name"),
            }
        )
    if path == "/api/eval-cases":
        return json_dumps(
            {
                "direction": "output",
                "path": path,
                "ok": payload.get("ok"),
                "case_type": payload.get("case_type"),
                "record_id": payload.get("record_id"),
                "dataset_path": payload.get("path"),
                "counts": (payload.get("overview") or {}).get("counts"),
            }
        )
    return json_dumps({"direction": "output", "path": path})


def recent_log_lines(limit: int = 200) -> list[str]:
    if not LOG_PATH.exists():
        return []
    lines = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-limit:]


def render_log_page(limit: int = 200) -> str:
    lines = recent_log_lines(limit)
    body = "\n".join(lines) if lines else "暂无日志。"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>麻将馆试用日志</title>
  <style>
    body {{ margin: 0; padding: 24px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: #f7f8f5; color: #1f2a24; }}
    .bar {{ display: flex; align-items: baseline; justify-content: space-between; gap: 16px; margin-bottom: 16px; }}
    h1 {{ font-size: 20px; margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    a {{ color: #286955; text-decoration: none; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #fff; border: 1px solid #d9ded6; border-radius: 8px; padding: 16px; line-height: 1.5; }}
  </style>
</head>
<body>
  <div class="bar">
    <h1>麻将馆试用日志</h1>
    <a href="/">返回控制台</a>
  </div>
  <pre>{html.escape(body)}</pre>
</body>
</html>"""


def truncate_text(value: str, limit: int) -> str:
    text = value.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


class TrialStore:
    def __init__(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._migrate()
        self._seed_if_empty()
        self._backfill_customer_genders()
        self._backfill_state_transition_events()

    def _migrate(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS customers (
                id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                contact TEXT NOT NULL DEFAULT '',
                preferred_games TEXT NOT NULL DEFAULT '[]',
                preferred_levels TEXT NOT NULL DEFAULT '[]',
                usual_start_hours TEXT NOT NULL DEFAULT '[]',
                gender TEXT NOT NULL DEFAULT 'unknown',
                smoke_preference TEXT NOT NULL DEFAULT 'any',
                response_speed TEXT NOT NULL DEFAULT 'medium',
                response_rate REAL NOT NULL DEFAULT 0.5,
                last_invited_at TEXT,
                last_arrived_at TEXT,
                invite_count INTEGER NOT NULL DEFAULT 0,
                response_count INTEGER NOT NULL DEFAULT 0,
                arrival_count INTEGER NOT NULL DEFAULT 0,
                fatigue_score REAL NOT NULL DEFAULT 0,
                no_contact INTEGER NOT NULL DEFAULT 0,
                notes TEXT NOT NULL DEFAULT '',
                usual_party_size INTEGER,
                usual_party_size_confidence REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS trial_games (
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
            CREATE TABLE IF NOT EXISTS outbox (
                id TEXT PRIMARY KEY,
                game_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                customer_name TEXT NOT NULL,
                message_text TEXT NOT NULL,
                status TEXT NOT NULL,
                score REAL NOT NULL DEFAULT 0,
                reasons TEXT NOT NULL DEFAULT '[]',
                warnings TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                game_id TEXT,
                outbox_id TEXT,
                customer_id TEXT,
                feedback_type TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS followup_messages (
                id TEXT PRIMARY KEY,
                game_id TEXT NOT NULL,
                related_outbox_id TEXT,
                recipient_id TEXT NOT NULL,
                recipient_name TEXT NOT NULL,
                message_text TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS controlled_actions (
                action_id TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                trace_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                proposed_by TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '',
                risk_level TEXT NOT NULL DEFAULT 'unknown',
                side_effect INTEGER NOT NULL DEFAULT 1,
                approval_required INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                arguments_json TEXT NOT NULL DEFAULT '{}',
                validation_json TEXT NOT NULL DEFAULT '{}',
                result_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                executed_at TEXT,
                error TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_controlled_actions_trace
                ON controlled_actions(trace_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_controlled_actions_stage
                ON controlled_actions(stage, status, created_at);
            CREATE TABLE IF NOT EXISTS approval_requests (
                id TEXT PRIMARY KEY,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                action_id TEXT,
                idempotency_key TEXT,
                risk_level TEXT NOT NULL DEFAULT 'medium',
                status TEXT NOT NULL DEFAULT 'pending',
                reviewer_id TEXT,
                reviewer_name TEXT,
                decision_reason TEXT NOT NULL DEFAULT '',
                original_message_text TEXT NOT NULL DEFAULT '',
                final_message_text TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                decided_at TEXT,
                UNIQUE(target_type, target_id)
            );
            CREATE INDEX IF NOT EXISTS idx_approval_requests_status
                ON approval_requests(status, created_at);
            CREATE INDEX IF NOT EXISTS idx_approval_requests_action
                ON approval_requests(action_id);
            CREATE TABLE IF NOT EXISTS trace_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                level TEXT NOT NULL,
                direction TEXT NOT NULL DEFAULT 'log',
                event TEXT NOT NULL DEFAULT '',
                stage TEXT NOT NULL DEFAULT '',
                schema_version TEXT NOT NULL DEFAULT '',
                payload_json TEXT NOT NULL DEFAULT '{}',
                content TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_trace_events_trace
                ON trace_events(trace_id, id);
            CREATE INDEX IF NOT EXISTS idx_trace_events_kind
                ON trace_events(direction, event, created_at);
            CREATE TABLE IF NOT EXISTS state_transition_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                from_status TEXT,
                to_status TEXT NOT NULL,
                event TEXT NOT NULL,
                allowed INTEGER NOT NULL DEFAULT 1,
                reason TEXT NOT NULL DEFAULT '',
                trace_id TEXT NOT NULL DEFAULT '',
                action_id TEXT NOT NULL DEFAULT '',
                state_machine_version TEXT NOT NULL DEFAULT '',
                schema_version TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_state_transition_events_entity
                ON state_transition_events(entity_type, entity_id, id);
            CREATE INDEX IF NOT EXISTS idx_state_transition_events_trace
                ON state_transition_events(trace_id, id);
            CREATE INDEX IF NOT EXISTS idx_state_transition_events_event
                ON state_transition_events(event, created_at);
            CREATE TABLE IF NOT EXISTS message_delivery_attempts (
                id TEXT PRIMARY KEY,
                outbox_id TEXT NOT NULL,
                approval_id TEXT,
                channel TEXT NOT NULL DEFAULT 'manual',
                recipient_id TEXT NOT NULL DEFAULT '',
                recipient_name TEXT NOT NULL DEFAULT '',
                message_text TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'sent',
                idempotency_key TEXT NOT NULL UNIQUE,
                action_id TEXT NOT NULL DEFAULT '',
                trace_id TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                delivered_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_message_delivery_attempts_outbox
                ON message_delivery_attempts(outbox_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_message_delivery_attempts_trace
                ON message_delivery_attempts(trace_id, created_at);
            CREATE TABLE IF NOT EXISTS runtime_policies (
                id TEXT PRIMARY KEY,
                policy_json TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL,
                updated_by TEXT NOT NULL DEFAULT '',
                reason TEXT NOT NULL DEFAULT ''
            );
            """
        )
        self._ensure_column("customers", "gender", "TEXT NOT NULL DEFAULT 'unknown'")
        self._ensure_column("trial_games", "archived_at", "TEXT")
        self._ensure_column("trial_games", "final_reason", "TEXT NOT NULL DEFAULT ''")
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {
            str(row["name"])
            for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column in columns:
            return
        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _backfill_state_transition_events(self) -> None:
        sources = [
            ("game", "trial_games", "id", "status", "created_at"),
            ("outbox", "outbox", "id", "status", "created_at"),
            ("followup", "followup_messages", "id", "status", "created_at"),
        ]
        for entity_type, table, id_column, status_column, created_column in sources:
            rows = self.conn.execute(
                f"SELECT {id_column} AS entity_id, {status_column} AS status, {created_column} AS created_at FROM {table}"
            ).fetchall()
            for row in rows:
                entity_id = str(row["entity_id"] or "")
                if not entity_id:
                    continue
                existing = self.conn.execute(
                    """
                    SELECT 1 FROM state_transition_events
                    WHERE entity_type = ? AND entity_id = ?
                    LIMIT 1
                    """,
                    (entity_type, entity_id),
                ).fetchone()
                if existing:
                    continue
                status = str(row["status"] or "").strip()
                if not status:
                    continue
                transition = {
                    "allowed": True,
                    "code": "migration_backfill",
                    "reason": f"{entity_type} 现有状态回填为 {status}。",
                    "entity_type": entity_type,
                    "from_status": None,
                    "to_status": status,
                    "event": "migration_backfill",
                    "state_machine_version": STATE_MACHINE_VERSION,
                }
                self.record_state_transition(
                    transition,
                    entity_id=entity_id,
                    metadata={"source_table": table, "backfilled": True},
                    stamp=str(row["created_at"] or now_tz().isoformat()),
                )
        self.conn.commit()

    def begin_controlled_action(self, action: dict[str, Any]) -> dict[str, Any]:
        validation = action.get("validation") if isinstance(action.get("validation"), dict) else {}
        allowed = bool(validation.get("allowed"))
        existing = self.controlled_action(str(action.get("idempotency_key") or ""))
        if existing and existing.get("status") == "executed":
            return {"execute": False, "duplicate": True, "record": existing, "result": existing.get("result") or {}}
        if existing and existing.get("status") == "executing":
            return {"execute": False, "duplicate": True, "record": existing, "result": existing.get("result") or {}}
        if not allowed:
            self._upsert_controlled_action(action, status="rejected")
            record = self.controlled_action(str(action.get("idempotency_key") or ""))
            return {"execute": False, "duplicate": False, "record": record, "result": {}}
        self._upsert_controlled_action(action, status="executing")
        record = self.controlled_action(str(action.get("idempotency_key") or ""))
        return {"execute": True, "duplicate": False, "record": record, "result": {}}

    def complete_controlled_action(
        self,
        action: dict[str, Any],
        *,
        result: dict[str, Any] | None = None,
        status: str = "executed",
        error: str = "",
    ) -> dict[str, Any]:
        stamp = now_tz().isoformat()
        idempotency_key = str(action.get("idempotency_key") or "")
        self.conn.execute(
            """
            UPDATE controlled_actions
            SET status = ?, updated_at = ?, executed_at = ?, result_json = ?, error = ?
            WHERE idempotency_key = ?
            """,
            (
                status,
                stamp,
                stamp if status in {"executed", "failed"} else None,
                json_dumps(result or {}),
                error,
                idempotency_key,
            ),
        )
        self.conn.commit()
        return self.controlled_action(idempotency_key) or {}

    def controlled_action(self, idempotency_key: str) -> dict[str, Any] | None:
        if not idempotency_key:
            return None
        row = self.conn.execute(
            "SELECT * FROM controlled_actions WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        return self._controlled_action_from_row(row) if row else None

    def controlled_actions(self, limit: int = 80) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM controlled_actions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._controlled_action_from_row(row) for row in rows]

    def runtime_policy(self) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT * FROM runtime_policies WHERE id = 'default'"
        ).fetchone()
        policy = default_runtime_policy()
        if row:
            saved = json_loads(row["policy_json"], {})
            if isinstance(saved, dict):
                policy.update(saved)
            policy["updated_at"] = row["updated_at"]
            policy["updated_by"] = row["updated_by"]
            policy["reason"] = row["reason"] or policy.get("reason") or ""
        policy["policy_version"] = RUNTIME_POLICY_VERSION
        policy["controlled_agent_mode"] = str(policy.get("controlled_agent_mode") or "trial")
        default_policy = default_runtime_policy()
        for key in [
            "read_only_mode",
            "state_writes_enabled",
            "delivery_enabled",
            "approval_enabled",
            "eval_writes_enabled",
            "llm_required_for_side_effect_tools",
            "llm_required_for_state_writes",
        ]:
            policy[key] = self._bool_policy_value(policy.get(key), bool(default_policy[key]))
        return policy

    def update_runtime_policy(self, payload: dict[str, Any]) -> dict[str, Any]:
        current = self.runtime_policy()
        allowed_keys = {
            "controlled_agent_mode",
            "read_only_mode",
            "state_writes_enabled",
            "delivery_enabled",
            "approval_enabled",
            "eval_writes_enabled",
            "llm_required_for_side_effect_tools",
            "llm_required_for_state_writes",
            "reason",
        }
        next_policy = dict(current)
        for key in allowed_keys:
            if key not in payload:
                continue
            if key == "reason":
                next_policy[key] = str(payload.get(key) or "")[:500]
            elif key == "controlled_agent_mode":
                raw_mode = str(payload.get(key) or next_policy.get(key) or "trial").strip().lower()
                next_policy[key] = "production" if raw_mode in {"prod", "production", "controlled", "strict"} else "trial"
            else:
                next_policy[key] = self._bool_policy_value(payload.get(key), bool(current.get(key)))
        if next_policy.get("controlled_agent_mode") == "production":
            if "llm_required_for_side_effect_tools" not in payload:
                next_policy["llm_required_for_side_effect_tools"] = True
            if "llm_required_for_state_writes" not in payload:
                next_policy["llm_required_for_state_writes"] = True
        next_policy["policy_version"] = RUNTIME_POLICY_VERSION
        stamp = (parse_dt(payload.get("now")) or now_tz()).isoformat()
        updated_by = str(payload.get("updated_by") or payload.get("reviewer_id") or "boss_manual").strip() or "boss_manual"
        reason = str(next_policy.get("reason") or "")
        self.conn.execute(
            """
            INSERT INTO runtime_policies (id, policy_json, updated_at, updated_by, reason)
            VALUES ('default', ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                policy_json=excluded.policy_json,
                updated_at=excluded.updated_at,
                updated_by=excluded.updated_by,
                reason=excluded.reason
            """,
            (json_dumps(next_policy), stamp, updated_by, reason),
        )
        self.conn.commit()
        return self.runtime_policy()

    def _bool_policy_value(self, value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "打开", "开启", "是"}:
                return True
            if normalized in {"0", "false", "no", "off", "关闭", "否"}:
                return False
        return default

    def _upsert_controlled_action(self, action: dict[str, Any], *, status: str) -> None:
        stamp = now_tz().isoformat()
        validation = action.get("validation") if isinstance(action.get("validation"), dict) else {}
        arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
        created_at = str(action.get("created_at") or stamp)
        self.conn.execute(
            """
            INSERT INTO controlled_actions (
                action_id, idempotency_key, trace_id, stage, tool_name,
                proposed_by, source, risk_level, side_effect, approval_required,
                status, reason, arguments_json, validation_json, result_json,
                created_at, updated_at, executed_at, error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', ?, ?, NULL, '')
            ON CONFLICT(idempotency_key) DO UPDATE SET
                action_id=excluded.action_id,
                trace_id=excluded.trace_id,
                stage=excluded.stage,
                tool_name=excluded.tool_name,
                proposed_by=excluded.proposed_by,
                source=excluded.source,
                risk_level=excluded.risk_level,
                side_effect=excluded.side_effect,
                approval_required=excluded.approval_required,
                status=excluded.status,
                reason=excluded.reason,
                arguments_json=excluded.arguments_json,
                validation_json=excluded.validation_json,
                updated_at=excluded.updated_at,
                error=''
            """,
            (
                action.get("action_id"),
                action.get("idempotency_key"),
                self._trace_id_from_action(action),
                action.get("stage"),
                action.get("tool_name"),
                action.get("proposed_by") or "",
                action.get("source") or "",
                action.get("risk_level") or "unknown",
                1 if action.get("side_effect") else 0,
                1 if action.get("approval_required") else 0,
                status,
                str(action.get("reason") or ""),
                json_dumps(arguments),
                json_dumps(validation),
                created_at,
                stamp,
            ),
        )
        self.conn.commit()

    def _controlled_action_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "action_id": row["action_id"],
            "idempotency_key": row["idempotency_key"],
            "trace_id": row["trace_id"],
            "stage": row["stage"],
            "tool_name": row["tool_name"],
            "proposed_by": row["proposed_by"],
            "source": row["source"],
            "risk_level": row["risk_level"],
            "side_effect": bool(row["side_effect"]),
            "approval_required": bool(row["approval_required"]),
            "status": row["status"],
            "reason": row["reason"],
            "arguments": json_loads(row["arguments_json"], {}),
            "validation": json_loads(row["validation_json"], {}),
            "result": json_loads(row["result_json"], {}),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "executed_at": row["executed_at"],
            "error": row["error"],
        }

    def _trace_id_from_action(self, action: dict[str, Any]) -> str:
        idempotency_key = str(action.get("idempotency_key") or "")
        if ":" in idempotency_key:
            return idempotency_key.split(":", 1)[0]
        return str(action.get("trace_id") or "")

    def _seed_if_empty(self) -> None:
        existing_ids = {
            str(row["id"])
            for row in self.conn.execute("SELECT id FROM customers").fetchall()
        }
        for item in SEED_CUSTOMERS:
            customer_id = self._safe_id(str(item.get("id") or item.get("display_name") or "customer"))
            if customer_id in existing_ids:
                continue
            self.upsert_customer(item)
            existing_ids.add(customer_id)

    def _backfill_customer_genders(self) -> None:
        rows = self.conn.execute(
            "SELECT id, display_name, notes, gender FROM customers WHERE gender IS NULL OR gender = '' OR gender = 'unknown'"
        ).fetchall()
        for row in rows:
            inferred = infer_gender_from_customer_text(str(row["display_name"] or ""), str(row["notes"] or ""))
            if inferred == "unknown":
                continue
            self.conn.execute("UPDATE customers SET gender = ? WHERE id = ?", (inferred, row["id"]))
        self.conn.commit()

    def customers(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM customers ORDER BY display_name").fetchall()
        return [self._customer_from_row(row) for row in rows]

    def customer(self, customer_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
        return self._customer_from_row(row) if row else None

    def upsert_customer(self, item: dict[str, Any]) -> dict[str, Any]:
        customer_id = self._safe_id(str(item.get("id") or item.get("display_name") or "customer"))
        display_name = str(item.get("display_name") or customer_id)
        preferred_games = list(item.get("preferred_games") or [])
        preferred_levels = list(item.get("preferred_levels") or [])
        usual_start_hours = [int(hour) for hour in item.get("usual_start_hours") or [] if str(hour).isdigit()]
        gender = normalize_gender(item.get("gender"))
        if gender == "unknown":
            gender = infer_gender_from_customer_text(display_name, str(item.get("notes") or ""))
        self.conn.execute(
            """
            INSERT INTO customers (
                id, display_name, contact, preferred_games, preferred_levels, usual_start_hours,
                gender, smoke_preference, response_speed, response_rate, last_invited_at, last_arrived_at,
                invite_count, response_count, arrival_count, fatigue_score, no_contact, notes,
                usual_party_size, usual_party_size_confidence
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                display_name=excluded.display_name,
                contact=excluded.contact,
                preferred_games=excluded.preferred_games,
                preferred_levels=excluded.preferred_levels,
                usual_start_hours=excluded.usual_start_hours,
                gender=excluded.gender,
                smoke_preference=excluded.smoke_preference,
                response_speed=excluded.response_speed,
                response_rate=excluded.response_rate,
                last_invited_at=COALESCE(excluded.last_invited_at, customers.last_invited_at),
                last_arrived_at=COALESCE(excluded.last_arrived_at, customers.last_arrived_at),
                fatigue_score=excluded.fatigue_score,
                no_contact=excluded.no_contact,
                notes=excluded.notes,
                usual_party_size=excluded.usual_party_size,
                usual_party_size_confidence=excluded.usual_party_size_confidence
            """,
            (
                customer_id,
                display_name,
                str(item.get("contact") or ""),
                json_dumps(preferred_games),
                json_dumps(preferred_levels),
                json_dumps(usual_start_hours),
                gender,
                str(item.get("smoke_preference") or "any"),
                str(item.get("response_speed") or "medium"),
                float(item.get("response_rate") or 0.5),
                item.get("last_invited_at"),
                item.get("last_arrived_at"),
                int(item.get("invite_count") or 0),
                int(item.get("response_count") or 0),
                int(item.get("arrival_count") or 0),
                float(item.get("fatigue_score") or 0),
                1 if item.get("no_contact") else 0,
                str(item.get("notes") or ""),
                item.get("usual_party_size"),
                float(item.get("usual_party_size_confidence") or 0),
            ),
        )
        self.conn.commit()
        return self.customer(customer_id) or {}

    def create_game(
        self,
        game_id: str,
        status: str,
        organizer_id: str,
        organizer_name: str,
        source_text: str,
        parsed: dict[str, Any],
        reply_text: str,
        missing_fields: list[str],
        notes: list[str],
    ) -> None:
        existing = self.conn.execute("SELECT status FROM trial_games WHERE id = ?", (game_id,)).fetchone()
        transition = require_state_transition(
            entity_type="game",
            current_status=str(existing["status"]) if existing else None,
            next_status=status,
            event="create_game",
        )
        stamp = now_tz().isoformat()
        self.conn.execute(
            """
            INSERT INTO trial_games (
                id, created_at, updated_at, status, organizer_id, organizer_name,
                source_text, parsed_json, reply_text, missing_fields, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                updated_at=excluded.updated_at,
                status=excluded.status,
                parsed_json=excluded.parsed_json,
                reply_text=excluded.reply_text,
                missing_fields=excluded.missing_fields,
                notes=excluded.notes
            """,
            (
                game_id,
                stamp,
                stamp,
                status,
                organizer_id,
                organizer_name,
                source_text,
                json_dumps(parsed),
                reply_text,
                json_dumps(missing_fields),
                json_dumps(notes),
            ),
        )
        self.record_state_transition(
            transition,
            entity_id=game_id,
            metadata={"organizer_id": organizer_id, "source": "create_game"},
            stamp=stamp,
        )
        self.conn.commit()

    def create_outbox(
        self,
        game_id: str,
        customer_id: str,
        customer_name: str,
        message_text: str,
        score: float,
        reasons: list[str],
        warnings: list[str],
    ) -> str:
        existing = self.conn.execute(
            "SELECT id FROM outbox WHERE game_id = ? AND customer_id = ?",
            (game_id, customer_id),
        ).fetchone()
        if existing:
            return str(existing["id"])
        outbox_id = f"out_{game_id[-6:]}_{customer_id}"
        transition = require_state_transition(
            entity_type="outbox",
            current_status=None,
            next_status="待审批",
            event="create_pending_outbox",
        )
        stamp = now_tz().isoformat()
        self.conn.execute(
            """
            INSERT INTO outbox (
                id, game_id, customer_id, customer_name, message_text, status,
                score, reasons, warnings, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, '待审批', ?, ?, ?, ?, ?)
            """,
            (
                outbox_id,
                game_id,
                customer_id,
                customer_name,
                message_text,
                score,
                json_dumps(reasons),
                json_dumps(warnings),
                stamp,
                stamp,
            ),
        )
        self.record_state_transition(
            transition,
            entity_id=outbox_id,
            metadata={"game_id": game_id, "customer_id": customer_id},
            stamp=stamp,
        )
        self.conn.commit()
        return outbox_id

    def create_approval_request(
        self,
        *,
        target_type: str,
        target_id: str,
        action_id: str | None,
        idempotency_key: str | None,
        risk_level: str,
        original_message_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        safe_target_type = self._safe_id(target_type or "target")
        approval_id = f"approval_{safe_target_type}_{self._safe_id(target_id)}"
        stamp = now_tz().isoformat()
        self.conn.execute(
            """
            INSERT INTO approval_requests (
                id, target_type, target_id, action_id, idempotency_key, risk_level,
                status, original_message_text, final_message_text, metadata_json,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?)
            ON CONFLICT(target_type, target_id) DO UPDATE SET
                action_id=COALESCE(approval_requests.action_id, excluded.action_id),
                idempotency_key=COALESCE(approval_requests.idempotency_key, excluded.idempotency_key),
                risk_level=excluded.risk_level,
                original_message_text=CASE
                    WHEN approval_requests.status = 'pending' THEN excluded.original_message_text
                    ELSE approval_requests.original_message_text
                END,
                final_message_text=CASE
                    WHEN approval_requests.status = 'pending' THEN excluded.final_message_text
                    ELSE approval_requests.final_message_text
                END,
                metadata_json=CASE
                    WHEN approval_requests.status = 'pending' THEN excluded.metadata_json
                    ELSE approval_requests.metadata_json
                END,
                updated_at=excluded.updated_at
            """,
            (
                approval_id,
                safe_target_type,
                target_id,
                action_id,
                idempotency_key,
                risk_level or "medium",
                original_message_text,
                original_message_text,
                json_dumps(metadata or {}),
                stamp,
                stamp,
            ),
        )
        self.conn.commit()
        return self.approval_for_target(safe_target_type, target_id) or {
            "id": approval_id,
            "target_type": safe_target_type,
            "target_id": target_id,
            "status": "pending",
        }

    def approval_for_target(self, target_type: str, target_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM approval_requests WHERE target_type = ? AND target_id = ?",
            (target_type, target_id),
        ).fetchone()
        return self._approval_from_row(row) if row else None

    def approval_request(self, approval_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM approval_requests WHERE id = ?", (approval_id,)).fetchone()
        return self._approval_from_row(row) if row else None

    def recent_approvals(self, limit: int = 80) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM approval_requests ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._approval_from_row(row) for row in rows]

    def trace_events(self, trace_id: str, limit: int = 300) -> list[dict[str, Any]]:
        safe_trace_id = str(trace_id or "").strip()
        if not safe_trace_id:
            return []
        rows = self.conn.execute(
            """
            SELECT * FROM trace_events
            WHERE trace_id = ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (safe_trace_id, max(1, min(int(limit or 300), 1000))),
        ).fetchall()
        return [self._trace_event_from_row(row) for row in rows]

    def recent_traces(self, limit: int = 40) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT
                trace_id,
                MIN(created_at) AS first_seen_at,
                MAX(created_at) AS last_seen_at,
                COUNT(*) AS event_count,
                SUM(CASE WHEN direction = 'llm' THEN 1 ELSE 0 END) AS llm_event_count,
                SUM(CASE WHEN direction = 'tool' THEN 1 ELSE 0 END) AS tool_event_count,
                SUM(CASE WHEN level = 'ERROR' THEN 1 ELSE 0 END) AS error_count
            FROM trace_events
            GROUP BY trace_id
            ORDER BY last_seen_at DESC
            LIMIT ?
            """,
            (max(1, min(int(limit or 40), 200)),),
        ).fetchall()
        return [
            {
                "trace_id": row["trace_id"],
                "first_seen_at": row["first_seen_at"],
                "last_seen_at": row["last_seen_at"],
                "event_count": int(row["event_count"] or 0),
                "llm_event_count": int(row["llm_event_count"] or 0),
                "tool_event_count": int(row["tool_event_count"] or 0),
                "error_count": int(row["error_count"] or 0),
            }
            for row in rows
        ]

    def record_state_transition(
        self,
        transition: dict[str, Any],
        *,
        entity_id: str,
        trace_id: str | None = None,
        action_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        stamp: str | None = None,
    ) -> dict[str, Any]:
        record = {
            **transition,
            "entity_id": str(entity_id),
            "trace_id": str(trace_id or ""),
            "action_id": str(action_id or ""),
            "schema_version": STATE_TRANSITION_EVENT_SCHEMA_VERSION,
            "metadata": metadata or {},
        }
        self.conn.execute(
            """
            INSERT INTO state_transition_events (
                created_at, entity_type, entity_id, from_status, to_status,
                event, allowed, reason, trace_id, action_id,
                state_machine_version, schema_version, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stamp or now_tz().isoformat(),
                str(record.get("entity_type") or ""),
                str(entity_id),
                record.get("from_status"),
                str(record.get("to_status") or ""),
                str(record.get("event") or ""),
                1 if record.get("allowed") else 0,
                str(record.get("reason") or ""),
                str(trace_id or ""),
                str(action_id or ""),
                str(record.get("state_machine_version") or STATE_MACHINE_VERSION),
                STATE_TRANSITION_EVENT_SCHEMA_VERSION,
                json_dumps(metadata or {}),
            ),
        )
        return record

    def state_transition_events(
        self,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
        trace_id: str | None = None,
        limit: int = 120,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if entity_type:
            clauses.append("entity_type = ?")
            params.append(str(entity_type))
        if entity_id:
            clauses.append("entity_id = ?")
            params.append(str(entity_id))
        if trace_id:
            clauses.append("trace_id = ?")
            params.append(str(trace_id))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit or 120), 1000)))
        rows = self.conn.execute(
            f"""
            SELECT * FROM state_transition_events
            {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [self._state_transition_event_from_row(row) for row in rows]

    def delivery_attempt(self, delivery_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM message_delivery_attempts WHERE id = ?",
            (delivery_id,),
        ).fetchone()
        return self._delivery_attempt_from_row(row) if row else None

    def delivery_attempt_by_idempotency_key(self, idempotency_key: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM message_delivery_attempts WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        return self._delivery_attempt_from_row(row) if row else None

    def delivery_attempts_for_outbox(self, outbox_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM message_delivery_attempts WHERE outbox_id = ? ORDER BY created_at DESC",
            (outbox_id,),
        ).fetchall()
        return [self._delivery_attempt_from_row(row) for row in rows]

    def recent_delivery_attempts(self, limit: int = 80) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM message_delivery_attempts ORDER BY created_at DESC LIMIT ?",
            (max(1, min(int(limit or 80), 300)),),
        ).fetchall()
        return [self._delivery_attempt_from_row(row) for row in rows]

    def execute_outbox_delivery(self, payload: dict[str, Any]) -> dict[str, Any]:
        outbox_id = str(payload.get("outbox_id") or "").strip()
        if not outbox_id:
            raise ValueError("缺少 outbox_id")
        idempotency_key = str(payload.get("idempotency_key") or "").strip()
        if not idempotency_key:
            raise ValueError("缺少发送幂等键")
        existing = self.delivery_attempt_by_idempotency_key(idempotency_key)
        if existing:
            return {"ok": True, "deduplicated": True, "delivery": existing}

        row = self.conn.execute("SELECT * FROM outbox WHERE id = ?", (outbox_id,)).fetchone()
        if not row:
            raise ValueError("找不到待发送 outbox")
        outbox_item = self._outbox_from_row(row)
        approval = outbox_item.get("approval") if isinstance(outbox_item.get("approval"), dict) else None
        if not approval or approval.get("status") != "approved":
            raise ValueError("只有审批通过的草稿才能执行发送")
        current_status = str(outbox_item.get("status") or "")
        if current_status not in {"已审批", "已复制", "已发送"}:
            raise ValueError(f"当前状态 {current_status} 不能执行发送")

        stamp = (parse_dt(payload.get("now")) or now_tz()).isoformat()
        trace_id = str(payload.get("trace_id") or "")
        action_id = str(payload.get("action_id") or "")
        channel = str(payload.get("channel") or "manual").strip() or "manual"
        final_message = str(payload.get("message_text") or approval.get("final_message_text") or row["message_text"] or "")
        delivery_id = "delivery_" + hashlib.sha256(f"{idempotency_key}:{outbox_id}".encode("utf-8")).hexdigest()[:16]
        transition: dict[str, Any] | None = None
        if current_status != "已发送":
            transition = require_state_transition(
                entity_type="outbox",
                current_status=current_status,
                next_status="已发送",
                event="execute_send",
            )
            self.record_state_transition(
                transition,
                entity_id=outbox_id,
                trace_id=trace_id,
                action_id=action_id,
                metadata={"channel": channel, "approval_id": approval.get("id")},
                stamp=stamp,
            )
            self.conn.execute(
                "UPDATE outbox SET status = '已发送', message_text = ?, updated_at = ? WHERE id = ?",
                (final_message, stamp, outbox_id),
            )
        self.conn.execute(
            """
            INSERT INTO message_delivery_attempts (
                id, outbox_id, approval_id, channel, recipient_id, recipient_name,
                message_text, status, idempotency_key, action_id, trace_id, error,
                metadata_json, created_at, updated_at, delivered_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'sent', ?, ?, ?, '', ?, ?, ?, ?)
            """,
            (
                delivery_id,
                outbox_id,
                approval.get("id"),
                channel,
                str(row["customer_id"] or ""),
                str(row["customer_name"] or ""),
                final_message,
                idempotency_key,
                action_id,
                trace_id,
                json_dumps(payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}),
                stamp,
                stamp,
                stamp,
            ),
        )
        self.conn.execute(
            """
            INSERT INTO feedback (created_at, game_id, outbox_id, customer_id, feedback_type, notes)
            VALUES (?, ?, ?, ?, 'sent', ?)
            """,
            (stamp, row["game_id"], outbox_id, row["customer_id"], f"发送通道：{channel}"),
        )
        self.conn.commit()
        delivery = self.delivery_attempt(delivery_id) or {"id": delivery_id}
        return {
            "ok": True,
            "deduplicated": False,
            "delivery": delivery,
            "outbox_item": self.outbox_item(outbox_id),
            "state_transition": transition,
        }

    def decide_approval(self, payload: dict[str, Any]) -> dict[str, Any]:
        approval_id = str(payload.get("approval_id") or "").strip()
        target_type = str(payload.get("target_type") or "").strip()
        target_id = str(payload.get("target_id") or "").strip()
        if approval_id:
            approval = self.approval_request(approval_id)
        elif target_type and target_id:
            approval = self.approval_for_target(target_type, target_id)
        else:
            raise ValueError("缺少 approval_id 或 target_type/target_id")
        if not approval:
            raise ValueError("找不到审批请求")

        decision = str(payload.get("decision") or payload.get("status") or "").strip().lower()
        decision_map = {
            "approve": "approved",
            "approved": "approved",
            "同意": "approved",
            "通过": "approved",
            "reject": "rejected",
            "rejected": "rejected",
            "拒绝": "rejected",
        }
        status = decision_map.get(decision)
        if status not in {"approved", "rejected"}:
            raise ValueError("审批结果只能是 approved/rejected")
        stamp = (parse_dt(payload.get("now")) or now_tz()).isoformat()
        final_message = str(payload.get("final_message_text") or approval.get("final_message_text") or approval.get("original_message_text") or "")
        reviewer_id = str(payload.get("reviewer_id") or "boss_manual").strip() or "boss_manual"
        reviewer_name = str(payload.get("reviewer_name") or "老板").strip() or "老板"
        reason = str(payload.get("reason") or payload.get("decision_reason") or "").strip()
        self.conn.execute(
            """
            UPDATE approval_requests
            SET status = ?, reviewer_id = ?, reviewer_name = ?, decision_reason = ?,
                final_message_text = ?, updated_at = ?, decided_at = ?
            WHERE id = ?
            """,
            (status, reviewer_id, reviewer_name, reason, final_message, stamp, stamp, approval["id"]),
        )
        target_status = "已审批" if status == "approved" else "审批拒绝"
        if approval["target_type"] == "outbox":
            target_row = self.conn.execute(
                "SELECT status FROM outbox WHERE id = ?",
                (approval["target_id"],),
            ).fetchone()
            transition = require_state_transition(
                entity_type="outbox",
                current_status=str(target_row["status"]) if target_row else None,
                next_status=target_status,
                event="approval_decision",
            )
            self.conn.execute(
                "UPDATE outbox SET status = ?, message_text = ?, updated_at = ? WHERE id = ?",
                (target_status, final_message, stamp, approval["target_id"]),
            )
            self.record_state_transition(
                transition,
                entity_id=str(approval["target_id"]),
                metadata={"approval_id": approval["id"], "decision": status},
                stamp=stamp,
            )
        elif approval["target_type"] == "followup":
            target_row = self.conn.execute(
                "SELECT status FROM followup_messages WHERE id = ?",
                (approval["target_id"],),
            ).fetchone()
            transition = require_state_transition(
                entity_type="followup",
                current_status=str(target_row["status"]) if target_row else None,
                next_status=target_status,
                event="approval_decision",
            )
            self.conn.execute(
                "UPDATE followup_messages SET status = ?, message_text = ?, updated_at = ? WHERE id = ?",
                (target_status, final_message, stamp, approval["target_id"]),
            )
            self.record_state_transition(
                transition,
                entity_id=str(approval["target_id"]),
                metadata={"approval_id": approval["id"], "decision": status},
                stamp=stamp,
            )
        self.conn.commit()
        updated = self.approval_request(approval["id"]) or approval
        return {"ok": True, "approval": updated}

    def create_followup_message(
        self,
        *,
        game_id: str,
        related_outbox_id: str | None,
        recipient_id: str,
        recipient_name: str,
        message_text: str,
        reason: str,
    ) -> dict[str, Any]:
        safe_recipient = self._safe_id(recipient_id or recipient_name or "recipient")
        source = self._safe_id(related_outbox_id or game_id)
        followup_id = f"follow_{source[-10:]}_{safe_recipient}"
        existing = self.followup_message(followup_id)
        if existing and str(existing.get("status") or "") != "待审批":
            return existing
        transition = require_state_transition(
            entity_type="followup",
            current_status=str(existing.get("status") or "") if existing else None,
            next_status="待审批",
            event="create_pending_followup",
        )
        stamp = now_tz().isoformat()
        self.conn.execute(
            """
            INSERT INTO followup_messages (
                id, game_id, related_outbox_id, recipient_id, recipient_name,
                message_text, status, reason, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, '待审批', ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                message_text=excluded.message_text,
                status='待审批',
                reason=excluded.reason,
                updated_at=excluded.updated_at
            """,
            (
                followup_id,
                game_id,
                related_outbox_id,
                safe_recipient,
                recipient_name,
                message_text,
                reason,
                stamp,
                stamp,
            ),
        )
        self.record_state_transition(
            transition,
            entity_id=followup_id,
            metadata={"game_id": game_id, "related_outbox_id": related_outbox_id, "recipient_id": recipient_id},
            stamp=stamp,
        )
        self.conn.commit()
        return self.followup_message(followup_id) or {
            "id": followup_id,
            "game_id": game_id,
            "related_outbox_id": related_outbox_id,
            "recipient_id": safe_recipient,
            "recipient_name": recipient_name,
            "message_text": message_text,
            "status": "待审批",
            "reason": reason,
            "created_at": stamp,
            "updated_at": stamp,
        }

    def games(self, *, include_final: bool = False) -> list[dict[str, Any]]:
        if include_final:
            rows = self.conn.execute(
                "SELECT * FROM trial_games ORDER BY created_at DESC LIMIT 30"
            ).fetchall()
        else:
            final_statuses = tuple(sorted(FINAL_GAME_STATUSES))
            placeholders = ",".join("?" for _ in final_statuses)
            rows = self.conn.execute(
                f"""
                SELECT * FROM trial_games
                WHERE status NOT IN ({placeholders})
                ORDER BY created_at DESC
                LIMIT 30
                """,
                final_statuses,
            ).fetchall()
        return [self._game_from_row(row) for row in rows]

    def clear_current_games(self, *, reason: str = "老板手动清空当前局看板") -> dict[str, Any]:
        final_statuses = tuple(sorted(FINAL_GAME_STATUSES))
        placeholders = ",".join("?" for _ in final_statuses)
        rows = self.conn.execute(
            f"""
            SELECT id, status FROM trial_games
            WHERE status NOT IN ({placeholders})
            ORDER BY created_at DESC
            """,
            final_statuses,
        ).fetchall()
        game_ids = [str(row["id"]) for row in rows]
        if not game_ids:
            return {"ok": True, "cleared_count": 0, "cleared_game_ids": []}

        stamp = now_tz().isoformat()
        transitions: list[dict[str, Any]] = []
        id_placeholders = ",".join("?" for _ in game_ids)
        for row in rows:
            transition = require_state_transition(
                entity_type="game",
                current_status=str(row["status"] or ""),
                next_status="已取消",
                event="clear_board",
            )
            recorded = self.record_state_transition(
                transition,
                entity_id=str(row["id"]),
                metadata={"reason": reason},
                stamp=stamp,
            )
            transitions.append(recorded)
            self.conn.execute(
                """
                UPDATE trial_games
                SET status = '已取消', updated_at = ?, archived_at = ?, final_reason = ?
                WHERE id = ?
                """,
                (stamp, stamp, reason, str(row["id"])),
            )
        outbox_rows = self.conn.execute(
            f"""
            SELECT id, status FROM outbox
            WHERE game_id IN ({id_placeholders})
            """,
            game_ids,
        ).fetchall()
        for row in outbox_rows:
            transition = state_transition_verdict(
                entity_type="outbox",
                current_status=str(row["status"] or ""),
                next_status="局取消",
                event="clear_board",
            )
            if not transition["allowed"]:
                continue
            recorded = self.record_state_transition(
                transition,
                entity_id=str(row["id"]),
                metadata={"reason": reason},
                stamp=stamp,
            )
            transitions.append(recorded)
            self.conn.execute(
                "UPDATE outbox SET status = '局取消', updated_at = ? WHERE id = ?",
                (stamp, str(row["id"])),
            )
        for game_id in game_ids:
            self.conn.execute(
                """
                INSERT INTO feedback (created_at, game_id, outbox_id, customer_id, feedback_type, notes)
                VALUES (?, ?, NULL, NULL, 'board_cleared', ?)
                """,
                (stamp, game_id, reason),
            )
        self.conn.commit()
        return {
            "ok": True,
            "cleared_count": len(game_ids),
            "cleared_game_ids": game_ids,
            "state_machine_version": STATE_MACHINE_VERSION,
            "state_transitions": transitions,
        }

    def run_lifecycle(self, *, now: datetime | None = None) -> list[dict[str, Any]]:
        current = now or now_tz()
        archived: list[dict[str, Any]] = []
        active_games = self.games()
        for game in active_games:
            timeout_reason = self._timeout_failure_reason(game, current)
            if timeout_reason:
                archived.append(self._archive_game(game["id"], timeout_reason, current))
                continue
            declined_reason = self._all_declined_failure_reason(game)
            if declined_reason:
                archived.append(self._archive_game(game["id"], declined_reason, current))
        return archived

    def archived_games(self, limit: int = 10) -> list[dict[str, Any]]:
        final_statuses = tuple(sorted(FINAL_GAME_STATUSES))
        placeholders = ",".join("?" for _ in final_statuses)
        rows = self.conn.execute(
            f"""
            SELECT * FROM trial_games
            WHERE status IN ({placeholders})
            ORDER BY COALESCE(archived_at, updated_at, created_at) DESC
            LIMIT ?
            """,
            (*final_statuses, limit),
        ).fetchall()
        return [self._game_from_row(row) for row in rows]

    def _archive_game(self, game_id: str, reason: str, now: datetime) -> dict[str, Any]:
        stamp = now.isoformat()
        row = self.conn.execute("SELECT status FROM trial_games WHERE id = ?", (game_id,)).fetchone()
        if not row:
            return {"id": game_id, "final_reason": reason}
        transition = require_state_transition(
            entity_type="game",
            current_status=str(row["status"] or ""),
            next_status="已取消",
            event="auto_archive_game",
        )
        self.record_state_transition(
            transition,
            entity_id=game_id,
            metadata={"reason": reason},
            stamp=stamp,
        )
        self.conn.execute(
            """
            UPDATE trial_games
            SET status = '已取消', updated_at = ?, archived_at = ?, final_reason = ?
            WHERE id = ? AND status NOT IN ('已成局', '已取消')
            """,
            (stamp, stamp, reason, game_id),
        )
        outbox_rows = self.conn.execute(
            "SELECT id, status FROM outbox WHERE game_id = ?",
            (game_id,),
        ).fetchall()
        for outbox_row in outbox_rows:
            transition = state_transition_verdict(
                entity_type="outbox",
                current_status=str(outbox_row["status"] or ""),
                next_status="局取消",
                event="auto_archive_game",
            )
            if not transition["allowed"]:
                continue
            self.record_state_transition(
                transition,
                entity_id=str(outbox_row["id"]),
                metadata={"game_id": game_id, "reason": reason},
                stamp=stamp,
            )
            self.conn.execute(
                "UPDATE outbox SET status = '局取消', updated_at = ? WHERE id = ?",
                (stamp, str(outbox_row["id"])),
            )
        self.conn.execute(
            """
            INSERT INTO feedback (created_at, game_id, outbox_id, customer_id, feedback_type, notes)
            VALUES (?, ?, NULL, NULL, 'game_auto_archived', ?)
            """,
            (stamp, game_id, reason),
        )
        self.conn.commit()
        row = self.conn.execute("SELECT * FROM trial_games WHERE id = ?", (game_id,)).fetchone()
        return self._game_from_row(row) if row else {"id": game_id, "final_reason": reason}

    def _timeout_failure_reason(self, game: dict[str, Any], now: datetime) -> str | None:
        parsed = game.get("parsed") if isinstance(game.get("parsed"), dict) else {}
        start_at = parse_dt(parsed.get("start_at"))
        if not start_at:
            return None
        deadline = start_at + timedelta(minutes=GAME_EXPIRE_GRACE_MINUTES)
        if now <= deadline:
            return None
        confirmed = self._confirmed_count(game)
        target_missing = parsed.get("missing_count")
        if target_missing is None:
            target_missing = max(0, 4 - int(parsed.get("current_player_count") or 0))
        if confirmed >= int(target_missing or 0):
            return None
        summary = parsed.get("summary") or game.get("source_text") or game["id"]
        return f"{summary} 超过开局时间 {GAME_EXPIRE_GRACE_MINUTES} 分钟仍未补齐，自动归档。"

    def _all_declined_failure_reason(self, game: dict[str, Any]) -> str | None:
        outbox = game.get("outbox") or []
        if not outbox:
            return None
        statuses = {str(item.get("status") or "") for item in outbox}
        if statuses and statuses <= DECLINED_OUTBOX_STATUSES:
            parsed = game.get("parsed") if isinstance(game.get("parsed"), dict) else {}
            summary = parsed.get("summary") or game.get("source_text") or game["id"]
            return f"{summary} 已邀约候选人均拒绝或暂不参加，自动归档。"
        return None

    def _confirmed_count(self, game: dict[str, Any]) -> int:
        return sum(
            1
            for item in game.get("outbox") or []
            if str(item.get("status") or "") in {"已确认", "已到店"}
        )

    def outbox_for_game(self, game_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM outbox WHERE game_id = ? ORDER BY score DESC, created_at",
            (game_id,),
        ).fetchall()
        return [self._outbox_from_row(row) for row in rows]

    def recent_outbox(self, limit: int = 60) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM outbox ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._outbox_from_row(row) for row in rows]

    def outbox_item(self, outbox_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM outbox WHERE id = ?", (outbox_id,)).fetchone()
        return self._outbox_from_row(row) if row else None

    def followups_for_game(self, game_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM followup_messages WHERE game_id = ? ORDER BY created_at DESC",
            (game_id,),
        ).fetchall()
        return [self._followup_from_row(row) for row in rows]

    def recent_followups(self, limit: int = 60) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM followup_messages ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._followup_from_row(row) for row in rows]

    def followup_message(self, followup_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM followup_messages WHERE id = ?", (followup_id,)).fetchone()
        return self._followup_from_row(row) if row else None

    def record_feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        feedback_type = str(payload.get("feedback_type") or "")
        outbox_id = payload.get("outbox_id")
        game_id = payload.get("game_id")
        customer_id = payload.get("customer_id")
        notes = str(payload.get("notes") or "")
        stamp_dt = parse_dt(payload.get("now")) or now_tz()
        stamp = stamp_dt.isoformat()
        transitions: list[dict[str, Any]] = []
        if outbox_id:
            row = self.conn.execute("SELECT * FROM outbox WHERE id = ?", (outbox_id,)).fetchone()
            if row:
                customer_id = customer_id or row["customer_id"]
                game_id = game_id or row["game_id"]
                next_status = self._outbox_status_for_feedback(feedback_type)
                transition = require_state_transition(
                    entity_type="outbox",
                    current_status=str(row["status"] or ""),
                    next_status=next_status,
                    event=f"feedback:{feedback_type}",
                )
                recorded = self.record_state_transition(
                    transition,
                    entity_id=str(outbox_id),
                    trace_id=payload.get("trace_id"),
                    metadata={"feedback_type": feedback_type, "game_id": game_id, "customer_id": customer_id},
                    stamp=stamp,
                )
                transitions.append(recorded)
                self.conn.execute(
                    "UPDATE outbox SET status = ?, updated_at = ? WHERE id = ?",
                    (next_status, stamp, outbox_id),
                )

        if game_id and feedback_type in {"game_success", "game_cancelled"}:
            status = "已成局" if feedback_type == "game_success" else "已取消"
            final_reason = notes.strip()
            if not final_reason:
                final_reason = "老板标记已成局。" if feedback_type == "game_success" else self._manual_cancel_reason(str(game_id))
            row = self.conn.execute("SELECT status FROM trial_games WHERE id = ?", (game_id,)).fetchone()
            transition = require_state_transition(
                entity_type="game",
                current_status=str(row["status"] or "") if row else None,
                next_status=status,
                event=f"feedback:{feedback_type}",
            )
            recorded = self.record_state_transition(
                transition,
                entity_id=str(game_id),
                trace_id=payload.get("trace_id"),
                metadata={"feedback_type": feedback_type, "final_reason": final_reason},
                stamp=stamp,
            )
            transitions.append(recorded)
            self.conn.execute(
                """
                UPDATE trial_games
                SET status = ?, updated_at = ?, archived_at = ?, final_reason = ?
                WHERE id = ?
                """,
                (status, stamp, stamp, final_reason, game_id),
            )

        if customer_id:
            self._update_customer_stats(customer_id, feedback_type, stamp)
            profile_note = str(payload.get("profile_note") or notes or "").strip()
            if profile_note:
                self._append_customer_note(str(customer_id), profile_note, stamp)

        self.conn.execute(
            """
            INSERT INTO feedback (created_at, game_id, outbox_id, customer_id, feedback_type, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (stamp, game_id, outbox_id, customer_id, feedback_type, notes),
        )
        auto_success = self._auto_mark_success_if_full(str(game_id), stamp) if game_id else None
        if auto_success and isinstance(auto_success.get("state_transition"), dict):
            transitions.append(auto_success["state_transition"])
        self.conn.commit()
        self.run_lifecycle(now=stamp_dt)
        return {
            "ok": True,
            "auto_success": auto_success,
            "state_machine_version": STATE_MACHINE_VERSION,
            "state_transitions": transitions,
        }

    def _auto_mark_success_if_full(self, game_id: str, stamp: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM trial_games WHERE id = ?", (game_id,)).fetchone()
        if not row:
            return None
        game = self._game_from_row(row)
        if str(game.get("status") or "") in FINAL_GAME_STATUSES:
            return None
        parsed = game.get("parsed") if isinstance(game.get("parsed"), dict) else {}
        missing_count = parsed.get("missing_count")
        if missing_count is None:
            return None
        confirmed = self._confirmed_count(game)
        try:
            target = int(missing_count)
        except (TypeError, ValueError):
            return None
        if target <= 0 or confirmed < target:
            return None
        summary = parsed.get("summary") or game.get("source_text") or game_id
        final_reason = f"{summary} 已确认 {confirmed} 人，缺口已补齐，系统自动标记成局。"
        transition = require_state_transition(
            entity_type="game",
            current_status=str(game.get("status") or ""),
            next_status="已成局",
            event="auto_mark_success_if_full",
        )
        recorded = self.record_state_transition(
            transition,
            entity_id=game_id,
            metadata={"confirmed_count": confirmed, "missing_count": target, "final_reason": final_reason},
            stamp=stamp,
        )
        self.conn.execute(
            """
            UPDATE trial_games
            SET status = '已成局', updated_at = ?, archived_at = ?, final_reason = ?
            WHERE id = ? AND status NOT IN ('已成局', '已取消')
            """,
            (stamp, stamp, final_reason, game_id),
        )
        self.conn.execute(
            """
            INSERT INTO feedback (created_at, game_id, outbox_id, customer_id, feedback_type, notes)
            VALUES (?, ?, NULL, NULL, 'game_auto_success', ?)
            """,
            (stamp, game_id, final_reason),
        )
        return {
            "game_id": game_id,
            "status": "已成局",
            "final_reason": final_reason,
            "state_transition": recorded,
        }

    def _manual_cancel_reason(self, game_id: str) -> str:
        row = self.conn.execute("SELECT * FROM trial_games WHERE id = ?", (game_id,)).fetchone()
        if not row:
            return "老板标记局取消。"
        game = self._game_from_row(row)
        parsed = game.get("parsed") if isinstance(game.get("parsed"), dict) else {}
        summary = parsed.get("summary") or game.get("source_text") or game_id
        outbox = game.get("outbox") or []
        if outbox:
            counts: dict[str, int] = {}
            for item in outbox:
                status = str(item.get("status") or "")
                counts[status] = counts.get(status, 0) + 1
            status_text = "，".join(f"{status}{count}人" for status, count in sorted(counts.items()))
            return f"{summary} 老板标记局取消；邀约反馈：{status_text}。"
        return f"{summary} 老板标记局取消。"

    def _append_customer_note(self, customer_id: str, note: str, stamp: str) -> None:
        customer = self.customer(customer_id)
        if not customer:
            return
        clean = re.sub(r"\s+", " ", note).strip()
        if not clean:
            return
        existing = str(customer.get("notes") or "")
        entry = f"{stamp[:10]}反馈：{clean}"
        if entry in existing:
            return
        updated = f"{existing}；{entry}" if existing else entry
        self.conn.execute(
            "UPDATE customers SET notes = ? WHERE id = ?",
            (updated[-800:], customer_id),
        )

    def recap(self) -> dict[str, Any]:
        today = now_tz().date().isoformat()
        games = self.conn.execute(
            "SELECT status, COUNT(*) count FROM trial_games WHERE substr(created_at, 1, 10) = ? GROUP BY status",
            (today,),
        ).fetchall()
        outbox = self.conn.execute(
            "SELECT status, COUNT(*) count FROM outbox WHERE substr(created_at, 1, 10) = ? GROUP BY status",
            (today,),
        ).fetchall()
        top_customers = self.conn.execute(
            """
            SELECT display_name, response_rate, last_invited_at
            FROM customers
            WHERE no_contact = 0
            ORDER BY response_rate DESC, display_name
            LIMIT 5
            """
        ).fetchall()
        over_invited = self.conn.execute(
            """
            SELECT display_name, invite_count, last_invited_at
            FROM customers
            WHERE last_invited_at IS NOT NULL
            ORDER BY last_invited_at DESC
            LIMIT 5
            """
        ).fetchall()
        return {
            "games_by_status": {row["status"]: row["count"] for row in games},
            "outbox_by_status": {row["status"]: row["count"] for row in outbox},
            "top_customers": [dict(row) for row in top_customers],
            "recent_invited": [dict(row) for row in over_invited],
            "suggestions": self._recap_suggestions(over_invited),
        }

    def _recap_suggestions(self, rows: list[sqlite3.Row]) -> list[str]:
        suggestions: list[str] = []
        for row in rows:
            last_invited = parse_dt(row["last_invited_at"])
            if not last_invited:
                continue
            hours = (now_tz() - last_invited).total_seconds() / 3600
            if hours < 24 and row["invite_count"] >= 3:
                suggestions.append(f"{row['display_name']} 最近邀约较多，明天少打扰。")
        if not suggestions:
            suggestions.append("今天可以继续观察哪些客户回复快，优先补全画像。")
        return suggestions

    def _update_customer_stats(self, customer_id: str, feedback_type: str, stamp: str) -> None:
        customer = self.customer(customer_id)
        if not customer:
            return
        invite_delta = 1 if feedback_type in {"copied", "sent"} else 0
        response_delta = 1 if feedback_type in {"accepted", "arrived", "declined", "ask_later", "candidate_question", "candidate_negotiation"} else 0
        arrival_delta = 1 if feedback_type == "arrived" else 0
        no_contact = 1 if feedback_type == "do_not_disturb" else int(customer["no_contact"])
        last_invited_at = stamp if feedback_type in {"copied", "sent"} else customer["last_invited_at"]
        last_arrived_at = stamp if feedback_type == "arrived" else customer["last_arrived_at"]
        self.conn.execute(
            """
            UPDATE customers
            SET invite_count = invite_count + ?,
                response_count = response_count + ?,
                arrival_count = arrival_count + ?,
                response_rate = CASE
                    WHEN invite_count + ? <= 0 THEN response_rate
                    ELSE MIN(1.0, MAX(0.0, CAST(response_count + ? AS REAL) / CAST(invite_count + ? AS REAL)))
                END,
                last_invited_at = ?,
                last_arrived_at = ?,
                no_contact = ?,
                fatigue_score = CASE
                    WHEN ? IN ('copied','sent') THEN MIN(100, fatigue_score + 10)
                    WHEN ? IN ('arrived','accepted') THEN MAX(0, fatigue_score - 8)
                    ELSE fatigue_score
                END
            WHERE id = ?
            """,
            (
                invite_delta,
                response_delta,
                arrival_delta,
                invite_delta,
                response_delta,
                invite_delta,
                last_invited_at,
                last_arrived_at,
                no_contact,
                feedback_type,
                feedback_type,
                customer_id,
            ),
        )

    def _outbox_status_for_feedback(self, feedback_type: str) -> str:
        return {
            "copied": "已复制",
            "sent": "已发送",
            "accepted": "已确认",
            "arrived": "已到店",
            "declined": "拒绝",
            "no_reply": "未回复",
            "ask_later": "下次再问",
            "candidate_question": "待确认",
            "candidate_negotiation": "待协商",
            "do_not_disturb": "别再打扰",
        }.get(feedback_type, feedback_type or "已反馈")

    def _customer_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "display_name": row["display_name"],
            "contact": row["contact"],
            "preferred_games": json_loads(row["preferred_games"], []),
            "preferred_levels": json_loads(row["preferred_levels"], []),
            "usual_start_hours": json_loads(row["usual_start_hours"], []),
            "gender": normalize_gender(row["gender"]),
            "gender_label": GENDER_LABELS.get(normalize_gender(row["gender"]), "未知"),
            "smoke_preference": row["smoke_preference"],
            "response_speed": row["response_speed"],
            "response_rate": row["response_rate"],
            "last_invited_at": row["last_invited_at"],
            "last_arrived_at": row["last_arrived_at"],
            "invite_count": row["invite_count"],
            "response_count": row["response_count"],
            "arrival_count": row["arrival_count"],
            "fatigue_score": row["fatigue_score"],
            "no_contact": bool(row["no_contact"]),
            "notes": row["notes"],
            "usual_party_size": row["usual_party_size"],
            "usual_party_size_confidence": row["usual_party_size_confidence"],
        }

    def _game_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        game = {
            "id": row["id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "archived_at": row["archived_at"],
            "final_reason": row["final_reason"],
            "status": row["status"],
            "organizer_id": row["organizer_id"],
            "organizer_name": row["organizer_name"],
            "source_text": row["source_text"],
            "parsed": json_loads(row["parsed_json"], {}),
            "reply_text": row["reply_text"],
            "missing_fields": json_loads(row["missing_fields"], []),
            "notes": json_loads(row["notes"], []),
        }
        game["outbox"] = self.outbox_for_game(row["id"])
        game["followups"] = self.followups_for_game(row["id"])
        confirmed_count = self._confirmed_count(game)
        remaining_missing_count = self._remaining_missing_count(game, confirmed_count)
        game["confirmed_count"] = confirmed_count
        game["remaining_missing_count"] = remaining_missing_count
        game["active_player_count"] = self._active_player_count(game, confirmed_count)
        game["live_summary"] = self._live_game_summary(game)
        if isinstance(game.get("parsed"), dict):
            game["parsed"]["confirmed_count"] = confirmed_count
            game["parsed"]["remaining_missing_count"] = remaining_missing_count
            game["parsed"]["active_player_count"] = game["active_player_count"]
            game["parsed"]["live_summary"] = game["live_summary"]
        game["participants"] = self._participants_for_game(game)
        return game

    def _remaining_missing_count(self, game: dict[str, Any], confirmed_count: int) -> int | None:
        parsed = game.get("parsed") if isinstance(game.get("parsed"), dict) else {}
        missing_count = parsed.get("missing_count")
        if missing_count is None:
            return None
        try:
            return max(0, int(missing_count) - confirmed_count)
        except (TypeError, ValueError):
            return None

    def _active_player_count(self, game: dict[str, Any], confirmed_count: int) -> int | None:
        parsed = game.get("parsed") if isinstance(game.get("parsed"), dict) else {}
        current_count = parsed.get("current_player_count")
        if current_count is None:
            return None
        try:
            return min(4, int(current_count) + confirmed_count)
        except (TypeError, ValueError):
            return None

    def _live_game_summary(self, game: dict[str, Any]) -> str:
        parsed = game.get("parsed") if isinstance(game.get("parsed"), dict) else {}
        game_label = str(parsed.get("game_label") or GAME_TYPE_LABELS.get(str(parsed.get("game_type") or ""), "")).strip()
        level = str(parsed.get("level") or "").strip()
        level_text = f"{level}档" if level and not level.endswith("档") else level
        start_time = str(parsed.get("start_time") or "").strip()
        remaining = game.get("remaining_missing_count")
        if isinstance(remaining, int):
            missing_text = f"缺{remaining}" if remaining > 0 else "人齐"
        else:
            missing_text = ""
        rules = [
            str(rule).strip()
            for rule in parsed.get("rules") or []
            if str(rule).strip()
            and str(rule).strip() not in {game_label, "杭麻", "川麻", "麻将"}
        ]
        parts = [game_label, level_text, start_time, missing_text, *rules]
        summary = " ".join(part for part in parts if part)
        return summary or str(parsed.get("summary") or game.get("source_text") or game.get("id") or "")

    def _participants_for_game(self, game: dict[str, Any]) -> list[dict[str, Any]]:
        parsed = game.get("parsed") if isinstance(game.get("parsed"), dict) else {}
        current_count = parsed.get("current_player_count")
        participants = [
            {
                "customer_id": game.get("organizer_id"),
                "customer_name": game.get("organizer_name"),
                "role": "发起人",
                "status": "已在局内",
                "count": 1,
            }
        ]
        if isinstance(current_count, int) and current_count > 1:
            participants.append(
                {
                    "customer_id": None,
                    "customer_name": f"{game.get('organizer_name') or '发起人'}同行未留名",
                    "role": "同行",
                    "status": "未留名",
                    "count": current_count - 1,
                }
            )
        for item in game.get("outbox") or []:
            if str(item.get("status") or "") not in {"已确认", "已到店"}:
                continue
            participants.append(
                {
                    "customer_id": item.get("customer_id"),
                    "customer_name": item.get("customer_name"),
                    "role": "候选人",
                    "status": item.get("status"),
                    "count": 1,
                }
            )
        return participants

    def _outbox_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        customer = self.customer(str(row["customer_id"] or ""))
        gender = normalize_gender((customer or {}).get("gender"))
        approval = self.approval_for_target("outbox", row["id"])
        return {
            "id": row["id"],
            "game_id": row["game_id"],
            "customer_id": row["customer_id"],
            "customer_name": row["customer_name"],
            "gender": gender,
            "gender_label": GENDER_LABELS.get(gender, "未知"),
            "message_text": row["message_text"],
            "status": row["status"],
            "approval_status": approval_status_label((approval or {}).get("status") or row["status"]),
            "approval": approval,
            "score": row["score"],
            "reasons": json_loads(row["reasons"], []),
            "warnings": json_loads(row["warnings"], []),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "conversation": self._candidate_conversation_for_outbox(row["id"]),
            "deliveries": self.delivery_attempts_for_outbox(row["id"]),
        }

    def _followup_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        approval = self.approval_for_target("followup", row["id"])
        return {
            "id": row["id"],
            "game_id": row["game_id"],
            "related_outbox_id": row["related_outbox_id"],
            "recipient_id": row["recipient_id"],
            "recipient_name": row["recipient_name"],
            "message_text": row["message_text"],
            "status": row["status"],
            "approval_status": approval_status_label((approval or {}).get("status") or row["status"]),
            "approval": approval,
            "reason": row["reason"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _approval_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "target_type": row["target_type"],
            "target_id": row["target_id"],
            "action_id": row["action_id"],
            "idempotency_key": row["idempotency_key"],
            "risk_level": row["risk_level"],
            "status": row["status"],
            "reviewer_id": row["reviewer_id"],
            "reviewer_name": row["reviewer_name"],
            "decision_reason": row["decision_reason"],
            "original_message_text": row["original_message_text"],
            "final_message_text": row["final_message_text"],
            "metadata": json_loads(row["metadata_json"], {}),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "decided_at": row["decided_at"],
        }

    def _trace_event_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "trace_id": row["trace_id"],
            "created_at": row["created_at"],
            "level": row["level"],
            "direction": row["direction"],
            "event": row["event"],
            "stage": row["stage"],
            "schema_version": row["schema_version"],
            "payload": json_loads(row["payload_json"], {}),
            "content": row["content"],
        }

    def _state_transition_event_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "created_at": row["created_at"],
            "entity_type": row["entity_type"],
            "entity_id": row["entity_id"],
            "from_status": row["from_status"],
            "to_status": row["to_status"],
            "event": row["event"],
            "allowed": bool(row["allowed"]),
            "reason": row["reason"],
            "trace_id": row["trace_id"],
            "action_id": row["action_id"],
            "state_machine_version": row["state_machine_version"],
            "schema_version": row["schema_version"],
            "metadata": json_loads(row["metadata_json"], {}),
        }

    def _delivery_attempt_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "outbox_id": row["outbox_id"],
            "approval_id": row["approval_id"],
            "channel": row["channel"],
            "recipient_id": row["recipient_id"],
            "recipient_name": row["recipient_name"],
            "message_text": row["message_text"],
            "status": row["status"],
            "idempotency_key": row["idempotency_key"],
            "action_id": row["action_id"],
            "trace_id": row["trace_id"],
            "error": row["error"],
            "metadata": json_loads(row["metadata_json"], {}),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "delivered_at": row["delivered_at"],
        }

    def _candidate_conversation_for_outbox(self, outbox_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT created_at, feedback_type, notes
            FROM feedback
            WHERE outbox_id = ?
            ORDER BY id
            """,
            (outbox_id,),
        ).fetchall()
        conversation: list[dict[str, Any]] = []
        for row in rows:
            notes = str(row["notes"] or "")
            candidate_text = ""
            boss_reply = ""
            classification: dict[str, Any] = {}
            payload = json_loads(notes, {})
            if isinstance(payload, dict) and payload.get("kind") == "candidate_message":
                candidate_text = str(payload.get("candidate_text") or "")
                boss_reply = str(payload.get("boss_reply") or "")
                raw_classification = payload.get("classification")
                classification = raw_classification if isinstance(raw_classification, dict) else {}
            else:
                match = re.search(r"候选人回复：(.*?)；系统建议老板回复：(.*)", notes)
                if match:
                    candidate_text = match.group(1).strip()
                    boss_reply = match.group(2).strip()
            if not candidate_text and not boss_reply:
                continue
            conversation.append(
                {
                    "created_at": row["created_at"],
                    "feedback_type": row["feedback_type"],
                    "status": self._outbox_status_for_feedback(str(row["feedback_type"] or "")),
                    "candidate_text": candidate_text,
                    "boss_reply": boss_reply,
                    "classification": classification,
                }
            )
        return conversation

    def _safe_id(self, value: str) -> str:
        value = value.strip().lower()
        value = re.sub(r"[^a-z0-9_\u4e00-\u9fa5]+", "_", value)
        return value.strip("_") or f"customer_{int(datetime.now().timestamp())}"


class BossTrialService:
    def __init__(
        self,
        store: TrialStore,
        cache: RedisCache | None = None,
        cache_prefix: str = CACHE_PREFIX,
    ) -> None:
        self.store = store
        self.cache = cache
        self.cache_prefix = cache_prefix or "mahjong:trial"
        self._local_cache: dict[str, tuple[float, Any]] = {}
        self.composer = MessageComposer()
        self.llm_config = LLMConfig.from_env()
        self.llm_budget_manager = LLMBudgetManager.from_env() if self.llm_config else None
        llm_resolver = (
            OpenAICompatibleLLMResolver(
                self.llm_config,
                budget_manager=self.llm_budget_manager,
                audit_logger=write_llm_audit_log,
            )
            if self.llm_config and self.llm_budget_manager
            else None
        )
        self.responder = AgentResponder(invite_limit=8, llm_resolver=llm_resolver)
        self.followup_context_builder = TrialWorkflowFollowupContextBuilder(
            parse_datetime=parse_dt,
            text_normalizer=self._normalize_pool_query_text,
            memory_ttl_seconds=SHORT_MEMORY_TTL_SECONDS,
        )
        self.short_memory_text_merger = TrialShortMemoryTextMerger(
            parse_datetime=parse_dt,
            is_pool_inquiry_text=self._is_pool_inquiry_text,
            is_explicit_grouping_request=self._is_explicit_grouping_request,
            merge_window_seconds=SHORT_MEMORY_MERGE_WINDOW_SECONDS,
            critical_fields=set(CRITICAL_FIELDS),
        )
        self.tool_plan_prompt_builder = TrialToolPlanPromptBuilder()
        self.tool_call_normalizer = TrialToolCallNormalizer()
        self.tool_action_proposal_factory = TrialToolActionProposalFactory(
            protocol_version=CONTROLLED_AGENT_PROTOCOL_VERSION,
            tool_policy=self._tool_policy,
        )
        self.tool_action_validator = TrialToolActionValidator(
            critical_fields=set(CRITICAL_FIELDS),
            tool_spec_for_stage=tool_spec_for_stage,
            tool_specs_for_stage=self._tool_specs_for_stage,
            runtime_policy_getter=self.store.runtime_policy,
            runtime_policy_validation_override=self._runtime_policy_validation_override,
            trusted_action_proposer=trusted_action_proposer,
        )
        self.trial_tool_gateway = TrialToolGateway(
            validated_action_lookup=self._validated_tool_action_record,
            action_executor=self._execute_controlled_action,
        )
        self.trial_tool_request_factory = TrialToolRequestFactory()
        self.trial_tool_orchestration_service = TrialToolOrchestrationService(
            callbacks=TrialToolOrchestrationCallbacks(
                llm_tool_plan=self._llm_tool_plan,
                action_plan_view=self._action_plan_view,
                single_action_plan_view=self._single_action_plan_view,
                tool_requested=self._tool_requested,
                replace_action_plan_view=self._replace_action_plan_view,
                search_current_open_games_tool=self._search_current_open_games_tool,
                has_start_time_ambiguity=self._has_start_time_ambiguity,
                is_explicit_grouping_request=self._is_explicit_grouping_request,
                user_semantic_action_record=self._user_semantic_action_record,
                is_grouping_confirmation_followup=self._is_grouping_confirmation_followup,
                stable_request_game_id=self._stable_request_game_id,
                should_search_existing_pool=self._should_search_existing_pool,
                skipped_tool_result=self._skipped_tool_result,
                rejected_tool_result=self._rejected_tool_result,
                search_candidate_customers_tool=self._search_candidate_customers_tool,
                candidate_recommendations_from_tool=self._candidate_recommendations_from_tool,
                send_message_tool=self._send_message_tool,
            ),
            critical_fields=set(CRITICAL_FIELDS),
        )
        self.trial_game_state_creation_adapter = TrialGameStateCreationAdapter(
            callbacks=TrialGameStateCreationCallbacks(
                game_status_label=self._game_status_label,
                workflow_state_action_record=self._workflow_state_action_record,
                execute_controlled_action=self._execute_controlled_action,
                create_game_state_write=self._create_game_state_write,
                compact_action_record=self._compact_action_record,
                cache_game=self._cache_game,
                single_action_plan_view=self._single_action_plan_view,
            )
        )
        self.trial_reply_draft_adapter = TrialReplyDraftAdapter(
            callbacks=TrialReplyDraftCallbacks(
                suggested_reply=self._suggested_reply,
                update_sender_memory_after_reply=self._update_sender_memory_after_reply,
            )
        )
        self.trial_reply_rule_policy = TrialReplyRulePolicy(
            callbacks=TrialReplyRulePolicyCallbacks(
                pool_match_reply=self._pool_match_reply,
                follow_up_text=self._follow_up_text,
                should_search_existing_pool=self._should_search_existing_pool,
                is_explicit_grouping_request=self._is_explicit_grouping_request,
                pool_no_match_reply=self._pool_no_match_reply,
                brief_ack_reply=self._brief_ack_reply,
            )
        )
        self.controlled_runtime = build_controlled_runtime(
            core=self.responder.core,
            config=ControlledRuntimeConfig(
                trace_jsonl_path=ROOT / "logs" / "controlled_workflow_trace.jsonl",
                short_memory_ttl_seconds=SHORT_MEMORY_TTL_SECONDS,
                short_memory_max_records=20,
                fail_closed_without_llm=True,
            ),
        )
        self.reload_customers()

    def reload_customers(self) -> None:
        self.responder.core.store.customers.clear()
        for customer in self.store.customers():
            self.responder.core.upsert_customer(self._profile_from_customer(customer))

    def state(self, *, now: datetime | None = None) -> dict[str, Any]:
        archived = self.store.run_lifecycle(now=now)
        state = {
            "customers": self.store.customers(),
            "games": self.store.games(),
            "recent_archived_games": self.store.archived_games(),
            "recent_outbox": self.store.recent_outbox(),
            "recent_followups": self.store.recent_followups(),
            "recent_approvals": self.store.recent_approvals(),
            "recent_controlled_actions": self.store.controlled_actions(),
            "recent_state_transitions": self.store.state_transition_events(limit=80),
            "recent_delivery_attempts": self.store.recent_delivery_attempts(),
            "runtime_policy": self.store.runtime_policy(),
            "recap": self.store.recap(),
            "cache": self._cache_status(),
            "evals": self.eval_overview(),
            "lifecycle": {"archived_count": len(archived), "archived_game_ids": [item.get("id") for item in archived]},
        }
        self._cache_set("state", state, ttl_seconds=STATE_CACHE_TTL_SECONDS)
        return state

    def trace_view(self, trace_id: str | None = None, *, limit: int = 300) -> dict[str, Any]:
        safe_trace_id = str(trace_id or "").strip()
        if safe_trace_id:
            events = self.store.trace_events(safe_trace_id, limit=limit)
            return {
                "trace_id": safe_trace_id,
                "schema_version": TRACE_EVENT_SCHEMA_VERSION,
                "event_count": len(events),
                "events": events,
            }
        traces = self.store.recent_traces(limit=min(max(int(limit or 40), 1), 200))
        return {
            "schema_version": TRACE_EVENT_SCHEMA_VERSION,
            "trace_count": len(traces),
            "traces": traces,
        }

    def state_transition_view(
        self,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
        trace_id: str | None = None,
        limit: int = 120,
    ) -> dict[str, Any]:
        events = self.store.state_transition_events(
            entity_type=entity_type,
            entity_id=entity_id,
            trace_id=trace_id,
            limit=limit,
        )
        return {
            "schema_version": STATE_TRANSITION_EVENT_SCHEMA_VERSION,
            "event_count": len(events),
            "events": events,
        }

    def runtime_policy(self) -> dict[str, Any]:
        return {
            "ok": True,
            "policy": self.store.runtime_policy(),
        }

    def update_runtime_policy(self, payload: dict[str, Any]) -> dict[str, Any]:
        trace_id = str(payload.get("trace_id") or make_trace_id())
        now = parse_dt(payload.get("now")) or now_tz()
        action = self._workflow_state_action_record(
            trace_id=trace_id,
            stage="runtime_policy",
            action_name="update_runtime_policy",
            arguments={
                key: payload.get(key)
                for key in [
                    "controlled_agent_mode",
                    "read_only_mode",
                    "state_writes_enabled",
                    "delivery_enabled",
                    "approval_enabled",
                    "eval_writes_enabled",
                    "llm_required_for_side_effect_tools",
                    "llm_required_for_state_writes",
                    "reason",
                ]
                if key in payload
            },
            proposed_by="boss_manual",
            source="runtime_policy_console",
            risk_level="high",
            approval_required=True,
            reason="老板更新运行时安全策略，后端记录为受控配置变更。",
            now=now,
            validation={
                "allowed": True,
                "code": "runtime_policy_update_allowed",
                "reason": "运行时策略更新允许执行；只读模式也必须允许关闭安全开关或恢复服务。",
                "notes": ["策略会影响后续副作用动作门禁。"],
            },
        )
        result = self._execute_controlled_action(
            action,
            lambda: {"ok": True, "policy": self.store.update_runtime_policy({**payload, "now": now.isoformat()})},
        )
        result["agent_actions"] = [
            self._single_action_plan_view(stage="runtime_policy", source="runtime_policy_console", action=action)
        ]
        return result

    def _stable_request_game_id(self, trace_id: str) -> str:
        digest = hashlib.sha256(f"trial-game:{trace_id}".encode("utf-8")).hexdigest()[:12]
        return f"game_{digest}"

    def analyze_controlled(self, payload: dict[str, Any]) -> dict[str, Any]:
        return TrialControlledEntryAdapter(
            workflow_service=self.controlled_runtime.service,
            response_adapter=TrialControlledResponseAdapter(
                persistence_adapter=TrialControlledPersistenceAdapter(
                    store=self.store,
                    action_record_factory=self._workflow_state_action_record,
                    action_executor=self._execute_controlled_action,
                    action_plan_projector=self._single_action_plan_view,
                    game_lookup=self._game_by_id,
                    approval_status_labeler=approval_status_label,
                )
            ),
            request_builder=TrialControlledRequestBuilder(
                trace_id_factory=make_trace_id,
                now_factory=now_tz,
                parse_datetime=parse_dt,
            ),
            customer_reloader=self.reload_customers,
            lifecycle_runner=lambda now: self.store.run_lifecycle(now=now),
        ).analyze(payload)

    def save_customer(self, payload: dict[str, Any]) -> dict[str, Any]:
        trace_id = str(payload.get("trace_id") or make_trace_id())
        now = parse_dt(payload.get("now")) or now_tz()
        action = self._workflow_state_action_record(
            trace_id=trace_id,
            stage="profile_update",
            action_name="upsert_customer_profile",
            arguments={
                "customer_id": payload.get("id") or payload.get("customer_id"),
                "display_name": payload.get("display_name") or payload.get("name"),
                "fields": sorted(str(key) for key in payload.keys() if key not in {"trace_id", "now"}),
            },
            proposed_by="boss_manual",
            source="boss_manual",
            risk_level="medium",
            approval_required=True,
            reason="老板手动维护客户画像，后端记录为受控状态写入。",
            now=now,
            validation={
                "allowed": True,
                "code": "manual_approved",
                "reason": "老板手动提交画像更新，视为已审批。",
                "notes": ["客户画像会影响后续候选人推荐，需要进入审计日志。"],
            },
        )
        customer = self._execute_controlled_action(action, lambda: self.store.upsert_customer(payload))
        if not customer.get("rejected"):
            self.reload_customers()
            if customer.get("id"):
                self._cache_set(f"customer:{customer['id']}", customer, ttl_seconds=GAME_CACHE_TTL_SECONDS)
        customer["agent_actions"] = [
            self._single_action_plan_view(stage="profile_update", source="boss_manual", action=action)
        ]
        return customer

    def analyze(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.reload_customers()
        text = str(payload.get("text") or "").strip()
        if not text:
            raise ValueError("消息不能为空")
        trace_id = str(payload.get("trace_id") or make_trace_id())
        sender_id = str(payload.get("sender_id") or "trial_customer")
        sender_name = str(payload.get("sender_name") or "试用客户")
        conversation_id = str(
            payload.get("conversation_id")
            or payload.get("conversationId")
            or "boss_trial"
        ).strip() or "boss_trial"
        now = parse_dt(payload.get("now")) or now_tz()
        self.store.run_lifecycle(now=now)
        sender_memory = self._sender_memory(conversation_id, sender_id, now)
        workflow_followup_context = self._workflow_followup_context(sender_memory, text, now)
        effective_text = self._effective_text(sender_memory, text, now)
        pool_inquiry = self._is_pool_inquiry_text(text) and not self._is_explicit_grouping_request(text, text, None)
        message = Message(
            text=effective_text,
            sender_id=sender_id,
            sender_name=sender_name,
            channel_id=conversation_id,
            channel_type=ChannelType.MANUAL,
            sent_at=now,
            metadata={
                "conversation_id": conversation_id,
                "trace_id": trace_id,
                "pool_inquiry_detected": pool_inquiry,
                "workflow_followup_context": workflow_followup_context,
            },
        )
        decision = self.responder.respond(message, now=now)
        if effective_text != text:
            decision.notes.append("已合并 Redis 短期记忆中的同一客户近期碎片消息")
        game = self.responder.core.store.games.get(decision.game_id) if decision.game_id else None
        if game is None:
            game = self._materialize_contextual_game_from_llm_slots(
                trace_id=trace_id,
                decision=decision,
                workflow_followup_context=workflow_followup_context,
                sender_id=sender_id,
                sender_name=sender_name,
                conversation_id=conversation_id,
                source_message_id=message.id,
                now=now,
            )
        else:
            self._merge_workflow_context_into_game(
                trace_id=trace_id,
                game=game,
                decision=decision,
                workflow_followup_context=workflow_followup_context,
            )
        if game:
            self._apply_trial_inferences(
                game,
                effective_text,
                sender_id,
                now=now,
                source_text=text,
                sender_memory=sender_memory,
            )

        missing_fields = self._missing_fields(game, decision)
        tool_orchestration = self.trial_tool_orchestration_service.run(
            TrialToolOrchestrationInput(
                trace_id=trace_id,
                sender_id=sender_id,
                sender_name=sender_name,
                source_text=text,
                effective_text=effective_text,
                workflow_followup_context=workflow_followup_context,
                decision=decision,
                game=game,
                missing_fields=missing_fields,
                decision_action=decision.action.value,
                pool_inquiry=pool_inquiry,
                now=now,
            )
        )
        action_plans = tool_orchestration.action_plans
        pool_tool_result = tool_orchestration.pool_tool_result
        candidate_tool_result = tool_orchestration.candidate_tool_result
        send_tool_result = tool_orchestration.send_tool_result
        tool_results = tool_orchestration.tool_results
        pool_matches = tool_orchestration.pool_matches
        use_existing_pool = tool_orchestration.use_existing_pool
        user_action_record = tool_orchestration.user_action_record
        user_action_validation = tool_orchestration.user_action_validation
        effective_user_action = tool_orchestration.effective_user_action
        should_materialize_game = tool_orchestration.should_materialize_game
        inquiry_without_materialized_game = tool_orchestration.inquiry_without_materialized_game
        response_missing_fields = tool_orchestration.response_missing_fields
        recommendations = tool_orchestration.recommendations
        outbox = tool_orchestration.outbox

        parsed = self._game_to_dict(game) if game else {}
        parsed["semantic_action"] = {
            "source": user_action_record.get("source"),
            "proposed_action": (user_action_record.get("arguments") or {}).get("proposed_action"),
            "effective_action": user_action_validation.get("effective_action"),
            "confidence": (user_action_record.get("arguments") or {}).get("confidence"),
            "validation": {
                "allowed": user_action_validation.get("allowed"),
                "code": user_action_validation.get("code"),
                "reason": user_action_validation.get("reason"),
                "notes": user_action_validation.get("notes") or [],
            },
        }
        if pool_matches:
            parsed["level_options"] = self._level_options_from_query(game, effective_text, sender_id)
            parsed["smoke_options"] = self._smoke_options_from_query(game, effective_text)
            parsed["tool_decision"] = {
                "tool_name": pool_tool_result.get("tool_name"),
                "called": pool_tool_result.get("called"),
                "call_reason": pool_tool_result.get("call_reason"),
                "result_count": pool_tool_result.get("result_count"),
            }
        effective_action = (
            "match_existing_game"
            if use_existing_pool
            else "inquire_existing_game"
            if inquiry_without_materialized_game
            else self._effective_intent_action(decision.action.value, game, missing_fields, outbox)
        )
        parsed["intent_action"] = effective_action
        parsed["user_intent"] = self._user_intent_label(effective_action)

        self._remember_sender(
            sender_id=sender_id,
            sender_name=sender_name,
            conversation_id=conversation_id,
            text=text,
            effective_text=effective_text,
            parsed=parsed,
            missing_fields=response_missing_fields,
            decision=decision.to_dict(),
            game_id=pool_matches[0]["game_id"] if use_existing_pool else (game.id if should_materialize_game else None),
            trace_id=trace_id,
            now=now,
        )

        reply_draft = self.trial_reply_draft_adapter.draft(
            TrialReplyDraftInput(
                conversation_id=conversation_id,
                sender_id=sender_id,
                sender_name=sender_name,
                source_text=text,
                effective_text=effective_text,
                trace_id=trace_id,
                game=game,
                workflow_followup_context=workflow_followup_context,
                missing_fields=response_missing_fields,
                decision_reply=decision.reply_text,
                parsed=parsed,
                recommendations=recommendations,
                outbox=outbox,
                pool_matches=pool_matches,
                tool_results=tool_results,
                now=now,
            )
        )
        suggested_reply = reply_draft.suggested_reply
        if should_materialize_game:
            create_game_state = self.trial_game_state_creation_adapter.create(
                TrialCreateGameStateInput(
                    trace_id=trace_id,
                    game=game,
                    sender_id=sender_id,
                    sender_name=sender_name,
                    source_text=text,
                    parsed=parsed,
                    suggested_reply=suggested_reply,
                    fallback_reply_text=decision.reply_text,
                    missing_fields=missing_fields,
                    decision_notes=list(decision.notes),
                    user_action_record=user_action_record,
                    effective_user_action=effective_user_action,
                    outbox=outbox,
                    now=now,
                )
            )
            action_plans.append(create_game_state.action_plan)
        elif game and (use_existing_pool or not should_materialize_game):
            self.responder.core.store.games.pop(game.id, None)

        return {
            "trace_id": trace_id,
            "conversation_id": conversation_id,
            "sender_id": sender_id,
            "sender_name": sender_name,
            "source_text": text,
            "decision": decision.to_dict(),
            "parsed": parsed,
            "missing_fields": response_missing_fields,
            "follow_up": self._follow_up_text(response_missing_fields, decision.reply_text, sender_id=sender_id, game=game),
            "suggested_reply": suggested_reply,
            "group_draft": "" if not should_materialize_game else (decision.draft_group_post or (self.composer.group_post(game) if game else "")),
            "candidates": [self._candidate_to_dict(item) for item in recommendations],
            "outbox": outbox,
            "pool_matches": pool_matches,
            "agent_actions": action_plans,
            "tool_results": self._tool_results_for_prompt(tool_results),
            "used_short_memory": effective_text != text,
            "effective_text": effective_text,
            "state": self.state(now=now),
        }

    def _materialize_contextual_game_from_llm_slots(
        self,
        *,
        trace_id: str,
        decision: Any,
        workflow_followup_context: dict[str, Any],
        sender_id: str,
        sender_name: str,
        conversation_id: str,
        source_message_id: str,
        now: datetime,
    ) -> GameRequest | None:
        semantic = getattr(decision, "semantic_proposal", None)
        if not isinstance(semantic, dict) or semantic.get("source") != "llm":
            return None
        proposed_action = self._normalize_user_semantic_action(semantic.get("proposed_action"))
        confidence = self._safe_float(semantic.get("confidence")) or 0.0
        if proposed_action != "create_game" or confidence < 0.72:
            return None
        if not isinstance(workflow_followup_context, dict) or not workflow_followup_context:
            return None
        previous_reply = str(workflow_followup_context.get("previous_system_suggested_reply") or "").strip()
        if not previous_reply:
            return None
        previous_game = workflow_followup_context.get("previous_game")
        if not isinstance(previous_game, dict):
            previous_game = {}
        slots = semantic.get("slots") if isinstance(semantic.get("slots"), dict) else {}
        if not previous_game and not slots:
            return None

        rules = self._unique_strings([str(item) for item in previous_game.get("rules") or []])
        play_options = self._unique_strings([str(item) for item in previous_game.get("play_options") or []])
        notes = self._unique_strings([str(item) for item in previous_game.get("notes") or []])
        ambiguities = self._unique_strings([str(item) for item in previous_game.get("ambiguities") or []])
        notes.append("LLM 语义提案结合上一轮组局上下文，物化为待后端校验的局对象")

        game_type = str(previous_game.get("game_type") or "mahjong")
        ruleset = previous_game.get("ruleset")
        variant = previous_game.get("variant")
        level = previous_game.get("level")
        base_score = self._safe_float(previous_game.get("base_score"))
        cap_score = self._safe_float(previous_game.get("cap_score"))
        start_at = parse_dt(str(previous_game.get("start_at") or "")) if previous_game.get("start_at") else None
        duration_hours = self._safe_float(previous_game.get("duration_hours"))
        current_player_count = self._safe_int(previous_game.get("current_player_count"))
        missing_count = self._safe_int(previous_game.get("missing_count"))

        slot_game_type = self._semantic_slot_value(slots.get("game_type"))
        if self._semantic_slot_usable(slots.get("game_type"), min_confidence=0.7) and isinstance(slot_game_type, str):
            if slot_game_type and slot_game_type != "unknown":
                game_type = slot_game_type
        if game_type and game_type != "mahjong" and not ruleset:
            ruleset = game_type
        game_label = GAME_TYPE_LABELS.get(game_type, "")
        if game_label and game_label != "麻将":
            rules.append(game_label)

        slot_variant = self._semantic_slot_value(slots.get("variant"))
        if self._semantic_slot_usable(slots.get("variant"), min_confidence=0.7) and isinstance(slot_variant, str):
            if slot_variant and slot_variant != "unknown":
                variant = slot_variant
        variant_label = VARIANT_LABELS.get(str(variant or ""), "")
        if variant_label:
            play_options.append(variant_label)

        slot_level = self._semantic_slot_value(slots.get("level"))
        if self._semantic_slot_usable(slots.get("level"), min_confidence=0.7) and slot_level not in (None, "", "unknown"):
            level = str(slot_level).strip()
        if level:
            base_score = self._safe_float(level) if base_score is None else base_score

        start_time_mode = self._semantic_slot_value(slots.get("start_time_mode"))
        if start_time_mode == "people_ready" and self._semantic_slot_usable(slots.get("start_time_mode"), min_confidence=0.7):
            start_at = None
            rules.append("人齐开")
            play_options = [item for item in play_options if item != "固定时间"]
            ambiguities = [item for item in ambiguities if "上午还是下午" not in item and "已经过了" not in item]
        else:
            slot_start_time = self._semantic_slot_value(slots.get("start_time"))
            if (
                isinstance(slot_start_time, str)
                and re.fullmatch(r"\d{1,2}:\d{2}", slot_start_time)
                and self._semantic_slot_usable(slots.get("start_time"), min_confidence=0.75)
            ):
                hour, minute = [int(part) for part in slot_start_time.split(":", 1)]
                candidate_start = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if candidate_start < now - timedelta(minutes=30):
                    ambiguity = f"{hour}点 已经过了"
                    if ambiguity not in ambiguities:
                        ambiguities.append(ambiguity)
                else:
                    start_at = candidate_start

        duration_mode = self._semantic_slot_value(slots.get("duration_mode"))
        if duration_mode == "overnight" and self._semantic_slot_usable(slots.get("duration_mode"), min_confidence=0.7):
            duration_hours = None
            rules.append("通宵")
        else:
            slot_duration = self._semantic_slot_value(slots.get("duration_hours"))
            slot_duration_float = self._safe_float(slot_duration)
            if slot_duration_float and self._semantic_slot_usable(slots.get("duration_hours"), min_confidence=0.75):
                duration_hours = slot_duration_float
                rules = [item for item in rules if item != "通宵"]

        smoke = self._semantic_slot_value(slots.get("smoke"))
        if self._semantic_slot_usable(slots.get("smoke"), min_confidence=0.7) and isinstance(smoke, str):
            smoke_rule = {
                "any": "烟况都可",
                "no_smoke": "无烟",
                "smoke_ok": "可吸烟",
            }.get(smoke)
            if smoke_rule:
                rules = [item for item in rules if item not in {"无烟", "可吸烟", "烟况都可"}]
                rules.append(smoke_rule)

        slot_known_players = self._semantic_slot_value(slots.get("known_players"))
        slot_known_int = self._safe_int(slot_known_players)
        if slot_known_int and self._semantic_slot_usable(slots.get("known_players"), min_confidence=0.7):
            current_player_count = max(1, min(4, slot_known_int))
        slot_missing = self._semantic_slot_value(slots.get("missing_count"))
        slot_missing_int = self._safe_int(slot_missing)
        if slot_missing_int is not None and self._semantic_slot_usable(slots.get("missing_count"), min_confidence=0.7):
            missing_count = max(0, min(3, slot_missing_int))
        if current_player_count is not None and missing_count is None:
            missing_count = max(0, 4 - current_player_count)
        if missing_count is not None and current_player_count is None:
            current_player_count = max(1, 4 - missing_count)

        game = GameRequest(
            organizer_id=sender_id,
            organizer_name=sender_name,
            channel_id=conversation_id,
            source_message_id=source_message_id,
            status=GameStatus.NEED_CLARIFICATION,
            game_type=game_type or "mahjong",
            ruleset=str(ruleset) if ruleset else None,
            variant=str(variant) if variant else None,
            current_player_count=current_player_count,
            missing_count=missing_count,
            level=str(level) if level else None,
            base_score=base_score,
            cap_score=cap_score,
            start_at=start_at,
            start_time_confidence=0.0 if start_at is None else 0.8,
            duration_hours=duration_hours,
            rules=self._unique_strings(rules),
            play_options=self._unique_strings(play_options),
            notes=self._unique_strings(notes),
            ambiguities=self._unique_strings(ambiguities),
        )
        if not game.ruleset and game.game_type != "mahjong":
            game.ruleset = game.game_type
        if game.status == GameStatus.NEED_CLARIFICATION and not (set(self._missing_fields(game, decision)) & CRITICAL_FIELDS):
            game.status = GameStatus.OPEN
        decision.notes.append("后端已根据 LLM 槽位和上一轮上下文生成待校验局对象")
        write_tool_audit_log(
            trace_id,
            "state_materialization",
            {
                "source": "llm_slots",
                "stage": "contextual_game_materialization",
                "allowed": True,
                "proposed_action": proposed_action,
                "confidence": confidence,
                "previous_missing_fields": workflow_followup_context.get("previous_missing_fields") or [],
                "materialized_game": self._game_to_dict(game),
            },
        )
        return game

    def _merge_workflow_context_into_game(
        self,
        *,
        trace_id: str,
        game: GameRequest,
        decision: Any,
        workflow_followup_context: dict[str, Any],
    ) -> None:
        semantic = getattr(decision, "semantic_proposal", None)
        if not isinstance(semantic, dict) or semantic.get("source") != "llm":
            return
        proposed_action = self._normalize_user_semantic_action(semantic.get("proposed_action"))
        confidence = self._safe_float(semantic.get("confidence")) or 0.0
        if proposed_action != "create_game" or confidence < 0.72:
            return
        if not isinstance(workflow_followup_context, dict) or not workflow_followup_context:
            return
        previous_game = workflow_followup_context.get("previous_game")
        if not isinstance(previous_game, dict) or not previous_game:
            return

        before = self._game_to_dict(game)
        previous_game_type = str(previous_game.get("game_type") or "").strip()
        if game.game_type == "mahjong" and previous_game_type and previous_game_type != "mahjong":
            game.game_type = previous_game_type
        if not game.ruleset and previous_game.get("ruleset"):
            game.ruleset = str(previous_game.get("ruleset"))
        if not game.variant and previous_game.get("variant"):
            game.variant = str(previous_game.get("variant"))
        if game.level is None and previous_game.get("level"):
            game.level = str(previous_game.get("level"))
            game.base_score = self._safe_float(game.level) if game.base_score is None else game.base_score
        if game.current_player_count is None:
            game.current_player_count = self._safe_int(previous_game.get("current_player_count"))
        if game.missing_count is None:
            game.missing_count = self._safe_int(previous_game.get("missing_count"))

        previous_start_mode = str(previous_game.get("start_time_mode") or "").strip()
        if game.start_at is None and not self._has_flexible_start(game):
            if previous_start_mode == "people_ready" or str(previous_game.get("start_time") or "") == "人齐开":
                game.rules.append("人齐开")
            elif previous_game.get("start_at"):
                game.start_at = parse_dt(str(previous_game.get("start_at") or ""))
                if game.start_at:
                    game.start_time_confidence = max(float(game.start_time_confidence or 0.0), 0.8)

        previous_duration_mode = str(previous_game.get("duration_mode") or "").strip()
        if not self._has_duration_strategy(game):
            previous_duration = self._safe_float(previous_game.get("duration_hours"))
            if previous_duration:
                game.duration_hours = previous_duration
            elif previous_duration_mode == "overnight" or str(previous_game.get("duration_text") or "") == "通宵":
                game.rules.append("通宵")

        current_smoke_rules = {"无烟", "可吸烟", "烟况都可"} & set(game.rules or [])
        for rule in [str(item) for item in previous_game.get("rules") or []]:
            if rule in {"无烟", "可吸烟", "烟况都可"} and current_smoke_rules:
                continue
            if rule and rule not in game.rules:
                game.rules.append(rule)
        for option in [str(item) for item in previous_game.get("play_options") or []]:
            if option and option not in game.play_options:
                game.play_options.append(option)
        for ambiguity in [str(item) for item in previous_game.get("ambiguities") or []]:
            if ambiguity and ambiguity not in game.ambiguities:
                game.ambiguities.append(ambiguity)

        game.rules = self._unique_strings(game.rules)
        game.play_options = self._unique_strings(game.play_options)
        game.ambiguities = self._unique_strings(game.ambiguities)
        after = self._game_to_dict(game)
        if before == after:
            return
        decision.notes.append("后端已将上一轮工作流上下文合并进当前局对象")
        write_tool_audit_log(
            trace_id,
            "state_materialization",
            {
                "source": "workflow_context_merge",
                "stage": "contextual_game_merge",
                "allowed": True,
                "proposed_action": proposed_action,
                "confidence": confidence,
                "before": before,
                "after": after,
            },
        )

    def _semantic_slot_value(self, slot: Any) -> Any:
        if isinstance(slot, dict):
            return slot.get("value")
        return slot

    def _semantic_slot_confidence(self, slot: Any) -> float:
        if not isinstance(slot, dict):
            return 0.0
        return self._safe_float(slot.get("confidence")) or 0.0

    def _semantic_slot_source(self, slot: Any) -> str:
        if not isinstance(slot, dict):
            return ""
        return str(slot.get("source") or "").strip().lower()

    def _semantic_slot_usable(self, slot: Any, *, min_confidence: float) -> bool:
        if not isinstance(slot, dict):
            return False
        if bool(slot.get("needs_confirmation")):
            return False
        if self._semantic_slot_confidence(slot) < min_confidence:
            return False
        return self._semantic_slot_source(slot) not in {"", "unknown"}

    def feedback(self, payload: dict[str, Any]) -> dict[str, Any]:
        trace_id = str(payload.get("trace_id") or make_trace_id())
        now = parse_dt(payload.get("now")) or now_tz()
        feedback_type = str(payload.get("feedback_type") or "").strip()
        is_send_bypass = feedback_type == "sent" and bool(payload.get("outbox_id"))
        action = self._workflow_state_action_record(
            trace_id=trace_id,
            stage="manual_feedback",
            action_name="record_manual_feedback",
            arguments={
                "game_id": payload.get("game_id"),
                "outbox_id": payload.get("outbox_id"),
                "customer_id": payload.get("customer_id"),
                "feedback_type": feedback_type,
            },
            proposed_by="boss_manual",
            source="boss_manual",
            risk_level="high" if is_send_bypass else ("medium" if feedback_type in {"accepted", "arrived", "do_not_disturb"} else "low"),
            approval_required=True,
            reason="老板手动标记邀约反馈，后端记录为受控状态写入。",
            now=now,
            validation={
                "allowed": not is_send_bypass,
                "code": "send_requires_delivery_gateway" if is_send_bypass else "manual_approved",
                "reason": "发送动作必须走 /api/send-outbox，不能通过 feedback 直接标记已发送。"
                if is_send_bypass
                else "老板手动反馈候选人状态，视为已审批。",
                "notes": ["发送是高风险动作，需要审批通过、发送幂等和 delivery 账本。"]
                if is_send_bypass
                else ["反馈会更新局状态和客户画像统计。"],
            },
        )
        result = self._execute_controlled_action(action, lambda: self.store.record_feedback(payload))
        if result.get("ok") and not result.get("deduplicated"):
            self.reload_customers()
        game_id = payload.get("game_id")
        if game_id:
            self._cache_existing_game(str(game_id))
        result["agent_actions"] = [
            self._single_action_plan_view(stage="manual_feedback", source="boss_manual", action=action)
        ]
        result["state"] = self.state(now=now)
        return result

    def send_outbox(self, payload: dict[str, Any]) -> dict[str, Any]:
        return TrialOutboxDeliveryAdapter(
            outbox_lookup=self.store.outbox_item,
            delivery_executor=self.store.execute_outbox_delivery,
            action_record_factory=self._workflow_state_action_record,
            action_executor=self._execute_controlled_action,
            action_plan_projector=self._single_action_plan_view,
            state_loader=lambda now: self.state(now=now),
            trace_id_factory=make_trace_id,
            now_factory=now_tz,
            parse_datetime=parse_dt,
            customer_reloader=self.reload_customers,
            game_cache_updater=self._cache_existing_game,
        ).send(payload)

    def approval_decision(self, payload: dict[str, Any]) -> dict[str, Any]:
        return TrialApprovalDecisionAdapter(
            approval_executor=self.store.decide_approval,
            action_record_factory=self._workflow_state_action_record,
            action_executor=self._execute_controlled_action,
            action_plan_projector=self._single_action_plan_view,
            state_loader=lambda now: self.state(now=now),
            trace_id_factory=make_trace_id,
            now_factory=now_tz,
            parse_datetime=parse_dt,
            game_cache_updater=self._cache_existing_game,
        ).decide(payload)

    def candidate_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        fact_service = CandidateReplyFactService(parse_datetime=parse_dt)
        semantic_resolver = CandidateSemanticResolverService(
            fallback_classifier=fact_service.classify_reply,
            llm_config=self.llm_config,
            budget_manager=self.llm_budget_manager,
            audit_logger=write_llm_audit_log,
            customer_lookup=self.store.customer,
            confirmed_count_provider=self._confirmed_count,
            urlopen=urllib.request.urlopen,
        )
        semantic_proposer = CandidateSemanticProposalAdapter(
            fallback_proposal_factory=semantic_resolver.fallback_proposal,
            llm_proposal_factory=semantic_resolver.llm_proposal,
        )
        action_validator = CandidateActionProposalValidator(
            fallback_classifier=fact_service.classify_reply,
            negotiation_classifier=fact_service.classify_negotiation,
            game_full_checker=self._game_is_full_for_new_candidate,
            extracted_fact_applier=fact_service.apply_extracted_negotiation_facts,
            final_game_statuses=set(FINAL_GAME_STATUSES),
        )
        feedback_action_service = CandidateFeedbackActionService(
            protocol_version=CONTROLLED_AGENT_PROTOCOL_VERSION,
            runtime_policy_validator=self._runtime_policy_validation_override,
            state_write_policy_validator=self._state_write_proposal_validation_override,
            action_compactor=self._compact_action_record,
            tool_audit_logger=write_tool_audit_log,
            final_game_statuses=set(FINAL_GAME_STATUSES),
        )
        reply_service = CandidateReplyDraftService(confirmed_count_provider=self._confirmed_count)
        return TrialCandidateMessageAdapter(
            outbox_lookup=self.store.outbox_item,
            game_lookup=self._game_by_id,
            semantic_proposal_factory=semantic_proposer.propose,
            proposal_validator=action_validator.validate,
            candidate_reply_factory=reply_service.fallback_reply,
            candidate_reply_guard=reply_service.guard_reply,
            candidate_action_factory=feedback_action_service.build,
            organizer_followup_factory=self._organizer_followup_for_candidate_negotiation,
            action_executor=self._execute_controlled_action,
            action_plan_projector=self._single_action_plan_view,
            feedback_recorder=self.store.record_feedback,
            state_loader=lambda now: self.state(now=now),
            trace_id_factory=make_trace_id,
            now_factory=now_tz,
            parse_datetime=parse_dt,
            customer_reloader=self.reload_customers,
            game_cache_updater=self._cache_existing_game,
            json_dumper=json_dumps,
        ).handle(payload)

    def clear_board(self, payload: dict[str, Any]) -> dict[str, Any]:
        trace_id = str(payload.get("trace_id") or make_trace_id())
        now = parse_dt(payload.get("now")) or now_tz()
        reason = str(payload.get("reason") or "老板手动清空当前局看板").strip()
        active_games = [
            item for item in self.store.games() if str(item.get("status") or "") not in FINAL_GAME_STATUSES
        ]
        action = self._workflow_state_action_record(
            trace_id=trace_id,
            stage="clear_board",
            action_name="archive_current_games",
            arguments={
                "reason": reason,
                "scope": "active_games",
            },
            proposed_by="boss_manual",
            source="boss_manual",
            risk_level="high",
            approval_required=True,
            reason="老板手动清空当前局看板，所有未结束局会归档为已取消。",
            now=now,
            validation={
                "allowed": True,
                "code": "manual_approved",
                "reason": "老板手动触发清空看板，视为已审批。",
                "notes": ["高风险状态写入：会批量归档当前未结束牌局。"],
            },
        )
        result = self._execute_controlled_action(
            action,
            lambda: self.store.clear_current_games(reason=reason),
        )
        result["agent_actions"] = [
            self._single_action_plan_view(stage="clear_board", source="boss_manual", action=action)
        ]
        result["state"] = self.state(now=now)
        return result

    def clear_short_memory(self, payload: dict[str, Any]) -> dict[str, Any]:
        conversation_id = str(
            payload.get("conversation_id")
            or payload.get("conversationId")
            or "boss_trial"
        ).strip() or "boss_trial"
        sender_id = str(payload.get("sender_id") or payload.get("senderId") or "anonymous").strip() or "anonymous"
        reason = str(payload.get("reason") or "老板手动清空当前客户短期记忆").strip()
        name = f"conversation:{conversation_id}:sender:{sender_id}:memory"
        key = self._cache_key(name)
        before = self._cache_get(name, [])
        before_count = len(before) if isinstance(before, list) else 0
        deleted = 0
        if self.cache:
            try:
                deleted = int(self.cache.delete(key))
            except RedisCacheError as exc:
                print(f"Redis cache delete skipped: {exc}")
        else:
            deleted = 1 if key in self._local_cache else 0
            self._local_cache.pop(key, None)
        return {
            "ok": True,
            "conversation_id": conversation_id,
            "sender_id": sender_id,
            "cache_key": key,
            "cleared_count": before_count,
            "deleted_keys": deleted,
            "reason": reason,
            "state": self.state(),
        }

    def _create_game_state_write(
        self,
        *,
        game: GameRequest,
        status: str,
        organizer_id: str,
        organizer_name: str,
        source_text: str,
        parsed: dict[str, Any],
        reply_text: str,
        missing_fields: list[str],
        notes: list[Any],
        outbox: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.store.create_game(
            game_id=game.id,
            status=status,
            organizer_id=organizer_id,
            organizer_name=organizer_name,
            source_text=source_text,
            parsed=parsed,
            reply_text=reply_text,
            missing_fields=missing_fields,
            notes=notes,
        )
        return {
            "ok": True,
            "game_id": game.id,
            "status": status,
            "outbox_count": len(outbox or []),
        }

    def _game_by_id(self, game_id: str) -> dict[str, Any] | None:
        for game in self.store.games(include_final=True):
            if str(game.get("id") or "") == game_id:
                return game
        return None

    def _user_semantic_action_record(
        self,
        *,
        trace_id: str,
        decision,
        game: GameRequest | None,
        missing_fields: list[str],
        explicit_grouping_request: bool,
        use_existing_pool: bool,
        pool_tool_result: dict[str, Any],
        now: datetime,
    ) -> dict[str, Any]:
        proposal = self._user_semantic_action_proposal(
            decision=decision,
            game=game,
            missing_fields=missing_fields,
            explicit_grouping_request=explicit_grouping_request,
            use_existing_pool=use_existing_pool,
            pool_tool_result=pool_tool_result,
        )
        validation = self._validate_user_semantic_action_proposal(
            proposal=proposal,
            game=game,
            missing_fields=missing_fields,
            explicit_grouping_request=explicit_grouping_request,
            use_existing_pool=use_existing_pool,
            pool_tool_result=pool_tool_result,
        )
        stable_payload = json.dumps(
            {
                "trace_id": trace_id,
                "stage": "user_semantic_action",
                "tool_name": "propose_user_action",
                "arguments": {
                    "proposed_action": proposal.get("proposed_action"),
                    "effective_action": validation.get("effective_action"),
                    "decision_action": getattr(getattr(decision, "action", None), "value", ""),
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        action_hash = hashlib.sha256(stable_payload.encode("utf-8")).hexdigest()[:16]
        action = {
            "action_id": f"act_{action_hash}",
            "idempotency_key": f"{trace_id}:user_semantic_action:propose_user_action:{action_hash}",
            "protocol": CONTROLLED_AGENT_PROTOCOL_VERSION,
            "stage": "user_semantic_action",
            "tool_name": "propose_user_action",
            "arguments": {
                "intent": proposal.get("intent"),
                "proposed_action": proposal.get("proposed_action"),
                "confidence": proposal.get("confidence"),
                "decision_action": getattr(getattr(decision, "action", None), "value", ""),
                "critical_missing_fields": sorted(set(missing_fields) & CRITICAL_FIELDS),
            },
            "proposed_by": str(proposal.get("source") or "unknown"),
            "source": str(proposal.get("source") or "unknown"),
            "risk_level": "low",
            "side_effect": False,
            "approval_required": False,
            "reason": str(proposal.get("reasoning_summary") or validation.get("reason") or "用户消息语义动作提案。")[:240],
            "created_at": now.isoformat(),
            "validation": validation,
        }
        allowed = bool(validation.get("allowed"))
        write_tool_audit_log(
            trace_id,
            "action_validation",
            {
                "protocol": CONTROLLED_AGENT_PROTOCOL_VERSION,
                "stage": "user_semantic_action",
                "source": action["source"],
                "proposed_count": 1,
                "allowed_count": 1 if allowed else 0,
                "rejected_count": 0 if allowed else 1,
                "validated_actions": [self._compact_action_record(action)] if allowed else [],
                "rejected_actions": [self._compact_action_record(action)] if not allowed else [],
                "effective_action": validation.get("effective_action"),
            },
        )
        return action

    def _user_semantic_action_proposal(
        self,
        *,
        decision,
        game: GameRequest | None,
        missing_fields: list[str],
        explicit_grouping_request: bool,
        use_existing_pool: bool,
        pool_tool_result: dict[str, Any],
    ) -> dict[str, Any]:
        semantic = getattr(decision, "semantic_proposal", None)
        if isinstance(semantic, dict) and semantic.get("source") == "llm":
            action = self._normalize_user_semantic_action(semantic.get("proposed_action"))
            if action == "unknown":
                action = self._fallback_user_semantic_action(
                    decision=decision,
                    game=game,
                    missing_fields=missing_fields,
                    explicit_grouping_request=explicit_grouping_request,
                    use_existing_pool=use_existing_pool,
                    pool_tool_result=pool_tool_result,
                )
            return {
                "source": "llm",
                "intent": semantic.get("intent"),
                "proposed_action": action,
                "confidence": self._safe_float(semantic.get("confidence")) or 0.0,
                "reasoning_summary": semantic.get("reasoning_summary") or "LLM 根据用户消息提出下一步动作。",
                "raw": semantic,
            }
        action = self._fallback_user_semantic_action(
            decision=decision,
            game=game,
            missing_fields=missing_fields,
            explicit_grouping_request=explicit_grouping_request,
            use_existing_pool=use_existing_pool,
            pool_tool_result=pool_tool_result,
        )
        return {
            "source": "rules",
            "intent": getattr(getattr(decision, "action", None), "value", "unknown"),
            "proposed_action": action,
            "confidence": float(getattr(decision, "confidence", 0.0) or 0.0),
            "reasoning_summary": "LLM 未提供语义动作提案，使用后端安全兜底动作。",
            "raw": {},
        }

    def _fallback_user_semantic_action(
        self,
        *,
        decision,
        game: GameRequest | None,
        missing_fields: list[str],
        explicit_grouping_request: bool,
        use_existing_pool: bool,
        pool_tool_result: dict[str, Any],
    ) -> str:
        decision_action = str(getattr(getattr(decision, "action", None), "value", ""))
        if decision_action == "human_review":
            return "human_review"
        if decision_action == "ignore":
            return "ignore"
        if decision_action in {"accept_seat", "join_game"}:
            return "join_game"
        if decision_action in {"close_game", "decline_invite"}:
            return "cancel_game"
        if use_existing_pool:
            return "search_existing_games"
        if game and explicit_grouping_request and not (set(missing_fields) & CRITICAL_FIELDS):
            return "create_game"
        if pool_tool_result.get("called") is True:
            return "search_existing_games"
        if game or decision_action in {"ask_clarification", "create_pending_game", "queue_invites", "create_game"}:
            return "ask_clarification"
        return "unknown"

    def _validate_user_semantic_action_proposal(
        self,
        *,
        proposal: dict[str, Any],
        game: GameRequest | None,
        missing_fields: list[str],
        explicit_grouping_request: bool,
        use_existing_pool: bool,
        pool_tool_result: dict[str, Any],
    ) -> dict[str, Any]:
        proposed_action = self._normalize_user_semantic_action(proposal.get("proposed_action"))
        source = str(proposal.get("source") or "unknown")
        confidence = self._safe_float(proposal.get("confidence")) or 0.0
        critical_missing = sorted(set(missing_fields) & CRITICAL_FIELDS)
        notes: list[str] = []
        effective_action = proposed_action
        allowed = True
        code = "allowed"
        reason = "用户语义动作提案通过后端校验。"
        llm_contextual_create = source == "llm" and confidence >= 0.72

        if source == "llm" and proposed_action in {"create_game", "search_existing_games", "join_game", "cancel_game"} and confidence < 0.62:
            allowed = False
            effective_action = "ask_clarification"
            code = "low_confidence_downgrade"
            reason = f"LLM 动作置信度 {confidence:.2f} 低于阈值，降级为追问/人工确认。"
        elif proposed_action == "create_game":
            if (
                self.store.runtime_policy().get("llm_required_for_state_writes")
                and not trusted_action_proposer(source)
            ):
                allowed = False
                effective_action = "human_review"
                code = "runtime_policy_llm_required_for_state_write"
                reason = "当前生产策略要求业务状态写入必须由 LLM 或人工明确提案，拒绝后端兜底建局。"
                notes.append("后端可继续记录审计和建议转人工，但不能自动创建当前局。")
            elif game is None:
                allowed = False
                effective_action = "ask_clarification"
                code = "missing_game_context"
                reason = "没有可落库的组局对象，拒绝创建局。"
            elif use_existing_pool:
                allowed = False
                effective_action = "search_existing_games"
                code = "existing_pool_preferred"
                reason = "当前已有匹配局，优先回复现有局，不重复创建新局。"
            elif critical_missing:
                allowed = False
                effective_action = "ask_clarification"
                code = "critical_slots_missing"
                reason = "组局关键信息不足，拒绝创建局。"
                notes.append("缺少：" + "、".join(critical_missing))
            elif not explicit_grouping_request and not llm_contextual_create:
                allowed = False
                effective_action = "search_existing_games" if pool_tool_result.get("called") else "ask_clarification"
                code = "not_explicit_grouping_request"
                reason = "用户未明确要求老板帮忙组局，不能自动创建局。"
            elif not explicit_grouping_request and llm_contextual_create:
                notes.append("LLM 基于上一轮工作流上下文提出创建组局需求，后端按置信度和状态机继续校验。")
        elif proposed_action == "search_existing_games":
            if (
                game is not None
                and explicit_grouping_request
                and not use_existing_pool
                and pool_tool_result.get("called") is True
                and int(pool_tool_result.get("result_count") or 0) == 0
                and not critical_missing
            ):
                effective_action = "create_game"
                code = "pool_no_match_create_game"
                reason = "现有局搜索无匹配，且用户已明确要求组局、关键信息齐全，继续创建组局并进入候选人搜索。"
                notes.append("search_existing_games 被后端升级为 create_game，但仍只会创建待审批邀约，不会直接外发。")
            elif not (use_existing_pool or pool_tool_result.get("called") is True):
                allowed = False
                effective_action = "ask_clarification"
                code = "search_not_supported_by_context"
                reason = "当前上下文没有触发现有局搜索，降级为追问。"
        elif proposed_action == "ask_clarification":
            effective_action = "ask_clarification"
        elif proposed_action in {"human_review", "ignore", "join_game", "cancel_game"}:
            effective_action = proposed_action
        else:
            allowed = False
            effective_action = "ask_clarification" if game else "unknown"
            code = "unknown_action"
            reason = "模型或兜底动作不在白名单内，拒绝执行。"

        if source != "llm":
            notes.append("当前动作为后端安全兜底提案。")
        return {
            "allowed": allowed,
            "code": code,
            "reason": reason,
            "notes": notes,
            "effective_action": effective_action,
            "proposed_action": proposed_action,
            "confidence": confidence,
        }

    def _normalize_user_semantic_action(self, value: Any) -> str:
        action = str(value or "").strip().lower()
        aliases = {
            "search_existing": "search_existing_games",
            "search_current_open_games": "search_existing_games",
            "find_existing_game": "search_existing_games",
            "create_new_game": "create_game",
            "queue_invites": "create_game",
            "find_players": "create_game",
            "clarify": "ask_clarification",
            "ask_followup": "ask_clarification",
            "manual_review": "human_review",
            "silent": "ignore",
            "no_reply": "ignore",
        }
        action = aliases.get(action, action)
        if action in {
            "search_existing_games",
            "create_game",
            "ask_clarification",
            "cancel_game",
            "join_game",
            "human_review",
            "ignore",
            "unknown",
        }:
            return action
        return "unknown"

    def _workflow_state_action_record(
        self,
        *,
        trace_id: str,
        stage: str,
        action_name: str,
        arguments: dict[str, Any],
        proposed_by: str,
        source: str,
        risk_level: str,
        approval_required: bool,
        reason: str,
        now: datetime,
        validation: dict[str, Any] | None = None,
        action_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        stable_payload = json.dumps(
            {
                "trace_id": trace_id,
                "stage": stage,
                "tool_name": action_name,
                "arguments": arguments,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        action_hash = hashlib.sha256(stable_payload.encode("utf-8")).hexdigest()[:16]
        verdict = validation if isinstance(validation, dict) else {}
        if "allowed" not in verdict:
            verdict = {
                "allowed": True,
                "code": "allowed",
                "reason": "动作通过后端校验。",
                "notes": [],
                **verdict,
            }
        verdict.setdefault("notes", [])
        policy_verdict = self._runtime_policy_validation_override(stage=stage, action_name=action_name)
        if policy_verdict:
            original_notes = [str(item) for item in verdict.get("notes") or []]
            verdict = {
                **verdict,
                **policy_verdict,
                "notes": original_notes + [str(item) for item in policy_verdict.get("notes") or []],
            }
        state_write_policy_verdict = self._state_write_proposal_validation_override(
            stage=stage,
            action_name=action_name,
            proposed_by=proposed_by,
            source=source,
        )
        if state_write_policy_verdict:
            original_notes = [str(item) for item in verdict.get("notes") or []]
            verdict = {
                **verdict,
                **state_write_policy_verdict,
                "notes": original_notes + [str(item) for item in state_write_policy_verdict.get("notes") or []],
            }
        action = {
            "action_id": action_id or f"act_{action_hash}",
            "idempotency_key": idempotency_key or f"{trace_id}:{stage}:{action_name}:{action_hash}",
            "protocol": CONTROLLED_AGENT_PROTOCOL_VERSION,
            "stage": stage,
            "tool_name": action_name,
            "arguments": arguments,
            "proposed_by": proposed_by,
            "source": source,
            "risk_level": risk_level,
            "side_effect": True,
            "approval_required": approval_required,
            "reason": reason[:240],
            "created_at": now.isoformat(),
            "validation": verdict,
        }
        allowed = bool(verdict.get("allowed"))
        write_tool_audit_log(
            trace_id,
            "action_validation",
            {
                "protocol": CONTROLLED_AGENT_PROTOCOL_VERSION,
                "stage": stage,
                "source": source,
                "proposed_count": 1,
                "allowed_count": 1 if allowed else 0,
                "rejected_count": 0 if allowed else 1,
                "validated_actions": [self._compact_action_record(action)] if allowed else [],
                "rejected_actions": [self._compact_action_record(action)] if not allowed else [],
            },
        )
        return action

    def _runtime_policy_validation_override(
        self,
        *,
        stage: str,
        action_name: str,
        side_effect: bool = True,
        approval_required: bool = False,
    ) -> dict[str, Any] | None:
        if stage == "runtime_policy":
            return None
        policy = self.store.runtime_policy()
        if policy.get("read_only_mode") and side_effect:
            return {
                "allowed": False,
                "code": "runtime_policy_read_only",
                "reason": "当前运行时策略为只读模式，禁止执行副作用动作。",
                "notes": [str(policy.get("reason") or "")[:240]],
                "runtime_policy": policy,
            }
        if (stage in STATE_WRITE_STAGES or side_effect) and not policy.get("state_writes_enabled", True):
            return {
                "allowed": False,
                "code": "runtime_policy_state_writes_disabled",
                "reason": "当前运行时策略禁止状态写入。",
                "notes": [str(policy.get("reason") or "")[:240]],
                "runtime_policy": policy,
            }
        if action_name == "execute_outbox_delivery" and not policy.get("delivery_enabled", True):
            return {
                "allowed": False,
                "code": "runtime_policy_delivery_disabled",
                "reason": "当前运行时策略禁止执行外发。",
                "notes": [str(policy.get("reason") or "")[:240]],
                "runtime_policy": policy,
            }
        if (stage == "approval_decision" or approval_required) and not policy.get("approval_enabled", True):
            return {
                "allowed": False,
                "code": "runtime_policy_approval_disabled",
                "reason": "当前运行时策略禁止审批动作。",
                "notes": [str(policy.get("reason") or "")[:240]],
                "runtime_policy": policy,
            }
        if stage == "eval_case" and not policy.get("eval_writes_enabled", True):
            return {
                "allowed": False,
                "code": "runtime_policy_eval_writes_disabled",
                "reason": "当前运行时策略禁止写入评测数据。",
                "notes": [str(policy.get("reason") or "")[:240]],
                "runtime_policy": policy,
            }
        return None

    def _state_write_proposal_validation_override(
        self,
        *,
        stage: str,
        action_name: str,
        proposed_by: str,
        source: str,
    ) -> dict[str, Any] | None:
        if stage == "runtime_policy":
            return None
        policy = self.store.runtime_policy()
        if not policy.get("llm_required_for_state_writes"):
            return None
        if stage not in STATE_WRITE_STAGES:
            return None
        if trusted_action_proposer(proposed_by, source):
            return None
        return {
            "allowed": False,
            "code": "runtime_policy_llm_required_for_state_write",
            "reason": "当前生产策略要求业务状态写入必须由 LLM 或人工明确提案，拒绝后端兜底写入。",
            "notes": [str(policy.get("reason") or "")[:240], f"stage={stage}", f"action={action_name}"],
            "runtime_policy": policy,
        }

    def _execute_controlled_action(self, action: dict[str, Any], operation) -> dict[str, Any]:
        claim = self.store.begin_controlled_action(action)
        ledger = {
            "status": (claim.get("record") or {}).get("status"),
            "duplicate": bool(claim.get("duplicate")),
            "execute": bool(claim.get("execute")),
        }
        action["ledger"] = ledger
        if not claim.get("execute"):
            result = claim.get("result") if isinstance(claim.get("result"), dict) else {}
            if claim.get("duplicate"):
                deduped = {**result, "deduplicated": True, "ok": result.get("ok", True)}
                action["ledger"] = {**ledger, "status": (claim.get("record") or {}).get("status") or "executed"}
                return deduped
            return {
                "ok": False,
                "rejected": True,
                "reason": action.get("validation", {}).get("reason"),
            }
        try:
            result = operation()
        except Exception as exc:
            record = self.store.complete_controlled_action(
                action,
                result={"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            action["ledger"] = {"status": record.get("status"), "duplicate": False, "execute": True}
            raise
        record = self.store.complete_controlled_action(action, result=result, status="executed")
        action["ledger"] = {"status": record.get("status"), "duplicate": False, "execute": True}
        return result

    def _single_action_plan_view(self, *, stage: str, source: str, action: dict[str, Any]) -> dict[str, Any]:
        validation = action.get("validation") if isinstance(action.get("validation"), dict) else {}
        allowed = bool(validation.get("allowed"))
        return {
            "protocol": action.get("protocol") or CONTROLLED_AGENT_PROTOCOL_VERSION,
            "stage": stage,
            "source": source,
            "fallback_used": source != "llm",
            "reasoning_summary": action.get("reason"),
            "validated_actions": [self._compact_action_record(action)] if allowed else [],
            "rejected_actions": [self._compact_action_record(action)] if not allowed else [],
        }

    def _confirmed_count(self, game: dict[str, Any] | None) -> int:
        if not isinstance(game, dict):
            return 0
        outbox = game.get("outbox")
        if not isinstance(outbox, list):
            game_id = str(game.get("id") or "")
            outbox = self.store.outbox_for_game(game_id) if game_id else []
        return sum(
            1
            for item in outbox
            if str(item.get("status") or "") in {"已确认", "已到店"}
        )

    def _game_is_full_for_new_candidate(self, game: dict[str, Any] | None) -> bool:
        if not isinstance(game, dict):
            return False
        parsed = game.get("parsed") if isinstance(game.get("parsed"), dict) else {}
        missing_count = parsed.get("missing_count")
        if not isinstance(missing_count, int):
            return False
        return self._confirmed_count(game) >= max(0, missing_count)

    def _organizer_followup_for_candidate_negotiation(
        self,
        *,
        trace_id: str,
        classification: dict[str, Any],
        candidate_text: str,
        suggested_candidate_reply: str,
        outbox_item: dict[str, Any],
        game: dict[str, Any] | None,
        now: datetime,
    ) -> dict[str, Any] | None:
        draft_service = OrganizerFollowupDraftService(
            llm_config=self.llm_config,
            budget_manager=self.llm_budget_manager,
            audit_logger=write_llm_audit_log,
            urlopen=urllib.request.urlopen,
        )
        return TrialOrganizerFollowupAdapter(
            fallback_factory=draft_service.fallback_message,
            draft_factory=draft_service.draft,
            text_guard=draft_service.guard_message,
            tool_plan_validator=self._validate_tool_plan,
            validated_action_lookup=self._validated_tool_action_record,
            action_executor=self._execute_controlled_action,
            followup_state_writer=self._create_pending_followup_state_write,
            plan_projector=self._action_plan_view,
            tool_audit_logger=write_tool_audit_log,
        ).create(
            trace_id=trace_id,
            classification=classification,
            candidate_text=candidate_text,
            suggested_candidate_reply=suggested_candidate_reply,
            outbox_item=outbox_item,
            game=game,
            now=now,
        )

    def _create_pending_followup_state_write(
        self,
        *,
        action: dict[str, Any],
        game_id: str,
        related_outbox_id: str,
        recipient_id: str,
        recipient_name: str,
        message_text: str,
        reason: str,
        draft_source: str,
    ) -> dict[str, Any]:
        followup = self.store.create_followup_message(
            game_id=game_id,
            related_outbox_id=related_outbox_id,
            recipient_id=recipient_id,
            recipient_name=recipient_name,
            message_text=message_text,
            reason=reason,
        )
        approval = self.store.create_approval_request(
            target_type="followup",
            target_id=str(followup.get("id") or ""),
            action_id=str(action.get("action_id") or ""),
            idempotency_key=str(action.get("idempotency_key") or ""),
            risk_level=str(action.get("risk_level") or "high"),
            original_message_text=message_text,
            metadata={
                "game_id": game_id,
                "related_outbox_id": related_outbox_id,
                "recipient_id": recipient_id,
                "recipient_name": recipient_name,
                "draft_source": draft_source,
                "tool_name": "send_message",
                "execution_mode": "create_pending_followup",
            },
        )
        return {**followup, "approval": approval, "approval_status": approval_status_label(approval.get("status"))}

    def manual_create_game(self, payload: dict[str, Any]) -> dict[str, Any]:
        return TrialManualGameAdapter(
            action_record_factory=self._workflow_state_action_record,
            action_executor=self._execute_controlled_action,
            action_plan_projector=self._single_action_plan_view,
            game_state_writer=self._manual_create_game_state_write,
            game_lookup=self._game_by_id,
            state_loader=lambda now: self.state(now=now),
            trace_id_factory=make_trace_id,
            now_factory=now_tz,
            parse_datetime=parse_dt,
            timezone=TZ,
            action_compactor=self._compact_action_record,
            active_game_statuses=set(ACTIVE_GAME_STATUSES),
            final_game_statuses=set(FINAL_GAME_STATUSES),
            game_cache_updater=self._cache_existing_game,
        ).create(payload)

    def _manual_create_game_state_write(
        self,
        *,
        game_id: str,
        status: str,
        organizer_id: str,
        organizer_name: str,
        source_text: str,
        parsed: dict[str, Any],
        notes: list[Any],
    ) -> dict[str, Any]:
        self.store.create_game(
            game_id=game_id,
            status=status,
            organizer_id=organizer_id,
            organizer_name=organizer_name,
            source_text=source_text,
            parsed=parsed,
            reply_text="老板手动创建局，暂无系统建议回复。",
            missing_fields=[],
            notes=notes,
        )
        return {"ok": True, "game_id": game_id, "status": status}

    def _safe_float(self, value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _safe_int(self, value: Any) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def eval_overview(self) -> dict[str, Any]:
        return {
            "paths": {
                "golden": str(GOLDEN_DATASET_PATH),
                "boss_trial_golden": str(BOSS_TRIAL_GOLDEN_PATH),
                "badcase": str(BADCASE_PATH),
                "few_shot": str(FEW_SHOT_EXAMPLES_PATH),
                "skills": str(SKILL_LIBRARY_PATH),
            },
            "counts": {
                "golden": count_jsonl_records(GOLDEN_DATASET_PATH),
                "boss_trial_golden": count_jsonl_records(BOSS_TRIAL_GOLDEN_PATH),
                "badcase": count_jsonl_records(BADCASE_PATH),
                "few_shot": count_jsonl_records(FEW_SHOT_EXAMPLES_PATH),
                "skills": count_jsonl_records(SKILL_LIBRARY_PATH),
            },
            "recent": {
                "golden": read_jsonl_records(GOLDEN_DATASET_PATH, limit=3),
                "boss_trial_golden": read_jsonl_records(BOSS_TRIAL_GOLDEN_PATH, limit=3),
                "badcase": read_jsonl_records(BADCASE_PATH, limit=3),
                "few_shot": read_jsonl_records(FEW_SHOT_EXAMPLES_PATH, limit=3),
                "skills": read_jsonl_records(SKILL_LIBRARY_PATH, limit=3),
            },
            "runner": "PYTHONPATH=src python scripts/run_scenario_eval.py",
        }

    def record_eval_case(self, payload: dict[str, Any]) -> dict[str, Any]:
        case_type = str(payload.get("case_type") or payload.get("kind") or "").strip().lower()
        if case_type not in {"badcase", "golden", "few_shot"}:
            raise ValueError("case_type 必须是 badcase、golden 或 few_shot")
        trace_id = str(payload.get("source_trace_id") or payload.get("trace_id") or make_trace_id())
        now = parse_dt(payload.get("now")) or now_tz()
        action = self._workflow_state_action_record(
            trace_id=trace_id,
            stage="eval_case",
            action_name="record_eval_case",
            arguments={
                "case_type": case_type,
                "source_trace_id": payload.get("source_trace_id"),
                "has_analysis": isinstance(payload.get("analysis"), dict),
            },
            proposed_by="boss_manual",
            source="boss_trial_console",
            risk_level="medium",
            approval_required=True,
            reason="老板归档 badcase/golden/few-shot，后端记录为受控评测数据写入。",
            now=now,
            validation={
                "allowed": True,
                "code": "manual_approved",
                "reason": "老板手动归档评测样本，视为已审批。",
                "notes": ["评测数据会影响后续回归和 few-shot，应进入受控动作账本。"],
            },
        )
        result = self._execute_controlled_action(
            action,
            lambda: self._record_eval_case_state_write(payload, case_type=case_type, trace_id=trace_id, stamp=now),
        )
        result["agent_actions"] = [
            self._single_action_plan_view(stage="eval_case", source="boss_trial_console", action=action)
        ]
        return result

    def _record_eval_case_state_write(
        self,
        payload: dict[str, Any],
        *,
        case_type: str,
        trace_id: str,
        stamp: datetime,
    ) -> dict[str, Any]:
        analysis = payload.get("analysis") if isinstance(payload.get("analysis"), dict) else {}
        source_text = str(
            payload.get("text")
            or analysis.get("source_text")
            or analysis.get("effective_text")
            or ""
        ).strip()
        if not source_text:
            raise ValueError("归档样本缺少原始消息文本")

        trace_id = str(payload.get("source_trace_id") or analysis.get("trace_id") or trace_id or "")
        record_id = str(payload.get("id") or f"{case_type}_{stamp.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}")
        sender_id = str(payload.get("sender_id") or analysis.get("sender_id") or "boss_trial_user")
        sender_name = str(payload.get("sender_name") or analysis.get("sender_name") or "")
        note = str(payload.get("note") or payload.get("notes") or "").strip()
        tags = self._eval_tags(payload, analysis, case_type)
        actual = self._actual_from_analysis(analysis)

        if case_type == "badcase":
            path = BADCASE_PATH
            record = {
                "schema_version": 1,
                "kind": "badcase",
                "id": record_id,
                "name": str(payload.get("name") or "试用台归档 badcase"),
                "source": "boss_trial_console",
                "trace_id": trace_id,
                "observed_at": stamp.isoformat(),
                "tags": tags,
                "text": source_text,
                "sender_id": sender_id,
                "sender_name": sender_name,
                "metadata": {
                    "conversation_id": analysis.get("conversation_id"),
                    "effective_text": analysis.get("effective_text"),
                    "used_short_memory": analysis.get("used_short_memory"),
                },
                "expected": payload.get("expected") or {},
                "actual": actual,
                "triage_status": "new",
                "note": note,
            }
        elif case_type == "golden":
            path = GOLDEN_DATASET_PATH
            expected = self._expected_for_golden(payload, analysis)
            record = {
                "schema_version": 1,
                "kind": "golden",
                "id": record_id,
                "name": str(payload.get("name") or "试用台确认样本"),
                "source": "boss_trial_console",
                "trace_id": trace_id,
                "created_at": stamp.isoformat(),
                "tags": tags,
                "text": source_text,
                "sender_id": sender_id,
                "metadata": {
                    "conversation_id": analysis.get("conversation_id"),
                    "effective_text": analysis.get("effective_text"),
                    "used_short_memory": analysis.get("used_short_memory"),
                },
                "expected": expected,
                "note": note,
            }
            if sender_name:
                record["sender_name"] = sender_name
        else:
            path = FEW_SHOT_EXAMPLES_PATH
            suggested = analysis.get("suggested_reply") if isinstance(analysis.get("suggested_reply"), dict) else {}
            parsed = analysis.get("parsed") if isinstance(analysis.get("parsed"), dict) else {}
            reply_text = str(payload.get("reply_text") or suggested.get("text") or "").strip()
            if not reply_text:
                raise ValueError("few-shot 样本缺少可复用回复")
            record = {
                "schema_version": 1,
                "kind": "few_shot",
                "id": record_id,
                "name": str(payload.get("name") or "老板认可话术"),
                "source": "boss_trial_console",
                "trace_id": trace_id,
                "created_at": stamp.isoformat(),
                "tags": tags,
                "customer_message": source_text,
                "parsed": parsed.get("summary") or self._few_shot_parsed_text(parsed, analysis),
                "reply_text": reply_text,
                "reasoning_summary": str(suggested.get("reasoning_summary") or ""),
                "note": note,
            }

        append_jsonl_record(path, record)
        overview = self.eval_overview()
        return {
            "ok": True,
            "case_type": case_type,
            "record_id": record_id,
            "path": str(path),
            "record": record,
            "overview": overview,
        }

    def _eval_tags(self, payload: dict[str, Any], analysis: dict[str, Any], case_type: str) -> list[str]:
        raw_tags = payload.get("tags") or []
        if isinstance(raw_tags, str):
            raw_tags = re.split(r"[,，、\s]+", raw_tags)
        tags = [str(item).strip() for item in raw_tags if str(item).strip()]
        tags.append(case_type)
        decision = analysis.get("decision") if isinstance(analysis.get("decision"), dict) else {}
        action = decision.get("action")
        if action:
            tags.append(str(action))
        if analysis.get("used_short_memory"):
            tags.append("short_memory")
        parsed = analysis.get("parsed") if isinstance(analysis.get("parsed"), dict) else {}
        game_label = parsed.get("game_label")
        if game_label:
            tags.append(str(game_label))
        return list(dict.fromkeys(tags))

    def _actual_from_analysis(self, analysis: dict[str, Any]) -> dict[str, Any]:
        decision = analysis.get("decision") if isinstance(analysis.get("decision"), dict) else {}
        suggested = analysis.get("suggested_reply") if isinstance(analysis.get("suggested_reply"), dict) else {}
        return {
            "decision": decision,
            "parsed": analysis.get("parsed") or {},
            "missing_fields": analysis.get("missing_fields") or [],
            "suggested_reply": suggested,
            "group_draft": analysis.get("group_draft") or "",
            "candidate_count": len(analysis.get("candidates") or []),
            "outbox_count": len(analysis.get("outbox") or []),
        }

    def _expected_for_golden(self, payload: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
        expected = payload.get("expected") if isinstance(payload.get("expected"), dict) else {}
        if expected:
            return {key: value for key, value in expected.items() if value not in (None, "", [])}
        decision = analysis.get("decision") if isinstance(analysis.get("decision"), dict) else {}
        action = str(decision.get("action") or "").strip()
        should_reply = decision.get("should_reply")
        result: dict[str, Any] = {}
        if action:
            result["action"] = action
        if should_reply is not None:
            result["should_reply"] = bool(should_reply)
        return result

    def _few_shot_parsed_text(self, parsed: dict[str, Any], analysis: dict[str, Any]) -> str:
        fields = [
            parsed.get("game_label"),
            parsed.get("level"),
            parsed.get("start_time"),
        ]
        if parsed.get("missing_count") is not None:
            fields.append(f"缺{parsed.get('missing_count')}")
        rules = parsed.get("rules") if isinstance(parsed.get("rules"), list) else []
        fields.extend(rules[:3])
        missing = analysis.get("missing_fields") or []
        if missing:
            fields.append("待确认：" + "、".join(str(item) for item in missing))
        return "，".join(str(item) for item in fields if item) or "系统未解析出完整组局条件"

    def _few_shot_examples(self) -> list[dict[str, Any]]:
        dynamic: list[dict[str, Any]] = []
        for record in read_jsonl_records(FEW_SHOT_EXAMPLES_PATH, limit=20):
            if record.get("kind") != "few_shot":
                continue
            customer_message = str(record.get("customer_message") or "").strip()
            reply_text = str(record.get("reply_text") or "").strip()
            if not customer_message or not reply_text:
                continue
            item = {
                "name": str(record.get("name") or "老板认可话术"),
                "source": f"试用台采集：{record.get('id')}",
                "customer_message": customer_message,
                "parsed": str(record.get("parsed") or ""),
                "reply_text": reply_text,
            }
            conditions = str(record.get("conditions") or "").strip()
            if conditions:
                item["conditions"] = conditions
            dynamic.append(item)
        return [*BOSS_REPLY_FEW_SHOTS, *dynamic[-8:]]

    def _active_skills(
        self,
        *,
        stage: str,
        source_text: str = "",
        effective_text: str = "",
        game: GameRequest | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        parts = [source_text, effective_text]
        if game:
            parts.extend(
                [
                    self._summary(game),
                    self._game_label(game),
                    " ".join(game.rules),
                    " ".join(game.play_options),
                    " ".join(game.ambiguities),
                ]
            )
        return select_relevant_skills(
            stage=stage,
            text="\n".join(part for part in parts if part),
            path=SKILL_LIBRARY_PATH,
            limit=limit,
        )

    def _cache_status(self) -> dict[str, Any]:
        return {
            "redis_enabled": self.cache is not None,
            "local_fallback_enabled": self.cache is None,
            "prefix": self.cache_prefix,
            "purpose": "短期记忆、当前局快照、最新页面状态缓存",
            "short_memory_ttl_seconds": SHORT_MEMORY_TTL_SECONDS,
            "game_cache_ttl_seconds": GAME_CACHE_TTL_SECONDS,
        }

    def _cache_key(self, name: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", name).strip("_")
        return f"{self.cache_prefix}:{safe}"

    def _cache_set(self, name: str, value: Any, *, ttl_seconds: int) -> None:
        if not self.cache:
            self._local_cache[self._cache_key(name)] = (
                now_tz().timestamp() + max(1, int(ttl_seconds)),
                value,
            )
            return
        try:
            self.cache.set_json(self._cache_key(name), value, ttl_seconds=ttl_seconds)
        except RedisCacheError as exc:
            print(f"Redis cache write skipped: {exc}")

    def _cache_get(self, name: str, default: Any) -> Any:
        key = self._cache_key(name)
        if not self.cache:
            item = self._local_cache.get(key)
            if item is None:
                return default
            expires_at, value = item
            if now_tz().timestamp() > expires_at:
                self._local_cache.pop(key, None)
                return default
            return value
        try:
            return self.cache.get_json(key, default)
        except RedisCacheError as exc:
            print(f"Redis cache read skipped: {exc}")
            return default

    def _sender_memory(self, conversation_id: str, sender_id: str, now: datetime) -> list[dict[str, Any]]:
        rows = self._cache_get(f"conversation:{conversation_id}:sender:{sender_id}:memory", [])
        if not isinstance(rows, list):
            return []
        recent: list[dict[str, Any]] = []
        for row in rows[-12:]:
            if not isinstance(row, dict):
                continue
            created_at = parse_dt(str(row.get("at") or ""))
            if not created_at:
                continue
            if (now - created_at).total_seconds() <= SHORT_MEMORY_TTL_SECONDS:
                recent.append(row)
        return recent

    def _workflow_followup_context(
        self,
        memory: list[dict[str, Any]],
        text: str,
        now: datetime,
    ) -> dict[str, Any]:
        return self.followup_context_builder.build(memory, text, now)

    def _is_grouping_confirmation_followup(
        self,
        workflow_followup_context: dict[str, Any] | None,
        text: str,
    ) -> bool:
        return self.followup_context_builder.is_grouping_confirmation_followup(workflow_followup_context, text)

    def _effective_text(self, memory: list[dict[str, Any]], text: str, now: datetime) -> str:
        return self.short_memory_text_merger.build(memory, text, now)

    def _memory_row_has_pending_goal(self, row: dict[str, Any]) -> bool:
        return self.short_memory_text_merger.memory_row_has_pending_goal(row)

    def _should_merge_short_memory(self, fragments: list[str], text: str) -> bool:
        return self.short_memory_text_merger.should_merge(fragments, text)

    def _remember_sender(
        self,
        *,
        sender_id: str,
        sender_name: str,
        conversation_id: str,
        text: str,
        effective_text: str,
        parsed: dict[str, Any],
        missing_fields: list[str],
        decision: dict[str, Any],
        game_id: str | None,
        trace_id: str,
        now: datetime,
    ) -> None:
        rows = self._sender_memory(conversation_id, sender_id, now)
        current_text = text.strip()
        if current_text:
            rows = [row for row in rows if str(row.get("text") or "").strip() != current_text]
        rows.append(
            {
                "at": now.isoformat(),
                "conversation_id": conversation_id,
                "sender_id": sender_id,
                "sender_name": sender_name,
                "trace_id": trace_id,
                "text": text,
                "effective_text": effective_text,
                "used_short_memory": effective_text != text,
                "game_id": game_id,
                "parsed_summary": parsed.get("summary"),
                "missing_fields": missing_fields,
                "action": decision.get("action"),
            }
        )
        self._cache_set(
            f"conversation:{conversation_id}:sender:{sender_id}:memory",
            rows[-12:],
            ttl_seconds=SHORT_MEMORY_TTL_SECONDS,
        )

    def _update_sender_memory_after_reply(
        self,
        *,
        conversation_id: str,
        sender_id: str,
        trace_id: str,
        suggested_reply: dict[str, Any],
        parsed: dict[str, Any],
        tool_results: dict[str, Any],
        pool_matches: list[dict[str, Any]],
        now: datetime,
    ) -> None:
        rows = self._sender_memory(conversation_id, sender_id, now)
        for row in reversed(rows):
            if str(row.get("trace_id") or "") != trace_id:
                continue
            row["suggested_reply"] = {
                "text": str(suggested_reply.get("text") or ""),
                "source": suggested_reply.get("source"),
                "model": suggested_reply.get("model"),
                "reasoning_summary": suggested_reply.get("reasoning_summary"),
                "status": suggested_reply.get("status"),
            }
            row["system_suggested_reply"] = str(suggested_reply.get("text") or "")
            row["suggested_reasoning"] = suggested_reply.get("reasoning_summary")
            row["parsed"] = {
                key: parsed.get(key)
                for key in [
                    "intent_action",
                    "user_intent",
                    "summary",
                    "game_label",
                    "level",
                    "start_time",
                    "start_time_mode",
                    "duration_text",
                    "current_player_count",
                    "missing_count",
                    "rules",
                    "smoke_options",
                    "level_options",
                ]
                if key in parsed
            }
            row["intent_action"] = parsed.get("intent_action")
            row["user_intent"] = parsed.get("user_intent")
            row["game"] = {
                key: parsed.get(key)
                for key in [
                    "game_type",
                    "game_label",
                    "ruleset",
                    "variant",
                    "level",
                    "start_at",
                    "start_time",
                    "start_time_mode",
                    "duration_hours",
                    "duration_mode",
                    "duration_text",
                    "current_player_count",
                    "missing_count",
                    "rules",
                    "play_options",
                    "ambiguities",
                    "summary",
                ]
                if key in parsed
            }
            row["tool_results"] = self._short_tool_results_for_memory(tool_results, pool_matches)
            break
        self._cache_set(
            f"conversation:{conversation_id}:sender:{sender_id}:memory",
            rows[-12:],
            ttl_seconds=SHORT_MEMORY_TTL_SECONDS,
        )

    def _short_tool_results_for_memory(
        self,
        tool_results: dict[str, Any],
        pool_matches: list[dict[str, Any]],
    ) -> dict[str, Any]:
        pool = tool_results.get("search_current_open_games") if isinstance(tool_results, dict) else {}
        candidates = tool_results.get("search_candidate_customers") if isinstance(tool_results, dict) else {}
        sender = tool_results.get("send_message") if isinstance(tool_results, dict) else {}
        return {
            "search_current_open_games": {
                "called": bool(isinstance(pool, dict) and pool.get("called")),
                "result_count": int(pool.get("result_count") or 0) if isinstance(pool, dict) else 0,
                "matches": [
                    {
                        "game_id": item.get("game_id"),
                        "summary": item.get("summary"),
                        "reply_text": item.get("reply_text"),
                    }
                    for item in pool_matches[:3]
                    if isinstance(item, dict)
                ],
            },
            "search_candidate_customers": {
                "called": bool(isinstance(candidates, dict) and candidates.get("called")),
                "result_count": int(candidates.get("result_count") or 0) if isinstance(candidates, dict) else 0,
            },
            "send_message": {
                "called": bool(isinstance(sender, dict) and sender.get("called")),
                "result_count": int(sender.get("result_count") or 0) if isinstance(sender, dict) else 0,
                "direct_send_executed": bool(isinstance(sender, dict) and sender.get("direct_send_executed")),
            },
        }

    def _cache_game(
        self,
        game: GameRequest,
        outbox: list[dict[str, Any]],
        *,
        status: str,
        source_text: str,
    ) -> None:
        self._cache_set(
            f"game:{game.id}",
            {
                "id": game.id,
                "status": status,
                "source_text": source_text,
                "parsed": self._game_to_dict(game),
                "outbox": outbox,
                "updated_at": now_tz().isoformat(),
            },
            ttl_seconds=GAME_CACHE_TTL_SECONDS,
        )

    def _cache_existing_game(self, game_id: str) -> None:
        for game in self.store.games():
            if game["id"] == game_id:
                self._cache_set(
                    f"game:{game_id}",
                    {**game, "updated_at": now_tz().isoformat()},
                    ttl_seconds=GAME_CACHE_TTL_SECONDS,
                )
                return

    def _profile_from_customer(self, customer: dict[str, Any]) -> CustomerProfile:
        games = customer["preferred_games"]
        preferences: list[PlayPreference] = []
        has_caiqiao = self._customer_has_label(customer, "财敲")
        if self._customer_has_label(customer, "杭麻") or has_caiqiao:
            preferences.append(
                PlayPreference(
                    game_type="hangzhou_mahjong",
                    preferred_levels=customer["preferred_levels"],
                    preferred_rulesets=["hangzhou_mahjong"],
                    preferred_variants=["caiqiao"] if has_caiqiao else [],
                    preferred_play_options=["财敲"] if has_caiqiao else [],
                )
            )
        if self._customer_has_label(customer, "川麻") or self._customer_has_label(customer, "幺鸡"):
            preferences.append(
                PlayPreference(
                    game_type="sichuan_mahjong",
                    preferred_levels=customer["preferred_levels"],
                    preferred_rulesets=["sichuan_mahjong"],
                    preferred_variants=["yaoji"] if self._customer_has_label(customer, "幺鸡") else [],
                    preferred_play_options=[
                        label
                        for label in ["幺鸡", "素鸡", "幺鸡47", "换三张", "定缺"]
                        if self._customer_has_label(customer, label)
                    ],
                )
            )
        for label, game_type in [
            ("红中", "hongzhong_mahjong"),
            ("捉鸡", "zhuoji_mahjong"),
            ("湖南麻将", "hunan_mahjong"),
        ]:
            if self._has_game(games, label):
                preferences.append(
                    PlayPreference(
                        game_type=game_type,
                        preferred_levels=customer["preferred_levels"],
                        preferred_rulesets=[game_type],
                    )
                )
        return CustomerProfile(
            id=customer["id"],
            display_name=customer["display_name"],
            preferred_levels=customer["preferred_levels"],
            play_preferences=preferences,
            tags=[
                *customer["preferred_games"],
                *[label for label in ["财敲", "换三张", "定缺", "幺鸡", "素鸡", "幺鸡47"] if self._customer_has_label(customer, label)],
                *customer["preferred_levels"],
                customer.get("gender_label") or "",
            ],
            smoke_free_preference={"no_smoke": True, "smoke_ok": False}.get(customer["smoke_preference"]),
            usual_party_size=customer.get("usual_party_size"),
            usual_party_size_confidence=float(customer.get("usual_party_size_confidence") or 0),
            usual_start_hours=customer["usual_start_hours"],
            no_contact=customer["no_contact"],
            last_invited_at=parse_dt(customer["last_invited_at"]),
            max_games_per_day=2 if customer["fatigue_score"] < 35 else 1,
            invite_cooldown_hours=8 if customer["fatigue_score"] >= 50 else 6,
            metadata={
                "contact": customer["contact"],
                "response_speed": customer["response_speed"],
                "response_rate": customer["response_rate"],
                "last_arrived_at": customer["last_arrived_at"],
                "notes": customer["notes"],
                "gender": customer.get("gender") or "unknown",
                "gender_label": customer.get("gender_label") or "未知",
            },
        )

    def _apply_trial_inferences(
        self,
        game: GameRequest,
        text: str,
        sender_id: str,
        *,
        now: datetime,
        source_text: str = "",
        sender_memory: list[dict[str, Any]] | None = None,
    ) -> None:
        normalized = text.lower()
        past_time_ambiguity = self._past_time_ambiguity_from_text(text, now)
        if past_time_ambiguity and past_time_ambiguity not in game.ambiguities:
            game.ambiguities.append(past_time_ambiguity)
        if past_time_ambiguity:
            note = "用户给出的开局时间已早于当前时间，必须确认是否改期或是否指明天。"
            if note not in game.notes:
                game.notes.append(note)
        self._apply_time_resolution_policy(game, now=now)
        confirmed_party_size = self._confirmed_party_size_from_affirmative_followup(
            source_text=source_text,
            sender_id=sender_id,
            sender_memory=sender_memory or [],
        )
        explicit_party = self._explicit_party_counts_from_text(f"{source_text}\n{text}")
        requires_party_confirmation = self._requires_party_size_confirmation(normalized)
        if explicit_party is not None:
            current, missing, raw = explicit_party
            game.current_player_count = current
            game.missing_count = missing
            note = f"试用台按用户原文明确人数 {raw} 纠正为 {current}缺{missing}"
            if note not in game.notes:
                game.notes.append(note)
            requires_party_confirmation = False
        elif confirmed_party_size is not None:
            game.current_player_count = confirmed_party_size
            game.missing_count = max(0, game.seats_total - confirmed_party_size)
            note = f"用户短答确认上一轮人数追问，按画像确认 {confirmed_party_size} 人"
            if note not in game.notes:
                game.notes.append(note)
            requires_party_confirmation = False
        if requires_party_confirmation and (game.current_player_count is not None or game.missing_count is not None):
            game.current_player_count = None
            game.missing_count = None
            game.notes.append("试用台清理未由当前原文明确支持的人数/缺口推断")
        if game.current_player_count is None and game.missing_count is None:
            customer = self.responder.core.store.customers.get(sender_id)
            if (
                not requires_party_confirmation
                and customer
                and customer.usual_party_size
                and customer.usual_party_size_confidence >= TRIAL_PROFILE_PARTY_SIZE_CONFIDENCE_THRESHOLD
            ):
                game.current_player_count = customer.usual_party_size
                game.missing_count = max(0, game.seats_total - customer.usual_party_size)
                game.notes.append(f"试用台根据客户画像推断 {customer.usual_party_size} 人")
            elif requires_party_confirmation:
                game.notes.append("用户说组一桌但未说明现在几个人，需确认人数")
        if game.current_player_count is not None and game.missing_count is not None:
            game.ambiguities = [item for item in game.ambiguities if "几缺几" not in item and "几个人" not in item]
        self._apply_candidate_composition_preference(game, f"{source_text}\n{text}")
        if game.status == GameStatus.NEED_CLARIFICATION and not (set(self._missing_fields(game, None)) & CRITICAL_FIELDS):
            game.status = GameStatus.OPEN

    def _apply_candidate_composition_preference(self, game: GameRequest, text: str) -> None:
        preference = self._candidate_composition_preference_from_text(text, game.missing_count)
        if not preference:
            return
        genders = [gender for gender in preference.get("preferred_candidate_genders") or [] if gender in GENDER_LABELS]
        if not genders:
            return
        game.rules = [
            rule
            for rule in game.rules
            if not re.search(r"(男|女).*玩家|玩家.*(男|女)|一男一女|一女一男|性别", str(rule))
        ]
        note = (
            f"{GENDER_NOTE_PREFIX}性别={','.join(genders)}；"
            f"强度={'soft' if preference.get('soft', True) else 'hard'}；"
            f"来源={preference.get('source_phrase') or '用户原文'}"
        )
        game.notes = [item for item in game.notes if not str(item).startswith(GENDER_NOTE_PREFIX)]
        game.notes.append(note)

    def _candidate_composition_preference_from_text(
        self,
        text: str,
        missing_count: int | None = None,
    ) -> dict[str, Any]:
        compact = re.sub(r"\s+", "", text or "")
        if not compact or not re.search(r"男|女", compact):
            return {}
        if re.search(r"男女(都可|都行|不限|无所谓)|性别(不限|无所谓)", compact):
            return {}
        genders: list[str] = []
        source_phrase = ""
        if "一男一女" in compact or "一女一男" in compact:
            genders = ["male", "female"]
            source_phrase = "一男一女" if "一男一女" in compact else "一女一男"
        elif re.search(r"(两个|2个|两位|2位|两个?人).{0,2}(男|男生|男士)", compact):
            genders = ["male", "male"]
            source_phrase = "两个男"
        elif re.search(r"(两个|2个|两位|2位|两个?人).{0,2}(女|女生|女士)", compact):
            genders = ["female", "female"]
            source_phrase = "两个女"
        elif re.search(r"(来|找|要|缺|补).{0,6}(男|男生|男士).{0,6}(女|女生|女士)", compact) or re.search(
            r"(来|找|要|缺|补).{0,6}(女|女生|女士).{0,6}(男|男生|男士)", compact
        ):
            genders = ["male", "female"]
            source_phrase = "男女组合"
        if not genders:
            return {}
        if isinstance(missing_count, int) and missing_count > 0:
            genders = genders[:missing_count]
        return {
            "preferred_candidate_genders": genders,
            "gender_labels": [GENDER_LABELS.get(gender, "未知") for gender in genders],
            "soft": True,
            "source_phrase": source_phrase,
        }

    def _candidate_composition_preference_from_game(self, game: GameRequest | None) -> dict[str, Any]:
        if not game:
            return {}
        for note in game.notes:
            text = str(note)
            if not text.startswith(GENDER_NOTE_PREFIX):
                continue
            match = re.search(r"性别=([a-z_,]+)", text)
            if not match:
                continue
            genders = [normalize_gender(item) for item in match.group(1).split(",")]
            genders = [gender for gender in genders if gender in {"male", "female"}]
            if not genders:
                continue
            return {
                "preferred_candidate_genders": genders,
                "gender_labels": [GENDER_LABELS.get(gender, "未知") for gender in genders],
                "soft": "强度=hard" not in text,
                "source_note": text,
            }
        return {}

    def _explicit_party_counts_from_text(self, text: str) -> tuple[int, int, str] | None:
        normalized = re.sub(r"\s+", "", text.lower())
        patterns: list[tuple[str, int, int, str]] = [
            (r"(?<!\d)(371)(?!\d)", 3, 1, "371"),
            (r"三(?:缺|差|等)一|3(?:缺|差|等)1", 3, 1, "三缺一"),
            (r"(?<!\d)(272)(?!\d)", 2, 2, "272"),
            (r"(?:二|两|2)(?:缺|差|等)(?:二|两|2)", 2, 2, "二缺二"),
            (r"(?<!\d)(173)(?!\d)", 1, 3, "173"),
            (r"一(?:缺|差|等)三|1(?:缺|差|等)3", 1, 3, "一缺三"),
        ]
        for pattern, current, missing, label in patterns:
            if re.search(pattern, normalized):
                return current, missing, label
        return None

    def _apply_time_resolution_policy(self, game: GameRequest, *, now: datetime) -> None:
        if game.start_at is None or not game.ambiguities:
            return
        kept: list[str] = []
        resolved: list[str] = []
        for ambiguity in game.ambiguities:
            decision = self._time_resolution_decision(game, ambiguity, now=now)
            if decision.get("accepted"):
                resolved.append(ambiguity)
                game.start_time_confidence = max(
                    float(game.start_time_confidence or 0),
                    float(decision.get("confidence") or 0),
                )
                note = f"时间消歧：{ambiguity}，{decision.get('reason')}，按 {game.start_at.strftime('%H:%M')} 处理。"
                if note not in game.notes:
                    game.notes.append(note)
            else:
                kept.append(ambiguity)
        if resolved:
            game.ambiguities = kept

    def _time_resolution_decision(self, game: GameRequest, ambiguity: str, *, now: datetime) -> dict[str, Any]:
        if "上午还是下午" not in ambiguity or game.start_at is None:
            return {"accepted": False, "confidence": 0.0, "reason": "不是可自动消歧的上午/下午候选"}
        if game.start_at.date() != now.date():
            return {"accepted": False, "confidence": 0.0, "reason": "候选时间不是今天，仍需确认"}
        minutes_until_start = (game.start_at - now).total_seconds() / 60
        if minutes_until_start < -15:
            return {"accepted": False, "confidence": 0.0, "reason": "候选时间已经明显早于当前时间"}
        if minutes_until_start > LOCAL_TIME_MAX_LOOKAHEAD_HOURS * 60:
            return {"accepted": False, "confidence": 0.0, "reason": "候选时间距离当前过远"}
        if now.hour < LOCAL_AFTERNOON_CONTEXT_HOUR:
            return {"accepted": False, "confidence": 0.0, "reason": "当前还未进入下午业务语境"}
        confidence = 0.82
        if 0 <= minutes_until_start <= 180:
            confidence = 0.88
        accepted = confidence >= TIME_RESOLUTION_CONFIDENCE_THRESHOLD
        return {
            "accepted": accepted,
            "confidence": confidence,
            "reason": "当前已是下午，且解析候选是今天接下来可开局的最近时间",
        }

    def _confirmed_party_size_from_affirmative_followup(
        self,
        *,
        source_text: str,
        sender_id: str,
        sender_memory: list[dict[str, Any]],
    ) -> int | None:
        if not self._is_affirmative_short_answer(source_text):
            return None
        for row in reversed(sender_memory[-6:]):
            missing_fields = row.get("missing_fields") or []
            if not isinstance(missing_fields, list) or "known_players" not in missing_fields:
                continue
            customer = self.responder.core.store.customers.get(sender_id)
            if not customer:
                return None
            party_size = customer.usual_party_size
            confidence = float(customer.usual_party_size_confidence or 0)
            if isinstance(party_size, int) and 1 <= party_size <= 4 and confidence >= TRIAL_PROFILE_PARTY_SIZE_CONFIDENCE_THRESHOLD:
                return party_size
            return None
        return None

    def _is_affirmative_short_answer(self, text: str) -> bool:
        normalized = re.sub(r"[\s，。,.!?！？~～]+", "", text.strip().lower())
        return normalized in {"是", "是的", "对", "对的", "嗯", "嗯嗯", "嗯呢", "没错", "对啊", "对呀"}

    def _past_time_ambiguity_from_text(self, text: str, now: datetime) -> str | None:
        normalized = text.lower()
        if re.search(r"(明天|明晚|明儿|后天)", normalized):
            return None
        probe = Message(
            text=text,
            sender_id="time_probe",
            sender_name="time_probe",
            channel_id="time_probe",
            channel_type=ChannelType.MANUAL,
            sent_at=now,
        )
        extraction = self.responder.core.parser.parse(probe, now=now)
        if not extraction.game:
            return None
        for ambiguity in extraction.game.ambiguities:
            if "已经过了" in ambiguity:
                return ambiguity
        return None

    def _requires_party_size_confirmation(self, text: str) -> bool:
        if self._has_explicit_party_size(text):
            return False
        return bool(re.search(r"(帮我|帮忙|给我).*(组|找|摇).*(一桌|一局)|组一桌|找一桌", text))

    def _has_explicit_party_size(self, text: str) -> bool:
        return bool(
            re.search(
                r"(371|三\s*(?:缺|差|等)\s*一|3\s*(?:缺|差|等)\s*1|272|二\s*(?:缺|差|等)\s*二|两\s*(?:缺|差|等)\s*两|2\s*(?:缺|差|等)\s*2|"
                r"173|一\s*(?:缺|差|等)\s*三|1\s*(?:缺|差|等)\s*3|缺\s*[一二两三123])",
                text,
            )
            or re.search(r"(?<!\d)(1|一)\s*(?:个|位)?\s*人(?!\d)", text)
            or re.search(r"(?<!\d)(2|二|两|俩)\s*(?:个|位)?\s*人(?!\d)", text)
            or re.search(r"(?<!\d)(3|三)\s*(?:个|位)?\s*人(?!\d)", text)
            or re.search(r"(?<!\d)(4|四)\s*(?:个|位)?\s*人(?!\d)", text)
        )

    def _missing_fields(self, game: GameRequest | None, decision: Any) -> list[str]:
        if game is None:
            return ["play_type", "start_time", "stake", "known_players"]
        fields: list[str] = []
        if game.game_type == "mahjong" and not game.ruleset and not game.variant:
            fields.append("play_type")
        if self._has_start_time_ambiguity(game) or (game.start_at is None and not self._has_flexible_start(game)):
            fields.append("start_time")
        if game.level is None:
            fields.append("stake")
        if game.current_player_count is None or game.missing_count is None:
            fields.append("known_players")
        if "无烟" not in game.rules and "可吸烟" not in game.rules and "烟况都可" not in game.rules:
            fields.append("smoke")
        if not self._has_duration_strategy(game):
            fields.append("duration")
        return fields

    def _has_flexible_start(self, game: GameRequest | None) -> bool:
        if game is None:
            return False
        return "人齐开" in set(game.rules or []) or "人齐开" in set(game.play_options or [])

    def _start_time_mode(self, game: GameRequest | None) -> str:
        if game is None:
            return "unknown"
        if game.start_at is not None:
            return "fixed"
        if self._has_flexible_start(game):
            return "people_ready"
        return "unknown"

    def _start_time_display(self, game: GameRequest | None) -> str | None:
        if game is None:
            return None
        if game.start_at is not None:
            return game.start_at.strftime("%H:%M")
        if self._has_flexible_start(game):
            return "人齐开"
        return None

    def _has_duration_strategy(self, game: GameRequest | None) -> bool:
        if game is None:
            return False
        return game.duration_hours is not None or "通宵" in set(game.rules or []) or "通宵" in set(game.play_options or [])

    def _duration_mode(self, game: GameRequest | None) -> str:
        if game is None:
            return "unknown"
        if game.duration_hours is not None:
            return "fixed"
        if "通宵" in set(game.rules or []) or "通宵" in set(game.play_options or []):
            return "overnight"
        return "unknown"

    def _has_start_time_ambiguity(self, game: GameRequest | None) -> bool:
        if game is None:
            return False
        return any("已经过了" in item or "上午还是下午" in item for item in game.ambiguities)

    def _start_time_question(self, game: GameRequest | None) -> str:
        if game:
            for ambiguity in game.ambiguities:
                if "已经过了" in ambiguity:
                    raw = ambiguity.split(" 已经过了", 1)[0].strip()
                    return f"你说的{raw}是明天吗，还是改其他时间？" if raw else "这个时间已经过了，是明天还是改其他时间？"
                if "上午还是下午" in ambiguity:
                    raw = ambiguity.split(" 是上午还是下午", 1)[0].strip()
                    return f"你说的{raw}是上午还是下午？" if raw else "这个时间是上午还是下午？"
        return "大概几点开？"

    def _follow_up_text(
        self,
        missing_fields: list[str],
        fallback: str,
        *,
        sender_id: str | None = None,
        game: GameRequest | None = None,
    ) -> str:
        if not missing_fields:
            return ""
        profile = self._customer_profile_for_prompt(sender_id) if sender_id else {}
        preferred_levels = profile.get("preferred_levels") if isinstance(profile.get("preferred_levels"), list) else []
        default_level = preferred_levels[0] if len(preferred_levels) == 1 else None
        party_question = "你这边现在几个人？"
        usual_party_size = profile.get("usual_party_size")
        usual_party_confidence = float(profile.get("usual_party_size_confidence") or 0) if profile else 0
        if usual_party_size == 1 and usual_party_confidence >= TRIAL_PROFILE_PARTY_SIZE_CONFIDENCE_THRESHOLD:
            party_question = "你一个人吗？"
        elif isinstance(usual_party_size, int) and usual_party_size > 1 and usual_party_confidence >= TRIAL_PROFILE_PARTY_SIZE_CONFIDENCE_THRESHOLD:
            party_question = f"你们{usual_party_size}个人吗？"
        questions = {
            "play_type": "今天还是打杭麻吗？",
            "start_time": self._start_time_question(game),
            "stake": f"还是按你常打的 {default_level} 吗？" if default_level else "打多大合适？",
            "known_players": party_question,
            "smoke": "烟这边有要求吗，无烟还是都可以？",
            "duration": "大概打几个小时？",
        }
        question_parts = [questions[field] for field in missing_fields if field in questions]
        if not question_parts:
            return fallback
        if "start_time" in missing_fields and self._has_start_time_ambiguity(game):
            if len(question_parts) == 1:
                return question_parts[0]
            return f"{question_parts[0]}另外{''.join(question_parts[1:3])}"
        return "可以，我先帮你看。" + " ".join(question_parts[:4])

    def _llm_tool_plan(
        self,
        *,
        trace_id: str,
        stage: str,
        sender_id: str,
        sender_name: str,
        source_text: str,
        effective_text: str,
        workflow_followup_context: dict[str, Any] | None,
        game: GameRequest | None,
        missing_fields: list[str],
        decision_action: str,
        tool_results: dict[str, Any],
        now: datetime,
    ) -> dict[str, Any]:
        fallback = self._fallback_tool_plan(
            stage=stage,
            source_text=source_text,
            effective_text=effective_text,
            game=game,
            missing_fields=missing_fields,
            decision_action=decision_action,
        )
        if self._should_use_deterministic_tool_plan(
            stage=stage,
            source_text=source_text,
            effective_text=effective_text,
            game=game,
        ):
            return self._validate_tool_plan(
                trace_id=trace_id,
                plan={
                    **fallback,
                    "source": "rules",
                    "fallback_used": False,
                    "reasoning_summary": "强规则命中当前局池查询，使用确定性只读工具计划，跳过 LLM。",
                },
                stage=stage,
                game=game,
                missing_fields=missing_fields,
                tool_results=tool_results,
                now=now,
            )
        if not self.llm_config or not self.llm_budget_manager:
            return self._validate_tool_plan(
                trace_id=trace_id,
                plan=fallback,
                stage=stage,
                game=game,
                missing_fields=missing_fields,
                tool_results=tool_results,
                now=now,
            )

        available_tools = self._tool_specs_for_stage(stage)
        if not available_tools:
            return self._validate_tool_plan(
                trace_id=trace_id,
                plan={**fallback, "reasoning_summary": "当前阶段没有可用工具。"},
                stage=stage,
                game=game,
                missing_fields=missing_fields,
                tool_results=tool_results,
                now=now,
            )

        max_tokens = min(self.llm_config.max_completion_tokens, 260)
        prompt_input = TrialToolPlanPromptInput(
            stage=stage,
            now=now,
            sender_id=sender_id,
            sender_name=sender_name,
            customer_profile=self._customer_profile_for_prompt(sender_id),
            source_text=source_text,
            effective_text=effective_text,
            workflow_followup_context=workflow_followup_context or {},
            text_normalization=self._text_normalization_for_prompt(source_text, effective_text),
            decision_action=decision_action,
            parsed_game=self._game_to_dict(game) if game else {},
            missing_fields=missing_fields,
            critical_fields=set(CRITICAL_FIELDS),
            available_tools=available_tools,
            tool_registry_version=TOOL_REGISTRY_VERSION,
            existing_tool_results=self._tool_results_for_prompt(tool_results) if tool_results else {},
            active_skills=self._active_skills(
                stage="tool_planning",
                source_text=source_text,
                effective_text=effective_text,
                game=game,
            ),
        )
        payload = self.tool_plan_prompt_builder.build_payload(
            prompt_input,
            model=self.llm_config.model,
            temperature=min(self.llm_config.temperature, 0.2),
            max_tokens=max_tokens,
            thinking_enabled=self.llm_config.thinking_enabled,
            response_format=self.llm_config.response_format,
        )

        budget_decision = self.llm_budget_manager.reserve(
            key="boss_trial_tool_plan",
            model=self.llm_config.model,
            prompt=payload,
            max_completion_tokens=max_tokens,
        )
        if not budget_decision.allowed:
            write_llm_audit_log(
                trace_id,
                "llm_budget_denied",
                {
                    "stage": "tool_planning",
                    "tool_stage": stage,
                    "provider": self.llm_config.provider,
                    "model": self.llm_config.model,
                    "budget": budget_decision.to_dict(),
                    "fallback_plan": fallback,
                },
            )
            return self._validate_tool_plan(
                trace_id=trace_id,
                plan={
                    **self._fail_closed_tool_plan(
                        fallback,
                        llm_source="budget_denied",
                        reason="LLM 工具规划预算不足，已 fail-closed，仅保留只读工具。",
                    ),
                    "budget": budget_decision.to_dict(),
                },
                stage=stage,
                game=game,
                missing_fields=missing_fields,
                tool_results=tool_results,
                now=now,
            )

        write_llm_audit_log(
            trace_id,
            "llm_request",
            {
                "stage": "tool_planning",
                "tool_stage": stage,
                "provider": self.llm_config.provider,
                "model": self.llm_config.model,
                "base_url": self.llm_config.base_url,
                "timeout_seconds": self.llm_config.timeout_seconds,
                "budget": budget_decision.to_dict(),
                "payload": payload,
            },
        )
        request = urllib.request.Request(
            f"{self.llm_config.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.llm_config.api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.llm_config.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
                usage = usage_from_response(data, self.llm_config.model)
                self.llm_budget_manager.commit(budget_decision.reservation_id, usage)
        except urllib.error.HTTPError as exc:
            write_llm_audit_log(
                trace_id,
                "llm_error",
                {
                    "stage": "tool_planning",
                    "tool_stage": stage,
                    "provider": self.llm_config.provider,
                    "model": self.llm_config.model,
                    "error": f"HTTPError {exc.code}: {exc.reason}",
                    "fallback_plan": fallback,
                },
            )
            return self._validate_tool_plan(
                trace_id=trace_id,
                plan=self._fail_closed_tool_plan(
                    fallback,
                    llm_source="llm_http_error",
                    reason=f"LLM 工具规划失败 HTTP {exc.code}，已 fail-closed，仅保留只读工具。",
                ),
                stage=stage,
                game=game,
                missing_fields=missing_fields,
                tool_results=tool_results,
                now=now,
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            write_llm_audit_log(
                trace_id,
                "llm_error",
                {
                    "stage": "tool_planning",
                    "tool_stage": stage,
                    "provider": self.llm_config.provider,
                    "model": self.llm_config.model,
                    "error": f"{type(exc).__name__}: {exc}",
                    "fallback_plan": fallback,
                },
            )
            return self._validate_tool_plan(
                trace_id=trace_id,
                plan=self._fail_closed_tool_plan(
                    fallback,
                    llm_source="llm_error",
                    reason=f"LLM 工具规划失败：{type(exc).__name__}，已 fail-closed，仅保留只读工具。",
                ),
                stage=stage,
                game=game,
                missing_fields=missing_fields,
                tool_results=tool_results,
                now=now,
            )

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        write_llm_audit_log(
            trace_id,
            "llm_response",
            {
                "stage": "tool_planning",
                "tool_stage": stage,
                "provider": self.llm_config.provider,
                "model": self.llm_config.model,
                "usage": asdict(usage) if usage else None,
                "raw_content": content,
            },
        )
        parsed = self._parse_llm_reply_json(
            content,
            trace_id=trace_id,
            stage="tool_planning",
            schema_hint='{"tool_calls":[{"tool_name":str,"arguments":{},"reason":str}],"reasoning_summary":str}',
        )
        tool_calls = self._normalize_tool_calls(parsed.get("tool_calls"), available_tools)
        if not tool_calls:
            return self._validate_tool_plan(
                trace_id=trace_id,
                plan={
                    **self._fail_closed_tool_plan(
                        fallback,
                        llm_source="invalid_or_empty_tool_plan",
                        reason="LLM 未返回有效工具计划，已 fail-closed，仅保留只读工具。",
                    ),
                    "llm_reasoning_summary": str(parsed.get("reasoning_summary") or "")[:240],
                },
                stage=stage,
                game=game,
                missing_fields=missing_fields,
                tool_results=tool_results,
                now=now,
            )
        return self._validate_tool_plan(
            trace_id=trace_id,
            plan={
                "source": "llm",
                "stage": stage,
                "fallback_used": False,
                "tool_calls": tool_calls,
                "reasoning_summary": str(parsed.get("reasoning_summary") or "LLM 根据当前目标选择工具。")[:240],
            },
            stage=stage,
            game=game,
            missing_fields=missing_fields,
            tool_results=tool_results,
            now=now,
        )

    def _fallback_tool_plan(
        self,
        *,
        stage: str,
        source_text: str,
        effective_text: str,
        game: GameRequest | None,
        missing_fields: list[str],
        decision_action: str,
    ) -> dict[str, Any]:
        tool_calls: list[dict[str, Any]] = []
        if stage == "before_open_game_search" and self._should_search_existing_pool(source_text, effective_text, game):
            tool_calls.append(
                {
                    "tool_name": "search_current_open_games",
                    "arguments": {},
                    "reason": self._pool_tool_call_reason(source_text, effective_text, game, decision_action),
                    "requested_by": "backend_fallback",
                }
            )
        if stage == "after_open_game_search" and game and not (set(missing_fields) & CRITICAL_FIELDS):
            tool_calls.extend(
                [
                    {
                        "tool_name": "search_candidate_customers",
                        "arguments": {},
                        "reason": "关键信息已足够且没有可拼局，搜索候选人。",
                        "requested_by": "backend_fallback",
                    },
                    {
                        "tool_name": "send_message",
                        "arguments": {"execution_mode": "create_pending_outbox"},
                        "reason": "有候选人时创建待审批邀约草稿，不直接发送。",
                        "requested_by": "backend_fallback",
                    },
                ]
            )
        if stage == "after_candidate_search" and game and not (set(missing_fields) & CRITICAL_FIELDS):
            tool_calls.append(
                {
                    "tool_name": "send_message",
                    "arguments": {"execution_mode": "create_pending_outbox"},
                    "reason": "候选人搜索已有结果，创建待审批邀约草稿，不直接发送。",
                    "requested_by": "backend_fallback",
                }
            )
        return {
            "source": "backend_fallback",
            "stage": stage,
            "fallback_used": True,
            "tool_calls": tool_calls,
            "reasoning_summary": "模型工具规划不可用或不完整，使用后端兜底策略。",
        }

    def _should_use_deterministic_tool_plan(
        self,
        *,
        stage: str,
        source_text: str,
        effective_text: str,
        game: GameRequest | None,
    ) -> bool:
        if stage != "before_open_game_search":
            return False
        text = self._normalize_pool_query_text(f"{source_text}\n{effective_text}")
        return (
            self._is_pool_inquiry_text(text)
            and self._should_search_existing_pool(source_text, effective_text, game)
            and not self._is_explicit_grouping_request(source_text, effective_text, game)
        )

    def _fail_closed_tool_plan(
        self,
        fallback: dict[str, Any],
        *,
        llm_source: str,
        reason: str,
    ) -> dict[str, Any]:
        safe_calls: list[dict[str, Any]] = []
        for call in fallback.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            tool_name = str(call.get("tool_name") or "")
            if bool(TOOL_REGISTRY.get(tool_name, {}).get("side_effect")):
                continue
            safe_calls.append({**call, "reason": str(call.get("reason") or reason)})
        return {
            **fallback,
            "source": "backend_fallback",
            "fallback_used": True,
            "llm_source": llm_source,
            "tool_calls": safe_calls,
            "reasoning_summary": reason,
        }

    def _tool_specs_for_stage(self, stage: str) -> list[dict[str, Any]]:
        return tool_specs_for_stage(stage)

    def _normalize_tool_calls(self, raw_calls: Any, available_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.tool_call_normalizer.normalize(raw_calls, available_tools)

    def _validate_tool_plan(
        self,
        *,
        trace_id: str,
        plan: dict[str, Any],
        stage: str,
        game: GameRequest | None,
        missing_fields: list[str],
        tool_results: dict[str, Any],
        now: datetime,
    ) -> dict[str, Any]:
        raw_calls = plan.get("tool_calls") if isinstance(plan.get("tool_calls"), list) else []
        allowed_calls: list[dict[str, Any]] = []
        proposals: list[dict[str, Any]] = []
        validated_actions: list[dict[str, Any]] = []
        rejected_actions: list[dict[str, Any]] = []
        for index, call in enumerate(raw_calls):
            if not isinstance(call, dict):
                continue
            proposal = self._tool_action_proposal(
                call=call,
                index=index,
                stage=stage,
                source=str(plan.get("source") or "unknown"),
                trace_id=trace_id,
                now=now,
            )
            verdict = self._validate_tool_action(
                proposal=proposal,
                game=game,
                missing_fields=missing_fields,
                tool_results=tool_results,
            )
            action_record = {**proposal, "validation": verdict}
            proposals.append(action_record)
            if verdict["allowed"]:
                allowed_calls.append(
                    {
                        "tool_name": proposal["tool_name"],
                        "arguments": verdict.get("effective_arguments") or proposal.get("arguments") or {},
                        "reason": proposal.get("reason") or verdict.get("reason") or "动作已通过后端校验。",
                        "requested_by": proposal.get("proposed_by") or plan.get("source") or "unknown",
                        "action_id": proposal["action_id"],
                        "idempotency_key": proposal["idempotency_key"],
                        "risk_level": proposal.get("risk_level"),
                        "approval_required": proposal.get("approval_required"),
                    }
                )
                validated_actions.append(action_record)
            else:
                rejected_actions.append(action_record)

        result = {
            **plan,
            "stage": stage,
            "tool_calls": allowed_calls,
            "action_protocol": CONTROLLED_AGENT_PROTOCOL_VERSION,
            "action_proposals": proposals,
            "validated_actions": validated_actions,
            "rejected_actions": rejected_actions,
        }
        write_tool_audit_log(
            trace_id,
            "action_validation",
            {
                "protocol": CONTROLLED_AGENT_PROTOCOL_VERSION,
                "stage": stage,
                "source": result.get("source"),
                "proposed_count": len(proposals),
                "allowed_count": len(validated_actions),
                "rejected_count": len(rejected_actions),
                "validated_actions": [
                    self._compact_action_record(item) for item in validated_actions
                ],
                "rejected_actions": [
                    self._compact_action_record(item) for item in rejected_actions
                ],
            },
        )
        return result

    def _tool_action_proposal(
        self,
        *,
        call: dict[str, Any],
        index: int,
        stage: str,
        source: str,
        trace_id: str,
        now: datetime,
    ) -> dict[str, Any]:
        return self.tool_action_proposal_factory.build(
            call=call,
            index=index,
            stage=stage,
            source=source,
            trace_id=trace_id,
            now=now,
        )

    def _validate_tool_action(
        self,
        *,
        proposal: dict[str, Any],
        game: GameRequest | None,
        missing_fields: list[str],
        tool_results: dict[str, Any],
    ) -> dict[str, Any]:
        return self.tool_action_validator.validate(
            proposal=proposal,
            game=game,
            missing_fields=missing_fields,
            tool_results=tool_results,
        )

    def _tool_policy(self, tool_name: str, stage: str) -> dict[str, Any]:
        item = tool_spec_for_stage(tool_name, stage)
        if item:
            return {
                "risk_level": item.get("risk_level") or "unknown",
                "side_effect": bool(item.get("side_effect")),
                "approval_required": bool(item.get("side_effect")) or "approval_required" in str(item.get("policy") or ""),
            }
        return {"risk_level": "unknown", "side_effect": False, "approval_required": False}

    def _compact_action_record(self, item: dict[str, Any]) -> dict[str, Any]:
        validation = item.get("validation") if isinstance(item.get("validation"), dict) else {}
        ledger = item.get("ledger") if isinstance(item.get("ledger"), dict) else {}
        return {
            "action_id": item.get("action_id"),
            "tool_name": item.get("tool_name"),
            "proposed_by": item.get("proposed_by"),
            "risk_level": item.get("risk_level"),
            "approval_required": item.get("approval_required"),
            "allowed": validation.get("allowed"),
            "code": validation.get("code"),
            "reason": validation.get("reason"),
            "notes": validation.get("notes") or [],
            "idempotency_key": item.get("idempotency_key"),
            "ledger_status": ledger.get("status"),
            "deduplicated": bool(ledger.get("duplicate")),
        }

    def _action_plan_view(self, plan: dict[str, Any]) -> dict[str, Any]:
        return {
            "protocol": plan.get("action_protocol") or CONTROLLED_AGENT_PROTOCOL_VERSION,
            "stage": plan.get("stage"),
            "source": plan.get("source"),
            "fallback_used": bool(plan.get("fallback_used")),
            "reasoning_summary": plan.get("reasoning_summary"),
            "llm_source": plan.get("llm_source"),
            "validated_actions": [
                self._compact_action_record(item)
                for item in plan.get("validated_actions") or []
                if isinstance(item, dict)
            ],
            "rejected_actions": [
                self._compact_action_record(item)
                for item in plan.get("rejected_actions") or []
                if isinstance(item, dict)
            ],
        }

    def _tool_requested(self, tool_plan: dict[str, Any], tool_name: str) -> bool:
        return any(
            isinstance(item, dict) and item.get("tool_name") == tool_name
            for item in tool_plan.get("tool_calls") or []
        )

    def _validated_tool_action_record(self, tool_plan: dict[str, Any] | None, tool_name: str) -> dict[str, Any] | None:
        if not isinstance(tool_plan, dict):
            return None
        for item in tool_plan.get("validated_actions") or []:
            if isinstance(item, dict) and item.get("tool_name") == tool_name:
                return item
        return None

    def _replace_action_plan_view(self, action_plans: list[dict[str, Any]], tool_plan: dict[str, Any]) -> None:
        stage = tool_plan.get("stage")
        for index in range(len(action_plans) - 1, -1, -1):
            if action_plans[index].get("stage") == stage:
                action_plans[index] = self._action_plan_view(tool_plan)
                return
        action_plans.append(self._action_plan_view(tool_plan))

    def _tool_request_source(self, tool_plan: dict[str, Any] | None, tool_name: str) -> str:
        if not tool_plan:
            return "backend"
        for item in tool_plan.get("tool_calls") or []:
            if isinstance(item, dict) and item.get("tool_name") == tool_name:
                return str(item.get("requested_by") or tool_plan.get("source") or "llm")
        return str(tool_plan.get("source") or "backend")

    def _execute_tool_gateway(
        self,
        *,
        tool_name: str,
        tool_plan: dict[str, Any] | None,
        request: dict[str, Any],
        operation,
        rejected_result: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        return self.trial_tool_gateway.execute(
            tool_name=tool_name,
            tool_plan=tool_plan,
            request=request,
            operation=operation,
            rejected_result=rejected_result,
        )

    def _search_current_open_games_tool(
        self,
        *,
        trace_id: str,
        query_game: GameRequest | None,
        source_text: str,
        effective_text: str,
        sender_id: str,
        decision_action: str,
        llm_requested: bool | None = None,
        tool_plan: dict[str, Any] | None = None,
        now: datetime,
    ) -> dict[str, Any]:
        text = self._normalize_pool_query_text(f"{source_text}\n{effective_text}")
        should_call = self._should_search_existing_pool(source_text, effective_text, query_game)
        if llm_requested is not None:
            should_call = bool(llm_requested)
        tool_name = "search_current_open_games"
        level_options = self._level_options_from_query(query_game, text, sender_id)
        smoke_preference = self._smoke_preference_from_query(query_game, text)
        smoke_options = self._smoke_options_from_query(query_game, text)
        window = self._time_window_from_query(query_game, text, now)
        query_game_type = self._query_game_type(query_game, text)
        request = self.trial_tool_request_factory.current_open_games(
            called=should_call,
            requested_by=self._tool_request_source(tool_plan, tool_name),
            tool_plan_source=(tool_plan or {}).get("source"),
            decision_action=decision_action,
            call_reason=self._pool_tool_call_reason(source_text, effective_text, query_game, decision_action)
            if should_call
            else "当前语义不是找现成局/可拼局，跳过工具。",
            sender_id=sender_id,
            game_type=query_game_type,
            level_options=level_options,
            smoke_preference=smoke_preference,
            smoke_options=smoke_options,
            time_window=window,
            source_text=source_text,
        )
        write_tool_audit_log(trace_id, "tool_request", request)
        if not should_call:
            result = {**request, "matches": [], "result_count": 0}
            write_tool_audit_log(
                trace_id,
                "tool_response",
                {
                    "tool_name": tool_name,
                    "called": False,
                    "result_count": 0,
                    "matches": [],
                },
            )
            return result

        result, gateway_action = self._execute_tool_gateway(
            tool_name=tool_name,
            tool_plan=tool_plan,
            request=request,
            rejected_result={"matches": [], "result_count": 0},
            operation=lambda: self._search_current_open_games_state_read(
                request=request,
                query_game=query_game,
                source_text=source_text,
                effective_text=effective_text,
                sender_id=sender_id,
                now=now,
            ),
        )
        write_tool_audit_log(
            trace_id,
            "tool_response",
            {
                "tool_name": tool_name,
                "called": bool(result.get("called")),
                "result_count": int(result.get("result_count") or 0),
                "action_id": result.get("action_id") or (gateway_action or {}).get("action_id"),
                "idempotency_key": result.get("idempotency_key") or (gateway_action or {}).get("idempotency_key"),
                "deduplicated": bool(result.get("deduplicated")),
                "rejected": bool(result.get("rejected")),
                "matches": [
                    {
                        "game_id": item.get("game_id"),
                        "summary": item.get("summary"),
                        "score": item.get("score"),
                        "level_match_type": item.get("level_match_type"),
                        "reasons": item.get("reasons"),
                    }
                    for item in list(result.get("matches") or [])[:5]
                ],
            },
        )
        return result

    def _search_current_open_games_state_read(
        self,
        *,
        request: dict[str, Any],
        query_game: GameRequest | None,
        source_text: str,
        effective_text: str,
        sender_id: str,
        now: datetime,
    ) -> dict[str, Any]:
        matches = self._match_existing_games(
            query_game=query_game,
            source_text=source_text,
            effective_text=effective_text,
            sender_id=sender_id,
            now=now,
        )
        return {
            **request,
            "ok": True,
            "called": True,
            "matches": matches,
            "result_count": len(matches),
        }

    def _pool_tool_call_reason(
        self,
        source_text: str,
        effective_text: str,
        query_game: GameRequest | None,
        decision_action: str,
    ) -> str:
        text = self._normalize_pool_query_text(f"{source_text}\n{effective_text}")
        if self._is_near_start_pool_query(text):
            return "用户在问人齐开/现成快开局，应按当前到 90 分钟内搜索当前局池。"
        if re.search(r"有人.*(打|搓|麻将)|有.*(局|桌|麻将)", text):
            return "用户在问有没有现成牌局，应先查当前局池。"
        if decision_action in {"join_game", "find_players"} and re.search(r"下班|晚上|今天|麻将|打牌|有局", text):
            return "语义解析为牌局意向，且像是在找可加入/可拼的局。"
        if query_game and query_game.start_at is None:
            return "组局条件不完整，先查是否已有局可承接，减少重复摇人。"
        return "命中当前局池搜索策略。"

    def _search_candidate_customers_tool(
        self,
        *,
        trace_id: str,
        game: GameRequest,
        now: datetime,
        tool_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        tool_name = "search_candidate_customers"
        composition_preference = self._candidate_composition_preference_from_game(game)
        request = self.trial_tool_request_factory.candidate_customers(
            requested_by=self._tool_request_source(tool_plan, tool_name),
            tool_plan_source=(tool_plan or {}).get("source"),
            game_id=game.id,
            game_type=game.game_type,
            game_label=self._game_label(game),
            level=game.level,
            start_at=game.start_at,
            rules=game.rules,
            missing_count=game.missing_count,
            organizer_id=game.organizer_id,
            candidate_composition_preference=composition_preference,
        )
        write_tool_audit_log(trace_id, "tool_request", request)
        result, gateway_action = self._execute_tool_gateway(
            tool_name=tool_name,
            tool_plan=tool_plan,
            request=request,
            rejected_result={"result_count": 0, "candidates": []},
            operation=lambda: self._search_candidate_customers_state_read(
                request=request,
                game=game,
                now=now,
            ),
        )
        write_tool_audit_log(
            trace_id,
            "tool_response",
            {
                "tool_name": tool_name,
                "called": bool(result.get("called")),
                "result_count": int(result.get("result_count") or 0),
                "action_id": result.get("action_id") or (gateway_action or {}).get("action_id"),
                "idempotency_key": result.get("idempotency_key") or (gateway_action or {}).get("idempotency_key"),
                "deduplicated": bool(result.get("deduplicated")),
                "rejected": bool(result.get("rejected")),
                "candidates": [
                    {
                        "customer_id": item.get("customer_id"),
                        "display_name": item.get("display_name"),
                        "gender": item.get("gender"),
                        "score": item.get("score"),
                        "reasons": item.get("reasons"),
                        "warnings": item.get("warnings"),
                    }
                    for item in list(result.get("candidates") or [])[:8]
                ],
            },
        )
        return result

    def _search_candidate_customers_state_read(
        self,
        *,
        request: dict[str, Any],
        game: GameRequest,
        now: datetime,
    ) -> dict[str, Any]:
        recommendations = self._recommend(game, now)
        candidates = [self._candidate_to_dict(item) for item in recommendations]
        return {
            **request,
            "ok": True,
            "called": True,
            "result_count": len(candidates),
            "candidates": candidates,
        }

    def _send_message_tool(
        self,
        *,
        trace_id: str,
        game: GameRequest,
        recommendations: list[CandidateRecommendation],
        now: datetime,
        tool_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        tool_name = "send_message"
        request = self.trial_tool_request_factory.pending_outbox_message(
            called=bool(recommendations),
            requested_by=self._tool_request_source(tool_plan, tool_name),
            tool_plan_source=(tool_plan or {}).get("source"),
            game_id=game.id,
            recipient_count=len(recommendations[:8]),
        )
        write_tool_audit_log(trace_id, "tool_request", request)
        if not recommendations:
            result = {**request, "result_count": 0, "outbox": []}
            write_tool_audit_log(
                trace_id,
                "tool_response",
                {
                    "tool_name": tool_name,
                    "called": False,
                    "result_count": 0,
                    "outbox": [],
                    "direct_send_executed": False,
                },
            )
            return result

        prepared_send_action = self._validated_tool_action_record(tool_plan, tool_name)
        result, send_action = self._execute_tool_gateway(
            tool_name=tool_name,
            tool_plan=tool_plan,
            request=request,
            rejected_result={
                "result_count": 0,
                "direct_send_executed": False,
                "outbox": [],
                "reason": "send_message 未通过后端动作校验，拒绝创建待审批 outbox。",
            },
            operation=lambda: self._create_pending_outbox_state_write(
                request={
                    **request,
                    "action_id": (prepared_send_action or {}).get("action_id"),
                    "idempotency_key": (prepared_send_action or {}).get("idempotency_key"),
                },
                trace_id=trace_id,
                game=game,
                recommendations=recommendations[:8],
                now=now,
            ),
        )
        result = {**request, **result}
        write_tool_audit_log(
            trace_id,
            "tool_response",
            {
                "tool_name": tool_name,
                "called": bool(result.get("called", True)),
                "result_count": int(result.get("result_count") or 0),
                "approval_required": True,
                "direct_send_executed": False,
                "action_id": result.get("action_id"),
                "idempotency_key": result.get("idempotency_key"),
                "deduplicated": bool(result.get("deduplicated")),
                "outbox": [
                    {
                        "id": item.get("id"),
                        "game_id": item.get("game_id"),
                        "customer_id": item.get("customer_id"),
                        "customer_name": item.get("customer_name"),
                        "approval_status": item.get("approval_status"),
                        "draft_source": item.get("draft_source"),
                    }
                    for item in list(result.get("outbox") or [])[:8]
                ],
            },
        )
        return result

    def _create_pending_outbox_state_write(
        self,
        *,
        request: dict[str, Any],
        trace_id: str,
        game: GameRequest,
        recommendations: list[CandidateRecommendation],
        now: datetime,
    ) -> dict[str, Any]:
        action_id = str(request.get("action_id") or "")
        idempotency_key = str(request.get("idempotency_key") or "")
        risk_level = str(request.get("risk_level") or "high")
        invite_drafts = self._candidate_invite_drafts(
            trace_id=trace_id,
            game=game,
            recommendations=recommendations,
            now=now,
        )
        outbox: list[dict[str, Any]] = []
        for recommendation in recommendations:
            customer = self.store.customer(recommendation.customer_id) or {}
            gender = normalize_gender(customer.get("gender"))
            draft = invite_drafts.get(recommendation.customer_id) or {}
            message_text = str(draft.get("message_text") or "").strip()
            if not message_text:
                message_text = self._private_invite_text(game, recommendation.display_name)
            outbox_id = self.store.create_outbox(
                game_id=game.id,
                customer_id=recommendation.customer_id,
                customer_name=recommendation.display_name,
                message_text=message_text,
                score=recommendation.score,
                reasons=recommendation.reasons,
                warnings=recommendation.warnings,
            )
            approval = self.store.create_approval_request(
                target_type="outbox",
                target_id=outbox_id,
                action_id=action_id,
                idempotency_key=idempotency_key,
                risk_level=risk_level,
                original_message_text=message_text,
                metadata={
                    "game_id": game.id,
                    "customer_id": recommendation.customer_id,
                    "customer_name": recommendation.display_name,
                    "draft_source": draft.get("source") or "rules",
                    "tool_name": "send_message",
                    "execution_mode": "create_pending_outbox",
                },
            )
            outbox.append(
                {
                    "id": outbox_id,
                    "game_id": game.id,
                    "customer_id": recommendation.customer_id,
                    "customer_name": recommendation.display_name,
                    "gender": gender,
                    "gender_label": GENDER_LABELS.get(gender, "未知"),
                    "message_text": message_text,
                    "status": "待审批",
                    "approval_status": approval_status_label(approval.get("status")),
                    "approval": approval,
                    "approval_required": True,
                    "direct_send_executed": False,
                    "draft_source": draft.get("source") or "rules",
                    "draft_reasoning": draft.get("reasoning_summary") or "",
                    "score": recommendation.score,
                    "reasons": recommendation.reasons,
                    "warnings": recommendation.warnings,
                }
            )
        return {
            **request,
            "ok": True,
            "called": True,
            "result_count": len(outbox),
            "direct_send_executed": False,
            "outbox": outbox,
        }

    def _candidate_recommendations_from_tool(self, tool_result: dict[str, Any]) -> list[CandidateRecommendation]:
        recommendations: list[CandidateRecommendation] = []
        for item in tool_result.get("candidates") or []:
            if not isinstance(item, dict):
                continue
            customer_id = str(item.get("customer_id") or "").strip()
            display_name = str(item.get("display_name") or "").strip()
            if not customer_id or not display_name:
                continue
            recommendations.append(
                CandidateRecommendation(
                    customer_id=customer_id,
                    display_name=display_name,
                    score=float(item.get("score") or 0),
                    reasons=[str(value) for value in item.get("reasons") or []],
                    warnings=[str(value) for value in item.get("warnings") or []],
                )
            )
        return recommendations

    def _skipped_tool_result(
        self,
        tool_name: str,
        reason: str,
        *,
        risk_level: str = "low",
        approval_required: bool = False,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "tool_name": tool_name,
            "called": False,
            "risk_level": risk_level,
            "approval_required": approval_required,
            "call_reason": reason,
            "result_count": 0,
        }
        if tool_name == "search_candidate_customers":
            result["candidates"] = []
        if tool_name == "send_message":
            result["direct_send_allowed"] = False
            result["direct_send_executed"] = False
            result["execution_mode"] = "not_called"
            result["outbox"] = []
        return result

    def _rejected_tool_result(
        self,
        trace_id: str,
        tool_name: str,
        reason: str,
        *,
        risk_level: str = "low",
        approval_required: bool = False,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "tool_name": tool_name,
            "called": False,
            "rejected": True,
            "risk_level": risk_level,
            "approval_required": approval_required,
            "validation_error": reason,
            "result_count": 0,
        }
        if tool_name == "search_candidate_customers":
            result["candidates"] = []
        if tool_name == "send_message":
            result["direct_send_allowed"] = False
            result["direct_send_executed"] = False
            result["execution_mode"] = "rejected"
            result["outbox"] = []
        write_tool_audit_log(trace_id, "tool_rejected", result)
        return result

    def _match_existing_games(
        self,
        *,
        query_game: GameRequest | None,
        source_text: str,
        effective_text: str,
        sender_id: str,
        now: datetime,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        if not self._should_search_existing_pool(source_text, effective_text, query_game):
            return []
        text = self._normalize_pool_query_text(f"{source_text}\n{effective_text}")
        level_options = self._level_options_from_query(query_game, text, sender_id)
        explicit_levels = self._explicit_level_options_from_text(text)
        smoke_preference = self._smoke_preference_from_query(query_game, text)
        window = self._time_window_from_query(query_game, text, now)
        query_game_type = self._query_game_type(query_game, text)

        matches: list[dict[str, Any]] = []
        for row in self.store.games():
            if str(row.get("organizer_id") or "") == sender_id:
                continue
            parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}
            if not parsed:
                continue
            if str(row.get("status") or "") == "已满":
                continue
            start_at = parse_dt(str(parsed.get("start_at") or ""))
            if start_at is None or start_at < now - timedelta(minutes=30):
                continue
            open_slots = self._open_slots_for_existing_game(row)
            if open_slots <= 0:
                continue
            level = str(parsed.get("level") or "").strip()
            if level_options and level and level not in level_options:
                continue
            if not self._smoke_compatible(smoke_preference, list(parsed.get("rules") or [])):
                continue
            game_type = str(parsed.get("game_type") or "")
            if query_game_type and game_type and query_game_type != game_type:
                continue

            score = 0
            reasons: list[str] = []
            if query_game_type and game_type == query_game_type:
                score += 20
                reasons.append(f"玩法匹配{GAME_TYPE_LABELS.get(game_type, game_type)}")
            level_match_type = "unknown"
            if level:
                if explicit_levels and level in explicit_levels:
                    score += 35
                    level_match_type = "exact"
                    reasons.append(f"档位精确匹配 {level}")
                elif explicit_levels and level_options and level in level_options:
                    score += 16
                    level_match_type = "profile_fallback"
                    reasons.append(f"画像可接受 {level}")
                elif not level_options or level in level_options:
                    score += 26
                    level_match_type = "compatible"
                    reasons.append(f"档位匹配 {level}")
            if self._time_compatible(start_at, window):
                score += 25
                reasons.append("时间段匹配")
            if self._smoke_compatible(smoke_preference, list(parsed.get("rules") or [])):
                score += 15
                reasons.append(self._smoke_match_reason(smoke_preference, list(parsed.get("rules") or [])))
            score += min(10, open_slots * 3)
            reasons.append(f"还缺 {open_slots} 人")

            match = {
                "game_id": row.get("id"),
                "status": row.get("status"),
                "summary": parsed.get("summary") or self._existing_game_summary(parsed, open_slots),
                "game_label": parsed.get("game_label") or GAME_TYPE_LABELS.get(game_type, game_type),
                "start_time": start_at.strftime("%H:%M"),
                "start_at": start_at.isoformat(),
                "level": level,
                "level_match_type": level_match_type,
                "requested_levels": explicit_levels,
                "rules": list(parsed.get("rules") or []),
                "missing_count": open_slots,
                "organizer_id": row.get("organizer_id"),
                "organizer_name": row.get("organizer_name"),
                "score": score,
                "reasons": [reason for reason in reasons if reason],
            }
            match["reply_text"] = self._pool_match_reply(match)
            matches.append(match)
        return sorted(matches, key=lambda item: item["score"], reverse=True)[:limit]

    def _should_search_existing_pool(
        self,
        source_text: str,
        effective_text: str,
        query_game: GameRequest | None,
    ) -> bool:
        text = self._normalize_pool_query_text(f"{source_text}\n{effective_text}")
        if self._is_explicit_grouping_request(source_text, effective_text, query_game):
            return False
        if self._is_near_start_pool_query(text):
            return True
        if self._is_pool_inquiry_text(text):
            return True
        return bool(
            query_game
            and query_game.start_at is None
            and (query_game.level or "烟况都可" in query_game.rules or re.search(r"0\.5|一块|1块|无烟|有烟", text))
            and re.search(r"有人|有局|麻将|打牌|搓", text)
        )

    def _is_pool_inquiry_text(self, text: str) -> bool:
        normalized = self._normalize_pool_query_text(text)
        if self._is_near_start_pool_query(normalized):
            return True
        return bool(
            re.search(
                r"(有没有|有没|有没有人|有人吗|有人.*(?:打|搓|麻将|牌)|有.*(?:局|桌|麻将)|"
                r"现成局|通常局|普通局|下班.*麻将|晚上.*麻将|想.*(?:打|搓).*(?:麻将|牌))",
                normalized,
            )
        )

    def _is_explicit_grouping_request(
        self,
        source_text: str,
        effective_text: str,
        game: GameRequest | None,
    ) -> bool:
        text = self._normalize_pool_query_text(f"{source_text}\n{effective_text}")
        explicit_action = bool(
            re.search(
                r"(帮我|帮忙|给我|替我|麻烦).{0,12}(组|找|摇|约|问).{0,8}(人|局|桌|搭子|选手)?|"
                r"(组一桌|组一个|组个|组个局|开一桌|摇人|摇下人|帮.*问人|帮.*找人|给我.*组)",
                text,
            )
        )
        if explicit_action:
            return True
        if self._is_pool_inquiry_text(text):
            return False
        has_explicit_party = self._has_explicit_party_size(text)
        if not has_explicit_party:
            return False
        has_grouping_signal = bool(
            re.search(r"麻将|杭麻|财敲|川麻|红中|捉鸡|幺鸡|0\.5|一块|1块|两块|2块|无烟|有烟|\d{1,2}[:.点]\d{0,2}", text)
        )
        return has_grouping_signal or bool(game and (game.level or game.start_at or game.game_type != "mahjong"))

    def _level_options_from_query(
        self,
        query_game: GameRequest | None,
        text: str,
        sender_id: str,
    ) -> list[str]:
        levels = self._explicit_level_options_from_text(text)
        if levels and self._pool_search_allows_profile_level_fallback(text):
            levels.extend(self._compatible_profile_level_fallbacks(sender_id, levels))
            return self._unique_strings(levels)
        if levels:
            return levels
        if query_game and query_game.level:
            return [query_game.level]
        customer = self.store.customer(sender_id)
        if customer:
            return self._unique_strings([str(item) for item in customer.get("preferred_levels") or []])[:3]
        return []

    def _explicit_level_options_from_text(self, text: str) -> list[str]:
        text = self._normalize_pool_query_text(text)
        text = self._strip_duration_expressions_for_level_parse(text)
        levels: list[str] = []
        aliases = {
            "五毛": "0.5",
            "半块": "0.5",
            "半": "0.5",
            "一块": "1",
            "1块": "1",
            "两块": "2",
            "2块": "2",
        }
        for label, level in aliases.items():
            if label in text:
                levels.append(level)
        for match in re.finditer(r"(?<![\d:：])(\d+(?:\.\d+)?(?:-\d+)?)(?!\d)", text):
            value = match.group(1)
            after = text[match.end() : match.end() + 1]
            if after in {"点", "时", ":"}:
                continue
            if "-" not in value:
                try:
                    numeric_value = float(value)
                    if numeric_value <= 0 or numeric_value > 5:
                        continue
                except ValueError:
                    continue
            levels.append(value)
        return self._unique_strings(levels)

    def _strip_duration_expressions_for_level_parse(self, text: str) -> str:
        text = re.sub(r"\d+(?:\.\d+)?\s*(?:个)?\s*(?:小时|钟头|h|H)", " ", text)
        text = re.sub(r"[一二两三四五六七八九十半]+(?:个)?\s*(?:小时|钟头)", " ", text)
        return text

    def _pool_search_allows_profile_level_fallback(self, text: str) -> bool:
        text = self._normalize_pool_query_text(text)
        if re.search(r"只(?:想)?打|就(?:想)?打|不要|不打|别.*一块|别.*1块|必须", text):
            return False
        return self._is_near_start_pool_query(text)

    def _compatible_profile_level_fallbacks(self, sender_id: str, explicit_levels: list[str]) -> list[str]:
        customer = self.store.customer(sender_id)
        if not customer:
            return []
        explicit_numeric: list[float] = []
        for item in explicit_levels:
            if "-" in item:
                continue
            try:
                explicit_numeric.append(float(item))
            except ValueError:
                continue
        fallbacks: list[str] = []
        for item in customer.get("preferred_levels") or []:
            level = str(item or "").strip()
            if not level or level in explicit_levels or "-" in level:
                continue
            try:
                value = float(level)
            except ValueError:
                continue
            if not explicit_numeric or any(abs(value - requested) <= 0.5 for requested in explicit_numeric):
                fallbacks.append(level)
            if len(fallbacks) >= 2:
                break
        return fallbacks

    def _smoke_options_from_query(self, query_game: GameRequest | None, text: str) -> list[str]:
        text = self._normalize_pool_query_text(text)
        preference = self._smoke_preference_from_query(query_game, text)
        if preference == "any":
            return ["无烟", "可吸烟"]
        if preference == "no_smoke":
            return ["无烟"]
        if preference == "smoke_ok":
            return ["可吸烟"]
        return []

    def _smoke_preference_from_query(self, query_game: GameRequest | None, text: str) -> str:
        text = self._normalize_pool_query_text(text)
        if re.search(r"烟.*(都可|都行|都可以|随便|无所谓)|有烟无烟.*(都可|都行)|可烟|烟也都可", text):
            return "any"
        if "烟况都可" in (query_game.rules if query_game else []):
            return "any"
        if re.search(r"无烟|不抽|禁烟", text) or (query_game and "无烟" in query_game.rules):
            return "no_smoke"
        if re.search(r"有烟|可吸烟|能抽|抽烟", text) or (query_game and "可吸烟" in query_game.rules):
            return "smoke_ok"
        return "unknown"

    def _time_window_from_query(
        self,
        query_game: GameRequest | None,
        text: str,
        now: datetime,
    ) -> tuple[datetime, datetime] | None:
        text = self._normalize_pool_query_text(text)
        if query_game and query_game.start_at:
            return query_game.start_at - timedelta(minutes=90), query_game.start_at + timedelta(minutes=90)
        if self._is_near_start_pool_query(text):
            return now, now + timedelta(minutes=90)
        day = now.date()
        if re.search(r"下班|晚上|今晚", text):
            return (
                datetime(day.year, day.month, day.day, 17, 0, tzinfo=TZ),
                datetime(day.year, day.month, day.day, 23, 30, tzinfo=TZ),
            )
        if "下午" in text:
            return (
                datetime(day.year, day.month, day.day, 12, 0, tzinfo=TZ),
                datetime(day.year, day.month, day.day, 18, 30, tzinfo=TZ),
            )
        if re.search(r"上午|早上|早晨", text):
            return (
                datetime(day.year, day.month, day.day, 8, 0, tzinfo=TZ),
                datetime(day.year, day.month, day.day, 12, 30, tzinfo=TZ),
            )
        return None

    def _is_near_start_pool_query(self, text: str) -> bool:
        text = self._normalize_pool_query_text(text)
        return bool(
            re.search(
                r"(人齐\s*开|齐开|人齐|现成局|现成|马上.*(开|打|搓)|一会.*(开|打|搓)|"
                r"现在.*(有|有没有).*(局|桌|麻将|人.*打|人齐)|有没有.*(人齐|齐开))",
                text,
            )
        )

    def _normalize_pool_query_text(self, text: str) -> str:
        return normalize_mahjong_text(text).text

    def _text_normalization_for_prompt(self, source_text: str, effective_text: str = "") -> dict[str, Any]:
        raw = str(effective_text or source_text or "")
        result = normalize_mahjong_text(raw)
        return {
            "raw_text": result.raw_text,
            "normalized_text": result.text,
            "changed": result.text != result.raw_text,
            "changed_rule_ids": result.changed_rule_ids(),
            "changes": [
                {
                    "rule_id": change.rule_id,
                    "before": change.before,
                    "after": change.after,
                    "reason": change.reason,
                }
                for change in result.changes[:8]
            ],
            "policy": "这是低风险文本标准化证据，不是业务事实；模型仍需要结合原文、客户画像、局池和上下文判断真实语义。",
        }

    def _time_compatible(self, start_at: datetime, window: tuple[datetime, datetime] | None) -> bool:
        if window is None:
            return True
        return window[0] <= start_at <= window[1]

    def _query_game_type(self, query_game: GameRequest | None, text: str) -> str | None:
        if query_game and query_game.game_type and query_game.game_type != "mahjong":
            return query_game.game_type
        if self._source_mentions_non_default_play(text):
            if re.search(r"川麻|四川麻|换三张|定缺|幺鸡|妖鸡|素鸡", text):
                return "sichuan_mahjong"
            if "红中" in text:
                return "hongzhong_mahjong"
            if "捉鸡" in text:
                return "zhuoji_mahjong"
            if "湖南" in text:
                return "hunan_mahjong"
        return "hangzhou_mahjong"

    def _open_slots_for_existing_game(self, row: dict[str, Any]) -> int:
        parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}
        try:
            missing_count = int(parsed.get("missing_count") or 0)
        except (TypeError, ValueError):
            missing_count = 0
        confirmed = sum(
            1
            for item in row.get("outbox") or []
            if str(item.get("status") or "") in {"已确认", "已到店"}
        )
        return max(0, missing_count - confirmed)

    def _smoke_compatible(self, preference: str, rules: list[str]) -> bool:
        if preference in {"unknown", "any"}:
            return True
        if preference == "no_smoke":
            return "无烟" in rules or "烟况都可" in rules
        if preference == "smoke_ok":
            return "可吸烟" in rules or "有烟" in rules or "烟况都可" in rules
        return True

    def _smoke_match_reason(self, preference: str, rules: list[str]) -> str:
        if preference == "any":
            return "烟况都可匹配"
        if preference == "no_smoke" and "无烟" in rules:
            return "无烟匹配"
        if preference == "smoke_ok" and ("可吸烟" in rules or "有烟" in rules):
            return "可烟匹配"
        return "烟况不冲突"

    def _pool_match_reply(self, match: dict[str, Any]) -> str:
        level = str(match.get("level") or "").strip()
        smoke = self._smoke_text_from_rules(list(match.get("rules") or []))
        level_text = self._level_text_for_reply(level)
        level_smoke = "".join(part for part in [level_text, smoke] if part)
        pieces = [str(match.get("start_time") or "").strip(), level_smoke]
        core = " ".join(part for part in pieces if part)
        missing = match.get("missing_count")
        missing_text = self._missing_text_for_reply(missing)
        requested_levels = [str(item) for item in match.get("requested_levels") or [] if str(item).strip()]
        if match.get("level_match_type") == "profile_fallback" and requested_levels and core:
            requested = "/".join(self._level_text_for_reply(item) for item in requested_levels)
            return f"{requested}的暂时没有诶，{core}的，{missing_text}。"
        if core:
            return f"有的，{core}，{missing_text}。要我帮你确认吗？"
        return f"有个局比较合适，{missing_text}。要我帮你确认吗？"

    def _pool_no_match_reply(self, source_text: str, effective_text: str, sender_id: str) -> str:
        text = f"{source_text}\n{effective_text}"
        requested = self._explicit_level_options_from_text(text)
        if requested:
            level_text = "/".join(self._level_text_for_reply(item) for item in requested)
            return f"{level_text}的暂时没有诶。要组一个吗？"
        return "现在没有诶，要组一个吗？"

    def _level_text_for_reply(self, level: str) -> str:
        if level == "1":
            return "1块"
        if level == "2":
            return "2块"
        return level

    def _missing_text_for_reply(self, missing: Any) -> str:
        try:
            count = int(missing)
        except (TypeError, ValueError):
            count = 0
        return {1: "三缺一了", 2: "二缺二了", 3: "一缺三了"}.get(count, f"还缺 {count} 个" if count else "人数快齐了")

    def _smoke_text_from_rules(self, rules: list[str]) -> str:
        if "无烟" in rules:
            return "无烟"
        if "可吸烟" in rules or "有烟" in rules:
            return "有烟"
        if "烟况都可" in rules:
            return "烟都可"
        return ""

    def _existing_game_summary(self, parsed: dict[str, Any], open_slots: int) -> str:
        parts = [
            str(parsed.get("game_label") or "").strip(),
            str(parsed.get("level") or "").strip(),
            str(parsed.get("start_time") or "").strip(),
            f"缺{open_slots}" if open_slots else "",
            self._smoke_text_from_rules(list(parsed.get("rules") or [])),
        ]
        return " ".join(part for part in parts if part)

    def _unique_strings(self, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            item = str(value or "").strip()
            if item and item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def _customer_profile_for_prompt(self, sender_id: str | None) -> dict[str, Any]:
        if not sender_id:
            return {}
        customer = self.responder.core.store.customers.get(sender_id)
        if customer is None:
            return {}
        default_assumption = self._customer_default_assumption(customer)
        return {
            "id": customer.id,
            "display_name": customer.display_name,
            "preferred_levels": list(customer.preferred_levels[:8]),
            "default_assumption": default_assumption,
            "play_preferences": [
                {
                    "game_type": preference.game_type,
                    "preferred_levels": list(preference.preferred_levels[:8]),
                    "preferred_rulesets": list(preference.preferred_rulesets[:8]),
                    "preferred_variants": list(preference.preferred_variants[:8]),
                    "preferred_play_options": list(preference.preferred_play_options[:12]),
                }
                for preference in customer.play_preferences[:8]
            ],
            "tags": list(customer.tags[:20]),
            "smoke_free_preference": customer.smoke_free_preference,
            "usual_party_size": customer.usual_party_size,
            "usual_party_size_confidence": customer.usual_party_size_confidence,
            "usual_start_hours": list(customer.usual_start_hours[:12]),
            "metadata": {
                key: customer.metadata.get(key)
                for key in ["response_speed", "response_rate", "notes"]
                if key in customer.metadata
            },
        }

    def _customer_default_assumption(self, customer: CustomerProfile) -> dict[str, Any]:
        default_region = os.environ.get("MAHJONG_DEFAULT_REGION", "hangzhou").strip().lower()
        default_game_type = {
            "hangzhou": "hangzhou_mahjong",
            "hz": "hangzhou_mahjong",
            "杭州": "hangzhou_mahjong",
            "sichuan": "sichuan_mahjong",
            "sc": "sichuan_mahjong",
            "四川": "sichuan_mahjong",
            "成都": "sichuan_mahjong",
        }.get(default_region, "hangzhou_mahjong")
        default_region_label = "四川" if default_game_type == "sichuan_mahjong" else "杭州"
        preferences = [preference for preference in customer.play_preferences if preference.game_type != "mahjong"]
        preferred = next((preference for preference in preferences if preference.game_type == default_game_type), None)
        if preferred is None and len({preference.game_type for preference in preferences}) == 1:
            preferred = preferences[0]
        game_type = preferred.game_type if preferred else default_game_type
        variant = preferred.preferred_variants[0] if preferred and len(preferred.preferred_variants) == 1 else None
        game_label = GAME_TYPE_LABELS.get(game_type, game_type)
        variant_label = VARIANT_LABELS.get(variant or "")
        if variant_label and variant_label not in game_label:
            game_label = f"{game_label}{variant_label}"
        level = None
        if preferred and preferred.preferred_levels:
            level = preferred.preferred_levels[0]
        elif customer.preferred_levels:
            level = customer.preferred_levels[0]
        return {
            "default_region": default_region_label,
            "default_region_game_type": default_game_type,
            "game_type": game_type,
            "game_label": game_label,
            "level": level,
            "reason": f"{default_region_label}门店未明确玩法时默认按{GAME_TYPE_LABELS.get(default_game_type, default_game_type)}理解；非默认玩法通常需要用户明说。",
            "reply_style": "未明确玩法时用确认式说法，例如“还是按老样子/杭麻给你看”，不要问“杭麻还是川麻”。",
        }

    def _suggested_reply(
        self,
        *,
        source_text: str,
        effective_text: str,
        trace_id: str,
        sender_id: str,
        sender_name: str,
        game: GameRequest | None,
        workflow_followup_context: dict[str, Any] | None,
        missing_fields: list[str],
        decision_reply: str,
        recommendations: list[CandidateRecommendation],
        outbox: list[dict[str, Any]],
        pool_matches: list[dict[str, Any]],
        tool_results: dict[str, Any],
        now: datetime,
    ) -> dict[str, Any]:
        rule_decision = self.trial_reply_rule_policy.decide(
            TrialReplyRulePolicyInput(
                source_text=source_text,
                effective_text=effective_text,
                sender_id=sender_id,
                sender_name=sender_name,
                game=game,
                workflow_followup_context=workflow_followup_context,
                missing_fields=missing_fields,
                decision_reply=decision_reply,
                recommendations=recommendations,
                outbox=outbox,
                pool_matches=pool_matches,
                tool_results=tool_results,
            )
        )
        fallback = rule_decision.fallback
        fallback_text = str(fallback.get("text") or "")
        if rule_decision.skip_llm:
            return fallback
        llm_result = self._llm_suggested_reply(
            source_text=source_text,
            effective_text=effective_text,
            trace_id=trace_id,
            sender_id=sender_id,
            sender_name=sender_name,
            game=game,
            workflow_followup_context=workflow_followup_context,
            missing_fields=missing_fields,
            recommendations=recommendations,
            outbox=outbox,
            pool_matches=pool_matches,
            tool_results=tool_results,
            fallback=fallback_text,
            now=now,
        )
        return llm_result or fallback

    def _llm_suggested_reply(
        self,
        *,
        source_text: str,
        effective_text: str,
        trace_id: str,
        sender_id: str,
        sender_name: str,
        game: GameRequest | None,
        workflow_followup_context: dict[str, Any] | None,
        missing_fields: list[str],
        recommendations: list[CandidateRecommendation],
        outbox: list[dict[str, Any]],
        pool_matches: list[dict[str, Any]],
        tool_results: dict[str, Any],
        fallback: str,
        now: datetime,
    ) -> dict[str, Any] | None:
        if not self.llm_config or not self.llm_budget_manager:
            return None
        max_tokens = min(self.llm_config.max_completion_tokens, 180)
        system_prompt = """你是麻将馆老板的微信回复起草助手。
你只能生成“待老板审批”的建议回复，不能说已经私聊、已经群发、已经确认房间。
不能承诺优惠、不能处理抽水/赌资/结算/收付款。
回复要自然、短、像真实老板发给客户的话。
可以利用客户画像中的稳定偏好来组织问法，比如“还是按你常打的0.5？”“今天还是杭麻吗？”。
本店默认地区是杭州：用户没有明确说川麻/换三张/定缺/幺鸡时，不要问“杭麻还是川麻”，默认按杭麻或客户画像里的杭麻细分来确认。
如果客户画像里同时有杭麻和川麻，也不要把二者平铺给客户选；除非客户原话表达“玩法都可以/想换玩法/川麻”，否则按杭州默认和高频偏好组织回复。
不要暴露系统内部字段，不要说“缺失字段、槽位、当前人数、档位、烟况”这类产品/后台词。
如果 matched_existing_games 不为空，说明后端已经在当前局池里找到了可拼/可加入的局，优先回复已有局；不要继续问“大概几点”“打多大”“有烟无烟”等已经可由匹配局覆盖的问题。
回复已有局时，只能简短提时间、档位、烟况、还缺几人，并询问是否要帮他确认；不要说评分、推荐原因、候选人姓名。
如果 matched_existing_games 有多条，从中选择一条最适合当前用户语义的局，并在 selected_pool_game_id 里返回它的 game_id；不能编造工具结果之外的 game_id。
回复优先级：missing_fields 非空时，必须先自然追问缺的信息；这时不要套用“现在没有，要不要组一个”的现有局无匹配话术。
只有 missing_fields 为空、matched_existing_games 为空、且当前只是咨询现有局时，search_current_open_games result_count=0 才回复“暂时没有，要不要帮你组一个”。
prompt.text_normalization 是后端提供的低风险文本标准化证据，不是业务事实；例如“0。5/0，5/0 5/0、5”在麻将语境和画像支持时可按 0.5 理解，但不确定时要自然追问。
工具语义：search_current_open_games 和 search_candidate_customers 是只读搜索工具；send_message 是高风险工具，当前只能创建待审批 outbox，不能直接发送。
如果 tool_results.send_message.direct_send_executed=false，不要说“已经发了/已经通知了/已经私聊了”，只能说“我帮你问问/我先帮你确认”。
如果用户已经明确表达“帮我组一桌/帮忙找人/摇人”，且时间、玩法、档位、人数、烟况等关键信息已经足够，就只做极简确认，例如“好的，我帮你问问。”；不要复述杭麻/财敲/0.5/两点/无烟等条件。
“帮我组一桌/组一桌”不代表客户已经有三个人；如果 missing_fields 包含 known_players，要自然追问人数，比如“你一个人吗？”，不要说已经帮他问人。
如果 missing_fields 包含 start_time，且 parsed_game.ambiguities 提示时间已经过了或上午下午不明确，只能追问时间，例如“你说的两点是明天吗，还是改其他时间？”，不能说已经帮他问人。
信息不全时，一条回复里最多问 3 个关键问题；优先问会影响能不能组局的问题。
参考 few_shot_examples 的语气和边界，但不要照抄；有 conditions 时先判断，不满足不能套用。
reasoning_summary 只能写一句简短判断依据，不要输出长篇思维链。
只输出 JSON：
{"reply_text":"一句可复制给当前客户的回复","selected_pool_game_id":"可选，只能来自工具返回的 game_id","risk_level":"low|medium|high","reasoning_summary":"一句话说明为什么这样回复","notes":["简短说明"]}"""
        prompt = {
            "task": "根据客户原话、系统解析结果、缺失字段和候选人，起草给当前客户的一条建议回复。",
            "now": now.strftime("%Y-%m-%d %H:%M:%S"),
            "few_shot_examples": self._few_shot_examples(),
            "active_skills": self._active_skills(
                stage="reply_draft",
                source_text=source_text,
                effective_text=effective_text,
                game=game,
            ),
            "sender_name": sender_name,
            "customer_profile": self._customer_profile_for_prompt(sender_id),
            "source_text": source_text,
            "effective_text": effective_text,
            "workflow_followup_context": workflow_followup_context or {},
            "text_normalization": self._text_normalization_for_prompt(source_text, effective_text),
            "parsed_game": self._game_to_dict(game) if game else {},
            "missing_fields": missing_fields,
            "reply_style_hint": self._reply_style_hint(game, missing_fields, source_text, effective_text),
            "tool_results": self._tool_results_for_prompt(tool_results),
            "matched_existing_games": pool_matches[:5],
            "candidate_recommendations": [
                {
                    "name": item.display_name,
                    "score": item.score,
                    "reasons": item.reasons[:4],
                    "warnings": item.warnings[:3],
                }
                for item in recommendations[:6]
            ],
            "pending_invites": [
                {
                    "customer_name": item.get("customer_name"),
                    "approval_status": item.get("approval_status") or item.get("status"),
                    "message_text": item.get("message_text"),
                }
                for item in outbox[:6]
            ],
            "fallback_reply": fallback,
            "rules": [
                "如果信息不全，按老板口吻自然追问，不要列出字段名。",
                "优先参考 active_skills；skill 是业务经验，最终回复仍必须遵守安全边界和审批状态。",
                "如果 reply_style_hint.mode 是 brief_ack，只输出极简确认，不要复述已明确条件。",
                "如果 matched_existing_games 不为空，优先基于最匹配的现有局回复，不要先追问时间。",
                "如果工具 search_current_open_games 已返回候选局，只能在这些候选局里选择，不要创造新局。",
                "如果 missing_fields 非空，优先追问缺的信息；不要回复“现在没有，要组一个吗”。",
                "只有 missing_fields 为空且用户只是在问有没有现成局时，工具 search_current_open_games 无候选局才回复“现在没有，要组一个吗”。",
                "search_candidate_customers 只表示找到了候选人，不代表已经联系。",
                "send_message 当前只能创建待审批 outbox；direct_send_executed=false 时禁止说已经发送。",
                "如果客户画像有稳定偏好，可以用“老样子/常打的”进行确认式追问，但不能强行说已经确认。",
                "杭州门店里，客户没说川麻时默认杭麻；不要问“杭麻还是川麻”。",
                "如果信息已足够且有候选人，告诉客户先帮他问人，但不要说已经发送。",
                "如果没有候选人，说明先帮他留意相近局。",
                "不要输出候选人的内部评分。",
            ],
        }
        payload = {
            "model": self.llm_config.model,
            "temperature": min(self.llm_config.temperature, 0.4),
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        }
        if self.llm_config.thinking_enabled is not None:
            payload["thinking"] = {"type": "enabled" if self.llm_config.thinking_enabled else "disabled"}
        if self.llm_config.response_format:
            payload["response_format"] = {"type": self.llm_config.response_format}

        budget_decision = self.llm_budget_manager.reserve(
            key="boss_trial_draft",
            model=self.llm_config.model,
            prompt=payload,
            max_completion_tokens=max_tokens,
        )
        if not budget_decision.allowed:
            write_llm_audit_log(
                trace_id,
                "llm_budget_denied",
                {
                    "stage": "reply_draft",
                    "provider": self.llm_config.provider,
                    "model": self.llm_config.model,
                    "budget": budget_decision.to_dict(),
                },
            )
            return {
                "text": fallback,
                "source": "rules",
                "model": self.llm_config.model,
                "needs_approval": True,
                "status": "待审批",
                "notes": [f"LLM 预算不足，使用规则兜底：{budget_decision.reason}"],
                "budget": budget_decision.to_dict(),
            }

        write_llm_audit_log(
            trace_id,
            "llm_request",
            {
                "stage": "reply_draft",
                "provider": self.llm_config.provider,
                "model": self.llm_config.model,
                "base_url": self.llm_config.base_url,
                "timeout_seconds": self.llm_config.timeout_seconds,
                "budget": budget_decision.to_dict(),
                "payload": payload,
            },
        )

        request = urllib.request.Request(
            f"{self.llm_config.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.llm_config.api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.llm_config.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            write_llm_audit_log(
                trace_id,
                "llm_error",
                {
                    "stage": "reply_draft",
                    "provider": self.llm_config.provider,
                    "model": self.llm_config.model,
                    "error": f"HTTP {exc.code}",
                    "budget": budget_decision.to_dict(),
                },
            )
            return {
                "text": fallback,
                "source": "rules",
                "model": self.llm_config.model,
                "needs_approval": True,
                "status": "待审批",
                "notes": [f"LLM 起草失败，使用规则兜底：HTTP {exc.code}"],
                "budget": budget_decision.to_dict(),
            }
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            write_llm_audit_log(
                trace_id,
                "llm_error",
                {
                    "stage": "reply_draft",
                    "provider": self.llm_config.provider,
                    "model": self.llm_config.model,
                    "error": f"{type(exc).__name__}: {exc}",
                    "budget": budget_decision.to_dict(),
                },
            )
            return {
                "text": fallback,
                "source": "rules",
                "model": self.llm_config.model,
                "needs_approval": True,
                "status": "待审批",
                "notes": [f"LLM 起草失败，使用规则兜底：{type(exc).__name__}"],
                "budget": budget_decision.to_dict(),
            }

        actual_usage = usage_from_response(data, self.llm_config.model)
        self.llm_budget_manager.commit(budget_decision.reservation_id, actual_usage)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        write_llm_audit_log(
            trace_id,
            "llm_response",
            {
                "stage": "reply_draft",
                "provider": self.llm_config.provider,
                "model": self.llm_config.model,
                "raw_response": data,
                "content": content,
                "usage": actual_usage.to_dict() if actual_usage else None,
            },
        )
        parsed = self._parse_llm_reply_json(
            content,
            trace_id=trace_id,
            stage="reply_draft",
            schema_hint='{"reply_text":str,"selected_pool_game_id":str|null,"risk_level":str,"reasoning_summary":str,"notes":[]}',
        )
        reply_text = str(parsed.get("reply_text") or "").strip()
        if not reply_text:
            return {
                "text": fallback,
                "source": "rules",
                "model": self.llm_config.model,
                "needs_approval": True,
                "status": "待审批",
                "notes": ["LLM 未返回可用回复，使用规则兜底。"],
                "budget": budget_decision.to_dict(),
            }
        reply_text = self._guard_suggested_reply(
            reply_text,
            source_text=source_text,
            effective_text=effective_text,
            sender_id=sender_id,
            game=game,
            missing_fields=missing_fields,
            pool_matches=pool_matches,
            tool_results=tool_results,
        )
        notes = parsed.get("notes") if isinstance(parsed.get("notes"), list) else []
        selected_pool_game_id = self._valid_selected_pool_game_id(
            parsed.get("selected_pool_game_id"),
            pool_matches,
        )
        reasoning_summary = str(
            parsed.get("reasoning_summary")
            or parsed.get("reason")
            or (notes[0] if notes else "")
        ).strip()
        result = {
            "text": truncate_text(reply_text, 500),
            "source": "llm",
            "model": self.llm_config.model,
            "needs_approval": True,
            "status": "待审批",
            "risk_level": str(parsed.get("risk_level") or "low"),
            "selected_pool_game_id": selected_pool_game_id,
            "reasoning_summary": reasoning_summary,
            "notes": [str(item) for item in notes[:3]] or ["LLM 已生成建议回复，老板确认后再发送。"],
            "budget": budget_decision.to_dict(),
        }
        write_llm_audit_log(
            trace_id,
            "llm_parsed",
            {
                "stage": "reply_draft",
                "provider": self.llm_config.provider,
                "model": self.llm_config.model,
                "parsed": result,
            },
        )
        return result

    def _reply_style_hint(
        self,
        game: GameRequest | None,
        missing_fields: list[str],
        source_text: str,
        effective_text: str = "",
    ) -> dict[str, Any]:
        if self._should_use_brief_ack(game, missing_fields, source_text, effective_text):
            return {
                "mode": "brief_ack",
                "instruction": "客户组局意图和条件已明确，只回复“好的，我帮你问问。”这类极简确认，不要复述条件。",
                "forbidden_repetition": ["玩法", "档位", "时间", "烟况", "人数"],
            }
        return {"mode": "normal"}

    def _should_use_brief_ack(
        self,
        game: GameRequest | None,
        missing_fields: list[str],
        source_text: str,
        effective_text: str = "",
    ) -> bool:
        if game is None or missing_fields:
            return False
        text = self._normalize_pool_query_text(f"{source_text}\n{effective_text}")
        return bool(re.search(r"帮.*(组|找|问|摇)|组一桌|摇下人|找.*人", text))

    def _brief_ack_reply(self) -> str:
        return "好的，我帮你问问。"

    def _guard_suggested_reply(
        self,
        reply_text: str,
        *,
        source_text: str,
        effective_text: str = "",
        sender_id: str,
        game: GameRequest | None = None,
        missing_fields: list[str] | None = None,
        pool_matches: list[dict[str, Any]] | None = None,
        tool_results: dict[str, Any] | None = None,
    ) -> str:
        if pool_matches:
            if re.search(r"大概几点|几点能到|打多大|烟.*要求|无烟还是", reply_text):
                return self._pool_match_reply(pool_matches[0])
            if not re.search(r"\d{1,2}:\d{2}|还缺|帮你问|帮你确认", reply_text):
                return self._pool_match_reply(pool_matches[0])
            return reply_text
        if missing_fields:
            follow_up = self._follow_up_text(missing_fields or [], reply_text, sender_id=sender_id, game=game)
            if follow_up and (
                self._looks_like_pool_no_match_reply(reply_text)
                or self._reply_promises_invite_action(reply_text)
                or not self._reply_addresses_missing_fields(reply_text, missing_fields or [])
            ):
                return follow_up
        pool_result = (tool_results or {}).get("search_current_open_games") if isinstance(tool_results, dict) else {}
        if (
            isinstance(pool_result, dict)
            and pool_result.get("called") is True
            and int(pool_result.get("result_count") or 0) == 0
            and self._should_search_existing_pool(source_text, effective_text, game)
            and re.search(r"大概几点|几点能到|打多大|烟.*要求|无烟还是|几个人|明天|改其他时间", reply_text)
        ):
            return self._pool_no_match_reply(source_text, effective_text, sender_id)
        if "start_time" in (missing_fields or []) and self._has_start_time_ambiguity(game):
            return self._follow_up_text(missing_fields or [], reply_text, sender_id=sender_id, game=game)
        if "known_players" in (missing_fields or []) and not re.search(r"一个人|几个人|几位|几缺几|几缺", reply_text):
            return self._follow_up_text(missing_fields or [], reply_text, sender_id=sender_id, game=game)
        if self._reply_promises_invite_action(reply_text) and not self._has_pending_invite_outbox(tool_results):
            follow_up = self._follow_up_text(missing_fields or [], reply_text, sender_id=sender_id, game=game)
            if follow_up:
                return follow_up
            if game is None and self._is_explicit_grouping_request(source_text, effective_text, game):
                return self._follow_up_text(
                    self._missing_fields(None, None),
                    reply_text,
                    sender_id=sender_id,
                    game=game,
                )
            return "好的，我先帮你留意下。"
        if self._should_use_brief_ack(game, missing_fields or [], source_text, effective_text):
            return self._brief_ack_reply() if self._has_pending_invite_outbox(tool_results) else "好的，我先帮你留意下。"
        if self._source_mentions_non_default_play(source_text):
            return reply_text
        profile = self._customer_profile_for_prompt(sender_id)
        default_assumption = profile.get("default_assumption") if isinstance(profile, dict) else {}
        if not isinstance(default_assumption, dict):
            return reply_text
        if default_assumption.get("default_region_game_type") != "hangzhou_mahjong":
            return reply_text
        if "川麻" not in reply_text or "杭麻" not in reply_text:
            return reply_text
        label = str(default_assumption.get("game_label") or "杭麻")
        guarded = reply_text
        guarded = re.sub(r"想打\s*杭麻(?:财敲)?\s*还是\s*川麻[？?]?", f"还是按{label}给你看？", guarded)
        guarded = re.sub(r"打\s*杭麻(?:财敲)?\s*还是\s*川麻[？?]?", f"按{label}给你看？", guarded)
        guarded = re.sub(r"杭麻(?:财敲)?\s*还是\s*川麻", f"{label}", guarded)
        guarded = re.sub(r"川麻\s*还是\s*杭麻(?:财敲)?", f"{label}", guarded)
        return guarded

    def _reply_promises_invite_action(self, reply_text: str) -> bool:
        return bool(re.search(r"帮你问|帮.*问人|帮.*摇|摇人|问问|先问", reply_text))

    def _looks_like_pool_no_match_reply(self, reply_text: str) -> bool:
        return bool(
            re.search(
                r"(现在|暂时|目前).{0,8}(没有|没).{0,8}(要不要|要).{0,4}组|"
                r"(没有|没).{0,8}(现成|对应|合适).{0,8}局.{0,8}(要不要|要).{0,4}组",
                reply_text,
            )
        )

    def _reply_addresses_missing_fields(self, reply_text: str, missing_fields: list[str]) -> bool:
        if not missing_fields:
            return True
        patterns = {
            "start_time": r"几点|时间|早上|上午|中午|下午|晚上|今晚|明天|人齐开|到店",
            "stake": r"打多大|多大|档位|档|0\\.5|五毛|半块|一块|1块|两块|2块|几块",
            "smoke": r"烟|无烟|有烟|抽",
            "duration": r"几小时|多久|多长|时长|通宵|小时",
            "known_players": r"一个人|几个人|几位|几缺几|几缺|现在.*人|你们.*人",
            "play_type": r"玩法|杭麻|财敲|川麻|红中|捉鸡|幺鸡|麻将",
        }
        checked = [field for field in missing_fields if field in patterns]
        if not checked:
            return True
        return any(re.search(patterns[field], reply_text) for field in checked)

    def _has_pending_invite_outbox(self, tool_results: dict[str, Any] | None) -> bool:
        if not isinstance(tool_results, dict):
            return False
        sender = tool_results.get("send_message")
        if not isinstance(sender, dict):
            return False
        if sender.get("called") is not True:
            return False
        if int(sender.get("result_count") or 0) > 0:
            return True
        return bool(sender.get("outbox"))

    def _valid_selected_pool_game_id(
        self,
        value: Any,
        pool_matches: list[dict[str, Any]],
    ) -> str | None:
        if not pool_matches:
            return None
        selected = str(value or "").strip()
        allowed = {str(item.get("game_id") or "") for item in pool_matches}
        if selected and selected in allowed:
            return selected
        return str(pool_matches[0].get("game_id") or "") or None

    def _tool_results_for_prompt(self, tool_results: dict[str, Any]) -> dict[str, Any]:
        pool = dict(tool_results.get("search_current_open_games") or {})
        candidates = dict(tool_results.get("search_candidate_customers") or {})
        sender = dict(tool_results.get("send_message") or {})
        pool["matches"] = [
            {
                "game_id": item.get("game_id"),
                "summary": item.get("summary"),
                "start_time": item.get("start_time"),
                "level": item.get("level"),
                "level_match_type": item.get("level_match_type"),
                "requested_levels": item.get("requested_levels"),
                "rules": item.get("rules"),
                "missing_count": item.get("missing_count"),
                "score": item.get("score"),
                "reasons": item.get("reasons"),
                "reply_text": item.get("reply_text"),
            }
            for item in list(pool.get("matches") or [])[:5]
            if isinstance(item, dict)
        ]
        candidates["candidates"] = [
            {
                "customer_id": item.get("customer_id"),
                "display_name": item.get("display_name"),
                "score": item.get("score"),
                "reasons": list(item.get("reasons") or [])[:3],
                "warnings": list(item.get("warnings") or [])[:2],
            }
            for item in list(candidates.get("candidates") or [])[:8]
            if isinstance(item, dict)
        ]
        sender["outbox"] = [
            {
                "id": item.get("id"),
                "game_id": item.get("game_id"),
                "customer_id": item.get("customer_id"),
                "customer_name": item.get("customer_name"),
                "approval_status": item.get("approval_status"),
                "direct_send_executed": item.get("direct_send_executed"),
                "draft_source": item.get("draft_source"),
            }
            for item in list(sender.get("outbox") or [])[:8]
            if isinstance(item, dict)
        ]
        for item in [pool, candidates, sender]:
            item.pop("source_text", None)
            query = item.get("query")
            if isinstance(query, dict):
                query.pop("source_text", None)
            for key in list(item.keys()):
                if key not in {
                    "tool_name",
                    "called",
                    "rejected",
                    "risk_level",
                    "approval_required",
                    "direct_send_allowed",
                    "direct_send_executed",
                    "execution_mode",
                    "requested_by",
                    "tool_plan_source",
                    "call_reason",
                    "validation_error",
                    "result_count",
                    "matches",
                    "candidates",
                    "outbox",
                    "query",
                }:
                    item.pop(key, None)
        return {
            "search_current_open_games": pool,
            "search_candidate_customers": candidates,
            "send_message": sender,
        }

    def _source_mentions_non_default_play(self, text: str) -> bool:
        return bool(re.search(r"川麻|四川麻|换三张|定缺|幺鸡|妖鸡|素鸡|捉鸡|湖南|红中|重庆", text))

    def _parse_llm_reply_json(
        self,
        content: str,
        *,
        trace_id: str | None = None,
        stage: str = "unknown",
        schema_hint: str = "",
        retry: bool = True,
    ) -> dict[str, Any]:
        parsed, error = self._parse_llm_reply_json_once(content)
        if parsed is not None:
            return parsed
        if trace_id:
            write_llm_audit_log(
                trace_id,
                "llm_parse_failed",
                {
                    "stage": stage,
                    "provider": self.llm_config.provider if self.llm_config else None,
                    "model": self.llm_config.model if self.llm_config else None,
                    "error": error,
                    "content_excerpt": truncate_text(str(content or ""), 1200),
                },
            )
        if (
            not retry
            or not self.llm_config
            or not self.llm_budget_manager
            or not getattr(self.llm_config, "parse_retry_enabled", True)
        ):
            return {}
        return self._repair_llm_reply_json(
            content,
            trace_id=trace_id or "unknown",
            stage=stage,
            schema_hint=schema_hint,
        )

    def _parse_llm_reply_json_once(self, content: str) -> tuple[dict[str, Any] | None, str]:
        body = str(content or "").strip()
        if not body:
            return None, "empty_content"
        try:
            raw = json.loads(body)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", body, flags=re.S)
            if not match:
                return None, "no_json_object_found"
            try:
                raw = json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                return None, f"invalid_json: {exc}"
        if not isinstance(raw, dict):
            return None, "json_root_not_object"
        return raw, "ok"

    def _repair_llm_reply_json(
        self,
        content: str,
        *,
        trace_id: str,
        stage: str,
        schema_hint: str,
    ) -> dict[str, Any]:
        if not self.llm_config or not self.llm_budget_manager:
            return {}
        max_tokens = max(
            64,
            min(
                int(getattr(self.llm_config, "parse_retry_max_tokens", 256) or 256),
                int(self.llm_config.max_completion_tokens or 256),
            ),
        )
        repair_input = {
            "stage": stage,
            "schema_hint": schema_hint or "输出必须是 JSON object。",
            "malformed_output": truncate_text(str(content or ""), 4000),
        }
        payload = {
            "model": self.llm_config.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是 JSON 修复器。只修复格式，不补编业务事实。"
                        "根据 schema_hint 把 malformed_output 修成一个合法 JSON object。"
                        "只输出 JSON，不要解释，不要 Markdown。"
                    ),
                },
                {"role": "user", "content": json.dumps(repair_input, ensure_ascii=False)},
            ],
        }
        if self.llm_config.thinking_enabled is not None:
            payload["thinking"] = {"type": "enabled" if self.llm_config.thinking_enabled else "disabled"}
        if self.llm_config.response_format:
            payload["response_format"] = {"type": self.llm_config.response_format}

        budget_decision = self.llm_budget_manager.reserve(
            key=f"boss_trial_json_repair:{stage}",
            model=self.llm_config.model,
            prompt=payload,
            max_completion_tokens=max_tokens,
        )
        if not budget_decision.allowed:
            write_llm_audit_log(
                trace_id,
                "llm_retry_budget_denied",
                {
                    "stage": stage,
                    "provider": self.llm_config.provider,
                    "model": self.llm_config.model,
                    "budget": budget_decision.to_dict(),
                },
            )
            return {}

        write_llm_audit_log(
            trace_id,
            "llm_retry_request",
            {
                "stage": stage,
                "provider": self.llm_config.provider,
                "model": self.llm_config.model,
                "base_url": self.llm_config.base_url,
                "timeout_seconds": self.llm_config.timeout_seconds,
                "budget": budget_decision.to_dict(),
                "payload": payload,
            },
        )
        request = urllib.request.Request(
            f"{self.llm_config.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.llm_config.api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.llm_config.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            write_llm_audit_log(
                trace_id,
                "llm_retry_timeout",
                {
                    "stage": stage,
                    "provider": self.llm_config.provider,
                    "model": self.llm_config.model,
                    "error": f"{type(exc).__name__}: {exc}",
                    "budget": budget_decision.to_dict(),
                },
            )
            return {}
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as exc:
            write_llm_audit_log(
                trace_id,
                "llm_retry_error",
                {
                    "stage": stage,
                    "provider": self.llm_config.provider,
                    "model": self.llm_config.model,
                    "error": f"{type(exc).__name__}: {exc}",
                    "budget": budget_decision.to_dict(),
                },
            )
            return {}

        actual_usage = usage_from_response(data, self.llm_config.model)
        self.llm_budget_manager.commit(budget_decision.reservation_id, actual_usage)
        repaired_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        write_llm_audit_log(
            trace_id,
            "llm_retry_response",
            {
                "stage": stage,
                "provider": self.llm_config.provider,
                "model": self.llm_config.model,
                "raw_response": data,
                "content": repaired_content,
                "usage": actual_usage.to_dict() if actual_usage else None,
            },
        )
        parsed, error = self._parse_llm_reply_json_once(str(repaired_content or ""))
        if parsed is None:
            write_llm_audit_log(
                trace_id,
                "llm_retry_parse_failed",
                {
                    "stage": stage,
                    "provider": self.llm_config.provider,
                    "model": self.llm_config.model,
                    "error": error,
                    "content_excerpt": truncate_text(str(repaired_content or ""), 1200),
                },
            )
            return {}
        write_llm_audit_log(
            trace_id,
            "llm_retry_parsed",
            {
                "stage": stage,
                "provider": self.llm_config.provider,
                "model": self.llm_config.model,
                "parsed": parsed,
            },
        )
        return parsed

    def _recommend(self, game: GameRequest, now: datetime) -> list[CandidateRecommendation]:
        recommendations: list[CandidateRecommendation] = []
        for customer in self.store.customers():
            if customer["no_contact"] or customer["id"] == game.organizer_id:
                continue
            score = 0.0
            reasons: list[str] = []
            warnings: list[str] = []
            levels = customer["preferred_levels"]
            if self._customer_matches_game(game, customer):
                score += 30
                reasons.append(f"常打{self._game_label(game)}")
            elif game.game_type != "mahjong":
                score -= 50
                warnings.append("画像里没有这个玩法偏好")

            if game.level and game.level in levels:
                score += 20
                reasons.append(f"常打 {game.level} 档")
            elif game.level and self._near_level(game.level, levels):
                score += 10
                reasons.append("档位接近")

            if game.start_at and game.start_at.hour in customer["usual_start_hours"]:
                score += 20
                reasons.append(f"{game.start_at.hour}:00 左右常来")
            elif game.start_at and any(abs(game.start_at.hour - hour) <= 1 for hour in customer["usual_start_hours"]):
                score += 10
                reasons.append("时间段接近")

            if "无烟" in game.rules:
                if customer["smoke_preference"] == "no_smoke":
                    score += 10
                    reasons.append("无烟偏好匹配")
                elif customer["smoke_preference"] == "smoke_ok":
                    score -= 10
                    warnings.append("可能更习惯有烟局")
            elif customer["smoke_preference"] == "any":
                score += 3
                reasons.append("烟况要求灵活")

            last_invited_at = parse_dt(customer["last_invited_at"])
            if last_invited_at is None:
                score += 10
                reasons.append("近期未邀约")
            else:
                hours = (now - last_invited_at).total_seconds() / 3600
                if hours >= 24:
                    score += 10
                    reasons.append(f"{int(hours // 24)} 天未邀约")
                else:
                    score -= 20
                    warnings.append("最近 24 小时已邀约")

            if customer["response_rate"] >= 0.7 or customer["response_speed"] == "fast":
                score += 10
                reasons.append("响应率高")

            if customer["fatigue_score"] >= 60:
                score -= 20
                warnings.append("疲劳度较高")

            if score >= 20:
                recommendations.append(
                    CandidateRecommendation(
                        customer_id=customer["id"],
                        display_name=customer["display_name"],
                        score=round(score, 1),
                        reasons=reasons or ["基础条件可尝试"],
                        warnings=warnings,
                    )
                )
        sorted_recommendations = sorted(recommendations, key=lambda item: item.score, reverse=True)
        return self._apply_candidate_composition_ranking(game, sorted_recommendations)

    def _apply_candidate_composition_ranking(
        self,
        game: GameRequest,
        recommendations: list[CandidateRecommendation],
    ) -> list[CandidateRecommendation]:
        preference = self._candidate_composition_preference_from_game(game)
        desired_genders = [
            gender
            for gender in preference.get("preferred_candidate_genders") or []
            if gender in {"male", "female"}
        ]
        if not desired_genders or not recommendations:
            return recommendations
        remaining = list(recommendations)
        selected: list[CandidateRecommendation] = []
        unmet: list[str] = []
        for desired_gender in desired_genders:
            match_index = next(
                (
                    index
                    for index, item in enumerate(remaining)
                    if self._candidate_gender(item.customer_id) == desired_gender
                ),
                None,
            )
            if match_index is None:
                unmet.append(desired_gender)
                continue
            item = remaining.pop(match_index)
            reason = f"符合候选组合偏好：{GENDER_LABELS.get(desired_gender, '未知')}"
            if reason not in item.reasons:
                item.reasons.append(reason)
            selected.append(item)
        if unmet and remaining:
            labels = "、".join(GENDER_LABELS.get(gender, "未知") for gender in unmet)
            warning = f"候选组合偏好未完全满足：暂无匹配{labels}候选"
            if warning not in remaining[0].warnings:
                remaining[0].warnings.append(warning)
        return selected + remaining

    def _candidate_gender(self, customer_id: str) -> str:
        customer = self.store.customer(customer_id)
        return normalize_gender((customer or {}).get("gender"))

    def _game_to_dict(self, game: GameRequest | None) -> dict[str, Any]:
        if game is None:
            return {}
        return {
            "id": game.id,
            "status": game.status.value,
            "game_type": game.game_type,
            "game_label": self._game_label(game),
            "ruleset": game.ruleset,
            "variant": game.variant,
            "variant_label": VARIANT_LABELS.get(game.variant or "", game.variant),
            "level": game.level,
            "base_score": game.base_score,
            "cap_score": game.cap_score,
            "start_at": game.start_at.isoformat() if game.start_at else None,
            "start_time": self._start_time_display(game),
            "start_time_mode": self._start_time_mode(game),
            "start_time_confidence": game.start_time_confidence,
            "duration_hours": game.duration_hours,
            "duration_mode": self._duration_mode(game),
            "duration_text": self._duration_invite_term(game) or None,
            "current_player_count": game.current_player_count,
            "missing_count": game.missing_count,
            "rules": game.rules,
            "play_options": game.play_options,
            "ambiguities": game.ambiguities,
            "notes": game.notes,
            "candidate_composition_preference": self._candidate_composition_preference_from_game(game),
            "summary": self._summary(game),
        }

    def _user_intent_label(self, action: str) -> str:
        return {
            "ask_clarification": "想打/想组局，信息待确认",
            "inquire_existing_game": "咨询现有局",
            "create_pending_game": "明确组局，先入待组局",
            "match_existing_game": "匹配已有局/可拼局",
            "create_game": "创建组局需求",
            "queue_invites": "找人组局",
            "accept_seat": "接受邀约/报名入局",
            "decline_invite": "拒绝邀约",
            "close_game": "取消或关闭组局",
            "ignore": "无关消息，无需回复",
            "human_review": "高风险或不确定，转人工",
        }.get(action, action or "-")

    def _effective_intent_action(
        self,
        action: str,
        game: GameRequest | None,
        missing_fields: list[str],
        outbox: list[dict[str, Any]],
    ) -> str:
        if game and (set(missing_fields) & CRITICAL_FIELDS) and action != "human_review":
            return "ask_clarification"
        if game and not (set(missing_fields) & CRITICAL_FIELDS):
            if outbox:
                return "queue_invites"
            if action == "ask_clarification":
                return "create_pending_game"
        return action

    def _candidate_invite_drafts(
        self,
        *,
        trace_id: str,
        game: GameRequest,
        recommendations: list[CandidateRecommendation],
        now: datetime,
    ) -> dict[str, dict[str, Any]]:
        if not recommendations:
            return {}
        llm_drafts = self._llm_candidate_invite_drafts(
            trace_id=trace_id,
            game=game,
            recommendations=recommendations,
            now=now,
        )
        drafts: dict[str, dict[str, Any]] = {}
        for recommendation in recommendations:
            raw = llm_drafts.get(recommendation.customer_id, {})
            message_text = str(raw.get("message_text") or "").strip()
            guarded = self._guard_private_invite_text(
                message_text,
                game=game,
                customer_name=recommendation.display_name,
            )
            source = "llm" if message_text and guarded == message_text else "rules"
            drafts[recommendation.customer_id] = {
                "message_text": guarded,
                "source": source,
                "reasoning_summary": raw.get("reasoning_summary") or (
                    "LLM 邀约草稿已通过后端校验。" if source == "llm" else "LLM 未返回合规草稿，使用规则兜底。"
                ),
            }
        return drafts

    def _llm_candidate_invite_drafts(
        self,
        *,
        trace_id: str,
        game: GameRequest,
        recommendations: list[CandidateRecommendation],
        now: datetime,
    ) -> dict[str, dict[str, Any]]:
        if not self.llm_config or not self.llm_budget_manager:
            return {}
        max_tokens = min(self.llm_config.max_completion_tokens, 700)
        system_prompt = """你是麻将馆老板的私聊邀约起草助手。
你只能为候选人生成“待老板审批”的私聊草稿，不能说已经确认、已经占座、已经发出。
草稿要非常短，像老板微信手写。
默认不要透露玩法细分、房间状态、还有几个人、谁发起的局、推荐原因、候选人评分。
禁止出现：有一桌、缺1位、缺一位、缺2位、缺二位、缺3位、缺三位、三缺一、二缺二、方便来吗。
杭州门店默认杭麻，杭麻/财敲不需要写给候选人；除非是川麻/红中/捉鸡/湖南等非默认玩法才可简短写玩法。
如果 public_invite_terms.duration 存在，邀约里要带上时长。
推荐格式：{候选人昵称}，14:00，0.5无烟，约4小时，打吗？
只输出 JSON：
{"drafts":[{"customer_id":"候选人ID","message_text":"一句待审批私聊草稿","reasoning_summary":"一句话说明"}]}"""
        prompt = {
            "task": "给每个候选人生成一条极简私聊邀约草稿。",
            "now": now.strftime("%Y-%m-%d %H:%M:%S"),
            "active_skills": self._active_skills(stage="invite_draft", game=game),
            "public_invite_terms": self._public_invite_terms(game),
            "private_fields_do_not_disclose": {
                "missing_count": game.missing_count,
                "current_player_count": game.current_player_count,
                "organizer_id": game.organizer_id,
                "game_label": self._game_label(game),
                "ruleset": game.ruleset,
                "variant": game.variant,
            },
            "candidates": [
                {
                    "customer_id": item.customer_id,
                    "display_name": item.display_name,
                    "reasons": item.reasons[:3],
                    "warnings": item.warnings[:2],
                }
                for item in recommendations
            ],
            "rules": [
                "优先参考 active_skills；skill 是运营经验，不能覆盖隐私和审批边界。",
                "每条必须包含候选人昵称。",
                "每条尽量只包含时间、档位、烟况、明确时长和“打吗”。",
                "如果 public_invite_terms.duration 存在，必须带上时长。",
                "不要写“方便来吗”。",
                "不要写“缺几位/三缺一/二缺二”。",
                "不要写“有一桌”。",
                "默认杭麻/财敲不写玩法；非默认玩法才写玩法。",
                "不要透露发起人是谁，也不要透露推荐原因。",
            ],
        }
        payload = {
            "model": self.llm_config.model,
            "temperature": min(self.llm_config.temperature, 0.3),
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        }
        if self.llm_config.thinking_enabled is not None:
            payload["thinking"] = {"type": "enabled" if self.llm_config.thinking_enabled else "disabled"}
        if self.llm_config.response_format:
            payload["response_format"] = {"type": self.llm_config.response_format}

        budget_decision = self.llm_budget_manager.reserve(
            key="boss_trial_invites",
            model=self.llm_config.model,
            prompt=payload,
            max_completion_tokens=max_tokens,
        )
        if not budget_decision.allowed:
            write_llm_audit_log(
                trace_id,
                "llm_budget_denied",
                {
                    "stage": "invite_drafts",
                    "provider": self.llm_config.provider,
                    "model": self.llm_config.model,
                    "budget": budget_decision.to_dict(),
                },
            )
            return {}

        write_llm_audit_log(
            trace_id,
            "llm_request",
            {
                "stage": "invite_drafts",
                "provider": self.llm_config.provider,
                "model": self.llm_config.model,
                "base_url": self.llm_config.base_url,
                "timeout_seconds": self.llm_config.timeout_seconds,
                "budget": budget_decision.to_dict(),
                "payload": payload,
            },
        )

        request = urllib.request.Request(
            f"{self.llm_config.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.llm_config.api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.llm_config.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            write_llm_audit_log(
                trace_id,
                "llm_error",
                {
                    "stage": "invite_drafts",
                    "provider": self.llm_config.provider,
                    "model": self.llm_config.model,
                    "error": f"HTTP {exc.code}",
                    "budget": budget_decision.to_dict(),
                },
            )
            return {}
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            write_llm_audit_log(
                trace_id,
                "llm_error",
                {
                    "stage": "invite_drafts",
                    "provider": self.llm_config.provider,
                    "model": self.llm_config.model,
                    "error": f"{type(exc).__name__}: {exc}",
                    "budget": budget_decision.to_dict(),
                },
            )
            return {}

        actual_usage = usage_from_response(data, self.llm_config.model)
        self.llm_budget_manager.commit(budget_decision.reservation_id, actual_usage)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        write_llm_audit_log(
            trace_id,
            "llm_response",
            {
                "stage": "invite_drafts",
                "provider": self.llm_config.provider,
                "model": self.llm_config.model,
                "raw_response": data,
                "content": content,
                "usage": actual_usage.to_dict() if actual_usage else None,
            },
        )
        parsed = self._parse_llm_reply_json(
            content,
            trace_id=trace_id,
            stage="invite_draft",
            schema_hint='{"drafts":[{"customer_id":str,"message_text":str,"reasoning_summary":str}]}',
        )
        drafts: dict[str, dict[str, Any]] = {}
        for item in parsed.get("drafts") or []:
            if not isinstance(item, dict):
                continue
            customer_id = str(item.get("customer_id") or "").strip()
            message_text = str(item.get("message_text") or "").strip()
            if not customer_id or not message_text:
                continue
            drafts[customer_id] = {
                "message_text": message_text,
                "source": "llm",
                "reasoning_summary": str(item.get("reasoning_summary") or "").strip(),
            }
        write_llm_audit_log(
            trace_id,
            "llm_parsed",
            {
                "stage": "invite_drafts",
                "provider": self.llm_config.provider,
                "model": self.llm_config.model,
                "draft_count": len(drafts),
                "drafts": drafts,
            },
        )
        return drafts

    def _public_invite_terms(self, game: GameRequest) -> dict[str, Any]:
        smoke = "无烟" if "无烟" in game.rules else ("可烟" if "可吸烟" in game.rules else "")
        level_smoke = "".join(part for part in [game.level or "", smoke] if part)
        game_label = ""
        if game.game_type not in {"mahjong", "hangzhou_mahjong"}:
            game_label = self._game_label(game)
        return {
            "start_time": self._start_time_display(game) or "",
            "level_smoke": level_smoke,
            "game_label": game_label,
            "duration": self._duration_invite_term(game),
        }

    def _duration_invite_term(self, game: GameRequest) -> str:
        duration = game.duration_hours
        if duration is None:
            return "通宵" if self._duration_mode(game) == "overnight" else ""
        if float(duration).is_integer():
            value = str(int(duration))
        else:
            value = f"{float(duration):g}"
        return f"约{value}小时"

    def _candidate_to_dict(self, item: CandidateRecommendation) -> dict[str, Any]:
        gender = self._candidate_gender(item.customer_id)
        return {
            "customer_id": item.customer_id,
            "display_name": item.display_name,
            "gender": gender,
            "gender_label": GENDER_LABELS.get(gender, "未知"),
            "score": item.score,
            "reasons": item.reasons,
            "warnings": item.warnings,
        }

    def _private_invite_text(self, game: GameRequest, customer_name: str) -> str:
        terms = self._public_invite_terms(game)
        parts = [terms["start_time"], terms["game_label"], terms["level_smoke"], terms["duration"]]
        core = "，".join(str(part) for part in parts if part)
        return f"{customer_name}，{core}，打吗？" if core else f"{customer_name}，打吗？"

    def _guard_private_invite_text(
        self,
        message_text: str,
        *,
        game: GameRequest,
        customer_name: str,
    ) -> str:
        fallback = self._private_invite_text(game, customer_name)
        text = re.sub(r"\s+", "", message_text.strip())
        if not text:
            return fallback
        forbidden_patterns = [
            r"有一桌",
            r"缺\s*[一二三四五六七八九十0-9]\s*位?",
            r"[一二三四五六七八九十0-9]\s*缺\s*[一二三四五六七八九十0-9]",
            r"方便来吗",
            r"推荐",
            r"评分",
            r"发起",
            r"张哥",
            r"杭麻",
            r"财敲",
        ]
        if any(re.search(pattern, text) for pattern in forbidden_patterns):
            return fallback
        duration_term = self._duration_invite_term(game)
        if duration_term and duration_term.replace("约", "") not in text:
            return fallback
        if len(text) > 38:
            return fallback
        if "打吗" not in text:
            return fallback
        if customer_name and not text.startswith(f"{customer_name}，"):
            text = f"{customer_name}，{text}"
        return text

    def _summary(self, game: GameRequest) -> str:
        parts = [self._game_label(game)]
        if game.level:
            parts.append(f"{game.level}档")
        if game.start_at:
            parts.append(game.start_at.strftime("%H:%M"))
        elif self._has_flexible_start(game):
            parts.append("人齐开")
        if game.missing_count is not None:
            parts.append(f"缺{game.missing_count}")
        visible_rules = [rule for rule in game.rules if rule not in {self._game_label(game), "杭麻", "川麻"}]
        if visible_rules:
            parts.append("、".join(visible_rules))
        return " ".join(part for part in parts if part)

    def _game_status_label(self, game: GameRequest, missing_fields: list[str], has_outbox: bool) -> str:
        if missing_fields:
            return "待补充"
        if has_outbox:
            return "邀约中"
        if game.status == GameStatus.CONFIRMED:
            return "已满"
        return "待组局"

    def _game_label(self, game: GameRequest) -> str:
        labels = []
        if game.game_type in GAME_TYPE_LABELS and game.game_type != "mahjong":
            labels.append(GAME_TYPE_LABELS[game.game_type])
        if game.variant in VARIANT_LABELS:
            labels.append(VARIANT_LABELS[game.variant])
        return " ".join(labels) or "麻将"

    def _customer_matches_game(self, game: GameRequest, customer: dict[str, Any]) -> bool:
        games = customer["preferred_games"]
        label = self._game_label(game)
        if any(item in label or label in item for item in games):
            return True
        if game.game_type == "hangzhou_mahjong" and self._customer_has_label(customer, "杭麻"):
            return True
        if game.game_type == "sichuan_mahjong" and self._customer_has_label(customer, "川麻"):
            return True
        if game.variant == "caiqiao" and self._customer_has_label(customer, "财敲"):
            return True
        return False

    def _customer_has_label(self, customer: dict[str, Any], label: str) -> bool:
        if self._has_game(customer["preferred_games"], label):
            return True
        return label in str(customer.get("notes") or "")

    def _near_level(self, level: str, levels: list[str]) -> bool:
        try:
            current = float(level.split("-", 1)[0])
        except ValueError:
            return False
        for item in levels:
            try:
                value = float(item.split("-", 1)[0])
            except ValueError:
                continue
            if abs(current - value) <= 0.5:
                return True
        return False

    def _has_game(self, games: list[str], label: str) -> bool:
        return any(label in item or item in label for item in games)


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>麻将馆组局试用台</title>
  <style>
    :root {
      --bg: #f6f7f4;
      --panel: #ffffff;
      --line: #d9ded8;
      --text: #20251f;
      --muted: #687266;
      --brand: #2f6f5e;
      --brand-soft: #e6f1ed;
      --warn: #9a5b14;
      --bad: #9f2f2f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 12px 18px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfa;
      position: sticky;
      top: 0;
      z-index: 10;
    }
    h1 { font-size: 18px; margin: 0; }
    h2 { font-size: 15px; margin: 0 0 10px; }
    h3 { font-size: 14px; margin: 14px 0 8px; }
    button {
      border: 1px solid #b8c5bc;
      background: #fff;
      color: var(--text);
      border-radius: 6px;
      padding: 7px 10px;
      cursor: pointer;
      white-space: nowrap;
    }
    button.primary { background: var(--brand); color: #fff; border-color: var(--brand); }
    button.ghost { background: var(--brand-soft); border-color: #c5d8d0; color: #214d41; }
    button.danger { color: var(--bad); }
    input, textarea, select {
      width: 100%;
      border: 1px solid #cbd3ca;
      border-radius: 6px;
      padding: 8px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }
    textarea { resize: vertical; min-height: 150px; }
    .app {
      display: grid;
      grid-template-columns: minmax(260px, 340px) minmax(360px, 1fr) minmax(360px, 1fr);
      gap: 12px;
      padding: 12px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }
    .stack { display: grid; gap: 10px; }
    .row { display: flex; gap: 8px; align-items: center; }
    .row > * { flex: 1; }
    .toolbar { display: flex; gap: 8px; flex-wrap: wrap; }
    .kv { display: grid; grid-template-columns: 92px 1fr; gap: 6px 10px; }
    .kv div:nth-child(odd) { color: var(--muted); }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 2px 7px;
      border-radius: 999px;
      background: #eef1ed;
      margin: 0 4px 4px 0;
      font-size: 12px;
      color: #374138;
    }
    .pill.warn { background: #fff4dc; color: var(--warn); }
    .pill.good { background: var(--brand-soft); color: var(--brand); }
    .muted { color: var(--muted); }
    .draft, pre {
      white-space: pre-wrap;
      word-break: break-word;
      background: #f5f7f4;
      border: 1px solid #e1e5df;
      border-radius: 6px;
      padding: 9px;
      margin: 6px 0;
    }
    .draft.compact {
      padding: 7px;
      margin: 4px 0 0;
      min-height: 36px;
    }
    .message-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin: 8px 0;
    }
    .manual-game-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }
    .candidate, .game, .customer {
      border: 1px solid #e0e5de;
      border-radius: 7px;
      padding: 10px;
      margin-bottom: 8px;
      background: #fff;
    }
    .conversation {
      display: grid;
      gap: 6px;
      margin-top: 6px;
    }
    .turn {
      border-left: 3px solid #d7e5dc;
      background: #f8faf7;
      padding: 7px 9px;
      border-radius: 4px;
    }
    .candidate strong, .game strong { font-size: 15px; }
    .bottom {
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 12px;
      padding: 0 12px 14px;
    }
    .customer-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 8px;
      max-height: 360px;
      overflow: auto;
    }
    .tiny { font-size: 12px; }
    @media (max-width: 1120px) {
      .app, .bottom { grid-template-columns: 1fr; }
      .message-grid { grid-template-columns: 1fr; }
      .manual-game-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>麻将馆组局试用台</h1>
    <div class="toolbar">
      <span id="cacheStatus" class="pill">缓存状态读取中</span>
      <button class="ghost" onclick="fillSample('weak')">弱意图样例</button>
      <button class="ghost" onclick="fillSample('clear')">明确组局样例</button>
      <button onclick="loadState()">刷新</button>
    </div>
  </header>

  <section class="app">
    <section class="panel stack">
      <h2>输入客户消息</h2>
      <div class="row">
        <input id="senderName" placeholder="客户昵称，如张哥" value="张哥" />
        <input id="senderId" placeholder="客户ID，可用微信备注" value="zhang" />
      </div>
      <input id="conversationId" placeholder="会话ID，如 group_a / private_zhang" value="boss_trial" />
      <textarea id="messageText">下午两点 0.5 无烟杭麻，打4小时，帮我组一桌</textarea>
      <div class="toolbar">
        <button class="primary" onclick="analyze()">解析消息</button>
        <button onclick="copyText('messageText')">复制原文</button>
        <button class="ghost" onclick="clearShortMemory()">清空短期记忆</button>
      </div>
      <div class="muted tiny">第一版只生成草稿，不自动群发、不自动私发、不确认房间。</div>
    </section>

    <section class="panel stack">
      <h2>识别出的组局条件</h2>
      <div id="parsedBox" class="muted">等待解析。</div>
      <h3>建议回复（待审批）</h3>
      <div id="suggestedReplyMeta" class="tiny muted"></div>
      <div id="followUpBox" class="draft muted">解析后生成给当前客户的建议回复。</div>
      <h3>群发草稿</h3>
      <div id="groupDraftBox" class="draft muted">信息明确后生成群发草稿。</div>
      <div class="toolbar">
        <button onclick="copyRendered('followUpBox')">复制建议回复</button>
        <button onclick="copyRendered('groupDraftBox')">复制群发</button>
      </div>
    </section>

    <section class="panel stack">
      <h2>当前匹配和待审批邀约</h2>
      <div id="candidateBox" class="muted">解析后显示可拼局或候选人。</div>
    </section>
  </section>

  <section class="bottom">
    <section class="panel">
      <div class="row">
        <h2>当前局看板</h2>
        <button class="danger" onclick="clearBoard()">清空当前局</button>
      </div>
      <h3>手动创建局</h3>
      <div class="manual-game-grid">
        <select id="manualGameType">
          <option value="hangzhou_mahjong">杭麻</option>
          <option value="sichuan_mahjong">川麻</option>
          <option value="hongzhong_mahjong">红中</option>
          <option value="zhuoji_mahjong">捉鸡</option>
          <option value="hunan_mahjong">湖南麻将</option>
        </select>
        <input id="manualVariant" placeholder="细分，如财敲/换三张，可空" value="财敲" />
        <input id="manualStartTime" type="time" />
        <input id="manualLevel" placeholder="档位，如0.5/1/2-16" value="0.5" />
        <input id="manualCurrentPlayers" type="number" min="0" max="4" placeholder="当前人数" value="3" />
        <input id="manualMissingCount" type="number" min="0" max="4" placeholder="缺口" value="1" />
        <input id="manualDurationHours" type="number" min="1" max="12" step="0.5" placeholder="时长/小时" value="4" />
        <select id="manualSmoke">
          <option value="no_smoke">无烟</option>
          <option value="smoke_ok">有烟</option>
          <option value="any">烟况都可</option>
        </select>
        <select id="manualStatus">
          <option value="待组局">待组局</option>
          <option value="邀约中">邀约中</option>
          <option value="已满">已满</option>
        </select>
        <input id="manualOrganizerName" placeholder="发起人/来源" value="老板手动创建" />
      </div>
      <textarea id="manualSourceText" style="min-height: 72px; margin-top: 8px;" placeholder="来源说明，如：电话里李姐说六点有烟1块，三缺一"></textarea>
      <div class="toolbar" style="margin: 8px 0 12px;">
        <button class="primary" onclick="manualCreateGame()">创建到看板</button>
      </div>
      <div id="gameBoard" class="muted">暂无当前局。</div>
    </section>
    <section class="panel">
      <h2>今日复盘</h2>
      <div id="recapBox" class="muted">暂无复盘。</div>
    </section>
  </section>

  <section class="bottom">
    <section class="panel stack">
      <h2>评测样本沉淀</h2>
      <textarea id="evalNote" placeholder="写清楚老板判断：哪里错、哪里对、希望以后怎么处理。"></textarea>
      <div class="row">
        <select id="evalExpectedAction">
          <option value="">golden 默认沿用当前动作</option>
          <option value="ask_clarification">追问信息</option>
          <option value="create_pending_game">进入待组局</option>
          <option value="create_game">创建组局</option>
          <option value="queue_invites">推荐并生成邀约</option>
          <option value="accept_seat">接受入局</option>
          <option value="decline_invite">拒绝入局</option>
          <option value="close_game">关闭局</option>
          <option value="ignore">静默</option>
          <option value="human_review">转人工</option>
        </select>
        <input id="evalTags" placeholder="标签，逗号分隔，如弱意图,张哥,杭州" />
      </div>
      <div class="toolbar">
        <button class="danger" onclick="recordEvalCase('badcase')">归档 badcase</button>
        <button class="ghost" onclick="recordEvalCase('golden')">加入 golden</button>
        <button onclick="recordEvalCase('few_shot')">采集 few-shot</button>
      </div>
      <div id="evalResult" class="muted tiny">先解析一条消息，再把结果沉淀成评测数据。</div>
    </section>
    <section class="panel">
      <h2>可观测与评测入口</h2>
      <div id="evalOverview" class="muted">评测数据读取中。</div>
      <div class="toolbar" style="margin-top: 10px;">
        <a href="/logs" target="_blank"><button>查看日志</button></a>
        <a href="/api/logs" target="_blank"><button>JSON 日志</button></a>
        <a href="/api/eval-cases" target="_blank"><button>评测数据</button></a>
      </div>
    </section>
  </section>

  <section class="bottom">
    <section class="panel">
      <h2>客户画像管理</h2>
      <div class="row">
        <input id="customerName" placeholder="昵称" />
        <input id="customerContact" placeholder="微信备注/联系方式" />
      </div>
      <div class="row">
        <select id="customerGender">
          <option value="unknown">性别未知</option>
          <option value="male">男</option>
          <option value="female">女</option>
        </select>
      </div>
      <div class="row">
        <input id="customerGames" placeholder="常打大类，逗号分隔，如杭麻,川麻；财敲/换三张等细分写备注" />
        <input id="customerLevels" placeholder="常打档位，如0.5,1" />
      </div>
      <div class="row">
        <input id="customerHours" placeholder="常来时间，如14,19,20" />
        <select id="customerSmoke">
          <option value="any">烟况都可</option>
          <option value="no_smoke">无烟偏好</option>
          <option value="smoke_ok">可有烟</option>
        </select>
      </div>
      <textarea id="customerNotes" placeholder="备注，如常一个人来、少打扰"></textarea>
      <button class="primary" onclick="saveCustomer()">保存客户</button>
    </section>
    <section class="panel">
      <h2>常客列表</h2>
      <div id="customerList" class="customer-grid"></div>
    </section>
  </section>

  <script>
    let currentAnalysis = null;
    window.latestState = null;

    const samples = {
      weak: "老板，今天下班有人打麻将吗？0.5或者1都行，烟也都可",
      clear: "下午两点 0.5 无烟杭麻，打4小时，一缺三，帮我组一桌"
    };

    function fillSample(name) {
      document.getElementById("messageText").value = samples[name];
    }

    async function api(path, options = {}) {
      const res = await fetch(path, {
        headers: {"Content-Type": "application/json"},
        ...options
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.message || data.error || "请求失败");
      return data;
    }

    async function loadState() {
      const state = await api("/api/state");
      renderState(state);
    }

    async function analyze() {
      const payload = {
        sender_name: document.getElementById("senderName").value,
        sender_id: document.getElementById("senderId").value,
        conversation_id: document.getElementById("conversationId").value,
        text: document.getElementById("messageText").value
      };
      currentAnalysis = await api("/api/analyze", {
        method: "POST",
        body: JSON.stringify(payload)
      });
      renderAnalysis(currentAnalysis);
      renderState(currentAnalysis.state);
    }

    function renderAnalysis(result) {
      const parsed = result.parsed || {};
      const missing = result.missing_fields || [];
      const rules = (parsed.rules || []).map(item => `<span class="pill good">${escapeHtml(item)}</span>`).join("");
      const composition = parsed.candidate_composition_preference || {};
      const compositionText = (composition.gender_labels || []).length ? (composition.gender_labels || []).join("、") : "-";
      const fields = [
        ["会话ID", result.conversation_id || "-"],
        ["用户意向", parsed.user_intent || intentLabel(result.decision?.action)],
        ["玩法", parsed.game_label || "-"],
        ["时间", parsed.start_time || "-"],
        ["档位", parsed.level || "-"],
        ["时长", parsed.duration_text || (parsed.duration_hours ? `${parsed.duration_hours} 小时` : "-")],
        ["当前人数", parsed.current_player_count ?? "-"],
        ["缺口", parsed.missing_count ?? "-"],
        ["烟况/规则", rules || "-"],
        ["候选偏好", compositionText]
      ];
      document.getElementById("parsedBox").innerHTML = `
        <div class="kv">${fields.map(([k, v]) => `<div>${k}</div><div>${v}</div>`).join("")}</div>
        <div>${missing.map(item => `<span class="pill warn">缺 ${fieldLabel(item)}</span>`).join("")}</div>
        <div class="muted tiny">${escapeHtml(parsed.summary || result.decision?.reply_text || "")}</div>
      `;
      const suggested = result.suggested_reply || {};
      const sourceText = suggested.source === "llm" ? `LLM：${suggested.model || "-"}` : "规则兜底";
      document.getElementById("suggestedReplyMeta").innerHTML = `
        <span class="pill warn">${escapeHtml(suggested.status || "待审批")}</span>
        <span class="pill">${escapeHtml(sourceText)}</span>
        ${(suggested.notes || []).map(note => `<span class="pill">${escapeHtml(note)}</span>`).join("")}
      `;
      document.getElementById("followUpBox").textContent = suggested.text || result.follow_up || "信息基本够，可以生成候选和草稿。";
      document.getElementById("groupDraftBox").textContent = result.group_draft || "暂未生成群发草稿。";
      renderCandidates(result.outbox || [], result.pool_matches || []);
    }

    function renderCandidates(outbox, poolMatches) {
      const box = document.getElementById("candidateBox");
      if (!outbox.length && !poolMatches.length) {
        box.innerHTML = `<div class="muted">暂无可拼局或待审批邀约。通常是因为时间、档位或人数还没补齐。</div>`;
        return;
      }
      const poolHtml = poolMatches.length ? `
        <h3>可拼/可加入的局</h3>
        ${poolMatches.map(item => `
          <div class="candidate">
            <div class="row">
              <strong>${escapeHtml(item.summary || "匹配局")} <span class="pill">${escapeHtml(String(item.score || 0))}分</span></strong>
              <span class="pill good">可拼局</span>
            </div>
            <div>${(item.reasons || []).map(reason => `<span class="pill good">${escapeHtml(reason)}</span>`).join("")}</div>
            <div class="draft" id="pool-${item.game_id}">${escapeHtml(item.reply_text || "")}</div>
            <div class="toolbar">
              <button onclick="copyPoolReply('${item.game_id}')">复制回复</button>
            </div>
          </div>
        `).join("")}
      ` : "";
      const outboxHtml = outbox.length ? `
        <h3>推荐候选人和待审批邀约</h3>
        ${outbox.map(item => `
        <div class="candidate">
          <div class="row">
            <strong>${escapeHtml(item.customer_name)} <span class="pill">${item.score}分</span></strong>
            <span class="pill">${escapeHtml(item.gender_label || "未知")}</span>
            <span class="pill warn">${escapeHtml(item.approval_status || item.status || "待审批")}</span>
          </div>
          <div>${(item.reasons || []).map(reason => `<span class="pill good">${escapeHtml(reason)}</span>`).join("")}</div>
          <div>${(item.warnings || []).map(reason => `<span class="pill warn">${escapeHtml(reason)}</span>`).join("")}</div>
          <div class="draft" id="draft-${item.id}">${escapeHtml(item.message_text)}</div>
          <div class="toolbar">
            <button onclick="copyDraft('${item.id}')">复制邀约</button>
            <button onclick="approvalDecision('${item.approval?.id || ""}', 'approved')">审批通过</button>
            <button class="danger" onclick="approvalDecision('${item.approval?.id || ""}', 'rejected')">审批拒绝</button>
            <button onclick="sendOutbox('${item.id}')">已发送</button>
            <button onclick="feedback('${item.id}', '${item.game_id}', '${item.customer_id}', 'accepted')">已确认</button>
            <button onclick="feedback('${item.id}', '${item.game_id}', '${item.customer_id}', 'arrived')">已到店</button>
            <button onclick="feedback('${item.id}', '${item.game_id}', '${item.customer_id}', 'declined')">拒绝</button>
            <button onclick="feedback('${item.id}', '${item.game_id}', '${item.customer_id}', 'no_reply')">未回复</button>
            <button class="danger" onclick="feedback('${item.id}', '${item.game_id}', '${item.customer_id}', 'do_not_disturb')">别再打扰</button>
          </div>
          <div class="message-grid">
            <input id="candidate-reply-${item.id}" placeholder="模拟候选人回复，如：可以 / 今天不来 / 几点啊" />
            <button onclick="candidateReply('${item.id}')">模拟回复</button>
          </div>
          <div class="draft compact" id="candidate-result-${item.id}">候选人回复后，系统会给出老板下一句建议。</div>
          <div class="conversation" id="candidate-conversation-${item.id}">
            ${renderCandidateConversation(item.conversation || [])}
          </div>
        </div>
        `).join("")}
      ` : "";
      box.innerHTML = poolHtml + outboxHtml;
    }

    function renderCandidateConversation(turns) {
      if (!turns.length) {
        return `<div class="muted tiny">暂无模拟会话。</div>`;
      }
      return turns.map((turn, index) => `
        <div class="turn">
          <div class="muted tiny">第 ${index + 1} 轮 · ${escapeHtml(turn.status || turn.feedback_type || "")}</div>
          <div><strong>候选人：</strong>${escapeHtml(turn.candidate_text || "-")}</div>
          <div><strong>老板：</strong>${escapeHtml(turn.boss_reply || "-")}</div>
        </div>
      `).join("");
    }

    async function copyDraft(id) {
      await navigator.clipboard.writeText(document.getElementById(`draft-${id}`).textContent);
      const item = findOutbox(id);
      if (item) await feedback(id, item.game_id, item.customer_id, "copied");
    }

    async function copyPoolReply(gameId) {
      await navigator.clipboard.writeText(document.getElementById(`pool-${gameId}`).textContent);
    }

    function findOutbox(id) {
      if (!currentAnalysis) return null;
      return (currentAnalysis.outbox || []).find(item => item.id === id);
    }

    async function approvalDecision(approvalId, decision) {
      if (!approvalId) {
        alert("这条草稿还没有审批请求，请刷新后再试。");
        return;
      }
      const data = await api("/api/approval-decision", {
        method: "POST",
        body: JSON.stringify({approval_id: approvalId, decision})
      });
      const approval = data.approval || {};
      if (currentAnalysis?.outbox && approval.target_type === "outbox") {
        currentAnalysis.outbox = currentAnalysis.outbox.map(item => (
          item.id === approval.target_id
            ? {...item, approval, approval_status: approvalStatusLabel(approval.status), status: approvalStatusLabel(approval.status), message_text: approval.final_message_text || item.message_text}
            : item
        ));
        renderCandidates(currentAnalysis.outbox || [], currentAnalysis.pool_matches || []);
      }
      renderState(data.state);
    }

    async function feedback(outboxId, gameId, customerId, feedbackType) {
      const data = await api("/api/feedback", {
        method: "POST",
        body: JSON.stringify({
          outbox_id: outboxId || null,
          game_id: gameId || null,
          customer_id: customerId || null,
          feedback_type: feedbackType
        })
      });
      renderState(data.state);
    }

    async function candidateReply(outboxId) {
      const input = document.getElementById(`candidate-reply-${outboxId}`);
      const text = (input?.value || "").trim();
      if (!text) {
        alert("请输入候选人的回复内容。");
        return;
      }
      const data = await api("/api/candidate-message", {
        method: "POST",
        body: JSON.stringify({outbox_id: outboxId, text})
      });
      const candidate = data.candidate_message || {};
      if (currentAnalysis?.outbox && data.outbox_item) {
        currentAnalysis.outbox = currentAnalysis.outbox.map(item => (
          item.id === outboxId ? {...item, ...data.outbox_item, status: data.outbox_item.status, approval_status: data.outbox_item.status} : item
        ));
        renderCandidates(currentAnalysis.outbox || [], currentAnalysis.pool_matches || []);
      }
      const freshResultBox = document.getElementById(`candidate-result-${outboxId}`);
      if (freshResultBox) {
        const source = candidate.reply_source === "llm" ? `LLM：${candidate.model || "-"}` : "规则兜底";
        const followup = data.organizer_followup;
        freshResultBox.innerHTML = `
          <div>识别：${escapeHtml(candidate.status || candidate.intent || "-")}；${escapeHtml(source)}；老板建议回复：${escapeHtml(candidate.suggested_boss_reply || "-")}</div>
          ${followup ? `<div class="draft compact"><strong>待问${escapeHtml(followup.recipient_name || "发起人")}：</strong>${escapeHtml(followup.message_text || "-")}</div>` : ""}
        `;
      }
      renderState(data.state);
    }

    async function gameFeedback(gameId, feedbackType) {
      const data = await api("/api/feedback", {
        method: "POST",
        body: JSON.stringify({game_id: gameId, feedback_type: feedbackType})
      });
      renderState(data.state);
    }

    async function clearBoard() {
      if (!confirm("确定清空当前局看板吗？历史日志和复盘仍会保留。")) return;
      const data = await api("/api/clear-board", {
        method: "POST",
        body: JSON.stringify({reason: "老板在试用台手动清空当前局看板"})
      });
      renderState(data.state);
      alert(`已清空 ${data.cleared_count || 0} 个当前局。`);
    }

    async function clearShortMemory() {
      const senderId = document.getElementById("senderId").value || "anonymous";
      const conversationId = document.getElementById("conversationId").value || "boss_trial";
      if (!confirm(`确定清空 ${conversationId} / ${senderId} 的短期记忆吗？客户画像、当前局和日志不会删除。`)) return;
      const data = await api("/api/clear-short-memory", {
        method: "POST",
        body: JSON.stringify({
          sender_id: senderId,
          conversation_id: conversationId,
          reason: "老板在试用台手动清空当前客户短期记忆"
        })
      });
      renderState(data.state);
      alert(`已清空 ${data.cleared_count || 0} 条短期记忆。`);
    }

    async function manualCreateGame() {
      const payload = {
        organizer_id: "boss_manual",
        organizer_name: document.getElementById("manualOrganizerName").value || "老板手动创建",
        game_type: document.getElementById("manualGameType").value,
        variant: normalizeVariant(document.getElementById("manualVariant").value),
        level: document.getElementById("manualLevel").value,
        start_time: document.getElementById("manualStartTime").value,
        current_player_count: Number(document.getElementById("manualCurrentPlayers").value),
        missing_count: Number(document.getElementById("manualMissingCount").value),
        duration_hours: Number(document.getElementById("manualDurationHours").value),
        smoke: document.getElementById("manualSmoke").value,
        status: document.getElementById("manualStatus").value,
        source_text: document.getElementById("manualSourceText").value
      };
      const data = await api("/api/manual-create-game", {
        method: "POST",
        body: JSON.stringify(payload)
      });
      renderState(data.state);
      document.getElementById("manualSourceText").value = "";
    }

    function normalizeVariant(value) {
      const text = String(value || "").trim();
      return {
        "财敲": "caiqiao",
        "幺鸡": "yaoji",
        "妖鸡": "yaoji",
        "素鸡": "suji",
        "幺鸡47": "yaoji_47"
      }[text] || "";
    }

    async function recordEvalCase(caseType) {
      if (!currentAnalysis) {
        alert("请先解析一条消息，再归档评测样本。");
        return;
      }
      const expectedAction = document.getElementById("evalExpectedAction").value;
      const expected = {};
      if (caseType === "golden" && expectedAction) {
        expected.action = expectedAction;
        expected.should_reply = Boolean(currentAnalysis.decision?.should_reply);
      }
      const data = await api("/api/eval-cases", {
        method: "POST",
        body: JSON.stringify({
          case_type: caseType,
          source_trace_id: currentAnalysis.trace_id,
          sender_id: currentAnalysis.sender_id,
          sender_name: currentAnalysis.sender_name,
          text: currentAnalysis.source_text || document.getElementById("messageText").value,
          note: document.getElementById("evalNote").value,
          tags: splitFreeText(document.getElementById("evalTags").value),
          expected,
          analysis: currentAnalysis
        })
      });
      document.getElementById("evalResult").textContent = `${caseType} 已写入：${data.path}，id=${data.record_id}`;
      renderEvalOverview(data.overview || {});
    }

    function renderState(state) {
      window.latestState = state || {};
      renderCache(state.cache || {});
      renderGames(state.games || []);
      renderRecap(state.recap || {});
      renderCustomers(state.customers || []);
      renderEvalOverview(state.evals || {});
    }

    function renderCache(cache) {
      const status = document.getElementById("cacheStatus");
      if (!status) return;
      if (cache.redis_enabled) {
        status.className = "pill good";
        status.textContent = "Redis 短期记忆已启用";
      } else {
        status.className = "pill warn";
        status.textContent = "仅 SQLite，Redis 未启用";
      }
    }

    function renderGames(games) {
      const box = document.getElementById("gameBoard");
      if (!games.length) {
        box.textContent = "暂无当前局。";
        return;
      }
      box.innerHTML = games.map(game => {
        const outbox = game.outbox || [];
        const followups = game.followups || [];
        const confirmed = game.confirmed_count ?? outbox.filter(item => ["已确认", "已到店"].includes(item.status)).length;
        const missing = game.parsed?.missing_count;
        const stillMissing = game.remaining_missing_count ?? (missing == null ? "-" : Math.max(0, missing - confirmed));
        const duration = game.parsed?.duration_text ? `，${game.parsed.duration_text}` : (game.parsed?.duration_hours ? `，约 ${game.parsed.duration_hours} 小时` : "");
        const participants = game.participants || [];
        const title = game.live_summary || game.parsed?.live_summary || dynamicGameSummary(game, stillMissing);
        return `
          <div class="game">
            <strong>${escapeHtml(title)}</strong>
            <span class="pill">${escapeHtml(game.status)}</span>
            <div class="muted tiny">已确认 ${confirmed} 人，还缺 ${stillMissing} 人${escapeHtml(duration)}</div>
            <div>${participants.map(item => `<span class="pill">${escapeHtml(item.customer_name || "-")}：${escapeHtml(item.status || item.role || "")}${item.count ? ` x${escapeHtml(item.count)}` : ""}</span>`).join("")}</div>
            <div class="message-grid">
              <div>
                <div class="muted tiny">用户消息</div>
                <div class="draft compact">${escapeHtml(game.source_text || "-")}</div>
              </div>
              <div>
                <div class="muted tiny">系统建议回复</div>
                <div class="draft compact">${escapeHtml(game.reply_text || "-")}</div>
              </div>
            </div>
            <div>${outbox.slice(0, 6).map(item => `<span class="pill">${escapeHtml(item.customer_name)}：${escapeHtml(item.status)}</span>`).join("")}</div>
            ${followups.length ? `
              <div class="conversation">
                <div class="muted tiny">待协商确认</div>
                ${followups.slice(0, 5).map(item => `
                  <div class="turn">
                    <div><strong>给${escapeHtml(item.recipient_name || "-")}：</strong>${escapeHtml(item.message_text || "-")}</div>
                    <div class="muted tiny">${escapeHtml(item.status || "待审批")} · ${escapeHtml(item.reason || "")}</div>
                  </div>
                `).join("")}
              </div>
            ` : ""}
            <div class="toolbar">
              <button onclick="gameFeedback('${game.id}', 'game_success')">已成局</button>
              <button onclick="gameFeedback('${game.id}', 'game_cancelled')">局取消</button>
            </div>
          </div>
        `;
      }).join("");
    }

    function dynamicGameSummary(game, stillMissing) {
      const parsed = game.parsed || {};
      const rules = (parsed.rules || []).filter(rule => !["杭麻", "川麻", "麻将", parsed.game_label].includes(rule));
      const level = parsed.level ? `${parsed.level}档` : "";
      const missing = stillMissing === "-" ? "" : (Number(stillMissing) > 0 ? `缺${stillMissing}` : "人齐");
      return [parsed.game_label, level, parsed.start_time, missing, ...rules].filter(Boolean).join(" ") || parsed.summary || game.source_text || "-";
    }

    function renderRecap(recap) {
      const games = recap.games_by_status || {};
      const outbox = recap.outbox_by_status || {};
      const top = recap.top_customers || [];
      const archived = window.latestState?.recent_archived_games || [];
      document.getElementById("recapBox").innerHTML = `
        <div class="kv">
          <div>今日组局</div><div>${Object.values(games).reduce((a, b) => a + b, 0)} 次</div>
          <div>邀约草稿</div><div>${Object.values(outbox).reduce((a, b) => a + b, 0)} 条</div>
          <div>成局</div><div>${games["已成局"] || 0} 次</div>
        </div>
        <h3>响应较好客户</h3>
        <div>${top.map(item => `<span class="pill good">${escapeHtml(item.display_name)} ${Math.round(item.response_rate * 100)}%</span>`).join("") || "暂无"}</div>
        <h3>建议</h3>
        <div>${(recap.suggestions || []).map(item => `<div class="muted">- ${escapeHtml(item)}</div>`).join("")}</div>
        <h3>最近归档局</h3>
        <div>${archived.slice(0, 5).map(game => `
          <div class="draft compact">
            <strong>${escapeHtml(game.parsed?.summary || game.source_text || game.id)}</strong>
            <span class="pill">${escapeHtml(game.status)}</span>
            <div class="muted tiny">${escapeHtml(game.final_reason || "暂无归档原因")}</div>
          </div>
        `).join("") || "暂无归档局。"}</div>
      `;
    }

    function renderEvalOverview(evals) {
      const box = document.getElementById("evalOverview");
      if (!box) return;
      const counts = evals.counts || {};
      const paths = evals.paths || {};
      box.innerHTML = `
        <div class="kv">
          <div>golden</div><div>${counts.golden ?? 0} 条</div>
          <div>boss-trial golden</div><div>${counts.boss_trial_golden ?? 0} 条</div>
          <div>badcase</div><div>${counts.badcase ?? 0} 条</div>
          <div>few-shot</div><div>${counts.few_shot ?? 0} 条</div>
          <div>skills</div><div>${counts.skills ?? 0} 条</div>
        </div>
        <h3>文件路径</h3>
        <div class="tiny muted">golden：${escapeHtml(paths.golden || "-")}</div>
        <div class="tiny muted">boss-trial golden：${escapeHtml(paths.boss_trial_golden || "-")}</div>
        <div class="tiny muted">badcase：${escapeHtml(paths.badcase || "-")}</div>
        <div class="tiny muted">few-shot：${escapeHtml(paths.few_shot || "-")}</div>
        <div class="tiny muted">skills：${escapeHtml(paths.skills || "-")}</div>
        <h3>回归命令</h3>
        <pre>${escapeHtml(evals.runner || "PYTHONPATH=src python scripts/run_scenario_eval.py")}</pre>
      `;
    }

    function renderCustomers(customers) {
      const box = document.getElementById("customerList");
      box.innerHTML = customers.map(customer => `
        <div class="customer">
          <strong>${escapeHtml(customer.display_name)}</strong>
          <span class="pill">${escapeHtml(customer.gender_label || "未知")}</span>
          ${customer.no_contact ? '<span class="pill warn">勿扰</span>' : ''}
          <div class="muted tiny">${escapeHtml(customer.contact || "")}</div>
          <div>${(customer.preferred_games || []).map(item => `<span class="pill">${escapeHtml(item)}</span>`).join("")}</div>
          <div>${(customer.preferred_levels || []).map(item => `<span class="pill">${escapeHtml(item)}档</span>`).join("")}</div>
          <div class="muted tiny">响应率 ${Math.round((customer.response_rate || 0) * 100)}%，最近邀约 ${formatTime(customer.last_invited_at)}</div>
          <button onclick='editCustomer(${JSON.stringify(customer).replaceAll("'", "&#39;")})'>编辑</button>
        </div>
      `).join("");
    }

    function editCustomer(customer) {
      document.getElementById("customerName").value = customer.display_name || "";
      document.getElementById("customerContact").value = customer.contact || "";
      document.getElementById("customerGender").value = customer.gender || "unknown";
      document.getElementById("customerGames").value = (customer.preferred_games || []).join(",");
      document.getElementById("customerLevels").value = (customer.preferred_levels || []).join(",");
      document.getElementById("customerHours").value = (customer.usual_start_hours || []).join(",");
      document.getElementById("customerSmoke").value = customer.smoke_preference || "any";
      document.getElementById("customerNotes").value = customer.notes || "";
      document.getElementById("senderName").value = customer.display_name || "";
      document.getElementById("senderId").value = customer.id || "";
    }

    async function saveCustomer() {
      const name = document.getElementById("customerName").value.trim();
      const payload = {
        id: name,
        display_name: name,
        contact: document.getElementById("customerContact").value,
        gender: document.getElementById("customerGender").value,
        preferred_games: splitInput("customerGames"),
        preferred_levels: splitInput("customerLevels"),
        usual_start_hours: splitInput("customerHours").map(Number).filter(Number.isFinite),
        smoke_preference: document.getElementById("customerSmoke").value,
        notes: document.getElementById("customerNotes").value,
        response_rate: 0.5
      };
      await api("/api/customers", {method: "POST", body: JSON.stringify(payload)});
      await loadState();
    }

    async function sendOutbox(outboxId) {
      const data = await api("/api/send-outbox", {
        method: "POST",
        body: JSON.stringify({outbox_id: outboxId, channel: "manual"})
      });
      if (currentAnalysis?.outbox && data.outbox_item) {
        currentAnalysis.outbox = currentAnalysis.outbox.map(item => (
          item.id === outboxId
            ? {...item, ...data.outbox_item, approval_status: data.outbox_item.approval_status || data.outbox_item.status}
            : item
        ));
        renderCandidates(currentAnalysis.outbox || [], currentAnalysis.pool_matches || []);
      }
      await loadState();
    }

    function splitInput(id) {
      return document.getElementById(id).value.split(/[,，、\s]+/).map(s => s.trim()).filter(Boolean);
    }

    function splitFreeText(value) {
      return String(value || "").split(/[,，、\s]+/).map(s => s.trim()).filter(Boolean);
    }

    async function copyText(id) {
      await navigator.clipboard.writeText(document.getElementById(id).value);
    }

    async function copyRendered(id) {
      await navigator.clipboard.writeText(document.getElementById(id).textContent);
    }

    function fieldLabel(value) {
      return {
        play_type: "玩法",
        start_time: "时间",
        stake: "档位",
        known_players: "人数",
        smoke: "烟况",
        duration: "时长"
      }[value] || value;
    }

    function intentLabel(action) {
      return {
        ask_clarification: "想打/想组局，信息待确认",
        create_pending_game: "明确组局，先入待组局",
        create_game: "创建组局需求",
        queue_invites: "找人组局",
        accept_seat: "接受邀约/报名入局",
        decline_invite: "拒绝邀约",
        close_game: "取消或关闭组局",
        ignore: "无关消息，无需回复",
        human_review: "高风险或不确定，转人工"
      }[action] || action || "-";
    }

    function formatTime(value) {
      if (!value) return "-";
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return "-";
      return `${date.getMonth() + 1}-${date.getDate()} ${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
    }

    function approvalStatusLabel(value) {
      return {
        pending: "待审批",
        approved: "已审批",
        rejected: "审批拒绝"
      }[String(value || "").toLowerCase()] || value || "待审批";
    }

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, char => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }[char]));
    }

    loadState().catch(err => alert(err.message));
  </script>
</body>
</html>
"""


class BossTrialHandler(BaseHTTPRequestHandler):
    service: BossTrialService

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/logs"}:
            self.send_response(200)
            self.send_header("content-type", "text/html; charset=utf-8")
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._html(HTML)
            return
        if parsed.path == "/logs":
            self._html(render_log_page())
            return
        if parsed.path == "/api/logs":
            trace_id = make_trace_id()
            lines = recent_log_lines()
            self._json(
                {
                    "trace_id": trace_id,
                    "path": str(LOG_PATH),
                    "line_count": len(lines),
                    "lines": lines,
                }
            )
            return
        if parsed.path == "/api/traces":
            trace_id = make_trace_id()
            params = parse_qs(parsed.query)
            requested_trace_id = str((params.get("trace_id") or params.get("traceId") or [""])[0] or "").strip()
            limit_raw = str((params.get("limit") or ["300"])[0] or "300")
            try:
                limit = int(limit_raw)
            except ValueError:
                limit = 300
            write_io_log(
                trace_id,
                "INFO",
                json_dumps(
                    {
                        "direction": "input",
                        "path": parsed.path,
                        "requested_trace_id": requested_trace_id,
                        "limit": limit,
                    }
                ),
            )
            result = self.service.trace_view(requested_trace_id, limit=limit)
            result["api_trace_id"] = trace_id
            result["requested_trace_id"] = requested_trace_id
            write_io_log(
                trace_id,
                "INFO",
                json_dumps(
                    {
                        "direction": "output",
                        "path": parsed.path,
                        "requested_trace_id": requested_trace_id,
                        "event_count": result.get("event_count"),
                        "trace_count": result.get("trace_count"),
                    }
                ),
            )
            self._json(result)
            return
        if parsed.path == "/api/state-transitions":
            trace_id = make_trace_id()
            params = parse_qs(parsed.query)
            entity_type = str((params.get("entity_type") or params.get("entityType") or [""])[0] or "").strip()
            entity_id = str((params.get("entity_id") or params.get("entityId") or [""])[0] or "").strip()
            requested_trace_id = str((params.get("trace_id") or params.get("traceId") or [""])[0] or "").strip()
            limit_raw = str((params.get("limit") or ["120"])[0] or "120")
            try:
                limit = int(limit_raw)
            except ValueError:
                limit = 120
            write_io_log(
                trace_id,
                "INFO",
                json_dumps(
                    {
                        "direction": "input",
                        "path": parsed.path,
                        "entity_type": entity_type,
                        "entity_id": entity_id,
                        "requested_trace_id": requested_trace_id,
                        "limit": limit,
                    }
                ),
            )
            result = self.service.state_transition_view(
                entity_type=entity_type or None,
                entity_id=entity_id or None,
                trace_id=requested_trace_id or None,
                limit=limit,
            )
            result["api_trace_id"] = trace_id
            write_io_log(
                trace_id,
                "INFO",
                json_dumps(
                    {
                        "direction": "output",
                        "path": parsed.path,
                        "event_count": result.get("event_count"),
                    }
                ),
            )
            self._json(result)
            return
        if parsed.path == "/api/runtime-policy":
            trace_id = make_trace_id()
            write_io_log(trace_id, "INFO", json_dumps({"direction": "input", "path": parsed.path}))
            result = self.service.runtime_policy()
            result["trace_id"] = trace_id
            write_io_log(trace_id, "INFO", log_output_content(parsed.path, result))
            self._json(result)
            return
        if parsed.path == "/api/state":
            trace_id = make_trace_id()
            write_io_log(trace_id, "INFO", json_dumps({"direction": "input", "path": parsed.path}))
            state = self.service.state()
            state["trace_id"] = trace_id
            write_io_log(trace_id, "INFO", log_output_content(parsed.path, state))
            self._json(state)
            return
        if parsed.path == "/api/eval-cases":
            trace_id = make_trace_id()
            write_io_log(trace_id, "INFO", json_dumps({"direction": "input", "path": parsed.path}))
            result = self.service.eval_overview()
            result["trace_id"] = trace_id
            write_io_log(trace_id, "INFO", log_output_content(parsed.path, {"ok": True, "overview": result}))
            self._json(result)
            return
        self._json({"error": "not_found"}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        trace_id = make_trace_id()
        try:
            body = self._read_json()
            trace_id = str(body.get("trace_id") or trace_id)
            body["trace_id"] = trace_id
            write_io_log(trace_id, "INFO", log_input_content(parsed.path, body))
            if parsed.path == "/api/analyze":
                result = (
                    self.service.analyze_controlled(body)
                    if use_controlled_trial_workflow(body)
                    else self.service.analyze(body)
                )
                result["trace_id"] = trace_id
                write_io_log(trace_id, "INFO", log_output_content(parsed.path, result))
                self._json(result)
                return
            if parsed.path == "/api/customers":
                result = self.service.save_customer(body)
                result["trace_id"] = trace_id
                write_io_log(trace_id, "INFO", log_output_content(parsed.path, result))
                self._json(result)
                return
            if parsed.path == "/api/feedback":
                result = self.service.feedback(body)
                result["trace_id"] = trace_id
                write_io_log(trace_id, "INFO", log_output_content(parsed.path, result))
                self._json(result)
                return
            if parsed.path == "/api/approval-decision":
                result = self.service.approval_decision(body)
                result["trace_id"] = trace_id
                write_io_log(trace_id, "INFO", log_output_content(parsed.path, result))
                self._json(result)
                return
            if parsed.path == "/api/send-outbox":
                result = self.service.send_outbox(body)
                result["trace_id"] = trace_id
                write_io_log(trace_id, "INFO", log_output_content(parsed.path, result))
                self._json(result)
                return
            if parsed.path == "/api/runtime-policy":
                result = self.service.update_runtime_policy(body)
                result["trace_id"] = trace_id
                write_io_log(trace_id, "INFO", log_output_content(parsed.path, result))
                self._json(result)
                return
            if parsed.path == "/api/candidate-message":
                result = self.service.candidate_message(body)
                result["trace_id"] = trace_id
                write_io_log(trace_id, "INFO", log_output_content(parsed.path, result))
                self._json(result)
                return
            if parsed.path == "/api/clear-board":
                result = self.service.clear_board(body)
                result["trace_id"] = trace_id
                write_io_log(trace_id, "INFO", log_output_content(parsed.path, result))
                self._json(result)
                return
            if parsed.path == "/api/clear-short-memory":
                result = self.service.clear_short_memory(body)
                result["trace_id"] = trace_id
                write_io_log(trace_id, "INFO", log_output_content(parsed.path, result))
                self._json(result)
                return
            if parsed.path == "/api/manual-create-game":
                result = self.service.manual_create_game(body)
                result["trace_id"] = trace_id
                write_io_log(trace_id, "INFO", log_output_content(parsed.path, result))
                self._json(result)
                return
            if parsed.path == "/api/eval-cases":
                result = self.service.record_eval_case(body)
                result["trace_id"] = trace_id
                write_io_log(trace_id, "INFO", log_output_content(parsed.path, result))
                self._json(result)
                return
        except Exception as exc:
            payload = {"trace_id": trace_id, "error": type(exc).__name__, "message": str(exc)}
            write_io_log(
                trace_id,
                "ERROR",
                json_dumps(
                    {
                        "direction": "output",
                        "path": parsed.path,
                        "error": payload["error"],
                        "message": payload["message"],
                    }
                ),
            )
            self._json(payload, status=400)
            return
        self._json({"error": "not_found"}, status=404)

    def log_message(self, format: str, *args: Any) -> None:
        print(format % args)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8")) if raw else {}

    def _html(self, body: str, status: int = 200) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def build_redis_cache_from_env() -> RedisCache | None:
    redis_url = os.environ.get("MAHJONG_REDIS_URL", DEFAULT_REDIS_URL).strip()
    if redis_url.lower() in {"", "0", "false", "off", "none", "disabled"}:
        print("Redis cache disabled by MAHJONG_REDIS_URL.")
        return None
    timeout = float(os.environ.get("MAHJONG_REDIS_TIMEOUT_SECONDS", "0.3"))
    try:
        cache = RedisCache.from_url(redis_url, timeout_seconds=timeout)
        cache.ping()
    except (RedisCacheError, ValueError) as exc:
        print(f"Redis cache unavailable, continue with SQLite only: {exc}")
        return None
    print(f"Redis cache enabled: {_redact_redis_url(redis_url)}")
    return cache


def _redact_redis_url(redis_url: str) -> str:
    parsed = urlparse(redis_url)
    if not parsed.password:
        return redis_url
    username = parsed.username or ""
    auth = f"{username}:***@" if username else "***@"
    host = parsed.hostname or "127.0.0.1"
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{auth}{host}{port}{parsed.path or ''}"


def load_local_env(path: pathlib.Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


def main() -> None:
    load_local_env()
    store = TrialStore(DB_PATH)
    cache = build_redis_cache_from_env()
    BossTrialHandler.service = BossTrialService(store, cache=cache)
    server = ThreadingHTTPServer(("127.0.0.1", 8790), BossTrialHandler)
    print("Boss trial app listening on http://127.0.0.1:8790")
    print("All outbound messages are drafts only. Use copy buttons for manual sending.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nBoss trial app stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
