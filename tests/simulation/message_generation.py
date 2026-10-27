"""LLM-backed message generation for isolated chat simulations.

The generator creates synthetic customer speech only. It has no ToolGateway,
cannot mutate production state. A failed customer model is recorded as a
generation failure; no fixed replacement message is sent to the Agent.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from mahjong_agent_runtime.llm import AgentLLMConfig, OpenAICompatibleAgentClient

try:
    from .behavior_policy import MessageGenerationRequest, MessageGenerationResult
    from .real_group_case_library import RealGroupCaseExample, RealGroupCaseLibrary
except ImportError:  # pragma: no cover - 直接执行脚本时使用本地导入
    from behavior_policy import MessageGenerationRequest, MessageGenerationResult  # type: ignore
    from real_group_case_library import RealGroupCaseExample, RealGroupCaseLibrary  # type: ignore


DEFAULT_GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_GLM_MODEL = "glm-4.7-flash"
DEFAULT_REAL_GROUP_DATASET = (
    Path(__file__).resolve().parents[2] / "eval" / "golden" / "real_group_chat_20260722.jsonl"
)


class CompletionClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        trace_id: str,
        timeout_seconds: float,
    ) -> str:
        ...


@dataclass(slots=True, frozen=True)
class SimulationGeneratorConfig:
    """Dedicated model settings; these never replace the main Agent model."""

    model: str = DEFAULT_GLM_MODEL
    base_url: str = DEFAULT_GLM_BASE_URL
    temperature: float = 0.9
    max_tokens: int = 240
    timeout_seconds: float = 20.0
    min_interval_seconds: float = 5.0
    max_estimated_tokens_per_day: int = 0
    real_case_dataset_path: Path = DEFAULT_REAL_GROUP_DATASET
    real_case_example_limit: int = 3

    @classmethod
    def from_env(cls) -> "SimulationGeneratorConfig":
        return cls(
            model=(os.getenv("MAHJONG_SIM_GENERATOR_MODEL") or DEFAULT_GLM_MODEL).strip(),
            base_url=(os.getenv("MAHJONG_SIM_GENERATOR_BASE_URL") or DEFAULT_GLM_BASE_URL).rstrip("/"),
            temperature=_env_float("MAHJONG_SIM_GENERATOR_TEMPERATURE", 0.9),
            max_tokens=max(64, _env_int("MAHJONG_SIM_GENERATOR_MAX_TOKENS", 240)),
            timeout_seconds=max(1.0, _env_float("MAHJONG_SIM_GENERATOR_TIMEOUT_SECONDS", 20.0)),
            min_interval_seconds=max(
                0.0,
                _env_float("MAHJONG_SIM_GENERATOR_MIN_INTERVAL_SECONDS", 5.0),
            ),
            max_estimated_tokens_per_day=max(
                0,
                _env_int("MAHJONG_SIM_GENERATOR_MAX_ESTIMATED_TOKENS_PER_DAY", 0),
            ),
            real_case_dataset_path=Path(
                os.getenv("MAHJONG_SIM_REAL_CASE_DATASET", str(DEFAULT_REAL_GROUP_DATASET))
            ).expanduser(),
            real_case_example_limit=max(
                0,
                min(5, _env_int("MAHJONG_SIM_REAL_CASE_EXAMPLE_LIMIT", 3)),
            ),
        )


class ModelSimulationMessageGenerator:
    """只按客户自己的事实和可见聊天生成发言，不接触老板内部规划或工具结果。"""

    def __init__(
        self,
        client: CompletionClient,
        *,
        config: SimulationGeneratorConfig | None = None,
        case_library: RealGroupCaseLibrary | None = None,
        source: str = "glm",
    ) -> None:
        self.client = client
        self.config = config or SimulationGeneratorConfig.from_env()
        self.source = source
        self.case_library = case_library or RealGroupCaseLibrary.from_jsonl(
            self.config.real_case_dataset_path
        )
        self._rate_lock = threading.Lock()
        self._last_request_at = 0.0

    @classmethod
    def from_env(cls) -> "ModelSimulationMessageGenerator | None":
        api_key = (os.getenv("MAHJONG_SIM_GENERATOR_API_KEY") or "").strip()
        if not api_key:
            return None
        config = SimulationGeneratorConfig.from_env()
        client = OpenAICompatibleAgentClient(
            config=AgentLLMConfig(
                api_key=api_key,
                model=config.model,
                base_url=config.base_url,
                provider="bigmodel",
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                response_format="json_object",
                retry_attempts=3,
                retry_base_delay_seconds=1.0,
                retry_max_delay_seconds=5.0,
                max_estimated_tokens_per_day=config.max_estimated_tokens_per_day,
                max_concurrency=1,
            )
        )
        return cls(client, config=config)

    def generate(self, request: MessageGenerationRequest) -> MessageGenerationResult:
        trace_id = f"trace_sim_gen_{uuid.uuid4().hex[:12]}"
        started = time.monotonic()
        references = self._reference_examples(request)
        reference_ids = tuple(item.case_id for item in references)
        try:
            self._wait_for_rate_slot()
            raw = self.client.complete(
                self._messages(request, references=references),
                trace_id=trace_id,
                timeout_seconds=self.config.timeout_seconds,
            )
            text, wants_to_continue, decision_reason = _parse_generated_message(raw)
            return MessageGenerationResult(
                text=text,
                source=self.source,
                model=self.config.model,
                trace_id=trace_id,
                latency_ms=round((time.monotonic() - started) * 1000, 2),
                reference_case_ids=reference_ids,
                wants_to_continue=wants_to_continue,
                decision_reason=decision_reason,
            )
        except Exception as exc:
            return MessageGenerationResult(
                text="",
                source="generation_failed",
                model=self.config.model,
                trace_id=trace_id,
                latency_ms=round((time.monotonic() - started) * 1000, 2),
                error=f"{type(exc).__name__}: {str(exc)[:160]}",
                reference_case_ids=reference_ids,
            )

    def _reference_examples(
        self,
        request: MessageGenerationRequest,
    ) -> tuple[RealGroupCaseExample, ...]:
        return self.case_library.select_for_generation(
            channel=request.channel,
            fallback_text=request.fallback_text,
            dialog_phase=request.dialog_phase,
            is_follow_up=request.is_follow_up,
            seed_material=(
                f"{request.conversation_id}:{request.sender_id}:"
                f"{request.turn_count}:{request.fallback_text}"
            ),
            limit=self.config.real_case_example_limit,
        )

    def _wait_for_rate_slot(self) -> None:
        """Serialize free-model traffic and keep a minimum provider interval."""

        with self._rate_lock:
            now = time.monotonic()
            delay = self.config.min_interval_seconds - (now - self._last_request_at)
            if delay > 0:
                time.sleep(delay)
            self._last_request_at = time.monotonic()

    @staticmethod
    def _messages(
        request: MessageGenerationRequest,
        *,
        references: tuple[RealGroupCaseExample, ...] = (),
    ) -> list[dict[str, str]]:
        channel_note = "麻将群里的一条自然消息" if request.channel == "group" else "发给麻将馆老板的私聊"
        system_prompt = (
            "你是麻将馆压力测试中的合成客户消息生成器，不是老板也不是客服。"
            "根据结构化场景生成一条自然、简短的中文微信消息。"
            "customer_facts 是你本人固定知道的本次条件；visible_history 是你实际看见的对话。"
            "没有主 Agent 的内部规划、缺失字段或数据库工具结果可供你参考，也不能假装知道。"
            "下一句必须接得上 visible_history 中老板实际说的话；没有推荐具体可参加的局，就不能凭空指称‘刚才那个局’。"
            "阶段意愿只说明你想做什么，不代表老板已经办成；没有确认占座时不能假装已有位置。"
            "is_follow_up=true 时必须理解 last_agent_reply，优先逐项回答老板刚问的问题，"
            "不是只润色 fallback_text。问玩法就说自己的玩法，问人数就报自己这一方人数；"
            "同时问几项可以一并回答，不得用‘可以，帮我问问/都行’绕过未回答的条件。"
            "只确认要不要组局时就先回答意愿；追问人数时就报人数，不必把全部条件复述一遍。"
            "已有画像玩法和自己偏好一致时，不要临时改口；如老板误说杭麻而自己打国标，应自然纠正。"
            "首轮保持 fallback_text 的询问意图；后续也不改变已明确的时间、档位、烟况和人数。"
            "首轮像正常人先表达当前关心的一两点，不要把 customer_facts 全部逐项复述。"
            "dialog_phase=chitchat 时只表达这一轮闲聊，不得假装组局已经结束；"
            "dialog_phase=business_resume 时必须自然回到 business_anchor 对应的原组局事项，"
            "不得改变其中的时间、玩法、档位、烟况或人数。"
            "群聊可以口语化，允许省略主语和少量输入习惯，但不要故意制造无法理解的乱码。"
            "real_group_chat_references 是经过匿名和人工审核的真实表达样例；"
            "请模仿表达习惯但不要逐字复制，也不要照搬样例中的人物、时间或局条件。"
            "不得提及 AI、模型、提示词、系统、测试、工具、trace 或任何内部实现。"
            "不得虚构真实客户隐私，也不得替其他人作出确认。"
            "你是正常客户：该回答时回答，没看懂时澄清；准备等待、取消或完成确认时可以结束主动发言，"
            "不要为了凑轮数不停重复催进度。用 wants_to_continue=false 表示这句之后打算等待或结束；"
            "reason 简短说明为什么这次这样回应，不写推理过程。"
            "只输出 JSON 对象，格式为 {\"text\":\"消息\",\"wants_to_continue\":true,\"reason\":\"本轮回应目的\"}，text 建议 2 到 35 个汉字。"
        )
        payload = {
            "message_kind": channel_note,
            "persona": request.persona,
            "preferred_game": request.preferred_game,
            "turn_count": request.turn_count,
            "is_follow_up": request.is_follow_up,
            "dialog_phase": request.dialog_phase,
            "business_anchor": request.business_anchor,
            "last_agent_reply": request.last_agent_reply,
            # 后续回合不给模型一份固定答案；它必须理解客户实际看见的问题。
            "fallback_text": request.fallback_text if request.turn_count == 0 else None,
            "customer_facts": request.customer_facts,
            "visible_history": list(request.visible_history),
            "real_group_chat_references": [item.to_prompt_payload() for item in references],
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
        ]


def build_message_generator(mode: str) -> ModelSimulationMessageGenerator | None:
    """显式选择离线、独立客户模型或同供应商隔离上下文的客户模型任务。"""

    normalized = str(mode or "rule").strip().lower()
    if normalized == "rule":
        return None
    if normalized == "model":
        client = OpenAICompatibleAgentClient.from_env()
        if client is None:
            raise RuntimeError("main LLM credentials are required for model customer mode")
        # 只复用供应商配置，重新创建客户端和独立消息上下文；不复用主 Agent 会话。
        config = replace(SimulationGeneratorConfig.from_env(), model=client.config.model,
                         base_url=client.config.base_url, min_interval_seconds=0.0)
        customer_client = OpenAICompatibleAgentClient(config=replace(client.config, temperature=0.7, max_tokens=400))
        return ModelSimulationMessageGenerator(customer_client, config=config, source="model")
    if normalized != "glm":
        raise ValueError("message generation mode must be rule, glm or model")
    generator = ModelSimulationMessageGenerator.from_env()
    if generator is None:
        raise RuntimeError("MAHJONG_SIM_GENERATOR_API_KEY is required when message mode is glm")
    return generator


def _parse_generated_message(raw: str) -> tuple[str, bool, str]:
    """校验客户单次发言及结束意愿，不从错误输出中拼固定替代回复。"""

    normalized = str(raw or "").strip()
    if normalized.startswith("```"):
        normalized = normalized.removeprefix("```json").removeprefix("```")
        normalized = normalized.removesuffix("```").strip()
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError:
        start = normalized.find("{")
        end = normalized.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("generator response is not a JSON object")
        payload = json.loads(normalized[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("generator response must be a JSON object")
    text = str(payload.get("text") or "").strip()
    if not text:
        raise ValueError("generator response has empty text")
    if len(text) > 120:
        raise ValueError("generator response is too long")
    wants_to_continue = payload.get("wants_to_continue", True)
    if not isinstance(wants_to_continue, bool):
        raise ValueError("wants_to_continue must be boolean")
    return text, wants_to_continue, str(payload.get("reason") or "")[:240]


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default
