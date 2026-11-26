"""为独立仿真解析 OpenCode Go 和 DeepSeek 官方模型连接。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import tomllib

from mahjong_agent_runtime.llm import AgentLLMConfig, OpenAICompatibleAgentClient, OPENCODE_GO_BASE_URL
from mahjong_agent_runtime.review_config import build_reply_review_client

from .venue_write_guard import VenueWriteGuard


@dataclass(slots=True)
class VenueDeepSeekResponsesConfig(AgentLLMConfig):
    """为沙盒明确传递 Responses 的 DeepSeek 推理选项。"""

    responses_no_reasoning: bool = False

    def _request_payload(self, messages):
        """复用 Runtime 的消息和协议封装，只补仿真实验明确指定的无推理选项。"""
        endpoint, payload = AgentLLMConfig._request_payload(self, messages)
        if (self.provider in {"opencode_go", "deepseek"} and self.api_protocol == "responses"
                and self.thinking_mode == "disabled" and self.responses_no_reasoning
                and self.model.rsplit("/", 1)[-1].startswith("deepseek-")):
            payload["reasoning"] = {"effort": "none"}
        return endpoint, payload


def opencode_config(config_dir: Path, *, max_concurrency: int) -> AgentLLMConfig:
    """从命名模型配置读取模型，仅复用环境变量或显式配置的本地纯文本凭证。"""
    profile_path = config_dir / "opencode-go.config.toml"
    settings_path = config_dir / "config.toml"
    explicit_key = os.environ.get("OPENCODE_GO_API_KEY") or os.environ.get("OPENCODE_API_KEY")
    explicit_model = os.environ.get("OPENCODE_GO_MODEL")
    if not profile_path.is_file() and not settings_path.is_file() and not (explicit_key and explicit_model):
        raise FileNotFoundError("未找到本机 OpenCode Go 配置")
    settings = tomllib.loads(settings_path.read_text(encoding="utf-8")) if settings_path.is_file() else {}
    profile = tomllib.loads(profile_path.read_text(encoding="utf-8")) if profile_path.is_file() else {}
    providers = {**settings.get("model_providers", {}), **profile.get("model_providers", {})}
    provider = providers.get("opencode-go") or providers.get("opencode_go") or {}
    if not profile and not provider and not explicit_key:
        raise FileNotFoundError("未配置本机 OpenCode Go")
    api_key = explicit_key
    env_key = provider.get("env_key")
    if not api_key and isinstance(env_key, str):
        api_key = os.environ.get(env_key)
    if not api_key:
        # 支持本机已配置的单文件 cat 读取方式。直接读文件，避免执行配置中的任意命令。
        auth = provider.get("auth") or {}
        args = auth.get("args") or []
        command = str(auth.get("command") or "")
        key_file = Path(args[0]).expanduser() if command in {"cat", "/bin/cat", "/usr/bin/cat"} and len(args) == 1 else None
        if key_file is not None and key_file.is_absolute() and key_file.is_file():
            api_key = key_file.read_text(encoding="utf-8").strip()
    model = explicit_model or profile.get("model")
    if not model and settings.get("model_provider") in {"opencode-go", "opencode_go"}:
        model = settings.get("model")
    if not api_key or not model:
        raise ValueError("OpenCode Go 需要可读的本机凭证及 OPENCODE_GO_MODEL 或本机配置中的模型")
    reasoning = os.environ.get("OPENCODE_GO_RESPONSE_REASONING")
    if reasoning is None:
        reasoning = "none" if str(model).rsplit("/", 1)[-1].startswith("deepseek-") else "provider-default"
    if reasoning not in {"provider-default", "none", "low", "high", "max"}:
        raise ValueError("OPENCODE_GO_RESPONSE_REASONING 必须为 provider-default、none、low、high 或 max")
    explicit_effort = reasoning if reasoning in {"low", "high", "max"} else None
    return VenueDeepSeekResponsesConfig(
        api_key=api_key, model=str(model), provider="opencode_go", base_url=OPENCODE_GO_BASE_URL,
        api_protocol=str(provider.get("wire_api") or "responses"),
        max_tokens=max(1, int(os.environ.get("OPENCODE_GO_MAX_OUTPUT_TOKENS", "7200"))),
        max_concurrency=max_concurrency, retry_attempts=2,
        thinking_mode="enabled" if explicit_effort else "disabled",
        reasoning_effort=explicit_effort, responses_no_reasoning=reasoning == "none",
    )


def deepseek_official_config(*, max_concurrency: int) -> AgentLLMConfig:
    """只将官方 DeepSeek 密钥发给固定官方地址，仿真各组件使用同一 Flash 版本。"""
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("MAHJONG_DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DeepSeek 官方仿真需要 DEEPSEEK_API_KEY 或 MAHJONG_DEEPSEEK_API_KEY")
    return VenueDeepSeekResponsesConfig(
        api_key=api_key, model="deepseek-flash", provider="deepseek", base_url="https://api.deepseek.com",
        api_protocol="responses", max_tokens=7200, max_concurrency=max_concurrency,
        retry_attempts=2, thinking_mode="disabled", responses_no_reasoning=True,
    )


def opencode_auxiliary(config: AgentLLMConfig, *, max_tokens: int, workers: int) -> OpenAICompatibleAgentClient:
    """同一官方连接可复用认证；角色、审查与主 Agent 各自持有请求上下文和限额。"""
    return OpenAICompatibleAgentClient(replace(config, max_tokens=max_tokens, max_concurrency=workers))


def venue_planner_config(config: AgentLLMConfig, *, reasoning_low: bool) -> AgentLLMConfig:
    """仅对仿真的规划客户端做低推理对照，角色与独立判定沿用原连接。"""
    return replace(config, thinking_mode="enabled", reasoning_effort="low") if reasoning_low else config


def venue_reply_review_no_reasoning(config: AgentLLMConfig) -> OpenAICompatibleAgentClient:
    """保留完整的独立审查及预算，只为沙盒 OpenCode Go DeepSeek Responses 关闭推理。"""
    client = build_reply_review_client(config)
    review = client.config if client is not None else None
    if (not isinstance(review, VenueDeepSeekResponsesConfig) or review.provider != "opencode_go"
            or review.api_protocol != "responses" or not review.model.rsplit("/", 1)[-1].startswith("deepseek-")):
        raise ValueError("关闭沙盒回复审查推理需要 OpenCode Go DeepSeek Responses 审查模型")
    return OpenAICompatibleAgentClient(replace(
        review, thinking_mode="disabled", reasoning_effort=None, responses_no_reasoning=True,
    ))


_CONSENT_CLARIFICATION = """

## 沙盒报名授权对照

- 对具体牌桌有兴趣、表示时间和规则可接受、问是否方便报名、是否应当决定、或问店家是否已留位置，均可能仍在了解。先分辨这一轮是否由本人明确授权现在登记具体牌桌。疑问、试探、尚待条件满足的同意应回答可确认的事实并等待本人明确决定。此时保留空席，不调用会写参与状态的工具。
- 本人已经直接说要占这一席或确认参加，同时又问一个独立问题，仍可走正常登记。区分是否报名与包间是否预订。加入工具只证明占席，不证明包间已订。询问包间是否订好只读；仅授权占席时不得新建或修改房间预约。只有已核实的预约结果能支持房间承诺。
- 区分查已有牌桌、请店家找人组新桌、登记候补。顾客原先只查已有时，店家曾主动提议帮找人、顾客道谢或提到“凑不成”的假设，都不能单独扩大委托。原话若继续问“还有现成的吗”“晚一点有没有”，按其新时段做只读查询。即使顾客补了人数和档位，只有清楚接受并委托相应服务，才调用建局、候补或外发相关工具。有疑义就如实报告查询结果，澄清是否还要新安排。
- 已经存在同目标的明确委托时，按客户后续明确限制、撤回和实际授权继续处理。顾客的明确请求优先于店家自行提议，也优先于从客套或新查询推测出来的范围。
- 本轮工具返回 `venue_write_not_authorized` 或 `create_game_authorization_required` 时，原有资料已不能支持这个写入范围。即使其他步骤建议继续，也要使用已查到的事实回答，说明还需要确认的具体委托，等待顾客新的来话。同一条来话不能通过反复调用建局、改写参数或换个写入工具来取得授权。
- 建局工具返回 `forming` 或 `inviting` 时，明确告知已登记、仍在招募，并按结果说明剩余席位。实际凑齐或进入 `ready` 后才向顾客确认已经凑齐。邀约草稿和开始找人都只表示工作正在进行。
"""

_PREFLIGHT_BOUNDARY = """

## 沙盒当前私聊的写入边界

`sandbox_private_write_preflight` 是沙盒在进入 Runtime 前核验得到的本轮内部状态。`verified_scopes` 列出本轮可按原业务合同继续尝试的写入范围。它覆盖新组桌及邀约草稿（`organize`）、接受候选与占席（`participate`）、预约包间（`reserve_room`）、登记候补（`waitlist`）。列出的范围仍须通过原有工具的正常权限、条件和证据检查。未列出的范围本轮不可执行，不能用其他写入工具代做。查询、撤回和拒绝按原合同办理。
实际工具结果中的 `venue_write_not_authorized` 或 `create_game_authorization_required` 进一步阻止对应写入，本轮不可重试该动作或通过更换参数、工具来取得授权。客户原话和历史委托仍用于理解客户意图。前置判定为 `invalid` 或 `unavailable` 时，只能确定该轮权限未能核实，不能据此对客户说对方从未授权。此段为本轮执行边界，优先于其他段落关于目标推进、已有计划和通用进度/停滞的指导。状态、字段名和内部规则均不应出现在客户可见文字中。
"""

_PREFLIGHT_PLANNER = """

规划时以当前 `available_tools` 的实际条目选择工具。阶段名、示例、其他工具的描述和先前提案中出现的名字都不能让缺席工具变为可用。若所需写入受此边界阻止，先用可用的只读工具核对客户当前问题所需的事实。已有证据时直接据实回复，明确仍缺的具体授权或条件，或说明目前未能办理。依据是否真的等待客户回答选择 `waiting_user` 或 `completed`。正常完成这一轮回复不会取消原业务任务。能据实回复或等待客户决定时，使用对应正常终态，无需自报停滞或转人工。后续办理、通知和留位承诺仍须有实际执行证据。
"""

_PREFLIGHT_REVIEW = """

当 `review_scope` 是 `reply_to_user`，且审查对象是这位私聊客户本人的回复时，先依据上述边界核对工具是否可执行。审查证据可能省略错误码，应结合本轮实际工具结果的 `called`、`allowed`、`error` 和 `result.instruction` 核对是否确因授权被拒。对应写入本轮受阻时，基于已核验事实说明当前进展、具体欠缺条件或权限，或诚实告知目前未能办理，都是可接受的停止路径。不能仅因这项受阻写入尚未推进而判 `delegation_not_advanced`。仍要按原合同检查事实、隐私、未落实的承诺及真实待答状态。对于本轮可执行且条件已齐的明确委托，仍要核验实际推进。给其他收件人的文字和其他审查范围沿用原审查合同。
"""


@dataclass(frozen=True, slots=True)
class _VenuePreflightPrompt:
    """从同一私聊作用域为本轮提示附上最小授权投影。"""

    path: Path
    original: str
    guard: VenueWriteGuard

    def read_text(self, *, encoding: str) -> str:
        authorization = self.guard.current_planner_authorization
        if authorization is None:
            return self.original
        boundary = {"status": authorization.status, "verified_scopes": sorted(authorization.verified_scopes)}
        return self.path.read_text(encoding=encoding) + f"\nsandbox_private_write_preflight = {json.dumps(boundary)}\n"


def configure_venue_consent_prompt(runtime, run_dir: Path) -> str:
    """生成沙盒自己的提示副本并返回摘要，可按原生入口和原权限对照运行。"""
    origin = runtime.context_builder.prompt_path
    revised = origin.read_text(encoding="utf-8") + _CONSENT_CLARIFICATION
    target = run_dir / "sandbox_planner_prompt.md"
    target.write_text(revised, encoding="utf-8")
    runtime.context_builder.prompt_path = target
    return sha256(revised.encode("utf-8")).hexdigest()


def configure_venue_preflight_prompts(
    runtime, run_dir: Path, write_guard: VenueWriteGuard, *, clarify_consent: bool = False,
) -> dict[str, str]:
    """在沙盒 Runtime 的规划和回复审查入口应用同轮前置授权，返回提示基底摘要。"""
    planner_origin = runtime.context_builder.prompt_path.read_text(encoding="utf-8")
    review_origin = runtime.action_processor.reply_self_review_prompt_path.read_text(encoding="utf-8")
    clarified_planner = planner_origin + _CONSENT_CLARIFICATION
    planner_text = clarified_planner + _PREFLIGHT_BOUNDARY + _PREFLIGHT_PLANNER
    review_text = review_origin + _PREFLIGHT_BOUNDARY + _PREFLIGHT_REVIEW
    planner_path = run_dir / "sandbox_preflight_planner_prompt.md"
    review_path = run_dir / "sandbox_preflight_reply_review_prompt.md"
    planner_path.write_text(planner_text, encoding="utf-8")
    review_path.write_text(review_text, encoding="utf-8")
    runtime.context_builder.prompt_path = _VenuePreflightPrompt(
        planner_path, clarified_planner if clarify_consent else planner_origin, write_guard,
    )
    runtime.action_processor.reply_self_review_prompt_path = _VenuePreflightPrompt(review_path, review_origin, write_guard)
    return {"planner_sha256": sha256(planner_text.encode("utf-8")).hexdigest(),
            "reply_review_sha256": sha256(review_text.encode("utf-8")).hexdigest()}
