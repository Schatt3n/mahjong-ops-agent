"""为仿真解析本机 OpenCode Go 模型与受限凭证来源。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import os
from pathlib import Path
import tomllib

from mahjong_agent_runtime.llm import AgentLLMConfig, OpenAICompatibleAgentClient, OPENCODE_GO_BASE_URL


@dataclass(slots=True)
class VenueOpenCodeConfig(AgentLLMConfig):
    """为沙盒明确传递 Responses 的 DeepSeek 推理选项。"""

    responses_no_reasoning: bool = False

    def _request_payload(self, messages):
        """复用 Runtime 的消息和协议封装，只补仿真实验明确指定的无推理选项。"""
        endpoint, payload = AgentLLMConfig._request_payload(self, messages)
        if (self.provider == "opencode_go" and self.api_protocol == "responses"
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
    return VenueOpenCodeConfig(
        api_key=api_key, model=str(model), provider="opencode_go", base_url=OPENCODE_GO_BASE_URL,
        api_protocol=str(provider.get("wire_api") or "responses"),
        max_tokens=max(1, int(os.environ.get("OPENCODE_GO_MAX_OUTPUT_TOKENS", "7200"))),
        max_concurrency=max_concurrency, retry_attempts=2,
        thinking_mode="enabled" if explicit_effort else "disabled",
        reasoning_effort=explicit_effort, responses_no_reasoning=reasoning == "none",
    )


def opencode_auxiliary(config: AgentLLMConfig, *, max_tokens: int, workers: int) -> OpenAICompatibleAgentClient:
    """同一官方连接可复用认证；角色、审查与主 Agent 各自持有请求上下文和限额。"""
    return OpenAICompatibleAgentClient(replace(config, max_tokens=max_tokens, max_concurrency=workers))


def venue_planner_config(config: AgentLLMConfig, *, reasoning_low: bool) -> AgentLLMConfig:
    """仅对仿真的规划客户端做低推理对照，角色与独立判定沿用原连接。"""
    return replace(config, thinking_mode="enabled", reasoning_effort="low") if reasoning_low else config


_CONSENT_CLARIFICATION = """

## 沙盒报名授权对照

- 对具体牌桌有兴趣、表示时间和规则可接受、问是否方便报名、是否应当决定、或问店家是否已留位置，均可能仍在了解。先分辨这一轮是否由本人明确授权现在登记具体牌桌。疑问、试探、尚待条件满足的同意应回答可确认的事实并等待本人明确决定。此时保留空席，不调用会写参与状态的工具。
- 本人已经直接说要占这一席或确认参加，同时又问一个独立问题，仍可走正常登记。区分是否报名与包间是否预订。加入工具只证明占席，不证明包间已订。询问包间是否订好只读；仅授权占席时不得新建或修改房间预约。只有已核实的预约结果能支持房间承诺。
- 区分查已有牌桌、请店家找人组新桌、登记候补。顾客原先只查已有时，店家曾主动提议帮找人、顾客道谢或提到“凑不成”的假设，都不能单独扩大委托。原话若继续问“还有现成的吗”“晚一点有没有”，按其新时段做只读查询。即使顾客补了人数和档位，只有清楚接受并委托相应服务，才调用建局、候补或外发相关工具。有疑义就如实报告查询结果，澄清是否还要新安排。
- 已经存在同目标的明确委托时，按客户后续明确限制、撤回和实际授权继续处理。顾客的明确请求优先于店家自行提议，也优先于从客套或新查询推测出来的范围。
"""


def configure_venue_consent_prompt(runtime, run_dir: Path) -> str:
    """生成沙盒自己的提示副本并返回摘要，可按原生入口和原权限对照运行。"""
    origin = runtime.context_builder.prompt_path
    revised = origin.read_text(encoding="utf-8") + _CONSENT_CLARIFICATION
    target = run_dir / "sandbox_planner_prompt.md"
    target.write_text(revised, encoding="utf-8")
    runtime.context_builder.prompt_path = target
    return sha256(revised.encode("utf-8")).hexdigest()
