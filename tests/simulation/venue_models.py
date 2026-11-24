"""为仿真解析本机 OpenCode Go 模型与受限凭证来源。"""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import tomllib

from mahjong_agent_runtime.llm import AgentLLMConfig, OpenAICompatibleAgentClient, OPENCODE_GO_BASE_URL


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
    return AgentLLMConfig(
        api_key=api_key, model=str(model), provider="opencode_go", base_url=OPENCODE_GO_BASE_URL,
        api_protocol=str(provider.get("wire_api") or "responses"),
        max_tokens=4000, max_concurrency=max_concurrency, retry_attempts=2,
    )


def opencode_auxiliary(config: AgentLLMConfig, *, max_tokens: int, workers: int) -> OpenAICompatibleAgentClient:
    """同一官方连接可复用认证；角色、审查与主 Agent 各自持有请求上下文和限额。"""
    return OpenAICompatibleAgentClient(replace(config, max_tokens=max_tokens, max_concurrency=workers))
