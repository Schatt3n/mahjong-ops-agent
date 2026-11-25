"""在临时配置中核对仿真模型凭证优先级和官方连接边界。"""

import json
import os
from pathlib import Path
import subprocess
import tomllib
from types import SimpleNamespace

import pytest

from mahjong_agent_runtime.llm import OPENCODE_GO_BASE_URL

from .venue_models import (configure_venue_consent_prompt, deepseek_official_config, opencode_auxiliary,
                           opencode_config, venue_planner_config)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """避免配置测试读取真实环境变量，也不允许其中任何命令被执行。"""
    for variable in ("OPENCODE_GO_API_KEY", "OPENCODE_API_KEY", "OPENCODE_GO_MODEL",
                     "OPENCODE_GO_RESPONSE_REASONING", "TEST_VENUE_PROVIDER_KEY",
                     "DEEPSEEK_API_KEY", "MAHJONG_DEEPSEEK_API_KEY"):
        monkeypatch.delenv(variable, raising=False)

    def unexpected_command(*args, **kwargs):
        pytest.fail("fixture credential command must never execute")

    monkeypatch.setattr(os, "system", unexpected_command)
    monkeypatch.setattr(os, "popen", unexpected_command)
    monkeypatch.setattr(subprocess, "run", unexpected_command)
    monkeypatch.setattr(subprocess, "Popen", unexpected_command)


def _profile(config_dir, *, command="cat", args=(), model="gpt-6-luna"):
    fields = [
        f"model = {json.dumps(model)}",
        "[model_providers.opencode-go]",
        'base_url = "https://untrusted.example.invalid/ignore-this"',
        'env_key = "TEST_VENUE_PROVIDER_KEY"',
        'wire_api = "responses"',
        "[model_providers.opencode-go.auth]",
        f"command = {json.dumps(command)}",
        f"args = {json.dumps(list(args))}",
    ]
    (config_dir / "opencode-go.config.toml").write_text("\n".join(fields) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("credentials", "expected"),
    [
        ({"OPENCODE_GO_API_KEY": "dedicated", "OPENCODE_API_KEY": "generic", "TEST_VENUE_PROVIDER_KEY": "named"}, "dedicated"),
        ({"OPENCODE_API_KEY": "generic", "TEST_VENUE_PROVIDER_KEY": "named"}, "generic"),
        ({"TEST_VENUE_PROVIDER_KEY": "named"}, "named"),
        ({}, "fixture-file-key"),
    ],
)
def test_opencode_prefers_environment_and_uses_only_official_endpoint(tmp_path, monkeypatch, credentials, expected):
    key_path = tmp_path / "test-token"
    key_path.write_text("  fixture-file-key\n", encoding="utf-8")
    _profile(tmp_path, command="/usr/bin/cat", args=(str(key_path),))
    for variable, value in credentials.items():
        monkeypatch.setenv(variable, value)
    monkeypatch.setenv("OPENCODE_GO_MODEL", "model-from-environment")

    config = opencode_config(tmp_path, max_concurrency=4)
    assert config.api_key == expected
    assert config.model == "model-from-environment"
    assert config.provider == "opencode_go"
    assert config.base_url == OPENCODE_GO_BASE_URL
    assert config.api_protocol == "responses"
    assert config.max_concurrency == 4


def test_profile_overrides_general_model_and_auxiliary_has_independent_limit(tmp_path):
    (tmp_path / "config.toml").write_text(
        'model = "general-model"\nmodel_provider = "opencode_go"\n', encoding="utf-8",
    )
    key_path = tmp_path / "test-token"
    key_path.write_text("fixture-key", encoding="utf-8")
    _profile(tmp_path, args=(str(key_path),))

    config = opencode_config(tmp_path, max_concurrency=5)
    auxiliary = opencode_auxiliary(config, max_tokens=900, workers=2)
    assert config.model == "gpt-6-luna"
    assert auxiliary.config.model == config.model
    assert auxiliary.config.api_key == config.api_key
    assert auxiliary.config.base_url == config.base_url == OPENCODE_GO_BASE_URL
    assert (auxiliary.config.max_tokens, auxiliary.config.max_concurrency) == (900, 2)
    assert (config.max_tokens, config.max_concurrency) == (7200, 5)


def test_deepseek_official_uses_the_dedicated_key_on_the_same_responses_protocol(monkeypatch):
    """现有店家代理地址和通用密钥不能改变官方对照的请求路径。"""
    monkeypatch.setenv("MAHJONG_LLM_BASE_URL", "https://unrelated.example.invalid")
    monkeypatch.setenv("MAHJONG_LLM_API_KEY", "unrelated-key")
    with pytest.raises(ValueError, match="DeepSeek 官方仿真需要"):
        deepseek_official_config(max_concurrency=4)

    monkeypatch.setenv("MAHJONG_DEEPSEEK_API_KEY", "project-deepseek-fixture")
    config = deepseek_official_config(max_concurrency=4)
    assert config.api_key == "project-deepseek-fixture"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "official-fixture")
    config = deepseek_official_config(max_concurrency=4)
    assert (config.provider, config.model, config.api_key) == ("deepseek", "deepseek-flash", "official-fixture")
    endpoint, request = config._request_payload([{"role": "user", "content": "本地合同检查"}])
    assert endpoint == "https://api.deepseek.com/responses"
    assert request["reasoning"] == {"effort": "none"} and request["store"] is False
    assert config.max_concurrency == 4


def test_general_config_can_supply_its_own_provider_alias(tmp_path, monkeypatch):
    (tmp_path / "config.toml").write_text(
        'model_provider = "opencode_go"\nmodel = "general-go-model"\n'
        '[model_providers.opencode_go]\nenv_key = "TEST_VENUE_PROVIDER_KEY"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_VENUE_PROVIDER_KEY", "fixture-key")

    config = opencode_config(tmp_path, max_concurrency=1)
    assert config.model == "general-go-model"
    assert config.api_key == "fixture-key"
    assert config.base_url == OPENCODE_GO_BASE_URL


@pytest.mark.parametrize(("setting", "effort"), [(None, "none"), ("provider-default", None), ("low", "low")])
def test_deepseek_experiment_sends_the_actual_responses_effort(tmp_path, monkeypatch, setting, effort):
    """验证真实请求和诊断均按实验配置发送，不读取其它模型的推理缺省。"""
    _profile(tmp_path, model="deepseek-v4.1-flash")
    monkeypatch.setenv("TEST_VENUE_PROVIDER_KEY", "fixture-key")
    if setting is not None:
        monkeypatch.setenv("OPENCODE_GO_RESPONSE_REASONING", setting)
    config = opencode_config(tmp_path, max_concurrency=1)
    auxiliary = opencode_auxiliary(config, max_tokens=600, workers=1)
    assert config.request_diagnostics()["reasoning_effort"] == effort
    assert auxiliary.config.request_diagnostics()["reasoning_effort"] == effort
    monkeypatch.setenv("OPENCODE_GO_MODEL", "unrelated-model")
    different = opencode_config(tmp_path, max_concurrency=1)
    assert different.request_diagnostics()["reasoning_effort"] != "none"


def test_launcher_keeps_planner_consent_changes_inside_the_sandbox(tmp_path, monkeypatch):
    """验证规划推理配置与提示文件的独立副本。"""
    _profile(tmp_path, model="deepseek-v4.1-flash")
    monkeypatch.setenv("TEST_VENUE_PROVIDER_KEY", "fixture-key")
    base = opencode_config(tmp_path, max_concurrency=1)
    planner = venue_planner_config(base, reasoning_low=True)
    assert planner.request_diagnostics()["reasoning_effort"] == "low"
    assert base.request_diagnostics()["reasoning_effort"] == "none"

    original = tmp_path / "original.md"
    original.write_text("原生规划提示", encoding="utf-8")
    runtime = SimpleNamespace(context_builder=SimpleNamespace(prompt_path=original))
    digest = configure_venue_consent_prompt(runtime, tmp_path)
    projected = runtime.context_builder.prompt_path.read_text(encoding="utf-8")
    assert len(digest) == 64 and original.read_text(encoding="utf-8") == "原生规划提示"
    assert projected.startswith("原生规划提示") and "明确授权" in projected


@pytest.mark.parametrize("command, args", [
    ("/bin/sh", ("{file}",)),
    ("cat", ("{file}", "ignored")),
    ("cat", ("relative-key",)),
    ("/bin/cat", ("{missing}",)),
])
def test_configured_auth_cannot_execute_or_read_other_than_one_absolute_file(tmp_path, command, args):
    key_path = tmp_path / "test-token"
    key_path.write_text("fixture-key", encoding="utf-8")
    refs = {"{file}": str(key_path), "{missing}": str(tmp_path / "missing")}
    _profile(tmp_path, command=command, args=tuple(refs.get(value, value) for value in args))

    with pytest.raises(ValueError, match="OpenCode Go"):
        opencode_config(tmp_path, max_concurrency=1)


def test_missing_or_invalid_opencode_config_reports_a_parseable_error(tmp_path: Path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="OpenCode Go"):
        opencode_config(tmp_path, max_concurrency=1)
    profile = tmp_path / "opencode-go.config.toml"
    profile.write_text("[model_providers.\n", encoding="utf-8")
    with pytest.raises(tomllib.TOMLDecodeError):
        opencode_config(tmp_path, max_concurrency=1)
    profile.unlink()
    (tmp_path / "config.toml").write_text('model_provider = "different-provider"\nmodel = "other-model"\n',
                                         encoding="utf-8")
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "fixture-key")
    with pytest.raises(ValueError, match="OpenCode Go"):
        opencode_config(tmp_path, max_concurrency=1)
