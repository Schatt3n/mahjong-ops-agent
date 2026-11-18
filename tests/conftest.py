"""隔离本机微信接入配置，避免测试读取实际运营账号的授权。"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_saved_wechat_setup(monkeypatch, tmp_path):
    """每项测试默认没有已保存账号；需要配置的测试可显式覆盖该路径。"""
    monkeypatch.setenv("MAHJONG_AGENT_WECHAT_CONFIG_PATH", str(tmp_path / "wechat_setup.json"))
