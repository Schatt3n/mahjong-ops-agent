"""从本地环境文件加载默认配置，且不覆盖进程中已显式设置的变量。"""

from __future__ import annotations

import os
from pathlib import Path


def read_dotenv_values(path: Path | str) -> dict[str, str]:
    """读取本机配置而不修改进程环境，用于整组校验凭证和接口地址。"""
    env_path = Path(path)
    values: dict[str, str] = {}
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return values
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values.setdefault(key, value.strip().strip("'").strip('"'))
    return values


def load_dotenv_defaults(path: Path | str) -> dict[str, str]:
    """只补齐进程中未设置的变量，并返回此次实际加载的配置。"""
    loaded = {key: value for key, value in read_dotenv_values(path).items() if key not in os.environ}
    os.environ.update(loaded)
    return loaded
