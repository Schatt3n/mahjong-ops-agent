"""从本地环境文件加载默认配置，且不覆盖进程中已显式设置的变量。"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv_defaults(path: Path | str) -> dict[str, str]:
    """加载dotenvdefaults，并恢复为领域层可使用的结构。

"""
    env_path = Path(path)
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        normalized = value.strip().strip("'").strip('"')
        os.environ[key] = normalized
        loaded[key] = normalized
    return loaded
