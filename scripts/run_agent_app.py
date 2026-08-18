#!/usr/bin/env python3
"""正式启动入口：组装依赖后运行麻将馆运营进程。

导航：从 :func:`main` 开始读；应用组装与生命周期在
``scripts.agent_runtime_app.MahjongAgentApplication``，业务主循环在
``mahjong_agent_runtime.runtime.AgentRuntime``。
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agent_runtime_app import MahjongAgentApplication  # noqa: E402


def main() -> None:
    """组装并运行应用；这里不解析消息，也不承载麻将业务规则。

    上游：命令行执行本文件时，底部的 ``__main__`` 分支调用这里。
    下游：``from_environment`` 是组合根，``run`` 启动调度器和 HTTP 服务并阻塞。
    """
    MahjongAgentApplication.from_environment().run()


if __name__ == "__main__":
    main()
