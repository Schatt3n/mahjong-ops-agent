#!/bin/bash
# 兼容从 Finder、Codex 或终端启动，统一使用仓库内的启动入口。
set -eu
cd -- "$(dirname -- "$0")"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

if [[ "${1:-}" == --stop && ! -e data/wechat-stack.lock ]]; then
    printf '%s\n' '当前没有由一键入口管理的服务。'
    exit 0
fi

python_bin="${MAHJONG_WECHAT_PYTHON_BIN:-}"
if [[ -n "$python_bin" ]]; then
    if ! "$python_bin" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then
        printf '%s\n' 'MAHJONG_WECHAT_PYTHON_BIN 必须指向 Python 3.11 或更高版本。' >&2
        exit 1
    fi
else
    for candidate in .venv/bin/python python3 python3.11; do
        if "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
            python_bin="$candidate"
            break
        fi
    done
    if [[ -z "$python_bin" ]]; then
        python_bin="$(bash scripts/bootstrap_wechat_tool.sh python)"
    fi
fi
exec "$python_bin" scripts/start_wechat_stack.py "$@"
