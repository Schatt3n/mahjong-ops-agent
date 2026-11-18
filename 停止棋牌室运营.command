#!/bin/bash
# Finder 停止入口，与 Codex 停止操作使用同一套退出逻辑。
exec /bin/bash "$(dirname -- "$0")/启动棋牌室运营.command" --stop
