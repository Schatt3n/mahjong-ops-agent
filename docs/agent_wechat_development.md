# AgentWechat Fork 开发

客户端和本地 Hub 源码位于 [integrations/agent-wechat](../integrations/agent-wechat)。从本仓库克隆后即可编辑，修改跟随麻将馆 Agent 仓库提交。导入的上游基线为 [cdb6635](https://github.com/thisnick/agent-wechat/commit/cdb6635ab698743ee92b9efe62bd54bce9bfe636)，其上包含本地实验期间的改动。原项目与随附组件的许可证保留在各自目录。进展和后续验收项见 [实验日记](agent_wechat_experiment_log.md)。

| 改动 | 源码目录 |
| --- | --- |
| 桌面聊天界面 | `packages/client-ui` |
| macOS 应用壳、启动和本地 Hub 管理 | `packages/desktop` |
| 接收、发件队列、联系人和本地数据库 | `packages/native-client-rust/crates` |
| Hub HTTP 客户端和消息接口类型 | `packages/shared` |

以下命令在 AgentWechat 子目录执行。使用 Node 24.22.1 或更新版本、pnpm 9、仓库指定的 Rust 1.93.1 和 CMake。模拟器的媒体验收需要 PATH 中的 `ffmpeg` 和 `ffprobe`。构建桌面端还需要 Apple Silicon Mac 上的 Xcode 命令行工具。

```bash
cd integrations/agent-wechat
pnpm install --frozen-lockfile
pnpm --filter @agent-wechat/shared build
pnpm --filter @agent-wechat/client-ui test
cargo build --locked --profile test --manifest-path packages/native-client-rust/Cargo.toml \
  -p wx-native -p wechat-simulator --bin wx --bin wx-dev
node scripts/native-simulator-smoke.mjs
```

构建与现有 Fork 使用同一应用标识的 Apple Silicon 桌面包：

```bash
pnpm --filter @agent-wechat/desktop prepare:sidecars \
  --target aarch64-apple-darwin --wx "$PWD/packages/native-client-rust/target/debug/wx"
pnpm --filter @agent-wechat/desktop app:build --debug --target aarch64-apple-darwin \
  --bundles app --no-sign --config "$PWD/packages/desktop/src-tauri/tauri.fork.json"
```

产物在 `packages/desktop/src-tauri/target/aarch64-apple-darwin/debug/bundle/macos/`。使用现有应用标识时，替换前先退出正在运行的 Fork。临时签名变化可能触发 macOS 钥匙串重新授权。本地调试包使用 `--debug` 构建，其后台自动更新检查不会运行。上游发布工作流保留在客户端子目录供参考，GitHub Actions 只会发现仓库根目录的工作流。

微信账号资料、聊天记录、钥匙串、访问令牌、数据库和原始日志保存在本机运行目录。麻将馆 Agent 接入时通过 Hub 的本地 API，个人微信通道仍是灰度实验。客户端源码并入仓库无需启用运营 Agent 的自动外发。
