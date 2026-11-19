# agent-wechat 上游部署与适配器开发

[agent-wechat](https://github.com/thisnick/agent-wechat) 按发布版本独立安装、运行和升级。
麻将馆 Agent 通过本仓库的 [Node 适配器](../integrations/agent-wechat-adapter/README.md)
调用上游公开的本地 HTTP/SSE API。完整架构、端口、数据流和能力边界见
[agent-wechat 通道架构](agent-wechat-adapter.md)。

## 上游源码和接口校验

从 [上游 Releases](https://github.com/thisnick/agent-wechat/releases) 获取安装包，按对应 tag
查看源码。适配器的 `contract-lock.json` 基于 v2.5.0（commit `ce5c068`），
`npm run contract:check` 默认读取该 tag 的 OpenAPI。升级前可传入目标版本的 OpenAPI
执行检查。早期导入的 v2.3.2 快照可在本仓库历史提交 `d8bc379` 中查阅。

## 上游部署

```bash
npm install -g @agent-wechat/cli
wx serve                  # 默认 127.0.0.1:6174
# 首次使用：新建独立 Profile（全平台）
wx accounts create --alias mahjong-work
wx auth login
wx accounts list
```

`--official` 只支持 Linux 接管已有官方微信 Profile，使用前必须退出官方应用并备份。
macOS/Windows 不支持该模式，不能把官方客户端的数据目录直接交给 daemon。

API token 默认位于 `~/.local/share/agent-wechat/api-token`。上游除 `/health` 外的路由都需要
`Authorization: Bearer <token>`，daemon 只应监听回环地址。

上游自己的微信账号资料、消息库、钥匙串和日志留在它的运行目录，不进入本仓库。不要把真实
账号、消息正文、二维码、token 或数据库写入测试夹具和 Git。

Runtime 侧接入时显式设置通道专属路由：

```bash
export MAHJONG_AGENT_WECHAT_ROUTE_SCOPE=incoming_only
export MAHJONG_WECHATY_OUTBOUND_BASE_URL=http://127.0.0.1:8792
```

`MAHJONG_WECHATY_ROUTE_SCOPE` 缺省仍为 `self_only`；agent-wechat 缺省为
`incoming_only`，因为适配器默认过滤自消息。两者互不继承，避免切换通道时误改历史行为。

## 适配器开发

适配器只依赖 Node.js 20+，不需要 Rust、pnpm、Xcode 或上游源码构建：

```bash
cd integrations/agent-wechat-adapter
npm test
npm run contract:check
node src/main.mjs --dry-run
```

需要修改接入行为时，只改本目录。应保持以下边界：

- 只使用上游公开 HTTP/SSE API，不读取 daemon 的数据库、锁或缓存；
- 业务画像、记忆、会话和权限留在 Python Runtime；
- 外发默认关闭，`needs_review` 不自动重试；
- 目标无法唯一确认时返回候选，不按前缀或模糊结果自动发送；
- 会话命名空间沿用 `wechaty:contact:` / `wechaty:room:`，但观察记录保留真实通道名；
- 通道路由默认值按通道独立，不能让 agent-wechat 继承 wechaty 的 `self_only`；
- 新依赖的上游路由或 schema 必须同步更新合同锁和测试。

## Python Runtime 回归

仓库根目录执行：

```bash
python3 -m pytest -q \
  tests/test_channel_observations.py tests/test_agent_app.py \
  tests/test_wechaty_input_gate.py tests/test_api_reply_delivery.py
```

真实微信验收不属于单元测试：需要独立工作号、扫码、私聊和群聊双向收发、重启续传、断网
重放、发送回执和账号风控观察。未完成这些验收前，不应把通道标记为生产可用。

历史实验记录见 [AgentWechat 实验日记](agent_wechat_experiment_log.md)。该文档记录的是早期
内嵌 Fork 阶段的尝试，结论仍可作为风险参考，但构建方式已不代表当前结构。
