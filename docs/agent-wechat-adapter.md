# agent-wechat 通道架构与接入

本文说明麻将馆运营 Agent 如何接入 [agent-wechat](https://github.com/thisnick/agent-wechat)。
结论是：**上游独立部署，本仓库不 fork；业务仓库只保留薄适配器和冻结的只读参考。**

## 职责边界

```text
┌────────────────────────────────────────────────────────────┐
│ Python Runtime（本仓库）                                    │
│ 用户画像 / 短期与长期记忆 / 业务状态 / 权限 / 回复审查      │
└───────────────▲───────────────────────────┬────────────────┘
                │ POST raw 入站              │ POST /send 出站
┌───────────────┴───────────────────────────▼────────────────┐
│ Node 适配器（integrations/agent-wechat-adapter）             │
│ SSE 跟随 / 消息规范化 / 游标 / 重放队列 / 发送幂等 / 消歧    │
└───────────────▲───────────────────────────┬────────────────┘
                │ HTTP + SSE                 │ 受控发送
┌───────────────┴───────────────────────────▼────────────────┐
│ agent-wechat daemon（上游，独立安装和升级）                  │
│ 微信登录 / 本地消息库 / 联系人同步 / 发送队列                │
└───────────────────────────────────────────────────────────┘
```

每层只通过公开接口通信：

- Python Runtime 不导入 agent-wechat 的 Rust crate，不读取它的 SQLite、锁文件或缓存。
- Node 适配器不修改上游源码，不解析内部数据库，只使用上游 HTTP/SSE API。
- 上游使用独立版本、独立进程和独立账号目录；升级不需要把整棵源码合并回业务仓库。

`integrations/agent-wechat/` 是冻结的 v2.3.2 源码副本，仅作历史只读参考，不参与构建，
也不是当前运行链路的一部分。适配器和 `contract-lock.json` 对齐并已校验的上游发布版本是
v2.5.0（commit `ce5c068`）。

## 端口与数据流

| 端口 | 服务 | 用途 |
| --- | --- | --- |
| `6174` | 上游 agent-wechat daemon | 账号、消息、联系人、群、事件流、发送 |
| `8790` | Python Runtime | `/api/channels/agent-wechat/raw` 与业务 API |
| `8792` | Node 适配器控制面 | `/send`、`/send-channel`、`/auto-send`、`/resolve`、`/health` |

### 入站

```text
上游 SSE message.committed
  -> 适配器 GET /v1/accounts/{a}/messages/{id}
  -> 区分老板跨设备回复、适配器外发回声和客户消息
  -> 映射为 Wechaty 兼容的 raw 载荷
  -> POST http://127.0.0.1:8790/api/channels/agent-wechat/raw
  -> SQLite 持久接收，HTTP 202，适配器推进游标
  -> 后台写入私聊老板上下文或处理客户消息
```

适配器收到持久接收确认就推进游标。入站路由和到期输入批次各自最多并行处理四个会话，同一队列中的同一会话按顺序消费。
不同会话可以同时生成回复或同时审查。主模型和回复审查各用自己的模型请求并发上限。
原始消息也写到 `logs/agent_wechat_weixin_raw.jsonl`。消息详情拉取或持久接收失败时，适配器保留游标，重连后按原顺序补读。
客户之后接着问时，即使上一次业务任务已经归档，Agent 仍能参考同一私聊里近期的老板回复及相关对话。当前局况和实际执行仍由业务工具确认。

### 出站

```text
业务状态 -> 回复生成 -> 末端内容审查 -> Runtime 外发客户端
  -> POST http://127.0.0.1:8792/send
  -> 适配器解析目标（wxid / 群 id 或唯一完全匹配的备注）
  -> 自动回复复查老板是否已有回复
  -> 上游 POST /v1/accounts/{a}/sends + Idempotency-Key
  -> 轮询发送状态
```

适配器复用现有 wechaty 桥接的 `/health`、`/send`、`/send-channel`、`/auto-send` 合同，
因此 Runtime 业务代码不需要知道上游协议。只需要把
`MAHJONG_WECHATY_OUTBOUND_BASE_URL` 指向 `http://127.0.0.1:8792`。
入站触发的自动回复会先保存审查结果，再由独立的串行外发线程调用仅限本机且需令牌的 `/deliver-ingress-reply`。
外发等待期间，入站处理线程仍可处理其他会话。同一会话等待该次外发得到结果后继续消费。
交接失败时同一来源会在后台重试，发送仍使用同一个幂等键。
私聊中，老板的回复同步到 daemon 后，会挡住该轮尚未发送的自动回复。群聊按对原消息的精确引用判断。

## 会话命名空间与 wxid

agent-wechat 和 wechaty 操作的是同一个工作微信号、同一批 `wxid` 和群 id。为了让既有
看板、群聚合和私聊历史在切换通道时不中断，适配器把会话 id 继续写成：

```text
私聊：wechaty:contact:<wxid>
群聊：wechaty:room:<room-id>@chatroom
```

但原始证据、通道观察、身份记录和外发草稿仍保留真实通道名 `agent-wechat`，不会把两个
通道的事实混在一起。

`threadId` / `senderId` 是通道寻址键，不是业务客户主键。业务侧应维护自己的
`customer_id`，通过映射表绑定 `(channel, external_id)`；备注、昵称和群名只作展示。
微信身份可能变化，当前适配器**不做自动身份迁移**。如果 `wxid` 变化，只能用手机号、
备注、历史聊天、业务凭证等证据人工重新确认，然后更新映射，不能按名字自动合并。

## 找人、重名与群聊

- 先按精确 id 匹配，再按唯一的**完全同名**匹配；备注、昵称、显示名和微信号都可参与
  展示与检索。
- 重名、仅前缀命中、包含匹配或远程搜索结果无法唯一确定时，返回 HTTP `409` 和候选，
  绝不自动挑一个收件人。
- 私聊使用对方 `wxid`；群聊必须使用以 `@chatroom` 结尾的群 id。
- 群聊不是共享私聊上下文。群消息进入群会话链，不能把群成员的个人画像或私聊记忆直接
  拼进群回复。
- 上游联系人检索没有拼音和纠错；中文同音字、输入错误仍需要上层确认。

## 安全闸门

外发默认全关，必须同时打开两个开关才会真正发送：

```bash
MAHJONG_AGENT_WECHAT_SEND_ENABLED=true
MAHJONG_AGENT_WECHAT_AUTO_SEND_REPLY=true
```

其他约束：

- 客户可见文本必须经过 Runtime 末端过滤，适配器不承担业务审查。
- 同一源消息的自动回复会用稳定幂等键去重；只有上游 `accepted` 才记入投递账本。
- `needs_review` 表示消息是否到达未知，只告警并要求人工核对聊天窗口，禁止自动重发。
- 控制面只监听回环地址；配置 `MAHJONG_AGENT_WECHAT_CONTROL_TOKEN` 后，除 `/health`
  外都需要同一个令牌。
- 个人微信接入是非官方灰度通道，存在账号风控；必须使用独立工作号、小范围白名单和人工
  兜底，不承诺稳定 SLA。

## 启动顺序

1. 安装并启动上游，完成扫码登录：

```bash
npm install -g @agent-wechat/cli
wx serve                 # 默认 127.0.0.1:6174
# 首次使用：新建独立 Profile（全平台）
wx accounts create --alias mahjong-work
wx auth login
wx accounts list
```

`wx auth login --official` 仅在 Linux 接管已有官方微信数据时使用；必须先退出官方微信、
备份 Profile，并在 daemon 运行期间保持官方应用关闭。macOS 和 Windows 只能新建独立
Profile，且新 Profile 不会获得完整历史聊天记录。

2. 启动 Python Runtime，确认 `GET /api/health` 正常，并设置入站与出站配置。

```bash
export MAHJONG_AGENT_WECHAT_ROUTE_SCOPE=incoming_only
export MAHJONG_WECHATY_OUTBOUND_BASE_URL=http://127.0.0.1:8792
```

`MAHJONG_WECHATY_ROUTE_SCOPE` 与 `MAHJONG_AGENT_WECHAT_ROUTE_SCOPE` 缺省值彼此独立：
前者保持 `self_only`，后者为 `incoming_only`。客户消息按各通道的入口范围处理，老板回复走独立的会话记录入口。

3. 启动适配器；先使用 `--dry-run` 验证能读到事件，再打开受控投递：

```bash
cd integrations/agent-wechat-adapter
node src/main.mjs --dry-run
node src/main.mjs
```

4. 先保持发送开关关闭，验收私聊和群聊入站、联系人消歧和观察记录；确认后再逐账号打开
   外发。完整环境变量见
   [适配器 README](../integrations/agent-wechat-adapter/README.md)。

## 当前能力

| 能力 | 状态 |
| --- | --- |
| 私聊 / 群聊文本入站 | 支持 |
| 私聊 / 群聊文本外发 | 支持，受双开关和幂等控制 |
| 联系人 / 群检索与消歧 | 支持，重名返回候选 |
| 老板跨设备回复 | 私聊记录为会话上下文；群聊保留原始观察，按引用阻止重复回复 |
| 适配器外发回声、系统消息 | 默认过滤，可为排查打开转发 |
| 图片、语音、视频、文件、引用 | 仅透传消息 id、类型、可读正文和元数据；未下载媒体、未做 OCR/转写 |
| 朋友圈 | 上游当前没有路由，不支持 |
| 添加好友 | 上游当前没有路由，不支持 |
| 修改备注 | 上游当前没有写路由，不支持 |
| 群管理 | 上游有部分路由，但适配器未开放 |

朋友圈、加好友和修改备注都不是当前 MVP 的必需项。它们属于低频操作，后续可以用独立的
Linux UI 自动化通道补齐，不应混入本适配器的可靠发送链路。

## 上游升级

跟随 GitHub Release / `v*` tag，不跟随 `main`。升级前先跑合同检查：

```bash
cd integrations/agent-wechat-adapter
npm run contract:check  # 默认校验固定发布的 v2.5.0 OpenAPI
node scripts/check-upstream.mjs --file /path/to/new/openapi.json
```

准备升级到新 tag 时，先对新版 OpenAPI 跑检查，再显式更新锁和版本元数据：

```bash
node scripts/write-contract-lock.mjs --url <new-openapi-url> \
  --version vX.Y.Z --commit <commit-sha>
```

锁文件覆盖适配器依赖的 11 条路由、6 个 schema 的必填字段/字段存在性、发送幂等头和
发送状态集合。新增路由不阻断，删除或修改依赖项会非零退出。合同无漂移后再做真实账号
灰度，不能仅凭单元测试升级生产。

## 故障处置

| 现象 | 处置 |
| --- | --- |
| Runtime 不可达 | 适配器保留最后确认的游标，恢复后重连补读。旧重放队列可通过 `POST /replay-spool` 导入 |
| SSE 断线 | 适配器指数退避重连，最长 30 秒；从最后游标续传 |
| 游标过期 | 定位上游最早仍保留的事件，记录已清理的区间，并用 JSON 分页补齐保留窗口。首次启动且历史重放关闭时仍从日志末尾开始 |
| 发送 `needs_review` | 人工查看手机/微信窗口，确认是否到达；不得自动重试 |
| 目标重名或找不到 | 使用返回的候选和精确 id 重新确认，不要放宽成模糊匹配 |
| 账号风控或掉线 | 关闭发送开关，人工接管；恢复后重新验证入站再开外发 |

## 验证状态

当前已通过 Python 通道回归、Node 适配器单元测试和 OpenAPI 合同检查，但**尚未完成真实
工作号的扫码、私聊/群聊收发和长期在线验收**。在真实灰度完成前，只能把本通道视为开发
联调可用，不能对外承诺生产稳定性。
