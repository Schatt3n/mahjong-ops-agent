# agent-wechat 通道适配器

把 [agent-wechat](https://github.com/thisnick/agent-wechat) 的本地 HTTP/SSE API 翻译成
Mahjong Agent Runtime 的原始通道合同。上游按发布版本独立部署、独立升级；本目录不
修改、不 fork 上游源码，也不读取上游的数据库、锁文件或内部缓存。

当前适配器对齐并已校验的上游版本是 `v2.5.0`（commit `ce5c068`）。仓库内
`integrations/agent-wechat/` 的 `v2.3.2` 副本只作历史只读参考，不参与构建和合同检查。

> 个人微信接入属于非官方、灰度通道。请使用单独的工作号，先小范围验证，并保留店员
> 通过官方微信人工兜底的能力。本项目不承诺规避平台风控。

## 边界

| 能力 | 状态 |
| --- | --- |
| 私聊 / 群聊文本入站 | 支持，转为 `wechaty:contact:` / `wechaty:room:` 会话命名空间 |
| 图片、语音、视频、文件、引用等入站 | 透传消息 id、类型、可读正文和元数据；未下载媒体，未做 OCR / 转写 |
| 联系人 / 群检索与消歧 | 支持，重名或仅前缀命中必须上层确认 |
| 文本外发 | 支持，带幂等键；`needs_review` 只告警，禁止自动重试 |
| 朋友圈、加好友 | **上游当前 API 没有这些路由，本适配器不支持** |
| 修改备注、群管理 | 上游当前没有备注写路由；群管理仅有部分路由且本适配器未开放 |

本适配器只做协议翻译、可靠投递和本地闸门。用户画像、短期/长期记忆、会话状态、业务
权限和客户可见文本审查全部留在 Python Runtime。

## 部署

### 1. 启动上游 daemon

```bash
npm install -g @agent-wechat/cli
wx serve             # 默认 127.0.0.1:6174
# 首次使用：新建独立 Profile（Linux / macOS / Windows 均可）
wx accounts create --alias mahjong-work
wx auth login
wx accounts list
```

`wx auth login --official` 只适用于 Linux 上接管已有官方微信数据；使用前必须退出官方
微信并备份 Profile，daemon 运行期间保持官方应用关闭。macOS/Windows 不支持接管官方
客户端数据，只能新建独立 Profile，新 Profile 不会带回完整历史聊天记录。

API token 默认在 `~/.local/share/agent-wechat/api-token`。上游 daemon 只监听回环地址，
所有路由除 `/health` 外都需要 `Authorization: Bearer <token>`。

### 2. 启动适配器

```bash
cd integrations/agent-wechat-adapter
MAHJONG_AGENT_WECHAT_RAW_ENDPOINT=http://127.0.0.1:8790/api/channels/agent-wechat/raw \
node src/main.mjs
```

启动时会确认只有一个可用账号；多账号必须显式设置 `MAHJONG_AGENT_WECHAT_ACCOUNT`。
默认从事件日志尾部开始，不重放历史消息。

只读联调：

```bash
node src/main.mjs --dry-run
```

`--dry-run` 只打印准备投递的消息摘要，不写 Runtime、不发送任何消息。

### 3. 接回 Python Runtime

入站由适配器主动 POST 到 Runtime；出站复用 Runtime 现有的 wechaty 外发客户端，只需
把地址指向适配器控制面：

```bash
# Runtime 侧
export MAHJONG_WECHATY_OUTBOUND_BASE_URL=http://127.0.0.1:8792
# agent-wechat 适配器默认只投递客户消息，Runtime 侧对应使用 incoming_only
export MAHJONG_AGENT_WECHAT_ROUTE_SCOPE=incoming_only
# 若设置了 MAHJONG_AGENT_WECHAT_CONTROL_TOKEN，则两处必须一致
export MAHJONG_WECHATY_OUTBOUND_TOKEN=...
```

两个桥接通道的路由默认值彼此独立：`MAHJONG_WECHATY_ROUTE_SCOPE` 缺省为
`self_only`，`MAHJONG_AGENT_WECHAT_ROUTE_SCOPE` 缺省为 `incoming_only`。这样
agent-wechat 灰度时客户消息不会被 `self_only` 拦截，也不会改变 wechaty 既有配置。
允许值为 `self_only`、`incoming_only`、`all`。

适配器控制面实现与现有 wechaty 桥接相同的 `/health`、`/send`、`/send-channel`、
`/auto-send` 合同，因此不需要改 Runtime 的业务代码。外发默认全关；必须同时打开
发送总闸和自动外发开关，手动发送才会生效。

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MAHJONG_AGENT_WECHAT_BASE_URL` | `http://127.0.0.1:6174` | 上游 daemon 地址 |
| `MAHJONG_AGENT_WECHAT_TOKEN` | 空 | 上游 token；为空时读 token 文件 |
| `MAHJONG_AGENT_WECHAT_TOKEN_FILE` | `~/.local/share/agent-wechat/api-token` | token 文件 |
| `MAHJONG_AGENT_WECHAT_ACCOUNT` | 空 | 账号 id 或 alias；多账号时必填 |
| `MAHJONG_AGENT_WECHAT_RAW_ENDPOINT` | `http://127.0.0.1:8790/api/channels/agent-wechat/raw` | Runtime 入站地址 |
| `MAHJONG_AGENT_WECHAT_AGENT_TOKEN` | 空 | 投递 Runtime 的 token；回退 `MAHJONG_AGENT_API_TOKEN` 或 `data/runtime_api_token` |
| `MAHJONG_AGENT_WECHAT_CONTROL_HOST` | `127.0.0.1` | 控制面监听地址，不要改成公网 |
| `MAHJONG_AGENT_WECHAT_CONTROL_PORT` | `8792` | 控制面端口 |
| `MAHJONG_AGENT_WECHAT_CONTROL_TOKEN` | 空 | 控制面 token；为空时仅依赖回环绑定 |
| `MAHJONG_AGENT_WECHAT_STATE_DIR` | `data/agent_wechat_adapter` | 游标、账本、重放队列 |
| `MAHJONG_AGENT_WECHAT_SEND_ENABLED` | `false` | 发送总闸初始值 |
| `MAHJONG_AGENT_WECHAT_AUTO_SEND_REPLY` | `false` | 自动外发初始值 |
| `MAHJONG_AGENT_WECHAT_FORWARD_SELF` | `false` | 是否把自消息交给 Runtime |
| `MAHJONG_AGENT_WECHAT_FORWARD_SYSTEM` | `false` | 是否转发系统提示 |
| `MAHJONG_AGENT_WECHAT_REPLAY_HISTORY` | `false` | 首次启动是否从游标 0 回放 |
| `MAHJONG_AGENT_WECHAT_START_CURSOR` | 空 | 显式指定起始游标 |
| `MAHJONG_AGENT_WECHAT_DRY_RUN` | `false` | 只读模式 |
| `MAHJONG_AGENT_WECHAT_SEND_WAIT_SECONDS` | `60` | 等待发送终态的时间 |
| `MAHJONG_AGENT_WECHAT_CONTACT_CACHE_TTL_SECONDS` | `600` | 联系人缓存 TTL |
| `MAHJONG_AGENT_WECHAT_THREAD_CACHE_TTL_SECONDS` | `300` | 会话展示信息缓存 TTL |
| `MAHJONG_AGENT_WECHAT_LOG_LEVEL` | `info` | `debug` / `info` / `warn` / `error` |

## 控制面

只监听回环。配置了 `MAHJONG_AGENT_WECHAT_CONTROL_TOKEN` 后，除 `GET /health` 外都必须
带 `X-Mahjong-Agent-Token` 或 `Authorization: Bearer`。

| 路由 | 用途 |
| --- | --- |
| `GET /health` | 开关状态、账号、缓存数量、当前游标 |
| `GET/POST /send-channel` | 发送总闸 |
| `GET/POST /auto-send` | 自动外发开关 |
| `GET /contacts`、`GET /rooms` | 缓存中的联系人和群 |
| `GET /resolve?query=&kind=` | 按名字消歧，返回候选与 `recommendedId` |
| `POST /send` | 兼容 Runtime 的文本外发合同 |
| `POST /replay-spool` | 手动重放 Runtime 暂时不可用期间积压的入站 |

`POST /send` 示例：

```json
{
  "to": "wxid_xxx 或唯一备注名 或 123@chatroom",
  "target_type": "contact",
  "text": "晚上三缺一，来吗？",
  "source_trace_id": "biz-anchor-123"
}
```

- 目标先按精确 id 匹配，再按唯一完全匹配的名字匹配；
- 重名、仅前缀命中或模糊命中返回 HTTP 409 和候选列表，绝不自动挑人；
- 有 `source_trace_id` 时幂等键跨重试稳定；没有锚点时使用一次性键，避免误去重。

## 可靠性

- **事件游标**：每处理完一条消息才推进游标，原子写入 `events_cursor.json`。
- **Runtime 不可用**：载荷落 `ingress_spool/`，恢复后按顺序重放，成功才删除。
- **发送去重**：`delivery_ledger.jsonl` 只记录上游 `accepted` 的回复。
- **结果不明**：`needs_review` 表示微信是否收到未知，必须人工核对聊天窗口，禁止自动重试。
- **游标过期**：先用 JSON 分页追平；仍失败则快进到尾部并明确记录丢弃窗口。
- **SSE 断线**：指数退避重连，最长 30 秒。

## 身份与 wxid

适配器把上游 `threadId`（私聊为对方 username，群聊以 `@chatroom` 结尾）作为通道寻址键，
同时把备注、昵称、显示名作为展示字段传递。备注和昵称可以随时变化，不能当业务主键。
业务侧应维护自己的 `customer_id`，再通过稳定映射表绑定通道身份；需要时用历史聊天记录
做人工确认。本轮没有实现自动身份迁移。

## 上游升级与合同检查

上游按发布版本独立升级。本仓库的 `integrations/agent-wechat/` 只是冻结的只读参考，
不参与构建，也不随业务改动。升级步骤：

```bash
# 1. 用新版本 OpenAPI 检查破坏性变化
node scripts/check-upstream.mjs --file /path/to/new/openapi.json
# 或直接拉取
node scripts/check-upstream.mjs --url https://example.com/openapi.json

# 2. 确认无漂移后更新锁文件，并显式记录新版本
node scripts/write-contract-lock.mjs --url https://example.com/openapi.json \
  --version vX.Y.Z --commit <commit-sha>
```

检查覆盖适配器真正依赖的 11 条路由、6 个 schema 的必填字段/字段存在性、发送幂等头和
发送状态集合。新增路由不会报警，删改依赖项会非零退出。

不带参数运行 `npm run contract:check` 时，默认从固定 tag 读取 v2.5.0 的 OpenAPI，而不是
使用仓库内冻结的 v2.3.2 副本。

## 测试

```bash
npm test              # Node 单元测试，无需上游或真实微信
npm run contract:check
```

Python 侧通道回归：

```bash
cd ../..
python3 -m pytest -q \
  tests/test_channel_observations.py tests/test_agent_app.py \
  tests/test_wechaty_input_gate.py tests/test_api_reply_delivery.py
```
