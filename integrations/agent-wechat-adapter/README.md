# agent-wechat 通道适配器

把 [agent-wechat](https://github.com/thisnick/agent-wechat) 的本地 HTTP/SSE API 翻译成
Mahjong Agent Runtime 的原始通道合同。上游按发布版本独立部署、独立升级；本目录不
修改、不 fork 上游源码，也不读取上游的数据库、锁文件或内部缓存。

适配器的接口锁基于上游 `v2.5.0`（commit `ce5c068`）的 OpenAPI。
上游 daemon 从 [官方发布渠道](https://github.com/thisnick/agent-wechat/releases) 独立安装。

> 个人微信接入属于非官方、灰度通道。请使用单独的运营测试号，先小范围验证，并保留店员
> 通过官方微信人工兜底的能力。本项目不承诺规避平台风控。

## 边界

| 能力 | 状态 |
| --- | --- |
| 私聊 / 群聊文本入站 | 支持，转为 `wechaty:contact:` / `wechaty:room:` 会话命名空间 |
| 图片、语音、视频、文件、引用等入站 | 透传消息 id、类型、可读正文和元数据；未下载媒体，未做 OCR / 转写 |
| 联系人 / 群检索与消歧 | 支持，重名或仅前缀命中必须上层确认 |
| 文本外发 | 支持，带幂等键；`needs_review` 只告警，禁止自动重试 |
| 朋友圈、加好友 | 本适配器未接入 |
| 修改备注、群管理 | 上游当前没有备注写路由；群管理仅有部分路由且本适配器未开放 |

本适配器只做协议翻译、可靠投递和本地闸门。用户画像、短期/长期记忆、会话状态、业务
权限和客户可见文本审查全部留在 Python Runtime。

## 部署

本项目的推荐平台和验证状态见 [环境要求](../../README.md#1-环境要求)。下文的 Profile 说明针对上游 `wx` 的登录方式。

### 1. 启动上游 daemon

```bash
npm install -g @agent-wechat/cli
wx serve             # 默认 127.0.0.1:6174
# 首次使用上游 wx 时新建独立 Profile
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
# 在项目根目录运行
MAHJONG_AGENT_WECHAT_RAW_ENDPOINT=http://127.0.0.1:8790/api/channels/agent-wechat/raw \
node integrations/agent-wechat-adapter/src/main.mjs
```

启动时会确认只有一个可用账号；多账号必须显式设置 `MAHJONG_AGENT_WECHAT_ACCOUNT`。
默认从事件日志尾部开始，不重放历史消息。

只读联调：

```bash
node integrations/agent-wechat-adapter/src/main.mjs --dry-run
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
`/auto-send` 合同。外发默认关闭。配置定向外发名单后，手动发送只需要发送总闸，自动回复还需打开自动外发开关。

### 定向群自动回复

将 Runtime 和适配器的 `MAHJONG_AGENT_WECHAT_GROUP_REPLY_POLICY_PATH` 指向同一个本机 JSON 文件。填入上游账号真实 ID、精确 `room_ids` 及带时区的 `enabled_at`。同时把获准外发的群加入同账号的外发名单。授权前的消息仍可入站观察，自动回复只处理启用后的入站文本。

定向回复只允许确认诉求或追问，经话题分类、客户可见内容审查和确定性过滤后外发，有上游服务器消息 ID 时可引用原消息。托管群继续走既有的独立业务入口。要观测其他群，可将 Runtime 的 `MAHJONG_AGENT_WECHAT_TOPIC_OBSERVE_PATH` 指向含 `observe_all_unmanaged_rooms` 或精确群条目的本机文件。部署配置、账号 ID 和凭证存于源码树之外。

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
| `MAHJONG_AGENT_WECHAT_GROUP_REPLY_POLICY_PATH` | `data/agent_wechat_group_auto_reply.local.json` | Runtime 和适配器共同读取的群回复授权 |
| `MAHJONG_AGENT_WECHAT_OUTBOUND_ALLOWLIST_PATH` | 未配置 | 本机按账号绑定的外发名单。文件使用 `account_id` 和 `recipients`，每项包含 `thread_id`、`label` 和可选的 `aliases`。启用总闸后可定向手动发送，自动回复仍由自己的开关控制。 |
| `MAHJONG_AGENT_WECHAT_FORWARD_SELF` | `false` | 是否把其余本人消息转发为观察记录；已识别的老板跨设备消息始终转发 |
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
| `POST /deliver-ingress-reply` | Runtime 交回已审查的入站自动回复，使用控制面令牌或 Runtime API 令牌 |
| `POST /replay-spool` | 手动重放原同步模式遗留的入站记录 |

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

- 每条入站请求使用 `Prefer: respond-async`。Runtime 把原始消息和待办状态写入 SQLite 后返回 HTTP 202，适配器随即推进**事件游标**。入站路由最多并行处理四个会话，各会话按落库顺序处理。重复来源消息沿用首次记录，重启后会续处理待办。
- 自动回复开启时，适配器也只等待持久接收确认。Runtime 保存待发结果，单独串行交给适配器执行发送授权、老板已回复检查及发送去重。其他会话的入站处理可以继续。回调一时不可用时，后台按原消息重试交接。
- 消息详情或持久接收失败会保留游标，从上一个已确认位置重连补读。旧 `ingress_spool/` 仍可手动重放。
- **发送去重**：`delivery_ledger.jsonl` 只记录上游 `accepted` 的回复。
- **结果不明**：`needs_review` 表示微信是否收到未知，必须人工核对聊天窗口，禁止自动重试。
- **游标过期**时定位上游最早仍保留的事件，记录已清理的区间，再用 JSON 分页补齐保留窗口。读取或处理失败时保留已确认的游标。
- **SSE 断线**：指数退避重连，最长 30 秒。

## 身份与 wxid

适配器把上游 `threadId`（私聊为对方 username，群聊以 `@chatroom` 结尾）作为通道寻址键，
同时把备注、昵称、显示名作为展示字段传递。备注和昵称可以随时变化，不能当业务主键。
业务侧应维护自己的 `customer_id`，再通过稳定映射表绑定通道身份；需要时用历史聊天记录
做人工确认。本轮没有实现自动身份迁移。

## 上游升级与合同检查

先按下面的步骤检查目标版本的接口，再升级独立运行的上游 daemon。

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

不带参数运行 `npm run contract:check` 时，默认从上游 v2.5.0 tag 读取 OpenAPI。

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
