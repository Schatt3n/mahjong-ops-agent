# Wechaty 微信原始消息通道

本文档记录 Wechaty 作为微信消息通道的试验方案。当前目标是验证能否监听真实微信私聊和群聊消息，并把原始消息转发到麻将运营 runtime。

## 当前定位

- Wechaty 只作为消息通道 SDK。
- 具体能否监听真实微信，取决于底层 Puppet。
- 当前第一版只接收和记录消息，不自动回复、不自动发送。
- Mahjong Agent Runtime 仍然负责业务理解、工具调用、状态管理、审查和发件箱。

## 目录

Wechaty bridge 位于：

```bash
/Users/wangjie/Desktop/mahjong_agent_core/integrations/wechaty/mahjong-wechaty-bridge
```

Runtime 原始消息接口：

```text
POST http://127.0.0.1:8790/api/channels/wechaty/raw
GET  http://127.0.0.1:8790/api/channels/wechaty/raw?limit=20
```

原始消息日志：

```bash
/Users/wangjie/Desktop/mahjong_agent_core/logs/wechaty_weixin_raw.jsonl
```

## 安装

本机当前没有系统级 Node，但 Codex runtime 自带 Node 和 pnpm。可以这样安装：

```bash
cd /Users/wangjie/Desktop/mahjong_agent_core/integrations/wechaty/mahjong-wechaty-bridge
/Users/wangjie/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/pnpm install
```

## 启动

先启动麻将 runtime：

```bash
cd /Users/wangjie/Desktop/mahjong_agent_core
python scripts/run_agent_app.py
```

再启动 Wechaty bridge：

```bash
cd /Users/wangjie/Desktop/mahjong_agent_core/integrations/wechaty/mahjong-wechaty-bridge
/Users/wangjie/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/pnpm start
```

如果需要指定转发地址：

```bash
export MAHJONG_WECHATY_RAW_ENDPOINT='http://127.0.0.1:8790/api/channels/wechaty/raw'
```

## 测试期路由范围

Wechaty 原始消息会全部写入 `logs/wechaty_weixin_raw.jsonl`，但不一定都会进入 Agent。当前测试阶段默认只让自己发出的消息进入 Agent，避免误处理真实客户消息：

```bash
export MAHJONG_WECHATY_ROUTE_SCOPE=self_only
```

可选值：

- `self_only`：只处理 `self_message=true` 的消息，适合本机测试。
- `incoming_only`：只处理别人发来的消息，适合真实试用前的小范围验证。
- `all`：自己和别人发的文本消息都会进 Agent，一般不建议生产使用，容易形成回环。

## Puppet 选择

Wechaty 是统一 SDK，真正决定能不能登录、能不能收群聊的是 Puppet。

### 默认 Puppet

如果不设置 `WECHATY_PUPPET`，Wechaty 会使用默认策略。这个方式最简单，但不保证能登录真实微信，也不保证能收群聊。

```bash
pnpm start
```

### Web 微信 Puppet

Web 微信方式可能无法登录很多账号，尤其是较新的微信号。能不能用要现场验证。

```bash
export WECHATY_PUPPET=wechaty-puppet-wechat
pnpm add wechaty-puppet-wechat
pnpm start
```

### PadLocal Puppet

PadLocal 能力更接近完整个人微信，通常需要 token，也可能有费用和风控问题。

```bash
export WECHATY_PUPPET=wechaty-puppet-padlocal
export WECHATY_PUPPET_PADLOCAL_TOKEN='your-token'
pnpm add wechaty-puppet-padlocal
pnpm start
```

### Mac/Windows Hook 类 Puppet

这类方案更接近“监听当前桌面微信”，但维护成本和风控风险最高。Mac 方向需要单独验证对应 Puppet 是否仍可用、是否维护、是否支持当前微信版本。

## 当前 payload 字段

bridge 会尽量保留原始信息：

- `channel`
- `platform_name`
- `puppet`
- `conversation_id`
- `message_id`
- `source_message_id`
- `message_type`
- `is_room`
- `room`
- `sender_id`
- `sender_name`
- `talker`
- `listener`
- `text`
- `raw_text`
- `self_message`
- `payload`

## 验证步骤

1. 启动麻将 runtime。
2. 启动 Wechaty bridge。
3. 根据终端二维码扫码登录。
4. 用另一个微信给该账号发私聊消息。
5. 在麻将 runtime 页面点击 `Wechaty 原始消息`。
6. 再测试群聊消息，看是否能拿到 `is_room=true`、`room.id` 和 `room.topic`。

## 是否满足麻将馆目标

只有当以下测试通过，才算接近可用：

- 能稳定登录老板微信或备用微信。
- 能收到私聊消息。
- 能收到群聊消息。
- 群聊消息能拿到稳定群 ID。
- 群聊消息能拿到发送人 ID。
- 消息有稳定 `message_id`，便于幂等去重。
- 图片、语音、文件至少能拿到可处理的 payload。
- 断线后可以恢复，或者能明确报警。

如果默认 Puppet 无法做到这些，需要继续验证 PadLocal、Mac Hook、Windows Hook 或其他通道。

## 参考

- Wechaty Puppet Providers：https://wechaty.js.org/docs/puppet-providers/
- Wechaty PadLocal：https://wechaty.js.org/docs/puppet-providers/padlocal
- Wechaty Web 微信限制：https://wechaty.js.org/docs/puppet-providers/wechat
