# Hermes 微信原始消息通道

目标：先只把 Hermes / Weixin 网关收到的原始消息转发到本项目，不让 Hermes 自己进入后续 Agent 派发。

## 本项目接收端

- `POST /api/channels/hermes/raw`：接收 Hermes 插件转发的原始消息。
- `GET /api/channels/hermes/raw?limit=20`：查看最近收到的原始消息。
- 原始消息日志：`logs/hermes_weixin_raw.jsonl`。
- trace 日志：`logs/agent_runtime_trace.log`，事件名为 `hermes_raw_message_received`。

## Hermes 插件

插件路径：

```bash
/Users/wangjie/Desktop/mahjong_agent_core/integrations/hermes/mahjong_raw_weixin_bridge
```

建议软链接到 Hermes 用户插件目录：

```bash
mkdir -p ~/.hermes/plugins
ln -sfn /Users/wangjie/Desktop/mahjong_agent_core/integrations/hermes/mahjong_raw_weixin_bridge ~/.hermes/plugins/mahjong-raw-weixin-bridge
```

如果本机还没有 Hermes，需要先安装：

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

启用插件：

```bash
hermes plugins enable mahjong-raw-weixin-bridge
```

配置微信网关并扫码：

```bash
hermes gateway setup
hermes gateway
```

## 环境变量

- `MAHJONG_HERMES_RAW_ENDPOINT`：默认 `http://127.0.0.1:8790/api/channels/hermes/raw`。
- `MAHJONG_HERMES_BRIDGE_SKIP`：默认 `1`，转发后跳过 Hermes 后续派发。
- `MAHJONG_HERMES_BRIDGE_FAIL_CLOSED`：默认 `1`，转发失败时也跳过 Hermes 后续派发，避免误触发其他 Agent。
- `MAHJONG_HERMES_BRIDGE_PLATFORMS`：默认 `weixin,wechat`。
- `MAHJONG_HERMES_RAW_LOG_PATH`：默认 `logs/hermes_weixin_raw.jsonl`。

## 验证方式

启动麻将项目服务后，先用本地请求验证接收端：

```bash
curl -s -X POST http://127.0.0.1:8790/api/channels/hermes/raw \
  -H 'Content-Type: application/json' \
  -d '{"platform":"weixin","text":"测试Hermes原始消息","message_id":"local_test_001","source":{"user_id":"u1","chat_id":"c1","chat_type":"dm"}}'

curl -s http://127.0.0.1:8790/api/channels/hermes/raw?limit=5
```

如果能看到 `records`，说明项目侧已经能接住原始消息。后续只需要验证 Hermes 微信网关是否能把真实微信消息推过来。
