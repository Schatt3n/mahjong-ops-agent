# AstrBot 微信原始消息通道

本文档记录当前项目接入 AstrBot 的方式。当前目标不是让 AstrBot 直接替麻将馆回复，而是先把微信私聊、群聊的原始消息稳定转发到麻将运营 runtime，作为后续 Agent 主流程的消息输入通道。

## 当前定位

- AstrBot 只作为微信消息网关。
- Mahjong Agent Runtime 仍然是业务主流程入口。
- 第一阶段只接收和记录原始消息，不自动回复微信。
- 插件默认会拦截 AstrBot 自己的后续处理，避免 AstrBot 内置 Agent 抢先回复。

## 本机安装位置

AstrBot 安装在：

```bash
/Users/wangjie/Desktop/AstrBot
```

Python 虚拟环境：

```bash
/Users/wangjie/Desktop/AstrBot/venv
```

麻将项目中的 AstrBot 插件源码：

```bash
/Users/wangjie/Desktop/mahjong_agent_core/integrations/astrbot/mahjong_channel_bridge
```

AstrBot 插件目录中的软链接：

```bash
/Users/wangjie/Desktop/AstrBot/data/plugins/mahjong_channel_bridge
```

## 启动方式

先启动麻将 runtime：

```bash
cd /Users/wangjie/Desktop/mahjong_agent_core
python scripts/run_agent_app.py
```

再启动 AstrBot：

```bash
cd /Users/wangjie/Desktop/AstrBot
venv/bin/python main.py
```

AstrBot WebUI：

```text
http://127.0.0.1:6185
```

麻将 runtime 控制台：

```text
http://127.0.0.1:8790
```

## AstrBot WebUI 登录

AstrBot 首次启动会在日志里打印初始账号和密码。本机当前启动日志中显示：

```text
username: astrbot
password: zniV9jwAJ8sqlhciZhXsDLrr
```

如果后续 AstrBot 重置了配置，以最新启动日志为准。

## 微信接入步骤

进入 AstrBot WebUI 后：

1. 新建机器人。
2. 选择个人微信通道，通常是 `weixin_oc`。
3. 按页面提示扫码登录微信。
4. 用微信私聊或群聊发送测试消息。
5. 在麻将 runtime 页面点击 `AstrBot 原始消息` 查看最近转发结果。

也可以直接访问接口：

```bash
curl 'http://127.0.0.1:8790/api/channels/astrbot/raw?limit=20'
```

## Runtime 接口

AstrBot 插件会把消息 POST 到：

```text
POST http://127.0.0.1:8790/api/channels/astrbot/raw
```

查看最近消息：

```text
GET http://127.0.0.1:8790/api/channels/astrbot/raw?limit=20
```

消息会追加写入：

```bash
/Users/wangjie/Desktop/mahjong_agent_core/logs/astrbot_weixin_raw.jsonl
```

每条消息也会进入 trace 日志，事件名为：

```text
astrbot_raw_message_received
```

## 插件环境变量

AstrBot 插件支持以下环境变量：

```bash
# 原始消息转发目标
export MAHJONG_ASTRBOT_RAW_ENDPOINT='http://127.0.0.1:8790/api/channels/astrbot/raw'

# 请求超时秒数
export MAHJONG_ASTRBOT_BRIDGE_TIMEOUT_SECONDS='3'

# 是否阻止 AstrBot 继续处理该消息，默认开启，避免 AstrBot 内置 Agent 回复
export MAHJONG_ASTRBOT_BRIDGE_STOP_EVENT='1'
```

麻将 runtime 支持自定义 AstrBot 原始消息日志路径：

```bash
export MAHJONG_ASTRBOT_RAW_LOG_PATH='/path/to/astrbot_weixin_raw.jsonl'
```

## 当前 payload 字段

插件会尽量保留 AstrBot 事件里的原始信息：

- `channel`
- `platform_name`
- `platform_id`
- `message_type`
- `conversation_id`
- `session_id`
- `group_id`
- `sender_id`
- `sender_name`
- `self_id`
- `message_id`
- `source_message_id`
- `text`
- `outline`
- `message_chain`
- `raw_message`
- `message_obj`

其中 `message_chain`、`raw_message`、`message_obj` 用于后续支持图片、语音、文件、群聊上下文等能力。

## 后续演进

当前只是消息通道打通。后续应按生产链路继续做：

- 将 AstrBot 原始消息转换成统一 `InboundMessage`。
- 给私聊、群聊分别建立稳定的 `conversation_id`。
- 做消息幂等、顺序控制和短时间合并，避免用户分多条发消息时主流程过早执行。
- 将文本、图片、语音分别进入不同的理解链路。
- 输出仍先走待审批发件箱，不直接自动发微信。
- 当审查、幂等、限频、人工接管稳定后，再考虑自动发送。

## 参考文档

- AstrBot 个人微信接入：https://docs.astrbot.app/platform/weixin_oc.html
- AstrBot 消息事件开发：https://docs.astrbot.app/dev/star/guides/listen-message-event.html
- AstrBot 消息发送开发：https://docs.astrbot.app/dev/star/guides/send-message.html
