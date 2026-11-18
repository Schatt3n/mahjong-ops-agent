# 微信接入试验记录 · 2026-11-13

本轮在一台 Apple M1 Mac 上验证个人微信接入。目标是让运营 Agent 能确认目标会话，接收和发送普通私聊、群聊消息，并尽可能把登录会话留在自己的设备。前一轮的模拟与安卓登录记录见 [2026-11-11 试验记录](../trial.log)。

当前继续用 **GeWe 做小范围灰度**。用户已确认通过 GeWe 发出一条群消息，真实入站回调、私聊双向收发和长时间在线仍待验收。**即时兜底是店员通过官方微信人工发送**。FastBridge 和 iPhone 镜像保留为本地辅助方案，现有证据还不足以启用自动切换或自动补发。

## 本轮结果

| 路线 | 已验证 | 尚未验证 |
| --- | --- | --- |
| GeWe 托管 API | 用户报告一条群消息实际发出。发送接口使用收件方 ID。 | 真实私聊、群聊入站 Webhook、端到端时延、掉线后的恢复、长期成本。没有核实私有化是否包含能在自有 M1 上运行的完整微信登录节点。 |
| 官方 Mac 微信 4.2.0 + FastBridge 1.7.1 | 微信客户端签名为腾讯。FastBridge 从发布标签源码构建成原生 ARM64，39 项上游测试通过，临时 stdio MCP 可以枚举工具，真实微信进程的 `wechat_status` 成功。 | `wechat_read` 未读到消息，发送和监听均未验收。后续用户报告 Mac 微信强制下线，提示提到账号安全。触发原因未知，测试已停止。 |
| AgentWechat 2.3.2 | 校验发布包后启动原生 macOS ARM64 应用和本地后台程序，到达扫码步骤。 | 扫码登录未完成验收，手机端的设备类型、私聊和群聊收发未知。应用与本地后台程序已退出。 |
| wechat-cli `3162883` | 在 M1 从 Go 源码编译出 macOS ARM64 程序，命令行可启动，四项离线协议测试通过。源码中的 `tablet` 指 Android 平板。 | 未在本机进行真实账号扫码或收发。作者报告过扫码、同步和服务端接受一条文本，普通微信群双向收发仍缺少本机证据。 |
| WeChatPadPro v2.01 / 旧 iPad 实现 | 核对 WeChatPadPro 发布包 SHA-256，确认包内确有 macOS ARM64 可执行文件。旧 Ipad860 公开了部分 Go 代码，其随附 Linux 加密库为 x86-64。 | 没有运行 WeChatPadPro 或登录测试。同代 861 版本曾有用户报告扫码时被拒绝，环境为 Windows Docker。 |
| 官方 Linux 微信 | 核对腾讯安装包元数据为 `4.1.13.26`、`arm64`，并确认本机支持硬件虚拟化。 | 未创建 Linux 虚拟机或登录。桌面通知和辅助功能能读到多少真实消息未知。 |

FastBridge 的状态调用单次约 **57 毫秒**，只验证进程和辅助功能可用。读取文件传输助手的首轮调用约 **1.56 秒**，结果为 `WECHAT_FOCUS_FAILED`。当时官方微信显示朋友圈，FastBridge 未能切到聊天。用户手动打开后报告下线。测试期间没有发送消息，也没有完成真实消息读取。两次单次计时不代表收发速度或吞吐量。

GeWe 的公开文档提供 Mac 登录选项、发送 API 和消息回调。手机显示 Mac 在线可以作为登录类型的验收项。私有化还要单独核对微信登录会话由谁持有，尤其是执行节点、会话凭据和消息存储的位置。AidNet 文档描述的是网络出口，不能据此确认会话随之搬到本机。

## 兜底用法

1. **需要马上回复时，走人工。** 由店员在官方手机或 Mac 微信里选定会话，核对 Agent 给出的回复草稿，再手动发送。GeWe 的某条发送结果不明时，先核对目标会话是否已经出现该条消息，避免重复补发。
2. **iPhone 镜像可辅助单条发送。** 此前用镜像 MCP 做过发送测试，用户观察到一轮约一分钟三条。小微的多收件人请求曾只展示首个发送确认卡，可保留逐条核对的操作。Mac 通知接收在此前实验中未稳定捕获，暂不依赖通知自动收消息。
3. **FastBridge 保留为本机试验方案。** 下次从微信聊天页开始，先在文件传输助手验证读取，再测试指定私聊和测试群。它的多会话监听依赖已加载的会话列表预览和未读变化，需要用实际消息核对是否漏读。遇到再次强制下线或安全提示时结束当轮测试，并记录提示范围。

## 下一轮验收

- **先补 GeWe 收发闭环。** 只在明确授权的测试私聊和群里各验证一次接收与发送。用手机会话和回调消息逐条核对发送人、会话、正文、时间与消息 ID，分别记录接口返回和对方实际收到的结果。
- **再测可靠性与成本。** 短暂断网、重启和消息集中到达时，分别统计缺失、重复和恢复时间。每店每天约 1,000 条是估算负载，后续用它评估通道费用、设备占用和人工接管成本。本轮没有测得可用于报价的成本或吞吐量。

## 参考资料

- [GeWe 登录说明](https://doc.geweapi.com/api-139908279)、[Webhook 事件](https://doc.geweapi.com/doc-3146208)、[企业方案](https://www.geweapi.com/pricing)、[AidNet 说明](https://doc.geweapi.com/api-483017826)
- [WeChat FastBridge v1.7.1](https://github.com/huj28-creator/wechat-fastbridge/releases/tag/v1.7.1)、[iPhone 镜像 MCP](https://github.com/orshih6/iphone-mirror-mcp)
- [AgentWechat v2.3.2 发布资产](https://github.com/thisnick/agent-wechat/releases/expanded_assets/v2.3.2)、[平台验收说明](https://github.com/thisnick/agent-wechat/blob/v2.3.2/PLATFORM-ACCEPTANCE.md)
- [wechat-cli 固定提交](https://github.com/kevmo314/wechat-cli/tree/316288343500c1bfd741e17707f357a0778a2f0f)、[WeChatPadPro v2.01 发布资产](https://github.com/WeChatPadPro/WeChatPadPro/releases/expanded_assets/v2.01)、[861 登录报告](https://github.com/langbot-app/LangBot/issues/2239)
