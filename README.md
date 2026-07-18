# Mahjong Ops Agent

面向麻将馆、棋牌室等本地生活私域运营场景的目标驱动 Agent。

系统接收 Web 或微信消息，由主模型围绕“帮助客户找到合适的局”自主规划下一步，按需查询房态和现有局、创建组局需求、推荐候选人、生成邀约、记录反馈并推进状态。后端不替模型编写业务决策树，只负责工具合同、权限、状态机、幂等、并发、持久化、审查和审计。

> 当前阶段：本地优先的小范围试运行版本。已经完成 Web 控制台、SQLite 持久化、真实模型调用和 WeChaty 白名单灰度测试；微信个人号通道属于非官方接入方式，不等同于微信官方生产通道。

## 为什么做这个项目

线下棋牌室的组局运营高度依赖老板经验，但真实对话往往不完整、不规范并且持续变化：

- 客户可能说“今晚 0.5 有人吗”“371”“人齐开”，而不是填写固定表单。
- 多条碎片消息之间会穿插闲聊、引用回复、改时间、改烟况和临时取消。
- 老客户的玩法、档位、烟况、常来时段和同桌关系会影响匹配结果。
- 查询、邀约、确认和取消可能同时发生，人数不能重复计算，消息不能串会话。
- 对外回复要像老板说话，同时不能泄露模型、工具、审批、内部备注和其他客户隐私。

因此，本项目没有把业务做成一组不断增长的 `if-else`，而是采用“模型负责理解和规划，后端负责可信执行”的架构。

## 当前能力

### 业务能力

- 识别找局、组局、补充条件、确认、拒绝、协商、取消和闲聊等消息。
- 支持玩法、底注/封顶、时间、人齐开、时长、烟况、人数结构和多座位需求。
- 查询当前局和房间库存，匹配现有局或创建新局。
- 根据画像、近期邀约、疲劳度和客户关系推荐候选人。
- 生成待审批邀约，记录候选人反馈并更新局内人数和状态。
- 支持局的开始/结束时间、超时取消、房间释放、失败归档和状态回溯。

### Agent 能力

- 目标驱动主循环：模型动态决定工具、顺序和下一步动作。
- 多轮上下文：包含最近对话、checkpoint、用户画像、关系、任务记忆、局况和工具结果。
- 碎片输入聚合：在信息尚未形成完整意图时短暂等待，静默超时后重新触发判断。
- 上下文治理：预算预检、当前 loop 去重、决策投影和长对话摘要。
- 死循环保护：识别重复观察、短周期循环和连续无进展，先要求模型重规划，再安全终止。
- 客户可见文本处理：业务决策、自然话术生成和信息泄露审查相互分离。

### 工程能力

- 同一会话串行、消息幂等、工具幂等、可恢复 lease 和会话版本控制。
- SQLite 原子事务；可选 Redis 分布式协调锁。
- HTTP 鉴权、请求体限制、并发限制和频率限制。
- WeChaty 接入具备白名单、外发开关、投递去重、失败暂存和重放能力。
- 全链路记录 trace、上下文、模型输出、工具调用、状态迁移和最终结果。
- 持续沉淀 badcase、golden dataset、few-shot examples 和回归评测。

## 架构

```mermaid
flowchart LR
  subgraph Channels["消息通道"]
    Web["Web 控制台 / HTTP API"]
    WeChaty["WeChaty 私聊与群聊"]
  end

  subgraph Boundary["输入边界"]
    Buffer["碎片消息持久化聚合"]
    Gate["业务 / 闲聊 / 等待 / 忽略"]
  end

  subgraph Agent["目标驱动主 Agent"]
    Runtime["AgentRuntime<br/>锁、幂等、会话版本"]
    Context["Context Builder<br/>检索、投影、预算、摘要"]
    Loop["Agent Loop<br/>规划 -> 工具 -> 观察 -> 重规划"]
    Progress["Progress Monitor<br/>循环与无进展检测"]
  end

  subgraph Execution["可信执行"]
    Contract["输出合同校验"]
    Gateway["Tool Gateway<br/>schema、权限、幂等"]
    Visible["话术生成与内容审查"]
  end

  subgraph State["状态与质量资产"]
    SQLite["SQLite<br/>局、房间、画像、记忆、草稿"]
    Trace["Trace / Audit"]
    Eval["Badcase / Golden / Eval"]
  end

  Web --> Buffer
  WeChaty --> Buffer
  Buffer --> Gate
  Gate --> Runtime
  Runtime --> Context
  Context --> Loop
  Loop --> Contract
  Contract --> Gateway
  Gateway --> SQLite
  Gateway --> Loop
  Loop --> Progress
  Loop --> Visible
  Visible --> Web
  Visible --> WeChaty
  Runtime --> Trace
  Gateway --> Trace
  Trace --> Eval
```

这不是多 Agent 系统。系统只有一个负责完成业务目标的主 Agent；输入分流、摘要、话术生成和内容审查是边界清晰的一次性模型任务，不拥有独立业务目标和状态。

详细代码链路见 [Agent Runtime 架构解析](docs/runtime_loop_design.md)，完整系统说明见 [系统讲解文档](docs/system_explanation.html)。

## 主 Agent Loop

主循环刻意保持简单：

```text
handle user message
  -> acquire conversation lock
  -> check message idempotency
  -> build context
  -> call LLM
  -> validate AgentAction contract
  -> if tool calls:
       execute through ToolGateway
       append real tool results
       check progress
       continue loop
  -> else:
       generate and review customer-visible reply
       persist result
       stop
```

模型每一轮必须返回结构化 `AgentAction`，声明：

- 当前目标和目标状态。
- 简短的决策依据。
- 是否需要调用工具，以及工具参数和调用原因。
- 是否可以停止、还剩哪些工作、是否依赖工具结果。
- 面向客户的最终回复，或者明确等待用户/转人工。

后端不会执行合同不合法的工具调用，也不会接受模型直接修改数据库。

## 上下文与记忆

每次调用主模型时，`AgentContextBuilder` 根据当前目标组装有限上下文：

| 上下文 | 作用 |
| --- | --- |
| 当前消息与输入窗口 | 保留本轮原始片段、引用消息、静默超时和批次版本 |
| 最近对话 | 提供短期语言上下文，保持业务与闲聊可穿插 |
| 会话 checkpoint | 保存长对话压缩后的目标、事实、待办和待确认问题 |
| 用户画像与客户关系 | 提供稳定偏好、历史同桌关系和明确冲突 |
| 当前任务记忆 | 保存“这一次不和 C 打”等尚未写入长期画像的约束 |
| 当前局与房态 | 只加载与当前会话或用户相关的有界决策投影 |
| 可用工具 | 由后端按当前权限动态提供工具 schema |
| 上轮工具结果 | 让模型基于真实查询和状态变化继续规划 |

摘要默认在以下情况下触发：

- 最近对话达到 12 轮，且距离上次摘要至少 6 轮，并且粗估超过 3000 tokens。
- 调用主模型前，上下文粗估超过单次预算的 85%。

摘要不会替代业务状态。局、人数、房间和邀约仍以数据库为准；checkpoint 只帮助模型恢复目标和对话事实。

## 工具

| 工具 | 副作用 | 作用 |
| --- | --- | --- |
| `search_current_games` | 无 | 查询当前局并计算加入后的座位状态 |
| `check_room_availability` | 无 | 查询指定时间段的房间库存 |
| `reserve_room` | 有 | 为有效局创建房间预留 |
| `search_customers` | 无 | 按画像、疲劳度和关系搜索候选人 |
| `create_game` | 有 | 创建待组局记录，不直接发送消息 |
| `create_invite_drafts` | 有 | 为候选人生成待审批邀约草稿 |
| `create_outbound_message_drafts` | 有 | 创建通道无关的外发草稿 |
| `record_candidate_reply` | 有 | 记录确认、拒绝、协商、到店等反馈 |
| `update_game_status` | 有 | 按状态机推进、取消或归档局 |
| `record_user_memory` | 有 | 写入任务记忆或待确认长期画像候选 |
| `update_context_checkpoint` | 有 | 更新会话 checkpoint |
| `record_badcase` | 有 | 归档失败样本和评测候选 |

所有有副作用的工具都会经过 schema、主体权限、资源归属、状态机、幂等和并发版本校验。

## 快速启动

### 1. 环境要求

- Python 3.11+
- DeepSeek 或其他 OpenAI-compatible 模型 API
- Node.js + pnpm，仅在启用 WeChaty 桥接时需要
- Redis 可选；单机默认不依赖 Redis

### 2. 安装

```bash
git clone git@github.com:Schatt3n/mahjong-ops-agent.git
cd mahjong-ops-agent
python -m pip install -e ".[dev]"
```

需要 Redis 协调锁时：

```bash
python -m pip install -e ".[dev,distributed]"
```

### 3. 配置模型

项目启动时会读取根目录 `.env`，且不会覆盖已经存在的环境变量：

```bash
MAHJONG_LLM_PROVIDER=deepseek
MAHJONG_LLM_MODEL=<your-model-name>
MAHJONG_LLM_API_KEY=<your-api-key>
MAHJONG_LLM_BASE_URL=https://api.deepseek.com

# 推荐为写入 API 配置鉴权；不要提交真实 token
MAHJONG_AGENT_API_TOKEN=<local-api-token>

# 可选：房间和 Redis
MAHJONG_ROOM_IDS=room_1,room_2,room_3
MAHJONG_REDIS_URL=redis://127.0.0.1:6379/0
```

### 4. 启动服务

```bash
python scripts/run_agent_app.py
```

打开：

- 控制台：<http://127.0.0.1:8790/>
- 健康检查：<http://127.0.0.1:8790/api/health>
- Runtime 信息：<http://127.0.0.1:8790/api/runtime>

### 5. 发送测试消息

设置了 API token 时，请附带 `Authorization`：

```bash
curl -X POST http://127.0.0.1:8790/api/message \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <local-api-token>' \
  -d '{
    "conversation_id": "trial_001",
    "sender_id": "customer_001",
    "sender_name": "",
    "message_id": "message_001",
    "text": "今晚七点0.5无烟，帮我组一个"
  }'
```

## WeChaty 灰度通道

WeChaty 桥只负责消息收发，不承载业务决策：

```bash
cd integrations/wechaty/mahjong-wechaty-bridge
pnpm install

export MAHJONG_AGENT_API_TOKEN=<local-api-token>
export MAHJONG_WECHATY_SEND_ENABLED=false
export MAHJONG_WECHATY_AUTO_SEND_REPLY=false
pnpm start
```

默认行为：

- 接收微信原始消息并转发到 `/api/channels/wechaty/raw`。
- 按 `conversation_id + sender_id` 聚合碎片输入。
- 默认仅允许白名单或测试范围进入主 Agent。
- 发送通道和自动回复默认关闭，需要分别显式开启。
- 失败入站消息进入本地 spool，成功外发写入投递账本，避免重复发送。

建议始终使用测试号、小范围白名单和人工审批。个人微信机器人存在账号风控，不应把非官方协议接入视为稳定 SLA 通道。

## API

| 方法 | 路径 | 作用 |
| --- | --- | --- |
| `GET` | `/api/health` | 健康检查 |
| `GET` | `/api/runtime` | 查看运行配置和组件状态 |
| `POST` | `/api/message` | Web/API 消息入口 |
| `GET` | `/api/state` | 查看局、草稿、画像和记忆 |
| `GET` | `/api/traces?trace_id=...` | 查询指定 trace |
| `GET` | `/api/logs?limit=...` | 查询日志尾部 |
| `POST` | `/api/invite-drafts/action` | 审批、拒绝或发送邀约草稿 |
| `POST` | `/api/badcases` | 手工标记 badcase |
| `POST` | `/api/reset-state` | 清空本地测试状态和记忆 |
| `POST` | `/api/channels/wechaty/raw` | WeChaty 原始消息入口 |

设置 `MAHJONG_AGENT_API_TOKEN` 后，除健康检查和静态页面外的受保护接口需要 Bearer Token 或 `X-Mahjong-Agent-Token`。

## 数据与可观测

默认本地数据：

```text
data/agent_runtime.sqlite3         # 主状态库
logs/agent_runtime_trace.log       # Agent 全链路 trace
logs/wechaty_weixin_raw.jsonl      # 微信原始消息日志
runtime_data/                      # 本地评测报告与临时数据库
```

SQLite 保存客户画像、客户关系、局、房间、邀约草稿、状态迁移、对话、checkpoint、任务记忆、消息结果、幂等账本和待处理输入批次。Redis 只在配置后承担跨进程协调，不是业务真相来源。

每条链路可以通过 `trace_id` 回溯：

```text
用户输入 -> 上下文构建 -> 模型请求/响应 -> 工具调用/结果
        -> 状态迁移 -> 话术生成/审查 -> 最终回复
```

日志格式：

```text
traceId-yyyy-mm-dd hh:mm:ss-loglevel: content
```

## 测试与评测

完整测试：

```bash
PYTHONPATH=src python -m pytest -q
```

确定性回归、边界检查、badcase 覆盖和 golden dataset 校验：

```bash
PYTHONPATH=src python scripts/run_evals.py
```

显式调用真实模型的老板对话评测：

```bash
PYTHONPATH=src python scripts/run_real_owner_chat_live_eval.py
```

最近一次完整验证结果：

- 自动化测试：`237 passed, 1 skipped`
- Agent 确定性回归：`138/138`
- 真实 DeepSeek 老板对话场景：`10/10`
- badcase 回归覆盖：`fixed=14, open=0`

质量资产位于 `eval/`：

```text
eval/badcases/                         # 失败样本
eval/regression/                       # 确定性回归集
eval/golden/                           # 真实聊天 golden dataset
eval/few_shot_examples.jsonl           # 认可的话术样本
```

基本原则：先把问题沉淀成可复现样本，再通过提示词、合同、工具、数据模型或通用运行时机制解决，不把每个 badcase 补成业务分支。

## 代码入口

```text
scripts/run_agent_app.py                  # 服务启动入口
scripts/agent_runtime_app.py              # Web、HTTP API、通道适配

src/mahjong_agent_runtime/
  runtime.py                              # 锁、消息幂等、会话版本
  loop.py                                 # 主 Agent Loop
  context.py                              # 上下文构建与决策投影
  lifecycle.py                            # 预算预检、摘要、上下文重建
  processing.py                           # AgentAction、工具和回复处理
  tools.py                                # Tool Gateway 与工具注册
  progress.py                             # 死循环和无进展检测
  summary.py                              # checkpoint 摘要
  copywriting.py                          # 客户可见话术生成
  visibility.py                           # 客户可见信息审查
  sqlite_store.py                         # SQLite 持久化
  coordination.py                         # 本机与 Redis 协调锁
  prompts/                                # 主模型及一次性模型任务提示词

integrations/wechaty/                     # 微信消息桥
eval/                                     # badcase、golden、回归和 few-shot
tests/                                    # 单元、边界和生产不变量测试
```

## 生产边界

当前系统已经具备生产化 Agent 所需的核心运行时能力，但仍需区分“代码能力”和“外部依赖能力”：

- SQLite 适合单店、几百名客户和本地 MacBook 部署；多节点部署需要统一数据库、Redis 锁和正式迁移方案。
- WeChaty 已验证消息桥接链路，但个人微信协议稳定性和账号风控无法由本项目保证。
- 自动外发应在真实数据持续回归、误发率达标和业务方确认后逐步放开。
- 资金、优惠、纠纷和隐私敏感操作不应授权给模型自动执行。
- 模型输出无法做到绝对确定，关键状态始终以工具结果和数据库为准。

## 开发准则

1. 主 Agent 负责目标规划，后端不维护业务决策树。
2. 模型只能提出动作，不能绕过 Tool Gateway 修改状态。
3. 工具结果必须回喂模型，客户回复不能虚构未执行动作。
4. 所有副作用动作必须具备权限、幂等、状态机和审计能力。
5. 所有客户可见文本都要经过专用生成与审查合同。
6. 修复必须进入测试、badcase、golden dataset 或 eval 回归。
7. 不把微信个人号灰度验证包装成官方生产接入。
