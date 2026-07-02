# Mahjong Ops Workflow

棋牌室自助运营助手。

面向自主棋牌室/麻将馆的可控运营 Workflow 核心引擎。

这个项目沉淀了一套可以接入真实微信、企业微信、小程序、客服系统和 CRM 的“麻将馆运营自动化中台”。它负责理解用户组局意图、维护客户画像、管理待组局队列、推荐合适客户、生成邀约草稿，并在高并发和分布式场景下保证消息不串、不乱序、可观测、可回溯。

更准确地说，当前项目是一个 **agentic workflow**：LLM 负责结合上下文做语义理解、动作提案和回复草稿；流程、状态推进、幂等、客户锁、outbox、风险拦截和审计由确定性后端控制。它不是一个可以绕过后端边界、自己随意改数据库和发送消息的完全自治 Agent。

代码里的 `mahjong_agent` 包名和 `AgentRuntime` 等类名暂时保留为历史实现名，后续可以做无破坏兼容迁移；对外产品名和仓库名建议使用 `mahjong-ops-workflow`。

## Agent Runtime V2

项目已经新增一条独立 V2 主链路，用来重做“模型自主决策工具”的 Agent 版本。V2 不复用旧 parser、旧 workflow、旧 guard 作为主执行链路；旧系统只作为业务参考。

V2 原则：

- LLM 负责理解用户、判断目标、决定调用哪些工具。
- 后端负责工具 schema 校验、权限、幂等、状态机、并发、预算、日志审计。
- 不再用业务 if-else 修麻将语义。
- 每一次模型输入、模型输出、工具调用、工具结果、状态变化都可追溯。
- 回复不对进入 eval/badcase，不直接硬编码修一句话。
- 同一会话串行处理，同一 `message_id` 重复进入时走消息结果账本，不重复调用模型或执行工具。
- 工具 contract 提供结构化组局条件和客户可见文案校验；参数错误会作为 tool result 回到模型，由模型修正，而不是后端补业务语义 if-else。
- `StatePolicyV2` 负责局和邀约草稿的状态合法性，模型不能绕过状态机直接落库。
- `ContextPackingPolicyV2` 负责上下文预算和裁剪审计，避免多轮对话无限塞进模型窗口。
- LLM 调用失败会记录 `llm_error` 并中断本轮工具执行，返回人工兜底回复。

V2 文档见 [docs/agent_runtime_v2.md](docs/agent_runtime_v2.md)。

V2 状态默认写入 `data/agent_runtime_v2.sqlite3`，trace 默认写入 `logs/agent_runtime_v2_trace.jsonl`，badcase 默认写入 `eval/badcases/agent_runtime_v2_badcases.jsonl`。

本地启动：

```bash
set -a
source .env
set +a
/Users/wangjie/Documents/Codex/tools/miniforge3/bin/python scripts/run_agent_v2_app.py
```

默认地址：

```text
http://127.0.0.1:8790/
```

## 解决的问题

线下麻将馆的运营本质上是一个高频、多线程、强上下文的人力调度问题：

- 群聊和私聊里有人用各种口语、缩写、图片、语音表达想打麻将。
- 同时存在多个局，时间、玩法、档位、人数、烟否、时长都不同。
- 老板需要判断哪些局可以合并，哪些人适合邀请，哪些人今天已经打过不能再打扰。
- 用户会报名、取消、说满了、临时改时间，所有状态都要同步。
- 单个客户不能被重复拉进多个有效局。
- 线上系统必须幂等、保序、可审计，不能因为节点重启丢消息。

本项目把这些经验沉淀为结构化状态机、规则解析、生产级上下文处理器、LLM 语义动作提案、客户推荐和可靠运行时。

## 核心能力

- 组局意图识别：识别找人、报名、取消、满员、潜在咨询、无关消息。
- 玩法解析：支持杭麻/财敲、川麻、幺鸡、红中、捉鸡、湖南麻将等本地玩法体系。
- 口语缩写解析：例如 `cq371` = 杭麻财敲三缺一，`川麻216` = 川麻 2-16 档。
- 底注/封顶结构化：`216` 解析为底注 2、封顶 16。
- 待组局队列：信息明确但缺时间时，先入队列并继续追问，不乱发邀约。
- 房态约束：如果目标开局时间满房，会建议最快可用时间并暂停邀约，避免承诺不存在的房间。
- 客户画像：记录客户偏好玩法、档位、无烟偏好、常见同行人数、常打时段、疲劳度和打扰频率。
- 候选人推荐：结合玩法偏好、档位、烟否、活跃时段、疲劳度和客户锁推荐人选。
- 局隔离和客户锁：防止客户被重复拉入多个有效局。
- 生命周期管理：自动处理待确认、开放、邀约中、占位、已确认、已完成、已过期、已取消。
- 草稿生成：生成群发和私聊邀约草稿，默认进入人工确认 outbox。
- 碎片消息聚合：同一用户短时间连发“老板 / 今天下午 / 有没有打麻将的 / 0.5或者1都行”等碎片时，会合并理解后再追问；如果老客户画像里高置信记录为通常一个人来，会按 `173` 处理而不是重复问人数。
- 生产级上下文处理器：每次 LLM 调用前统一裁剪、脱敏、组装、预算控制和记录上下文 digest。
- LLM 语义动作提案：结合当前消息、原邀约、当前局、客户画像和历史对话，输出 `semantic_type`、`proposed_action`、置信度、回复草稿和简短原因。
- 后端状态校验：后端校验 LLM 提案的动作白名单、置信度、状态机合法性、局是否已满、候选人是否属于该局 outbox，再决定是否落库。
- 显式状态机：后端维护 `state_machine.v1`，局状态、候选邀约状态和 followup 状态都必须按迁移表推进；终态局不能被重开，已确认邀约不能倒退为未回复。
- 状态迁移账本：局、outbox 和 followup 的状态推进会写入 `state_transition_events`，保留实体、事件、原状态、目标状态、原因和 metadata，支持按实体回放。
- 工具注册中心：后端维护 `tool_registry.v1`，按阶段只向 LLM 注入可用工具和参数 schema；非法工具、非法参数和直接发送请求会被 Tool Gateway 拒绝、剔除或降级。
- 受控动作协议：主流程工具调用和关键状态写入已经使用 `controlled_agent.v1`，每轮返回 `agent_actions`，记录动作提案、校验结果、风险等级、审批要求和幂等键。
- 工具规划 fail-closed：已配置 LLM 时，如果工具规划超时、预算拒绝或返回空计划，后端只允许保留只读工具，不会用 fallback 继续创建待审批外发草稿。
- 动作账本：只读搜索工具、关键状态写入和待审批消息草稿创建都会进入 SQLite `controlled_actions` 持久化账本，记录 `executing/executed/rejected/failed` 状态和执行结果。
- 审批请求账本：候选人邀约 outbox 和发起人 followup 草稿会同步写入 `approval_requests`，保留审批人、审批时间、最终文案、拒绝原因和关联动作 ID；审批通过只代表草稿可发，不等于已经真实发送。
- 发送执行账本：审批通过后的发送必须走 `/api/send-outbox`，写入 `message_delivery_attempts`；发送幂等键不依赖 traceId，重复点击或网络重试不会重复发送。
- 运行时安全策略：后端维护 `runtime_policy.v1`，支持只读模式、禁止状态写入、禁止发送、禁止审批、禁止 eval 写入，以及生产模式下要求副作用工具和业务状态写入必须由 LLM/人工明确提案；策略更新本身也进入受控动作账本，副作用工具和候选人反馈等状态写入会在执行前被策略门禁拦截。
- Trace 回放账本：输入、输出、LLM 请求/响应、工具请求/响应会同步写入 `trace_events`，可通过 `/api/traces?trace_id=...` 按链路回放。
- 候选人反馈受控写入：候选人回复会先形成 `record_candidate_feedback` 动作，后端归一化和校验后才写入状态；协商类回复只创建待审批 followup。
- 工作流写入受控：自动建局、老板手动建局、手动反馈、客户画像更新、清空当前局看板都会经过持久化执行门禁；同一幂等键重复提交不会重复落库，清空看板属于高风险批量归档动作。
- 高风险动作拦截：`send_message` 当前只能创建待审批 outbox/followup，即使模型请求直接发送，也会被后端降级；草稿创建本身也经过账本幂等门禁。
- 可观测运行时：结构化日志、指标、审计事件、上下文快照、异常和超时 fail-closed。
- 可靠处理：平台消息幂等、短窗口语义去重、会话保序、SQLite 持久化、outbox 幂等键、状态快照。

完整功能矩阵见 [docs/product_capabilities.md](docs/product_capabilities.md)。
真实微信聊天记录复盘见 [docs/wechat_record_analysis.md](docs/wechat_record_analysis.md)。
模型评估、训练与持续改进方案见 [docs/agent_training_and_improvement.md](docs/agent_training_and_improvement.md)。

## 生产架构

```mermaid
flowchart LR
    A["微信/企微/小程序/客服 Adapter"] --> B["Message 标准化"]
    B --> C["InputGate 幂等/去重/保序"]
    C --> D["ContextBuilder 上下文构建"]
    D --> E["SemanticResolver LLM语义contract"]
    E --> F["ProposedAction 动作提案"]
    F --> G["ActionValidator 后端校验"]
    G --> H["ToolOrchestrator 工具编排"]
    H --> I["StateMachine 状态机落库"]
    I --> J["ReplyPolicy 回复生成"]
    J --> K["ReplyGuard 安全闸"]
    K --> L["Pending Outbox 待老板审批"]
    L --> M["Trace / Eval / Audit"]
```

生产级架构说明见 [docs/production_architecture.md](docs/production_architecture.md)。
当前默认老板试用台入口对应的受控链路说明见 [docs/architecture_refactor_plan.md](docs/architecture_refactor_plan.md)。

## LLM 接入

系统会把 LLM 放在受控的语义动作提案位置：模型可以根据上下文提出意图、槽位、工具调用建议、候选人反馈动作和回复草稿；后端负责校验、落库、幂等、权限、预算和审计。规则解析仍用于低成本结构化抽取和无模型降级，但不再要求后端用大量 `if-else` 覆盖所有自然语言。

新的受控工作流统一从 `build_controlled_runtime()` 组装：

```python
from mahjong_agent import build_controlled_runtime

runtime = build_controlled_runtime()
result = runtime.service.handle_message(message)
```

这条入口会串起 `ContextBuilder -> SemanticResolver -> ActionValidator -> ToolOrchestrator -> StateMachine -> ReplyPolicy -> ReplyGuard`，并把 trace 写入 JSONL。默认 trace 路径是 `logs/controlled_workflow_trace.jsonl`，可用环境变量覆盖：

```bash
export MAHJONG_TRACE_JSONL_PATH="logs/controlled_workflow_trace.jsonl"
```

如果没有配置 LLM，受控运行时默认 fail-closed：不继续编造动作，而是转人工并写 trace。

最小配置：

```bash
export MAHJONG_LLM_API_KEY="your-api-key"
export MAHJONG_LLM_MODEL="your-model"
```

使用通义千问/阿里云百炼：

```bash
export DASHSCOPE_API_KEY="your-dashscope-api-key"
export MAHJONG_LLM_MODEL="qwen-plus"
export MAHJONG_LLM_TIMEOUT_SECONDS=60
```

设置 `DASHSCOPE_API_KEY` 时，系统会自动使用 `qwen` provider。你也可以显式设置 `MAHJONG_LLM_PROVIDER=qwen` 和 `MAHJONG_LLM_API_KEY`。

`MAHJONG_LLM_PROVIDER=qwen` 会默认使用 OpenAI-compatible endpoint：

```text
https://dashscope.aliyuncs.com/compatible-mode/v1
```

使用其他 OpenAI-compatible 服务：

```bash
export MAHJONG_LLM_BASE_URL="https://your-provider.example/v1"
```

使用 GLM / Z.ai / BigModel：

```bash
export MAHJONG_LLM_PROVIDER="zai"
export MAHJONG_LLM_API_KEY="your-api-key"
export MAHJONG_LLM_MODEL="glm-4.7-flash"
```

`MAHJONG_LLM_PROVIDER=zai` 默认使用：

```text
https://api.z.ai/api/paas/v4
```

使用 DeepSeek：

```bash
export MAHJONG_LLM_PROVIDER="deepseek"
export MAHJONG_LLM_API_KEY="your-api-key"
export MAHJONG_LLM_MODEL="deepseek-v4-flash"
```

`MAHJONG_LLM_PROVIDER=deepseek` 默认使用：

```text
https://api.deepseek.com
```

并默认关闭 thinking、开启 JSON output，适合当前这种结构化语义解析。

### LLM 预算管理

模型调用不是无上限资源。系统会在每次 LLM 调用前做预算预留，调用后用 provider 返回的 usage 回填真实 token；如果预算不足，会停止模型调用并 fail-closed 转人工或走规则降级。

常用配置：

```bash
export MAHJONG_LLM_MAX_CALLS_PER_DAY=1000
export MAHJONG_LLM_MAX_TOKENS_PER_DAY=200000
export MAHJONG_LLM_MAX_TOKENS_PER_CALL=16000
export MAHJONG_LLM_MAX_COMPLETION_TOKENS=512
```

如果要按金额控制，还需要配置模型价格。单位是每 1000 token 的价格：

```bash
export MAHJONG_LLM_INPUT_PRICE_PER_1K=0.001
export MAHJONG_LLM_OUTPUT_PRICE_PER_1K=0.002
export MAHJONG_LLM_MAX_COST_PER_DAY=5
```

当前预算管理是单进程内存版，适合本地试用和单机部署。多进程/分布式生产环境要把预算计数器迁移到 Redis 或数据库，并按 `tenant_id + 日期` 做原子扣减。

### 全自动 Agent Runtime

本地试用可以开启新的目标驱动 Agent Loop：

```bash
export MAHJONG_AUTONOMOUS_AGENT_ENABLED=1
```

开启后，控制台输入不再走“语义解析 + 后端阶段推进”的旧链路，而是由 LLM 在每一轮决定下一步是调用工具还是回复用户。后端只负责工具 schema、权限、幂等、状态机、审计和高风险动作边界。页面返回里的 `workflow.engine` 会显示 `autonomous_agent.v1`。

### 判断模型是否够用

真实开发流程里，不靠主观感觉判断模型是否好用，而是按任务和指标验收：

1. 明确模型职责：本项目里 LLM 负责语义理解、动作提案、工具选择建议、归一化和草稿生成，不直接改状态、不直接发消息。
2. 建立 golden dataset：用 [eval/golden/scenario_golden.jsonl](eval/golden/scenario_golden.jsonl) 覆盖明确组局、弱意图、模糊时间、玩法行话、多模态转写、敏感资金、报名接受等关键场景。
3. 离线跑评估：比较不同模型的意图准确率、字段抽取准确率、JSON 合法率、转人工率、平均延迟和平均成本。
4. 试用阶段跑影子模式：模型只给建议，不影响真实发送；人工结果回写到 [eval/badcases/badcases.jsonl](eval/badcases/badcases.jsonl)。
5. 设上线阈值：例如高风险场景 100% 转人工、JSON 合法率 > 99%、关键字段准确率 > 95%、P95 延迟在可接受范围内、单店日成本不超过预算。
6. 持续迭代：高频稳定错误进规则/词典，模糊表达进 prompt/golden，低频复杂问题保留人工审核。

所以 GLM-4.7-Flash、Qwen 或其他模型是否够用，最终要以同一套评估集和预算数据对比，而不是只看榜单。

测试 LLM 是否能真实调用：

```bash
PYTHONPATH=src python scripts/run_llm_smoke_test.py
```

`MAHJONG_LLM_TIMEOUT_SECONDS` 是单次模型请求超时。控制台、聊天室和 API 服务的外层 workflow 超时会默认读取它并额外增加 5 秒缓冲；也可以通过 `MAHJONG_AGENT_TIMEOUT_SECONDS` 单独覆盖。

LLM 只允许输出结构化 JSON 或规范化文本，不能直接改状态、不能直接发消息、不能直接占位。所有状态变更仍由核心状态机完成。

## API 入口

生产系统可以把任意上游消息通道转换为统一请求：

```bash
curl -X POST http://127.0.0.1:8787/respond \
  -H "Content-Type: application/json" \
  -d '{
    "text": "cq371 0.5 19.30 无烟",
    "sender_id": "u_123",
    "sender_name": "张哥",
    "channel_id": "group_main",
    "channel_type": "wechat_group",
    "source_message_id": "wechat-msg-001",
    "sequence": 1,
    "now": "2026-06-16T18:00:00+08:00"
  }'
```

返回值是结构化 `ReplyDecision`，包含：

- `action`：本轮动作，如追问、建局、生成邀约、接受报名、转人工、静默。
- `reply_text`：给当前用户的回复。
- `draft_group_post`：群发草稿。
- `invitation_drafts`：私聊邀约草稿。
- `needs_human_review`：是否需要人工确认。
- `notes`：决策证据、LLM 解释、风险提示。

## 控制台输入 + 微信测试输出

在不依赖个人微信实时读消息的情况下，可以先用控制台作为输入通道，用测试路由把所有输出改写到自己的微信号 `radon_1`：

```bash
PYTHONPATH=src python scripts/run_console_agent.py \
  --reset \
  --dispatch wechat-test \
  --output-channel wechat \
  --test-recipient radon_1
```

如果要让控制台输入真的走 LLM，把 LLM 环境变量放在同一个 shell 里再启动；千问等较慢模型建议设置 30-60 秒超时：

```bash
export DASHSCOPE_API_KEY="your-dashscope-api-key"
export MAHJONG_LLM_MODEL="qwen3.7-plus"
export MAHJONG_LLM_TIMEOUT_SECONDS=60

PYTHONPATH=src python scripts/run_console_agent.py \
  --reset \
  --dispatch wechat-test \
  --output-channel wechat \
  --test-recipient radon_1
```

运行后在终端输入真实群消息即可。Workflow 会按完整链路处理：

```text
控制台输入 -> IncomingEnvelope -> DurableProcessor -> AgentRuntime
-> AgentResponder -> outbox_events -> OutputRouter -> radon_1
```

注意：这个控制台脚本仍是早期 adapter/输出路由验证工具，适合验证“输入输出解耦”和 Mac 微信测试发送；老板试用台 `/api/analyze` 默认已经走 `ControlledWorkflowService` 受控链路。后续会把控制台入口也迁移到同一条受控工作流。

默认是 dry-run，只会在终端打印将要发送到 `radon_1` 的内容。若要调用 Mac 微信 UI 自动化发送，需要显式开启：

```bash
PYTHONPATH=src python scripts/run_console_agent.py \
  --dispatch wechat-test \
  --output-channel wechat \
  --test-recipient radon_1 \
  --wechat-live-send
```

实际发送脚本是 [scripts/send_wechat_mac.py](scripts/send_wechat_mac.py)。它依赖 macOS Accessibility 权限和微信客户端 UI，适合测试执行器，不适合作为长期唯一生产通道。

当前输入和输出已经解耦：控制台、API、微信、企微、小红书、抖音都应被视为 adapter。核心引擎只处理标准 `Message` 和 `outbox_events`，不绑定具体平台。

## 运行方式

启动老板试用 Web 台：

```bash
PYTHONPATH=src python scripts/run_boss_trial_app.py
```

打开：

```text
http://127.0.0.1:8792
```

这个入口面向麻将馆老板试用，不追求完整平台能力。第一版只做可操作闭环：

- 客户画像维护，内置 50 个模拟常客，可手动新增/编辑。
- 粘贴客户消息并解析玩法、时间、档位、人数、烟况。
- 信息不全时生成追问建议。
- 按规则推荐候选人，并解释推荐原因。
- 为每个候选人生成私聊邀约草稿。
- 发件箱支持复制话术、标记已发送、拒绝、未回复、已确认、已到店、别再打扰。
- 当前局看板展示已邀请谁、谁回复了、还缺几人、是否成局。
- 今日复盘展示识别/邀约/成局情况和少打扰建议。

试用台默认不接微信、不自动发送、不自动确认房间、不处理资金和纠纷。所有外发只生成草稿，由老板复制粘贴或手动确认。

### 本机 Redis 短期缓存

本机 Redis 用于短期记忆和组局缓存，SQLite 仍然是长期事实库。Redis 不可用时，试用台会自动降级为只使用 SQLite。

当前本机安装位置：

```bash
~/.local/redis/bin/redis-server
~/.local/redis/bin/redis-cli
```

启动 Redis：

```bash
~/.local/redis/bin/redis-server ~/.local/redis/redis.conf
```

验证 Redis：

```bash
~/.local/redis/bin/redis-cli -h 127.0.0.1 -p 6379 ping
```

正常返回：

```text
PONG
```

停止 Redis：

```bash
~/.local/redis/bin/redis-cli -h 127.0.0.1 -p 6379 shutdown
```

试用台默认读取：

```bash
export MAHJONG_REDIS_URL="redis://127.0.0.1:6379/0"
```

Redis 当前缓存三类信息：

- `mahjong:trial:state`：最新页面状态快照，默认 5 分钟过期。
- `mahjong:trial:sender:<sender_id>:memory`：同一客户近期碎片消息短期记忆，默认 2 小时过期。
- `mahjong:trial:game:<game_id>`：当前局快照和邀约草稿，默认 24 小时过期。

### 输入输出日志

老板试用台会记录输入和输出日志，路径：

```text
logs/boss_trial_io.log
```

日志格式固定为：

```text
traceId-time(yyyy-mm-dd hh:mm:ss)-loglevel: content
```

实际示例：

```text
trace_20260627103005_ab12cd34-2026-06-27 10:30:05-INFO: {"direction":"input","path":"/api/analyze","sender_id":"zhang","sender_name":"张哥","text":"下午两点 0.5 无烟杭麻，帮我组一桌"}
trace_20260627103005_ab12cd34-2026-06-27 10:30:05-INFO: {"direction":"output","path":"/api/analyze","action":"create_game","missing_fields":[],"candidate_count":8,"outbox_count":8,"used_short_memory":false}
```

同一次请求的输入、输出、异常都使用同一个 `traceId`。日志内容只记录关键字段和摘要，不把完整客户画像列表全部打进去。

启动 API 服务：

```bash
PYTHONPATH=src python scripts/run_agent_server.py
```

开发和验收时可以启动本地对话验证工具：

```bash
PYTHONPATH=src python scripts/run_chatroom.py
```

本地对话验证工具仅用于开发和质量验收，不是生产架构本身。

## 测试

安装标准测试工具：

```bash
python -m pip install -e ".[dev]"
```

标准 pytest 入口：

```bash
python -m pytest -q
```

项目自带 runner 和场景评估：

```bash
PYTHONPATH=src python scripts/run_tests.py
PYTHONPATH=src python scripts/run_scenario_eval.py
```

受控 Agent 本地验收门禁会串起密钥扫描、语法检查、离线 pytest、场景评估和项目自带 runner：

```bash
PYTHONPATH=src python scripts/run_controlled_agent_acceptance.py
```

本地试用默认是 `trial` 策略，允许规则兜底生成待审批草稿。生产受控模式建议显式打开：

```bash
export MAHJONG_CONTROLLED_AGENT_MODE=production
```

生产模式会默认打开：

```text
llm_required_for_side_effect_tools=true
llm_required_for_state_writes=true
```

这意味着没有 LLM 或人工动作提案时，后端只能追问、转人工、写审计和执行只读搜索，不能自动建局、确认候选人或创建待审批外发草稿。

DeepSeek 集成测试是显式开启的真实模型调用。普通 `pytest` 和普通 `run_tests.py` 不会联网、不消耗模型预算；只有下面命令会强制调用 DeepSeek：

```bash
export MAHJONG_DEEPSEEK_API_KEY="your-deepseek-api-key"
export MAHJONG_RUN_DEEPSEEK_INTEGRATION=1
python -m pytest -q -m integration
PYTHONPATH=src python scripts/run_deepseek_integration_test.py
PYTHONPATH=src python scripts/run_tests.py --with-deepseek
PYTHONPATH=src python scripts/run_controlled_agent_acceptance.py --with-deepseek
```

其中 `python -m pytest -q -m integration` 是 pytest 集成测试入口；如果没有设置 `MAHJONG_RUN_DEEPSEEK_INTEGRATION=1`，该测试会跳过，避免普通离线回归误调真实模型。

DeepSeek 集成测试会强制使用：

```text
provider=deepseek
model=deepseek-v4-flash
base_url=https://api.deepseek.com
```

它只做语义解析 smoke test，不会写业务数据库、不会创建 outbox、不会发送消息。测试必须拿到 provider 返回的 token usage，否则不能证明真实模型响应成功。调用日志写入 `logs/deepseek_integration.log`，API key 会脱敏，不会出现在日志里。可以用这些环境变量控制成本和超时：

```bash
export MAHJONG_DEEPSEEK_TIMEOUT_SECONDS=30
export MAHJONG_DEEPSEEK_MAX_CALLS=2
export MAHJONG_DEEPSEEK_MAX_TOKENS=12000
export MAHJONG_DEEPSEEK_MAX_COST=1
```

场景评估现在读取 [eval/golden/scenario_golden.jsonl](eval/golden/scenario_golden.jsonl)。它是稳定回归集，失败代表行为回归或预期需要评审。

测试、试用或真实运营中发现的新问题先写入 [eval/badcases/badcases.jsonl](eval/badcases/badcases.jsonl)。修复确认后，再把它整理进 golden dataset 或 `eval/regression/`。

手动追加样本：

```bash
PYTHONPATH=src python scripts/record_eval_case.py \
  --dataset badcase \
  --id weak_intent_new_001 \
  --name "新的弱意图表达" \
  --text "晚点有没有人搓一把" \
  --sender-id eval_user \
  --expected-action ask_clarification \
  --contains "帮你看看"
```

运行评估并把失败样本追加到 badcase：

```bash
PYTHONPATH=src python scripts/run_scenario_eval.py --record-failures
```

当前覆盖：

- 玩法解析和本地行话
- 模糊时间追问
- 待组局队列
- 满房时间协商
- 玩法偏好推荐
- 客户疲劳度
- 客户锁
- 邀约接受和废弃
- 局生命周期
- 多模态 metadata
- 敏感词转人工
- 运行时异常和超时
- 持久化幂等和会话保序
- LLM 兜底路径

## 当前实现状态

这是一个可运行的核心引擎版本，已经具备生产系统中最关键的领域建模和可靠性骨架。

已经完成：

- 规则解析和结构化状态机。
- 客户画像和推荐。
- 待组局队列。
- outbox 草稿。
- 输入/输出 adapter 边界。
- 控制台输入 runner。
- OutputRouter 和微信测试输出到 `radon_1`。
- ContextBuilder 上下文处理器。
- LLM 接入点。
- 运行时保护。
- SQLite 持久化、幂等、保序、审计。

尚未完成：

- 真实微信/企业微信/小红书/抖音生产级消息收发 adapter。
- 生产数据库迁移。
- 多租户权限和后台管理。
- 真实图片 OCR / 语音 ASR 管线。
- 网页搜索和外部工具注册中心。
- LLM token 成本统计。
- 商业化管理后台。

项目路线图见 [docs/roadmap.md](docs/roadmap.md)。

## 适合的商业化形态

- 单店麻将馆 AI 运营助理。
- 多店连锁棋牌室运营中台。
- SaaS 化客户画像和组局调度系统。
- 微信群/私域流量运营 Workflow。
- 面向地方玩法的垂直行业 agentic workflow 模板。

核心商业价值在于减少老板/店员反复拉人、确认、追问、记偏好的人工成本，同时提高组局成功率和客户触达质量。
