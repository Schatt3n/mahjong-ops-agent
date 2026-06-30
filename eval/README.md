# 评估集、Badcase 和 Golden Dataset

这个目录用于沉淀麻将馆运营 workflow 的长期质量资产。

## 目录分工

- `golden/scenario_golden.jsonl`：稳定、已确认正确的底层 workflow 回归评估集。每一条都应该长期通过，失败就代表系统行为发生了回归，或者预期需要重新评审。
- `golden/boss_trial_golden.jsonl`：试用台端到端评估集，覆盖页面建议回复、老板话术风格、候选推荐展示等不属于底层 `AgentResponder` 的行为。
- `badcases/badcases.jsonl`：测试、试用或真实运营中发现的失败样本、边界样本和争议样本。它是待处理队列，不默认作为发布阻塞条件。
- `regression/`：从 badcase 修复后沉淀出来的专项回归集。适合放“曾经线上/试用中明确失败过，修复后必须永远防回归”的样本。
- `regression/controlled_workflow_regression.jsonl`：受控工作流专项回归集，使用固定 semantic contract 验证 `ContextBuilder -> SemanticResolver -> ActionValidator -> ToolOrchestrator -> StateMachine -> ReplyPolicy -> ReplyGuard -> Trace`，不依赖真实 LLM 的随机输出。
- `few_shot_examples.jsonl`：老板认可的话术样例。运行试用台时会被动态读取，作为 LLM 起草回复的 few-shot examples，但它不等同于回归评估集。
- `../skills/mahjong_operations_skills.jsonl`：可复用的运营 skill。它描述“遇到某类场景应该怎么判断和行动”，会被动态注入语义解析、工具规划、回复起草和邀约草稿阶段。

## 什么时候写入

- 发现系统回复明显不对：先写入 `badcases/badcases.jsonl`。
- 发现真实老板表达方式很常见，但当前评估集没有覆盖：写入 `badcases/badcases.jsonl`，修复后提升到 `golden/scenario_golden.jsonl` 或 `regression/`。
- 修复一个 badcase 后：把稳定预期整理成 golden case，并保留原 badcase 的 `source_scenario_id` 或备注。
- 新增玩法、规则、通道、状态机能力时：同步补 golden case，避免后续改坏。
- 发现“页面建议回复/老板话术/候选展示”这类试用台问题：先写入 `badcases/badcases.jsonl`，修复后补到 `golden/boss_trial_golden.jsonl`，并让 `scripts/run_tests.py` 跑过。
- 老板确认某句回复“像我会说的话”：写入 `few_shot_examples.jsonl`，用于改善后续回复风格。
- 发现某条规则不是单个样本，而是一类可复用运营经验：写入 `../skills/mahjong_operations_skills.jsonl`，例如“过期时间必须确认”“弱意图先查当前局池”。
- 试用台页面可以直接归档三类数据：点 `归档 badcase`、`加入 golden` 或 `采集 few-shot`，系统会保留 traceId，便于回查输入、提示词、模型输出和人工判断。

## JSONL 样本格式

单轮样本：

```json
{"schema_version":1,"kind":"golden","id":"weak_intent_001","name":"弱组局咨询追问","tags":["弱意图"],"text":"今天下班有人打麻将吗","sender_id":"passerby","expected":{"action":"ask_clarification","contains":"帮你看看能不能拼一桌"}}
```

多步样本：

```json
{"schema_version":1,"kind":"golden","id":"invitation_accept_001","name":"被邀请用户接受","steps":[{"name":"创建局","text":"今晚7点 0.5 三缺一 无烟","sender_id":"host"},{"name":"被邀用户确认","text":"我来","sender_id":"__first_invited_customer__","expected":{"action":"accept_seat","contains":"人数已齐"}}]}
```

few-shot 话术样本：

```json
{"schema_version":1,"kind":"few_shot","id":"trial_good_reply_001","customer_message":"张哥，下午有人打吗","parsed":"老客户，杭州默认杭麻，档位按常打 0.5 确认","reply_text":"张哥，下午我帮你按老样子0.5杭麻看下？你大概几点方便？"}
```

skill 样本：

```json
{"schema_version":1,"kind":"operation_skill","id":"time_ambiguity_guard","stages":["semantic_resolution","tool_planning","reply_draft"],"triggers":["两点","已经过了"],"instructions":["如果用户给出的时间早于当前时间，不能自动改成明天。"],"risk_controls":["时间未确认时禁止候选人搜索"]}
```

试用台端到端样本：

```json
{"schema_version":1,"kind":"boss_trial_golden","id":"clear_complete_request_concise_ack_001","input":{"sender_name":"张哥","sender_id":"zhang","text":"下午两点 0.5 无烟杭麻，帮我组一桌"},"expected":{"parsed_user_intent":"找人组局","suggested_reply_exact":"好的，我帮你问问。","forbidden_in_suggested_reply":["杭麻","0.5","两点","无烟"]}}
```

## 运行评估

```bash
PYTHONPATH=src python scripts/run_scenario_eval.py
```

运行受控工作流专项回归：

```bash
PYTHONPATH=src python scripts/run_controlled_workflow_eval.py
```

运行全部评估入口：

```bash
PYTHONPATH=src python scripts/run_evals.py
```

使用指定数据集：

```bash
PYTHONPATH=src python scripts/run_scenario_eval.py --dataset eval/golden/scenario_golden.jsonl
```

把失败样本自动追加到 badcase：

```bash
PYTHONPATH=src python scripts/run_scenario_eval.py --record-failures
```

试用台 golden 样本由项目测试入口执行：

```bash
PYTHONPATH=src python scripts/run_tests.py
```

## 质量规则

- golden case 必须脱敏，不放真实手机号、微信号、地址和完整聊天截图原文。
- golden case 要写稳定预期，不要写“当前实现刚好这样返回”的偶然文本。
- badcase 可以保留更多现场信息，但仍然要脱敏。
- 先覆盖高频、高风险、高价值场景：弱意图、明确组局、取消、改时间、满房、重复消息、客户锁、敏感资金、人工审核。
