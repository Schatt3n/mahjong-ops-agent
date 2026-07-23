# 评估集、Badcase 和 Golden Dataset

这个目录只保留当前 `mahjong_agent_runtime` 主链路相关的质量资产。

## 目录分工

- `badcases/badcases.jsonl`：当前 Agent Runtime 在测试、试用或真实微信灰度中发现的失败样本和边界样本。
- `regression/agent_runtime_regression.jsonl`：当前主链路专项回归集，验证“模型负责目标理解和工具规划，后端负责 schema、权限、状态、幂等和审计”。
- `golden/real_owner_chat_golden.jsonl`：真实老板聊天转写出的长对话 golden dataset，用于验证闲聊和业务组局穿插时上下文仍能接回。
- `golden/real_owner_chat_transcript_20260704.md`：真实聊天截图的可读转写。
- `golden/real_owner_chat_timeline_20260705_20260718.json`：12 张真实老板聊天截图清洗出的多日机器可读时间线，保留 86 条文本、7 个业务片段、时间、终态、业务轮次、事实、评测 case 和回放脚本；表情包及图片已排除。
- `golden/real_owner_chat_transcript_20260705_20260718.md`：上述多日时间线的可读转写，说明简单/复杂场景所需轮次、取消边界和推荐测试输入。
- `golden/fragmented_input_golden.jsonl`：碎片化输入边界样本，验证等待、超时重触发、发送者隔离和聚合后一次处理。
- `golden/real_group_chat_20260722.jsonl`：真实微信群观察数据清洗出的高置信群聊 Gold，覆盖看板解析、状态演进、碎片聚合、引用更新、撤回和确定性噪声过滤。
- `adversarial/privacy_isolation.jsonl`：跨会话隐私对抗样本，一条一个攻击 case，与执行脚本解耦。
- `adversarial/real_group_chat_20260722.jsonl`：真实群聊中的指代、跨角色多轮和麻将黑话歧义样本；在领域问题确认前只能作为待复核对抗集，不能作为硬真值。
- `adversarial/agent_planning_discoveries.jsonl`：真实模型模拟中发现、尚未闭环的规划失败样本；保留输入、工具轨迹、来源 trace 和可接受结果，供后续规划/无进展评测复现。
- `few_shot_examples.jsonl`：老板认可的话术样例，用于改善后续回复风格。
- `tests/test_context_summary_quality.py`：上下文压缩决策一致性评测，比较完整历史与 checkpoint 路径的状态、工具调用和回复。

## 写入规则

- 发现当前主链路回复、工具调用或状态推进有问题，先写入 `badcases/badcases.jsonl`。
- 修复 badcase 后，必须补 `regression_refs`，指向 `agent_runtime_regression`、`real_owner_chat_golden`、`live_eval` 或具体 `pytest` 用例。
- 老板确认某句回复“像我会说的话”，写入 `few_shot_examples.jsonl`。
- 真实长对话样本优先沉淀到 `real_owner_chat_golden.jsonl`，用于验证上下文、闲聊分流和多轮恢复。
- 多日真实私聊必须按独立业务目标切分为 episode，并保存原始时间顺序。业务轮次按“一方围绕同一动作的连续消息 + 另一方连续回应”计数，不能把老板连续发送的多个短气泡误算为多个完整轮次。
- 轮次统计只能描述当前样本，不能从少量成功片段推导稳定成局率或平均耗时；自动回归应优先断言少追问、状态正确、取消生效、参与者不混淆和工具闭环。
- 真实群聊数据必须先匿名化：房间和发送者使用别名，源消息仅保留截断 SHA-256；禁止提交微信 ID、昵称、头像、签名、完整 payload 和现有分类器标签。
- 只有业务含义及期望动作都明确的群聊样本才能进入 Gold；“那个一”“3块是否等于368”等仍需老板确认的样本进入 Adversarial，并保留 `open_questions`。
- 用户把一句需求拆成多条发送时，写入 `fragmented_input_golden.jsonl`；不能靠新增麻将关键词 `if-else` 修复，应由输入边界模型和通用并发合同解决。
- Agent 重复调用同一工具、在短周期动作间来回切换或连续没有状态/信息进展时，必须补 `ProgressMonitor` 回归；检测器只比较动作、结果和状态变化，不写麻将业务 `if-else`。
- 真实模拟中首次发现但尚未稳定修复的规划问题先写入 `agent_planning_discoveries.jsonl`，状态保持 `open`；只有重复复现通过并建立自动回归后，才迁入 fixed badcase 闭环。
- 修改摘要提示词、checkpoint 合同或上下文裁剪逻辑时，必须运行压缩决策一致性评测；不能只断言摘要已生成。
- 越权询问、提示词注入、格式诱导和间接暗示等攻击必须写入 `adversarial/`；仅使用合成身份和 canary，不提交真实客户私聊。

## 运行评估

运行当前主链路默认评估：

```bash
PYTHONPATH=src python scripts/run_evals.py
```

单独校验真实群聊数据的 JSON 合同、匿名化和分层状态：

```bash
python scripts/validate_real_group_chat_dataset.py
PYTHONPATH=src python -m pytest -q tests/test_real_group_chat_dataset.py
```

校验真实老板私聊的基础 JSONL 与多日时间线：

```bash
PYTHONPATH=src python scripts/validate_real_owner_chat_golden.py
PYTHONPATH=src python -m pytest -q \
  tests/test_real_owner_chat_golden.py \
  tests/test_real_owner_chat_golden_validator.py \
  tests/test_real_owner_chat_timeline.py
```

校验器只负责数据格式与匿名化，不能证明 Agent 能正确处理样本。真实群聊 Gold 还必须回放到实际的 `OwnerMessageParser`、`QuickFilter`、`MessageAccumulator`、`SessionRouter` 和 `GroupSessionClassifier`：

```bash
# 不调用模型：看板解析、状态演进、引用、撤回和噪声过滤
PYTHONPATH=src python scripts/run_real_group_chat_flow_eval.py --strict

# 调用当前配置的模型，并把语义样本继续送入主 Agent 验证工具选择
PYTHONPATH=src python scripts/run_real_group_chat_flow_eval.py --live --strict \
  --report-path runtime_data/real_group_chat_flow_eval_live.json
```

报告逐 case 保存 `expected`、`actual`、差异、模型调用次数、模型耗时、主 Agent 工具调用和 trace 步骤。`cq` 规范化为“杭麻/财敲”，“川麻换三”规范化为“川麻/换三张”；`568/368/132` 等三位码在场馆确认规则前仅保存在 `rule_code`，不得推断成底注、封顶或番数。

单独运行上下文压缩质量评测：

```bash
PYTHONPATH=src python -m pytest -q tests/test_context_summary_quality.py
```

该评测覆盖失败历史、临时约束、子任务续接和工具结果依赖，并包含摘要漏约束的反向对照。二进制核心指标为 `DecisionConsistencyReport.consistent`；差异会写入 `context_summary_decision_consistency` trace，便于定位状态、工具参数或回复在哪一项发生漂移。

运行真实模型 live 评估：

```bash
MAHJONG_LLM_PROVIDER=deepseek MAHJONG_LLM_MODEL=deepseek-v4-flash DEEPSEEK_API_KEY=*** PYTHONPATH=src python scripts/run_real_owner_chat_live_eval.py --strict
```

运行跨会话隐私对抗评测：

```bash
PYTHONPATH=src python scripts/run_privacy_isolation_live_eval.py --strict
```

评测器默认读取 `adversarial/privacy_isolation.jsonl`；可以用 `--case direct_reason` 单独调试某个 case，或用 `--cases-path` 切换到其他对抗数据集。

也可以把 live 评估接到主链路总评估里：

```bash
MAHJONG_LLM_PROVIDER=deepseek MAHJONG_LLM_MODEL=deepseek-v4-flash DEEPSEEK_API_KEY=*** PYTHONPATH=src python scripts/run_evals.py --live-real-owner
```

追加 `--live-real-group` 可把真实群聊语义样本也纳入总评估。

`scripts/check_badcase_regression_coverage.py` 会审计所有 fixed badcase 是否已经闭环到当前回归资产，避免问题只停留在日志里。
