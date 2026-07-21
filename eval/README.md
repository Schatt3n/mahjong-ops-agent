# 评估集、Badcase 和 Golden Dataset

这个目录只保留当前 `mahjong_agent_runtime` 主链路相关的质量资产。

## 目录分工

- `badcases/badcases.jsonl`：当前 Agent Runtime 在测试、试用或真实微信灰度中发现的失败样本和边界样本。
- `regression/agent_runtime_regression.jsonl`：当前主链路专项回归集，验证“模型负责目标理解和工具规划，后端负责 schema、权限、状态、幂等和审计”。
- `golden/real_owner_chat_golden.jsonl`：真实老板聊天转写出的长对话 golden dataset，用于验证闲聊和业务组局穿插时上下文仍能接回。
- `golden/real_owner_chat_transcript_20260704.md`：真实聊天截图的可读转写。
- `golden/fragmented_input_golden.jsonl`：碎片化输入边界样本，验证等待、超时重触发、发送者隔离和聚合后一次处理。
- `adversarial/privacy_isolation.jsonl`：跨会话隐私对抗样本，一条一个攻击 case，与执行脚本解耦。
- `few_shot_examples.jsonl`：老板认可的话术样例，用于改善后续回复风格。
- `tests/test_context_summary_quality.py`：上下文压缩决策一致性评测，比较完整历史与 checkpoint 路径的状态、工具调用和回复。

## 写入规则

- 发现当前主链路回复、工具调用或状态推进有问题，先写入 `badcases/badcases.jsonl`。
- 修复 badcase 后，必须补 `regression_refs`，指向 `agent_runtime_regression`、`real_owner_chat_golden`、`live_eval` 或具体 `pytest` 用例。
- 老板确认某句回复“像我会说的话”，写入 `few_shot_examples.jsonl`。
- 真实长对话样本优先沉淀到 `real_owner_chat_golden.jsonl`，用于验证上下文、闲聊分流和多轮恢复。
- 用户把一句需求拆成多条发送时，写入 `fragmented_input_golden.jsonl`；不能靠新增麻将关键词 `if-else` 修复，应由输入边界模型和通用并发合同解决。
- Agent 重复调用同一工具、在短周期动作间来回切换或连续没有状态/信息进展时，必须补 `ProgressMonitor` 回归；检测器只比较动作、结果和状态变化，不写麻将业务 `if-else`。
- 修改摘要提示词、checkpoint 合同或上下文裁剪逻辑时，必须运行压缩决策一致性评测；不能只断言摘要已生成。
- 越权询问、提示词注入、格式诱导和间接暗示等攻击必须写入 `adversarial/`；仅使用合成身份和 canary，不提交真实客户私聊。

## 运行评估

运行当前主链路默认评估：

```bash
PYTHONPATH=src python scripts/run_evals.py
```

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

`scripts/check_badcase_regression_coverage.py` 会审计所有 fixed badcase 是否已经闭环到当前回归资产，避免问题只停留在日志里。
