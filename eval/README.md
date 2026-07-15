# 评估集、Badcase 和 Golden Dataset

这个目录只保留当前 `mahjong_agent_runtime` 主链路相关的质量资产。

## 目录分工

- `badcases/badcases.jsonl`：当前 Agent Runtime 在测试、试用或真实微信灰度中发现的失败样本和边界样本。
- `regression/agent_runtime_regression.jsonl`：当前主链路专项回归集，验证“模型负责目标理解和工具规划，后端负责 schema、权限、状态、幂等和审计”。
- `golden/real_owner_chat_golden.jsonl`：真实老板聊天转写出的长对话 golden dataset，用于验证闲聊和业务组局穿插时上下文仍能接回。
- `golden/real_owner_chat_transcript_20260704.md`：真实聊天截图的可读转写。
- `golden/fragmented_input_golden.jsonl`：碎片化输入边界样本，验证等待、超时重触发、发送者隔离和聚合后一次处理。
- `few_shot_examples.jsonl`：老板认可的话术样例，用于改善后续回复风格。

## 写入规则

- 发现当前主链路回复、工具调用或状态推进有问题，先写入 `badcases/badcases.jsonl`。
- 修复 badcase 后，必须补 `regression_refs`，指向 `agent_runtime_regression`、`real_owner_chat_golden`、`live_eval` 或具体 `pytest` 用例。
- 老板确认某句回复“像我会说的话”，写入 `few_shot_examples.jsonl`。
- 真实长对话样本优先沉淀到 `real_owner_chat_golden.jsonl`，用于验证上下文、闲聊分流和多轮恢复。
- 用户把一句需求拆成多条发送时，写入 `fragmented_input_golden.jsonl`；不能靠新增麻将关键词 `if-else` 修复，应由输入边界模型和通用并发合同解决。
- Agent 重复调用同一工具、在短周期动作间来回切换或连续没有状态/信息进展时，必须补 `ProgressMonitor` 回归；检测器只比较动作、结果和状态变化，不写麻将业务 `if-else`。

## 运行评估

运行当前主链路默认评估：

```bash
PYTHONPATH=src python scripts/run_evals.py
```

运行真实模型 live 评估：

```bash
MAHJONG_LLM_PROVIDER=deepseek MAHJONG_LLM_MODEL=deepseek-v4-flash DEEPSEEK_API_KEY=*** PYTHONPATH=src python scripts/run_real_owner_chat_live_eval.py --strict
```

也可以把 live 评估接到主链路总评估里：

```bash
MAHJONG_LLM_PROVIDER=deepseek MAHJONG_LLM_MODEL=deepseek-v4-flash DEEPSEEK_API_KEY=*** PYTHONPATH=src python scripts/run_evals.py --live-real-owner
```

`scripts/check_badcase_regression_coverage.py` 会审计所有 fixed badcase 是否已经闭环到当前回归资产，避免问题只停留在日志里。
