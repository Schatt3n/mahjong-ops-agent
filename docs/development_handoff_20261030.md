# 2026-10-30 开发交接：在个人电脑继续验证

当前版本是用户明确要求先提交、推送的开发检查点，**完整门禁尚未通过，不能作为已验收版本**。用户原话：“先通过推送吧，把没有完成的任务存进去，我晚上回家用我自己的电脑跑”。这次提交例外不修改 `AGENTS.md` 的后续验收要求。

## 已实现的范围

- 主模型保留上一轮有效目标、计划与工具结果；用户放宽时间、给出范围或改变条件时重新判断，开放自然语言仍交给模型理解。
- 查询参数、己方人数与目标牌桌人数分别建模和校验；后端继续守住权限、幂等、人数、状态与隐私边界。
- 客户可见审查区分服务提议、承接责任和已执行声明；房态和有限复查使用真实工具与持久化证据。
- 补充输入先持久化并提升版本；过期运行在模型、工具、审查、摘要及最终输出边界停止。已提交的状态和已交付的历史保留。
- 本地 API、手动模拟面板支持处理中继续输入，处理重复请求、乱序响应和待发输出失效。
- 全量真实模型门禁加入独立语义判定；后端安全拒绝并纠正的动作合同错误单列计数，真实语义/权限/数据错误仍阻断。
- DeepSeek 配置支持 `enabled/high`；响应正文和分块读取执行绝对截止时间，网络重试共用总预算。

运行中修正是协作式的：当前模型调用返回后检查新版本，不是立即取消远端推理。交付保证目前覆盖本地 API/手动面板，未扩展为真实微信端到端自动外发保证。DNS、连接、TLS 和响应头阶段仍使用 urllib 原有超时机制。

## 最后一次验证结果

本机完整运行编号：`20261030T092732524329Z_x7rxf607`（Full18）。运行期间 348 个源码/资产文件及模型配置保持不变，完整门禁返回非零。

| 范围 | 结果 |
| --- | --- |
| 全量 pytest | 1854 passed，2 skipped |
| 离线评测、中文注释、边界、diff 检查 | 通过 |
| 老板对话与独立语义 | 34/34，通过；动作合同错误 0 |
| 群聊三组回放 | 12/12、4/4、7/7，通过；其中包含确定性分流场景 |
| 真实并发回放 | **66/68，失败 2** |
| 运行中补充输入并改变目标 | 通过 |
| 受控客户可见审查 | 14/14，通过 |
| 六类咨询 | 6/6，通过 |
| 隐私对抗 | **未完成**；按用户交接要求中止，保存了 4 条原始执行轨迹，不能计为整组通过 |

两个 pytest 跳过项是未配置 Redis 与未启用单独的可选 DeepSeek 冒烟测试；必须执行的真实模型集合另有上述结果。并发中 3 次动作合同错误均有完整证据证明安全恢复，未恢复错误 0；它们不是本轮两例失败的原因。共享运行客户端共 338 次模型调用，失败 0、HTTP 重试/超时/恢复均 0；这些指标不含各场景独立语义判定器，也不是供应商 token 账单。

## 待办一：闲聊被业务问句关键词误杀

- 场景：`casual_chat_should_not_pollute_business_state`，并发第 2 次。
- 输入：询问老板是否每天人工组局、是否愿意使用 AI 帮忙组局。
- 实际回复：“组局确实费脑子，谁几点有空、打多大、抽不抽烟，一堆事都得记着，有人帮着张罗肯定省心不少。”
- 实际无工具调用，原 Game 槽位保持，独立语义 8 项全部通过。失败是 runner 的裸字符串 `打多大` 命中；回复是在列举运营事项，没有向客户追问档位。
- 定位：`scripts/run_real_owner_chat_live_eval.py` 的该场景 `forbidden_reply_contains`、`validate_result` 与 `reply_semantic_expectation`；修订链在 `scripts/validate_real_owner_chat_golden.py`。
- 建议：将业务追问/招揽裸词限制转为明确语义规则，禁止把闲聊当作个人组局需求而定向追问玩法、档位、人数或招揽；允许泛论运营事项。身份、隐私、工具与真实状态硬检查继续保留。
- 需要先处理资产追溯：该原始人工 case 没有 `existing_hard_assertions_snapshot` 或 `live_scenario_id`，不能直接删硬断言。设计追加式 baseline 元数据与严格 revision 链，校验唯一来源映射、旧快照和新执行合同；保留原 JSONL 字节和原失败报告。增加正反例，验证新语义条目真正进入独立判定器，不再加一套关键词规则。

**尚未实施该修复。** `action_contract_errors_should_be_safely_recovered` 的伴随失败来自整体质量状态联动；这个案例实际动作合同错误为 0，不应误诊为另一合同问题。

## 待办二：无登记局咨询漏答当前结果

- 场景：`profile_range_empty_inquiry`，并发第 2 次。
- 输入：“下午有人打吗”；可靠画像为红中、1～3 块、烟不限。
- 已执行同玩法/范围、不限时段的查询，完整结果为空。最终只回复：“下午几点、几个人？要我帮你问问有没有人吗？”
- 独立语义正确判定：提议帮忙问人符合要求，但遗漏用户询问的当前登记局况。原查询、原输入和原期望不能改成迎合这条回复。
- 定位：`src/mahjong_agent_runtime/domains/tools/shared.py` 的 `current_game_search_reply_contract`；`src/mahjong_agent_runtime/prompts/agent_runtime_system.md` 的查询/缺项说明。
- 当前合同一处说不应先下“个人适配/不适配”结论，另一处扩大为澄清前不要判断 availability，主提示词还有澄清前不要附带“暂时没有登记中的局”的限制。这些文字冲突可能促成漏答，需结合原轨迹进一步核验，而非据此断言所有漏答都来自同一原因。
- 最小修复方向：先回答已有证据能够回答的查询事实，再补后续动作真正需要的缺项。完整空超集可支持对应范围当前无登记局；未知个人适配、未查询人群、非空或截断结果仍不能外推。没有登记局不等于没有潜在牌友；咨询不自动授权建局/邀约，已有授权也不因空查询失效。
- 相关回归：`tests/test_inquiry_evidence_scope.py`、`tests/test_query_goal_scope_contract.py`、现有 `profile_range_empty_inquiry` 与查询范围/审查集合。沉淀原失败到 adversarial 资产，保持原语义和硬断言。

**尚未实施该修复。** 没有采纳尚未验证的临时补丁，也没有降低验收标准。

## 可移植证据与本机数据边界

[匿名失败摘录](../eval/handoffs/20261030_full18_failures.json) 随仓库提交，包含两例原输入、画像最小投影、实际回复、工具结果摘要、失败项、逐项语义判定和部分结构化事件。它不是完整 trace，也不替代既有 golden。

完整失败报告、原始 trace、运行数据库和 `.env` 保留在原电脑，未提交到 Git。原报告路径为 `runtime_data/eval_runs/20261030T092732524329Z_x7rxf607/`；个人电脑不必拥有这些运行数据即可重新执行隔离夹具。不要把临时项目快照或数据库整体加入 Git。本地旧服务本次未重启。

## 在个人电脑继续

安装 Python（本次验证使用 CPython 3.12）及 Node.js，并确认 `node` 在 PATH 中，供前端交互回归测试使用。只做本地测试无需安装或启用 WeChaty。

```bash
git clone https://github.com/Schatt3n/mahjong-ops-agent.git
cd mahjong-ops-agent
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

已有 checkout 先保存自己的改动，再同步 `main`。在根目录自行配置忽略提交的 `.env`，填入自己的有效密钥：

```dotenv
MAHJONG_LLM_PROVIDER=deepseek
MAHJONG_LLM_MODEL=deepseek-flash
MAHJONG_LLM_BASE_URL=https://api.deepseek.com
MAHJONG_LLM_API_KEY=<自行填写，禁止提交>
MAHJONG_LLM_THINKING_MODE=enabled
MAHJONG_LLM_REASONING_EFFORT=high
MAHJONG_LLM_MAX_COMPLETION_TOKENS=6400
MAHJONG_LLM_MAX_CONCURRENCY=3
MAHJONG_AGENT_MAX_TOKENS_PER_CALL=64000
MAHJONG_AGENT_LLM_TIMEOUT_SECONDS=90
MAHJONG_LLM_ATTEMPT_TIMEOUT_SECONDS=60
```

独立语义判定器固定关闭思考，不能改为自己给被测输出背书。现有环境变量优先于 `.env`，注意终端里是否残留其他配置。

先阅读 `AGENTS.md`、本交接与匿名证据。针对两个问题调试可用以下命令，每次使用新目录保留旧证据；真实模型有变化，不保证每次都会复现同一句回复：

```bash
RUN_LABEL="$(date +%Y%m%dT%H%M%S)"
PYTHONPATH=src .venv/bin/python scripts/run_real_owner_chat_live_eval.py \
  --scenario profile_range_empty_inquiry \
  --scenario casual_chat_should_not_pollute_business_state \
  --repeat 2 --timeout-seconds 90 --strict \
  --db-path "runtime_data/home-debug-$RUN_LABEL/owner.sqlite3" \
  --report-path "runtime_data/home-debug-$RUN_LABEL/report.json"
```

修复后执行完整验证，不能用挑选用例或未改代码的反复重跑冒充修复：

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python scripts/run_evals.py
.venv/bin/python scripts/check_chinese_docstrings.py
git diff --check
PYTHONPATH=src .venv/bin/python scripts/run_evals.py --full
```

待办顺序：修复两项 → 新增可追溯回归/资产 → 全量复跑（包括完整隐私集合）→ 如实记录门禁与安全恢复次数 → 更新原 badcase 的 `verification.full_gate` → 提交最终验收版本。

手工体验可运行 `PYTHONPATH=src .venv/bin/python scripts/run_agent_app.py`，按启动日志提供的地址访问 `/simulation`。当前检查点仍有上述已知失败，启动成功不等于完整验收通过。
