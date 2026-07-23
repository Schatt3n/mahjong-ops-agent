# 2026-07-23 微信群监听数据清洗复核

## 数据边界

- 原始来源为本机仅接收模式下的微信群监听日志，原始日志不进入 Git。
- 当日共观察到 84 条群事件、6 个群；文本业务信号主要来自两个组局群。
- 数据集只保留业务文本、相对时间、匿名角色与截断 SHA-256 来源引用。
- 微信 ID、群名、成员昵称、头像、签名、完整 payload 和原始消息 ID 均未写入测试资产。

## 分层结果

Gold 共 4 条：

1. `cq173 人齐开 烟都可1块`：紧凑财敲报局，验证烟况“不限”。
2. `红中3爆 371，人齐开`：紧凑红中报局，验证特殊规则与金额不臆造。
3. 引用 `财敲一块 272 11点开` 后更新为 `财敲1块 371 11点开 有烟`：验证更新同一局而非创建重复局。
4. `财敲一块 371 有烟人齐开`：验证人数进展和人齐开语义。

Adversarial 共 4 条：

1. 群成员直接 @ 另一成员并说三缺一：缺玩法、档位、时间和烟况，不能自动建局。
2. `散约371 371`：存在本地域歧义，暂不写死解释。
3. 运营招募公告：不应误建成局。
4. 与麻将无关的短群聊：作为噪声过滤候选，尚不强制写成规则。

## 截图批次去重

本轮同时复核了两批老板私聊截图。关键文本已存在于：

- `eval/golden/real_owner_chat_transcript_20260627_20260719_owner_b.md`
- `eval/golden/real_owner_chat_timeline_20260627_20260719_owner_b.json`

因此不重复写入 Gold，避免同一真实话术被重复计权。截图只用于人工核验既有转写的时间线和内容覆盖。

## 本轮发现并闭环的问题

旧实现只支持“引用某局后说人齐/满了”，无法处理引用旧报局后补充或更新人数、时间、烟况。真实消息会把 `272` 和更新后的 `371` 解析成两个局。

修复后，引用关系先定位原局，再把本条可解析字段合并到原 `BoardItem`；目标条目的稳定 ID 和看板编号不变，人数、时间、烟况等字段按新消息更新。新增 `quoted_requirement_update` 生产链路回归，明确要求最终只有一个逻辑局。

## 复核运行

```bash
python scripts/validate_real_group_chat_dataset.py
PYTHONPATH=src python scripts/run_real_group_chat_flow_eval.py \
  --dataset eval/golden/real_group_chat_20260723.jsonl \
  --report-path runtime_data/real_group_chat_flow_eval_20260723_report.json \
  --strict
PYTHONPATH=src python -m pytest -q tests/test_real_group_chat_dataset.py
```

## 验证结果

- 真实群聊数据校验：24 条通过，其中 Gold 16 条、Adversarial 8 条。
- 2026-07-23 生产链路回放：4/4 通过，失败 0 条。
- 全量自动化回归：554 passed，1 skipped。
- 新增资产不包含微信 ID、群名、昵称、完整消息 ID 或原始 payload。
