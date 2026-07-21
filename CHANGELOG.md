# Changelog

## 2.0.0 - 2026-07-21

### 架构重构

- 将存储端口、内存后端和 SQLite 后端拆分到 `stores/`，保留旧导入路径兼容。
- 将局、客户、关系、参与者和邀约规则提取到 `domains/`。
- 将 Tool Gateway、schema、注册表和处理器拆分到 `domains/tools/`。
- 将上下文构建拆分为消息、会话、客户、关系、局况和工具结果等独立 builder。
- 将主循环、单步执行、动作处理、工具调度、上下文生命周期和客户可见处理拆分到 `services/`。
- 将 `runtime.py` 收敛为消息入口和组合根，历史私有方法由 `runtime_compat.py` 适配。
- 将客户可见审查合同从模型调用处理器中拆出，避免生成、调用和合同解析混在同一文件。

### 数据模型

- 新增 `runtime_game_participants` 联结表，以 `(game_id, customer_id)` 为复合主键并记录 `status`、`joined_at` 等参与事实。
- `runtime_games.payload` 不再持久化 `participants`、`parties` 或座位派生字段；读取时从参与者表恢复领域视图。
- 为历史嵌套参与者数据提供幂等迁移，并为 `game_id` 配置级联删除外键。
- `create_game`、`join_game` 及候选人状态更新统一通过归一化参与者持久化路径读写。

### 验证

- 全量自动化测试：`326 passed, 1 skipped`。
- Agent 确定性回归：9 个场景、138 项检查全部通过。
- badcase 回归覆盖：`fixed=24, open=0`；golden dataset 校验通过。
- 8 类确定性并发不变量全部通过。
- 与重构前提交 `4a2354b` 对比，同机 3 轮中位数墙钟变化 `+0.72%`，低于 `5%` 退化阈值。

### 兼容性

- `mahjong_agent_runtime.store`、`sqlite_store`、`tools`、`processing`、`lifecycle` 和 `loop` 保留为轻量兼容 facade。
- 包级 `InMemoryAgentStore`、`SQLiteAgentStore`、`ToolGateway`、`ActionProcessor` 和 `AgentLoop` 公共导入保持不变。
