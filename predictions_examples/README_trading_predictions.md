# 交易策略阶段预测数据交接说明

本目录包含最终选优模型 LightGBM 的两份逐小时预测结果。文件由已验收的正式实验产物原样复制，没有重新训练、修改预测值或删除审计行。

## 2025 独立验证预测

- 文件：`lightgbm_predictions_2025_validation.csv`
- 用途：在 2025 validation 上开发和比较交易策略。
- 行数：8,760。
- 主要字段：`datetime`、`actual_spread`、`predicted_spread`、`residual`、`absolute_error`、实际/预测方向和极端小时标记。
- SHA-256：`2C1918D22874CAF2AB37949771D6BE2FD3959070A2AA7D3FE6289C19DC4A3D3F`

## 2026 H1 最终滚动预测

- 文件：`lightgbm_predictions_2026_H1_walkforward.csv`
- 用途：最终独立测试或冻结策略后的 2026 H1 评价。
- 行数：4,338；其中 4,336 行 `target_available=1`，2 行为缺失 RT 目标的审计行。
- 预测方式：27 个 weekly expanding-window 批次，终点为 2026-06-30 18:00 CDT。
- 主要字段：`datetime`、`week_id`、`train_end`、`actual_spread`、`predicted_spread`、`residual`、`absolute_error`、实际/预测方向、`target_available` 和 `target_missing_reason`。
- SHA-256：`450D86AF48241E39BD6A86B17E09D7039EA83E155CE9ED71D60319B9D328CB3B`

## 使用注意

交易策略应先在 2025 文件上开发和冻结，不应利用 2026 结果继续选择阈值、规则或模型。评价 2026 实际收益时仅使用 `target_available == 1` 的行；两条缺失目标记录应保留用于时间轴审计，但不得用于 PnL 或策略指标计算。
