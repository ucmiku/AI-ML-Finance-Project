# 交易团队交付说明：C1统一预测表

## 1. 交付对象

正式交付给交易同学的预测表：

- Parquet: `phase_C_model_selection_validation/outputs/prediction_agent_interface/C1_unified_prediction_table_2025_oof_v3.parquet`
- CSV: `phase_C_model_selection_validation/outputs/prediction_agent_interface/C1_unified_prediction_table_2025_oof_v3.csv`
- Manifest: `phase_C_model_selection_validation/outputs/prediction_agent_interface/C1_unified_prediction_table_manifest_v3.json`

该表是 C1 完整系统的 2025 OOF 预测结果，保留两个组件的原始输出：

- B2A XGBoost Regression Head
- B2B XGBoost 5/20 Classifier Head

## 2. 字段说明

| 字段 | 来源 | 用途 |
|---|---|---|
| `delivery_hour_utc` | 交付小时主键 | 对齐交易小时 |
| `delivery_time_local` / `delivery_date_local` | 时间字段 | 本地日、月度、峰谷分析 |
| `predicted_spread` | B2A XGBoost回归头 | 价差幅度、仓位和风险过滤研究 |
| `p_c1`-`p_c5` | B2B分类头 | 完整五分类概率 |
| `p_negative` | `p_c1 + p_c2` | INC信号概率 |
| `p_neutral` | `p_c3` | No-trade概率 |
| `p_positive` | `p_c4 + p_c5` | DEC信号概率 |
| `p_outer` | `p_c1 + p_c5` | 两端极端概率 |
| `predicted_class` | B2B分类头 | 分类结果 |
| `confidence` | 最大类别概率 | 信号强度 |
| `signal_base` | 当前0.60阈值规则 | 基准交易信号，-1=INC, 0=No Trade, 1=DEC |
| `recommended_action_base` | `signal_base`映射 | INC / NO_TRADE / DEC |
| `actual_spread` | 真实2025 OOF结果 | 历史回测专用 |
| `actual_class` | 真实五分类标签 | 历史分类评价 |
| `fixed_extreme_weather_flag` | 特征数据 | 极端天气分组评估 |
| `target_extreme20` / `target_extreme50` | 真实spread标签 | 尾部风险和尖峰分析 |
| `net_pnl` | 当前基准规则回测 | 历史回测结果 |
| `run_id` / `model_version` | 实验记录 | 可复现性 |
| `regression_head` / `classification_head` | 模型组件记录 | 组件级研究 |

## 3. 当前基准交易规则

当前基准策略仍然只使用 B2B 分类概率：

```text
if p_positive >= 0.60 and p_positive > p_negative:
    signal_base = +1  # DEC
elif p_negative >= 0.60 and p_negative > p_positive:
    signal_base = -1  # INC
else:
    signal_base = 0   # No Trade
```

交易成本与回测假设：

- 每个执行小时 1 MWh
- commission = 2 USD/MWh
- slippage = abs(spread) * 0.005
- capture rate = 65%
- 不按signal变化次数收费
- 不使用连续持仓限制

## 4. 2025基准表现

| 模型 | Trades | Direction Precision | Total PnL | PnL/MWh | Sharpe | Max DD | Profitable Months |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 base | 1174 | 0.665 | 3439.00 | 2.93 | 1.58 | -1791.07 | 8 |

## 5. 可继续研究的交易规则

交易同学可以基于同一张表测试：

- 只用 B2B 概率。
- B2B信号 + B2A方向确认。
- B2B信号 + B2A幅度过滤。
- 不同概率阈值。
- 不同仓位大小或分层仓位。
- 用 `confidence` 做仓位缩放。
- 用 `fixed_extreme_weather_flag` 做风险开关。
- 用 `p_outer` 识别两端尖峰机会。

注意：这些都是交易策略研究，不应回头修改模型训练参数。

## 6. 风险提示

2026 weekly结果显示，C1收益主要集中在1月和极端天气小时：

- 2026 H1 Total PnL: 5705
- 去掉1月后 PnL: -835
- 去掉Top 5交易日后 PnL: -1743
- Extreme-weather PnL: 6355
- Normal-weather PnL: -650

因此该模型更像极端天气和尖峰机会捕捉器，而不是均匀全天候盈利策略。

