# Prediction Agent v2 工作总结与结果评价

## 1. 项目目标

本轮 Prediction Agent v2 的目标是构建无天气泄漏风险的 ERCOT DART spread 预测系统。正式预测任务采用五分类：

- C1: spread < -20
- C2: -20 <= spread < -5
- C3: -5 <= spread <= 5
- C4: 5 < spread <= 20
- C5: spread > 20

核心约束：

- 不复用 v1 的 fitted model、scaler、Optuna study、SHAP、预测或回测结果。
- 模型输入来自 v2 冻结数据集。
- 2025 用三折 expanding-window OOF 做模型选择。
- 2026 只在明确授权后用于 weekly walk-forward 评价。
- 正式前端部署只保留一个 Prediction Agent：C1。

## 2. 数据与特征

正式 v2 数据已完成泄漏安全重建与冻结。模型输入表排除了17个无源payload小时，不插值、不补造源数据。

特征分为四层：

| 特征集 | 内容 | 特征数 |
|---|---|---:|
| Z0 | 历史价差、日历、天然气 | 28 |
| Z1 | Z0 + 负荷、风电、光伏、可再生、净负荷 | 101 |
| Z2 | Z1 + 原始日前天气预测 | 123 |
| Z3 | Z2 + Ramp、持续性、极端天气、交互特征 | 138 used / 166 whitelist |

说明：消融实验中只使用 `fold_fitted=0` 且已经存在于冻结表中的特征，避免在全表提前拟合 fold-specific 天气异常。

## 3. B组单模型结果

| 实验 | 模型 | 任务 | Macro-F1 | MAE | Log Loss | Direction Acc | PnL | Sharpe | 评价 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| B1A | Ridge | 连续 | 0.218 | 17.002 | NA | 0.454 | 0 | NA | 弱基线 |
| B1B | Logistic 5/20 | 分类 | 0.305 | 12.354 | 1.522 | 0.451 | 0 | NA | 可用但弱 |
| B2A | XGBoost Regression | 连续 | 0.177 | 10.930 | NA | 0.491 | 0 | NA | 连续头可用 |
| B2B | XGBoost 5/20 | 分类 | 0.361 | 11.272 | 1.228 | 0.513 | 0 | NA | 最佳单分类模型 |
| B3A | LightGBM Regression import | 连续 | 0.096 | 11.292 | 2.258 | 0.316 | 3205 | 2.599 | 极保守交易头 |
| B3B | LightGBM 5/20 import | 分类 | 0.354 | NA | 1.279 | 0.505 | 1832 | 0.790 | 接近B2B但回撤差 |
| B4A | Seq2Seq LSTM Continuous | 连续 | 0.161 | 10.807 | NA | 0.510 | 0 | NA | MAE最低但无交易触发 |
| B4B | Seq2Seq LSTM 5/20 | 分类 | 0.325 | NA | 1.391 | 0.487 | 2376 | 1.243 | 最佳深度分类 |
| B4C | Seq2Seq LSTM Multi-task | 多任务 | 0.323 | 11.141 | 1.396 | 0.479 | 2111 | 1.097 | 可用但弱于C1 |
| B5A | TFT Distribution | 连续 | 0.160 | 11.974 | NA | 0.463 | -3010 | -2.324 | 经济失败 |
| B5B | TFT 5/20 | 分类 | 0.180 | NA | 1.382 | 0.395 | -1674 | -2.404 | 分类弱 |
| B5C | TFT Multi-task | 多任务 | NA | NA | NA | NA | NA | NA | FAILED_ARCHIVED |

B组结论：

- 统计上，B2B XGBoost 5/20 是最强分类头。
- 连续误差上，B4A LSTM MAE略好，但预测太保守，按 ±10 交易阈值没有触发交易。
- TFT本轮没有形成可比较优势。

## 4. C组完整系统

| 系统 | 组成 | Macro-F1 | Log Loss | Direction Acc | PnL | Sharpe | Max DD | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| C1 | B2A XGBoost回归头 + B2B XGBoost分类头 | 0.361 | 1.228 | 0.513 | 3439 | 1.583 | -1791 | 正式胜出 |
| C2 | B4C LSTM Multi-task | 0.323 | 1.396 | 0.479 | 2111 | 1.097 | -1941 | 次优深度系统 |
| C3 | Multi-task TFT | NA | NA | NA | NA | NA | NA | 失败归档 |
| C4 | B4A LSTM连续头 + B2B分类头 | 0.361 | 1.228 | 0.513 | 3439 | 1.583 | -1791 | exploratory head-swap |

C4与C1分类和交易指标完全一致，因为二者使用同一个 B2B 分类概率头。区别只在连续 spread 预测头：

| 系统 | 连续头 | Spread MAE | RMSE | R2 | Sign Direction Acc |
|---|---|---:|---:|---:|---:|
| C1 | B2A XGBoost | 10.930 | 39.668 | 0.013 | 0.607 |
| C4 | B4A LSTM | 10.807 | 39.913 | 0.001 | 0.605 |

## 5. C1 Z0-Z3消融

| 特征集 | Macro-F1 | Log Loss | Spread MAE | Extreme Weather Macro-F1 | Total PnL | Trades | Sharpe | Max DD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Z0 | 0.298 | 1.274 | 11.070 | 0.289 | 1788 | 898 | 0.868 | -1505 |
| Z1 | 0.359 | 1.231 | 10.888 | 0.338 | 3038 | 1140 | 1.152 | -1916 |
| Z2 | 0.361 | 1.229 | 10.966 | 0.336 | 3118 | 1226 | 1.161 | -1868 |
| Z3 | 0.361 | 1.228 | 10.930 | 0.330 | 3439 | 1174 | 1.290 | -1791 |

消融结论：

- 最大提升来自 Z1：负荷、风、光伏、净负荷显著提升分类能力。
- Z2加入原始天气后，Macro-F1略高。
- Z3对Macro-F1提升不大，但经济表现最好，说明Ramp、持续性和极端天气交互更像交易筛选器，而不是单纯提高平均分类准确率。

## 6. 2026 Weekly Walk-forward

按用户授权，对 C1 与 C4 做了 2026 H1 weekly expanding-window 测试。

| 系统 | Rows | Macro-F1 | Log Loss | Spread MAE | PnL | Trades | Sharpe | Max DD | Profitable Months |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 4197 | 0.329 | 1.283 | 14.858 | 5705 | 441 | 1.677 | -2246 | 4 |
| C4 | 4197 | 0.329 | 1.283 | 15.377 | 5705 | 441 | 1.677 | -2246 | 4 |

2026评价：

- C1连续头优于C4连续头。
- C1与C4交易完全一致，因为交易信号仍由B2B分类概率产生。
- 2026收益主要来自1月和极端天气小时：去掉1月后PnL为 -835，去掉Top 5交易日后PnL为 -1743。
- 因此2026表现更像尖峰/极端天气捕捉，不是均匀稳健盈利。

## 7. SHAP解释结果

解释对象为正式 C1。SHAP结果显示：

- Load / Net-load 是最核心变量组。
- 对负向极端概率，重要变量包括 `net_load_z30_same_hour`、`load_system_z30_same_hour`、`solar_ramp_1h_mw`、`hour_sin`、`is_evening_peak`。
- 对正向极端概率，重要变量包括 `net_load_ramp_1h_mw`、`net_load_ramp_3h_mw`、`spread_asof_lag168`、`cloud_cover_dfw_mean_pct`。
- 对连续spread，重要变量包括 `spread_asof_roll_mad24`、`load_south_central_mw`、`cloud_cover_dfw_mean_pct`、`net_load_z30_same_hour`。

解释结论：模型主要依赖负荷、净负荷、ramp、日历峰谷结构和历史spread波动。天气变量更多是放大器，尤其影响正向尖峰和极端交易小时。SHAP是模型解释，不构成因果证明。

## 8. 最终建议

正式 Prediction Agent 采用：

`C1 = B2A XGBoost Regression Head + B2B XGBoost 5/20 Classification Head`

部署侧只部署 C1。历史页面可以展示B1-B5、C1-C4、LightGBM联合探索模型的静态对比，但不应把所有模型都部署为实时服务。

