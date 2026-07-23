# B组单模型比较报告

- run_id: `B6_single_model_comparison_v3_f5730506`
- data_hash: `f5730506707c2f227f6208bb6bc00ca4c0c45fe5a23c3148c1e9c2c04cfa0717`
- 范围：仅2025 OOF；未访问2026。
- 规则：失败或 NOT_COMPARABLE 的实验只归档，不进入正式排名。

## 单模型汇总表

| experiment_id | model_name | task | status | comparable | rows | mae | rmse | r2 | direction_accuracy | macro_f1_5_20 | log_loss_5_20 | catastrophic_reversal_rate | total_pnl | trade_count | daily_sharpe | max_drawdown |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1A | Ridge Regression | continuous | COMPLETED | True | 8760 | 17.0023 | 59.016 | -1.18448 | 0.453653 | 0.21806 |  | 0.129795 | 0 | 0 |  | 0 |
| B1B | Logistic 5/20 | classification_5_20 | COMPLETED | True | 8760 | 12.3539 | 40.4156 | -0.0244873 | 0.451256 | 0.305034 | 1.52176 | 0.205822 | 0 | 0 |  | 0 |
| B2A | XGBoost Regression | continuous | COMPLETED | True | 8760 | 10.9302 | 39.6681 | 0.0130599 | 0.490639 | 0.177039 |  | 0.00856164 | 0 | 0 |  | 0 |
| B2B | XGBoost 5/20 | classification_5_20 | COMPLETED | True | 8760 | 11.2722 | 39.8099 | 0.00598958 | 0.512671 | 0.360937 | 1.22803 | 0.143151 | 0 | 0 |  | 0 |
| B3A | LightGBM Regression Import | continuous | COMPLETED_IMPORTED | True | 8760 | 11.2918 | 39.6667 | 0.0131307 | 0.315868 | 0.0959202 | 2.25783 | 0.215068 | 3205.31 | 104 | 2.59943 | -102.614 |
| B3B | LightGBM 5/20 Import | classification_5_20 | COMPLETED_IMPORTED | True | 8760 |  |  |  | 0.505137 | 0.354309 | 1.27909 | 0.146689 | 1832.07 | 1595 | 0.790284 | -2472.42 |
| B4A | Seq2Seq LSTM Continuous | continuous | COMPLETED | True | 8760 | 10.8074 | 39.9135 | 0.000812377 | 0.51016 | 0.160716 |  | 0.0259132 | 0 | 0 |  | 0 |
| B4B | Seq2Seq LSTM 5/20 | classification_5_20 | COMPLETED | True | 8760 |  |  |  | 0.486986 | 0.325082 | 1.39098 | 0.175799 | 2376.08 | 894 | 1.24336 | -1628.59 |
| B4C | Seq2Seq LSTM Multi-task | multitask | COMPLETED | True | 8760 | 11.1411 | 39.9286 | 5.28756e-05 | 0.479338 | 0.323436 | 1.39609 | 0.179224 | 2110.94 | 928 | 1.09669 | -1940.96 |
| B5A | TFT Distribution | continuous | COMPLETED | True | 8760 | 11.9738 | 40.3248 | -0.0198899 | 0.4629 | 0.160154 |  | 0.0769406 | -3009.57 | 1049 | -2.32393 | -3151.84 |
| B5B | TFT 5/20 | classification_5_20 | COMPLETED | True | 8760 |  |  |  | 0.394863 | 0.180231 | 1.38214 | 0.109132 | -1673.91 | 466 | -2.40413 | -1703.46 |
| B5C | TFT Multi-task | multitask | FAILED_ARCHIVED | False | 0 |  |  |  |  |  |  |  |  |  |  |  |

## B6结论

- 最佳连续/分布单任务：`B4A`。
- 最佳五分类单任务：`B2B`。
- 最佳深度候选：`B4B`。
- `B5C` 已按失败归档，原因是当前可运行 TFT baseline 不支持真正单模型 multi-task trainer；未用拼接结果冒充。

## 生成文件

- `D:\NUS SOC2026\ERCOT_Virtual_Bidding_Project\02_prediction_agent_v2\phase_C_model_selection_validation\outputs\B_single_models\B6_single_model_comparison_v3.csv`
- `D:\NUS SOC2026\ERCOT_Virtual_Bidding_Project\02_prediction_agent_v2\phase_C_model_selection_validation\outputs\B_single_models\B6_task_winners_v3.json`