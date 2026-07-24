# C4 exploratory head-swap报告

- C4 = B4A LSTM continuous OOF predictions + B2B XGBoost 5/20 classifier OOF probabilities。
- 未重新训练模型，未运行Optuna，未修改阈值。
- 严格检查B4A、B2B、C1的2025 OOF UTC时间戳一一对齐；未使用inner join静默删样本。
- 标记：exploratory head-swap，不覆盖正式C1。

## Prediction Metrics

| system_id | run_id | rows | accuracy | macro_f1 | balanced_accuracy | log_loss | mean_abs_class_distance | catastrophic_reversal_rate | direction_accuracy | extreme_weather_macro_f1 | normal_weather_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1_best_boosting_complete_system | C4_head_swap_exploratory_v3_f5730506 | 8760 | 0.466895 | 0.360937 | 0.366854 | 1.22803 | 0.774543 | 0.143151 | 0.512671 | 0.330285 | 0.358517 |
| C4_exploratory_head_swap | C4_head_swap_exploratory_v3_f5730506 | 8760 | 0.466895 | 0.360937 | 0.366854 | 1.22803 | 0.774543 | 0.143151 | 0.512671 | 0.330285 | 0.358517 |

## Economic Metrics

| system_id | run_id | total_pnl | total_return | trade_count | coverage | direction_precision | pnl_per_mwh | sharpe | sortino | max_drawdown | cvar_95_daily | win_rate | profit_factor | maximum_single_loss | profitable_months | inc_count | dec_count | inc_pnl | dec_pnl | extreme_weather_pnl | normal_weather_pnl | extreme_weather_tail20_pnl | extreme_weather_tail50_pnl | pnl_ex_top5_days | top1_day_share | top5_day_share | top1_hour_share | top5_hour_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1_best_boosting_complete_system | C4_head_swap_exploratory_v3_f5730506 | 3439 | 0.03439 | 1174 | 0.134018 | 0.665247 | 2.9293 | 1.5828 | 1.33389 | -1791.07 | -219.546 | 0.601363 | 1.40891 | -969.374 | 8 | 974 | 200 | 3576.56 | -137.56 | 3876.12 | -437.12 | 3505.97 | 0 | 921.499 | 0.176153 | 0.732044 | 0.125373 | 0.42571 |
| C4_exploratory_head_swap | C4_head_swap_exploratory_v3_f5730506 | 3439 | 0.03439 | 1174 | 0.134018 | 0.665247 | 2.9293 | 1.5828 | 1.33389 | -1791.07 | -219.546 | 0.601363 | 1.40891 | -969.374 | 8 | 974 | 200 | 3576.56 | -137.56 | 3876.12 | -437.12 | 3505.97 | 0 | 921.499 | 0.176153 | 0.732044 | 0.125373 | 0.42571 |

## Trade Overlap

| comparison | hours | both_trade | both_same_signal | c1_only_trade | c4_only_trade | opposite_trade | trade_jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1_vs_C4 | 8760 | 1174 | 8760 | 0 | 0 | 0 | 1 |

## 文件

- `D:\NUS SOC2026\ERCOT_Virtual_Bidding_Project\02_prediction_agent_v2\phase_C_model_selection_validation\outputs\C_combination_systems\C4_exploratory_head_swap_oof_2025_v3.parquet`
- `D:\NUS SOC2026\ERCOT_Virtual_Bidding_Project\02_prediction_agent_v2\phase_C_model_selection_validation\outputs\prediction_quality_gate\C4_head_swap_prediction_metrics_2025_v3.csv`
- `D:\NUS SOC2026\ERCOT_Virtual_Bidding_Project\02_prediction_agent_v2\phase_C_model_selection_validation\outputs\economic_value_gate\C4_head_swap_economic_metrics_2025_v3.csv`
- `D:\NUS SOC2026\ERCOT_Virtual_Bidding_Project\02_prediction_agent_v2\phase_C_model_selection_validation\outputs\economic_value_gate\C4_head_swap_monthly_pnl_2025_v3.csv`
- `D:\NUS SOC2026\ERCOT_Virtual_Bidding_Project\02_prediction_agent_v2\phase_C_model_selection_validation\outputs\economic_value_gate\C4_head_swap_trade_overlap_2025_v3.csv`
- `D:\NUS SOC2026\ERCOT_Virtual_Bidding_Project\02_prediction_agent_v2\phase_C_model_selection_validation\outputs\economic_value_gate\C4_head_swap_disagreement_hours_2025_v3.csv`
