# C组完整预测系统比较报告

- run_id: `C_system_quality_econ_v3_f5730506`
- C1：B2A XGBoost连续预测 + B2B XGBoost 5/20概率。
- C2：B4C Seq2Seq LSTM multi-task。
- C3：TFT multi-task失败归档，不进入比较。

## Prediction Quality

| system_id | model_family | rows | accuracy | macro_f1 | balanced_accuracy | log_loss | mean_abs_class_distance | catastrophic_reversal_rate | direction_accuracy | outer_macro_f1 | extreme_weather_macro_f1 | normal_weather_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1_best_boosting_complete_system | XGBoost | 8760 | 0.466895 | 0.360937 | 0.366854 | 1.22803 | 0.774543 | 0.143151 | 0.512671 | 0.36951 | 0.330285 | 0.358517 |
| C2_multitask_lstm_complete_system | Seq2Seq_LSTM | 8760 | 0.391438 | 0.323436 | 0.384994 | 1.39609 | 0.980137 | 0.179224 | 0.479338 | 0.514287 | 0.280727 | 0.323975 |

## Economic Gate

| system_id | total_pnl | total_return | trade_count | coverage | pnl_per_mwh | sharpe | sortino | max_drawdown | win_rate | profit_factor | maximum_single_loss | inc_trade_count | dec_trade_count | inc_pnl | dec_pnl | profitable_months | extreme_weather_pnl | normal_weather_pnl |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1_best_boosting_complete_system | 3439 | 0.03439 | 1174 | 0.134018 | 2.9293 | 1.5828 | 1.33389 | -1791.07 | 0.601363 | 1.40891 | -969.374 | 974 | 200 | 3576.56 | -137.56 | 8 | 3876.12 | -437.12 |
| C2_multitask_lstm_complete_system | 2110.94 | 0.0211094 | 928 | 0.105936 | 2.27472 | 1.09669 | 0.81608 | -1940.96 | 0.588362 | 1.29258 | -969.374 | 723 | 205 | 2201.11 | -90.1708 | 9 | 2422.49 | -311.554 |
