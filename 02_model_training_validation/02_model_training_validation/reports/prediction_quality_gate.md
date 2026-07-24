# Prediction Quality Gate

主排序使用 Macro-F1，其次 Log Loss、灾难性反转率。普通 Accuracy 不作为主排名依据。

| system_id | model_family | rows | accuracy | macro_f1 | balanced_accuracy | log_loss | mean_abs_class_distance | catastrophic_reversal_rate | direction_accuracy | outer_macro_f1 | extreme_weather_macro_f1 | normal_weather_macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1_best_boosting_complete_system | XGBoost | 8760 | 0.466895 | 0.360937 | 0.366854 | 1.22803 | 0.774543 | 0.143151 | 0.512671 | 0.36951 | 0.330285 | 0.358517 |
| C2_multitask_lstm_complete_system | Seq2Seq_LSTM | 8760 | 0.391438 | 0.323436 | 0.384994 | 1.39609 | 0.980137 | 0.179224 | 0.479338 | 0.514287 | 0.280727 | 0.323975 |
