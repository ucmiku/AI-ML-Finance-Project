# C1 2026 H1 Temporal Robustness Evaluation

本文件补充 C1 在 2026 H1 的 weekly expanding walk-forward 预测和冻结规则时间稳健性结果。此次整理只复用已有结果，没有重新训练、没有运行 Optuna、没有修改模型参数或交易阈值，也没有覆盖 2025 OOF 文件。

## Research Label

**frozen-rule post-hoc temporal robustness evaluation**

由于团队此前已经查看和讨论过部分 2026 结果，本结果不得称为 untouched holdout。2026 H1 只用于时间稳健性评价，不得反向用于调整 2025 模型、参数、特征或阈值。

## Model Definition

C1 = B2A XGBoost Regression continuous head + B2B XGBoost 5/20 classifier probability head。

- B2A 输出：`predicted_spread`
- B2B 输出：`p_c1` 至 `p_c5`
- `p_negative = p_c1 + p_c2`
- `p_neutral = p_c3`
- `p_positive = p_c4 + p_c5`
- `p_positive >= 0.60 -> DEC`
- `p_negative >= 0.60 -> INC`
- otherwise -> `NO_TRADE`

## Source Files Found

- Prediction: `D:\NUS SOC2026\ERCOT_Virtual_Bidding_Project\02_prediction_agent_v2\phase_C_model_selection_validation\delivery_packages\trading_handoff_C1_2026_H1_v1\C1_unified_prediction_table_2026_H1_walkforward_v1.parquet`
- Weekly manifest: `D:\NUS SOC2026\ERCOT_Virtual_Bidding_Project\02_prediction_agent_v2\phase_C_model_selection_validation\outputs\final_2026\C1_C4_weekly_manifest_2026_v3.csv`
- Run script: `D:\NUS SOC2026\ERCOT_Virtual_Bidding_Project\02_prediction_agent_v2\phase_C_model_selection_validation\scripts\run_c1_c4_weekly_2026_v3.py`
- B2A frozen params: `D:\NUS SOC2026\ERCOT_Virtual_Bidding_Project\02_prediction_agent_v2\phase_C_model_selection_validation\outputs\experiments\B2_regression_formal_v3r1_f5730506\best_params.json`
- B2B frozen params: `D:\NUS SOC2026\ERCOT_Virtual_Bidding_Project\02_prediction_agent_v2\phase_C_model_selection_validation\outputs\experiments\B2_classifier_5_20_formal_v3r1_f5730506\best_params.json`

## Coverage And Checks

- Rows: 4197
- UTC range: 2026-01-01 06:00:00+00:00 to 2026-07-01 04:00:00+00:00
- Walk-forward weeks: 27
- Matches model-ready 2026 H1 `evaluation_eligible=1`: True
- UTC key unique: True
- UTC sorted: True
- Probability sum bad rows: 0
- Direction probability formula bad rows: negative=0, neutral=0, positive=0
- Training cutoff after prediction hour rows: 0
- Weekly expanding walk-forward: True
- Single 2025 model post-hoc prediction: False

## Metric Snapshots

Prediction metrics are in `metrics/2026_h1_temporal_robustness/c1_prediction_metrics_2026_h1.csv`.
Economic metrics are in `metrics/2026_h1_temporal_robustness/c1_economic_metrics_2026_h1.csv`.
Monthly metrics are in `metrics/2026_h1_temporal_robustness/c1_monthly_metrics_2026_h1.csv`.
Robustness checks are in `metrics/2026_h1_temporal_robustness/c1_robustness_checks_2026_h1.csv`.

## Important Notes

- 2025 OOF remains the model-selection basis.
- 2026 H1 is a frozen-rule temporal robustness check only.
- The 0.60 probability threshold was not tuned on 2026.
- The current Git package does not include weekly model artifacts by default. The source experiment directory contains weekly C1/C4 model files, but they are not required for reviewing predictions and metrics and would add unnecessary weight to the Git module.
