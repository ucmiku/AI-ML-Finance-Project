# Explainability and Error Attribution

- run_id: `explainability_v3_f5730506`
- final_prediction_agent: `C1_best_boosting_complete_system`
- evaluation_period: `2025_OOF / validation_fold_3解释样本`
- 解释器：TreeSHAP；若某输出失败，已在源表中标记为XGBoostGainFallback。
- 注意：SHAP及梯度归因属于模型解释，不构成天气导致电价变化的因果证明。

## 生成内容

- Global SHAP / feature importance top-20源表与PNG。
- 固定特征聚合组：Historical Spread, Load, Wind, Solar and Net Load, Raw Weather, Extreme Weather, Gas, Calendar。
- 本地案例选择表：正确极端盈利机会、漏报大价差机会、方向错误亏损、正常天气日。

## 警告与限制

- 无

## 消融限制

Z0-Z3消融没有已完成训练输出，本轮只保存范围缩减说明，未把空结果加入正式比较。
