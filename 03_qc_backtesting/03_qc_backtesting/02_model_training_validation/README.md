# 02 Model Training Validation

This module contains the compact Git handoff for the ERCOT Prediction Agent v2 model work.

## Final Model

The final Prediction Agent is **C1_XGBoost_Prediction_Agent v3**.

C1 has two heads:

- **B2A XGBoost Regression**: continuous `predicted_spread` output.
- **B2B XGBoost 5/20 Classifier**: five-class probabilities and the current benchmark trading signal.

B2B is the best single classification model from the 2025 OOF comparison. C1 is the final Prediction Agent selected using **2025 OOF only**. No 2026 result is used here for model selection or threshold tuning.

## Signal Rule

The current C1 benchmark signal is generated only from B2B classification probabilities:

- `p_negative = p_c1 + p_c2`
- `p_neutral = p_c3`
- `p_positive = p_c4 + p_c5`
- `p_positive >= 0.60` and greater than `p_negative`: `DEC`
- `p_negative >= 0.60` and greater than `p_positive`: `INC`
- otherwise: `NO_TRADE`

B2A currently provides spread magnitude, explanation support, and future risk/filtering extensions. It does not change the current benchmark signal.

## Important Notes

- C4 is an exploratory head-swap robustness test: it replaces the B2A continuous head with B4A LSTM continuous predictions while retaining the B2B classifier signal.
- C4 classification and economic results match C1 because the signal is still driven by B2B probabilities.
- C4 has slightly lower continuous MAE, but C1 remains simpler and better overall for deployment because C4 does not improve RMSE, R2, direction accuracy, or the benchmark signal.
- TFT did not produce a comparable advantage in this round.
- SHAP values are model explanations, not causal proof. Weather variables appear more like amplifiers, especially for positive spikes and extreme trading hours.
- The 2025 OOF table ends at `2026-01-01 05:00 UTC` because the ERCOT local 2025 delivery year converts into UTC that way.

## Install

```bash
pip install -r requirements-model.txt
```

## Packaged Artifacts

- `models/c1_prediction_agent/`: C1 fold models, feature schema, class mapping, thresholds and metadata.
- `predictions/c1_2025_oof_predictions.csv`: unified 2025 OOF prediction table for trading/front-end research handoff.
- `metrics/`: model comparisons, economic metrics, ablation summaries and SHAP tables.
- `reports/`: concise project reports and SHAP figures.
- `src/`: final training/evaluation/ablation/explainability scripts and a clean inference helper.

## Run Inference

Use a feature table containing the frozen feature columns listed in `models/c1_prediction_agent/feature_schema.json`.

```bash
python src/inference/c1_inference.py --input path/to/features.parquet --output predictions/new_predictions.csv --fold validation_fold_3
```

The output includes continuous spread, five class probabilities, aggregate direction probabilities, predicted class, confidence and `INC` / `DEC` / `NO_TRADE` signal.
