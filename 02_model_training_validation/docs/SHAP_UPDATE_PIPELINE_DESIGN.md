# SHAP Update Pipeline Design

## Daily Job

1. Load latest approved C1 model artifact.
2. Load frozen feature schema.
3. Load daily prediction feature table.
4. Validate feature presence and order.
5. Run C1 prediction or load same-day C1 prediction output.
6. Compute SHAP values separately for each output head.
7. Save local row-level SHAP table.
8. Aggregate to daily feature ranking.
9. Update weekly and monthly rolling rankings.
10. Generate dependence source tables for selected top features.
11. Publish CSV/Parquet files for frontend or API.

## Weekly/Monthly Aggregation

Weekly/monthly outputs should be aggregation of local SHAP values already computed with the relevant model version. If the model retrains weekly, record model_version in every output row.

## Output Heads

Keep output heads separate:

- `spread_regression`
- `negative_probability`
- `neutral_probability`
- `positive_probability`

Do not add SHAP values across output heads.

## Local Explanation

For one prediction hour, take the top N absolute SHAP values for the relevant output head. Return feature name, feature value, SHAP value, rank, feature group and prediction context.

## Dependence/Relationship Data

For a selected variable, return many rows with:

- feature value on x-axis,
- SHAP value on y-axis,
- optional color by signal, hour, p_positive or p_negative.

This allows frontend to show nonlinear threshold and regime patterns.
