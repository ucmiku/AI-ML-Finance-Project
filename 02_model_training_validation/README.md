# C1 SHAP Update Pipeline Handoff

This package defines how to produce daily, weekly and monthly SHAP outputs for frontend display.

It is a design and interface package. It does **not** train models, run Optuna, or recompute SHAP during packaging.

## Why This Is Needed

The existing SHAP handoff is static. For product use, the frontend needs regularly updated explanation data:

- daily feature ranking,
- weekly/monthly rolling ranking,
- local explanation for one delivery hour,
- single-feature dependence/relationship source tables.

SHAP is not a separate model. SHAP values are computed from the trained C1 model plus the feature rows being explained.

## Required Inputs for Real Execution

- Approved C1 model artifacts.
- Frozen C1 feature schema and feature order.
- Daily prediction feature table.
- C1 prediction outputs for the same delivery hours.
- Runtime with `shap`, `xgboost`, `pandas`, `numpy`, `joblib`.

## Output Tables

- `shap_daily_feature_ranking.csv`
- `shap_weekly_feature_ranking.csv`
- `shap_monthly_feature_ranking.csv`
- `shap_local_explanations.csv`
- `shap_dependence_daily.csv`
- `shap_dependence_weekly.csv`
- `shap_dependence_monthly.csv`

Schemas are in `schemas/`.

## Frontend Usage

Use ranking tables for leaderboard cards. Use local explanations for a single prediction hour. Use dependence tables to draw scatter plots showing nonlinear or regime-dependent relationships.

## Caution

Do not say SHAP proves causality. Use wording like "model drivers", "model sensitivity" and "risk context".
