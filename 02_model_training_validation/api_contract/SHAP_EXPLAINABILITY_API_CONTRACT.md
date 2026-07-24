# SHAP Explainability API Contract

## GET /explainability/ranking

Query parameters:

- `window`: daily, weekly, monthly
- `date`: YYYY-MM-DD
- `output_head`: spread_regression, negative_probability, neutral_probability, positive_probability
- `top_n`: optional integer, default 20

Returns rows matching `schemas/shap_feature_ranking_schema.csv`.

## GET /explainability/local

Query parameters:

- `delivery_hour_utc`
- `output_head`
- `top_n`: optional integer, default 10

Returns rows matching `schemas/shap_local_explanation_schema.csv`.

## GET /explainability/dependence

Query parameters:

- `feature_name`
- `window`: daily, weekly, monthly
- `date`: YYYY-MM-DD
- `output_head`
- `color_by`: optional, e.g. signal, p_positive, ercot_local_hour

Returns rows matching `schemas/shap_dependence_schema.csv`.

## Error Rules

- Unknown feature: HTTP 404 or 422.
- Missing date/window/head: HTTP 422.
- No available SHAP output for requested date: HTTP 404.
