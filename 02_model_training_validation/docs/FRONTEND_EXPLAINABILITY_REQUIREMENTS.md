# Frontend Explainability Requirements

## Recommended Pages

1. Global explanation dashboard.
2. Daily/weekly/monthly top-driver leaderboard.
3. Single prediction explanation panel.
4. Feature relationship explorer.

## API/UI Controls

- date selector,
- window selector: daily / weekly / monthly,
- output head selector,
- feature selector,
- delivery hour selector.

## Suggested Visuals

- Top driver bar chart from ranking table.
- Feature group stacked bar chart from ranking table grouped by feature_group.
- Dependence scatter: x = feature_value, y = shap_value, color = signal or hour.
- Local explanation waterfall-like table for selected hour.

## Copywriting Guardrail

Use "model used", "model driver", "associated in this model".
Avoid "caused", "proves", "will lead to".
