# Variable Relationship Display

For a single variable relationship chart:

- x-axis: `feature_value`
- y-axis: `shap_value`
- color: `signal`, `p_positive`, `p_negative`, or local hour
- filter: output head and date window

Interpretation:

- Positive SHAP means the feature pushed that output head upward for that row.
- Negative SHAP means the feature pushed that output head downward for that row.
- Curved or threshold-like scatter patterns indicate nonlinear model behavior.
- Clustered colors indicate regime dependence.

Do not infer causality from the plot.
