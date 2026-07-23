# MODEL_STATUS

`deployment_status = "demo_three_fold_ensemble"`

`final_refit_available = false`

No true final-refit B2A/B2B all-data production artifact was found. This package therefore uses a demo three-fold ensemble.

Regression prediction uses the average of the three B2A fold model outputs. Classification probabilities use the average of the three B2B fold probability vectors.

This is usable for course demonstration and FastAPI integration. It is not a final production model refit on all allowed training data.

Local SHAP is available for `spread_regression`, `predicted_class`, and component class heads `C1` to `C5`. Do not fabricate additive SHAP values for `p_negative` or `p_positive`.

SHAP is model explanation, not causal proof.
