# FASTAPI_HANDOFF

Import:

```python
from src.inference.c1_inference import load_c1_agent
agent = load_c1_agent()
```

Core methods:

- `agent.predict_one(features, delivery_hour_utc=None)`
- `agent.predict_batch(records)`
- `agent.explain_local(features, head="predicted_class", top_k=10)`
- `agent.get_global_shap(head=None, top_k=20)`
- `agent.metadata()`

Suggested FastAPI endpoints:

- `POST /predict`
- `POST /predict-batch`
- `POST /explain/local`
- `GET /explain/global`
- `GET /model/metadata`

See `examples/fastapi_integration_example.py`.

Signal rule:

- `p_negative = p_c1 + p_c2`
- `p_neutral = p_c3`
- `p_positive = p_c4 + p_c5`
- `p_positive >= 0.60` and greater than `p_negative`: `DEC`
- `p_negative >= 0.60` and greater than `p_positive`: `INC`
- otherwise: `NO_TRADE`

SHAP is model explanation, not causal proof. Keep output heads separate.
