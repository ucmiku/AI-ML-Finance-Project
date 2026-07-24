# Streamlit Integration Guide

This module intentionally does not alter the existing Streamlit app. It provides a standalone React workbench that can be embedded.

## Option A: Development iframe

Run the React workbench:

```bash
cd deliverables/ercot_map_workbench_v3/frontend
npm install
npm run dev
```

Then in Streamlit:

```python
from deliverables.ercot_map_workbench_v3.integration.streamlit_embed import render_ercot_map_workbench

render_ercot_map_workbench("http://127.0.0.1:5178", height=900)
```

## Option B: Static build hosted by FastAPI or any static server

```bash
cd deliverables/ercot_map_workbench_v3/frontend
npm run build
```

Serve `frontend/dist` from the existing platform server or a small static file server, then iframe that URL from Streamlit.

## Data API contract

The React app requests JSON only:

- `GET /api/v1/health`
- `GET /api/v1/dashboard/snapshot`
- `GET /api/v1/dashboard/timeseries`
- `GET /api/v1/map/layers`

If the API is unavailable, the frontend automatically falls back to its local demo snapshot.

React must not read parquet directly. A production adapter should convert parquet/model outputs into the snapshot JSON contract outside the React app.

## Mock API

For development only:

```bash
cd deliverables/ercot_map_workbench_v3
python -m pip install fastapi uvicorn
python -m uvicorn integration.mock_api_server:app --host 127.0.0.1 --port 8787
```

Then run the frontend with `npm run dev`.
