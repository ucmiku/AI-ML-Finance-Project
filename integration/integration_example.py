from __future__ import annotations

import os

import streamlit as st

from streamlit_embed import render_ercot_map_workbench


st.set_page_config(page_title="ERCOT Map Workbench v3", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stHeader"], [data-testid="stToolbar"] { display: none; }
    .block-container { padding: 0 !important; max-width: none !important; }
    iframe { display: block; }
    </style>
    """,
    unsafe_allow_html=True,
)

workbench_url = os.getenv("ERCOT_MAP_WORKBENCH_URL", "http://127.0.0.1:5178")
render_ercot_map_workbench(workbench_url, height=int(os.getenv("ERCOT_MAP_WORKBENCH_HEIGHT", "900")))
