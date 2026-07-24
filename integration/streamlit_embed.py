from __future__ import annotations

import os

import streamlit as st
import streamlit.components.v1 as components


def render_ercot_map_workbench(
    url: str | None = None,
    height: int = 900,
) -> None:
    """Embed the standalone React ERCOT map workbench in Streamlit.

    During development, run the React app first:

        cd deliverables/ercot_map_workbench_v3/frontend
        npm install
        npm run dev

    Then call this helper from the main Streamlit app.
    """

    workbench_url = url or os.getenv("ERCOT_MAP_WORKBENCH_URL", "http://127.0.0.1:5178")
    components.iframe(workbench_url, height=height, scrolling=False)


if __name__ == "__main__":
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
    render_ercot_map_workbench(height=920)
