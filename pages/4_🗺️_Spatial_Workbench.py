import streamlit as st
from components.agent_ui import render_global_copilot
from integration.streamlit_embed import render_ercot_map_workbench

# --- Page Configuration ---
st.set_page_config(
    page_title="Spatial Workbench", 
    page_icon="🗺️", 
    layout="wide"
)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⏱️ Spatial Operations")
    st.markdown("Explore regional risk, weather forecast points, and training distributions.")
    st.markdown("---")
    render_global_copilot()

# --- Main Page Content ---
st.markdown("### 🗺️ ERCOT Spatial Data Distribution & Weather Workbench")
st.markdown("Interactive visualization of training data distribution, weather forecast nodes, and regional risk levels across ERCOT hubs.")
st.markdown("---")

# 调用后端同学封装好的 React 工作台组件（连接本地 Vite 开发端口 5178）
render_ercot_map_workbench("http://127.0.0.1:5178", height=900)