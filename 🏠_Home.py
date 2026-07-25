import streamlit as st
from utils import init_database
from components.agent_ui import render_global_copilot
from integration.streamlit_embed import render_ercot_map_workbench
import base64
import os
from components.theme import inject_custom_css, render_sidebar_logo


# 1. 必须确保 layout="wide"
st.set_page_config(
    page_title="GridWise | Trading Terminal", 
    page_icon="⚡", 
    layout="wide"
)
inject_custom_css()

# 检查持久化登录 Token（刷新页面不会掉线）
if "token" in st.query_params and st.query_params["token"] == "valid":
    st.session_state['logged_in'] = True
elif 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# 2. 页面标题 (保持原样)
st.markdown("## ⚡ GridWise Quantitative Trading Terminal")
st.markdown("Welcome to the ERCOT North Hub DAM-RTM Spread Prediction System.")
st.markdown("---")

# 3. 侧边栏：全局设置与 AI 助手
# ==========================================
# 🌟 2. 侧边栏 Logo 渲染（带有毛玻璃遮罩 + 彻底解决文字缺失）
# ==========================================
with st.sidebar:
    render_sidebar_logo()
    
    # 1. 优先渲染 AI 浮窗对话框，让它排在最上方
    render_global_copilot()
    
    st.markdown("---")
    
    # 2. 其次是全局设置
    st.markdown("### ⚙️ Global Settings")
    api_key = st.text_input(
        "DeepSeek API Key:", 
        value="sk-8080fcbf46f3459895cd3f8daed48535", 
        type="password",                            
        help="Required for Agent and Strategy Generation"
    )
    
    if api_key:
        st.session_state['api_key'] = api_key
        st.success("✅ API Key is loaded and ready.")
        
    st.markdown("---")
    
    # --- 左下角：极简用户状态模块 ---
    if st.session_state['logged_in']:
        # 已登录：简单的圆形默认头像 + 邮箱号
        st.markdown("""
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                <div style="width: 36px; height: 36px; border-radius: 50%; background-color: #E2E8F0; color: #475569; display: flex; align-items: center; justify-content: center; font-size: 18px; margin-right: 12px;">👤</div>
                <div style="color: #1E293B; font-weight: 500; font-size: 15px;">user@gridwise.com</div>
            </div>
        """, unsafe_allow_html=True)
        # 退出按钮
        if st.button("Sign Out", use_container_width=True):
            st.session_state['logged_in'] = False
            if "token" in st.query_params:
                del st.query_params["token"]
            st.rerun()
    else:
        # 未登录：参考图片高度还原的清爽样式 (白底描边 Sign In，深底主色 Create Account)
        st.markdown("<p style='color: #64748B; font-size: 14px; margin-bottom: 10px; font-weight: 500;'>Not logged in</p>", unsafe_allow_html=True)
        # 点击模拟登录
        if st.button("Sign In", use_container_width=True):
            st.session_state['logged_in'] = True
            st.query_params["token"] = "valid"
            st.rerun()


# ==========================================
# 4. 板块 1: 地图交互看板 (挪至最上方)
# ==========================================
st.markdown("### 🗺️ ERCOT Spatial Overview")
st.markdown("Interactive visualization of regional risk levels and weather forecast nodes.")

# 地图渲染，可根据视觉效果微调 height (如 700 或 750)
render_ercot_map_workbench("http://127.0.0.1:5178", height=750)

st.markdown("---")

# ==========================================
# 5. 板块 2: 核心功能导航区 (Module Navigation)
# ==========================================
# 移除原有的 Emoji，改用干净的文本
st.markdown("<h3 style='margin-bottom: 24px;'>Core Modules</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

# 结合第一步中定义的 .mercury-card CSS 类
with col1:
    st.markdown("""
        <div class="mercury-card">
            <h4 class="mercury-card-title">Live Forecast</h4>
            <p class="mercury-card-text">
                Real-time ERCOT DAM-RTM spread predictions powered by the C1 Agent. Monitor 24-hour price trajectories and extreme weather flags.
            </p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div class="mercury-card">
            <h4 class="mercury-card-title">Trading Agent</h4>
            <p class="mercury-card-text">
                LLM-driven trading assistant utilizing DeepSeek. Formulate strategies, analyze risk, and interact via natural language.
            </p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div class="mercury-card">
            <h4 class="mercury-card-title">Model Metrics</h4>
            <p class="mercury-card-text">
                Comprehensive backtest analysis, strategy leaderboard, actual vs predicted tracking, and Explainable AI feature attribution.
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# 6. 板块 3: 系统架构说明区 (System Architecture)
# ==========================================
st.markdown("### 🧠 System Architecture")
st.markdown(
    """
    * **Prediction Engine:** `C1_XGBoost_Prediction_Agent v3`
    * **Model Architecture:** B2A XGBoost Regressor (Spread Magnitude) + B2B XGBoost Classifier (Directional Probability)
    * **Primary Strategy:** `ExtremeWeather_Only` & `B2B_Optimal_070`
    * **Target Market:** ERCOT North Hub (DAM vs RTM Arbitrage)
    * **LLM Backend:** DeepSeek Integration enabled
    """
)