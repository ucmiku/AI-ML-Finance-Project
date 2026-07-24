import streamlit as st
from utils import init_database
from components.agent_ui import render_global_copilot

# 1. 全局页面配置 (必须在第一行)
st.set_page_config(
    page_title="GridWise | Trading Terminal", 
    page_icon="⚡", 
    layout="wide"
)

# 2. 页面标题
st.markdown("## ⚡ GridWise Quantitative Trading Terminal")
st.markdown("Welcome to the ERCOT North Hub DAM-RTM Spread Prediction System.")
st.markdown("---")

# 3. 侧边栏：全局设置与 AI 助手
with st.sidebar:
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
        
    st.markdown("### 🗄️ System Status")
    db_status = init_database()
    st.info(db_status)

# 4. 核心功能导航区 (Module Navigation)
st.markdown("### 🧭 Core Modules")

col1, col2, col3 = st.columns(3)

# 核心技巧：在外层 div 的 style 里面加上 height: 240px; 强制对齐
with col1:
    st.markdown("""
        <div style="background-color: #162B3F; padding: 24px; border-radius: 12px; height: 200px;">
            <h4 style="color: #63B3ED; margin-top: 0;">📈 Live Forecast</h4>
            <p style="color: #90CDF4; font-size: 15px; line-height: 1.5;">
                Real-time ERCOT DAM-RTM spread predictions powered by the C1 Agent. Monitor 24-hour price trajectories, extreme weather flags, and confidence intervals.
            </p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div style="background-color: #183825; padding: 24px; border-radius: 12px; height: 200px;">
            <h4 style="color: #68D391; margin-top: 0;">🤖 Trading Agent</h4>
            <p style="color: #9AE6B4; font-size: 15px; line-height: 1.5;">
                LLM-driven trading assistant utilizing DeepSeek. Formulate strategies, analyze risk, and interact with the quantitative engine via natural language.
            </p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div style="background-color: #3C3D1C; padding: 24px; border-radius: 12px; height: 200px;">
            <h4 style="color: #F6E05E; margin-top: 0;">🧪 Model Metrics</h4>
            <p style="color: #FAF089; font-size: 15px; line-height: 1.5;">
                Comprehensive backtest analysis, strategy leaderboard, actual vs predicted tracking, and Explainable AI (SHAP) feature attribution.
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# 5. 系统架构说明区 (System Architecture)
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