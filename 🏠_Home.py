import streamlit as st
from utils import init_database
from components.agent_ui import render_global_copilot
from integration.streamlit_embed import render_ercot_map_workbench

# 1. 必须确保 layout="wide"
st.set_page_config(
    page_title="GridWise | Trading Terminal", 
    page_icon="⚡", 
    layout="wide"
)

# ==========================================
# 🌟 修复：使用 fixed 定位将 Logo 牢牢固定在侧边栏最上方
# ==========================================
st.markdown("""
    <style>
    /* 给侧边栏顶部的默认页面导航条留出足够的空间 */
    [data-testid="stSidebarNav"] {
        padding-top: 80px !important; 
    }
    
    /* 将 Logo 容器相对于浏览器窗口固定，强制在最顶端 */
    .sidebar-logo-container {
        position: fixed;
        top: 25px;
        left: 20px;
        width: 250px;
        z-index: 999999;
        display: flex;
        align-items: center;
        padding-bottom: 15px;
        border-bottom: 1px solid #E2E8F0;
    }
    </style>
""", unsafe_allow_html=True)

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
with st.sidebar:
    # --- 顶部的 GridWise 品牌标识 ---
    st.markdown("""
        <div class="sidebar-logo-container">
            <div style='background-color: #10B981; color: white; border-radius: 8px; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-size: 20px; margin-right: 12px; box-shadow: 0 2px 4px rgba(16, 185, 129, 0.3);'>
                ⚡
            </div>
            <h2 style='margin: 0; color: #1E293B; font-weight: 800; font-size: 24px; letter-spacing: -0.5px;'>GridWise</h2>
        </div>
    """, unsafe_allow_html=True)

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
st.markdown("### 🧭 Core Modules")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div style="background-color: #FFFFFF; padding: 24px; border-radius: 8px; border: 1px solid #E2E8F0; border-left: 4px solid #3B82F6; height: 200px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <h4 style="color: #1E293B; margin-top: 0;">📈 Live Forecast</h4>
            <p style="color: #64748B; font-size: 14px; line-height: 1.6;">
                Real-time ERCOT DAM-RTM spread predictions powered by the C1 Agent. Monitor 24-hour price trajectories, extreme weather flags, and confidence intervals.
            </p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div style="background-color: #FFFFFF; padding: 24px; border-radius: 8px; border: 1px solid #E2E8F0; border-left: 4px solid #10B981; height: 200px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <h4 style="color: #1E293B; margin-top: 0;">🤖 Trading Agent</h4>
            <p style="color: #64748B; font-size: 14px; line-height: 1.6;">
                LLM-driven trading assistant utilizing DeepSeek. Formulate strategies, analyze risk, and interact with the quantitative engine via natural language.
            </p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div style="background-color: #FFFFFF; padding: 24px; border-radius: 8px; border: 1px solid #E2E8F0; border-left: 4px solid #F59E0B; height: 200px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <h4 style="color: #1E293B; margin-top: 0;">🧪 Model Metrics</h4>
            <p style="color: #64748B; font-size: 14px; line-height: 1.6;">
                Comprehensive backtest analysis, strategy leaderboard, actual vs predicted tracking, and Explainable AI (SHAP) feature attribution.
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