import streamlit as st
from utils import init_database

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

# 3. 侧边栏：全局设置
with st.sidebar:
    st.markdown("### ⚙️ Global Settings")
    
    # 核心修改：设置 value 为你的 API Key，并保持 type="password"
    api_key = st.text_input(
        "DeepSeek API Key:", 
        value="sk-8080fcbf46f3459895cd3f8daed48535", # 默认填充
        type="password",                            # 以密码形式隐藏
        help="Required for Agent and Strategy Generation"
    )
    
    # 只要 api_key 有值（包括默认值），页面一加载就会自动存入全局状态
    if api_key:
        st.session_state['api_key'] = api_key
        st.success("✅ API Key is loaded and ready.")
        
    st.markdown("### 🗄️ System Status")
    db_status = init_database()
    st.info(db_status)

# 4. 导航提示
st.info("👈 Please select a module from the sidebar to begin.")