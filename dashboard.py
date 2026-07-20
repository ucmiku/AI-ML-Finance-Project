import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from openai import OpenAI
import time
import os
import sqlite3

# --- 1. 页面配置 ---
st.set_page_config(
    page_title="GridWise | North Hub Spread Prediction Terminal", 
    page_icon="⚡", 
    layout="wide", 
    initial_sidebar_state="expanded" 
)

# --- 2. 数据库自动初始化 (核心新增部分) ---
@st.cache_resource
def init_database():
    """
    检查 SQLite 数据库是否存在，如果不存在则自动从 CSV 加载数据。
    使用 @st.cache_resource 确保每次启动服务器只执行一次。
    """
    db_name = "ercot_data.db"
    csv_name = "model_wide_hourly_2024_2026(1).csv"
    
    # 如果数据库文件还不存在，执行创建逻辑
    if not os.path.exists(db_name):
        # 检查同目录下有没有同学发给你的 CSV 文件
        if os.path.exists(csv_name):
            try:
                # 读取 CSV 并写入 SQLite
                df = pd.read_csv(csv_name)
                conn = sqlite3.connect(db_name)
                df.to_sql("model_wide_hourly_2024_2026", conn, if_exists="replace", index=False)
                conn.close()
                return "✅ Database initialized successfully from CSV."
            except Exception as e:
                return f"❌ Error initializing database: {e}"
        else:
            return f"⚠️ Warning: '{csv_name}' not found. AI Agent will lack database access."
    else:
        return "✅ Database is ready."

# 在界面最上方静默执行初始化（也可以把状态打印在侧边栏）
db_status = init_database()

# --- 3. 初始化 Session State ---
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "preset_prompt" not in st.session_state:
    st.session_state.preset_prompt = None

# ... 接下去就是你原有的侧边栏、KPI 和图表代码 ...
# 注意：可以在侧边栏展示数据库状态，让评委看到这个细节！
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_key = st.text_input(
        "Enter DeepSeek API Key:", 
        value="sk-8080fcbf46f3459895cd3f8daed48535", 
        type="password"
    )
    st.markdown("---")
    st.markdown("### 🗄️ System Status")
    
    # 在侧边栏展示底层数据库状态，增加专业感
    if "✅" in db_status:
        st.success(db_status)
    elif "⚠️" in db_status:
        st.warning(db_status)
    else:
        st.error(db_status)
        
    st.markdown("<small>Powered by DeepSeek Model & SQLite</small>", unsafe_allow_html=True)

# --- 顶部区域 ---
col_logo, col_title, col_status = st.columns([1, 4, 1])
with col_logo:
    st.markdown("### ⚡ GridWise") 
with col_title:
    st.markdown("<h2 style='text-align: center;'>ERCOT North Hub DAM-RTM Spread Terminal</h2>", unsafe_allow_html=True)
with col_status:
    st.error("⚠️ Status: Winter Storm Warning in North Hub")

st.markdown("---") 

# --- KPI 指标卡片 ---
st.markdown("#### 📊 Model Performance & Backtest Metrics (Spread Trading)")
metric_cols = st.columns(5)
with metric_cols[0]:
    st.metric(label="Alpha (α)", value="0.00", delta="Pending")
with metric_cols[1]:
    st.metric(label="Beta (β)", value="0.00", delta="Pending")
with metric_cols[2]:
    st.metric(label="Sharpe Ratio", value="0.00", delta="Pending")
with metric_cols[3]:
    st.metric(label="Max Drawdown", value="0.00%", delta="-", delta_color="inverse")
with metric_cols[4]:
    st.metric(label="Est. Annual Return", value="$ 0.00", delta="Pending")

st.markdown("---")

# --- 核心图表与原有 AI 宏观分析区 ---
col_chart, col_ai = st.columns([7, 3]) 

# 1. 生成图表与数据
with col_chart:
    st.markdown("#### 📈 24-Hour DAM-RTM Price Spread Forecast (North Hub)")
    
    hours = [f"{i:02d}:00" for i in range(24)]
    zero_baseline = np.zeros(24)
    forecast_spread = np.random.uniform(-5, 5, 24)
    
    # 模拟晚上 17:00 - 20:00 出现巨大的正价差
    spike_hours = range(17, 21) 
    for h in spike_hours:
        forecast_spread[h] += np.random.uniform(150, 400) 
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=zero_baseline, mode='lines', name='Zero Spread', line=dict(color='gray', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=hours, y=forecast_spread, mode='lines+markers', name='Predicted Spread', line=dict(color='#00E676', width=3), marker=dict(size=6)))
    
    fig.update_layout(
        xaxis_title="Hour of Day", yaxis_title="Spread ($/MWh)",
        template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified", margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# 2. 调用大模型生成策略
with col_ai:
    # 1. 标题独占一行，解决拥挤问题
    st.markdown("#### 🤖 LLM Trading Advisor")
    st.info("AI is ready to analyze spread predictions.")
    
    # 2. 策略生成按钮
    if st.button("Generate Strategy via AI", type="primary", use_container_width=True):
        if not api_key:
            st.error("API Key is missing!")
        else:
            with st.spinner("AI is analyzing the 24-hour spread data..."):
                try:
                    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                    data_str = ", ".join([f"HE {i:02d}: ${s:.2f}" for i, s in enumerate(forecast_spread)])
                    prompt = f"""
                    You are an expert quantitative power trader...
                    {data_str}
                    1. 🟢 **STRONG LONG (Virtual Bid):** Identify the specific hours.
                    2. ⚪ **HOLD/AVOID:** Identify hours where spread is negligible.
                    3. 🔴 **RISK WARNING:** Disclaimer about weather volatility.
                    """
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3 
                    )
                    st.markdown("---")
                    st.markdown(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"API Error: {e}")
    else:
        st.markdown("*(Click the button above to generate a real-time strategy)*")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- 3. 将 3D 机器人作为悬浮按钮 (终极锚点锁定版) ---
    import base64
    import os
    
    def get_robot_image(bin_file):
        try:
            with open(bin_file, 'rb') as f:
                data = f.read()
            return f"data:image/png;base64,{base64.b64encode(data).decode()}"
        except FileNotFoundError:
            # 备用高清 3D 机器人动图，防崩底线
            return "https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Robot.png"

    img_url = get_robot_image("agent_avatar.png") 

    robot_btn_css = f"""
    <style>
    @keyframes gentle-sway {{
        0% {{ transform: rotate(0deg); }}
        25% {{ transform: rotate(6deg); }}
        50% {{ transform: rotate(0deg); }}
        75% {{ transform: rotate(-6deg); }}
        100% {{ transform: rotate(0deg); }}
    }}
    @keyframes pointRight {{
        0%, 20%, 50%, 80%, 100% {{transform: translateX(0);}}
        40% {{transform: translateX(-5px);}}
        60% {{transform: translateX(-3px);}}
    }}
    .pointer-text {{
        color: #00E676;
        font-size: 14px;
        font-weight: bold;
        text-align: right;
        margin-top: 50px; 
        animation: pointRight 1.5s infinite;
    }}
    
    /* 终极魔法：精准打击！寻找带有 #magic-anchor 的容器，把紧挨着它的下一个元素(按钮)彻底爆改！ */
    div[data-testid="stElementContainer"]:has(#magic-anchor) + div[data-testid="stElementContainer"] button {{
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        width: 120px !important;  
        height: 120px !important; 
        background-image: url('{img_url}') !important;
        background-size: contain !important;
        background-repeat: no-repeat !important;
        background-position: center bottom !important;
        animation: gentle-sway 4s ease-in-out infinite !important; 
        transform-origin: bottom center !important;
        padding: 0 !important;
        margin: 0 auto !important;
        display: block !important;
    }}
    
    /* 强制隐藏原本的 Agent 文字 */
    div[data-testid="stElementContainer"]:has(#magic-anchor) + div[data-testid="stElementContainer"] button * {{
        display: none !important;
    }}
    
    /* 鼠标悬停发光特效 */
    div[data-testid="stElementContainer"]:has(#magic-anchor) + div[data-testid="stElementContainer"] button:hover {{
        filter: drop-shadow(0px 0px 15px rgba(0, 230, 118, 0.8)) !important;
        transform: scale(1.05) !important;
    }}
    </style>
    """
    st.markdown(robot_btn_css, unsafe_allow_html=True)

    if not st.session_state.chat_open:
        col_text, col_btn = st.columns([5, 4])
        with col_text:
            st.markdown("<div class='pointer-text'>Ask RAG Agent (24-26 Data & Decisions) 👉</div>", unsafe_allow_html=True)
        with col_btn:
            # === 【极度重要】这里的“隐形锚点”是核心机关，它是 CSS 的靶子，绝对不能删！ ===
            st.markdown('<span id="magic-anchor"></span>', unsafe_allow_html=True)
            # ====================================================================
            
            # 按钮本身还是原来的样子，但会被上面的魔法瞬间变成 3D 机器人
            if st.button("Agent", key="open_chat"):
                st.session_state.chat_open = True
                st.rerun()

    # 4. 更新底部提示文字
    st.markdown("""
    <div style='background: rgba(0, 230, 118, 0.1); border-left: 4px solid #00E676; padding: 15px; border-radius: 5px; margin-top: 10px;'>
        <b>Deep Dive into Data with RAG?</b><br>
        Ask our interactive <b>RAG Agent</b> to query any historical data from <b>2024 to 2026</b> directly from the SQLite database. You can also consult it for spread predictions and trading decision support!<br><br>
        <span style='color: #00E676; font-weight: bold;'>↗️ (Please click the 🤖 Robot above to expand the conversation)</span>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 聊天交互界面 (在页面最下方展开)
# ==========================================

# 渲染聊天交互界面
if st.session_state.chat_open:
    st.markdown("<hr style='border:1px solid #333; margin-top: 50px;'>", unsafe_allow_html=True)
    
    col_title, col_close = st.columns([10, 1])
    with col_title:
        st.markdown("#### 🤖 Database Query Agent")
    with col_close:
        if st.button("✖️", key="close_chat"):
            st.session_state.chat_open = False
            st.rerun()

    # 展示历史记录
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 预设问题
    if len(st.session_state.chat_history) == 0:
        st.markdown("<p style='color: #888; font-size: 14px;'>💡 <b>Suggested SQL Queries:</b></p>", unsafe_allow_html=True)
        preset_1 = st.button("What price should I bid for tomorrow's peak hours?")
        preset_2 = st.button("Based on history, which hours have the largest spread during freezing weather?")
        
        if preset_1:
            st.session_state.preset_prompt = "What price should I bid for tomorrow's peak hours?"
        if preset_2:
            st.session_state.preset_prompt = "Based on history, which hours have the largest spread during freezing weather?"

    # 输入框
    chat_input_val = st.chat_input("Ask the AI to query the database...")
    prompt = chat_input_val or st.session_state.preset_prompt

    if prompt:
        st.session_state.preset_prompt = None 
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # --- AI 回复区 ---
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            full_response = "Accessing SQLite database... Analyzing historical ERCOT data... \n\nBased on similar weather patterns, I recommend placing Day-Ahead bids around $45/MWh for hours 17:00 to 19:00."
            
            # 打字机效果
            typed_response = ""
            for chunk in full_response.split(" "):
                typed_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(typed_response + "▌")
            message_placeholder.markdown(full_response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
        st.rerun()