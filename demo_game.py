import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import random

st.set_page_config(page_title="North Hub Survival", layout="wide", initial_sidebar_state="collapsed")

# ==========================================
# 工具函数：图片与撒钱特效引擎
# ==========================================
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()
        return f"data:image/jpeg;base64,{encoded_string}"
    except FileNotFoundError:
        return ""

def render_background(image_file, zoom=False, dark=False):
    bg_url = get_base64_image(image_file)
    if not bg_url:
        return
    transform_css = "transform: scale(3.5) translate(5%, 10%);" if zoom else "transform: scale(1);"
    filter_css = "filter: brightness(0.3) blur(2px);" if dark else ("filter: brightness(0.4) blur(2px);" if zoom else "filter: brightness(0.8);")
    st.markdown(f"""
    <style>
    .stApp::before {{
        content: ""; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background-image: url("{bg_url}"); background-size: cover; background-position: center;
        z-index: -999; transition: transform 2.5s ease-in-out, filter 2s ease;
        {transform_css} {filter_css}
    }}
    </style>
    """, unsafe_allow_html=True)

# 专属：定制化漫天撒金币特效
def falling_coins_effect():
    css = """
    <style>
    .coin-container {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        pointer-events: none; z-index: 9999; overflow: hidden;
    }
    .coin {
        position: absolute; top: -50px;
        animation-name: fallAndSpin;
        animation-timing-function: linear;
        animation-iteration-count: infinite;
    }
    @keyframes fallAndSpin {
        0% { transform: translateY(-50px) rotate(0deg); opacity: 1; }
        100% { transform: translateY(110vh) rotate(360deg); opacity: 0.5; }
    }
    </style>
    <div class="coin-container">
    """
    emojis = ['💰', '🪙', '✨', '💵']
    for _ in range(45):
        emoji = random.choice(emojis)
        left = random.randint(0, 100)
        duration = random.uniform(2.5, 6.0) # 下落速度随机
        delay = random.uniform(0, 3.0)      # 延迟出现随机
        size = random.randint(25, 50)       # 图标大小随机
        css += f'<div class="coin" style="left: {left}%; font-size: {size}px; animation-duration: {duration}s; animation-delay: {delay}s;">{emoji}</div>'
    css += "</div>"
    st.markdown(css, unsafe_allow_html=True)

# ==========================================
# 核心全局 CSS 引擎
# ==========================================
st.markdown("""
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: transparent !important; }
#MainMenu {visibility: hidden;} header {visibility: hidden;} footer {visibility: hidden;}
.block-container {padding: 0rem; max-width: 100%;}

/* 覆盖 Streamlit Primary 按钮的默认荧光色 */
button[data-testid="baseButton-primary"] {
    background-color: #2C3E50 !important; /* 沉稳的深蓝灰 */
    color: #E2E8F0 !important;            /* 柔和的灰白色 */
    border: 1px solid #4A5568 !important; /* 细腻的边框 */
    transition: all 0.3s ease;
}

button[data-testid="baseButton-primary"]:hover {
    background-color: #34495E !important; 
    border-color: #3498DB !important;     /* 悬浮时带出一点专业金融蓝 */
    box-shadow: 0 0 10px rgba(52, 152, 219, 0.4) !important;
}

/* === Scene 0: 高级动态加载页面（增强版） === */
.scene0-bg {
    position: fixed; 
    top: 0; left: 0; width: 100vw; height: 100vh;
    /* 保持科幻地球背景图 */
    background-image: url('https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop');
    background-size: cover;
    background-position: center;
    z-index: -1001;
    /* 【修改】：时间从40s缩短到18s，让你明显看到它在平移和呼吸 */
    animation: slowPan 18s ease-in-out infinite alternate;
}

.scene0-overlay {
    position: fixed; 
    top: 0; left: 0; width: 100vw; height: 100vh;
    /* 【修改】：大幅降低遮罩的黑暗度，让背后的地球亮出来！保留极轻微的扫描线 */
    background: linear-gradient(to bottom, rgba(10, 5, 20, 0.1), rgba(0, 0, 0, 0.45)),
                repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0, 230, 118, 0.03) 2px, rgba(0, 230, 118, 0.03) 4px);
    z-index: -1000;
}

@keyframes slowPan {
    0% { transform: scale(1) translate(0, 0); }
    /* 【修改】：加大放大倍率（1.25倍）和平移距离（-4%），增强景深漫游感 */
    100% { transform: scale(1.25) translate(-4%, -4%); }
}

.alert-box {
    margin-top: 5vh; text-align: center; 
    animation: pulseAlert 2s infinite, floatDown 1s ease-out forwards;
}

@keyframes floatDown {
    from { opacity: 0; transform: translateY(-30px); }
    to { opacity: 1; transform: translateY(0); }
}

/* 首页规则终端卡片 - 保持高级毛玻璃质感 */
.rule-card {
    background: rgba(15, 20, 30, 0.45); 
    backdrop-filter: blur(12px);        
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 82, 82, 0.4);
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.8), inset 0 0 20px rgba(255, 82, 82, 0.1);
    border-radius: 12px;
    padding: 25px 35px;
    margin: 20px auto;
    max-width: 900px;
    color: #E0E0E0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    animation: floatUp 1.2s ease-out forwards;
    opacity: 0; 
}

@keyframes floatUp {
    from { opacity: 0; transform: translateY(40px); }
    to { opacity: 1; transform: translateY(0); }
}

/* 首页规则终端卡片 */
.rule-card {
    background: rgba(15, 20, 30, 0.85);
    border: 1px solid #FF5252;
    box-shadow: 0 0 15px rgba(255, 82, 82, 0.2);
    border-radius: 10px;
    padding: 20px 30px;
    margin: 20px auto;
    max-width: 900px;
    color: #E0E0E0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.rule-card h4 {
    margin-top: 10px;
    margin-bottom: 8px;
    font-weight: 600;
}

/* === 引导小手 === */
.pointer-hand {
    font-size: 50px; text-align: center; opacity: 0;
    animation: fadeInHand 0.5s forwards 0.5s, bounceHand 1s infinite alternate 0.5s;
    text-shadow: 0 0 10px rgba(255,255,255,0.5);
    margin-top: -10px; margin-bottom: 20px;
}
@keyframes fadeInHand { to { opacity: 1; } }
@keyframes bounceHand { 0% { transform: translateY(15px); } 100% { transform: translateY(0px); } }

/* 通用 UI 组件 */
.dialogue-box {
    position: fixed; bottom: 3%; left: 10%; width: 80%;
    background: rgba(10, 15, 25, 0.95); border: 1px solid #444; border-radius: 8px;
    padding: 20px 30px; z-index: 100; box-shadow: 0 4px 20px rgba(0,0,0,0.8);
    color: #FFF; font-size: 18px; line-height: 1.5;
}
.content-wrapper { background: rgba(0, 0, 0, 0.85); padding: 30px; border-radius: 10px; margin-top: 30px; border: 1px solid #333; }

/* 科幻风数据解释框 */
.info-box {
    background: rgba(0, 230, 118, 0.08); border-left: 4px solid #00E676;
    padding: 15px 25px; margin-bottom: 20px; border-radius: 0 5px 5px 0;
    font-family: 'Courier New', Courier, monospace; color: #E0E0E0; font-size: 15px;
}

/* 结算特效：纯正金黄色辉光 */
.golden-glow {
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    background: radial-gradient(circle, rgba(255, 235, 59, 0.15) 0%, rgba(255, 215, 0, 0.4) 100%);
    box-shadow: inset 0 0 150px rgba(255, 215, 0, 0.6); mix-blend-mode: color-dodge; z-index: -10; pointer-events: none;
    animation: goldPulse 2s infinite alternate;
}
@keyframes goldPulse { 0% {opacity: 0.6;} 100% {opacity: 1;} }

.freeze-effect {
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    background: radial-gradient(circle, rgba(173,216,230,0.1) 0%, rgba(0,0,139,0.6) 100%);
    box-shadow: inset 0 0 200px rgba(255,255,255,0.7); backdrop-filter: grayscale(80%) blur(1px); z-index: -10; pointer-events: none;
}
</style>
""", unsafe_allow_html=True)

if 'scene' not in st.session_state:
    st.session_state.scene = 0 

hours_str = [f"{i:02d}:00" for i in range(24)]
true_spreads = np.array([1.5, -1.0, 0.5, 0.0, -2.0, 3.5, 8.0, 4.0, 1.5, -3.0, 0.0, 5.0, 15.0, 80.0, 350.0, 650.0, 720.0, 580.0, 200.0, 45.0, 10.0, 2.0, -1.0, 0.5])
ai_profit = np.sum(true_spreads[np.argsort(true_spreads)[-3:]])
temps_2m = np.array([5, 3, 1, -2, -5, -8, -10, -12, -14, -15, -16, -15, -13, -11, -10, -12, -15, -18, -20, -19, -17, -15, -12, -10])
wind_speed_10m = np.array([5, 6, 8, 10, 12, 15, 18, 20, 15, 10, 8, 5, 4, 3, 2, 2, 1, 1, 2, 3, 5, 6, 8, 10])
wind_gusts_10m = wind_speed_10m * 1.5
precipitation = np.array([0, 0, 0, 2, 5, 10, 15, 25, 20, 10, 5, 2, 0, 0, 0, 5, 15, 25, 10, 5, 0, 0, 0, 0])
humidity_2m = np.array([60, 65, 70, 75, 80, 85, 90, 95, 95, 90, 85, 80, 75, 70, 75, 80, 85, 90, 95, 95, 90, 85, 80, 75])

# ==========================================
# 场景 0：全新升级的高级加载与交易指南首页 (全英文修复版)
# ==========================================
if st.session_state.scene == 0:
    st.markdown("<div class='scene0-bg'></div>", unsafe_allow_html=True)
    st.markdown("<div class='scene0-overlay'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='alert-box'>
        <h1 style='color: #FF1744; font-size: 42px; font-family: monospace;'>⚠️ WARNING: GRID EMERGENCY SIMULATION</h1>
        <h3 style='color: #FF5252; letter-spacing: 2px;'>ERCOT North Hub Virtual Trading Terminal</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # 修复了空行导致的解析bug，并全部替换为纯英文
    st.markdown("""
    <div class='rule-card'>
        <h4 style='color: #00E676;'>⚡ BACKGROUND: WHAT IS VIRTUAL TRADING?</h4>
        <p style='font-size: 14px; line-height: 1.6;'>
            In the Texas <b>ERCOT</b> market, traders don't need physical power plants to arbitrage between the <b>Day-Ahead</b> and <b>Real-Time</b> markets:<br>
            • <b>Mechanism:</b> Bids must be submitted before the 10:00 AM deadline the day prior.<br>
            • <b>Settlement:</b> You buy power at the locked Day-Ahead price and sell it simultaneously at the actual Real-Time price.<br>
            • <b>The Goal:</b> If extreme weather causes Real-Time prices to spike, you capture the massive price spread!
        </p>
        <h4 style='color: #00BFFF; margin-top: 15px;'>🎮 RULES & YOUR MISSION</h4>
        <ol style='font-size: 14px; line-height: 1.6; margin-bottom: 10px; padding-left: 20px;'>
            <li>Analyze the terminal's <b>24-hour multidimensional weather forecast</b> (Temp, Wind, Rain, Humidity).</li>
            <li>Predict when the grid will stress the most and <b>lock in exactly 3 hours</b>.</li>
            <li>Execute your trades and see if human intuition can beat our <b>AI Machine Learning Model</b>!</li>
        </ol>
        <p style='font-size: 13px; color: #FFD700; background: rgba(255, 215, 0, 0.1); padding: 8px 12px; border-radius: 5px; margin-top: 15px;'>
            💡 <b>WHY DO WE NEED AN AI TRADING MODEL?</b><br>
            Power prices during extreme weather are highly non-linear. Human intuition easily misjudges complex meteorological data, leading to massive losses. This simulation will show you exactly why our Machine Learning model is essential for capturing subtle patterns and maximizing returns!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2,1,2])
    with col2:
        if st.button("INITIALIZE SIMULATION", type="primary", use_container_width=True):
            st.session_state.scene = 1
            st.rerun()
        st.markdown("<div class='pointer-hand'>👆</div>", unsafe_allow_html=True)

# ==========================================
# 场景 1
# ==========================================
elif st.session_state.scene == 1:
    render_background("room.jpg", zoom=False)
    
    st.markdown("""
    <div class='dialogue-box'>
        <b>[System Alert] 9:00 AM - DAY AHEAD</b><br><br>
        A once-in-a-century winter storm is bearing down on the North Hub.<br>
        You need to review the <b>multi-dimensional weather forecast</b> and place Day-Ahead Virtual Bids.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top: 5vh;'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
         if st.button("Access Weather & Trading Terminal 💻", use_container_width=True):
             st.session_state.scene = 2
             st.rerun()
         st.markdown("<div class='pointer-hand'>👆</div>", unsafe_allow_html=True)

# ==========================================
# 场景 2：天气分析与交易执行终端
# ==========================================
elif st.session_state.scene == 2:
    # 确保初始化选中时间的列表
    if 'selected_hours' not in st.session_state:
        st.session_state.selected_hours = []

    render_background("room.jpg", zoom=True)
    st.markdown("<div class='content-wrapper'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #00BFFF; text-align: center; margin-bottom: 20px;'>ERCOT ADVANCED METEOROLOGICAL TERMINAL (FEB 13)</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <b>💡 SYSTEM GUIDE: How Weather Impacts Grid Prices</b><br>
        • <b>Temperature 🌡️:</b> Deep freezes spike heating demand and can freeze natural gas pipelines (low supply, high demand).<br>
        • <b>Wind 💨:</b> Normally produces cheap energy, but extreme ice storms can freeze wind turbines, crashing the power supply.<br>
        • <b>Precipitation ❄️:</b> Freezing rain and snow cover solar panels and weigh down power lines, causing local outages.<br>
        • <b>Humidity 💧:</b> High humidity accelerates dangerous ice formation on grid infrastructure.
    </div>
    """, unsafe_allow_html=True)
    
    # 任务提示框
    st.markdown("""
    <div style='background: rgba(255, 215, 0, 0.15); border-left: 4px solid #FFD700; padding: 15px 25px; margin-bottom: 20px; border-radius: 0 5px 5px 0;'>
        <h4 style='color: #FFD700; margin: 0 0 10px 0; font-family: monospace;'>⚡ MISSION: PREDICT THE PRICE SPIKE</h4>
        <p style='color: #E0E0E0; margin: 0; font-size: 15px;'>
            Your goal is to find the breaking points of the grid. <br>
            <b>👉 ACTION: Click the hour buttons on the timeline below</b> to lock in exactly <b>3 hours</b> where you predict electricity prices will <b>SKYROCKET!</b>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 绘制气象图表（去掉之前的点击交互，恢复纯展示模式）
    # 绘制气象图表
    fig_weather = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 温度线：保持原样（原本的亮蓝色比较清晰）
    fig_weather.add_trace(go.Scatter(x=hours_str, y=temps_2m, mode='lines+markers', name='Temp 2m (°C)', line=dict(color='#00BFFF', width=3)), secondary_y=False)
    
    # 🌟 修改 1：加粗湿度线，提高不透明度至 0.8，改为更显眼的虚线
    fig_weather.add_trace(go.Scatter(x=hours_str, y=humidity_2m, mode='lines', name='Rel. Humidity (%)', line=dict(color='rgba(255,255,255,0.8)', width=2, dash='dash')), secondary_y=False)
    
    # 🌟 修改 2：将降水柱状图改为带点蓝调的半透明色，提高辨识度
    fig_weather.add_trace(go.Bar(x=hours_str, y=precipitation, name='Precipitation (mm)', marker_color='rgba(135, 206, 250, 0.4)'), secondary_y=True)
    
    # 风速线：保持原样
    fig_weather.add_trace(go.Scatter(x=hours_str, y=wind_speed_10m, mode='lines', name='Wind Spd (m/s)', line=dict(color='#FFA500', width=2)), secondary_y=True)
    
    fig_weather.update_layout(
        template="plotly_dark", 
        # 🌟 修改 3：给图表增加一个深色半透明背景，隔绝背后的杂乱图片
        plot_bgcolor='rgba(15, 20, 30, 0.75)', 
        paper_bgcolor='rgba(15, 20, 30, 0.75)', 
        height=350, 
        margin=dict(t=10, b=10, l=10, r=10), 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        # 🌟 修改 4：强制图表内部的所有坐标系和图例文字为纯白色，略微放大
        font=dict(color="#FFFFFF", size=13)
    )
    st.plotly_chart(fig_weather, use_container_width=True)
    
    # ==========================================
    # 🌟 全新设计的底部交互式时间轴按钮组
    # ==========================================
    st.markdown(f"<h4 style='color: #DDD; text-align: center; margin-top: 10px;'>TARGET HOURS SELECTED: {len(st.session_state.selected_hours)} / 3</h4>", unsafe_allow_html=True)
    
    # 使用 24 列来平均分配按钮，模拟 X 轴
    cols = st.columns(24)
    for i, hour in enumerate(hours_str):
        is_selected = hour in st.session_state.selected_hours
        
        # 根据选中状态改变按钮颜色 (primary = 红色/高亮, secondary = 默认灰色)
        btn_type = "primary" if is_selected else "secondary"
        
        with cols[i]:
            # 为了在一排显示下，按钮文字缩简为小时数，如 "00", "01"
            display_hour = f"{i:02d}" 
            if st.button(display_hour, key=f"btn_{hour}", type=btn_type, use_container_width=True):
                # 切换选中状态逻辑
                if is_selected:
                    st.session_state.selected_hours.remove(hour)
                    st.rerun()
                else:
                    if len(st.session_state.selected_hours) < 3:
                        st.session_state.selected_hours.append(hour)
                        st.rerun()
                    else:
                        st.toast("⚠️ Maximum 3 hours reached! Please deselect one first.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # ==========================================
    # 🌟 选满 3 个后自动弹出执行面板
    # ==========================================
    if len(st.session_state.selected_hours) == 3:
        st.success("✅ Targets Locked. Ready for execution.")
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            if st.button("EXECUTE BIDS", type="primary", use_container_width=True):
                # 记录玩家选择的下标和利润，用于结算界面
                st.session_state.user_picks_idx = [hours_str.index(h) for h in st.session_state.selected_hours]
                st.session_state.user_profit = sum([true_spreads[i] for i in st.session_state.user_picks_idx])
                st.session_state.scene = 3
                st.rerun()
            st.markdown("<div class='pointer-hand'>👆</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
# ==========================================
# 场景 3
# ==========================================
elif st.session_state.scene == 3:
    render_background("snow.jpg", dark=True)
    is_win = st.session_state.user_profit >= (ai_profit * 0.7)

    if is_win:
        falling_coins_effect() 
        st.markdown("<div class='golden-glow'></div>", unsafe_allow_html=True) 
    else:
        st.snow() 
        st.markdown("<div class='freeze-effect'></div>", unsafe_allow_html=True) 
    
    st.markdown("<div class='content-wrapper'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: white;'>MARKET SETTLEMENT REPORT</h2>", unsafe_allow_html=True)
    
    score_col1, score_col2 = st.columns(2)
    with score_col1:
        color = "#00E676" if is_win else "#FF5252"
        st.markdown(f"<h3 style='color: {color};'>👤 Human Trader: ${st.session_state.user_profit:.2f}</h3>", unsafe_allow_html=True)
    with score_col2:
        st.markdown(f"<h3 style='color: #FFD700;'>🤖 AI Model (Simulation): ${ai_profit:.2f}</h3>", unsafe_allow_html=True)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours_str, y=true_spreads, mode='lines', name='Actual Spread', line=dict(color='rgba(255,255,255,0.7)', width=2)))
    fig.add_trace(go.Scatter(
        x=[hours_str[i] for i in st.session_state.user_picks_idx],
        y=[true_spreads[i] for i in st.session_state.user_picks_idx],
        mode='markers', name='Your Trades', marker=dict(color='#FF5252', size=16, symbol='cross')
    ))
    ai_picks_idx = np.argsort(true_spreads)[-3:]
    fig.add_trace(go.Scatter(
        x=[hours_str[i] for i in ai_picks_idx],
        y=[true_spreads[i] for i in ai_picks_idx],
        mode='markers', name='AI Trades', marker=dict(color='#FFD700', size=18, symbol='star', line=dict(color='white', width=1))
    ))
    fig.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height: 180px; width: 100%;'></div>", unsafe_allow_html=True)
    
    dialogue_text = (
        "<b>[Victory]</b> Incredible intuition! You successfully navigated the chaos today. But human intuition doesn't scale, and fatigue makes us prone to errors. <br><br><b>How do we automate this success and execute it with mathematical precision every single day? Let us show you our real Machine Learning Model...</b>" 
        if is_win else 
        "<b>[System Failure]</b> Your intuition was overwhelmed by the complexity of the data. This is exactly the limitation of human trading during extreme events.<br><br><b>But don't worry, this is exactly why we built our solution. Are you ready to see how our real Machine Learning Model solves this? Let's dive in...</b>"
    )
    
    st.markdown(f"<div class='dialogue-box'>{dialogue_text}</div>", unsafe_allow_html=True)
    
    st.markdown("<div style='position: fixed; bottom: 4%; right: 2%; z-index: 100;'>", unsafe_allow_html=True)
    if st.button("REBOOT GAME"):
        st.session_state.scene = 0
        st.rerun()
    st.markdown("<div class='pointer-hand' style='font-size: 30px;'>👆</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)