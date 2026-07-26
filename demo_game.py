import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import os

# ==========================================
# 0. 工具函数与配置
# ==========================================
st.set_page_config(page_title="GridWise Climate Game", layout="wide", initial_sidebar_state="collapsed")

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode()
        ext = image_path.split('.')[-1]
        mime = "image/png" if ext.lower() == "png" else "image/jpeg"
        return f"data:{mime};base64,{encoded_string}"
    return ""

def render_background(image_path):
    bg_url = get_base64_image(image_path)
    if bg_url:
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("{bg_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """, unsafe_allow_html=True)

# 初始化 Session State
if 'scene' not in st.session_state:
    st.session_state.scene = 0
if 'selected_hours' not in st.session_state:
    st.session_state.selected_hours = []
# 🌟 核心：专门用于控制切页动效的计数器
if 'anim_counter' not in st.session_state:
    st.session_state.anim_counter = 0

# 回调函数
def change_scene(new_scene):
    st.session_state.scene = new_scene
    st.session_state.anim_counter += 1  # 切换场景，触发动画

def toggle_hour(hour):
    if hour in st.session_state.selected_hours:
        st.session_state.selected_hours.remove(hour)
    elif len(st.session_state.selected_hours) < 3:
        st.session_state.selected_hours.append(hour)
    # 注意：选时间不增加计数器，保证页面不闪烁！

def settle_market():
    st.session_state.user_picks_idx = [hours_str.index(h) for h in st.session_state.selected_hours]
    st.session_state.user_profit = sum([true_spreads[i] for i in st.session_state.user_picks_idx])
    st.session_state.scene = 3
    st.session_state.anim_counter += 1  # 切换场景，触发动画

def restart_game():
    st.session_state.selected_hours = []
    st.session_state.scene = 0
    st.session_state.anim_counter += 1  # 切换场景，触发动画

# ==========================================
# 1. 核心全局 CSS & 🌟 动态转场动效注入
# ==========================================
# 动态获取当前计数器，强制浏览器认为这是一个全新的动画！
t_id = st.session_state.anim_counter

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=Inter:wght@400;600;700&display=swap');

#MainMenu {{visibility: hidden;}} header {{visibility: hidden;}} footer {{visibility: hidden;}}

/* 🌟 全局容器从暗到亮的交错动效 (时间拉长至 2.5s，更从容优雅) */
@keyframes globalFadeAndBrighten_{t_id} {{
    0% {{ opacity: 0; filter: brightness(0.2); }}
    100% {{ opacity: 1; filter: brightness(1); }}
}}
.block-container {{ 
    padding-top: 3rem; padding-bottom: 3rem; 
    animation: globalFadeAndBrighten_{t_id} 2.5s cubic-bezier(0.2, 0.8, 0.2, 1) forwards !important;
}}

/* 🌟 内部元素的优雅上浮动效 (时间拉长至 2.0s) */
@keyframes elegantFadeUp_{t_id} {{
    0% {{ opacity: 0; transform: translateY(40px); }}
    100% {{ opacity: 1; transform: translateY(0); }}
}}
.animate-fade-up {{
    animation: elegantFadeUp_{t_id} 2.0s cubic-bezier(0.2, 0.8, 0.2, 1) forwards !important;
    opacity: 0;
}}

/* 延迟层级同步拉长，保持错落有致的呼吸感 */
.delay-1 {{ animation-delay: 0.4s; opacity: 0; animation-fill-mode: forwards; }}
.delay-2 {{ animation-delay: 0.8s; opacity: 0; animation-fill-mode: forwards; }}

/* 卡片样式 */
.glass-card {{
    background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border-radius: 16px; padding: 40px; max-width: 550px; margin: 4vh auto 30px auto;
    box-shadow: 0 15px 45px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.6);
    text-align: center; border: 1px solid rgba(255, 255, 255, 0.5);
}}

.glass-card-wide {{
    background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border-radius: 16px; padding: 30px; max-width: 950px; margin: 2vh auto 30px auto;
    box-shadow: 0 15px 45px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,0.6);
    border: 1px solid rgba(255, 255, 255, 0.6);
}}

.glass-card-full {{
    background: rgba(255, 255, 255, 0.9); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border-radius: 12px; padding: 25px 30px; margin: 0 0 20px 0; width: 100%;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05), inset 0 1px 0 rgba(255,255,255,0.6);
    border: 1px solid rgba(255, 255, 255, 0.6);
}}

.bulletin-board {{
    background-color: #F8F7F2; border: 1px solid #D6D3C4; border-radius: 12px; padding: 25px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08), inset 0 0 40px rgba(139,115,85,0.02); margin-bottom: 20px;
}}

/* 字体排版 */
.glass-card h1, .glass-card-wide h1, .glass-card-full h1 {{ font-family: 'Lora', serif; color: #111827; font-size: 38px; margin-bottom: 15px; line-height: 1.2; }}
.glass-card h3, .glass-card-wide h3 {{ font-family: 'Inter', sans-serif; font-size: 13px; font-weight: 700; color: #047857; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 10px; }}
.glass-card p, .glass-card-wide p {{ font-family: 'Inter', sans-serif; font-size: 16px; color: #374151; line-height: 1.6; }}

/* 按钮样式 */
div[data-testid="stButton"] > button {{
    border-radius: 6px !important; padding: 6px 0 !important; font-size: 12px !important; font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important; white-space: nowrap !important; min-width: 100% !important; transition: all 0.3s ease !important;
    border: 1px solid #D6D3C4 !important;
}}

div[data-testid="stButton"] > button[data-testid="baseButton-secondary"] {{
    background: #FFFFFF !important; color: #44403C !important; box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
}}

div[data-testid="stButton"] > button[data-testid="baseButton-primary"] {{
    background: #047857 !important; color: #FFFFFF !important; border-color: #047857 !important;
    box-shadow: 0 4px 12px rgba(4, 120, 87, 0.3) !important;
}}

div[data-testid="stButton"] > button:hover {{ transform: translateY(-2px); border-color: #047857 !important; }}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. 核心全量数据
# ==========================================
hours_str = [f"{i:02d}:00" for i in range(24)]
true_spreads = np.array([1.5, -1.0, 0.5, 0.0, -2.0, 3.5, 8.0, 4.0, 1.5, -3.0, 0.0, 5.0, 15.0, 80.0, 350.0, 650.0, 720.0, 580.0, 200.0, 45.0, 10.0, 2.0, -1.0, 0.5])
ai_profit = np.sum(true_spreads[np.argsort(true_spreads)[-3:]])

temps_2m = np.array([5, 3, 1, -2, -5, -8, -10, -12, -14, -15, -16, -15, -13, -11, -10, -12, -15, -18, -20, -19, -17, -15, -12, -10])
wind_speed_10m = np.array([5, 6, 8, 10, 12, 15, 18, 20, 15, 10, 8, 5, 4, 3, 2, 2, 1, 1, 2, 3, 5, 6, 8, 10])
precipitation = np.array([0, 0, 0, 2, 5, 10, 15, 25, 20, 10, 5, 2, 0, 0, 0, 5, 15, 25, 10, 5, 0, 0, 0, 0])
humidity_2m = np.array([60, 65, 70, 75, 80, 85, 90, 95, 95, 90, 85, 80, 75, 70, 75, 80, 85, 90, 95, 95, 90, 85, 80, 75])

# ==========================================
# Scene 0: Start
# ==========================================
if st.session_state.scene == 0:
    render_background("bg0.png")
    st.markdown("""
<div class="glass-card animate-fade-up">
    <h3>The Grid Crisis Game</h3>
    <h1>Can you predict the breaking point by tomorrow?</h1>
    <p>See if human intuition can save your portfolio from the worst effects of an unprecedented winter storm.</p>
</div>
""", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.button("Start Simulation", key="btn_start", type="primary", on_click=change_scene, args=(1,), use_container_width=True)

# ==========================================
# Scene 1: The Context
# ==========================================
elif st.session_state.scene == 1:
    render_background("bg1.png")
    st.markdown("""
<div class="glass-card animate-fade-up" style="text-align: left;">
    <h3>Round 1 of 3 • The Briefing</h3>
    <h1 style="font-size: 30px;">A polar vortex is descending upon Texas.</h1>
    <p><strong>Date: Feb 13, 2021.</strong> Meteorologists have just issued a dire warning. The storm threatens to freeze natural gas wellheads and ice over wind turbines.</p>
    <p>As an energy trader, you participate in <strong>Virtual Trading</strong>. You must buy power in the Day-Ahead market and sell it in Real-Time.</p>
    <div class="animate-fade-up delay-1" style="background: rgba(4, 120, 87, 0.1); padding: 15px; border-left: 4px solid #047857; border-radius: 4px; margin-top: 20px;">
        <p style="margin:0; font-size: 14px; font-weight: 600;">Your Goal: Analyze the weather data on the next page and select exactly 3 hours to deploy your capital.</p>
    </div>
</div>
""", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.button("Analyze Weather Data", key="btn_analyze", type="primary", on_click=change_scene, args=(2,), use_container_width=True)

# ==========================================
# Scene 2: The Data Analysis & Execution
# ==========================================
elif st.session_state.scene == 2:
    render_background("bg2.png")
    st.markdown("""
<div class="glass-card-wide animate-fade-up">
    <h3 style="text-align: center;">Round 2 of 3 • Market Execution</h3>
    <h1 style="font-size: 26px; text-align: center; margin-bottom: 25px;">Meteorological Forecast Dashboard</h1>
    <div class="bulletin-board animate-fade-up delay-1" style="padding: 20px 25px;">
        <div style="background: rgba(255, 255, 255, 0.95); border-left: 4px solid #047857; padding: 12px 18px; margin-bottom: 15px; border-radius: 6px; text-align: left;">
            <p style="margin: 0 0 4px 0; font-size: 13px; font-weight: 700; color: #047857;">💡 SYSTEM GUIDE: How Weather Impacts Grid Prices</p>
            <p style="margin: 0; font-size: 12px; color: #44403C; line-height: 1.5;">
                • <b>Temp 🌡️:</b> Deep freezes spike heating demand & freeze natural gas pipelines.<br>
                • <b>Wind 💨:</b> Extreme ice storms can freeze turbine blades, crashing power supply.<br>
                • <b>Precipitation ❄️:</b> Freezing rain & snow weigh down power lines & cover solar panels.<br>
                • <b>Humidity 💧:</b> High humidity accelerates dangerous ice formation on grid infrastructure.
            </p>
        </div>
""", unsafe_allow_html=True)
    
    board_bg_color = '#F8F7F2'
    fig_weather = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_weather.add_trace(go.Scatter(x=hours_str, y=temps_2m, mode='lines+markers', name='Temp (°C)', line=dict(color='#B91C1C', width=3)), secondary_y=False)
    fig_weather.add_trace(go.Scatter(x=hours_str, y=humidity_2m, mode='lines', name='Humidity (%)', line=dict(color='#475569', width=2, dash='dot')), secondary_y=False)
    fig_weather.add_trace(go.Bar(x=hours_str, y=precipitation, name='Precip (mm)', marker_color='rgba(56, 189, 248, 0.35)'), secondary_y=True)
    fig_weather.add_trace(go.Scatter(x=hours_str, y=wind_speed_10m, mode='lines', name='Wind (m/s)', line=dict(color='#047857', width=3)), secondary_y=True)
    
    fig_weather.update_layout(
        template="plotly_white", height=300, margin=dict(t=10, b=10, l=45, r=45), 
        plot_bgcolor=board_bg_color, paper_bgcolor=board_bg_color,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
        font=dict(family="Inter", color="#292524", size=11),
        xaxis=dict(showgrid=True, gridcolor='#E7E5E4'),
        yaxis=dict(showgrid=True, gridcolor='#E7E5E4', title="Temp (°C) / Humidity (%)"),
        yaxis2=dict(showgrid=False, title="Wind (m/s) / Precip (mm)")
    )
    st.plotly_chart(fig_weather, use_container_width=True)
    
    st.markdown(f"""
        <div style="border-top: 1px dashed #D6D3C4; margin-top: 10px; padding-top: 10px;">
            <p style='text-align: center; font-weight: 700; color: #44403C; margin-bottom: 12px;'>
                Select 3 Target Hours: <span style='color:#047857;'>{len(st.session_state.selected_hours)} / 3</span>
            </p>
        </div>
""", unsafe_allow_html=True)
    
    cols = st.columns(24)
    for i, hour in enumerate(hours_str):
        with cols[i]:
            btn_type = "primary" if hour in st.session_state.selected_hours else "secondary"
            st.button(f"{i:02d}", key=f"btn_{hour}", type=btn_type, on_click=toggle_hour, args=(hour,))
            
    st.markdown("""
    </div>
</div>
""", unsafe_allow_html=True) 

    if len(st.session_state.selected_hours) == 3:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.button("Lock Trades & Settle Market", key="btn_settle", type="primary", on_click=settle_market, use_container_width=True)
            
# ==========================================
# Scene 3: The Detailed Settlement
# ==========================================
elif st.session_state.scene == 3:
    render_background("bg3.png")
    is_win = st.session_state.user_profit >= (ai_profit * 0.7)
    board_bg_color = '#F8F7F2'
    
    st.markdown("""
<div class="glass-card-full animate-fade-up">
    <div style="border-bottom: 2px solid #111827; padding-bottom: 10px;">
        <h1 style="font-size: 28px; margin: 0; text-align: left;">Summary: Market Settlement</h1>
    </div>
</div>
""", unsafe_allow_html=True)
    
    col_chart, col_stats = st.columns([2, 1])
    
    with col_chart:
        st.markdown("""
    <div class="bulletin-board animate-fade-up delay-1" style="padding: 20px; margin-bottom: 0;">
        <p style='font-weight: 700; color: #292524; margin-bottom: 5px;'>Actual Market Spread ($/MWh)</p>
""", unsafe_allow_html=True)
        
        fig_res = go.Figure()
        fig_res.add_trace(go.Bar(x=hours_str, y=true_spreads, name='Spread', marker_color='#D6D3C4'))
        
        user_y = [true_spreads[i] for i in st.session_state.user_picks_idx]
        user_x = [hours_str[i] for i in st.session_state.user_picks_idx]
        fig_res.add_trace(go.Scatter(x=user_x, y=user_y, mode='markers', name='Your Trades', marker=dict(color='#B91C1C', size=16, symbol='x')))
        
        ai_picks_idx = np.argsort(true_spreads)[-3:]
        ai_y = [true_spreads[i] for i in ai_picks_idx]
        ai_x = [hours_str[i] for i in ai_picks_idx]
        fig_res.add_trace(go.Scatter(x=ai_x, y=ai_y, mode='markers', name='AI Model', marker=dict(color='#047857', size=14)))
        
        fig_res.update_layout(
            template="plotly_white", height=300, margin=dict(t=0, b=0, l=0, r=0), 
            plot_bgcolor=board_bg_color, paper_bgcolor=board_bg_color,
            legend=dict(orientation="h", y=1.1, font=dict(color="#292524", weight="bold"))
        )
        st.plotly_chart(fig_res, use_container_width=True)
        st.markdown("""
    </div>
""", unsafe_allow_html=True)

    with col_stats:
        st.markdown(f"""
    <div class="bulletin-board animate-fade-up delay-1" style="padding: 20px; margin-bottom: 15px;">
        <p style="font-size: 13px; font-weight: 800; color: #57534E; text-transform: uppercase; margin: 0;">Human Trader PnL</p>
        <h2 style="font-family: 'Lora', serif; font-size: 38px; color: {'#047857' if is_win else '#B91C1C'}; margin: 5px 0;">${st.session_state.user_profit:.2f}</h2>
    </div>
    <div class="bulletin-board animate-fade-up delay-2" style="padding: 20px; margin-bottom: 25px;">
        <p style="font-size: 13px; font-weight: 800; color: #57534E; text-transform: uppercase; margin: 0;">GridWise AI PnL</p>
        <h2 style="font-family: 'Lora', serif; font-size: 38px; color: #1C1917; margin: 5px 0;">${ai_profit:.2f}</h2>
    </div>
""", unsafe_allow_html=True)
        
        st.button("Restart Simulation", key="btn_restart", type="primary", on_click=restart_game, use_container_width=True)

    st.markdown("""
<div class="glass-card-full animate-fade-up delay-2" style="display: flex; gap: 20px; padding: 25px; margin-top: 25px; text-align: left;">
    <div style="flex: 1;">
        <p style="font-weight: 800; font-size: 15px; margin-bottom: 8px; color: #047857;">✅ AI Attribution</p>
        <p style="font-size: 14px; color: #44403C; margin: 0; line-height: 1.6;">Our model detected a subtle non-linear collapse in wind generation precisely when evening heating demand spiked, locking in peak profitability.</p>
    </div>
    <div style="flex: 1;">
        <p style="font-weight: 800; font-size: 15px; margin-bottom: 8px; color: #D97706;">⚠️ Alpha Tracker</p>
        <p style="font-size: 14px; color: #44403C; margin: 0; line-height: 1.6;">Human intuition struggles to process multidimensional arrays simultaneously. Quantitative models scale this flawlessly across massive datasets.</p>
    </div>
</div>
""", unsafe_allow_html=True)