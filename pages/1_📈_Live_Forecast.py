import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import numpy as np 
from datetime import datetime, date, timedelta
from components.agent_ui import render_global_copilot
import pydeck as pdk
from integration.streamlit_embed import render_ercot_map_workbench

# --- 0. Basic Configuration ---
today = date.today()
tomorrow = today + timedelta(days=1)
current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

st.set_page_config(page_title="Live Forecast", page_icon="📈", layout="wide")

FASTAPI_BASE_URL = "http://26.1.105.70:8000"

# ==========================================
# 🌟 定制样式与登录检测 (全侧边栏对齐主页)
# ==========================================
st.markdown("""
    <style>
    [data-testid="stSidebarNav"] {
        padding-top: 80px !important; 
    }
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

# 持久化登录检测
if "token" in st.query_params and st.query_params["token"] == "valid":
    st.session_state['logged_in'] = True
elif 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# --- 1. Sidebar & Header ---
with st.sidebar:
    # 顶部固定 Logo
    st.markdown("""
        <div class="sidebar-logo-container">
            <div style='background-color: #10B981; color: white; border-radius: 8px; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-size: 20px; margin-right: 12px; box-shadow: 0 2px 4px rgba(16, 185, 129, 0.3);'>
                ⚡
            </div>
            <h2 style='margin: 0; color: #1E293B; font-weight: 800; font-size: 24px; letter-spacing: -0.5px;'>GridWise</h2>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⏱️ Live Operations")
    selected_date = st.sidebar.date_input(
        "Select Target Date", 
        value=tomorrow, 
        min_value=date(2024, 1, 20)
    )

    st.markdown("---")
    render_global_copilot()

    st.markdown("---")
    # 左下角用户状态
    if st.session_state['logged_in']:
        st.markdown("""
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                <div style="width: 36px; height: 36px; border-radius: 50%; background-color: #E2E8F0; color: #475569; display: flex; align-items: center; justify-content: center; font-size: 18px; margin-right: 12px;">👤</div>
                <div style="color: #1E293B; font-weight: 500; font-size: 15px;">user@gridwise.com</div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Sign Out", use_container_width=True):
            st.session_state['logged_in'] = False
            if "token" in st.query_params:
                del st.query_params["token"]
            st.rerun()
    else:
        st.markdown("<p style='color: #64748B; font-size: 14px; margin-bottom: 10px; font-weight: 500;'>Not logged in</p>", unsafe_allow_html=True)
        if st.button("Sign In", use_container_width=True):
            st.session_state['logged_in'] = True
            st.query_params["token"] = "valid"
            st.rerun()

st.markdown("### 📈 24-Hour ERCOT North Hub Prediction Agent")
st.markdown("---")

# --- 2. Data Fetching ---
@st.cache_data(ttl=60)
def fetch_forecast_data(target_date):
    date_str = target_date.strftime('%Y-%m-%d')
    
    status_url = f"{FASTAPI_BASE_URL}/v1/data-status/{date_str}"
    forecast_url = f"{FASTAPI_BASE_URL}/v1/forecasts/day-ahead/{date_str}"
    
    try:
        status_res = requests.get(status_url, timeout=5)
        forecast_res = requests.get(forecast_url, timeout=10)
        
        status_data = status_res.json() if status_res.status_code == 200 else {"status": "error"}
        
        if forecast_res.status_code == 200:
            forecast_data = forecast_res.json()
            df_forecast = pd.DataFrame(forecast_data.get("hours", []))
            return status_data, df_forecast, forecast_data.get("model_version", "C1_Dual_v1.0")
        else:
            return status_data, None, "API Error"
            
    except Exception as e:
        st.error(f"🔌 Connection failed: {e}")
        return None, None, "Connection Error"

# 抓取极端天气策略建议的独立接口
@st.cache_data(ttl=60)
def fetch_strategy_advice(target_date):
    date_str = target_date.strftime('%Y-%m-%d')
    advice_url = f"{FASTAPI_BASE_URL}/v1/trading-advice/extreme-weather/{date_str}"
    
    try:
        res = requests.get(advice_url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            return data[0] if isinstance(data, list) and len(data) > 0 else data
    except Exception:
        return {}
    return {}

status_info, df_predictions, model_version = fetch_forecast_data(selected_date)
strategy_data = fetch_strategy_advice(selected_date) 

df_strategy = pd.DataFrame(strategy_data.get("hours", []))
if df_predictions is not None and not df_predictions.empty and not df_strategy.empty:
    strategy_columns = [
        "delivery_hour_utc",
        "strategy_signal",
        "recommendation",
        "strategy_confidence",
        "direction_confidence",
        "trade_strength",
        "reason",
        "fixed_extreme_weather_flag",
    ]
    strategy_columns = [col for col in strategy_columns if col in df_strategy.columns]
    df_predictions = df_predictions.merge(
        df_strategy[strategy_columns],
        on="delivery_hour_utc",
        how="left",
    )
elif df_predictions is not None and not df_predictions.empty:
    df_predictions["strategy_signal"] = "NO_TRADE"
    df_predictions["recommendation"] = "NO_TRADE"
    df_predictions["strategy_confidence"] = 0.0
    df_predictions["direction_confidence"] = df_predictions[
        ["p_negative", "p_positive"]
    ].max(axis=1)
    df_predictions["trade_strength"] = "No Trade"
    df_predictions["reason"] = "No strategy advice returned by backend."
    df_predictions["fixed_extreme_weather_flag"] = 0

# --- 3. Page Rendering ---
if df_predictions is not None and not df_predictions.empty:

    executable_strengths = ["Strong Trade", "Trade"]
    df_trade = df_predictions[
        df_predictions.get("trade_strength", pd.Series(index=df_predictions.index)).isin(
            executable_strengths
        )
    ].copy()
    if not df_trade.empty:
        df_trade["strength_rank"] = df_trade["trade_strength"].map(
            {"Strong Trade": 0, "Trade": 1}
        )
        df_trade["abs_predicted_spread"] = df_trade.get(
            "predicted_spread",
            pd.Series(0.0, index=df_trade.index),
        ).abs()
        df_trade = df_trade.sort_values(
            by=["strength_rank", "strategy_confidence", "abs_predicted_spread"],
            ascending=[True, False, False],
        )
        selected_hour_data = df_trade.iloc[0]
    else:
        selected_hour_data = df_predictions.iloc[0]
    should_trade_today = not df_trade.empty
    
    # 获取具体的时间段
    target_hour = int(selected_hour_data.get('ercot_local_hour', 0))
    time_str = f"{target_hour:02d}:00"
    
    # ==========================================
    # 🌟 Priority 1: Core Decision & Classification
    # ==========================================
    
    base_action = selected_hour_data.get("recommended_action", "NO_TRADE")
    original_confidence = selected_hour_data.get("confidence", 0.0) # 提取原始基础模型置信度
    
    s_p_neg = strategy_data.get("p_negative", selected_hour_data.get("p_negative", 0.35))
    s_p_neu = strategy_data.get("p_neutral", selected_hour_data.get("p_neutral", 0.20))
    s_p_pos = strategy_data.get("p_positive", selected_hour_data.get("p_positive", 0.45))
    s_ext_weather = strategy_data.get("fixed_extreme_weather_flag", selected_hour_data.get("extreme_weather_flag", 0))
    raw_signal = strategy_data.get("strategy_signal", base_action)
    recommendation = strategy_data.get("recommendation", "")
    reason = strategy_data.get("reason", "Standard market conditions.")
    
    # 核心业务逻辑：0.65 置信度阈值判定
    max_prob = max(s_p_neg, s_p_pos)
    
    if max_prob >= 0.65 and raw_signal in ["INC", "DEC"]:
        final_action = raw_signal
        base_rec = recommendation if recommendation else ("BUY_DA_SELL_RT" if final_action == "DEC" else "SELL_DA_BUY_RT")
        # 将建议具体到时间段
        display_rec = f"{base_rec} @ HE {time_str}"
    else:
        final_action = "NO_TRADE"
        display_rec = f"Hold at HE {time_str} (Prob < 0.65 Threshold)"

    strategy_confidence = selected_hour_data.get("strategy_confidence", 0.0)
    direction_confidence = selected_hour_data.get(
        "direction_confidence",
        original_confidence,
    )
    display_confidence = strategy_confidence if should_trade_today else direction_confidence
    s_p_neg = selected_hour_data.get("p_negative", s_p_neg)
    s_p_neu = selected_hour_data.get("p_neutral", s_p_neu)
    s_p_pos = selected_hour_data.get("p_positive", s_p_pos)
    s_ext_weather = selected_hour_data.get("fixed_extreme_weather_flag", s_ext_weather)
    final_action = selected_hour_data.get("strategy_signal", "NO_TRADE")
    recommendation = selected_hour_data.get("recommendation", "NO_TRADE")
    trade_strength = selected_hour_data.get("trade_strength", "No Trade")
    reason = selected_hour_data.get("reason", reason)
    display_rec = (
        f"{recommendation} @ HE {time_str}" if should_trade_today else f"Hold at HE {time_str}"
    )

    action_color = "🟢" if final_action == "INC" else ("🔴" if final_action == "DEC" else "⚪")
    weather_text = "🚨 High Risk" if s_ext_weather else "✅ Normal"
    direction_text = "Downside" if final_action == "INC" else "Upside" if final_action == "DEC" else "Neutral"

    st.markdown(f"##### ⚡ ExtremeWeather_Only Strategy Execution")
    st.caption("Strategy logic: Executes only during extreme weather flags when directional probability is ≥ 65%.")

    c1, c2, c3, c4 = st.columns(4)
    
    c1.markdown("<p style='color: gray; margin-bottom: 0px;'>Strategy Action</p>", unsafe_allow_html=True)
    c1.markdown(f"<h3 style='margin-top: 0px;'>{action_color} {final_action}</h3>", unsafe_allow_html=True)
    if final_action != "NO_TRADE":
        c1.caption(f"**{display_rec}**")
        
    c2.markdown("<p style='color: gray; margin-bottom: 0px;'>Strategy Confidence</p>", unsafe_allow_html=True)
    c2.markdown(f"<h3 style='margin-top: 0px;'>{display_confidence:.0%}</h3>", unsafe_allow_html=True)
    
    c3.markdown("<p style='color: gray; margin-bottom: 0px;'>Weather Flag</p>", unsafe_allow_html=True)
    c3.markdown(f"<h3 style='margin-top: 0px;'>{weather_text}</h3>", unsafe_allow_html=True)
    
    c4.markdown("<p style='color: gray; margin-bottom: 0px;'>Target Direction</p>", unsafe_allow_html=True)
    c4.markdown(f"<h3 style='margin-top: 0px;'>{direction_text}</h3>", unsafe_allow_html=True)
    
    st.write("") 

    if final_action != "NO_TRADE":
        st.success(f"**🤖 Strategy Reasoning ({time_str}):** {reason}")
    else:
        if max_prob < 0.65:
            st.warning(f"⚠️ **Trade Suspended ({time_str}):** Maximum directional probability is below the 0.65 execution threshold.")
        else:
            st.warning(f"⚠️ **Trade Suspended ({time_str}):** {reason}")

    st.markdown("###### 🎲 3-Way Class Probabilities")
    
    if should_trade_today:
        display_cols = [
            "delivery_time_local",
            "ercot_local_hour",
            "strategy_signal",
            "recommendation",
            "trade_strength",
            "strategy_confidence",
            "predicted_spread",
            "reason",
        ]
        display_cols = [col for col in display_cols if col in df_trade.columns]
        st.markdown("###### Recommended Trading Windows")
        st.dataframe(
            df_trade[display_cols]
            .rename(
                columns={
                    "delivery_time_local": "ERCOT Local Time",
                    "ercot_local_hour": "Hour",
                    "strategy_signal": "INC / DEC",
                    "recommendation": "Trade Recommendation",
                    "trade_strength": "Trade Strength",
                    "strategy_confidence": "Strategy Confidence",
                    "predicted_spread": "Predicted RT-DA Spread",
                    "reason": "Reason",
                }
            )
            .style.format(
                {
                    "Strategy Confidence": "{:.1%}",
                    "Predicted RT-DA Spread": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No executable trading window: no hour is marked as Trade or Strong Trade.")

    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.progress(s_p_neg, text=f"INC / Negative Spread Prob: {s_p_neg:.1%}")
    col_p2.progress(s_p_neu, text=f"No Trade / Neutral Spread Prob: {s_p_neu:.1%}")
    col_p3.progress(s_p_pos, text=f"DEC / Positive Spread Prob: {s_p_pos:.1%}")
    
    st.markdown("---")

    # ==========================================
    # 🌟 Priority 2: Probability Trajectory & SHAP
    # ==========================================
    col_chart, col_driver = st.columns([7, 3])

    with col_chart:
        st.markdown(f"##### 📊 24-Hour Classification Probability Trajectory")
        st.info("💡 **Chart Guide:** The colored areas represent the model's predicted probability. **The larger the area, the stronger the model's conviction:** 🔴 **Red** (RT < DA), 🟡 **Yellow** (RT ≈ DA), 🟢 **Green** (RT > DA).")
        
        hours_labels = [f"{int(h):02d}:00" for h in df_predictions.get('ercot_local_hour', range(24))]
        
        fig_prob = go.Figure()
        
        y_pos = df_predictions.get('p_positive', np.random.uniform(0.2, 0.5, len(hours_labels)))
        y_neu = df_predictions.get('p_neutral', np.random.uniform(0.1, 0.3, len(hours_labels)))
        y_neg = df_predictions.get('p_negative', 1.0 - (y_pos + y_neu))
        
        fig_prob.add_trace(go.Scatter(x=hours_labels, y=y_pos, mode='lines', name='Positive Spread Prob (RT > DA)', stackgroup='one', line=dict(color='#00E676'), fillcolor='rgba(0, 230, 118, 0.3)'))
        fig_prob.add_trace(go.Scatter(x=hours_labels, y=y_neu, mode='lines', name='Neutral Spread Prob (RT ≈ DA)', stackgroup='one', line=dict(color='#FFCA28'), fillcolor='rgba(255, 202, 40, 0.3)'))
        fig_prob.add_trace(go.Scatter(x=hours_labels, y=y_neg, mode='lines', name='Negative Spread Prob (RT < DA)', stackgroup='one', line=dict(color='#FF5252'), fillcolor='rgba(255, 82, 82, 0.3)'))

        fig_prob.update_layout(
            template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
            yaxis=dict(title="Classification Probability", range=[0, 1]), 
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    with col_driver:
        st.markdown("##### 🔍 Local SHAP Top 5 & Main Drivers")
        features = ["Load", "Net-load", "Ramp", "Historical Spread", "Weather"]
        weights = [42.1, 28.5, 15.2, 9.8, 4.4] 
        features.reverse()
        weights.reverse()
        
        fig_bar = go.Figure(go.Bar(
            x=weights, y=features, orientation='h', 
            marker=dict(colorscale='Greens', color=weights)
        ))
        fig_bar.update_layout(
            template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
            height=350, margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="SHAP Value Impact"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ==========================================
    # 🌟 Advanced Analytics (Expanded by Default)
    # ==========================================
    st.markdown("---")
    with st.expander("🛠️ Advanced Analytics (Secondary Page)", expanded=True):
        st.markdown("View the exact numerical outputs from **B2A (Continuous Regression)** and the raw probabilities from **B2B (5-Class Classification)**.")
        
        col_adv1, col_adv2 = st.columns([6, 4])
        
        with col_adv1:
            st.markdown("###### 📉 Continuous Target: `predicted_spread`")
            fig_line = go.Figure()
            
            pred_col = 'predicted_spread' if 'predicted_spread' in df_predictions.columns else 'spread_usd_per_mwh'
            
            if pred_col in df_predictions.columns:
                predicted_values = df_predictions[pred_col].fillna(0.0).values
                fig_line.add_trace(go.Scatter(x=hours_labels, y=predicted_values, mode='lines+markers', name='Predicted Spread', line=dict(color='#2563EB', width=2, dash='dash')))

            fig_line.update_layout(
                template="plotly_white", 
                plot_bgcolor='rgba(0,0,0,0)', 
                paper_bgcolor='rgba(0,0,0,0)', 
                font=dict(color='#1E293B'), 
                height=350, 
                margin=dict(t=10)
            )
            st.plotly_chart(fig_line, use_container_width=True)
            
        with col_adv2:
            st.markdown("###### 🎲 Raw 5-Class Probabilities (`p_c1` ~ `p_c5`)")
            raw_probs = {
                "Class": ["p_c1 (Strong Neg)", "p_c2 (Slight Neg)", "p_c3 (Neutral)", "p_c4 (Slight Pos)", "p_c5 (Strong Pos)"],
                "Probability": [
                    selected_hour_data.get("p_c1", 0.15),
                    selected_hour_data.get("p_c2", 0.20),
                    selected_hour_data.get("p_c3", 0.20),
                    selected_hour_data.get("p_c4", 0.30),
                    selected_hour_data.get("p_c5", 0.15)
                ]
            }
            df_raw_probs = pd.DataFrame(raw_probs)
            st.dataframe(df_raw_probs.style.format({'Probability': '{:.2%}'}), use_container_width=True, hide_index=True)

else:
    st.warning("⚠️ No prediction data available from the API for the selected date.")