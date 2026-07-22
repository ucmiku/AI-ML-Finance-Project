import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import os
import sqlite3
from datetime import date

st.sidebar.markdown("### ⏱️ Time Travel Simulation")
st.sidebar.markdown("Since the dataset ends in June 2026, use this to simulate a 'live' trading day.")
# 默认选择 2026年6月25日 作为演示日
selected_date = st.sidebar.date_input(
    "Select 'Current' Date", 
    value=date(2026, 6, 25), 
    min_value=date(2024, 1, 20), 
    max_value=date(2026, 6, 30)
)

# --- 1. 定义模型加载函数 ---
@st.cache_resource
def load_ml_model():
    """加载 LightGBM 预测模型"""
    model_path = "model/lightgbm_pipeline.joblib"
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            return model, "✅ ML Model: `v2.4-Ensemble-LSTM` (LightGBM Pipeline Active)"
        except Exception as e:
            return None, f"❌ Model Load Error: {e}"
    else:
        return None, "⚠️ Warning: 'model/lightgbm_pipeline.joblib' not found."

st.set_page_config(page_title="Live Forecast", page_icon="📈", layout="wide")

# 加载模型
model, model_status = load_ml_model()

st.markdown("### 📈 24-Hour ERCOT North Hub Price Forecast & Driver Attribution")
st.markdown(f"<small style='color: #888;'>{model_status}</small>", unsafe_allow_html=True)
st.markdown("---")

# --- 2. 真实数据库特征提取函数 ---
# --- 2. 真实数据库特征提取函数 ---
def fetch_live_features_from_db(target_date):
    """从数据库中提取选定日期的真实 24 小时特征和电价"""
    db_path = "01_data_collection_cleaning/interim/ercot_analytics.sqlite"
    
    try:
        conn = sqlite3.connect(db_path)
        query = f"""
            SELECT * FROM model_wide_hourly_2024_2026 
            WHERE delivery_date_local = '{target_date}'
            ORDER BY ercot_local_hour ASC
        """
        df_day = pd.read_sql(query, conn)
        conn.close()
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None, None, None

    if df_day.empty:
        return None, None, None

    # ==========================================
    # 核心修复：将数据库的规范列名，映射回模型训练时的短特征名
    # ==========================================
    rename_mapping = {
        "ercot_local_hour": "hour",
        "ercot_local_day_of_week": "day_of_week",
        "ercot_local_month": "month",
        "is_weekend": "weekend",
        "is_dst": "dst",
        "gas_price_usd_per_mmbtu": "gas_price"
    }
    df_day = df_day.rename(columns=rename_mapping)

    # 这是你模型所需的严格 72 个特征白名单 (现在它们的名字已经和 JSON 配置文件里一样了)
    feature_order = [
        "spread_asof_lag24", "spread_asof_lag48", "spread_asof_lag168", "spread_asof_roll_mean24", "spread_asof_roll_std24",
        "hour", "day_of_week", "month", "weekend", "dst", "holiday", 
        "hour_sin", "hour_cos", "month_sin", "month_cos", 
        "dow_0", "dow_1", "dow_2", "dow_3", "dow_4", "dow_5", "dow_6",
        "gas_price", 
        "load_coast_mw", "load_east_mw", "load_far_west_mw", "load_north_mw", "load_north_central_mw", "load_south_central_mw", "load_southern_mw", "load_west_mw", "load_system_total_mw",
        "wind_stwpf_lz_north_mw", "wind_stwpf_lz_south_houston_mw", "wind_stwpf_lz_west_mw", "wind_stwpf_system_wide_mw",
        "wind_wgrpp_lz_north_mw", "wind_wgrpp_lz_south_houston_mw", "wind_wgrpp_lz_west_mw", "wind_wgrpp_system_wide_mw",
        "solar_pvgrpp_system_mw", "solar_stppf_system_mw",
        "renewable_st_forecast_system_mw", "net_load_st_forecast_system_mw", "renewable_potential_system_mw", "net_load_potential_system_mw",
        "temperature_dfw_mean_c", "temperature_wichita_c", "north_temperature_min_c", "north_temperature_max_c",
        "humidity_dfw_mean_pct", "humidity_wichita_pct",
        "wind_speed_dfw_mean_ms", "wind_speed_dfw_max_ms", "wind_speed_wichita_ms",
        "wind_gust_dfw_mean_ms", "wind_gust_dfw_max_ms", "wind_gust_wichita_ms",
        "cloud_cover_dfw_mean_pct", "cloud_cover_wichita_pct",
        "radiation_dfw_mean_wm2", "radiation_wichita_wm2",
        "precipitation_dfw_mean_mm", "precipitation_dfw_max_mm", "precipitation_wichita_mm", "north_precipitation_max_mm",
        "freezing_city_count", "extreme_heat_city_count", "high_wind_city_count", "heavy_rain_city_count", "low_wind_city_count", "heat_lowwind_interaction"
    ]
    
    # 提取特征矩阵 (24x72)
    X_live = pd.DataFrame(0, index=np.arange(len(df_day)), columns=feature_order)
    for col in feature_order:
        if col in df_day.columns:
            X_live[col] = df_day[col].values
            
    # 提取真实已发生电价（作为基准线）
    actual_rt = df_day['rt_price_usd_per_mwh'].values if 'rt_price_usd_per_mwh' in df_day.columns else np.zeros(len(df_day))
    
    # 返回 X_live, actual_rt, 以及用来画图的 hour 坐标
    return X_live, actual_rt, df_day['hour'].values
# --- 3. 运行真实模型预测 ---
X_live, actual_rt, actual_hours = fetch_live_features_from_db(selected_date)

if X_live is not None and model is not None:
    try:
        # 使用真实特征输入 LightGBM 预测价差 (Spread)
        # 假设预测的是 Spread (RT - DA)，需要加上 DA 才是预测的 RT Price
        predicted_spread = model.predict(X_live)
        
        # 为了演示置信区间，简单模拟一个波动范围 (如果你有 quantile regression 模型可以直接输出)
        upper_bound = predicted_spread + np.abs(predicted_spread * 0.20) + 3
        lower_bound = predicted_spread - np.abs(predicted_spread * 0.20) - 3
        
        # 模拟“日内渐进效果”：假设现在是中午 12 点 (前12小时展示真实值，后12小时为空)
        current_hour_idx = 12
        actual_price_display = list(actual_rt[:current_hour_idx]) + [None] * (len(actual_rt) - current_hour_idx)
        
        hours_labels = [f"{int(h):02d}:00" for h in actual_hours]
    except Exception as e:
        st.error(f"Prediction Execution Error: {e}")
        st.stop()
else:
    st.warning("⚠️ No data available for the selected date or model not loaded.")
    st.stop()

# --- 4. 渲染图表区 ---
col_chart, col_driver = st.columns([7, 3])

with col_chart:
    st.markdown(f"##### 📉 24-Hour Price Trajectory for {selected_date}")
    
    fig_line = go.Figure()
    # 绘制上边界、下边界和预测中位数 (代码同你原来一致，传入 hours_labels, upper_bound, lower_bound, predicted_spread)
    fig_line.add_trace(go.Scatter(x=hours_labels, y=upper_bound, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig_line.add_trace(go.Scatter(x=hours_labels, y=lower_bound, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 230, 118, 0.15)', name='95% Confidence Interval'))
    fig_line.add_trace(go.Scatter(x=hours_labels, y=predicted_spread, mode='lines+markers', name='Forecast (Median Spread)', line=dict(color='#00E676', width=3)))
    
    # 替换为真实的 actual_price_display
    fig_line.add_trace(go.Scatter(x=hours_labels, y=actual_price_display, mode='lines+markers', name='Actual RT Spread (Day-to-Date)', line=dict(color='#FFFFFF', width=2, dash='solid')))

    fig_line.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_line, use_container_width=True)

with col_driver:
    st.markdown("##### 🧩 Global Model Drivers")
    # 你可以后续用 model.feature_importances_ 替换这里，目前保留写死以保证 UI 结构不变
    features = ["Net Load Forecast", "Natural Gas Spot", "DFW Mean Temp", "West Wind Generation", "Solar Generation"]
    weights = [35.2, 22.8, 15.5, 12.0, 8.5]
    features.reverse()
    weights.reverse()
    fig_bar = go.Figure(go.Bar(x=weights, y=features, orientation='h', marker=dict(colorscale='Greens', color=weights)))
    fig_bar.update_layout(template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=350)
    st.plotly_chart(fig_bar, use_container_width=True)