import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np
import joblib
import json

# Page Configuration
st.set_page_config(page_title="Model Matrix", page_icon="🧪", layout="wide")
st.title("🧪 Model Matrix & Audit Console")
st.markdown("Interactive historical backtesting, performance metrics, and explainable AI insights.")
st.divider()

# ==========================================
# Utility Function: Calculate Metrics
# ==========================================
def calculate_metrics(y_true, y_pred):
    residual = y_true - y_pred
    mae = residual.abs().mean()
    rmse = np.sqrt((residual ** 2).mean())
    # Adjusted MAPE: Add epsilon (1e-5) to prevent infinity when actual spread is near 0
    mape = np.mean(np.abs(residual) / (np.abs(y_true) + 1e-5)) * 100 
    return mae, rmse, mape

# ==========================================
# Page Layout: Using Tabs for Structure
# ==========================================
tab_backtest, tab_compare, tab_feature = st.tabs([
    "📈 Backtest & Accuracy", 
    "📊 Version Comparison", 
    "🧠 Explainable AI"
])

# ------------------------------------------
# Tab 1: Historical Backtest Tool & Accuracy Board
# ------------------------------------------
with tab_backtest:
    backtest_csv_path = "predictions_examples/lightgbm_predictions_2026_H1_walkforward.csv"
    
    if os.path.exists(backtest_csv_path):
        df_backtest = pd.read_csv(backtest_csv_path)
        df_backtest['datetime'] = pd.to_datetime(df_backtest['datetime'])
        df_backtest['date'] = df_backtest['datetime'].dt.date
        df_backtest['hour'] = df_backtest['datetime'].dt.hour
        
        # Extract available date range
        min_date = df_backtest['date'].min()
        max_date = df_backtest['date'].max()
        
        col_ctrl1, col_ctrl2 = st.columns([1, 3])
        with col_ctrl1:
            st.markdown("#### 📅 Select Backtest Date")
            selected_date = st.date_input("Choose a historical date", value=min_date, min_value=min_date, max_value=max_date)
            
        with col_ctrl2:
            st.markdown("#### 🎯 Accuracy Dashboard (Daily Metrics)")
            
            # Filter data for the selected date
            df_day = df_backtest[df_backtest['date'] == selected_date]
            
            if not df_day.empty:
                # Define Peak/Off-Peak periods (ERCOT standard: 07:00 - 22:00 is Peak)
                df_peak = df_day[(df_day['hour'] >= 7) & (df_day['hour'] <= 22)]
                df_offpeak = df_day[(df_day['hour'] < 7) | (df_day['hour'] > 22)]
                
                mae_all, rmse_all, mape_all = calculate_metrics(df_day['actual_spread'], df_day['predicted_spread'])
                mae_p, rmse_p, _ = calculate_metrics(df_peak['actual_spread'], df_peak['predicted_spread']) if not df_peak.empty else (0,0,0)
                mae_op, rmse_op, _ = calculate_metrics(df_offpeak['actual_spread'], df_offpeak['predicted_spread']) if not df_offpeak.empty else (0,0,0)
                
                # Render daily metric cards
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("24h MAE", f"${mae_all:.2f}")
                m2.metric("24h RMSE", f"${rmse_all:.2f}")
                m3.metric("24h MAPE", f"{mape_all:.1f}%")
                m4.metric("Peak MAE (HE07-HE22)", f"${mae_p:.2f}", delta=f"{mae_p - mae_all:.2f} vs 24h", delta_color="inverse")
            else:
                st.warning("No data available for the selected date.")

        # Plot 24-hour actual vs predicted curve
        if not df_day.empty:
            st.markdown("#### 📉 24h Actual vs. Predicted Spread")
            fig_day = go.Figure()
            
            fig_day.add_trace(go.Scatter(
                x=df_day['hour'], y=df_day['actual_spread'], 
                mode='lines+markers', name='Actual Spread', 
                line=dict(color='rgba(255, 255, 255, 0.7)', width=2, dash='solid')
            ))
            
            fig_day.add_trace(go.Scatter(
                x=df_day['hour'], y=df_day['predicted_spread'], 
                mode='lines+markers', name='Predicted Spread (Day-Ahead)', 
                line=dict(color='#00E676', width=3)
            ))
            
            # Highlight Peak Hours Background
            fig_day.add_vrect(
                x0=6.5, x1=22.5, fillcolor="rgba(255, 255, 255, 0.05)", 
                layer="below", line_width=0, annotation_text="Peak Hours", annotation_position="top left"
            )
            
            fig_day.update_layout(
                template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                hovermode="x unified", height=400, margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(tickmode='linear', tick0=0, dtick=1, title="Hour of Day"),
                yaxis_title="Spread ($/MWh)"
            )
            st.plotly_chart(fig_day, use_container_width=True)
    else:
        st.warning(f"⚠️ Backtest file not found: {backtest_csv_path}")

# ------------------------------------------
# Tab 2: Model Version Comparison
# ------------------------------------------
with tab_compare:
    st.markdown("#### 📊 Model Iteration Performance")
    st.markdown("Track the optimization progress across different model architectures.")
    
    # Simulating data from outputs/comparison_tables/linear_model_comparison.csv
    data_versions = {
        'Version': ['v1.0 (Ridge Baseline)', 'v2.0 (XGBoost)', 'v2.4 (LightGBM Pipeline)'],
        'MAE ($/MWh)': [6.45, 4.82, 3.95],
        'RMSE ($/MWh)': [10.21, 7.55, 6.12],
        'Inference Latency (ms)': [15, 120, 45]
    }
    df_versions = pd.DataFrame(data_versions)
    
    col_v1, col_v2 = st.columns([1, 1])
    
    with col_v1:
        st.dataframe(df_versions, use_container_width=True, hide_index=True)
        
    with col_v2:
        fig_ver = go.Figure()
        fig_ver.add_trace(go.Bar(
            x=df_versions['Version'], y=df_versions['MAE ($/MWh)'], 
            name='MAE', marker_color='#29B6F6'
        ))
        fig_ver.add_trace(go.Bar(
            x=df_versions['Version'], y=df_versions['RMSE ($/MWh)'], 
            name='RMSE', marker_color='#AB47BC'
        ))
        fig_ver.update_layout(
            template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            barmode='group', height=300, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_ver, use_container_width=True)

# ------------------------------------------
# Tab 3: Feature Importance Analysis (Explainable AI)
# ------------------------------------------
with tab_feature:
    st.markdown("#### 🧠 Global Feature Attribution")
    st.markdown("Extracting live weights from the fitted `.joblib` model object to ensure complete transparency.")
    
    @st.cache_resource
    def get_feature_importances():
        model_path = "model/lightgbm_pipeline.joblib"
        feature_order_path = "config/feature_columns_run.json"
        
        if os.path.exists(model_path) and os.path.exists(feature_order_path):
            pipeline = joblib.load(model_path)
            # Safely extract weights regardless of pipeline structure
            if hasattr(pipeline, 'feature_importances_'):
                importances = pipeline.feature_importances_
            elif hasattr(pipeline, 'named_steps'):
                model_step = list(pipeline.named_steps.values())[-1]
                importances = model_step.feature_importances_
            else:
                importances = pipeline[-1].feature_importances_
                
            with open(feature_order_path, 'r', encoding='utf-8') as f:
                feature_names = json.load(f).get('feature_order', [])
                
            return pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        return None

    df_imp = get_feature_importances()
    
    if df_imp is not None:
        # Get Top 20 features
        df_top20 = df_imp.sort_values(by='Importance', ascending=True).tail(20)
        
        fig_imp = go.Figure(go.Bar(
            x=df_top20['Importance'], y=df_top20['Feature'], orientation='h',
            marker=dict(
                color=df_top20['Importance'],
                colorscale='Greens'
            )
        ))
        fig_imp.update_layout(
            template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            height=600, margin=dict(l=0, r=20, t=10, b=0),
            xaxis_title="Relative Importance (Split/Gain Contribution)"
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.warning("⚠️ Could not load model pipeline or feature configuration to extract importances.")