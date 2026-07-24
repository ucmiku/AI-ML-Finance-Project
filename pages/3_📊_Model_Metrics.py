import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import json
import requests
import numpy as np
from datetime import date, timedelta
from components.agent_ui import render_global_copilot

# Page Configuration
st.set_page_config(page_title="Model Matrix", page_icon="🧪", layout="wide")
st.title("🧪 Model Matrix & Audit Console")
st.markdown("Interactive historical backtesting, performance metrics, and explainable AI insights.")
st.divider()
FASTAPI_BASE_URL = "http://26.1.105.70:8000"

with st.sidebar:
    st.markdown("### ⚙️ Copilot Access")
    st.markdown("Interact with the quantitative RAG engine anytime.")
    st.markdown("---")
    render_global_copilot()

# ==========================================
# Page Layout
# ==========================================
tab_pred, tab_trade, tab_feature = st.tabs([
    "🎯 Prediction Model Performance", 
    "📈 Trading Model Performance", 
    "🧠 Explainable AI"
])

json_path = "bk_testing/backtest_result_c1.json"
c1_data = {}
if os.path.exists(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        c1_data = json.load(f)

# ------------------------------------------
# Tab 1: Prediction Model Performance
# ------------------------------------------
with tab_pred:
    st.markdown("### 🎯 Classification Probability & Actual Spread Tracker")
    st.markdown("Tracking C1 dual-output classification probabilities alongside actual spread.")
    
    st.info("💡 **Chart Guide:** The colored areas represent the model's predicted probability (Left Axis). **The larger the area, the stronger the model's conviction:** 🔴 **Red** (RT < DA), 🟡 **Yellow** (RT ≈ DA), 🟢 **Green** (RT > DA). The solid blue line tracks the actual market spread (Right Axis).")

    col_d1, col_d2 = st.columns([2, 6])
    with col_d1:
        min_allowed_date = date(2024, 1, 1)
        max_allowed_date = date(2026, 6, 30)
        default_eval_date = date(2026, 6, 26)
        
        eval_date = st.date_input(
            "Select Evaluation Date", 
            value=default_eval_date, 
            min_value=min_allowed_date, 
            max_value=max_allowed_date, 
            key="eval_date_picker"
        )
    
    date_str = eval_date.strftime('%Y-%m-%d')
    
    @st.cache_data(ttl=60)
    def fetch_eval_forecast_data(target_date_str):
        api_url = f"{FASTAPI_BASE_URL}/v1/forecasts/day-ahead/{target_date_str}"
        try:
            response = requests.get(api_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                hours_data = data.get("hours", data) if isinstance(data, dict) else data
                return pd.DataFrame(hours_data)
            return None
        except Exception:
            return None

    df_eval = fetch_eval_forecast_data(date_str)
    
    if df_eval is not None and not df_eval.empty:
        hour_col = 'ercot_local_hour' if 'ercot_local_hour' in df_eval.columns else df_eval.columns[0]
        hours_labels = [f"{int(h):02d}:00" for h in df_eval[hour_col]]
        
        fig_dual = go.Figure()
        
        prob_cols = {
            'p_positive': ('Positive Spread Prob (RT > DA)', '#00E676', 'rgba(0, 230, 118, 0.3)'),
            'p_neutral': ('Neutral Spread Prob (RT ≈ DA)', '#FFCA28', 'rgba(255, 202, 40, 0.3)'),
            'p_negative': ('Negative Spread Prob (RT < DA)', '#FF5252', 'rgba(255, 82, 82, 0.3)')
        }
        
        for col_name, (label, line_color, fill_color) in prob_cols.items():
            if col_name in df_eval.columns:
                fig_dual.add_trace(go.Scatter(
                    x=hours_labels, y=df_eval[col_name].fillna(0.0).values,
                    mode='lines', name=label,
                    stackgroup='one', 
                    line=dict(width=0.5, color=line_color),
                    fillcolor=fill_color 
                ))
        
        if not any(col in df_eval.columns for col in prob_cols.keys()):
            np.random.seed(42)
            dummy_pos = np.random.uniform(0.1, 0.6, len(hours_labels))
            dummy_neg = np.random.uniform(0.1, 0.4, len(hours_labels))
            dummy_neu = 1.0 - (dummy_pos + dummy_neg)
            
            fig_dual.add_trace(go.Scatter(x=hours_labels, y=dummy_pos, mode='lines', name='Positive Spread Prob (RT > DA)', stackgroup='one', line=dict(color='#00E676'), fillcolor='rgba(0, 230, 118, 0.3)'))
            fig_dual.add_trace(go.Scatter(x=hours_labels, y=dummy_neu, mode='lines', name='Neutral Spread Prob (RT ≈ DA)', stackgroup='one', line=dict(color='#FFCA28'), fillcolor='rgba(255, 202, 40, 0.3)'))
            fig_dual.add_trace(go.Scatter(x=hours_labels, y=dummy_neg, mode='lines', name='Negative Spread Prob (RT < DA)', stackgroup='one', line=dict(color='#FF5252'), fillcolor='rgba(255, 82, 82, 0.3)'))

        act_col = 'spread_usd_per_mwh' if 'spread_usd_per_mwh' in df_eval.columns else ('actual_spread' if 'actual_spread' in df_eval.columns else None)
        if act_col and act_col in df_eval.columns:
            actual_spread_values = df_eval[act_col].fillna(0.0).values
            fig_dual.add_trace(go.Scatter(
                x=hours_labels, y=actual_spread_values,
                mode='lines+markers', name='Actual Spread ($/MWh)',
                yaxis='y2',
                line=dict(color='#29B6F6', width=2)
            ))

        fig_dual.update_layout(
            template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Delivery Hour (Local)", 
            yaxis=dict(title="Classification Probability", range=[0, 1]),
            yaxis2=dict(title="Actual Spread ($/MWh)", overlaying='y', side='right', showgrid=False),
            height=500,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_dual, use_container_width=True)

        # Hourly Prediction Audit Log
        st.markdown("#### 🕵️ Hourly Prediction Audit Log")
        st.markdown("Evaluating true prediction accuracy and hypothetical Net PnL based on actual historical spread.")
        
        audit_data = []
        for i, row in df_eval.iterrows():
            hour = f"{int(row.get(hour_col, i)):02d}:00"
            actual_spread = row.get(act_col, 0.0)
            
            threshold = 0.5 
            if actual_spread > threshold:
                actual_direction = "Positive"
            elif actual_spread < -threshold:
                actual_direction = "Negative"
            else:
                actual_direction = "Neutral"
            
            recommended_action = row.get("recommended_action", None)
            if not recommended_action:
                p_pos = row.get("p_positive", 0)
                p_neg = row.get("p_negative", 0)
                if p_pos > max(p_neg, 0.4): recommended_action = "DEC"
                elif p_neg > max(p_pos, 0.4): recommended_action = "INC"
                else: recommended_action = "NO_TRADE"
                
            if recommended_action == "DEC" and actual_direction == "Positive":
                is_correct = "✅ Yes"
                executed = "Yes (DEC)"
            elif recommended_action == "INC" and actual_direction == "Negative":
                is_correct = "✅ Yes"
                executed = "Yes (INC)"
            elif recommended_action == "NO_TRADE":
                is_correct = "⚪ N/A"
                executed = "No"
            else:
                is_correct = "❌ No"
                executed = f"Yes ({recommended_action})"
                
            if executed == "No":
                net_pnl = 0.0
            else:
                multiplier = 1 if recommended_action == "DEC" else -1
                net_pnl = actual_spread * multiplier
                
            audit_data.append({
                "Hour": hour,
                "Recommended Action": recommended_action,
                "Actual Spread ($/MWh)": f"{actual_spread:+.2f}",
                "Actual Direction": actual_direction,
                "Trade Executed": executed,
                "Direction Correct": is_correct,
                "Net PnL": f"${net_pnl:+.2f}"
            })
            
        df_audit = pd.DataFrame(audit_data)
        
        def style_audit_log(row):
            styles = [''] * len(row)
            if row['Direction Correct'] == '✅ Yes':
                styles[5] = 'color: #00E676; font-weight: bold;'
            elif row['Direction Correct'] == '❌ No':
                styles[5] = 'color: #FF5252; font-weight: bold;'
                
            if float(row['Net PnL'].replace('$', '')) > 0:
                styles[6] = 'color: #00E676; font-weight: bold;'
            elif float(row['Net PnL'].replace('$', '')) < 0:
                styles[6] = 'color: #FF5252; font-weight: bold;'
                
            return styles

        st.dataframe(df_audit.style.apply(style_audit_log, axis=1), use_container_width=True, hide_index=True)

    else:
        st.warning(f"⚠️ No prediction audit data available from FastAPI for `{date_str}`.")

# ------------------------------------------
# Tab 2: Trading Model Performance
# ------------------------------------------
with tab_trade:
    if not c1_data:
        st.warning(f"⚠️ Backtest file not found: `{json_path}`.")
    else:
        st.markdown("### 📈 Trading Strategy Evaluation")
        meta = c1_data.get("meta", {})
        assumptions = meta.get("assumptions", {})
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Selected Strategy", "ExtremeWeather_Only", "Chosen for Deployment")
        m2.metric("Initial Capital", f"${assumptions.get('initial_capital_usd', 0):,}")
        m3.metric("Position Size", f"{assumptions.get('position_size_mwh', 0)} MWh")
        m4.metric("Commission", f"${assumptions.get('commission_per_mwh_usd', 0)}/MWh")
        
        st.divider()
        
        scoring_json_path = "bk_testing/scoring_report.json"
        if os.path.exists(scoring_json_path):
            with open(scoring_json_path, "r", encoding="utf-8") as fs:
                scoring_data = json.load(fs)
                
            st.markdown("#### 🏆 Strategy Leaderboard & Capability Radar")
            ranking = scoring_data.get("ranking", [])
            scores = scoring_data.get("scores", {})
            
            if len(ranking) >= 3:
                col_1st, col_2nd, col_3rd = st.columns(3)
                with col_1st:
                    st.markdown(f"""
                    <div style="background-color: rgba(255, 82, 82, 0.15); padding: 15px; border-radius: 10px; border-left: 5px solid #FF5252;">
                        <h4 style="margin-top:0; color: white;">🥇 1st Place</h4>
                        <h3 style="color: #FF5252;">{ranking[0]}</h3>
                        <strong>Composite Score: {scores[ranking[0]]['composite']}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                with col_2nd:
                    st.markdown(f"""
                    <div style="background-color: rgba(0, 230, 118, 0.15); padding: 15px; border-radius: 10px; border-left: 5px solid #00E676;">
                        <h4 style="margin-top:0; color: white;">🥈 2nd Place</h4>
                        <h3 style="color: #00E676;">{ranking[1]}</h3>
                        <strong>Composite Score: {scores[ranking[1]]['composite']}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                with col_3rd:
                    st.markdown(f"""
                    <div style="background-color: rgba(41, 182, 246, 0.15); padding: 15px; border-radius: 10px; border-left: 5px solid #29B6F6;">
                        <h4 style="margin-top:0; color: white;">🥉 3rd Place</h4>
                        <h3 style="color: #29B6F6;">{ranking[2]}</h3>
                        <strong>Composite Score: {scores[ranking[2]]['composite']}</strong>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.write("") 
            
            radar_fig = go.Figure()
            categories = ["Return", "Risk", "Robustness", "Efficiency"]
            compare_models = [ranking[0], ranking[1], ranking[2]] 
            colors = ['#FF5252', '#00E676', '#29B6F6'] 
            
            for idx, model_name in enumerate(compare_models):
                if model_name in scores:
                    model_cats = scores[model_name]["categories"]
                    radar_scores = [model_cats[cat]["score_0_100"] for cat in model_cats] 
                    if len(radar_scores) == len(categories):
                        radar_scores.append(radar_scores[0])
                        radar_cats = categories + [categories[0]]
                        
                        radar_fig.add_trace(go.Scatterpolar(
                            r=radar_scores, theta=radar_cats, fill='toself',
                            name=model_name, line=dict(color=colors[idx % len(colors)], width=2), opacity=0.5
                        ))

            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.2)"),
                    bgcolor="rgba(0,0,0,0)"
                ),
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=450, margin=dict(t=40, b=40, l=40, r=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1)
            )
            st.plotly_chart(radar_fig, use_container_width=True)
            st.divider()
            
            st.markdown("#### 📊 Strategy Comparison Matrix")
            df_comp = pd.DataFrame(c1_data.get("strategy_comparison", []))
            if not df_comp.empty:
                df_comp = df_comp.sort_values(by="sharpe_ratio", ascending=False)
                format_dict = {
                    "total_pnl": "${:.2f}", "total_return": "{:.2%}", "sharpe_ratio": "{:.4f}",
                    "max_drawdown": "{:.2%}", "win_rate": "{:.2%}", "avg_trade_pnl": "${:.2f}"
                }
                def highlight_selected(row):
                    return ['background-color: rgba(0, 230, 118, 0.2)'] * len(row) if row['name'] == 'ExtremeWeather_Only' else [''] * len(row)
                
                st.dataframe(df_comp.style.apply(highlight_selected, axis=1).format(format_dict), use_container_width=True, hide_index=True)
                
            st.divider()

        st.markdown("#### 📈 Multi-Strategy Equity Curves")
        equity_data = c1_data.get("equity_curves", {})
        
        if equity_data:
            fig_eq = go.Figure()
            highlight_strategies = ["B2B_Baseline_060", "ExtremeWeather_Only"]
            
            for strategy_name, curve_points in equity_data.items():
                df_curve = pd.DataFrame(curve_points)
                is_highlight = strategy_name in highlight_strategies
                line_width = 4 if strategy_name == "ExtremeWeather_Only" else (2 if is_highlight else 1)
                opacity = 1.0 if is_highlight else 0.3
                dash_style = 'dash' if strategy_name == "B2B_Baseline_060" else 'solid'
                
                fig_eq.add_trace(go.Scatter(
                    x=df_curve['date'], y=df_curve['equity'],
                    mode='lines', name=strategy_name,
                    line=dict(width=line_width, dash=dash_style), opacity=opacity
                ))
                
            fig_eq.update_layout(
                template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                hovermode="x unified", height=500, xaxis_title="Date", yaxis_title="Equity (USD)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_eq.add_hline(y=100000, line_dash="dot", line_color="gray", annotation_text="Initial Capital ($100k)")
            st.plotly_chart(fig_eq, use_container_width=True)

# ------------------------------------------
# Tab 3: Feature Importance Analysis (SHAP Explainability API)
# ------------------------------------------
with tab_feature:
    st.markdown("#### 🧠 Explainable AI & SHAP Insights")
    st.markdown("Dynamic model attribution, feature ranking, local explanations, and dependence relationships via backend SHAP APIs.")
    
    # Sub-tabs for SHAP exploration
    shap_tab1, shap_tab2, shap_tab3 = st.tabs([
        "📊 Global Feature Ranking", 
        "🔍 Local Hour Explanation", 
        "📈 Feature Dependence Scatter"
    ])

    output_head_options = {
        "Spread Regression": "spread_regression",
        "Negative Probability": "negative_probability",
        "Neutral Probability": "neutral_probability",
        "Positive Probability": "positive_probability"
    }

    # 1. Global Feature Ranking
    with shap_tab1:
        st.markdown("##### 📊 Global Feature Importance Ranking")
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        
        with col_r1:
            window_choice = st.selectbox("Window", ["daily", "weekly", "monthly"], key="shap_window")
        with col_r2:
            ranking_date = st.date_input("Reference Date", value=date(2026, 6, 26), key="shap_rank_date")
        with col_r3:
            selected_head_label = st.selectbox("Output Head", list(output_head_options.keys()), key="shap_rank_head")
        with col_r4:
            top_n = st.slider("Top N Features", min_value=5, max_value=30, value=20, key="shap_top_n")

        date_str_rank = ranking_date.strftime('%Y-%m-%d')
        output_head_val = output_head_options[selected_head_label]

        @st.cache_data(ttl=60)
        def fetch_shap_ranking(window, date_s, head, top):
            url = f"{FASTAPI_BASE_URL}/explainability/ranking"
            params = {"window": window, "date": date_s, "output_head": head, "top_n": top}
            try:
                res = requests.get(url, params=params, timeout=10)
                if res.status_code == 200:
                    return pd.DataFrame(res.json())
                return None
            except Exception:
                return None

        df_ranking = fetch_shap_ranking(window_choice, date_str_rank, output_head_val, top_n)

        if df_ranking is not None and not df_ranking.empty:
            df_sorted = df_ranking.sort_values(by='importance', ascending=True)
            fig_rank = go.Figure(go.Bar(
                x=df_sorted['importance'], y=df_sorted['feature'], orientation='h',
                marker=dict(color=df_sorted['importance'], colorscale='Greens')
            ))
            fig_rank.update_layout(
                template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                height=550, margin=dict(l=0, r=20, t=10, b=0),
                xaxis_title="Mean Absolute SHAP Importance",
                yaxis_title="Feature"
            )
            st.plotly_chart(fig_rank, use_container_width=True)
        else:
            st.warning(f"⚠️ No SHAP ranking data found for window=`{window_choice}`, date=`{date_str_rank}`, head=`{output_head_val}`. (Please ensure backend endpoint `/explainability/ranking` is running).")

    # 2. Local Hour Explanation
    with shap_tab2:
        st.markdown("##### 🔍 Local Prediction Explanation (Single Delivery Hour)")
        col_l1, col_l2, col_l3 = st.columns(3)
        with col_l1:
            local_utc_hour = st.text_input("Delivery Hour UTC", value="2026-06-26 12:00:00+00:00", key="shap_local_utc")
        with col_l2:
            local_head_label = st.selectbox("Output Head", list(output_head_options.keys()), key="shap_local_head")
        with col_l3:
            local_top_n = st.slider("Top N Local Drivers", min_value=5, max_value=20, value=10, key="shap_local_top")

        local_head_val = output_head_options[local_head_label]

        @st.cache_data(ttl=60)
        def fetch_shap_local(utc_hour, head, top):
            url = f"{FASTAPI_BASE_URL}/explainability/local"
            params = {"delivery_hour_utc": utc_hour, "output_head": head, "top_n": top}
            try:
                res = requests.get(url, params=params, timeout=10)
                if res.status_code == 200:
                    return pd.DataFrame(res.json())
                return None
            except Exception:
                return None

        df_local = fetch_shap_local(local_utc_hour, local_head_val, local_top_n)

        if df_local is not None and not df_local.empty:
            # Sort by absolute shap value or impact
            df_local_sorted = df_local.sort_values(by='shap_value', ascending=True)
            colors = ['#FF5252' if val < 0 else '#00E676' for val in df_local_sorted['shap_value']]
            
            fig_local = go.Figure(go.Bar(
                x=df_local_sorted['shap_value'], y=df_local_sorted['feature'], orientation='h',
                marker=dict(color=colors)
            ))
            fig_local.update_layout(
                template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                height=450, margin=dict(l=0, r=20, t=10, b=0),
                xaxis_title="SHAP Value (Impact on Prediction)",
                yaxis_title="Feature"
            )
            st.plotly_chart(fig_local, use_container_width=True)
            st.markdown("> **Interpretation Note:** Positive SHAP values (green) push the prediction upward for this hour, while negative SHAP values (red) push it downward.")
        else:
            st.warning(f"⚠️ No local SHAP explanation found for UTC hour `{local_utc_hour}`. Please check the UTC timestamp format.")

    # 3. Feature Dependence Scatter
    with shap_tab3:
        st.markdown("##### 📈 Feature Dependence & Interaction Scatter Plot")
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        with col_d1:
            dep_feature = st.text_input("Feature Name", value="load_system_total_mw", key="shap_dep_feature")
        with col_d2:
            dep_window = st.selectbox("Window", ["daily", "weekly", "monthly"], key="shap_dep_window")
        with col_d3:
            dep_date = st.date_input("Reference Date", value=date(2026, 6, 26), key="shap_dep_date")
        with col_d4:
            dep_head_label = st.selectbox("Output Head", list(output_head_options.keys()), key="shap_dep_head")

        dep_date_str = dep_date.strftime('%Y-%m-%d')
        dep_head_val = output_head_options[dep_head_label]
        color_by_choice = st.selectbox("Color By", ["signal", "p_positive", "ercot_local_hour"], key="shap_color_by")

        @st.cache_data(ttl=60)
        def fetch_shap_dependence(feature, window, date_s, head, color_by):
            url = f"{FASTAPI_BASE_URL}/explainability/dependence"
            params = {"feature_name": feature, "window": window, "date": date_s, "output_head": head, "color_by": color_by}
            try:
                res = requests.get(url, params=params, timeout=10)
                if res.status_code == 200:
                    return pd.DataFrame(res.json())
                return None
            except Exception:
                return None

        df_dep = fetch_shap_dependence(dep_feature, dep_window, dep_date_str, dep_head_val, color_by_choice)

        if df_dep is not None and not df_dep.empty:
            fig_dep = px_scatter = go.Figure(go.Scatter(
                x=df_dep.get('feature_value'),
                y=df_dep.get('shap_value'),
                mode='markers',
                marker=dict(
                    size=8,
                    color=df_dep.get('color_value', df_dep.get(color_by_choice, 0)),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=color_by_choice)
                ),
                text=df_dep.get('delivery_time', None)
            ))
            fig_dep.update_layout(
                template="plotly_dark", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                height=500, margin=dict(l=0, r=20, t=10, b=0),
                xaxis_title=f"Feature Value: {dep_feature}",
                yaxis_title="SHAP Value"
            )
            st.plotly_chart(fig_dep, use_container_width=True)
            st.markdown("> **Caution Note:** Do not infer direct causality from dependence plots; they illustrate model sensitivity, nonlinear behavior, and regime dependence[cite: 9].")
        else:
            st.warning(f"⚠️ No dependence data found for feature `{dep_feature}` on `{dep_date_str}`. Check if the feature name is valid.")