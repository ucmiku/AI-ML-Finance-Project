import streamlit as st
import time
import pandas as pd
import json
import pickle 
import os

st.set_page_config(page_title="GridWise AI Copilot", page_icon="🤖", layout="wide")
# --- 记忆持久化配置 ---
HISTORY_CACHE_FILE = "agent_memory.pkl"

def load_memory():
    """从本地读取聊天记忆"""
    if os.path.exists(HISTORY_CACHE_FILE):
        try:
            with open(HISTORY_CACHE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return []
    return []

def save_memory(history):
    """将聊天记忆保存到本地"""
    with open(HISTORY_CACHE_FILE, "wb") as f:
        pickle.dump(history, f)

# --- 1. Page Header (SaaS Layout) ---
st.markdown("### 🤖 GridWise Intelligence Copilot & Decision Terminal")
st.markdown(
    "<small style='color: #888;'>Integrating Live DB + ML Predictions + ERCOT Market Knowledge Base</small>", 
    unsafe_allow_html=True
)
st.markdown("---")

# Security Check
api_key = st.session_state.get('api_key', None)
if not api_key:
    st.warning("⚠️ **System Lock:** Please return to the Main Page and enter your DeepSeek API Key to unlock the decision terminal.")
    st.stop()

# --- 2. Sidebar: Source Observability ---
with st.sidebar:
    st.markdown("### 🔍 Active Knowledge Sources")
    st.success("📊 Real-Time DB: `model_wide_hourly_2024_2026` (Connected)")
    st.success("🤖 ML Model: `v2.4-Ensemble-LSTM` (Active)")
    st.success("📜 Knowledge Base: `ERCOT Nodal Protocols 2026` (Loaded)")
    st.markdown("---")
    
    # 修改这里的清除按钮逻辑
    if st.button("🧹 Clear Conversation History", use_container_width=True):
        st.session_state.chat_history = []
        if os.path.exists(HISTORY_CACHE_FILE):
            os.remove(HISTORY_CACHE_FILE) # 物理删除记忆文件
        st.rerun()

# --- 3. Initialize Global State ---
if "chat_history" not in st.session_state:
    # 以前是 = []，现在改为从文件读取
    st.session_state.chat_history = load_memory()

# --- 4. Scene Quick Commands (High-Frequency Queries) ---
st.markdown("##### ⚡ Quick Decision Commands")
cmd_cols = st.columns(4)

with cmd_cols[0]:
    if st.button("🔋 Optimal BESS Dispatch", use_container_width=True, help="Scenario: Strategy Recommendation"):
        st.session_state.preset_prompt = "Analyze the forecasted DAM-RTM spread for ERCOT North Hub tomorrow and recommend the optimal charge/discharge strategy for a Battery Energy Storage System (BESS)."
with cmd_cols[1]:
    if st.button("🚨 Price Spike Alert", use_container_width=True, help="Scenario: Risk Warning"):
        st.session_state.preset_prompt = "Identify any extreme high-price risks in the RTM for the next 24 hours driven by weather anomalies or load surges."
with cmd_cols[2]:
    if st.button("📈 Weekly Arbitrage Check", use_container_width=True, help="Scenario: Market Analysis"):
        st.session_state.preset_prompt = "Evaluate the spread arbitrage opportunities at the North Hub based on the DAM and RTM forecasts for the upcoming week."
with cmd_cols[3]:
    if st.button("📜 ERCOT Protocol Query", use_container_width=True, help="Scenario: Rule Retrieval"):
        st.session_state.preset_prompt = "Retrieve the latest ERCOT protocols regarding Virtual Bidding constraints and settlement rules."

# --- 5. Render Chat History ---
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Render structured dataframe and sources if they exist in history
        if "dataframe" in message:
            st.markdown("**📋 Raw Query Results:**")
            st.dataframe(message["dataframe"], use_container_width=True)
        if "sources" in message:
            st.caption(f"🔗 **Sources:** {message['sources']}")

# --- 6. Core Chat Input & Logic ---
prompt_input = st.chat_input("Ask anything to the decision terminal...")
prompt = prompt_input or st.session_state.get("preset_prompt", None)

if prompt:
    # Reset preset state
    if "preset_prompt" in st.session_state:
        st.session_state.preset_prompt = None
        
    # 1. Display User Input
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # 2. AI Response Area
    with st.chat_message("assistant"):
        text_placeholder = st.empty()
        
        with st.spinner("Agent syntax parsing & executing DB/ML query..."):
            try:
                # 🛠️ For production, replace with your actual backend call:
                # raw_response = get_sql_agent_response(prompt, api_key=api_key)
                
                time.sleep(1.5) # Simulating execution time
                
                # Mock AI text response in professional trading English
                text_analysis = """
### 📊 Quantitative Strategy Output
Based on the `v2.4-Ensemble-LSTM` model's DAM and RTM spread predictions for ERCOT North Hub tomorrow (July 21, 2026), the system has identified a significant intraday dual-peak arbitrage opportunity.

> 🔋 **Optimal BESS Dispatch Recommendation:**
> * **Charging Period:** **01:00 - 05:00**. Execute bulk charging. The model predicts negative or extremely low RTM prices averaging **$28/MWh**.
> * **Discharging Period:** **14:00 - 18:00**. Dispatch at full capacity. The evening peak load surge will widen the spread, with forecasted average prices reaching **$82/MWh**.
> * **Economic Evaluation:** The estimated arbitrage spread per cycle is **$54/MWh**. After accounting for a 15% round-trip efficiency loss, the net revenue remains highly favorable.

### 🚨 Trading Risk Notice
Weather forecasts for HE 17:00 indicate a risk of localized severe convection in North Texas. RTM prices may experience transient spikes (exceeding $300/MWh). It is recommended to inject power linearly in tranches and avoid over-exposing positions within a single trading interval.
                """
                
                # Mock Database DataFrame output
                mock_df = pd.DataFrame({
                    "HE (Hour Ending)": [f"{i:02d}:00" for i in range(1, 7)],
                    "DAM Predicted ($/MWh)": [32.5, 30.1, 28.4, 27.9, 29.0, 35.6],
                    "RTM Predicted ($/MWh)": [29.0, 26.2, 24.0, 25.5, 31.0, 42.1],
                    "Spread Forecast ($/MWh)": [-3.5, -3.9, -4.4, -2.4, +2.0, +6.5]
                })
                
                mock_sources = "📁 DB Table: `model_wide_hourly_2024_2026` | 🤖 ML Model: `v2.4-LSTM` | 📜 ERCOT Protocol Sec 4.4"
                
                # --- Rendering Logic ---
                # A. Render Text Analysis
                text_placeholder.markdown(text_analysis)
                
                # B. Render Raw Data Table
                st.markdown("**📋 Raw Database Outputs**")
                st.dataframe(mock_df, use_container_width=True)
                
                # C. Render Source Citations
                st.markdown(
                    f"<div style='background-color: rgba(255, 255, 255, 0.05); padding: 8px 12px; border-radius: 4px; border-left: 3px solid #00E676; margin-top: 15px;'>"
                    f"<small style='color: #bbb;'>🔗 <b>Sources:</b> {mock_sources}</small>"
                    f"</div>", 
                    unsafe_allow_html=True
                )
                
                # 3. Save full structured response to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": text_analysis,
                    "dataframe": mock_df,
                    "sources": mock_sources
                })
                
                # 新增这一行：只要有新对话，立刻物理存档
                save_memory(st.session_state.chat_history)
                
            except Exception as e:
                error_msg = f"❌ **Agent Execution Error:** {str(e)}"
                text_placeholder.markdown(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                # 报错信息也存下来
                save_memory(st.session_state.chat_history)
                
    st.rerun()