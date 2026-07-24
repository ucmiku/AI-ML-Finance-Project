import streamlit as st
import pickle
import os
import requests
import json

FASTAPI_AGENT_URL = "https://ai-ml-finance-project.onrender.com/v1/agent/ask"
HISTORY_CACHE_FILE = "agent_memory.pkl"

def load_memory():
    if os.path.exists(HISTORY_CACHE_FILE):
        try:
            with open(HISTORY_CACHE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return []
    return []

def save_memory(history):
    with open(HISTORY_CACHE_FILE, "wb") as f:
        pickle.dump(history, f)

# 处理用户提交的回调函数，彻底解决无限循环
def handle_agent_submit():
    user_msg = st.session_state.sidebar_agent_input
    if user_msg.strip():
        # 1. 把用户的问题加入历史记录
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        # 2. 标记有待处理的问题
        st.session_state.pending_agent_query = user_msg
        # 3. 立即清空输入框，斩断死循环
        st.session_state.sidebar_agent_input = ""

def render_global_copilot():
    """将 RAG 智能体完美嵌入侧边栏，并默认展开"""
    
    with st.expander("💬 GridWise AI Copilot (DeepSeek)", expanded=True):
        st.caption("Powered by LangChain Multi-Tool Agent")
        
        api_key = st.session_state.get('api_key', None)
        if not api_key:
            st.warning("⚠️ Enter DeepSeek API Key in Global Settings first.")
            return

        if st.button("🧹 Clear Chat History", key="clear_chat_sidebar"):
            st.session_state.chat_history = []
            if os.path.exists(HISTORY_CACHE_FILE):
                os.remove(HISTORY_CACHE_FILE)
            st.rerun()
            
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = load_memory()
            
        # 聊天记录展示区
        chat_container = st.container(height=350)
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # 使用回调机制的输入框
        st.text_input("Ask about ERCOT data...", key="sidebar_agent_input", on_change=handle_agent_submit)
        
        # 如果有待处理的问题，执行请求
        if st.session_state.get("pending_agent_query"):
            query = st.session_state.pending_agent_query
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("🤖 Analyzing Database & Live Context..."):
                        try:
                            # 获取当前页面的预测上下文 (如果是空说明用户还没看预测页面)
                            live_context = st.session_state.get("current_prediction_context", "No real-time prediction data is currently viewed by the user.")
                            
                            payload = {
                                "question": query, 
                                "api_key": api_key,
                                "dashboard_context": live_context # 喂给大模型的“页面正在展示的内容”
                            }
                            
                            response = requests.post(FASTAPI_AGENT_URL, json=payload, timeout=60)
                            if response.status_code == 200:
                                ai_response = response.json().get("answer", "No response.")
                            else:
                                ai_response = f"⚠️ Error {response.status_code}: {response.text}"
                            
                            clean_response = agent_response.replace("*", "")
                            st.markdown(clean_response)
                            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                            save_memory(st.session_state.chat_history)
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                            
            # 请求完成后，清理待处理标记并刷新渲染
            st.session_state.pending_agent_query = None
            st.rerun()