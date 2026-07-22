import streamlit as st
import pickle 
import os
import requests

# 👉 确保此处导入了你刚才的后端函数。如果你的文件名不是 backend.py，请修改这里的 'backend'。
from rag_backend import get_sql_agent_response

st.set_page_config(page_title="GridWise AI Copilot", page_icon="🤖", layout="wide")

# --- 1. 记忆持久化配置 ---
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

# --- 2. 页面与安全检查 ---
st.markdown("### 🤖 GridWise Intelligence Copilot & Decision Terminal")
st.markdown("<small style='color: #888;'>Powered by LangChain SQL Agent + DeepSeek LLM</small>", unsafe_allow_html=True)
st.markdown("---")

# 安全检查：获取主页存入的 API Key
api_key = st.session_state.get('api_key', None)
if not api_key:
    st.warning("⚠️ **System Lock:** Please return to the Main Page and enter your DeepSeek API Key to unlock the decision terminal.")
    st.stop()

# --- 3. 侧边栏 ---
with st.sidebar:
    st.markdown("### 🔍 Active Knowledge Sources")
    st.success("📊 Real-Time DB: Connected via LangChain")
    st.success("🤖 DeepSeek LLM: Active")
    st.markdown("---")
    
    if st.button("🧹 Clear Conversation History", use_container_width=True):
        st.session_state.chat_history = []
        if os.path.exists(HISTORY_CACHE_FILE):
            os.remove(HISTORY_CACHE_FILE)
        st.rerun()

# --- 4. 初始化状态与渲染历史对话 ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_memory()

# 新增：用于存放语音转写后的文本草稿
if "draft_prompt" not in st.session_state:
    st.session_state.draft_prompt = None

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. 核心聊天逻辑 ---
st.markdown("---")

# 1. 渲染语音录制按钮
voice_text = speech_to_text(
    language='en-US',          
    start_prompt="🎙️ Click to Record Voice",
    stop_prompt="🛑 Stop Recording & Edit",
    just_once=True,
    key='voice_input'
)

# 2. 如果录音结束并返回了文本，存入草稿箱并刷新 UI
if voice_text:
    st.session_state.draft_prompt = voice_text
    st.rerun()

prompt_to_execute = None

# 3. 动态 UI 切换：如果有语音草稿，则显示“编辑区”；否则显示正常的聊天输入框
if st.session_state.draft_prompt is not None:
    st.info("💡 Voice captured! Edit your prompt below before sending:")
    # 提供一个多行文本框供用户修改语音识别结果
    edited_text = st.text_area("📝 Edit Prompt:", value=st.session_state.draft_prompt, height=100)
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("🚀 Send", use_container_width=True):
            prompt_to_execute = edited_text
            st.session_state.draft_prompt = None # 发送后清空草稿
    with col2:
        if st.button("🗑️ Cancel", use_container_width=True):
            st.session_state.draft_prompt = None
            st.rerun() # 取消后直接刷新页面，回到初始状态
else:
    # 正常的键盘输入流
    text_input = st.chat_input("Ask about ERCOT historical data (or use the mic above 🎙️)...")
    if text_input:
        prompt_to_execute = text_input

# ================= 执行 Agent 逻辑 =================
if prompt_to_execute:
    # 渲染用户提问
    st.session_state.chat_history.append({"role": "user", "content": prompt_to_execute})
    with st.chat_message("user"):
        st.markdown(prompt_to_execute)
        
    # 渲染 AI 回复
    with st.chat_message("assistant"):
        with st.spinner("🤖 Cloud Agent is analyzing the database..."):
            try:
                # 替换为调用后端的 FastAPI 接口
                api_url = "http://127.0.0.1:8000/v1/agent/ask" # 替换为他实际给你的网址
                payload = {
                    "question": prompt_to_execute,
                    "api_key": api_key
                }
                
                # 发送请求给后端
                response = requests.post(api_url, json=payload, timeout=60)
                
                if response.status_code == 200:
                    ai_response = response.json().get("answer", "No response from agent.")
                else:
                    ai_response = f"⚠️ Backend Error: {response.text}"
                
                st.markdown(ai_response)
                
                # 存入记忆
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                save_memory(st.session_state.chat_history)
                
            except Exception as e:
                error_msg = f"❌ **Network Execution Error:** {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                save_memory(st.session_state.chat_history)
                
    st.rerun() # 执行完毕后刷新页面，保持 UI 干净