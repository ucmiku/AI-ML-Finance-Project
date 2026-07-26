import streamlit as st
import requests
from datetime import datetime
from utils import init_database
from components.agent_ui import render_global_copilot
from integration.streamlit_embed import render_ercot_map_workbench
from components.theme import inject_custom_css, render_sidebar_logo
# 1. 在顶部导入部分，加上忽略警告的代码，防止控制台刷屏
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 1. 页面配置
st.set_page_config(
    page_title="GridWise | Trading Terminal", 
    page_icon=":material/bolt:", 
    layout="wide"
)
inject_custom_css()

# 检查持久化登录 Token
if "token" in st.query_params and st.query_params["token"] == "valid":
    st.session_state['logged_in'] = True
elif 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

# ==========================================
# 2. 核心功能：NewsAPI 实时抓取 (带缓存防超额)
# ==========================================
@st.cache_data(ttl=3600)
def fetch_live_market_news(api_key):
    if not api_key:
        return []
    
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "(ERCOT OR \"Texas grid\" OR \"natural gas\")",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 6,
        "apiKey": api_key
    }
    try:
        # 【关键修复】加上 verify=False 绕过证书拦截
        response = requests.get(url, params=params, timeout=10, verify=False)
        if response.status_code == 200:
            return response.json().get("articles", [])
        else:
            st.error(f"News API Error: {response.json().get('message', 'Unknown Error')}")
            return []
    except Exception as e:
        st.error(f"Failed to fetch news: {e}")
        return []

# ==========================================
# 3. 页面标题
# ==========================================
st.markdown("""
    <h2 style='display: flex; align-items: center;'>
        <img src="https://api.iconify.design/icon-park-outline/lightning.svg?color=%2310B981" width="32" style="margin-right: 12px;"> 
        GridWise Quantitative Trading Terminal
    </h2>
""", unsafe_allow_html=True)
st.markdown("Welcome to the ERCOT North Hub DAM-RTM Spread Prediction System.")
st.markdown("---")

# ==========================================
# 4. 侧边栏：全局设置与 AI 助手
# ==========================================
with st.sidebar:
    render_sidebar_logo()
    
    st.markdown("---")
    st.page_link("Home.py", label="Terminal Home", icon=":material/dashboard:")
    st.page_link("pages/Live_Forecast.py", label="Live Forecast", icon=":material/show_chart:")
    st.page_link("pages/Model_Metrics.py", label="Model Metrics", icon=":material/analytics:")
    st.markdown("---")
        
    st.markdown("Copilot Access")
    st.markdown("Interact with the quantitative RAG engine anytime.")
    render_global_copilot()
    
    # Global Settings
    st.markdown("""
        <h3 style='display: flex; align-items: center; font-size: 16px;'>
            <img src="https://api.iconify.design/icon-park-outline/config.svg?color=%2322AF88" width="24" style="margin-right: 8px;"> 
            Global Settings
        </h3>
    """, unsafe_allow_html=True)
    
    api_key = st.text_input(
        "DeepSeek API Key:", 
        value="sk-8080fcbf46f3459895cd3f8daed48535", 
        type="password",                            
        help="Required for Agent and Strategy Generation"
    )
    if api_key:
        st.session_state['api_key'] = api_key
        
    news_api_key = st.text_input(
        "NewsAPI Key:", 
        value="19f00766ce7840209e29c52b905e166c", 
        type="password",                            
        help="Get it for free at newsapi.org to enable live market feeds."
    )
    if news_api_key:
        st.session_state['news_api_key'] = news_api_key
        
    st.markdown("---")
    
    # 用户状态模块
    if st.session_state['logged_in']:
        st.markdown("""
            <div style="display: flex; align-items: center; margin-bottom: 15px;">
                <div style="width: 36px; height: 36px; border-radius: 50%; background-color: #E2E8F0; display: flex; align-items: center; justify-content: center; margin-right: 12px;">
                    <img src="https://api.iconify.design/icon-park-outline/user.svg?color=%23475569" width="20">
                </div>
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

# ==========================================
# 5. 页面主体 Tabs
# ==========================================
tab_overview, tab_news = st.tabs([
    ":material/dashboard: Terminal Overview", 
    ":material/newspaper: Market & Policy Context"
])

with tab_overview:
    st.markdown("""
        <h3 style='display: flex; align-items: center;'>
            <img src="https://api.iconify.design/icon-park-outline/map-draw.svg?color=%232563EB" width="26" style="margin-right: 10px;"> 
            ERCOT Spatial Overview
        </h3>
    """, unsafe_allow_html=True)
    st.markdown("Interactive visualization of regional risk levels and weather forecast nodes.")
    render_ercot_map_workbench("http://127.0.0.1:5178", height=750)
    st.markdown("---")

    st.markdown("<h3 style='margin-bottom: 24px;'>Core Modules</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class="mercury-card morandi-sage">
                <h4 class="mercury-card-title">Live Forecast</h4>
                <p class="mercury-card-text">
                    Real-time ERCOT DAM-RTM spread predictions powered by the C1 Agent. Monitor 24-hour price trajectories and extreme weather flags.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="mercury-card morandi-rose">
                <h4 class="mercury-card-title">Trading Agent</h4>
                <p class="mercury-card-text">
                    LLM-driven trading assistant utilizing DeepSeek. Formulate strategies, analyze risk, and interact via natural language.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="mercury-card morandi-haze">
                <h4 class="mercury-card-title">Model Metrics</h4>
                <p class="mercury-card-text">
                    Comprehensive backtest analysis, strategy leaderboard, actual vs predicted tracking, and Explainable AI feature attribution.
                </p>
            </div>
        """, unsafe_allow_html=True)

with tab_news:
    st.markdown("### :material/newspaper: Live Energy Market Intelligence")
    st.markdown("Real-time feed monitoring Texas power grid conditions, extreme weather events, and natural gas markets.")
    
    # 悬浮卡片样式 (已放大标题并修复下划线)
    # 悬浮卡片样式 (彻底修复字体大小、颜色与等宽字体问题)
    st.markdown("""
        <style>
        /* 强制去掉链接自带的下划线和默认排版 */
        a.news-card {
            display: flex; flex-direction: row; align-items: center; 
            padding: 18px; margin-bottom: 16px; background-color: #FFFFFF; 
            border-radius: 12px; border: 1px solid #E5E7EB; 
            box-shadow: 0 1px 3px rgba(0,0,0,0.05); 
            transition: all 0.2s ease-in-out;
            cursor: pointer;
            text-decoration: none !important; 
        }
        
        a.news-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 16px -4px rgba(0,0,0,0.1);
            border-color: #D1D5DB;
        }
        
        a.news-card img {
            width: 150px; height: 100px; object-fit: cover; 
            border-radius: 8px; margin-right: 24px;
        }
        
        .news-tag {
            display: inline-block; padding: 4px 10px; border-radius: 4px; 
            font-size: 11px !important; font-weight: 700 !important; margin-bottom: 8px;
            background-color: #D1FAE5 !important; color: #065F46 !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        /* 🌟 核心修复：直接指定 a 标签下的 h4，强制 24px 大字号、极深黑灰色、正常字体 */
        a.news-card h4 { 
            margin: 0 0 8px 0 !important; 
            color: #0F172A !important; 
            font-size: 24px !important; 
            font-weight: 800 !important; 
            font-family: 'Inter', sans-serif !important; 
            text-decoration: none !important;
            line-height: 1.3 !important;
            transition: color 0.2s ease;
        }
        
        /* 鼠标悬浮时，标题变成主题蓝 */
        a.news-card:hover h4 {
            color: #2563EB !important;
        }
        
        /* 强制正文颜色加深，字号稍微调大 */
        a.news-card p { 
            margin: 0 0 10px 0 !important; 
            color: #334155 !important; 
            font-size: 15px !important; 
            line-height: 1.5 !important; 
            display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; 
            text-decoration: none !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        /* 日期/来源 颜色加深 */
        a.news-card span.news-meta { 
            font-size: 13px !important; 
            color: #64748B !important; 
            font-weight: 600 !important; 
            font-family: 'JetBrains Mono', monospace !important; 
            text-decoration: none !important;
        }
        </style>
    """, unsafe_allow_html=True)

    current_news_key = st.session_state.get('news_api_key', '')
    
    if not current_news_key:
        st.warning(":material/warning: Please enter your NewsAPI Key in the Global Settings (sidebar) to activate the live feed.")
    else:
        with st.spinner("Fetching latest energy markets data..."):
            articles = fetch_live_market_news(current_news_key)
            
        if not articles:
            st.info("No recent news found or API limit reached. Showing historical context instead.")
        else:
            for article in articles:
                # 处理缺失数据并设置默认封面图
                title = article.get("title", "No Title")
                desc = article.get("description", "No description available.")
                url = article.get("url", "#")
                source = article.get("source", {}).get("name", "Unknown Source")
                
                # 格式化时间
                raw_date = article.get("publishedAt", "")
                if raw_date:
                    try:
                        dt = datetime.strptime(raw_date, "%Y-%m-%dT%H:%M:%SZ")
                        date_str = dt.strftime("%b %d, %Y - %H:%M UTC")
                    except:
                        date_str = raw_date
                else:
                    date_str = "Recent"

                # 兜底图片
                img_url = article.get("urlToImage")
                if not img_url:
                    img_url = "https://images.unsplash.com/photo-1542273917363-3b1817f69a2d?auto=format&fit=crop&q=80&w=300&h=200"

                st.markdown(f"""
                    <a href="{url}" target="_blank" class="news-card">
                        <img src="{img_url}" alt="News Image" onerror="this.src='https://images.unsplash.com/photo-1478265409131-1f65c88f965c?auto=format&fit=crop&q=80&w=300&h=200';">
                        <div class="news-content">
                            <span class="news-meta">{date_str} &nbsp;•&nbsp; {source}</span>
                            <h4>{title}</h4>
                            <p>{desc}</p>
                            <span>{date_str} &nbsp;•&nbsp; {source}</span>
                        </div>
                    </a>
                """, unsafe_allow_html=True)
            
            if st.button(":material/sync: Force Refresh Cache"):
                fetch_live_market_news.clear()
                st.rerun()