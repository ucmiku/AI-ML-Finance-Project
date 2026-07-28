# components/theme.py
import streamlit as st
import base64
import os

def get_base64_image(image_path):
    """安全读取本地图片并转为 Base64"""
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

def inject_custom_css():
    """全局注入 Mercury 风格高科技浅色 CSS 主题"""
    st.markdown("""
        <style>
        
        /* ==========================================
           1. 字体与全局背景
           ========================================== */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
        
        .stApp {
            background-color: #FBFBFB !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', sans-serif !important;
            color: #111827 !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em !important;
        }
        
        [data-testid="stHeader"] {
            background-color: transparent !important;
        }

        /* ==========================================
           2. 彻底隐藏 Streamlit 默认导航栏 & 侧边栏置顶
           ========================================== */
        /* 🌟 核心修复：隐藏自动生成的灰色无图标导航栏 */
        [data-testid="stSidebarNav"] {
            display: none !important; 
        }

        [data-testid="stSidebarContent"] {
            padding-top: 20px !important;
        }

        [data-testid="stSidebar"] {
            contain: layout !important; 
        }

        [data-testid="stSidebarCollapseButton"] {
            z-index: 1000000 !important;
            position: fixed !important;
            top: 14px !important;
            right: 12px !important;
        }

        [data-testid="stSidebarCollapseButton"] button {
            background: transparent !important;
            border: none !important;
            color: #475569 !important;
        }

        .sidebar-logo-container {
            position: fixed !important;
            top: 0px !important;
            left: 0px !important;
            width: 100% !important;
            height: 60px !important;
            z-index: 999999 !important;
            
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
            justify-content: flex-start !important;
            padding-left: 18px !important;
            padding-right: 50px !important;
            
            background: rgba(248, 250, 252, 0.88) !important; 
            backdrop-filter: blur(12px) !important;
            -webkit-backdrop-filter: blur(12px) !important;
            border-bottom: 1px solid rgba(226, 232, 240, 0.8) !important;
            box-sizing: border-box !important;
        }

        .sidebar-logo-text {
            margin: 0 !important;
            padding: 0 !important;
            color: #0F172A !important;
            font-weight: 800 !important;
            font-size: 24px !important;
            line-height: 1 !important;
            letter-spacing: -0.5px !important;
            white-space: nowrap !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        /* ==========================================
           3. 数据表格与卡片 (Morandi Style)
           ========================================== */
        /* 卡片基础骨架 */
        .mercury-card {
            padding: 24px;
            border-radius: 12px;
            height: 200px;
            transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            backdrop-filter: blur(10px); /* 毛玻璃透明感 */
        }
        
        .mercury-card-title {
            color: #111827;
            font-size: 18px;
            font-weight: 600;
            margin-top: 0;
            margin-bottom: 8px;
        }
        
        .mercury-card-text {
            color: #475569;
            font-size: 14px;
            line-height: 1.6;
        }

        /* 莫兰迪一：鼠尾草绿 (Sage Green) */
        .morandi-sage {
            background: rgba(184, 196, 185, 0.15);
            border: 1px solid rgba(184, 196, 185, 0.4);
        }
        .morandi-sage:hover {
            border-color: rgba(184, 196, 185, 0.9);
            box-shadow: 0 10px 25px -5px rgba(184, 196, 185, 0.3);
            transform: translateY(-2px);
        }

        /* 莫兰迪二：茱萸粉 (Dusty Rose) */
        .morandi-rose {
            background: rgba(212, 196, 199, 0.15);
            border: 1px solid rgba(212, 196, 199, 0.4);
        }
        .morandi-rose:hover {
            border-color: rgba(212, 196, 199, 0.9);
            box-shadow: 0 10px 25px -5px rgba(212, 196, 199, 0.3);
            transform: translateY(-2px);
        }

        /* 莫兰迪三：雾霾蓝 (Haze Blue) */
        .morandi-haze {
            background: rgba(185, 196, 208, 0.15);
            border: 1px solid rgba(185, 196, 208, 0.4);
        }
        .morandi-haze:hover {
            border-color: rgba(185, 196, 208, 0.9);
            box-shadow: 0 10px 25px -5px rgba(185, 196, 208, 0.3);
            transform: translateY(-2px);
        }

        /* ==========================================
           4. 按钮与提示框 (Buttons & Alerts)
           ========================================== */
        [data-testid="stButton"] button {
            background-color: #FFFFFF !important;
            color: #111827 !important;
            border: 1px solid #D1D5DB !important;
            border-radius: 6px !important;
            padding: 8px 16px !important;
            font-weight: 500 !important;
            font-size: 14px !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
        }
        
        [data-testid="stButton"] button:hover {
            border-color: #9CA3AF !important;
            background-color: #F9FAFB !important;
            color: #000000 !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08) !important;
            transform: translateY(-1px) !important;
        }
        
        [data-testid="stButton"] button:active {
            transform: scale(0.97) !important;
            box-shadow: none !important;
            background-color: #F3F4F6 !important;
        }
        
        [data-testid="stAlert"] {
            border-radius: 8px !important;
            border: 1px solid #E5E7EB !important;
            padding: 12px 16px !important;
            background-color: #FFFFFF !important; 
            box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
            border-left-width: 4px !important;
        }

        /* ==========================================
           5. 板块专属页面加载浮入动效
           ========================================== */
        @keyframes terminalLoadBottom {
            0% { opacity: 0; transform: translateY(120px); }
            100% { opacity: 1; transform: translateY(0); }
        }

        @keyframes terminalLoadLeft {
            0% { opacity: 0; transform: translateX(-120px); }
            100% { opacity: 1; transform: translateX(0); }
        }

        @keyframes terminalLoadRight {
            0% { opacity: 0; transform: translateX(120px); }
            100% { opacity: 1; transform: translateX(0); }
        }

        [data-testid="stMain"] [data-testid="stVerticalBlock"] > div {
            opacity: 0;
            animation: terminalLoadBottom 2.2s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }

        [data-testid="stMain"] [data-testid="stVerticalBlock"] > div:nth-child(3n + 2) {
            animation-name: terminalLoadLeft;
        }

        [data-testid="stMain"] [data-testid="stVerticalBlock"] > div:nth-child(3n + 3) {
            animation-name: terminalLoadRight;
        }
        /* ==========================================
           6. 全局留白压缩与 Tabs 选项卡优化
           ========================================== */
        /* 压缩主页面顶部巨大的默认留白 */
        .block-container {
            padding-top: 3rem !important;
            padding-bottom: 2rem !important;
        }

        /* 优化 Tabs 的视觉，增加底部边框作为轨道，拉开间距 */
        [data-baseweb="tab-list"] {
            gap: 32px;
            border-bottom: 1px solid #E5E7EB;
        }
        [data-baseweb="tab"] {
            padding-bottom: 12px !important;
            font-size: 16.5px !important;
            font-weight: 600 !important;
        }

        /* ==========================================
           7. 赋予 st.container(border=True) 悬浮卡片质感
           ========================================== */
        /* 拦截 Streamlit 原生边框容器，强制赋予纯白背景和阴影 */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03) !important;
            padding: 0.5rem !important; /* 内部增加呼吸感 */
            transition: box-shadow 0.3s ease, border-color 0.3s ease !important;
        }
        
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06) !important;
            border-color: #CBD5E1 !important;
        }
        </style>
    """, unsafe_allow_html=True)

def render_sidebar_logo():
    """渲染侧边栏顶部自适应毛玻璃 Logo 遮罩（已移除多余的 st.page_link）"""
    icon_base64 = get_base64_image("pages/icon.png")
    
    if icon_base64:
        img_html = f'<img src="data:image/png;base64,{icon_base64}" style="width: 36px; height: 36px; border-radius: 8px; margin-right: 12px; object-fit: cover; box-shadow: 0 1px 3px rgba(0,0,0,0.1); display: block;">'
    else:
        img_html = '<div style="background:#10B981; width:36px; height:36px; border-radius:8px; display:flex; align-items:center; justify-content:center; margin-right:12px; font-size:18px; color:white;">⚡</div>'

    st.sidebar.markdown(f"""
        <div class="sidebar-logo-container">
            {img_html}
            <span class="sidebar-logo-text">GridWise</span>
        </div>
        <div style="height: 45px;"></div>
    """, unsafe_allow_html=True)