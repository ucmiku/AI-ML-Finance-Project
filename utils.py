import streamlit as st
import os
import sqlite3
import pandas as pd
import joblib

@st.cache_resource
def init_database():
    """检查并初始化 SQLite 数据库"""
    db_name = "ercot_data.db"
    csv_name = "model_wide_hourly_2024_2026(1).csv"
    
    if not os.path.exists(db_name):
        if os.path.exists(csv_name):
            try:
                df = pd.read_csv(csv_name)
                conn = sqlite3.connect(db_name)
                df.to_sql("model_wide_hourly_2024_2026", conn, if_exists="replace", index=False)
                conn.close()
                return "✅ Database ready (Loaded from CSV)"
            except Exception as e:
                return f"❌ Database Error: {e}"
        else:
            return f"⚠️ Warning: '{csv_name}' missing. DB empty."
    return "✅ Database is active and connected."

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