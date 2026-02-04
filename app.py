import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf

# 設定網頁標題
st.set_page_config(page_title="專業交易診斷 v26", layout="wide")

def safe_read(file):
    try:
        df = pd.read_csv(file, encoding='utf-8')
    except:
        file.seek(0)
        df = pd.read_csv(file, encoding='cp950')
    df.columns = df.columns.str.strip()
    return df

def to_num(series):
    return pd.to_numeric(series.astype(str).str.replace(r'[^0-9.-]', '', regex=True), errors='coerce').fillna(0)

st.title("🛡️ 專業期貨交易診斷系統 (穩定版)")

f1 = st.sidebar.file_uploader("1. 資金餘額 (CSV)", type="csv")
f2 = st.sidebar.file_uploader("2. 交易明細 (CSV)", type="csv")

if f1 and f2:
    try:
        # 讀取資料
        db, dt = safe_read(f1), safe_read(f2)
        
        # 1. 資金數據清理
        db['Total Net'] = to_num(db['Total Net'])
        db['Date'] = pd.to_datetime(db['Date'], errors='coerce').dt.normalize()
        db = db.dropna(subset=['Date', 'Total Net']).sort_values('Date')
        
        # 2. 每日資金處理 (處理出入金)
        db_daily = db.groupby('Date')['Total Net'].last().reset_index()
        db_daily['Raw_Ret'] = db_daily['Total Net'].pct_change().fillna(0)
        # 過濾異常波動 (>20% 視為出入金)
        db_daily['User_Ret'] = db_daily['Raw_Ret'].apply(lambda x: x if abs(x) < 0.2 else 0)

        # 3. 抓取大盤數據
        start_date = db_daily['Date'].min().strftime('%Y-%m-%d')
        end_date = (db_daily['Date'].max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        
        @st.cache_data
        def get_tw_bench(s, e
