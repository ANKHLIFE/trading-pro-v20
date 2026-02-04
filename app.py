import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf

st.set_page_config(page_title="專業交易診斷 v27", layout="wide")

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

st.title("🛡️ 專業期貨交易診斷系統 (穩定修復版)")

f1 = st.sidebar.file_uploader("1. 資金餘額 (CSV)", type="csv")
f2 = st.sidebar.file_uploader("2. 交易明細 (CSV)", type="csv")

if f1 and f2:
    try:
        db, dt = safe_read(f1), safe_read(f2)
        
        # 1. 資金處理
        db['Total Net'] = to_num(db['Total Net'])
        db['Date'] = pd.to_datetime(db['Date'], errors='coerce').dt.normalize()
        db = db.dropna(subset=['Date', 'Total Net']).sort_values('Date')
        db_daily = db.groupby('Date')['Total Net'].last().reset_index()
        db_daily['User_Ret'] = db_daily['Total Net'].pct_change().fillna(0)
        # 過濾異常 (出入金過濾放寬至 30% 以免誤殺)
        db_daily['User_Ret'] = db_daily['User_Ret'].apply(lambda x: x if abs(x) < 0.3 else 0)

        # 2. 抓取大盤 (修正語法括號)
        start_d = db_daily['Date'].min().strftime('%Y-%m-%d')
        end_d = (db_daily['Date'].max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        
        @st.cache_data
        def get_tw_bench(s, e):
            df_yf = yf.download("^TWII", start=s, end=e, progress=False)
            if df_yf.empty: return pd.Series()
            col = 'Adj Close' if 'Adj Close' in df_yf.columns else df_yf.columns[0]
            bench = df_yf[col]
            if isinstance(bench, pd.DataFrame): bench = bench.iloc[:, 0]
            return bench.astype(float).pct_change().fillna(0)

        market_ret = get_tw_bench(start_d, end_d)
        market_ret.index = market_ret.index.normalize()

        # 3.
