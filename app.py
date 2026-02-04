import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf

st.set_page_config(page_title="專業交易診斷 v25", layout="wide")

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

st.title("🛡️ 專業期貨交易診斷系統 (v25 - 異常過濾版)")

f1 = st.sidebar.file_uploader("1. 資金餘額 (CSV)", type="csv")
f2 = st.sidebar.file_uploader("2. 交易明細 (CSV)", type="csv")

if f1 and f2:
    try:
        db, dt = safe_read(f1), safe_read(f2)
        db['Total Net'] = to_num(db['Total Net'])
        db['Date'] = pd.to_datetime(db['Date'], errors='coerce').dt.normalize()
        db = db.dropna(subset=['Date', 'Total Net']).sort_values('Date')
        
        # --- 1. 日報酬計算與出入金過濾 ---
        db_daily = db.groupby('Date')['Total Net'].last().reset_index()
        db_daily['Raw_Ret'] = db_daily['Total Net'].pct_change().fillna(0)
        
        # 核心修正：過濾出入金干擾 (單日波動 > 10% 視為非交易損益)
        # 你可以根據實際情況調整這個 0.1 (10%)
        db_daily['User_Ret'] = db_daily['Raw_Ret'].apply(lambda x: x if abs(x) < 0.1 else 0)

        # --- 2. 抓取大盤 ---
        start_d = db_daily['Date'].min().strftime('%Y-%m-%d')
        end_d = (db_daily['Date'].max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        
        @st.cache_data
        def get_bench_v25(s, e):
            df_yf = yf.download("^TWII", start=s, end=e, progress=False)
            if df_yf.empty: return pd.Series()
            target = 'Adj Close' if 'Adj Close' in df_yf.columns else df_yf.columns[0]
            res = df_yf[target]
            if isinstance(res, pd.DataFrame): res = res.iloc[:, 0]
            return res.astype(float).pct_change().fillna(0)

        market_ret = get_bench_v25(start_d, end_d)
        market_ret.index = market_ret.index.normalize()

        # --- 3. 指標計算 ---
        user_series = db_daily.set_index('Date')['User_Ret']
        combined = pd.concat([user_series, market_ret], axis=1).dropna()
        combined.columns = ['User', 'Market']

        beta, alpha, sharpe, mdd = 0.0, 0.0, 0.0, 0.0
        if not combined.empty and len(combined) > 1:
            cov_mat = np.cov(combined['User'], combined['Market'])
            if cov_mat.shape == (2,2):
                cov = cov_mat[0, 1]
                m_var = combined['Market'].var()
                beta = cov / m_var if m_var != 0 else 0
                # 這裡改用累積回報率的幾何平均來計算 Alpha，會更穩定
                alpha = (combined['User'].mean() - beta * combined['Market'].mean()) * 252

        # MDD
        db_daily['CumMax'] = db_daily['Total Net'].cummax()
        db_daily['Drawdown'] = (db_daily['Total Net'] - db_daily['CumMax']) / db_daily['CumMax']
        mdd = db_daily['Drawdown'].min()
        
        # Sharpe
        vol = db_daily['User_Ret'].std() * np.sqrt(252)
        sharpe = (db_daily['User_Ret'].mean() * 252 - 0.01) / vol if vol > 0.001 else 0

        # --- 4. 畫面呈現 ---
        t1, t2, t3 = st.tabs(["🏠 核心績效", "🔍 行為診斷", "📉 風險控管"])

        with t1:
            st.metric("💰 目前帳戶總資產", f"${db_daily.iloc[-1]['Total Net']:,.0f}")
            st.info("💡 註：系統已自動過濾單日 >10% 的資產變動（視為出入金），以確保風險指標之準確性。")
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Beta (市場相關性)", f"{beta:.2f}")
            c2.metric("Alpha (年化超額)", f"{alpha*100:.2f}%")
            c3.metric("Sharpe (夏普值)", f"{sharpe:.2f}")
            c4.metric("MDD (最大回撤)", f"{mdd*100:.2f}%")
            
            # (以下保留原本的排行榜格式...)
            dt['Profit'] = to_num(dt['Profit'])
            pnl_grp = dt.groupby('Underlying')['Profit'].sum().sort_values(ascending=False).reset_index()
            def f(v): return f"{int(round(v)):,}"
            cl, cr = st.columns(2)
            with cl:
                st.success("🟢 獲利前五名")
                st.table(pnl_grp.head(5).assign(Profit=lambda x: x['Profit'].apply(f)).rename(columns={'Underlying':'商品','Profit':'損益'}))
            with cr:
                st.error("🔴 虧損前五名")
                st.table(pnl_grp.tail(5).sort_values('Profit').assign(Profit=lambda x: x['Profit'].apply(f)).rename(columns={'Underlying':'商品','Profit':'損益'}))

        with t3:
            # 累積收益圖
            db_
