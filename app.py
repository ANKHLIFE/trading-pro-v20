import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf

st.set_page_config(page_title="專業交易診斷 v21", layout="wide")

def safe_read(file):
    try:
        df = pd.read_csv(file, encoding='utf-8')
    except:
        file.seek(0)
        df = pd.read_csv(file, encoding='cp950')
    df.columns = df.columns.str.strip()
    return df

def to_num(series):
    return pd.to_numeric(series.astype(str).str.replace('[^0-9.-]', '', regex=True), errors='coerce').fillna(0)

st.title("🛡️ 專業期貨交易診斷系統 (穩定對齊版)")

f1 = st.sidebar.file_uploader("1. 資金餘額 (CSV)", type="csv")
f2 = st.sidebar.file_uploader("2. 交易明細 (CSV)", type="csv")

if f1 and f2:
    try:
        db, dt = safe_read(f1), safe_read(f2)
        dt['Profit'] = to_num(dt['Profit'])
        db['Total Net'] = to_num(db['Total Net'])
        db['Date'] = pd.to_datetime(db['Date'], errors='coerce')
        dt['Sell Date'] = pd.to_datetime(dt['Sell Date'], errors='coerce')
        dt['Buy Date'] = pd.to_datetime(dt['Buy Date'], errors='coerce')
        
        db = db.dropna(subset=['Total Net', 'Date']).sort_values('Date')
        dt = dt.dropna(subset=['Underlying', 'Profit'])

        # --- 數據計算核心 ---
        db['Returns'] = db['Total Net'].pct_change().fillna(0)
        
        # 抓取大盤數據
        try:
            start_d = db['Date'].min().strftime('%Y-%m-%d')
            end_d = (db['Date'].max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            bench_data = yf.download("^TWII", start=start_d, end=end_d, progress=False)['Adj Close']
            
            # 關鍵修正：確保 bench_data 是 Series 且處理多層索引問題
            if isinstance(bench_data, pd.DataFrame):
                bench_data = bench_data.iloc[:, 0]
            
            bench_ret = bench_data.pct_change().dropna()
            # 對齊日期
            combined = pd.DataFrame({'User': db.set_index('Date')['Returns'], 'Market': bench_ret}).dropna()
        except:
            combined = pd.DataFrame()

        # --- 指標計算 (防錯處理) ---
        beta, alpha, sharpe, mdd = 0.0, 0.0, 0.0, 0.0
        
        if not combined.empty and len(combined) > 2:
            cov = combined.cov().iloc[0, 1]
            m_var = combined['Market'].var()
            beta = cov / m_var if m_var != 0 else 0
            alpha = (combined['User'].mean() - beta * combined['Market'].mean()) * 252
        
        # MDD 計算
        db['CumMax'] = db['Total Net'].cummax()
        db['Drawdown'] = (db['Total Net'] - db['CumMax']) / db['CumMax']
        mdd = db['Drawdown'].min()
        
        # Sharpe (年化)
        vol = db['Returns'].std() * np.sqrt(252)
        sharpe = (db['Returns'].mean() * 252 - 0.01) / vol if vol > 0.0001 else 0

        # --- 分頁介面 ---
        tab1, tab2, tab3 = st.tabs(["🏠 核心績效", "🔍 行為診斷", "📉 風險控管"])

        with tab1:
            st.metric("💰 目前帳戶總資產", f"${db.iloc[-1]['Total Net']:,.0f}")
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Beta (市場相關性)", f"{beta:.2f}")
            c2.metric("Alpha (年化超額)", f"{alpha*100:.2f}%")
            c3.metric("Sharpe (夏普值)", f"{sharpe:.2f}")
            c4.metric("MDD (最大回撤)", f"{mdd*100:.2f}%")
            
            # 排行榜 (格式化)
            dt['Type'] = dt['Underlying'].apply(lambda x: '程式' if '小台' in str(x) else '手動')
            pnl_grp = dt.groupby('Underlying')['Profit'].sum().sort_values(ascending=False).reset_index()
            def fmt_c(v): return f"{int(round(v)):,}"
            
            col_l, col_r = st.columns(2)
            with col_l:
                st.success("🟢 獲利前五名")
                t5 = pnl_grp.head(5).copy(); t5['Profit'] = t5['Profit'].apply(fmt_c)
                st.table(t5.rename(columns={'Underlying':'商品', 'Profit':'損益'}))
            with col_r:
                st.error("🔴 虧損前五名")
                b5 = pnl_grp.tail(5).sort_values('Profit').copy(); b5['Profit'] = b5['Profit'].apply(fmt_c)
                st.table(b5.rename(columns={'Underlying':'商品', 'Profit':'損益'}))

        with tab3:
            st.subheader("📊 資金回撤壓力圖")
            fig_dd = px.area(db, x='Date', y='Drawdown', color_discrete_sequence=['#EF553B'])
            st.plotly_chart(fig_dd, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ 程式執行遇到問題: {e}")
else:
    st.info("👈 請上傳 CSV 檔案開始診斷")
