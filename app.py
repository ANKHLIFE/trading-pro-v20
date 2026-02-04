import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf

st.set_page_config(page_title="專業交易診斷 v20", layout="wide")

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

st.title("🛡️ 專業期貨交易診斷系統 (大盤對比版)")

f1 = st.sidebar.file_uploader("1. 資金餘額 (CSV)", type="csv")
f2 = st.sidebar.file_uploader("2. 交易明細 (CSV)", type="csv")

if f1 and f2:
    try:
        db, dt = safe_read(f1), safe_read(f2)
        dt['Profit'] = to_num(dt['Profit'])
        db['Total Net'] = to_num(db['Total Net'])
        db['Date'] = pd.to_datetime(db['Date'], errors='coerce')
        db = db.dropna(subset=['Total Net', 'Date']).sort_values('Date')
        
        # --- 自動抓取大盤數據 (^TWII) ---
        start_date = db['Date'].min().strftime('%Y-%m-%d')
        end_date = db['Date'].max().strftime('%Y-%m-%d')
        
        @st.cache_data # 快取數據避免重複下載
        def get_bench_data(start, end):
            bench = yf.download("^TWII", start=start, end=end)['Adj Close']
            return bench.pct_change().dropna()

        bench_returns = get_bench_data(start_date, end_date)
        
        # --- 計算風險指標 ---
        db['Returns'] = db['Total Net'].pct_change().fillna(0)
        
        # 為了計算 Alpha/Beta，需要將個人回報與大盤回報對齊日期
        combined = pd.DataFrame({'User': db.set_index('Date')['Returns'], 'Market': bench_returns}).dropna()
        
        if len(combined) > 2:
            # Beta: 協方差 / 方差
            covariance = combined.cov().iloc[0, 1]
            market_variance = combined['Market'].var()
            beta = covariance / market_variance
            
            # Alpha: 個人回報 - (Beta * 大盤回報)
            # 這裡簡化為日平均超額收益並年化
            alpha = (combined['User'].mean() - beta * combined['Market'].mean()) * 252
            
            # MDD
            db['CumMax'] = db['Total Net'].cummax()
            db['Drawdown'] = (db['Total Net'] - db['CumMax']) / db_s['CumMax'] if 'db_s' in locals() else (db['Total Net'] - db['CumMax']) / db['CumMax']
            mdd = db['Drawdown'].min()
            
            # Sharpe
            vol = db['Returns'].std() * np.sqrt(252)
            sharpe = (db['Returns'].mean() * 252 - 0.01) / vol if vol != 0 else 0

        # --- 分頁呈現 ---
        t1, t2, t3 = st.tabs(["🏠 核心績效", "🔍 行為診斷", "📉 風險控管"])

        with t1:
            st.metric("💰 目前帳戶總資產", f"${db.iloc[-1]['Total Net']:,.0f}")
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Beta (市場相關性)", f"{beta:.2f}")
            c2.metric("Alpha (超額收益)", f"{alpha*100:.2f}%")
            c3.metric("Sharpe (夏普值)", f"{sharpe:.2f}")
            c4.metric("MDD (最大回撤)", f"{mdd*100:.2f}%")
            
            st.info(f"💡 Beta 為 {beta:.2f} 代表大盤漲 1%，你的資產變動約 {beta:.2f}%。Alpha 代表你靠個人技術贏過大盤的年化報酬率。")

        with t3:
            st.subheader("📉 資產對比與回撤分析")
            # 累積報酬率對比圖
            db['User_Cum'] = (1 + db['Returns']).cumprod()
            market_cum = (1 + bench_returns).cumprod()
            
            fig_compare = px.line(title="個人資產 vs 大盤累積回報")
            fig_compare.add_scatter(x=db['Date'], y=db['User_Cum'], name="你的帳戶")
            fig_compare.add_scatter(x=market_cum.index, y=market_cum.values, name="台股大盤")
            st.plotly_chart(fig_compare, use_container_width=True)
            
            st.plotly_chart(px.area(db, x='Date', y='Drawdown', title="歷史回撤圖"), use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ 數據抓取或計算出錯: {e}")
else:
    st.info("👈 請上傳 CSV 檔案，系統將自動連線 Yahoo Finance 抓取對比數據。")
