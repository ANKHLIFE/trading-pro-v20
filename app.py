import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf

st.set_page_config(page_title="專業交易診斷 v24", layout="wide")

def safe_read(file):
    try:
        df = pd.read_csv(file, encoding='utf-8')
    except:
        file.seek(0)
        df = pd.read_csv(file, encoding='cp950')
    df.columns = df.columns.str.strip()
    return df

def to_num(series):
    # 先轉為字串，移除所有非數字/點/負號的字元，再轉為 float
    return pd.to_numeric(series.astype(str).str.replace(r'[^0-9.-]', '', regex=True), errors='coerce').fillna(0)

st.title("🛡️ 專業期貨交易診斷系統 (v24)")

f1 = st.sidebar.file_uploader("1. 資金餘額 (CSV)", type="csv")
f2 = st.sidebar.file_uploader("2. 交易明細 (CSV)", type="csv")

if f1 and f2:
    try:
        db, dt = safe_read(f1), safe_read(f2)
        
        # --- 1. 核心型別轉換 (防止 str / str 錯誤) ---
        db['Total Net'] = to_num(db['Total Net'])
        db['Date'] = pd.to_datetime(db['Date'], errors='coerce').dt.normalize()
        db = db.dropna(subset=['Date', 'Total Net']).sort_values('Date')
        
        dt['Profit'] = to_num(dt['Profit'])
        dt['Sell Date'] = pd.to_datetime(dt['Sell Date'], errors='coerce').dt.normalize()

        # --- 2. 處理每日資金數據 ---
        # 確保 Total Net 是 float，並計算日回報率
        db_daily = db.groupby('Date')['Total Net'].last().reset_index()
        db_daily['Total Net'] = db_daily['Total Net'].astype(float)
        db_daily['User_Ret'] = db_daily['Total Net'].pct_change().fillna(0)

        # --- 3. 抓取大盤數據 ---
        start_d = db_daily['Date'].min().strftime('%Y-%m-%d')
        end_d = (db_daily['Date'].max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        
        @st.cache_data
        def get_bench(s, e):
            df_yf = yf.download("^TWII", start=s, end=e, progress=False)
            if df_yf.empty: return pd.Series()
            # 處理多層索引與欄位名稱
            target_col = 'Adj Close' if 'Adj Close' in df_yf.columns else ('Close' if 'Close' in df_yf.columns else df_yf.columns[0])
            res = df_yf[target_col]
            if isinstance(res, pd.DataFrame): res = res.iloc[:, 0]
            return res.astype(float).pct_change().fillna(0)

        market_ret = get_bench(start_d, end_d)
        market_ret.index = market_ret.index.normalize()

        # --- 4. 指標計算 ---
        user_series = db_daily.set_index('Date')['User_Ret']
        combined = pd.concat([user_series, market_ret], axis=1).dropna()
        combined.columns = ['User', 'Market']

        beta, alpha, sharpe, mdd = 0.0, 0.0, 0.0, 0.0
        if not combined.empty and len(combined) > 2:
            cov = np.cov(combined['User'], combined['Market'])[0, 1]
            m_var = combined['Market'].var()
            beta = cov / m_var if m_var != 0 else 0
            alpha = (combined['User'].mean() - beta * combined['Market'].mean()) * 252

        # MDD 計算 (強制轉換為 float 避免 str/str 錯誤)
        db_daily['CumMax'] = db_daily['Total Net'].cummax()
        db_daily['Drawdown'] = (db_daily['Total Net'].astype(float) - db_daily['CumMax'].astype(float)) / db_daily['CumMax'].astype(float)
        db_daily['Drawdown'] = db_daily['Drawdown'].fillna(0)
        mdd = db_daily['Drawdown'].min()
        
        vol = db_daily['User_Ret'].std() * np.sqrt(252)
        sharpe = (db_daily['User_Ret'].mean() * 252 - 0.01) / vol if vol > 0.001 else 0

        # --- 5. 畫面呈现 ---
        t1, t2, t3 = st.tabs(["🏠 核心績效", "🔍 行為診斷", "📉 風險控管"])

        with t1:
            st.metric("💰 目前帳戶總資產", f"${db_daily.iloc[-1]['Total Net']:,.0f}")
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Beta (市場相關性)", f"{beta:.2f}")
            c2.metric("Alpha (超額收益)", f"{alpha*100:.2f}%")
            c3.metric("Sharpe (夏普值)", f"{sharpe:.2f}")
            c4.metric("MDD (最大回撤)", f"{mdd*100:.2f}%")
            
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
            st.subheader("📉 累積收益對比")
            db_daily['User_Cum'] = (1 + db_daily['User_Ret']).cumprod()
            market_cum = (1 + market_ret[market_ret.index >= db_daily['Date'].min()]).cumprod()
            
            fig = px.line(title="累積收益率走勢 (個人 vs 大盤)")
            fig.add_scatter(x=db_daily['Date'], y=db_daily['User_Cum'], name="你的帳戶")
            fig.add_scatter(x=market_cum.index, y=market_cum.values, name="台股大盤")
            st.plotly_chart(fig, use_container_width=True)
            
            st.plotly_chart(px.area(db_daily, x='Date', y='Drawdown', title="歷史回撤圖 (Underwater Chart)", color_discrete_sequence=['#EF553B']), use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ 診斷出錯: {e}")
else:
    st.info("👈 請上傳 CSV 檔案開始分析")
