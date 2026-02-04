import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf

# 基本網頁設定
st.set_page_config(page_title="專業交易診斷 v28", layout="wide")

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

st.title("🛡️ 專業期貨交易診斷系統")

f1 = st.sidebar.file_uploader("1. 資金餘額 (CSV)", type="csv")
f2 = st.sidebar.file_uploader("2. 交易明細 (CSV)", type="csv")

if f1 and f2:
    try:
        # --- 數據讀取與清理 ---
        db, dt = safe_read(f1), safe_read(f2)
        db['Total Net'] = to_num(db['Total Net'])
        db['Date'] = pd.to_datetime(db['Date'], errors='coerce').dt.normalize()
        db = db.dropna(subset=['Date', 'Total Net']).sort_values('Date')
        
        # 處理每日資金與報酬率
        db_daily = db.groupby('Date')['Total Net'].last().reset_index()
        db_daily['Raw_Ret'] = db_daily['Total Net'].pct_change().fillna(0)
        # 過濾異常出入金 (單日 > 30%)
        db_daily['User_Ret'] = db_daily['Raw_Ret'].apply(lambda x: x if abs(x) < 0.3 else 0)

        # --- 大盤數據對齊 ---
        start_d = db_daily['Date'].min().strftime('%Y-%m-%d')
        end_d = (db_daily['Date'].max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        
        @st.cache_data
        def get_market(s, e):
            df_yf = yf.download("^TWII", start=s, end=e, progress=False)
            if df_yf.empty: return pd.Series()
            col = 'Adj Close' if 'Adj Close' in df_yf.columns else df_yf.columns[0]
            bench = df_yf[col]
            if isinstance(bench, pd.DataFrame): bench = bench.iloc[:, 0]
            return bench.astype(float).pct_change().fillna(0)

        market_ret = get_market(start_d, end_d)
        market_ret.index = market_ret.index.normalize()

        # --- 風險指標計算 ---
        user_s = db_daily.set_index('Date')['User_Ret']
        combined = pd.concat([user_s, market_ret], axis=1).dropna()
        combined.columns = ['User', 'Market']

        beta, alpha, sharpe, mdd = 0.0, 0.0, 0.0, 0.0
        if not combined.empty and len(combined) > 2:
            cov = np.cov(combined['User'], combined['Market'])[0, 1]
            m_var = combined['Market'].var()
            beta = cov / m_var if m_var != 0 else 0
            alpha = (combined['User'].mean() - beta * combined['Market'].mean()) * 252
        
        db_daily['CumMax'] = db_daily['Total Net'].cummax()
        db_daily['Drawdown'] = (db_daily['Total Net'] - db_daily['CumMax']) / db_daily['CumMax']
        mdd = db_daily['Drawdown'].min()
        vol = db_daily['User_Ret'].std() * np.sqrt(252)
        sharpe = (db_daily['User_Ret'].mean() * 252 - 0.01) / vol if vol > 0.001 else 0

        # --- 介面呈現 ---
        t1, t2, t3 = st.tabs(["🏠 核心績效", "🔍 行為診斷", "📉 風險控管"])

        with t1:
            st.metric("💰 目前帳戶總資產", f"${db_daily.iloc[-1]['Total Net']:,.0f}")
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Beta", f"{beta:.2f}")
            c2.metric("Alpha (年化)", f"{alpha*100:.2f}%")
            c3.metric("Sharpe", f"{sharpe:.2f}")
            c4.metric("MDD", f"{mdd*100:.2f}%")
            
            dt['Profit'] = to_num(dt['Profit'])
            pnl_grp = dt.groupby('Underlying')['Profit'].sum().sort_values(ascending=False).reset_index()
            def fmt(v): return f"{int(round(v)):,}"
            
            cl, cr = st.columns(2)
            with cl:
                st.success("🟢 獲利前五名")
                st.table(pnl_grp.head(5).assign(Profit=lambda x: x['Profit'].apply(fmt)).rename(columns={'Underlying':'商品','Profit':'損益'}))
            with cr:
                st.error("🔴 虧損前五名")
                st.table(pnl_grp.tail(5).sort_values('Profit').assign(Profit=lambda x: x['Profit'].apply(fmt)).rename(columns={'Underlying':'商品','Profit':'損益'}))

        with t3:
            st.subheader("📉 績效與風險分析")
            db_daily['User_Cum'] = (1 + db_daily['User_Ret']).cumprod()
            m_cum = (1 + market_ret[market_ret.index >= db_daily['Date'].min()]).cumprod()
            fig = px.line(title="累積收益率走勢 (與大盤對比)")
            fig.add_scatter(x=db_daily['Date'], y=db_daily['User_Cum'], name="你的帳戶")
            fig.add_scatter(x=m_cum.index, y=m_cum.values, name="台股大盤")
            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(px.area(db_daily, x='Date', y='Drawdown', title="歷史回撤圖"), use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ 診斷出錯，請檢查 CSV 格式或日期欄位。報錯訊息: {e}")
else:
    st.info("👈 請在左側上傳「資金餘額」與「交易明細」CSV 檔案。")
