import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import warnings
import os  # 新增：用來建立記憶檔案

warnings.filterwarnings("ignore")

# ─── 網頁基本設定 ───
st.set_page_config(page_title="4H SMC 狙擊雷達", page_icon="🎯", layout="wide")
st.title("🎯 4H 全市場 SMC 狙擊雷達")
st.markdown("精準捕捉 **OB 訂單塊回踩** 與 **WaveTrend 恐慌轉折**。支援美股、台股與加密貨幣。")

# ─── 記憶功能：讀取與儲存收藏名單 ───
FAV_FILE = "favorites.txt"
DEFAULT_FAVS = "SPY, QQQ, NVDA, TSLA, 2330.TW, 0050.TW, BTC-USD, ^TWII, NQ=F, ES=F"

def load_favorites():
    if os.path.exists(FAV_FILE):
        with open(FAV_FILE, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content: return content
    return DEFAULT_FAVS

def save_favorites(fav_str):
    with open(FAV_FILE, "w", encoding="utf-8") as f:
        f.write(fav_str)

# ─── 核心運算函數 ───
@st.cache_data(ttl=3600)
def get_us_indices():
    def fetch_wiki(url):
        try:
            storage_options = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
            dfs = pd.read_html(url, storage_options=storage_options)
            for df in dfs:
                if 'Symbol' in df.columns: return df['Symbol'].str.replace('.', '-', regex=False).tolist()
                if 'Ticker' in df.columns: return df['Ticker'].str.replace('.', '-', regex=False).tolist()
        except Exception:
            pass
        return []
    
    sp500 = fetch_wiki("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    ndx   = fetch_wiki("https://en.wikipedia.org/wiki/Nasdaq-100")
    sox   = fetch_wiki("https://en.wikipedia.org/wiki/PHLX_Semiconductor_Sector")
    
    combined = list(dict.fromkeys([t for t in (sp500 + ndx + sox) if isinstance(t, str) and len(t) <= 7]))
    
    if len(combined) < 100:
        backup = [
            "SPY", "QQQ", "DIA", "IWM", "SOXX", "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", 
            "AMD", "WDC", "MU", "AVGO", "SMCI", "INTC", "QCOM", "TSM", "ASML", "ARM", "NVO",
            "PLTR", "MSTR", "COIN", "MARA", "DJT", "SPCE", "RDDT", "HOOD", "SNOW",
            "CEG", "VST", "NEE", "IONQ", "RKLB", "ASTS", "BTC-USD", "ETH-USD", "SOL-USD"
        ]
        return backup, False 
    return combined, True 

def rma(series, length): return series.ewm(alpha=1.0/length, adjust=False).mean()

def resample_to_tv_4h(df_1h):
    df = df_1h.copy()
    if df.index.tz is not None: df.index = df.index.tz_convert('America/New_York')
    else: df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert('America/New_York')
    date_group = df.index.date
    session_group = np.where(df.index.hour < 13, 0, 1)
    df_4h = df.groupby([date_group, session_group]).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
    df_4h.index = [pd.Timestamp(f"{d} {'09:30' if s==0 else '13:30'}").tz_localize('America/New_York') for d, s in df_4h.index]
    return df_4h.sort_index()

def scan_4h_signals(df, recent_k):
    if df is None or len(df) < 40: return None
    close, high, low, open_p = df['Close'], df['High'], df['Low'], df['Open']
    n = len(df)

    h_close_22 = close.rolling(window=22).max()
    wvf = ((h_close_22 - low) / h_close_22) * 100.0
    wvf_avg, wvf_std = wvf.rolling(20).mean(), wvf.rolling(20).std(ddof=0)
    wvf_ext = (wvf >= wvf_avg + 2.0 * wvf_std) | (wvf >= wvf.rolling(50).max() * 0.85)

    diff = close.diff().fillna(0)
    pos_diff, neg_diff = diff.clip(lower=0), (-diff).clip(lower=0)
    overbuy_count, oversell_count, rsi_sum = pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index)

    for l in range(10, 21):
        den = rma(diff.abs(), l).replace(0, np.nan)
        rsi_i = (100.0 * (rma(pos_diff, l) / den)).fillna(50.0)
        rsi_sum += rsi_i
        overbuy_count += (rsi_i > 70.0).astype(float)
        oversell_count += (rsi_i < 30.0).astype(float)

    N = 11
    avg_rsi = rsi_sum / N
    per_over, per_under = (overbuy_count / N) * 100.0, 100.0 - (oversell_count / N) * 100.0
    buy_rsi_ma, sell_rsi_ma = np.zeros(n), np.zeros(n)
    buy_rsi_ma[0], sell_rsi_ma[0] = avg_rsi.iloc[0], avg_rsi.iloc[0]
    ob_r, os_r, avg_v = (overbuy_count / N).values, (oversell_count / N).values, avg_rsi.values

    for i in range(1, n):
        buy_rsi_ma[i] = buy_rsi_ma[i-1] + ob_r[i] * (avg_v[i] - buy_rsi_ma[i-1])
        sell_rsi_ma[i] = sell_rsi_ma[i-1] + os_r[i] * (avg_v[i] - sell_rsi_ma[i-1])

    rsi_red_below = (oversell_count > 0.0) & (per_under <= sell_rsi_ma)
    rsi_strong_green = (overbuy_count > 0.0) & (per_over >= buy_rsi_ma) & (per_over >= 60.0)

    wt_ap = (high + low + close) / 3.0
    wt_esa = wt_ap.ewm(span=10, adjust=False).mean()
    wt_d = (wt_ap - wt_esa).abs().ewm(span=10, adjust=False).mean().replace(0, 0.0001)
    wt1 = ((wt_ap - wt_esa) / (0.015 * wt_d)).ewm(span=21, adjust=False).mean()
    wt2 = wt1.rolling(4).mean()

    rt_basis = close.ewm(span=55, adjust=False).mean()
    rt_vola = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1).ewm(span=55, adjust=False).mean()
    rt_upper, rt_lower = rt_basis + 4.0 * rt_vola, rt_basis - 4.0 * rt_vola

    red_dec = ((per_under > per_under.shift(1)) | (per_under == 100.0)) & (per_under.shift(1) < 100.0)
    mb_combo_bar = red_dec & wvf_ext
    top_cond_bar = (per_over < per_over.shift(1)) & (per_over.shift(1) > 0.0) & (wt1.shift(1) > 53.0)

    os_ext_trig, os_warn_trig, ob_warn_trig, wt_dead_trig = False, False, False, False
    top_fired_in_cycle, mb_fired_in_cycle = False, False
    rt_last_sig, rt_blw_cnt, rt_abv_cnt = -1000, 0, 0
    bull_obs, bear_obs, latest_signals = [], [], []
    last_ph, last_pl, ph_idx, pl_idx = float('inf'), -float('inf'), -1, -1

    for i in range(2, n):
        if low.iloc[i] < rt_lower.iloc[i]: rt_blw_cnt += 1
        else: rt_blw_cnt = 0
        if high.iloc[i] > rt_upper.iloc[i]: rt_abv_cnt += 1
        else: rt_abv_cnt = 0

        if (per_under.iloc[i] < 100.0) and not (per_under.iloc[i-1] < 100.0): mb_fired_in_cycle = False
        if (per_over.iloc[i] > 0.0) and not (per_over.iloc[i-1] > 0.0): top_fired_in_cycle = False
        if wt1.iloc[i] > -60.0: os_ext_trig = False
        if wt1.iloc[i] > -53.0: os_warn_trig = False
        if wt1.iloc[i] <= 53.0: ob_warn_trig = False
        if wt1.iloc[i] <= 60.0: wt_dead_trig = False

        size = 5
        if i >= size * 2:
            if max([high.iloc[j] for j in range(i - size*2, i + 1)]) == high.iloc[i - size]:
                last_ph, ph_idx = high.iloc
