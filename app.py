import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import warnings
import os

warnings.filterwarnings("ignore")

# ─── 網頁基本設定 ───
st.set_page_config(page_title="4H SMC 狙擊雷達", page_icon="🎯", layout="wide")
st.title("🎯 4H 全市場 SMC 狙擊雷達")
st.markdown("精準捕捉 **OB 訂單塊回踩** 與 **WaveTrend 恐慌轉折**。支援美股、台股與加密貨幣。")

# ─── 記憶功能：讀取與儲存收藏名單 ───
FAV_FILE = "favorites.txt"
DEFAULT_FAVS = "SPY, QQQ, NVDA, TSLA, 2330.TW, 0050.TW, BTC-USD, BZ=F, ^TWII, NQ=F"

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
                last_ph, ph_idx = high.iloc[i - size], i - size
            if min([low.iloc[j] for j in range(i - size*2, i + 1)]) == low.iloc[i - size]:
                last_pl, pl_idx = low.iloc[i - size], i - size

            if close.iloc[i] > last_ph and ph_idx != -1:
                start_idx = max(0, pl_idx)
                ob_idx, min_l = start_idx, low.iloc[start_idx]
                for j in range(start_idx, i + 1):
                    if low.iloc[j] < min_l: min_l, ob_idx = low.iloc[j], j
                bull_obs.append({'top': high.iloc[ob_idx], 'bot': low.iloc[ob_idx]})
                last_ph = float('inf')

            if close.iloc[i] < last_pl and pl_idx != -1:
                start_idx = max(0, ph_idx)
                ob_idx, max_h = start_idx, high.iloc[start_idx]
                for j in range(start_idx, i + 1):
                    if high.iloc[j] > max_h: max_h, ob_idx = high.iloc[j], j
                bear_obs.append({'top': high.iloc[ob_idx], 'bot': low.iloc[ob_idx]})
                last_pl = -float('inf')

        bull_obs = [ob for ob in bull_obs if close.iloc[i] >= ob['bot']][-5:]
        bear_obs = [ob for ob in bear_obs if close.iloc[i] <= ob['top']][-5:]

        # ─── 訊號產生與時間/點位標記 ───
        curr_sigs = []
        t_str = df.index[i].strftime('%m/%d %H:%M') # 取得 K 棒時間
        c_p = f"{close.iloc[i]:.2f}"                # 取得當下收盤價

        if (wt1.iloc[i] > wt2.iloc[i]) and (wt1.iloc[i-1] <= wt2.iloc[i-1]) and (wt1.iloc[i-1] <= -60.0) and wvf_ext.iloc[i-1] and rsi_red_below.iloc[i-1]: 
            curr_sigs.append(f"🟢【恐慌抄底】({t_str} @ {c_p})")
        if (wt1.iloc[i] <= -60.0) and wvf_ext.iloc[i] and rsi_red_below.iloc[i] and not os_ext_trig:
            curr_sigs.append(f"🟣【極度超賣】({t_str} @ {c_p})"); os_ext_trig = True
        if (mb_combo_bar.iloc[i] or mb_combo_bar.iloc[i-1] or mb_combo_bar.iloc[i-2]) and (close.iloc[i] > close.iloc[i-1]) and (close.iloc[i-1] > close.iloc[i-2]) and not mb_fired_in_cycle:
            curr_sigs.append(f"🟡【Market Bottom】({t_str} @ {c_p})"); mb_fired_in_cycle = True
        if (wt1.iloc[i] <= -53.0) and wvf_ext.iloc[i] and not os_warn_trig:
            curr_sigs.append(f"🟪【超賣預警】({t_str} @ {c_p})"); os_warn_trig = True
        if (top_cond_bar.iloc[i] or top_cond_bar.iloc[i-1] or top_cond_bar.iloc[i-2]) and (close.iloc[i] < open_p.iloc[i]) and (close.iloc[i] < close.iloc[i-1]) and (low.iloc[i] <= low.iloc[i-1]) and (close.iloc[i-1] < open_p.iloc[i-1]) and (close.iloc[i-1] < close.iloc[i-2]) and (low.iloc[i-1] <= low.iloc[i-2]) and not top_fired_in_cycle:
            curr_sigs.append(f"⚠️【頂部預警】({t_str} @ {c_p})"); top_fired_in_cycle = True
        if (wt1.iloc[i] < wt2.iloc[i]) and (wt1.iloc[i-1] >= wt2.iloc[i-1]) and (wt1.iloc[i-1] > 60.0) and not wt_dead_trig:
            curr_sigs.append(f"🟠【WT 死叉】({t_str} @ {c_p})"); wt_dead_trig = True
        if (wt1.iloc[i] >= 53.0) and not ob_warn_trig:
            curr_sigs.append(f"🟡【超買預警】({t_str} @ {c_p})"); ob_warn_trig = True

        can_fire_rt = (i - rt_last_sig) >= 10
        if (close.iloc[i] > rt_lower.iloc[i]) and ((low.iloc[i] < rt_lower.iloc[i] and close.iloc[i-1] > rt_lower.iloc[i] and rt_blw_cnt <= 10) or (close.iloc[i-1] < rt_lower.iloc[i] and rt_blw_cnt <= 10)) and can_fire_rt and (oversell_count.iloc[i] > 0):
            curr_sigs.append(f"🛡️【空頭陷阱】({t_str} @ {c_p})"); rt_last_sig = i
        if (close.iloc[i] < rt_upper.iloc[i]) and ((high.iloc[i] > rt_upper.iloc[i] and close.iloc[i-1] < rt_upper.iloc[i] and rt_abv_cnt <= 10) or (close.iloc[i-1] > rt_upper.iloc[i] and rt_abv_cnt <= 10)) and can_fire_rt and (overbuy_count.iloc[i] > 0):
            curr_sigs.append(f"⚠️【多頭陷阱】({t_str} @ {c_p})"); rt_last_sig = i

        if i >= n - recent_k:
            for ob in bull_obs:
                if low.iloc[i-1] >= ob['top'] and low.iloc[i] <= ob['top']: 
                    curr_sigs.append(f"🧱【OB支撐回踩】({t_str} 踩 {ob['top']:.2f})")
            for ob in bear_obs:
                if high.iloc[i-1] <= ob['bot'] and high.iloc[i] >= ob['bot']: 
                    curr_sigs.append(f"🧱【OB壓力測試】({t_str} 測 {ob['bot']:.2f})")
            
            for sig in curr_sigs:
                if sig not in latest_signals: latest_signals.append(sig)

    if latest_signals:
        return {"標的代號": df.name if hasattr(df, 'name') else "", "最新收盤": f"{close.iloc[-1]:.2f}", "4H WT快線": round(float(wt1.iloc[-1]), 1), "觸發指標通知": " ｜ ".join(latest_signals)}
    return None

def format_vol(val):
    if val == 0 or pd.isna(val): return "N/A"
    if val >= 1e9: return f"{val/1e9:.2f} B"
    if val >= 1e6: return f"{val/1e6:.2f} M"
    return str(val)

# ─── 網頁側邊欄設定 ───
st.sidebar.header("⚙️ 雷達掃描設定")

saved_favs = load_favorites()

st.sidebar.subheader("📌 收藏標的 (自訂清單)")
fav_input = st.sidebar.text_area(
    "請輸入股票代號 (用半形逗號分隔)：",
    value=saved_favs,
    help="台股大盤請輸入 ^TWII，布蘭特原油 BZ=F，納斯達克期貨 NQ=F，S&P500期貨 ES=F。"
)

if st.sidebar.button("💾 儲存收藏清單"):
    save_favorites(fav_input)
    st.sidebar.success("清單已儲存！")

st.sidebar.subheader("🌍 全市場掃描擴充")
scan_us_indices = st.sidebar.checkbox("同步掃描美股三大指數 (約 900 檔)", value=False)

st.sidebar.subheader("🎯 訊號觸發條件")
recent_k = st.sidebar.number_input("掃描近 N 根 4H K線", min_value=1, max_value=10, value=3)

st.sidebar.subheader("📅 底層資料參數")
scan_days = st.sidebar.slider("載入歷史天數", min_value=30, max_value=90, value=60)

scan_btn = st.sidebar.button("🚀 立即啟動掃描", use_container_width=True)

# ─── 主程式掃描執行 ───
if scan_btn:
    favorites = [t.strip().upper() for t in fav_input.split(",") if t.strip()]
    all_tickers = favorites.copy()
    
    if scan_us_indices:
        with st.spinner("🔄 正在取得美股三大指數成分股清單..."):
            us_stocks, success = get_us_indices()
            if success:
                st.success("✅ 成功從維基百科取得 900+ 檔美股清單！")
            else:
                st.warning("⚠️ 雲端主機被維基百科阻擋爬蟲，已自動切換至內建的「Top 40 龍頭備用清單」！")
            all_tickers.extend(us_stocks)
    
    all_tickers = list(dict.fromkeys(all_tickers)) 
    
    if not all_tickers:
        st.error("❌ 找不到要掃描的標的，請確認自訂清單是否正確！")
        st.stop()

    st.info(f"🚀 準備就緒！即將掃描共 {len(all_tickers)} 檔標的...")
    progress_bar = st.progress(0)
    status_text = st.empty()

    chunk_size = 80
    triggered_results = []
    start_time = time.time()

    for i in range(0, len(all_tickers), chunk_size):
        batch = all_tickers[i:i+chunk_size]
        status_text.text(f"⏳ 正在下載與分析 K 線: {i} / {len(all_tickers)} ...")

        try:
            data = yf.download(batch, period=f"{scan_days}d", interval="1h", group_by='ticker', progress=False, threads=True)
            for t in batch:
                try:
                    df_t = data[t] if len(batch) > 1 else data
                    df_t = df_t.dropna(subset=['Close'])
                    if len(df_t) > 40:
                        df_t.name = t
                        res = scan_4h_signals(resample_to_tv_4h(df_t), recent_k)
                        if res:
                            res["標的代號"] = t
                            triggered_results.append(res)
                except: continue
        except: continue

        progress_bar.progress(min((i + chunk_size) / len(all_tickers), 1.0))

    status_text.text("🔍 掃描完畢！正在抓取交易量與排版...")

    if triggered_results:
        for res in triggered_results:
            ticker = res['標的代號']
            try:
                info = yf.Ticker(ticker).info
                res['Raw_Volume'] = info.get('averageVolume', info.get('volume', 0))
            except: res['Raw_Volume'] = 0

            res['日均交易量'] = format_vol(res['Raw_Volume'])
            res['Priority'] = 1 if ticker in favorites else 0
            if ticker in favorites: res['標的代號'] = f"⭐ {ticker}"

        df_report = pd.DataFrame(triggered_results)
        df_report = df_report.sort_values(by=['Priority', 'Raw_Volume'], ascending=[False, False])
        df_report = df_report.drop(columns=['Raw_Volume', 'Priority'])

        st.success(f"🎯 掃描完成！共發現 {len(df_report)} 檔觸發訊號。(耗時: {int(time.time() - start_time)} 秒)")
        st.dataframe(df_report[["標的代號", "日均交易量", "最新收盤", "4H WT快線", "觸發指標通知"]], use_container_width=True, height=600)
    else:
        st.warning(f"🎯 目前掃描清單內，近 {recent_k} 根 4H K線均未觸發指標。(耗時: {int(time.time() - start_time)} 秒)")
