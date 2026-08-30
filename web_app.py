"""
IDX Stock Screener - Versi Web (Streamlit)
=============================================
Ini "kulit" web dari stock_screener.py. Logika screening-nya SAMA PERSIS,
cuma ditampilkan lewat browser, bukan lewat teks di Command Prompt.

Cara pakai:
1. Pastikan file ini ada di FOLDER YANG SAMA dengan stock_screener.py
2. pip install streamlit plotly  (kalau belum)
3. Jalankan lewat Command Prompt: streamlit run web_app.py
   (BUKAN "python web_app.py" - Streamlit punya cara jalanin sendiri)
4. Browser bakal kebuka otomatis ke http://localhost:8501
"""

import streamlit as st
import pandas as pd
import ta
import base64
import os
import json
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

# Import semua fungsi yang udah dibuat di stock_screener.py
# (file ini HARUS ada di folder yang sama)
from stock_screener import (
    fetch_batch_data,
    fetch_data,
    compute_signals,
    compute_screener_results,
    compute_fundamental_score,
    compute_trade_levels,
    find_support_resistance,
    fetch_all_idx_tickers,
    fetch_macro_context,
    check_leading_lagging,
    compute_bandarmology_score,
    GOAPI_API_KEY,
    KONGLOMERAT_GROUPS,
    INDICATOR_DESCRIPTIONS,
    BROKER_SUMMARY_DESCRIPTION,
    BROKER_SMART_MONEY,
    BROKER_RETAIL,
    BROKER_INSIDER_MAP,
    WATCHLIST as DEFAULT_WATCHLIST,
)

# Indikator SKOR tambahan (di luar 3 screener utama) yang bisa dipilih user.
AVAILABLE_INDICATORS = {
    "rsi_oversold": "RSI Oversold",
    "macd_cross": "MACD Golden Cross",
    "volume_spike": "Volume Spike",
    "above_ma20": "Harga di atas MA20",
    "uptrend_ma": "MA20 > MA50 (uptrend)",
    "bollinger_riding": "Riding Upper Bollinger Band",
    "near_support": "Dekat Area Support",
    "vcp_pattern": "Pola VCP",
    "ema_riding": "EMA9 Riding",
    "stochrsi_support": "StochRSI Oversold di Support",
    "sr_role_reversal": "Support/Resistance Role Reversal",
    "range_sideways": "Sedang Sideways/Ranging",
}

# 3 screener utama. Saham ditampilkan kalau lolos SALAH SATU screener
# yang dicentang aktif. Watchlist-nya SELALU pakai LQ45 + konglomerat
# bawaan dari stock_screener.py (nggak ada lagi input manual).
SCREENER_INFO = {
    "day_trade": {
        "label": "📊 Day Trade",
        "help": "Volume ≥1.5x rata-rata, MACD histogram positif, harga di atas MA20, "
                "RSI 40-70, value transaksi > Rp1 miliar.",
    },
    "bsjp": {
        "label": "🌆 BSJP (Beli Sore Jual Pagi)",
        "help": "Naik ≥5%, volume breakout ≥2x MA20, harga di atas MA5 & Open, "
                "value transaksi > Rp5 miliar, bukan saham gocap.",
    },
    "bpjs": {
        "label": "🌅 BPJS (Beli Pagi Jual Sore)",
        "help": "Versi lebih longgar dari BSJP, biasa dicek 30 menit sebelum market buka.",
    },
}

# Mapping indikator mana yang "senasib" konsepnya sama tiap screener -
# dipakai buat auto-centang dua arah (screener <-> indikator).
SCREENER_INDICATOR_MAP = {
    "day_trade": ["volume_spike", "macd_cross", "above_ma20"],
    "bsjp": ["volume_spike", "above_ma20", "ema_riding"],
    "bpjs": ["volume_spike", "ema_riding"],
}

# Reverse map: tiap indikator -> daftar screener yang memuatnya.
INDICATOR_TO_SCREENERS = {}
for _scr_key, _ind_keys in SCREENER_INDICATOR_MAP.items():
    for _ind_key in _ind_keys:
        INDICATOR_TO_SCREENERS.setdefault(_ind_key, []).append(_scr_key)


def sync_indicators_from_screener(screener_key: str):
    """Callback: pas screener dicentang, auto-centang indikator yang terkait."""
    if st.session_state.get(f"scr_{screener_key}"):
        for ind_key in SCREENER_INDICATOR_MAP.get(screener_key, []):
            st.session_state[f"chk_{ind_key}"] = True


def sync_screener_from_indicator(indicator_key: str):
    """Callback: pas indikator dicentang, auto-centang screener yang terkait."""
    if st.session_state.get(f"chk_{indicator_key}"):
        for screener_key in INDICATOR_TO_SCREENERS.get(indicator_key, []):
            st.session_state[f"scr_{screener_key}"] = True


# Peta kriteria (key dari evidence screener) -> di baris/subplot mana
# kotak bukti harus digambar, dan berapa candle terakhir yang dilingkupi.
# "row" mengacu ke urutan subplot: 1=Harga, 2=Volume, 3=RSI, 4=MACD.
KEY_BOX_MAP = {
    "bsjp_naik5":  {"row": 1, "candles": 2},
    "bsjp_volume": {"row": 2, "candles": 2},
    "bsjp_ma5":    {"row": 1, "candles": 6},
    "bsjp_open":   {"row": 1, "candles": 1},
    "bsjp_value":  {"row": 2, "candles": 1},
    "bsjp_gocap":  {"row": 1, "candles": 1},
    "bpjs_ma5":    {"row": 1, "candles": 6},
    "bpjs_naik5":  {"row": 1, "candles": 2},
    "bpjs_open":   {"row": 1, "candles": 1},
    "bpjs_volume": {"row": 2, "candles": 2},
    "bpjs_value":  {"row": 2, "candles": 1},
    "dt_volume":   {"row": 2, "candles": 2},
    "dt_macd":     {"row": 4, "candles": 5},
    "dt_ma20":     {"row": 1, "candles": 21},
    "dt_rsi":      {"row": 3, "candles": 3},
    "dt_value":    {"row": 2, "candles": 1},
}


# =========================================================================
# RIWAYAT SCREENING (Google Sheets sebagai database eksternal, lewat
# jembatan Google Apps Script - TANPA Google Cloud Console/service account)
# - Riwayat PERMANEN (nggak auto-kehapus tiap hari) karena udah pindah ke
#   penyimpanan eksternal, bukan file lokal server lagi.
# - Dipakai buat 2 hal: (1) log semua screening (single & massal/all),
#   (2) cache broker summary per-ticker per-hari biar kuota GoAPI hemat
#   (kalau ticker udah pernah di-screening HARI INI, datanya dipakai ulang).
# =========================================================================
HISTORY_HEADERS = ["Tanggal", "Mode", "Waktu", "Ticker", "Harga", "RSI",
                    "ScreenerLolos", "BrokerSummaryStatus", "DataJSON"]

HISTORY_FALLBACK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "riwayat_screening_lokal.json")


def _today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _format_reasons_bullets(reasons: list) -> str:
    """
    Susun list alasan jadi poin-poin rapi (satu alasan per baris, diawali
    "•"), bukan digabung koma jadi satu paragraf panjang. Dipakai di
    tampilan app MAUPUN pas disimpan ke Google Sheets - kalau cell di
    Sheets di-set "Wrap text", ini bakal kebaca sebagai list ke bawah,
    bukan satu baris ngga jelas.
    """
    if not reasons:
        return "-"
    return "\n".join(f"• {r}" for r in reasons if r)


def _get_apps_script_config():
    """
    Balikin (url, token) Web App Google Apps Script dari Streamlit Secrets,
    atau (None, None) kalau belum diisi (app tetap jalan normal, fallback
    ke penyimpanan lokal - lihat catatan di _load_history_local).

    CARA SETUP (sekali aja, lihat panduan lengkap dari saya):
    1. Tempel skrip 'apps_script_riwayat.gs' ke Google Sheet kamu lewat
       Extensions > Apps Script, lalu Deploy sebagai Web App.
    2. Isi st.secrets dengan "APPS_SCRIPT_URL" (URL Web App hasil Deploy)
       dan "APPS_SCRIPT_TOKEN" (token rahasia yang kamu set di skrip itu)
       di Streamlit Cloud -> Settings -> Secrets.
    """
    try:
        url = st.secrets.get("APPS_SCRIPT_URL")
        token = st.secrets.get("APPS_SCRIPT_TOKEN")
    except Exception:
        return None, None
    if not url or not token:
        return None, None
    return url, token


def _load_history_local() -> dict:
    if not os.path.exists(HISTORY_FALLBACK_FILE):
        return {"rows": []}
    try:
        with open(HISTORY_FALLBACK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"rows": []}


def _save_history_local(data: dict):
    try:
        os.makedirs(os.path.dirname(HISTORY_FALLBACK_FILE), exist_ok=True)
        with open(HISTORY_FALLBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, default=str)
    except Exception:
        pass


def _append_history_row(mode: str, ticker: str = "", harga="", rsi="", screener_lolos: str = "",
                         broker_status: str = "-", data_obj=None) -> bool:
    """Simpan 1 baris riwayat. Balikin True kalau berhasil ke Google Sheets, False kalau fallback lokal."""
    row = {
        "Tanggal": _today_str(), "Mode": mode, "Waktu": datetime.now().strftime("%H:%M:%S"),
        "Ticker": ticker, "Harga": harga, "RSI": rsi, "ScreenerLolos": screener_lolos,
        "BrokerSummaryStatus": broker_status,
        "DataJSON": json.dumps(data_obj, default=str) if data_obj is not None else "",
    }
    url, token = _get_apps_script_config()
    if url:
        try:
            payload = dict(row)
            payload["token"] = token
            resp = requests.post(url, json=payload, timeout=10)
            if resp.ok and "error" not in resp.json():
                return True
        except Exception:
            pass
    local = _load_history_local()
    local.setdefault("rows", []).append(row)
    _save_history_local(local)
    return False


def _read_history_rows(mode: str = None, date_filter: str = None) -> list:
    url, token = _get_apps_script_config()
    if url:
        try:
            params = {"token": token}
            if mode:
                params["mode"] = mode
            if date_filter:
                params["date"] = date_filter
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            if "rows" in data:
                return data["rows"]
        except Exception:
            pass
    records = _load_history_local().get("rows", [])
    if date_filter:
        records = [r for r in records if r.get("Tanggal") == date_filter]
    if mode:
        records = [r for r in records if r.get("Mode") == mode]
    return records


def _get_cached_bandar_today(ticker: str):
    """Cari data broker summary ticker ini yang udah pernah diambil HARI INI (buat hemat kuota)."""
    rows = _read_history_rows(mode="single", date_filter=_today_str())
    for r in reversed(rows):
        if r.get("Ticker") == ticker and r.get("DataJSON"):
            try:
                return json.loads(r["DataJSON"]), r.get("Waktu")
            except Exception:
                continue
    return None, None


# =========================================================================
# BATAS HARIAN GoAPI - biar kuota nggak abis tanpa sadar
# =========================================================================
# INFO PEMAKAIAN GoAPI (bukan batasan keras - cuma info transparansi)
# =========================================================================
# Nggak ada blokir di sini. Tiap pengecekan Broker Summary itu sendiri udah
# didesain hemat: 1x panggilan API buat broker summary biasa, dan Buy-the-Dip
# cuma nambah MAKSIMAL 1x panggilan lagi (cek hari merah paling baru aja,
# bukan tiap hari merah) - lihat check_buy_the_dip_accumulation() di
# stock_screener.py. Counter di bawah cuma buat kamu pantau, nggak menghalangi apa-apa.

def _count_goapi_calls_today() -> int:
    """
    Hitung PERSIS berapa kali GoAPI udah kepanggil hari ini, dari riwayat
    (semua mode). Cuma hitung baris yang statusnya "Diambil baru" (bukan
    dari cache), dan ambil angka "api_calls_made" dari DataJSON masing-masing
    (fallback ke 1 kalau field itu nggak ada, misal riwayat versi lama).
    """
    rows = _read_history_rows(date_filter=_today_str())
    total = 0
    for r in rows:
        if str(r.get("BrokerSummaryStatus", "")).startswith("Diambil baru") and r.get("DataJSON"):
            try:
                data = json.loads(r["DataJSON"])
                total += int(data.get("api_calls_made", 1))
            except Exception:
                total += 1
    return total


def build_evidence_chart(df: pd.DataFrame, ticker: str, trade_levels: dict = None, highlight_item: dict = None):
    """
    Bikin grafik 4 panel (Harga+MA, Volume, RSI, MACD) sebagai BUKTI VISUAL.
    Kalau highlight_item diisi (dict evidence yang punya "key" & "label"),
    gambar KOTAK MERAH di panel & rentang candle yang jadi sumber indikator
    itu - bukan cuma nandain harga penutupan terakhir.
    """
    chart_df = df.tail(90)

    rsi_series = ta.momentum.RSIIndicator(df["Close"], window=14).rsi().tail(90)
    macd_hist_series = ta.trend.MACD(df["Close"]).macd_diff().tail(90)

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.025,
        row_heights=[0.46, 0.18, 0.18, 0.18],
        subplot_titles=(f"{ticker} — Harga, MA & Level Trading", "Volume", "RSI (14)", "MACD Histogram"),
    )

    # --- Row 1: Harga ---
    fig.add_trace(go.Candlestick(
        x=chart_df.index, open=chart_df["Open"], high=chart_df["High"],
        low=chart_df["Low"], close=chart_df["Close"], name="Harga",
    ), row=1, col=1)

    ma5 = df["Close"].rolling(5).mean().tail(90)
    ma20 = df["Close"].rolling(20).mean().tail(90)
    fig.add_trace(go.Scatter(x=chart_df.index, y=ma5, name="MA5",
                              line=dict(color="orange", width=1.3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_df.index, y=ma20, name="MA20",
                              line=dict(color="blue", width=1.3)), row=1, col=1)

    try:
        sr = find_support_resistance(df)
        for lvl in sr.get("support", []):
            fig.add_hline(y=lvl, line_dash="dot", line_color="gray", opacity=0.4, row=1, col=1)
        for lvl in sr.get("resistance", []):
            fig.add_hline(y=lvl, line_dash="dot", line_color="gray", opacity=0.4, row=1, col=1)
    except Exception:
        pass

    if trade_levels:
        fig.add_hline(y=trade_levels["support"], line_dash="solid", line_color="#2ecc71",
                      opacity=0.8, annotation_text=f"Support {trade_levels['support']:.0f}",
                      annotation_position="left", row=1, col=1)
        fig.add_hline(y=trade_levels["stop_loss"], line_dash="solid", line_color="#e74c3c",
                      opacity=0.8, annotation_text=f"Stop Loss {trade_levels['stop_loss']:.0f}",
                      annotation_position="left", row=1, col=1)
        fig.add_hline(y=trade_levels["take_profit_1"], line_dash="dash", line_color="#3498db",
                      opacity=0.8, annotation_text=f"TP1 {trade_levels['take_profit_1']:.0f}",
                      annotation_position="left", row=1, col=1)
        fig.add_hline(y=trade_levels["take_profit_2"], line_dash="dash", line_color="#9b59b6",
                      opacity=0.8, annotation_text=f"TP2 {trade_levels['take_profit_2']:.0f}",
                      annotation_position="left", row=1, col=1)

    # --- Row 2: Volume ---
    vol_colors = ["#2ecc71" if c >= o else "#e74c3c"
                  for c, o in zip(chart_df["Close"], chart_df["Open"])]
    fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["Volume"], name="Volume",
                          marker_color=vol_colors), row=2, col=1)
    vol_ma20 = df["Volume"].rolling(20).mean().tail(90)
    fig.add_trace(go.Scatter(x=chart_df.index, y=vol_ma20, name="Vol MA20",
                              line=dict(color="#8888ff", width=1, dash="dot")), row=2, col=1)

    # --- Row 3: RSI ---
    fig.add_trace(go.Scatter(x=chart_df.index, y=rsi_series, name="RSI",
                              line=dict(color="#9b59b6", width=1.3)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#e74c3c", opacity=0.5, row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#2ecc71", opacity=0.5, row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    # --- Row 4: MACD Histogram ---
    macd_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in macd_hist_series]
    fig.add_trace(go.Bar(x=chart_df.index, y=macd_hist_series, name="MACD Hist",
                          marker_color=macd_colors), row=4, col=1)
    fig.add_hline(y=0, line_color="gray", opacity=0.6, row=4, col=1)

    # --- Kotak bukti (bukan penanda harga akhir) ---
    if highlight_item:
        key = highlight_item.get("key")
        label = highlight_item.get("label", "")
        box_info = KEY_BOX_MAP.get(key)
        if box_info and len(chart_df.index) > 1:
            n = min(box_info["candles"], len(chart_df.index))
            freq = chart_df.index[-1] - chart_df.index[-2]
            pad = freq / 2
            x0 = chart_df.index[-n] - pad
            x1 = chart_df.index[-1] + pad
            target_row = box_info["row"]
            fig.add_vrect(
                x0=x0, x1=x1, row=target_row, col=1,
                fillcolor="rgba(255, 215, 0, 0.22)",
                line=dict(color="#e63946", width=2.5),
                annotation_text=f"📦 {label}", annotation_position="top left",
                annotation=dict(bgcolor="#FFF3CD", bordercolor="#e63946", font=dict(size=11)),
            )

    fig.update_layout(
        height=820, xaxis_rangeslider_visible=False,
        showlegend=True, margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# =========================================================================
# PENGATURAN TAMPILAN HALAMAN
# =========================================================================

st.set_page_config(page_title="KingBill Stock Screener", page_icon="👑", layout="wide")

# --- Splash screen singkat, muncul sekali per sesi ---
if "intro_shown" not in st.session_state:
    st.session_state.intro_shown = False

if not st.session_state.intro_shown:
    intro_placeholder = st.empty()
    intro_placeholder.markdown(
        """
        <style>
        @keyframes fadeScaleIn {
            0%   { opacity: 0; transform: scale(0.85); }
            60%  { opacity: 1; transform: scale(1.03); }
            100% { opacity: 1; transform: scale(1); }
        }
        @keyframes shimmer {
            0%   { background-position: -200% center; }
            100% { background-position: 200% center; }
        }
        .kb-intro-wrap {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 60vh;
            text-align: center;
            animation: fadeScaleIn 1.1s ease-out;
        }
        .kb-crown {
            font-size: 3.2rem;
            margin-bottom: 0.2rem;
            animation: fadeScaleIn 1.1s ease-out;
        }
        .kb-title {
            font-size: 2.6rem;
            font-weight: 800;
            letter-spacing: 2px;
            background: linear-gradient(90deg, #B8860B, #FFD700, #FFF3B0, #FFD700, #B8860B);
            background-size: 400% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: shimmer 2.5s linear infinite;
            margin: 0;
        }
        .kb-subtitle {
            font-size: 1.05rem;
            color: #888;
            margin-top: 0.6rem;
            letter-spacing: 1px;
        }
        </style>
        <div class="kb-intro-wrap">
            <div class="kb-crown">👑</div>
            <p class="kb-title">WELCOME TO<br>KINGBILL STOCK SCREENER</p>
            <p class="kb-subtitle">Instrumen screening saham IDX — sedang menyiapkan...</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    time.sleep(2.2)
    intro_placeholder.empty()
    st.session_state.intro_shown = True

def _load_hero_banner_base64() -> str:
    """
    Baca foto hero banner (assets/hero_banner.jpg, satu folder sama file ini)
    dan encode base64 buat dipasang sebagai CSS background-image. Kalau
    filenya belum ada, balikin string kosong biar app tetap jalan (fallback
    ke judul teks biasa).
    """
    hero_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "hero_banner.jpg")
    if not os.path.exists(hero_path):
        return ""
    with open(hero_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


_hero_b64 = _load_hero_banner_base64()

if _hero_b64:
    st.markdown(
        f"""
        <style>
        @keyframes kbFadeUp {{
            0%   {{ opacity: 0; transform: translateY(14px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes kbShimmer {{
            0%   {{ background-position: -200% center; }}
            100% {{ background-position: 200% center; }}
        }}
        @keyframes kbPulse {{
            0%, 100% {{ box-shadow: 0 0 0 0 rgba(255,215,0,0.35); }}
            50%      {{ box-shadow: 0 0 0 5px rgba(255,215,0,0); }}
        }}
        .kb-hero {{
            position: relative;
            width: 100%;
            border-radius: 14px;
            overflow: hidden;
            background-image: url("data:image/jpeg;base64,{_hero_b64}");
            background-size: cover;
            background-position: center;
            min-height: 280px;
            display: flex;
            align-items: center;
            padding: 0 2.4rem;
            margin-bottom: 0.6rem;
            border-bottom: 4px solid #B8860B;
        }}
        .kb-hero-text {{
            max-width: 660px;
            animation: kbFadeUp 0.9s ease-out;
        }}
        .kb-hero-eyebrow {{
            display: inline-block;
            color: #FFD700;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 2.5px;
            text-transform: uppercase;
            margin: 0 0 0.5rem 0;
            opacity: 0.9;
        }}
        .kb-hero-title {{
            font-size: 2.5rem;
            font-weight: 800;
            letter-spacing: 1px;
            background: linear-gradient(90deg, #B8860B, #FFD700, #FFF3B0, #FFD700, #B8860B);
            background-size: 400% auto;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: kbShimmer 3s linear infinite;
            margin: 0 0 0.5rem 0;
        }}
        .kb-hero-sub {{
            color: #f0f0f0;
            font-size: 1.03rem;
            margin: 0 0 1rem 0;
            line-height: 1.55;
            max-width: 560px;
        }}
        .kb-badge-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 0.9rem;
        }}
        .kb-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: rgba(255, 215, 0, 0.1);
            border: 1px solid rgba(255, 215, 0, 0.55);
            color: #FFE9A8;
            font-size: 0.82rem;
            font-weight: 600;
            padding: 0.28rem 0.75rem;
            border-radius: 999px;
            backdrop-filter: blur(2px);
        }}
        .kb-hero-cta {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            color: #FFD700;
            font-size: 0.9rem;
            font-weight: 700;
            border: 1.5px solid #FFD700;
            padding: 0.4rem 1rem;
            border-radius: 8px;
            animation: kbPulse 2.4s infinite;
        }}
        </style>
        <div class="kb-hero">
            <div class="kb-hero-text">
                <p class="kb-hero-eyebrow">IDX Stock Intelligence Tool</p>
                <p class="kb-hero-title">👑 KingBill Stock Screener</p>
                <p class="kb-hero-sub">Screening saham IDX otomatis pakai sinyal teknikal &amp;
                data broker summary REAL — biar riset kamu lebih cepat, tanpa gantikan
                keputusan trading kamu sendiri.</p>
                <div class="kb-badge-row">
                    <span class="kb-badge">🎯 3 Screener Siap Pakai</span>
                    <span class="kb-badge">📡 Broker Summary Real-Time</span>
                    <span class="kb-badge">📊 900+ Saham IDX</span>
                </div>
                <span class="kb-hero-cta">👇 Mulai screening di bawah</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.title("👑 KingBill Stock Screener")
    st.caption("Instrumen screening berbasis sinyal teknikal — bukan rekomendasi investasi")

st.warning(
    "⚠️ Ini alat bantu screening berbasis indikator teknikal historis, BUKAN prediksi "
    "harga yang pasti. Data delay 15-20 menit. Level TP/SL dihitung dari support/resistance "
    "historis, bukan jaminan harga akan bergerak ke situ. Selalu riset tambahan sebelum "
    "ambil keputusan trading."
)

# =========================================================================
# SIDEBAR - PENGATURAN (di kiri layar)
# =========================================================================

st.sidebar.header("⚙️ Pengaturan")

scan_all = st.sidebar.checkbox(
    "🌐 Scan SEMUA saham IDX (~900+)",
    value=False,
    help="Kalau dicentang, semua saham IDX yang tercatat bakal di-scan. "
         "Kalau nggak, pakai daftar bawaan LQ45 + saham konglomerat "
         "(lihat WATCHLIST di stock_screener.py). Prosesnya jauh lebih "
         "lama kalau scan semua (bisa beberapa menit).",
)

st.sidebar.caption(
    f"Mode: {'Semua saham IDX (~900+)' if scan_all else f'Watchlist bawaan ({len(DEFAULT_WATCHLIST)} saham: LQ45 + konglomerat)'}"
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Pilih Screener")
st.sidebar.caption(
    "Saham ditampilkan kalau lolos SALAH SATU screener yang dicentang. "
    "Centang screener otomatis ikut nyentang indikator terkait di bawah, "
    "dan sebaliknya."
)

active_screeners = []
for key, info in SCREENER_INFO.items():
    checked = st.sidebar.checkbox(
        info["label"], value=True, help=info["help"], key=f"scr_{key}",
        on_change=sync_indicators_from_screener, args=(key,),
    )
    if checked:
        active_screeners.append(key)

st.sidebar.markdown("---")
st.sidebar.subheader("✅ Indikator Skor Tambahan")
st.sidebar.caption("Di luar screener utama - nambah poin skor buat rangking (opsional).")

if "selected_indicators" not in st.session_state:
    st.session_state.selected_indicators = list(AVAILABLE_INDICATORS.keys())

col_a, col_b = st.sidebar.columns(2)
if col_a.button("Pilih Semua", width='stretch'):
    st.session_state.selected_indicators = list(AVAILABLE_INDICATORS.keys())
    for k in AVAILABLE_INDICATORS.keys():
        st.session_state[f"chk_{k}"] = True
if col_b.button("Kosongkan", width='stretch'):
    st.session_state.selected_indicators = []
    for k in AVAILABLE_INDICATORS.keys():
        st.session_state[f"chk_{k}"] = False

selected_indicators = []
for key, label in AVAILABLE_INDICATORS.items():
    related_screeners = INDICATOR_TO_SCREENERS.get(key, [])
    help_text = None
    if related_screeners:
        related_labels = ", ".join(SCREENER_INFO[s]["label"] for s in related_screeners)
        help_text = f"Terkait sama screener: {related_labels} - nyentang ini otomatis ikut nyentang screener itu juga."
    checked = st.sidebar.checkbox(
        label,
        value=(key in st.session_state.selected_indicators),
        key=f"chk_{key}",
        help=help_text,
        on_change=sync_screener_from_indicator, args=(key,),
    )
    if checked:
        selected_indicators.append(key)
st.session_state.selected_indicators = selected_indicators

include_news = st.sidebar.checkbox(
    "📰 Sertakan Sentimen Berita (penguat fundamental)",
    value=True,
    help="Cek berita terkini (maks 31 hari) soal emiten. Pakai analisis AI kalau API "
         "key sudah diisi di Streamlit Secrets, kalau belum fallback ke keyword matching. "
         "Link berita selalu disertakan buat verifikasi manual.",
)

include_macro = st.sidebar.checkbox(
    "🌍 Cek Konteks Makro (Dow/Nikkei/KOSPI/VIX)",
    value=True,
    help="Cek kondisi bursa global (Video 29) sebelum screening - malam bursa Amerika, "
         "pagi Nikkei & KOSPI. Cuma dicek sekali per klik screening, bukan per saham.",
)

goapi_configured = "ISI_" not in GOAPI_API_KEY and bool(GOAPI_API_KEY)

# =========================================================================
# PIN AKSES GoAPI - biar nggak sembarang orang yang buka app ini bisa
# manggil API (yang notabene kamu bayar) tanpa izin kamu dulu.
#
# PIN-nya DIISI LEWAT STREAMLIT SECRETS (panel web Streamlit Cloud), BUKAN
# ditulis langsung di kode ini - soalnya kode ini ada di GitHub, kalau
# repo-nya public (atau suatu saat di-publicin), siapa pun bisa lihat PIN-nya
# kalau ditulis di sini. Cara isi: buka app kamu di Streamlit Cloud > titik
# tiga > Settings > Secrets, tambahkan baris:
#   GOAPI_ACCESS_PIN = "080603"
# (ganti "080603" sesuka kamu). Simpan, app auto-restart, PIN langsung aktif.
# =========================================================================
try:
    GOAPI_ACCESS_PIN = st.secrets.get("GOAPI_ACCESS_PIN")
except Exception:
    GOAPI_ACCESS_PIN = None

if "goapi_unlocked" not in st.session_state:
    st.session_state.goapi_unlocked = False

if goapi_configured and GOAPI_ACCESS_PIN and not st.session_state.goapi_unlocked:
    with st.sidebar.expander("🔒 PIN Akses Broker Summary (GoAPI)", expanded=False):
        _pin_input = st.text_input(
            "Masukkan PIN buat buka fitur Broker Summary:", type="password", key="goapi_pin_input",
        )
        if _pin_input:
            if _pin_input == GOAPI_ACCESS_PIN:
                st.session_state.goapi_unlocked = True
                st.success("✅ PIN benar - fitur Broker Summary terbuka.")
            else:
                st.error("❌ PIN salah.")
elif goapi_configured and not GOAPI_ACCESS_PIN:
    st.sidebar.caption(
        "⚠️ PIN akses Broker Summary belum di-set (tambahin GOAPI_ACCESS_PIN "
        "di Streamlit Secrets) - fitur ini masih kebuka buat siapa aja yang buka web ini."
    )

# goapi_configured yang dipakai di seluruh app SEKARANG juga mensyaratkan
# PIN udah benar (KALAU PIN-nya di-set) - kalau belum, semua toggle Broker
# Summary tetap disabled walaupun API key-nya udah keisi.
goapi_configured = goapi_configured and (not GOAPI_ACCESS_PIN or st.session_state.goapi_unlocked)

include_bandarmology = False  # Broker Summary (GoAPI) SENGAJA dimatikan buat mode Screening
                               # Massal - biar kuota API nggak kepakai per-saham x puluhan
                               # saham. Fitur ini cuma bisa diaktifkan di bagian
                               # "🔎 Screening Satu Saham" di bawah, biar hasilnya per saham
                               # bisa dijelaskan & divalidasi detail satu-satu.
st.sidebar.caption(
    "📡 Broker Summary (GoAPI) nggak tersedia di mode ini biar kuota API nggak boros "
    "kalau nge-scan banyak saham sekaligus. Buka bagian **'🔎 Screening Satu Saham'** "
    "di bawah kalau mau lihat data broker summary lengkap + validasinya per saham."
    + ("" if "ISI_" not in GOAPI_API_KEY and bool(GOAPI_API_KEY) else " (API key GoAPI juga belum diisi di Streamlit Secrets.)")
)

run_button = st.sidebar.button("🔍 Jalankan Screening", type="primary", width='stretch')

# =========================================================================
# AREA UTAMA - HASIL SCREENING
# =========================================================================

if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.last_run = None
    st.session_state.evidence_map = {}
    st.session_state.chart_data = {}
    st.session_state.trade_levels_map = {}
    st.session_state.macro_context = None

if run_button:
    if not active_screeners:
        st.sidebar.error("Pilih minimal 1 screener dulu.")
        st.stop()

    if include_macro:
        with st.spinner("Mengecek konteks makro (bursa global)..."):
            st.session_state.macro_context = fetch_macro_context()
    else:
        st.session_state.macro_context = None

    if scan_all:
        with st.spinner("Mengambil daftar semua saham IDX..."):
            watchlist = fetch_all_idx_tickers()
        if not watchlist:
            st.error("Gagal mengambil daftar saham IDX. Cek koneksi internet, atau coba lagi nanti.")
            st.stop()
    else:
        watchlist = DEFAULT_WATCHLIST

    results = []
    evidence_map = {}
    chart_data = {}
    trade_levels_map = {}
    progress_bar = st.progress(0, text="Memulai screening (mode batch)...")

    batch_size = 50
    total_batches = (len(watchlist) + batch_size - 1) // batch_size

    for b in range(total_batches):
        batch = watchlist[b * batch_size: (b + 1) * batch_size]
        progress_bar.progress(
            (b + 1) / total_batches,
            text=f"Memproses batch {b + 1}/{total_batches} ({len(batch)} saham)...",
        )
        batch_data = fetch_batch_data(batch)

        for ticker, df in batch_data.items():
            screeners = compute_screener_results(df)

            passed_any = any(screeners[s]["passed"] for s in active_screeners if s in screeners)
            if not passed_any:
                continue

            tech = compute_signals(df, selected_indicators=selected_indicators)
            all_reasons = list(tech["reasons"])
            total_score = tech["score"]

            news_url = None
            if include_news:
                fund = compute_fundamental_score(ticker)
                total_score += fund["score"]
                all_reasons += fund["reasons"]
                news_url = fund.get("search_url")

            # Broker Summary (GoAPI) SENGAJA nggak dipanggil di sini (mode
            # massal) - biar kuota API nggak kepakai per-saham x puluhan
            # saham sekaligus. Fitur ini cuma jalan di "Screening Satu Saham".

            lolos_tags = []
            for s in active_screeners:
                if screeners.get(s, {}).get("passed"):
                    lolos_tags.append(SCREENER_INFO[s]["label"].split(" ", 1)[1])

            evidence_map[ticker] = {s: screeners[s]["evidence"] for s in active_screeners if s in screeners}
            chart_data[ticker] = df
            try:
                trade_levels_map[ticker] = compute_trade_levels(df)
            except Exception:
                trade_levels_map[ticker] = None

            results.append({
                "Ticker": ticker,
                "Harga": tech["price"],
                "Skor": total_score,
                "RSI": round(tech["rsi"], 1),
                "Lolos Screener": ", ".join(lolos_tags) if lolos_tags else "-",
                "Alasan": _format_reasons_bullets(all_reasons),
                "Link Berita": news_url or "",
                "_screener_keys": [s for s in active_screeners if screeners.get(s, {}).get("passed")],
            })

    progress_bar.empty()
    st.session_state.results = pd.DataFrame(results)
    st.session_state.evidence_map = evidence_map
    st.session_state.chart_data = chart_data
    st.session_state.trade_levels_map = trade_levels_map
    st.session_state.last_run = datetime.now().strftime("%d %b %Y, %H:%M:%S")

    # Log riwayat run screening MASSAL ini ke Google Sheets - SATU BARIS
    # PER SAHAM (bukan digepyok jadi 1 baris berisi JSON semua saham kayak
    # sebelumnya), biar kebaca rapi langsung di Sheets. Plus 1 baris ringkasan
    # di awal buat rekap total. Dibatasin ke top 30 by skor biar nggak
    # kelamaan (tiap baris = 1 request ke Google Sheets).
    MAX_ROWS_LOGGED_PER_SCAN = 30
    _top_for_log = sorted(results, key=lambda r: r["Skor"], reverse=True)[:MAX_ROWS_LOGGED_PER_SCAN]

    _append_history_row(
        mode="all_summary", ticker="RINGKASAN", harga="", rsi="",
        screener_lolos=f"{len(results)} saham lolos dari {len(watchlist)} discan "
                        f"({', '.join(SCREENER_INFO[s]['label'] for s in active_screeners)})",
        broker_status="-",
        data_obj={
            "watchlist_count": len(watchlist),
            "active_screeners": [SCREENER_INFO[s]["label"] for s in active_screeners],
            "total_lolos": len(results),
            "rows_logged": len(_top_for_log),
        },
    )

    if _top_for_log:
        with st.spinner(f"Menyimpan {len(_top_for_log)} baris riwayat ke Sheets (1 baris per saham)..."):
            for r in _top_for_log:
                _append_history_row(
                    mode="all", ticker=r["Ticker"], harga=r["Harga"], rsi=r["RSI"],
                    screener_lolos=r["Lolos Screener"],
                    broker_status="-",
                    data_obj={"skor": r["Skor"], "alasan": r["Alasan"], "link_berita": r["Link Berita"]},
                )

if st.session_state.results is not None:
    st.caption(f"Terakhir dijalankan: {st.session_state.last_run}")

    if st.session_state.macro_context:
        mc = st.session_state.macro_context
        icon = {1: "🟢", -1: "🔴", 0: "🟡"}[mc["sentiment_score"]]
        with st.expander(f"{icon} Konteks Makro: {mc['summary']}", expanded=False):
            if mc["indices"]:
                mc_cols = st.columns(len(mc["indices"]))
                for i, (sym, data) in enumerate(mc["indices"].items()):
                    mc_cols[i].metric(data["name"], f"{data['pct_change']:+.2f}%")
            else:
                st.caption("Data makro tidak tersedia saat ini.")
            st.caption(
                "Malam: bursa Amerika (Dow/S&P/Nasdaq/VIX). Pagi: Nikkei & KOSPI "
                "(buka ~2 jam lebih awal, dianggap lebih relate ke IHSG karena "
                "sama-sama regional Asia)."
            )

    df_results = st.session_state.results
    evidence_map = st.session_state.evidence_map

    col1, col2, col3 = st.columns(3)
    col1.metric("Saham lolos screener", len(df_results))
    col2.metric("Skor tertinggi", int(df_results["Skor"].max()) if len(df_results) else 0)
    col3.metric("Skor rata-rata", round(df_results["Skor"].mean(), 1) if len(df_results) else 0)

    st.subheader("Kandidat Hasil Screening")
    if len(df_results) > 0:
        display_cols = ["Ticker", "Harga", "Skor", "RSI", "Lolos Screener", "Alasan", "Link Berita"]
        col_config = {
            "Harga": st.column_config.NumberColumn(format="%d"),
            "Link Berita": st.column_config.LinkColumn("Cek Berita Manual", display_text="🔗 Buka"),
        }
        any_table_shown = False
        for screener_key in ["day_trade", "bsjp", "bpjs"]:
            if screener_key not in active_screeners:
                continue
            screener_label = SCREENER_INFO[screener_key]["label"]
            df_screener = df_results[df_results["_screener_keys"].apply(lambda keys, k=screener_key: k in keys)]
            df_screener = df_screener.sort_values("Skor", ascending=False)
            st.markdown(f"##### {screener_label} — {len(df_screener)} saham lolos")
            if len(df_screener) > 0:
                st.dataframe(df_screener[display_cols], width='stretch', hide_index=True, column_config=col_config)
                any_table_shown = True
            else:
                st.caption("Nggak ada saham yang lolos screener ini hari ini.")
        if not any_table_shown:
            st.caption("Belum ada saham yang lolos screener manapun.")

        with st.expander("📜 Riwayat Screening Massal (Screening All) — Log Semua Run", expanded=False):
            st.caption(
                "Log ini permanen (tersimpan di Google Sheets kalau sudah kamu setting, "
                "kalau belum fallback ke file lokal server) - nggak kehapus tiap hari."
            )
            all_runs = _read_history_rows(mode="all")
            if all_runs:
                run_rows = [{"Tanggal": r.get("Tanggal"), "Waktu": r.get("Waktu"),
                             "Ringkasan": r.get("ScreenerLolos")} for r in reversed(all_runs)]
                st.dataframe(pd.DataFrame(run_rows), width='stretch', hide_index=True)
            else:
                st.caption("Belum ada riwayat run screening massal tersimpan.")

        st.markdown("---")
        st.subheader("🔍 Bukti Validitas per Saham")
        st.caption(
            "Pilih saham, lalu klik salah satu kriteria indikator di bawah — grafiknya "
            "otomatis kasih penanda dan deskripsi kriteria itu muncul di bawahnya."
        )

        pilihan_ticker = st.selectbox("Pilih saham:", df_results["Ticker"].tolist())

        # Leading-Lagging: kalau ticker ini masuk grup konglomerat, cek status-nya
        ticker_group = None
        for gname, members in KONGLOMERAT_GROUPS.items():
            if pilihan_ticker in members:
                ticker_group = (gname, members)
                break
        if ticker_group:
            gname, members = ticker_group
            with st.spinner(f"Cek status leading-lagging grup '{gname}'..."):
                # pakai data yang udah ke-fetch kalau ada, kalau nggak fetch on-demand
                group_batch = {t: st.session_state.chart_data[t]
                                for t in members if t in st.session_state.chart_data}
                missing = [t for t in members if t not in group_batch]
                if missing:
                    group_batch.update(fetch_batch_data(missing))
            ll = check_leading_lagging(pilihan_ticker, group_batch)
            if ll["triggered"]:
                st.info(f"🔗 **Leading-Lagging**: {ll['value']}")
            else:
                st.caption(f"🔗 Leading-Lagging: {ll['value']}")

        if goapi_configured:
            st.caption(
                f"📡 Mau lihat data Broker Summary buat **{pilihan_ticker}**? Buka bagian "
                "**'🔎 Screening Satu Saham'** di bawah, ketik kodenya di sana - broker "
                "summary cuma jalan per-saham biar kuota GoAPI-nya hemat."
            )

        criteria_options = ["(Nggak ada yang dipilih - tampilan grafik biasa)"]
        criteria_lookup = {}
        if pilihan_ticker in evidence_map:
            for screener_key, evidence_list in evidence_map[pilihan_ticker].items():
                screener_label = SCREENER_INFO[screener_key]["label"]
                for item in evidence_list:
                    status_icon = "✅" if item["passed"] else "❌"
                    display = f"{status_icon} [{screener_label}] {item['label']}"
                    criteria_options.append(display)
                    criteria_lookup[display] = item

        pilihan_kriteria = st.selectbox("Klik untuk pilih indikator/kriteria:", criteria_options)

        trade_levels = st.session_state.trade_levels_map.get(pilihan_ticker)
        highlight_item = criteria_lookup.get(pilihan_kriteria)

        if pilihan_ticker and pilihan_ticker in st.session_state.chart_data:
            st.plotly_chart(
                build_evidence_chart(
                    st.session_state.chart_data[pilihan_ticker], pilihan_ticker,
                    trade_levels=trade_levels, highlight_item=highlight_item,
                ),
                width='stretch',
            )
            if highlight_item and highlight_item.get("key") not in KEY_BOX_MAP:
                st.caption("ℹ️ Kriteria ini belum punya area kotak spesifik di chart - tampilan di atas grafik biasa.")

        if pilihan_kriteria in criteria_lookup:
            item = criteria_lookup[pilihan_kriteria]
            desc = item.get("description") or "Belum ada deskripsi buat kriteria ini."
            status_text = "✅ **LOLOS**" if item["passed"] else "❌ **TIDAK LOLOS**"
            st.info(f"**{item['label']}** — {status_text}\n\n"
                    f"📝 {desc}\n\n"
                    f"📊 Nilai aktual: `{item['value']}`")

        if trade_levels:
            st.markdown("#### 🎯 Level Trading (perkiraan, bukan jaminan)")
            lc1, lc2, lc3, lc4 = st.columns(4)
            lc1.metric("Support", f"{trade_levels['support']:,.0f}")
            lc2.metric("Stop Loss", f"{trade_levels['stop_loss']:,.0f}",
                       delta=f"{(trade_levels['stop_loss']/trade_levels['price']-1)*100:.1f}%",
                       delta_color="inverse")
            lc3.metric("Take Profit 1", f"{trade_levels['take_profit_1']:,.0f}",
                       delta=f"{(trade_levels['take_profit_1']/trade_levels['price']-1)*100:.1f}%")
            lc4.metric("Take Profit 2", f"{trade_levels['take_profit_2']:,.0f}",
                       delta=f"{(trade_levels['take_profit_2']/trade_levels['price']-1)*100:.1f}%")
            if trade_levels.get("risk_reward_1"):
                st.caption(
                    f"Risk:Reward ke TP1 ≈ 1:{trade_levels['risk_reward_1']}, "
                    f"ke TP2 ≈ 1:{trade_levels['risk_reward_2']}. Support/resistance "
                    f"dihitung dari data historis - selalu cek ulang manual sebelum entry."
                )

        st.markdown("---")
        if pilihan_ticker and pilihan_ticker in evidence_map:
            for screener_key, evidence_list in evidence_map[pilihan_ticker].items():
                screener_label = SCREENER_INFO[screener_key]["label"]
                with st.expander(f"{screener_label} — semua kriteria", expanded=False):
                    if not evidence_list:
                        st.caption("Tidak ada data (kemungkinan data historis kurang).")
                        continue
                    ev_df = pd.DataFrame(evidence_list)[["label", "passed", "value"]]
                    ev_df["passed"] = ev_df["passed"].map({True: "✅ Lolos", False: "❌ Tidak"})
                    ev_df = ev_df.rename(columns={
                        "label": "Kriteria", "passed": "Status", "value": "Nilai Aktual",
                    })
                    st.dataframe(ev_df, width='stretch', hide_index=True)
    else:
        st.info("Nggak ada saham yang lolos screener yang dipilih. Coba ganti pilihan screener di sidebar.")
else:
    st.info("👈 Pilih screener & klik tombol **'Jalankan Screening'** di sidebar kiri untuk mulai.")


# =========================================================================
# SCREENING SATU SAHAM (cek cepat, nggak perlu scan semua watchlist dulu)
# =========================================================================

st.markdown("---")
st.markdown("---")
st.header("🔎 Screening Satu Saham")
st.caption("Cek satu saham langsung tanpa nunggu scan semua watchlist selesai dulu.")

col_input, col_button = st.columns([3, 1])
single_ticker_input = col_input.text_input(
    "Kode saham (contoh: BBCA, TLKM, GOTO):", value="", key="single_ticker_input",
    placeholder="Ketik kode saham tanpa .JK",
).strip()
single_cek_button = col_button.button("🔎 Cek Saham Ini", type="primary", width='stretch')

single_col_a, single_col_b = st.columns(2)
include_bandarmology_single = single_col_a.checkbox(
    "📡 Sertakan Broker Summary (GoAPI)", value=goapi_configured, disabled=not goapi_configured,
    help="Data broker net-buy/sell REAL dari GoAPI.IO buat SATU saham ini aja. Kalau "
         "saham ini SUDAH pernah di-cek hari ini, datanya dipakai dari cache (nggak "
         "manggil API lagi) biar kuota hemat. " + ("Isi GOAPI_API_KEY di Streamlit Secrets dulu."
         if not goapi_configured else "API key terdeteksi."),
)
include_buy_the_dip_single = single_col_b.checkbox(
    "📉 Cek juga Buy-the-Dip (30 hari terakhir)", value=False, disabled=not include_bandarmology_single,
    help="Cek apakah broker smart money akumulasi pas harga lagi turun tajam. HEMAT API: "
         "cuma cek 1 hari merah paling baru dalam 30 hari terakhir (bukan tiap hari merah), "
         "jadi maksimal +1x panggilan API doang - atau 0x kalau nggak ada hari merah sama sekali.",
)
if not goapi_configured:
    st.caption("⚠️ GOAPI_API_KEY belum diisi di Streamlit Secrets, atau PIN akses belum dimasukkan/benar - Broker Summary nggak akan jalan.")
else:
    _calls_today = _count_goapi_calls_today()
    st.caption(
        f"ℹ️ **Kuota GoAPI terpakai hari ini: {_calls_today} panggilan** (info doang, bukan "
        f"batasan - tiap pengecekan di sini emang udah didesain hemat: maksimal 1 panggilan "
        f"buat Broker Summary + 1 lagi kalau Buy-the-Dip diaktifkan). Cek dari cache TIDAK "
        f"ikut dihitung ke sini."
    )

_ticker_preview = single_ticker_input.upper() if single_ticker_input else None
if _ticker_preview and not _ticker_preview.endswith(".JK"):
    _ticker_preview += ".JK"
_cached_bandar_preview, _cached_bandar_time = (_get_cached_bandar_today(_ticker_preview)
                                                if _ticker_preview else (None, None))
force_refresh_bandar = False
if _cached_bandar_preview:
    st.caption(
        f"📦 **{_ticker_preview}** udah pernah di-screening hari ini pukul "
        f"**{_cached_bandar_time or '-'}** - broker summary bakal dipakai dari cache "
        "(hemat kuota GoAPI), bukan manggil API lagi."
    )
    force_refresh_bandar = st.checkbox(
        "🔄 Paksa ambil ulang data broker summary (abaikan cache hari ini)",
        value=False, key="force_refresh_bandar",
    )

if "single_result" not in st.session_state:
    st.session_state.single_result = None
    st.session_state.single_df = None
    st.session_state.single_trade_levels = None

if single_cek_button and single_ticker_input:
    ticker_full = single_ticker_input.upper()
    if not ticker_full.endswith(".JK"):
        ticker_full += ".JK"

    with st.spinner(f"Menganalisis {ticker_full}..."):
        df_single = fetch_data(ticker_full)

        if df_single.empty:
            st.session_state.single_result = {"found": False, "ticker": ticker_full}
        else:
            screeners_single = compute_screener_results(df_single)
            tech_single = compute_signals(df_single)
            trade_levels_single = compute_trade_levels(df_single)

            fund_single = None
            if include_news:
                fund_single = compute_fundamental_score(ticker_full)

            bandar_single = None
            bandar_from_cache = False
            cached_bandar, cached_bandar_time = _get_cached_bandar_today(ticker_full)
            # Cache dianggap valid buat request ini kalau: ada, dan (nggak minta
            # buy-the-dip ATAU buy-the-dip-nya udah pernah kesimpan juga), dan
            # user nggak nge-klik "paksa ambil ulang".
            cache_ok = bool(
                cached_bandar and not force_refresh_bandar
                and (not include_buy_the_dip_single or cached_bandar.get("buy_the_dip"))
            )
            if include_bandarmology_single and goapi_configured:
                if cache_ok:
                    bandar_single = cached_bandar
                    bandar_from_cache = True
                else:
                    # Nggak ada gate/batasan di sini - pengecekannya sendiri
                    # udah didesain hemat (maksimal 1 panggilan API buat
                    # broker summary + 1 lagi kalau Buy-the-Dip diaktifkan,
                    # lihat check_buy_the_dip_accumulation() di stock_screener.py).
                    bandar_single = compute_bandarmology_score(
                        ticker_full, df_single, include_buy_the_dip=include_buy_the_dip_single,
                    )

            ll_single = None
            for gname, members in KONGLOMERAT_GROUPS.items():
                if ticker_full in members:
                    group_batch_single = fetch_batch_data(members)
                    ll_single = check_leading_lagging(ticker_full, group_batch_single)
                    break

            st.session_state.single_result = {
                "found": True,
                "ticker": ticker_full,
                "screeners": screeners_single,
                "tech": tech_single,
                "fund": fund_single,
                "bandar": bandar_single,
                "bandar_from_cache": bandar_from_cache,
                "leading_lagging": ll_single,
            }
            st.session_state.single_df = df_single
            st.session_state.single_trade_levels = trade_levels_single

            # Simpan riwayat ke Google Sheets (permanen). DataJSON cuma diisi
            # kalau broker summary BARU diambil (bukan dari cache) - biar nggak
            # dobel-nyimpen data yang sama berkali-kali.
            screeners_passed_labels = [SCREENER_INFO[k]["label"] for k, v in screeners_single.items() if v["passed"]]
            broker_status = "-"
            data_to_log = None
            if include_bandarmology_single and goapi_configured:
                if bandar_from_cache:
                    broker_status = f"Cache (diambil {cached_bandar_time})"
                else:
                    broker_status = "Diambil baru"
                    data_to_log = bandar_single
            _append_history_row(
                mode="single", ticker=ticker_full,
                harga=float(tech_single.get("price", 0) or 0), rsi=float(tech_single.get("rsi", 0) or 0),
                screener_lolos=", ".join(screeners_passed_labels) or "-",
                broker_status=broker_status, data_obj=data_to_log,
            )

single_result = st.session_state.single_result

if single_result is None:
    st.caption("👆 Ketik kode saham dan klik tombol buat mulai analisis.")
elif not single_result.get("found"):
    st.error(f"❌ Data untuk {single_result['ticker']} tidak ditemukan (kode salah, atau saham kurang dari 30 hari transaksi).")
else:
    ticker = single_result["ticker"]
    tech = single_result["tech"]
    df_single = st.session_state.single_df
    trade_levels_single = st.session_state.single_trade_levels

    st.success(f"**{ticker}** — Harga sekarang: **{tech['price']:,.0f}** | RSI: **{tech['rsi']:.1f}**")

    # Status di semua screener
    scr_cols = st.columns(len(SCREENER_INFO))
    for i, (skey, sinfo) in enumerate(SCREENER_INFO.items()):
        passed = single_result["screeners"].get(skey, {}).get("passed", False)
        scr_cols[i].metric(sinfo["label"], "✅ Lolos" if passed else "❌ Tidak")

    if single_result.get("leading_lagging"):
        ll = single_result["leading_lagging"]
        if ll["triggered"]:
            st.info(f"🔗 **Leading-Lagging**: {ll['value']}")
        else:
            st.caption(f"🔗 Leading-Lagging: {ll['value']}")

    if single_result.get("fund") and single_result["fund"]["reasons"]:
        st.info(f"📰 **Berita**: {single_result['fund']['reasons'][0]}")
        if single_result["fund"].get("search_url"):
            st.caption(f"[🔗 Cek semua berita terkait]({single_result['fund']['search_url']})")

    if single_result.get("bandar") and single_result["bandar"].get("available"):
        bandar = single_result["bandar"]
        acc = bandar.get("broker_accumulation") or {}
        dip = bandar.get("buy_the_dip")

        st.markdown("### 📡 Broker Summary (GoAPI) — Detail & Validasi")
        if single_result.get("bandar_from_cache"):
            st.caption("📦 Data ini dari **cache hari ini** (saham ini sudah pernah di-screening sebelumnya hari ini), bukan panggilan API baru - hemat kuota GoAPI.")

        with st.expander("📖 Apa itu Broker Summary & cara membacanya?", expanded=True):
            st.caption(acc.get("description") or BROKER_SUMMARY_DESCRIPTION)
            st.markdown(
                "**Cara memvalidasi hasil di bawah ini:**\n"
                "1. Cek tabel **'Data Mentah per Broker'** — ini rekap net-buy/sell "
                "LANGSUNG dari GoAPI, belum diolah kesimpulannya. Verifikasi manual di sini.\n"
                "2. Cek tabel **'Kriteria Bandarmology'** — tiap baris adalah satu kriteria "
                "penilaian (mis. siapa net-buyer terbesar, kategorinya apa) dengan status "
                "Lolos/Tidak, biar nggak cuma percaya kesimpulan mentah-mentah.\n"
                "3. Cocokkan kode broker di tabel data mentah dengan **'Tabel Referensi "
                "Klasifikasi Broker'** di bawah - biar tahu kenapa suatu broker dianggap "
                "'Smart Money' atau 'Retail'.\n"
                "4. Ingat: ini indikator KONTEKS tambahan, bukan sinyal beli/jual "
                "berdiri sendiri - selalu gabungkan dengan analisis teknikal & fundamental."
            )

        if bandar["reasons"]:
            for r in bandar["reasons"]:
                st.info(f"**Kesimpulan**: {r}")
        else:
            st.caption("Nggak ada sinyal akumulasi smart money terdeteksi hari ini.")

        evidence_bandar = acc.get("evidence") or []
        if evidence_bandar:
            st.markdown(f"**📋 Kriteria Bandarmology ({acc.get('date', '-')}):**")
            ev_bandar_df = pd.DataFrame(evidence_bandar)[["label", "passed", "value"]]
            ev_bandar_df["passed"] = ev_bandar_df["passed"].map({True: "✅ Lolos", False: "❌ Tidak"})
            ev_bandar_df = ev_bandar_df.rename(columns={"label": "Kriteria", "passed": "Status", "value": "Nilai Aktual"})
            st.dataframe(ev_bandar_df, width='stretch', hide_index=True)
            pilihan_kriteria_bandar = st.selectbox(
                "Klik kriteria buat lihat penjelasan lengkapnya:",
                ["(Nggak ada yang dipilih)"] + [e["label"] for e in evidence_bandar],
                key="bandar_kriteria_select",
            )
            for e in evidence_bandar:
                if e["label"] == pilihan_kriteria_bandar:
                    status_text = "✅ **LOLOS**" if e["passed"] else "❌ **TIDAK LOLOS**"
                    st.info(f"**{e['label']}** — {status_text}\n\n📝 {e['description']}\n\n📊 Nilai aktual: `{e['value']}`")

        raw_table = acc.get("raw_table") or []
        if raw_table:
            st.markdown(f"**🧾 Data Mentah per Broker ({acc.get('date', '-')}) — langsung dari GoAPI, belum diolah:**")
            st.dataframe(pd.DataFrame(raw_table), width='stretch', hide_index=True)
            n_smart = sum(1 for r in raw_table if "Smart Money" in r["Kategori"])
            n_retail = sum(1 for r in raw_table if "Retail" in r["Kategori"])
            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("Total Broker Aktif", len(raw_table))
            bc2.metric("Net Smart Money (Rp)", f"{acc.get('total_smart_net', 0):,.0f}")
            bc3.metric("Net Retail (Rp)", f"{acc.get('total_retail_net', 0):,.0f}")
        else:
            st.caption("Data mentah broker per kode nggak tersedia (cek kuota/koneksi GoAPI).")

        with st.expander("📚 Tabel Referensi Klasifikasi Broker (buat validasi manual)", expanded=False):
            st.caption(
                "Referensi dari riset manual - dipakai buat nge-kategorikan tiap kode broker "
                "di tabel data mentah di atas. Bisa berubah sewaktu-waktu, cek ulang berkala."
            )
            ref_rows = []
            for code, name in BROKER_SMART_MONEY.items():
                ref_rows.append({"Kode": code, "Nama Sekuritas": name, "Kategori": "🟢 Smart Money"})
            for code, name in BROKER_RETAIL.items():
                ref_rows.append({"Kode": code, "Nama Sekuritas": name, "Kategori": "🔵 Retail"})
            for code, name in BROKER_INSIDER_MAP.items():
                ref_rows.append({"Kode": code, "Nama Sekuritas": f"Terkait {name}", "Kategori": "🟡 Insider/Konglomerat"})
            st.dataframe(pd.DataFrame(ref_rows), width='stretch', hide_index=True)

        if dip:
            with st.expander("📉 Buy-the-Dip Check (30 hari terakhir)", expanded=False):
                st.caption(dip.get("description", ""))
                status_text = "✅ **TERDETEKSI**" if dip["triggered"] else "❌ **Nggak terdeteksi**"
                st.info(f"{status_text} — {dip['value']}")
                dip_evidence = dip.get("evidence") or []
                if dip_evidence:
                    st.markdown("**Rincian per hari yang harganya turun:**")
                    st.dataframe(pd.DataFrame(dip_evidence), width='stretch', hide_index=True)
                else:
                    st.caption("Nggak ada hari dengan penurunan harga ≥1% dalam 30 hari terakhir.")
        elif include_bandarmology_single and not include_buy_the_dip_single:
            st.caption("ℹ️ Centang 'Cek juga Buy-the-Dip' di atas kalau mau lihat analisis akumulasi saat harga turun.")

    st.markdown("#### Semua Kriteria per Screener")
    single_criteria_options = ["(Nggak ada yang dipilih - tampilan grafik biasa)"]
    single_criteria_lookup = {}
    for screener_key, sdata in single_result["screeners"].items():
        screener_label = SCREENER_INFO.get(screener_key, {}).get("label", screener_key)
        with st.expander(f"{screener_label} — {'✅ Lolos' if sdata['passed'] else '❌ Tidak lolos'}", expanded=False):
            if not sdata["evidence"]:
                st.caption("Tidak ada data.")
                continue
            ev_df = pd.DataFrame(sdata["evidence"])[["label", "passed", "value"]]
            ev_df["passed"] = ev_df["passed"].map({True: "✅ Lolos", False: "❌ Tidak"})
            ev_df = ev_df.rename(columns={"label": "Kriteria", "passed": "Status", "value": "Nilai Aktual"})
            st.dataframe(ev_df, width='stretch', hide_index=True)
        for item in sdata["evidence"]:
            status_icon = "✅" if item["passed"] else "❌"
            display = f"{status_icon} [{screener_label}] {item['label']}"
            single_criteria_options.append(display)
            single_criteria_lookup[display] = item

    st.markdown("#### 🔍 Bukti Visual per Kriteria")
    pilihan_kriteria_single = st.selectbox(
        "Klik untuk kotakkan kriteria ini di chart:", single_criteria_options, key="single_kriteria_select",
    )
    highlight_item_single = single_criteria_lookup.get(pilihan_kriteria_single)
    if highlight_item_single:
        desc = highlight_item_single.get("description") or "Belum ada deskripsi buat kriteria ini."
        status_text = "✅ **LOLOS**" if highlight_item_single["passed"] else "❌ **TIDAK LOLOS**"
        st.info(f"**{highlight_item_single['label']}** — {status_text}\n\n"
                f"📝 {desc}\n\n📊 Nilai aktual: `{highlight_item_single['value']}`")

    st.plotly_chart(
        build_evidence_chart(df_single, ticker, trade_levels=trade_levels_single, highlight_item=highlight_item_single),
        width='stretch',
    )
    if highlight_item_single and highlight_item_single.get("key") not in KEY_BOX_MAP:
        st.caption("ℹ️ Kriteria ini belum punya area kotak spesifik di chart - tampilan di atas grafik biasa.")

    st.markdown("#### 🎯 Level Trading (perkiraan, bukan jaminan)")
    lc1, lc2, lc3, lc4 = st.columns(4)
    lc1.metric("Support", f"{trade_levels_single['support']:,.0f}")
    lc2.metric("Stop Loss", f"{trade_levels_single['stop_loss']:,.0f}",
               delta=f"{(trade_levels_single['stop_loss']/trade_levels_single['price']-1)*100:.1f}%",
               delta_color="inverse")
    lc3.metric("Take Profit 1", f"{trade_levels_single['take_profit_1']:,.0f}",
               delta=f"{(trade_levels_single['take_profit_1']/trade_levels_single['price']-1)*100:.1f}%")
    lc4.metric("Take Profit 2", f"{trade_levels_single['take_profit_2']:,.0f}",
               delta=f"{(trade_levels_single['take_profit_2']/trade_levels_single['price']-1)*100:.1f}%")


# --- Riwayat Screening Hari Ini (di luar blok if, biar selalu kelihatan) ---
st.markdown("#### 📜 Riwayat Screening Satu Saham Hari Ini")
_hist_rows_today = _read_history_rows(mode="single", date_filter=_today_str())
if _hist_rows_today:
    hist_display = [{
        "Waktu": r.get("Waktu"), "Ticker": r.get("Ticker"),
        "Harga": f"{float(r.get('Harga') or 0):,.0f}" if r.get("Harga") not in ("", None) else "-",
        "RSI": f"{float(r.get('RSI') or 0):.1f}" if r.get("RSI") not in ("", None) else "-",
        "Screener Lolos": r.get("ScreenerLolos") or "-",
        "Broker Summary": r.get("BrokerSummaryStatus") or "-",
    } for r in reversed(_hist_rows_today)]
    st.dataframe(pd.DataFrame(hist_display), width='stretch', hide_index=True)
    _gs_active = _get_apps_script_config()[0] is not None
    st.caption(
        ("📗 Tersimpan di Google Sheets - riwayat ini PERMANEN, nggak kehapus tiap hari."
         if _gs_active else
         "📁 Google Sheets belum di-setting, riwayat ini sementara disimpan di file lokal server "
         "(bisa hilang kalau server restart). Lihat panduan setup Google Sheets biar permanen.")
        + " Broker Summary yang statusnya 'Diambil' nggak akan manggil API lagi kalau kamu cek "
          "ulang saham yang sama hari ini, kecuali centang 'Paksa ambil ulang'."
    )
else:
    st.caption("Belum ada saham yang di-screening hari ini.")


# =========================================================================
# CATATAN BUAT KAMU (baca ini)
# =========================================================================
#
# CARA JALANIN:
#   streamlit run web_app.py   (BUKAN "python web_app.py")
#
# CARA BERHENTIKAN:
#   Ctrl+C di Command Prompt yang lagi jalanin server
#
# SOAL WATCHLIST:
#   Selalu pakai daftar bawaan (LQ45 + konglomerat) kecuali centang
#   "Scan SEMUA saham IDX". Buat ubah daftar bawaan, edit LQ45_LIST
#   atau KONGLOMERAT_GROUPS di stock_screener.py.
