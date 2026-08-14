"""
IDX Stock Screener - Versi Web (Streamlit)
=============================================
Ini "kulit" web dari stock_screener.py. Logika screening-nya SAMA PERSIS,
cuma ditampilkan lewat browser, bukan lewat teks di Command Prompt.

Cara pakai:
1. Pastikan file ini ada di FOLDER YANG SAMA dengan stock_screener.py
2. pip install streamlit  (kalau belum)
3. Jalankan lewat Command Prompt: streamlit run web_app.py
   (BUKAN "python web_app.py" - Streamlit punya cara jalanin sendiri)
4. Browser bakal kebuka otomatis ke http://localhost:8501
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# Import semua fungsi yang udah dibuat di stock_screener.py
# (file ini HARUS ada di folder yang sama)
from stock_screener import (
    fetch_batch_data,
    compute_signals,
    compute_screener_results,
    compute_fundamental_score,
    fetch_all_idx_tickers,
    find_support_resistance,
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
}

# 3 screener utama. Saham ditampilkan kalau lolos SALAH SATU screener
# yang dicentang aktif.
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


def build_evidence_chart(df: pd.DataFrame, ticker: str):
    """
    Bikin grafik candlestick + MA5/MA20 + volume + garis support/resistance,
    sebagai BUKTI VISUAL kenapa saham ini lolos screener - biar nggak cuma
    percaya angka di tabel, tapi bisa lihat sendiri bentuk grafiknya.
    """
    chart_df = df.tail(90)  # 90 hari terakhir biar nggak kepadetan

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=(f"{ticker} — Harga & Moving Average", "Volume"),
    )

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

    # Garis support/resistance, biar konsisten sama indikator "Dekat Area Support"
    try:
        sr = find_support_resistance(df)
        for lvl in sr.get("support", []):
            fig.add_hline(y=lvl, line_dash="dot", line_color="green", opacity=0.6,
                          annotation_text="Support", annotation_position="right", row=1, col=1)
        for lvl in sr.get("resistance", []):
            fig.add_hline(y=lvl, line_dash="dot", line_color="red", opacity=0.6,
                          annotation_text="Resistance", annotation_position="right", row=1, col=1)
    except Exception:
        pass

    vol_colors = ["#2ecc71" if c >= o else "#e74c3c"
                  for c, o in zip(chart_df["Close"], chart_df["Open"])]
    fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["Volume"], name="Volume",
                          marker_color=vol_colors), row=2, col=1)

    fig.update_layout(
        height=520, xaxis_rangeslider_visible=False,
        showlegend=True, margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig

# =========================================================================
# PENGATURAN TAMPILAN HALAMAN
# =========================================================================

st.set_page_config(
    page_title="Screener Saham IDX",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Screener Saham IDX")
st.caption("Instrumen screening berbasis sinyal teknikal — bukan rekomendasi investasi")

st.warning(
    "⚠️ Ini alat bantu screening berbasis indikator teknikal historis, BUKAN prediksi "
    "harga yang pasti. Data delay 15-20 menit. Selalu riset tambahan sebelum ambil "
    "keputusan trading."
)

# =========================================================================
# SIDEBAR - PENGATURAN (di kiri layar)
# =========================================================================

st.sidebar.header("⚙️ Pengaturan")

scan_all = st.sidebar.checkbox(
    "🌐 Scan SEMUA saham IDX (~900+)",
    value=False,
    help="Kalau dicentang, watchlist di bawah diabaikan dan semua saham IDX yang "
         "tercatat bakal di-scan. Prosesnya jauh lebih lama (bisa beberapa menit).",
)

if scan_all:
    st.sidebar.info("Mode: scan semua saham IDX. Watchlist manual di bawah di-nonaktifkan.")
    watchlist_text = ""
else:
    watchlist_text = st.sidebar.text_area(
        "Watchlist saham (pisahkan pakai koma)",
        value=", ".join(DEFAULT_WATCHLIST),
        height=120,
        help="Format: KODESAHAM.JK — misal BBCA.JK, TLKM.JK",
    )

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Pilih Screener")
st.sidebar.caption("Saham ditampilkan kalau lolos SALAH SATU screener yang dicentang.")

active_screeners = []
for key, info in SCREENER_INFO.items():
    checked = st.sidebar.checkbox(info["label"], value=True, help=info["help"], key=f"scr_{key}")
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
if col_b.button("Kosongkan", width='stretch'):
    st.session_state.selected_indicators = []

selected_indicators = []
for key, label in AVAILABLE_INDICATORS.items():
    checked = st.sidebar.checkbox(
        label,
        value=(key in st.session_state.selected_indicators),
        key=f"chk_{key}",
    )
    if checked:
        selected_indicators.append(key)
st.session_state.selected_indicators = selected_indicators

include_news = st.sidebar.checkbox(
    "📰 Sertakan Sentimen Berita (penguat fundamental)",
    value=True,
    help="Cek berita terkini (maks 31 hari) soal emiten. Pakai analisis AI kalau API "
         "key sudah diisi di stock_screener.py, kalau belum fallback ke keyword matching. "
         "Link berita selalu disertakan buat verifikasi manual.",
)

run_button = st.sidebar.button("🔍 Jalankan Screening", type="primary", width='stretch')

st.sidebar.markdown("---")
if not scan_all:
    watchlist_preview = [t.strip().upper() for t in watchlist_text.split(",") if t.strip()]
    st.sidebar.caption(f"Jumlah saham di watchlist: {len(watchlist_preview)}")

# =========================================================================
# AREA UTAMA - HASIL SCREENING
# =========================================================================

if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.last_run = None
    st.session_state.evidence_map = {}
    st.session_state.chart_data = {}

if run_button:
    if not active_screeners:
        st.sidebar.error("Pilih minimal 1 screener dulu.")
        st.stop()

    if scan_all:
        with st.spinner("Mengambil daftar semua saham IDX..."):
            watchlist = fetch_all_idx_tickers()
        if not watchlist:
            st.error("Gagal mengambil daftar saham IDX. Cek koneksi internet, atau coba lagi nanti.")
            st.stop()
    else:
        watchlist = [t.strip().upper() for t in watchlist_text.split(",") if t.strip()]

    results = []
    evidence_map = {}  # {ticker: {"day_trade": [...], "bsjp": [...], "bpjs": [...]}}
    chart_data = {}  # {ticker: dataframe} buat render grafik nanti
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

            lolos_tags = []
            for s in active_screeners:
                if screeners.get(s, {}).get("passed"):
                    lolos_tags.append(SCREENER_INFO[s]["label"].split(" ", 1)[1])  # buang emoji

            evidence_map[ticker] = {s: screeners[s]["evidence"] for s in active_screeners if s in screeners}
            chart_data[ticker] = df

            results.append({
                "Ticker": ticker,
                "Harga": tech["price"],
                "Skor": total_score,
                "RSI": round(tech["rsi"], 1),
                "Lolos Screener": ", ".join(lolos_tags) if lolos_tags else "-",
                "Alasan": ", ".join(all_reasons) if all_reasons else "-",
                "Link Berita": news_url or "",
            })

    progress_bar.empty()
    st.session_state.results = pd.DataFrame(results)
    st.session_state.evidence_map = evidence_map
    st.session_state.chart_data = chart_data
    st.session_state.last_run = datetime.now().strftime("%d %b %Y, %H:%M:%S")

# Tampilkan hasil kalau sudah pernah dijalankan
if st.session_state.results is not None:
    st.caption(f"Terakhir dijalankan: {st.session_state.last_run}")

    df_results = st.session_state.results
    evidence_map = st.session_state.evidence_map

    col1, col2, col3 = st.columns(3)
    col1.metric("Saham lolos screener", len(df_results))
    col2.metric("Skor tertinggi", int(df_results["Skor"].max()) if len(df_results) else 0)
    col3.metric("Skor rata-rata", round(df_results["Skor"].mean(), 1) if len(df_results) else 0)

    st.subheader("Kandidat Hasil Screening")
    if len(df_results) > 0:
        st.dataframe(
            df_results.sort_values("Skor", ascending=False),
            width='stretch',
            hide_index=True,
            column_config={
                "Harga": st.column_config.NumberColumn(format="%d"),
                "Link Berita": st.column_config.LinkColumn(
                    "Cek Berita Manual", display_text="🔗 Buka"
                ),
            },
        )

        st.markdown("---")
        st.subheader("🔍 Bukti Validitas per Saham")
        st.caption(
            "Pilih saham buat lihat angka detail tiap kriteria screener — "
            "biar kamu bisa verifikasi sendiri, bukan cuma percaya label 'lolos'."
        )

        pilihan_ticker = st.selectbox("Pilih saham:", df_results["Ticker"].tolist())

        if pilihan_ticker and pilihan_ticker in st.session_state.chart_data:
            st.plotly_chart(
                build_evidence_chart(st.session_state.chart_data[pilihan_ticker], pilihan_ticker),
                width='stretch',
            )

        if pilihan_ticker and pilihan_ticker in evidence_map:
            for screener_key, evidence_list in evidence_map[pilihan_ticker].items():
                screener_label = SCREENER_INFO[screener_key]["label"]
                with st.expander(f"{screener_label} — bukti kriteria", expanded=False):
                    if not evidence_list:
                        st.caption("Tidak ada data (kemungkinan data historis kurang).")
                        continue
                    ev_df = pd.DataFrame(evidence_list)
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
# CATATAN BUAT KAMU (baca ini)
# =========================================================================
#
# CARA JALANIN:
#   Di Command Prompt, pindah ke folder file ini, lalu ketik:
#   streamlit run web_app.py
#
#   BUKAN "python web_app.py" — kalau kamu jalanin dengan cara itu,
#   nanti error atau nggak ada apa-apa yang muncul. Streamlit punya
#   command sendiri buat "menyalakan" server webnya.
#
# CARA BERHENTIKAN:
#   Balik ke Command Prompt yang lagi jalanin server, tekan Ctrl+C
#
# KALAU MAU DIAKSES DARI HP JUGA (masih di jaringan WiFi yang sama):
#   Command Prompt bakal nampilin 2 alamat pas dijalankan:
#   - Local URL (cuma bisa dibuka di laptop ini)
#   - Network URL (bisa dibuka dari HP, asal masih 1 WiFi)
#   Buka Network URL itu dari browser HP kamu.
