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
    compute_trade_levels,
    find_support_resistance,
    INDICATOR_DESCRIPTIONS,
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


def build_evidence_chart(df: pd.DataFrame, ticker: str, trade_levels: dict = None, highlight_label: str = None):
    """
    Bikin grafik candlestick + MA5/MA20 + volume + garis support/resistance
    + level TP/SL, sebagai BUKTI VISUAL. Kalau highlight_label diisi, kasih
    penanda panah di candle terakhir nunjukkin kriteria mana yang lagi dipilih.
    """
    chart_df = df.tail(90)  # 90 hari terakhir biar nggak kepadetan

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=(f"{ticker} — Harga, MA & Level Trading", "Volume"),
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

    # Garis support/resistance historis (abu-abu tipis, latar belakang saja)
    try:
        sr = find_support_resistance(df)
        for lvl in sr.get("support", []):
            fig.add_hline(y=lvl, line_dash="dot", line_color="gray", opacity=0.4, row=1, col=1)
        for lvl in sr.get("resistance", []):
            fig.add_hline(y=lvl, line_dash="dot", line_color="gray", opacity=0.4, row=1, col=1)
    except Exception:
        pass

    # Garis level trading (support/SL/TP1/TP2) - lebih tegas warnanya
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

    # Penanda kalau ada indikator yang lagi dipilih - panah nunjuk ke candle terakhir
    if highlight_label:
        last_date = chart_df.index[-1]
        last_high = chart_df["High"].iloc[-1]
        fig.add_annotation(
            x=last_date, y=last_high, text=f"📍 {highlight_label}",
            showarrow=True, arrowhead=2, arrowcolor="black", arrowwidth=2,
            ax=0, ay=-60, bgcolor="#FFF3CD", bordercolor="black", borderwidth=1,
            font=dict(size=12), row=1, col=1,
        )

    vol_colors = ["#2ecc71" if c >= o else "#e74c3c"
                  for c, o in zip(chart_df["Close"], chart_df["Open"])]
    fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["Volume"], name="Volume",
                          marker_color=vol_colors), row=2, col=1)

    fig.update_layout(
        height=560, xaxis_rangeslider_visible=False,
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

# =========================================================================
# AREA UTAMA - HASIL SCREENING
# =========================================================================

if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.last_run = None
    st.session_state.evidence_map = {}
    st.session_state.chart_data = {}
    st.session_state.trade_levels_map = {}

if run_button:
    if not active_screeners:
        st.sidebar.error("Pilih minimal 1 screener dulu.")
        st.stop()

    if scan_all:
        with st.spinner("Mengambil daftar semua saham IDX..."):
            from stock_screener import fetch_all_idx_tickers as _fetch_all
            watchlist = _fetch_all()
        if not watchlist:
            st.error("Gagal mengambil daftar saham IDX. Cek koneksi internet, atau coba lagi nanti.")
            st.stop()
    else:
        watchlist = DEFAULT_WATCHLIST

    results = []
    evidence_map = {}       # {ticker: {"day_trade": [...], "bsjp": [...], "bpjs": [...]}}
    chart_data = {}         # {ticker: dataframe} buat render grafik nanti
    trade_levels_map = {}   # {ticker: {"support":.., "stop_loss":.., "take_profit_1":.., ...}}
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
                "Alasan": ", ".join(all_reasons) if all_reasons else "-",
                "Link Berita": news_url or "",
            })

    progress_bar.empty()
    st.session_state.results = pd.DataFrame(results)
    st.session_state.evidence_map = evidence_map
    st.session_state.chart_data = chart_data
    st.session_state.trade_levels_map = trade_levels_map
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
            "Pilih saham, lalu klik salah satu kriteria indikator di bawah — grafiknya "
            "otomatis kasih penanda dan deskripsi kriteria itu muncul di bawahnya."
        )

        pilihan_ticker = st.selectbox("Pilih saham:", df_results["Ticker"].tolist())

        # Kumpulkan semua kriteria dari screener yang lolos buat saham ini,
        # jadi satu daftar yang bisa dipilih (biar bisa "klik indikator -> ke grafik")
        criteria_options = ["(Nggak ada yang dipilih - tampilan grafik biasa)"]
        criteria_lookup = {}  # {display_text: {"label":.., "description":.., "screener":..}}
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
        highlight_label = None
        if pilihan_kriteria in criteria_lookup:
            highlight_label = criteria_lookup[pilihan_kriteria]["label"]

        if pilihan_ticker and pilihan_ticker in st.session_state.chart_data:
            st.plotly_chart(
                build_evidence_chart(
                    st.session_state.chart_data[pilihan_ticker], pilihan_ticker,
                    trade_levels=trade_levels, highlight_label=highlight_label,
                ),
                width='stretch',
            )

        # Deskripsi kriteria yang dipilih
        if pilihan_kriteria in criteria_lookup:
            item = criteria_lookup[pilihan_kriteria]
            desc = item.get("description") or "Belum ada deskripsi buat kriteria ini."
            status_text = "✅ **LOLOS**" if item["passed"] else "❌ **TIDAK LOLOS**"
            st.info(f"**{item['label']}** — {status_text}\n\n"
                    f"📝 {desc}\n\n"
                    f"📊 Nilai aktual: `{item['value']}`")

        # Level trading: Support, Stop Loss, TP1, TP2
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

        # Tabel bukti lengkap (semua kriteria, semua screener) tetap ditampilkan di bawah
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
#
# SOAL WATCHLIST:
#   Watchlist manual di web app udah DIHAPUS - sekarang selalu pakai
#   daftar bawaan (LQ45 + saham konglomerat) dari stock_screener.py,
#   kecuali kamu centang "Scan SEMUA saham IDX". Kalau mau ubah daftar
#   bawaannya, edit LQ45_LIST atau KONGLOMERAT_GROUPS di stock_screener.py.
