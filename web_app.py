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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

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
    analyze_single_stock,
    check_leading_lagging,
    fetch_all_idx_tickers,
    INDICATOR_DESCRIPTIONS,
    TRADE_STYLE_PARAMS,
    KONGLOMERAT_GROUPS,
    WATCHLIST as DEFAULT_WATCHLIST,
)

# Indikator SKOR tambahan (di luar screener utama) yang bisa dipilih user.
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

# 7 gaya screener, disesuaikan sama materi video (day trade, scalping, swing,
# ARA hunter, position trading) + BSJP/BPJS ala Stockbit.
SCREENER_INFO = {
    "scalping": {
        "label": "⚡ Scalping",
        "help": "Volume ≥1.2x rata-rata, RSI 45-65, harga di atas MA5, spread High-Low >1.5%. "
                "Horizon: menit-jam.",
        "style_key": "scalping",
    },
    "day_trade": {
        "label": "📊 Day Trade",
        "help": "Volume ≥1.5x rata-rata, MACD histogram positif, harga di atas MA20, "
                "RSI 40-70, value transaksi > Rp1 miliar. Horizon: 1 hari.",
        "style_key": "day_trade",
    },
    "bsjp": {
        "label": "🌆 BSJP (Beli Sore Jual Pagi)",
        "help": "Naik ≥5%, volume breakout ≥2x MA20, harga di atas MA5 & Open, "
                "value transaksi > Rp5 miliar, bukan saham gocap.",
        "style_key": "bsjp",
    },
    "bpjs": {
        "label": "🌅 BPJS (Beli Pagi Jual Sore)",
        "help": "Versi lebih longgar dari BSJP, biasa dicek 30 menit sebelum market buka.",
        "style_key": "bpjs",
    },
    "swing_trading": {
        "label": "📈 Swing Trading",
        "help": "MA20 > MA50, harga di atas keduanya, RSI 45-65, MACD histogram positif, "
                "volume wajar. Horizon: beberapa hari-minggu.",
        "style_key": "swing_trading",
    },
    "ara_hunter": {
        "label": "🚀 ARA Hunter ⚠️",
        "help": "Kenaikan mendekati batas ARA, volume ≥3x rata-rata. PALING BERISIKO - "
                "rawan reversal tajam & susah dijual saat exit.",
        "style_key": "ara_hunter",
    },
    "position_trading": {
        "label": "🏔️ Position Trading",
        "help": "MA50 > MA100, harga dekat puncak 200 hari, RSI belum overbought ekstrem. "
                "Horizon: bulanan.",
        "style_key": "position_trading",
    },
}

# Kategori tiap key kriteria -> baris chart mana yang di-highlight pas dipilih
def evidence_chart_row(key: str) -> int:
    k = (key or "").lower()
    if "macd" in k:
        return 2
    if "rsi" in k or "stoch" in k:
        return 3
    if "volume" in k or "value" in k:
        return 4
    return 1  # default: harga/MA/support/resistance/open


def build_evidence_chart(df: pd.DataFrame, ticker: str, trade_levels: dict = None,
                          highlight_label: str = None, highlight_key: str = None):
    """
    Grafik 4-panel: Harga+MA (1), MACD (2), RSI (3), Volume (4).
    Kalau ada kriteria yang dipilih, gambar KOTAK (vrect) di panel yang
    relevan mengelilingi candle/bar terakhir - ini "bukti visual" beneran,
    bukan cuma panah nunjuk ke harga penutupan.
    """
    chart_df = df.tail(90)

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=[0.45, 0.18, 0.18, 0.19],
        subplot_titles=(f"{ticker} — Harga, MA & Level Trading", "MACD", "RSI", "Volume"),
    )

    # --- Row 1: Candlestick + MA + S/R + trade levels ---
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
                      opacity=0.8, annotation_text=f"SL {trade_levels['stop_loss']:.0f}",
                      annotation_position="left", row=1, col=1)
        fig.add_hline(y=trade_levels["take_profit_1"], line_dash="dash", line_color="#3498db",
                      opacity=0.8, annotation_text=f"TP1 {trade_levels['take_profit_1']:.0f}",
                      annotation_position="left", row=1, col=1)
        fig.add_hline(y=trade_levels["take_profit_2"], line_dash="dash", line_color="#9b59b6",
                      opacity=0.8, annotation_text=f"TP2 {trade_levels['take_profit_2']:.0f}",
                      annotation_position="left", row=1, col=1)

    # --- Row 2: MACD ---
    macd_obj = ta.trend.MACD(df["Close"])
    macd_line = macd_obj.macd().tail(90)
    macd_signal = macd_obj.macd_signal().tail(90)
    macd_hist = macd_obj.macd_diff().tail(90)
    fig.add_trace(go.Scatter(x=chart_df.index, y=macd_line, name="MACD",
                              line=dict(color="#2980b9", width=1.3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=chart_df.index, y=macd_signal, name="Signal",
                              line=dict(color="#e67e22", width=1.3)), row=2, col=1)
    hist_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in macd_hist]
    fig.add_trace(go.Bar(x=chart_df.index, y=macd_hist, name="Histogram",
                          marker_color=hist_colors, opacity=0.5), row=2, col=1)

    # --- Row 3: RSI ---
    rsi_series = ta.momentum.RSIIndicator(df["Close"], window=14).rsi().tail(90)
    fig.add_trace(go.Scatter(x=chart_df.index, y=rsi_series, name="RSI",
                              line=dict(color="#8e44ad", width=1.3)), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", opacity=0.4, row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", opacity=0.4, row=3, col=1)

    # --- Row 4: Volume ---
    vol_colors = ["#2ecc71" if c >= o else "#e74c3c"
                  for c, o in zip(chart_df["Close"], chart_df["Open"])]
    fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["Volume"], name="Volume",
                          marker_color=vol_colors), row=4, col=1)
    vol_ma20 = df["Volume"].rolling(20).mean().tail(90)
    fig.add_trace(go.Scatter(x=chart_df.index, y=vol_ma20, name="Vol MA20",
                              line=dict(color="black", width=1, dash="dot")), row=4, col=1)

    # --- KOTAK BUKTI: highlight beberapa bar terakhir di panel yang relevan ---
    if highlight_key and len(chart_df) >= 3:
        target_row = evidence_chart_row(highlight_key)
        box_start = chart_df.index[-4]
        box_end = chart_df.index[-1]
        fig.add_vrect(
            x0=box_start, x1=box_end, row=target_row, col=1,
            fillcolor="yellow", opacity=0.25, line_width=2, line_color="#B8860B",
        )
        # kasih label kecil di atas kotak
        fig.add_annotation(
            x=chart_df.index[-2], y=1, yref=f"y{target_row} domain" if target_row > 1 else "y domain",
            text=f"📦 {highlight_label}", showarrow=False, bgcolor="#FFF3CD",
            bordercolor="black", borderwidth=1, font=dict(size=11),
            row=target_row, col=1,
        )

    fig.update_layout(
        height=780, xaxis_rangeslider_visible=False,
        showlegend=True, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def render_evidence_and_levels(ticker: str, df: pd.DataFrame, evidence_by_screener: dict,
                                default_style: str = "day_trade", key_prefix: str = ""):
    """
    Bagian UI yang dipakai bersama oleh mode Screening Massal & Screening
    Satu Saham: dropdown pilih kriteria -> grafik dengan kotak bukti +
    deskripsi, plus level trading yang bisa diganti gaya-nya.
    """
    criteria_options = ["(Nggak ada yang dipilih - tampilan grafik biasa)"]
    criteria_lookup = {}
    for screener_key, evidence_list in evidence_by_screener.items():
        screener_label = SCREENER_INFO.get(screener_key, {}).get("label", screener_key)
        for item in evidence_list:
            status_icon = "✅" if item["passed"] else "❌"
            display = f"{status_icon} [{screener_label}] {item['label']}"
            criteria_options.append(display)
            criteria_lookup[display] = item

    pilihan_kriteria = st.selectbox(
        "🔎 Klik untuk pilih indikator/kriteria (grafik & deskripsi otomatis mengikuti):",
        criteria_options, key=f"{key_prefix}_kriteria",
    )

    style_options = list(SCREENER_INFO.keys())
    style_labels = [SCREENER_INFO[s]["label"] for s in style_options]
    default_idx = style_options.index(default_style) if default_style in style_options else 0
    pilihan_style_label = st.selectbox(
        "🎯 Level trading disesuaikan gaya:", style_labels, index=default_idx,
        key=f"{key_prefix}_style",
    )
    pilihan_style = style_options[style_labels.index(pilihan_style_label)]

    trade_levels = compute_trade_levels(df, style=pilihan_style)

    highlight_label, highlight_key = None, None
    if pilihan_kriteria in criteria_lookup:
        item = criteria_lookup[pilihan_kriteria]
        highlight_label = item["label"]
        highlight_key = item.get("key")

    st.plotly_chart(
        build_evidence_chart(df, ticker, trade_levels=trade_levels,
                              highlight_label=highlight_label, highlight_key=highlight_key),
        width='stretch',
    )

    if pilihan_kriteria in criteria_lookup:
        item = criteria_lookup[pilihan_kriteria]
        desc = item.get("description") or "Belum ada deskripsi buat kriteria ini."
        status_text = "✅ **LOLOS**" if item["passed"] else "❌ **TIDAK LOLOS**"
        st.info(f"**{item['label']}** — {status_text}\n\n"
                f"📝 {desc}\n\n"
                f"📊 Nilai aktual: `{item['value']}`\n\n"
                f"📦 Kotak kuning di grafik nunjukkin candle/bar yang jadi bukti kriteria ini.")

    st.markdown(f"#### 🎯 Level Trading — gaya {SCREENER_INFO[pilihan_style]['label']}")
    lc1, lc2, lc3, lc4 = st.columns(4)
    lc1.metric("Support", f"{trade_levels['support']:,.0f}")
    lc2.metric("Stop Loss", f"{trade_levels['stop_loss']:,.0f}", delta=f"{trade_levels['sl_pct']:.1f}%",
               delta_color="inverse")
    lc3.metric("Take Profit 1", f"{trade_levels['take_profit_1']:,.0f}", delta=f"{trade_levels['tp1_pct']:.1f}%")
    lc4.metric("Take Profit 2", f"{trade_levels['take_profit_2']:,.0f}", delta=f"{trade_levels['tp2_pct']:.1f}%")
    if trade_levels.get("risk_reward_1"):
        st.caption(
            f"Risk:Reward ke TP1 ≈ 1:{trade_levels['risk_reward_1']}, ke TP2 ≈ 1:{trade_levels['risk_reward_2']}. "
            f"Jarak SL/TP udah disesuaikan gaya trading yang dipilih (scalping = rapat, "
            f"position trading = lebar) - tetap cek ulang manual sebelum entry."
        )

    st.markdown("---")
    for screener_key, evidence_list in evidence_by_screener.items():
        screener_label = SCREENER_INFO.get(screener_key, {}).get("label", screener_key)
        with st.expander(f"{screener_label} — semua kriteria", expanded=False):
            if not evidence_list:
                st.caption("Tidak ada data (kemungkinan data historis kurang).")
                continue
            ev_df = pd.DataFrame(evidence_list)[["label", "passed", "value"]]
            ev_df["passed"] = ev_df["passed"].map({True: "✅ Lolos", False: "❌ Tidak"})
            ev_df = ev_df.rename(columns={"label": "Kriteria", "passed": "Status", "value": "Nilai Aktual"})
            st.dataframe(ev_df, width='stretch', hide_index=True)


# =========================================================================
# PENGATURAN TAMPILAN HALAMAN
# =========================================================================

st.set_page_config(page_title="Screener Saham IDX", page_icon="📈", layout="wide")

st.title("📈 Screener Saham IDX")
st.caption("Instrumen screening berbasis sinyal teknikal — bukan rekomendasi investasi")

st.warning(
    "⚠️ Ini alat bantu screening berbasis indikator teknikal historis, BUKAN prediksi "
    "harga yang pasti. Data delay 15-20 menit. Level TP/SL dihitung dari support/resistance "
    "historis, bukan jaminan harga akan bergerak ke situ. Screener **ARA Hunter** khususnya "
    "berisiko tinggi. Selalu riset tambahan sebelum ambil keputusan trading."
)

tab_massal, tab_satu = st.tabs(["🔍 Screening Massal", "🔎 Screening Satu Saham"])

# =========================================================================
# TAB 1: SCREENING MASSAL (banyak saham sekaligus)
# =========================================================================

with tab_massal:
    st.sidebar.header("⚙️ Pengaturan — Screening Massal")

    scan_all = st.sidebar.checkbox(
        "🌐 Scan SEMUA saham IDX (~900+)", value=False,
        help="Kalau nggak dicentang, pakai daftar bawaan LQ45 + saham konglomerat.",
    )
    st.sidebar.caption(
        f"Mode: {'Semua saham IDX (~900+)' if scan_all else f'Watchlist bawaan ({len(DEFAULT_WATCHLIST)} saham)'}"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Pilih Screener")
    st.sidebar.caption("Saham ditampilkan kalau lolos SALAH SATU screener yang dicentang.")

    active_screeners = []
    for key, info in SCREENER_INFO.items():
        checked = st.sidebar.checkbox(info["label"], value=(key in ("day_trade", "bsjp", "bpjs")),
                                       help=info["help"], key=f"scr_{key}")
        if checked:
            active_screeners.append(key)

    st.sidebar.markdown("---")
    st.sidebar.subheader("✅ Indikator Skor Tambahan")
    if "selected_indicators" not in st.session_state:
        st.session_state.selected_indicators = list(AVAILABLE_INDICATORS.keys())
    col_a, col_b = st.sidebar.columns(2)
    if col_a.button("Pilih Semua", width='stretch'):
        st.session_state.selected_indicators = list(AVAILABLE_INDICATORS.keys())
    if col_b.button("Kosongkan", width='stretch'):
        st.session_state.selected_indicators = []
    selected_indicators = []
    for key, label in AVAILABLE_INDICATORS.items():
        checked = st.sidebar.checkbox(label, value=(key in st.session_state.selected_indicators), key=f"chk_{key}")
        if checked:
            selected_indicators.append(key)
    st.session_state.selected_indicators = selected_indicators

    include_news = st.sidebar.checkbox(
        "📰 Sertakan Sentimen Berita", value=True,
        help="Berita maks 31 hari. Pakai AI kalau API key sudah diisi di stock_screener.py.",
    )

    run_button = st.sidebar.button("🔍 Jalankan Screening", type="primary", width='stretch')

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
                st.error("Gagal mengambil daftar saham IDX. Cek koneksi internet.")
                st.stop()
        else:
            watchlist = DEFAULT_WATCHLIST

        results = []
        evidence_map = {}
        chart_data = {}
        progress_bar = st.progress(0, text="Memulai screening...")
        batch_size = 50
        total_batches = (len(watchlist) + batch_size - 1) // batch_size

        for b in range(total_batches):
            batch = watchlist[b * batch_size: (b + 1) * batch_size]
            progress_bar.progress((b + 1) / total_batches,
                                   text=f"Memproses batch {b + 1}/{total_batches}...")
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

                lolos_tags = [SCREENER_INFO[s]["label"] for s in active_screeners
                              if screeners.get(s, {}).get("passed")]

                evidence_map[ticker] = {s: screeners[s]["evidence"] for s in active_screeners if s in screeners}
                chart_data[ticker] = df

                results.append({
                    "Ticker": ticker, "Harga": tech["price"], "Skor": total_score,
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
                df_results.sort_values("Skor", ascending=False), width='stretch', hide_index=True,
                column_config={
                    "Harga": st.column_config.NumberColumn(format="%d"),
                    "Link Berita": st.column_config.LinkColumn("Cek Berita Manual", display_text="🔗 Buka"),
                },
            )

            st.markdown("---")
            st.subheader("🔍 Bukti Validitas per Saham")
            pilihan_ticker = st.selectbox("Pilih saham:", df_results["Ticker"].tolist(), key="massal_ticker")

            if pilihan_ticker in st.session_state.chart_data:
                default_style = "day_trade"
                for s in active_screeners:
                    if evidence_map.get(pilihan_ticker, {}).get(s):
                        if any(e["passed"] for e in evidence_map[pilihan_ticker][s]):
                            default_style = SCREENER_INFO[s]["style_key"]
                            break
                render_evidence_and_levels(
                    pilihan_ticker, st.session_state.chart_data[pilihan_ticker],
                    evidence_map.get(pilihan_ticker, {}), default_style=default_style,
                    key_prefix="massal",
                )
        else:
            st.info("Nggak ada saham yang lolos screener yang dipilih.")
    else:
        st.info("👈 Pilih screener & klik tombol **'Jalankan Screening'** di sidebar kiri untuk mulai.")


# =========================================================================
# TAB 2: SCREENING SATU SAHAM
# =========================================================================

with tab_satu:
    st.subheader("🔎 Cek Satu Saham Secara Mendalam")
    st.caption("Masukkan kode saham buat lihat SEMUA screener, indikator, dan level trading sekaligus.")

    col_input, col_button = st.columns([3, 1])
    ticker_input = col_input.text_input(
        "Kode saham (contoh: BBCA, TLKM, GOTO):", value="", key="single_ticker_input",
        placeholder="Ketik kode saham tanpa .JK",
    ).strip()
    cek_button = col_button.button("🔎 Cek Saham Ini", type="primary", width='stretch')

    if "single_result" not in st.session_state:
        st.session_state.single_result = None

    if cek_button and ticker_input:
        with st.spinner(f"Menganalisis {ticker_input.upper()}..."):
            ticker_full = ticker_input.upper() if ticker_input.upper().endswith(".JK") else ticker_input.upper() + ".JK"

            # Cek dulu apakah masuk grup konglomerat, kalau iya ambil data
            # anggota grup lain juga buat cek leading-lagging
            group_batch = None
            for group_name, members in KONGLOMERAT_GROUPS.items():
                if ticker_full in members:
                    group_batch = fetch_batch_data(members)
                    break

            result = analyze_single_stock(ticker_full, batch_data_for_group=group_batch)
            st.session_state.single_result = result

    result = st.session_state.single_result

    if result is None:
        st.info("👆 Ketik kode saham dan klik tombol buat mulai analisis.")
    elif not result.get("found"):
        st.error(f"❌ {result.get('error', 'Saham tidak ditemukan')}")
    else:
        ticker = result["ticker"]
        df = result["df"]

        st.success(f"**{ticker}** — Harga sekarang: **{result['price']:,.0f}** | RSI: **{result['rsi']:.1f}**")

        # Ringkasan status semua screener
        st.markdown("#### Status di Semua Screener")
        screener_cols = st.columns(len(SCREENER_INFO))
        for i, (skey, sinfo) in enumerate(SCREENER_INFO.items()):
            passed = result["screeners"].get(skey, {}).get("passed", False)
            screener_cols[i].metric(sinfo["label"], "✅ Lolos" if passed else "❌ Tidak")

        # Leading-lagging kalau ada
        if result.get("leading_lagging"):
            ll = result["leading_lagging"]
            if ll["triggered"]:
                st.info(f"🔗 **Leading-Lagging**: {ll['value']}")
            else:
                st.caption(f"🔗 Leading-Lagging: {ll['value']}")

        # Berita
        if result.get("fundamental", {}).get("search_url"):
            fund = result["fundamental"]
            if fund["reasons"]:
                st.info(f"📰 **Berita**: {fund['reasons'][0]}")
            st.caption(f"[🔗 Cek semua berita terkait]({fund['search_url']})")

        st.markdown("---")

        # Evidence + chart + trade levels, pakai semua screener (lolos maupun tidak)
        # supaya user bisa lihat kenapa gagal juga
        evidence_by_screener = {skey: result["screeners"][skey]["evidence"]
                                 for skey in SCREENER_INFO.keys() if skey in result["screeners"]}

        default_style = "day_trade"
        for skey in SCREENER_INFO.keys():
            if result["screeners"].get(skey, {}).get("passed"):
                default_style = SCREENER_INFO[skey]["style_key"]
                break

        render_evidence_and_levels(ticker, df, evidence_by_screener,
                                    default_style=default_style, key_prefix="single")

        # Indikator skor tambahan juga ditampilkan di sini
        st.markdown("---")
        st.markdown("#### Indikator Skor Tambahan (di luar screener utama)")
        ind_rows = []
        for key, label in AVAILABLE_INDICATORS.items():
            ind = result["indicators"].get(key, {})
            ind_rows.append({
                "Indikator": label,
                "Status": "✅ Aktif" if ind.get("triggered") else "❌ Tidak aktif",
                "Detail": ind.get("detail") or "-",
            })
        st.dataframe(pd.DataFrame(ind_rows), width='stretch', hide_index=True)


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
# TAB "Screening Satu Saham":
#   Ketik kode saham tanpa .JK (misal "BBCA"), klik "Cek Saham Ini".
#   Kalau saham itu termasuk salah satu grup konglomerat (Prajogo, Bakrie,
#   dll - lihat KONGLOMERAT_GROUPS di stock_screener.py), otomatis dicek
#   juga status leading-lagging-nya dibanding saham lain segrup.
#
# SOAL WATCHLIST (mode Screening Massal):
#   Selalu pakai daftar bawaan (LQ45 + konglomerat) kecuali centang
#   "Scan SEMUA saham IDX". Buat ubah daftar bawaan, edit LQ45_LIST
#   atau KONGLOMERAT_GROUPS di stock_screener.py.
