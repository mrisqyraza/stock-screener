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
from datetime import datetime

# Import semua fungsi yang udah dibuat di stock_screener.py
# (file ini HARUS ada di folder yang sama)
from stock_screener import (
    fetch_batch_data,
    compute_signals,
    compute_all_indicators,
    compute_addon_screens,
    compute_fundamental_score,
    compute_sentiment_score,
    fetch_all_idx_tickers,
    WATCHLIST as DEFAULT_WATCHLIST,
)

# Daftar indikator SKOR yang bisa dipilih/dicentang user. Key-nya harus
# sama persis dengan key di compute_all_indicators() -> "indicators" dict.
# BSJP & BPJS SENGAJA TIDAK di sini - itu add-on screening terpisah,
# bukan poin skor (lihat ADDON_SCREENS di bawah).
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

# Peringatan disclaimer, selalu ditampilkan di atas
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

min_score = st.sidebar.slider(
    "Skor minimum ditampilkan",
    min_value=0, max_value=len(AVAILABLE_INDICATORS) + 1, value=3,
    help="Makin tinggi, makin ketat filternya (cuma saham dengan banyak sinyal yang muncul)",
)

st.sidebar.markdown("---")
st.sidebar.subheader("✅ Pilih Indikator")
st.sidebar.caption("Cuma indikator yang dicentang yang dihitung ke skor.")

if "selected_indicators" not in st.session_state:
    # default: semua kecentang
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
    help="Cek judul berita terkini soal emiten, dihitung sebagai sinyal tambahan "
         "(bukan analisis fundamental mendalam - cuma keyword matching sederhana). "
         "Link berita selalu disertakan biar bisa kamu cek manual sendiri.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("➕ Add-on Screening")
st.sidebar.caption(
    "Filter TERPISAH dari skor indikator di atas - saham WAJIB lolos kriteria "
    "ini kalau dicentang (bukan nambah poin skor, tapi nyaring lolos/tidak)."
)

require_bsjp = st.sidebar.checkbox(
    "Wajib lolos BSJP (Beli Sore Jual Pagi)",
    value=False,
    help="Naik ≥5%, volume breakout ≥2x MA20, harga di atas MA5 & Open, "
         "value transaksi > Rp5 miliar.",
)
require_bpjs = st.sidebar.checkbox(
    "Wajib lolos BPJS (Beli Pagi Jual Sore)",
    value=False,
    help="Versi lebih longgar dari BSJP, biasa dicek 30 menit sebelum market buka.",
)

run_button = st.sidebar.button("🔍 Jalankan Screening", type="primary", width='stretch')

st.sidebar.markdown("---")
if not scan_all:
    watchlist_preview = [t.strip().upper() for t in watchlist_text.split(",") if t.strip()]
    st.sidebar.caption(f"Jumlah saham di watchlist: {len(watchlist_preview)}")

# =========================================================================
# AREA UTAMA - HASIL SCREENING
# =========================================================================

# session_state dipakai biar hasil screening nggak hilang tiap kali
# ada interaksi lain di halaman (misal geser slider)
if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.last_run = None

if run_button:
    if scan_all:
        with st.spinner("Mengambil daftar semua saham IDX..."):
            watchlist = fetch_all_idx_tickers()
        if not watchlist:
            st.error("Gagal mengambil daftar saham IDX. Cek koneksi internet, atau coba lagi nanti.")
            st.stop()
        st.caption(f"Ditemukan {len(watchlist)} saham tercatat di IDX.")
    else:
        watchlist = [t.strip().upper() for t in watchlist_text.split(",") if t.strip()]

    results = []
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
            addons = compute_addon_screens(df)

            # Filter add-on: kalau dicentang, saham WAJIB lolos, kalau nggak di-skip
            if require_bsjp and not addons["bsjp"]["passed"]:
                continue
            if require_bpjs and not addons["bpjs"]["passed"]:
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

            addon_tags = []
            if addons["bsjp"]["passed"]:
                addon_tags.append("BSJP")
            if addons["bpjs"]["passed"]:
                addon_tags.append("BPJS")

            results.append({
                "Ticker": ticker,
                "Harga": tech["price"],
                "Skor": total_score,
                "RSI": round(tech["rsi"], 1),
                "Add-on": ", ".join(addon_tags) if addon_tags else "-",
                "Alasan": ", ".join(all_reasons) if all_reasons else "-",
                "Link Berita": news_url or "",
            })

    progress_bar.empty()
    st.session_state.results = pd.DataFrame(results)
    st.session_state.last_run = datetime.now().strftime("%d %b %Y, %H:%M:%S")

# Tampilkan hasil kalau sudah pernah dijalankan
if st.session_state.results is not None:
    st.caption(f"Terakhir dijalankan: {st.session_state.last_run}")

    df_results = st.session_state.results
    df_filtered = df_results[df_results["Skor"] >= min_score].sort_values("Skor", ascending=False)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total saham dicek", len(df_results))
    col2.metric("Lolos filter skor", len(df_filtered))
    col3.metric("Skor rata-rata", round(df_results["Skor"].mean(), 1) if len(df_results) else 0)

    max_possible_score = len(AVAILABLE_INDICATORS) + (1 if include_news else 0)
    st.subheader(f"Kandidat dengan skor ≥ {min_score}")
    if len(df_filtered) > 0:
        st.dataframe(
            df_filtered,
            width='stretch',
            hide_index=True,
            column_config={
                "Harga": st.column_config.NumberColumn(format="%d"),
                "Skor": st.column_config.ProgressColumn(min_value=0, max_value=max(max_possible_score, 1)),
                "Link Berita": st.column_config.LinkColumn(
                    "Cek Berita Manual", display_text="🔗 Buka"
                ),
            },
        )
        st.caption(
            "💡 Kolom 'Link Berita' membuka pencarian Google News buat emiten itu — "
            "dipakai buat verifikasi manual, karena deteksi sentimen di sini cuma "
            "keyword matching sederhana, bukan pembacaan konteks yang akurat."
        )
    else:
        st.info("Nggak ada saham yang lolos filter skor ini. Coba turunin 'Skor minimum' di sidebar kiri.")

    with st.expander("Lihat semua hasil (termasuk yang skornya rendah)"):
        st.dataframe(df_results.sort_values("Skor", ascending=False), width='stretch', hide_index=True)
else:
    st.info("👈 Klik tombol **'Jalankan Screening'** di sidebar kiri untuk mulai.")


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
