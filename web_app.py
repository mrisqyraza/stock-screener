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
    compute_fundamental_score,
    compute_sentiment_score,
    fetch_all_idx_tickers,
    WATCHLIST as DEFAULT_WATCHLIST,
)

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
    min_value=0, max_value=8, value=3,
    help="Makin tinggi, makin ketat filternya (cuma saham dengan banyak sinyal yang muncul)",
)

run_button = st.sidebar.button("🔍 Jalankan Screening", type="primary", use_container_width=True)

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
            tech = compute_signals(df)
            fund = compute_fundamental_score(ticker)
            sent = compute_sentiment_score(ticker)

            total_score = tech["score"] + fund["score"] + sent["score"]
            all_reasons = tech["reasons"] + fund["reasons"] + sent["reasons"]

            results.append({
                "Ticker": ticker,
                "Harga": tech["price"],
                "Skor": total_score,
                "RSI": round(tech["rsi"], 1),
                "Alasan": ", ".join(all_reasons) if all_reasons else "-",
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

    st.subheader(f"Kandidat dengan skor ≥ {min_score}")
    if len(df_filtered) > 0:
        st.dataframe(
            df_filtered,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Harga": st.column_config.NumberColumn(format="%d"),
                "Skor": st.column_config.ProgressColumn(min_value=0, max_value=8),
            },
        )
    else:
        st.info("Nggak ada saham yang lolos filter skor ini. Coba turunin 'Skor minimum' di sidebar kiri.")

    with st.expander("Lihat semua hasil (termasuk yang skornya rendah)"):
        st.dataframe(df_results.sort_values("Skor", ascending=False), use_container_width=True, hide_index=True)
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
