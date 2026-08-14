"""
IDX Stock Screener - Starter Instrument
=========================================
Deteksi saham dengan sinyal teknikal kuat (RSI, MACD, volume spike, MA crossover)
lalu kirim notifikasi ke Telegram.

Cara pakai:
1. pip install yfinance pandas ta requests
2. Isi WATCHLIST dengan ticker IDX (pakai suffix .JK)
3. Isi TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID (lihat panduan di bawah)
4. Jalankan manual: python stock_screener.py
5. Jadwalkan otomatis via cron / GitHub Actions (lihat catatan di bagian bawah file)

PENTING - DISCLAIMER:
Ini alat screening berbasis sinyal teknikal historis, BUKAN prediksi harga
yang akurat. Data dari Yahoo Finance untuk saham IDX biasanya delay 15-20 menit.
Selalu lakukan riset tambahan sebelum ambil keputusan trading. Ini bukan
rekomendasi investasi.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import requests
import time
import io
from datetime import datetime

# =========================================================================
# KONFIGURASI - edit bagian ini
# =========================================================================

# Konstituen LQ45 (saham paling likuid di IDX), periode 3 Agu - 30 Okt 2026.
# Cek ulang berkala karena BEI evaluasi ulang tiap 3 bulan.
LQ45_LIST = [
    "AADI.JK", "ADMR.JK", "ADRO.JK", "AKRA.JK", "AMMN.JK", "AMRT.JK",
    "ANTM.JK", "ASII.JK", "BBCA.JK", "BBNI.JK", "BBRI.JK", "BMRI.JK",
    "BRPT.JK", "BUMI.JK", "CPIN.JK", "CUAN.JK", "DEWA.JK", "EMTK.JK",
    "ESSA.JK", "EXCL.JK", "GOTO.JK", "HRTA.JK", "ICBP.JK", "INCO.JK",
    "INDF.JK", "INDY.JK", "INKP.JK", "ISAT.JK", "ITMG.JK", "JPFA.JK",
    "KLBF.JK", "MAPI.JK", "MBMA.JK", "MDKA.JK", "MEDC.JK", "NCKL.JK",
    "PGAS.JK", "PGEO.JK", "PTBA.JK", "SCMA.JK", "TLKM.JK", "UNTR.JK",
    "UNVR.JK", "WIFI.JK",
]

# Referensi saham per grup konglomerat — buat deteksi leading-lagging
# dan tambahan watchlist di luar LQ45. Cek ulang berkala, kepemilikan bisa berubah.
KONGLOMERAT_GROUPS = {
    "Prajogo Pangestu (Barito Group)": [
        "BRPT.JK", "TPIA.JK", "CUAN.JK", "BREN.JK", "PTRO.JK", "CDIA.JK", "GZCO.JK",
    ],
    "Bakrie Group": [
        "BUMI.JK", "BRMS.JK", "BNBR.JK", "ELTY.JK", "DEWA.JK", "ENRG.JK",
        "UNSP.JK", "VIVA.JK", "VKTR.JK", "MDIA.JK", "JGLE.JK",
    ],
    "Happy Hapsoro": [
        "RAJA.JK", "RATU.JK", "BUVA.JK", "MINA.JK",
    ],
    "Haji Isam (Jhonlin Group)": [
        "JARR.JK", "PGUN.JK", "TEBE.JK", "PACK.JK",
    ],
    "Djarum Group (Hartono Bersaudara)": [
        "BBCA.JK", "TOWR.JK", "SUPR.JK", "BELI.JK", "RANC.JK", "DATA.JK", "HEAL.JK",
    ],
}
WATCHLIST_KONGLOMERAT = [ticker for group in KONGLOMERAT_GROUPS.values() for ticker in group]

# Watchlist final: gabungan LQ45 + saham konglomerat, duplikat dibuang otomatis
# (urutan aslinya nggak penting karena tiap saham diproses satu-satu)
WATCHLIST = sorted(set(LQ45_LIST + WATCHLIST_KONGLOMERAT))

TELEGRAM_BOT_TOKEN = "ISI_TOKEN_BOT_KAMU"   # dari @BotFather di Telegram
TELEGRAM_CHAT_ID = "ISI_CHAT_ID_KAMU"       # dari @userinfobot di Telegram

# Threshold sinyal - bisa disetel sesuai preferensi
RSI_OVERSOLD = 35
VOLUME_SPIKE_MULTIPLIER = 1.8   # volume hari ini vs rata-rata 20 hari
LOOKBACK_DAYS = "3mo"

# Ukuran batch buat download data sekaligus. Jangan diset terlalu besar
# (Yahoo Finance bisa nolak/rate-limit kalau kebanyakan dalam 1 request).
BATCH_SIZE = 50
BATCH_DELAY_SECONDS = 1.5  # jeda antar batch, biar "sopan" ke server


# =========================================================================
# MODUL AMBIL SEMUA TICKER IDX (opsional, buat scan semua saham)
# =========================================================================

# Sumber: dataset publik emiten IDX (kode, nama, papan pencatatan).
# Diambil live tiap kali fungsi ini dipanggil, BUKAN disimpan hardcode di sini,
# supaya datanya selalu yang paling baru dan reponya nggak berat.
IDX_ALL_TICKERS_URL = "https://raw.githubusercontent.com/wildangunawan/Dataset-Saham-IDX/master/List%20Emiten/all.csv"


def fetch_all_idx_tickers(exclude_boards: list = None) -> list:
    """
    Ambil daftar SEMUA kode saham yang tercatat di IDX (~900+ ticker),
    lalu tambahin suffix .JK biar siap dipakai yfinance.

    exclude_boards: papan pencatatan yang mau di-skip, misal
      ["Pemantauan Khusus"] buat exclude saham yang lagi dalam pengawasan khusus.
      Default: nggak exclude apa-apa (ambil semua).

    Kalau gagal ambil (nggak ada internet / sumbernya down), fungsi ini
    return list kosong dan kasih tau lewat print — jangan bikin program
    crash total gara-gara ini.
    """
    try:
        resp = requests.get(IDX_ALL_TICKERS_URL, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        print(f"[ERROR] Gagal ambil daftar semua ticker IDX: {e}")
        return []

    if exclude_boards:
        df = df[~df["listingBoard"].isin(exclude_boards)]

    tickers = [f"{code.strip()}.JK" for code in df["code"].astype(str)]
    return tickers


# Set True kalau mau screening SEMUA saham IDX, bukan cuma LQ45 + konglomerat.
# HATI-HATI: ini bakal narik ~900 saham, proses jauh lebih lama (beberapa menit)
# meskipun udah pakai batch download.
SCAN_SEMUA_SAHAM_IDX = False

if SCAN_SEMUA_SAHAM_IDX:
    _all_tickers = fetch_all_idx_tickers()
    if _all_tickers:
        WATCHLIST = sorted(set(_all_tickers))
    # kalau gagal ambil (nggak ada internet dsb), WATCHLIST tetap
    # pakai LQ45 + konglomerat yang udah didefinisikan di atas (fallback aman)


# =========================================================================
# MODUL TEKNIKAL
# =========================================================================

def fetch_data(ticker: str) -> pd.DataFrame:
    """Ambil data OHLCV historis (delayed) dari Yahoo Finance - SATU ticker.
    Masih dipakai kalau kamu butuh ambil 1 saham doang secara terpisah."""
    df = yf.download(ticker, period=LOOKBACK_DAYS, interval="1d", progress=False)
    if df.empty or len(df) < 30:
        return pd.DataFrame()
    # yfinance kadang mengembalikan MultiIndex kolom
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_batch_data(tickers: list, batch_size: int = None, delay: float = None) -> dict:
    """
    Ambil data OHLCV buat BANYAK ticker sekaligus, dipecah jadi batch kecil.

    Ini jauh lebih cepat daripada fetch_data() satu-satu, dan lebih aman
    dari resiko rate-limit karena requestnya nggak sekaligus semua.

    Return: dict {ticker: dataframe}. Ticker yang datanya gagal/kurang
    otomatis di-skip (nggak bikin keseluruhan proses berhenti).
    """
    batch_size = batch_size or BATCH_SIZE
    delay = delay if delay is not None else BATCH_DELAY_SECONDS

    result = {}
    total_batches = (len(tickers) + batch_size - 1) // batch_size

    for b in range(total_batches):
        batch = tickers[b * batch_size: (b + 1) * batch_size]
        print(f"[INFO] Batch {b + 1}/{total_batches} ({len(batch)} ticker)...")

        try:
            if len(batch) == 1:
                # yfinance beda formatnya kalau cuma 1 ticker
                raw = yf.download(batch[0], period=LOOKBACK_DAYS, interval="1d", progress=False)
                candidates = {batch[0]: raw}
            else:
                raw = yf.download(
                    batch, period=LOOKBACK_DAYS, interval="1d",
                    group_by="ticker", progress=False, threads=True,
                )
                candidates = {t: (raw[t] if t in raw.columns.get_level_values(0) else pd.DataFrame())
                              for t in batch}
        except Exception as e:
            print(f"[ERROR] Batch {b + 1} gagal total: {e}")
            continue

        for ticker, df in candidates.items():
            df = df.dropna(how="all")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and len(df) >= 30:
                result[ticker] = df

        # jeda antar batch (skip kalau ini batch terakhir)
        if b < total_batches - 1:
            time.sleep(delay)

    print(f"[INFO] Selesai: {len(result)}/{len(tickers)} ticker berhasil diambil datanya.")
    return result


def check_bollinger_riding(df: pd.DataFrame, days_check: int = 3) -> bool:
    """
    Cek apakah harga sedang "riding" (nempel & jalan) di upper band Bollinger.
    Konsep dari video: ini tanda tren KUAT, bukan tanda overbought buat jual.

    Caranya: ambil N hari terakhir, cek apakah harga penutupan tiap hari
    itu deket (95%+) dari garis upper band di hari yang sama.
    """
    close = df["Close"]
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    upper_band = bb.bollinger_hband()

    # ambil beberapa hari terakhir
    recent_close = close.iloc[-days_check:]
    recent_upper = upper_band.iloc[-days_check:]

    if recent_upper.isna().any():
        return False  # data belum cukup buat hitung BB

    # harga dianggap "riding" kalau selalu di atas 95% dari upper band
    riding = (recent_close >= recent_upper * 0.95).all()
    return bool(riding)


def find_support_resistance(df: pd.DataFrame, window: int = 5, cluster_pct: float = 0.02) -> dict:
    """
    Cari level support & resistance dari data harga historis.

    Logikanya sederhana:
    1. Cari titik "lembah" (support candidate) = titik dengan harga Low
       yang paling rendah dibanding beberapa hari sebelum & sesudahnya
    2. Cari titik "puncak" (resistance candidate) = sebaliknya
    3. Kelompokkan (cluster) titik-titik yang harganya berdekatan
       (dalam radius cluster_pct) jadi satu level
    4. Cuma level yang "disentuh" minimal 2 kali yang dianggap valid
       (support/resistance asli, bukan kebetulan)
    """
    lows = df["Low"].values
    highs = df["High"].values
    n = len(lows)

    pivot_lows = []
    pivot_highs = []
    for i in range(window, n - window):
        if lows[i] == min(lows[i - window: i + window + 1]):
            pivot_lows.append(lows[i])
        if highs[i] == max(highs[i - window: i + window + 1]):
            pivot_highs.append(highs[i])

    def cluster_levels(levels):
        if not levels:
            return []
        levels = sorted(levels)
        clusters = [[levels[0]]]
        for lvl in levels[1:]:
            if abs(lvl - clusters[-1][-1]) / clusters[-1][-1] < cluster_pct:
                clusters[-1].append(lvl)
            else:
                clusters.append([lvl])
        # cuma ambil cluster yang disentuh >= 2 kali (level yang "diuji" berulang)
        return [sum(c) / len(c) for c in clusters if len(c) >= 2]

    support_levels = cluster_levels(pivot_lows)
    resistance_levels = cluster_levels(pivot_highs)

    return {"support": support_levels, "resistance": resistance_levels}


def check_near_support(price: float, support_levels: list, tolerance_pct: float = 0.03) -> bool:
    """Cek apakah harga sekarang lagi deket (dalam radius tolerance_pct) sama salah satu level support."""
    for lvl in support_levels:
        if abs(price - lvl) / lvl <= tolerance_pct:
            return True
    return False


def detect_vcp(df: pd.DataFrame, window: int = 5, min_pullbacks: int = 2) -> bool:
    """
    Deteksi pola VCP (Volatility Contraction Pattern) versi sederhana.

    Konsep dari video: harga membentuk beberapa koreksi (pullback) yang
    besarnya SEMAKIN MENGECIL tiap fase, sebelum akhirnya breakout.

    Caranya:
    1. Cari titik puncak & lembah bergantian (swing point)
    2. Hitung besar tiap pullback (dari puncak ke lembah berikutnya, dalam %)
    3. Cek apakah pullback-pullback terakhir itu polanya mengecil terus
    """
    highs = df["High"].values
    lows = df["Low"].values
    n = len(highs)

    swings = []  # list of (index, price, tipe) tipe: 'high' atau 'low'
    for i in range(window, n - window):
        if highs[i] == max(highs[i - window: i + window + 1]):
            swings.append((i, highs[i], "high"))
        elif lows[i] == min(lows[i - window: i + window + 1]):
            swings.append((i, lows[i], "low"))

    swings.sort(key=lambda x: x[0])

    # cari pasangan high->low berurutan (itu yang disebut "pullback")
    pullback_depths = []
    for j in range(len(swings) - 1):
        cur, nxt = swings[j], swings[j + 1]
        if cur[2] == "high" and nxt[2] == "low":
            depth_pct = (cur[1] - nxt[1]) / cur[1]
            pullback_depths.append(depth_pct)

    if len(pullback_depths) < min_pullbacks:
        return False

    # ambil beberapa pullback terakhir, cek apakah trennya mengecil
    last_pullbacks = pullback_depths[-min_pullbacks:]
    is_contracting = all(
        last_pullbacks[i] > last_pullbacks[i + 1] for i in range(len(last_pullbacks) - 1)
    )
    return is_contracting


def _evaluate_conditions(conditions: list) -> tuple:
    """
    Helper: dari list kondisi [{"label", "passed", "value"}, ...],
    hitung apakah SEMUA lolos, dan susun teks bukti buat ditampilkan.
    """
    all_passed = all(c["passed"] for c in conditions)
    return all_passed, conditions


def check_bsjp(df: pd.DataFrame) -> dict:
    """
    Screener 'BSJP' (Beli Sore Jual Pagi) ala Stockbit — nyari saham yang baru
    breakout volume & harga di atas rata-rata, tanda minat beli kuat sore hari.

    Return dict berisi status lolos/tidak PLUS bukti angka tiap kriteria,
    biar bisa diverifikasi manual (bukan cuma "true/false" doang).
    """
    close = df["Close"]
    open_ = df["Open"]
    volume = df["Volume"]

    price = close.iloc[-1]
    prev_close = close.iloc[-2]
    today_open = open_.iloc[-1]
    today_volume = volume.iloc[-1]
    prev_volume = volume.iloc[-2]
    vol_ma20 = volume.rolling(20).mean().iloc[-1]
    price_ma5 = close.rolling(5).mean().iloc[-1]
    value_transaksi = price * today_volume
    pct_change = (price / prev_close - 1) * 100

    conditions = [
        {
            "label": "Naik ≥5% dari kemarin",
            "passed": bool(price >= 1.05 * prev_close),
            "value": f"{pct_change:+.1f}% (harga {price:.0f} vs kemarin {prev_close:.0f})",
        },
        {
            "label": "Volume breakout (≥2x MA20 & ≥1x kemarin)",
            "passed": bool((today_volume >= 2 * vol_ma20) and (today_volume >= prev_volume)),
            "value": f"{today_volume:,.0f} (MA20: {vol_ma20:,.0f}, kemarin: {prev_volume:,.0f})",
        },
        {
            "label": "Harga ≥ MA5",
            "passed": bool(price >= price_ma5),
            "value": f"{price:.0f} vs MA5 {price_ma5:.0f}",
        },
        {
            "label": "Harga ≥ Open (nggak turun dari open)",
            "passed": bool(price >= today_open),
            "value": f"{price:.0f} vs Open {today_open:.0f}",
        },
        {
            "label": "Value transaksi > Rp5 miliar",
            "passed": bool(value_transaksi > 5_000_000_000),
            "value": f"Rp{value_transaksi:,.0f}",
        },
        {
            "label": "Harga sebelumnya > 50 (bukan saham gocap)",
            "passed": bool(prev_close > 50),
            "value": f"{prev_close:.0f}",
        },
    ]
    passed, evidence = _evaluate_conditions(conditions)
    return {
        "passed": passed,
        "evidence": evidence,
        "detail": "Lolos screener BSJP (Beli Sore Jual Pagi)" if passed else None,
    }


def check_bpjs(df: pd.DataFrame) -> dict:
    """
    Screener 'BPJS' (Beli Pagi Jual Sore) ala Stockbit — versi lebih longgar
    dari BSJP, dicek 30 menit sebelum market buka.
    """
    close = df["Close"]
    open_ = df["Open"]
    volume = df["Volume"]

    price = close.iloc[-1]
    prev_close = close.iloc[-2]
    today_open = open_.iloc[-1]
    today_volume = volume.iloc[-1]
    prev_volume = volume.iloc[-2]
    price_ma5 = close.rolling(5).mean().iloc[-1]
    value_transaksi = price * today_volume
    pct_change = (price / prev_close - 1) * 100

    conditions = [
        {
            "label": "Harga ≥ MA5",
            "passed": bool(price >= price_ma5),
            "value": f"{price:.0f} vs MA5 {price_ma5:.0f}",
        },
        {
            "label": "Naik ≥5% dari kemarin",
            "passed": bool(price >= 1.05 * prev_close),
            "value": f"{pct_change:+.1f}% (harga {price:.0f} vs kemarin {prev_close:.0f})",
        },
        {
            "label": "Harga ≥ Open",
            "passed": bool(price >= today_open),
            "value": f"{price:.0f} vs Open {today_open:.0f}",
        },
        {
            "label": "Volume ≥ 0.2x kemarin",
            "passed": bool(today_volume >= 0.2 * prev_volume),
            "value": f"{today_volume:,.0f} vs 0.2x kemarin ({0.2 * prev_volume:,.0f})",
        },
        {
            "label": "Value transaksi > Rp5 miliar",
            "passed": bool(value_transaksi > 5_000_000_000),
            "value": f"Rp{value_transaksi:,.0f}",
        },
    ]
    passed, evidence = _evaluate_conditions(conditions)
    return {
        "passed": passed,
        "evidence": evidence,
        "detail": "Lolos screener BPJS (Beli Pagi Jual Sore)" if passed else None,
    }


def check_day_trade(df: pd.DataFrame) -> dict:
    """
    Screener 'Day Trade' — kombinasi sinyal buat trading intraday/harian
    (beda dari BSJP/BPJS yang formula persis Stockbit). Disusun dari
    indikator² yang udah ada: momentum, volume, dan tren jangka pendek.

    Kriteria (kombinasi harus semua benar):
    - Volume hari ini >= 1.5x rata-rata 20 hari (minat beli aktif)
    - MACD histogram positif (momentum bullish jangka pendek)
    - Harga di atas MA20 (tren jangka pendek naik)
    - RSI antara 40-70 (momentum sehat, belum overbought ekstrem)
    - Value transaksi > Rp1 miliar (likuiditas minimum buat day trade)
    """
    close = df["Close"]
    volume = df["Volume"]

    price = close.iloc[-1]
    today_volume = volume.iloc[-1]
    vol_ma20 = volume.rolling(20).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    value_transaksi = price * today_volume

    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    macd_hist = ta.trend.MACD(close).macd_diff().iloc[-1]

    conditions = [
        {
            "label": "Volume ≥1.5x rata-rata 20 hari",
            "passed": bool(today_volume >= 1.5 * vol_ma20),
            "value": f"{today_volume:,.0f} vs 1.5x MA20 ({1.5 * vol_ma20:,.0f})",
        },
        {
            "label": "MACD histogram positif (momentum naik)",
            "passed": bool(macd_hist > 0),
            "value": f"{macd_hist:.2f}",
        },
        {
            "label": "Harga di atas MA20",
            "passed": bool(price > ma20),
            "value": f"{price:.0f} vs MA20 {ma20:.0f}",
        },
        {
            "label": "RSI antara 40-70 (momentum sehat)",
            "passed": bool(40 <= rsi <= 70),
            "value": f"RSI {rsi:.1f}",
        },
        {
            "label": "Value transaksi > Rp1 miliar",
            "passed": bool(value_transaksi > 1_000_000_000),
            "value": f"Rp{value_transaksi:,.0f}",
        },
    ]
    passed, evidence = _evaluate_conditions(conditions)
    return {
        "passed": passed,
        "evidence": evidence,
        "detail": "Lolos screener Day Trade" if passed else None,
    }


def compute_all_indicators(df: pd.DataFrame) -> dict:
    """
    Hitung SEMUA indikator teknikal satu-satu, return dict per indikator
    (bukan langsung dijumlah). Ini dipakai biar UI web bisa kasih checkbox
    per indikator — user pilih sendiri mana yang mau dihitung ke skor.

    Setiap entry formatnya: {"triggered": bool, "label": str, "detail": str or None}

    CATATAN: BSJP dan BPJS TIDAK dimasukkan ke sini. Dua itu bukan indikator
    yang ikut dijumlah ke skor, tapi ADD-ON SCREENING terpisah — lihat
    compute_addon_screens(). Alasannya: BSJP/BPJS itu sendiri udah kombinasi
    beberapa kondisi sekaligus (bukan 1 sinyal tunggal), jadi lebih pas
    diperlakukan sebagai filter "lolos/tidak", bukan poin tambahan skor.
    """
    close = df["Close"]
    volume = df["Volume"]

    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    macd = ta.trend.MACD(close)
    macd_line = macd.macd().iloc[-1]
    macd_signal = macd.macd_signal().iloc[-1]
    macd_bullish_cross = (
        macd.macd().iloc[-2] < macd.macd_signal().iloc[-2]
        and macd_line > macd_signal
    )

    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else None
    price = close.iloc[-1]

    avg_volume_20 = volume.rolling(20).mean().iloc[-1]
    today_volume = volume.iloc[-1]
    volume_ratio = today_volume / avg_volume_20 if avg_volume_20 > 0 else 0

    bb_riding = check_bollinger_riding(df)
    sr_levels = find_support_resistance(df)
    near_support = check_near_support(price, sr_levels["support"])
    vcp_detected = detect_vcp(df)

    indicators = {
        "rsi_oversold": {
            "triggered": bool(rsi < RSI_OVERSOLD),
            "label": "RSI Oversold",
            "detail": f"RSI {rsi:.1f} (< {RSI_OVERSOLD})" if rsi < RSI_OVERSOLD else None,
        },
        "macd_cross": {
            "triggered": bool(macd_bullish_cross),
            "label": "MACD Golden Cross",
            "detail": "MACD bullish crossover" if macd_bullish_cross else None,
        },
        "volume_spike": {
            "triggered": bool(volume_ratio >= VOLUME_SPIKE_MULTIPLIER),
            "label": "Volume Spike",
            "detail": f"Volume {volume_ratio:.1f}x rata-rata" if volume_ratio >= VOLUME_SPIKE_MULTIPLIER else None,
        },
        "above_ma20": {
            "triggered": bool(price > ma20),
            "label": "Harga di atas MA20",
            "detail": "Harga di atas MA20" if price > ma20 else None,
        },
        "uptrend_ma": {
            "triggered": bool(ma50 is not None and ma20 > ma50),
            "label": "MA20 > MA50 (uptrend)",
            "detail": "MA20 > MA50 (uptrend)" if (ma50 is not None and ma20 > ma50) else None,
        },
        "bollinger_riding": {
            "triggered": bool(bb_riding),
            "label": "Riding Upper Bollinger Band",
            "detail": "Riding upper Bollinger Band (tren kuat)" if bb_riding else None,
        },
        "near_support": {
            "triggered": bool(near_support),
            "label": "Dekat Area Support",
            "detail": "Harga dekat area support" if near_support else None,
        },
        "vcp_pattern": {
            "triggered": bool(vcp_detected),
            "label": "Pola VCP",
            "detail": "Pola VCP terdeteksi (pullback mengecil)" if vcp_detected else None,
        },
    }

    return {
        "price": price,
        "rsi": rsi,
        "volume_ratio": volume_ratio,
        "support_levels": sr_levels["support"],
        "resistance_levels": sr_levels["resistance"],
        "indicators": indicators,
    }


def compute_screener_results(df: pd.DataFrame) -> dict:
    """
    Jalankan SEMUA screener (Day Trade, BSJP, BPJS) sekaligus buat 1 saham.
    Masing-masing independen - saham bisa lolos satu, dua, atau ketiganya.

    Return: {"day_trade": {...}, "bsjp": {...}, "bpjs": {...}}
    Tiap entry berisi "passed", "evidence" (list bukti per kriteria), "detail".
    """
    results = {}
    for key, fn in [("day_trade", check_day_trade), ("bsjp", check_bsjp), ("bpjs", check_bpjs)]:
        try:
            results[key] = fn(df)
        except Exception as e:
            results[key] = {"passed": False, "evidence": [], "detail": None, "error": str(e)}
    return results


def compute_signals(df: pd.DataFrame, selected_indicators: list = None) -> dict:
    """
    Wrapper di atas compute_all_indicators() — hitung skor & alasan
    berdasarkan indikator yang DIPILIH aja (kalau None, pakai SEMUA indikator).

    selected_indicators: list of key, misal ["rsi_oversold", "macd_cross", "bsjp"]
    """
    all_data = compute_all_indicators(df)
    indicators = all_data["indicators"]

    keys_to_use = selected_indicators if selected_indicators is not None else list(indicators.keys())

    score = 0
    reasons = []
    for key in keys_to_use:
        ind = indicators.get(key)
        if ind and ind["triggered"]:
            score += 1
            reasons.append(ind["detail"] or ind["label"])

    return {
        "price": all_data["price"],
        "rsi": all_data["rsi"],
        "volume_ratio": all_data["volume_ratio"],
        "support_levels": all_data["support_levels"],
        "resistance_levels": all_data["resistance_levels"],
        "indicators": indicators,
        "score": score,
        "reasons": reasons,
    }


# =========================================================================
# MODUL NOTIFIKASI
# =========================================================================

def send_telegram_message(text: str):
    if "ISI_" in TELEGRAM_BOT_TOKEN or "ISI_" in TELEGRAM_CHAT_ID:
        print("[SKIP] Telegram belum dikonfigurasi. Pesan:\n", text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, data=payload, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Gagal kirim Telegram: {e}")


# =========================================================================
# MODUL FUNDAMENTAL & BERITA (analisis berita sebagai penguat fundamental+teknikal)
# =========================================================================

# --- Opsional: isi ini kalau mau analisis berita pakai AI beneran (Claude),
# bukan cuma keyword matching. Dapetin API key di console.anthropic.com.
# Ada biaya kecil per analisis (bayar sesuai pemakaian, bukan gratis).
# Kalau dibiarkan "ISI_..." (default), otomatis fallback ke keyword matching.
ANTHROPIC_API_KEY = "ISI_API_KEY_ANTHROPIC_KAMU_KALAU_MAU_ANALISIS_AI"
ANTHROPIC_MODEL = "claude-sonnet-4-6"

# Berita yang lebih tua dari ini (hari) diabaikan - biar analisisnya
# selalu berdasarkan info terkini, bukan berita basi.
NEWS_MAX_AGE_DAYS = 31

# Kata kunci sederhana buat fallback keyword matching (kalau AI nggak dipakai).
# Ini pendekatan kasar, bukan NLP canggih - anggap sebagai sinyal lemah.
POSITIVE_NEWS_KEYWORDS = [
    "laba naik", "laba melonjak", "untung", "cuan", "ekspansi", "akuisisi",
    "kenaikan", "melesat", "menguat", "dividen", "kontrak baru", "kerja sama",
    "rekor", "pertumbuhan", "optimis", "prospek cerah", "borong", "capai target",
]
NEGATIVE_NEWS_KEYWORDS = [
    "rugi", "turun tajam", "anjlok", "gugatan", "korupsi", "phk", "delisting",
    "suspensi", "gagal bayar", "utang", "penurunan", "penyelidikan", "skandal",
    "kebakaran", "kecelakaan", "sanksi", "denda", "restrukturisasi",
]
# Kata negasi - kalau ini muncul TEPAT SEBELUM keyword, artinya dibalik.
# Contoh: "tidak untung" -> jangan dihitung positif.
NEGATION_WORDS = ["tidak", "bukan", "belum", "tanpa", "gagal", "batal"]


def fetch_news_headlines(query: str, max_articles: int = 15, max_age_days: int = None) -> list:
    """
    Ambil judul + LINK + TANGGAL berita soal 'query' dari Google News RSS,
    difilter cuma yang berumur <= max_age_days (default: NEWS_MAX_AGE_DAYS).

    Return: list of dict [{"title": str, "link": str, "published": datetime}, ...]
    Diurutkan dari yang PALING BARU. Kalau gagal/nggak ada internet, return
    list kosong (nggak bikin program crash).
    """
    import urllib.parse
    import re as _re
    from email.utils import parsedate_to_datetime
    from datetime import timezone

    max_age_days = max_age_days if max_age_days is not None else NEWS_MAX_AGE_DAYS
    cutoff = datetime.now(timezone.utc) - pd.Timedelta(days=max_age_days)

    url = (
        "https://news.google.com/rss/search?q="
        + urllib.parse.quote(query)
        + "&hl=id&gl=ID&ceid=ID:id"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        items = _re.findall(r"<item>(.*?)</item>", resp.text, flags=_re.S)
        results = []
        for item in items:
            title_match = _re.search(r"<title>(.*?)</title>", item, flags=_re.S)
            link_match = _re.search(r"<link>(.*?)</link>", item, flags=_re.S)
            date_match = _re.search(r"<pubDate>(.*?)</pubDate>", item, flags=_re.S)
            if not (title_match and link_match):
                continue

            published = None
            if date_match:
                try:
                    published = parsedate_to_datetime(date_match.group(1).strip())
                    if published.tzinfo is None:
                        published = published.replace(tzinfo=timezone.utc)
                except Exception:
                    published = None

            # Skip berita yang lebih tua dari batas umur, ATAU yang nggak
            # ada tanggalnya sama sekali (lebih aman di-skip daripada nebak)
            if published is None or published < cutoff:
                continue

            results.append({
                "title": title_match.group(1).strip(),
                "link": link_match.group(1).strip(),
                "published": published,
            })
            if len(results) >= max_articles:
                break

        results.sort(key=lambda x: x["published"], reverse=True)
        return results
    except Exception as e:
        print(f"[WARN] Gagal ambil berita untuk '{query}': {e}")
        return []


def build_news_search_url(query: str) -> str:
    """
    Bikin link Google News yang bisa dibuka manual di browser buat
    verifikasi sendiri.
    """
    import urllib.parse
    return "https://news.google.com/search?q=" + urllib.parse.quote(query) + "&hl=id&gl=ID&ceid=ID:id"


def _keyword_sentiment_fallback(headlines: list) -> dict:
    """
    Fallback KALAU nggak ada API key AI: keyword matching yang sedikit
    lebih pintar dari versi sebelumnya - sekarang cek NEGASI juga.
    Contoh: "tidak untung" -> nggak dihitung positif lagi.

    Tetap kasar dan gampang salah baca konteks rumit - anggap sinyal LEMAH.
    """
    pos_count = 0
    neg_count = 0
    matched_titles = []

    for h in headlines:
        words = h["title"].lower().split()
        text = h["title"].lower()

        for kw in POSITIVE_NEWS_KEYWORDS:
            if kw in text:
                idx = text.find(kw)
                # cek 3 kata sebelum keyword, ada negasi atau nggak
                before = text[:idx].split()[-3:]
                if any(neg in before for neg in NEGATION_WORDS):
                    neg_count += 1  # dibalik jadi negatif
                else:
                    pos_count += 1
                    matched_titles.append(h["title"])

        for kw in NEGATIVE_NEWS_KEYWORDS:
            if kw in text:
                idx = text.find(kw)
                before = text[:idx].split()[-3:]
                if any(neg in before for neg in NEGATION_WORDS):
                    pos_count += 1  # "tidak rugi" -> dianggap positif
                else:
                    neg_count += 1

    net_sentiment = pos_count - neg_count
    return {
        "triggered": net_sentiment > 0,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "summary": None,  # fallback nggak bisa bikin ringkasan asli, cuma hitungan
        "matched_titles": matched_titles[:3],
    }


def _ai_news_analysis(ticker: str, headlines: list) -> dict:
    """
    Analisis berita pakai Claude API - baca judul-judul berita beneran
    (bukan cuma cocokin kata), kasih penilaian relevansi ke fundamental
    & teknikal saham, plus ringkasan singkat kenapa.

    Return None kalau API key belum diisi atau request gagal (fallback
    otomatis ke keyword matching di compute_news_sentiment).
    """
    if "ISI_" in ANTHROPIC_API_KEY or not ANTHROPIC_API_KEY:
        return None

    headlines_text = "\n".join(f"- {h['title']}" for h in headlines)
    prompt = f"""Kamu menganalisis berita terbaru (maks {NEWS_MAX_AGE_DAYS} hari terakhir) soal saham {ticker} di Bursa Efek Indonesia.

Judul-judul berita:
{headlines_text}

Analisis apakah berita-berita ini, secara keseluruhan, memperkuat atau memperlemah keyakinan untuk MEMBELI saham ini, dari sisi fundamental (kinerja bisnis, kondisi keuangan) maupun teknikal (katalis yang bisa gerakkan harga jangka pendek).

Jawab HANYA dalam format JSON persis seperti ini, tanpa teks lain:
{{"sentimen": "positif" atau "negatif" atau "netral", "kekuatan_sinyal": angka 1-5, "ringkasan": "1-2 kalimat alasan singkat dalam bahasa Indonesia"}}"""

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["content"][0]["text"].strip()
        # buang markdown code fence kalau ada
        text = text.replace("```json", "").replace("```", "").strip()

        import json
        parsed = json.loads(text)

        sentimen = parsed.get("sentimen", "netral")
        kekuatan = int(parsed.get("kekuatan_sinyal", 0))
        ringkasan = parsed.get("ringkasan", "")

        return {
            "triggered": sentimen == "positif" and kekuatan >= 3,
            "pos_count": None,
            "neg_count": None,
            "summary": f"[Analisis AI] {ringkasan} (sentimen: {sentimen}, kekuatan: {kekuatan}/5)",
            "matched_titles": [],
        }
    except Exception as e:
        print(f"[WARN] Analisis AI gagal untuk '{ticker}': {e} — fallback ke keyword matching")
        return None


def compute_news_sentiment(query: str, ticker: str = None) -> dict:
    """
    Analisis sentimen berita terbaru (maks NEWS_MAX_AGE_DAYS hari) buat
    dipakai sebagai PENGUAT fundamental + teknikal.

    Coba pakai AI (Claude) dulu kalau API key udah diisi. Kalau nggak
    (atau gagal), otomatis fallback ke keyword matching yang udah
    diperbaiki (dengan deteksi negasi).
    """
    search_url = build_news_search_url(query)
    headlines = fetch_news_headlines(query)

    if not headlines:
        return {
            "triggered": False, "label": "Sentimen Berita", "detail": None,
            "headlines": [], "search_url": search_url,
        }

    result = _ai_news_analysis(ticker or query, headlines)
    used_ai = result is not None
    if result is None:
        result = _keyword_sentiment_fallback(headlines)

    detail = None
    if result["triggered"]:
        if used_ai:
            detail = f"{result['summary']} — {len(headlines)} berita ({NEWS_MAX_AGE_DAYS} hari terakhir), cek manual: {search_url}"
        else:
            detail = (
                f"Berita cenderung positif ({result['pos_count']} sinyal positif vs "
                f"{result['neg_count']} negatif dari {len(headlines)} judul, "
                f"{NEWS_MAX_AGE_DAYS} hari terakhir) — cek manual: {search_url}"
            )

    return {
        "triggered": result["triggered"],
        "label": "Sentimen Berita Positif" + (" (AI)" if used_ai else " (keyword)"),
        "detail": detail,
        "headlines": headlines[:5],
        "search_url": search_url,
        "used_ai": used_ai,
    }


def compute_fundamental_score(ticker: str) -> dict:
    """
    Skor fundamental dari sinyal berita (news sentiment) sebagai penguat.
    Nama perusahaan dipakai buat query berita - pakai kode ticker aja
    dulu (tanpa .JK) karena biasanya itu yang paling relevan hasil pencariannya.

    Selalu menyertakan search_url biar kamu bisa cek manual sendiri -
    ingat, deteksi sentimennya cuma keyword matching kasar.
    """
    company_query = ticker.replace(".JK", "") + " saham"
    news = compute_news_sentiment(company_query, ticker=ticker)

    score = 1 if news["triggered"] else 0
    reasons = [news["detail"]] if news["detail"] else []

    return {
        "score": score,
        "reasons": reasons,
        "news_headlines": news.get("headlines", []),
        "search_url": news.get("search_url"),
    }


def compute_sentiment_score(ticker: str) -> dict:
    """
    Placeholder terpisah buat sentimen sosial media (beda dari berita resmi).
    Belum diimplementasi - butuh data platform sosial media yang biasanya
    berbayar/proprietary (lihat dokumentasi kebutuhan data).
    """
    return {"score": 0, "reasons": []}


# =========================================================================
# MAIN
# =========================================================================

def run_screener(active_screeners: list = None):
    """
    active_screeners: list dari "day_trade", "bsjp", "bpjs" yang mau diaktifkan.
    Saham ditampilkan kalau lolos SALAH SATU dari screener yang aktif.
    Default (None): semua 3 screener aktif.
    """
    active_screeners = active_screeners if active_screeners is not None else ["day_trade", "bsjp", "bpjs"]

    print(f"=== Screening dijalankan: {datetime.now()} ===")
    print(f"Total saham di watchlist: {len(WATCHLIST)}")
    print(f"Screener aktif: {', '.join(active_screeners)}")

    batch_data = fetch_batch_data(WATCHLIST)
    results = []

    for ticker, df in batch_data.items():
        screeners = compute_screener_results(df)

        # saham lolos kalau match SALAH SATU screener yang aktif
        passed_any = any(screeners[s]["passed"] for s in active_screeners if s in screeners)
        if not passed_any:
            continue

        tech = compute_signals(df)
        fund = compute_fundamental_score(ticker)

        total_score = tech["score"] + fund["score"]
        all_reasons = list(tech["reasons"]) + list(fund["reasons"])

        lolos_screener = []
        for s in active_screeners:
            if screeners.get(s, {}).get("passed"):
                lolos_screener.append(s)
                all_reasons.append(screeners[s]["detail"])

        results.append({
            "ticker": ticker,
            "price": tech["price"],
            "score": total_score,
            "lolos_screener": lolos_screener,
            "screener_evidence": {s: screeners[s]["evidence"] for s in active_screeners if s in screeners},
            "reasons": all_reasons,
        })

    # Urutkan dari skor tertinggi
    results.sort(key=lambda x: x["score"], reverse=True)
    candidates = results

    if candidates:
        msg_lines = ["*Screening Result*", ""]
        for c in candidates:
            msg_lines.append(
                f"*{c['ticker']}* — skor {c['score']}\n"
                f"Harga: {c['price']:.0f}\n"
                f"Alasan: {', '.join(c['reasons'])}\n"
            )
        message = "\n".join(msg_lines)
    else:
        message = "Screening selesai — tidak ada kandidat kuat hari ini."

    print(message)
    send_telegram_message(message)


if __name__ == "__main__":
    run_screener()


# =========================================================================
# CATATAN SETUP OTOMATIS (baca ini)
# =========================================================================
#
# A) SETUP TELEGRAM BOT (gratis, 5 menit):
#    1. Chat @BotFather di Telegram -> /newbot -> ikuti instruksi -> dapat TOKEN
#    2. Chat @userinfobot -> dapat CHAT_ID kamu
#    3. Isi TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID di atas
#
# B) JADWALKAN OTOMATIS - Opsi 1: Cron di komputer/server sendiri
#    Edit crontab: `crontab -e`
#    Jalankan tiap jam bursa (misal jam 10, 13, 15 WIB hari kerja):
#    0 10,13,15 * * 1-5 /usr/bin/python3 /path/ke/stock_screener.py
#
# C) JADWALKAN OTOMATIS - Opsi 2: GitHub Actions (gratis, cloud, "set & forget")
#    Buat repo, taruh file ini di dalamnya, lalu buat file
#    .github/workflows/screener.yml dengan schedule (cron) yang trigger
#    `pip install -r requirements.txt && python stock_screener.py`.
#    Simpan TOKEN & CHAT_ID sebagai GitHub Secrets, jangan hardcode di kode
#    kalau repo public.
#
# D) PENGEMBANGAN SELANJUTNYA
#    - Tambah lebih banyak indikator (Bollinger Bands, Stochastic, ADX)
#    - Isi compute_fundamental_score() dengan data rasio (PER, PBV, ROE)
#    - Isi compute_sentiment_score() dengan scraping berita/RSS
#    - Tambah backtest sederhana untuk validasi seberapa akurat sinyal ini
#      secara historis SEBELUM dipakai dengan uang sungguhan
