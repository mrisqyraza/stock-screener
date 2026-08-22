"""
IDX Stock Screener - Instrumen Screening Saham
=========================================
Deteksi saham dengan sinyal teknikal, screener bergaya Stockbit (BSJP/BPJS),
dan Day Trade, lalu kirim notifikasi ke Telegram.

Cara pakai:
1. pip install yfinance pandas ta scipy requests
2. Isi TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID (lihat panduan di bawah)
3. Jalankan manual: python stock_screener.py
4. Jadwalkan otomatis via cron / GitHub Actions (lihat catatan di bagian bawah file)

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

# Referensi saham per grup konglomerat — tambahan watchlist di luar LQ45.
# Cek ulang berkala, kepemilikan bisa berubah.
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

# Watchlist final: gabungan LQ45 + saham konglomerat, duplikat dibuang otomatis.
WATCHLIST = sorted(set(LQ45_LIST + WATCHLIST_KONGLOMERAT))

TELEGRAM_BOT_TOKEN = "ISI_TOKEN_BOT_KAMU"   # dari @BotFather di Telegram
TELEGRAM_CHAT_ID = "ISI_CHAT_ID_KAMU"       # dari @userinfobot di Telegram

# Threshold sinyal - bisa disetel sesuai preferensi
RSI_OVERSOLD = 35
VOLUME_SPIKE_MULTIPLIER = 1.8   # volume hari ini vs rata-rata 20 hari
LOOKBACK_DAYS = "3mo"

# Ukuran batch buat download data sekaligus.
BATCH_SIZE = 50
BATCH_DELAY_SECONDS = 1.5  # jeda antar batch, biar "sopan" ke server


# =========================================================================
# MODUL AMBIL SEMUA TICKER IDX (opsional, buat scan semua saham)
# =========================================================================

IDX_ALL_TICKERS_URL = "https://raw.githubusercontent.com/wildangunawan/Dataset-Saham-IDX/master/List%20Emiten/all.csv"


def fetch_all_idx_tickers(exclude_boards: list = None) -> list:
    """
    Ambil daftar SEMUA kode saham yang tercatat di IDX (~900+ ticker),
    lalu tambahin suffix .JK biar siap dipakai yfinance.
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


# =========================================================================
# MODUL TEKNIKAL
# =========================================================================

def fetch_data(ticker: str) -> pd.DataFrame:
    """Ambil data OHLCV historis (delayed) dari Yahoo Finance - SATU ticker."""
    df = yf.download(ticker, period=LOOKBACK_DAYS, interval="1d", progress=False)
    if df.empty or len(df) < 30:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_batch_data(tickers: list, batch_size: int = None, delay: float = None) -> dict:
    """
    Ambil data OHLCV buat BANYAK ticker sekaligus, dipecah jadi batch kecil.
    Jauh lebih cepat daripada fetch_data() satu-satu, dan lebih aman dari
    resiko rate-limit karena requestnya nggak sekaligus semua.
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

        if b < total_batches - 1:
            time.sleep(delay)

    print(f"[INFO] Selesai: {len(result)}/{len(tickers)} ticker berhasil diambil datanya.")
    return result


def check_bollinger_riding(df: pd.DataFrame, days_check: int = 3) -> bool:
    """
    Cek apakah harga sedang "riding" (nempel & jalan) di upper band Bollinger.
    Ini tanda tren KUAT, bukan tanda overbought buat jual.
    """
    close = df["Close"]
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    upper_band = bb.bollinger_hband()

    recent_close = close.iloc[-days_check:]
    recent_upper = upper_band.iloc[-days_check:]

    if recent_upper.isna().any():
        return False

    riding = (recent_close >= recent_upper * 0.95).all()
    return bool(riding)


def find_support_resistance(df: pd.DataFrame, window: int = 5, cluster_pct: float = 0.02) -> dict:
    """
    Cari level support & resistance dari data harga historis.

    1. Cari titik "lembah" (support candidate) & "puncak" (resistance candidate)
    2. Kelompokkan (cluster) titik-titik yang harganya berdekatan jadi satu level
    3. Cuma level yang "disentuh" minimal 2 kali yang dianggap valid
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
        return [sum(c) / len(c) for c in clusters if len(c) >= 2]

    support_levels = cluster_levels(pivot_lows)
    resistance_levels = cluster_levels(pivot_highs)

    return {"support": support_levels, "resistance": resistance_levels}


def check_near_support(price: float, support_levels: list, tolerance_pct: float = 0.03) -> bool:
    """Cek apakah harga sekarang lagi deket sama salah satu level support."""
    for lvl in support_levels:
        if abs(price - lvl) / lvl <= tolerance_pct:
            return True
    return False


def detect_vcp(df: pd.DataFrame, window: int = 5, min_pullbacks: int = 2) -> bool:
    """
    Deteksi pola VCP (Volatility Contraction Pattern) versi sederhana:
    harga membentuk beberapa koreksi (pullback) yang SEMAKIN MENGECIL
    tiap fase, sebelum akhirnya breakout.
    """
    highs = df["High"].values
    lows = df["Low"].values
    n = len(highs)

    swings = []
    for i in range(window, n - window):
        if highs[i] == max(highs[i - window: i + window + 1]):
            swings.append((i, highs[i], "high"))
        elif lows[i] == min(lows[i - window: i + window + 1]):
            swings.append((i, lows[i], "low"))

    swings.sort(key=lambda x: x[0])

    pullback_depths = []
    for j in range(len(swings) - 1):
        cur, nxt = swings[j], swings[j + 1]
        if cur[2] == "high" and nxt[2] == "low":
            depth_pct = (cur[1] - nxt[1]) / cur[1]
            pullback_depths.append(depth_pct)

    if len(pullback_depths) < min_pullbacks:
        return False

    last_pullbacks = pullback_depths[-min_pullbacks:]
    is_contracting = all(
        last_pullbacks[i] > last_pullbacks[i + 1] for i in range(len(last_pullbacks) - 1)
    )
    return is_contracting


def _evaluate_conditions(conditions: list) -> tuple:
    """Helper: dari list kondisi, hitung apakah SEMUA lolos, susun bukti."""
    all_passed = all(c["passed"] for c in conditions)
    return all_passed, conditions


def check_bsjp(df: pd.DataFrame) -> dict:
    """
    Screener 'BSJP' (Beli Sore Jual Pagi) ala Stockbit — nyari saham yang baru
    breakout volume & harga di atas rata-rata, tanda minat beli kuat sore hari.
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
            "key": "bsjp_naik5",
            "label": "Naik ≥5% dari kemarin",
            "passed": bool(price >= 1.05 * prev_close),
            "value": f"{pct_change:+.1f}% (harga {price:.0f} vs kemarin {prev_close:.0f})",
            "description": "Harga hari ini minimal 5% lebih tinggi dari penutupan kemarin - "
                            "tanda ada dorongan beli yang kuat, bukan cuma naik tipis biasa.",
        },
        {
            "key": "bsjp_volume",
            "label": "Volume breakout (≥2x MA20 & ≥1x kemarin)",
            "passed": bool((today_volume >= 2 * vol_ma20) and (today_volume >= prev_volume)),
            "value": f"{today_volume:,.0f} (MA20: {vol_ma20:,.0f}, kemarin: {prev_volume:,.0f})",
            "description": "Volume transaksi hari ini minimal 2x rata-rata 20 hari DAN lebih "
                            "tinggi dari kemarin - tanda minat beli meledak, bukan cuma noise harian.",
        },
        {
            "key": "bsjp_ma5",
            "label": "Harga ≥ MA5",
            "passed": bool(price >= price_ma5),
            "value": f"{price:.0f} vs MA5 {price_ma5:.0f}",
            "description": "Harga sekarang di atas rata-rata 5 hari terakhir - konfirmasi "
                            "momentum jangka pendek masih condong naik.",
        },
        {
            "key": "bsjp_open",
            "label": "Harga ≥ Open (nggak turun dari open)",
            "passed": bool(price >= today_open),
            "value": f"{price:.0f} vs Open {today_open:.0f}",
            "description": "Harga penutupan nggak turun di bawah harga pembukaan hari ini - "
                            "tanda pembeli tetap mendominasi sepanjang sesi.",
        },
        {
            "key": "bsjp_value",
            "label": "Value transaksi > Rp5 miliar",
            "passed": bool(value_transaksi > 5_000_000_000),
            "value": f"Rp{value_transaksi:,.0f}",
            "description": "Total nilai transaksi (harga x volume) hari ini di atas Rp5 miliar - "
                            "filter biar sahamnya cukup likuid, bukan saham yang jarang ditransaksikan.",
        },
        {
            "key": "bsjp_gocap",
            "label": "Harga sebelumnya > 50 (bukan saham gocap)",
            "passed": bool(prev_close > 50),
            "value": f"{prev_close:.0f}",
            "description": "Harga saham di atas Rp50 - menghindari saham 'gocap' (harga terendah "
                            "IDX) yang pergerakannya sering nggak wajar/gampang dimanipulasi.",
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
            "key": "bpjs_ma5",
            "label": "Harga ≥ MA5",
            "passed": bool(price >= price_ma5),
            "value": f"{price:.0f} vs MA5 {price_ma5:.0f}",
            "description": "Harga sekarang di atas rata-rata 5 hari terakhir - konfirmasi "
                            "momentum jangka pendek masih condong naik.",
        },
        {
            "key": "bpjs_naik5",
            "label": "Naik ≥5% dari kemarin",
            "passed": bool(price >= 1.05 * prev_close),
            "value": f"{pct_change:+.1f}% (harga {price:.0f} vs kemarin {prev_close:.0f})",
            "description": "Harga hari ini minimal 5% lebih tinggi dari penutupan kemarin - "
                            "tanda ada dorongan beli yang kuat.",
        },
        {
            "key": "bpjs_open",
            "label": "Harga ≥ Open",
            "passed": bool(price >= today_open),
            "value": f"{price:.0f} vs Open {today_open:.0f}",
            "description": "Harga penutupan nggak turun di bawah harga pembukaan hari ini.",
        },
        {
            "key": "bpjs_volume",
            "label": "Volume ≥ 0.2x kemarin",
            "passed": bool(today_volume >= 0.2 * prev_volume),
            "value": f"{today_volume:,.0f} vs 0.2x kemarin ({0.2 * prev_volume:,.0f})",
            "description": "Volume hari ini minimal seperlima volume kemarin - kriteria yang "
                            "sengaja lebih longgar dari BSJP karena dicek pagi hari, sebelum "
                            "volume harian terbentuk penuh.",
        },
        {
            "key": "bpjs_value",
            "label": "Value transaksi > Rp5 miliar",
            "passed": bool(value_transaksi > 5_000_000_000),
            "value": f"Rp{value_transaksi:,.0f}",
            "description": "Total nilai transaksi hari ini di atas Rp5 miliar - filter likuiditas "
                            "minimum.",
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
    Screener 'Day Trade' — kombinasi sinyal buat trading intraday/harian.

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
            "key": "dt_volume",
            "label": "Volume ≥1.5x rata-rata 20 hari",
            "passed": bool(today_volume >= 1.5 * vol_ma20),
            "value": f"{today_volume:,.0f} vs 1.5x MA20 ({1.5 * vol_ma20:,.0f})",
            "description": "Volume hari ini minimal 1.5x rata-rata 20 hari terakhir - tanda "
                            "minat transaksi lagi aktif, penting buat day trade biar gampang "
                            "keluar-masuk posisi.",
        },
        {
            "key": "dt_macd",
            "label": "MACD histogram positif (momentum naik)",
            "passed": bool(macd_hist > 0),
            "value": f"{macd_hist:.2f}",
            "description": "Histogram MACD positif artinya garis MACD lagi di atas garis sinyal - "
                            "momentum jangka pendek condong naik.",
        },
        {
            "key": "dt_ma20",
            "label": "Harga di atas MA20",
            "passed": bool(price > ma20),
            "value": f"{price:.0f} vs MA20 {ma20:.0f}",
            "description": "Harga sekarang di atas rata-rata 20 hari - konfirmasi tren jangka "
                            "pendek masih naik.",
        },
        {
            "key": "dt_rsi",
            "label": "RSI antara 40-70 (momentum sehat)",
            "passed": bool(40 <= rsi <= 70),
            "value": f"RSI {rsi:.1f}",
            "description": "RSI di zona 40-70 - nggak lagi jenuh jual (di bawah 40) atau jenuh "
                            "beli ekstrem (di atas 70), momentum dianggap masih 'sehat' buat entry.",
        },
        {
            "key": "dt_value",
            "label": "Value transaksi > Rp1 miliar",
            "passed": bool(value_transaksi > 1_000_000_000),
            "value": f"Rp{value_transaksi:,.0f}",
            "description": "Filter likuiditas minimum - meski lebih longgar dari BSJP/BPJS, "
                            "day trade tetap butuh saham yang cukup aktif ditransaksikan.",
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

    CATATAN: BSJP, BPJS, Day Trade TIDAK dimasukkan ke sini. Itu bukan
    indikator yang ikut dijumlah ke skor, tapi SCREENER terpisah — lihat
    compute_screener_results().
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


def compute_signals(df: pd.DataFrame, selected_indicators: list = None) -> dict:
    """
    Wrapper di atas compute_all_indicators() — hitung skor & alasan
    berdasarkan indikator yang DIPILIH aja (kalau None, pakai SEMUA indikator).
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


def compute_screener_results(df: pd.DataFrame) -> dict:
    """
    Jalankan SEMUA screener (Day Trade, BSJP, BPJS) sekaligus buat 1 saham.
    Masing-masing independen - saham bisa lolos satu, dua, atau ketiganya.
    """
    results = {}
    for key, fn in [("day_trade", check_day_trade), ("bsjp", check_bsjp), ("bpjs", check_bpjs)]:
        try:
            results[key] = fn(df)
        except Exception as e:
            results[key] = {"passed": False, "evidence": [], "detail": None, "error": str(e)}
    return results


def compute_trade_levels(df: pd.DataFrame) -> dict:
    """
    Hitung level SUPPORT, STOP LOSS, TAKE PROFIT 1 & 2 buat 1 saham,
    berdasarkan level support/resistance historis + prinsip asymmetric
    bet (risk kecil, reward lebih besar).

    - Support = level support terdekat DI BAWAH harga sekarang
      (fallback: 5% di bawah harga kalau nggak ketemu level valid)
    - Stop Loss = sedikit di bawah support (buffer 2%)
    - Take Profit 1 = resistance terdekat DI ATAS harga sekarang
      (fallback: risk x1.5 dari harga sekarang)
    - Take Profit 2 = resistance berikutnya di atas TP1
      (fallback: risk x3 dari harga sekarang)
    """
    close = df["Close"]
    price = close.iloc[-1]
    sr = find_support_resistance(df)

    supports_below = sorted([s for s in sr["support"] if s < price], reverse=True)
    resistances_above = sorted([r for r in sr["resistance"] if r > price])

    support = supports_below[0] if supports_below else price * 0.95
    stop_loss = support * 0.98
    risk = max(price - stop_loss, price * 0.01)

    if len(resistances_above) >= 1:
        tp1 = resistances_above[0]
    else:
        tp1 = price + risk * 1.5

    if len(resistances_above) >= 2:
        tp2 = resistances_above[1]
    else:
        tp2 = max(tp1 + risk * 1.5, price + risk * 3)

    reward1 = tp1 - price
    reward2 = tp2 - price

    return {
        "price": round(float(price), 0),
        "support": round(float(support), 0),
        "stop_loss": round(float(stop_loss), 0),
        "take_profit_1": round(float(tp1), 0),
        "take_profit_2": round(float(tp2), 0),
        "risk_reward_1": round(float(reward1 / risk), 2) if risk > 0 else None,
        "risk_reward_2": round(float(reward2 / risk), 2) if risk > 0 else None,
    }


# Deskripsi tiap indikator dalam bahasa sederhana, dipakai di web app.
INDICATOR_DESCRIPTIONS = {
    "rsi_oversold": (
        "RSI (Relative Strength Index) ngukur seberapa 'jenuh jual' suatu saham. "
        "Kalau RSI di bawah 35, itu tanda harga udah turun cukup dalam dan secara "
        "statistik lebih rawan mantul naik - tapi bukan jaminan, bisa aja terus turun."
    ),
    "macd_cross": (
        "MACD Golden Cross terjadi ketika garis MACD memotong garis sinyal dari "
        "bawah ke atas - biasa dibaca sebagai awal momentum naik jangka pendek."
    ),
    "volume_spike": (
        "Volume Spike artinya volume transaksi hari ini jauh di atas rata-rata "
        "20 hari terakhir. Ini tanda ada minat beli/jual yang tiba-tiba membesar."
    ),
    "above_ma20": (
        "Harga di atas MA20 (rata-rata 20 hari) nunjukkin tren jangka pendek "
        "masih condong naik dibanding sebulan terakhir."
    ),
    "uptrend_ma": (
        "MA20 di atas MA50 nunjukkin tren jangka pendek lebih kuat dari tren "
        "jangka menengah - kombinasi ini sering dipakai buat konfirmasi arah tren."
    ),
    "bollinger_riding": (
        "Riding upper Bollinger Band artinya harga 'nempel' dan jalan di "
        "sepanjang garis atas band volatilitas - ini justru tanda tren KUAT, "
        "bukan tanda kemahalan/overbought seperti yang sering disalahpahami."
    ),
    "near_support": (
        "Harga sekarang lagi deket sama level support historis - level yang "
        "berkali-kali jadi 'lantai' harga di masa lalu."
    ),
    "vcp_pattern": (
        "VCP (Volatility Contraction Pattern) itu pola di mana tiap koreksi "
        "harga makin mengecil - tanda tekanan jual makin lemah. Sering muncul "
        "sebelum breakout menurut pendekatan trader Mark Minervini."
    ),
    "day_trade": (
        "Screener Day Trade nyari saham dengan kombinasi volume aktif, momentum "
        "MACD positif, tren MA20 naik, RSI di zona sehat, dan likuiditas cukup."
    ),
    "bsjp": (
        "BSJP (Beli Sore Jual Pagi) nyari saham yang baru breakout kencang sore "
        "hari dengan asumsi momentum berlanjut ke pembukaan besok pagi."
    ),
    "bpjs": (
        "BPJS (Beli Pagi Jual Sore) versi lebih longgar dari BSJP, dicek "
        "sebelum market buka buat nyari kandidat berpotensi lanjut naik."
    ),
}


# =========================================================================
# MODUL FUNDAMENTAL & BERITA (analisis berita sebagai penguat fundamental+teknikal)
# =========================================================================

ANTHROPIC_API_KEY = "ISI_API_KEY_ANTHROPIC_KAMU_KALAU_MAU_ANALISIS_AI"
ANTHROPIC_MODEL = "claude-sonnet-4-6"

NEWS_MAX_AGE_DAYS = 31

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
NEGATION_WORDS = ["tidak", "bukan", "belum", "tanpa", "gagal", "batal"]


def fetch_news_headlines(query: str, max_articles: int = 15, max_age_days: int = None) -> list:
    """
    Ambil judul + LINK + TANGGAL berita soal 'query' dari Google News RSS,
    difilter cuma yang berumur <= max_age_days (default: NEWS_MAX_AGE_DAYS).
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
    """Bikin link Google News yang bisa dibuka manual di browser buat verifikasi sendiri."""
    import urllib.parse
    return "https://news.google.com/search?q=" + urllib.parse.quote(query) + "&hl=id&gl=ID&ceid=ID:id"


def _keyword_sentiment_fallback(headlines: list) -> dict:
    """
    Fallback KALAU nggak ada API key AI: keyword matching yang cek NEGASI juga.
    Contoh: "tidak untung" -> nggak dihitung positif lagi.
    """
    pos_count = 0
    neg_count = 0
    matched_titles = []

    for h in headlines:
        text = h["title"].lower()

        for kw in POSITIVE_NEWS_KEYWORDS:
            if kw in text:
                idx = text.find(kw)
                before = text[:idx].split()[-3:]
                if any(neg in before for neg in NEGATION_WORDS):
                    neg_count += 1
                else:
                    pos_count += 1
                    matched_titles.append(h["title"])

        for kw in NEGATIVE_NEWS_KEYWORDS:
            if kw in text:
                idx = text.find(kw)
                before = text[:idx].split()[-3:]
                if any(neg in before for neg in NEGATION_WORDS):
                    pos_count += 1
                else:
                    neg_count += 1

    net_sentiment = pos_count - neg_count
    return {
        "triggered": net_sentiment > 0,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "summary": None,
        "matched_titles": matched_titles[:3],
    }


def _ai_news_analysis(ticker: str, headlines: list) -> dict:
    """
    Analisis berita pakai Claude API (opsional) - baca judul berita beneran,
    kasih penilaian relevansi ke fundamental & teknikal. Return None kalau
    API key belum diisi (fallback otomatis ke keyword matching).
    """
    if "ISI_" in ANTHROPIC_API_KEY or not ANTHROPIC_API_KEY:
        return None

    headlines_text = "\n".join(f"- {h['title']}" for h in headlines)
    prompt = f"""Kamu menganalisis berita terbaru (maks {NEWS_MAX_AGE_DAYS} hari terakhir) soal saham {ticker} di Bursa Efek Indonesia.

Judul-judul berita:
{headlines_text}

Analisis apakah berita-berita ini, secara keseluruhan, memperkuat atau memperlemah keyakinan untuk MEMBELI saham ini, dari sisi fundamental maupun teknikal.

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
    Analisis sentimen berita terbaru (maks NEWS_MAX_AGE_DAYS hari).
    Coba pakai AI (Claude) dulu kalau API key udah diisi, kalau nggak
    fallback ke keyword matching.
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
    Selalu menyertakan search_url biar kamu bisa cek manual sendiri.
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
    berbayar/proprietary.
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

    results.sort(key=lambda x: x["score"], reverse=True)
    candidates = results

    if candidates:
        msg_lines = ["*Screening Result*", ""]
        for c in candidates:
            msg_lines.append(
                f"*{c['ticker']}* — skor {c['score']} | lolos: {', '.join(c['lolos_screener'])}\n"
                f"Harga: {c['price']:.0f}\n"
                f"Alasan: {', '.join(c['reasons'])}\n"
            )
        message = "\n".join(msg_lines)
    else:
        message = "Screening selesai — tidak ada kandidat hari ini."

    print(message)
    send_telegram_message(message)


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
#    0 10,13,15 * * 1-5 /usr/bin/python3 /path/ke/stock_screener.py
#
# C) JADWALKAN OTOMATIS - Opsi 2: GitHub Actions (gratis, cloud)
#    Buat file .github/workflows/screener.yml dengan schedule (cron) yang
#    trigger `pip install -r requirements.txt && python stock_screener.py`.
#    Simpan TOKEN & CHAT_ID sebagai GitHub Secrets, jangan hardcode di kode
#    kalau repo public.
