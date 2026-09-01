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
import os
import json
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
# Diperkecil dari 50 -> 15 dan delay dinaikin, biar lebih "sopan" ke Yahoo
# Finance dan ngurangin resiko kena rate limit (401 Invalid Crumb) yang
# sering muncul kalau nge-hit banyak ticker sekaligus dari IP shared hosting
# kayak Streamlit Cloud. Scan tetap MENYELURUH ke semua ticker di watchlist -
# cuma dipecah jadi batch lebih kecil & lebih pelan, bukan dikurangin cakupannya.
BATCH_SIZE = 15
BATCH_DELAY_SECONDS = 3.0  # jeda antar batch
BATCH_RETRY_DELAY_SECONDS = 8.0  # jeda ekstra sebelum retry kalau satu batch gagal total/kosong semua
BATCH_MAX_RETRIES = 1  # berapa kali retry per batch kalau kena gagal total/kosong (bukan per-ticker gagal biasa)


# =========================================================================
# MODUL CACHE SAHAM DELISTED
# =========================================================================
# Tujuan: saham yang udah kebukti nggak ada datanya di Yahoo Finance (delisted/
# suspend permanen) DITANDAI, disimpan permanen di file JSON, dan otomatis
# DILEWATIN (nggak di-hit ke yfinance lagi) di screening-screening berikutnya.
# Ini murni PENAMBAHAN di atas fetch_data()/fetch_batch_data() yang udah ada -
# alur & hasil normalnya nggak berubah sama sekali buat ticker yang sehat.
#
# Kenapa nggak langsung ditandai delisted sekali gagal?
# Karena gagal ambil data bisa juga gara-gara rate limit / gangguan sementara
# dari Yahoo Finance (lihat pesan "Invalid Crumb" / HTTP 401 di log kamu),
# BUKAN berarti sahamnya beneran delisted. Makanya baru ditandai PERMANEN
# setelah gagal DELISTED_MISS_THRESHOLD kali berturut-turut (lintas beberapa
# kali run screening), biar nggak salah tandai.
# =========================================================================

DELISTED_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "delisted_tickers.json")
DELISTED_MISS_THRESHOLD = 3  # gagal berturut-turut sebanyak ini baru ditandai delisted otomatis


def _load_delisted_store() -> dict:
    if not os.path.exists(DELISTED_CACHE_FILE):
        return {"delisted": {}, "miss_streak": {}}
    try:
        with open(DELISTED_CACHE_FILE, "r") as f:
            data = json.load(f)
        data.setdefault("delisted", {})
        data.setdefault("miss_streak", {})
        return data
    except Exception as e:
        print(f"[WARN] Gagal baca cache delisted ({DELISTED_CACHE_FILE}), mulai dari kosong: {e}")
        return {"delisted": {}, "miss_streak": {}}


def _save_delisted_store(data: dict):
    try:
        os.makedirs(os.path.dirname(DELISTED_CACHE_FILE), exist_ok=True)
        with open(DELISTED_CACHE_FILE, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] Gagal simpan cache delisted: {e}")


def get_delisted_tickers() -> dict:
    """Balikin dict {ticker: {tanggal_ditandai, alasan}} - semua saham yang udah ditandai delisted."""
    return _load_delisted_store()["delisted"]


def is_ticker_delisted(ticker: str) -> bool:
    return ticker in _load_delisted_store()["delisted"]


def mark_ticker_delisted(ticker: str, reason: str = "Ditandai manual"):
    """Tandai satu ticker sebagai delisted (misal manual dari UI web app)."""
    data = _load_delisted_store()
    data["delisted"][ticker] = {
        "tanggal_ditandai": datetime.now().strftime("%Y-%m-%d"),
        "alasan": reason,
    }
    data["miss_streak"].pop(ticker, None)
    _save_delisted_store(data)


def unmark_ticker_delisted(ticker: str):
    """Hapus tanda delisted (misal ternyata salah tandai) - dicek lagi mulai screening berikutnya."""
    data = _load_delisted_store()
    data["delisted"].pop(ticker, None)
    data["miss_streak"].pop(ticker, None)
    _save_delisted_store(data)


def _record_fetch_result(ticker: str, success: bool):
    """Dipanggil tiap kali fetch_data()/fetch_batch_data() selesai proses SATU ticker.
    Nge-track gagal berturut-turut, dan auto-tandai delisted kalau udah kelewat threshold."""
    data = _load_delisted_store()
    if success:
        if ticker in data["miss_streak"]:
            data["miss_streak"].pop(ticker, None)
            _save_delisted_store(data)
        return

    streak = data["miss_streak"].get(ticker, 0) + 1
    if streak >= DELISTED_MISS_THRESHOLD and ticker not in data["delisted"]:
        data["delisted"][ticker] = {
            "tanggal_ditandai": datetime.now().strftime("%Y-%m-%d"),
            "alasan": f"Auto: gagal ambil data {streak}x berturut-turut (kemungkinan delisted/suspend)",
        }
        data["miss_streak"].pop(ticker, None)
        print(f"[INFO] {ticker} ditandai DELISTED otomatis setelah {streak}x gagal berturut-turut.")
    else:
        data["miss_streak"][ticker] = streak
    _save_delisted_store(data)


# =========================================================================
# MODUL AMBIL SEMUA TICKER IDX (opsional, buat scan semua saham)
# =========================================================================

IDX_ALL_TICKERS_URL = "https://raw.githubusercontent.com/wildangunawan/Dataset-Saham-IDX/master/List%20Emiten/all.csv"


def check_yfinance_backend() -> dict:
    """
    Cek apakah yfinance beneran lagi pakai curl_cffi (TLS/browser
    impersonation) atau fallback ke requests biasa (lebih gampang kena
    rate-limit/block dari Yahoo Finance). Murni buat diagnostik, dipanggil
    sekali pas startup app - nggak ngubah cara fetch_data()/fetch_batch_data()
    kerja, karena yfinance versi ini (>=1.x) OTOMATIS pakai curl_cffi sendiri
    kalau library-nya kedetect terinstall - kita nggak perlu setting session
    manual kayak versi lama.
    """
    try:
        from yfinance import _http as _yf_http
        active = bool(getattr(_yf_http, "HAS_CURL_CFFI", False))
        return {
            "curl_cffi_aktif": active,
            "keterangan": (
                "yfinance pakai curl_cffi dengan browser TLS impersonation (Chrome) - "
                "konfigurasi paling tahan rate-limit."
                if active else
                "yfinance FALLBACK ke requests biasa (curl_cffi nggak kedetect) - lebih "
                "rawan kena rate-limit/block dari Yahoo Finance. Cek curl_cffi ada di "
                "requirements.txt dan ke-install dengan benar."
            ),
        }
    except Exception as e:
        return {
            "curl_cffi_aktif": None,
            "keterangan": f"Nggak bisa dicek (kemungkinan versi yfinance beda struktur internal): {e}",
        }


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
    if is_ticker_delisted(ticker):
        # Udah ditandai delisted sebelumnya - nggak usah buang request ke yfinance lagi.
        return pd.DataFrame()

    df = yf.download(ticker, period=LOOKBACK_DAYS, interval="1d", progress=False)
    if df.empty or len(df) < 30:
        _record_fetch_result(ticker, success=False)
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    _record_fetch_result(ticker, success=True)
    return df


def _download_one_batch(batch: list):
    """Satu kali percobaan download buat satu batch ticker. Dipisah dari
    fetch_batch_data() biar bisa dipanggil ulang (retry) kalau gagal total."""
    if len(batch) == 1:
        raw = yf.download(batch[0], period=LOOKBACK_DAYS, interval="1d", progress=False)
        return {batch[0]: raw}
    raw = yf.download(
        batch, period=LOOKBACK_DAYS, interval="1d",
        group_by="ticker", progress=False, threads=True,
    )
    return {t: (raw[t] if t in raw.columns.get_level_values(0) else pd.DataFrame())
            for t in batch}


def fetch_batch_data(tickers: list, batch_size: int = None, delay: float = None) -> dict:
    """
    Ambil data OHLCV buat BANYAK ticker sekaligus, dipecah jadi batch kecil.
    Jauh lebih cepat daripada fetch_data() satu-satu, dan lebih aman dari
    resiko rate-limit karena requestnya nggak sekaligus semua.

    Scan tetap MENYELURUH ke semua ticker yang dikasih (kecuali yang udah
    ditandai delisted) - batch kecil & retry di sini cuma soal SEBERAPA HATI-HATI
    cara ambilnya, bukan ngurangin cakupan saham yang di-screening.
    """
    batch_size = batch_size or BATCH_SIZE
    delay = delay if delay is not None else BATCH_DELAY_SECONDS

    # Buang dulu ticker yang udah ditandai delisted - nggak usah dikirim ke
    # yfinance sama sekali, biar kuota request nggak kebuang percuma.
    delisted_now = get_delisted_tickers()
    tickers_to_fetch = [t for t in tickers if t not in delisted_now]
    skipped_delisted = len(tickers) - len(tickers_to_fetch)
    if skipped_delisted:
        print(f"[INFO] {skipped_delisted} ticker dilewati (sudah ditandai delisted): "
              f"{', '.join(sorted(set(tickers) & set(delisted_now.keys())))}")

    result = {}
    total_batches = (len(tickers_to_fetch) + batch_size - 1) // batch_size

    for b in range(total_batches):
        batch = tickers_to_fetch[b * batch_size: (b + 1) * batch_size]
        print(f"[INFO] Batch {b + 1}/{total_batches} ({len(batch)} ticker)...")

        candidates = {}
        for attempt in range(BATCH_MAX_RETRIES + 1):
            try:
                candidates = _download_one_batch(batch)
            except Exception as e:
                print(f"[ERROR] Batch {b + 1} gagal total (percobaan {attempt + 1}): {e}")
                candidates = {}

            # Batch dianggap "gagal total" kalau SEMUA ticker di batch ini
            # kosong - biasanya tanda kena rate-limit sesaat (bukan delisted
            # beneran), jadi layak di-retry sekali dengan jeda lebih lama
            # SEBELUM dianggap gagal permanen buat masing-masing ticker.
            batch_all_empty = candidates and all(
                (df is None or df.dropna(how="all").empty) for df in candidates.values()
            )
            if candidates and not batch_all_empty:
                break
            if attempt < BATCH_MAX_RETRIES:
                print(f"[INFO] Batch {b + 1} kosong semua, retry dalam {BATCH_RETRY_DELAY_SECONDS}s...")
                time.sleep(BATCH_RETRY_DELAY_SECONDS)

        for ticker, df in candidates.items():
            df = df.dropna(how="all")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and len(df) >= 30:
                result[ticker] = df
                _record_fetch_result(ticker, success=True)
            else:
                _record_fetch_result(ticker, success=False)

        if not candidates:
            # Batch gagal total (exception terus di semua percobaan) - tetap
            # catat miss buat tiap ticker di batch ini biar streak delisted jalan.
            for ticker in batch:
                _record_fetch_result(ticker, success=False)

        if b < total_batches - 1:
            time.sleep(delay)

    skip_note = f" ({skipped_delisted} dilewati krn delisted)" if skipped_delisted else ""
    print(f"[INFO] Selesai: {len(result)}/{len(tickers_to_fetch)} ticker berhasil diambil datanya{skip_note}.")
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


def check_ema_riding(df: pd.DataFrame, span: int = 9, days_check: int = 3) -> dict:
    """
    EMA Riding (Video 13) - beda dari MA biasa, EMA lebih responsif ke harga
    terbaru. Harga yang "nempel & jalan" di atas EMA9 selama beberapa hari
    dianggap tanda tren kuat masih berlanjut.
    """
    close = df["Close"]
    ema = close.ewm(span=span, adjust=False).mean()
    recent_close = close.iloc[-days_check:]
    recent_ema = ema.iloc[-days_check:]
    riding = bool((recent_close >= recent_ema).all())
    return {
        "triggered": riding,
        "value": f"Harga {close.iloc[-1]:.0f} vs EMA{span} {ema.iloc[-1]:.0f} "
                  f"({days_check} hari terakhir di atas EMA)",
    }


def check_stochrsi_at_support(df: pd.DataFrame) -> dict:
    """
    Kombinasi Stochastic RSI oversold + harga dekat support (Video 24).
    Sinyal dianggap lebih kuat kalau dua-duanya terjadi bersamaan.
    """
    close = df["Close"]
    price = close.iloc[-1]
    stoch_rsi = ta.momentum.StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
    stoch_val = stoch_rsi.stochrsi_k().iloc[-1] * 100

    sr = find_support_resistance(df)
    near_support = check_near_support(price, sr["support"])
    oversold = bool(stoch_val < 20)

    triggered = oversold and near_support
    return {
        "triggered": triggered,
        "value": f"StochRSI {stoch_val:.1f} ({'oversold' if oversold else 'normal'}), "
                  f"{'dekat' if near_support else 'jauh dari'} support",
    }


def check_sr_role_reversal(df: pd.DataFrame, lookback: int = 40, tolerance_pct: float = 0.02) -> dict:
    """
    Support/Resistance Role Reversal (Video 28, 30) - cek apakah level yang
    dulunya resistance sekarang udah ditembus & harga sekarang deket situ
    (jadi dianggap support baru).
    """
    if len(df) < lookback:
        return {"triggered": False, "value": "Data historis kurang buat cek role reversal"}

    old_df = df.iloc[:-20]
    recent_price = df["Close"].iloc[-1]

    try:
        old_sr = find_support_resistance(old_df)
    except Exception:
        return {"triggered": False, "value": "Gagal hitung level historis"}

    reversal_levels = []
    for r in old_sr["resistance"]:
        if r < recent_price and abs(recent_price - r) / r <= tolerance_pct * 3:
            reversal_levels.append(r)

    triggered = len(reversal_levels) > 0
    detail_val = (
        f"Level {reversal_levels[0]:.0f} dulu resistance, sekarang ditembus & "
        f"jadi support baru (harga sekarang {recent_price:.0f})"
        if triggered else "Belum terdeteksi pola role reversal di data historis"
    )
    return {"triggered": triggered, "value": detail_val}


def check_range_sideways(df: pd.DataFrame, window: int = 20, max_range_pct: float = 0.08) -> dict:
    """
    Range Trading (Video 4) - deteksi saham yang lagi bergerak sideways,
    cocok buat strategi beli-di-support jual-di-resistance dalam range yang sama.
    """
    recent = df.tail(window)
    high = recent["High"].max()
    low = recent["Low"].min()
    avg = recent["Close"].mean()
    range_pct = (high - low) / avg if avg > 0 else 999

    triggered = bool(range_pct <= max_range_pct)
    return {
        "triggered": triggered,
        "value": f"Range {range_pct*100:.1f}% dari harga rata-rata ({window} hari) - "
                  f"{'sideways/ranging' if triggered else 'trending, bukan sideways'}",
    }


def suggest_trailing_stop(df: pd.DataFrame, trail_pct: float = 5.0) -> dict:
    """
    Trailing Stop (Video 20) - hitung level trailing stop SAAT INI berdasarkan
    harga tertinggi 20 hari terakhir sebagai proxy "puncak". BUKAN pelacakan
    posisi beneran (script ini stateless) - anggap ini kalkulator bantu.
    """
    recent_high = df["Close"].tail(20).max()
    current_price = df["Close"].iloc[-1]
    trailing_stop_level = recent_high * (1 - trail_pct / 100)

    return {
        "recent_high": round(float(recent_high), 0),
        "current_price": round(float(current_price), 0),
        "trailing_stop_level": round(float(trailing_stop_level), 0),
        "trail_pct": trail_pct,
        "note": f"Puncak 20 hari terakhir: {recent_high:.0f}. Trailing stop {trail_pct}% "
                f"ada di {trailing_stop_level:.0f}. Update ulang tiap ada rekor tertinggi baru.",
    }


def check_leading_lagging(ticker: str, batch_data: dict, min_leader_gain_pct: float = 5.0) -> dict:
    """
    Leading-Lagging antar saham segrup (Video 2, 6) - cek apakah ticker ini
    berada dalam satu grup konglomerat dengan saham lain yang HARI INI udah
    naik signifikan (leader), sementara ticker ini sendiri belum banyak
    bergerak - berpotensi jadi 'penyusul' (laggard).
    """
    group_name = None
    group_members = []
    for gname, members in KONGLOMERAT_GROUPS.items():
        if ticker in members:
            group_name = gname
            group_members = members
            break

    if group_name is None:
        return {"triggered": False, "value": "Saham ini nggak termasuk grup konglomerat yang terdaftar"}

    leaders = []
    for member in group_members:
        if member == ticker or member not in batch_data:
            continue
        mdf = batch_data[member]
        if len(mdf) < 2:
            continue
        pct = (mdf["Close"].iloc[-1] / mdf["Close"].iloc[-2] - 1) * 100
        if pct >= min_leader_gain_pct:
            leaders.append((member, pct))

    self_pct = 0.0
    if ticker in batch_data and len(batch_data[ticker]) >= 2:
        self_df = batch_data[ticker]
        self_pct = (self_df["Close"].iloc[-1] / self_df["Close"].iloc[-2] - 1) * 100

    triggered = len(leaders) > 0 and self_pct < min_leader_gain_pct
    if triggered:
        leader_text = ", ".join(f"{m} (+{p:.1f}%)" for m, p in leaders)
        detail = f"Grup '{group_name}': {leader_text} udah naik duluan, {ticker} baru {self_pct:+.1f}% - berpotensi menyusul"
    else:
        detail = f"Grup '{group_name}': belum ada leader signifikan hari ini, atau {ticker} sendiri udah ikut naik"

    return {"triggered": triggered, "value": detail, "group": group_name, "leaders": leaders}


# =========================================================================
# MODUL KONTEKS MAKRO (Video 29)
# =========================================================================

MACRO_INDICES = {
    "^DJI": "Dow Jones", "^GSPC": "S&P 500", "^IXIC": "Nasdaq",
    "^VIX": "VIX (indeks volatilitas)", "^N225": "Nikkei 225", "^KS11": "KOSPI",
}


def fetch_macro_context() -> dict:
    """
    Cek kondisi bursa global sebelum screening IHSG - malam (bursa Amerika)
    dan pagi (Nikkei & KOSPI, buka ~2 jam lebih awal, dianggap lebih relate
    karena sama-sama regional Asia).
    """
    result = {}
    try:
        raw = yf.download(list(MACRO_INDICES.keys()), period="5d", interval="1d",
                           group_by="ticker", progress=False, threads=True)
    except Exception as e:
        print(f"[WARN] Gagal ambil data makro: {e}")
        return {"indices": {}, "sentiment_score": 0, "summary": "Data makro tidak tersedia"}

    positive_count = 0
    negative_count = 0
    for symbol, name in MACRO_INDICES.items():
        try:
            idf = raw[symbol] if len(MACRO_INDICES) > 1 else raw
            idf = idf.dropna(how="all")
            if len(idf) < 2:
                continue
            pct = (idf["Close"].iloc[-1] / idf["Close"].iloc[-2] - 1) * 100
            result[symbol] = {"name": name, "pct_change": round(float(pct), 2)}
            if symbol == "^VIX":
                if pct > 5:
                    negative_count += 1
                elif pct < -5:
                    positive_count += 1
            else:
                if pct > 0.3:
                    positive_count += 1
                elif pct < -0.3:
                    negative_count += 1
        except Exception:
            continue

    sentiment_score = 1 if positive_count > negative_count else (-1 if negative_count > positive_count else 0)
    summary = {1: "Kondisi makro cenderung positif", -1: "Kondisi makro cenderung negatif",
               0: "Kondisi makro netral/campuran"}[sentiment_score]

    return {"indices": result, "sentiment_score": sentiment_score, "summary": summary}



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
    ema_result = check_ema_riding(df)
    stochrsi_result = check_stochrsi_at_support(df)
    role_reversal_result = check_sr_role_reversal(df)
    range_result = check_range_sideways(df)

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
        "ema_riding": {
            "triggered": ema_result["triggered"],
            "label": "EMA9 Riding",
            "detail": ema_result["value"] if ema_result["triggered"] else None,
        },
        "stochrsi_support": {
            "triggered": stochrsi_result["triggered"],
            "label": "StochRSI Oversold di Support",
            "detail": stochrsi_result["value"] if stochrsi_result["triggered"] else None,
        },
        "sr_role_reversal": {
            "triggered": role_reversal_result["triggered"],
            "label": "Support/Resistance Role Reversal",
            "detail": role_reversal_result["value"] if role_reversal_result["triggered"] else None,
        },
        "range_sideways": {
            "triggered": range_result["triggered"],
            "label": "Sedang Sideways/Ranging",
            "detail": range_result["value"] if range_result["triggered"] else None,
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


def compute_position_sizing(modal_total: float, risk_pct: float, trade_levels: dict) -> dict:
    """
    Position sizing pakai aturan risiko-per-trade (mis. 2% rule) - hitung
    berapa lot maksimal yang boleh dibeli berdasarkan modal & toleransi
    risiko, plus strategi entry bertahap 2-3-5 (masuk sedikit dulu,
    tambah kalau konfirmasi, sisa jadi cadangan).

    modal_total: total modal trading (Rp)
    risk_pct: berapa persen dari modal yang rela di-resiko-kan PER TRADE
              (bukan per saham beli habis) - lazimnya 1-2%.
    trade_levels: hasil dari compute_trade_levels(), butuh "price" & "stop_loss"
    """
    price = trade_levels["price"]
    stop_loss = trade_levels["stop_loss"]
    jarak_cutloss_pct = (price - stop_loss) / price * 100 if price > 0 else 0

    risiko_rp = modal_total * (risk_pct / 100)
    max_position_rp = risiko_rp / (jarak_cutloss_pct / 100) if jarak_cutloss_pct > 0 else 0
    # 1 lot = 100 lembar saham di IDX
    max_lot = int(max_position_rp / (price * 100)) if price > 0 else 0
    max_lot = max(max_lot, 0)

    plan_2_3_5 = {
        "entry_awal_20pct": {"lot": max(round(max_lot * 0.2), 1) if max_lot > 0 else 0, "porsi": "20%"},
        "konfirmasi_30pct": {"lot": max(round(max_lot * 0.3), 0) if max_lot > 0 else 0, "porsi": "30%"},
        "cadangan_50pct":   {"lot": max(round(max_lot * 0.5), 0) if max_lot > 0 else 0, "porsi": "50%"},
    }

    actual_position_rp = max_lot * price * 100
    return {
        "modal_total": modal_total,
        "risk_pct": risk_pct,
        "jarak_cutloss_pct": round(float(jarak_cutloss_pct), 2),
        "risiko_rp": round(float(risiko_rp), 0),
        "max_position_rp": round(float(max_position_rp), 0),
        "max_lot": max_lot,
        "actual_position_rp": round(float(actual_position_rp), 0),
        "plan_2_3_5": plan_2_3_5,
        "pct_of_modal": round(float(actual_position_rp / modal_total * 100), 1) if modal_total else 0,
        "too_risky": max_lot == 0,
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
    "ema_riding": (
        "EMA (Exponential Moving Average) mirip MA biasa tapi lebih responsif ke "
        "harga terbaru. Harga yang 'nempel & jalan' di atas EMA9 beberapa hari "
        "berturut-turut nunjukkin tren jangka pendek yang sedang kuat."
    ),
    "stochrsi_support": (
        "Kombinasi dua sinyal sekaligus: Stochastic RSI oversold (di bawah 20) "
        "DAN harga lagi deket level support historis. Dua sinyal ini muncul "
        "bersamaan dianggap lebih meyakinkan daripada masing-masing sendirian."
    ),
    "sr_role_reversal": (
        "Level yang dulunya jadi resistance, setelah ditembus naik, berubah "
        "fungsi jadi support baru. Pola klasik buat konfirmasi breakout yang "
        "beneran kuat."
    ),
    "range_sideways": (
        "Saham lagi bergerak dalam rentang harga sempit (nggak tren naik/turun "
        "tajam) - cocok buat strategi beli di area bawah range, jual di area "
        "atasnya."
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

# Sama kayak GOAPI_API_KEY di bawah - isi lewat Streamlit Secrets
# (GOAPI_API_KEY / ANTHROPIC_API_KEY), jangan ditulis langsung di sini.
try:
    import streamlit as _st
    ANTHROPIC_API_KEY = _st.secrets.get("ANTHROPIC_API_KEY", "ISI_API_KEY_ANTHROPIC_KAMU_KALAU_MAU_ANALISIS_AI")
except Exception:
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


def compute_fundamental_health_score(ticker: str) -> dict:
    """
    Skor kesehatan fundamental SUNGGUHAN - beda dari compute_fundamental_score()
    di atas yang isinya cuma sentimen berita. Ini pakai rasio keuangan asli:
    PER, PBV, DER, ROA, dan profitabilitas net income - ditarik dari
    yfinance .info (gratis, sumber sama yang udah dipakai buat harga).

    Melengkapi pilar ke-4 (Fundamental) yang sebelumnya kosong di screener
    ini - sebelumnya cuma ada 3 pilar (Teknikal, Money Flow, Sentimen).
    """
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        return {"available": False, "error": str(e), "score": 0, "max_score": 5, "checks": {}}

    if not info or info.get("regularMarketPrice") is None:
        return {"available": False, "error": "Data fundamental tidak ditemukan", "score": 0, "max_score": 5, "checks": {}}

    per = info.get("trailingPE")
    pbv = info.get("priceToBook")
    der = info.get("debtToEquity")
    roa = info.get("returnOnAssets")
    net_income = info.get("netIncomeToCommon")

    checks = {
        "profitable": {
            "passed": net_income is not None and net_income > 0,
            "label": "Perusahaan profit (net income positif)",
            "value": f"Rp{net_income:,.0f}" if net_income is not None else "Data tidak tersedia",
        },
        "per_wajar": {
            "passed": per is not None and 0 < per <= 20,
            "label": "PER wajar (0-20x, hindari yang kemahalan)",
            "value": f"{per:.1f}x" if per is not None else "Data tidak tersedia",
        },
        "pbv_sehat": {
            "passed": pbv is not None and pbv > 0.5,
            "label": "PBV sehat (>0.5x, PBV kelewat rendah bisa jadi red flag)",
            "value": f"{pbv:.2f}x" if pbv is not None else "Data tidak tersedia",
        },
        "der_rendah": {
            "passed": der is not None and der < 100,
            "label": "DER rendah (<100%, utang nggak kebesaran vs ekuitas)",
            "value": f"{der:.1f}%" if der is not None else "Data tidak tersedia",
        },
        "roa_positif": {
            "passed": roa is not None and roa > 0,
            "label": "ROA positif (aset perusahaan menghasilkan laba)",
            "value": f"{roa*100:.1f}%" if roa is not None else "Data tidak tersedia",
        },
    }
    score = sum(1 for c in checks.values() if c["passed"])

    return {
        "available": True,
        "score": score,
        "max_score": len(checks),
        "checks": checks,
        "per": per, "pbv": pbv, "der": der, "roa": roa, "net_income": net_income,
        "summary": f"{score}/{len(checks)} kriteria fundamental terpenuhi",
    }

def compute_dcf_fair_value(ticker: str, growth1: float = 0.10, growth2: float = 0.05,
                            discount: float = 0.10, terminal: float = 0.03) -> dict:
    """
    DCF (Discounted Cash Flow) 2-tahap sederhana - struktur & urutan
    perhitungannya sama kayak kalkulator DCF manual: FCF sekarang ->
    proyeksi growth tahap 1 (5 tahun) -> growth tahap 2 (5 tahun
    berikutnya) -> terminal value -> present value semuanya -> kurangi
    net debt -> bagi jumlah saham beredar -> harga wajar per saham.

    Default growth1=10%, growth2=5%, discount=10%, terminal=3% - ini
    ASUMSI, bukan angka pasti. Ganti sesuai keyakinan kamu sendiri soal
    prospek perusahaannya - hasil DCF sangat sensitif ke angka-angka ini.
    """
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        return {"available": False, "error": str(e)}

    fcf = info.get("freeCashflow")
    total_debt = info.get("totalDebt") or 0
    total_cash = info.get("totalCash") or 0
    net_debt = total_debt - total_cash
    shares = info.get("sharesOutstanding")
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")

    if not fcf or not shares:
        return {
            "available": False,
            "error": "Data Free Cash Flow atau jumlah saham beredar nggak tersedia dari Yahoo "
                     "Finance buat saham ini - DCF nggak bisa dihitung otomatis.",
        }

    pv_total = 0.0
    cf = fcf
    for yr in range(1, 11):
        g = growth1 if yr <= 5 else growth2
        cf = cf * (1 + g)
        pv_total += cf / (1 + discount) ** yr

    if discount <= terminal:
        return {"available": False, "error": "Discount rate harus lebih besar dari terminal growth rate."}

    terminal_value = (cf * (1 + terminal)) / (discount - terminal)
    pv_terminal = terminal_value / (1 + discount) ** 10
    equity_value = pv_total + pv_terminal - net_debt
    fair_value = equity_value / shares

    mos_pct = ((fair_value - current_price) / fair_value * 100) if (fair_value and current_price) else None
    harga_beli_ideal = fair_value * (1 - 0.30) if fair_value else None  # target MOS 30%, umum dipakai

    return {
        "available": True,
        "fcf_now": fcf,
        "net_debt": net_debt,
        "shares_outstanding": shares,
        "fair_value": round(float(fair_value), 0) if fair_value else None,
        "current_price": current_price,
        "mos_pct": round(float(mos_pct), 1) if mos_pct is not None else None,
        "harga_beli_ideal_mos30": round(float(harga_beli_ideal), 0) if harga_beli_ideal else None,
        "assumptions": {"growth1": growth1, "growth2": growth2, "discount": discount, "terminal": terminal},
        "note": (
            "DCF sensitif banget ke asumsi growth rate & discount rate - anggap ini SALAH SATU "
            "sudut pandang valuasi, bukan angka pasti. Selalu bandingkan juga sama PER/PBV dan "
            "kondisi bisnis riil perusahaannya."
        ),
    }



    """
    Placeholder terpisah buat sentimen sosial media (beda dari berita resmi).
    Belum diimplementasi - butuh data platform sosial media yang biasanya
    berbayar/proprietary.
    """
    return {"score": 0, "reasons": []}


# =========================================================================
# MODUL BANDARMOLOGY (GoAPI.IO - broker summary real-time)
# =========================================================================
#
# CATATAN JUJUR soal cakupan modul ini:
# GoAPI.IO menyediakan data BROKER SUMMARY (ringkasan net-buy/sell per
# broker per hari), TAPI TIDAK menyediakan data orderbook level-2 (bid/offer
# per level harga) atau data tick-by-tick. Jadi modul ini BISA:
#   - Broker Summary / Aksi Broker (Video 3, 16, 27)
#   - Buy-the-dip saat broker akumulasi crash (Video 27)
#   - Smart Money vs Retail (Video 15)
#   - Broker terkait tokoh/insider (Video 21, 22)
# Modul ini TIDAK BISA (masih butuh data proprietary lain):
#   - Bid/Offer Imbalance, Lot/Frequency Ratio, Timestamp Clustering,
#     Order-flow shift buy/sell
#
# PENTING - SOAL KUOTA API:
# Tiap panggilan fungsi di modul ini = 1 request ke GoAPI. Kalau dipakai di
# mode "Screening Massal" buat puluhan saham sekaligus, kuota API kamu bisa
# cepat habis (apalagi kalau masih paket Free Trial). Makanya modul ini
# HARUS diaktifkan manual lewat toggle, defaultnya OFF.

# Isi API key GoAPI kamu di STREAMLIT SECRETS (panel web Streamlit Cloud),
# BUKAN ditulis langsung di baris ini - soalnya file ini ada di GitHub,
# kalau repo-nya public siapa pun bisa lihat key-nya kalau ditulis di sini.
# Cara isi: buka app kamu di Streamlit Cloud > titik tiga > Settings >
# Secrets, tambahkan baris:
#   GOAPI_API_KEY = "key_asli_kamu_dari_goapi.io"
# Kalau belum di-set di Secrets, fallback ke placeholder di bawah (otomatis
# kedeteksi sebagai "belum dikonfigurasi" oleh app, fitur broker summary
# nonaktif tapi app tetap jalan normal).
try:
    import streamlit as _st
    GOAPI_API_KEY = _st.secrets.get("GOAPI_API_KEY", "ISI_API_KEY_GOAPI_KAMU")
except Exception:
    # Modul ini dijalankan di luar Streamlit (mis. script/testing biasa) -
    # nggak ada st.secrets, fallback ke placeholder.
    GOAPI_API_KEY = "ISI_API_KEY_GOAPI_KAMU"
GOAPI_BASE_URL = "https://api.goapi.io/stock/idx"

# Referensi klasifikasi broker (dari riset manual, bisa berubah - cek ulang berkala)
BROKER_SMART_MONEY = {
    "AK": "UBS Sekuritas Indonesia",
    "BK": "J.P. Morgan Sekuritas Indonesia",
    "BB": "Verdhana Sekuritas Indonesia",
    "YU": "CGS International Sekuritas Indonesia",
    "ZP": "Maybank Sekuritas Indonesia",
    "RX": "Macquarie Sekuritas Indonesia",
}
BROKER_RETAIL = {
    "XC": "Ajaib Sekuritas Asia",
    "XL": "Stockbit Sekuritas Digital",
    "YP": "Mirae Asset Sekuritas Indonesia",
    "PD": "Indo Premier Sekuritas",
    "CP": "KB Valbury Sekuritas",
}
BROKER_INSIDER_MAP = {
    "DX": "Prajogo Pangestu",
    "HP": "Prajogo Pangestu",
    "MG": "Aguan (Sugianto Kusuma)",
    "RF": "Aguan (Sugianto Kusuma)",
    "LG": "Boy Thohir (Garibaldi Thohir)",
}

# Penjelasan umum indikator Broker Summary - ditampilkan di web app biar
# jelas indikator ini artinya apa & datanya dari mana (bukan cuma angka
# tanpa konteks, biar kuota GoAPI yang dibayar nggak sia-sia).
BROKER_SUMMARY_DESCRIPTION = (
    "Broker Summary (Bandarmology) merekap transaksi net-buy/net-sell tiap "
    "sekuritas (kode broker) buat 1 saham di 1 hari tertentu, datanya REAL "
    "dari GoAPI.IO (bukan estimasi). Net-buy = broker itu beli lebih banyak "
    "daripada jual hari itu (net positif), net-sell = sebaliknya (net negatif). "
    "Broker di kelompok 'Smart Money' (mis. UBS, JP Morgan, Verdhana, CGS, "
    "Maybank, Macquarie) sering dipakai institusi/asing buat masuk saham "
    "secara terencana, sementara broker 'Retail' (mis. Ajaib, Stockbit, "
    "Mirae, Indo Premier, KB Valbury) didominasi transaksi investor ritel. "
    "Indikator ini TRIGGERED (dianggap sinyal akumulasi) kalau net-buy "
    "TERBESAR hari itu datang dari broker smart money. Ini bukan jaminan "
    "harga naik - anggap sebagai konteks tambahan siapa yang lagi aktif "
    "beli/jual, bukan sinyal berdiri sendiri."
)


def fetch_goapi(endpoint: str, params: dict = None) -> dict:
    """
    Helper generic buat manggil REST API GoAPI.IO. Return None kalau API
    key belum diisi atau request gagal (nggak bikin program crash).
    """
    if "ISI_" in GOAPI_API_KEY or not GOAPI_API_KEY:
        return None
    try:
        resp = requests.get(
            GOAPI_BASE_URL + endpoint,
            headers={"X-API-KEY": GOAPI_API_KEY},
            params=params or {},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[WARN] GoAPI request gagal ({endpoint}): {e}")
        return None


def fetch_broker_summary(symbol: str, date: str = None) -> list:
    """Ambil broker summary 1 saham buat 1 tanggal (default: hari ini)."""
    symbol_clean = symbol.replace(".JK", "")
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    data = fetch_goapi(f"/{symbol_clean}/broker_summary", {"date": date})
    if not data:
        return []
    return data.get("data", {}).get("results", []) or []


def check_broker_accumulation(symbol: str, date: str = None) -> dict:
    """
    Broker Summary / Aksi Broker (Video 3, 16, 27) - pakai data REAL dari
    GoAPI. Cek broker mana net-buy terbesar hari itu, apakah itu termasuk
    'smart money', dan broker mana yang net-sell terbesar (biasanya retail).

    Selain kesimpulan (triggered/value), fungsi ini juga nyimpen "raw_table":
    rekap net-buy/sell PER BROKER langsung dari hasil API GoAPI - ini bukti
    mentahnya, bukan cuma kalimat kesimpulan, biar bisa diverifikasi manual
    dan kuota API yang udah kepakai kelihatan hasilnya.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    results = fetch_broker_summary(symbol, date)
    if not results:
        return {
            "triggered": False,
            "value": "Data broker summary tidak tersedia (cek API key/kuota GoAPI, atau tanggal ini libur bursa)",
            "date": date,
            "description": BROKER_SUMMARY_DESCRIPTION,
            "raw_table": [],
            "evidence": [],
        }

    broker_net = {}
    for r in results:
        code = r.get("code") or (r.get("broker") or {}).get("code")
        side = str(r.get("side") or "").lower()
        value = r.get("value") or 0
        if code is None:
            continue
        broker_net.setdefault(code, 0)
        if side in ("buy", "b"):
            broker_net[code] += value
        elif side in ("sell", "s"):
            broker_net[code] -= value

    if not broker_net:
        return {
            "triggered": False,
            "value": "Data broker summary kosong buat tanggal ini",
            "date": date,
            "description": BROKER_SUMMARY_DESCRIPTION,
            "raw_table": [],
            "evidence": [],
        }

    sorted_brokers = sorted(broker_net.items(), key=lambda x: x[1], reverse=True)
    top_buyer_code, top_buyer_val = sorted_brokers[0]
    top_seller_code, top_seller_val = sorted_brokers[-1]

    is_smart_money_buying = top_buyer_code in BROKER_SMART_MONEY
    is_retail_selling = top_seller_code in BROKER_RETAIL

    triggered = bool(is_smart_money_buying and top_buyer_val > 0)

    smart_name = BROKER_SMART_MONEY.get(top_buyer_code, top_buyer_code)
    insider_note = f" (terkait {BROKER_INSIDER_MAP[top_buyer_code]})" if top_buyer_code in BROKER_INSIDER_MAP else ""

    detail = (
        f"Net-buy terbesar: {top_buyer_code} ({smart_name}){insider_note} Rp{top_buyer_val:,.0f} | "
        f"Net-sell terbesar: {top_seller_code} Rp{abs(top_seller_val):,.0f}"
        + (" - broker smart money lagi akumulasi" if is_smart_money_buying else "")
    )

    # Bukti langsung per broker (raw_table) - ini yang ditampilkan sebagai
    # tabel di web app, bukan cuma disimpulkan lewat teks.
    raw_table = []
    total_smart_net = 0.0
    total_retail_net = 0.0
    for code, net_val in sorted_brokers:
        if code in BROKER_SMART_MONEY:
            kategori = "🟢 Smart Money"
            nama = BROKER_SMART_MONEY[code]
            total_smart_net += net_val
        elif code in BROKER_RETAIL:
            kategori = "🔵 Retail"
            nama = BROKER_RETAIL[code]
            total_retail_net += net_val
        else:
            kategori = "-"
            nama = BROKER_INSIDER_MAP.get(code, "-")
        raw_table.append({
            "Kode Broker": code,
            "Nama Sekuritas": nama,
            "Kategori": kategori,
            "Net Value (Rp)": round(float(net_val), 0),
            "Sisi": "Net Buy" if net_val > 0 else ("Net Sell" if net_val < 0 else "Netral"),
        })

    total_all_net = sum(v for _, v in sorted_brokers)
    n_buy_side = sum(1 for _, v in sorted_brokers if v > 0)
    n_sell_side = sum(1 for _, v in sorted_brokers if v < 0)

    # Evidence berbentuk POIN/TABEL (sama polanya kayak screener BSJP/BPJS/Day
    # Trade) - biar hasil bandarmology bisa divalidasi kriteria per kriteria,
    # bukan cuma dibaca dari 1 kalimat kesimpulan.
    evidence = [
        {
            "key": "bandar_top_buyer",
            "label": "Net-Buy Terbesar Hari Ini",
            "passed": bool(top_buyer_val > 0),
            "value": f"{top_buyer_code} ({smart_name}) — Rp{top_buyer_val:,.0f}",
            "description": "Broker dengan selisih beli-jual (net) TERBESAR hari itu. Ini "
                            "'pemain utama' yang paling banyak nambah posisi net di saham ini.",
        },
        {
            "key": "bandar_top_buyer_category",
            "label": "Kategori Net-Buyer Terbesar",
            "passed": is_smart_money_buying,
            "value": ("🟢 Smart Money" if is_smart_money_buying else
                      ("🔵 Retail" if top_buyer_code in BROKER_RETAIL else "❓ Belum diklasifikasi")) + insider_note,
            "description": "LOLOS kalau net-buyer terbesar termasuk broker 'smart money' "
                            "(institusi/asing besar, lihat tabel referensi di bawah) - "
                            "dianggap sinyal akumulasi yang lebih meyakinkan dibanding kalau "
                            "yang beli cuma broker retail.",
        },
        {
            "key": "bandar_top_seller",
            "label": "Net-Sell Terbesar Hari Ini",
            "passed": is_retail_selling,
            "value": f"{top_seller_code} — Rp{abs(top_seller_val):,.0f}"
                      + (" (Retail)" if is_retail_selling else ""),
            "description": "Broker dengan net-sell (jual bersih) terbesar. Kalau ini broker "
                            "RETAIL sementara net-buyer terbesar adalah smart money, polanya "
                            "sering diartikan 'retail panic-sell, institusi akumulasi'.",
        },
        {
            "key": "bandar_breadth",
            "label": "Sebaran Broker (Buy vs Sell)",
            "passed": bool(n_buy_side >= n_sell_side),
            "value": f"{n_buy_side} broker net-buy vs {n_sell_side} broker net-sell (dari {len(sorted_brokers)} broker aktif)",
            "description": "Berapa banyak broker yang net-buy vs net-sell hari itu. Kalau "
                            "mayoritas broker net-buy, minat beli tersebar luas (bukan cuma "
                            "1-2 broker), biasanya lebih sehat.",
        },
        {
            "key": "bandar_net_total",
            "label": "Net Value Total Semua Broker",
            "passed": bool(total_all_net > 0),
            "value": f"Rp{total_all_net:,.0f}",
            "description": "Total net-buy dikurangi net-sell semua broker digabung. Positif "
                            "artinya hari itu lebih banyak aksi beli ketimbang jual secara "
                            "keseluruhan (across semua broker, bukan cuma yang terbesar).",
        },
    ]

    return {
        "triggered": triggered,
        "value": detail,
        "date": date,
        "description": BROKER_SUMMARY_DESCRIPTION,
        "top_buyer": top_buyer_code,
        "top_buyer_value": round(float(top_buyer_val), 0),
        "top_seller": top_seller_code,
        "top_seller_value": round(float(top_seller_val), 0),
        "is_smart_money_buying": is_smart_money_buying,
        "is_retail_selling": is_retail_selling,
        "raw_table": raw_table,
        "evidence": evidence,
        "total_smart_net": round(total_smart_net, 0),
        "total_retail_net": round(total_retail_net, 0),
    }


BROKER_AVERAGING_DESCRIPTION = (
    "Cek broker tertentu (biasanya top buyer dari Broker Summary) apakah lagi "
    "'averaging UP' (harga rata-rata beli mereka naik dari hari ke hari - "
    "sinyal makin yakin/makin agresif akumulasi) atau 'averaging DOWN' (harga "
    "rata-rata beli turun - kurang meyakinkan, bisa jadi cuma nyoba-nyoba). "
    "PERINGATAN KUOTA: fungsi ini manggil API sebanyak jumlah hari yang dicek "
    "(default 5 hari) - jauh lebih boros dari pengecekan Broker Summary biasa, "
    "makanya ini OPSIONAL dan terpisah, bukan otomatis jalan tiap screening."
)


def check_broker_averaging_trend(symbol: str, broker_code: str, lookback_days: int = 5) -> dict:
    """
    Cek apakah broker tertentu sedang averaging UP (sinyal positif lebih
    kuat - makin yakin, makin agresif) atau averaging DOWN (kurang
    meyakinkan) selama beberapa hari terakhir, berdasarkan data harian
    broker summary dari GoAPI.

    PERINGATAN KUOTA: manggil API sebanyak `lookback_days` kali (default 5).
    Pakai secukupnya, jangan dipanggil otomatis buat semua saham sekaligus.
    """
    dates_checked = []
    avg_prices = []
    lots = []

    for i in range(lookback_days):
        d = (datetime.now() - pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        rows = fetch_broker_summary(symbol, d)
        if not rows:
            continue
        buy_rows = [
            r for r in rows
            if (r.get("code") == broker_code) and str(r.get("side", "")).lower() in ("buy", "b")
        ]
        if not buy_rows:
            continue
        total_val = sum((r.get("value") or 0) for r in buy_rows)
        total_lot = sum((r.get("lot") or 0) for r in buy_rows)
        if total_lot > 0:
            dates_checked.append(d)
            avg_prices.append(total_val / total_lot)
            lots.append(total_lot)

    # urutkan dari yang paling lama ke paling baru (kronologis)
    combined = sorted(zip(dates_checked, avg_prices, lots), key=lambda x: x[0])
    dates_checked = [c[0] for c in combined]
    avg_prices = [c[1] for c in combined]
    lots = [c[2] for c in combined]

    if len(avg_prices) < 2:
        return {
            "available": True,
            "trend": "data_kurang",
            "accumulation_count": len(avg_prices),
            "meets_min_count": False,
            "signal": "belum_cukup_data",
            "description": BROKER_AVERAGING_DESCRIPTION,
            "detail": f"Broker {broker_code} cuma kedeteksi net-buy di {len(avg_prices)} dari "
                      f"{lookback_days} hari terakhir - belum cukup buat nentuin tren averaging.",
            "dates": dates_checked, "avg_prices": avg_prices,
        }

    is_averaging_up = avg_prices[-1] > avg_prices[0]
    meets_min_count = len(avg_prices) >= 2  # syarat minimal akumulasi 2x

    signal = "positif_kuat" if (is_averaging_up and meets_min_count) else "kurang_menarik"
    trend_label = "averaging UP" if is_averaging_up else "averaging DOWN"

    return {
        "available": True,
        "trend": "averaging_up" if is_averaging_up else "averaging_down",
        "accumulation_count": len(avg_prices),
        "meets_min_count": meets_min_count,
        "signal": signal,
        "description": BROKER_AVERAGING_DESCRIPTION,
        "detail": (
            f"Broker {broker_code} {trend_label} selama {len(avg_prices)} hari yang "
            f"kedeteksi ({dates_checked[0]} s/d {dates_checked[-1]}): harga rata-rata beli "
            f"dari Rp{avg_prices[0]:,.0f} jadi Rp{avg_prices[-1]:,.0f}."
        ),
        "dates": dates_checked, "avg_prices": [round(float(p), 0) for p in avg_prices], "lots": lots,
    }

BUY_THE_DIP_DESCRIPTION = (
    "Cek apakah broker smart money net-buy BESAR justru pas harga saham lagi "
    "turun tajam (>=1% dalam sehari). Pola ini sering diartikan sebagai 'buy "
    "the dip' institusi - mereka manfaatin harga turun buat masuk lebih "
    "murah, bukan ikut panic-sell. HEMAT KUOTA: fungsi ini cuma cek 1 HARI "
    "MERAH PALING BARU dalam 30 hari terakhir (bukan semua hari merah), jadi "
    "cuma 1x panggilan API per pengecekan - atau 0x kalau nggak ada hari "
    "yang harganya turun tajam sama sekali dalam periode itu."
)


def check_buy_the_dip_accumulation(df: pd.DataFrame, symbol: str, lookback_days: int = 30) -> dict:
    """
    Buy-the-dip saat broker akumulasi crash (Video 27) - HEMAT API: cuma
    cek 1 hari merah PALING BARU dalam lookback_days (bukan tiap hari
    merah), jadi cuma 1x panggilan API per pengecekan (atau 0x kalau nggak
    ada hari yang harganya turun tajam sama sekali).
    """
    if len(df) < 2:
        return {"triggered": False, "value": "Data harga kurang", "description": BUY_THE_DIP_DESCRIPTION, "evidence": []}

    recent = df.tail(lookback_days)
    close = df["Close"]

    # Cari SEMUA hari merah dulu - ini GRATIS (dari data harga Yahoo Finance
    # yang udah ke-fetch), belum manggil GoAPI sama sekali di tahap ini.
    red_days = []
    for date in recent.index:
        idx = df.index.get_loc(date)
        if idx == 0:
            continue
        pct_change = (close.iloc[idx] / close.iloc[idx - 1] - 1) * 100
        if pct_change <= -1:  # turun >=1%
            red_days.append((date, pct_change))

    if not red_days:
        return {
            "triggered": False,
            "value": f"Nggak ada hari yang harganya turun ≥1% dalam {lookback_days} hari terakhir - "
                     f"nggak ada yang perlu dicek, 0 panggilan API dipakai.",
            "description": BUY_THE_DIP_DESCRIPTION,
            "evidence": [],
        }

    # Ambil yang PALING BARU aja - 1x panggilan API total, bukan per hari merah.
    latest_date, latest_pct = red_days[-1]
    date_str = latest_date.strftime("%Y-%m-%d")
    acc = check_broker_accumulation(symbol, date_str)

    checked_days = [{
        "Tanggal": date_str,
        "Perubahan Harga": f"{latest_pct:+.1f}%",
        "Net-Buyer Terbesar": acc.get("top_buyer", "-"),
        "Nilai Akumulasi (Rp)": f"Rp{acc.get('top_buyer_value', 0):,.0f}" if acc.get("top_buyer") else "-",
        "Smart Money Akumulasi?": "✅ Ya" if acc.get("is_smart_money_buying") else "❌ Tidak",
    }]

    triggered = bool(acc.get("is_smart_money_buying"))
    if triggered:
        detail = (f"Tgl {date_str}: harga turun {latest_pct:.1f}%, tapi broker smart money "
                  f"({acc.get('top_buyer')}) justru net-buy besar")
    else:
        detail = (
            f"Hari merah paling baru ({date_str}, {latest_pct:+.1f}%) dicek - nggak ada tanda "
            f"akumulasi smart money. Catatan: cuma hari terbaru yang dicek buat hemat kuota "
            f"(ada {len(red_days)} hari merah total dalam {lookback_days} hari terakhir, tapi "
            f"cuma 1 yang dicek ke API)."
        )

    return {
        "triggered": triggered,
        "value": detail,
        "description": BUY_THE_DIP_DESCRIPTION,
        "evidence": checked_days,
    }


def check_entry_conditions(price_above_ma50: bool, fundamental_score: int, fundamental_max: int,
                            bandarmology_triggered: bool, risk_reward_1: float = None) -> dict:
    """
    Checklist 4-kondisi wajib sebelum entry - rangkum sinyal dari 4 pilar
    sekaligus (Teknikal, Fundamental, Money Flow/Bandarmology, Risk:Reward)
    jadi satu checklist ringkas, bukan sinyal baru - ini cuma nge-gabungin
    hasil dari fungsi-fungsi lain yang udah ada.

    Semua parameter dihitung DI LUAR fungsi ini (di web_app.py) dari hasil
    compute_trade_levels, compute_fundamental_health_score, dan
    compute_bandarmology_score / check_broker_accumulation yang udah jalan.
    """
    fundamental_ok = fundamental_max > 0 and (fundamental_score / fundamental_max) >= 0.6  # minimal 60% kriteria fundamental lolos
    rr_ok = (risk_reward_1 or 0) >= 2

    conditions = {
        "trend": {
            "passed": bool(price_above_ma50),
            "label": "Trend: harga di atas MA50 (bukan downtrend)",
        },
        "fundamental": {
            "passed": bool(fundamental_ok),
            "label": f"Fundamental: minimal 60% kriteria sehat terpenuhi ({fundamental_score}/{fundamental_max})",
        },
        "bandarmology": {
            "passed": bool(bandarmology_triggered),
            "label": "Money Flow: ada tanda akumulasi smart money",
        },
        "risk_reward": {
            "passed": bool(rr_ok),
            "label": f"Risk:Reward minimal 1:2 ke TP1 ({risk_reward_1 if risk_reward_1 else '-'})",
        },
    }
    met_count = sum(1 for c in conditions.values() if c["passed"])

    return {
        "conditions": conditions,
        "met_count": met_count,
        "total": 4,
        "all_met": met_count == 4,
        "summary": f"{met_count}/4 kondisi wajib terpenuhi" + (" - siap entry" if met_count == 4 else ""),
    }

def compute_bandarmology_score(ticker: str, df: pd.DataFrame, include_buy_the_dip: bool = False) -> dict:
    """
    Skor tambahan dari data broker summary GoAPI - OPSIONAL, cuma jalan
    kalau GOAPI_API_KEY udah diisi. Ini yang dipanggil dari web app / script
    utama, bukan fungsi individual di atas.
    """
    if "ISI_" in GOAPI_API_KEY or not GOAPI_API_KEY:
        return {"score": 0, "reasons": [], "available": False, "detail": None,
                "description": BROKER_SUMMARY_DESCRIPTION}

    # PENTING: jangan pakai tanggal "hari ini" dari jam sistem - kalau hari
    # ini libur bursa/weekend atau GoAPI belum publish data buat hari
    # berjalan, hasilnya bakal kosong. Pakai tanggal TRANSAKSI TERAKHIR yang
    # beneran valid dari data harga (df) yang udah kita fetch dari Yahoo
    # Finance - ini jauh lebih reliable buat mastiin raw_table ke-isi.
    last_trading_date = df.index[-1].strftime("%Y-%m-%d") if len(df) else None
    acc = check_broker_accumulation(ticker, date=last_trading_date)
    score = 0
    reasons = []
    if acc["triggered"]:
        score += 1
        reasons.append(acc["value"])

    dip_result = None
    if include_buy_the_dip:
        dip_result = check_buy_the_dip_accumulation(df, ticker)
        if dip_result["triggered"]:
            score += 1
            reasons.append(dip_result["value"])

    # Hitung persis berapa kali GoAPI kepanggil di run ini - dipakai buat
    # ngelacak kuota harian di sisi web app (bukan tebakan/estimasi kasar).
    api_calls_made = 1  # 1 panggilan buat check_broker_accumulation di atas
    if dip_result is not None:
        api_calls_made += len(dip_result.get("evidence") or [])

    return {
        "score": score,
        "reasons": reasons,
        "available": True,
        "broker_accumulation": acc,
        "buy_the_dip": dip_result,
        "api_calls_made": api_calls_made,
    }


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
