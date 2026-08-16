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
LOOKBACK_DAYS = "1y"  # dinaikkan dari 3mo biar cukup buat Position Trading (butuh MA100 & data 200 hari)

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
    Sinyal dianggap lebih kuat kalau dua-duanya terjadi bersamaan,
    dibanding cuma salah satu.
    """
    close = df["Close"]
    price = close.iloc[-1]
    stoch_rsi = ta.momentum.StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
    stoch_val = stoch_rsi.stochrsi_k().iloc[-1] * 100  # skala 0-100

    sr = find_support_resistance(df)
    near_support = check_near_support(price, sr["support"])
    oversold = bool(stoch_val < 20)

    triggered = oversold and near_support
    return {
        "triggered": triggered,
        "value": f"StochRSI {stoch_val:.1f} ({'oversold' if oversold else 'normal'}), "
                  f"{'dekat' if near_support else 'jauh dari'} support",
    }


def check_sr_role_reversal(df: pd.DataFrame, lookback: int = 60, tolerance_pct: float = 0.02) -> dict:
    """
    Support/Resistance Role Reversal (Video 28, 30) - cek apakah level yang
    dulunya resistance sekarang udah beberapa kali "dites" jadi support
    (atau sebaliknya), setelah harga breakout ngelewatin level itu.

    Caranya disederhanakan: ambil level S/R dari separuh awal data historis
    (sebelum breakout), cek apakah harga SEKARANG ada di dekat salah satu
    level itu TAPI posisinya udah kebalikan dari fungsi asalnya
    (level dulu resistance, sekarang harga ada di atasnya & deket = jadi support).
    """
    if len(df) < lookback:
        return {"triggered": False, "value": "Data historis kurang buat cek role reversal"}

    old_df = df.iloc[:-20]  # data sebelum 20 hari terakhir
    recent_price = df["Close"].iloc[-1]

    try:
        old_sr = find_support_resistance(old_df)
    except Exception:
        return {"triggered": False, "value": "Gagal hitung level historis"}

    # cari resistance lama yang sekarang ada DI BAWAH harga (udah ditembus)
    # dan harga sekarang deket situ (jadi dianggap support baru)
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
    Range Trading (Video 4) - deteksi saham yang lagi bergerak sideways
    (nggak tren naik/turun tajam), cocok buat strategi beli-di-support
    jual-di-resistance dalam range yang sama.

    Caranya: ukur (harga tertinggi - harga terendah) / harga rata-rata
    dalam N hari terakhir. Kalau rentangnya sempit (di bawah max_range_pct),
    dianggap sideways.
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
        "range_high": float(high),
        "range_low": float(low),
    }


def suggest_trailing_stop(df: pd.DataFrame, trail_pct: float = 5.0) -> dict:
    """
    Trailing Stop (Video 20) - hitung level trailing stop SAAT INI berdasarkan
    harga tertinggi yang pernah dicapai sejak entry hipotetis (disederhanakan:
    pakai harga tertinggi 20 hari terakhir sebagai proxy "puncak").

    Ini BUKAN pelacakan posisi beneran (soalnya script ini stateless, nggak
    nyimpen kapan kamu entry) - anggap ini kalkulator bantu, isi manual
    tanggal & harga entry kamu sendiri kalau mau lebih presisi.
    """
    recent_high = df["Close"].tail(20).max()
    current_price = df["Close"].iloc[-1]
    trailing_stop_level = recent_high * (1 - trail_pct / 100)

    return {
        "recent_high": round(float(recent_high), 0),
        "current_price": round(float(current_price), 0),
        "trailing_stop_level": round(float(trailing_stop_level), 0),
        "trail_pct": trail_pct,
        "note": f"Kalau harga puncak 20 hari terakhir adalah {recent_high:.0f}, "
                f"trailing stop {trail_pct}% ada di {trailing_stop_level:.0f}. "
                f"Update ulang tiap ada rekor tertinggi baru.",
    }


def check_leading_lagging(ticker: str, batch_data: dict, min_leader_gain_pct: float = 5.0) -> dict:
    """
    Leading-Lagging antar saham segrup (Video 2, 6) - cek apakah ticker ini
    berada dalam satu grup konglomerat dengan saham lain yang HARI INI udah
    naik signifikan (jadi 'leader'), sementara ticker ini sendiri belum
    banyak bergerak - artinya berpotensi jadi 'penyusul' (laggard).

    batch_data: dict {ticker: dataframe} - data yang UDAH diambil buat semua
    saham di watchlist (biar nggak fetch ulang satu-satu).
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

    # cek saham ini sendiri belum naik signifikan
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
    "^DJI": "Dow Jones",
    "^GSPC": "S&P 500",
    "^IXIC": "Nasdaq",
    "^VIX": "VIX (indeks volatilitas)",
    "^N225": "Nikkei 225",
    "^KS11": "KOSPI",
}


def fetch_macro_context() -> dict:
    """
    Cek kondisi bursa global sebelum screening IHSG - malam (bursa Amerika)
    dan pagi (Nikkei & KOSPI, buka ~2 jam lebih awal dari IHSG dan dianggap
    lebih relate karena sama-sama regional Asia).

    Return: dict per indeks + skor sentimen makro keseluruhan (-1/0/+1).
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
                # VIX itu kebalikan: naik = takut/negatif buat saham
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


def check_scalping(df: pd.DataFrame) -> dict:
    """
    Screener 'Scalping' (Video 14) - timeframe super pendek, cari momentum
    kecil yang sering muncul. Threshold lebih longgar dari Day Trade karena
    targetnya cuma pergerakan kecil dalam waktu singkat.

    Kriteria:
    - Volume >= 1.2x rata-rata 20 hari (lebih longgar dari day trade)
    - RSI antara 45-65 (zona netral-bullish, momentum kecil cukup)
    - Harga di atas MA5 (tren sangat jangka pendek naik)
    - Spread High-Low hari ini cukup besar (>1.5%) - ada pergerakan buat di-scalp
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    price = close.iloc[-1]
    today_volume = volume.iloc[-1]
    vol_ma20 = volume.rolling(20).mean().iloc[-1]
    ma5 = close.rolling(5).mean().iloc[-1]
    today_high = high.iloc[-1]
    today_low = low.iloc[-1]
    spread_pct = (today_high - today_low) / price * 100 if price > 0 else 0

    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]

    conditions = [
        {
            "key": "scalp_volume",
            "label": "Volume ≥1.2x rata-rata 20 hari",
            "passed": bool(today_volume >= 1.2 * vol_ma20),
            "value": f"{today_volume:,.0f} vs 1.2x MA20 ({1.2 * vol_ma20:,.0f})",
            "description": "Volume cukup aktif buat scalping - nggak perlu se-ekstrem day trade "
                            "karena target pergerakannya juga lebih kecil.",
        },
        {
            "key": "scalp_rsi",
            "label": "RSI antara 45-65 (momentum netral-bullish)",
            "passed": bool(45 <= rsi <= 65),
            "value": f"RSI {rsi:.1f}",
            "description": "Zona RSI moderat - cukup ada momentum tapi belum ekstrem, cocok "
                            "buat entry-exit cepat berkali-kali dalam sehari.",
        },
        {
            "key": "scalp_ma5",
            "label": "Harga di atas MA5",
            "passed": bool(price > ma5),
            "value": f"{price:.0f} vs MA5 {ma5:.0f}",
            "description": "Tren sangat jangka pendek (5 hari) masih naik - relevan buat "
                            "scalping yang horizonnya cuma hitungan menit-jam.",
        },
        {
            "key": "scalp_spread",
            "label": "Spread High-Low hari ini >1.5%",
            "passed": bool(spread_pct > 1.5),
            "value": f"{spread_pct:.2f}% (High {today_high:.0f}, Low {today_low:.0f})",
            "description": "Ada pergerakan harga yang cukup lebar hari ini - scalping butuh "
                            "volatilitas intraday, kalau spread-nya kecil susah cari profit cepat.",
        },
    ]
    passed, evidence = _evaluate_conditions(conditions)
    return {
        "passed": passed,
        "evidence": evidence,
        "detail": "Lolos screener Scalping" if passed else None,
    }


def check_swing_trading(df: pd.DataFrame) -> dict:
    """
    Screener 'Swing Trading' (Video 14) - horizon beberapa hari sampai
    beberapa minggu, pakai timeframe harian. Threshold lebih ketat/jarang
    dari day trade, tapi target keuntungan per sinyal lebih besar.

    Kriteria:
    - MA20 > MA50 (uptrend jangka menengah)
    - Harga di atas MA20 DAN MA50 (posisi kuat dalam tren)
    - RSI antara 45-65 (momentum sehat, belum extend terlalu jauh)
    - MACD histogram positif (konfirmasi momentum)
    - Volume >= rata-rata 20 hari (minimal nggak sepi)
    """
    close = df["Close"]
    volume = df["Volume"]

    if len(close) < 50:
        return {"passed": False, "evidence": [], "detail": None}

    price = close.iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    today_volume = volume.iloc[-1]
    vol_ma20 = volume.rolling(20).mean().iloc[-1]

    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
    macd_hist = ta.trend.MACD(close).macd_diff().iloc[-1]

    conditions = [
        {
            "key": "swing_trend",
            "label": "MA20 > MA50 (uptrend jangka menengah)",
            "passed": bool(ma20 > ma50),
            "value": f"MA20 {ma20:.0f} vs MA50 {ma50:.0f}",
            "description": "Tren jangka menengah (20 vs 50 hari) sedang naik - dasar utama "
                            "buat swing trading yang megang posisi beberapa hari-minggu.",
        },
        {
            "key": "swing_position",
            "label": "Harga di atas MA20 & MA50",
            "passed": bool(price > ma20 and price > ma50),
            "value": f"{price:.0f} vs MA20 {ma20:.0f} & MA50 {ma50:.0f}",
            "description": "Posisi harga kuat, di atas dua rata-rata sekaligus - konfirmasi "
                            "tambahan tren jangka menengah benar-benar solid.",
        },
        {
            "key": "swing_rsi",
            "label": "RSI antara 45-65",
            "passed": bool(45 <= rsi <= 65),
            "value": f"RSI {rsi:.1f}",
            "description": "Momentum sehat tapi belum overextend - swing trading butuh ruang "
                            "gerak masih naik, bukan yang udah mepet jenuh beli.",
        },
        {
            "key": "swing_macd",
            "label": "MACD histogram positif",
            "passed": bool(macd_hist > 0),
            "value": f"{macd_hist:.2f}",
            "description": "Konfirmasi momentum masih bullish di timeframe harian.",
        },
        {
            "key": "swing_volume",
            "label": "Volume ≥ rata-rata 20 hari",
            "passed": bool(today_volume >= vol_ma20),
            "value": f"{today_volume:,.0f} vs MA20 {vol_ma20:,.0f}",
            "description": "Minimal nggak lagi sepi transaksi - swing trading nggak butuh "
                            "volume se-ekstrem day trade, tapi tetap harus wajar.",
        },
    ]
    passed, evidence = _evaluate_conditions(conditions)
    return {
        "passed": passed,
        "evidence": evidence,
        "detail": "Lolos screener Swing Trading" if passed else None,
    }


def check_ara_hunter(df: pd.DataFrame, ara_threshold_pct: float = 20.0) -> dict:
    """
    Screener 'ARA Hunter' - nyari saham yang mendekati/kena ARA (Auto Reject
    Atas, batas kenaikan harga maksimum harian di IDX). Gaya trading yang
    umum di kalangan trader retail Indonesia, ngejar saham yang lagi "gocap
    naik" atau breakout ekstrem disertai volume gede.

    PERINGATAN KHUSUS: gaya ini termasuk PALING BERISIKO dari semua screener
    di sini. Saham yang deket ARA sering susah dijual (nggak ada lawan beli
    pas mau exit), rawan reversal tajam begitu ARA "jebol", dan beberapa
    saham dengan pola auto-reject berulang pernah kena suspensi BEI karena
    dianggap pergerakan nggak wajar.

    Kriteria:
    - Kenaikan hari ini >= 70% dari batas ARA (proxy: >=14% asumsi ARA 20%,
      makin dekat makin ketat lagi kriterianya)
    - Volume >= 3x rata-rata 20 hari (lonjakan ekstrem)
    - Harga di atas Open (nggak balik turun dari pembukaan)
    - Value transaksi > Rp3 miliar (filter minimal, meski gaya ini emang biasa
      terjadi di saham kurang likuid - makanya risikonya tinggi)
    """
    close = df["Close"]
    open_ = df["Open"]
    volume = df["Volume"]

    price = close.iloc[-1]
    prev_close = close.iloc[-2]
    today_open = open_.iloc[-1]
    today_volume = volume.iloc[-1]
    vol_ma20 = volume.rolling(20).mean().iloc[-1]
    value_transaksi = price * today_volume
    pct_change = (price / prev_close - 1) * 100

    ara_proxy_threshold = ara_threshold_pct * 0.7  # 70% dari batas ARA umum

    conditions = [
        {
            "key": "ara_gain",
            "label": f"Kenaikan hari ini ≥{ara_proxy_threshold:.0f}% (mendekati ARA)",
            "passed": bool(pct_change >= ara_proxy_threshold),
            "value": f"{pct_change:+.1f}%",
            "description": f"Kenaikan mendekati batas ARA umum ({ara_threshold_pct:.0f}%) - "
                            "tanda saham lagi diserbu beli secara ekstrem.",
        },
        {
            "key": "ara_volume",
            "label": "Volume ≥3x rata-rata 20 hari",
            "passed": bool(today_volume >= 3 * vol_ma20),
            "value": f"{today_volume:,.0f} vs 3x MA20 ({3 * vol_ma20:,.0f})",
            "description": "Lonjakan volume ekstrem - ciri khas saham yang lagi diburu rame-rame "
                            "menuju ARA.",
        },
        {
            "key": "ara_open",
            "label": "Harga di atas Open",
            "passed": bool(price >= today_open),
            "value": f"{price:.0f} vs Open {today_open:.0f}",
            "description": "Nggak ada tanda profit taking besar dari pembukaan - momentum ARA "
                            "masih terjaga sampai saat ini.",
        },
        {
            "key": "ara_value",
            "label": "Value transaksi > Rp3 miliar",
            "passed": bool(value_transaksi > 3_000_000_000),
            "value": f"Rp{value_transaksi:,.0f}",
            "description": "Filter likuiditas minimal - meski saham ARA sering kurang likuid, "
                            "ini buat menghindari yang benar-benar nggak ada pasar.",
        },
    ]
    passed, evidence = _evaluate_conditions(conditions)
    return {
        "passed": passed,
        "evidence": evidence,
        "detail": ("⚠️ Lolos screener ARA Hunter - RISIKO TINGGI, baca peringatan"
                   if passed else None),
    }


def check_position_trading(df: pd.DataFrame) -> dict:
    """
    Screener 'Position Trading' - horizon paling panjang (bulanan), fokus
    ke tren besar bukan momentum jangka pendek. Pakai MA50 & MA100 sebagai
    acuan, lebih toleran ke noise harian.

    Kriteria:
    - MA50 > MA100 (uptrend jangka panjang)
    - Harga di atas MA50 (posisi masih dalam tren naik besar)
    - Harga dalam 25% dari titik tertinggi 200 hari (bukan udah jatuh jauh
      dari puncak - masih dalam fase uptrend yang sehat)
    - RSI mingguan-ish (pakai RSI 14 hari biasa) di bawah 75 (belum
      overbought ekstrem jangka panjang)
    """
    close = df["Close"]

    if len(close) < 100:
        return {"passed": False, "evidence": [], "detail": None}

    price = close.iloc[-1]
    ma50 = close.rolling(50).mean().iloc[-1]
    ma100 = close.rolling(100).mean().iloc[-1]
    high_200 = close.tail(min(200, len(close))).max()
    pct_from_high = (price / high_200 - 1) * 100

    rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]

    conditions = [
        {
            "key": "pos_trend",
            "label": "MA50 > MA100 (uptrend jangka panjang)",
            "passed": bool(ma50 > ma100),
            "value": f"MA50 {ma50:.0f} vs MA100 {ma100:.0f}",
            "description": "Tren besar (50 vs 100 hari) masih naik - dasar utama position "
                            "trading yang megang posisi berbulan-bulan.",
        },
        {
            "key": "pos_above_ma50",
            "label": "Harga di atas MA50",
            "passed": bool(price > ma50),
            "value": f"{price:.0f} vs MA50 {ma50:.0f}",
            "description": "Posisi harga masih dalam tren naik jangka menengah-panjang.",
        },
        {
            "key": "pos_near_high",
            "label": "Dalam 25% dari titik tertinggi 200 hari",
            "passed": bool(pct_from_high >= -25),
            "value": f"{pct_from_high:+.1f}% dari puncak 200 hari ({high_200:.0f})",
            "description": "Harga belum jatuh terlalu jauh dari puncaknya - masih dalam fase "
                            "uptrend yang sehat, bukan udah masuk downtrend besar.",
        },
        {
            "key": "pos_rsi",
            "label": "RSI di bawah 75 (belum overbought ekstrem)",
            "passed": bool(rsi < 75),
            "value": f"RSI {rsi:.1f}",
            "description": "Belum jenuh beli ekstrem - masih ada ruang buat posisi jangka "
                            "panjang tanpa risiko koreksi tajam dalam waktu dekat.",
        },
    ]
    passed, evidence = _evaluate_conditions(conditions)
    return {
        "passed": passed,
        "evidence": evidence,
        "detail": "Lolos screener Position Trading" if passed else None,
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


# Parameter jarak SL/TP per gaya trading - dipakai compute_trade_levels()
# biar level yang dihasilkan "terjangkau" sesuai horizon waktu tiap gaya.
TRADE_STYLE_PARAMS = {
    "scalping": {"support_fallback_pct": 0.02, "sl_buffer_pct": 0.008, "tp1_mult": 1.0, "tp2_mult": 2.0, "min_risk_pct": 0.005},
    "day_trade": {"support_fallback_pct": 0.05, "sl_buffer_pct": 0.02, "tp1_mult": 1.5, "tp2_mult": 3.0, "min_risk_pct": 0.01},
    "bsjp": {"support_fallback_pct": 0.05, "sl_buffer_pct": 0.02, "tp1_mult": 1.5, "tp2_mult": 3.0, "min_risk_pct": 0.01},
    "bpjs": {"support_fallback_pct": 0.05, "sl_buffer_pct": 0.02, "tp1_mult": 1.5, "tp2_mult": 3.0, "min_risk_pct": 0.01},
    "ara_hunter": {"support_fallback_pct": 0.03, "sl_buffer_pct": 0.015, "tp1_mult": 1.0, "tp2_mult": 1.5, "min_risk_pct": 0.01},
    "swing_trading": {"support_fallback_pct": 0.08, "sl_buffer_pct": 0.03, "tp1_mult": 2.0, "tp2_mult": 4.0, "min_risk_pct": 0.02},
    "position_trading": {"support_fallback_pct": 0.15, "sl_buffer_pct": 0.05, "tp1_mult": 3.0, "tp2_mult": 6.0, "min_risk_pct": 0.03},
}


def compute_trade_levels(df: pd.DataFrame, style: str = "day_trade") -> dict:
    """
    Hitung level SUPPORT, STOP LOSS, TAKE PROFIT 1 & 2 buat 1 saham,
    disesuaikan sama GAYA TRADING-nya - biar levelnya "terjangkau" dan
    masuk akal buat dipakai entry beneran, bukan target yang kejauhan
    atau kedeketan dari style yang dipilih.

    style: "scalping", "day_trade", "bsjp", "bpjs", "swing_trading",
           "position_trading", "ara_hunter"

    Logika dasarnya sama (support/resistance historis + asymmetric bet),
    tapi jarak SL/TP-nya diskalakan sesuai horizon waktu tiap gaya:
    - Scalping: SL/TP sangat rapat (hitungan menit-jam)
    - Day Trade/BSJP/BPJS/ARA Hunter: rapat, horizon 1 hari
    - Swing Trading: lebih lebar, horizon beberapa hari-minggu
    - Position Trading: paling lebar, horizon bulanan
    """
    params = TRADE_STYLE_PARAMS.get(style, TRADE_STYLE_PARAMS["day_trade"])

    close = df["Close"]
    price = close.iloc[-1]
    sr = find_support_resistance(df)

    supports_below = sorted([s for s in sr["support"] if s < price], reverse=True)
    resistances_above = sorted([r for r in sr["resistance"] if r > price])

    fallback_support = price * (1 - params["support_fallback_pct"])
    # ambil support terdekat, tapi jangan sampai lebih jauh dari fallback-nya
    # (biar SL nggak kejauhan buat gaya trading rapat kayak scalping)
    if supports_below:
        support = max(supports_below[0], fallback_support)
    else:
        support = fallback_support

    stop_loss = support * (1 - params["sl_buffer_pct"])
    min_risk = price * params["min_risk_pct"]
    risk = max(price - stop_loss, min_risk)
    stop_loss = price - risk  # samakan lagi biar konsisten sama risk minimum

    fallback_tp1 = price + risk * params["tp1_mult"]
    fallback_tp2 = price + risk * params["tp2_mult"]

    if resistances_above:
        # ambil resistance terdekat, tapi jangan lebih jauh dari fallback
        # (biar TP tetap "terjangkau" sesuai gaya trading, nggak kejauhan)
        tp1 = min(resistances_above[0], fallback_tp1) if style in ("scalping", "ara_hunter") else resistances_above[0]
        tp1 = max(tp1, price + min_risk)  # minimal tetap di atas harga
    else:
        tp1 = fallback_tp1

    if len(resistances_above) >= 2:
        tp2 = resistances_above[1]
        tp2 = min(tp2, fallback_tp2) if style in ("scalping", "ara_hunter") else tp2
    else:
        tp2 = max(tp1 + risk * (params["tp2_mult"] - params["tp1_mult"]), fallback_tp2)

    reward1 = tp1 - price
    reward2 = tp2 - price

    return {
        "style": style,
        "price": round(float(price), 0),
        "support": round(float(support), 0),
        "stop_loss": round(float(stop_loss), 0),
        "take_profit_1": round(float(tp1), 0),
        "take_profit_2": round(float(tp2), 0),
        "risk_reward_1": round(float(reward1 / risk), 2) if risk > 0 else None,
        "risk_reward_2": round(float(reward2 / risk), 2) if risk > 0 else None,
        "sl_pct": round(float((stop_loss / price - 1) * 100), 2),
        "tp1_pct": round(float((tp1 / price - 1) * 100), 2),
        "tp2_pct": round(float((tp2 / price - 1) * 100), 2),
    }


# Deskripsi tiap indikator dalam bahasa sederhana, dipakai di web app
# buat nampilin penjelasan pas user klik/pilih indikator tertentu.
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
        "20 hari terakhir. Ini tanda ada minat beli/jual yang tiba-tiba membesar, "
        "sering muncul pas ada berita atau akumulasi besar."
    ),
    "above_ma20": (
        "Harga di atas MA20 (rata-rata 20 hari) nunjukkin tren jangka pendek "
        "masih condong naik dibanding sebulan terakhir."
    ),
    "uptrend_ma": (
        "MA20 di atas MA50 nunjukkin tren jangka pendek lebih kuat dari tren "
        "jangka menengah - kombinasi dua rata-rata ini sering dipakai buat "
        "konfirmasi arah tren yang lebih meyakinkan."
    ),
    "bollinger_riding": (
        "Riding upper Bollinger Band artinya harga 'nempel' dan jalan di "
        "sepanjang garis atas band volatilitas - ini justru tanda tren KUAT, "
        "bukan tanda kemahalan/overbought seperti yang sering disalahpahami."
    ),
    "near_support": (
        "Harga sekarang lagi deket sama level support historis - level yang "
        "berkali-kali jadi 'lantai' harga di masa lalu. Area ini secara "
        "statistik lebih rawan jadi titik pantul."
    ),
    "vcp_pattern": (
        "VCP (Volatility Contraction Pattern) itu pola di mana tiap koreksi "
        "harga makin mengecil - tanda tekanan jual makin lemah. Sering muncul "
        "sebelum breakout menurut pendekatan trader Mark Minervini."
    ),
    "day_trade": (
        "Screener Day Trade nyari saham dengan kombinasi volume aktif, momentum "
        "MACD positif, tren MA20 naik, RSI di zona sehat (nggak oversold/overbought "
        "ekstrem), dan likuiditas cukup buat masuk-keluar dengan cepat di hari yang sama."
    ),
    "bsjp": (
        "BSJP (Beli Sore Jual Pagi) nyari saham yang baru breakout kencang sore "
        "hari - naik signifikan disertai lonjakan volume - dengan asumsi momentum "
        "berlanjut ke pembukaan besok pagi."
    ),
    "bpjs": (
        "BPJS (Beli Pagi Jual Sore) versi lebih longgar dari BSJP, biasa dicek "
        "sebelum market buka buat nyari kandidat yang berpotensi lanjut naik "
        "sepanjang hari itu."
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
        "Level yang dulunya jadi resistance (harga sering mantul turun di situ), "
        "setelah ditembus naik, berubah fungsi jadi support baru. Ini pola klasik "
        "yang sering dipakai buat konfirmasi breakout yang beneran kuat."
    ),
    "range_sideways": (
        "Saham lagi bergerak dalam rentang harga sempit (nggak tren naik/turun "
        "tajam) - cocok buat strategi beli di area bawah range, jual di area "
        "atasnya, ulang-ulang selama masih sideways."
    ),
    "scalping": (
        "Screener Scalping nyari saham dengan momentum kecil tapi sering muncul - "
        "volume cukup aktif, RSI netral-bullish, tren sangat jangka pendek naik, "
        "dan pergerakan harga intraday yang cukup lebar buat di-scalp berkali-kali."
    ),
    "swing_trading": (
        "Screener Swing Trading nyari saham dengan tren jangka menengah kuat "
        "(MA20 di atas MA50), momentum masih sehat, cocok buat dipegang beberapa "
        "hari sampai beberapa minggu."
    ),
    "ara_hunter": (
        "Screener ARA Hunter nyari saham yang mendekati batas ARA (Auto Reject "
        "Atas) disertai lonjakan volume ekstrem. PALING BERISIKO dari semua "
        "screener - rawan reversal tajam dan susah dijual saat mau exit."
    ),
    "position_trading": (
        "Screener Position Trading nyari saham dengan tren jangka panjang kuat "
        "(MA50 di atas MA100), masih dekat titik tertinggi historis, cocok buat "
        "dipegang berbulan-bulan."
    ),
}



def compute_screener_results(df: pd.DataFrame) -> dict:
    """
    Jalankan SEMUA screener (Day Trade, BSJP, BPJS) sekaligus buat 1 saham.
    Masing-masing independen - saham bisa lolos satu, dua, atau ketiganya.

    Return: {"day_trade": {...}, "bsjp": {...}, "bpjs": {...}}
    Tiap entry berisi "passed", "evidence" (list bukti per kriteria), "detail".
    """
    results = {}
    screener_fns = [
        ("day_trade", check_day_trade), ("bsjp", check_bsjp), ("bpjs", check_bpjs),
        ("scalping", check_scalping), ("swing_trading", check_swing_trading),
        ("ara_hunter", check_ara_hunter), ("position_trading", check_position_trading),
    ]
    for key, fn in screener_fns:
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


def analyze_single_stock(ticker: str, batch_data_for_group: dict = None) -> dict:
    """
    Analisis LENGKAP satu saham - dipakai buat fitur "Screening Satu-per-Satu".
    Beda dari run_screener() yang scan banyak saham sekaligus, ini fokus
    ke SATU ticker dan kasih semua info sekaligus: semua indikator, semua
    screener (day trade/scalping/swing/dst), level trading per gaya,
    berita, dan status leading-lagging kalau tickernya masuk grup konglomerat.

    batch_data_for_group: opsional, dict {ticker: df} data saham lain dalam
    grup yang sama (buat cek leading-lagging). Kalau nggak dikasih,
    leading-lagging di-skip.
    """
    ticker = ticker.upper().strip()
    if not ticker.endswith(".JK"):
        ticker += ".JK"

    df = fetch_data(ticker)
    if df.empty:
        return {"ticker": ticker, "found": False, "error": "Data tidak ditemukan atau saham kurang dari 30 hari transaksi"}

    all_indicators = compute_all_indicators(df)
    screeners = compute_screener_results(df)
    fundamental = compute_fundamental_score(ticker)

    trade_levels_per_style = {
        style: compute_trade_levels(df, style=style) for style in TRADE_STYLE_PARAMS.keys()
    }

    leading_lagging = None
    if batch_data_for_group:
        try:
            leading_lagging = check_leading_lagging(ticker, batch_data_for_group)
        except Exception:
            leading_lagging = None

    return {
        "ticker": ticker,
        "found": True,
        "df": df,
        "price": all_indicators["price"],
        "rsi": all_indicators["rsi"],
        "indicators": all_indicators["indicators"],
        "support_levels": all_indicators["support_levels"],
        "resistance_levels": all_indicators["resistance_levels"],
        "screeners": screeners,
        "fundamental": fundamental,
        "trade_levels_per_style": trade_levels_per_style,
        "leading_lagging": leading_lagging,
    }



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
