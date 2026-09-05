"""vSynapse v3.1 — Binance Futures scanner, 1 file.

Isi: fetch data (async), indikator teknikal, confluence scoring,
setup engine (breakout/pullback/continuation/extended), filter ADX,
multi-timeframe scanning + MTF agreement bonus, risk filter,
cooldown/dedup per (symbol, timeframe), notifikasi Telegram, dan
entry point CLI.

Perubahan dari v3 (lihat catatan detail di tiap bagian):
1. Scan sekarang jalan independen di SETIAP timeframe di config
   ("timeframes"), bukan cuma 1 TF utama + 1 TF filter biner. Setiap
   TF bisa menghasilkan sinyal sendiri (15m, 1h, atau 4h — bukan
   melulu 15m).
2. Konfluensi multi-timeframe sekarang jadi BONUS SKOR (kalau 2+ TF
   searah), bukan syarat lolos/gugur — supaya sinyal 1 TF yang kuat
   tidak otomatis dibuang cuma karena TF lain kebetulan netral.
3. Tambah "Setup Engine" ringan: tiap sinyal dilabeli salah satu dari
   BREAKOUT / PULLBACK / CONTINUATION / EXTENDED, dengan bonus/penalti
   skor sesuai jenisnya (lihat classify_setup()).
4. Tambah filter ADX: sinyal ditolak duluan kalau tren dianggap terlalu
   lemah/choppy (ADX di bawah ambang), sebelum indikator lain dihitung.
5. Fix bug: BinanceFuturesClient sekarang benar-benar punya
   get_klines_paginated() (dipanggil backtest.py untuk --limit >1500,
   sebelumnya tidak ada method-nya sama sekali -> AttributeError).

Perubahan v3.2 (fokus kualitas sinyal & setup):
6. Gate histori minimum: symbol dengan closed candle < scanning.min_history_bars
   di-skip, supaya coin baru listing (EMA200/ADX belum konvergen) tidak ikut
   menghasilkan sinyal yang bias.
7. ADX sekarang juga jadi bonus skor bertingkat (scoring.weights.adx_strength),
   bukan cuma gate biner lolos/gugur di risk.adx_min — ADX >= risk.adx_strong
   dapat bonus penuh, di antara adx_min & adx_strong bonusnya diskala linear.
8. SL sekarang berbasis struktur (swing low/high N-bar terakhir + buffer ATR
   kecil), bukan cuma ATR flat — dipakai kalau lebih jauh dari ATR stop biasa,
   di-cap di risk.structure_sl_max_atr_mult supaya R:R tidak jebol saat
   struktur jauh (mis. tren EXTENDED). TP tetap proporsional risk_reward_min
   terhadap SL yang baru ini, jadi kontrak reward:risk tidak berubah.

Dijalankan lewat: python scanner.py --out synaptic_candidates.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import aiohttp
import pandas as pd
import yaml

BASE_URL = "https://www.binance.com"  # fapi.binance.com sering diblokir IP datacenter


# ---------------------------------------------------------------------------
# Data client
# ---------------------------------------------------------------------------

@dataclass
class Kline:
    symbol: str
    timeframe: str
    df: pd.DataFrame


class BinanceFuturesClient:
    def __init__(self, session: aiohttp.ClientSession | None = None):
        self._session = session
        self._owns_session = session is None

    async def __aenter__(self) -> "BinanceFuturesClient":
        if self._session is None:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            self._session = aiohttp.ClientSession(headers=headers)
        return self

    async def __aexit__(self, *exc):
        if self._owns_session and self._session:
            await self._session.close()

    async def get_active_symbols(self, quote_asset: str = "USDT") -> list[str]:
        url = f"{BASE_URL}/fapi/v1/exchangeInfo"
        async with self._session.get(url) as resp:
            data = await resp.json()
        if "symbols" not in data:
            raise RuntimeError(
                f"Binance API tidak mengembalikan data yang diharapkan. "
                f"Status: {resp.status}, Response: {data}"
            )
        return [
            s["symbol"]
            for s in data["symbols"]
            if s["quoteAsset"] == quote_asset and s["status"] == "TRADING"
        ]

    async def get_24h_volume(self, symbol: str) -> float:
        url = f"{BASE_URL}/fapi/v1/ticker/24hr"
        async with self._session.get(url, params={"symbol": symbol}) as resp:
            data = await resp.json()
        return float(data.get("quoteVolume", 0))

    @staticmethod
    def _parse_klines_df(raw: list) -> pd.DataFrame:
        df = pd.DataFrame(
            raw,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades",
                "taker_buy_base", "taker_buy_quote", "ignore",
            ],
        )
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        return df

    async def get_klines(self, symbol: str, interval: str, limit: int = 300) -> Kline:
        url = f"{BASE_URL}/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        async with self._session.get(url, params=params) as resp:
            raw = await resp.json()
        df = self._parse_klines_df(raw)
        return Kline(symbol=symbol, timeframe=interval, df=df)

    async def get_klines_paginated(self, symbol: str, interval: str, total_limit: int) -> Kline:
        """Ambil kline lebih dari batas 1500/request Binance dengan beberapa
        request mundur dari waktu sekarang (pakai endTime), lalu digabung
        jadi satu Kline utuh. Dipakai backtest.py saat --limit > 1500 —
        sebelumnya method ini dipanggil tapi tidak pernah didefinisikan,
        jadi selalu crash AttributeError begitu limit-nya besar."""
        url = f"{BASE_URL}/fapi/v1/klines"
        all_raw: list = []
        end_time: int | None = None
        remaining = total_limit

        while remaining > 0:
            batch_limit = min(remaining, 1500)
            params = {"symbol": symbol, "interval": interval, "limit": batch_limit}
            if end_time is not None:
                params["endTime"] = end_time
            async with self._session.get(url, params=params) as resp:
                raw = await resp.json()
            if not raw:
                break
            all_raw = raw + all_raw
            remaining -= len(raw)
            end_time = int(raw[0][0]) - 1  # mundur sebelum candle paling awal batch ini
            if len(raw) < batch_limit:
                break  # data historis sudah habis

        df = self._parse_klines_df(all_raw)
        df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
        if len(df) > total_limit:
            df = df.iloc[-total_limit:].reset_index(drop=True)
        return Kline(symbol=symbol, timeframe=interval, df=df)

    async def get_klines_many(
        self, symbols: list[str], interval: str, limit: int = 300
    ) -> list[Kline]:
        tasks = [self.get_klines(s, interval, limit) for s in symbols]
        return await asyncio.gather(*tasks, return_exceptions=False)


def drop_unclosed_candle(df: pd.DataFrame) -> pd.DataFrame:
    """Buang candle terakhir kalau itu masih 'sedang berjalan' (belum closed).

    Binance selalu menyertakan candle yang sedang terbentuk sebagai baris
    terakhir kalau di-query real-time. Kalau ini tidak dibuang, skor/sinyal
    dihitung dari harga yang masih bisa berubah kapan saja sampai candle itu
    benar-benar tutup — sumber utama sinyal yang "berubah-ubah" tiap scan."""
    if df.empty or "close_time" not in df.columns:
        return df
    now = pd.Timestamp.now(tz="UTC")
    last_close_time = df["close_time"].iloc[-1]
    if pd.notna(last_close_time) and last_close_time > now:
        return df.iloc[:-1].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Indikator teknikal
# ---------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    result = pd.Series(index=series.index, dtype=float)
    no_loss = avg_loss == 0
    no_gain = avg_gain == 0
    normal = ~no_loss & ~no_gain

    rs = avg_gain[normal] / avg_loss[normal]
    result[normal] = 100 - (100 / (1 + rs))
    result[no_loss & ~no_gain] = 100.0
    result[no_gain & ~no_loss] = 0.0
    result[no_gain & no_loss] = 50.0
    return result


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 2.5) -> pd.Series:
    hl2 = (df["high"] + df["low"]) / 2
    atr_val = atr(df, period)
    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val

    trend = pd.Series(index=df.index, dtype=int)
    trend.iloc[0] = 1
    for i in range(1, len(df)):
        if df["close"].iloc[i] > upper_band.iloc[i - 1]:
            trend.iloc[i] = 1
        elif df["close"].iloc[i] < lower_band.iloc[i - 1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]
    return trend


def volume_spike(volume: pd.Series, lookback: int = 20, factor: float = 1.5) -> pd.Series:
    avg = volume.rolling(lookback).mean()
    return volume > (avg * factor)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — mengukur KEKUATAN tren, bukan arahnya.

    Dipakai sebagai gate di score_at(): kalau ADX di bawah ambang, sinyal
    ditolak lebih dulu sebelum indikator lain dihitung. Ini mengatasi kasus
    EMA200 + MACD + Supertrend kebetulan align sesaat saat market sideways
    (choppy) — ketiganya trend-following dan bisa "sepakat" sesaat tanpa
    ada tren nyata yang layak ditradingkan."""
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
    ).max(axis=1)

    smoothed_tr = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, 1e-9)
    smoothed_plus_dm = plus_dm.ewm(alpha=1 / period, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
    minus_di = 100 * (smoothed_minus_dm / smoothed_tr)

    di_sum = (plus_di + minus_di).replace(0, 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    return dx.ewm(alpha=1 / period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Confluence scoring
# ---------------------------------------------------------------------------

@dataclass
class SignalResult:
    symbol: str
    direction: str  # "LONG" | "SHORT" | "NONE"
    score: float
    timeframe: str = ""
    setup_type: str = ""  # "BREAKOUT" | "PULLBACK" | "CONTINUATION" | "EXTENDED" | ""
    reasons: list[str] = field(default_factory=list)
    entry: float | None = None
    sl: float | None = None
    tp: float | None = None


def compute_indicators(df: pd.DataFrame, cfg: dict) -> dict:
    """Hitung semua indikator SEKALI di seluruh histori yang tersedia.

    Krusial buat akurasi: indikator berbasis EWM (EMA, MACD, Supertrend, ATR,
    ADX) butuh histori panjang supaya nilainya konvergen ke rata-rata yang
    stabil, bukan bias ke titik mulai data yang arbitrer. Backtest versi
    sebelumnya memotong ulang window 200 bar tiap iterasi lalu menghitung
    EMA200 dari nol di situ — 200 bar jelas tidak cukup buat EMA span=200
    konvergen, hasilnya bias dan kadang salah arah dibanding EMA200 "asli"."""
    close = df["close"]
    ind_cfg = cfg["indicators"]

    macd_line, signal_line, _ = macd(
        close, ind_cfg["macd"]["fast"], ind_cfg["macd"]["slow"], ind_cfg["macd"]["signal"],
    )
    vol_cfg = ind_cfg.get("volume_spike", {})
    adx_cfg = ind_cfg.get("adx", {})

    return {
        "ema200": ema(close, ind_cfg["ema"]["period"]),
        "macd_line": macd_line,
        "signal_line": signal_line,
        "supertrend": supertrend(df, ind_cfg["supertrend"]["period"], ind_cfg["supertrend"]["multiplier"]),
        "rsi": rsi(close, ind_cfg["rsi"]["period"]),
        "vol_spike": volume_spike(
            df["volume"], vol_cfg.get("lookback", 20), vol_cfg.get("factor", 1.5)
        ),
        "atr": atr(df, ind_cfg["atr"]["period"]),
        "adx": adx(df, adx_cfg.get("period", 14)),
    }


def classify_setup(df: pd.DataFrame, ind: dict, i: int, direction: str, cfg: dict) -> str:
    """Klasifikasikan jenis setup di bar ke-i, sesuai arah yang sudah
    ditentukan sebelumnya oleh score_at():

    - BREAKOUT: harga close menembus level tertinggi/terendah N-bar
      terakhir searah sinyal — momentum baru, biasanya paling layak dikejar.
    - PULLBACK: harga sedang berada dekat/menyentuh garis EMA200 (area
      value), belum breakout — entry lebih murah, R:R biasanya lebih bagus.
    - EXTENDED: harga sudah terlalu jauh dari EMA200 (dalam satuan ATR) —
      rawan exhaustion/reversal jangka pendek, prioritas diturunkan.
    - CONTINUATION: tidak masuk 3 kategori di atas — tren jalan normal,
      bukan di titik entry yang istimewa.

    Ini pengganti sederhana untuk "Setup Engine" (BREAKOUT/PULLBACK/
    CONTINUATION/EXTENDED) yang sebelumnya cuma jadi arah+skor mentah tanpa
    label sama sekali."""
    se_cfg = cfg.get("setup_engine", {})
    structure_lookback = se_cfg.get("structure_lookback", 20)
    extended_atr_mult = se_cfg.get("extended_atr_mult", 3.5)
    pullback_atr_mult = se_cfg.get("pullback_atr_mult", 1.0)

    close = df["close"].iloc[i]
    atr_val = ind["atr"].iloc[i]
    ema_val = ind["ema200"].iloc[i]
    dist_ema_atr = abs(close - ema_val) / atr_val if atr_val > 0 else 0.0

    lookback_start = max(0, i - structure_lookback)
    prior_high = df["high"].iloc[lookback_start:i].max() if i > lookback_start else close
    prior_low = df["low"].iloc[lookback_start:i].min() if i > lookback_start else close

    if direction == "LONG" and pd.notna(prior_high) and close > prior_high:
        return "BREAKOUT"
    if direction == "SHORT" and pd.notna(prior_low) and close < prior_low:
        return "BREAKOUT"

    if dist_ema_atr <= pullback_atr_mult:
        return "PULLBACK"
    if dist_ema_atr >= extended_atr_mult:
        return "EXTENDED"
    return "CONTINUATION"


def score_at(
    df: pd.DataFrame, ind: dict, i: int, symbol: str, cfg: dict, timeframe: str = ""
) -> SignalResult:
    """Skor 1 bar spesifik (index i) pakai indikator yang sudah dihitung
    sebelumnya lewat compute_indicators(). Dipisah dari score_symbol supaya
    backtest bisa hitung indikator sekali lalu skor banyak titik, bukan
    hitung ulang tiap titik (lihat compute_indicators).

    Alur (v3.2):
    0. Gate ADX — kalau tren dianggap terlalu lemah (choppy), tolak duluan
       sebelum indikator lain dihitung sama sekali.
    1. Arah (`direction`) ditentukan dari 3 indikator trend-following inti
       (EMA200, MACD, Supertrend) — dan tidak bisa dibalik lagi oleh RSI.
    2. RSI cuma MENGUATKAN arah yang sudah ditentukan (momentum
       confirmation), RSI netral (40-60) tidak dapat bonus.
    3. Volume spike cuma dapat bonus kalau searah candle dengan sinyal.
    4. ADX strength — bonus skor bertingkat kalau ADX jauh di atas adx_min
       (tren kuat), beda dari ADX yang cuma baru lolos gate di langkah 0.
    5. Setup Engine memberi label + bonus/penalti sesuai jenis setup
       (breakout/pullback/continuation/extended) — lihat classify_setup().
    6. SL dihitung dari ATR ATAU struktur (swing low/high N-bar terakhir),
       dipilih yang lebih jauh dari entry lalu di-cap — bukan ATR flat saja.
       TP tetap proporsional risk_reward_min terhadap SL final ini.
    """
    adx_min = cfg.get("risk", {}).get("adx_min", 0)
    if adx_min and pd.notna(ind["adx"].iloc[i]) and ind["adx"].iloc[i] < adx_min:
        return SignalResult(
            symbol=symbol,
            direction="NONE",
            score=0.0,
            timeframe=timeframe,
            reasons=[
                f"ADX ({ind['adx'].iloc[i]:.1f}) di bawah ambang {adx_min} — "
                "market dianggap choppy, sinyal ditolak"
            ],
        )

    w = cfg["scoring"]["weights"]
    price = df["close"].iloc[i]
    open_price = df["open"].iloc[i]

    long_score, short_score = 0.0, 0.0
    reasons: list[str] = []

    # --- 1. Tentukan arah dari 3 indikator trend-following inti ---------
    trend_long, trend_short = 0.0, 0.0

    ema_up = price > ind["ema200"].iloc[i]
    if ema_up:
        trend_long += w["ema_trend"]
    else:
        trend_short += w["ema_trend"]

    macd_up = ind["macd_line"].iloc[i] > ind["signal_line"].iloc[i]
    if macd_up:
        trend_long += w["macd_cross"]
    else:
        trend_short += w["macd_cross"]

    st_up = ind["supertrend"].iloc[i] == 1
    if st_up:
        trend_long += w["supertrend"]
    else:
        trend_short += w["supertrend"]

    direction = "LONG" if trend_long >= trend_short else "SHORT"
    long_score, short_score = trend_long, trend_short

    if direction == "LONG":
        reasons.append("Harga di atas EMA200 (uptrend)" if ema_up else "Harga di bawah EMA200 (tapi indikator lain condong LONG)")
        reasons.append("MACD line di atas signal line (momentum naik)" if macd_up else "MACD line di bawah signal line (tapi indikator lain condong LONG)")
        reasons.append("Supertrend menunjukkan uptrend" if st_up else "Supertrend menunjukkan downtrend (tapi indikator lain condong LONG)")
    else:
        reasons.append("Harga di bawah EMA200 (downtrend)" if not ema_up else "Harga di atas EMA200 (tapi indikator lain condong SHORT)")
        reasons.append("MACD line di bawah signal line (momentum turun)" if not macd_up else "MACD line di atas signal line (tapi indikator lain condong SHORT)")
        reasons.append("Supertrend menunjukkan downtrend" if not st_up else "Supertrend menunjukkan uptrend (tapi indikator lain condong SHORT)")

    # --- 2. RSI cuma menguatkan arah yang sudah ditentukan, tidak pernah
    #        melawannya. RSI netral (40-60) tidak dapat bonus sama sekali
    #        karena tidak informatif untuk arah mana pun. -----------------
    r = ind["rsi"].iloc[i]
    if direction == "LONG" and 50 < r <= 70:
        long_score += w["rsi_confluence"]
        reasons.append(f"RSI ({r:.0f}) menguatkan momentum naik")
    elif direction == "SHORT" and 30 <= r < 50:
        short_score += w["rsi_confluence"]
        reasons.append(f"RSI ({r:.0f}) menguatkan momentum turun")
    elif direction == "LONG" and r > 70:
        reasons.append(f"RSI ({r:.0f}) overbought — tidak dapat bonus, hati-hati chasing")
    elif direction == "SHORT" and r < 30:
        reasons.append(f"RSI ({r:.0f}) oversold — tidak dapat bonus, hati-hati chasing")
    else:
        reasons.append(f"RSI netral ({r:.0f}), tidak menambah skor")

    # --- 3. Volume spike cuma dapat bonus kalau candle-nya searah sinyal -
    has_volume_spike = bool(ind["vol_spike"].iloc[i])
    candle_bullish = price > open_price
    candle_bearish = price < open_price
    if has_volume_spike and direction == "LONG" and candle_bullish:
        long_score += w["volume_spike"]
        reasons.append("Volume spike di candle bullish (menguatkan sinyal LONG)")
    elif has_volume_spike and direction == "SHORT" and candle_bearish:
        short_score += w["volume_spike"]
        reasons.append("Volume spike di candle bearish (menguatkan sinyal SHORT)")
    elif has_volume_spike:
        reasons.append("Volume spike terdeteksi tapi arah candle tidak sesuai sinyal — diabaikan")

    # --- 4. ADX strength: bonus bertingkat, beda dari gate biner di atas -
    # ADX yang baru lolos adx_min (mis. 21) tidak sekuat ADX 45 — di sini
    # ADX >= risk.adx_strong dapat bonus penuh, di antaranya diskala linear.
    adx_strength_weight = w.get("adx_strength", 0)
    adx_val = ind["adx"].iloc[i]
    if adx_strength_weight and pd.notna(adx_val):
        adx_strong = cfg.get("risk", {}).get("adx_strong", (adx_min or 20) + 15)
        if adx_val >= adx_strong:
            adx_bonus = adx_strength_weight
        elif adx_min and adx_val > adx_min and adx_strong > adx_min:
            adx_bonus = adx_strength_weight * (adx_val - adx_min) / (adx_strong - adx_min)
        else:
            adx_bonus = 0.0
        if adx_bonus > 0:
            if direction == "LONG":
                long_score += adx_bonus
            else:
                short_score += adx_bonus
            reasons.append(f"ADX ({adx_val:.1f}) tren kuat (+{adx_bonus:.1f} skor)")

    # --- 5. Setup Engine: label jenis setup + bonus/penalti skor ---------
    setup_type = classify_setup(df, ind, i, direction, cfg)
    setup_bonus = cfg["scoring"].get("setup_bonus", {}).get(setup_type.lower(), 0)
    if direction == "LONG":
        long_score += setup_bonus
    else:
        short_score += setup_bonus
    sign = "+" if setup_bonus >= 0 else ""
    reasons.append(f"Setup terdeteksi: {setup_type} ({sign}{setup_bonus} skor)")

    final_score = long_score if direction == "LONG" else short_score

    if final_score < cfg["scoring"]["min_score_to_trigger"]:
        return SignalResult(
            symbol=symbol, direction="NONE", score=final_score,
            timeframe=timeframe, setup_type=setup_type, reasons=reasons,
        )

    # --- 6. SL berbasis struktur, bukan ATR flat -------------------------
    # ATR-flat stop bisa taruh SL persis di tengah zona swing low/high yang
    # baru saja terbentuk (rawan kesapu noise/wick). Di sini SL digeser ke
    # luar swing low/high N-bar terakhir (structure_lookback) + buffer ATR
    # kecil, TAPI cuma dipakai kalau itu lebih jauh dari ATR stop biasa, dan
    # di-cap ke structure_sl_max_atr_mult supaya R:R tidak jebol saat
    # struktur jauh (mis. setup EXTENDED). TP tetap proporsional terhadap
    # risk_reward_min dari SL final ini, jadi kontrak reward:risk tak berubah.
    se_cfg = cfg.get("setup_engine", {})
    struct_lookback = se_cfg.get("structure_lookback", 20)
    sl_buffer = ind["atr"].iloc[i] * se_cfg.get("structure_sl_buffer_atr_mult", 0.25)
    atr_sl_dist = ind["atr"].iloc[i] * cfg["risk"]["atr_multiplier_sl"]
    max_sl_dist = atr_sl_dist * cfg.get("risk", {}).get("structure_sl_max_atr_mult", 2.5)

    lb_start = max(0, i - struct_lookback)
    if direction == "LONG":
        structural_level = df["low"].iloc[lb_start:i].min() if i > lb_start else None
        structural_dist = (price - structural_level) + sl_buffer if pd.notna(structural_level) else atr_sl_dist
    else:
        structural_level = df["high"].iloc[lb_start:i].max() if i > lb_start else None
        structural_dist = (structural_level - price) + sl_buffer if pd.notna(structural_level) else atr_sl_dist

    sl_dist = max(atr_sl_dist, structural_dist)
    if sl_dist > max_sl_dist:
        sl_dist = max_sl_dist
        reasons.append(f"SL struktur terlalu jauh, di-cap {cfg.get('risk', {}).get('structure_sl_max_atr_mult', 2.5)}x ATR")
    elif sl_dist > atr_sl_dist:
        reasons.append(f"SL digeser ke luar struktur {struct_lookback}-bar terakhir (bukan cuma ATR flat)")

    if direction == "LONG":
        sl = price - sl_dist
        tp = price + sl_dist * cfg["risk"]["risk_reward_min"]
    else:
        sl = price + sl_dist
        tp = price - sl_dist * cfg["risk"]["risk_reward_min"]

    return SignalResult(
        symbol=symbol,
        direction=direction,
        score=round(final_score, 1),
        timeframe=timeframe,
        setup_type=setup_type,
        reasons=reasons,
        entry=round(price, 6),
        sl=round(sl, 6),
        tp=round(tp, 6),
    )


def score_symbol(df: pd.DataFrame, symbol: str, cfg: dict, timeframe: str = "") -> SignalResult:
    """Wrapper: hitung indikator di seluruh df lalu skor bar TERAKHIR.
    Dipakai scanner (live) & chart.py — interface tidak berubah dari versi
    sebelumnya (parameter `timeframe` opsional, default "" tetap backward
    compatible). Untuk backtest, pakai compute_indicators()+score_at()
    langsung (lihat backtest.py) supaya indikator dihitung sekali, bukan
    berulang tiap titik simulasi."""
    ind = compute_indicators(df, cfg)
    return score_at(df, ind, len(df) - 1, symbol, cfg, timeframe)


# ---------------------------------------------------------------------------
# Risk filter
# ---------------------------------------------------------------------------

def passes_risk_filter(signal: SignalResult, cfg: dict) -> bool:
    if signal.direction == "NONE" or signal.entry is None:
        return False

    risk = abs(signal.entry - signal.sl)
    reward = abs(signal.tp - signal.entry)
    if risk == 0:
        return False

    rr = reward / risk
    return rr >= cfg["risk"]["risk_reward_min"] - 1e-6  # toleransi pembulatan


def position_size(equity: float, risk_pct: float, entry: float, sl: float) -> float:
    risk_amount = equity * risk_pct
    per_unit_risk = abs(entry - sl)
    if per_unit_risk == 0:
        return 0.0
    return risk_amount / per_unit_risk


# ---------------------------------------------------------------------------
# Cooldown / dedup state — sekarang per (symbol, timeframe)
# ---------------------------------------------------------------------------

def load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def save_state(path: str, state: dict) -> None:
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def _state_key(symbol: str, timeframe: str) -> str:
    return f"{symbol}|{timeframe}"


def is_in_cooldown(state: dict, symbol: str, direction: str, timeframe: str, cooldown_hours: float) -> bool:
    """Cooldown sekarang per (symbol, timeframe) — bukan per symbol saja.
    Sebelumnya sinyal 15m yang lagi cooldown ikut memblokir sinyal baru di
    1h/4h untuk symbol yang sama, padahal keduanya independen."""
    entry = state.get(_state_key(symbol, timeframe))
    if not entry or entry.get("direction") != direction:
        return False
    last_time = datetime.fromisoformat(entry["timestamp"])
    return datetime.now(timezone.utc) - last_time < timedelta(hours=cooldown_hours)


def mark_signaled(state: dict, symbol: str, direction: str, timeframe: str) -> None:
    state[_state_key(symbol, timeframe)] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "direction": direction,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Market regime filter
# ---------------------------------------------------------------------------

async def get_market_regime(client: "BinanceFuturesClient", cfg: dict) -> str:
    """Tentukan rezim pasar makro dari 1 simbol bellwether (default BTCUSDT)
    di timeframe tinggi, dipakai buat menyaring sinyal yang melawan arus
    utama pasar.

    Alasan: backtest batch nunjukkin trend-following system ini lemah secara
    STATISTIK SIGNIFIKAN kalau ambil sinyal berlawanan arah dengan rezim
    dominan (SHORT saat market lagi bull luas: t-stat -3.46 dari 51 trade,
    win rate 0% di beberapa simbol) — bukan cuma kebetulan noise di 1-2 coin,
    kejadian di semua 10 simbol yang dites. BULL/BEAR cuma diklaim kalau ADX
    di timeframe itu cukup kuat (di atas regime_filter.adx_min); kalau tidak,
    dianggap NEUTRAL dan tidak menyaring arah manapun."""
    regime_cfg = cfg.get("regime_filter", {})
    symbol = regime_cfg.get("symbol", "BTCUSDT")
    timeframe = regime_cfg.get("timeframe", "4h")
    adx_threshold = regime_cfg.get("adx_min", 25)

    try:
        kline = await client.get_klines(symbol, timeframe, limit=cfg.get("scanning", {}).get("klines_limit", 300))
    except Exception:
        return "NEUTRAL"  # gagal fetch -> jangan blokir scan gara-gara ini, biarkan lolos netral

    closed = drop_unclosed_candle(kline.df)
    if len(closed) < 50:
        return "NEUTRAL"

    ind = compute_indicators(closed, cfg)
    i = len(closed) - 1
    price = closed["close"].iloc[i]
    ema_up = price > ind["ema200"].iloc[i]
    st_up = ind["supertrend"].iloc[i] == 1
    adx_val = ind["adx"].iloc[i]

    if pd.isna(adx_val) or adx_val < adx_threshold:
        return "NEUTRAL"
    if ema_up and st_up:
        return "BULL"
    if not ema_up and not st_up:
        return "BEAR"
    return "NEUTRAL"  # EMA & Supertrend tidak sepakat -> jangan jadikan dasar filter


# ---------------------------------------------------------------------------
# Notifikasi Telegram
# ---------------------------------------------------------------------------

def format_signal_message(signal: SignalResult) -> str:
    tf_label = f" [{signal.timeframe}]" if signal.timeframe else ""
    setup_label = f" · {signal.setup_type}" if signal.setup_type else ""
    lines = [
        f"*{signal.symbol}*{tf_label} — {signal.direction}{setup_label} (score {signal.score})",
        f"Entry: `{signal.entry}`",
        f"SL: `{signal.sl}`",
        f"TP: `{signal.tp}`",
    ]
    if signal.reasons:
        lines.append("Alasan: " + "; ".join(signal.reasons))
    return "\n".join(lines)


async def send_telegram_message(text: str, cfg: dict) -> None:
    tg_cfg = cfg["notify"]["telegram"]
    if not tg_cfg.get("enabled"):
        return

    token = os.environ.get(tg_cfg["bot_token_env"])
    chat_id = os.environ.get(tg_cfg["chat_id_env"])
    if not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}

    async with aiohttp.ClientSession() as session, session.post(url, json=payload) as resp:
        await resp.read()


async def send_telegram_photo(photo_path: str, caption: str, cfg: dict) -> None:
    tg_cfg = cfg["notify"]["telegram"]
    if not tg_cfg.get("enabled"):
        return

    token = os.environ.get(tg_cfg["bot_token_env"])
    chat_id = os.environ.get(tg_cfg["chat_id_env"])
    if not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    with open(photo_path, "rb") as photo_file:
        data = aiohttp.FormData()
        data.add_field("chat_id", str(chat_id))
        data.add_field("caption", caption)
        data.add_field("parse_mode", "Markdown")
        data.add_field(
            "photo",
            photo_file,
            filename=os.path.basename(photo_path),
            content_type="image/png",
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as resp:
                await resp.read()


async def send_telegram_document(file_path: str, caption: str, cfg: dict) -> None:
    """Kirim 1 file (misal .zip berisi kumpulan chart) sebagai dokumen Telegram."""
    tg_cfg = cfg["notify"]["telegram"]
    if not tg_cfg.get("enabled"):
        return

    token = os.environ.get(tg_cfg["bot_token_env"])
    chat_id = os.environ.get(tg_cfg["chat_id_env"])
    if not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/sendDocument"
    # Caption Telegram dibatasi ~1024 karakter untuk dokumen.
    safe_caption = caption[:1024]
    with open(file_path, "rb") as doc_file:
        data = aiohttp.FormData()
        data.add_field("chat_id", str(chat_id))
        data.add_field("caption", safe_caption)
        data.add_field("parse_mode", "Markdown")
        data.add_field(
            "document",
            doc_file,
            filename=os.path.basename(file_path),
            content_type="application/zip",
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as resp:
                await resp.read()


def zip_charts(
    chart_paths: list[str],
    summary_text: str | None = None,
    out_dir: str = "charts",
) -> str | None:
    """Kumpulkan semua chart hasil scan ke dalam 1 file .zip, plus file teks
    ringkasan.txt berisi detail tiap sinyal di dalam zip yang sama. Return
    path zip, atau None kalau tidak ada apa pun (chart maupun ringkasan)
    untuk di-zip."""
    if not chart_paths and not summary_text:
        return None

    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    zip_path = os.path.join(out_dir, f"scan_{timestamp}.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for chart_path in chart_paths:
            if os.path.exists(chart_path):
                zf.write(chart_path, arcname=os.path.basename(chart_path))
        if summary_text:
            zf.writestr("ringkasan.txt", summary_text)

    return zip_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


async def run_scan(cfg: dict, out_path: str) -> list[dict]:
    results = []
    captions: list[str] = []
    chart_paths: list[str] = []

    scan_cfg = cfg.get("scanning", {})
    state_path = scan_cfg.get("state_path", "signal_state.json")
    cooldown_hours = scan_cfg.get("cooldown_hours", 4)
    klines_limit = scan_cfg.get("klines_limit", 300)
    min_history_bars = scan_cfg.get("min_history_bars", 250)
    state = load_state(state_path)

    auto_generate_charts = cfg.get("chart", {}).get("auto_generate", True)
    timeframes = cfg["timeframes"]
    mtf_bonus_weight = cfg["scoring"]["weights"].get("mtf_agreement", 0)

    async with BinanceFuturesClient() as client:
        symbols = await client.get_active_symbols(cfg["exchange"]["quote_asset"])

        volumes = await asyncio.gather(*(client.get_24h_volume(s) for s in symbols))
        min_vol = cfg["exchange"]["min_volume_usdt_24h"]
        active_symbols = [s for s, v in zip(symbols, volumes) if v >= min_vol]

        # --- Rezim pasar makro (BTCUSDT) — dicek SEKALI di awal, dipakai
        # buat menyaring sinyal yang melawan arus utama di semua simbol.
        regime_cfg = cfg.get("regime_filter", {})
        regime = await get_market_regime(client, cfg) if regime_cfg.get("enabled", False) else "NEUTRAL"

        # --- Scan SETIAP timeframe secara independen ------------------------
        # Sebelumnya cuma timeframes[0] (15m) yang pernah menghasilkan sinyal;
        # 1h cuma jadi filter biner dan 4h sama sekali tidak pernah dipakai.
        # Sekarang tiap TF di config di-scan sendiri-sendiri dengan pipeline
        # yang identik, jadi setup bisa muncul dari TF manapun yang layak.
        per_tf_candidates: dict[str, list[tuple[SignalResult, Kline]]] = {}
        for tf in timeframes:
            klines = await client.get_klines_many(active_symbols, tf, limit=klines_limit)
            tf_candidates: list[tuple[SignalResult, Kline]] = []
            for kline in klines:
                closed_df = drop_unclosed_candle(kline.df)
                if len(closed_df) < min_history_bars:
                    # Histori terlalu pendek (mis. coin baru listing) -> EMA200/ADX
                    # belum konvergen, indikator trend-following jadi bias/tidak reliable.
                    continue
                signal = score_symbol(closed_df, kline.symbol, cfg, timeframe=tf)
                if signal.direction == "NONE":
                    continue
                if regime == "BULL" and signal.direction == "SHORT":
                    continue  # lawan arus BTC bull -> data menunjukkan ini secara sistematis buruk
                if regime == "BEAR" and signal.direction == "LONG":
                    continue
                if not passes_risk_filter(signal, cfg):
                    continue
                if is_in_cooldown(state, kline.symbol, signal.direction, tf, cooldown_hours):
                    continue
                tf_candidates.append((signal, kline))
            per_tf_candidates[tf] = tf_candidates

        # --- MTF agreement: sekarang jadi BONUS skor, bukan syarat lolos ----
        # (dulu: 1h WAJIB searah baru sinyal 15m lolos; kalau 1h netral,
        # sinyal 15m yang sebenarnya kuat ikut terbuang percuma.)
        direction_map: dict[str, dict[str, str]] = {}
        for tf, cands in per_tf_candidates.items():
            for signal, kline in cands:
                direction_map.setdefault(kline.symbol, {})[tf] = signal.direction

        flat_candidates: list[tuple[SignalResult, Kline]] = []
        for tf, cands in per_tf_candidates.items():
            for signal, kline in cands:
                agree_tfs = [
                    t for t, d in direction_map.get(kline.symbol, {}).items()
                    if t != tf and d == signal.direction
                ]
                if agree_tfs and mtf_bonus_weight:
                    bonus = mtf_bonus_weight * len(agree_tfs)
                    signal.score = round(signal.score + bonus, 1)
                    signal.reasons.append(
                        f"Searah dengan TF {', '.join(agree_tfs)} (+{bonus} MTF agreement)"
                    )
                flat_candidates.append((signal, kline))

        # --- Satu simbol -> satu setup terbaik lintas TF ---------------------
        # Supaya Top 5 tidak diborong 1 koin yang sinyalnya tumpang tindih
        # di beberapa TF sekaligus; yang dipilih adalah skor tertinggi.
        best_per_symbol: dict[str, tuple[SignalResult, Kline]] = {}
        for signal, kline in flat_candidates:
            current = best_per_symbol.get(kline.symbol)
            if current is None or signal.score > current[0].score:
                best_per_symbol[kline.symbol] = (signal, kline)

        candidates = sorted(best_per_symbol.values(), key=lambda c: c[0].score, reverse=True)
        top_candidates = candidates[: cfg["risk"]["max_signals_per_run"]]

        for signal, kline in top_candidates:
            results.append(signal.__dict__)
            mark_signaled(state, kline.symbol, signal.direction, signal.timeframe)

            captions.append(format_signal_message(signal))

            if auto_generate_charts:
                import chart as chart_module  # lazy import, hindari circular import

                os.makedirs("charts", exist_ok=True)
                chart_path = f"charts/{kline.symbol}_{signal.timeframe}.png"
                chart_module.build_chart(
                    kline.df,
                    kline.symbol,
                    signal.timeframe,
                    signal,
                    cfg,
                    chart_path,
                )
                chart_paths.append(chart_path)

    save_state(state_path, state)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # --- Kirim ke Telegram: cukup 1 file zip per scan, tanpa pesan teks terpisah ---
    summary = None
    if captions:
        summary = f"Scan selesai — {len(results)} sinyal ditemukan\n\n" + "\n\n".join(captions)

    zip_path = zip_charts(chart_paths, summary_text=summary)
    if zip_path:
        n_chart = len(chart_paths)
        caption = (
            f"{n_chart} chart sinyal (detail lengkap ada di ringkasan.txt dalam zip)"
            if n_chart
            else "Ringkasan sinyal (lihat ringkasan.txt dalam zip)"
        )
        await send_telegram_document(zip_path, caption=caption, cfg=cfg)

    return results


def main():
    parser = argparse.ArgumentParser(description="vSynapse v3.1 scanner (multi-timeframe)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="synaptic_candidates.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = asyncio.run(run_scan(cfg, args.out))
    print(f"Ditemukan {len(results)} sinyal. Disimpan ke {args.out}")


if __name__ == "__main__":
    main()
