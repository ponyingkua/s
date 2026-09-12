from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from scanner import BinanceFuturesClient, load_config, drop_unclosed_candle
from chart import _fetch_and_build_multi

OUT_DIR = "analysis_output"
FETCH_MAX_RETRIES = 3
FETCH_RETRY_BACKOFF_SECONDS = 1.5


def normalize_symbol(raw: str, quote_asset: str) -> str:
    s = raw.strip().upper()
    if not s:
        raise ValueError("Symbol cannot be empty")
    return s if s.endswith(quote_asset) else f"{s}{quote_asset}"


# ============================================================
# Dynamic Precision Rounding (Edge Case #2 Fix)
# ============================================================

def _round_price(value: float) -> float:
    if value == 0 or pd.isna(value):
        return 0.0
    abs_val = abs(value)
    if abs_val < 0.0001:
        digits = 8
    elif abs_val < 0.01:
        digits = 6
    elif abs_val < 1:
        digits = 4
    elif abs_val < 100:
        digits = 3
    else:
        digits = 2
    return round(float(value), digits)


# ============================================================
# Core Technical Indicators
# ============================================================

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    return out


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ef = series.ewm(span=fast, adjust=False).mean()
    es = series.ewm(span=slow, adjust=False).mean()
    line = ef - es
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def _bollinger(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    width = (upper - lower) / mid.replace(0, np.nan) * 100
    denom = upper - lower
    pb = pd.Series(np.where(denom == 0, 0.5, (series - lower) / denom), index=series.index)
    return mid, upper, lower, width, pb


def _obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff().fillna(0))
    return (direction * df["volume"]).cumsum()


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period).mean()


def _ema_slope(series: pd.Series, lookback: int = 5) -> float:
    if len(series) < lookback + 1 or pd.isna(series.iloc[-1]):
        return 0.0
    base = max(abs(float(series.iloc[-lookback])), 1e-12)
    return float(series.iloc[-1] - series.iloc[-lookback]) / base


# ============================================================
# Swing Points & Price Structure
# ============================================================

def _swing_points(df: pd.DataFrame, left: int = 3, right: int = 3):
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    sh, sl = [], []
    n = len(df)
    for i in range(left, n - right):
        if highs[i] == np.max(highs[i - left : i + right + 1]):
            sh.append((i, highs[i]))
        if lows[i] == np.min(lows[i - left : i + right + 1]):
            sl.append((i, lows[i]))
    return sh, sl


def _levels(df: pd.DataFrame, lookback: int = 100):
    x = df.tail(min(lookback, len(df)))
    sh, sl = _swing_points(x, 2, 2)
    resistance = max([v for _, v in sh[-5:]], default=float(x["high"].max()))
    support = min([v for _, v in sl[-5:]], default=float(x["low"].min()))
    return float(support), float(resistance)


def _structure_analysis(df: pd.DataFrame) -> dict:
    sh, sl = _swing_points(df, 3, 3)
    notes = []
    bull = bear = 0.0

    if len(sh) >= 2 and len(sl) >= 2:
        hh = sh[-1][1] > sh[-2][1]
        hl = sl[-1][1] > sl[-2][1]
        lh = sh[-1][1] < sh[-2][1]
        ll = sl[-1][1] < sl[-2][1]

        if hh and hl:
            bull += 3.0
            notes.append("Struktur market: Higher-High & Higher-Low konfirmasi tren naik.")
        elif lh and ll:
            bear += 3.0
            notes.append("Struktur market: Lower-High & Lower-Low konfirmasi tren turun.")
        elif hh or hl:
            bull += 1.4
            notes.append("Struktur cenderung bullish (HH/HL belum sempurna).")
        elif lh or ll:
            bear += 1.4
            notes.append("Struktur cenderung bearish (LH/LL belum sempurna).")

    support, resistance = _levels(df)
    prior_support, prior_resistance = _levels(df.iloc[:-1]) if len(df) > 25 else (support, resistance)
    last_close = float(df["close"].iloc[-1])
    last_high = float(df["high"].iloc[-1])
    last_low = float(df["low"].iloc[-1])

    if last_close > prior_resistance and last_high > prior_resistance:
        bull += 2.2
        notes.append(f"Harga bertahan di atas resistance terdahulu ({prior_resistance:.8g}).")
    elif last_close < prior_support and last_low < prior_support:
        bear += 2.2
        notes.append(f"Harga tertahan di bawah support terdahulu ({prior_support:.8g}).")

    total = bull + bear
    if total == 0:
        label = "NEUTRAL"
    elif bull / total >= 0.65:
        label = "BULLISH"
    elif bear / total >= 0.65:
        label = "BEARISH"
    else:
        label = "MIXED"

    return {
        "label": label,
        "bull": round(bull, 2),
        "bear": round(bear, 2),
        "support": support,
        "resistance": resistance,
        "notes": notes,
    }


# ============================================================
# Direction Analysis Engine
# ============================================================

def _direction_analysis(df: pd.DataFrame) -> dict:
    close, volume = df["close"], df["volume"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean() if len(df) >= 200 else None
    
    rsi = _rsi(close)
    _, _, hist = _macd(close)
    _, _, _, bb_width, pb = _bollinger(close)
    obv = _obv(df)
    atr = _atr(df)

    bull = bear = 0.0
    notes = []
    p = float(close.iloc[-1])

    # Analisis Tren EMA
    trend = 0.0
    e20, e50 = float(ema20.iloc[-1]), float(ema50.iloc[-1])
    trend += 1.5 if e20 > e50 else -1.5 if e20 < e50 else 0
    s20, s50 = _ema_slope(ema20), _ema_slope(ema50)
    trend += 1.0 if s20 > 0 and s50 > 0 else -1.0 if s20 < 0 and s50 < 0 else 0
    
    if ema200 is not None and pd.notna(ema200.iloc[-1]):
        trend += 1.3 if p > float(ema200.iloc[-1]) else -1.3

    if trend > 0:
        bull += min(trend, 3.0)
        notes.append("Struktur EMA mendukung posisi beli.")
    elif trend < 0:
        bear += min(abs(trend), 3.0)
        notes.append("Struktur EMA mendukung posisi jual.")

    # Analisis RSI
    r = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None
    if r is not None:
        if 52 <= r < 68:
            bull += 1.2
            notes.append(f"RSI ({r:.1f}) mendukung momentum bullish.")
        elif r >= 68:
            bull += 0.4
            notes.append(f"RSI ({r:.1f}) memasuki area overbought.")
        elif 32 < r <= 48:
            bear += 1.2
            notes.append(f"RSI ({r:.1f}) mendukung momentum bearish.")
        elif r <= 32:
            bear += 0.4
            notes.append(f"RSI ({r:.1f}) memasuki area oversold.")

    # Analisis MACD
    h = float(hist.iloc[-1]) if pd.notna(hist.iloc[-1]) else None
    hp = float(hist.iloc[-2]) if pd.notna(hist.iloc[-2]) else None
    if h is not None and hp is not None:
        if h > 0 and h > hp:
            bull += 1.3
            notes.append("Histogram MACD positif dan melebarkan penguatan.")
        elif h < 0 and h < hp:
            bear += 1.3
            notes.append("Histogram MACD negatif dan melebarkan pelemahan.")
        elif h > 0:
            bull += 0.5
        elif h < 0:
            bear += 0.5

    # Analisis Volume
    if len(volume) >= 20:
        vma = float(volume.rolling(20).mean().iloc[-1])
        vr = float(volume.iloc[-1]) / vma if vma > 0 else 0
        up = float(close.iloc[-1]) > float(close.iloc[-2])
        if vr >= 1.4:
            if up:
                bull += 1.0
                notes.append(f"Lonjakan volume mengonfirmasi kenaikan ({vr:.1f}x rata-rata).")
            else:
                bear += 1.0
                notes.append(f"Lonjakan volume mengonfirmasi penurunan ({vr:.1f}x rata-rata).")

    # Analisis OBV
    if len(obv) >= 12:
        delta = float(obv.iloc[-1] - obv.iloc[-12])
        if delta > 0:
            bull += 0.7
        elif delta < 0:
            bear += 0.7

    total = bull + bear
    if total == 0:
        bias = "NEUTRAL"
    else:
        ratio = bull / total
        if ratio >= 0.68:
            bias = "BULLISH STRONG"
        elif ratio >= 0.55:
            bias = "BULLISH"
        elif ratio <= 0.32:
            bias = "BEARISH STRONG"
        elif ratio <= 0.45:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL / MIXED"

    return {
        "bias": bias,
        "bull": round(bull, 2),
        "bear": round(bear, 2),
        "rsi": r,
        "atr": float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else None,
        "atr_pct": float(atr.iloc[-1] / p * 100) if pd.notna(atr.iloc[-1]) and p else None,
        "bb_width_pct": float(bb_width.iloc[-1]) if pd.notna(bb_width.iloc[-1]) else None,
        "notes": notes,
    }


# ============================================================
# Setup Detection Logic (Edge Case #1 Fix)
# ============================================================

def _detect_setup(df: pd.DataFrame, structure: dict, direction: dict) -> dict:
    p = float(df["close"].iloc[-1])
    atr = direction["atr"] or p * 0.01
    support, resistance = structure["support"], structure["resistance"]
    bull_bias = direction["bull"] > direction["bear"]
    bear_bias = direction["bear"] > direction["bull"]
    notes = []
    setup, score = "NONE", 0.0

    last = df.iloc[-1]
    # Guard: Minimal range tidak boleh 0 atau berharga ekstrem kecil (mencegah ZeroDivision)
    rng = max(float(last.high - last.low), p * 0.0005)
    body_ratio = abs(float(last.close - last.open)) / rng

    if bull_bias and p > resistance and body_ratio >= 0.40:
        setup, score = "BREAKOUT", 3.0
        notes.append("Breakout bullish terkonfirmasi dengan candle solid.")
    elif bear_bias and p < support and body_ratio >= 0.40:
        setup, score = "BREAKDOWN", 3.0
        notes.append("Breakdown bearish terkonfirmasi dengan candle solid.")
    else:
        if bull_bias and abs(p - support) <= max(1.2 * atr, p * 0.008):
            setup, score = "PULLBACK / RETEST", 2.6
            notes.append("Harga melakukan retest ke area support dalam tren naik.")
        elif bear_bias and abs(p - resistance) <= max(1.2 * atr, p * 0.008):
            setup, score = "PULLBACK / RETEST", 2.6
            notes.append("Harga melakukan retest ke area resistance dalam tren turun.")

        upper_wick = float(last.high - max(last.open, last.close))
        lower_wick = float(min(last.open, last.close) - last.low)
        if bear_bias and (upper_wick / rng) >= 0.40 and last.high >= resistance:
            setup, score = "REJECTION", max(score, 2.5)
            notes.append("Penolakan ekor atas (wick rejection) terlihat di area resistance.")
        elif bull_bias and (lower_wick / rng) >= 0.40 and last.low <= support:
            setup, score = "REJECTION", max(score, 2.5)
            notes.append("Penolakan ekor bawah (wick rejection) terlihat di area support.")

    if setup == "NONE":
        if bull_bias and structure["label"] == "BULLISH":
            setup, score = "CONTINUATION", 2.0
            notes.append("Struktur dan tren sejalan mendukung kelanjutan tren naik.")
        elif bear_bias and structure["label"] == "BEARISH":
            setup, score = "CONTINUATION", 2.0
            notes.append("Struktur dan tren sejalan mendukung kelanjutan tren turun.")

    quality = "HIGH" if score >= 2.8 else "MEDIUM" if score >= 2.0 else "LOW"
    return {"type": setup, "quality": quality, "score": score, "notes": notes}


# ============================================================
# Dynamic Level Calculator (Edge Case #3 Fix)
# ============================================================

def _build_levels(df: pd.DataFrame, direction: dict, structure: dict, setup: dict) -> dict:
    p = float(df["close"].iloc[-1])
    atr = direction["atr"] or (p * 0.01)
    support, resistance = structure["support"], structure["resistance"]
    is_long = direction["bull"] > direction["bear"]
    is_short = direction["bear"] > direction["bull"]

    if not (is_long or is_short) or setup["type"] == "NONE":
        return {"direction": "NONE", "entry": None, "sl": None, "tp1": None, "tp2": None}

    if is_long:
        entry_lo = _round_price(min(p, p - 0.2 * atr))
        entry_hi = _round_price(max(p, p + 0.1 * atr))
        
        anchor_sl = support if (support is not None and support < p) else (p - atr)
        sl = _round_price(min(anchor_sl - 0.2 * atr, p - 1.2 * atr))
        
        # Guard: Pastikan SL selalu di bawah entry_lo
        if sl >= entry_lo:
            sl = _round_price(entry_lo - 1.2 * atr)
            
        risk = max(entry_hi - sl, p * 0.005)
        tp1 = _round_price(entry_hi + (1.5 * risk))
        tp2 = _round_price(entry_hi + (2.8 * risk))

        # Guard: Pastikan TP2 selalu lebih tinggi dari TP1
        if tp2 <= tp1:
            tp2 = _round_price(tp1 + (0.5 * risk))

        return {
            "direction": "LONG",
            "entry": (entry_lo, entry_hi),
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
        }

    else:
        entry_hi = _round_price(max(p, p + 0.2 * atr))
        entry_lo = _round_price(min(p, p - 0.1 * atr))
        
        anchor_sl = resistance if (resistance is not None and resistance > p) else (p + atr)
        sl = _round_price(max(anchor_sl + 0.2 * atr, p + 1.2 * atr))
        
        # Guard: Pastikan SL selalu di atas entry_hi
        if sl <= entry_hi:
            sl = _round_price(entry_hi + 1.2 * atr)
            
        risk = max(sl - entry_lo, p * 0.005)
        tp1 = _round_price(entry_lo - (1.5 * risk))
        tp2 = _round_price(entry_lo - (2.8 * risk))

        # Guard: Pastikan TP2 selalu lebih rendah dari TP1
        if tp2 >= tp1:
            tp2 = _round_price(tp1 - (0.5 * risk))

        return {
            "direction": "SHORT",
            "entry": (entry_lo, entry_hi),
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
        }


def analyze_timeframe(df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
    direction = _direction_analysis(df)
    structure = _structure_analysis(df)
    setup = _detect_setup(df, structure, direction)

    conf_bull_strong = direction["bull"] - direction["bear"] >= 1.5
    conf_bear_strong = direction["bear"] - direction["bull"] >= 1.5
    divergence_notes = []

    if structure["label"] == "BULLISH" and conf_bear_strong:
        final = "NONE"
        divergence_notes.append(
            "Divergensi: Struktur HH/HL Bullish, tetapi indikator momentum sangat Bearish. Posisi diabaikan."
        )
    elif structure["label"] == "BEARISH" and conf_bull_strong:
        final = "NONE"
        divergence_notes.append(
            "Divergensi: Struktur LH/LL Bearish, tetapi indikator momentum sangat Bullish. Posisi diabaikan."
        )
    elif structure["label"] == "BULLISH" and direction["bull"] >= direction["bear"]:
        final = "LONG"
    elif structure["label"] == "BEARISH" and direction["bear"] >= direction["bull"]:
        final = "SHORT"
    elif conf_bull_strong:
        final = "LONG"
    elif conf_bear_strong:
        final = "SHORT"
    else:
        final = "NONE"

    levels = _build_levels(df, direction, structure, setup) if final != "NONE" else {
        "direction": "NONE", "entry": None, "sl": None, "tp1": None, "tp2": None
    }

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "direction": final,
        "bias": direction["bias"],
        "structure": structure,
        "direction_analysis": direction,
        "setup": setup,
        "levels": levels,
        "divergence_notes": divergence_notes,
    }


def _select_best_setup(per_tf: dict):
    candidates = []
    for tf, info in per_tf.items():
        if "error" in info or info.get("direction") == "NONE":
            continue
        da, su = info["direction_analysis"], info["setup"]
        strength = abs(da["bull"] - da["bear"])
        agree = len(info.get("mtf_agree_tfs", []))
        candidates.append(((su["score"], strength, agree), tf, info))

    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0], reverse=True)
    _, best_tf, best_info = candidates[0]
    return best_tf, best_info


async def _fetch_tf(client: "BinanceFuturesClient", symbol: str, tf: str, klines_limit: int):
    last_err = None
    for attempt in range(FETCH_MAX_RETRIES):
        try:
            kline = await client.get_klines(symbol, tf, limit=klines_limit)
            df = drop_unclosed_candle(kline.df)
            if len(df) < 60:
                return tf, None, f"Hanya ada {len(df)} candle tertutup; minimal butuh 60 candle."
            return tf, df, None
        except Exception as exc:
            last_err = exc
            if attempt < FETCH_MAX_RETRIES - 1:
                delay = FETCH_RETRY_BACKOFF_SECONDS * (2 ** attempt)
                print(
                    f"[warn] {symbol} {tf}: Fetch gagal ({exc}); mencoba ulang dalam "
                    f"{delay:.1f}s (percobaan {attempt + 1}/{FETCH_MAX_RETRIES})"
                )
                await asyncio.sleep(delay)
    print(f"[error] Gagal mengambil data {symbol} {tf} setelah {FETCH_MAX_RETRIES} percobaan: {last_err}")
    return tf, None, str(last_err)


async def analyze_symbol(symbol: str, cfg: dict) -> dict:
    data_cfg = cfg.get("scanning", {})
    klines_limit = int(data_cfg.get("klines_limit", 300))
    timeframes = cfg.get("timeframes", ["1h"])
    per_tf = {}
    dfs = {}

    async with BinanceFuturesClient() as client:
        results = await asyncio.gather(
            *(_fetch_tf(client, symbol, tf, klines_limit) for tf in timeframes),
            return_exceptions=True,
        )

    for tf, res in zip(timeframes, results):
        if isinstance(res, Exception):
            print(f"[error] Gagal menganalisis {symbol} {tf}: {res}")
            per_tf[tf] = {"error": str(res)}
            continue
        _, df, err = res
        if err is not None:
            per_tf[tf] = {"error": err}
            continue
        try:
            dfs[tf] = df
            info = analyze_timeframe(df, symbol, tf)
            info["mtf_agree_tfs"] = []
            per_tf[tf] = info
        except Exception as exc:
            print(f"[error] Gagal menganalisis {symbol} {tf}: {exc}")
            per_tf[tf] = {"error": str(exc)}

    directions = {tf: x["direction"] for tf, x in per_tf.items() if "direction" in x and x["direction"] != "NONE"}
    for tf, info in per_tf.items():
        if "direction" in info and info["direction"] != "NONE":
            info["mtf_agree_tfs"] = [t for t, d in directions.items() if t != tf and d == info["direction"]]

    best = _select_best_setup(per_tf)
    return {
        "symbol": symbol,
        "per_tf": per_tf,
        "dfs": dfs,
        "best_tf": best[0] if best else None,
        "best": best[1] if best else None,
    }


def _mtf_alignment(per_tf: dict) -> dict:
    actionable = [(tf, x["direction"]) for tf, x in per_tf.items() if "direction" in x and x["direction"] != "NONE"]
    longs = sum(d == "LONG" for _, d in actionable)
    shorts = sum(d == "SHORT" for _, d in actionable)
    overall = None
    if actionable:
        overall = "bullish" if longs > shorts else "bearish" if shorts > longs else "mixed"
    return {"longs": longs, "shorts": shorts, "overall": overall}


def compose_analysis_text(result: dict) -> str:
    symbol = result["symbol"]
    per_tf = result["per_tf"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Analisis Teknikal Independen: {symbol}",
        f"Waktu: {now}",
        "",
        "Analisis ini berjalan mandiri, bebas dari sistem skoring atau filter bawaan scanner.",
        "",
    ]

    best_tf, best_info = result.get("best_tf"), result.get("best")
    lines.append("## 🎯 Best Setup")
    if best_info is None:
        lines.append("Tidak ditemukan setup yang aman/actionable di semua timeframe (semua NONE).")
    else:
        su, lv = best_info["setup"], best_info["levels"]
        lo, hi = lv["entry"]
        lines.append(f"Timeframe: {best_tf}")
        lines.append(f"Arah: {best_info['direction']}")
        lines.append(f"Setup: {su['type']} ({su['quality']}, Skor {su['score']})")
        lines.append(f"Area Entry: {lo} - {hi}")
        lines.append(f"Stop Loss (SL): {lv['sl']}")
        lines.append(f"Take Profit 1 (TP1): {lv['tp1']}")
        lines.append(f"Take Profit 2 (TP2): {lv['tp2']}")
        if best_info.get("mtf_agree_tfs"):
            lines.append(f"Konfirmasi TF Lain: {', '.join(best_info['mtf_agree_tfs'])}")
    lines.append("")
    lines.append("## Rincian Per-Timeframe")
    lines.append("")

    for tf, info in per_tf.items():
        lines.append(f"## {tf}")
        if "error" in info:
            lines += [f"Error Analisis: {info['error']}", ""]
            continue

        da, st, su, lv = info["direction_analysis"], info["structure"], info["setup"], info["levels"]
        lines.append(f"Arah Signal: {info['direction']}")
        lines.append(f"Bias Teknikal: {info['bias']}")
        lines.append(f"Struktur Price Action: {st['label']}")
        lines.append(f"Skor Konfluens: Bullish {da['bull']} / Bearish {da['bear']}")
        if da["rsi"] is not None:
            lines.append(f"RSI: {da['rsi']:.1f}")
        if da["atr_pct"] is not None:
            lines.append(f"ATR Volatilitas: {da['atr_pct']:.2f}%")
        if da["bb_width_pct"] is not None:
            lines.append(f"Lebar Bollinger: {da['bb_width_pct']:.2f}%")
        for note in st["notes"] + da["notes"] + su["notes"] + info.get("divergence_notes", []):
            lines.append(f"- {note}")
        lines.append(f"Setup Terbaik: {su['type']} ({su['quality']})")

        if lv["direction"] != "NONE":
            lo, hi = lv["entry"]
            lines.append(f"Area Entry: {lo} - {hi}")
            lines.append(f"Stop Loss: {lv['sl']}")
            lines.append(f"TP1: {lv['tp1']}")
            lines.append(f"TP2: {lv['tp2']}")
        if info.get("mtf_agree_tfs"):
            lines.append(f"Konfirmasi MTF: {', '.join(info['mtf_agree_tfs'])}")
        lines.append("")

    mtf = _mtf_alignment(per_tf)
    lines.append("## Ringkasan Multi-Timeframe (MTF)")
    if mtf["overall"] is None:
        lines.append("Tidak ada keselarasan arah yang cukup kuat antar timeframe.")
    else:
        lines.append(f"LONG: {mtf['longs']} timeframe")
        lines.append(f"SHORT: {mtf['shorts']} timeframe")
        lines.append(f"Kesimpulan MTF: {mtf['overall'].upper()}")
    return "\n".join(lines)


def compose_console_summary(result: dict) -> str:
    symbol = result["symbol"]
    per_tf = result["per_tf"]
    lines = [f"[analyze] Ringkasan {symbol}:"]

    best_tf, best_info = result.get("best_tf"), result.get("best")
    if best_info is None:
        lines.append("[analyze]   SETUP TERBAIK: Tidak ada (semua TF Netral/Divergen)")
    else:
        su, lv = best_info["setup"], best_info["levels"]
        lo, hi = lv["entry"]
        lines.append(
            f"[analyze]   SETUP TERBAIK -> {best_tf}: {best_info['direction']} | "
            f"{su['type']} ({su['quality']}) | Entry {lo}-{hi} | SL {lv['sl']} | "
            f"TP1 {lv['tp1']} | TP2 {lv['tp2']}"
        )

    for tf, info in per_tf.items():
        if "error" in info:
            lines.append(f"[analyze]   {tf}: ERROR - {info['error']}")
            continue
        su, lv = info["setup"], info["levels"]
        line = f"[analyze]   {tf}: {info['direction']} | {info['bias']} | {su['type']} ({su['quality']})"
        if lv["direction"] != "NONE":
            lo, hi = lv["entry"]
            line += f" | Entry {lo}-{hi} SL {lv['sl']} TP1 {lv['tp1']} TP2 {lv['tp2']}"
        if info.get("divergence_notes"):
            line += " | ⚠ DIVERGENSI (Struktur vs Momentum)"
        lines.append(line)

    mtf = _mtf_alignment(per_tf)
    if mtf["overall"] is None:
        lines.append("[analyze]   MTF: Tidak ada penyesuaian tren yang jelas")
    else:
        lines.append(f"[analyze]   MTF: {mtf['longs']} LONG / {mtf['shorts']} SHORT -> {mtf['overall'].upper()}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="vSynapse — Engine Analisis MTF Independen")
    parser.add_argument("--symbol", required=True, help="Kode koin, contoh: ZEC atau ZECUSDT")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    symbol = normalize_symbol(args.symbol, cfg["exchange"]["quote_asset"])
    timeframes = cfg.get("timeframes", ["1h"])

    print(f"[analyze] Memulai analisis independen untuk {symbol}")
    print(f"[analyze] Timeframe yang diproses: {', '.join(timeframes)}")
    result = asyncio.run(analyze_symbol(symbol, cfg))
    text = compose_analysis_text(result)
    print(compose_console_summary(result))

    os.makedirs(OUT_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(OUT_DIR, f"analysis_{result['symbol']}_{timestamp}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[analyze] Selesai. Laporan Markdown disimpan ke {md_path}")

    chart_path = os.path.join(OUT_DIR, f"{symbol}_multi.png")
    try:
        asyncio.run(_fetch_and_build_multi(symbol, timeframes, cfg, chart_path, notify=False))
        print(f"[analyze] Gambar chart MTF disimpan ke {chart_path}")
    except Exception as exc:
        print(f"[warn] Gagal membuat gambar chart MTF {symbol}: {exc}")

    per_tf = result["per_tf"]
    if per_tf and all("error" in info for info in per_tf.values()):
        print(f"[analyze] Semua {len(per_tf)} timeframe gagal di-fetch/analisis. Exit non-zero.")
        sys.exit(1)


if __name__ == "__main__":
    main()
