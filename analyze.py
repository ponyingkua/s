from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Infrastructure only. The analysis engine below does NOT use scanner scoring,
# scanner filters, scanner setup classification, or market-regime decisions.
from scanner import BinanceFuturesClient, load_config, drop_unclosed_candle

# Dipakai apa adanya dari chart.py untuk generate chart MTF otomatis setelah
# analisa selesai -- tidak ada logika chart.py yang diubah/diduplikasi di sini.
from chart import _fetch_and_build_multi

OUT_DIR = "analysis_output"

# Retry/backoff untuk fetch klines per timeframe -- lihat _fetch_tf().
# Backoff eksponensial: percobaan ke-n (0-indexed) menunggu
# FETCH_RETRY_BACKOFF_SECONDS * 2**n sebelum retry berikutnya.
FETCH_MAX_RETRIES = 3
FETCH_RETRY_BACKOFF_SECONDS = 1.5


def normalize_symbol(raw: str, quote_asset: str) -> str:
    s = raw.strip().upper()
    if not s:
        raise ValueError("Symbol cannot be empty")
    return s if s.endswith(quote_asset) else f"{s}{quote_asset}"


# ============================================================
# Indicators
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
# Price structure
# ============================================================

def _swing_points(df: pd.DataFrame, left: int = 3, right: int = 3):
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    sh, sl = [], []
    for i in range(left, len(df) - right):
        if highs[i] >= np.max(highs[i-left:i+right+1]):
            sh.append((i, highs[i]))
        if lows[i] <= np.min(lows[i-left:i+right+1]):
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
            notes.append("Higher-high / higher-low structure is intact.")
        elif lh and ll:
            bear += 3.0
            notes.append("Lower-high / lower-low structure is intact.")
        elif hh or hl:
            bull += 1.4
            notes.append("Recent swing structure leans bullish but is not fully aligned.")
        elif lh or ll:
            bear += 1.4
            notes.append("Recent swing structure leans bearish but is not fully aligned.")

    support, resistance = _levels(df)
    prior_support, prior_resistance = _levels(df.iloc[:-1]) if len(df) > 25 else (support, resistance)
    last_close = float(df["close"].iloc[-1])
    last_high = float(df["high"].iloc[-1])
    last_low = float(df["low"].iloc[-1])

    if last_close > prior_resistance and last_high > prior_resistance:
        bull += 2.2
        notes.append(f"Price is holding above prior resistance ({prior_resistance:.8g}).")
    elif last_close < prior_support and last_low < prior_support:
        bear += 2.2
        notes.append(f"Price is holding below prior support ({prior_support:.8g}).")

    total = bull + bear
    if total == 0:
        label = "NEUTRAL"
    elif bull / total >= 0.67:
        label = "BULLISH"
    elif bear / total >= 0.67:
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
# Independent direction/confluence engine
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

    # Trend is deliberately one coherent component to avoid double-counting EMA facts.
    trend = 0.0
    e20, e50 = float(ema20.iloc[-1]), float(ema50.iloc[-1])
    trend += 1.5 if e20 > e50 else -1.5 if e20 < e50 else 0
    s20, s50 = _ema_slope(ema20), _ema_slope(ema50)
    trend += 1.0 if s20 > 0 and s50 > 0 else -1.0 if s20 < 0 and s50 < 0 else 0
    if ema200 is not None and pd.notna(ema200.iloc[-1]):
        trend += 1.3 if p > float(ema200.iloc[-1]) else -1.3
    else:
        notes.append(
            f"EMA200 not available (only {len(df)} candles of history); "
            f"trend score uses EMA20/EMA50 alignment only."
        )
    if trend > 0:
        bull += min(trend, 3.0)
        notes.append("Trend structure favors buyers.")
    elif trend < 0:
        bear += min(abs(trend), 3.0)
        notes.append("Trend structure favors sellers.")

    r = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None
    if r is not None:
        if 55 <= r < 70:
            bull += 1.2
            notes.append(f"RSI {r:.1f} supports bullish momentum.")
        elif r >= 70:
            bull += 0.4
            notes.append(f"RSI {r:.1f} shows strong momentum with elevated extension risk.")
        elif 30 < r <= 45:
            bear += 1.2
            notes.append(f"RSI {r:.1f} favors bearish momentum.")
        elif r <= 30:
            bear += 0.4
            notes.append(f"RSI {r:.1f} is deeply weak, increasing bounce risk.")
        else:
            notes.append(f"RSI {r:.1f} is transitional.")

    h = float(hist.iloc[-1]) if pd.notna(hist.iloc[-1]) else None
    hp = float(hist.iloc[-2]) if pd.notna(hist.iloc[-2]) else None
    if h is not None and hp is not None:
        if h > 0 and h > hp:
            bull += 1.3
            notes.append("MACD histogram is positive and expanding.")
        elif h < 0 and h < hp:
            bear += 1.3
            notes.append("MACD histogram is negative and expanding.")
        elif h > 0:
            bull += 0.5
        elif h < 0:
            bear += 0.5

    pb_now = float(pb.iloc[-1]) if pd.notna(pb.iloc[-1]) else None
    if pb_now is not None and h is not None:
        if pb_now > 1 and h > 0:
            bull += 0.6
            notes.append("Price is pressing above the upper Bollinger area with positive momentum.")
        elif pb_now < 0 and h < 0:
            bear += 0.6
            notes.append("Price is pressing below the lower Bollinger area with negative momentum.")

    if len(volume) >= 20:
        vma = float(volume.rolling(20).mean().iloc[-1])
        vr = float(volume.iloc[-1]) / vma if vma > 0 else 0
        up = float(close.iloc[-1]) > float(close.iloc[-2])
        if vr >= 1.5:
            if up:
                bull += 1.0
                notes.append(f"Volume expansion confirms the latest upward candle ({vr:.1f}x average).")
            else:
                bear += 1.0
                notes.append(f"Volume expansion confirms the latest downward candle ({vr:.1f}x average).")
        elif vr >= 1.15:
            bull += 0.35 if up else 0
            bear += 0.35 if not up else 0

    if len(obv) >= 12:
        delta = float(obv.iloc[-1] - obv.iloc[-12])
        if delta > 0:
            bull += 0.7
            notes.append("OBV is rising across the recent window.")
        elif delta < 0:
            bear += 0.7
            notes.append("OBV is falling across the recent window.")

    total = bull + bear
    if total == 0:
        bias = "NEUTRAL"
    else:
        ratio = bull / total
        if ratio >= 0.70:
            bias = "BULLISH STRONG"
        elif ratio >= 0.57:
            bias = "BULLISH"
        elif ratio <= 0.30:
            bias = "BEARISH STRONG"
        elif ratio <= 0.43:
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
# Setup selection
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
    rng = max(float(last.high - last.low), 1e-12)
    body_ratio = abs(float(last.close - last.open)) / rng

    if bull_bias and p > resistance and body_ratio >= 0.45:
        setup, score = "BREAKOUT", 3.0
        notes.append("Bullish breakout has a decisive candle behind it.")
    elif bear_bias and p < support and body_ratio >= 0.45:
        setup, score = "BREAKDOWN", 3.0
        notes.append("Bearish breakdown has a decisive candle behind it.")
    else:
        if bull_bias and abs(p - support) <= max(1.2 * atr, p * 0.006):
            setup, score = "PULLBACK / RETEST", 2.6
            notes.append("Price is close to structural support while directional pressure remains bullish.")
        elif bear_bias and abs(p - resistance) <= max(1.2 * atr, p * 0.006):
            setup, score = "PULLBACK / RETEST", 2.6
            notes.append("Price is close to structural resistance while directional pressure remains bearish.")

        upper_wick = float(last.high - max(last.open, last.close))
        lower_wick = float(min(last.open, last.close) - last.low)
        if bear_bias and upper_wick / rng >= 0.45 and last.high >= resistance:
            setup, score = "REJECTION", max(score, 2.5)
            notes.append("Upper-wick rejection is visible at structural resistance.")
        elif bull_bias and lower_wick / rng >= 0.45 and last.low <= support:
            setup, score = "REJECTION", max(score, 2.5)
            notes.append("Lower-wick rejection is visible at structural support.")

    if setup == "NONE":
        if bull_bias and structure["label"] == "BULLISH":
            setup, score = "CONTINUATION", 2.0
            notes.append("Trend and structure remain aligned for continuation.")
        elif bear_bias and structure["label"] == "BEARISH":
            setup, score = "CONTINUATION", 2.0
            notes.append("Trend and structure remain aligned for continuation.")

    quality = "HIGH" if score >= 2.8 else "MEDIUM" if score >= 2.0 else "LOW"
    return {"type": setup, "quality": quality, "score": score, "notes": notes}


# ============================================================
# Levels: structure + volatility, not scanner RR
# ============================================================

def _round_price(value: float) -> float:
    if value == 0:
        return 0.0
    digits = max(2, int(6 - np.floor(np.log10(abs(value)))))
    return round(float(value), digits)


def _build_levels(df: pd.DataFrame, direction: dict, structure: dict, setup: dict) -> dict:
    p = float(df["close"].iloc[-1])
    atr = direction["atr"] or p * 0.01
    support, resistance = structure["support"], structure["resistance"]
    is_long = direction["bull"] > direction["bear"]
    is_short = direction["bear"] > direction["bull"]

    if not (is_long or is_short) or setup["type"] == "NONE":
        return {"direction": "NONE", "entry": None, "sl": None, "tp1": None, "tp2": None}

    if is_long:
        anchor = support if support is not None and support < p else p - atr
        sl = min(anchor - 0.25 * atr, p - atr)
        lo = min(p, max(anchor, p - 0.45 * atr))
        hi = max(p, lo + 0.15 * atr)
        risk = max(hi - sl, 0.25 * atr)
        return {
            "direction": "LONG",
            "entry": (_round_price(lo), _round_price(hi)),
            "sl": _round_price(sl),
            "tp1": _round_price(hi + risk),
            "tp2": _round_price(hi + 1.8 * risk),
        }

    anchor = resistance if resistance is not None and resistance > p else p + atr
    sl = max(anchor + 0.25 * atr, p + atr)
    hi = max(p, min(anchor, p + 0.45 * atr))
    lo = min(p, hi - 0.15 * atr)
    risk = max(sl - lo, 0.25 * atr)
    return {
        "direction": "SHORT",
        "entry": (_round_price(lo), _round_price(hi)),
        "sl": _round_price(sl),
        "tp1": _round_price(lo - risk),
        "tp2": _round_price(lo - 1.8 * risk),
    }


def analyze_timeframe(df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
    direction = _direction_analysis(df)
    structure = _structure_analysis(df)
    setup = _detect_setup(df, structure, direction)

    conf_bull_strong = direction["bull"] - direction["bear"] >= 1.5
    conf_bear_strong = direction["bear"] - direction["bull"] >= 1.5
    divergence_notes = []

    # Structure has priority. Indicator confluence confirms rather than
    # gates -- but confluence is never allowed to flip the call to the
    # OPPOSITE side of an intact structure (e.g. HH/HL still intact yet
    # RSI/MACD/volume/OBV are bearish-strong). That combination is a
    # structure/momentum divergence: flagged explicitly and treated as NOT
    # actionable (NONE) rather than silently emitting a signal against the
    # very structure that was just confirmed.
    if structure["label"] == "BULLISH" and conf_bear_strong:
        final = "NONE"
        divergence_notes.append(
            f"Divergence: structure is BULLISH (HH/HL intact) but indicator "
            f"confluence is bearish-strong (bull {direction['bull']} / bear "
            f"{direction['bear']}); not treated as an actionable SHORT."
        )
    elif structure["label"] == "BEARISH" and conf_bull_strong:
        final = "NONE"
        divergence_notes.append(
            f"Divergence: structure is BEARISH (LH/LL intact) but indicator "
            f"confluence is bullish-strong (bull {direction['bull']} / bear "
            f"{direction['bear']}); not treated as an actionable LONG."
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


# ============================================================
# Best-setup selection (single verdict across all analyzed TFs)
# ============================================================

def _select_best_setup(per_tf: dict):
    """Pilih SATU setup terbaik dari seluruh timeframe yang diminta user.
    Timeframe dengan error fetch atau direction NONE tidak diikutkan.
    Ranking (berurutan): skor kualitas setup dari _detect_setup, lalu
    kekuatan confluence (selisih bull/bear direction_analysis), lalu jumlah
    timeframe lain yang searah (mtf_agree_tfs) sebagai tie-breaker terakhir.
    Return (timeframe, info) milik pemenang, atau None kalau tidak ada
    satupun timeframe yang actionable."""
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
    """Fetch & validasi satu timeframe. Retry dengan backoff eksponensial
    (FETCH_MAX_RETRIES percobaan) untuk kegagalan transient (timeout, rate
    limit dari Binance) -- percobaan terakhir yang gagal dilaporkan apa
    adanya. Tidak pernah melempar exception ke caller: sukses maupun gagal
    sama-sama dikembalikan sebagai (tf, df, error), df/error salah satunya
    None, supaya analyze_symbol bisa memproses hasil tiap timeframe secara
    seragam setelah asyncio.gather selesai."""
    last_err = None
    for attempt in range(FETCH_MAX_RETRIES):
        try:
            kline = await client.get_klines(symbol, tf, limit=klines_limit)
            df = drop_unclosed_candle(kline.df)
            if len(df) < 60:
                return tf, None, f"Only {len(df)} closed bars available; minimum 60 required."
            return tf, df, None
        except Exception as exc:
            last_err = exc
            if attempt < FETCH_MAX_RETRIES - 1:
                delay = FETCH_RETRY_BACKOFF_SECONDS * (2 ** attempt)
                print(
                    f"[warn] {symbol} {tf}: fetch failed ({exc}); retrying in "
                    f"{delay:.1f}s (attempt {attempt + 1}/{FETCH_MAX_RETRIES})"
                )
                await asyncio.sleep(delay)
    print(f"[error] Failed to fetch {symbol} {tf} after {FETCH_MAX_RETRIES} attempts: {last_err}")
    return tf, None, str(last_err)


async def analyze_symbol(symbol: str, cfg: dict) -> dict:
    # Only data-fetch settings are borrowed from config. No scanner thresholds,
    # filters, scores, or regime decisions are consulted.
    data_cfg = cfg.get("scanning", {})
    klines_limit = int(data_cfg.get("klines_limit", 300))
    timeframes = cfg.get("timeframes", ["1h"])
    per_tf = {}
    dfs = {}

    async with BinanceFuturesClient() as client:
        # Tiap timeframe independen satu sama lain (tidak ada dependency),
        # jadi menunggu satu-satu di sini murni overhead -- di-fetch paralel
        # lewat asyncio.gather. return_exceptions=True adalah jaring pengaman
        # tambahan; _fetch_tf sendiri sudah menangkap semua exception-nya
        # secara internal dan tidak pernah melempar keluar.
        results = await asyncio.gather(
            *(_fetch_tf(client, symbol, tf, klines_limit) for tf in timeframes),
            return_exceptions=True,
        )

    for tf, res in zip(timeframes, results):
        if isinstance(res, Exception):
            print(f"[error] Failed to analyze {symbol} {tf}: {res}")
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
            print(f"[error] Failed to analyze {symbol} {tf}: {exc}")
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
        f"# Independent Technical Analysis: {symbol}",
        f"Time: {now}",
        "",
        "Analysis is independent of scanner filters, scanner scores, and scanner setup classification.",
        "",
    ]

    best_tf, best_info = result.get("best_tf"), result.get("best")
    lines.append("## 🎯 Best Setup")
    if best_info is None:
        lines.append("No actionable setup was found on any analyzed timeframe (all NONE).")
    else:
        su, lv = best_info["setup"], best_info["levels"]
        lo, hi = lv["entry"]
        lines.append(f"Timeframe: {best_tf}")
        lines.append(f"Direction: {best_info['direction']}")
        lines.append(f"Setup: {su['type']} ({su['quality']}, score {su['score']})")
        lines.append(f"Entry: {lo} - {hi}")
        lines.append(f"SL: {lv['sl']}")
        lines.append(f"TP1: {lv['tp1']}")
        lines.append(f"TP2: {lv['tp2']}")
        if best_info.get("mtf_agree_tfs"):
            lines.append(f"Confirmed by: {', '.join(best_info['mtf_agree_tfs'])}")
        lines.append("Chosen over the other timeframe(s) by setup quality, confluence strength, and MTF agreement.")
    lines.append("")
    lines.append("## Per-timeframe breakdown")
    lines.append("")

    for tf, info in per_tf.items():
        lines.append(f"## {tf}")
        if "error" in info:
            lines += [f"Analysis error: {info['error']}", ""]
            continue

        da, st, su, lv = info["direction_analysis"], info["structure"], info["setup"], info["levels"]
        lines.append(f"Direction: {info['direction']}")
        lines.append(f"Technical bias: {info['bias']}")
        lines.append(f"Structure: {st['label']}")
        lines.append(f"Confluence: bullish {da['bull']} / bearish {da['bear']}")
        if da["rsi"] is not None:
            lines.append(f"RSI: {da['rsi']:.1f}")
        if da["atr_pct"] is not None:
            lines.append(f"ATR: {da['atr_pct']:.2f}%")
        if da["bb_width_pct"] is not None:
            lines.append(f"Bollinger width: {da['bb_width_pct']:.2f}%")
        for note in st["notes"] + da["notes"] + su["notes"] + info.get("divergence_notes", []):
            lines.append(f"- {note}")
        lines.append(f"Best setup: {su['type']} ({su['quality']})")

        if lv["direction"] != "NONE":
            lo, hi = lv["entry"]
            lines.append(f"Entry: {lo} - {hi}")
            lines.append(f"SL: {lv['sl']}")
            lines.append(f"TP1: {lv['tp1']}")
            lines.append(f"TP2: {lv['tp2']}")
        if info.get("mtf_agree_tfs"):
            lines.append(f"MTF confirmation: {', '.join(info['mtf_agree_tfs'])}")
        lines.append("")

    mtf = _mtf_alignment(per_tf)
    lines.append("## MTF Summary")
    if mtf["overall"] is None:
        lines.append("No sufficiently clear directional alignment was found.")
    else:
        lines.append(f"LONG: {mtf['longs']} timeframe(s)")
        lines.append(f"SHORT: {mtf['shorts']} timeframe(s)")
        lines.append(f"Overall MTF read: {mtf['overall']}")
    return "\n".join(lines)


def compose_console_summary(result: dict) -> str:
    # Concise CI/console view: one line per timeframe plus the MTF verdict.
    # The full breakdown (indicators, notes, structure) stays in the markdown file only.
    symbol = result["symbol"]
    per_tf = result["per_tf"]
    lines = [f"[analyze] {symbol} summary:"]

    best_tf, best_info = result.get("best_tf"), result.get("best")
    if best_info is None:
        lines.append("[analyze]   BEST SETUP: none (no actionable timeframe)")
    else:
        su, lv = best_info["setup"], best_info["levels"]
        lo, hi = lv["entry"]
        lines.append(
            f"[analyze]   BEST SETUP -> {best_tf}: {best_info['direction']} | "
            f"{su['type']} ({su['quality']}) | entry {lo}-{hi} SL {lv['sl']} "
            f"TP1 {lv['tp1']} TP2 {lv['tp2']}"
        )

    for tf, info in per_tf.items():
        if "error" in info:
            lines.append(f"[analyze]   {tf}: ERROR - {info['error']}")
            continue
        su, lv = info["setup"], info["levels"]
        line = f"[analyze]   {tf}: {info['direction']} | {info['bias']} | {su['type']} ({su['quality']})"
        if lv["direction"] != "NONE":
            lo, hi = lv["entry"]
            line += f" | entry {lo}-{hi} SL {lv['sl']} TP1 {lv['tp1']} TP2 {lv['tp2']}"
        if info.get("divergence_notes"):
            line += " | ⚠ DIVERGENCE (structure vs momentum)"
        lines.append(line)

    mtf = _mtf_alignment(per_tf)
    if mtf["overall"] is None:
        lines.append("[analyze]   MTF: no clear directional alignment")
    else:
        lines.append(f"[analyze]   MTF: {mtf['longs']} LONG / {mtf['shorts']} SHORT -> {mtf['overall']}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="vSynapse — independent single-token MTF analyzer")
    parser.add_argument("--symbol", required=True, help="Coin code, e.g. ZEC or ZECUSDT")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    symbol = normalize_symbol(args.symbol, cfg["exchange"]["quote_asset"])
    timeframes = cfg.get("timeframes", ["1h"])

    print(f"[analyze] Independent analysis started for {symbol}")
    print(f"[analyze] Timeframes: {', '.join(timeframes)}")
    result = asyncio.run(analyze_symbol(symbol, cfg))
    text = compose_analysis_text(result)
    print(compose_console_summary(result))

    os.makedirs(OUT_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(OUT_DIR, f"analysis_{result['symbol']}_{timestamp}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[analyze] Finished. Markdown saved to {md_path}")

    # Chart MTF otomatis untuk simbol yang dianalisa, memakai
    # _fetch_and_build_multi dari chart.py apa adanya (fetch klines sendiri,
    # timeframe mengikuti config yang sama dipakai analisa di atas). Disimpan
    # di OUT_DIR yang sama dengan markdown-nya (bukan folder terpisah) supaya
    # artifact workflow tetap satu folder.
    # notify=False: alur analyze.py ini cuma untuk hasil yang masuk artifact
    # workflow, bukan dikirim ke Telegram (itu tetap jadi urusan chart.py
    # sendiri lewat mode CLI/scanner-nya). Dibuat best-effort: kalau gagal
    # (mis. rate limit / symbol bermasalah), tidak menggagalkan seluruh run
    # analisa yang sudah berhasil di atas.
    chart_path = os.path.join(OUT_DIR, f"{symbol}_multi.png")
    try:
        asyncio.run(_fetch_and_build_multi(symbol, timeframes, cfg, chart_path, notify=False))
        print(f"[analyze] Chart MTF saved to {chart_path}")
    except Exception as exc:
        print(f"[warn] Gagal membuat chart MTF untuk {symbol}: {exc}")

    # Kalau SEMUA timeframe gagal (fetch atau analisa), laporan tetap ditulis
    # di atas untuk jejak, tapi exit code dibuat non-zero supaya CI/workflow
    # membedakan "beneran gagal dapat data" (symbol salah ketik, API down)
    # dari "berhasil analisa tapi memang tidak ada setup actionable".
    per_tf = result["per_tf"]
    if per_tf and all("error" in info for info in per_tf.values()):
        print(f"[analyze] All {len(per_tf)} timeframe(s) failed to fetch/analyze. Exiting non-zero.")
        sys.exit(1)


if __name__ == "__main__":
    main()
