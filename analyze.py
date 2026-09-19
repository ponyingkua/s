from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from scanner import (
    BASE_URL,
    BinanceFuturesClient,
    drop_unclosed_candle,
    get_market_regime,
    load_config,
)

OUT_DIR = "analysis_output"
FETCH_MAX_RETRIES = 3
FETCH_RETRY_BACKOFF_SECONDS = 1.5
MAX_SL_ATR_MULT = 2.5  # SL tidak boleh lebih dari N x ATR dari harga saat ini


def normalize_symbol(raw: str, quote_asset: str) -> str:
    s = raw.strip().upper()
    if not s:
        raise ValueError("Symbol cannot be empty")
    return s if s.endswith(quote_asset) else f"{s}{quote_asset}"


def parse_symbols(raw: str) -> list[str]:
    cleaned = raw.replace(" ", ",")
    while ",," in cleaned:
        cleaned = cleaned.replace(",,", ",")
    cleaned = cleaned.strip(",")
    return [s.strip() for s in cleaned.split(",") if s.strip()]


def _decimals_from_price(price: float) -> int:
    """Disalin dari chart.py::decimals_from_price - supaya analyze.py tidak
    perlu `import chart` cuma untuk format angka (dulu di-import lokal di
    dalam _fmt_price & _round_price di bawah, dipanggil berkali-kali per
    analisa)."""
    p = abs(float(price))
    if p < 0.0001:
        return 8
    if p < 0.001:
        return 7
    if p < 0.01:
        return 6
    if p < 0.1:
        return 5
    if p < 1:
        return 5
    if p < 10:
        return 4
    if p < 100:
        return 3
    return 2


def _fmt_price(value: float) -> str:
    return f"{float(value):.{_decimals_from_price(value)}f}"


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
    out = out.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
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


def _rsi_divergence(df: pd.DataFrame, rsi: pd.Series, lookback: int = 50) -> str | None:
    """Classic 2-point RSI/price divergence, checked against the last two
    confirmed swing highs (bearish) and swing lows (bullish) within the
    most recent `lookback` candles.

    Bearish: price prints a higher high while RSI prints a lower high at
    that same high -> upside momentum is fading even as price still rises.
    Bullish: price prints a lower low while RSI prints a higher low ->
    downside momentum is fading even as price still falls.

    Only counted when at least one of the two RSI readings is already in
    the same momentum zone used elsewhere in this file for RSI scoring
    (>=55 bullish zone boundary / <=45 bearish zone boundary) - divergence
    sitting entirely inside the 45-55 no-man's-land is too weak to be a
    meaningful signal and would mostly just add noise.

    Returns "bullish", "bearish", or None (including when both fire at
    once, which is not a clean read).
    """
    n = min(lookback, len(df))
    if n < 12:
        return None
    x = df.tail(n).reset_index(drop=True)
    rsi_x = rsi.tail(n).reset_index(drop=True)
    sh, sl = _swing_points(x, 3, 3)

    bearish = False
    if len(sh) >= 2:
        (i1, p1), (i2, p2) = sh[-2], sh[-1]
        raw1, raw2 = rsi_x.iloc[i1], rsi_x.iloc[i2]
        if pd.notna(raw1) and pd.notna(raw2):
            r1, r2 = float(raw1), float(raw2)
            if p2 > p1 and r2 < r1 and max(r1, r2) >= 55:
                bearish = True

    bullish = False
    if len(sl) >= 2:
        (i1, p1), (i2, p2) = sl[-2], sl[-1]
        raw1, raw2 = rsi_x.iloc[i1], rsi_x.iloc[i2]
        if pd.notna(raw1) and pd.notna(raw2):
            r1, r2 = float(raw1), float(raw2)
            if p2 < p1 and r2 > r1 and min(r1, r2) <= 45:
                bullish = True

    if bullish and not bearish:
        return "bullish"
    if bearish and not bullish:
        return "bearish"
    return None


def _volume_nodes(df: pd.DataFrame, lookback: int = 100, n_bins: int = 24):
    """Cheap volume profile: bucket close prices into n_bins over the lookback
    window and sum volume per bucket, returning bin centers sorted by volume
    descending. This approximates high-volume nodes (HVN) without needing
    intrabar/tick data -- good enough to flag price levels where a lot of
    activity actually happened, vs. swing S/R which only looks at 2 candles
    (the swing point itself) and ignores everything traded in between."""
    x = df.tail(min(lookback, len(df)))
    if len(x) < 10:
        return []
    lo, hi = float(x["low"].min()), float(x["high"].max())
    if hi <= lo:
        return []
    bin_edges = np.linspace(lo, hi, n_bins + 1)
    bin_vol = np.zeros(n_bins)
    # Distribute each candle's volume across the bins its range spans,
    # weighted by overlap -- better than dumping all volume onto the close.
    for _, row in x.iterrows():
        c_lo, c_hi, vol = float(row["low"]), float(row["high"]), float(row["volume"])
        if vol <= 0 or c_hi <= c_lo:
            continue
        lo_idx = np.searchsorted(bin_edges, c_lo, side="right") - 1
        hi_idx = np.searchsorted(bin_edges, c_hi, side="right") - 1
        lo_idx = max(0, min(lo_idx, n_bins - 1))
        hi_idx = max(0, min(hi_idx, n_bins - 1))
        if lo_idx == hi_idx:
            bin_vol[lo_idx] += vol
        else:
            span = c_hi - c_lo
            for b in range(lo_idx, hi_idx + 1):
                b_lo, b_hi = bin_edges[b], bin_edges[b + 1]
                overlap = max(0.0, min(c_hi, b_hi) - max(c_lo, b_lo))
                bin_vol[b] += vol * (overlap / span)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    order = np.argsort(bin_vol)[::-1]
    return [(float(centers[i]), float(bin_vol[i])) for i in order if bin_vol[i] > 0]


def _levels(df: pd.DataFrame, lookback: int = 100):
    x = df.tail(min(lookback, len(df)))
    sh, sl = _swing_points(x, 2, 2)
    price = float(x["close"].iloc[-1])

    res_candidates = [v for _, v in sh[-5:]]
    sup_candidates = [v for _, v in sl[-5:]]

    # High-volume nodes (HVN): price levels where the most volume actually
    # traded, independent of swing geometry. Used to pull the swing-based
    # level toward a spot with real liquidity behind it when one is nearby,
    # since a swing point with almost no volume traded there is a level in
    # name only. Kept as a nudge, not a replacement -- swing structure is
    # still the primary signal because it reflects where price actually
    # reversed, not just where volume happened to concentrate.
    nodes = _volume_nodes(x, lookback)
    top_nodes = [v for v, _ in nodes[:5]]

    def _hvn_adjust(candidates, direction_above: bool):
        """If a top volume node sits between price and the nearest swing
        candidate (same side), prefer the node -- it's a more defensible
        level because it reflects sustained activity, not a single wick."""
        side_candidates = [v for v in candidates if (v >= price) == direction_above]
        if not side_candidates:
            return None
        nearest = min(side_candidates, key=lambda v: abs(v - price))
        for node in top_nodes:
            is_between = (price < node < nearest) if direction_above else (nearest < node < price)
            if is_between:
                return node
        return nearest

    resistance = _hvn_adjust(res_candidates, True)
    if resistance is None:
        # Fall back to nearest high-volume node above price, then the window high.
        above_nodes = [v for v in top_nodes if v >= price]
        resistance = min(above_nodes, key=lambda v: abs(v - price)) if above_nodes else float(x["high"].max())

    support = _hvn_adjust(sup_candidates, False)
    if support is None:
        below_nodes = [v for v in top_nodes if v <= price]
        support = max(below_nodes, key=lambda v: abs(v - price)) if below_nodes else float(x["low"].min())

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

    # Breakout/breakdown-vs-prior-level check. Earlier this re-ran the full
    # _levels() on df.iloc[:-1] to get a "prior" support/resistance, but
    # _levels() picks whichever swing point is CLOSEST to the current price,
    # not a fixed level -- so dropping one candle can make it jump to a
    # completely different, unrelated swing point (not just shift slightly).
    # That produced notes like "holding below prior support" using a level
    # far from anything currently relevant, contradicting the direction/bias.
    # Using the second-to-last swing high/low already computed above instead
    # -- the actual prior structural point price just broke -- is stable
    # under a 1-candle change and matches what "prior level" should mean.
    last_close = float(df["close"].iloc[-1])
    last_high = float(df["high"].iloc[-1])
    last_low = float(df["low"].iloc[-1])

    if len(sh) >= 2:
        prior_swing_high = sh[-2][1]
        if last_close > prior_swing_high and last_high > prior_swing_high:
            bull += 2.2
            notes.append(f"Price is holding above prior swing resistance ({_fmt_price(prior_swing_high)}).")
    if len(sl) >= 2:
        prior_swing_low = sl[-2][1]
        if last_close < prior_swing_low and last_low < prior_swing_low:
            bear += 2.2
            notes.append(f"Price is holding below prior swing support ({_fmt_price(prior_swing_low)}).")

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

    trend = 0.0
    e20, e50 = float(ema20.iloc[-1]), float(ema50.iloc[-1])
    trend += 1.5 if e20 > e50 else -1.5 if e20 < e50 else 0
    s20, s50 = _ema_slope(ema20), _ema_slope(ema50)
    trend += 1.0 if s20 > 0 and s50 > 0 else -1.0 if s20 < 0 and s50 < 0 else 0

    atr_now = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else p * 0.01
    atr_pct = (atr_now / p * 100) if p else 0.0

    if ema200 is not None and pd.notna(ema200.iloc[-1]):
        e200 = float(ema200.iloc[-1])
        trend += 1.3 if p > e200 else -1.3
        dist_atr = abs(p - e200) / atr_now if atr_now > 0 else 0.0
        if dist_atr >= 1.0:
            trend += 0.4 if p > e200 else -0.4
            notes.append(
                f"Price is {dist_atr:.1f}x ATR {'above' if p > e200 else 'below'} EMA200."
            )
    else:
        notes.append(
            f"EMA200 not available (only {len(df)} candles of history); "
            f"trend score uses EMA20/EMA50 alignment only."
        )

    if atr_pct < 0.35:
        trend *= 0.85
        notes.append(f"ATR {atr_pct:.2f}% is compressed; trend weight reduced.")
    elif atr_pct > 3.5:
        notes.append(f"ATR {atr_pct:.2f}% is elevated; expect wider swings.")

    if trend > 0:
        bull += min(trend, 3.5)
        notes.append("Trend structure favors buyers.")
    elif trend < 0:
        bear += min(abs(trend), 3.5)
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
    else:
        notes.append("RSI not available (flat price over lookback window).")

    if r is not None:
        divergence = _rsi_divergence(df, rsi)
        if divergence == "bullish":
            bull += 0.8
            notes.append("Bullish RSI divergence: price made a lower low while RSI made a higher low.")
        elif divergence == "bearish":
            bear += 0.8
            notes.append("Bearish RSI divergence: price made a higher high while RSI made a lower high.")

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
            if up:
                bull += 0.35
            else:
                bear += 0.35

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
        "atr_pct": round(atr_pct, 4) if pd.notna(atr.iloc[-1]) else None,
        "bb_width_pct": float(bb_width.iloc[-1]) if pd.notna(bb_width.iloc[-1]) else None,
        "notes": notes,
    }


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

    # Directional conviction, used throughout to scale scores continuously
    # instead of the old fixed-per-type values. Ranges roughly 0..1: how
    # lopsided bull vs bear votes are (from _direction_analysis) reflects how
    # confident the underlying signal mix already is, so a BREAKOUT built on
    # near-unanimous bullish votes should outscore one built on a bare edge.
    total_bias = direction["bull"] + direction["bear"]
    conviction = abs(direction["bull"] - direction["bear"]) / total_bias if total_bias > 0 else 0.0

    if bull_bias and p > resistance and body_ratio >= 0.45:
        # Base 2.6..3.4 scaled by candle decisiveness (body_ratio can exceed
        # 0.45 up to ~1.0) and directional conviction, instead of a flat 3.0.
        setup = "BREAKOUT"
        score = 2.6 + 0.5 * min((body_ratio - 0.45) / 0.55, 1.0) + 0.3 * conviction
        notes.append("Bullish breakout has a decisive candle behind it.")
    elif bear_bias and p < support and body_ratio >= 0.45:
        setup = "BREAKDOWN"
        score = 2.6 + 0.5 * min((body_ratio - 0.45) / 0.55, 1.0) + 0.3 * conviction
        notes.append("Bearish breakdown has a decisive candle behind it.")
    else:
        if bull_bias and abs(p - support) <= max(1.2 * atr, p * 0.006):
            # Closer to support = stronger retest; scale 2.2..2.8 by proximity.
            dist_ratio = abs(p - support) / max(1.2 * atr, p * 0.006)
            setup, score = "PULLBACK / RETEST", 2.2 + 0.6 * (1.0 - min(dist_ratio, 1.0))
            notes.append("Price is close to structural support while directional pressure remains bullish.")
        elif bear_bias and abs(p - resistance) <= max(1.2 * atr, p * 0.006):
            dist_ratio = abs(p - resistance) / max(1.2 * atr, p * 0.006)
            setup, score = "PULLBACK / RETEST", 2.2 + 0.6 * (1.0 - min(dist_ratio, 1.0))
            notes.append("Price is close to structural resistance while directional pressure remains bearish.")

        upper_wick = float(last.high - max(last.open, last.close))
        lower_wick = float(min(last.open, last.close) - last.low)
        # A rejection wick becomes the primary setup only if nothing stronger
        # was already found; otherwise it's recorded as a secondary
        # confirmation note so the setup label and its notes never disagree.
        if bear_bias and upper_wick / rng >= 0.45 and last.high >= resistance:
            wick_ratio = upper_wick / rng
            rej_score = 2.0 + 0.6 * min((wick_ratio - 0.45) / 0.55, 1.0)
            if score >= rej_score:
                notes.append("Upper-wick rejection also visible at structural resistance (secondary confirmation).")
            else:
                setup = "REJECTION"
                notes.append("Upper-wick rejection is visible at structural resistance.")
            score = max(score, rej_score)
        elif bull_bias and lower_wick / rng >= 0.45 and last.low <= support:
            wick_ratio = lower_wick / rng
            rej_score = 2.0 + 0.6 * min((wick_ratio - 0.45) / 0.55, 1.0)
            if score >= rej_score:
                notes.append("Lower-wick rejection also visible at structural support (secondary confirmation).")
            else:
                setup = "REJECTION"
                notes.append("Lower-wick rejection is visible at structural support.")
            score = max(score, rej_score)

    if setup == "NONE":
        ema20 = df["close"].ewm(span=20, adjust=False).mean()
        slope20 = _ema_slope(ema20)
        _, _, hist = _macd(df["close"])
        h = float(hist.iloc[-1]) if pd.notna(hist.iloc[-1]) else 0.0
        rsi = _rsi(df["close"])
        r = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50.0

        if bull_bias and structure["label"] == "BULLISH":
            momentum_ok = slope20 > 0 and h >= 0
            rsi_ok = r < 75
            atr_ok = True
            if direction.get("atr_pct") is not None and direction["atr_pct"] < 0.30:
                atr_ok = False
                notes.append("CONTINUATION skipped: ATR too compressed for reliable follow-through.")
            if momentum_ok and rsi_ok and atr_ok:
                # Scale 1.6..2.4 by how much extra room RSI has and how firm
                # the MACD histogram push is, instead of a flat 2.2.
                rsi_room = (75 - r) / 75 if bull_bias else (r - 25) / 75
                hist_strength = min(abs(h) / (atr * 0.05 if atr else 1.0), 1.0) if atr else 0.0
                setup = "CONTINUATION"
                score = 1.6 + 0.4 * min(rsi_room, 1.0) + 0.4 * hist_strength
                notes.append("Trend, structure, and momentum remain aligned for continuation.")
            elif not (momentum_ok and rsi_ok):
                notes.append("Structure bullish but momentum/slope not supportive enough for CONTINUATION.")
        elif bear_bias and structure["label"] == "BEARISH":
            momentum_ok = slope20 < 0 and h <= 0
            rsi_ok = r > 25
            atr_ok = True
            if direction.get("atr_pct") is not None and direction["atr_pct"] < 0.30:
                atr_ok = False
                notes.append("CONTINUATION skipped: ATR too compressed for reliable follow-through.")
            if momentum_ok and rsi_ok and atr_ok:
                rsi_room = (r - 25) / 75
                hist_strength = min(abs(h) / (atr * 0.05 if atr else 1.0), 1.0) if atr else 0.0
                setup = "CONTINUATION"
                score = 1.6 + 0.4 * min(rsi_room, 1.0) + 0.4 * hist_strength
                notes.append("Trend, structure, and momentum remain aligned for continuation.")
            elif not (momentum_ok and rsi_ok):
                notes.append("Structure bearish but momentum/slope not supportive enough for CONTINUATION.")

    score = round(score, 2)
    quality = "HIGH" if score >= 2.8 else "MEDIUM" if score >= 2.0 else "LOW" if score > 0 else "NONE"
    return {"type": setup, "quality": quality, "score": score, "notes": notes}


def _round_price(value: float) -> float:
    if value == 0:
        return 0.0
    return round(float(value), _decimals_from_price(value))


def _build_levels(df: pd.DataFrame, direction: dict, structure: dict, setup: dict) -> dict:
    p = float(df["close"].iloc[-1])
    atr = direction["atr"] or p * 0.01
    support, resistance = structure["support"], structure["resistance"]
    is_long = direction["bull"] > direction["bear"]
    is_short = direction["bear"] > direction["bull"]

    if not (is_long or is_short) or setup["type"] == "NONE":
        return {"direction": "NONE", "entry": None, "sl": None, "tp1": None, "tp2": None}

    notes = []

    if is_long:
        anchor = support if support < p else p - atr
        sl_raw = min(anchor - 0.25 * atr, p - atr)
        sl_floor = p - MAX_SL_ATR_MULT * atr
        sl = max(sl_raw, sl_floor)
        if sl > sl_raw:
            notes.append(
                f"SL distance capped at {MAX_SL_ATR_MULT}x ATR "
                f"(structural support was farther away at {_fmt_price(sl_raw)})."
            )
        lo = min(p, max(anchor, p - 0.45 * atr))
        hi = max(p, lo + 0.15 * atr)
        risk = max(hi - sl, 0.25 * atr)
        tp1, tp2 = hi + risk, hi + 1.8 * risk
        if lo <= 0 or hi <= 0 or sl <= 0 or tp1 <= 0 or tp2 <= 0:
            return {
                "direction": "NONE", "entry": None, "sl": None, "tp1": None, "tp2": None,
                "notes": ["Setup discarded: computed entry/SL/TP were not usable (non-positive price)."],
            }
        return {
            "direction": "LONG",
            "entry": (_round_price(lo), _round_price(hi)),
            "sl": _round_price(sl),
            "tp1": _round_price(tp1),
            "tp2": _round_price(tp2),
            "notes": notes,
        }

    anchor = resistance if resistance > p else p + atr
    sl_raw = max(anchor + 0.25 * atr, p + atr)
    sl_cap = p + MAX_SL_ATR_MULT * atr
    sl = min(sl_raw, sl_cap)
    if sl < sl_raw:
        notes.append(
            f"SL distance capped at {MAX_SL_ATR_MULT}x ATR "
            f"(structural resistance was farther away at {_fmt_price(sl_raw)})."
        )
    hi = max(p, min(anchor, p + 0.45 * atr))
    lo = min(p, hi - 0.15 * atr)
    risk = max(sl - lo, 0.25 * atr)
    tp1, tp2 = lo - risk, lo - 1.8 * risk
    if lo <= 0 or hi <= 0 or sl <= 0 or tp1 <= 0 or tp2 <= 0:
        return {
            "direction": "NONE", "entry": None, "sl": None, "tp1": None, "tp2": None,
            "notes": ["Setup discarded: computed entry/SL/TP were not usable (non-positive price)."],
        }
    return {
        "direction": "SHORT",
        "entry": (_round_price(lo), _round_price(hi)),
        "sl": _round_price(sl),
        "tp1": _round_price(tp1),
        "tp2": _round_price(tp2),
        "notes": notes,
    }


def analyze_timeframe(df: pd.DataFrame, symbol: str, timeframe: str, deriv: dict | None = None) -> dict:
    direction = _direction_analysis(df)
    if deriv is not None:
        dbias = _derivatives_bias(deriv)
        direction["bull"] = round(direction["bull"] + dbias["bull"], 2)
        direction["bear"] = round(direction["bear"] + dbias["bear"], 2)
        direction["notes"] = direction["notes"] + dbias["notes"]
        direction["derivatives"] = {
            "funding_rate_pct": deriv.get("funding_rate_pct"),
            "open_interest": deriv.get("open_interest"),
            "oi_change_pct": deriv.get("oi_change_pct"),
            "ls_account_ratio": deriv.get("ls_account_ratio"),
        }
    structure = _structure_analysis(df)
    setup = _detect_setup(df, structure, direction)

    conf_bull_strong = direction["bull"] - direction["bear"] >= 1.5
    conf_bear_strong = direction["bear"] - direction["bull"] >= 1.5
    divergence_notes = []

    if structure["label"] == "BULLISH" and conf_bear_strong:
        final = "NONE"
        divergence_notes.append(
            f"Divergence: structure is BULLISH (HH/HL intact) but indicator "
            f"confluence is bearish-strong; not treated as an actionable SHORT."
        )
    elif structure["label"] == "BEARISH" and conf_bull_strong:
        final = "NONE"
        divergence_notes.append(
            f"Divergence: structure is BEARISH (LH/LL intact) but indicator "
            f"confluence is bullish-strong; not treated as an actionable LONG."
        )
    elif structure["label"] == "BULLISH" and direction["bull"] > direction["bear"]:
        final = "LONG"
    elif structure["label"] == "BEARISH" and direction["bear"] > direction["bull"]:
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
    if levels["direction"] == "NONE" and final != "NONE":
        # _build_levels discarded the setup (unrealistic SL/TP) — keep direction and
        # levels in sync so downstream selection/rendering never sees a mismatched state.
        final = "NONE"
    divergence_notes += levels.get("notes", [])

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


MTF_AGREE_BONUS = 0.5  # ranking-only bonus per other timeframe agreeing on direction; see _select_best_setup
REGIME_CONFLICT_PENALTY = 0.4  # ranking-only penalty when setup direction fights BTC structure/market regime


def _select_best_setup(per_tf: dict, market_regime: str = "NEUTRAL"):
    # With continuous setup scoring (see _detect_setup), a weak CONTINUATION
    # can genuinely score below 2.0 / land as LOW quality now, so this
    # weak-pool fallback is an active filter, not dead code.
    candidates = []
    weak = []
    for tf, info in per_tf.items():
        if "error" in info or info.get("direction") == "NONE":
            continue
        da, su = info["direction_analysis"], info["setup"]
        if su.get("type", "NONE") == "NONE":
            continue
        strength = abs(da["bull"] - da["bear"])
        agree = len(info.get("mtf_agree_tfs", []))
        rank_score = su["score"] + MTF_AGREE_BONUS * agree
        if info.get("btc_correlation", {}).get("conflict"):
            rank_score -= REGIME_CONFLICT_PENALTY
        row = ((rank_score, strength), tf, info)
        if su.get("quality") == "LOW" or su.get("score", 0) < 2.0:
            weak.append(row)
        else:
            candidates.append(row)

    pool = candidates if candidates else weak
    if not pool:
        return None
    pool.sort(key=lambda c: c[0], reverse=True)
    _, best_tf, best_info = pool[0]
    return best_tf, best_info


async def _fetch_tf(client: "BinanceFuturesClient", symbol: str, tf: str, klines_limit: int):
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
                await asyncio.sleep(delay)
    return tf, None, str(last_err)


# ---------------------------------------------------------------------------
# Derivatives data (funding rate / open interest / long-short ratio).
#
# Endpoint publik Binance Futures ini TIDAK ada di scanner.py (BinanceFuturesClient
# di sana cuma untuk klines + volume), jadi diambil langsung di sini pakai session
# aiohttp milik client yang sudah terbuka (client._session) -- tanpa menambah
# method baru ke class BinanceFuturesClient di scanner.py, biar scanner.py tidak
# disentuh sama sekali. Semua endpoint ini best-effort: kalau gagal (rate limit,
# symbol delisting dari data futures publik, dll) dikembalikan None dan dicatat
# sbg error per-field, tidak pernah membatalkan analisa teknikal utama.
# ---------------------------------------------------------------------------

async def _fetch_derivatives(client: "BinanceFuturesClient", symbol: str) -> dict:
    session = client._session
    out: dict = {
        "funding_rate": None,
        "funding_rate_pct": None,
        "next_funding_time": None,
        "open_interest": None,
        "oi_change_pct": None,
        "ls_account_ratio": None,
        "errors": [],
    }

    # --- Funding rate (current, from premiumIndex -- includes predicted next rate) ---
    try:
        url = f"{BASE_URL}/fapi/v1/premiumIndex"
        async with session.get(url, params={"symbol": symbol}) as resp:
            data = await resp.json()
        if resp.status == 200 and isinstance(data, dict) and "lastFundingRate" in data:
            rate = float(data["lastFundingRate"])
            out["funding_rate"] = rate
            out["funding_rate_pct"] = rate * 100
            out["next_funding_time"] = data.get("nextFundingTime")
        else:
            out["errors"].append(f"funding rate: unexpected response ({data})")
    except Exception as exc:
        out["errors"].append(f"funding rate: {exc}")

    # --- Open interest: current snapshot + ~5h ago from openInterestHist for a
    # cheap trend read (rising OI + rising price = new longs; rising OI + falling
    # price = new shorts; falling OI = position unwinding / closing, either side).
    try:
        url = f"{BASE_URL}/fapi/v1/openInterest"
        async with session.get(url, params={"symbol": symbol}) as resp:
            data = await resp.json()
        if resp.status == 200 and isinstance(data, dict) and "openInterest" in data:
            out["open_interest"] = float(data["openInterest"])
        else:
            out["errors"].append(f"open interest: unexpected response ({data})")
    except Exception as exc:
        out["errors"].append(f"open interest: {exc}")

    if out["open_interest"] is not None:
        try:
            hist_url = f"{BASE_URL}/futures/data/openInterestHist"
            async with session.get(
                hist_url, params={"symbol": symbol, "period": "1h", "limit": 6}
            ) as resp:
                hist = await resp.json()
            if resp.status == 200 and isinstance(hist, list) and len(hist) >= 2:
                oi_then = float(hist[0]["sumOpenInterest"])
                oi_now = float(hist[-1]["sumOpenInterest"])
                if oi_then > 0:
                    out["oi_change_pct"] = (oi_now - oi_then) / oi_then * 100
        except Exception as exc:
            out["errors"].append(f"open interest history: {exc}")

    # --- Global long/short account ratio (retail positioning; contrarian read at
    # extremes -- very crowded retail longs/shorts often precede a squeeze the
    # other way). Endpoint has ~30min data lag, treated as directional only. ---
    try:
        url = f"{BASE_URL}/futures/data/globalLongShortAccountRatio"
        async with session.get(
            url, params={"symbol": symbol, "period": "1h", "limit": 1}
        ) as resp:
            data = await resp.json()
        if resp.status == 200 and isinstance(data, list) and data:
            out["ls_account_ratio"] = float(data[-1]["longShortRatio"])
        else:
            out["errors"].append(f"long/short ratio: unexpected response ({data})")
    except Exception as exc:
        out["errors"].append(f"long/short ratio: {exc}")

    return out


def _derivatives_bias(deriv: dict) -> dict:
    """Convert raw derivatives numbers into a bull/bear score contribution
    plus human-readable notes, mirroring the scoring style of _direction_analysis.
    Every threshold here is a heuristic, not a law -- deliberately kept small
    relative to the technical score (max ~1.8 total) since derivatives data is
    a confirming/contrarian signal, not a primary directional one."""
    bull = bear = 0.0
    notes = []

    rate = deriv.get("funding_rate_pct")
    if rate is not None:
        if rate <= -0.05:
            bull += 0.5
            notes.append(f"Funding rate {rate:+.3f}% is meaningfully negative — shorts paying longs.")
        elif rate >= 0.08:
            bear += 0.5
            notes.append(f"Funding rate {rate:+.3f}% is elevated — longs paying a premium, crowded-long risk.")
        elif rate >= 0.04:
            bear += 0.2
            notes.append(f"Funding rate {rate:+.3f}% is mildly positive.")
        elif rate <= -0.02:
            bull += 0.2
            notes.append(f"Funding rate {rate:+.3f}% is mildly negative.")

    oi_chg = deriv.get("oi_change_pct")
    if oi_chg is not None:
        if abs(oi_chg) >= 3.0:
            notes.append(f"Open interest {'up' if oi_chg > 0 else 'down'} {abs(oi_chg):.1f}% over the last ~5h.")

    ratio = deriv.get("ls_account_ratio")
    if ratio is not None:
        # ratio = longs/shorts among retail accounts; >1 means more long accounts.
        if ratio >= 2.2:
            bear += 0.4
            notes.append(f"Retail long/short ratio {ratio:.2f} is crowded-long (contrarian bearish tilt).")
        elif ratio <= 0.55:
            bull += 0.4
            notes.append(f"Retail long/short ratio {ratio:.2f} is crowded-short (contrarian bullish tilt).")

    if deriv.get("errors"):
        notes.append(f"Derivatives data partially unavailable: {'; '.join(deriv['errors'])}")

    return {"bull": round(bull, 2), "bear": round(bear, 2), "notes": notes}


def _btc_correlation_note(symbol: str, deriv_regime: str, btc_dfs: dict, per_tf: dict) -> dict:
    """Soft correlation check against BTC, not a hard block. Altcoins that
    want to go LONG while BTC's own structure/regime on the same timeframe is
    BEARISH (or vice versa) get a cautionary note and a small score penalty in
    ranking (via _select_best_setup reading this field) rather than being
    filtered out outright -- alts do decouple from BTC sometimes, so this is
    a risk flag for the trader to weigh, not an automatic disqualifier."""
    notes_by_tf: dict = {}
    if symbol.upper().startswith("BTC"):
        return notes_by_tf

    for tf, info in per_tf.items():
        if "error" in info or info.get("direction") == "NONE":
            continue
        btc_df = btc_dfs.get(tf)
        if btc_df is None or len(btc_df) < 30:
            continue
        try:
            btc_direction = _direction_analysis(btc_df)
            btc_structure = _structure_analysis(btc_df)
        except Exception:
            continue
        btc_bull = btc_direction["bull"] > btc_direction["bear"]
        btc_bear = btc_direction["bear"] > btc_direction["bull"]
        btc_label = "BULLISH" if btc_bull else "BEARISH" if btc_bear else "NEUTRAL"

        sym_direction = info["direction"]
        conflict = (sym_direction == "LONG" and (btc_label == "BEARISH" or deriv_regime == "BEAR")) or (
            sym_direction == "SHORT" and (btc_label == "BULLISH" or deriv_regime == "BULL")
        )
        note = None
        if conflict:
            parts = []
            if btc_label != "NEUTRAL":
                parts.append(f"BTC {tf} structure is {btc_label}")
            if deriv_regime in ("BULL", "BEAR"):
                parts.append(f"overall market regime is {deriv_regime}")
            note = (
                f"Caution: {sym_direction} on {symbol} runs against "
                f"{' and '.join(parts)} — correlation risk, not a hard block."
            )
        notes_by_tf[tf] = {
            "btc_structure": btc_label,
            "conflict": conflict,
            "note": note,
        }
    return notes_by_tf


async def analyze_symbol(symbol: str, cfg: dict) -> dict:
    data_cfg = cfg.get("scanning", {})
    klines_limit = int(data_cfg.get("klines_limit", 300))
    timeframes = cfg.get("timeframes", ["1h"])
    per_tf = {}
    dfs = {}
    need_btc = not symbol.upper().startswith("BTC")

    async with BinanceFuturesClient() as client:
        awaitables = [
            asyncio.gather(
                *(_fetch_tf(client, symbol, tf, klines_limit) for tf in timeframes),
                return_exceptions=True,
            ),
            _fetch_derivatives(client, symbol),
            get_market_regime(client, cfg),
        ]
        if need_btc:
            awaitables.append(
                asyncio.gather(
                    *(_fetch_tf(client, "BTCUSDT", tf, klines_limit) for tf in timeframes),
                    return_exceptions=True,
                )
            )
        gathered = await asyncio.gather(*awaitables)
    results, deriv, regime = gathered[0], gathered[1], gathered[2]
    btc_results = gathered[3] if need_btc else []

    btc_dfs = {}
    for tf, res in zip(timeframes, btc_results):
        if isinstance(res, Exception):
            continue
        _, bdf, err = res
        if err is None and bdf is not None:
            btc_dfs[tf] = bdf

    for tf, res in zip(timeframes, results):
        if isinstance(res, Exception):
            per_tf[tf] = {"error": str(res)}
            continue
        _, df, err = res
        if err is not None:
            per_tf[tf] = {"error": err}
            continue
        try:
            dfs[tf] = df
            info = analyze_timeframe(df, symbol, tf, deriv=deriv)
            info["mtf_agree_tfs"] = []
            per_tf[tf] = info
        except Exception as exc:
            per_tf[tf] = {"error": str(exc)}

    directions = {tf: x["direction"] for tf, x in per_tf.items() if "direction" in x and x["direction"] != "NONE"}
    for tf, info in per_tf.items():
        if "direction" in info and info["direction"] != "NONE":
            info["mtf_agree_tfs"] = [t for t, d in directions.items() if t != tf and d == info["direction"]]

    btc_corr = _btc_correlation_note(symbol, regime, btc_dfs, per_tf)
    for tf, corr in btc_corr.items():
        per_tf[tf]["btc_correlation"] = corr
        if corr.get("note"):
            per_tf[tf]["direction_analysis"]["notes"].append(corr["note"])

    best = _select_best_setup(per_tf, market_regime=regime)
    return {
        "symbol": symbol,
        "per_tf": per_tf,
        "dfs": dfs,
        "market_regime": regime,
        "best_tf": best[0] if best else None,
        "best": best[1] if best else None,
    }


def compose_analysis_text(result: dict) -> str:
    symbol = result["symbol"]
    per_tf = result["per_tf"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Independent Technical Analysis: {symbol}",
        f"Time: {now}",
        f"Market regime (BTC 4h): {result.get('market_regime', 'NEUTRAL')}",
        "",
    ]

    best_tf, best_info = result.get("best_tf"), result.get("best")
    lines.append("## 🎯 Best Setup")
    if best_info is None:
        lines.append("No actionable setup was found on any analyzed timeframe.")
    else:
        su, lv = best_info["setup"], best_info["levels"]
        lo, hi = lv["entry"]
        lines.append(f"Timeframe: {best_tf}")
        lines.append(f"Direction: {best_info['direction']}")
        lines.append(f"Setup: {su['type']} ({su['quality']}, score {su['score']})")
        lines.append(f"Entry: {_fmt_price(lo)} - {_fmt_price(hi)}")
        lines.append(f"SL: {_fmt_price(lv['sl'])}")
        lines.append(f"TP1: {_fmt_price(lv['tp1'])}")
        lines.append(f"TP2: {_fmt_price(lv['tp2'])}")
        agree = best_info.get("mtf_agree_tfs", [])
        if agree:
            lines.append(f"MTF agreement: {', '.join(agree)}")
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
        deriv_info = da.get("derivatives")
        if deriv_info and deriv_info.get("funding_rate_pct") is not None:
            lines.append(
                f"Funding: {deriv_info['funding_rate_pct']:+.4f}% | "
                f"OI: {deriv_info.get('open_interest')} "
                f"({deriv_info.get('oi_change_pct') or 0:+.1f}% /5h) | "
                f"L/S ratio: {deriv_info.get('ls_account_ratio')}"
            )
        agree = info.get("mtf_agree_tfs", [])
        if agree:
            lines.append(f"MTF agreement: {', '.join(agree)}")
        for note in st["notes"] + da["notes"] + su["notes"] + info.get("divergence_notes", []):
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines)


def compose_console_summary(result: dict) -> str:
    symbol = result["symbol"]
    lines = [f"[analyze] {symbol} summary:"]

    best_tf, best_info = result.get("best_tf"), result.get("best")
    if best_info is None:
        lines.append("[analyze]   BEST SETUP: none")
    else:
        su, lv = best_info["setup"], best_info["levels"]
        lo, hi = lv["entry"]
        lines.append(
            f"[analyze]   BEST SETUP -> {best_tf}: {best_info['direction']} | "
            f"{su['type']} ({su['quality']}) | entry {_fmt_price(lo)}-{_fmt_price(hi)} SL {_fmt_price(lv['sl'])}"
        )
    return "\n".join(lines)


def _render_charts_for_format(
    result: dict,
    symbol: str,
    cfg: dict,
    timeframes: list[str],
    chart_type: str,
    chart_format: str,
) -> None:
    """Render chart MTF dan/atau drill-down untuk satu rasio (wide/square)."""
    from mtfk import build_mtfk_chart, single_mtfk

    square = chart_format == "square"
    suffix = "_square" if square else ""

    if chart_type in ("both", "mtf"):
        chart_path = os.path.join(OUT_DIR, f"{symbol}_multi{suffix}.png")
        try:
            build_mtfk_chart(
                dfs=result["dfs"],
                symbol=result["symbol"],
                timeframes=timeframes,
                per_tf=result["per_tf"],
                out_path=chart_path,
                cfg=cfg,
                square=square,
            )
            print(f"[analyze] Chart MTF saved to {chart_path}")
        except Exception as exc:
            print(f"[warn] Gagal membuat chart MTF untuk {symbol} ({chart_format}): {exc}")

    best_tf = result.get("best_tf")
    if chart_type in ("both", "single") and best_tf is not None:
        single_path = os.path.join(OUT_DIR, f"{symbol}_single_{best_tf}{suffix}.png")
        try:
            single_mtfk(
                df=result["dfs"][best_tf],
                symbol=result["symbol"],
                timeframe=best_tf,
                tf_info=result["per_tf"][best_tf],
                out_path=single_path,
                cfg=cfg,
                square=square,
            )
            print(f"[analyze] Chart drill-down ({best_tf}) saved to {single_path}")
        except Exception as exc:
            print(f"[warn] Gagal membuat chart drill-down untuk {symbol} ({best_tf}, {chart_format}): {exc}")


def run_one(symbol_raw: str, cfg: dict, chart_format: str = "wide", chart_type: str = "both") -> bool:
    quote = cfg["exchange"]["quote_asset"]
    try:
        symbol = normalize_symbol(symbol_raw, quote)
    except ValueError as exc:
        print(f"[analyze] ERROR: {exc}")
        return False

    timeframes = cfg.get("timeframes", ["1h"])
    print(f"[analyze] Independent analysis started for {symbol}")

    try:
        result = asyncio.run(analyze_symbol(symbol, cfg))
    except Exception as exc:
        print(f"[analyze] ✗ Gagal: {symbol} ({exc})")
        return False

    text = compose_analysis_text(result)
    print(compose_console_summary(result))

    os.makedirs(OUT_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(OUT_DIR, f"analysis_{result['symbol']}_{timestamp}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[analyze] Finished. Markdown saved to {md_path}")

    chart_formats = ["wide", "square"] if chart_format == "both" else [chart_format]
    for fmt in chart_formats:
        _render_charts_for_format(result, symbol, cfg, timeframes, chart_type, fmt)

    per_tf = result["per_tf"]
    if per_tf and all("error" in info for info in per_tf.values()):
        print(f"[analyze] All {len(per_tf)} timeframe(s) failed to fetch/analyze. Exiting non-zero.")
        return False

    print(f"[analyze] ✓ Selesai: {symbol}")
    return True


def main():
    parser = argparse.ArgumentParser(description="vSynapse — independent multi-token MTF analyzer")
    parser.add_argument("--symbol", default=None, help="Single coin code, e.g. RIVER or RIVERUSDT")
    parser.add_argument("--symbols", default=None, help="Comma/space separated symbols, e.g. ZEC,SOL,BTC")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--chart-format", choices=["wide", "square", "both"], default="wide")
    parser.add_argument("--chart-type", choices=["both", "single", "mtf"], default="both")
    args = parser.parse_args()

    if not args.symbol and not args.symbols:
        parser.error("Berikan --symbol atau --symbols")
    if args.symbol and args.symbols:
        parser.error("Gunakan salah satu saja: --symbol atau --symbols, jangan keduanya")

    raw = args.symbols if args.symbols else args.symbol
    symbols = parse_symbols(raw)
    if not symbols:
        print("ERROR: Tidak ada symbol yang diberikan")
        sys.exit(1)

    cfg = load_config(args.config)

    print(f"Symbols yang akan dianalisa: {','.join(symbols)}")
    print("----------------------------------------")

    failed = 0
    for sym in symbols:
        print("")
        print(f">>> Analyzing: {sym}")
        print("----------------------------------------")
        ok = run_one(sym, cfg, args.chart_format, args.chart_type)
        if not ok:
            failed += 1

    print("")
    print("========================================")
    print("")
    print("Isi analysis_output/ (markdown + chart MTF per token):")
    if os.path.isdir(OUT_DIR):
        for root, _, files in os.walk(OUT_DIR):
            for name in sorted(files):
                print(os.path.join(root, name))
    else:
        print("(folder analysis_output belum ada)")

    print("")
    if failed > 0:
        print(f"Selesai dengan {failed} kegagalan.")
        sys.exit(1)
    print("Semua analisa berhasil.")


if __name__ == "__main__":
    main()
