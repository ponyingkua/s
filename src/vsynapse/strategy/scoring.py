"""Confluence scoring: gabungkan beberapa sinyal indikator jadi satu skor 0-100,
bukan sinyal biner. Ini yang bikin hasil lebih robust dibanding versi lama
yang (kemungkinan) langsung trigger dari satu-dua kondisi saja.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from vsynapse.indicators import technical as ta


@dataclass
class SignalResult:
    symbol: str
    direction: str  # "LONG" | "SHORT" | "NONE"
    score: float
    reasons: list[str] = field(default_factory=list)
    entry: float | None = None
    sl: float | None = None
    tp: float | None = None


def score_symbol(df: pd.DataFrame, symbol: str, cfg: dict) -> SignalResult:
    w = cfg["scoring"]["weights"]
    close = df["close"]

    ema200 = ta.ema(close, cfg["indicators"]["ema"]["period"])
    macd_line, signal_line, _ = ta.macd(
        close,
        cfg["indicators"]["macd"]["fast"],
        cfg["indicators"]["macd"]["slow"],
        cfg["indicators"]["macd"]["signal"],
    )
    st = ta.supertrend(
        df,
        cfg["indicators"]["supertrend"]["period"],
        cfg["indicators"]["supertrend"]["multiplier"],
    )
    rsi_val = ta.rsi(close, cfg["indicators"]["rsi"]["period"])
    vol_spike = ta.volume_spike(df["volume"])
    atr_val = ta.atr(df, cfg["indicators"]["atr"]["period"])

    last = -1
    price = close.iloc[last]

    long_score, short_score = 0.0, 0.0
    long_reasons: list[str] = []
    short_reasons: list[str] = []

    # Trend (EMA200)
    if price > ema200.iloc[last]:
        long_score += w["ema_trend"]
        long_reasons.append("Harga di atas EMA200 (uptrend)")
    else:
        short_score += w["ema_trend"]
        short_reasons.append("Harga di bawah EMA200 (downtrend)")

    # MACD cross
    if macd_line.iloc[last] > signal_line.iloc[last]:
        long_score += w["macd_cross"]
        long_reasons.append("MACD line di atas signal line (momentum naik)")
    else:
        short_score += w["macd_cross"]
        short_reasons.append("MACD line di bawah signal line (momentum turun)")

    # Supertrend
    if st.iloc[last] == 1:
        long_score += w["supertrend"]
        long_reasons.append("Supertrend menunjukkan uptrend")
    else:
        short_score += w["supertrend"]
        short_reasons.append("Supertrend menunjukkan downtrend")

    # Volume spike (menguatkan arah dominan, bukan penentu arah)
    has_volume_spike = bool(vol_spike.iloc[last])
    if has_volume_spike:
        if long_score >= short_score:
            long_score += w["volume_spike"]
            long_reasons.append("Volume spike terdeteksi (menguatkan sinyal)")
        else:
            short_score += w["volume_spike"]
            short_reasons.append("Volume spike terdeteksi (menguatkan sinyal)")

    # RSI confluence: hindari entry di zona overbought/oversold ekstrem
    r = rsi_val.iloc[last]
    if 40 <= r <= 60:
        long_score += w["rsi_confluence"] / 2
        short_score += w["rsi_confluence"] / 2
        long_reasons.append(f"RSI netral ({r:.0f})")
        short_reasons.append(f"RSI netral ({r:.0f})")
    elif r < 40:
        long_score += w["rsi_confluence"]
        long_reasons.append(f"RSI oversold ({r:.0f}), potensi rebound")
    elif r > 60:
        short_score += w["rsi_confluence"]
        short_reasons.append(f"RSI overbought ({r:.0f}), potensi koreksi")

    direction = "NONE"
    final_score = 0.0
    reasons: list[str] = []
    if long_score >= short_score:
        direction, final_score, reasons = "LONG", long_score, long_reasons
    else:
        direction, final_score, reasons = "SHORT", short_score, short_reasons

    if final_score < cfg["scoring"]["min_score_to_trigger"]:
        return SignalResult(symbol=symbol, direction="NONE", score=final_score, reasons=reasons)

    sl_dist = atr_val.iloc[last] * cfg["risk"]["atr_multiplier_sl"]
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
        reasons=reasons,
        entry=round(price, 6),
        sl=round(sl, 6),
        tp=round(tp, 6),
    )