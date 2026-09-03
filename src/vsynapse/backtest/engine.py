"""Backtest engine sederhana: replay sinyal di atas data historis dan hitung
win rate, average R, dan equity curve. Ini yang paling penting untuk
memvalidasi bahwa strategi baru benar-benar lebih baik dari versi lama
sebelum dijalankan live.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from vsynapse.strategy.scoring import score_symbol


@dataclass
class Trade:
    symbol: str
    direction: str
    entry: float
    sl: float
    tp: float
    result: str  # "WIN" | "LOSS" | "OPEN"
    r_multiple: float


def backtest_symbol(
    df: pd.DataFrame, symbol: str, cfg: dict, window: int = 300
) -> list[Trade]:
    """Jalankan strategi secara rolling di atas histori, simulasikan tiap sinyal
    sampai kena SL atau TP di bar-bar berikutnya."""
    trades: list[Trade] = []

    for i in range(window, len(df) - 1):
        sub_df = df.iloc[i - window : i].reset_index(drop=True)
        signal = score_symbol(sub_df, symbol, cfg)
        if signal.direction == "NONE":
            continue

        future = df.iloc[i + 1 :]
        result, r_mult = _simulate_exit(signal, future)
        trades.append(
            Trade(
                symbol=symbol,
                direction=signal.direction,
                entry=signal.entry,
                sl=signal.sl,
                tp=signal.tp,
                result=result,
                r_multiple=r_mult,
            )
        )

    return trades


def _simulate_exit(signal, future_df: pd.DataFrame) -> tuple[str, float]:
    risk = abs(signal.entry - signal.sl)
    for _, bar in future_df.iterrows():
        if signal.direction == "LONG":
            if bar["low"] <= signal.sl:
                return "LOSS", -1.0
            if bar["high"] >= signal.tp:
                reward = abs(signal.tp - signal.entry)
                return "WIN", reward / risk
        else:
            if bar["high"] >= signal.sl:
                return "LOSS", -1.0
            if bar["low"] <= signal.tp:
                reward = abs(signal.entry - signal.tp)
                return "WIN", reward / risk
    return "OPEN", 0.0


def summarize(trades: list[Trade]) -> dict:
    closed = [t for t in trades if t.result != "OPEN"]
    if not closed:
        return {"total": 0, "win_rate": 0.0, "avg_r": 0.0}

    wins = [t for t in closed if t.result == "WIN"]
    return {
        "total": len(closed),
        "win_rate": round(len(wins) / len(closed) * 100, 1),
        "avg_r": round(sum(t.r_multiple for t in closed) / len(closed), 2),
    }
