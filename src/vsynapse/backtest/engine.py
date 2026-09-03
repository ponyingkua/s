"""Backtest engine: replay sinyal di atas data historis dan hitung
win rate, average R (net setelah fee), dan equity curve.

Dua perbaikan penting dari versi awal:
1. Skip-ahead — setelah sebuah trade dibuka, engine tidak mengevaluasi sinyal
   baru untuk simbol yang sama sampai trade itu selesai (kena SL/TP).
2. Fee simulation — biaya trading dikurangkan dari tiap hasil trade dalam
   satuan R, supaya avg-R mendekati kondisi nyata.
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
    result: str
    r_multiple: float


def backtest_symbol(
    df: pd.DataFrame, symbol: str, cfg: dict, window: int = 300
) -> list[Trade]:
    trades: list[Trade] = []
    fee_pct = cfg.get("backtest", {}).get("fee_round_trip_pct", 0.0)

    i = window
    while i < len(df) - 1:
        sub_df = df.iloc[i - window : i].reset_index(drop=True)
        signal = score_symbol(sub_df, symbol, cfg)

        if signal.direction == "NONE":
            i += 1
            continue

        future = df.iloc[i + 1 :].reset_index(drop=True)
        result, r_mult, exit_offset = _simulate_exit(signal, future, fee_pct)

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

        if result == "OPEN":
            break
        i = i + 1 + exit_offset + 1

    return trades


def _simulate_exit(signal, future_df: pd.DataFrame, fee_pct: float) -> tuple[str, float, int]:
    risk = abs(signal.entry - signal.sl)
    fee_r = (signal.entry * fee_pct) / risk if risk > 0 else 0.0

    for offset, (_, bar) in enumerate(future_df.iterrows()):
        if signal.direction == "LONG":
            if bar["low"] <= signal.sl:
                return "LOSS", -1.0 - fee_r, offset
            if bar["high"] >= signal.tp:
                reward = abs(signal.tp - signal.entry)
                return "WIN", (reward / risk) - fee_r, offset
        else:
            if bar["high"] >= signal.sl:
                return "LOSS", -1.0 - fee_r, offset
            if bar["low"] <= signal.tp:
                reward = abs(signal.entry - signal.tp)
                return "WIN", (reward / risk) - fee_r, offset

    return "OPEN", 0.0, len(future_df)


def summarize(trades: list[Trade]) -> dict:
    closed = [t for t in trades if t.result != "OPEN"]
    if not closed:
        return {"total": 0, "win_rate": 0.0, "avg_r": 0.0}

    wins = [t for t in closed if t.result == "WIN"]
    return {
        "total": len(closed),
        "win_rate": round(len(wins) / len(closed) * 100, 1),
        "avg_r": round(sum(t.r_multiple for t in closed) / len(closed), 3),
    }
