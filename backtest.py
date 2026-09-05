"""vSynapse v3 — backtest engine, 1 file. Mode single (1 simbol) atau
batch (banyak simbol + ringkasan gabungan).

Contoh:
  python backtest.py --symbol BTCUSDT --timeframe 1h
  python backtest.py --batch --timeframe 1h
  python backtest.py --batch --symbols BTCUSDT,ETHUSDT --timeframe 1h
"""
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass

import pandas as pd
import yaml

from scanner import BinanceFuturesClient, compute_indicators, score_at

DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "LTCUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT", "AVAXUSDT",
]


@dataclass
class Trade:
    symbol: str
    direction: str
    entry: float
    sl: float
    tp: float
    result: str  # "WIN" | "LOSS" | "OPEN"
    r_multiple: float  # sudah dikurangi fee


def backtest_symbol(df: pd.DataFrame, symbol: str, cfg: dict, warmup: int = 250) -> list[Trade]:
    """Skip-ahead: setelah trade dibuka, tidak evaluasi sinyal baru sampai
    trade itu selesai — supaya trade tidak saling tumpang tindih.

    Indikator dihitung SEKALI di seluruh df (bukan direset tiap iterasi
    kayak versi sebelumnya) — lihat catatan lengkap di
    scanner.compute_indicators(). `warmup` cuma menentukan titik mulai
    evaluasi (kasih ruang indikator konvergen dulu), bukan ukuran window
    perhitungan seperti parameter `window` di versi lama."""
    trades: list[Trade] = []
    fee_pct = cfg.get("backtest", {}).get("fee_round_trip_pct", 0.0)
    ind = compute_indicators(df, cfg)

    i = warmup
    while i < len(df) - 1:
        signal = score_at(df, ind, i, symbol, cfg)

        if signal.direction == "NONE":
            i += 1
            continue

        future = df.iloc[i + 1 :].reset_index(drop=True)
        result, r_mult, exit_offset = _simulate_exit(signal, future, fee_pct)

        trades.append(
            Trade(
                symbol=symbol, direction=signal.direction, entry=signal.entry,
                sl=signal.sl, tp=signal.tp, result=result, r_multiple=r_mult,
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


# ---------------------------------------------------------------------------
# Mode single
# ---------------------------------------------------------------------------

async def run_single(symbol: str, timeframe: str, limit: int, cfg: dict) -> None:
    async with BinanceFuturesClient() as client:
        kline = await client.get_klines(symbol, timeframe, limit=limit)

    trades = backtest_symbol(kline.df, symbol, cfg)
    summary = summarize(trades)
    print(json.dumps(summary, indent=2))

    with open("backtest_result.md", "w") as f:
        f.write(f"# Backtest — {symbol} ({timeframe})\n\n")
        f.write(f"- Total trade tertutup: **{summary['total']}**\n")
        f.write(f"- Win rate: **{summary['win_rate']}%**\n")
        f.write(f"- Rata-rata R multiple: **{summary['avg_r']}**\n\n")
        f.write("> Dihasilkan otomatis lewat GitHub Actions workflow_dispatch.\n")


# ---------------------------------------------------------------------------
# Mode batch
# ---------------------------------------------------------------------------

async def run_batch(symbols: list[str], timeframe: str, limit: int, cfg: dict) -> None:
    per_symbol_results: list[tuple[str, dict | None, str | None]] = []
    all_trades: list[Trade] = []

    async with BinanceFuturesClient() as client:
        for symbol in symbols:
            try:
                kline = await client.get_klines(symbol, timeframe, limit=limit)
            except Exception as exc:
                per_symbol_results.append((symbol, None, str(exc)))
                continue

            trades = backtest_symbol(kline.df, symbol, cfg)
            summary = summarize(trades)
            per_symbol_results.append((symbol, summary, None))
            all_trades.extend(trades)

    combined = summarize(all_trades)
    _write_batch_report(per_symbol_results, combined, timeframe)


def _write_batch_report(per_symbol_results, combined: dict, timeframe: str) -> None:
    lines = [f"# Backtest Gabungan ({timeframe})\n"]
    lines.append("| Simbol | Trade | Win Rate | Avg R |")
    lines.append("|---|---|---|---|")

    for symbol, summary, error in per_symbol_results:
        if error:
            lines.append(f"| {symbol} | - | - | error: {error[:40]} |")
        elif summary["total"] == 0:
            lines.append(f"| {symbol} | 0 | - | - |")
        else:
            lines.append(f"| {symbol} | {summary['total']} | {summary['win_rate']}% | {summary['avg_r']} |")

    lines.append("")
    lines.append("## Ringkasan Gabungan")
    lines.append("(semua trade dari semua simbol digabung jadi satu populasi)\n")
    lines.append(f"- Total trade: **{combined['total']}**")
    lines.append(f"- Win rate gabungan: **{combined['win_rate']}%**")
    lines.append(f"- Avg R gabungan: **{combined['avg_r']}**")
    lines.append("")
    lines.append(
        "> ⚠️ Angka gabungan lebih bisa dipercaya dibanding angka per-simbol "
        "individual, tapi tetap bukan jaminan performa live — belum "
        "memperhitungkan slippage atau funding rate."
    )

    report = "\n".join(lines)
    with open("backtest_batch_result.md", "w") as f:
        f.write(report)
    print(report)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true", help="Jalankan mode batch (banyak simbol)")
    parser.add_argument("--symbol", default="BTCUSDT", help="Simbol untuk mode single")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="Simbol dipisah koma untuk mode batch")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.batch:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        asyncio.run(run_batch(symbols, args.timeframe, args.limit, cfg))
    else:
        asyncio.run(run_single(args.symbol.upper(), args.timeframe, args.limit, cfg))


if __name__ == "__main__":
    main()