"""Jalankan backtest untuk banyak simbol sekaligus dan hasilkan ringkasan
gabungan (per simbol + gabungan semua simbol jadi satu populasi trade).
Dijalankan manual lewat GitHub Actions (workflow_dispatch) — bisa dipicu dari
app GitHub tanpa perlu komputer.
"""
from __future__ import annotations

import argparse
import asyncio

import yaml

from vsynapse.backtest.engine import backtest_symbol, summarize
from vsynapse.data.binance_client import BinanceFuturesClient

DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "LTCUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT", "AVAXUSDT",
]


async def run(symbols: list[str], timeframe: str, limit: int, config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    per_symbol_results: list[tuple[str, dict | None, str | None]] = []
    all_trades = []

    async with BinanceFuturesClient() as client:
        for symbol in symbols:
            try:
                kline = await client.get_klines(symbol, timeframe, limit=limit)
            except Exception as exc:
                per_symbol_results.append((symbol, None, str(exc)))
                continue

            trades = backtest_symbol(kline.df, symbol, cfg, window=200)
            summary = summarize(trades)
            per_symbol_results.append((symbol, summary, None))
            all_trades.extend(trades)

    combined = summarize(all_trades)
    write_report(per_symbol_results, combined, timeframe)


def write_report(
    per_symbol_results: list[tuple[str, dict | None, str | None]],
    combined: dict,
    timeframe: str,
) -> None:
    lines = [f"# Backtest Gabungan ({timeframe})\n"]
    lines.append("| Simbol | Trade | Win Rate | Avg R |")
    lines.append("|---|---|---|---|")

    for symbol, summary, error in per_symbol_results:
        if error:
            lines.append(f"| {symbol} | - | - | error: {error[:40]} |")
        elif summary["total"] == 0:
            lines.append(f"| {symbol} | 0 | - | - |")
        else:
            lines.append(
                f"| {symbol} | {summary['total']} | {summary['win_rate']}% | {summary['avg_r']} |"
            )

    lines.append("")
    lines.append("## Ringkasan Gabungan")
    lines.append("(semua trade dari semua simbol digabung jadi satu populasi)\n")
    lines.append(f"- Total trade: **{combined['total']}**")
    lines.append(f"- Win rate gabungan: **{combined['win_rate']}%**")
    lines.append(f"- Avg R gabungan: **{combined['avg_r']}**")
    lines.append("")
    lines.append(
        "> ⚠️ Angka gabungan lebih bisa dipercaya dibanding angka per-simbol "
        "individual (sample lebih besar), tapi tetap bukan jaminan performa live — "
        "data historis tidak menjamin masa depan, dan backtest ini belum "
        "memperhitungkan slippage atau funding rate."
    )

    report = "\n".join(lines)
    with open("backtest_batch_result.md", "w") as f:
        f.write(report)

    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Simbol dipisah koma, contoh: BTCUSDT,ETHUSDT,SOLUSDT",
    )
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    asyncio.run(run(symbols, args.timeframe, args.limit, args.config))