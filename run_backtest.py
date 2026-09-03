"""Jalankan backtest untuk satu simbol via GitHub Actions (workflow_dispatch),
supaya bisa dipicu langsung dari app GitHub di HP tanpa komputer.
"""
from __future__ import annotations

import argparse
import asyncio
import json

import yaml

from vsynapse.backtest.engine import backtest_symbol, summarize
from vsynapse.data.binance_client import BinanceFuturesClient


async def run(symbol: str, timeframe: str, limit: int, config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    async with BinanceFuturesClient() as client:
        kline = await client.get_klines(symbol, timeframe, limit=limit)

    trades = backtest_symbol(kline.df, symbol, cfg, window=200)
    summary = summarize(trades)
    print(json.dumps(summary, indent=2))

    with open("backtest_result.md", "w") as f:
        f.write(f"# Backtest — {symbol} ({timeframe})\n\n")
        f.write(f"- Total trade tertutup: **{summary['total']}**\n")
        f.write(f"- Win rate: **{summary['win_rate']}%**\n")
        f.write(f"- Rata-rata R multiple: **{summary['avg_r']}**\n\n")
        f.write("> Dihasilkan otomatis lewat GitHub Actions workflow_dispatch.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    asyncio.run(run(args.symbol, args.timeframe, args.limit, args.config))
