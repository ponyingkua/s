"""Entry point: jalankan scan penuh atas semua simbol aktif di Binance Futures."""
from __future__ import annotations

import argparse
import asyncio
import json

import yaml

from vsynapse.data.binance_client import BinanceFuturesClient
from vsynapse.notify.telegram import format_signal_message, send_telegram_message
from vsynapse.risk.management import passes_risk_filter
from vsynapse.strategy.scoring import score_symbol


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


async def run_scan(cfg: dict, out_path: str) -> list[dict]:
    results = []
    async with BinanceFuturesClient() as client:
        symbols = await client.get_active_symbols(cfg["exchange"]["quote_asset"])

        # Filter volume dulu biar nggak buang waktu analisis coin sepi
        volumes = await asyncio.gather(*(client.get_24h_volume(s) for s in symbols))
        min_vol = cfg["exchange"]["min_volume_usdt_24h"]
        active_symbols = [s for s, v in zip(symbols, volumes) if v >= min_vol]

        primary_tf = cfg["timeframes"][0]
        klines = await client.get_klines_many(active_symbols, primary_tf)

        for kline in klines:
            signal = score_symbol(kline.df, kline.symbol, cfg)
            if signal.direction == "NONE":
                continue
            if not passes_risk_filter(signal, cfg):
                continue

            results.append(signal.__dict__)
            await send_telegram_message(format_signal_message(signal), cfg)

            if len(results) >= cfg["risk"]["max_signals_per_run"]:
                break

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def main():
    parser = argparse.ArgumentParser(description="vSynapse v2 scanner")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="synaptic_candidates.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = asyncio.run(run_scan(cfg, args.out))
    print(f"Ditemukan {len(results)} sinyal. Disimpan ke {args.out}")


if __name__ == "__main__":
    main()
