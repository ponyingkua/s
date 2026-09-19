                if df is None:
                    continue
                print(f"  {symbol} {tf}: {len(df)} candles, running walk-forward...")
                regime_cache: dict = {}
                trades = backtest_symbol_tf(symbol, tf, df, btc_dfs_by_tf, btc_time_arrays, regime_cache, cfg)
                print(f"  {symbol} {tf}: {len(trades)} trades generated")
                all_trades.extend(trades)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_trades, f, indent=2, default=str)
    print(f"\nRaw trades saved to {out_path}")

    return all_trades


def main():
    parser = argparse.ArgumentParser(description="Walk-forward backtest for analyze.py")
    parser.add_argument("--symbols", type=str, required=True,
                         help="Comma-separated symbols, e.g. BTCUSDT,ETHUSDT,SOLUSDT")
    parser.add_argument("--timeframes", type=str, default="15m,1h,4h",
                         help="Comma-separated timeframes (default: 15m,1h,4h)")
    parser.add_argument("--days", type=int, default=180, help="History length in days (default: 180)")
    parser.add_argument("--output", type=str, default=None,
                         help="Output JSON path (default: backtest_output/trades_<timestamp>.json)")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]
    out_path = args.output or os.path.join(
        OUT_DIR, f"trades_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )

    approx_15m_candles = args.days * 96
    if "15m" in timeframes and approx_15m_candles > 5000:
        print(
            f"[note] {args.days}d of 15m data is ~{approx_15m_candles} candles per symbol. "
            f"analyze_timeframe() recomputes indicators on the full expanding slice at every "
            f"step (not incrementally), so 15m over long periods is the slowest part of this "
            f"run by a wide margin -- expect this to take a while for multiple symbols.\n"
        )

    trades = asyncio.run(run_backtest(symbols, timeframes, args.days, out_path))
    print_report(trades)


if __name__ == "__main__":
    main()
