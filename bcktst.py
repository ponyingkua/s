"""
bcktst.py -- Walk-forward backtest for analyze.py's per-symbol MTF analyzer.

Purpose
-------
analyze.py's scoring (direction bull/bear weights, setup-type score formulas,
overextension penalties, structural TP logic, etc.) is all hand-tuned by
reasoning about the code, never validated against what actually happened
historically. This script answers: "if analyze_timeframe() had fired this
signal at this point in time, using only data available up to that point,
what would have happened next -- SL hit first, or TP1?"

Method (walk-forward, no look-ahead)
-------------------------------------
For each (symbol, timeframe):
  1. Fetch a long history of closed candles once.
  2. Slide an expanding window forward one candle at a time. At index i,
     only df.iloc[:i+1] (candles up to and including i) is visible --
     candle i is treated as just-closed, nothing after it exists yet.
  3. Call analyze.analyze_timeframe() on that slice (unmodified, imported
     directly -- this script does not reimplement any scoring logic).
  4. If it returns an actionable LONG/SHORT: record entry/SL/TP1/TP2/setup,
     then scan forward candle-by-candle to see whether SL or TP1 is touched
     first (SL wins on same-candle ties -- conservative assumption).
  5. Jump the window past the trade's exit candle before continuing (skip-
     forward), so overlapping/duplicate trades from adjacent candles aren't
     double-counted while a position would still be open.
  6. Aggregate R-multiples by timeframe x setup_type x quality.

Known limitations (by design, not oversights -- see the printed report header)
--------------------------------------------------------------------------
- No historical funding rate / open interest / long-short ratio: Binance's
  premiumIndex/openInterest endpoints are live-snapshot only, not
  backfillable, so analyze_timeframe() is called with deriv=None here. Live
  scores include a small extra bull/bear contribution from that data
  (see _derivatives_bias) that this backtest cannot reproduce.
- BTC correlation and market regime ARE included (see below), computed the
  same walk-forward way from BTC's own OHLCV so as not to leak future BTC
  price action into a signal's evaluation.
- Fees and slippage are NOT modeled. Real trading costs would reduce every
  R-multiple somewhat, more so on 15m (more trades, more turnover) than 4h.
- One simplification vs. analyze_symbol(): the live code's mtf_agree_tfs
  bonus depends on multiple timeframes being analyzed together in the same
  run. This backtest processes each (symbol, timeframe) pair independently,
  so mtf_agree_tfs is always empty here -- the MTF_AGREE_BONUS ranking
  effect (which only matters for _select_best_setup, not for whether a
  signal fires) is not exercised. Setup detection and level-building are
  otherwise identical to live.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from scanner import BinanceFuturesClient, drop_unclosed_candle
import scanner as scanner_mod
import analyze

OUT_DIR = "backtest_output"

# Minimal indicator config for compute_indicators()/get_market_regime(), since
# this script intentionally doesn't depend on an external config.yaml -- these
# are scanner.py's own documented defaults, not tuned guesses.
DEFAULT_CFG = {
    "indicators": {
        "ema": {"period": 200},
        "macd": {"fast": 12, "slow": 26, "signal": 9},
        "supertrend": {"period": 10, "multiplier": 3.0},
        "rsi": {"period": 14},
        "atr": {"period": 14},
        "volume_spike": {"lookback": 20, "factor": 1.6},
        "adx": {"period": 14},
    },
    "regime_filter": {"symbol": "BTCUSDT", "timeframe": "4h", "adx_min": 25},
}

MIN_WARMUP_BARS = 250  # enough history for EMA200 to be meaningful
MAX_HOLD_BARS = {"15m": 4 * 24 * 5, "1h": 24 * 10, "4h": 6 * 20}  # ~5d/10d/20d cap
FETCH_LIMIT_BY_TF = {"15m": 4 * 24, "1h": 24, "4h": 6}  # candles/day, for --days math


def _tf_to_days_multiplier(tf: str) -> int:
    return FETCH_LIMIT_BY_TF.get(tf, 24)


async def fetch_history(client: BinanceFuturesClient, symbol: str, tf: str, days: int) -> pd.DataFrame | None:
    total_limit = days * _tf_to_days_multiplier(tf)
    try:
        kline = await client.get_klines_paginated(symbol, tf, total_limit)
    except Exception as exc:
        print(f"[warn] fetch failed {symbol} {tf}: {exc}")
        return None
    df = drop_unclosed_candle(kline.df)
    if len(df) < MIN_WARMUP_BARS + 20:
        print(f"[warn] {symbol} {tf}: only {len(df)} candles, need >= {MIN_WARMUP_BARS + 20}, skipping")
        return None
    return df.reset_index(drop=True)


def compute_regime_at(btc_df: pd.DataFrame, upto_idx: int, cfg: dict) -> str:
    """Same logic as scanner.get_market_regime(), but offline against a
    slice of already-fetched BTC 4h data ending at upto_idx (inclusive),
    instead of fetching live -- so a signal at time T is only ever compared
    against BTC's regime as it was AT time T, never later."""
    closed = btc_df.iloc[: upto_idx + 1]
    if len(closed) < 50:
        return "NEUTRAL"
    ind = scanner_mod.compute_indicators(closed, cfg)
    i = len(closed) - 1
    price = closed["close"].iloc[i]
    ema_up = price > ind["ema200"].iloc[i]
    st_up = ind["supertrend"].iloc[i] == 1
    adx_val = ind["adx"].iloc[i]
    adx_threshold = cfg.get("regime_filter", {}).get("adx_min", 25)
    if pd.isna(adx_val) or adx_val < adx_threshold:
        return "NEUTRAL"
    if ema_up and st_up:
        return "BULL"
    if not ema_up and not st_up:
        return "BEAR"
    return "NEUTRAL"


def _asof_index(btc_times: pd.DatetimeIndex, target_time) -> int | None:
    """Index of the last BTC candle whose open_time <= target_time -- the
    BTC data actually available at that moment, never a later one. Returns
    None if target_time is before all BTC data (can't happen once both
    series' warmup is aligned, but checked defensively).

    Takes a pd.DatetimeIndex (not a raw np.datetime64 array) specifically to
    stay timezone-aware -- converting a tz-aware pd.Timestamp to
    np.datetime64 silently drops the timezone, which can shift the searched
    position by the UTC offset and pick the wrong candle."""
    target = pd.Timestamp(target_time)
    pos = btc_times.searchsorted(target, side="right") - 1
    return int(pos) if pos >= 0 else None


def btc_correlation_at(symbol: str, sym_direction: str, tf: str,
                        btc_dfs_by_tf: dict, btc_time_arrays: dict,
                        target_time, market_regime: str) -> dict:
    """Offline counterpart to analyze._btc_correlation_note() for one signal
    at one point in time: same conflict rule (LONG vs BTC-structure-BEARISH-
    or-regime-BEAR, and the mirror for SHORT), but BTC's own direction is
    computed only from BTC candles up to target_time -- matching what the
    live per-timeframe BTC check would have seen at that same moment."""
    if symbol.upper().startswith("BTC"):
        return {"btc_structure": "N/A", "conflict": False}
    btc_df = btc_dfs_by_tf.get(tf)
    btc_times = btc_time_arrays.get(tf)
    if btc_df is None or btc_times is None:
        return {"btc_structure": "UNKNOWN", "conflict": False}
    idx = _asof_index(btc_times, target_time)
    if idx is None or idx < 30:
        return {"btc_structure": "UNKNOWN", "conflict": False}
    btc_slice = btc_df.iloc[: idx + 1]
    try:
        btc_direction = analyze._direction_analysis(btc_slice)
    except Exception:
        return {"btc_structure": "UNKNOWN", "conflict": False}
    btc_bull = btc_direction["bull"] > btc_direction["bear"]
    btc_bear = btc_direction["bear"] > btc_direction["bull"]
    btc_label = "BULLISH" if btc_bull else "BEARISH" if btc_bear else "NEUTRAL"
    conflict = (sym_direction == "LONG" and (btc_label == "BEARISH" or market_regime == "BEAR")) or (
        sym_direction == "SHORT" and (btc_label == "BULLISH" or market_regime == "BULL")
    )
    return {"btc_structure": btc_label, "conflict": conflict}


def simulate_exit(df: pd.DataFrame, entry_idx: int, direction: str, sl: float, tp1: float,
                   max_hold: int) -> dict:
    """Walk forward from the candle AFTER entry_idx, checking each candle's
    high/low against SL and TP1. If both are touched within the same
    candle, SL is assumed to have been hit first (conservative -- we can't
    know intrabar order from OHLC alone). If neither is touched within
    max_hold candles, the trade is closed at the last available close
    ("timeout"), which is neither a full win nor a full loss."""
    n = len(df)
    last_idx = min(entry_idx + max_hold, n - 1)
    for j in range(entry_idx + 1, last_idx + 1):
        hi = float(df["high"].iloc[j])
        lo = float(df["low"].iloc[j])
        if direction == "LONG":
            hit_sl = lo <= sl
            hit_tp = hi >= tp1
        else:
            hit_sl = hi >= sl
            hit_tp = lo <= tp1
        if hit_sl and hit_tp:
            return {"exit_idx": j, "exit_price": sl, "reason": "SL"}
        if hit_sl:
            return {"exit_idx": j, "exit_price": sl, "reason": "SL"}
        if hit_tp:
            return {"exit_idx": j, "exit_price": tp1, "reason": "TP1"}
    exit_price = float(df["close"].iloc[last_idx])
    return {"exit_idx": last_idx, "exit_price": exit_price, "reason": "TIMEOUT"}


def r_multiple(direction: str, entry: float, sl: float, exit_price: float) -> float:
    risk = abs(entry - sl)
    if risk <= 0:
        return 0.0
    if direction == "LONG":
        return (exit_price - entry) / risk
    return (entry - exit_price) / risk


def backtest_symbol_tf(symbol: str, tf: str, df: pd.DataFrame,
                        btc_dfs_by_tf: dict, btc_time_arrays: dict,
                        regime_cache: dict, cfg: dict) -> list[dict]:
    trades = []
    n = len(df)
    i = MIN_WARMUP_BARS
    max_hold = MAX_HOLD_BARS.get(tf, 100)
    error_count = 0
    first_error = None

    while i < n - 1:
        df_slice = df.iloc[: i + 1]
        try:
            info = analyze.analyze_timeframe(df_slice, symbol, tf, deriv=None)
        except Exception as exc:
            error_count += 1
            if first_error is None:
                first_error = str(exc)
            i += 1
            continue

        direction = info.get("direction", "NONE")
        if direction == "NONE":
            i += 1
            continue

        lv = info["levels"]
        if lv.get("entry") is None:
            i += 1
            continue

        entry_time = df["open_time"].iloc[i]

        # Market regime as of this candle, from BTC 4h -- cached per BTC
        # candle index so repeated lookups across many symbol candles that
        # fall within the same BTC 4h bar don't recompute it each time.
        regime = "NEUTRAL"
        btc_4h_times = btc_time_arrays.get("4h")
        if btc_4h_times is not None:
            btc_idx = _asof_index(btc_4h_times, entry_time)
            if btc_idx is not None:
                if btc_idx not in regime_cache:
                    regime_cache[btc_idx] = compute_regime_at(btc_dfs_by_tf["4h"], btc_idx, cfg)
                regime = regime_cache[btc_idx]

        corr = btc_correlation_at(symbol, direction, tf, btc_dfs_by_tf, btc_time_arrays, entry_time, regime)

        entry_lo, entry_hi = lv["entry"]
        entry_price = entry_hi if direction == "LONG" else entry_lo
        sl = lv["sl"]
        tp1 = lv["tp1"]

        outcome = simulate_exit(df, i, direction, sl, tp1, max_hold)
        outcome_r = r_multiple(direction, entry_price, sl, outcome["exit_price"])

        trades.append({
            "symbol": symbol,
            "timeframe": tf,
            "entry_time": entry_time.isoformat(),
            "exit_time": df["open_time"].iloc[outcome["exit_idx"]].isoformat(),
            "direction": direction,
            "setup_type": info["setup"]["type"],
            "setup_quality": info["setup"]["quality"],
            "setup_score": info["setup"]["score"],
            "structure_label": info["structure"]["label"],
            "bias": info["bias"],
            "market_regime": regime,
            "btc_conflict": corr["conflict"],
            "entry": entry_price,
            "sl": sl,
            "tp1": tp1,
            "exit_price": outcome["exit_price"],
            "exit_reason": outcome["reason"],
            "r_multiple": round(outcome_r, 4),
            "hold_bars": outcome["exit_idx"] - i,
        })

        # Skip forward past this trade's exit before looking for the next
        # signal -- a new position while one is conceptually still open
        # would double-count the same market move as two trades.
        i = outcome["exit_idx"] + 1

    if error_count > 0:
        print(f"  [warn] {symbol} {tf}: analyze_timeframe raised {error_count} time(s), "
              f"first error: {first_error}")

    return trades


def summarize(trades: list[dict], group_keys: list[str]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    grouped = df.groupby(group_keys, dropna=False)
    summary = grouped.agg(
        n=("r_multiple", "count"),
        win_rate=("r_multiple", lambda s: round((s > 0).mean() * 100, 1)),
        avg_r=("r_multiple", lambda s: round(s.mean(), 4)),
        sum_r=("r_multiple", lambda s: round(s.sum(), 2)),
        median_r=("r_multiple", lambda s: round(s.median(), 4)),
    ).reset_index()
    return summary.sort_values("n", ascending=False)


def print_report(trades: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("BACKTEST REPORT -- analyze.py walk-forward simulation")
    print("=" * 70)
    print(
        "\nLimitations: no historical funding/OI/L-S ratio (deriv=None); "
        "no fees/slippage modeled; mtf_agree_tfs always empty (each timeframe "
        "backtested independently). See file docstring for details.\n"
    )
    if not trades:
        print("No trades were generated at all -- check symbols/timeframes/history length.")
        return

    df = pd.DataFrame(trades)
    print(f"Total trades: {len(df)}  |  Overall win rate: {(df['r_multiple'] > 0).mean()*100:.1f}%  "
          f"|  Overall avg R: {df['r_multiple'].mean():.4f}  |  Sum R: {df['r_multiple'].sum():.2f}\n")

    print("-- By timeframe --")
    print(summarize(trades, ["timeframe"]).to_string(index=False))

    print("\n-- By timeframe x setup_type --")
    print(summarize(trades, ["timeframe", "setup_type"]).to_string(index=False))

    print("\n-- By timeframe x setup_quality --")
    print(summarize(trades, ["timeframe", "setup_quality"]).to_string(index=False))

    print("\n-- By btc_conflict (True = signal fired against BTC structure/regime) --")
    print(summarize(trades, ["btc_conflict"]).to_string(index=False))

    print("\n-- By exit_reason --")
    print(summarize(trades, ["exit_reason"]).to_string(index=False))


async def run_backtest(symbols: list[str], timeframes: list[str], days: int, out_path: str) -> list[dict]:
    cfg = DEFAULT_CFG
    all_trades: list[dict] = []

    async with BinanceFuturesClient() as client:
        print(f"Fetching BTCUSDT history ({days}d) for regime/correlation reference...")
        btc_dfs_by_tf = {}
        btc_time_arrays = {}
        needed_tfs = set(timeframes) | {"4h"}  # 4h always needed for regime
        for tf in needed_tfs:
            btc_df = await fetch_history(client, "BTCUSDT", tf, days)
            if btc_df is not None:
                btc_dfs_by_tf[tf] = btc_df
                btc_time_arrays[tf] = pd.DatetimeIndex(btc_df["open_time"])

        for symbol in symbols:
            for tf in timeframes:
                print(f"Fetching {symbol} {tf}...")
                df = btc_dfs_by_tf.get(tf) if symbol.upper() == "BTCUSDT" else await fetch_history(client, symbol, tf, days)
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
