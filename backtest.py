from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, field

import pandas as pd
import yaml

from scanner import (
    BinanceFuturesClient,
    compute_indicators,
    mtf_bonus_eligible,
    passes_regime_filter,
    passes_risk_filter,
    score_at,
)

DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "LTCUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT", "AVAXUSDT",
]

TF_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
    "1d": 1440,
}


def get_warmup(cfg: dict, fallback: int = 250) -> int:
    """
    Jumlah bar warm-up sebelum backtest mulai cari sinyal. Dibaca dari
    scanning.min_history_bars di config.yaml supaya konsisten dengan
    syarat history minimum yang dipakai scanner.py (run_scan), bukan
    angka hardcode terpisah yang gampang divergen dari config.
    """
    return int(cfg.get("scanning", {}).get("min_history_bars", fallback))


def limit_for_days(timeframe: str, days: float, warmup: int = 250) -> int:
    minutes = TF_MINUTES.get(timeframe, 60)
    trading_bars = int(days * 24 * 60 / minutes)
    return trading_bars + warmup


async def fetch_klines(client: BinanceFuturesClient, symbol: str, timeframe: str, limit: int):
    if limit <= 1500:
        return await client.get_klines(symbol, timeframe, limit=limit)
    return await client.get_klines_paginated(symbol, timeframe, total_limit=limit)


def _regime_limit_for(primary_timeframe: str, primary_limit: int, regime_timeframe: str,
                       min_warmup: int = 300) -> int:
    primary_minutes = TF_MINUTES.get(primary_timeframe, 60)
    regime_minutes = TF_MINUTES.get(regime_timeframe, 240)
    span_minutes = primary_minutes * primary_limit
    needed_bars = int(span_minutes / regime_minutes) + min_warmup
    return max(needed_bars, min_warmup)


def _mtf_limit_for(primary_timeframe: str, primary_limit: int, other_tf: str,
                   min_warmup: int = 250) -> int:
    """How many bars of another TF are needed to cover the same calendar span."""
    primary_minutes = TF_MINUTES.get(primary_timeframe, 60)
    other_minutes = TF_MINUTES.get(other_tf, 60)
    span_minutes = primary_minutes * primary_limit
    needed = int(span_minutes / other_minutes) + min_warmup
    return max(needed, min_warmup)


def compute_regime_series(df_regime: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    regime_cfg = cfg.get("regime_filter", {})
    adx_threshold = regime_cfg.get("adx_min", 25)

    ind = compute_indicators(df_regime, cfg)
    ema_up = df_regime["close"] > ind["ema200"]
    st_up = ind["supertrend"] == 1
    adx_val = ind["adx"]
    strong = adx_val >= adx_threshold

    regime = pd.Series("NEUTRAL", index=df_regime.index)
    regime[strong & ema_up & st_up] = "BULL"
    regime[strong & ~ema_up & ~st_up] = "BEAR"

    return pd.DataFrame({
        "open_time": df_regime["open_time"],
        "close_time": df_regime["close_time"],
        "regime": regime,
    })


def align_regime_to(primary_df: pd.DataFrame, regime_df: pd.DataFrame) -> pd.Series:
    left = primary_df[["open_time"]].reset_index(drop=True).sort_values("open_time")
    right = regime_df.sort_values("close_time").reset_index(drop=True)

    merged = pd.merge_asof(
        left, right, left_on="open_time", right_on="close_time", direction="backward",
    )
    merged["regime"] = merged["regime"].fillna("NEUTRAL")
    return merged.sort_index()["regime"].reset_index(drop=True)


def align_series_by_open_time(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    source_series: pd.Series,
    default: str = "NEUTRAL",
) -> pd.Series:
    """Align a time series without using any future source candle."""
    source = pd.DataFrame({
        "open_time": source_df["open_time"].reset_index(drop=True),
        "value": source_series.reset_index(drop=True),
    }).sort_values("open_time")
    left = target_df[["open_time"]].reset_index(drop=True).sort_values("open_time")
    merged = pd.merge_asof(
        left, source, on="open_time", direction="backward"
    )
    merged["value"] = merged["value"].fillna(default)
    return merged.sort_index()["value"].reset_index(drop=True)


def _build_mtf_direction_series(
    primary_df: pd.DataFrame,
    other_df: pd.DataFrame,
    other_tf: str,
    symbol: str,
    cfg: dict,
    warmup: int | None = None,
    regime_series: pd.Series | None = None,
) -> pd.Series:
    """
    For every bar on the primary timeframe, look up the most recent closed bar
    on `other_tf` and run score_at on it. Returns a Series of direction strings
    aligned to primary_df index ("LONG" / "SHORT" / "NONE").
    """
    if other_df is None or other_df.empty:
        return pd.Series("NONE", index=primary_df.index)

    warmup = warmup if warmup is not None else get_warmup(cfg)
    ind_other = compute_indicators(other_df, cfg)
    # Pre-compute direction for every bar on the other TF (after warmup)
    directions = []
    times = []
    for j in range(warmup, len(other_df)):
        sig = score_at(other_df, ind_other, j, symbol, cfg, timeframe=other_tf)
        if sig.direction != "NONE":
            if not passes_risk_filter(sig, cfg):
                sig.direction = "NONE"
            elif regime_series is not None:
                regime = regime_series.iloc[j]
                if not passes_regime_filter(sig.direction, regime, cfg):
                    sig.direction = "NONE"
        directions.append(sig.direction)
        times.append(other_df["close_time"].iloc[j])

    if not times:
        return pd.Series("NONE", index=primary_df.index)

    other_dir_df = pd.DataFrame({
        "close_time": times,
        "direction": directions,
    }).sort_values("close_time")

    left = primary_df[["open_time"]].reset_index(drop=True).sort_values("open_time")
    merged = pd.merge_asof(
        left, other_dir_df,
        left_on="open_time", right_on="close_time",
        direction="backward",
    )
    merged["direction"] = merged["direction"].fillna("NONE")
    return merged.sort_index()["direction"].reset_index(drop=True)


@dataclass
class Trade:
    symbol: str
    direction: str
    entry: float
    sl: float
    tp: float
    result: str
    r_multiple: float
    timeframe: str = ""
    setup_type: str = ""
    entry_time: str = ""
    exit_time: str = ""
    both_touched: bool = False
    mtf_bonus: float = 0.0
    mtf_agree_tfs: list = field(default_factory=list)
    # Skor akhir sinyal (setelah MTF bonus, sebelum dibulatkan lagi). Sebelumnya
    # tidak direkam sama sekali di trade log, jadi analisis "apakah skor lebih
    # tinggi = performa lebih baik" tidak bisa dilakukan dari trades_raw*.json
    # yang lama. Field ini mengisi celah itu untuk backtest ke depan.
    score: float = 0.0


def backtest_symbol(
    df: pd.DataFrame,
    symbol: str,
    cfg: dict,
    timeframe: str = "",
    warmup: int | None = None,
    regime_series: pd.Series | None = None,
    mtf_direction_map: dict[str, pd.Series] | None = None,
) -> list[Trade]:
    """
    mtf_direction_map: optional dict of {other_tf: Series of directions aligned to df}.
    When present, MTF agreement bonus is applied exactly like in the live scanner.

    Catatan paritas dengan live scanner (run_scan di scanner.py):
    - cooldown_hours DIREPLIKASI di bawah (per arah, dalam simbol+timeframe ini),
      supaya backtest tidak mengambil sinyal beruntun yang sebenarnya akan
      ditahan cooldown di live.
    - max_signals_per_regime_episode TIDAK direplikasi. Limit itu bersifat
      lintas-simbol dan lintas-run (dihitung dari state persisten di
      signal_state.json, digabung dari semua simbol dalam satu scan), jadi
      tidak bisa direkonstruksi secara bermakna di backtest satu simbol.
      Untuk simbol yang sinyalnya searah dengan regime_gated_direction, hasil
      backtest bisa menunjukkan frekuensi trade sedikit lebih tinggi daripada
      yang live akan izinkan.
    """
    trades: list[Trade] = []
    fee_pct = cfg.get("backtest", {}).get("fee_round_trip_pct", 0.0)
    mtf_weight = cfg.get("scoring", {}).get("weights", {}).get("mtf_agreement", 0)
    cooldown_hours = cfg.get("scanning", {}).get("cooldown_hours", 0)
    warmup = warmup if warmup is not None else get_warmup(cfg)
    ind = compute_indicators(df, cfg)

    # Cooldown state in-memory: kapan terakhir kali arah ini diambil sebagai
    # sinyal, di simbol+timeframe ini. Tidak pakai is_in_cooldown/mark_signaled
    # dari scanner.py karena keduanya berbasis datetime.now() (wall clock),
    # sedangkan di sini waktunya harus mengikuti waktu candle historis.
    last_signal_time: dict[str, pd.Timestamp] = {}

    i = warmup
    while i < len(df) - 1:
        signal = score_at(df, ind, i, symbol, cfg, timeframe=timeframe)

        if signal.direction == "NONE":
            i += 1
            continue

        if regime_series is not None:
            regime = regime_series.iloc[i]
            if not passes_regime_filter(signal.direction, regime, cfg):
                i += 1
                continue

        signal_time = df["close_time"].iloc[i] if "close_time" in df.columns else None
        if cooldown_hours and signal_time is not None:
            last_t = last_signal_time.get(signal.direction)
            if last_t is not None and (signal_time - last_t) < pd.Timedelta(hours=cooldown_hours):
                i += 1
                continue

        # --- MTF agreement bonus (mirrors live scanner logic) ---
        mtf_bonus = 0.0
        agree_tfs: list[str] = []
        if mtf_direction_map and mtf_weight and mtf_bonus_eligible(signal.setup_type, cfg):
            for other_tf, dir_series in mtf_direction_map.items():
                if i < len(dir_series) and dir_series.iloc[i] == signal.direction:
                    agree_tfs.append(other_tf)
            if agree_tfs:
                mtf_bonus = mtf_weight * len(agree_tfs)
                signal.score = round(signal.score + mtf_bonus, 1)
                signal.reasons.append(
                    f"Searah dengan TF {', '.join(agree_tfs)} (+{mtf_bonus} MTF agreement)"
                )

        # Re-check min_score after MTF bonus (same as live)
        if signal.score < cfg["scoring"]["min_score_to_trigger"]:
            i += 1
            continue

        # Pakai passes_risk_filter yang sama dengan live scanner (bukan
        # reimplement manual), supaya validasi no_entry/nan_price/invalid_order/
        # zero_risk/rr_too_low selalu match kalau logikanya berubah di scanner.py.
        if not passes_risk_filter(signal, cfg):
            i += 1
            continue

        if cooldown_hours and signal_time is not None:
            last_signal_time[signal.direction] = signal_time

        future = df.iloc[i + 1 :].reset_index(drop=True)
        tie_break = cfg.get("backtest", {}).get("intrabar_tie_break", "conservative")
        max_holding_bars = int(cfg.get("backtest", {}).get("max_holding_bars", 0))
        result, r_mult, exit_offset, both_touched = _simulate_exit(
            signal, future, fee_pct, tie_break, max_holding_bars=max_holding_bars
        )

        # The signal is generated after candle i closes, so entry_time must be
        # close_time. Reporting open_time made every trade appear one candle
        # earlier than the price at which it was actually entered.
        entry_time = str(df.iloc[i].get("close_time", df.iloc[i].get("open_time", df.index[i])))
        exit_idx = i + 1 + exit_offset
        exit_time = (
            str(df.iloc[exit_idx].get("close_time", df.iloc[exit_idx].get("open_time", df.index[exit_idx])))
            if exit_idx < len(df)
            else ""
        )

        trades.append(
            Trade(
                symbol=symbol, direction=signal.direction, entry=signal.entry,
                sl=signal.sl, tp=signal.tp, result=result, r_multiple=r_mult,
                timeframe=signal.timeframe, setup_type=signal.setup_type,
                entry_time=entry_time, exit_time=exit_time, both_touched=both_touched,
                mtf_bonus=mtf_bonus, mtf_agree_tfs=agree_tfs, score=signal.score,
            )
        )

        if result == "OPEN":
            break
        i = i + 1 + exit_offset + 1

    return trades


def _simulate_exit(
    signal,
    future_df: pd.DataFrame,
    fee_pct: float,
    tie_break: str = "conservative",
    max_holding_bars: int = 0,
) -> tuple[str, float, int, bool]:
    risk = abs(signal.entry - signal.sl)
    fee_r = (signal.entry * fee_pct) / risk if risk > 0 else 0.0

    bars_to_check = future_df
    if max_holding_bars > 0:
        bars_to_check = future_df.iloc[:max_holding_bars]

    for offset, (_, bar) in enumerate(bars_to_check.iterrows()):
        if signal.direction == "LONG":
            hit_sl = bool(bar["low"] <= signal.sl)
            hit_tp = bool(bar["high"] >= signal.tp)
        else:
            hit_sl = bool(bar["high"] >= signal.sl)
            hit_tp = bool(bar["low"] <= signal.tp)

        if hit_sl and hit_tp:
            win = _resolve_tie(tie_break, signal, bar)
        elif hit_sl:
            win = False
        elif hit_tp:
            win = True
        else:
            continue

        if win:
            reward = abs(signal.tp - signal.entry)
            return "WIN", (reward / risk) - fee_r, offset, hit_sl and hit_tp
        return "LOSS", -1.0 - fee_r, offset, hit_sl and hit_tp

    if max_holding_bars > 0 and len(future_df) >= max_holding_bars and len(bars_to_check) > 0:
        last_close = float(bars_to_check.iloc[-1]["close"])
        if signal.direction == "LONG":
            mark_r = (last_close - signal.entry) / risk
        else:
            mark_r = (signal.entry - last_close) / risk
        return "TIMEOUT", mark_r - fee_r, len(bars_to_check) - 1, False

    return "OPEN", 0.0, len(future_df), False


def _resolve_tie(tie_break: str, signal, bar) -> bool:
    if tie_break == "optimistic":
        return True
    if tie_break == "midpoint":
        toward_tp = abs(bar["close"] - signal.tp) < abs(bar["close"] - signal.sl)
        return toward_tp
    return False


def summarize(trades: list[Trade]) -> dict:
    closed = [t for t in trades if t.result != "OPEN"]
    if not closed:
        return {"total": 0, "win_rate": 0.0, "avg_r": 0.0, "tie_count": 0, "tie_pct": 0.0,
                "mtf_trades": 0, "mtf_pct": 0.0}
    wins = [t for t in closed if t.result == "WIN"]
    ties = [t for t in closed if t.both_touched]
    mtf_trades = [t for t in closed if t.mtf_bonus > 0]
    return {
        "total": len(closed),
        "win_rate": round(len(wins) / len(closed) * 100, 1),
        "avg_r": round(sum(t.r_multiple for t in closed) / len(closed), 3),
        "tie_count": len(ties),
        "tie_pct": round(len(ties) / len(closed) * 100, 1),
        "mtf_trades": len(mtf_trades),
        "mtf_pct": round(len(mtf_trades) / len(closed) * 100, 1),
    }


def summarize_by_setup(trades: list[Trade]) -> dict[str, dict]:
    by_setup: dict[str, list[Trade]] = {}
    for t in trades:
        if t.result == "OPEN":
            continue
        by_setup.setdefault(t.setup_type or "UNKNOWN", []).append(t)
    return {setup: summarize(ts) for setup, ts in by_setup.items()}


def summarize_by_direction(trades: list[Trade]) -> dict[str, dict]:
    by_dir: dict[str, list[Trade]] = {}
    for t in trades:
        if t.result == "OPEN":
            continue
        by_dir.setdefault(t.direction, []).append(t)
    return {d: summarize(ts) for d, ts in by_dir.items()}


async def _fetch_regime_df(client: BinanceFuturesClient, timeframe: str, limit: int, cfg: dict):
    regime_cfg = cfg.get("regime_filter", {})
    if not regime_cfg.get("enabled", False):
        return None
    regime_symbol = regime_cfg.get("symbol", "BTCUSDT")
    regime_tf = regime_cfg.get("timeframe", "4h")
    regime_limit = _regime_limit_for(timeframe, limit, regime_tf)
    regime_kline = await fetch_klines(client, regime_symbol, regime_tf, regime_limit)
    return compute_regime_series(regime_kline.df, cfg)


async def _fetch_mtf_data(
    client: BinanceFuturesClient,
    symbol: str,
    primary_tf: str,
    primary_limit: int,
    cfg: dict,
) -> dict[str, pd.DataFrame]:
    """Fetch klines for all other configured timeframes (for MTF agreement)."""
    all_tfs = cfg.get("timeframes", ["15m", "1h", "4h"])
    other_tfs = [tf for tf in all_tfs if tf != primary_tf]
    result = {}
    for tf in other_tfs:
        lim = _mtf_limit_for(primary_tf, primary_limit, tf)
        try:
            kline = await fetch_klines(client, symbol, tf, lim)
            result[tf] = kline.df
        except Exception as exc:
            print(f"  [MTF] gagal fetch {symbol} {tf}: {exc}")
    return result


def _build_mtf_map(
    primary_df: pd.DataFrame,
    mtf_dfs: dict[str, pd.DataFrame],
    symbol: str,
    cfg: dict,
    warmup: int | None = None,
    regime_series: pd.Series | None = None,
) -> dict[str, pd.Series]:
    warmup = warmup if warmup is not None else get_warmup(cfg)
    mtf_map = {}
    for tf, odf in mtf_dfs.items():
        other_regime = None
        if regime_series is not None:
            other_regime = align_series_by_open_time(
                odf, primary_df, regime_series, default="NEUTRAL"
            )
        mtf_map[tf] = _build_mtf_direction_series(
            primary_df, odf, tf, symbol, cfg, warmup=warmup,
            regime_series=other_regime,
        )
    return mtf_map


async def run_single(symbol: str, timeframe: str, limit: int, cfg: dict,
                     use_mtf: bool = True) -> None:
    async with BinanceFuturesClient() as client:
        kline = await fetch_klines(client, symbol, timeframe, limit)
        regime_df = await _fetch_regime_df(client, timeframe, limit, cfg)

        mtf_dfs = {}
        if use_mtf:
            print(f"Fetching MTF data for {symbol} ...")
            mtf_dfs = await _fetch_mtf_data(client, symbol, timeframe, limit, cfg)

    regime_series = align_regime_to(kline.df, regime_df) if regime_df is not None else None
    mtf_map = (
        _build_mtf_map(kline.df, mtf_dfs, symbol, cfg, regime_series=regime_series)
        if mtf_dfs
        else None
    )

    trades = backtest_symbol(
        kline.df, symbol, cfg, timeframe=timeframe,
        regime_series=regime_series, mtf_direction_map=mtf_map,
    )
    summary = summarize(trades)
    setup_breakdown = summarize_by_setup(trades)
    direction_breakdown = summarize_by_direction(trades)
    print(json.dumps(
        {"overall": summary, "by_setup": setup_breakdown, "by_direction": direction_breakdown},
        indent=2,
    ))

    with open("trades_raw.json", "w") as f:
        json.dump([t.__dict__ for t in trades], f, indent=2)

    with open("backtest_result.md", "w") as f:
        f.write(f"# Backtest — {symbol} ({timeframe})\n\n")
        if regime_series is not None:
            f.write("_Market regime filter: **aktif** (BTCUSDT)_\n")
        if mtf_map:
            f.write("_MTF agreement bonus: **aktif**_\n")
        f.write("\n")
        f.write(f"- Total trade tertutup: **{summary['total']}**\n")
        f.write(f"- Win rate: **{summary['win_rate']}%**\n")
        f.write(f"- Rata-rata R multiple: **{summary['avg_r']}**\n")
        f.write(
            f"- Trade dengan SL & TP kesentuh di candle yang sama: "
            f"**{summary['tie_count']}** ({summary['tie_pct']}%)\n"
        )
        f.write(
            f"- Trade yang dapat MTF bonus: "
            f"**{summary['mtf_trades']}** ({summary['mtf_pct']}%)\n\n"
        )
        f.write("## Breakdown per arah\n\n")
        f.write("| Arah | Trade | Win Rate | Avg R |\n")
        f.write("|---|---|---|---|\n")
        for d, s in direction_breakdown.items():
            f.write(f"| {d} | {s['total']} | {s['win_rate']}% | {s['avg_r']} |\n")
        f.write("\n## Breakdown per jenis setup\n\n")
        f.write("| Setup | Trade | Win Rate | Avg R |\n")
        f.write("|---|---|---|---|\n")
        for setup, s in setup_breakdown.items():
            f.write(f"| {setup} | {s['total']} | {s['win_rate']}% | {s['avg_r']} |\n")
        f.write(
            "\n> Dihasilkan otomatis lewat GitHub Actions workflow_dispatch. "
            "Detail per-trade ada di `trades_raw.json`. "
            "SUDAH termasuk: market regime filter, MTF agreement bonus "
            "(kalau diaktifkan di config / flag --mtf), dan cooldown_hours "
            "per arah (sama seperti live scanner). BELUM termasuk: "
            "`max_signals_per_regime_episode` — limit itu lintas-simbol dan "
            "lintas-run di live scanner sehingga tidak direplikasi di sini; "
            "untuk simbol yang searah dengan regime_gated_direction, frekuensi "
            "trade live bisa sedikit lebih rendah dari yang ditunjukkan backtest ini.\n"
        )


async def run_batch(symbols: list[str], timeframe: str, limit: int, cfg: dict,
                    use_mtf: bool = True) -> None:
    per_symbol_results: list[tuple[str, dict | None, str | None]] = []
    all_trades: list[Trade] = []

    async with BinanceFuturesClient() as client:
        regime_df = await _fetch_regime_df(client, timeframe, limit, cfg)

        for symbol in symbols:
            try:
                kline = await fetch_klines(client, symbol, timeframe, limit)
            except Exception as exc:
                per_symbol_results.append((symbol, None, str(exc)))
                continue

            mtf_dfs = {}
            if use_mtf:
                print(f"Fetching MTF data for {symbol} ...")
                mtf_dfs = await _fetch_mtf_data(client, symbol, timeframe, limit, cfg)

            regime_series = align_regime_to(kline.df, regime_df) if regime_df is not None else None
            mtf_map = (
                _build_mtf_map(
                    kline.df, mtf_dfs, symbol, cfg, regime_series=regime_series
                )
                if mtf_dfs
                else None
            )

            trades = backtest_symbol(
                kline.df, symbol, cfg, timeframe=timeframe,
                regime_series=regime_series, mtf_direction_map=mtf_map,
            )
            summary = summarize(trades)
            per_symbol_results.append((symbol, summary, None))
            all_trades.extend(trades)

    combined = summarize(all_trades)
    combined_by_setup = summarize_by_setup(all_trades)
    combined_by_direction = summarize_by_direction(all_trades)
    with open("trades_raw_batch.json", "w") as f:
        json.dump([t.__dict__ for t in all_trades], f, indent=2)
    _write_batch_report(
        per_symbol_results, combined, combined_by_setup, combined_by_direction,
        timeframe,
        regime_enabled=cfg.get("regime_filter", {}).get("enabled", False),
        mtf_enabled=use_mtf,
    )


def _write_batch_report(
    per_symbol_results, combined: dict, combined_by_setup: dict[str, dict],
    combined_by_direction: dict[str, dict], timeframe: str,
    regime_enabled: bool = False, mtf_enabled: bool = False,
) -> None:
    lines = [f"# Backtest Gabungan ({timeframe})\n"]
    if regime_enabled:
        lines.append("_Market regime filter: **aktif** (BTCUSDT)_")
    if mtf_enabled:
        lines.append("_MTF agreement bonus: **aktif**_")
    lines.append("")
    lines.append("| Simbol | Trade | Win Rate | Avg R | MTF% |")
    lines.append("|---|---|---|---|---|")

    for symbol, summary, error in per_symbol_results:
        if error:
            lines.append(f"| {symbol} | - | - | error: {error[:40]} | - |")
        elif summary["total"] == 0:
            lines.append(f"| {symbol} | 0 | - | - | - |")
        else:
            lines.append(
                f"| {symbol} | {summary['total']} | {summary['win_rate']}% | "
                f"{summary['avg_r']} | {summary.get('mtf_pct', 0)}% |"
            )

    lines.append("")
    lines.append("## Ringkasan Gabungan")
    lines.append("(semua trade dari semua simbol digabung jadi satu populasi)\n")
    lines.append(f"- Total trade: **{combined['total']}**")
    lines.append(f"- Win rate gabungan: **{combined['win_rate']}%**")
    lines.append(f"- Avg R gabungan: **{combined['avg_r']}**")
    lines.append(
        f"- Trade dengan SL & TP kesentuh di candle yang sama (ambigu): "
        f"**{combined['tie_count']}** ({combined['tie_pct']}%)"
    )
    lines.append(
        f"- Trade yang dapat MTF bonus: "
        f"**{combined.get('mtf_trades', 0)}** ({combined.get('mtf_pct', 0)}%)"
    )

    lines.append("")
    lines.append("## Breakdown per arah (gabungan semua simbol)")
    lines.append("| Arah | Trade | Win Rate | Avg R |")
    lines.append("|---|---|---|---|")
    for d, s in combined_by_direction.items():
        lines.append(f"| {d} | {s['total']} | {s['win_rate']}% | {s['avg_r']} |")

    lines.append("")
    lines.append("## Breakdown per jenis setup (gabungan semua simbol)")
    lines.append("| Setup | Trade | Win Rate | Avg R |")
    lines.append("|---|---|---|---|")
    for setup, s in combined_by_setup.items():
        lines.append(f"| {setup} | {s['total']} | {s['win_rate']}% | {s['avg_r']} |")

    lines.append("")
    lines.append(
        "> ⚠️ Angka gabungan lebih bisa dipercaya dibanding angka per-simbol "
        "individual, tapi tetap bukan jaminan performa live — belum "
        "memperhitungkan slippage atau funding rate. "
        "Regime filter, MTF agreement bonus, dan cooldown_hours per arah SUDAH "
        "termasuk (kalau diaktifkan). BELUM termasuk `max_signals_per_regime_episode` "
        "— limit cluster-sinyal itu bersifat lintas-simbol & lintas-run di live "
        "scanner, sehingga tidak direplikasi di backtest per-simbol ini; angka "
        "\"jumlah trade\" gabungan di atas bisa sedikit lebih optimis dari yang "
        "live akan izinkan untuk simbol-simbol yang searah regime. "
        "Baris \"tie\" di atas menunjukkan seberapa besar hasil bergantung pada "
        "asumsi tie-break intrabar (lihat `backtest.intrabar_tie_break` di config.yaml)."
    )

    report = "\n".join(lines)
    with open("backtest_batch_result.md", "w") as f:
        f.write(report)
    print(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true", help="Jalankan mode batch (banyak simbol)")
    parser.add_argument("--symbol", default="BTCUSDT", help="Simbol untuk mode single")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS),
                        help="Simbol dipisah koma untuk mode batch")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--days", type=float, default=None,
                        help="Target cakupan kalender (hari) buat window trading, "
                             "sama rata lintas timeframe. Kalau diisi, override --limit.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mtf", action="store_true", default=True,
                        help="Aktifkan MTF agreement bonus (default: on)")
    parser.add_argument("--no-mtf", action="store_true",
                        help="Matikan MTF agreement bonus")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    use_mtf = args.mtf and not args.no_mtf

    warmup = get_warmup(cfg)
    limit = limit_for_days(args.timeframe, args.days, warmup=warmup) if args.days is not None else args.limit
    if args.days is not None:
        print(f"--days {args.days} @ {args.timeframe} -> --limit {limit} "
              f"({args.days:.0f} hari trading + {warmup} warmup, dari scanning.min_history_bars)")
    if use_mtf:
        print("MTF agreement bonus: AKTIF")
    else:
        print("MTF agreement bonus: MATI")

    if args.batch:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        asyncio.run(run_batch(symbols, args.timeframe, limit, cfg, use_mtf=use_mtf))
    else:
        asyncio.run(run_single(args.symbol.upper(), args.timeframe, limit, cfg, use_mtf=use_mtf))


if __name__ == "__main__":
    main()
