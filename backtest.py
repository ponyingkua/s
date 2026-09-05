"""vSynapse v3.1 — backtest engine, 1 file. Mode single (1 simbol) atau
batch (banyak simbol + ringkasan gabungan).

Contoh:
  python backtest.py --symbol BTCUSDT --timeframe 1h
  python backtest.py --batch --timeframe 1h
  python backtest.py --batch --symbols BTCUSDT,ETHUSDT --timeframe 1h

Catatan penting soal parity dengan scanner.py (live):
- Backtest ini menjalankan 1 timeframe per run (--timeframe), sama seperti
  sebelumnya. score_at() sekarang menerima parameter `timeframe` supaya
  tiap Trade tercatat asalnya dari TF apa, dan setiap Trade juga dicatat
  `setup_type`-nya (breakout/pullback/continuation/extended) — sehingga
  bisa dilihat jenis setup mana yang paling profitable.
- Bonus skor "MTF agreement" di scanner.py (live) TIDAK direplikasi di sini,
  karena itu perlu menyelaraskan candle-close antar-TF pada tiap titik
  simulasi (rawan lookahead bias kalau salah implementasi). Jadi angka
  win-rate/avg-R di backtest ini murni dari 1 TF, tanpa bonus MTF —
  anggap sebagai baseline konservatif (skor live bisa sedikit lebih tinggi
  dari ini kalau kebetulan ada agreement dari TF lain).
- v3.2: Market Regime Filter SEKARANG direplikasi di backtest (lihat
  compute_regime_series/align_regime_to) — beda dengan MTF agreement bonus
  di atas, ini feasible dan penting untuk direplikasi karena backtest batch
  sebelumnya (174 trade) yang menemukan masalah SHORT-vs-LONG (t=-3.46) itu
  sendiri yang menyarankan filter ini; tanpa direplikasi di backtest, kita
  tidak akan pernah tahu apakah fix-nya beneran membantu atau tidak.
- v3.3: Regime filter di sini sekarang lewat passes_regime_filter() (impor
  dari scanner.py) yang sama persis dipakai scanner.py live — SHORT & LONG
  bisa punya mode berbeda (short_mode/long_mode di config), bukan lagi
  logika BULL/BEAR simetris yang di-hardcode di sini. Jalankan ulang
  --batch setelah update config untuk lihat apakah short_mode="bear_only"
  benar-benar memperbaiki angka SHORT yang sebelumnya minus di semua setup.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass

import pandas as pd
import yaml

from scanner import BinanceFuturesClient, compute_indicators, passes_regime_filter, score_at

DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "LTCUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT", "AVAXUSDT",
]

TF_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
    "1d": 1440,
}


async def fetch_klines(client: BinanceFuturesClient, symbol: str, timeframe: str, limit: int):
    """Pakai get_klines biasa kalau limit masih dalam batas 1 request Binance
    (<=1500), atau get_klines_paginated kalau lebih besar dari itu — supaya
    --limit besar (misal 5000) otomatis di-pagination tanpa perlu flag
    tambahan di CLI."""
    if limit <= 1500:
        return await client.get_klines(symbol, timeframe, limit=limit)
    return await client.get_klines_paginated(symbol, timeframe, total_limit=limit)


# ---------------------------------------------------------------------------
# Market regime filter (replikasi backtest dari scanner.get_market_regime)
# ---------------------------------------------------------------------------

def _regime_limit_for(primary_timeframe: str, primary_limit: int, regime_timeframe: str,
                       min_warmup: int = 300) -> int:
    """Hitung berapa banyak candle di timeframe regime yang perlu di-fetch
    supaya rentang waktunya menutupi seluruh rentang backtest primer, PLUS
    warmup ekstra biar indikator regime (EMA200/ADX) sendiri sempat
    konvergen sebelum titik awal backtest."""
    primary_minutes = TF_MINUTES.get(primary_timeframe, 60)
    regime_minutes = TF_MINUTES.get(regime_timeframe, 240)
    span_minutes = primary_minutes * primary_limit
    needed_bars = int(span_minutes / regime_minutes) + min_warmup
    return max(needed_bars, min_warmup)


def compute_regime_series(df_regime: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Hitung rezim (BULL/BEAR/NEUTRAL) di SETIAP bar df_regime — beda dari
    scanner.get_market_regime() yang cuma menjawab rezim "saat ini" (bar
    terakhir), backtest butuh rezim di titik waktu manapun sepanjang
    histori. Dipakai bareng align_regime_to() untuk disejajarkan ke
    timeframe primer tanpa lookahead."""
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
    """Sejajarkan rezim (dari TF lebih tinggi) ke tiap bar primary_df, pakai
    rezim dari candle regime TERAKHIR YANG SUDAH CLOSE pada/sebelum waktu
    open bar primer — supaya tidak lookahead (backtest tidak mengintip
    rezim dari candle regime yang faktanya belum kejadian/closed saat itu)."""
    left = primary_df[["open_time"]].reset_index(drop=True).sort_values("open_time")
    right = regime_df.sort_values("close_time").reset_index(drop=True)

    merged = pd.merge_asof(
        left, right, left_on="open_time", right_on="close_time", direction="backward",
    )
    merged["regime"] = merged["regime"].fillna("NEUTRAL")
    return merged.sort_index()["regime"].reset_index(drop=True)


@dataclass
class Trade:
    symbol: str
    direction: str
    entry: float
    sl: float
    tp: float
    result: str  # "WIN" | "LOSS" | "OPEN"
    r_multiple: float  # sudah dikurangi fee
    timeframe: str = ""
    setup_type: str = ""
    entry_time: str = ""
    exit_time: str = ""
    both_touched: bool = False  # True kalau SL & TP sama-sama kena di 1 candle


def backtest_symbol(
    df: pd.DataFrame, symbol: str, cfg: dict, timeframe: str = "", warmup: int = 250,
    regime_series: pd.Series | None = None,
) -> list[Trade]:
    """Skip-ahead: setelah trade dibuka, tidak evaluasi sinyal baru sampai
    trade itu selesai — supaya trade tidak saling tumpang tindih.

    Indikator dihitung SEKALI di seluruh df (bukan direset tiap iterasi
    kayak versi sebelumnya) — lihat catatan lengkap di
    scanner.compute_indicators(). `warmup` cuma menentukan titik mulai
    evaluasi (kasih ruang indikator konvergen dulu), bukan ukuran window
    perhitungan seperti parameter `window` di versi lama.

    `regime_series`, kalau diisi, harus punya panjang & indeks yang sama
    dengan df (lihat align_regime_to()) — dipakai buat menolak sinyal yang
    melawan rezim BTCUSDT, sama persis seperti scanner.py saat live."""
    trades: list[Trade] = []
    fee_pct = cfg.get("backtest", {}).get("fee_round_trip_pct", 0.0)
    ind = compute_indicators(df, cfg)

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

        risk = abs(signal.entry - signal.sl)
        reward = abs(signal.tp - signal.entry)
        rr_min = cfg["risk"]["risk_reward_min"]
        if risk == 0 or (reward / risk) < rr_min - 1e-6:
            i += 1
            continue

        future = df.iloc[i + 1 :].reset_index(drop=True)
        tie_break = cfg.get("backtest", {}).get("intrabar_tie_break", "conservative")
        result, r_mult, exit_offset, both_touched = _simulate_exit(
            signal, future, fee_pct, tie_break
        )

        entry_time = str(df.iloc[i].get("open_time", df.index[i]))
        exit_idx = i + 1 + exit_offset
        exit_time = str(df.iloc[exit_idx].get("open_time", df.index[exit_idx])) if exit_idx < len(df) else ""

        trades.append(
            Trade(
                symbol=symbol, direction=signal.direction, entry=signal.entry,
                sl=signal.sl, tp=signal.tp, result=result, r_multiple=r_mult,
                timeframe=signal.timeframe, setup_type=signal.setup_type,
                entry_time=entry_time, exit_time=exit_time, both_touched=both_touched,
            )
        )

        if result == "OPEN":
            break
        i = i + 1 + exit_offset + 1

    return trades


def _simulate_exit(
    signal, future_df: pd.DataFrame, fee_pct: float, tie_break: str = "conservative"
) -> tuple[str, float, int, bool]:
    """tie_break menentukan hasil kalau SL & TP sama-sama kesentuh di 1 candle
    (nggak bisa dipastikan urutan intrabar-nya tanpa data tick/lower-TF):
      - "conservative": selalu LOSS (asumsi terburuk, ini perilaku lama)
      - "optimistic":   selalu WIN (asumsi terbaik)
      - "midpoint":     lihat posisi close candle relatif ke entry —
                        kalau close lebih dekat ke arah TP, anggap WIN, kalau
                        tidak, anggap LOSS. Kompromi lebih realistis daripada
                        selalu pilih salah satu ekstrem.
    """
    risk = abs(signal.entry - signal.sl)
    fee_r = (signal.entry * fee_pct) / risk if risk > 0 else 0.0

    for offset, (_, bar) in enumerate(future_df.iterrows()):
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

    return "OPEN", 0.0, len(future_df), False


def _resolve_tie(tie_break: str, signal, bar) -> bool:
    if tie_break == "optimistic":
        return True
    if tie_break == "midpoint":
        toward_tp = abs(bar["close"] - signal.tp) < abs(bar["close"] - signal.sl)
        return toward_tp
    return False  # "conservative" (default, sama seperti perilaku lama)


def summarize(trades: list[Trade]) -> dict:
    closed = [t for t in trades if t.result != "OPEN"]
    if not closed:
        return {"total": 0, "win_rate": 0.0, "avg_r": 0.0}
    wins = [t for t in closed if t.result == "WIN"]
    ties = [t for t in closed if t.both_touched]
    return {
        "total": len(closed),
        "win_rate": round(len(wins) / len(closed) * 100, 1),
        "avg_r": round(sum(t.r_multiple for t in closed) / len(closed), 3),
        "tie_count": len(ties),
        "tie_pct": round(len(ties) / len(closed) * 100, 1),
    }


def summarize_by_setup(trades: list[Trade]) -> dict[str, dict]:
    """Breakdown win-rate/avg-R per jenis setup (breakout/pullback/
    continuation/extended) — supaya kelihatan jenis setup mana yang
    layak dipertahankan dan mana yang sebaiknya di-nonaktifkan/di-tuning
    lewat scoring.setup_bonus di config.yaml."""
    by_setup: dict[str, list[Trade]] = {}
    for t in trades:
        if t.result == "OPEN":
            continue
        by_setup.setdefault(t.setup_type or "UNKNOWN", []).append(t)
    return {setup: summarize(ts) for setup, ts in by_setup.items()}


def summarize_by_direction(trades: list[Trade]) -> dict[str, dict]:
    """Breakdown win-rate/avg-R per arah (LONG/SHORT) — supaya asimetri
    seperti yang ditemukan di batch sebelumnya (SHORT jauh lebih lemah dari
    LONG, t=-3.46) langsung kelihatan tiap kali backtest dijalankan ulang,
    tanpa perlu analisis manual dari trades_raw.json."""
    by_dir: dict[str, list[Trade]] = {}
    for t in trades:
        if t.result == "OPEN":
            continue
        by_dir.setdefault(t.direction, []).append(t)
    return {d: summarize(ts) for d, ts in by_dir.items()}


# ---------------------------------------------------------------------------
# Mode single
# ---------------------------------------------------------------------------

async def _fetch_regime_df(client: BinanceFuturesClient, timeframe: str, limit: int, cfg: dict):
    regime_cfg = cfg.get("regime_filter", {})
    if not regime_cfg.get("enabled", False):
        return None
    regime_symbol = regime_cfg.get("symbol", "BTCUSDT")
    regime_tf = regime_cfg.get("timeframe", "4h")
    regime_limit = _regime_limit_for(timeframe, limit, regime_tf)
    regime_kline = await fetch_klines(client, regime_symbol, regime_tf, regime_limit)
    return compute_regime_series(regime_kline.df, cfg)


async def run_single(symbol: str, timeframe: str, limit: int, cfg: dict) -> None:
    async with BinanceFuturesClient() as client:
        kline = await fetch_klines(client, symbol, timeframe, limit)
        regime_df = await _fetch_regime_df(client, timeframe, limit, cfg)

    regime_series = align_regime_to(kline.df, regime_df) if regime_df is not None else None
    trades = backtest_symbol(kline.df, symbol, cfg, timeframe=timeframe, regime_series=regime_series)
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
            f.write("_Market regime filter: **aktif** (BTCUSDT)_\n\n")
        f.write(f"- Total trade tertutup: **{summary['total']}**\n")
        f.write(f"- Win rate: **{summary['win_rate']}%**\n")
        f.write(f"- Rata-rata R multiple: **{summary['avg_r']}**\n")
        f.write(
            f"- Trade dengan SL & TP kesentuh di candle yang sama: "
            f"**{summary['tie_count']}** ({summary['tie_pct']}%)\n\n"
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
            "Detail per-trade ada di `trades_raw.json`. Angka di sini murni 1 "
            "timeframe (tanpa bonus MTF agreement yang dipakai scanner.py saat "
            "live), tapi SUDAH termasuk market regime filter kalau diaktifkan "
            "di config.\n"
        )


# ---------------------------------------------------------------------------
# Mode batch
# ---------------------------------------------------------------------------

async def run_batch(symbols: list[str], timeframe: str, limit: int, cfg: dict) -> None:
    per_symbol_results: list[tuple[str, dict | None, str | None]] = []
    all_trades: list[Trade] = []

    async with BinanceFuturesClient() as client:
        # Rezim di-fetch SEKALI buat seluruh batch (bukan per simbol) —
        # semua simbol pakai timeframe & limit yang sama, jadi rentang
        # waktunya nyaris identik, tidak perlu N-1 fetch tambahan yang sia-sia.
        regime_df = await _fetch_regime_df(client, timeframe, limit, cfg)

        for symbol in symbols:
            try:
                kline = await fetch_klines(client, symbol, timeframe, limit)
            except Exception as exc:
                per_symbol_results.append((symbol, None, str(exc)))
                continue

            regime_series = align_regime_to(kline.df, regime_df) if regime_df is not None else None
            trades = backtest_symbol(
                kline.df, symbol, cfg, timeframe=timeframe, regime_series=regime_series
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
        timeframe, regime_enabled=cfg.get("regime_filter", {}).get("enabled", False),
    )


def _write_batch_report(
    per_symbol_results, combined: dict, combined_by_setup: dict[str, dict],
    combined_by_direction: dict[str, dict], timeframe: str, regime_enabled: bool = False,
) -> None:
    lines = [f"# Backtest Gabungan ({timeframe})\n"]
    if regime_enabled:
        lines.append("_Market regime filter: **aktif** (BTCUSDT)_\n")
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
    lines.append(
        f"- Trade dengan SL & TP kesentuh di candle yang sama (ambigu): "
        f"**{combined['tie_count']}** ({combined['tie_pct']}%)"
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
        "memperhitungkan slippage atau funding rate, dan belum termasuk bonus "
        "MTF agreement yang dipakai scanner.py saat live (regime filter SUDAH "
        "termasuk kalau diaktifkan). Baris \"tie\" di atas menunjukkan seberapa "
        "besar hasil bergantung pada asumsi tie-break intrabar (lihat "
        "`backtest.intrabar_tie_break` di config.yaml). Kalau breakdown arah "
        "atau setup masih timpang jauh, pertimbangkan tuning lebih lanjut di "
        "config.yaml."
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
