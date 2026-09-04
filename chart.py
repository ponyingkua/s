"""vSynapse v3 — chart generator, 1 file.

Bikin gambar chart candlestick bergaya vSch.py (rasio 5:2) lengkap dengan
EMA, Supertrend, volume + volume MA, dan mark Entry/SL/TP — buat posting
ke Binance Square atau sosial media lain.

Layout 2 panel vertikal (harga besar di atas, volume di bawah) — persis
seperti vSch.py. Label Entry/SL/TP ditaruh langsung di sisi kanan chart
(kotak warna solid + teks putih, seperti vSch), bukan di kolom terpisah.
Panel MACD sudah dihilangkan.

Palet warna diambil langsung dari vSch.py, KECUALI warna background/panel
(tetap hitam, bukan putih seperti vSch aslinya) dan warna teks/axis yang
disesuaikan sedikit lebih terang supaya tetap kebaca di atas background
gelap (vSch aslinya didesain untuk background putih).

Contoh:
  python chart.py --symbol BTCUSDT --timeframe 1h
"""
from __future__ import annotations

import argparse
import asyncio
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from scanner import (
    BinanceFuturesClient,
    atr,
    ema,
    score_symbol,
    supertrend,
)

# ============================================================
# STYLE — palet warna dari vSch.py.
#
# Pengecualian (background tetap hitam, bukan putih seperti vSch):
#   BG, PANEL   -> tetap gelap
#   TEXT, AXIS  -> dinaikkan kecerahannya (versi vSch: "#212121"/"#555555"
#                  didesain buat background putih, jadi nyaris tak
#                  kelihatan kalau dipakai apa adanya di atas hitam)
# Semua warna lain (candle, EMA, Supertrend, level, grid, spine, volume
# MA) persis nilai hex yang sama seperti di vSch.py.
# ============================================================

BG = "#0d0d0f"
PANEL = "#0d0d0f"
GRID = "#9e9e9e"
TEXT = "#e8e8ec"
AXIS = "#9e9e9e"
SPINE = "#9e9e9e"

UP = "#26a69a"
DOWN = "#ef5350"

EMA_COLOR = "#1565c0"

ST_UP = "#2e7d32"
ST_DOWN = "#c62828"

ENTRY = "#1565c0"
TP1 = "#00897b"
SL = "#c62828"

VOLUME_MA = "#e65100"

CANDLE_WIDTH = 0.72


# ============================================================
# HELPERS (gaya format & label persis seperti vSch.py)
# ============================================================

def decimals_from_price(price: float) -> int:
    p = abs(float(price))
    if p < 0.0001:
        return 8
    if p < 0.001:
        return 7
    if p < 0.01:
        return 6
    if p < 0.1:
        return 5
    if p < 1:
        return 5
    if p < 10:
        return 4
    if p < 100:
        return 3
    return 2


def format_price(value: float, decimals: int) -> str:
    return f"{float(value):.{int(decimals)}f}"


def _place_level_labels(ax, levels: list, label_x: float) -> None:
    """Kotak label solid warna + teks putih, ditaruh langsung di sumbu
    harga (bukan kolom terpisah) — persis gaya vSch.py."""
    for item in levels:
        ax.text(
            label_x, item["level"], f" {item['text']} ",
            color="#ffffff",
            bbox=dict(facecolor=item["color"], edgecolor="none",
                      boxstyle="round,pad=0.32", alpha=0.95),
            va="center", ha="left", fontweight="bold", fontsize=8,
            zorder=8, clip_on=False,
        )


def _draw_candles(ax, df: pd.DataFrame) -> list:
    """Gambar candlestick gaya vSch (badan alpha 0.92, sumbu rounded-cap).
    Mengembalikan warna tiap candle supaya bisa dipakai ulang di panel volume."""
    colors = []
    for i in range(len(df)):
        row = df.iloc[i]
        open_p, close_p = float(row["open"]), float(row["close"])
        high_p, low_p = float(row["high"]), float(row["low"])
        color = UP if close_p >= open_p else DOWN
        colors.append(color)

        ax.plot([i, i], [low_p, high_p], color=color, linewidth=1.3,
                 solid_capstyle="round", zorder=5)

        body_bottom = min(open_p, close_p)
        body_height = max(abs(close_p - open_p), (high_p - low_p) * 0.012)
        ax.add_patch(Rectangle(
            (i - CANDLE_WIDTH / 2, body_bottom), CANDLE_WIDTH, body_height,
            facecolor=color, edgecolor=color, alpha=0.92, linewidth=0, zorder=6,
        ))
    return colors


def _draw_volume(ax, df: pd.DataFrame, colors: list) -> None:
    """Bar volume + garis volume MA(20) — persis gaya vSch."""
    for i in range(len(df)):
        ax.bar(i, float(df["volume"].iloc[i]), color=colors[i], alpha=0.30,
               width=CANDLE_WIDTH, linewidth=0, zorder=2)

    vol_ma = df["volume"].rolling(20, min_periods=1).mean()
    ax.plot(range(len(df)), vol_ma, color=VOLUME_MA, linewidth=1.2,
             alpha=0.70, zorder=3)


def _draw_supertrend(ax, df: pd.DataFrame, st_dir: pd.Series,
                      period: int, multiplier) -> None:
    """Garis Supertrend dipecah per arah pakai masking (bukan loop segmen
    manual) — persis pendekatan vSch — tanpa garis penghubung palsu saat
    arah berubah.

    Level (upper/lower band) dihitung pakai formula & parameter yang SAMA
    persis dengan scanner.supertrend() (hl2 +/- multiplier * atr(df, period)),
    supaya posisi garis selalu sinkron dengan arah (st_dir) yang dihasilkan
    scanner — bukan nilai default 10/3 yang di-hardcode terpisah."""
    hl2 = (df["high"] + df["low"]) / 2
    atr_val = atr(df, period)
    upper = hl2 + multiplier * atr_val
    lower = hl2 - multiplier * atr_val
    level = lower.where(st_dir == 1, upper)

    x = range(len(df))
    ax.plot(x, level.where(st_dir == 1), color=ST_UP, linewidth=1.4,
             label=f"Supertrend {period} / {multiplier}", zorder=3)
    ax.plot(x, level.where(st_dir == -1), color=ST_DOWN, linewidth=1.4, zorder=3)


def build_chart(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    signal,
    cfg: dict,
    out_path: str,
) -> str:
    chart_cfg = cfg.get("chart", {})
    n_show = chart_cfg.get("candles_shown", 120)

    plot_df = df.tail(n_show).reset_index(drop=True)

    ema_period = cfg["indicators"]["ema"]["period"]
    st_period = cfg["indicators"]["supertrend"]["period"]
    st_mult = cfg["indicators"]["supertrend"]["multiplier"]

    ema_full = ema(df["close"], ema_period).tail(n_show).reset_index(drop=True)
    st_dir_full = supertrend(df, st_period, st_mult).tail(n_show).reset_index(drop=True)

    # --- Ukuran figure: rasio 5:2 (tidak berubah dari sebelumnya) ---
    width_px = chart_cfg.get("width_px", 2000)
    height_ratio = chart_cfg.get("height_ratio", 0.4)  # 2000 x 800 = 5:2
    dpi = 150
    fig_w = width_px / dpi
    fig_h = (width_px * height_ratio) / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(BG)

    # 2 baris: harga (besar) / volume (kecil) — panel MACD dihilangkan.
    gs = GridSpec(
        2, 1, figure=fig,
        height_ratios=[4.2, 0.75],
        hspace=0.06,
        left=0.07, right=0.96, top=0.87, bottom=0.11,
    )
    ax_price = fig.add_subplot(gs[0, 0])
    ax_vol = fig.add_subplot(gs[1, 0], sharex=ax_price)

    for ax in (ax_price, ax_vol):
        ax.set_facecolor(PANEL)
        ax.grid(True, linestyle="-", alpha=0.35, color=GRID)
        ax.set_axisbelow(True)
        ax.tick_params(colors=AXIS, labelcolor=AXIS, labelsize=7.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(SPINE)
            ax.spines[side].set_linewidth(0.8)

    ax_price.tick_params(labelbottom=False)

    # --- Panel harga: candle + EMA + Supertrend ---
    colors = _draw_candles(ax_price, plot_df)
    ax_price.plot(range(len(plot_df)), ema_full, color=EMA_COLOR, linewidth=1.4,
                  label=f"EMA {ema_period}", zorder=4)
    _draw_supertrend(ax_price, plot_df, st_dir_full, st_period, st_mult)

    last_x = len(plot_df) - 1

    # --- Level Entry / SL / TP (garis putus-putus + kotak label) ---
    ref_price = signal.entry if signal.entry is not None else (
        signal.sl if signal.sl is not None else float(plot_df["close"].iloc[-1])
    )
    dec = decimals_from_price(ref_price)

    levels = []
    if signal.entry is not None:
        levels.append({"level": signal.entry, "color": ENTRY,
                        "text": f"ENTRY  {format_price(signal.entry, dec)}"})
    if signal.tp is not None:
        levels.append({"level": signal.tp, "color": TP1,
                        "text": f"TP  {format_price(signal.tp, dec)}"})
    if signal.sl is not None:
        levels.append({"level": signal.sl, "color": SL,
                        "text": f"SL  {format_price(signal.sl, dec)}"})

    for item in levels:
        ax_price.axhline(y=item["level"], color=item["color"], linestyle="--",
                          linewidth=1.0, alpha=0.70, zorder=2)

    # --- Y-limit harga: ikut sertakan level Entry/SL/TP + padding 16% ---
    level_values = [item["level"] for item in levels]
    y_low = min([float(plot_df["low"].min())] + level_values)
    y_high = max([float(plot_df["high"].max())] + level_values)
    y_span = max(y_high - y_low, abs(y_low) * 0.01 if y_low != 0 else 0.01)
    y_padding = y_span * 0.16
    ax_price.set_ylim(y_low - y_padding, y_high + y_padding)

    # --- Margin ekstra di kanan buat kolom label (bukan subplot terpisah) ---
    gap_from_candle = 4.0
    label_width_est = 13.0
    gap_from_edge = 1.8
    extra_margin = gap_from_candle + label_width_est + gap_from_edge
    label_x = last_x + gap_from_candle

    ax_price.set_xlim(-0.6, last_x + extra_margin)
    ax_vol.set_xlim(-0.6, last_x + extra_margin)

    _place_level_labels(ax_price, levels, label_x)

    # --- Panel volume ---
    _draw_volume(ax_vol, plot_df, colors)
    ax_vol.set_ylabel("Vol", color=AXIS, fontsize=8, labelpad=5)
    ax_price.set_ylabel("Price", color=AXIS, fontsize=8.5, labelpad=5)

    # --- Tanggal di sumbu-x bawah, dari kolom open_time (persis gaya vSch) ---
    if "open_time" in plot_df.columns and len(plot_df):
        tick_count = min(6, len(plot_df))
        ticks_idx = (
            np.linspace(0, last_x, tick_count, dtype=int) if tick_count > 1 else [0]
        )
        ax_vol.set_xticks(ticks_idx)
        tick_labels = []
        for t in ticks_idx:
            ts = plot_df["open_time"].iloc[int(t)]
            tick_labels.append(ts.strftime("%d %b  %H:%M") if pd.notna(ts) else str(t))
        ax_vol.set_xticklabels(tick_labels, fontsize=7.5, color=AXIS)

    legend = ax_price.legend(
        loc="upper left", fontsize=7.5, framealpha=0.95,
        facecolor=BG, edgecolor=SPINE, labelcolor=TEXT, borderpad=0.4,
    )
    legend.get_frame().set_linewidth(0.7)

    # --- Header & footer teks (gaya fig.text seperti vSch) ---
    fig.text(0.07, 0.965,
              f"{symbol}  ·  {timeframe}  ·  {signal.direction}  ·  Score {signal.score}",
              fontsize=13, fontweight="bold", color=TEXT, ha="left", va="top")
    fig.text(0.07, 0.02, f"BINANCE FUTURES  ·  {symbol}  ·  {timeframe}",
              fontsize=7, color=AXIS, ha="left", va="bottom")
    fig.text(0.96, 0.02, "Not financial advice",
              fontsize=7, color=AXIS, ha="right", va="bottom")

    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


async def _fetch_and_build(symbol: str, timeframe: str, cfg: dict, out_path: str) -> str:
    async with BinanceFuturesClient() as client:
        limit = max(400, cfg.get("chart", {}).get("candles_shown", 120) + 250)
        kline = await client.get_klines(symbol, timeframe, limit=limit)
    signal = score_symbol(kline.df, symbol, cfg)
    return build_chart(kline.df, symbol, timeframe, signal, cfg, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_path = args.out or f"charts/{args.symbol.upper()}_{args.timeframe}.png"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    result_path = asyncio.run(_fetch_and_build(args.symbol.upper(), args.timeframe, cfg, out_path))
    print(f"Chart disimpan ke {result_path}")


if __name__ == "__main__":
    main()
