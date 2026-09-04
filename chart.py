"""vSynapse v3 — chart generator, 1 file.

Bikin gambar chart candlestick gaya profesional (dark UI, rasio 5:2) lengkap
dengan EMA200, Supertrend, MACD, volume, dan mark Entry/SL/TP + zone box
transparan — buat posting ke Binance Square atau sosial media lain.

Layout dibagi 3 panel vertikal (harga besar di atas, volume, MACD di bawah)
supaya nggak ada elemen yang numpuk. Semua label harga (entry/SL/TP)
ditaruh di kolom terpisah di sisi kanan chart, di luar area candle.

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
import matplotlib.transforms
import pandas as pd
import yaml
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from scanner import (
    BinanceFuturesClient,
    atr,
    ema,
    macd,
    score_symbol,
    supertrend,
)


def _draw_decluttered_labels(ax_labels, level_defs: list, ymin: float, ymax: float, theme: dict,
                              min_gap_frac: float = 0.16) -> None:
    """Taruh label Entry/SL/TP di kolom terpisah dengan jarak minimum antar
    label (dalam fraksi tinggi chart), supaya tidak saling timpa walau
    harganya berdekatan. Tiap label tetap ditarik garis tipis ke posisi
    harga aslinya biar jelas keterkaitannya."""
    if not level_defs or ymax <= ymin:
        return

    items = sorted(level_defs, key=lambda t: t[1])  # urut naik berdasarkan harga
    fracs = [(value - ymin) / (ymax - ymin) for _, value, _ in items]

    # dorong ke atas kalau terlalu rapat
    for i in range(1, len(fracs)):
        if fracs[i] - fracs[i - 1] < min_gap_frac:
            fracs[i] = fracs[i - 1] + min_gap_frac
    # kalau meluber di atas 1.0, geser semua turun secukupnya
    if fracs[-1] > 0.95:
        shift = fracs[-1] - 0.95
        fracs = [max(0.05, f - shift) for f in fracs]

    trans = matplotlib.transforms.blended_transform_factory(ax_labels.transAxes, ax_labels.transAxes)

    for (label, value, color), frac in zip(items, fracs):
        real_frac = (value - ymin) / (ymax - ymin)

        # garis penghubung tipis dari posisi harga asli ke posisi label yang digeser
        ax_labels.plot([0, 0.1], [real_frac, frac], transform=trans, color=color, linewidth=0.8, alpha=0.7)

        ax_labels.annotate(
            f"{label}\n{value:.6g}",
            xy=(0.12, frac), xycoords="axes fraction",
            va="center", ha="left", fontsize=8, color=color,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=theme.get("background", "#0d0d0f"),
                      edgecolor=color, linewidth=0.8),
        )


def _draw_candles(ax, df: pd.DataFrame, theme: dict) -> None:
    width = 0.6
    for idx, row in df.iterrows():
        color = theme["bull"] if row["close"] >= row["open"] else theme["bear"]
        # sumbu (high-low)
        ax.plot([idx, idx], [row["low"], row["high"]], color=color, linewidth=1, zorder=2)
        # badan (open-close)
        lower = min(row["open"], row["close"])
        height = abs(row["close"] - row["open"]) or (row["high"] - row["low"]) * 0.01
        ax.add_patch(
            Rectangle(
                (idx - width / 2, lower), width, height,
                facecolor=color, edgecolor=color, zorder=3,
            )
        )


def _draw_supertrend(ax, df: pd.DataFrame, st: pd.Series, theme: dict) -> None:
    """Gambar supertrend sebagai segmen warna beda per arah, tanpa garis
    penghubung palsu saat arah berubah (biar tidak menyesatkan secara visual)."""
    hl2 = (df["high"] + df["low"]) / 2
    atr_val = atr(df, 10)
    upper = hl2 + 3 * atr_val
    lower = hl2 - 3 * atr_val

    seg_x, seg_y, seg_color = [], [], None
    for i in range(len(df)):
        level = lower.iloc[i] if st.iloc[i] == 1 else upper.iloc[i]
        color = theme["supertrend_up"] if st.iloc[i] == 1 else theme["supertrend_down"]
        if seg_color is not None and color != seg_color:
            ax.plot(seg_x, seg_y, color=seg_color, linewidth=1.6, zorder=4)
            seg_x, seg_y = [], []
        seg_x.append(df.index[i])
        seg_y.append(level)
        seg_color = color
    if seg_x:
        ax.plot(seg_x, seg_y, color=seg_color, linewidth=1.6, zorder=4)


def build_chart(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    signal,
    cfg: dict,
    out_path: str,
) -> str:
    chart_cfg = cfg.get("chart", {})
    theme = chart_cfg.get("theme", {})
    n_show = chart_cfg.get("candles_shown", 120)

    plot_df = df.tail(n_show).reset_index(drop=True)

    ema200_full = ema(df["close"], cfg["indicators"]["ema"]["period"]).tail(n_show).reset_index(drop=True)
    st_full = supertrend(
        df, cfg["indicators"]["supertrend"]["period"], cfg["indicators"]["supertrend"]["multiplier"]
    ).tail(n_show).reset_index(drop=True)
    macd_line_full, signal_line_full, hist_full = macd(
        df["close"], cfg["indicators"]["macd"]["fast"], cfg["indicators"]["macd"]["slow"],
        cfg["indicators"]["macd"]["signal"],
    )
    macd_line = macd_line_full.tail(n_show).reset_index(drop=True)
    signal_line = signal_line_full.tail(n_show).reset_index(drop=True)
    hist = hist_full.tail(n_show).reset_index(drop=True)

    width_px = chart_cfg.get("width_px", 2000)
    height_ratio = chart_cfg.get("height_ratio", 0.4)
    dpi = 150
    fig_w = width_px / dpi
    fig_h = (width_px * height_ratio) / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(theme.get("background", "#0d0d0f"))

    # 4 kolom: [panel utama lebar] [kolom label harga sempit]
    # 3 baris: harga (besar) / volume (kecil) / macd (kecil)
    gs = GridSpec(
        3, 2, figure=fig,
        width_ratios=[9, 1.4],
        height_ratios=[3.4, 1, 1],
        hspace=0.08, wspace=0.02,
        left=0.05, right=0.97, top=0.90, bottom=0.09,
    )

    ax_price = fig.add_subplot(gs[0, 0])
    ax_labels = fig.add_subplot(gs[0, 1], sharey=ax_price)
    ax_vol = fig.add_subplot(gs[1, 0], sharex=ax_price)
    ax_macd = fig.add_subplot(gs[2, 0], sharex=ax_price)

    for ax in (ax_price, ax_vol, ax_macd, ax_labels):
        ax.set_facecolor(theme.get("background", "#0d0d0f"))
        for spine in ax.spines.values():
            spine.set_color(theme.get("grid", "#1c1c22"))
        ax.tick_params(colors=theme.get("text", "#d8d8e0"), labelsize=8)

    ax_price.grid(True, color=theme.get("grid", "#1c1c22"), linewidth=0.5, alpha=0.6)

    # --- Panel harga ---
    _draw_candles(ax_price, plot_df, theme)
    ax_price.plot(range(len(plot_df)), ema200_full, color=theme.get("ema", "#f5c542"),
                  linewidth=1.3, label="EMA200", zorder=5)
    ax_price.plot([], [], color=theme.get("supertrend_up", "#2ecc71"), linewidth=1.6, label="Supertrend")
    _draw_supertrend(ax_price, plot_df, st_full, theme)

    # Zone box transparan: area antara entry dan SL (zona risiko)
    if signal.entry is not None and signal.sl is not None:
        lo, hi = sorted([signal.entry, signal.sl])
        ax_price.axhspan(lo, hi, color=theme.get("zone_fill", "#3498db"),
                          alpha=theme.get("zone_alpha", 0.12), zorder=1)

    # Garis Entry / SL / TP + label di kolom terpisah (tidak numpuk di atas candle)
    level_defs = []
    if signal.entry is not None:
        level_defs.append(("Entry", signal.entry, theme.get("entry", "#3498db")))
    if signal.sl is not None:
        level_defs.append(("SL", signal.sl, theme.get("sl", "#e74c3c")))
    if signal.tp is not None:
        level_defs.append(("TP", signal.tp, theme.get("tp", "#2ecc71")))

    for label, level, color in level_defs:
        ax_price.axhline(level, color=color, linewidth=1, linestyle="--", alpha=0.8, zorder=4)

    ax_labels.axis("off")
    ymin, ymax = ax_price.get_ylim()
    _draw_decluttered_labels(ax_labels, level_defs, ymin, ymax, theme)

    ax_price.legend(loc="upper left", fontsize=8, facecolor=theme.get("background", "#0d0d0f"),
                     edgecolor=theme.get("grid", "#1c1c22"), labelcolor=theme.get("text", "#d8d8e0"))

    title = f"{symbol}  ·  {timeframe}  ·  {signal.direction}  ·  Score {signal.score}"
    ax_price.set_title(title, color=theme.get("text", "#d8d8e0"), fontsize=13, loc="left", pad=10)
    ax_price.set_xticklabels([])
    ax_price.set_xlim(-1, len(plot_df))

    # --- Panel volume ---
    vol_colors = [
        theme.get("bull", "#2ecc71") if row["close"] >= row["open"] else theme.get("bear", "#e74c3c")
        for _, row in plot_df.iterrows()
    ]
    ax_vol.bar(range(len(plot_df)), plot_df["volume"], color=vol_colors, width=0.6, zorder=2)
    ax_vol.set_xticklabels([])
    ax_vol.set_ylabel("Vol", color=theme.get("text", "#d8d8e0"), fontsize=8)
    ax_vol.grid(True, color=theme.get("grid", "#1c1c22"), linewidth=0.4, alpha=0.4)

    # --- Panel MACD ---
    hist_colors = [theme.get("bull", "#2ecc71") if v >= 0 else theme.get("bear", "#e74c3c") for v in hist]
    ax_macd.bar(range(len(plot_df)), hist, color=hist_colors, width=0.6, alpha=0.6, zorder=2)
    ax_macd.plot(range(len(plot_df)), macd_line, color=theme.get("ema", "#f5c542"), linewidth=1, zorder=3)
    ax_macd.plot(range(len(plot_df)), signal_line, color=theme.get("text", "#d8d8e0"), linewidth=1, zorder=3)
    ax_macd.set_ylabel("MACD", color=theme.get("text", "#d8d8e0"), fontsize=8)
    ax_macd.grid(True, color=theme.get("grid", "#1c1c22"), linewidth=0.4, alpha=0.4)

    # kolom label kosong di baris volume & macd biar grid tetap rapi (tanpa isi)
    for row in (1, 2):
        ax_empty = fig.add_subplot(gs[row, 1])
        ax_empty.axis("off")

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