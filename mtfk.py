"""
mtfk.py — Multi-Timeframe Chart Visualizer for analyze.py

Fungsi utama:
- Memvisualisasikan hasil analisa dari analyze.py per timeframe.
- Mengadopsi styling layout dark-theme modern dari chart.py.
- Menampilkan penanda visual poin analisa:
  * EMA 20, 50, dan 200 (jika bar mencukupi).
  * Area Support & Resistance dari _levels().
  * Penanda Swing High (v) & Swing Low (^) dari _swing_points().
  * Volume & Volume MA 20.
- Tidak menampilkan garis Entry, TP, dan SL (murni visualisasi struktur & tren).
"""

from __future__ import annotations

import os
import argparse
import asyncio
import yaml
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Mengimpor modul analisis dan helper dasar
from analyze import analyze_symbol, normalize_symbol, _swing_points

# Skema Warna Dark UI (Menyesuaikan standar tampilan chart.py)
BG_COLOR = "#0D1117"
PANEL_BG = "#161B22"
GRID_COLOR = "#21262D"
TEXT_COLOR = "#C9D1D9"
MUTED_TEXT = "#8B949E"
BULL_COLOR = "#26A69A"
BEAR_COLOR = "#EF5350"
ACCENT_YELLOW = "#F1C40F"
ACCENT_BLUE = "#29B6F6"
ACCENT_PURPLE = "#AB47BC"


def render_tf_panel(
    ax_price: plt.Axes,
    ax_vol: plt.Axes,
    df: pd.DataFrame,
    tf_name: str,
    tf_info: dict,
):
    """Menggambar candlestick dan penanda analisa untuk 1 timeframe."""
    df = df.copy().reset_index(drop=True)
    n = len(df)
    x = np.arange(n)

    # 1. Plot Candlesticks
    for i in range(n):
        open_p, high_p, low_p, close_p = df.loc[i, ["open", "high", "low", "close"]]
        color = BULL_COLOR if close_p >= open_p else BEAR_COLOR
        # Wick
        ax_price.plot([i, i], [low_p, high_p], color=color, linewidth=1.1, alpha=0.85)
        # Body
        body_bottom = min(open_p, close_p)
        body_height = max(abs(close_p - open_p), (high_p - low_p) * 0.001)
        ax_price.bar(i, body_height, bottom=body_bottom, color=color, width=0.6, align="center")

    # 2. Indikator EMA (Sesuai perhitungan analyze.py)
    close = df["close"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    ax_price.plot(x, ema20, color=ACCENT_YELLOW, linewidth=1.2, label="EMA 20", alpha=0.9)
    ax_price.plot(x, ema50, color=ACCENT_BLUE, linewidth=1.2, label="EMA 50", alpha=0.9)

    if n >= 200:
        ema200 = close.ewm(span=200, adjust=False).mean()
        ax_price.plot(x, ema200, color=ACCENT_PURPLE, linewidth=1.2, label="EMA 200", alpha=0.9)

    # 3. Penanda Support & Resistance (Dari struktur analyze.py)
    if "structure" in tf_info:
        st = tf_info["structure"]
        sup, res = st.get("support"), st.get("resistance")
        if sup:
            ax_price.axhline(sup, color=BULL_COLOR, linestyle="--", linewidth=1.0, alpha=0.6)
        if res:
            ax_price.axhline(res, color=BEAR_COLOR, linestyle="--", linewidth=1.0, alpha=0.6)

    # 4. Penanda Swing High & Swing Low
    sh, sl = _swing_points(df, left=3, right=3)
    for idx, val in sh:
        if idx < n:
            ax_price.scatter(idx, val * 1.002, marker="v", color=BEAR_COLOR, s=15, alpha=0.75)
    for idx, val in sl:
        if idx < n:
            ax_price.scatter(idx, val * 0.998, marker="^", color=BULL_COLOR, s=15, alpha=0.75)

    # 5. Volume Subplot & Volume MA
    for i in range(n):
        open_p, close_p, vol = df.loc[i, ["open", "close", "volume"]]
        color = BULL_COLOR if close_p >= open_p else BEAR_COLOR
        ax_vol.bar(i, vol, color=color, width=0.6, alpha=0.6)

    vma = df["volume"].rolling(20).mean()
    ax_vol.plot(x, vma, color=ACCENT_YELLOW, linewidth=1.0, alpha=0.8)

    # 6. Styling Header Panel (Menampilkan Arah Bias & Jenis Setup)
    direction = tf_info.get("direction", "NONE")
    setup_type = tf_info.get("setup", {}).get("type", "NONE") if isinstance(tf_info.get("setup"), dict) else "NONE"
    
    dir_color = BULL_COLOR if direction == "LONG" else BEAR_COLOR if direction == "SHORT" else MUTED_TEXT
    title_str = f"{tf_name}  ·  {direction}  ·  {setup_type}"

    ax_price.set_title(title_str, color=dir_color, fontsize=11, fontweight="bold", loc="left", pad=8)

    for ax in (ax_price, ax_vol):
        ax.set_facecolor(PANEL_BG)
        ax.grid(True, color=GRID_COLOR, linestyle="-", linewidth=0.5, alpha=0.5)
        ax.tick_params(colors=MUTED_TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)

    ax_price.set_xlim(-1, n)
    ax_vol.set_xlim(-1, n)
    ax_price.xaxis.set_ticklabels([])


def build_mtfk_chart(
    dfs: dict[str, pd.DataFrame],
    symbol: str,
    timeframes: list[str],
    per_tf: dict[str, dict],
    out_path: str,
) -> str:
    """Membuat grid multi-timeframe secara dinamis berdasarkan data analyze.py."""
    valid_tfs = [tf for tf in timeframes if tf in dfs and tf in per_tf]
    num_tfs = len(valid_tfs)

    if num_tfs == 0:
        raise ValueError("Tidak ada data timeframe yang valid untuk membuat chart.")

    fig = plt.figure(figsize=(6 * num_tfs, 7), facecolor=BG_COLOR)
    gs = gridspec.GridSpec(2, num_tfs, height_ratios=[3.5, 1.0], hspace=0.05, wspace=0.15)

    for idx, tf in enumerate(valid_tfs):
        ax_price = fig.add_subplot(gs[0, idx])
        ax_vol = fig.add_subplot(gs[1, idx], sharex=ax_price)
        render_tf_panel(ax_price, ax_vol, dfs[tf], tf, per_tf[tf])

    # Header & Footer Utama
    fig.suptitle(
        f"{symbol.upper()}  ·  MULTI-TIMEFRAME ANALYSIS",
        color=TEXT_COLOR,
        fontsize=16,
        fontweight="bold",
        x=0.02,
        y=0.96,
        ha="left",
    )

    fig.text(0.02, 0.02, "BINANCE FUTURES · vSynapse Visualizer Engine", color=MUTED_TEXT, fontsize=8, ha="left")
    fig.text(
        0.98,
        0.02,
        "Indicators: EMA 20/50/200 | Volume MA | Support & Resistance | Swing Points",
        color=MUTED_TEXT,
        fontsize=8,
        ha="right",
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    return out_path


async def main():
    parser = argparse.ArgumentParser(description="Generate Multi-TF Chart Visualizer (mtfk.py)")
    parser.add_argument("--symbol", required=True, help="Kode koin, contoh: BTCUSDT atau RIVER")
    parser.add_argument("--config", default="config.yaml", help="Path ke file konfigurasi")
    parser.add_argument("--out", default="analysis_output/mtfk_chart.png", help="Path file output gambar")
    args = parser.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    quote = cfg.get("exchange", {}).get("quote_asset", "USDT")
    symbol = normalize_symbol(args.symbol, quote)

    # Menjalankan analisa independen menggunakan analyze.py
    res = await analyze_symbol(symbol, cfg)

    build_mtfk_chart(
        dfs=res["dfs"],
        symbol=res["symbol"],
        timeframes=cfg.get("timeframes", list(res["per_tf"].keys())),
        per_tf=res["per_tf"],
        out_path=args.out,
    )
    print(f"[mtfk] Berhasil membuat visual chart di: {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
