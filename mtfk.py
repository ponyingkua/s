"""
mtfk.py — Multi-Timeframe Chart Renderer

Murni fungsi build_mtfk_chart. Entry point tetap analyze.py.
Styling & helper gambar dari chart.py; indikator (EMA 20/50/200,
S/R, swing) dari hasil analyze.py.
"""

from __future__ import annotations

import os

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import chart

EMA20_COLOR = "#FFD54F"
EMA50_COLOR = "#29B6F6"
EMA200_COLOR = "#AB47BC"


def _draw_analyze_indicators(ax, df: pd.DataFrame, n_show: int, tf_info: dict,
                              swing_context: int = 30) -> None:
    from analyze import _swing_points  # lazy — hindari circular import

    close = df["close"]
    ema20 = close.ewm(span=20, adjust=False).mean().tail(n_show).to_numpy()
    ema50 = close.ewm(span=50, adjust=False).mean().tail(n_show).to_numpy()
    x = range(n_show)

    # Garis EMA lebih tipis & soft
    ax.plot(x, ema20, color=EMA20_COLOR, linewidth=0.85, alpha=0.90, zorder=4, solid_capstyle="round")
    ax.plot(x, ema50, color=EMA50_COLOR, linewidth=0.85, alpha=0.90, zorder=4, solid_capstyle="round")
    if len(df) >= 200:
        ema200 = close.ewm(span=200, adjust=False).mean().tail(n_show).to_numpy()
        ax.plot(x, ema200, color=EMA200_COLOR, linewidth=0.85, alpha=0.90, zorder=4, solid_capstyle="round")

    structure = tf_info.get("structure") or {}
    support, resistance = structure.get("support"), structure.get("resistance")
    if support:
        ax.axhline(support, color=chart.UP, linestyle="--", linewidth=0.9, alpha=0.50, zorder=3)
    if resistance:
        ax.axhline(resistance, color=chart.DOWN, linestyle="--", linewidth=0.9, alpha=0.50, zorder=3)

    work = df.tail(n_show + swing_context).reset_index(drop=True)
    offset = len(work) - n_show
    sh, sl = _swing_points(work, left=3, right=3)
    for idx, val in sh:
        if idx >= offset:
            ax.scatter(idx - offset, val * 1.004, marker="v", color=chart.DOWN, s=12, alpha=0.75, zorder=5)
    for idx, val in sl:
        if idx >= offset:
            ax.scatter(idx - offset, val * 0.996, marker="^", color=chart.UP, s=12, alpha=0.75, zorder=5)


def build_mtfk_chart(
    dfs: dict,
    symbol: str,
    timeframes: list,
    per_tf: dict,
    out_path: str,
    cfg: dict | None = None,
    square: bool = False,
) -> str:
    cfg = cfg or {}
    valid_tfs = [tf for tf in timeframes if tf in dfs and tf in per_tf]
    n_panels = len(valid_tfs)
    if n_panels == 0:
        raise ValueError("Tidak ada data timeframe yang valid untuk membuat chart.")

    chart_cfg = cfg.get("chart", {})
    width_px = chart_cfg.get("width_px", 2800)
    dpi = 200
    fig_w = width_px / dpi
    fig_h = fig_w if square else fig_w * 0.38
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(chart.BG)

    if square:
        outer = GridSpec(n_panels, 1, figure=fig, hspace=0.32,
                          left=0.09, right=0.93, top=0.90, bottom=0.055)
    else:
        outer = GridSpec(1, n_panels, figure=fig, wspace=0.14,
                          left=0.045, right=0.98, top=0.86, bottom=0.10)

    tick_fs = 9.0 if square else 6.5
    title_fs = 13.5 if square else 9.5
    price_fs = 11.0 if square else 8.0

    for idx, tf in enumerate(valid_tfs):
        df = dfs[tf]
        tf_info = per_tf[tf]
        has_error = "error" in tf_info

        n_show = chart.get_candles_shown(tf, cfg)
        plot_df = df.tail(n_show).reset_index(drop=True)

        cell = outer[idx, 0] if square else outer[0, idx]
        # Volume lebih pendek (mirip proporsi chart.py)
        inner = cell.subgridspec(2, 1, height_ratios=[5.8, 1], hspace=0.06)
        ax_p = fig.add_subplot(inner[0, 0])
        ax_v = fig.add_subplot(inner[1, 0], sharex=ax_p)

        for ax in (ax_p, ax_v):
            ax.set_facecolor(chart.PANEL)
            ax.grid(True, linestyle="-", alpha=0.55, color=chart.GRID, linewidth=0.35)
            ax.set_axisbelow(True)
            ax.tick_params(colors=chart.AXIS, labelcolor=chart.AXIS, labelsize=tick_fs)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(chart.SPINE)
                ax.spines[side].set_linewidth(0.55)
        ax_p.tick_params(labelbottom=False)

        colors = chart._draw_candles(ax_p, plot_df)
        if not has_error:
            _draw_analyze_indicators(ax_p, df, n_show, tf_info)

        last_x = len(plot_df) - 1
        y_low = float(plot_df["low"].min())
        y_high = float(plot_df["high"].max())
        y_span = max(y_high - y_low, abs(y_low) * 0.01 if y_low != 0 else 0.01)
        pad = y_span * 0.10
        ax_p.set_ylim(y_low - pad, y_high + pad)
        ax_p.set_xlim(-0.5, last_x + 0.5)
        ax_v.set_xlim(-0.5, last_x + 0.5)

        vol_lookback = cfg.get("indicators", {}).get("volume_spike", {}).get("lookback", 20)
        chart._draw_volume(ax_v, plot_df, colors, vol_lookback)

        direction = "NONE" if has_error else tf_info.get("direction", "NONE")
        setup_info = tf_info.get("setup") if isinstance(tf_info.get("setup"), dict) else {}
        setup_type = "NONE" if has_error else setup_info.get("type", "NONE")
        badge_color = chart.UP if direction == "LONG" else chart.DOWN if direction == "SHORT" else chart.AXIS
        setup_txt = f"  ·  {setup_type}" if setup_type and setup_type != "NONE" else ""
        ax_p.set_title(f"{tf}  ·  {direction}{setup_txt}", color=badge_color,
                        fontsize=title_fs, fontweight="bold", loc="left", pad=5)

        dec = chart.decimals_from_price(float(plot_df["close"].iloc[-1]))
        last_price = chart.format_price(plot_df["close"].iloc[-1], dec)
        ax_p.text(0.99, 0.03, last_price, transform=ax_p.transAxes, color=chart.TEXT,
                   fontsize=price_fs, fontweight="bold", ha="right", va="bottom", zorder=9)

        if has_error:
            ax_p.text(0.5, 0.5, "NO DATA", transform=ax_p.transAxes, color=chart.DOWN,
                       fontsize=title_fs, fontweight="bold", ha="center", va="center")

    header_fs = 20.0 if square else 16.5
    badge_fs = 17.0 if square else 13.5
    footer_fs = 10.0 if square else 6.5
    disclaimer_fs = 9.0 if square else 6.0

    fig.text(0.045, 0.95, f"{symbol}  ·  MULTI-TIMEFRAME", fontsize=header_fs,
              fontweight="bold", color=chart.TEXT, ha="left", va="top")
    ref_df = dfs[valid_tfs[0]]
    chart._draw_change_badge(fig, 0.975, 0.95, chart._calc_24h_change(ref_df), fontsize=badge_fs)
    fig.text(0.045, 0.02, f"BINANCE FUTURES  ·  {symbol}", fontsize=footer_fs,
              color=chart.AXIS, ha="left", va="bottom")
    fig.text(0.98, 0.02,
              "Chart-based analysis for educational purposes only. NOT FINANCIAL ADVICE, DYOR.",
              fontsize=disclaimer_fs, fontweight="bold", color=chart.TEXT, ha="right", va="bottom")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor(), dpi=dpi * 2)
    plt.close(fig)
    return out_path
