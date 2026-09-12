"""
vSynapse Multi-Timeframe Chart Generator for analyze.py
Visualizes exactly what analyze.py calculates per timeframe:
- EMA 20, 50, 200
- Support & Resistance Levels (from analyze.py _levels)
- Swing High & Swing Low markers (HH, HL, LH, LL)
- RSI & MACD Subplots (or Bollinger/Volume alignment)
- Layout match with clean dark UI style (similar to RIVERUSDT_multi.png)
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

from analyze import (
    analyze_symbol,
    normalize_symbol,
    _rsi,
    _macd,
    _bollinger,
    _swing_points,
    _levels
)

# Color Palette (Dark Theme matching output UI)
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
    """Renders single timeframe candlestick chart + indicators from analyze.py."""
    df = df.copy().reset_index(drop=True)
    n = len(df)
    x = np.arange(n)

    # 1. Plot Candlesticks
    for i in range(n):
        open_p, high_p, low_p, close_p = df.loc[i, ["open", "high", "low", "close"]]
        color = BULL_COLOR if close_p >= open_p else BEAR_COLOR
        # Wick
        ax_price.plot([i, i], [low_p, high_p], color=color, linewidth=1.2, alpha=0.85)
        # Body
        body_bottom = min(open_p, close_p)
        body_height = max(abs(close_p - open_p), (high_p - low_p) * 0.001)
        ax_price.bar(i, body_height, bottom=body_bottom, color=color, width=0.6, align="center")

    # 2. Indicators calculated as in analyze.py
    close = df["close"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    
    ax_price.plot(x, ema20, color=ACCENT_YELLOW, linewidth=1.2, label="EMA 20", alpha=0.9)
    ax_price.plot(x, ema50, color=ACCENT_BLUE, linewidth=1.2, label="EMA 50", alpha=0.9)

    if n >= 200:
        ema200 = close.ewm(span=200, adjust=False).mean()
        ax_price.plot(x, ema200, color=ACCENT_PURPLE, linewidth=1.2, label="EMA 200", alpha=0.9)

    # 3. Support & Resistance from analyze.py
    if "structure" in tf_info:
        st = tf_info["structure"]
        sup, res = st.get("support"), st.get("resistance")
        if sup:
            ax_price.axhline(sup, color=BULL_COLOR, linestyle="--", linewidth=1.0, alpha=0.6)
        if res:
            ax_price.axhline(res, color=BEAR_COLOR, linestyle="--", linewidth=1.0, alpha=0.6)

    # 4. Mark Swing Highs & Swing Lows (Visualizing Structure)
    sh, sl = _swing_points(df, left=3, right=3)
    for idx, val in sh:
        if idx < n:
            ax_price.scatter(idx, val * 1.002, marker="v", color=BEAR_COLOR, s=15, alpha=0.7)
    for idx, val in sl:
        if idx < n:
            ax_price.scatter(idx, val * 0.998, marker="^", color=BULL_COLOR, s=15, alpha=0.7)

    # 5. Volume Subplot
    for i in range(n):
        open_p, close_p, vol = df.loc[i, ["open", "close", "volume"]]
        color = BULL_COLOR if close_p >= open_p else BEAR_COLOR
        ax_vol.bar(i, vol, color=color, width=0.6, alpha=0.6)
    
    # Volume MA 20
    vma = df["volume"].rolling(20).mean()
    ax_vol.plot(x, vma, color=ACCENT_YELLOW, linewidth=1.0, alpha=0.8)

    # Styling Panel
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


def build_analyzed_multi_tf_card(
    dfs: dict[str, pd.DataFrame],
    symbol: str,
    timeframes: list[str],
    per_tf: dict[str, dict],
    out_path: str,
) -> str:
    """Creates a unified multi-timeframe grid matching analyze.py output."""
    valid_tfs = [tf for tf in timeframes if tf in dfs and tf in per_tf]
    num_tfs = len(valid_tfs)

    if num_tfs == 0:
        raise ValueError("No valid dataframes to render MTF chart.")

    fig = plt.figure(figsize=(6 * num_tfs, 7), facecolor=BG_COLOR)
    gs = gridspec.GridSpec(2, num_tfs, height_ratios=[3.5, 1.0], hspace=0.05, wspace=0.15)

    for idx, tf in enumerate(valid_tfs):
        ax_price = fig.add_subplot(gs[0, idx])
        ax_vol = fig.add_subplot(gs[1, idx], sharex=ax_price)
        render_tf_panel(ax_price, ax_vol, dfs[tf], tf, per_tf[tf])

    # Overall Header & Footer
    fig.suptitle(f"{symbol.upper()}  ·  MULTI-TIMEFRAME ANALYSIS", color=TEXT_COLOR, fontsize=16, fontweight="bold", x=0.02, y=0.96, ha="left")
    
    fig.text(0.02, 0.02, "BINANCE FUTURES · vSynapse Analysis Engine", color=MUTED_TEXT, fontsize=8, ha="left")
    fig.text(0.98, 0.02, "Visualized indicators: EMA 20/50/200, Volume MA, S/R Levels, Swing Markers", color=MUTED_TEXT, fontsize=8, ha="right")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    return out_path


async def main():
    parser = argparse.ArgumentParser(description="Generate Multi-TF Visual Chart for analyze.py")
    parser.add_argument("--symbol", required=True, help="Coin symbol (e.g. BTCUSDT)")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--out", default="analysis_output/multi_tf.png", help="Output path")
    args = parser.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}

    symbol = normalize_symbol(args.symbol, cfg.get("quote_asset", "USDT"))
    res = await analyze_symbol(symbol, cfg)
    
    build_analyzed_multi_tf_card(
        dfs=res["dfs"],
        symbol=res["symbol"],
        timeframes=cfg.get("timeframes", list(res["per_tf"].keys())),
        per_tf=res["per_tf"],
        out_path=args.out,
    )
    print(f"Successfully generated clean MTF analysis chart at: {args.out}")

if __name__ == "__main__":
    asyncio.run(main())