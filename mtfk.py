from __future__ import annotations

import os

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

import chart

EMA20_COLOR = "#FFD54F"
EMA50_COLOR = "#29B6F6"
EMA200_COLOR = "#AB47BC"

TINT_LONG = (0.15, 0.65, 0.45, 0.05)
TINT_SHORT = (0.85, 0.25, 0.25, 0.05)
TINT_NONE = (0.5, 0.5, 0.5, 0.03)


def _fmt_volume(value: float, _pos=None) -> str:
    v = abs(value)
    if v >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if v >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def _time_axis_labels(df: pd.DataFrame, n_show: int, tf: str):
    tail = df.tail(n_show)
    m = len(tail)
    if m == 0:
        return None
    ts = None
    if isinstance(tail.index, pd.DatetimeIndex):
        ts = list(tail.index)
    else:
        for col in ("timestamp", "open_time", "time", "date", "datetime"):
            if col in tail.columns:
                parsed = pd.to_datetime(tail[col], unit="ms", errors="coerce")
                if parsed.isna().all():
                    parsed = pd.to_datetime(tail[col], errors="coerce")
                if not parsed.isna().all():
                    ts = list(parsed)
                break
    if ts is None or len(ts) != m:
        return None

    fmt = "%d %b" if tf in ("4h", "1d") else "%H:%M"
    step = max(m // 6, 1)
    positions = list(range(0, m, step))
    if positions[-1] != m - 1:
        positions.append(m - 1)
    labels = [pd.Timestamp(ts[p]).strftime(fmt) for p in positions]

    dedup_pos, dedup_lab = [], []
    for i, (pos, lab) in enumerate(zip(positions, labels)):
        if dedup_lab and lab == dedup_lab[-1] and i != len(positions) - 1:
            continue
        dedup_pos.append(pos)
        dedup_lab.append(lab)
    return dedup_pos, dedup_lab


def _atr_last(df: pd.DataFrame, period: int = 14) -> float:
    prev = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period).mean()
    val = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else 0.0
    return val


def _draw_analyze_indicators(
    ax,
    df: pd.DataFrame,
    n_show: int,
    tf: str,
    tf_info: dict,
    label_fs: float = 6.5,
) -> None:
    close = df["close"]
    p = float(close.iloc[-1])
    ema20 = close.ewm(span=20, adjust=False).mean().tail(n_show).to_numpy()
    ema50 = close.ewm(span=50, adjust=False).mean().tail(n_show).to_numpy()
    x = range(n_show)

    ax.plot(
        x, ema20, color=EMA20_COLOR, linewidth=0.9, alpha=0.92,
        zorder=4, solid_capstyle="round",
    )
    ax.plot(
        x, ema50, color=EMA50_COLOR, linewidth=0.9, alpha=0.92,
        zorder=4, solid_capstyle="round",
    )

    if len(df) >= 200:
        ema200_s = close.ewm(span=200, adjust=False).mean()
        ema200_val = float(ema200_s.iloc[-1])
        atr = _atr_last(df)
        near = atr > 0 and abs(p - ema200_val) <= 2.0 * atr
        if tf in ("1h", "4h", "1d") or near:
            ax.plot(
                x, ema200_s.tail(n_show).to_numpy(),
                color=EMA200_COLOR, linewidth=0.9, alpha=0.88,
                zorder=4, solid_capstyle="round",
            )

    structure = tf_info.get("structure") or {}
    support = structure.get("support")
    resistance = structure.get("resistance")
    if support:
        s_val = float(support)
        ax.axhline(
            s_val, color=chart.UP, linestyle="--",
            linewidth=0.9, alpha=0.50, zorder=3,
        )
        ax.text(
            0.012, s_val,
            f"S {chart.format_price(s_val, chart.decimals_from_price(s_val))}",
            transform=ax.get_yaxis_transform(), color=chart.UP,
            fontsize=label_fs, fontweight="bold", ha="left", va="bottom",
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=chart.BG, edgecolor="none", alpha=0.7),
        )
    if resistance:
        r_val = float(resistance)
        ax.axhline(
            r_val, color=chart.DOWN, linestyle="--",
            linewidth=0.9, alpha=0.50, zorder=3,
        )
        ax.text(
            0.012, r_val,
            f"R {chart.format_price(r_val, chart.decimals_from_price(r_val))}",
            transform=ax.get_yaxis_transform(), color=chart.DOWN,
            fontsize=label_fs, fontweight="bold", ha="left", va="top",
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=chart.BG, edgecolor="none", alpha=0.7),
        )

    direction = tf_info.get("direction", "NONE")
    last_c = float(df["close"].iloc[-1])
    if direction == "LONG":
        ax.scatter(
            n_show - 1, last_c, marker="o", s=18,
            color=chart.UP, edgecolors=chart.TEXT, linewidths=0.4,
            zorder=6, alpha=0.95,
        )
    elif direction == "SHORT":
        ax.scatter(
            n_show - 1, last_c, marker="o", s=18,
            color=chart.DOWN, edgecolors=chart.TEXT, linewidths=0.4,
            zorder=6, alpha=0.95,
        )


def _nearest_level_text(price: float, support, resistance) -> str:
    candidates = []
    if support is not None:
        candidates.append(("S", float(support)))
    if resistance is not None:
        candidates.append(("R", float(resistance)))
    if not candidates or price == 0:
        return ""
    label, level = min(candidates, key=lambda t: abs(price - t[1]))
    pct = (price - level) / price * 100
    sign = "+" if pct >= 0 else ""
    return f"  ·  {sign}{pct:.2f}% → {label}"


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
    fig_h = fig_w if square else fig_w * 0.40
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(chart.BG)

    if square:
        outer = GridSpec(
            n_panels, 1, figure=fig, hspace=0.32,
            left=0.09, right=0.93, top=0.90, bottom=0.055,
        )
    else:
        outer = GridSpec(
            1, n_panels, figure=fig, wspace=0.16,
            left=0.045, right=0.98, top=0.85, bottom=0.11,
        )

    tick_fs = 9.0 if square else 6.5
    title_fs = 13.5 if square else 9.5
    price_fs = 11.0 if square else 7.5

    for idx, tf in enumerate(valid_tfs):
        df = dfs[tf]
        tf_info = per_tf[tf]
        has_error = "error" in tf_info

        n_show = max(30, chart.get_candles_shown(tf, cfg) // 2 + 10)
        plot_df = df.tail(n_show).reset_index(drop=True)

        cell = outer[idx, 0] if square else outer[0, idx]
        inner = cell.subgridspec(2, 1, height_ratios=[4, 1], hspace=0.08)
        ax_p = fig.add_subplot(inner[0, 0])
        ax_v = fig.add_subplot(inner[1, 0], sharex=ax_p)

        for ax in (ax_p, ax_v):
            ax.set_facecolor(chart.PANEL)
            ax.grid(True, linestyle="-", alpha=0.7, color=chart.GRID, linewidth=0.4)
            ax.set_axisbelow(True)
            ax.tick_params(colors=chart.AXIS, labelcolor=chart.AXIS, labelsize=tick_fs)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(chart.SPINE)
                ax.spines[side].set_linewidth(0.6)
        ax_p.tick_params(labelbottom=False)

        direction = "NONE" if has_error else tf_info.get("direction", "NONE")
        if direction == "LONG":
            tint = TINT_LONG
        elif direction == "SHORT":
            tint = TINT_SHORT
        else:
            tint = TINT_NONE
        ax_p.add_patch(
            Rectangle(
                (0, 0), 1, 1, transform=ax_p.transAxes,
                facecolor=tint, edgecolor="none", zorder=0,
            )
        )

        colors = chart._draw_candles(ax_p, plot_df)
        if not has_error:
            _draw_analyze_indicators(ax_p, df, n_show, tf, tf_info, tick_fs)

        last_x = len(plot_df) - 1
        y_low = float(plot_df["low"].min())
        y_high = float(plot_df["high"].max())
        y_span = max(y_high - y_low, abs(y_low) * 0.01 if y_low != 0 else 0.01)
        pad = y_span * 0.12
        ax_p.set_ylim(y_low - pad, y_high + pad)
        ax_p.set_xlim(-0.6, last_x + 0.6)
        ax_v.set_xlim(-0.6, last_x + 0.6)

        vol_lookback = cfg.get("indicators", {}).get("volume_spike", {}).get("lookback", 20)
        chart._draw_volume(ax_v, plot_df, colors, vol_lookback)

        ax_v.yaxis.set_major_formatter(FuncFormatter(_fmt_volume))
        ax_v.yaxis.get_offset_text().set_visible(False)

        time_ticks = _time_axis_labels(df, n_show, tf)
        if time_ticks:
            positions, labels = time_ticks
            ax_v.set_xticks(positions)
            ax_v.set_xticklabels(labels, fontsize=tick_fs, color=chart.AXIS)

        setup_info = tf_info.get("setup")
        setup_info = setup_info if isinstance(setup_info, dict) else {}
        setup_type = "NONE" if has_error else setup_info.get("type", "NONE")
        badge_color = (
            chart.UP if direction == "LONG"
            else chart.DOWN if direction == "SHORT"
            else chart.AXIS
        )
        setup_txt = f"  ·  {setup_type}" if setup_type and setup_type != "NONE" else ""
        ax_p.set_title(
            f"{tf}  ·  {direction}{setup_txt}",
            color=badge_color, fontsize=title_fs, fontweight="bold",
            loc="left", pad=6,
        )

        last_px = float(plot_df["close"].iloc[-1])
        dec = chart.decimals_from_price(last_px)
        last_price = chart.format_price(last_px, dec)
        structure = (tf_info.get("structure") or {}) if not has_error else {}
        dist_txt = ""
        if not has_error:
            dist_txt = _nearest_level_text(
                last_px, structure.get("support"), structure.get("resistance"),
            )
        ax_p.text(
            0.99, 0.03, f"{last_price}{dist_txt}",
            transform=ax_p.transAxes, color=chart.TEXT,
            fontsize=price_fs, fontweight="bold",
            ha="right", va="bottom", zorder=9,
        )

        if has_error:
            ax_p.text(
                0.5, 0.5, "NO DATA", transform=ax_p.transAxes, color=chart.DOWN,
                fontsize=title_fs, fontweight="bold", ha="center", va="center",
            )

    header_fs = 20.0 if square else 17.0
    badge_fs = 17.0 if square else 14.0
    footer_fs = 10.0 if square else 7.0
    disclaimer_fs = 9.0 if square else 6.5

    fig.text(
        0.045, 0.95, f"{symbol}  ·  MULTI-TIMEFRAME",
        fontsize=header_fs, fontweight="bold", color=chart.TEXT,
        ha="left", va="top",
    )
    ref_df = dfs[valid_tfs[0]]
    chart._draw_change_badge(
        fig, 0.975, 0.95, chart._calc_24h_change(ref_df), fontsize=badge_fs,
    )
    fig.text(
        0.045, 0.02, f"BINANCE FUTURES  ·  {symbol}",
        fontsize=footer_fs, color=chart.AXIS, ha="left", va="bottom",
    )
    fig.text(
        0.98, 0.02,
        "Chart-based analysis. NOT FINANCIAL ADVICE, DYOR.",
        fontsize=disclaimer_fs, fontweight="bold", color=chart.TEXT,
        ha="right", va="bottom",
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor(), dpi=dpi * 2)
    plt.close(fig)
    return out_path
