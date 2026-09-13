from __future__ import annotations

import os

import numpy as np
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

SR_ZONE_ALPHA = 0.07
RSI_LINE_COLOR = "#CE93D8"
RSI_BAND_COLOR = "#616161"


def _fmt_volume(value: float, _pos=None) -> str:
    v = abs(value)
    if v >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if v >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def _rsi_series(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI dihitung lokal di mtfk.py (bukan import dari analyze.py) khusus untuk
    kebutuhan visual garis time-series di single_mtfk -- menghindari circular
    import yang sama seperti kasus build_mtfk_chart sebelumnya."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    out = 100 - (100 / (1 + rs))
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    return out


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


def _place_level_labels_muted(ax, levels: list, label_x: float, min_gap: float) -> None:
    """Versi mtfk.py dari chart._place_level_labels: logika anti-tabrakan
    (declutter posisi) sama persis, tapi warna box dibuat muted (alpha lebih
    rendah, teks pakai warna aslinya bukan putih solid) dan fontsize sedikit
    lebih besar -- permintaan khusus untuk chart drill-down single_mtfk."""
    if not levels:
        return

    ordered = sorted(levels, key=lambda item: item["level"])
    positions = [item["level"] for item in ordered]

    for i in range(1, len(positions)):
        if positions[i] - positions[i - 1] < min_gap:
            positions[i] = positions[i - 1] + min_gap
    for i in range(len(positions) - 2, -1, -1):
        if positions[i + 1] - positions[i] < min_gap:
            positions[i] = positions[i + 1] - min_gap

    level_fontsize = 10.5
    for item, label_y in zip(ordered, positions):
        ax.text(
            label_x, label_y, item["text"],
            color=item["color"],
            va="center", ha="left", fontweight="bold", fontsize=level_fontsize,
            zorder=chart.Z_LEVEL_LABEL, clip_on=False,
            bbox=dict(
                boxstyle="square,pad=0.35",
                facecolor=chart.PANEL,
                edgecolor=item["color"],
                linewidth=1.0,
                alpha=0.72,
            ),
        )


def _draw_sr_zones(ax, support, resistance, atr: float) -> None:
    """Gambar support/resistance sebagai zone box (band tipis di sekitar level)
    -- fitur tambahan di luar chart.py, lebar zone proporsional ke ATR.
    S/R sendiri tidak diberi label teks (cuma garis+zone) -- label teks di
    chart ini dikhususkan untuk level trading ENTRY/SL/TP1/TP2 lewat
    _place_level_labels_muted."""
    zone_half = max(atr * 0.15, 1e-9)
    if support is not None:
        s_val = float(support)
        ax.axhspan(
            s_val - zone_half, s_val + zone_half,
            color=chart.UP, alpha=SR_ZONE_ALPHA, zorder=2, linewidth=0,
        )
        ax.axhline(s_val, color=chart.UP, linestyle="--", linewidth=0.9, alpha=0.45, zorder=3)
    if resistance is not None:
        r_val = float(resistance)
        ax.axhspan(
            r_val - zone_half, r_val + zone_half,
            color=chart.DOWN, alpha=SR_ZONE_ALPHA, zorder=2, linewidth=0,
        )
        ax.axhline(r_val, color=chart.DOWN, linestyle="--", linewidth=0.9, alpha=0.45, zorder=3)


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


def single_mtfk(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    tf_info: dict,
    out_path: str,
    cfg: dict | None = None,
) -> str:
    """Chart drill-down/zoom-in untuk 1 timeframe spesifik (biasanya best_tf
    dari analyze_symbol) -- reuse hasil analisa yang sudah ada di tf_info,
    tidak menghitung ulang struktur/setup dari analyze.py. Layout & elemen
    visual mencontek build_chart di chart.py (label level anti-tabrakan,
    swing HH/LH/HL/LL, x-axis ticks dari open_time, volume MA) supaya
    konsisten dengan chart default proyek -- bedanya: rasio panel tetap
    3 baris (price:volume:rsi = 6:1:1, volume & RSI dibuat kecil), dan
    tambahan S/R sebagai zone box + panel RSI kecil yang tidak ada di
    chart.py."""
    cfg = cfg or {}
    has_error = "error" in tf_info

    chart_cfg = cfg.get("chart", {})
    width_px = chart_cfg.get("width_px", 2800)
    dpi = 200
    fig_w = width_px / dpi
    fig_h = fig_w * 0.62
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(chart.BG)

    outer = GridSpec(
        1, 1, figure=fig,
        left=0.06, right=0.955, top=0.87, bottom=0.10,
    )
    # price : volume : rsi -- volume & rsi dibuat kecil, price mendominasi
    inner = outer[0, 0].subgridspec(3, 1, height_ratios=[6, 1, 1], hspace=0.10)
    ax_p = fig.add_subplot(inner[0, 0])
    ax_v = fig.add_subplot(inner[1, 0], sharex=ax_p)
    ax_r = fig.add_subplot(inner[2, 0], sharex=ax_p)

    tick_fs = 7.5
    title_fs = 19.0
    price_fs = 10.5

    for ax in (ax_p, ax_v, ax_r):
        ax.set_facecolor(chart.PANEL)
        ax.grid(True, linestyle="-", alpha=0.8, color=chart.GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(colors=chart.AXIS, labelcolor=chart.AXIS, labelsize=tick_fs)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(chart.SPINE)
            ax.spines[side].set_linewidth(0.8)
    ax_p.tick_params(labelbottom=False)
    ax_v.tick_params(labelbottom=False)

    # --- window candle: ikut pola chart.py (STRUCTURE_CONTEXT ekstra utk swing/BOS,
    # lalu _calculate_visible_range menentukan window final) ---
    n_show_max = chart.get_candles_shown(timeframe, cfg)
    work_len = n_show_max + chart.STRUCTURE_CONTEXT + 50
    work_df = df.tail(work_len).reset_index(drop=True)
    structure = chart._compute_structure(work_df)

    left_pad = 10 if timeframe == "15m" else 7
    start_idx, end_idx = chart._calculate_visible_range(
        work_df, structure, timeframe, max_candles=n_show_max, left_padding=left_pad,
    )
    plot_df = work_df.iloc[start_idx:end_idx + 1].reset_index(drop=True)
    offset = start_idx
    n_show = len(plot_df)

    direction = "NONE" if has_error else tf_info.get("direction", "NONE")

    colors = chart._draw_candles(ax_p, plot_df)

    setup_info = tf_info.get("setup") if not has_error else {}
    setup_info = setup_info if isinstance(setup_info, dict) else {}
    tf_levels = (tf_info.get("levels") or {}) if not has_error else {}
    tf_structure = (tf_info.get("structure") or {}) if not has_error else {}

    levels = []
    if not has_error and tf_levels.get("direction") in ("LONG", "SHORT"):
        entry = tf_levels.get("entry")
        sl = tf_levels.get("sl")
        tp1 = tf_levels.get("tp1")
        tp2 = tf_levels.get("tp2")
        ref_price = entry[0] if entry is not None else (sl if sl is not None else float(plot_df["close"].iloc[-1]))
        dec = chart.decimals_from_price(ref_price)

        if entry is not None:
            entry_mid = (entry[0] + entry[1]) / 2
            levels.append({"level": entry_mid, "color": chart.ENTRY,
                            "text": f"ENTRY  {chart.format_price(entry_mid, dec)}"})
        if tp1 is not None:
            levels.append({"level": tp1, "color": chart.TP1,
                            "text": f"TP1  {chart.format_price(tp1, dec)}"})
        if tp2 is not None:
            levels.append({"level": tp2, "color": chart.TP2,
                            "text": f"TP2  {chart.format_price(tp2, dec)}"})
        if sl is not None:
            levels.append({"level": sl, "color": chart.SL,
                            "text": f"SL  {chart.format_price(sl, dec)}"})

    if not has_error:
        close = plot_df["close"]
        ema20 = close.ewm(span=20, adjust=False).mean().to_numpy()
        ema50 = close.ewm(span=50, adjust=False).mean().to_numpy()
        x = range(n_show)
        ax_p.plot(x, ema20, color=EMA20_COLOR, linewidth=1.1, alpha=0.92,
                  zorder=chart.Z_EMA, solid_capstyle="round", label="EMA 20")
        ax_p.plot(x, ema50, color=EMA50_COLOR, linewidth=1.1, alpha=0.92,
                  zorder=chart.Z_EMA, solid_capstyle="round", label="EMA 50")

        atr_val = float((tf_info.get("direction_analysis") or {}).get("atr") or 0.0)
        if atr_val <= 0:
            atr_val = float(plot_df["close"].iloc[-1]) * 0.01
        _draw_sr_zones(ax_p, tf_structure.get("support"), tf_structure.get("resistance"), atr_val)

    for item in levels:
        ax_p.axhline(y=item["level"], color=item["color"], linestyle="--",
                     linewidth=1.1, alpha=0.75, zorder=chart.Z_LEVEL_LINE)

    last_x = n_show - 1

    zone_values = []
    support = tf_structure.get("support")
    resistance = tf_structure.get("resistance")
    if support is not None:
        zone_values.append(float(support))
    if resistance is not None:
        zone_values.append(float(resistance))
    level_values = [item["level"] for item in levels]

    y_low = min([float(plot_df["low"].min())] + level_values + zone_values)
    y_high = max([float(plot_df["high"].max())] + level_values + zone_values)
    y_span = max(y_high - y_low, abs(y_low) * 0.01 if y_low != 0 else 0.01)
    y_padding = y_span * 0.16
    ax_p.set_ylim(y_low - y_padding, y_high + y_padding)

    gap_from_candle = 4.0
    label_width_est = 13.0
    gap_from_edge = 0.4
    extra_margin = gap_from_candle + label_width_est + gap_from_edge
    label_x = last_x + gap_from_candle

    ax_p.set_xlim(-0.6, last_x + extra_margin)
    ax_v.set_xlim(-0.6, last_x + extra_margin)
    ax_r.set_xlim(-0.6, last_x + extra_margin)

    if not has_error:
        chart._draw_structure_labels(ax_p, structure["labeled_points"], offset, n_show, y_span)

    label_min_gap = (ax_p.get_ylim()[1] - ax_p.get_ylim()[0]) * 0.07
    _place_level_labels_muted(ax_p, levels, label_x, label_min_gap)

    vol_lookback = cfg.get("indicators", {}).get("volume_spike", {}).get("lookback", 20)
    chart._draw_volume(ax_v, plot_df, colors, vol_lookback)
    ax_v.yaxis.set_major_formatter(FuncFormatter(chart._format_volume_axis))
    ax_v.tick_params(labelsize=tick_fs * 0.9)

    rsi_series = _rsi_series(df["close"]).iloc[offset:offset + n_show].to_numpy()
    x = range(n_show)
    ax_r.plot(x, rsi_series, color=RSI_LINE_COLOR, linewidth=1.0, alpha=0.95, zorder=4)
    ax_r.axhline(70, color=RSI_BAND_COLOR, linestyle="--", linewidth=0.6, alpha=0.5, zorder=2)
    ax_r.axhline(30, color=RSI_BAND_COLOR, linestyle="--", linewidth=0.6, alpha=0.5, zorder=2)
    ax_r.set_ylim(0, 100)
    ax_r.set_yticks([30, 70])
    ax_r.tick_params(labelsize=tick_fs * 0.9)
    ax_r.text(
        0.004, 0.88, "RSI", transform=ax_r.transAxes, color=chart.AXIS,
        fontsize=tick_fs * 0.9, fontweight="bold", ha="left", va="top",
    )

    if "open_time" in plot_df.columns and len(plot_df):
        tick_count = min(6, len(plot_df))
        ticks_idx = np.linspace(0, last_x, tick_count, dtype=int) if tick_count > 1 else [0]
        ax_r.set_xticks(ticks_idx)
        tick_labels = []
        for t in ticks_idx:
            ts = plot_df["open_time"].iloc[int(t)]
            tick_labels.append(ts.strftime("%d %b  %H:%M") if pd.notna(ts) else str(t))
        ax_r.set_xticklabels(tick_labels, fontsize=tick_fs, color=chart.AXIS)

    if not has_error:
        legend = ax_p.legend(
            loc="upper left", fontsize=7.0, framealpha=0.95,
            facecolor=chart.BG, edgecolor=chart.SPINE, labelcolor=chart.TEXT, borderpad=0.4,
        )
        legend.get_frame().set_linewidth(0.7)

    setup_type = "NONE" if has_error else setup_info.get("type", "NONE")
    quality = setup_info.get("quality", "")
    badge_color = chart.UP if direction == "LONG" else chart.DOWN if direction == "SHORT" else chart.AXIS
    setup_txt = f"  ·  {setup_type}" + (f" ({quality})" if quality else "") if setup_type != "NONE" else ""
    header_extra = pd.Timestamp.now(tz="UTC").strftime("Updated %d %b %H:%M UTC")
    fig.text(
        0.06, 0.965, f"{symbol}  ·  {timeframe}  ·  {direction}{setup_txt}",
        color=badge_color, fontsize=title_fs, fontweight="bold", ha="left", va="top",
    )
    chart._draw_change_badge(fig, 0.95, 0.965, chart._calc_24h_change(df), fontsize=17.0)

    last_px = float(plot_df["close"].iloc[-1])
    dec = chart.decimals_from_price(last_px)
    last_price = chart.format_price(last_px, dec)
    dist_txt = "" if has_error else _nearest_level_text(last_px, support, resistance)
    ax_p.text(
        0.985, 0.03, f"{last_price}{dist_txt}",
        transform=ax_p.transAxes, color=chart.TEXT,
        fontsize=price_fs, fontweight="bold", ha="right", va="bottom", zorder=9,
    )

    if has_error:
        ax_p.text(
            0.5, 0.5, "NO DATA", transform=ax_p.transAxes, color=chart.DOWN,
            fontsize=title_fs, fontweight="bold", ha="center", va="center",
        )

    fig.text(
        0.06, 0.02, f"BINANCE FUTURES  ·  {symbol}  ·  {timeframe}  ·  SINGLE TF DRILL-DOWN  ·  {header_extra}",
        fontsize=7.0, color=chart.AXIS, ha="left", va="bottom",
    )
    fig.text(
        0.955, 0.013,
        "Chart-based analysis.\nNOT FINANCIAL ADVICE, DYOR.",
        fontsize=7.5, fontweight="bold", color=chart.TEXT, ha="right", va="bottom",
        linespacing=1.6,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor(), dpi=dpi * 2)
    plt.close(fig)
    return out_path
