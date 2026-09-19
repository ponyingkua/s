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
from matplotlib.transforms import offset_copy

BG = "#121417"
PANEL = "#121417"
GRID = "#3A3A3A"
TEXT = "#F2F2F2"
AXIS = "#B8B8B8"
SPINE = "#4A4A4A"
UP = "#26A69A"
DOWN = "#EF5350"

Z_CANDLE_WICK = 2.0
Z_CANDLE_BODY = 2.1
Z_EMA = 4.0
Z_STRUCT_LABEL = 5.5
Z_LEVEL_LINE = 6.0
Z_LEVEL_LABEL = 6.5

STRUCT_TEXT = "#BDBDBD"

MAX_CANDLES_BY_TF = {
    "15m": 48,   # ~12 jam
    "1h": 48,    # ~2 hari
    "4h": 42,    # ~7 hari
}


def get_candles_shown(timeframe: str, cfg: dict) -> int:
    chart_cfg = cfg.get("chart", {})
    return MAX_CANDLES_BY_TF.get(timeframe, chart_cfg.get("candles_shown", 120))


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


def _calc_24h_change(df: pd.DataFrame) -> float | None:
    if "open_time" not in df.columns or len(df) < 2:
        return None
    last_time = df["open_time"].iloc[-1]
    last_close = float(df["close"].iloc[-1])
    target_time = last_time - pd.Timedelta(hours=24)
    if df["open_time"].iloc[0] > target_time:
        return None
    ref_rows = df[df["open_time"] <= target_time]
    if ref_rows.empty:
        return None
    ref_close = float(ref_rows["close"].iloc[-1])
    if ref_close == 0:
        return None
    return (last_close / ref_close - 1.0) * 100.0


def _draw_change_badge(fig, x: float, y: float, change_pct: float | None,
                        fontsize: float = 15) -> None:
    if change_pct is None:
        return
    color = UP if change_pct >= 0 else DOWN
    text = f"{change_pct:+.2f}%  24H"
    fig.text(
        x, y, text, fontsize=fontsize, fontweight="bold", color=TEXT,
        ha="right", va="top", zorder=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=color, edgecolor=color,
                   linewidth=0, alpha=0.88),
    )


def _find_swings(df: pd.DataFrame, left: int = 2, right: int = 2):
    high = df["high"].values
    low = df["low"].values
    n = len(df)
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)

    for i in range(left, n - right):
        window_h = high[i - left:i + right + 1]
        if high[i] == window_h.max() and np.argmax(window_h) == left:
            swing_high[i] = True
        window_l = low[i - left:i + right + 1]
        if low[i] == window_l.min() and np.argmin(window_l) == left:
            swing_low[i] = True

    return swing_high, swing_low


def _label_structure(df: pd.DataFrame, swing_high, swing_low) -> list:
    points = []
    for i in range(len(df)):
        if swing_high[i]:
            points.append((i, float(df["high"].iloc[i]), "H"))
        if swing_low[i]:
            points.append((i, float(df["low"].iloc[i]), "L"))
    points.sort(key=lambda p: p[0])

    labeled = []
    last_high = None
    last_low = None
    for idx, price, typ in points:
        if typ == "H":
            if last_high is not None:
                label = "HH" if price > last_high else "LH"
                labeled.append({"index": idx, "price": price, "type": "H", "label": label})
            last_high = price
        else:
            if last_low is not None:
                label = "HL" if price > last_low else "LL"
                labeled.append({"index": idx, "price": price, "type": "L", "label": label})
            last_low = price

    return labeled


def _draw_structure_labels(ax, labeled_points: list, offset: int, plot_len: int, y_span: float,
                             square: bool = False) -> None:
    pad = y_span * 0.022
    fontsize = 12.5 if square else 7.5
    for pt in labeled_points:
        px = pt["index"] - offset
        if px < 0 or px >= plot_len:
            continue
        if pt["type"] == "H":
            ax.text(px, pt["price"] + pad, pt["label"], color=STRUCT_TEXT,
                    fontsize=fontsize, fontweight="bold", ha="center", va="bottom",
                    zorder=Z_STRUCT_LABEL, clip_on=False)
        else:
            ax.text(px, pt["price"] - pad, pt["label"], color=STRUCT_TEXT,
                    fontsize=fontsize, fontweight="bold", ha="center", va="top",
                    zorder=Z_STRUCT_LABEL, clip_on=False)


EMA20_COLOR = "#FFD54F"
EMA50_COLOR = "#29B6F6"
EMA200_COLOR = "#AB47BC"

TINT_LONG = (0.15, 0.65, 0.45, 0.05)
TINT_SHORT = (0.85, 0.25, 0.25, 0.05)
TINT_NONE = (0.5, 0.5, 0.5, 0.03)

RSI_COLOR = "#4DD0E1"
BB_COLOR = "#90A4AE"

SINGLE_CANDLE_WIDTH = 0.58


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
        if step > 1 and (m - 1 - positions[-1]) < step / 2:
            positions[-1] = m - 1
        else:
            positions.append(m - 1)
    labels = [pd.Timestamp(ts[p]).strftime(fmt) for p in positions]

    dedup_pos, dedup_lab = [], []
    for pos, lab in zip(positions, labels):
        if dedup_lab and lab == dedup_lab[-1]:
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


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return rsi.fillna(50.0)


def _compute_bollinger(df: pd.DataFrame, n_show: int, period: int = 20, std_mult: float = 2.0) -> dict:
    close = df["close"]
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return {
        "upper": upper.tail(n_show).to_numpy(),
        "lower": lower.tail(n_show).to_numpy(),
    }


def _macd_histogram_last(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    close = df["close"]
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    hist = line - line.ewm(span=signal, adjust=False).mean()
    h = float(hist.iloc[-1]) if pd.notna(hist.iloc[-1]) else None
    hp = float(hist.iloc[-2]) if len(hist) > 1 and pd.notna(hist.iloc[-2]) else None
    return h, hp


def _volume_ratio_last(df: pd.DataFrame, lookback: int = 20):
    if len(df) < lookback:
        return None
    vma = float(df["volume"].tail(lookback).mean())
    if vma <= 0:
        return None
    return float(df["volume"].iloc[-1]) / vma


def _compute_ema_set(df: pd.DataFrame, n_show: int, tf: str) -> dict:
    close = df["close"]
    p = float(close.iloc[-1])
    ema20 = close.ewm(span=20, adjust=False).mean().tail(n_show).to_numpy()
    ema50 = close.ewm(span=50, adjust=False).mean().tail(n_show).to_numpy()

    ema200 = None
    if len(df) >= 200:
        ema200_s = close.ewm(span=200, adjust=False).mean()
        ema200_val = float(ema200_s.iloc[-1])
        atr = _atr_last(df)
        near = atr > 0 and abs(p - ema200_val) <= 2.0 * atr
        if tf in ("1h", "4h", "1d") or near:
            ema200 = ema200_s.tail(n_show).to_numpy()

    return {"ema20": ema20, "ema50": ema50, "ema200": ema200}


def _draw_candles_single(ax, df: pd.DataFrame) -> list:
    colors = []
    for i in range(len(df)):
        row = df.iloc[i]
        open_p, close_p = float(row["open"]), float(row["close"])
        high_p, low_p = float(row["high"]), float(row["low"])
        color = UP if close_p >= open_p else DOWN
        colors.append(color)

        ax.plot(
            [i, i], [low_p, high_p], color=color, linewidth=1.3,
            solid_capstyle="round", zorder=Z_CANDLE_WICK,
        )
        body_bottom = min(open_p, close_p)
        body_height = max(abs(close_p - open_p), (high_p - low_p) * 0.02)
        ax.add_patch(Rectangle(
            (i - SINGLE_CANDLE_WIDTH / 2, body_bottom), SINGLE_CANDLE_WIDTH, body_height,
            facecolor=color, edgecolor=color, alpha=0.92, linewidth=0, zorder=Z_CANDLE_BODY,
        ))
    return colors


def _draw_volume_bars_no_ma(ax, df: pd.DataFrame, colors: list) -> None:
    for i in range(len(df)):
        ax.bar(
            i, float(df["volume"].iloc[i]), color=colors[i], alpha=0.48,
            width=SINGLE_CANDLE_WIDTH, linewidth=0, zorder=2,
        )


_LEGEND_ORDER = ("EMA 20", "EMA 50", "EMA 200")


def _ordered_legend_handles(ax):
    handles, labels = ax.get_legend_handles_labels()
    pairs = sorted(
        zip(handles, labels),
        key=lambda hl: _LEGEND_ORDER.index(hl[1]) if hl[1] in _LEGEND_ORDER else 99,
    )
    if not pairs:
        return [], []
    return zip(*pairs)


def _pick_legend_loc(level_fracs: list, left_candle_fracs: list | None = None) -> str:
    """Default legend spot is upper-left. If a support/resistance level sits
    in the top ~30% of the visible candle range, its dashed line + label ends
    up right where the legend box would be, so move the legend to lower-left
    instead. But only if the candles on the left edge (where the legend
    actually overlaps the plot) aren't themselves sitting low -- otherwise
    lower-left just trades one collision (legend vs. S/R line) for another
    (legend vs. candles), which a pure S/R check can't see coming."""
    top_conflict = any(f >= 0.70 for f in level_fracs)
    if not top_conflict:
        return "upper left"
    left_is_low = bool(left_candle_fracs) and (sum(left_candle_fracs) / len(left_candle_fracs)) <= 0.35
    return "upper left" if left_is_low else "lower left"


def _draw_colored_segments(
    fig,
    x0: float,
    y: float,
    parts: list[tuple[str, str]],
    fontsize: float,
    sep: str = "   ·   ",
) -> None:
    inv = fig.transFigure.inverted()
    x = x0
    for i, (text, color) in enumerate(parts):
        if i > 0:
            t = fig.text(x, y, sep, fontsize=fontsize, fontweight="bold", color=AXIS, ha="left", va="top")
            fig.canvas.draw()
            x = inv.transform((t.get_window_extent(renderer=fig.canvas.get_renderer()).x1, 0))[0]
        t = fig.text(x, y, text, fontsize=fontsize, fontweight="bold", color=color, ha="left", va="top")
        fig.canvas.draw()
        x = inv.transform((t.get_window_extent(renderer=fig.canvas.get_renderer()).x1, 0))[0]


def _draw_rsi_panel(ax, rsi_values, tick_fs: float = 7.5) -> None:
    x = list(range(len(rsi_values)))
    ax.axhspan(70, 100, color=DOWN, alpha=0.06, zorder=1)
    ax.axhspan(0, 30, color=UP, alpha=0.06, zorder=1)
    ax.axhline(70, color=AXIS, linestyle="--", linewidth=0.6, alpha=0.55, zorder=2)
    ax.axhline(30, color=AXIS, linestyle="--", linewidth=0.6, alpha=0.55, zorder=2)
    ax.axhline(50, color=AXIS, linestyle="-", linewidth=0.4, alpha=0.30, zorder=2)
    ax.fill_between(x, rsi_values, 50, color=RSI_COLOR, alpha=0.10, zorder=3, linewidth=0)
    ax.plot(
        x, rsi_values, color=RSI_COLOR, linewidth=1.3, alpha=0.95,
        zorder=4, solid_capstyle="round",
    )
    ax.set_ylim(0, 100)
    ax.set_yticks([30, 50, 70])
    ax.tick_params(labelsize=tick_fs)


def _resolve_sr_label_anchors(
    s_val: float | None,
    r_val: float | None,
    price_span_hint: float,
    square: bool = False,
) -> tuple[float | None, float | None]:
    if s_val is None or r_val is None:
        return s_val, r_val
    span = max(price_span_hint, 1e-9)
    min_gap = span * (0.12 if square else 0.14)
    lo, hi = min(s_val, r_val), max(s_val, r_val)
    if hi - lo >= min_gap:
        return s_val, r_val
    mid = (lo + hi) / 2.0
    lo_new, hi_new = mid - min_gap / 2.0, mid + min_gap / 2.0
    return (lo_new, hi_new) if s_val <= r_val else (hi_new, lo_new)


def _draw_indicators_single(
    ax,
    df: pd.DataFrame,
    n_show: int,
    tf: str,
    tf_info: dict,
    right_pad: float,
    square: bool = False,
) -> list:
    n_show = min(n_show, len(df))
    last_x = n_show - 1
    edge_margin = max(right_pad * 0.08, 0.35)
    label_x = last_x + right_pad - edge_margin
    marker_x = last_x + SINGLE_CANDLE_WIDTH / 2 + max(right_pad * 0.20, 0.5)

    emas = _compute_ema_set(df, n_show, tf)
    x = range(n_show)
    ema_values = list(emas["ema20"]) + list(emas["ema50"])

    if emas["ema200"] is not None:
        ax.plot(
            x, emas["ema200"], color=EMA200_COLOR, linewidth=1.4, alpha=0.92,
            zorder=Z_EMA, solid_capstyle="round", label="EMA 200",
        )
        ema_values.extend(list(emas["ema200"]))
    ax.plot(
        x, emas["ema50"], color=EMA50_COLOR, linewidth=1.4, alpha=0.95,
        zorder=Z_EMA + 1, solid_capstyle="round", label="EMA 50",
    )
    ax.plot(
        x, emas["ema20"], color=EMA20_COLOR, linewidth=1.4, alpha=0.95,
        zorder=Z_EMA + 2, solid_capstyle="round", label="EMA 20",
    )

    bb = _compute_bollinger(df, n_show)
    bb_upper, bb_lower = bb["upper"], bb["lower"]
    if not (np.all(pd.isna(bb_upper)) or np.all(pd.isna(bb_lower))):
        ax.plot(
            x, bb_upper, color=BB_COLOR, linewidth=1.0, alpha=0.55,
            zorder=Z_EMA - 1, solid_capstyle="round", label="BB(20,2)",
        )
        ax.plot(
            x, bb_lower, color=BB_COLOR, linewidth=1.0, alpha=0.55,
            zorder=Z_EMA - 1, solid_capstyle="round",
        )
        ax.fill_between(
            x, bb_lower, bb_upper, color=BB_COLOR, alpha=0.05,
            zorder=Z_EMA - 2, linewidth=0,
        )
        ema_values.extend([float(v) for v in bb_upper if pd.notna(v)])
        ema_values.extend([float(v) for v in bb_lower if pd.notna(v)])

    structure = tf_info.get("structure") or {}
    support = structure.get("support")
    resistance = structure.get("resistance")
    label_fs = 10.5 if square else 9.5
    marker_size = 7.5 if square else 7.0

    s_val = float(support) if support else None
    r_val = float(resistance) if resistance else None
    price_span_hint = float(df["high"].tail(n_show).max() - df["low"].tail(n_show).min())
    s_anchor, r_anchor = _resolve_sr_label_anchors(s_val, r_val, price_span_hint, square=square)

    level_values = []
    if s_val is not None:
        level_values += [s_val, s_anchor]
        ax.axhline(
            s_val, color=UP, linestyle="--",
            linewidth=1.4, alpha=0.85, zorder=Z_LEVEL_LINE,
        )
        support_trans = offset_copy(
            ax.transData, fig=ax.figure, x=0, y=marker_size / 2 + 0.6, units="points",
        )
        ax.plot(
            [marker_x], [s_anchor], marker="^", markersize=marker_size,
            color=UP, zorder=Z_LEVEL_LABEL, clip_on=False,
            transform=support_trans,
        )
        ax.text(
            label_x, s_anchor,
            f"SUPPORT  {format_price(s_val, decimals_from_price(s_val))}",
            color=UP,
            fontsize=label_fs, fontweight="bold", ha="right", va="center",
            zorder=Z_LEVEL_LABEL,
            bbox=dict(boxstyle="round,pad=0.22", facecolor=BG, edgecolor="none", alpha=0.80),
        )
    if r_val is not None:
        level_values += [r_val, r_anchor]
        ax.axhline(
            r_val, color=DOWN, linestyle="--",
            linewidth=1.4, alpha=0.85, zorder=Z_LEVEL_LINE,
        )
        resistance_trans = offset_copy(
            ax.transData, fig=ax.figure, x=0, y=-(marker_size / 2 + 0.6), units="points",
        )
        ax.plot(
            [marker_x], [r_anchor], marker="v", markersize=marker_size,
            color=DOWN, zorder=Z_LEVEL_LABEL, clip_on=False,
            transform=resistance_trans,
        )
        ax.text(
            label_x, r_anchor,
            f"RESISTANCE  {format_price(r_val, decimals_from_price(r_val))}",
            color=DOWN,
            fontsize=label_fs, fontweight="bold", ha="right", va="center",
            zorder=Z_LEVEL_LABEL,
            bbox=dict(boxstyle="round,pad=0.22", facecolor=BG, edgecolor="none", alpha=0.80),
        )

    direction = tf_info.get("direction", "NONE")
    last_c = float(df["close"].iloc[-1])
    if direction == "LONG":
        ax.scatter(
            n_show - 1, last_c, marker="o", s=(34 if square else 26),
            color=UP, edgecolors=TEXT, linewidths=0.6,
            zorder=6, alpha=0.95,
        )
    elif direction == "SHORT":
        ax.scatter(
            n_show - 1, last_c, marker="o", s=(34 if square else 26),
            color=DOWN, edgecolors=TEXT, linewidths=0.6,
            zorder=6, alpha=0.95,
        )

    # Report where support/resistance land within the visible candle range
    # (as a 0..1 fraction, 1 = top) so the caller can steer the legend away
    # from a level that would otherwise sit right underneath it.
    candle_lo = float(df["low"].tail(n_show).min())
    candle_hi = float(df["high"].tail(n_show).max())
    candle_span = max(candle_hi - candle_lo, 1e-12)
    level_fracs = []
    if s_val is not None:
        level_fracs.append((s_val - candle_lo) / candle_span)
    if r_val is not None:
        level_fracs.append((r_val - candle_lo) / candle_span)

    return level_values + ema_values, level_fracs


def _draw_trigger_highlight(ax, plot_df: pd.DataFrame, tf_info: dict, y_span: float) -> None:
    setup = tf_info.get("setup") or {}
    setup_type = setup.get("type", "NONE")
    if setup_type not in ("BREAKOUT", "BREAKDOWN", "REJECTION"):
        return
    if len(plot_df) == 0:
        return

    direction = tf_info.get("direction", "NONE")
    color = UP if direction == "LONG" else DOWN if direction == "SHORT" else AXIS

    idx = len(plot_df) - 1
    last = plot_df.iloc[-1]
    high, low = float(last["high"]), float(last["low"])

    half_w = SINGLE_CANDLE_WIDTH / 2 + 0.30
    pad_y = max(y_span * 0.012, (high - low) * 0.10)
    ax.add_patch(Rectangle(
        (idx - half_w, low - pad_y), half_w * 2, (high - low) + pad_y * 2,
        facecolor="none", edgecolor=color, linewidth=1.6, linestyle="--",
        alpha=0.9, zorder=Z_LEVEL_LABEL + 1,
    ))


def build_single_mtfk_chart(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    tf_info: dict,
    out_path: str,
    cfg: dict | None = None,
    square: bool = False,
) -> str:
    cfg = cfg or {}
    has_error = "error" in tf_info

    n_show = get_candles_shown(timeframe, cfg)
    plot_df = df.tail(n_show).reset_index(drop=True)

    chart_cfg = cfg.get("chart", {})
    width_px = chart_cfg.get("width_px", 2800)
    height_ratio = 1.0 if square else chart_cfg.get("height_ratio", 0.54)
    dpi = 200
    output_scale = 2
    fig_w = width_px / dpi
    fig_h = (width_px * height_ratio) / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(BG)

    gs = GridSpec(
        3, 1, figure=fig,
        height_ratios=[4.4, 0.9, 1.3],
        hspace=0.10,
        left=0.07, right=0.96, top=0.87, bottom=0.11,
    )
    ax_price = fig.add_subplot(gs[0, 0])
    ax_vol = fig.add_subplot(gs[1, 0], sharex=ax_price)
    ax_rsi = fig.add_subplot(gs[2, 0], sharex=ax_price)

    for ax in (ax_price, ax_vol, ax_rsi):
        ax.set_facecolor(PANEL)
        ax.grid(True, linestyle="-", alpha=0.55, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(colors=AXIS, labelcolor=AXIS, labelsize=7.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(SPINE)
            ax.spines[side].set_linewidth(0.8)

    ax_price.tick_params(labelbottom=False)
    ax_vol.tick_params(labelbottom=False)

    direction = "NONE" if has_error else tf_info.get("direction", "NONE")

    colors = _draw_candles_single(ax_price, plot_df)
    last_x = len(plot_df) - 1

    label_zone_frac = 0.22 if square else 0.19
    right_pad = label_zone_frac * (last_x + 0.6) / (1 - label_zone_frac)
    right_pad = max(right_pad, SINGLE_CANDLE_WIDTH * 8)
    ax_price.set_xlim(-0.6, last_x + right_pad)
    ax_vol.set_xlim(-0.6, last_x + right_pad)
    ax_rsi.set_xlim(-0.6, last_x + right_pad)

    range_values = []
    level_fracs = []
    if not has_error:
        range_values, level_fracs = _draw_indicators_single(
            ax_price, df, n_show, timeframe, tf_info, right_pad, square=square,
        )
    all_vals = [float(plot_df["low"].min()), float(plot_df["high"].max())] + range_values
    y_low = min(all_vals)
    y_high = max(all_vals)
    y_span = max(y_high - y_low, abs(y_low) * 0.01 if y_low != 0 else 0.01)
    y_padding = y_span * 0.18
    ax_price.set_ylim(y_low - y_padding, y_high + y_padding)

    if not has_error and len(plot_df) >= 6:
        swing_high, swing_low = _find_swings(plot_df, 2, 2)
        labeled_points = _label_structure(plot_df, swing_high, swing_low)
        if labeled_points:
            labeled_points = labeled_points[-6:]
            _draw_structure_labels(
                ax_price, labeled_points, offset=0, plot_len=len(plot_df),
                y_span=y_span, square=square,
            )

    if not has_error:
        _draw_trigger_highlight(ax_price, plot_df, tf_info, y_span)

    _draw_volume_bars_no_ma(ax_vol, plot_df, colors)
    ax_vol.yaxis.set_major_formatter(FuncFormatter(_fmt_volume))
    ax_vol.yaxis.get_offset_text().set_visible(False)
    ax_vol.set_ylabel("Vol", color=AXIS, fontsize=8, labelpad=5)
    ax_price.set_ylabel("Price", color=AXIS, fontsize=8.5, labelpad=5)

    rsi_tail = _compute_rsi(df["close"]).tail(n_show).to_numpy()
    _draw_rsi_panel(ax_rsi, rsi_tail)
    ax_rsi.set_ylabel("RSI", color=AXIS, fontsize=8, labelpad=5)

    time_ticks = _time_axis_labels(df, n_show, timeframe)
    if time_ticks:
        positions, labels = time_ticks
        ax_rsi.set_xticks(positions)
        ax_rsi.set_xticklabels(labels, fontsize=7.5, color=AXIS)

    if not has_error:
        n_left = max(int(len(plot_df) * 0.15), 3)
        left_candle_fracs = [
            (float(v) - (y_low - y_padding)) / (y_high + y_padding - (y_low - y_padding))
            for v in list(plot_df["high"].iloc[:n_left]) + list(plot_df["low"].iloc[:n_left])
        ]
        handles, labels = _ordered_legend_handles(ax_price)
        legend = ax_price.legend(
            handles, labels,
            loc=_pick_legend_loc(level_fracs, left_candle_fracs), fontsize=(13.5 if square else 7.5), framealpha=0.95,
            facecolor=BG, edgecolor=SPINE, labelcolor=TEXT, borderpad=0.4,
        )
        legend.get_frame().set_linewidth(0.7)

    setup_info = tf_info.get("setup")
    setup_info = setup_info if isinstance(setup_info, dict) else {}
    setup_type = "NONE" if has_error else setup_info.get("type", "NONE")
    setup_txt = f"  ·  {setup_type}" if setup_type and setup_type != "NONE" else ""
    header_extra = pd.Timestamp.now(tz="UTC").strftime("Updated %d %b %H:%M UTC")
    header_title = f"{symbol}  ·  {timeframe}  ·  {direction}{setup_txt}"

    header_fs = 20.0 if square else 16.0
    badge_fs = 17.5 if square else 13.5
    segment_fs = 12.0 if square else 9.0
    segment_y = 0.913 if square else 0.917

    fig.text(
        0.07, 0.965, header_title,
        fontsize=header_fs, fontweight="bold", color=TEXT,
        ha="left", va="top",
    )
    _draw_change_badge(fig, 0.96, 0.965, _calc_24h_change(df), fontsize=badge_fs)

    if not has_error:
        atr_pct = (tf_info.get("direction_analysis") or {}).get("atr_pct")
        hist, hist_prev = _macd_histogram_last(df)
        vol_ratio = _volume_ratio_last(df)
        mtf_agree = tf_info.get("mtf_agree_tfs") or []

        segments: list[tuple[str, str]] = []
        if atr_pct is not None:
            segments.append((f"ATR {atr_pct:.2f}%", AXIS))
        if hist is not None:
            arrow = "▲" if hist > 0 else "▼" if hist < 0 else "→"
            state = ""
            if hist_prev is not None:
                if abs(hist) > abs(hist_prev):
                    state = "Expanding"
                elif abs(hist) < abs(hist_prev):
                    state = "Contracting"
            macd_color = UP if hist > 0 else DOWN if hist < 0 else AXIS
            segments.append((f"MACD {arrow}" + (f" {state}" if state else ""), macd_color))
        if vol_ratio is not None:
            vol_color = EMA20_COLOR if vol_ratio >= 1.5 else AXIS
            segments.append((f"Vol {vol_ratio:.1f}x avg", vol_color))
        if mtf_agree:
            agree_color = UP if direction == "LONG" else DOWN if direction == "SHORT" else AXIS
            segments.append((f"MTF agree: {', '.join(mtf_agree)}", agree_color))

        if segments:
            _draw_colored_segments(fig, 0.07, segment_y, segments, fontsize=segment_fs)

    if square:
        footer_left = f"BINANCE FUTURES  ·  {symbol}  ·  {timeframe}\n{header_extra}"
        fig.text(
            0.07, 0.028, footer_left, fontsize=14.0, color=AXIS,
            ha="left", va="bottom", linespacing=1.6,
        )
    else:
        fig.text(
            0.07, 0.02, f"BINANCE FUTURES  ·  {symbol}  ·  {timeframe}  ·  {header_extra}",
            fontsize=8.5, color=AXIS, ha="left", va="bottom",
        )

    fig.text(
        0.96, 0.013,
        "Chart-based analysis.\nNOT FINANCIAL ADVICE, DYOR.",
        fontsize=(13.0 if square else 7.5), fontweight="bold", color=TEXT,
        ha="right", va="bottom", linespacing=1.7,
    )

    if has_error:
        ax_price.text(
            0.5, 0.5, "NO DATA", transform=ax_price.transAxes, color=DOWN,
            fontsize=(22.0 if square else 16), fontweight="bold", ha="center", va="center",
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor(), dpi=dpi * output_scale)
    plt.close(fig)
    return out_path


single_mtfk = build_single_mtfk_chart


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
    height_ratio = chart_cfg.get("height_ratio", 0.54)
    dpi = 200
    output_scale = 2
    fig_w = width_px / dpi

    single_fig_h = (width_px * height_ratio) / dpi
    header_frac, body_frac = 0.13, 0.76
    block_h_in = (header_frac + body_frac) * single_fig_h

    top_margin_in = 0.55
    bottom_margin_in = 0.85
    gap_in = 0.20

    fig_h = (
        top_margin_in
        + n_panels * block_h_in
        + max(n_panels - 1, 0) * gap_in
        + bottom_margin_in
    )

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(BG)

    top_frac = 1 - top_margin_in / fig_h
    bottom_frac = bottom_margin_in / fig_h
    outer = GridSpec(
        n_panels, 1, figure=fig,
        height_ratios=[1] * n_panels,
        hspace=gap_in / block_h_in,
        left=0.07, right=0.96, top=top_frac, bottom=bottom_frac,
    )

    header_fs = 20.0 if square else 16.0
    segment_fs = 12.0 if square else 9.0
    tick_fs = 7.5
    legend_fs = 13.5 if square else 7.5

    for idx, tf in enumerate(valid_tfs):
        df = dfs[tf]
        tf_info = per_tf[tf]
        has_error = "error" in tf_info

        n_show = get_candles_shown(tf, cfg)
        plot_df = df.tail(n_show).reset_index(drop=True)

        block = outer[idx, 0].subgridspec(2, 1, height_ratios=[header_frac, body_frac], hspace=0.0)
        header_ax = fig.add_subplot(block[0, 0])
        header_ax.axis("off")
        body = block[1, 0].subgridspec(3, 1, height_ratios=[4.4, 0.9, 1.3], hspace=0.10)
        ax_price = fig.add_subplot(body[0, 0])
        ax_vol = fig.add_subplot(body[1, 0], sharex=ax_price)
        ax_rsi = fig.add_subplot(body[2, 0], sharex=ax_price)

        for ax in (ax_price, ax_vol, ax_rsi):
            ax.set_facecolor(PANEL)
            ax.grid(True, linestyle="-", alpha=0.55, color=GRID, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(colors=AXIS, labelcolor=AXIS, labelsize=tick_fs)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(SPINE)
                ax.spines[side].set_linewidth(0.8)
        ax_price.tick_params(labelbottom=False)
        ax_vol.tick_params(labelbottom=False)

        direction = "NONE" if has_error else tf_info.get("direction", "NONE")

        colors = _draw_candles_single(ax_price, plot_df)
        last_x = len(plot_df) - 1

        label_zone_frac = 0.22 if square else 0.19
        right_pad = label_zone_frac * (last_x + 0.6) / (1 - label_zone_frac)
        right_pad = max(right_pad, SINGLE_CANDLE_WIDTH * 8)
        ax_price.set_xlim(-0.6, last_x + right_pad)
        ax_vol.set_xlim(-0.6, last_x + right_pad)
        ax_rsi.set_xlim(-0.6, last_x + right_pad)

        range_values = []
        level_fracs = []
        if not has_error:
            range_values, level_fracs = _draw_indicators_single(
                ax_price, df, n_show, tf, tf_info, right_pad, square=square,
            )

        all_vals = [float(plot_df["low"].min()), float(plot_df["high"].max())] + range_values
        y_low = min(all_vals)
        y_high = max(all_vals)
        y_span = max(y_high - y_low, abs(y_low) * 0.01 if y_low != 0 else 0.01)
        y_padding = y_span * 0.18
        ax_price.set_ylim(y_low - y_padding, y_high + y_padding)

        if not has_error and len(plot_df) >= 6:
            swing_high, swing_low = _find_swings(plot_df, 2, 2)
            labeled_points = _label_structure(plot_df, swing_high, swing_low)
            if labeled_points:
                labeled_points = labeled_points[-6:]
                _draw_structure_labels(
                    ax_price, labeled_points, offset=0, plot_len=len(plot_df),
                    y_span=y_span, square=square,
                )

        if not has_error:
            _draw_trigger_highlight(ax_price, plot_df, tf_info, y_span)

        _draw_volume_bars_no_ma(ax_vol, plot_df, colors)
        ax_vol.yaxis.set_major_formatter(FuncFormatter(_fmt_volume))
        ax_vol.yaxis.get_offset_text().set_visible(False)
        ax_vol.set_ylabel("Vol", color=AXIS, fontsize=8, labelpad=5)
        ax_price.set_ylabel("Price", color=AXIS, fontsize=8.5, labelpad=5)

        rsi_tail = _compute_rsi(df["close"]).tail(n_show).to_numpy()
        _draw_rsi_panel(ax_rsi, rsi_tail)
        ax_rsi.set_ylabel("RSI", color=AXIS, fontsize=8, labelpad=5)

        time_ticks = _time_axis_labels(df, n_show, tf)
        if time_ticks:
            positions, labels = time_ticks
            ax_rsi.set_xticks(positions)
            ax_rsi.set_xticklabels(labels, fontsize=7.5, color=AXIS)

        if not has_error:
            n_left = max(int(len(plot_df) * 0.15), 3)
            left_candle_fracs = [
                (float(v) - (y_low - y_padding)) / (y_high + y_padding - (y_low - y_padding))
                for v in list(plot_df["high"].iloc[:n_left]) + list(plot_df["low"].iloc[:n_left])
            ]
            handles, labels = _ordered_legend_handles(ax_price)
            legend = ax_price.legend(
                handles, labels,
                loc=_pick_legend_loc(level_fracs, left_candle_fracs), fontsize=legend_fs, framealpha=0.95,
                facecolor=BG, edgecolor=SPINE, labelcolor=TEXT, borderpad=0.4,
            )
            legend.get_frame().set_linewidth(0.7)

        setup_info = tf_info.get("setup")
        setup_info = setup_info if isinstance(setup_info, dict) else {}
        setup_type = "NONE" if has_error else setup_info.get("type", "NONE")
        setup_txt = f"  ·  {setup_type}" if setup_type and setup_type != "NONE" else ""
        header_title = f"{symbol}  ·  {tf}  ·  {direction}{setup_txt}"
        header_ax.text(
            0.0, 0.92, header_title, transform=header_ax.transAxes,
            fontsize=header_fs, fontweight="bold", color=TEXT,
            ha="left", va="top",
        )

        if not has_error:
            atr_pct = (tf_info.get("direction_analysis") or {}).get("atr_pct")
            hist, hist_prev = _macd_histogram_last(df)
            vol_ratio = _volume_ratio_last(df)
            mtf_agree = tf_info.get("mtf_agree_tfs") or []

            segments: list[tuple[str, str]] = []
            if atr_pct is not None:
                segments.append((f"ATR {atr_pct:.2f}%", AXIS))
            if hist is not None:
                arrow = "▲" if hist > 0 else "▼" if hist < 0 else "→"
                state = ""
                if hist_prev is not None:
                    if abs(hist) > abs(hist_prev):
                        state = "Expanding"
                    elif abs(hist) < abs(hist_prev):
                        state = "Contracting"
                macd_color = UP if hist > 0 else DOWN if hist < 0 else AXIS
                segments.append((f"MACD {arrow}" + (f" {state}" if state else ""), macd_color))
            if vol_ratio is not None:
                vol_color = EMA20_COLOR if vol_ratio >= 1.5 else AXIS
                segments.append((f"Vol {vol_ratio:.1f}x avg", vol_color))
            if mtf_agree:
                agree_color = UP if direction == "LONG" else DOWN if direction == "SHORT" else AXIS
                segments.append((f"MTF agree: {', '.join(mtf_agree)}", agree_color))

            if segments:
                header_pos = header_ax.get_position()
                seg_y = header_pos.y0 + 0.42 * (header_pos.y1 - header_pos.y0)
                _draw_colored_segments(fig, header_pos.x0, seg_y, segments, fontsize=segment_fs)

        if has_error:
            ax_price.text(
                0.5, 0.5, "NO DATA", transform=ax_price.transAxes, color=DOWN,
                fontsize=(22.0 if square else 16), fontweight="bold", ha="center", va="center",
            )

    ref_df = dfs[valid_tfs[0]]
    top_title_y = (fig_h - 0.16) / fig_h
    fig.text(
        0.07, top_title_y, f"{symbol}  ·  MULTI-TIMEFRAME",
        fontsize=header_fs, fontweight="bold", color=TEXT,
        ha="left", va="top",
    )
    badge_fs = 17.5 if square else 13.5
    _draw_change_badge(fig, 0.96, top_title_y, _calc_24h_change(ref_df), fontsize=badge_fs)

    footer_fs = 14.0 if square else 8.5
    disclaimer_fs = 13.0 if square else 7.5
    header_extra = pd.Timestamp.now(tz="UTC").strftime("Updated %d %b %H:%M UTC")
    footer_y = 0.16 / fig_h
    fig.text(
        0.07, footer_y,
        f"BINANCE FUTURES  ·  {symbol}  ·  {'/'.join(valid_tfs)}  ·  {header_extra}",
        fontsize=footer_fs, color=AXIS, ha="left", va="bottom",
    )
    fig.text(
        0.96, footer_y,
        "Chart-based analysis.\nNOT FINANCIAL ADVICE, DYOR.",
        fontsize=disclaimer_fs, fontweight="bold", color=TEXT,
        ha="right", va="bottom", linespacing=1.7,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor(), dpi=dpi * output_scale)
    plt.close(fig)
    return out_path
