"""vSynapse v3 — chart generator, 1 file.

Chart candlestick gaya vSch.py (rasio 3549:1600 ≈ 2.218:1): EMA, Supertrend, volume +
volume MA, level Entry/SL/TP, dan markup market structure (HH/HL/LH/LL,
zona Demand/Supply, BOS, confirmation candle, arrow target ke TP).

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
    format_signal_message,
    score_symbol,
    send_telegram_photo,
    supertrend,
)

# ============================================================
# STYLE
# ============================================================

BG = "#f5f5f7"
PANEL = "#f5f5f7"
GRID = "#b8b8c0"
TEXT = "#1a1a1e"
AXIS = "#5a5a63"
SPINE = "#c0c0c6"

UP = "#26a69a"
DOWN = "#ef5350"

EMA_COLOR = "#1565c0"

ST_UP = "#2e7d32"
ST_DOWN = "#c62828"

ENTRY = "#1565c0"
TP1 = "#00897b"
SL = "#c62828"

VOLUME_MA = "#e65100"

STRUCT_TEXT = "#c9c9d1"
DEMAND_FILL = "#1e7a52"
DEMAND_EDGE = "#26a69a"
SUPPLY_FILL = "#7a2626"
SUPPLY_EDGE = "#ef5350"
BOS_BULL = "#4fc3f7"
BOS_BEAR = "#ffab40"
CONFIRM_BULL = "#4fc3f7"
CONFIRM_BEAR = "#ffab40"
ARROW_BULL = "#4fc3f7"
ARROW_BEAR = "#ffab40"

CANDLE_WIDTH = 0.72

MAX_CANDLES_BY_TF = {
    "15m": 60,
    "1h": 48,
    "4h": 50,
}

STRUCTURE_CONTEXT = 30
MAX_BOS_EVENTS = 1
MIN_BOS_GAP_FRACTION = 0.10
MIN_LABEL_GAP_FRACTION = 0.07


def get_candles_shown(timeframe: str, cfg: dict) -> int:
    chart_cfg = cfg.get("chart", {})
    return MAX_CANDLES_BY_TF.get(timeframe, chart_cfg.get("candles_shown", 120))


# ============================================================
# HELPERS
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
    for i in range(len(df)):
        ax.bar(i, float(df["volume"].iloc[i]), color=colors[i], alpha=0.30,
               width=CANDLE_WIDTH, linewidth=0, zorder=2)

    vol_ma = df["volume"].rolling(20, min_periods=1).mean()
    ax.plot(range(len(df)), vol_ma, color=VOLUME_MA, linewidth=1.2,
             alpha=0.70, zorder=3)


def _draw_supertrend(ax, df: pd.DataFrame, st_dir: pd.Series,
                      period: int, multiplier) -> None:
    hl2 = (df["high"] + df["low"]) / 2
    atr_val = atr(df, period)
    upper = hl2 + multiplier * atr_val
    lower = hl2 - multiplier * atr_val
    level = lower.where(st_dir == 1, upper)

    x = range(len(df))
    ax.plot(x, level.where(st_dir == 1), color=ST_UP, linewidth=1.6,
             drawstyle="steps-mid", solid_joinstyle="round", solid_capstyle="round",
             label=f"Supertrend {period}/{multiplier}", zorder=3.5)
    ax.plot(x, level.where(st_dir == -1), color=ST_DOWN, linewidth=1.6,
             drawstyle="steps-mid", solid_joinstyle="round", solid_capstyle="round",
             zorder=3.5)


# ============================================================
# MARKET STRUCTURE
# ============================================================

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


def _detect_bos(df: pd.DataFrame, swing_high, swing_low) -> list:
    close = df["close"].values
    n = len(df)

    events = []
    last_swing_high = None
    last_swing_low = None

    for i in range(n):
        if swing_high[i]:
            last_swing_high = (i, float(df["high"].iloc[i]))
        if swing_low[i]:
            last_swing_low = (i, float(df["low"].iloc[i]))

        if last_swing_high is not None and i > last_swing_high[0]:
            if close[i] > last_swing_high[1]:
                events.append({
                    "idx": i, "direction": "bull",
                    "level": last_swing_high[1], "origin": last_swing_high[0],
                })
                last_swing_high = None

        if last_swing_low is not None and i > last_swing_low[0]:
            if close[i] < last_swing_low[1]:
                events.append({
                    "idx": i, "direction": "bear",
                    "level": last_swing_low[1], "origin": last_swing_low[0],
                })
                last_swing_low = None

    events.sort(key=lambda e: e["idx"])

    min_gap = max(3, int(n * MIN_BOS_GAP_FRACTION))
    kept = []
    last_kept_idx = None
    for ev in reversed(events):
        if last_kept_idx is None or (last_kept_idx - ev["idx"]) >= min_gap:
            kept.append(ev)
            last_kept_idx = ev["idx"]
        if len(kept) >= MAX_BOS_EVENTS:
            break
    kept.sort(key=lambda e: e["idx"])
    return kept


def _find_zones(df: pd.DataFrame, bos_events: list, swing_high_idxs, swing_low_idxs) -> list:
    open_ = df["open"].values
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    zones = []
    for ev in bos_events:
        idx = ev["idx"]
        if ev["direction"] == "bull":
            prior = [s for s in swing_low_idxs if s < idx]
            leg_start = prior[-1] if prior else max(0, idx - STRUCTURE_CONTEXT)
            candidates = [j for j in range(leg_start, idx) if close[j] < open_[j]]
            if not candidates:
                continue
            ob = candidates[-1]
            zones.append({
                "type": "demand", "start": ob, "bos_idx": idx,
                "top": float(high[ob]), "bottom": float(low[ob]),
            })
        else:
            prior = [s for s in swing_high_idxs if s < idx]
            leg_start = prior[-1] if prior else max(0, idx - STRUCTURE_CONTEXT)
            candidates = [j for j in range(leg_start, idx) if close[j] > open_[j]]
            if not candidates:
                continue
            ob = candidates[-1]
            zones.append({
                "type": "supply", "start": ob, "bos_idx": idx,
                "top": float(high[ob]), "bottom": float(low[ob]),
            })

    return zones


def _draw_structure_labels(ax, labeled_points: list, offset: int, plot_len: int, y_span: float) -> None:
    pad = y_span * 0.022
    for pt in labeled_points:
        px = pt["index"] - offset
        if px < 0 or px >= plot_len:
            continue
        if pt["type"] == "H":
            ax.text(px, pt["price"] + pad, pt["label"], color=STRUCT_TEXT,
                    fontsize=6.0, fontweight="bold", ha="center", va="bottom",
                    zorder=9, clip_on=False)
        else:
            ax.text(px, pt["price"] - pad, pt["label"], color=STRUCT_TEXT,
                    fontsize=6.0, fontweight="bold", ha="center", va="top",
                    zorder=9, clip_on=False)


def _draw_zones(ax, zones: list, offset: int, plot_len: int, last_x: int, y_span: float) -> None:
    pad_zone = max(y_span, 1e-9) * 0.055
    for z in zones:
        bos_px = z["bos_idx"] - offset
        if bos_px < -0.5:
            continue
        start_px = max(z["start"] - offset, -0.4)
        end_px = min(bos_px + 3, last_x + 0.4)
        if end_px <= start_px:
            end_px = start_px + 1

        is_demand = z["type"] == "demand"
        fill = DEMAND_FILL if is_demand else SUPPLY_FILL
        edge = DEMAND_EDGE if is_demand else SUPPLY_EDGE
        label = "Demand" if is_demand else "Supply"

        ax.add_patch(Rectangle(
            (start_px, z["bottom"]), end_px - start_px, z["top"] - z["bottom"],
            facecolor=fill, edgecolor=edge, alpha=0.24, linewidth=0.8,
            zorder=1.2,
        ))
        label_y = (z["bottom"] - pad_zone) if is_demand else (z["top"] + pad_zone)
        va = "top" if is_demand else "bottom"
        ax.text(max(start_px, 0), label_y, f" {label} ", color=edge,
                fontsize=6.0, fontweight="bold", ha="left", va=va,
                alpha=0.90, zorder=1.5, clip_on=False)


def _draw_bos_and_confirmation(ax, bos_events: list, offset: int, plot_df: pd.DataFrame) -> None:
    plot_len = len(plot_df)
    high = plot_df["high"].values
    low = plot_df["low"].values
    y_span = float(plot_df["high"].max() - plot_df["low"].min())
    pad_marker = max(y_span, 1e-9) * 0.012
    pad_label = max(y_span, 1e-9) * 0.075

    for ev in bos_events:
        idx_px = ev["idx"] - offset
        if idx_px < 0 or idx_px >= plot_len:
            continue
        origin_px = max(ev["origin"] - offset, -0.4)
        is_bull = ev["direction"] == "bull"
        color = BOS_BULL if is_bull else BOS_BEAR

        ax.plot([origin_px, idx_px], [ev["level"], ev["level"]], color=color,
                 linestyle=(0, (5, 3)), linewidth=1.0, alpha=0.80, zorder=4)

        marker_color = CONFIRM_BULL if is_bull else CONFIRM_BEAR
        if is_bull:
            ax.plot(idx_px, high[idx_px] + pad_marker, marker="^", color=marker_color,
                     markersize=5.0, markeredgecolor="#000000", markeredgewidth=0.5,
                     zorder=10, clip_on=False)
            label_y = max(ev["level"], high[idx_px]) + pad_label
        else:
            ax.plot(idx_px, low[idx_px] - pad_marker, marker="v", color=marker_color,
                     markersize=5.0, markeredgecolor="#000000", markeredgewidth=0.5,
                     zorder=10, clip_on=False)
            label_y = min(ev["level"], low[idx_px]) - pad_label

        ax.text(idx_px + 0.9, label_y, "BOS", color="#050505",
                fontsize=5.8, fontweight="bold",
                ha="left", va="center",
                bbox=dict(facecolor=color, edgecolor="none",
                          boxstyle="round,pad=0.12", alpha=0.95),
                zorder=9, clip_on=False)


def _draw_target_arrow(ax, plot_df: pd.DataFrame, tp_price, last_x: int) -> None:
    """Satu panah arah target, dari candle terakhir (harga close
    terkini) menuju level TP aktual sinyal. Opacity dijaga sedang
    (70-80%) supaya tidak mendominasi chart."""
    if tp_price is None:
        return

    current_price = float(plot_df["close"].iloc[-1])
    is_bull = tp_price >= current_price
    color = ARROW_BULL if is_bull else ARROW_BEAR

    x_start = last_x + 0.4
    x_end = last_x + 3.0

    ax.annotate(
        "", xy=(x_end, tp_price), xytext=(x_start, current_price),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, alpha=0.75,
                          shrinkA=1, shrinkB=1, mutation_scale=10),
        zorder=9,
    )


def _declutter_labels(labeled_points: list, min_gap: int) -> list:
    kept = []
    last_kept_idx = None
    for pt in sorted(labeled_points, key=lambda p: p["index"], reverse=True):
        if last_kept_idx is None or (last_kept_idx - pt["index"]) >= min_gap:
            kept.append(pt)
            last_kept_idx = pt["index"]
    kept.sort(key=lambda p: p["index"])
    return kept


def _compute_structure(work_df: pd.DataFrame) -> dict:
    swing_high, swing_low = _find_swings(work_df)
    labeled_points = _label_structure(work_df, swing_high, swing_low)
    swing_high_idxs = [i for i in range(len(work_df)) if swing_high[i]]
    swing_low_idxs = [i for i in range(len(work_df)) if swing_low[i]]
    bos_events = _detect_bos(work_df, swing_high, swing_low)
    zones = _find_zones(work_df, bos_events, swing_high_idxs, swing_low_idxs)

    occupied = set()
    for ev in bos_events:
        for i in (ev["idx"], ev["origin"]):
            occupied.update(range(i - 2, i + 3))
    labeled_points = [p for p in labeled_points if p["index"] not in occupied]

    min_label_gap = max(3, int(len(work_df) * MIN_LABEL_GAP_FRACTION))
    labeled_points = _declutter_labels(labeled_points, min_label_gap)

    return {
        "labeled_points": labeled_points,
        "bos_events": bos_events,
        "zones": zones,
    }


def build_chart(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    signal,
    cfg: dict,
    out_path: str,
) -> str:
    n_show = get_candles_shown(timeframe, cfg)

    work_df = df.tail(n_show + STRUCTURE_CONTEXT).reset_index(drop=True)
    offset = max(len(work_df) - n_show, 0)
    plot_df = work_df.tail(n_show).reset_index(drop=True)

    structure = _compute_structure(work_df)

    ema_period = cfg["indicators"]["ema"]["period"]
    st_period = cfg["indicators"]["supertrend"]["period"]
    st_mult = cfg["indicators"]["supertrend"]["multiplier"]

    ema_full = ema(df["close"], ema_period).tail(n_show).reset_index(drop=True)
    st_dir_full = supertrend(df, st_period, st_mult).tail(n_show).reset_index(drop=True)

    chart_cfg = cfg.get("chart", {})
    width_px = chart_cfg.get("width_px", 2000)
    # Rasio output dikunci 20:9 ≈ 2.22:1 (height_ratio = tinggi/lebar = 9/20).
    height_ratio = 9 / 20
    dpi = 150
    output_scale = 1  # output final 1x, layout/proporsi tidak berubah
    fig_w = width_px / dpi
    fig_h = (width_px * height_ratio) / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(BG)

    gs = GridSpec(
        2, 1, figure=fig,
        height_ratios=[4.4, 0.6],
        hspace=0.06,
        left=0.07, right=0.96, top=0.87, bottom=0.11,
    )
    ax_price = fig.add_subplot(gs[0, 0])
    ax_vol = fig.add_subplot(gs[1, 0], sharex=ax_price)

    for ax in (ax_price, ax_vol):
        ax.set_facecolor(PANEL)
        ax.grid(True, linestyle="-", alpha=0.6, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(colors=AXIS, labelcolor=AXIS, labelsize=7.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(SPINE)
            ax.spines[side].set_linewidth(0.8)

    ax_price.tick_params(labelbottom=False)

    colors = _draw_candles(ax_price, plot_df)
    ax_price.plot(range(len(plot_df)), ema_full, color=EMA_COLOR, linewidth=1.6,
                  solid_capstyle="round", label=f"EMA {ema_period}", zorder=4)
    _draw_supertrend(ax_price, plot_df, st_dir_full, st_period, st_mult)

    last_x = len(plot_df) - 1

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

    zone_values = []
    for z in structure["zones"]:
        zone_values.append(z["top"])
        zone_values.append(z["bottom"])

    level_values = [item["level"] for item in levels]
    y_low = min([float(plot_df["low"].min())] + level_values + zone_values)
    y_high = max([float(plot_df["high"].max())] + level_values + zone_values)
    y_span = max(y_high - y_low, abs(y_low) * 0.01 if y_low != 0 else 0.01)
    y_padding = y_span * 0.18
    ax_price.set_ylim(y_low - y_padding, y_high + y_padding)

    gap_from_candle = 4.0
    label_width_est = 13.0
    gap_from_edge = 0.4
    extra_margin = gap_from_candle + label_width_est + gap_from_edge
    label_x = last_x + gap_from_candle

    ax_price.set_xlim(-0.6, last_x + extra_margin)
    ax_vol.set_xlim(-0.6, last_x + extra_margin)

    plot_len = len(plot_df)
    _draw_zones(ax_price, structure["zones"], offset, plot_len, last_x, y_span)
    _draw_bos_and_confirmation(ax_price, structure["bos_events"], offset, plot_df)
    _draw_target_arrow(ax_price, plot_df, signal.tp, last_x)
    _draw_structure_labels(ax_price, structure["labeled_points"], offset, plot_len, y_span)

    _place_level_labels(ax_price, levels, label_x)

    _draw_volume(ax_vol, plot_df, colors)
    ax_vol.set_ylabel("Vol", color=AXIS, fontsize=8, labelpad=5)
    ax_price.set_ylabel("Price", color=AXIS, fontsize=8.5, labelpad=5)

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

    # Risk:Reward menggantikan Score di header (lebih relevan buat pembaca chart).
    rr_text = None
    if signal.entry is not None and signal.sl is not None and signal.tp is not None:
        risk = abs(signal.entry - signal.sl)
        reward = abs(signal.tp - signal.entry)
        if risk > 0:
            rr_text = f"RR 1:{reward / risk:.2f}"

    header_extra = rr_text if rr_text else pd.Timestamp.utcnow().strftime("Updated %d %b %H:%M UTC")

    fig.text(0.07, 0.965,
              f"{symbol}  ·  {timeframe}  ·  {signal.direction}  ·  {header_extra}",
              fontsize=18, fontweight="bold", color=TEXT, ha="left", va="top")
    fig.text(0.07, 0.02, f"BINANCE FUTURES  ·  {symbol}  ·  {timeframe}",
              fontsize=7, color=AXIS, ha="left", va="bottom")
    fig.text(0.96, 0.02, "Not financial advice",
              fontsize=7, color=AXIS, ha="right", va="bottom")

    fig.savefig(out_path, facecolor=fig.get_facecolor(), dpi=dpi * output_scale)
    plt.close(fig)
    return out_path


async def _fetch_and_build(symbol: str, timeframe: str, cfg: dict, out_path: str) -> str:
    async with BinanceFuturesClient() as client:
        n_show = get_candles_shown(timeframe, cfg)
        limit = max(400, n_show + STRUCTURE_CONTEXT + 250)
        kline = await client.get_klines(symbol, timeframe, limit=limit)
    signal = score_symbol(kline.df, symbol, cfg)
    result_path = build_chart(kline.df, symbol, timeframe, signal, cfg, out_path)

    # Khusus jalur CLI/manual (mis. workflow "Chart Generator (manual)"):
    # kirim chart tunggal ini ke Telegram sebagai foto. Tidak dipanggil dari
    # scanner.run_scan, yang mengirim hasil batch lewat 1 file zip.
    try:
        await send_telegram_photo(result_path, format_signal_message(signal), cfg)
    except Exception as exc:  # jangan sampai gagal kirim Telegram menghentikan workflow
        print(f"Gagal kirim chart ke Telegram: {exc}")

    return result_path


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
