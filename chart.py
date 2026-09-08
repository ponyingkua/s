"""vSynapse chart generator — candlestick + EMA/Supertrend/volume + market
structure (HH/HL/LH/LL, zona Demand/Supply, BOS, panah target TP).

Contoh: python chart.py --symbol BTCUSDT --timeframe 1h
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
)

# ============================================================
# STYLE
# ============================================================

BG = "#121417"
PANEL = "#121417"
GRID = "#3A3A3A"
TEXT = "#F2F2F2"
AXIS = "#B8B8B8"
SPINE = "#4A4A4A"

UP = "#26A69A"
DOWN = "#EF5350"

EMA_COLOR = "#FFD54F"

ST_UP = "#66BB6A"
ST_DOWN = "#EF5350"

ENTRY = "#42A5F5"
TP1 = "#26A69A"
SL = "#EF5350"

VOLUME_MA = "#FFB74D"

STRUCT_TEXT = "#BDBDBD"
DEMAND_FILL = "#1B5E40"
DEMAND_EDGE = UP
SUPPLY_FILL = "#6B1F1F"
SUPPLY_EDGE = DOWN

# Penanda arah (BOS, panah target) memakai warna candle yang sama supaya
# konsisten: bull = hijau (UP), bear = merah (DOWN).
BOS_BULL = UP
BOS_BEAR = DOWN
ARROW_BULL = UP
ARROW_BEAR = DOWN

# Palet garis untuk comparison chart (multi-simbol), berurutan supaya tiap
# simbol dapat warna beda dan tetap kebaca di background gelap.
COMPARE_PALETTE = [
    "#42A5F5", "#FFD54F", "#26A69A", "#EF5350",
    "#AB47BC", "#66BB6A", "#FFB74D", "#4FC3F7",
]

CANDLE_WIDTH = 0.8

MAX_CANDLES_BY_TF = {
    "15m": 50,   # ideal 45-55
    "1h": 60,    # ideal 55-65
    "4h": 45,    # ideal 40-50
}

STRUCTURE_CONTEXT = 30
MAX_BOS_EVENTS = 1
MIN_BOS_GAP_FRACTION = 0.10
MIN_LABEL_GAP_FRACTION = 0.18


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


def _calc_24h_change(df: pd.DataFrame) -> float | None:
    """Persentase perubahan harga 24 jam terakhir, dihitung langsung dari
    kline yang sudah di-fetch (tanpa request tambahan ke API). None kalau
    histori yang tersedia belum menutup 24 jam penuh."""
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
    """Badge % perubahan 24 jam, kotak solid warna hijau/merah di pojok
    header -- menggantikan posisi teks 'Updated ...' yang lama."""
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


def _place_level_labels(ax, levels: list, label_x: float, min_gap: float,
                         square: bool = False) -> None:
    # Kotak solid berwarna sesuai level (entry/tp/sl), teks putih tebal di
    # dalamnya supaya tetap terbaca jelas dan menonjol di layar kecil (HP).
    # Kalau 2+ level berdekatan (mis. ENTRY & SL cuma beda dikit), kotak
    # label bisa tumpang tindih kalau ditaruh persis di harga aslinya --
    # jadi posisi label di-declutter vertikal (dorong-atas lalu dorong-bawah)
    # supaya tidak saling menimpa. Level asli tetap ditunjukkan oleh garis
    # putus-putus (axhline) yang warnanya sama dengan kotak labelnya.
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

    level_fontsize = 15.5 if square else 9.5
    for item, label_y in zip(ordered, positions):
        ax.text(
            label_x, label_y, item["text"],
            color=TEXT,
            va="center", ha="left", fontweight="bold", fontsize=level_fontsize,
            zorder=8, clip_on=False,
            bbox=dict(
                boxstyle="square,pad=0.35",
                facecolor=item["color"],
                edgecolor=item["color"],
                linewidth=0,
                alpha=0.92,
            ),
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


def _draw_volume(ax, df: pd.DataFrame, colors: list, vol_ma_lookback: int = 20) -> None:
    for i in range(len(df)):
        ax.bar(i, float(df["volume"].iloc[i]), color=colors[i], alpha=0.48,
               width=CANDLE_WIDTH, linewidth=0, zorder=2)

    vol_ma = df["volume"].rolling(vol_ma_lookback, min_periods=1).mean()
    ax.plot(range(len(df)), vol_ma, color=VOLUME_MA, linewidth=1.2,
             alpha=0.80, zorder=3)


def _supertrend_trailing(df: pd.DataFrame, period: int, multiplier):
    """Hitung level Supertrend versi 'trailing band' (band cuma bergerak
    searah tren seperti trailing-stop, tidak dihitung ulang dari nol tiap
    candle) supaya garis yang digambar mulus, bukan zig-zag. Dipakai khusus
    untuk visual chart -- tidak memengaruhi skor/sinyal di scanner.py."""
    hl2 = (df["high"] + df["low"]) / 2
    atr_val = atr(df, period)
    basic_upper = hl2 + multiplier * atr_val
    basic_lower = hl2 - multiplier * atr_val

    close = df["close"]
    n = len(df)
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    trend = pd.Series(1, index=df.index, dtype=int)

    # ATR butuh `period` candle pertama untuk "pemanasan" dan bernilai NaN
    # sebelum itu. Rekursi trailing-band di bawah cuma valid begitu ATR
    # sudah terisi — kalau dipaksa mulai dari index 0 yang NaN, band jadi
    # macet permanen di NaN (perbandingan apa pun dengan NaN selalu False,
    # jadi cabang else "bawa nilai lama" yang selalu kepilih selamanya).
    # Ini penyebab garis Supertrend tidak pernah muncul di chart. Fix-nya:
    # rekursi baru mulai dari candle pertama yang ATR-nya sudah valid.
    first_valid = atr_val.first_valid_index()
    if first_valid is None:
        return pd.Series(np.nan, index=df.index), trend
    start_pos = df.index.get_loc(first_valid)

    for i in range(start_pos + 1, n):
        if basic_upper.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if basic_lower.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        if close.iloc[i] > final_upper.iloc[i - 1]:
            trend.iloc[i] = 1
        elif close.iloc[i] < final_lower.iloc[i - 1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]

    level = final_lower.where(trend == 1, final_upper)
    level.iloc[:start_pos] = np.nan  # belum ada ATR valid, jangan digambar
    return level, trend


def _draw_supertrend(ax, level: pd.Series, trend: pd.Series,
                      period: int, multiplier) -> None:
    x = range(len(level))
    ax.plot(x, level.where(trend == 1), color=ST_UP, linewidth=1.4, alpha=0.90,
             drawstyle="steps-post", solid_joinstyle="round",
             label=f"Supertrend {period}/{multiplier}", zorder=7)
    ax.plot(x, level.where(trend == -1), color=ST_DOWN, linewidth=1.4, alpha=0.90,
             drawstyle="steps-post", solid_joinstyle="round", zorder=7)


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


def _pad_zone_bounds(z: dict, y_span: float) -> tuple[float, float]:
    """Perbesar tinggi kotak S/D. Kotak mentah (persis wick candle OB) sering
    terlalu tipis kalau candle-nya kecil / low volatility. Dikasih tinggi
    minimum + padding ekstra, searah alami zona -- demand (beli) diperpanjang
    ke bawah, supply (jual) diperpanjang ke atas -- jadi kotak tetap merujuk
    ke candle OB yang sama persis, cuma lebih kelihatan jelas di chart."""
    top, bottom = z["top"], z["bottom"]
    is_demand = z["type"] == "demand"
    min_height = y_span * 0.10
    extra_pad = y_span * 0.045

    height = top - bottom
    if height < min_height:
        deficit = min_height - height
        if is_demand:
            bottom -= deficit
        else:
            top += deficit

    if is_demand:
        bottom -= extra_pad
    else:
        top += extra_pad

    return top, bottom


def _draw_structure_labels(ax, labeled_points: list, offset: int, plot_len: int, y_span: float,
                             square: bool = False) -> None:
    pad = y_span * 0.022
    fontsize = 12.5 if square else 6.0
    for pt in labeled_points:
        px = pt["index"] - offset
        if px < 0 or px >= plot_len:
            continue
        if pt["type"] == "H":
            ax.text(px, pt["price"] + pad, pt["label"], color=STRUCT_TEXT,
                    fontsize=fontsize, fontweight="bold", ha="center", va="bottom",
                    zorder=9, clip_on=False)
        else:
            ax.text(px, pt["price"] - pad, pt["label"], color=STRUCT_TEXT,
                    fontsize=fontsize, fontweight="bold", ha="center", va="top",
                    zorder=9, clip_on=False)


def _draw_zones(ax, zones: list, offset: int, plot_len: int, last_x: int, y_span: float,
                 square: bool = False) -> None:
    for z in zones:
        bos_px = z["bos_idx"] - offset
        if bos_px < -0.5:
            continue
        start_px = max(z["start"] - offset, -0.4)
        end_px = min(bos_px + 4, last_x + 0.4)
        if end_px <= start_px:
            end_px = start_px + 1.5

        is_demand = z["type"] == "demand"
        fill = DEMAND_FILL if is_demand else SUPPLY_FILL
        edge = DEMAND_EDGE if is_demand else SUPPLY_EDGE
        label = "D" if is_demand else "S"

        ax.add_patch(Rectangle(
            (start_px, z["bottom"]), end_px - start_px, z["top"] - z["bottom"],
            facecolor=fill, edgecolor=edge, alpha=0.40, linewidth=1.1,
            zorder=1.2,
        ))

        # Teks S/D — warna hitam (sama dengan background gelap)
        mid_x = (start_px + end_px) / 2
        mid_y = (z["top"] + z["bottom"]) / 2
        ax.text(mid_x, mid_y, label, color="#F2F2F2",
                fontsize=(14.0 if square else 7.5), fontweight="bold", ha="center", va="center",
                alpha=0.95, zorder=1.6, clip_on=False)


def _draw_bos_and_confirmation(ax, bos_events: list, offset: int, plot_df: pd.DataFrame,
                                 square: bool = False) -> None:
    """Penanda BOS sederhana: garis level + segitiga kecil + teks BOS."""
    plot_len = len(plot_df)
    high = plot_df["high"].values
    low = plot_df["low"].values
    y_span = float(plot_df["high"].max() - plot_df["low"].min())
    pad_marker = max(y_span, 1e-9) * 0.015
    pad_label = max(y_span, 1e-9) * 0.07

    for ev in bos_events:
        idx_px = ev["idx"] - offset
        if idx_px < 0 or idx_px >= plot_len:
            continue
        origin_px = max(ev["origin"] - offset, -0.4)
        is_bull = ev["direction"] == "bull"
        color = BOS_BULL if is_bull else BOS_BEAR

        # Garis level sederhana
        ax.plot([origin_px, idx_px], [ev["level"], ev["level"]], color=color,
                 linestyle=(0, (4, 3)), linewidth=1.1, alpha=0.85, zorder=4)

        # Segitiga kecil
        if is_bull:
            ax.plot(idx_px, high[idx_px] + pad_marker, marker="^", color=color,
                     markersize=6.0, zorder=10, clip_on=False)
            label_y = max(ev["level"], high[idx_px]) + pad_label
        else:
            ax.plot(idx_px, low[idx_px] - pad_marker, marker="v", color=color,
                     markersize=6.0, zorder=10, clip_on=False)
            label_y = min(ev["level"], low[idx_px]) - pad_label

        # Teks BOS sederhana — warna hitam
        ax.text(idx_px + 0.9, label_y, "BOS", color="#F2F2F2",
                fontsize=(12.5 if square else 6.0), fontweight="bold",
                ha="left", va="center",
                zorder=11, clip_on=False)


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
                          linestyle=(0, (5, 3)),
                          connectionstyle="arc3,rad=0.35",
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


def _calculate_visible_range(
    work_df: pd.DataFrame,
    structure: dict,
    timeframe: str,
    max_candles: int,
    left_padding: int = 8,
) -> tuple[int, int]:
    """Window selalu tetap `max_candles` lebar (kalau histori cukup) supaya
    jumlah candle yang tampil konsisten antar chart. Struktur terbaru (BOS +
    zone terkait) dipastikan tetap terlihat; kalau strukturnya dekat ujung
    kanan, sisa ruang di kiri diisi candle histori biasa, bukan dipersempit."""
    n = len(work_df)
    end_idx = n - 1

    important = []

    # Prioritas utama: BOS terbaru + zone yang terkait
    bos_events = structure.get("bos_events", [])
    zones = structure.get("zones", [])

    if bos_events:
        # Ambil BOS paling kanan (terbaru)
        latest_bos = max(bos_events, key=lambda e: e["idx"])
        important.append(latest_bos["origin"])
        important.append(latest_bos["idx"])

        # Zone yang terkait dengan BOS tersebut
        for z in zones:
            if abs(z["bos_idx"] - latest_bos["idx"]) <= 3:
                important.append(z["start"])
                important.append(z["bos_idx"])

    # Cadangan: kalau tidak ada BOS, pakai zone terakhir
    if not important and zones:
        latest_zone = max(zones, key=lambda z: z["bos_idx"])
        important.append(latest_zone["start"])
        important.append(latest_zone["bos_idx"])

    # Tambahkan hanya label yang dekat dengan struktur utama (maks 2 label terakhir)
    labeled = structure.get("labeled_points", [])
    if labeled and important:
        rightmost_important = max(important)
        recent_labels = [
            p for p in labeled
            if p["index"] >= rightmost_important - 25
        ]
        recent_labels = sorted(recent_labels, key=lambda p: p["index"])[-2:]
        for p in recent_labels:
            important.append(p["index"])

    if not important:
        start_idx = max(0, n - max_candles)
        return start_idx, end_idx

    leftmost = min(important)
    start_idx = max(0, leftmost - left_padding)

    # Batasi maksimum candle
    if (end_idx - start_idx + 1) > max_candles:
        start_idx = max(0, end_idx - max_candles + 1)

    # Isi penuh sampai max_candles: kalau struktur terbaru dekat ujung kanan,
    # window jangan sampai lebih sempit dari chart lain — geser start_idx ke
    # kiri sejauh histori memungkinkan.
    full_window_start = max(0, end_idx - max_candles + 1)
    start_idx = min(start_idx, full_window_start)

    return start_idx, end_idx


def build_chart(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    signal,
    cfg: dict,
    out_path: str,
    preset: str = "standard",
    hide_indicators: bool = False,
    square: bool = False,
) -> str:
    """Candle + EMA/Supertrend + volume + market structure (zona Demand/
    Supply, BOS, label HH/HL/LH/LL, panah arah target) selalu digambar.
    preset="clean" (dipakai jalur CLI/manual) menyembunyikan garis+label
    ENTRY/TP/SL saja -- signature `signal`/`preset` dipertahankan supaya
    pemanggilan langsung dari scanner.py (jalur otomatis) tidak perlu
    berubah; default preset="standard" = ENTRY/TP/SL tetap tampil seperti
    semula. hide_indicators dan square independen, bisa dipakai kapan saja.
    square=True membuat kanvas rasio 1:1 (cocok untuk post feed Binance
    Square/IG)."""
    show_levels = preset != "clean"
    n_show_max = get_candles_shown(timeframe, cfg)

    # Ambil window lebih lebar untuk deteksi structure
    work_len = n_show_max + STRUCTURE_CONTEXT + 50
    work_df = df.tail(work_len).reset_index(drop=True)

    # Hitung structure dulu
    structure = _compute_structure(work_df)

    # Hitung range visible yang cerdas berdasarkan structure
    left_pad = 10 if timeframe == "15m" else 7
    start_idx, end_idx = _calculate_visible_range(
        work_df,
        structure,
        timeframe,
        max_candles=n_show_max,
        left_padding=left_pad,
    )

    plot_df = work_df.iloc[start_idx : end_idx + 1].reset_index(drop=True)
    offset = start_idx

    # Perbesar tinggi kotak S/D memakai rentang harga yang benar-benar
    # tampil (plot_df), bukan tinggi candle OB itu sendiri, supaya ukuran
    # kotak konsisten antar chart. Di-mutate langsung di sini (sebelum
    # dipakai untuk hitung ylim maupun digambar) supaya keduanya konsisten.
    rough_span = float(plot_df["high"].max() - plot_df["low"].min())
    for z in structure["zones"]:
        z["top"], z["bottom"] = _pad_zone_bounds(z, rough_span)

    ema_period = cfg["indicators"]["ema"]["period"]
    st_period = cfg["indicators"]["supertrend"]["period"]
    st_mult = cfg["indicators"]["supertrend"]["multiplier"]

    # EMA dihitung dari `df` penuh (bukan work_df) supaya warm-up-nya lebih
    # panjang dan lebih akurat, lalu diambil `len(plot_df)` candle terakhir.
    # plot_df selalu berakhir di candle paling akhir dari df, jadi .tail()
    # ini selalu align dengan window yang ditampilkan.
    ema_full = ema(df["close"], ema_period).tail(len(plot_df)).reset_index(drop=True)

    st_level_work, st_trend_work = _supertrend_trailing(work_df, st_period, st_mult)
    st_level_full = st_level_work.iloc[start_idx:end_idx+1].reset_index(drop=True)
    st_trend_full = st_trend_work.iloc[start_idx:end_idx+1].reset_index(drop=True)

    chart_cfg = cfg.get("chart", {})
    width_px = chart_cfg.get("width_px", 2800)
    height_ratio = 1.0 if square else chart_cfg.get("height_ratio", 9 / 20)
    dpi = 200
    output_scale = 2  # render 2x lalu disimpan di dpi lebih tinggi supaya
                       # hasil PNG lebih tajam; proporsi/layout tidak berubah
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
        ax.grid(True, linestyle="-", alpha=0.8, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(colors=AXIS, labelcolor=AXIS, labelsize=7.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(SPINE)
            ax.spines[side].set_linewidth(0.8)

    ax_price.tick_params(labelbottom=False)

    colors = _draw_candles(ax_price, plot_df)
    if not hide_indicators:
        ax_price.plot(range(len(plot_df)), ema_full, color=EMA_COLOR, linewidth=1.6,
                      solid_capstyle="round", label=f"EMA {ema_period}", zorder=4)
        _draw_supertrend(ax_price, st_level_full, st_trend_full, st_period, st_mult)

    last_x = len(plot_df) - 1

    ref_price = signal.entry if signal.entry is not None else (
        signal.sl if signal.sl is not None else float(plot_df["close"].iloc[-1])
    )
    dec = decimals_from_price(ref_price)

    levels = []
    if show_levels:
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

    # Level Supertrend ikut dihitung supaya garisnya tidak ke-clip oleh
    # set_ylim ketika band-nya melebar keluar dari range harga/level/zone.
    st_values = [] if hide_indicators else [v for v in st_level_full.tolist() if pd.notna(v)]

    level_values = [item["level"] for item in levels]
    y_low = min([float(plot_df["low"].min())] + level_values + zone_values + st_values)
    y_high = max([float(plot_df["high"].max())] + level_values + zone_values + st_values)
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
    _draw_zones(ax_price, structure["zones"], offset, plot_len, last_x, y_span, square=square)
    _draw_bos_and_confirmation(ax_price, structure["bos_events"], offset, plot_df, square=square)
    _draw_target_arrow(ax_price, plot_df, signal.tp, last_x)
    _draw_structure_labels(ax_price, structure["labeled_points"], offset, plot_len, y_span, square=square)

    label_min_gap = (ax_price.get_ylim()[1] - ax_price.get_ylim()[0]) * 0.065
    _place_level_labels(ax_price, levels, label_x, label_min_gap, square=square)

    vol_ma_lookback = cfg.get("indicators", {}).get("volume_spike", {}).get("lookback", 20)
    _draw_volume(ax_vol, plot_df, colors, vol_ma_lookback)
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

    if not hide_indicators:
        legend = ax_price.legend(
            loc="upper left", fontsize=(13.5 if square else 7.5), framealpha=0.95,
            facecolor=BG, edgecolor=SPINE, labelcolor=TEXT, borderpad=0.4,
        )
        legend.get_frame().set_linewidth(0.7)

    header_extra = pd.Timestamp.now(tz="UTC").strftime("Updated %d %b %H:%M UTC")
    setup_label = f"  ·  {signal.setup_type}" if signal.setup_type else ""
    header_title = f"{symbol}  ·  {timeframe}  ·  {signal.direction}{setup_label}"

    fig.text(0.07, 0.965, header_title,
              fontsize=(22.0 if square else 18), fontweight="bold", color=TEXT, ha="left", va="top")
    _draw_change_badge(fig, 0.96, 0.965, _calc_24h_change(df), fontsize=(19.0 if square else 15))

    if square:
        footer_left = f"BINANCE FUTURES  ·  {symbol}  ·  {timeframe}\n{header_extra}"
        fig.text(0.07, 0.028, footer_left,
                  fontsize=12.5, color=AXIS, ha="left", va="bottom", linespacing=1.6)
    else:
        fig.text(0.07, 0.02, f"BINANCE FUTURES  ·  {symbol}  ·  {timeframe}  ·  {header_extra}",
                  fontsize=7, color=AXIS, ha="left", va="bottom")

    # Disclaimer kanan-bawah: satu blok teks 2 baris, font & alignment
    # seragam supaya rapi (sebelumnya 2 fig.text terpisah dengan ukuran
    # font berbeda-beda dan emoji yang bisa tampil sebagai kotak kosong
    # kalau font sistem tidak dukung emoji).
    fig.text(
        0.96, 0.013,
        "Chart-based analysis for educational purposes only.\nNOT FINANCIAL ADVICE, DYOR.",
        fontsize=(13.0 if square else 7.5), fontweight="bold", color=TEXT, ha="right", va="bottom",
        linespacing=1.7,
    )

    fig.savefig(out_path, facecolor=fig.get_facecolor(), dpi=dpi * output_scale)
    plt.close(fig)
    return out_path


def build_multi_tf_card(
    dfs: dict,
    symbol: str,
    timeframes: list,
    signals: dict,
    cfg: dict,
    out_path: str,
) -> str:
    """Kartu multi-timeframe: N panel berdampingan (candle+EMA+Supertrend+
    volume mini per timeframe) dalam 1 gambar -- untuk konten "gimana
    posisi multi-TF" di Binance Square. Tidak menyentuh build_chart/
    scanner.py sama sekali, cuma memakai ulang helper gambar yang sudah ada."""
    n_panels = len(timeframes)
    chart_cfg = cfg.get("chart", {})
    width_px = chart_cfg.get("width_px", 2800)
    dpi = 200
    fig_w = width_px / dpi
    fig_h = fig_w * 0.40
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(BG)

    outer = GridSpec(1, n_panels, figure=fig, wspace=0.16,
                      left=0.045, right=0.98, top=0.85, bottom=0.11)

    for col, tf in enumerate(timeframes):
        df = dfs[tf]
        signal = signals[tf]

        n_show = max(30, get_candles_shown(tf, cfg) // 2 + 10)
        work_len = n_show + STRUCTURE_CONTEXT + 30
        work_df = df.tail(work_len).reset_index(drop=True)
        structure = _compute_structure(work_df)
        start_idx, end_idx = _calculate_visible_range(
            work_df, structure, tf, max_candles=n_show, left_padding=6,
        )
        plot_df = work_df.iloc[start_idx:end_idx + 1].reset_index(drop=True)

        inner = outer[0, col].subgridspec(2, 1, height_ratios=[4, 1], hspace=0.08)
        ax_p = fig.add_subplot(inner[0, 0])
        ax_v = fig.add_subplot(inner[1, 0], sharex=ax_p)

        for ax in (ax_p, ax_v):
            ax.set_facecolor(PANEL)
            ax.grid(True, linestyle="-", alpha=0.7, color=GRID, linewidth=0.4)
            ax.set_axisbelow(True)
            ax.tick_params(colors=AXIS, labelcolor=AXIS, labelsize=6.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(SPINE)
                ax.spines[side].set_linewidth(0.6)
        ax_p.tick_params(labelbottom=False)

        colors = _draw_candles(ax_p, plot_df)

        ema_period = cfg["indicators"]["ema"]["period"]
        ema_vals = ema(df["close"], ema_period).tail(len(plot_df)).reset_index(drop=True)
        ax_p.plot(range(len(plot_df)), ema_vals, color=EMA_COLOR, linewidth=1.1, zorder=4)

        st_period = cfg["indicators"]["supertrend"]["period"]
        st_mult = cfg["indicators"]["supertrend"]["multiplier"]
        st_level_work, st_trend_work = _supertrend_trailing(work_df, st_period, st_mult)
        st_level = st_level_work.iloc[start_idx:end_idx + 1].reset_index(drop=True)
        st_trend = st_trend_work.iloc[start_idx:end_idx + 1].reset_index(drop=True)
        _draw_supertrend(ax_p, st_level, st_trend, st_period, st_mult)

        last_x = len(plot_df) - 1
        y_low = float(plot_df["low"].min())
        y_high = float(plot_df["high"].max())
        y_span = max(y_high - y_low, abs(y_low) * 0.01 if y_low != 0 else 0.01)
        pad = y_span * 0.12
        ax_p.set_ylim(y_low - pad, y_high + pad)
        ax_p.set_xlim(-0.6, last_x + 0.6)
        ax_v.set_xlim(-0.6, last_x + 0.6)

        vol_lookback = cfg.get("indicators", {}).get("volume_spike", {}).get("lookback", 20)
        _draw_volume(ax_v, plot_df, colors, vol_lookback)

        badge_color = UP if str(signal.direction).upper().startswith("LONG") else DOWN
        setup_txt = f"  ·  {signal.setup_type}" if getattr(signal, "setup_type", None) else ""
        ax_p.set_title(f"{tf}  ·  {signal.direction}{setup_txt}", color=badge_color,
                        fontsize=9.5, fontweight="bold", loc="left", pad=6)

        dec = decimals_from_price(float(plot_df["close"].iloc[-1]))
        last_price = format_price(plot_df["close"].iloc[-1], dec)
        ax_p.text(0.99, 0.03, last_price, transform=ax_p.transAxes, color=TEXT,
                   fontsize=8, fontweight="bold", ha="right", va="bottom", zorder=9)

    fig.text(0.045, 0.95, f"{symbol}  ·  MULTI-TIMEFRAME", fontsize=17,
              fontweight="bold", color=TEXT, ha="left", va="top")
    ref_df = dfs[timeframes[0]]
    _draw_change_badge(fig, 0.975, 0.95, _calc_24h_change(ref_df), fontsize=14)
    fig.text(0.045, 0.02, f"BINANCE FUTURES  ·  {symbol}", fontsize=7,
              color=AXIS, ha="left", va="bottom")
    fig.text(0.98, 0.02,
              "Chart-based analysis for educational purposes only. NOT FINANCIAL ADVICE, DYOR.",
              fontsize=6.5, fontweight="bold", color=TEXT, ha="right", va="bottom")

    fig.savefig(out_path, facecolor=fig.get_facecolor(), dpi=dpi * 2)
    plt.close(fig)
    return out_path


def build_comparison_chart(
    series: dict,
    symbols: list,
    timeframe: str,
    cfg: dict,
    out_path: str,
    square: bool = False,
) -> str:
    """Comparison chart: performa % beberapa simbol dinormalisasi dari
    candle pertama di window yang sama, ditumpuk 1 axes -- untuk konten
    "mana yang lebih kuat" di Binance Square. square=True -> kanvas 1:1."""
    chart_cfg = cfg.get("chart", {})
    width_px = chart_cfg.get("width_px", 2800)
    dpi = 200
    fig_w = width_px / dpi
    fig_h = fig_w if square else fig_w * (9 / 20)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    ax.grid(True, linestyle="-", alpha=0.8, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(colors=AXIS, labelcolor=AXIS, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(0.8)

    max_len = max(len(df) for df in series.values())
    all_pct = []
    end_labels = []
    for i, sym in enumerate(symbols):
        df = series[sym]
        base = float(df["close"].iloc[0])
        pct = (df["close"] / base - 1.0) * 100.0
        color = COMPARE_PALETTE[i % len(COMPARE_PALETTE)]
        ax.plot(range(len(pct)), pct, color=color, linewidth=1.8, alpha=0.95,
                 zorder=5, label=sym)
        end_labels.append({"level": float(pct.iloc[-1]), "color": color,
                            "text": f"{sym}  {pct.iloc[-1]:+.1f}%"})
        all_pct.extend(pct.tolist())

    ax.axhline(0, color=SPINE, linewidth=0.9, linestyle="-", alpha=0.6, zorder=2)

    last_x = max_len - 1
    y_low, y_high = min(all_pct), max(all_pct)
    y_span = max(y_high - y_low, 1.0)
    pad = y_span * 0.15
    ax.set_ylim(y_low - pad, y_high + pad)
    ax.set_xlim(-0.6, last_x + 9.0)

    label_min_gap = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.06
    _place_level_labels(ax, end_labels, last_x + 3.0, label_min_gap)

    ax.set_ylabel("Change (%)", color=AXIS, fontsize=8.5, labelpad=5)

    ref_df = max(series.values(), key=len)
    if "open_time" in ref_df.columns and max_len:
        tick_count = min(6, max_len)
        ticks_idx = np.linspace(0, last_x, tick_count, dtype=int) if tick_count > 1 else [0]
        ax.set_xticks(ticks_idx)
        tick_labels = []
        for t in ticks_idx:
            if t < len(ref_df):
                ts = ref_df["open_time"].iloc[int(t)]
                tick_labels.append(ts.strftime("%d %b  %H:%M") if pd.notna(ts) else str(t))
            else:
                tick_labels.append(str(t))
        ax.set_xticklabels(tick_labels, fontsize=7.5, color=AXIS)

    header = pd.Timestamp.now(tz="UTC").strftime("Updated %d %b %H:%M UTC")
    fig.text(0.06, 0.92, f"COMPARISON  ·  {timeframe}  ·  {header}", fontsize=16,
              fontweight="bold", color=TEXT, ha="left", va="top")
    fig.text(0.06, 0.02, "BINANCE FUTURES  ·  Normalized % Change", fontsize=7,
              color=AXIS, ha="left", va="bottom")
    fig.text(0.94, 0.02,
              "Chart-based analysis for educational purposes only.\nNOT FINANCIAL ADVICE, DYOR.",
              fontsize=7.5, fontweight="bold", color=TEXT, ha="right", va="bottom",
              linespacing=1.6)

    fig.subplots_adjust(left=0.06, right=0.88, top=0.82, bottom=0.12)
    fig.savefig(out_path, facecolor=fig.get_facecolor(), dpi=dpi * 2)
    plt.close(fig)
    return out_path


async def _fetch_and_build(
    symbol: str,
    timeframe: str,
    cfg: dict,
    out_path: str,
    hide_indicators: bool = False,
    square: bool = False,
) -> str:
    async with BinanceFuturesClient() as client:
        n_show = get_candles_shown(timeframe, cfg)
        limit = max(400, n_show + STRUCTURE_CONTEXT + 250)
        kline = await client.get_klines(symbol, timeframe, limit=limit)
    signal = score_symbol(kline.df, symbol, cfg, timeframe=timeframe)
    result_path = build_chart(
        kline.df, symbol, timeframe, signal, cfg, out_path,
        preset="clean",
        hide_indicators=hide_indicators,
        square=square,
    )

    # Khusus jalur CLI/manual (mis. workflow "Chart Generator (manual)"):
    # kirim chart tunggal ini ke Telegram sebagai foto. Tidak dipanggil dari
    # scanner.run_scan, yang mengirim hasil batch lewat 1 file zip.
    try:
        await send_telegram_photo(result_path, format_signal_message(signal), cfg)
    except Exception as exc:  # jangan sampai gagal kirim Telegram menghentikan workflow
        print(f"Gagal kirim chart ke Telegram: {exc}")

    return result_path


async def _fetch_and_build_multi(symbol: str, timeframes: list, cfg: dict, out_path: str) -> str:
    dfs = {}
    async with BinanceFuturesClient() as client:
        for tf in timeframes:
            n_show = get_candles_shown(tf, cfg)
            limit = max(400, n_show + STRUCTURE_CONTEXT + 250)
            kline = await client.get_klines(symbol, tf, limit=limit)
            dfs[tf] = kline.df

    signals = {tf: score_symbol(dfs[tf], symbol, cfg, timeframe=tf) for tf in timeframes}
    result_path = build_multi_tf_card(dfs, symbol, timeframes, signals, cfg, out_path)

    caption = f"{symbol} multi-timeframe: " + " | ".join(
        f"{tf} {signals[tf].direction}" for tf in timeframes
    )
    try:
        await send_telegram_photo(result_path, caption, cfg)
    except Exception as exc:
        print(f"Gagal kirim chart ke Telegram: {exc}")

    return result_path


async def _fetch_and_build_compare(
    symbols: list, timeframe: str, cfg: dict, out_path: str, lookback: int,
    square: bool = False,
) -> str:
    # Setiap simbol di-fetch terpisah dan dibungkus try/except sendiri --
    # kalau satu simbol invalid/gagal (mis. tidak listing di Binance
    # Futures), simbol lain tetap lanjut diproses alih-alih seluruh mode
    # compare langsung crash. Simbol yang gagal dicatat lalu dilaporkan
    # jelas (satu per satu, bukan digabung jadi satu string membingungkan).
    series = {}
    failed = []
    async with BinanceFuturesClient() as client:
        for sym in symbols:
            try:
                kline = await client.get_klines(sym, timeframe, limit=lookback + 5)
            except Exception as exc:
                failed.append((sym, str(exc)))
                print(f"Lewati {sym}: gagal mengambil klines ({exc})")
                continue
            series[sym] = kline.df.tail(lookback).reset_index(drop=True)

    ok_symbols = [s for s in symbols if s in series]

    if len(ok_symbols) < 2:
        detail = "; ".join(f"{sym}: {err}" for sym, err in failed) or "tidak ada data"
        raise RuntimeError(
            f"Mode compare butuh minimal 2 simbol valid, cuma dapat {len(ok_symbols)}. "
            f"Detail kegagalan -> {detail}"
        )

    if failed:
        skipped = ", ".join(sym for sym, _ in failed)
        print(f"Compare tetap lanjut tanpa simbol yang gagal: {skipped}")

    result_path = build_comparison_chart(series, ok_symbols, timeframe, cfg, out_path, square=square)

    caption = "Comparison: " + " vs ".join(ok_symbols) + f" ({timeframe})"
    if failed:
        caption += "  |  Dilewati: " + ", ".join(sym for sym, _ in failed)
    try:
        await send_telegram_photo(result_path, caption, cfg)
    except Exception as exc:
        print(f"Gagal kirim chart ke Telegram: {exc}")

    return result_path


def _parse_list(raw: str) -> list:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default=None)
    parser.add_argument("--mode", choices=["clean", "multi", "compare"],
                          default="clean",
                          help="clean = chart lengkap (candle+EMA/Supertrend+volume+zona Demand/"
                               "Supply+BOS+label struktur+panah target) tanpa garis ENTRY/TP/SL "
                               "(default -- mode 'standard' lama sudah digabung ke sini). "
                               "multi = kartu multi-timeframe 1 simbol. "
                               "compare = perbandingan % beberapa simbol.")
    parser.add_argument("--timeframes", default="15m,1h,4h",
                          help="Dipisah koma, dipakai untuk --mode multi")
    parser.add_argument("--symbols", default="",
                          help="Dipisah koma, min. 2 simbol, dipakai untuk --mode compare")
    parser.add_argument("--compare-lookback", type=int, default=100,
                          help="Jumlah candle untuk --mode compare")
    parser.add_argument("--hide-indicators", action="store_true",
                          help="Sembunyikan EMA & Supertrend (biasa dipakai bareng --mode clean)")
    parser.add_argument("--ratio", choices=["wide", "square"], default="wide",
                          help="wide = rasio asli chart (default). "
                               "square = kanvas 1:1 untuk post feed Binance Square/IG. "
                               "Berlaku untuk mode standard/clean/compare.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.mode == "multi":
        timeframes = _parse_list(args.timeframes)
        if not timeframes:
            parser.error("--timeframes tidak boleh kosong untuk --mode multi")
        out_path = args.out or f"charts/{args.symbol.upper()}_multi.png"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        result_path = asyncio.run(
            _fetch_and_build_multi(args.symbol.upper(), timeframes, cfg, out_path)
        )

    elif args.mode == "compare":
        symbols = [s.upper() for s in _parse_list(args.symbols)]
        if len(symbols) < 2:
            parser.error("--symbols perlu minimal 2 simbol dipisah koma untuk --mode compare")
        out_path = args.out or f"charts/compare_{'_'.join(symbols)}_{args.timeframe}.png"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        result_path = asyncio.run(
            _fetch_and_build_compare(
                symbols, args.timeframe, cfg, out_path, args.compare_lookback,
                square=(args.ratio == "square"),
            )
        )

    else:
        out_path = args.out or f"charts/{args.symbol.upper()}_{args.timeframe}.png"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        result_path = asyncio.run(_fetch_and_build(
            args.symbol.upper(), args.timeframe, cfg, out_path,
            hide_indicators=args.hide_indicators,
            square=(args.ratio == "square"),
        ))

    print(f"Chart disimpan ke {result_path}")


if __name__ == "__main__":
    main()
