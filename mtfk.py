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

import chart

EMA20_COLOR = "#FFD54F"
EMA50_COLOR = "#29B6F6"
EMA200_COLOR = "#AB47BC"

TINT_LONG = (0.15, 0.65, 0.45, 0.05)
TINT_SHORT = (0.85, 0.25, 0.25, 0.05)
TINT_NONE = (0.5, 0.5, 0.5, 0.03)

RSI_COLOR = "#4DD0E1"
BB_COLOR = "#90A4AE"

# Khusus single_mtfk: lebar body candle dibuat lebih ramping drpd sebelumnya
# supaya candle tampak lebih "pipih" (tinggi lebih dominan drpd lebar) -
# (chart.CANDLE_WIDTH=0.8 tidak diubah krn dipakai bareng oleh chart.py & multi-panel).
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


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss_safe = avg_loss.replace(0, np.nan)
    rs = avg_gain / avg_loss_safe
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.fillna(50.0)
    return rsi


def _compute_bollinger(df: pd.DataFrame, n_show: int, period: int = 20, std_mult: float = 2.0) -> dict:
    """Bollinger Bands (formula sama dengan analyze.py::_bollinger) - dihitung
    ulang di sini (bukan lewat tf_info) supaya mtfk.py tetap tidak bergantung
    pada field baru di hasil analyze.py; input cuma df OHLC yang sudah ada."""
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
    """Nilai histogram MACD candle terakhir & candle sebelumnya (formula sama
    dengan analyze.py::_macd) - dipakai untuk badge arah+expanding/contracting."""
    close = df["close"]
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    hist = line - line.ewm(span=signal, adjust=False).mean()
    h = float(hist.iloc[-1]) if pd.notna(hist.iloc[-1]) else None
    hp = float(hist.iloc[-2]) if len(hist) > 1 and pd.notna(hist.iloc[-2]) else None
    return h, hp


def _volume_ratio_last(df: pd.DataFrame, lookback: int = 20):
    """Rasio volume candle terakhir vs rata-rata `lookback` candle - dipakai
    untuk badge volume spike."""
    if len(df) < lookback:
        return None
    vma = float(df["volume"].tail(lookback).mean())
    if vma <= 0:
        return None
    return float(df["volume"].iloc[-1]) / vma


def _compute_ema_set(df: pd.DataFrame, n_show: int, tf: str) -> dict:
    """Hitung EMA20/EMA50/EMA200(kondisional) satu kali di sini, dipakai
    bersama oleh _draw_analyze_indicators (multi-panel) dan
    _draw_indicators_single (single) - supaya logika "kapan EMA200 layak
    ditampilkan" konsisten di kedua jenis chart (sebelumnya logic ATR/near
    ini terduplikasi & bisa drift kalau salah satu diubah)."""
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
    """Sama seperti chart._draw_candles tapi body candle dibuat lebih lebar
    (SINGLE_CANDLE_WIDTH) - khusus dipakai single_mtfk. Tidak mengubah
    chart.CANDLE_WIDTH itu sendiri (dipakai bareng oleh chart.py & multi-panel)."""
    colors = []
    for i in range(len(df)):
        row = df.iloc[i]
        open_p, close_p = float(row["open"]), float(row["close"])
        high_p, low_p = float(row["high"]), float(row["low"])
        color = chart.UP if close_p >= open_p else chart.DOWN
        colors.append(color)

        ax.plot(
            [i, i], [low_p, high_p], color=color, linewidth=1.3,
            solid_capstyle="round", zorder=chart.Z_CANDLE_WICK,
        )
        body_bottom = min(open_p, close_p)
        # Tinggi minimum dinaikkan (0.012 -> 0.02) supaya candle doji/nyaris-
        # doji tetap kelihatan sbg garis tipis-tinggi, bukan hilang jadi titik,
        # selaras dgn body yang sekarang lebih ramping.
        body_height = max(abs(close_p - open_p), (high_p - low_p) * 0.02)
        ax.add_patch(Rectangle(
            (i - SINGLE_CANDLE_WIDTH / 2, body_bottom), SINGLE_CANDLE_WIDTH, body_height,
            facecolor=color, edgecolor=color, alpha=0.92, linewidth=0, zorder=chart.Z_CANDLE_BODY,
        ))
    return colors


def _draw_volume_bars_no_ma(ax, df: pd.DataFrame, colors: list) -> None:
    """Sama seperti chart._draw_volume tapi tanpa garis moving-average -
    dipakai khusus di single_mtfk karena baris itu diganti panel RSI."""
    for i in range(len(df)):
        ax.bar(
            i, float(df["volume"].iloc[i]), color=colors[i], alpha=0.48,
            width=SINGLE_CANDLE_WIDTH, linewidth=0, zorder=2,
        )


_LEGEND_ORDER = ("EMA 20", "EMA 50", "EMA 200")


def _ordered_legend_handles(ax):
    """Ambil handles/labels legend dari ax lalu urutkan cepat->lambat
    (EMA 20, 50, 200) tanpa peduli urutan gambar aslinya. Dipisah dari
    urutan gambar krn urutan gambar sengaja dibalik (200 dulu, 20 terakhir)
    supaya EMA20 selalu di atas secara visual - urutan legend tetap harus
    konvensional (cepat ke lambat) biar tidak membingungkan."""
    handles, labels = ax.get_legend_handles_labels()
    pairs = sorted(
        zip(handles, labels),
        key=lambda hl: _LEGEND_ORDER.index(hl[1]) if hl[1] in _LEGEND_ORDER else 99,
    )
    if not pairs:
        return [], []
    return zip(*pairs)


def _draw_colored_segments(
    fig,
    x0: float,
    y: float,
    parts: list[tuple[str, str]],
    fontsize: float,
    sep: str = "   ·   ",
) -> None:
    """Gambar satu baris teks (figure-fraction) yang tiap bagiannya punya
    warna sendiri sesuai makna informasinya, dipisah `sep` berwarna netral -
    dipakai utk baris ringkas ATR/MACD/Vol/MTF di bawah header supaya tiap
    info langsung kebaca artinya dari warnanya, bukan cuma dari teksnya.
    Posisi x tiap bagian dihitung dari lebar render bagian sebelumnya (perlu
    beberapa kali fig.canvas.draw() - baris ini pendek & cuma digambar
    sekali per chart jadi overhead-nya kecil)."""
    inv = fig.transFigure.inverted()
    x = x0
    for i, (text, color) in enumerate(parts):
        if i > 0:
            t = fig.text(x, y, sep, fontsize=fontsize, fontweight="bold", color=chart.AXIS, ha="left", va="top")
            fig.canvas.draw()
            x = inv.transform((t.get_window_extent(renderer=fig.canvas.get_renderer()).x1, 0))[0]
        t = fig.text(x, y, text, fontsize=fontsize, fontweight="bold", color=color, ha="left", va="top")
        fig.canvas.draw()
        x = inv.transform((t.get_window_extent(renderer=fig.canvas.get_renderer()).x1, 0))[0]


def _draw_rsi_panel(ax, rsi_values, tick_fs: float = 7.5) -> None:
    x = list(range(len(rsi_values)))
    ax.axhspan(70, 100, color=chart.DOWN, alpha=0.06, zorder=1)
    ax.axhspan(0, 30, color=chart.UP, alpha=0.06, zorder=1)
    ax.axhline(70, color=chart.AXIS, linestyle="--", linewidth=0.6, alpha=0.55, zorder=2)
    ax.axhline(30, color=chart.AXIS, linestyle="--", linewidth=0.6, alpha=0.55, zorder=2)
    ax.axhline(50, color=chart.AXIS, linestyle="-", linewidth=0.4, alpha=0.30, zorder=2)
    ax.fill_between(x, rsi_values, 50, color=RSI_COLOR, alpha=0.10, zorder=3, linewidth=0)
    ax.plot(
        x, rsi_values, color=RSI_COLOR, linewidth=1.3, alpha=0.95,
        zorder=4, solid_capstyle="round",
    )
    ax.set_ylim(0, 100)
    ax.set_yticks([30, 50, 70])
    ax.tick_params(labelsize=tick_fs)


def _draw_analyze_indicators(
    ax,
    df: pd.DataFrame,
    n_show: int,
    tf: str,
    tf_info: dict,
    label_fs: float = 6.5,
    show_legend: bool = False,
) -> list:
    p = float(df["close"].iloc[-1])
    emas = _compute_ema_set(df, n_show, tf)
    x = range(n_show)

    # Urutan gambar: EMA200 (paling lambat) dulu di paling bawah, EMA20
    # (paling cepat/paling relevan) digambar PALING TERAKHIR supaya selalu
    # terlihat di atas - sebelumnya EMA200 digambar terakhir dan malah
    # menutupi persilangan EMA20/50 saat ketiganya berdekatan.
    if emas["ema200"] is not None:
        ax.plot(
            x, emas["ema200"], color=EMA200_COLOR, linewidth=1.1, alpha=0.85,
            zorder=chart.Z_EMA, solid_capstyle="round",
            label="EMA 200" if show_legend else None,
        )
    ax.plot(
        x, emas["ema50"], color=EMA50_COLOR, linewidth=1.1, alpha=0.92,
        zorder=chart.Z_EMA + 1, solid_capstyle="round",
        label="EMA 50" if show_legend else None,
    )
    ax.plot(
        x, emas["ema20"], color=EMA20_COLOR, linewidth=1.15, alpha=0.95,
        zorder=chart.Z_EMA + 2, solid_capstyle="round",
        label="EMA 20" if show_legend else None,
    )

    structure = tf_info.get("structure") or {}
    support = structure.get("support")
    resistance = structure.get("resistance")
    # Nilai S/R dikumpulkan & dikembalikan supaya build_mtfk_chart bisa
    # mengikutsertakannya saat menghitung batas atas/bawah sumbu-y panel ini
    # (persis seperti di _draw_indicators_single). Sebelumnya y-range cuma
    # dihitung dari high/low candle, jadi kalau S/R terletak di luar rentang
    # candle yang tampil, garis+labelnya ikut ter-render di luar axes (label
    # teks defaultnya tidak clip) - itu penyebab "R ..." kadang menimpa judul
    # TF di panel yang lain.
    level_values = []
    if support:
        s_val = float(support)
        level_values.append(s_val)
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
        level_values.append(r_val)
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

    if show_legend:
        handles, labels = _ordered_legend_handles(ax)
        legend = ax.legend(
            handles, labels,
            loc="upper left", fontsize=label_fs + 1.5, framealpha=0.85,
            facecolor=chart.BG, edgecolor=chart.SPINE, labelcolor=chart.TEXT,
            borderpad=0.35, handlelength=1.4,
        )
        legend.get_frame().set_linewidth(0.6)

    return level_values


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


def _draw_indicators_single(
    ax,
    df: pd.DataFrame,
    n_show: int,
    tf: str,
    tf_info: dict,
    right_pad: float,
    square: bool = False,
) -> list:
    """Varian _draw_analyze_indicators khusus single_mtfk: linewidth & style
    disesuaikan skala chart penuh. S/R digambar di ruang kosong sebelah kanan
    (antara candle terakhir & tepi chart, lihat `right_pad` di
    build_single_mtfk_chart) - bukan lagi menumpuk di atas candle - supaya
    labelnya selalu bersih terbaca terpisah dari data harga; legend EMA/BB
    dipindah ke kiri (lihat build_single_mtfk_chart) supaya kedua elemen
    tidak saling menumpuk di satu sudut. Tidak dipakai oleh build_mtfk_chart
    (multi-panel)."""
    last_x = n_show - 1
    # marker (segitiga) nempel tepat di ujung ruang kosong (dekat candle
    # terakhir), teks nilai S/R nempel ke tepi kanan chart - keduanya pakai
    # koordinat data (bukan fraksi axes) krn xlim final sudah diset sebelum
    # fungsi ini dipanggil, jadi posisinya presisi & konsisten walau
    # right_pad berubah-ubah mengikuti jumlah candle.
    edge_margin = max(right_pad * 0.08, 0.35)
    label_x = last_x + right_pad - edge_margin
    marker_x = last_x + SINGLE_CANDLE_WIDTH / 2 + max(right_pad * 0.20, 0.5)

    emas = _compute_ema_set(df, n_show, tf)
    x = range(n_show)
    ema_values = list(emas["ema20"]) + list(emas["ema50"])

    # Sama seperti _draw_analyze_indicators: EMA200 (lambat) di bawah, EMA20
    # (cepat) digambar terakhir supaya selalu di atas EMA50/EMA200.
    if emas["ema200"] is not None:
        ax.plot(
            x, emas["ema200"], color=EMA200_COLOR, linewidth=1.4, alpha=0.92,
            zorder=chart.Z_EMA, solid_capstyle="round", label="EMA 200",
        )
        ema_values.extend(list(emas["ema200"]))
    ax.plot(
        x, emas["ema50"], color=EMA50_COLOR, linewidth=1.4, alpha=0.95,
        zorder=chart.Z_EMA + 1, solid_capstyle="round", label="EMA 50",
    )
    ax.plot(
        x, emas["ema20"], color=EMA20_COLOR, linewidth=1.4, alpha=0.95,
        zorder=chart.Z_EMA + 2, solid_capstyle="round", label="EMA 20",
    )

    bb = _compute_bollinger(df, n_show)
    bb_upper, bb_lower = bb["upper"], bb["lower"]
    if not (np.all(pd.isna(bb_upper)) or np.all(pd.isna(bb_lower))):
        ax.plot(
            x, bb_upper, color=BB_COLOR, linewidth=1.0, alpha=0.55,
            zorder=chart.Z_EMA - 1, solid_capstyle="round", label="BB(20,2)",
        )
        ax.plot(
            x, bb_lower, color=BB_COLOR, linewidth=1.0, alpha=0.55,
            zorder=chart.Z_EMA - 1, solid_capstyle="round",
        )
        ax.fill_between(
            x, bb_lower, bb_upper, color=BB_COLOR, alpha=0.05,
            zorder=chart.Z_EMA - 2, linewidth=0,
        )
        ema_values.extend([float(v) for v in bb_upper if pd.notna(v)])
        ema_values.extend([float(v) for v in bb_lower if pd.notna(v)])

    structure = tf_info.get("structure") or {}
    support = structure.get("support")
    resistance = structure.get("resistance")
    label_fs = 10.5 if square else 8.5

    level_values = []
    if support:
        s_val = float(support)
        level_values.append(s_val)
        marker_size = 7.5 if square else 6.0
        ax.axhline(
            s_val, color=chart.UP, linestyle="--",
            linewidth=1.4, alpha=0.85, zorder=chart.Z_LEVEL_LINE,
        )
        support_trans = offset_copy(
            ax.transData, fig=ax.figure, x=0, y=marker_size / 2 + 0.6, units="points",
        )
        ax.plot(
            [marker_x], [s_val], marker="^", markersize=marker_size,
            color=chart.UP, zorder=chart.Z_LEVEL_LABEL, clip_on=False,
            transform=support_trans,
        )
        ax.text(
            label_x, s_val,
            f"SUPPORT  {chart.format_price(s_val, chart.decimals_from_price(s_val))}",
            color=chart.UP,
            fontsize=label_fs, fontweight="bold", ha="right", va="bottom",
            zorder=chart.Z_LEVEL_LABEL,
            bbox=dict(boxstyle="round,pad=0.22", facecolor=chart.BG, edgecolor="none", alpha=0.80),
        )
    if resistance:
        r_val = float(resistance)
        level_values.append(r_val)
        marker_size = 7.5 if square else 6.0
        ax.axhline(
            r_val, color=chart.DOWN, linestyle="--",
            linewidth=1.4, alpha=0.85, zorder=chart.Z_LEVEL_LINE,
        )
        resistance_trans = offset_copy(
            ax.transData, fig=ax.figure, x=0, y=-(marker_size / 2 + 0.6), units="points",
        )
        ax.plot(
            [marker_x], [r_val], marker="v", markersize=marker_size,
            color=chart.DOWN, zorder=chart.Z_LEVEL_LABEL, clip_on=False,
            transform=resistance_trans,
        )
        ax.text(
            label_x, r_val,
            f"RESISTANCE  {chart.format_price(r_val, chart.decimals_from_price(r_val))}",
            color=chart.DOWN,
            fontsize=label_fs, fontweight="bold", ha="right", va="top",
            zorder=chart.Z_LEVEL_LABEL,
            bbox=dict(boxstyle="round,pad=0.22", facecolor=chart.BG, edgecolor="none", alpha=0.80),
        )

    direction = tf_info.get("direction", "NONE")
    last_c = float(df["close"].iloc[-1])
    if direction == "LONG":
        ax.scatter(
            n_show - 1, last_c, marker="o", s=(34 if square else 26),
            color=chart.UP, edgecolors=chart.TEXT, linewidths=0.6,
            zorder=6, alpha=0.95,
        )
    elif direction == "SHORT":
        ax.scatter(
            n_show - 1, last_c, marker="o", s=(34 if square else 26),
            color=chart.DOWN, edgecolors=chart.TEXT, linewidths=0.6,
            zorder=6, alpha=0.95,
        )

    return level_values + ema_values


def _draw_trigger_highlight(ax, plot_df: pd.DataFrame, tf_info: dict, y_span: float) -> None:
    """Highlight candle pemicu setup. Di analyze.py::_detect_setup, hanya
    BREAKOUT/BREAKDOWN/REJECTION yang murni ditentukan dari body/wick candle
    TERAKHIR - PULLBACK/RETEST & CONTINUATION dipicu oleh posisi harga
    terhadap level/trend, bukan 1 candle spesifik, jadi tidak dihighlight."""
    setup = tf_info.get("setup") or {}
    setup_type = setup.get("type", "NONE")
    if setup_type not in ("BREAKOUT", "BREAKDOWN", "REJECTION"):
        return
    if len(plot_df) == 0:
        return

    direction = tf_info.get("direction", "NONE")
    color = chart.UP if direction == "LONG" else chart.DOWN if direction == "SHORT" else chart.AXIS

    idx = len(plot_df) - 1
    last = plot_df.iloc[-1]
    high, low = float(last["high"]), float(last["low"])

    # Cuma kotak putus-putus di sekeliling candle - tanpa label teks
    # mengambang, supaya tidak berpotensi tabrakan dengan legend EMA/BB di
    # sudut chart manapun. Jenis setup (BREAKOUT/BREAKDOWN/REJECTION) sudah
    # tertulis di header, kotak ini cukup menunjuk candle mana pemicunya.
    half_w = SINGLE_CANDLE_WIDTH / 2 + 0.30
    pad_y = max(y_span * 0.012, (high - low) * 0.10)
    ax.add_patch(Rectangle(
        (idx - half_w, low - pad_y), half_w * 2, (high - low) + pad_y * 2,
        facecolor="none", edgecolor=color, linewidth=1.6, linestyle="--",
        alpha=0.9, zorder=chart.Z_LEVEL_LABEL + 1,
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
    """Chart single-timeframe versi mtfk (dipakai analyze.py saat cuma 1 tf).
    Layout/rasio/jumlah candle/warna dibuat persis dengan chart.build_chart,
    hanya menambahkan panel RSI di bawah volume (volume tanpa garis MA20).
    Jumlah candle selalu memakai batas maksimum dari chart.get_candles_shown
    (tidak dipotong dinamis) supaya candle memenuhi chart dari batas kiri
    sampai batas kanan."""
    cfg = cfg or {}
    has_error = "error" in tf_info

    n_show = chart.get_candles_shown(timeframe, cfg)
    plot_df = df.tail(n_show).reset_index(drop=True)

    chart_cfg = cfg.get("chart", {})
    width_px = chart_cfg.get("width_px", 2800)
    # Default 9/20 (0.45) semula dipakai utk chart 2-panel (price+volume) di
    # chart.py. single_mtfk punya panel ke-3 (RSI), jadi kalau cfg tidak
    # menimpa height_ratio secara eksplisit, dipakai default sedikit lebih
    # tinggi (0.54) supaya 3 panel + header/footer tidak terlalu gepeng saat
    # ditampilkan kecil di preview chat HP. Kalau di config yaml kalian sudah
    # ada key chart.height_ratio eksplisit, itu tetap yang dipakai.
    height_ratio = 1.0 if square else chart_cfg.get("height_ratio", 0.54)
    dpi = 200
    output_scale = 2
    fig_w = width_px / dpi
    fig_h = (width_px * height_ratio) / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(chart.BG)

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
        ax.set_facecolor(chart.PANEL)
        ax.grid(True, linestyle="-", alpha=0.55, color=chart.GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(colors=chart.AXIS, labelcolor=chart.AXIS, labelsize=7.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(chart.SPINE)
            ax.spines[side].set_linewidth(0.8)

    ax_price.tick_params(labelbottom=False)
    ax_vol.tick_params(labelbottom=False)

    direction = "NONE" if has_error else tf_info.get("direction", "NONE")

    colors = _draw_candles_single(ax_price, plot_df)
    last_x = len(plot_df) - 1

    # Sisi kiri tetap mepet (histori lama tidak penting dilihat penuh), tapi
    # sisi kanan dilebarkan jadi zona kosong khusus label SUPPORT/RESISTANCE
    # - dihitung sbg fraksi TETAP dari lebar axes (label_zone_frac), bukan
    # cuma kelipatan lebar candle, krn lebar axes dlm pixel itu konstan
    # (width_px sama berapa pun n_show-nya) sedangkan lebar candle relatif
    # thd n_show berubah-ubah. Dgn fraksi tetap ini, zona label selalu
    # cukup lega dari candle manapun n_show-nya - konsekuensinya candle jadi
    # lebih ramping/pipih drpd sebelumnya, itu memang trade-off yg diambil
    # supaya labelnya tidak lagi ketiban candle terakhir.
    label_zone_frac = 0.22 if square else 0.19
    right_pad = label_zone_frac * (last_x + 0.6) / (1 - label_zone_frac)
    right_pad = max(right_pad, SINGLE_CANDLE_WIDTH * 8)
    ax_price.set_xlim(-0.6, last_x + right_pad)
    ax_vol.set_xlim(-0.6, last_x + right_pad)
    ax_rsi.set_xlim(-0.6, last_x + right_pad)

    range_values = []
    if not has_error:
        range_values = _draw_indicators_single(
            ax_price, df, n_show, timeframe, tf_info, right_pad, square=square,
        )

    all_vals = [float(plot_df["low"].min()), float(plot_df["high"].max())] + range_values
    y_low = min(all_vals)
    y_high = max(all_vals)
    y_span = max(y_high - y_low, abs(y_low) * 0.01 if y_low != 0 else 0.01)
    y_padding = y_span * 0.18
    ax_price.set_ylim(y_low - y_padding, y_high + y_padding)

    if not has_error and len(plot_df) >= 6:
        swing_high, swing_low = chart._find_swings(plot_df, 2, 2)
        labeled_points = chart._label_structure(plot_df, swing_high, swing_low)
        if labeled_points:
            # seperlunya saja - hanya beberapa titik swing paling relevan
            # (terbaru) supaya tidak menuh-menuhin chart.
            labeled_points = labeled_points[-6:]
            chart._draw_structure_labels(
                ax_price, labeled_points, offset=0, plot_len=len(plot_df),
                y_span=y_span, square=square,
            )

    if not has_error:
        _draw_trigger_highlight(ax_price, plot_df, tf_info, y_span)

    _draw_volume_bars_no_ma(ax_vol, plot_df, colors)
    ax_vol.yaxis.set_major_formatter(FuncFormatter(_fmt_volume))
    ax_vol.yaxis.get_offset_text().set_visible(False)
    ax_vol.set_ylabel("Vol", color=chart.AXIS, fontsize=8, labelpad=5)
    ax_price.set_ylabel("Price", color=chart.AXIS, fontsize=8.5, labelpad=5)

    rsi_period = cfg.get("indicators", {}).get("rsi", {}).get("period", 14)
    rsi_tail = _compute_rsi(df["close"], rsi_period).tail(n_show).to_numpy()
    _draw_rsi_panel(ax_rsi, rsi_tail)
    ax_rsi.set_ylabel("RSI", color=chart.AXIS, fontsize=8, labelpad=5)

    time_ticks = _time_axis_labels(df, n_show, timeframe)
    if time_ticks:
        positions, labels = time_ticks
        ax_rsi.set_xticks(positions)
        ax_rsi.set_xticklabels(labels, fontsize=7.5, color=chart.AXIS)

    if not has_error:
        # Legend ditaruh di kiri-atas (bukan kanan) supaya tidak menumpuk
        # dengan label SUPPORT/RESISTANCE yang sekarang ada di kanan.
        handles, labels = _ordered_legend_handles(ax_price)
        legend = ax_price.legend(
            handles, labels,
            loc="upper left", fontsize=(13.5 if square else 7.5), framealpha=0.95,
            facecolor=chart.BG, edgecolor=chart.SPINE, labelcolor=chart.TEXT, borderpad=0.4,
        )
        legend.get_frame().set_linewidth(0.7)

    setup_info = tf_info.get("setup")
    setup_info = setup_info if isinstance(setup_info, dict) else {}
    setup_type = "NONE" if has_error else setup_info.get("type", "NONE")
    setup_txt = f"  ·  {setup_type}" if setup_type and setup_type != "NONE" else ""
    header_extra = pd.Timestamp.now(tz="UTC").strftime("Updated %d %b %H:%M UTC")
    header_title = f"{symbol}  ·  {timeframe}  ·  {direction}{setup_txt}"

    # Header sedikit dikecilkan lagi drpd sebelumnya (18/22 -> 16/20) supaya
    # tidak dominan, dan jarak ke baris indikator di bawahnya dirapikan jadi
    # gap yang konsisten & lega (bukan mepet) - badge ikut diskalakan turun
    # supaya proporsinya tetap seimbang dgn teks header.
    header_fs = 20.0 if square else 16.0
    badge_fs = 17.5 if square else 13.5
    segment_fs = 12.0 if square else 9.0
    segment_y = 0.913 if square else 0.917

    fig.text(
        0.07, 0.965, header_title,
        fontsize=header_fs, fontweight="bold", color=chart.TEXT,
        ha="left", va="top",
    )
    chart._draw_change_badge(fig, 0.96, 0.965, chart._calc_24h_change(df), fontsize=badge_fs)

    if not has_error:
        atr_pct = (tf_info.get("direction_analysis") or {}).get("atr_pct")
        hist, hist_prev = _macd_histogram_last(df)
        vol_ratio = _volume_ratio_last(df)
        mtf_agree = tf_info.get("mtf_agree_tfs") or []

        # Tiap segmen diwarnai sesuai fungsinya sendiri (dulu semua satu
        # warna netral) - ATR murni info volatilitas jadi netral, MACD & MTF
        # agree ikut warna arah (UP/DOWN) krn keduanya sinyal condong ke satu
        # sisi, Vol disorot kuning cuma kalau lonjakannya signifikan (>=1.5x
        # rata-rata), selain itu tetap netral.
        segments: list[tuple[str, str]] = []
        if atr_pct is not None:
            segments.append((f"ATR {atr_pct:.2f}%", chart.AXIS))
        if hist is not None:
            arrow = "▲" if hist > 0 else "▼" if hist < 0 else "→"
            state = ""
            if hist_prev is not None:
                if abs(hist) > abs(hist_prev):
                    state = "Expanding"
                elif abs(hist) < abs(hist_prev):
                    state = "Contracting"
            macd_color = chart.UP if hist > 0 else chart.DOWN if hist < 0 else chart.AXIS
            segments.append((f"MACD {arrow}" + (f" {state}" if state else ""), macd_color))
        if vol_ratio is not None:
            vol_color = EMA20_COLOR if vol_ratio >= 1.5 else chart.AXIS
            segments.append((f"Vol {vol_ratio:.1f}x avg", vol_color))
        if mtf_agree:
            agree_color = chart.UP if direction == "LONG" else chart.DOWN if direction == "SHORT" else chart.AXIS
            segments.append((f"MTF agree: {', '.join(mtf_agree)}", agree_color))

        if segments:
            _draw_colored_segments(fig, 0.07, segment_y, segments, fontsize=segment_fs)

    if square:
        footer_left = f"BINANCE FUTURES  ·  {symbol}  ·  {timeframe}\n{header_extra}"
        fig.text(
            0.07, 0.028, footer_left, fontsize=14.0, color=chart.AXIS,
            ha="left", va="bottom", linespacing=1.6,
        )
    else:
        fig.text(
            0.07, 0.02, f"BINANCE FUTURES  ·  {symbol}  ·  {timeframe}  ·  {header_extra}",
            fontsize=8.5, color=chart.AXIS, ha="left", va="bottom",
        )

    fig.text(
        0.96, 0.013,
        "Chart-based analysis.\nNOT FINANCIAL ADVICE, DYOR.",
        fontsize=(13.0 if square else 7.5), fontweight="bold", color=chart.TEXT,
        ha="right", va="bottom", linespacing=1.7,
    )

    if has_error:
        ax_price.text(
            0.5, 0.5, "NO DATA", transform=ax_price.transAxes, color=chart.DOWN,
            fontsize=(22.0 if square else 16), fontweight="bold", ha="center", va="center",
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, facecolor=fig.get_facecolor(), dpi=dpi * output_scale)
    plt.close(fig)
    return out_path


# Alias: analyze.py mengimpor & memanggil fungsi ini dengan nama `single_mtfk`
# (keyword args df/symbol/timeframe/tf_info/out_path/cfg persis sama).
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
        level_values = []
        if not has_error:
            # Legend EMA cuma ditaruh SEKALI di panel pertama (warnanya sama
            # di semua panel) - drpd diulang di tiap panel & bikin sesak.
            level_values = _draw_analyze_indicators(
                ax_p, df, n_show, tf, tf_info, tick_fs, show_legend=(idx == 0),
            ) or []

        last_x = len(plot_df) - 1
        # level_values (support/resistance) ikut disertakan di perhitungan
        # batas y - sebelumnya y-range cuma dari high/low candle yg tampil,
        # jadi kalau S/R ada di luar rentang itu, garis+labelnya ter-render
        # di luar axes & bisa menimpa judul TF panel (bug yg dilaporkan).
        # Dgn level_values ikut masuk, S/R (dan labelnya) dijamin selalu
        # berada di dalam area chart, walau efeknya candle bisa tampak
        # sedikit lebih kecil/rapat kalau S/R jauh dari harga saat ini.
        all_vals = [float(plot_df["low"].min()), float(plot_df["high"].max())] + level_values
        y_low = min(all_vals)
        y_high = max(all_vals)
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
