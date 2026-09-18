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

# --- Disalin dari chart.py -------------------------------------------------
# mtfk.py sebelumnya `import chart` hanya untuk memakai 20 atribut di bawah
# ini (8 warna, 5 konstanta Z-order, 8 fungsi helper - lihat daftar di setiap
# kelompok). Supaya mtfk.py bisa dipakai/diuji tanpa chart.py sama sekali,
# semuanya disalin verbatim (nilai & logika identik dgn chart.py saat ini) -
# KONSEKUENSINYA: kalau salah satu definisi ini diubah di chart.py, mtfk.py
# TIDAK ikut berubah otomatis dan perlu disinkronkan manual di sini juga.

# Warna
BG = "#121417"
PANEL = "#121417"
GRID = "#3A3A3A"
TEXT = "#F2F2F2"
AXIS = "#B8B8B8"
SPINE = "#4A4A4A"
UP = "#26A69A"
DOWN = "#EF5350"

# Z-order
Z_CANDLE_WICK = 2.0
Z_CANDLE_BODY = 2.1
Z_EMA = 4.0
Z_STRUCT_LABEL = 5.5   # dipakai internal oleh _draw_structure_labels di bawah
Z_LEVEL_LINE = 6.0
Z_LEVEL_LABEL = 6.5

STRUCT_TEXT = "#BDBDBD"  # dipakai internal oleh _draw_structure_labels di bawah

MAX_CANDLES_BY_TF = {
    "15m": 80,   # ~20 jam
    "1h": 70,    # ~2.9 hari
    "4h": 60,    # ~10 hari
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
    fontsize = 12.5 if square else 6.0
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
# --- Akhir bagian yang disalin dari chart.py --------------------------------

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
        # Guarantee the final candle is represented, but don't just tack it
        # on: if the last regular tick is already within half a step of it,
        # move that tick to the true last index instead of adding a second
        # one right next to it. Appending unconditionally used to put two
        # ticks only 1-2 candles apart at the right edge, which rendered as
        # visibly cramped or even literally duplicated labels (e.g. "21:15"
        # immediately followed by "21:30", or "15 Sep" printed twice).
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
    """Formula & penanganan edge-case (avg_gain/avg_loss = 0) disalin PERSIS
    dari analyze.py::_rsi - dulu di sini pakai rumus fallback yang sedikit
    beda (mis. kasus avg_gain==0 & avg_loss==0 sempat jatuh ke 100, padahal
    di analyze.py itu 50), jadi kurva RSI di chart bisa tidak match dengan
    nilai RSI & catatan bias yang dipakai analyze.py untuk skoring/setup.
    Period JUGA tidak lagi dibaca dari cfg (lihat pemanggil di bawah) karena
    analyze.py::_rsi() selalu pakai 14 tanpa override cfg."""
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
    # fillna cuma jaring pengaman utk baris warmup paling awal (sebelum
    # min_periods tercapai) supaya tidak ada NaN yang masuk ke array plot -
    # analyze.py sendiri membiarkan baris warmup itu NaN krn tidak pernah
    # dipakai (cuma nilai RSI candle terakhir yang dibaca), tapi di sini
    # seluruh window n_show ikut digambar jadi perlu aman dari NaN.
    return rsi.fillna(50.0)


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
    """Hitung EMA20/EMA50/EMA200(kondisional) satu kali di sini, dipakai oleh
    _draw_indicators_single (dipanggil dari build_single_mtfk_chart MAUPUN
    build_mtfk_chart) - supaya logika "kapan EMA200 layak ditampilkan"
    konsisten di kedua jenis chart."""
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
        color = UP if close_p >= open_p else DOWN
        colors.append(color)

        ax.plot(
            [i, i], [low_p, high_p], color=color, linewidth=1.3,
            solid_capstyle="round", zorder=Z_CANDLE_WICK,
        )
        body_bottom = min(open_p, close_p)
        # Tinggi minimum dinaikkan (0.012 -> 0.02) supaya candle doji/nyaris-
        # doji tetap kelihatan sbg garis tipis-tinggi, bukan hilang jadi titik,
        # selaras dgn body yang sekarang lebih ramping.
        body_height = max(abs(close_p - open_p), (high_p - low_p) * 0.02)
        ax.add_patch(Rectangle(
            (i - SINGLE_CANDLE_WIDTH / 2, body_bottom), SINGLE_CANDLE_WIDTH, body_height,
            facecolor=color, edgecolor=color, alpha=0.92, linewidth=0, zorder=Z_CANDLE_BODY,
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
    """Tentukan titik anchor vertikal (dipakai bareng oleh marker & teks)
    untuk label SUPPORT/RESISTANCE.

    BUG yang diperbaiki fungsi ini: sebelumnya label SUPPORT & RESISTANCE
    selalu digambar tepat di harga aslinya (s_val/r_val) tanpa cek jarak -
    begitu kedua level berdekatan (mis. kasus KAITOUSDT 15m/1h/4h: S/R cuma
    beda ~0.15-0.3% dari harga saat market lagi konsolidasi sempit), kedua
    kotak label saling menimpa dan jadi tidak terbaca sama sekali (kotak
    yang digambar belakangan - RESISTANCE - menutupi penuh kotak SUPPORT).

    Kalau kedua level cukup berjauhan, anchor = harga aslinya masing-masing
    (tidak ada perubahan sama sekali). Kalau berdekatan, keduanya digeser
    simetris menjauhi titik tengah supaya kotak label tidak lagi saling
    menimpa. Garis putus-putus (axhline) TETAP digambar di harga asli -
    hanya anchor marker+teks yang digeser, jadi masih jelas warna mana
    mewakili level mana meski posisi labelnya sedikit "meleset" dari
    garisnya sendiri saat market sedang sangat sempit.

    `price_span_hint` = perkiraan rentang high-low candle yang sedang
    ditampilkan (dipakai sbg basis heuristik jarak minimum antar label,
    krn y-limit final axes belum ditentukan saat fungsi ini dipanggil)."""
    if s_val is None or r_val is None:
        return s_val, r_val
    span = max(price_span_hint, 1e-9)
    # Fraksi dipilih & divalidasi scr visual (bukan cuma dihitung dari tinggi
    # font) supaya cukup lega utk 2 baris label bold 8.5-10.5pt tanpa
    # menggeser anchor jauh-jauh saat kasusnya cuma sedikit berdekatan.
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
    """Gambar EMA/BB/S-R/marker arah untuk satu panel harga, skala penuh.
    S/R digambar di ruang kosong sebelah kanan (antara candle terakhir &
    tepi chart, lihat `right_pad` di pemanggil) - bukan menumpuk di atas
    candle - supaya labelnya selalu bersih terbaca terpisah dari data harga;
    legend EMA/BB ditaruh di kiri supaya tidak menumpuk dgn label S/R di
    kanan. Dipakai oleh KEDUA build_single_mtfk_chart (1 panel) dan
    build_mtfk_chart (tiap blok TF di chart multi-timeframe)."""
    # BUG yang diperbaiki: n_show di sini dulu dipakai APA ADANYA (nilai
    # nominal dari get_candles_shown, mis. 60 utk "1h") padahal candle yang
    # BENAR-BENAR digambar pemanggil cuma sebanyak len(plot_df) =
    # len(df.tail(n_show)) - kalau df yang di-fetch kebetulan lebih pendek
    # dari n_show (mis. simbol baru listing dgn histori < 60 kline, atau tf
    # custom yang tidak ada di MAX_CANDLES_BY_TF sehingga jatuh ke default
    # 120), maka x=range(n_show) (n_show titik) dipasangkan dgn array
    # EMA/BB hasil `.tail(n_show)` yang panjangnya cuma len(df) (< n_show)
    # -> matplotlib ValueError ("x and y must have same first dimension")
    # dan chart gagal dibuat. `last_x`/marker S-R/titik arah LONG-SHORT yang
    # dihitung dari n_show mentah juga jadi meleset dari posisi candle
    # terakhir yang sebenarnya. Clamp sekali di sini menyelaraskan SEMUA
    # pemakaian n_show di bawah dgn jumlah candle yang benar-benar tampil.
    n_show = min(n_show, len(df))
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

    # EMA200 (lambat) digambar dulu di paling bawah, EMA20 (cepat) digambar
    # PALING TERAKHIR supaya selalu terlihat di atas EMA50/EMA200 saat
    # ketiganya berdekatan.
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
    label_fs = 10.5 if square else 8.5
    marker_size = 7.5 if square else 6.0

    s_val = float(support) if support else None
    r_val = float(resistance) if resistance else None
    # Lihat docstring _resolve_sr_label_anchors: anchor cuma beda dari
    # s_val/r_val saat kedua level berdekatan, supaya kotak label tidak
    # saling menimpa. price_span_hint dari rentang high-low candle yang
    # tampil (proxy y-span, krn y-limit final axes belum diset di sini).
    price_span_hint = float(df["high"].tail(n_show).max() - df["low"].tail(n_show).min())
    s_anchor, r_anchor = _resolve_sr_label_anchors(s_val, r_val, price_span_hint, square=square)

    level_values = []
    if s_val is not None:
        # s_val (harga asli, utk axhline) DAN s_anchor (posisi label yg
        # sudah dipisah) dua-duanya masuk ke level_values supaya y-limit
        # akhir yang dihitung pemanggil selalu cukup lega menampung label,
        # walau anchornya digeser sedikit dari harga aslinya.
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
    color = UP if direction == "LONG" else DOWN if direction == "SHORT" else AXIS

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
    """Chart single-timeframe versi mtfk (dipakai analyze.py saat cuma 1 tf).
    Layout/rasio/jumlah candle/warna dibuat persis dengan chart.build_chart,
    hanya menambahkan panel RSI di bawah volume (volume tanpa garis MA20).
    Jumlah candle selalu memakai batas maksimum dari get_candles_shown
    (tidak dipotong dinamis) supaya candle memenuhi chart dari batas kiri
    sampai batas kanan."""
    cfg = cfg or {}
    has_error = "error" in tf_info

    n_show = get_candles_shown(timeframe, cfg)
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
        swing_high, swing_low = _find_swings(plot_df, 2, 2)
        labeled_points = _label_structure(plot_df, swing_high, swing_low)
        if labeled_points:
            # seperlunya saja - hanya beberapa titik swing paling relevan
            # (terbaru) supaya tidak menuh-menuhin chart.
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
        # Legend ditaruh di kiri-atas (bukan kanan) supaya tidak menumpuk
        # dengan label SUPPORT/RESISTANCE yang sekarang ada di kanan.
        handles, labels = _ordered_legend_handles(ax_price)
        legend = ax_price.legend(
            handles, labels,
            loc="upper left", fontsize=(13.5 if square else 7.5), framealpha=0.95,
            facecolor=BG, edgecolor=SPINE, labelcolor=TEXT, borderpad=0.4,
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
        fontsize=header_fs, fontweight="bold", color=TEXT,
        ha="left", va="top",
    )
    _draw_change_badge(fig, 0.96, 0.965, _calc_24h_change(df), fontsize=badge_fs)

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
    """Chart multi-timeframe. Tiap TF digambar sbg satu blok yang gaya &
    proporsinya PERSIS mengikuti build_single_mtfk_chart (lebar candle,
    panel RSI, zona label SUPPORT/RESISTANCE di kanan, Bollinger Band, EMA,
    label struktur swing, trigger highlight, legend) - blok-blok itu ditumpuk
    vertikal, dibungkus satu judul+badge di paling atas
    ("{symbol} · MULTI-TIMEFRAME") dan satu footer di paling bawah (mirip
    footer single, tapi menyebut semua TF).

    Sengaja pakai ULANG fungsi-fungsi gambar yang sama dgn single_mtfk
    (_draw_candles_single, _draw_indicators_single, _draw_rsi_panel, dst)
    dan bukan menulis versi baru, supaya kedua jenis chart ini tidak bisa
    drift satu sama lain - dan supaya perubahan di sini TIDAK menyentuh
    build_single_mtfk_chart sama sekali (fungsi itu tetap apa adanya)."""
    cfg = cfg or {}
    valid_tfs = [tf for tf in timeframes if tf in dfs and tf in per_tf]
    n_panels = len(valid_tfs)
    if n_panels == 0:
        raise ValueError("Tidak ada data timeframe yang valid untuk membuat chart.")

    chart_cfg = cfg.get("chart", {})
    width_px = chart_cfg.get("width_px", 2800)
    # Proporsi "badan" (price+vol+rsi) tiap blok TF SELALU pakai height_ratio
    # non-square (default 0.54) walau square=True - kalau ikut jadi 1.0 spt
    # di single, tinggi total gambar utk >1 TF bisa meledak (mis. 3 TF
    # square = puluhan inci tinggi, tidak praktis dilihat di HP). Di sini
    # `square` cuma memperbesar font/label (lihat header_fs, segment_fs,
    # dst di bawah, serta diteruskan ke _draw_indicators_single dkk persis
    # spt cara single memakainya).
    height_ratio = chart_cfg.get("height_ratio", 0.54)
    dpi = 200
    output_scale = 2
    fig_w = width_px / dpi

    # Rasio header:badan tiap blok TF disamakan dgn rasio top/bottom di
    # build_single_mtfk_chart (top=0.87 -> badan 0.76 dari total, header
    # 0.13 dari total) - supaya ukuran font & lebar candle blok ini identik
    # dgn chart single, hanya tanpa footernya sendiri (footer dipakai satu
    # kali saja di paling bawah gambar gabungan ini, bukan per-blok).
    single_fig_h = (width_px * height_ratio) / dpi
    header_frac, body_frac = 0.13, 0.76
    block_h_in = (header_frac + body_frac) * single_fig_h

    top_margin_in = 0.55    # ruang judul "{symbol} · MULTI-TIMEFRAME" + badge
    # 0.85in supaya label sumbu-waktu blok TERAKHIR (yg nempel di bawah
    # panel RSI paling bawah) tidak numpuk sama teks footer di bawahnya -
    # nilai ini menyamai proporsi ruang footer di build_single_mtfk_chart
    # (bottom=0.11 dari fig_h single ≈ 0.83in), krn di sana ruang yg sama
    # juga menampung tick label + footer + disclaimer.
    bottom_margin_in = 0.85
    gap_in = 0.20            # jarak antar blok TF

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
        if not has_error:
            range_values = _draw_indicators_single(
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
            handles, labels = _ordered_legend_handles(ax_price)
            legend = ax_price.legend(
                handles, labels,
                loc="upper left", fontsize=legend_fs, framealpha=0.95,
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
