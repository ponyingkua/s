from __future__ import annotations

import argparse
import asyncio
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from scanner import (
    BinanceFuturesClient,
    Kline,
    get_market_regime,
    get_min_score_to_trigger,
    is_in_cooldown,
    is_score_excluded,
    load_config,
    load_state,
    mtf_bonus_eligible,
    passes_regime_filter,
    passes_risk_filter_detailed,
    drop_unclosed_candle,
    score_symbol,
)

OUT_DIR = "analysis_output"
CHART_SUBDIR = "charts"

REASON_LABELS = {
    "tidak_ada_arah": "tidak ada arah yang jelas (indikator belum warm-up / skor LONG-SHORT dasi)",
    "skor_di_bawah_ambang": "skor di bawah ambang trigger",
    "masuk_excluded_band": "skor masuk rentang yang dikecualikan (excluded_score_bands)",
    "cooldown_aktif": "simbol/arah ini masih dalam masa cooldown dari sinyal sebelumnya",
}


def normalize_symbol(raw: str, quote_asset: str) -> str:
    s = raw.strip().upper()
    if not s:
        raise ValueError("Simbol tidak boleh kosong")
    if s.endswith(quote_asset):
        return s
    return f"{s}{quote_asset}"


def _reject_label(reason: str | None, regime: str) -> str:
    if reason is None:
        return ""
    if reason.startswith("regime_"):
        return f"ditolak regime filter (regime saat ini: {regime})"
    if reason.startswith("risk_"):
        return f"gagal risk filter ({reason.split('risk_', 1)[1]})"
    return REASON_LABELS.get(reason, reason)


# ============================================================
# Indicator helpers
# ============================================================

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI. Pure uptrend → 100, pure downtrend → 0 (tidak NaN)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask(avg_loss == 0, 100.0)
    rsi = rsi.mask(avg_gain == 0, 0.0)
    return rsi


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def _bollinger(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    width_pct = (upper - lower) / mid.replace(0, np.nan) * 100
    # Bandwidth = 0 → percent_b = 0.5 (netral), bukan NaN/inf
    denom = upper - lower
    percent_b = np.where(denom == 0, 0.5, (series - lower) / denom)
    percent_b = pd.Series(percent_b, index=series.index)
    return width_pct, percent_b


def _obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff().fillna(0))
    return (direction * df["volume"]).cumsum()


def _atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period).mean()
    return (atr / close.replace(0, np.nan)) * 100


def _ema_slope(series: pd.Series, lookback: int = 5) -> float:
    if len(series) < lookback + 1 or pd.isna(series.iloc[-1]):
        return 0.0
    return float(series.iloc[-1] - series.iloc[-lookback]) / max(abs(series.iloc[-lookback]), 1e-9)


# ============================================================
# Confluence analysis (laporan teknikal 1 token)
# ============================================================

def analyze_confluence(df: pd.DataFrame) -> dict:
    """Analisa teknikal mandiri (EMA/RSI/MACD/Bollinger/OBV/ATR).
    Dipakai khusus untuk laporan analyze.py — bukan filter lolos/tidak.
    """
    if len(df) < 30:
        return {
            "bias": "DATA KURANG",
            "bullish_votes": 0.0,
            "bearish_votes": 0.0,
            "notes": ["Data kurang dari 30 bar"],
            "volatility_pct": None,
            "bb_width_pct": None,
            "rsi": None,
        }

    close = df["close"]
    volume = df["volume"]

    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean() if len(close) >= 200 else None

    rsi14 = _rsi(close, 14)
    _, _, hist = _macd(close)
    bb_width, percent_b = _bollinger(close)
    atrp = _atr_pct(df)
    obv_series = _obv(df)

    bull = 0.0
    bear = 0.0
    notes: list[str] = []

    price_now = float(close.iloc[-1])
    ema20_now = float(ema20.iloc[-1])
    ema50_now = float(ema50.iloc[-1])
    rsi_now = float(rsi14.iloc[-1]) if pd.notna(rsi14.iloc[-1]) else None
    pb_now = float(percent_b.iloc[-1]) if pd.notna(percent_b.iloc[-1]) else None
    hist_now = float(hist.iloc[-1]) if pd.notna(hist.iloc[-1]) else None
    hist_prev = float(hist.iloc[-2]) if len(hist) >= 2 and pd.notna(hist.iloc[-2]) else None

    # 1. Struktur EMA
    if ema20_now > ema50_now:
        bull += 1.5
        notes.append("EMA20 di atas EMA50 (struktur jangka pendek naik)")
    else:
        bear += 1.5
        notes.append("EMA20 di bawah EMA50 (struktur jangka pendek turun)")

    if ema200 is not None and pd.notna(ema200.iloc[-1]):
        ema200_now = float(ema200.iloc[-1])
        if price_now > ema200_now:
            bull += 1.2
            notes.append("Harga di atas EMA200 (bias jangka panjang naik)")
        else:
            bear += 1.2
            notes.append("Harga di bawah EMA200 (bias jangka panjang turun)")

        if ema20_now > ema50_now > ema200_now:
            bull += 0.8
            notes.append("Full bullish alignment (EMA20 > EMA50 > EMA200)")
        elif ema20_now < ema50_now < ema200_now:
            bear += 0.8
            notes.append("Full bearish alignment (EMA20 < EMA50 < EMA200)")

    # 2. Slope EMA
    slope20 = _ema_slope(ema20, 5)
    slope50 = _ema_slope(ema50, 5)
    if slope20 > 0 and slope50 > 0:
        bull += 0.7
        notes.append("EMA sloping naik (momentum trend positif)")
    elif slope20 < 0 and slope50 < 0:
        bear += 0.7
        notes.append("EMA sloping turun (momentum trend negatif)")

    # 3. RSI
    if rsi_now is not None:
        if rsi_now > 55:
            bull += 1.0
            notes.append(f"RSI {rsi_now:.0f}: momentum condong naik")
        elif rsi_now < 45:
            bear += 1.0
            notes.append(f"RSI {rsi_now:.0f}: momentum condong turun")
        else:
            notes.append(f"RSI {rsi_now:.0f}: netral")

        if rsi_now >= 70:
            notes.append("RSI overbought (≥70) — hati-hati retrace")
        elif rsi_now <= 30:
            notes.append("RSI oversold (≤30) — potensi bounce")

    # 4. MACD
    if hist_now is not None and hist_prev is not None:
        if hist_now > 0 and hist_now >= hist_prev:
            bull += 1.0
            notes.append("Histogram MACD positif dan menguat")
        elif hist_now < 0 and hist_now <= hist_prev:
            bear += 1.0
            notes.append("Histogram MACD negatif dan melemah")
        else:
            notes.append("Momentum MACD belum searah jelas")

        if hist_prev <= 0 < hist_now:
            bull += 0.8
            notes.append("MACD bullish crossover")
        elif hist_prev >= 0 > hist_now:
            bear += 0.8
            notes.append("MACD bearish crossover")

    # 5. Bollinger %B
    if pb_now is not None:
        if pb_now > 0.75:
            bull += 0.9
            notes.append(f"%B tinggi ({pb_now:.2f}) — dekat/menembus upper Bollinger Band")
        elif pb_now < 0.25:
            bear += 0.9
            notes.append(f"%B rendah ({pb_now:.2f}) — dekat/menembus lower Bollinger Band")

    # 6. OBV + Volume
    if len(obv_series) >= 12:
        obv_slope = float(obv_series.iloc[-1] - obv_series.iloc[-12])
        vol_ma = volume.rolling(20).mean().iloc[-1]
        vol_now = float(volume.iloc[-1])

        if obv_slope > 0:
            bull += 0.8
            notes.append("OBV naik 12 candle terakhir (volume mendukung buyer)")
        elif obv_slope < 0:
            bear += 0.8
            notes.append("OBV turun 12 candle terakhir (volume mendukung seller)")

        if pd.notna(vol_ma) and vol_ma > 0 and vol_now > vol_ma * 1.3:
            if close.iloc[-1] > close.iloc[-2]:
                bull += 0.5
                notes.append("Volume expansion + close hijau")
            else:
                bear += 0.5
                notes.append("Volume expansion + close merah")

    # Bias
    total = bull + bear
    if total == 0:
        bias = "NETRAL"
    else:
        ratio = bull / total
        if ratio >= 0.65:
            bias = "BULLISH KUAT"
        elif ratio >= 0.55:
            bias = "BULLISH"
        elif ratio <= 0.35:
            bias = "BEARISH KUAT"
        elif ratio <= 0.45:
            bias = "BEARISH"
        else:
            bias = "NETRAL / CAMPURAN"

    return {
        "bias": bias,
        "bullish_votes": round(bull, 2),
        "bearish_votes": round(bear, 2),
        "notes": notes,
        "volatility_pct": round(float(atrp.iloc[-1]), 2) if pd.notna(atrp.iloc[-1]) else None,
        "bb_width_pct": round(float(bb_width.iloc[-1]), 2) if pd.notna(bb_width.iloc[-1]) else None,
        "rsi": rsi_now,
    }


async def analyze_symbol(symbol: str, cfg: dict, chart_format: str = "wide") -> dict:
    scan_cfg = cfg.get("scanning", {})
    klines_limit = scan_cfg.get("klines_limit", 300)
    min_history_bars = scan_cfg.get("min_history_bars", 260)
    cooldown_hours = scan_cfg.get("cooldown_hours", 4)
    state_path = scan_cfg.get("state_path", "signal_state.json")
    timeframes = cfg["timeframes"]
    state = load_state(state_path)

    per_tf: dict[str, dict] = {}

    async with BinanceFuturesClient() as client:
        regime_cfg = cfg.get("regime_filter", {})
        regime = await get_market_regime(client, cfg) if regime_cfg.get("enabled", False) else "NEUTRAL"

        for tf in timeframes:
            try:
                kline = await client.get_klines(symbol, tf, limit=klines_limit)
            except Exception as exc:
                print(f"[error] Gagal ambil data {symbol} {tf}: {exc}")
                per_tf[tf] = {"error": str(exc)}
                continue

            closed_df = drop_unclosed_candle(kline.df)
            if len(closed_df) < min_history_bars:
                print(
                    f"[warn] {symbol} {tf}: history {len(closed_df)} bar "
                    f"< minimum {min_history_bars} — skor/analisa mungkin kurang akurat"
                )

            signal = score_symbol(closed_df, symbol, cfg, timeframe=tf)

            try:
                confluence = analyze_confluence(closed_df)
            except Exception as exc:
                print(f"[warn] Gagal hitung confluence {symbol} {tf}: {exc}")
                confluence = {
                    "bias": "ERROR",
                    "bullish_votes": 0.0,
                    "bearish_votes": 0.0,
                    "notes": [],
                }

            info: dict = {
                "signal": signal,
                "kline": Kline(symbol=symbol, timeframe=tf, df=closed_df),
                "confluence": confluence,
                "is_candidate": False,
                "would_trigger": False,
                "reject_reason": None,
                "mtf_agree_tfs": [],
            }

            if signal.direction == "NONE":
                info["reject_reason"] = "tidak_ada_arah"
            elif not passes_regime_filter(signal.direction, regime, cfg):
                info["reject_reason"] = f"regime_{regime}"
            else:
                risk_ok, risk_reason = passes_risk_filter_detailed(signal, cfg)
                if not risk_ok:
                    info["reject_reason"] = f"risk_{risk_reason}"
                elif is_in_cooldown(state, symbol, signal.direction, tf, cooldown_hours):
                    info["reject_reason"] = "cooldown_aktif"
                else:
                    info["is_candidate"] = True

            per_tf[tf] = info

    mtf_bonus_weight = cfg["scoring"]["weights"].get("mtf_agreement", 0)
    direction_map = {
        tf: info["signal"].direction
        for tf, info in per_tf.items()
        if info.get("is_candidate")
    }

    for tf, info in per_tf.items():
        if not info.get("is_candidate"):
            continue
        signal = info["signal"]

        agree_tfs = [t for t, d in direction_map.items() if t != tf and d == signal.direction]
        if agree_tfs and mtf_bonus_weight and mtf_bonus_eligible(signal.setup_type, cfg):
            bonus = mtf_bonus_weight * len(agree_tfs)
            signal.score = round(signal.score + bonus, 1)
        info["mtf_agree_tfs"] = agree_tfs

        if signal.score < get_min_score_to_trigger(cfg, tf):
            info["reject_reason"] = "skor_di_bawah_ambang"
        elif is_score_excluded(cfg, signal.score, tf, signal.setup_type, signal.direction):
            info["reject_reason"] = "masuk_excluded_band"
        else:
            info["would_trigger"] = True

    # Chart hanya untuk TF yang punya arah
    os.makedirs(os.path.join(OUT_DIR, CHART_SUBDIR), exist_ok=True)
    chart_paths: list[str] = []
    for tf, info in per_tf.items():
        if "signal" not in info:
            continue
        signal = info["signal"]
        if signal.direction == "NONE":
            continue
        kline = info["kline"]
        try:
            import chart as chart_module

            is_square = chart_format == "square"
            suffix = "_square" if is_square else ""
            chart_path = os.path.join(OUT_DIR, CHART_SUBDIR, f"{symbol}_{tf}{suffix}.png")
            chart_module.build_chart(
                kline.df, symbol, tf, signal, cfg, chart_path, square=is_square,
            )
            chart_paths.append(chart_path)
            info["chart_path"] = chart_path
        except Exception as exc:
            print(f"[warn] Gagal bikin chart {symbol} {tf}: {exc}")

    return {"symbol": symbol, "regime": regime, "per_tf": per_tf, "chart_paths": chart_paths}


def compose_analysis_text(result: dict) -> str:
    symbol = result["symbol"]
    regime = result["regime"]
    per_tf = result["per_tf"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"# Analisa Teknikal {symbol}",
        f"Waktu: {now}   |   Regime BTC (4h): {regime}",
        "",
    ]

    bias_count: dict[str, int] = {}

    for tf, info in per_tf.items():
        lines.append(f"## Timeframe {tf}")

        if "error" in info:
            lines.append(f"Gagal ambil data: {info['error']}")
            lines.append("")
            continue

        signal = info["signal"]
        confluence = info.get("confluence", {})
        setup_label = f" ({signal.setup_type})" if getattr(signal, "setup_type", None) else ""
        lines.append(f"Arah: {signal.direction}{setup_label}")

        bull_v = confluence.get("bullish_votes", 0)
        bear_v = confluence.get("bearish_votes", 0)
        lines.append(
            f"Bias teknikal (EMA/RSI/MACD/Bollinger/OBV): {confluence.get('bias', '-')} "
            f"({bull_v} indikator bullish vs {bear_v} bearish)"
        )
        if confluence.get("volatility_pct") is not None:
            lines.append(f"Volatilitas (ATR): {confluence['volatility_pct']}% dari harga")
        if confluence.get("bb_width_pct") is not None:
            lines.append(f"Lebar Bollinger Band: {confluence['bb_width_pct']}%")
        if confluence.get("rsi") is not None:
            lines.append(f"RSI: {confluence['rsi']:.1f}")
        for note in confluence.get("notes", []):
            lines.append(f"- {note}")

        if info["would_trigger"]:
            lines.append("Status: LOLOS semua filter live scanner (setup ini akan memicu sinyal)")
        else:
            lines.append(f"Status: TIDAK trigger — {_reject_label(info['reject_reason'], regime)}")

        if info["mtf_agree_tfs"]:
            lines.append(f"Searah dengan TF: {', '.join(info['mtf_agree_tfs'])}")

        if "chart_path" in info:
            lines.append(f"Chart: {os.path.basename(info['chart_path'])}")

        lines.append("")

        if signal.direction != "NONE":
            bias_count[signal.direction] = bias_count.get(signal.direction, 0) + 1

    lines.append("## Ringkasan MTF")
    total_tf = len([tf for tf, info in per_tf.items() if "error" not in info])
    if bias_count:
        dominant = max(bias_count, key=bias_count.get)
        n_dominant = bias_count[dominant]
        lines.append(f"Bias dominan: {dominant} ({n_dominant}/{total_tf} timeframe searah)")
    else:
        lines.append("Tidak ada arah yang jelas di timeframe manapun saat ini.")

    triggered = [tf for tf, info in per_tf.items() if info.get("would_trigger")]
    if triggered:
        lines.append(f"Timeframe dengan setup valid (lolos filter): {', '.join(triggered)}")
    else:
        lines.append("Belum ada timeframe dengan setup yang lolos filter live scanner saat ini.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="vSynapse — analisa MTF untuk satu simbol spesifik (bukan full scan)"
    )
    parser.add_argument("--symbol", required=True, help="Kode koin, mis. ZEC atau ZECUSDT")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--chart-format",
        choices=["wide", "square"],
        default=None,
        help="Format chart yang digenerate. Default: mengikuti chart.format di config.yaml (fallback wide).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    quote_asset = cfg["exchange"]["quote_asset"]
    symbol = normalize_symbol(args.symbol, quote_asset)
    chart_format = args.chart_format or cfg.get("chart", {}).get("format", "wide")

    print(f"[analyze] Mulai analisa {symbol} — timeframe: {', '.join(cfg['timeframes'])}")
    result = asyncio.run(analyze_symbol(symbol, cfg, chart_format=chart_format))
    text = compose_analysis_text(result)

    print()
    print(text)

    os.makedirs(OUT_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(OUT_DIR, f"analysis_{result['symbol']}_{timestamp}.md")
    with open(md_path, "w") as f:
        f.write(text)

    n_charts = len(result["chart_paths"])
    print(f"\n[analyze] Selesai. {n_charts} chart + 1 file markdown tersimpan di {OUT_DIR}/")


if __name__ == "__main__":
    main()
