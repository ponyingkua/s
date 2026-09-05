"""vSynapse v3 — Binance Futures scanner, 1 file.

Isi: fetch data (async), indikator teknikal, confluence scoring,
risk filter, cooldown/dedup, notifikasi Telegram, dan entry point CLI.

Dijalankan lewat: python scanner.py --out synaptic_candidates.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import aiohttp
import pandas as pd
import yaml

BASE_URL = "https://www.binance.com"  # fapi.binance.com sering diblokir IP datacenter


# ---------------------------------------------------------------------------
# Data client
# ---------------------------------------------------------------------------

@dataclass
class Kline:
    symbol: str
    timeframe: str
    df: pd.DataFrame


class BinanceFuturesClient:
    def __init__(self, session: aiohttp.ClientSession | None = None):
        self._session = session
        self._owns_session = session is None

    async def __aenter__(self) -> "BinanceFuturesClient":
        if self._session is None:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            self._session = aiohttp.ClientSession(headers=headers)
        return self

    async def __aexit__(self, *exc):
        if self._owns_session and self._session:
            await self._session.close()

    async def get_active_symbols(self, quote_asset: str = "USDT") -> list[str]:
        url = f"{BASE_URL}/fapi/v1/exchangeInfo"
        async with self._session.get(url) as resp:
            data = await resp.json()
        if "symbols" not in data:
            raise RuntimeError(
                f"Binance API tidak mengembalikan data yang diharapkan. "
                f"Status: {resp.status}, Response: {data}"
            )
        return [
            s["symbol"]
            for s in data["symbols"]
            if s["quoteAsset"] == quote_asset and s["status"] == "TRADING"
        ]

    async def get_24h_volume(self, symbol: str) -> float:
        url = f"{BASE_URL}/fapi/v1/ticker/24hr"
        async with self._session.get(url, params={"symbol": symbol}) as resp:
            data = await resp.json()
        return float(data.get("quoteVolume", 0))

    async def get_klines(self, symbol: str, interval: str, limit: int = 300) -> Kline:
        url = f"{BASE_URL}/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        async with self._session.get(url, params=params) as resp:
            raw = await resp.json()

        df = pd.DataFrame(
            raw,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades",
                "taker_buy_base", "taker_buy_quote", "ignore",
            ],
        )
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = df[col].astype(float)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        return Kline(symbol=symbol, timeframe=interval, df=df)

    async def get_klines_many(
        self, symbols: list[str], interval: str, limit: int = 300
    ) -> list[Kline]:
        tasks = [self.get_klines(s, interval, limit) for s in symbols]
        return await asyncio.gather(*tasks, return_exceptions=False)


def drop_unclosed_candle(df: pd.DataFrame) -> pd.DataFrame:
    """Buang candle terakhir kalau itu masih 'sedang berjalan' (belum closed).

    Binance selalu menyertakan candle yang sedang terbentuk sebagai baris
    terakhir kalau di-query real-time. Kalau ini tidak dibuang, skor/sinyal
    dihitung dari harga yang masih bisa berubah kapan saja sampai candle itu
    benar-benar tutup — sumber utama sinyal yang "berubah-ubah" tiap scan."""
    if df.empty or "close_time" not in df.columns:
        return df
    now = pd.Timestamp.now(tz="UTC")
    last_close_time = df["close_time"].iloc[-1]
    if pd.notna(last_close_time) and last_close_time > now:
        return df.iloc[:-1].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Indikator teknikal
# ---------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    result = pd.Series(index=series.index, dtype=float)
    no_loss = avg_loss == 0
    no_gain = avg_gain == 0
    normal = ~no_loss & ~no_gain

    rs = avg_gain[normal] / avg_loss[normal]
    result[normal] = 100 - (100 / (1 + rs))
    result[no_loss & ~no_gain] = 100.0
    result[no_gain & ~no_loss] = 0.0
    result[no_gain & no_loss] = 50.0
    return result


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 2.5) -> pd.Series:
    hl2 = (df["high"] + df["low"]) / 2
    atr_val = atr(df, period)
    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val

    trend = pd.Series(index=df.index, dtype=int)
    trend.iloc[0] = 1
    for i in range(1, len(df)):
        if df["close"].iloc[i] > upper_band.iloc[i - 1]:
            trend.iloc[i] = 1
        elif df["close"].iloc[i] < lower_band.iloc[i - 1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]
    return trend


def volume_spike(volume: pd.Series, lookback: int = 20, factor: float = 1.5) -> pd.Series:
    avg = volume.rolling(lookback).mean()
    return volume > (avg * factor)


# ---------------------------------------------------------------------------
# Confluence scoring
# ---------------------------------------------------------------------------

@dataclass
class SignalResult:
    symbol: str
    direction: str  # "LONG" | "SHORT" | "NONE"
    score: float
    reasons: list[str] = field(default_factory=list)
    entry: float | None = None
    sl: float | None = None
    tp: float | None = None


def compute_indicators(df: pd.DataFrame, cfg: dict) -> dict:
    """Hitung semua indikator SEKALI di seluruh histori yang tersedia.

    Krusial buat akurasi: indikator berbasis EWM (EMA, MACD, Supertrend, ATR)
    butuh histori panjang supaya nilainya konvergen ke rata-rata yang stabil,
    bukan bias ke titik mulai data yang arbitrer. Backtest versi sebelumnya
    memotong ulang window 200 bar tiap iterasi lalu menghitung EMA200 dari
    nol di situ — 200 bar jelas tidak cukup buat EMA span=200 konvergen,
    hasilnya bias dan kadang salah arah dibanding EMA200 "asli"."""
    close = df["close"]
    ind_cfg = cfg["indicators"]

    macd_line, signal_line, _ = macd(
        close, ind_cfg["macd"]["fast"], ind_cfg["macd"]["slow"], ind_cfg["macd"]["signal"],
    )

    return {
        "ema200": ema(close, ind_cfg["ema"]["period"]),
        "macd_line": macd_line,
        "signal_line": signal_line,
        "supertrend": supertrend(df, ind_cfg["supertrend"]["period"], ind_cfg["supertrend"]["multiplier"]),
        "rsi": rsi(close, ind_cfg["rsi"]["period"]),
        "vol_spike": volume_spike(df["volume"]),
        "atr": atr(df, ind_cfg["atr"]["period"]),
    }


def score_at(df: pd.DataFrame, ind: dict, i: int, symbol: str, cfg: dict) -> SignalResult:
    """Skor 1 bar spesifik (index i) pakai indikator yang sudah dihitung
    sebelumnya lewat compute_indicators(). Dipisah dari score_symbol supaya
    backtest bisa hitung indikator sekali lalu skor banyak titik, bukan
    hitung ulang tiap titik (lihat compute_indicators)."""
    w = cfg["scoring"]["weights"]
    price = df["close"].iloc[i]

    long_score, short_score = 0.0, 0.0
    long_reasons: list[str] = []
    short_reasons: list[str] = []

    if price > ind["ema200"].iloc[i]:
        long_score += w["ema_trend"]
        long_reasons.append("Harga di atas EMA200 (uptrend)")
    else:
        short_score += w["ema_trend"]
        short_reasons.append("Harga di bawah EMA200 (downtrend)")

    if ind["macd_line"].iloc[i] > ind["signal_line"].iloc[i]:
        long_score += w["macd_cross"]
        long_reasons.append("MACD line di atas signal line (momentum naik)")
    else:
        short_score += w["macd_cross"]
        short_reasons.append("MACD line di bawah signal line (momentum turun)")

    if ind["supertrend"].iloc[i] == 1:
        long_score += w["supertrend"]
        long_reasons.append("Supertrend menunjukkan uptrend")
    else:
        short_score += w["supertrend"]
        short_reasons.append("Supertrend menunjukkan downtrend")

    has_volume_spike = bool(ind["vol_spike"].iloc[i])
    if has_volume_spike:
        if long_score >= short_score:
            long_score += w["volume_spike"]
            long_reasons.append("Volume spike terdeteksi (menguatkan sinyal)")
        else:
            short_score += w["volume_spike"]
            short_reasons.append("Volume spike terdeteksi (menguatkan sinyal)")

    r = ind["rsi"].iloc[i]
    if 40 <= r <= 60:
        long_score += w["rsi_confluence"] / 2
        short_score += w["rsi_confluence"] / 2
        long_reasons.append(f"RSI netral ({r:.0f})")
        short_reasons.append(f"RSI netral ({r:.0f})")
    elif r < 40:
        long_score += w["rsi_confluence"]
        long_reasons.append(f"RSI oversold ({r:.0f}), potensi rebound")
    elif r > 60:
        short_score += w["rsi_confluence"]
        short_reasons.append(f"RSI overbought ({r:.0f}), potensi koreksi")

    if long_score >= short_score:
        direction, final_score, reasons = "LONG", long_score, long_reasons
    else:
        direction, final_score, reasons = "SHORT", short_score, short_reasons

    if final_score < cfg["scoring"]["min_score_to_trigger"]:
        return SignalResult(symbol=symbol, direction="NONE", score=final_score, reasons=reasons)

    sl_dist = ind["atr"].iloc[i] * cfg["risk"]["atr_multiplier_sl"]
    if direction == "LONG":
        sl = price - sl_dist
        tp = price + sl_dist * cfg["risk"]["risk_reward_min"]
    else:
        sl = price + sl_dist
        tp = price - sl_dist * cfg["risk"]["risk_reward_min"]

    return SignalResult(
        symbol=symbol,
        direction=direction,
        score=round(final_score, 1),
        reasons=reasons,
        entry=round(price, 6),
        sl=round(sl, 6),
        tp=round(tp, 6),
    )


def score_symbol(df: pd.DataFrame, symbol: str, cfg: dict) -> SignalResult:
    """Wrapper: hitung indikator di seluruh df lalu skor bar TERAKHIR.
    Dipakai scanner (live) & chart.py — interface tidak berubah dari versi
    sebelumnya. Untuk backtest, pakai compute_indicators()+score_at()
    langsung (lihat backtest.py) supaya indikator dihitung sekali, bukan
    berulang tiap titik simulasi."""
    ind = compute_indicators(df, cfg)
    return score_at(df, ind, len(df) - 1, symbol, cfg)


# ---------------------------------------------------------------------------
# Risk filter
# ---------------------------------------------------------------------------

def passes_risk_filter(signal: SignalResult, cfg: dict) -> bool:
    if signal.direction == "NONE" or signal.entry is None:
        return False

    risk = abs(signal.entry - signal.sl)
    reward = abs(signal.tp - signal.entry)
    if risk == 0:
        return False

    rr = reward / risk
    return rr >= cfg["risk"]["risk_reward_min"] - 1e-6  # toleransi pembulatan


def position_size(equity: float, risk_pct: float, entry: float, sl: float) -> float:
    risk_amount = equity * risk_pct
    per_unit_risk = abs(entry - sl)
    if per_unit_risk == 0:
        return 0.0
    return risk_amount / per_unit_risk


# ---------------------------------------------------------------------------
# Cooldown / dedup state
# ---------------------------------------------------------------------------

def load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def save_state(path: str, state: dict) -> None:
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def is_in_cooldown(state: dict, symbol: str, direction: str, cooldown_hours: float) -> bool:
    entry = state.get(symbol)
    if not entry or entry.get("direction") != direction:
        return False
    last_time = datetime.fromisoformat(entry["timestamp"])
    return datetime.now(timezone.utc) - last_time < timedelta(hours=cooldown_hours)


def mark_signaled(state: dict, symbol: str, direction: str) -> None:
    state[symbol] = {
        "direction": direction,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Konfluensi multi-timeframe
# ---------------------------------------------------------------------------

def higher_tf_bias(df: pd.DataFrame, cfg: dict) -> str:
    """Tentukan bias tren di timeframe lebih tinggi, pakai EMA200 + Supertrend
    (dua indikator trend-following paling kuat di scoring). Dipakai buat
    filter konfluensi: tolak sinyal LONG di TF utama kalau TF lebih tinggi
    masih downtrend, dan sebaliknya — mengurangi sinyal palsu saat market
    sedang choppy/melawan tren besar."""
    closed = drop_unclosed_candle(df)
    if len(closed) < 2:
        return "NEUTRAL"

    ind = compute_indicators(closed, cfg)
    i = len(closed) - 1
    price = closed["close"].iloc[i]
    ema_up = price > ind["ema200"].iloc[i]
    st_up = ind["supertrend"].iloc[i] == 1

    if ema_up and st_up:
        return "LONG"
    if not ema_up and not st_up:
        return "SHORT"
    return "NEUTRAL"  # EMA & Supertrend tidak sepakat di TF ini — jangan pakai buat konfirmasi


# ---------------------------------------------------------------------------
# Notifikasi Telegram
# ---------------------------------------------------------------------------

def format_signal_message(signal: SignalResult) -> str:
    lines = [
        f"*{signal.symbol}* — {signal.direction} (score {signal.score})",
        f"Entry: `{signal.entry}`",
        f"SL: `{signal.sl}`",
        f"TP: `{signal.tp}`",
    ]
    if signal.reasons:
        lines.append("Alasan: " + "; ".join(signal.reasons))
    return "\n".join(lines)


async def send_telegram_message(text: str, cfg: dict) -> None:
    tg_cfg = cfg["notify"]["telegram"]
    if not tg_cfg.get("enabled"):
        return

    token = os.environ.get(tg_cfg["bot_token_env"])
    chat_id = os.environ.get(tg_cfg["chat_id_env"])
    if not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}

    async with aiohttp.ClientSession() as session, session.post(url, json=payload) as resp:
        await resp.read()


async def send_telegram_photo(photo_path: str, caption: str, cfg: dict) -> None:
    tg_cfg = cfg["notify"]["telegram"]
    if not tg_cfg.get("enabled"):
        return

    token = os.environ.get(tg_cfg["bot_token_env"])
    chat_id = os.environ.get(tg_cfg["chat_id_env"])
    if not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    with open(photo_path, "rb") as photo_file:
        data = aiohttp.FormData()
        data.add_field("chat_id", str(chat_id))
        data.add_field("caption", caption)
        data.add_field("parse_mode", "Markdown")
        data.add_field(
            "photo",
            photo_file,
            filename=os.path.basename(photo_path),
            content_type="image/png",
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as resp:
                await resp.read()


async def send_telegram_document(file_path: str, caption: str, cfg: dict) -> None:
    """Kirim 1 file (misal .zip berisi kumpulan chart) sebagai dokumen Telegram."""
    tg_cfg = cfg["notify"]["telegram"]
    if not tg_cfg.get("enabled"):
        return

    token = os.environ.get(tg_cfg["bot_token_env"])
    chat_id = os.environ.get(tg_cfg["chat_id_env"])
    if not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/sendDocument"
    # Caption Telegram dibatasi ~1024 karakter untuk dokumen.
    safe_caption = caption[:1024]
    with open(file_path, "rb") as doc_file:
        data = aiohttp.FormData()
        data.add_field("chat_id", str(chat_id))
        data.add_field("caption", safe_caption)
        data.add_field("parse_mode", "Markdown")
        data.add_field(
            "document",
            doc_file,
            filename=os.path.basename(file_path),
            content_type="application/zip",
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as resp:
                await resp.read()


def zip_charts(
    chart_paths: list[str],
    summary_text: str | None = None,
    out_dir: str = "charts",
) -> str | None:
    """Kumpulkan semua chart hasil scan ke dalam 1 file .zip, plus file teks
    ringkasan.txt berisi detail tiap sinyal di dalam zip yang sama. Return
    path zip, atau None kalau tidak ada apa pun (chart maupun ringkasan)
    untuk di-zip."""
    if not chart_paths and not summary_text:
        return None

    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    zip_path = os.path.join(out_dir, f"scan_{timestamp}.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for chart_path in chart_paths:
            if os.path.exists(chart_path):
                zf.write(chart_path, arcname=os.path.basename(chart_path))
        if summary_text:
            zf.writestr("ringkasan.txt", summary_text)

    return zip_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


async def run_scan(cfg: dict, out_path: str) -> list[dict]:
    results = []
    captions: list[str] = []
    chart_paths: list[str] = []

    scan_cfg = cfg.get("scanning", {})
    state_path = scan_cfg.get("state_path", "signal_state.json")
    cooldown_hours = scan_cfg.get("cooldown_hours", 4)
    state = load_state(state_path)

    auto_generate_charts = cfg.get("chart", {}).get("auto_generate", True)

    async with BinanceFuturesClient() as client:
        symbols = await client.get_active_symbols(cfg["exchange"]["quote_asset"])

        volumes = await asyncio.gather(*(client.get_24h_volume(s) for s in symbols))
        min_vol = cfg["exchange"]["min_volume_usdt_24h"]
        active_symbols = [s for s, v in zip(symbols, volumes) if v >= min_vol]

        primary_tf = cfg["timeframes"][0]
        klines = await client.get_klines_many(active_symbols, primary_tf)

        candidates: list[tuple[SignalResult, "Kline"]] = []
        for kline in klines:
            closed_df = drop_unclosed_candle(kline.df)
            if len(closed_df) < 2:
                continue
            signal = score_symbol(closed_df, kline.symbol, cfg)
            if signal.direction == "NONE":
                continue
            if not passes_risk_filter(signal, cfg):
                continue
            if is_in_cooldown(state, kline.symbol, signal.direction, cooldown_hours):
                continue

            candidates.append((signal, kline))

        # Filter konfluensi multi-timeframe: cuma fetch TF lebih tinggi buat
        # kandidat yang sudah lolos filter primer (efisien, bukan semua simbol).
        confluence_cfg = cfg.get("confluence", {})
        if confluence_cfg.get("require_higher_tf", False) and candidates:
            htf = confluence_cfg.get("higher_timeframe") or (
                cfg["timeframes"][-1] if len(cfg["timeframes"]) > 1 else primary_tf
            )
            if htf != primary_tf:
                htf_symbols = list({kline.symbol for _, kline in candidates})
                htf_klines = await client.get_klines_many(htf_symbols, htf)
                htf_bias_map = {k.symbol: higher_tf_bias(k.df, cfg) for k in htf_klines}
                candidates = [
                    (signal, kline) for signal, kline in candidates
                    if htf_bias_map.get(kline.symbol) == signal.direction
                ]

        # Ambil hanya yang terbaik (score tertinggi), bukan sekadar yang
        # pertama ditemukan saat iterasi symbol.
        candidates.sort(key=lambda c: c[0].score, reverse=True)
        top_candidates = candidates[: cfg["risk"]["max_signals_per_run"]]

        for signal, kline in top_candidates:
            results.append(signal.__dict__)
            mark_signaled(state, kline.symbol, signal.direction)

            captions.append(format_signal_message(signal))

            if auto_generate_charts:
                import chart as chart_module  # lazy import, hindari circular import

                os.makedirs("charts", exist_ok=True)
                chart_path = f"charts/{kline.symbol}_{primary_tf}.png"
                chart_module.build_chart(
                    kline.df,
                    kline.symbol,
                    primary_tf,
                    signal,
                    cfg,
                    chart_path,
                )
                chart_paths.append(chart_path)

    save_state(state_path, state)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # --- Kirim ke Telegram: cukup 1 file zip per scan, tanpa pesan teks terpisah ---
    summary = None
    if captions:
        summary = f"Scan selesai — {len(results)} sinyal ditemukan\n\n" + "\n\n".join(captions)

    zip_path = zip_charts(chart_paths, summary_text=summary)
    if zip_path:
        n_chart = len(chart_paths)
        caption = (
            f"{n_chart} chart sinyal (detail lengkap ada di ringkasan.txt dalam zip)"
            if n_chart
            else "Ringkasan sinyal (lihat ringkasan.txt dalam zip)"
        )
        await send_telegram_document(zip_path, caption=caption, cfg=cfg)

    return results


def main():
    parser = argparse.ArgumentParser(description="vSynapse v3 scanner")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="synaptic_candidates.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = asyncio.run(run_scan(cfg, args.out))
    print(f"Ditemukan {len(results)} sinyal. Disimpan ke {args.out}")


if __name__ == "__main__":
    main()
