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

BASE_URL = "https://www.binance.com"


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
        if resp.status != 200 or "symbols" not in data:
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
        if resp.status != 200 or not isinstance(data, dict):
            return 0.0
        return float(data.get("quoteVolume", 0))

    @staticmethod
    def _parse_klines_df(raw: list) -> pd.DataFrame:
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
        return df

    async def get_klines(self, symbol: str, interval: str, limit: int = 300) -> Kline:
        url = f"{BASE_URL}/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        async with self._session.get(url, params=params) as resp:
            raw = await resp.json()
        if resp.status != 200 or not isinstance(raw, list):
            raise RuntimeError(
                f"Gagal mengambil klines {symbol} {interval}: "
                f"status={resp.status}, response={raw}"
            )
        df = self._parse_klines_df(raw)
        return Kline(symbol=symbol, timeframe=interval, df=df)

    async def get_klines_paginated(self, symbol: str, interval: str, total_limit: int) -> Kline:
        url = f"{BASE_URL}/fapi/v1/klines"
        all_raw: list = []
        end_time: int | None = None
        remaining = total_limit

        while remaining > 0:
            batch_limit = min(remaining, 1500)
            params = {"symbol": symbol, "interval": interval, "limit": batch_limit}
            if end_time is not None:
                params["endTime"] = end_time
            async with self._session.get(url, params=params) as resp:
                raw = await resp.json()
            if resp.status != 200 or not isinstance(raw, list):
                raise RuntimeError(
                    f"Gagal mengambil klines {symbol} {interval}: "
                    f"status={resp.status}, response={raw}"
                )
            if not raw:
                break
            all_raw = raw + all_raw
            remaining -= len(raw)
            end_time = int(raw[0][0]) - 1
            if len(raw) < batch_limit:
                break

        df = self._parse_klines_df(all_raw)
        df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
        if len(df) > total_limit:
            df = df.iloc[-total_limit:].reset_index(drop=True)
        return Kline(symbol=symbol, timeframe=interval, df=df)

    async def get_klines_many(
        self, symbols: list[str], interval: str, limit: int = 300
    ) -> list[Kline]:
        tasks = [self.get_klines(s, interval, limit) for s in symbols]
        return await asyncio.gather(*tasks, return_exceptions=False)


def drop_unclosed_candle(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "close_time" not in df.columns:
        return df
    now = pd.Timestamp.now(tz="UTC")
    last_close_time = df["close_time"].iloc[-1]
    if pd.notna(last_close_time) and last_close_time > now:
        return df.iloc[:-1].reset_index(drop=True)
    return df


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

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
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 2.5) -> pd.Series:
    """Return the direction of the standard Supertrend calculation.

    The previous implementation compared price to the *raw* previous ATR
    bands.  Those bands are not the Supertrend bands: they must be carried
    forward and only move in the direction allowed by the prior close.  The
    old version therefore flipped too easily in noisy markets.
    """
    if df.empty:
        return pd.Series(dtype="int64", index=df.index)

    hl2 = (df["high"] + df["low"]) / 2
    atr_val = atr(df, period)
    basic_upper = hl2 + multiplier * atr_val
    basic_lower = hl2 - multiplier * atr_val
    final_upper = pd.Series(index=df.index, dtype=float)
    final_lower = pd.Series(index=df.index, dtype=float)
    trend = pd.Series(index=df.index, dtype="int64")

    first = df.index[0]
    final_upper.loc[first] = basic_upper.loc[first]
    final_lower.loc[first] = basic_lower.loc[first]
    trend.loc[first] = 1

    for i in range(1, len(df)):
        idx = df.index[i]
        prev_idx = df.index[i - 1]

        if (
            basic_upper.loc[idx] < final_upper.loc[prev_idx]
            or df["close"].loc[prev_idx] > final_upper.loc[prev_idx]
        ):
            final_upper.loc[idx] = basic_upper.loc[idx]
        else:
            final_upper.loc[idx] = final_upper.loc[prev_idx]

        if (
            basic_lower.loc[idx] > final_lower.loc[prev_idx]
            or df["close"].loc[prev_idx] < final_lower.loc[prev_idx]
        ):
            final_lower.loc[idx] = basic_lower.loc[idx]
        else:
            final_lower.loc[idx] = final_lower.loc[prev_idx]

        prev_trend = trend.loc[prev_idx]
        if prev_trend == -1 and df["close"].loc[idx] > final_upper.loc[prev_idx]:
            trend.loc[idx] = 1
        elif prev_trend == 1 and df["close"].loc[idx] < final_lower.loc[prev_idx]:
            trend.loc[idx] = -1
        else:
            trend.loc[idx] = prev_trend

    return trend


def volume_spike(volume: pd.Series, lookback: int = 20, factor: float = 1.5) -> pd.Series:
    # Compare the current candle against completed candles only.  Including
    # the current volume in its own baseline makes a genuine spike harder to
    # detect and makes the feature inconsistent with a live alert.
    avg = volume.shift(1).rolling(lookback, min_periods=lookback).mean()
    return volume > (avg * factor)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
    ).max(axis=1)

    smoothed_tr = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, 1e-9)
    smoothed_plus_dm = plus_dm.ewm(alpha=1 / period, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * (smoothed_plus_dm / smoothed_tr)
    minus_di = 100 * (smoothed_minus_dm / smoothed_tr)

    di_sum = (plus_di + minus_di).replace(0, 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


@dataclass
class SignalResult:
    symbol: str
    direction: str
    score: float
    timeframe: str = ""
    setup_type: str = ""
    reasons: list[str] = field(default_factory=list)
    entry: float | None = None
    sl: float | None = None
    tp: float | None = None


def compute_indicators(df: pd.DataFrame, cfg: dict) -> dict:
    close = df["close"]
    ind_cfg = cfg["indicators"]

    macd_line, signal_line, _ = macd(
        close, ind_cfg["macd"]["fast"], ind_cfg["macd"]["slow"], ind_cfg["macd"]["signal"],
    )
    vol_cfg = ind_cfg.get("volume_spike", {})
    adx_cfg = ind_cfg.get("adx", {})

    return {
        "ema200": ema(close, ind_cfg["ema"]["period"]),
        "macd_line": macd_line,
        "signal_line": signal_line,
        "supertrend": supertrend(df, ind_cfg["supertrend"]["period"], ind_cfg["supertrend"]["multiplier"]),
        "rsi": rsi(close, ind_cfg["rsi"]["period"]),
        "vol_spike": volume_spike(
            df["volume"], vol_cfg.get("lookback", 20), vol_cfg.get("factor", 1.5)
        ),
        "atr": atr(df, ind_cfg["atr"]["period"]),
        "adx": adx(df, adx_cfg.get("period", 14)),
    }


def get_setup_engine_param(cfg: dict, param: str, timeframe: str = "", default: float = 0.0) -> float:
    se_cfg = cfg.get("setup_engine", {})
    tf_cfg = se_cfg.get(timeframe, {}) if timeframe else {}
    if isinstance(tf_cfg, dict) and param in tf_cfg:
        return tf_cfg[param]
    return se_cfg.get(param, default)


def classify_setup(df: pd.DataFrame, ind: dict, i: int, direction: str, cfg: dict, timeframe: str = "") -> str:
    structure_lookback = int(get_setup_engine_param(cfg, "structure_lookback", timeframe, 20))
    extended_atr_mult = get_setup_engine_param(cfg, "extended_atr_mult", timeframe, 3.5)
    pullback_atr_mult = get_setup_engine_param(cfg, "pullback_atr_mult", timeframe, 1.0)
    breakout_buffer_atr_mult = get_setup_engine_param(
        cfg, "breakout_buffer_atr_mult", timeframe, 0.10
    )

    close = df["close"].iloc[i]
    atr_val = ind["atr"].iloc[i]
    ema_val = ind["ema200"].iloc[i]
    if pd.isna(close) or pd.isna(atr_val) or pd.isna(ema_val) or atr_val <= 0:
        return "UNKNOWN"
    dist_ema_atr = abs(close - ema_val) / atr_val if atr_val > 0 else 0.0

    lookback_start = max(0, i - structure_lookback)
    prior_high = df["high"].iloc[lookback_start:i].max() if i > lookback_start else close
    prior_low = df["low"].iloc[lookback_start:i].min() if i > lookback_start else close

    breakout_buffer = atr_val * breakout_buffer_atr_mult
    if direction == "LONG" and pd.notna(prior_high) and close > prior_high + breakout_buffer:
        return "BREAKOUT"
    if direction == "SHORT" and pd.notna(prior_low) and close < prior_low - breakout_buffer:
        return "BREAKOUT"

    on_trend_side = (
        (direction == "LONG" and close >= ema_val)
        or (direction == "SHORT" and close <= ema_val)
    )
    if on_trend_side and dist_ema_atr <= pullback_atr_mult:
        return "PULLBACK"
    if on_trend_side and dist_ema_atr >= extended_atr_mult:
        return "EXTENDED"
    return "CONTINUATION"


def get_setup_bonus(cfg: dict, direction: str, setup_type: str, timeframe: str = "") -> float:
    sb_cfg = cfg["scoring"].get("setup_bonus", {})

    tf_cfg = sb_cfg.get(timeframe, {}) if timeframe else {}
    if isinstance(tf_cfg, dict) and ("long" in tf_cfg or "short" in tf_cfg):
        dir_cfg = tf_cfg.get(direction.lower(), {})
        if setup_type.lower() in dir_cfg:
            return dir_cfg[setup_type.lower()]

    if "long" in sb_cfg or "short" in sb_cfg:
        dir_cfg = sb_cfg.get(direction.lower(), {})
        return dir_cfg.get(setup_type.lower(), 0)
    return sb_cfg.get(setup_type.lower(), 0)


def passes_regime_filter(direction: str, regime: str, cfg: dict) -> bool:
    """Return True if the proposed direction is allowed under current market regime."""
    regime_cfg = cfg.get("regime_filter", {})
    if direction == "SHORT":
        mode = regime_cfg.get("short_mode", "bear_only")
        if mode == "bear_only":
            return regime == "BEAR"
        if mode == "not_bull":
            return regime != "BULL"
        return True  # allow_all
    if direction == "LONG":
        mode = regime_cfg.get("long_mode", "not_bear")
        if mode == "bull_only":
            return regime == "BULL"
        if mode == "bull_or_neutral":
            return regime != "BEAR"
        if mode == "not_bear":
            return regime != "BEAR"
        return True
    return True


def get_regime_gated_direction(regime: str, cfg: dict) -> str | None:
    """Return the direction that should be rate-limited in the current regime episode."""
    regime_cfg = cfg.get("regime_filter", {})
    short_mode = regime_cfg.get("short_mode", "bear_only")
    if regime == "BEAR" and short_mode in ("bear_only", "not_bull"):
        # Both modes treat BEAR as SHORT's "home" regime, so a long BEAR
        # episode still needs the cluster cap - not just the strict
        # bear_only case. Without this, loosening short_mode to not_bull
        # would silently disable episode capping for shorts.
        return "SHORT"
    long_mode = regime_cfg.get("long_mode", "not_bear")
    if regime == "BULL" and long_mode in ("bull_only", "bull_or_neutral", "not_bear"):
        return "LONG"
    return None


def update_regime_episode(state: dict, regime: str) -> dict:
    episode = state.get("_regime_episode")
    if not isinstance(episode, dict) or episode.get("regime") != regime:
        episode = {
            "regime": regime,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "signal_count": 0,
        }
        state["_regime_episode"] = episode
    return episode


def score_at(
    df: pd.DataFrame, ind: dict, i: int, symbol: str, cfg: dict, timeframe: str = ""
) -> SignalResult:
    if i < 0 or i >= len(df):
        return SignalResult(
            symbol=symbol, direction="NONE", score=0.0, timeframe=timeframe,
            reasons=["Index candle tidak valid"],
        )

    required_values = [
        ind["ema200"].iloc[i],
        ind["macd_line"].iloc[i],
        ind["signal_line"].iloc[i],
        ind["supertrend"].iloc[i],
        ind["rsi"].iloc[i],
        ind["atr"].iloc[i],
        ind["adx"].iloc[i],
    ]
    if any(pd.isna(value) for value in required_values):
        return SignalResult(
            symbol=symbol, direction="NONE", score=0.0, timeframe=timeframe,
            reasons=["Indikator belum warm-up atau datanya tidak valid"],
        )

    adx_min = cfg.get("risk", {}).get("adx_min", 0)
    if adx_min and pd.notna(ind["adx"].iloc[i]) and ind["adx"].iloc[i] < adx_min:
        return SignalResult(
            symbol=symbol,
            direction="NONE",
            score=0.0,
            timeframe=timeframe,
            reasons=[
                f"ADX ({ind['adx'].iloc[i]:.1f}) di bawah ambang {adx_min} — "
                "market dianggap choppy, sinyal ditolak"
            ],
        )

    # Optional symbol blacklist
    blacklist = cfg.get("symbol_filter", {}).get("blacklist", [])
    if symbol in blacklist:
        return SignalResult(
            symbol=symbol, direction="NONE", score=0.0, timeframe=timeframe,
            reasons=[f"{symbol} ada di blacklist"],
        )

    w = cfg["scoring"]["weights"]
    price = df["close"].iloc[i]
    open_price = df["open"].iloc[i]

    long_score, short_score = 0.0, 0.0
    reasons: list[str] = []

    trend_long, trend_short = 0.0, 0.0
    alignment_long = 0
    alignment_short = 0

    ema_up = price > ind["ema200"].iloc[i]
    if ema_up:
        trend_long += w["ema_trend"]
        alignment_long += 1
    else:
        trend_short += w["ema_trend"]
        alignment_short += 1

    macd_up = ind["macd_line"].iloc[i] > ind["signal_line"].iloc[i]
    if macd_up:
        trend_long += w["macd_cross"]
        alignment_long += 1
    else:
        trend_short += w["macd_cross"]
        alignment_short += 1

    st_up = ind["supertrend"].iloc[i] == 1
    if st_up:
        trend_long += w["supertrend"]
        alignment_long += 1
    else:
        trend_short += w["supertrend"]
        alignment_short += 1

    direction = "LONG" if trend_long >= trend_short else "SHORT"
    long_score, short_score = trend_long, trend_short
    alignment = alignment_long if direction == "LONG" else alignment_short

    # Hard confluence filter
    min_alignment = cfg.get("scoring", {}).get("min_trend_alignment", 2)
    min_direction_margin = cfg.get("scoring", {}).get("min_direction_margin", 0)
    direction_margin = abs(trend_long - trend_short)
    if alignment < min_alignment:
        return SignalResult(
            symbol=symbol,
            direction="NONE",
            score=0.0,
            timeframe=timeframe,
            reasons=[
                f"Konfluensi tren lemah ({alignment}/{min_alignment} indikator inti searah) — ditolak"
            ],
        )
    if direction_margin < min_direction_margin:
        return SignalResult(
            symbol=symbol, direction="NONE", score=0.0, timeframe=timeframe,
            reasons=[
                f"Arah terlalu imbang ({direction_margin:.1f} < "
                f"{min_direction_margin} skor margin) — ditolak"
            ],
        )

    if direction == "LONG":
        reasons.append("Harga di atas EMA200 (uptrend)" if ema_up else "Harga di bawah EMA200 (tapi indikator lain condong LONG)")
        reasons.append("MACD line di atas signal line (momentum naik)" if macd_up else "MACD line di bawah signal line (tapi indikator lain condong LONG)")
        reasons.append("Supertrend menunjukkan uptrend" if st_up else "Supertrend menunjukkan downtrend (tapi indikator lain condong LONG)")
    else:
        reasons.append("Harga di bawah EMA200 (downtrend)" if not ema_up else "Harga di atas EMA200 (tapi indikator lain condong SHORT)")
        reasons.append("MACD line di bawah signal line (momentum turun)" if not macd_up else "MACD line di atas signal line (tapi indikator lain condong SHORT)")
        reasons.append("Supertrend menunjukkan downtrend" if not st_up else "Supertrend menunjukkan uptrend (tapi indikator lain condong SHORT)")

    reasons.append(f"Konfluensi tren: {alignment}/3 indikator inti searah")

    r = ind["rsi"].iloc[i]
    if direction == "LONG" and 50 < r <= 68:
        long_score += w["rsi_confluence"]
        reasons.append(f"RSI ({r:.0f}) menguatkan momentum naik")
    elif direction == "SHORT" and 32 <= r < 50:
        short_score += w["rsi_confluence"]
        reasons.append(f"RSI ({r:.0f}) menguatkan momentum turun")
    elif direction == "LONG" and r > 70:
        long_score -= 10
        reasons.append(f"RSI ({r:.0f}) overbought — penalti chasing (-10)")
    elif direction == "SHORT" and r < 30:
        short_score -= 10
        reasons.append(f"RSI ({r:.0f}) oversold — penalti chasing (-10)")
    else:
        reasons.append(f"RSI netral ({r:.0f}), tidak menambah skor")

    has_volume_spike = bool(ind["vol_spike"].iloc[i])
    candle_bullish = price > open_price
    candle_bearish = price < open_price
    if has_volume_spike and direction == "LONG" and candle_bullish:
        long_score += w["volume_spike"]
        reasons.append("Volume spike di candle bullish (menguatkan sinyal LONG)")
    elif has_volume_spike and direction == "SHORT" and candle_bearish:
        short_score += w["volume_spike"]
        reasons.append("Volume spike di candle bearish (menguatkan sinyal SHORT)")
    elif has_volume_spike:
        reasons.append("Volume spike terdeteksi tapi arah candle tidak sesuai sinyal — diabaikan")

    adx_strength_weight = w.get("adx_strength", 0)
    adx_val = ind["adx"].iloc[i]
    if adx_strength_weight and pd.notna(adx_val):
        adx_strong = cfg.get("risk", {}).get("adx_strong", (adx_min or 20) + 15)
        if adx_val >= adx_strong:
            adx_bonus = adx_strength_weight
        elif adx_min and adx_val > adx_min and adx_strong > adx_min:
            adx_bonus = adx_strength_weight * (adx_val - adx_min) / (adx_strong - adx_min)
        else:
            adx_bonus = 0.0
        if adx_bonus > 0:
            if direction == "LONG":
                long_score += adx_bonus
            else:
                short_score += adx_bonus
            reasons.append(f"ADX ({adx_val:.1f}) tren kuat (+{adx_bonus:.1f} skor)")

    setup_type = classify_setup(df, ind, i, direction, cfg, timeframe)
    if setup_type == "UNKNOWN":
        return SignalResult(
            symbol=symbol, direction="NONE", score=0.0, timeframe=timeframe,
            reasons=["Setup tidak dapat diklasifikasikan karena nilai indikator invalid"],
        )

    if (
        setup_type == "EXTENDED"
        and cfg.get("setup_engine", {}).get("block_extended", True)
    ):
        return SignalResult(
            symbol=symbol,
            direction="NONE",
            score=0.0,
            timeframe=timeframe,
            setup_type=setup_type,
            reasons=reasons + [
                f"Setup EXTENDED diblokir total di timeframe {timeframe} (historis sangat underperform)"
            ],
        )

    blocked_setups = cfg.get("setup_engine", {}).get("block_setups", {}).get(timeframe, [])
    if setup_type in blocked_setups:
        return SignalResult(
            symbol=symbol,
            direction="NONE",
            score=0.0,
            timeframe=timeframe,
            setup_type=setup_type,
            reasons=reasons + [
                f"Setup {setup_type} diblokir total di timeframe {timeframe} (histori backtest jelek — lihat setup_engine.block_setups)"
            ],
        )

    setup_bonus = get_setup_bonus(cfg, direction, setup_type, timeframe)
    if direction == "LONG":
        long_score += setup_bonus
    else:
        short_score += setup_bonus
    sign = "+" if setup_bonus >= 0 else ""
    reasons.append(f"Setup terdeteksi: {setup_type} ({sign}{setup_bonus} skor)")

    final_score = long_score if direction == "LONG" else short_score

    if final_score < cfg["scoring"]["min_score_to_trigger"]:
        return SignalResult(
            symbol=symbol, direction="NONE", score=final_score,
            timeframe=timeframe, setup_type=setup_type, reasons=reasons,
        )

    struct_lookback = int(
        get_setup_engine_param(cfg, "structure_lookback", timeframe, 20)
    )
    stop_lookback = int(
        get_setup_engine_param(
            cfg, "stop_lookback", timeframe, max(5, min(struct_lookback, 12))
        )
    )
    sl_buffer = ind["atr"].iloc[i] * get_setup_engine_param(cfg, "structure_sl_buffer_atr_mult", timeframe, 0.25)
    atr_sl_dist = ind["atr"].iloc[i] * cfg["risk"]["atr_multiplier_sl"]

    risk_cfg = cfg.get("risk", {})
    base_cap_mult = risk_cfg.get("structure_sl_max_atr_mult", 2.5)
    # These settings are multiples of raw ATR, not multiples of the already
    # multiplied ATR stop. The old code effectively allowed
    # 2.3 * 1.6 = 3.68 ATR while the config said 2.3 ATR.
    cap_mult = base_cap_mult
    if setup_type == "BREAKOUT":
        cap_mult = risk_cfg.get("structure_sl_max_atr_mult_breakout", base_cap_mult)
    max_sl_dist = ind["atr"].iloc[i] * max(cap_mult, cfg["risk"]["atr_multiplier_sl"])

    lb_start = max(0, i - stop_lookback)
    if direction == "LONG":
        structural_level = df["low"].iloc[lb_start:i].min() if i > lb_start else None
        structural_dist = (price - structural_level) + sl_buffer if pd.notna(structural_level) else atr_sl_dist
    else:
        structural_level = df["high"].iloc[lb_start:i].max() if i > lb_start else None
        structural_dist = (structural_level - price) + sl_buffer if pd.notna(structural_level) else atr_sl_dist

    if (
        structural_dist > max_sl_dist
        and risk_cfg.get("reject_structural_stop_too_wide", True)
    ):
        return SignalResult(
            symbol=symbol, direction="NONE", score=final_score,
            timeframe=timeframe, setup_type=setup_type,
            reasons=reasons + [
                f"Stop struktur terlalu jauh ({structural_dist / ind['atr'].iloc[i]:.2f} ATR "
                f"> batas {cap_mult:.2f} ATR) — setup ditolak"
            ],
        )

    sl_dist = max(atr_sl_dist, structural_dist)
    if sl_dist > max_sl_dist:
        sl_dist = max_sl_dist
        reasons.append(f"SL struktur di-cap {cap_mult}x ATR")
    elif sl_dist > atr_sl_dist:
        reasons.append(f"SL digeser ke luar struktur {stop_lookback}-bar terakhir")

    rr_min = cfg["risk"]["risk_reward_min"]
    target_lookback = int(get_setup_engine_param(cfg, "target_lookback", timeframe, struct_lookback * 2))
    tgt_start = max(0, i - target_lookback)

    if direction == "LONG":
        sl = price - sl_dist
        rr_tp = price + sl_dist * rr_min
        struct_target = df["high"].iloc[tgt_start:i].max() if i > tgt_start else None
        if pd.notna(struct_target) and (struct_target - price) >= sl_dist * rr_min:
            tp = struct_target
            reasons.append(
                f"TP diambil dari level struktur (prior high {target_lookback}-bar), "
                f"RR aktual {(tp - price) / sl_dist:.2f}"
            )
        else:
            tp = rr_tp
    else:
        sl = price + sl_dist
        rr_tp = price - sl_dist * rr_min
        struct_target = df["low"].iloc[tgt_start:i].min() if i > tgt_start else None
        if pd.notna(struct_target) and (price - struct_target) >= sl_dist * rr_min:
            tp = struct_target
            reasons.append(
                f"TP diambil dari level struktur (prior low {target_lookback}-bar), "
                f"RR aktual {(price - tp) / sl_dist:.2f}"
            )
        else:
            tp = rr_tp

    return SignalResult(
        symbol=symbol,
        direction=direction,
        score=round(final_score, 1),
        timeframe=timeframe,
        setup_type=setup_type,
        reasons=reasons,
        entry=round(price, 6),
        sl=round(sl, 6),
        tp=round(tp, 6),
    )


def score_symbol(df: pd.DataFrame, symbol: str, cfg: dict, timeframe: str = "") -> SignalResult:
    ind = compute_indicators(df, cfg)
    return score_at(df, ind, len(df) - 1, symbol, cfg, timeframe)


def passes_risk_filter(signal: SignalResult, cfg: dict) -> bool:
    if signal.direction == "NONE" or signal.entry is None:
        return False
    if signal.sl is None or signal.tp is None:
        return False
    if not all(pd.notna(value) for value in (signal.entry, signal.sl, signal.tp)):
        return False

    if signal.direction == "LONG" and not (signal.sl < signal.entry < signal.tp):
        return False
    if signal.direction == "SHORT" and not (signal.tp < signal.entry < signal.sl):
        return False

    risk = abs(signal.entry - signal.sl)
    reward = abs(signal.tp - signal.entry)
    if risk == 0:
        return False

    rr = reward / risk
    return rr >= cfg["risk"]["risk_reward_min"] - 1e-6


def position_size(equity: float, risk_pct: float, entry: float, sl: float) -> float:
    risk_amount = equity * risk_pct
    per_unit_risk = abs(entry - sl)
    if per_unit_risk == 0:
        return 0.0
    return risk_amount / per_unit_risk


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


def _state_key(symbol: str, timeframe: str) -> str:
    return f"{symbol}|{timeframe}"


def is_in_cooldown(state: dict, symbol: str, direction: str, timeframe: str, cooldown_hours: float) -> bool:
    entry = state.get(_state_key(symbol, timeframe))
    if not entry or entry.get("direction") != direction:
        return False
    last_time = datetime.fromisoformat(entry["timestamp"])
    return datetime.now(timezone.utc) - last_time < timedelta(hours=cooldown_hours)


def mark_signaled(state: dict, symbol: str, direction: str, timeframe: str) -> None:
    state[_state_key(symbol, timeframe)] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "direction": direction,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def get_market_regime(client: "BinanceFuturesClient", cfg: dict) -> str:
    regime_cfg = cfg.get("regime_filter", {})
    symbol = regime_cfg.get("symbol", "BTCUSDT")
    timeframe = regime_cfg.get("timeframe", "4h")
    adx_threshold = regime_cfg.get("adx_min", 25)

    try:
        kline = await client.get_klines(symbol, timeframe, limit=cfg.get("scanning", {}).get("klines_limit", 300))
    except Exception:
        return "NEUTRAL"

    closed = drop_unclosed_candle(kline.df)
    if len(closed) < 50:
        return "NEUTRAL"

    ind = compute_indicators(closed, cfg)
    i = len(closed) - 1
    price = closed["close"].iloc[i]
    ema_up = price > ind["ema200"].iloc[i]
    st_up = ind["supertrend"].iloc[i] == 1
    adx_val = ind["adx"].iloc[i]

    if pd.isna(adx_val) or adx_val < adx_threshold:
        return "NEUTRAL"
    if ema_up and st_up:
        return "BULL"
    if not ema_up and not st_up:
        return "BEAR"
    return "NEUTRAL"


def format_signal_message(signal: SignalResult) -> str:
    tf_label = f" [{signal.timeframe}]" if signal.timeframe else ""
    setup_label = f" · {signal.setup_type}" if signal.setup_type else ""
    lines = [
        f"*{signal.symbol}*{tf_label} — {signal.direction}{setup_label} (score {signal.score})",
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
    tg_cfg = cfg["notify"]["telegram"]
    if not tg_cfg.get("enabled"):
        return

    token = os.environ.get(tg_cfg["bot_token_env"])
    chat_id = os.environ.get(tg_cfg["chat_id_env"])
    if not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/sendDocument"
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
    klines_limit = scan_cfg.get("klines_limit", 300)
    min_history_bars = scan_cfg.get("min_history_bars", 250)
    state = load_state(state_path)

    auto_generate_charts = cfg.get("chart", {}).get("auto_generate", True)
    timeframes = cfg["timeframes"]
    mtf_bonus_weight = cfg["scoring"]["weights"].get("mtf_agreement", 0)

    async with BinanceFuturesClient() as client:
        symbols = await client.get_active_symbols(cfg["exchange"]["quote_asset"])

        volumes = await asyncio.gather(*(client.get_24h_volume(s) for s in symbols))
        min_vol = cfg["exchange"]["min_volume_usdt_24h"]
        active_symbols = [s for s, v in zip(symbols, volumes) if v >= min_vol]

        regime_cfg = cfg.get("regime_filter", {})
        regime = await get_market_regime(client, cfg) if regime_cfg.get("enabled", False) else "NEUTRAL"

        regime_episode = update_regime_episode(state, regime)
        regime_gated_direction = get_regime_gated_direction(regime, cfg)
        max_signals_per_episode = cfg.get("risk", {}).get("max_signals_per_regime_episode")

        per_tf_candidates: dict[str, list[tuple[SignalResult, Kline]]] = {}
        diag_totals = {
            "history_too_short": 0, "score_none": 0, "regime_rejected": 0,
            "risk_rejected": 0, "cooldown_rejected": 0, "passed": 0,
        }
        for tf in timeframes:
            klines = await client.get_klines_many(active_symbols, tf, limit=klines_limit)
            tf_candidates: list[tuple[SignalResult, Kline]] = []
            for kline in klines:
                closed_df = drop_unclosed_candle(kline.df)
                if len(closed_df) < min_history_bars:
                    diag_totals["history_too_short"] += 1
                    continue
                signal = score_symbol(closed_df, kline.symbol, cfg, timeframe=tf)
                if signal.direction == "NONE":
                    diag_totals["score_none"] += 1
                    continue
                if not passes_regime_filter(signal.direction, regime, cfg):
                    diag_totals["regime_rejected"] += 1
                    continue
                if not passes_risk_filter(signal, cfg):
                    diag_totals["risk_rejected"] += 1
                    continue
                if is_in_cooldown(state, kline.symbol, signal.direction, tf, cooldown_hours):
                    diag_totals["cooldown_rejected"] += 1
                    continue
                diag_totals["passed"] += 1
                tf_candidates.append((signal, kline))
            per_tf_candidates[tf] = tf_candidates

        print(
            f"[diag] regime={regime} | history_too_short={diag_totals['history_too_short']} "
            f"score_none={diag_totals['score_none']} regime_rejected={diag_totals['regime_rejected']} "
            f"risk_rejected={diag_totals['risk_rejected']} cooldown_rejected={diag_totals['cooldown_rejected']} "
            f"passed={diag_totals['passed']}"
        )

        direction_map: dict[str, dict[str, str]] = {}
        for tf, cands in per_tf_candidates.items():
            for signal, kline in cands:
                direction_map.setdefault(kline.symbol, {})[tf] = signal.direction

        flat_candidates: list[tuple[SignalResult, Kline]] = []
        for tf, cands in per_tf_candidates.items():
            for signal, kline in cands:
                agree_tfs = [
                    t for t, d in direction_map.get(kline.symbol, {}).items()
                    if t != tf and d == signal.direction
                ]
                if agree_tfs and mtf_bonus_weight:
                    bonus = mtf_bonus_weight * len(agree_tfs)
                    signal.score = round(signal.score + bonus, 1)
                    signal.reasons.append(
                        f"Searah dengan TF {', '.join(agree_tfs)} (+{bonus} MTF agreement)"
                    )
                flat_candidates.append((signal, kline))

        best_per_symbol: dict[str, tuple[SignalResult, Kline]] = {}
        for signal, kline in flat_candidates:
            current = best_per_symbol.get(kline.symbol)
            if current is None or signal.score > current[0].score:
                best_per_symbol[kline.symbol] = (signal, kline)

        candidates = sorted(best_per_symbol.values(), key=lambda c: c[0].score, reverse=True)

        max_per_run = cfg["risk"]["max_signals_per_run"]
        top_candidates: list[tuple[SignalResult, Kline]] = []
        for signal, kline in candidates:
            if len(top_candidates) >= max_per_run:
                break
            if (
                regime_gated_direction
                and signal.direction == regime_gated_direction
                and max_signals_per_episode is not None
                and regime_episode["signal_count"] >= max_signals_per_episode
            ):
                signal.reasons.append(
                    f"Ditahan: sudah {regime_episode['signal_count']} sinyal "
                    f"{regime_gated_direction} diambil di episode regime "
                    f"{regime} ini (limit {max_signals_per_episode}, terpisah "
                    "dari max_signals_per_run) — cegah cluster sinyal berkorelasi"
                )
                continue
            top_candidates.append((signal, kline))
            if regime_gated_direction and signal.direction == regime_gated_direction:
                regime_episode["signal_count"] += 1

        for signal, kline in top_candidates:
            results.append(signal.__dict__)
            mark_signaled(state, kline.symbol, signal.direction, signal.timeframe)

            captions.append(format_signal_message(signal))

            if auto_generate_charts:
                import chart as chart_module

                os.makedirs("charts", exist_ok=True)
                chart_path = f"charts/{kline.symbol}_{signal.timeframe}.png"
                chart_module.build_chart(
                    kline.df,
                    kline.symbol,
                    signal.timeframe,
                    signal,
                    cfg,
                    chart_path,
                )
                chart_paths.append(chart_path)

    save_state(state_path, state)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

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
    parser = argparse.ArgumentParser(description="vSynapse v3.1 scanner (multi-timeframe)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="synaptic_candidates.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = asyncio.run(run_scan(cfg, args.out))
    print(f"Ditemukan {len(results)} sinyal. Disimpan ke {args.out}")


if __name__ == "__main__":
    main()
