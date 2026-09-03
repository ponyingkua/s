"""Async client untuk mengambil data dari Binance Futures API.

Didesain async supaya bisa scan puluhan/ratusan simbol secara paralel,
jauh lebih cepat dibanding loop sekuensial di versi sebelumnya.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import aiohttp
import pandas as pd

BASE_URL = "https://www.binance.com"


@dataclass
class Kline:
    symbol: str
    timeframe: str
    df: pd.DataFrame  # columns: open_time, open, high, low, close, volume


class BinanceFuturesClient:
    def __init__(self, session: aiohttp.ClientSession | None = None):
        self._session = session
        self._owns_session = session is None

    async def __aenter__(self) -> "BinanceFuturesClient":
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *exc):
        if self._owns_session and self._session:
            await self._session.close()

    async def get_active_symbols(self, quote_asset: str = "USDT") -> list[str]:
        """Ambil semua simbol futures aktif dengan quote asset tertentu."""
        url = f"{BASE_URL}/fapi/v1/exchangeInfo"
        async with self._session.get(url) as resp:
            data = await resp.json()
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
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        return Kline(symbol=symbol, timeframe=interval, df=df)

    async def get_klines_many(
        self, symbols: list[str], interval: str, limit: int = 300
    ) -> list[Kline]:
        """Fetch klines untuk banyak simbol secara paralel."""
        tasks = [self.get_klines(s, interval, limit) for s in symbols]
        return await asyncio.gather(*tasks, return_exceptions=False)
