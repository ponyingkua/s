from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Infrastructure only. The analysis engine below does NOT use scanner scoring,
# scanner filters, scanner setup classification, or market-regime decisions.
from scanner import BinanceFuturesClient, load_config, drop_unclosed_candle

OUT_DIR = "analysis_output"

# Retry/backoff untuk fetch klines per timeframe -- lihat _fetch_tf().
# Backoff eksponensial: percobaan ke-n (0-indexed) menunggu
# FETCH_RETRY_BACKOFF_SECONDS * 2**n sebelum retry berikutnya.
FETCH_MAX_RETRIES = 3
FETCH_RETRY_BACKOFF_SECONDS = 1.5


def normalize_symbol(raw: str, quote_asset: str) -> str:
    s = raw.strip().upper()
    if not s:
        raise ValueError("Symbol cannot be empty")
    return s if s.endswith(quote_asset) else f"{s}{quote_asset}"


# ============================================================
# Indicators
# ============================================================

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    out = out.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    out = out.mask((avg_gain == 0) & (avg_loss > 0), 0.0)