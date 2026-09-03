import pandas as pd

from vsynapse.indicators import technical as ta


def _sample_series():
    return pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)


def test_ema_length_matches_input():
    result = ta.ema(_sample_series(), period=3)
    assert len(result) == 10


def test_rsi_bounds():
    result = ta.rsi(_sample_series(), period=3).dropna()
    assert (result >= 0).all() and (result <= 100).all()


def test_macd_returns_three_series():
    macd_line, signal_line, hist = ta.macd(_sample_series(), fast=2, slow=4, signal=2)
    assert len(macd_line) == len(signal_line) == len(hist) == 10


def test_volume_spike_is_boolean():
    volume = pd.Series([10] * 20 + [100])
    spikes = ta.volume_spike(volume, lookback=5, factor=1.5)
    assert spikes.dtype == bool
