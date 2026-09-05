# vSynapse v3

Binance Futures scanner — deteksi sinyal LONG/SHORT pakai confluence scoring
(EMA200, MACD, Supertrend, Volume, RSI) + filter konfluensi multi-timeframe,
lengkap dengan backtest dan chart generator gaya market-structure (HH/HL/LH/LL,
zona Demand/Supply, BOS). Jalan sepenuhnya lewat GitHub Actions.

## Struktur

```
scanner.py     # fetch data, indikator, scoring, konfluensi MTF, notify Telegram
backtest.py    # backtest single simbol atau batch banyak simbol
chart.py       # generate chart PNG (dark UI) buat posting
config.yaml    # semua parameter strategi & chart
```

## Cara Pakai

```bash
python scanner.py                                          # scan sekali
python backtest.py --symbol BTCUSDT --timeframe 1h         # backtest 1 simbol
python backtest.py --batch --timeframe 1h                  # backtest 10 simbol default
python chart.py --symbol BTCUSDT --timeframe 1h             # generate 1 chart
```

## Otomatisasi (GitHub Actions)

Semua di atas juga bisa dipicu manual dari tab **Actions**:
- **vSynapse Scan** — scan + kirim sinyal & chart ke Telegram
- **Backtest (manual)** — mode single atau batch
- **Chart Generator (manual)** — generate 1 chart on-demand

## Setup

1. `pip install -r requirements.txt`
2. Isi secret `TELEGRAM_BOT_TOKEN` & `TELEGRAM_CHAT_ID` di **Settings → Secrets and variables → Actions**
3. Sesuaikan bobot scoring, threshold, dan `confluence.require_higher_tf` di `config.yaml`

---

> ⚠️ Untuk riset & edukasi saja. **Not Financial Advice (NFA).** Selalu
> validasi lewat backtest sebelum dipakai live — hasil historis tidak
> menjamin performa ke depan, dan backtest belum memperhitungkan slippage
> atau funding rate.
