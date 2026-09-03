# 🧠 vSynapse v2

Binance Futures scanner v2 — versi lebih modular, teruji (unit test + backtest),
dan lebih kaya sinyal dibanding versi pertama (`Synaptic.py`).

## Yang Berubah dari v1

- **Confluence scoring** (0–100) menggantikan sinyal biner, lebih tahan noise.
- **Backtest engine** bawaan — validasi win rate sebelum jalan live.
- **Risk filter** otomatis berbasis R:R minimum & ATR-based SL/TP.
- **Async fetch** — scan lebih cepat untuk banyak simbol sekaligus.
- **Struktur modular** (`data/`, `indicators/`, `strategy/`, `risk/`, `notify/`, `backtest/`).
- **CI** (lint + unit test) otomatis lewat GitHub Actions.
- Notifikasi **Telegram** bawaan.

## Struktur Proyek

```
├── main.py                  # entry point scan
├── config.yaml               # semua parameter strategi
├── src/vsynapse/
│   ├── data/                 # Binance API client (async)
│   ├── indicators/           # EMA, RSI, MACD, Supertrend, ATR
│   ├── strategy/             # confluence scoring engine
│   ├── risk/                 # filter R:R & position sizing
│   ├── notify/                # Telegram notifier
│   └── backtest/              # backtest engine + summary stats
├── tests/                     # unit test (pytest)
└── .github/workflows/         # scan terjadwal + CI
```

## Menjalankan Lokal

```bash
pip install -r requirements.txt
PYTHONPATH=src python main.py --out synaptic_candidates.json
```

## Menjalankan Test

```bash
PYTHONPATH=src pytest -q
```

## Backtest

```python
from vsynapse.backtest.engine import backtest_symbol, summarize

trades = backtest_symbol(df, "BTCUSDT", cfg)
print(summarize(trades))
```

## Konfigurasi

Semua bobot indikator, threshold skor, dan aturan risk ada di `config.yaml` —
tidak perlu ubah kode untuk tuning strategi.

## Otomatisasi

Workflow `.github/workflows/scan.yml` menjalankan scan tiap 30 menit via
GitHub Actions (gratis, tanpa server). Set secret `TELEGRAM_BOT_TOKEN` dan
`TELEGRAM_CHAT_ID` di repo settings untuk mengaktifkan notifikasi.

---

> ⚠️ Untuk riset & edukasi saja. **Not Financial Advice (NFA).**
