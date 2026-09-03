"""Filter sinyal berdasarkan aturan risk management sebelum dikirim/dieksekusi."""
from __future__ import annotations

from vsynapse.strategy.scoring import SignalResult


def passes_risk_filter(signal: SignalResult, cfg: dict) -> bool:
    if signal.direction == "NONE" or signal.entry is None:
        return False

    risk = abs(signal.entry - signal.sl)
    reward = abs(signal.tp - signal.entry)
    if risk == 0:
        return False

    rr = reward / risk
    return rr >= cfg["risk"]["risk_reward_min"]


def position_size(equity: float, risk_pct: float, entry: float, sl: float) -> float:
    """Hitung ukuran posisi (dalam unit aset) berdasarkan % risk dari equity."""
    risk_amount = equity * risk_pct
    per_unit_risk = abs(entry - sl)
    if per_unit_risk == 0:
        return 0.0
    return risk_amount / per_unit_risk
