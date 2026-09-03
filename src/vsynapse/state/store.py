"""Cooldown/dedup: hindari kirim sinyal berulang untuk simbol+arah yang sama
selama masih dalam jendela cooldown. State disimpan ke file JSON yang
di-commit balik ke repo tiap scan (GitHub Actions runner stateless antar run,
jadi tanpa ini state akan selalu kosong tiap kali workflow jalan).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone


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
    """True kalau simbol+arah ini sudah pernah disinyalin dalam jendela cooldown."""
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
