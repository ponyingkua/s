"""Kirim sinyal ke Telegram. Token & chat_id diambil dari environment variable
(disetel via GitHub Actions secrets), bukan hardcode di kode.
"""
from __future__ import annotations

import os

import aiohttp

from vsynapse.strategy.scoring import SignalResult


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

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            await resp.read()
