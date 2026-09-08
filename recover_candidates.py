"""
Merekonstruksi histori synaptic_candidates.json dari commit-commit git lama.

Kenapa perlu ini: sebelum perbaikan, tiap run scanner menimpa total file
synaptic_candidates.json dengan hasil run itu saja (bukan digabung). Tapi
karena tiap run di-commit ke git, versi lama tiap run masih ada di histori
git -- hanya perlu dikumpulkan dari banyak commit sekaligus.

CARA PAKAI:
1. Jalankan skrip ini dari root folder repo scanner-mu (folder yang ada
   git-nya), lokal di komputer -- bukan di sini, karena saya tidak punya
   akses ke repo GitHub-mu.
   Contoh: git clone <url-repo-mu> && cd <repo> && python recover_candidates.py

2. Skrip akan cetak jumlah commit yang ditemukan & entri hasil gabungan,
   lalu menulis hasilnya ke file terpisah:
   synaptic_candidates_recovered.json

3. Cek isinya, lalu kalau sudah oke, kamu bisa gabung manual ke
   synaptic_candidates.json yang sekarang (atau timpa langsung kalau mau).

CATATAN:
- Dedupe di sini kasar (berdasarkan symbol+timeframe+direction+entry+score).
  Kalau ada sinyal yang kebetulan identik persis di run berbeda, hanya
  disimpan sekali. Cek ulang hasilnya sebelum dipakai untuk backtest.
- Skrip ini hanya MEMBACA git log, tidak mengubah/menimpa apa pun di repo.
"""

import json
import subprocess
import sys

FILE_PATH = "synaptic_candidates.json"
OUTPUT_PATH = "synaptic_candidates_recovered.json"


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    )
    return result.stdout


def main() -> None:
    try:
        run_git("rev-parse", "--is-inside-work-tree")
    except subprocess.CalledProcessError:
        print("Bukan folder git repo. Jalankan skrip ini di dalam repo-nya.")
        sys.exit(1)

    log_output = run_git(
        "log", "--follow", "--diff-filter=AM", "--format=%H", "--", FILE_PATH
    )
    commits = [c for c in log_output.splitlines() if c.strip()]
    if not commits:
        print(f"Tidak ada commit yang menyentuh {FILE_PATH}. Cek nama/path filenya.")
        sys.exit(1)

    print(f"Ditemukan {len(commits)} commit yang menyentuh {FILE_PATH}.")

    seen = set()
    combined: list[dict] = []
    skipped_commits = 0

    for commit in commits:
        try:
            content = run_git("show", f"{commit}:{FILE_PATH}")
        except subprocess.CalledProcessError:
            skipped_commits += 1
            continue

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            skipped_commits += 1
            continue

        if not isinstance(data, list):
            skipped_commits += 1
            continue

        for entry in data:
            if not isinstance(entry, dict):
                continue
            key = (
                entry.get("symbol"),
                entry.get("timeframe"),
                entry.get("direction"),
                entry.get("entry"),
                entry.get("score"),
            )
            if key in seen:
                continue
            seen.add(key)
            entry_with_source = dict(entry)
            entry_with_source["_recovered_from_commit"] = commit
            combined.append(entry_with_source)

    if skipped_commits:
        print(f"({skipped_commits} commit dilewati -- file kosong/rusak/bukan list di commit itu)")

    print(f"Total entri unik hasil gabungan: {len(combined)}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(combined, f, indent=2, default=str)

    print(f"Tersimpan ke {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
