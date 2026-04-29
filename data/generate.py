"""
generate.py — Clean and stats helpers for the Kaminoan dataset.

Usage:
    python data/generate.py clean [file]     # deduplicate + fix formatting
    python data/generate.py stats [file]     # show dataset stats
"""

import json
import sys
from pathlib import Path


def clean(path):
    """Deduplicate, fix formatting, remove bad lines."""
    raw_lines = Path(path).read_text().strip().split("\n")
    seen_en = set()
    clean_pairs = []
    bad = 0

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue

        try:
            pair = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue

        if "en" not in pair or "yoda" not in pair:
            bad += 1
            continue

        en = pair["en"].strip()
        yoda = pair["yoda"].strip()

        if not en or not yoda:
            bad += 1
            continue

        key = en.lower()
        if key in seen_en:
            continue
        seen_en.add(key)

        clean_pairs.append({"en": en, "yoda": yoda})

    with open(path, "w") as f:
        for pair in clean_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Cleaned: {len(raw_lines)} → {len(clean_pairs)} pairs ({bad} bad, {len(raw_lines) - len(clean_pairs) - bad} dupes)")


def stats(path):
    """Show dataset statistics."""
    pairs = [json.loads(l) for l in open(path) if l.strip()]
    en_lens = [len(p["en"].split()) for p in pairs]
    yoda_lens = [len(p["yoda"].split()) for p in pairs]

    print(f"Total pairs:     {len(pairs)}")
    print(f"Avg EN length:   {sum(en_lens)/max(len(en_lens),1):.1f} words")
    print(f"Avg Yoda length: {sum(yoda_lens)/max(len(yoda_lens),1):.1f} words")
    print(f"Total chars:     {sum(len(p['en']) + len(p['yoda']) for p in pairs):,}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]
    f = sys.argv[2] if len(sys.argv) > 2 else "data/yoda_osv.jsonl"

    if cmd == "clean":
        clean(f)
    elif cmd == "stats":
        stats(f)
    else:
        print(f"Unknown command: {cmd}")
