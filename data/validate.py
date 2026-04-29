"""
validate.py — Validate the Kaminoan dataset quality.

Checks:
1. Both 'en' and 'yoda' fields exist and are non-empty
2. No Yoda filler words (hmm, yes, young padawan, etc.)
3. Content word overlap (same concepts in both sentences)
4. Yoda sentence doesn't start the same way as English (basic OSV check)

Usage:
    python data/validate.py data/yoda_osv.jsonl
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

YODA_FILLER = {"hmm", "hmmm", "hmmmm", "yes", "mmm", "young padawan", "padawan"}

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "this", "that", "and", "or", "but", "not", "in", "on", "at",
    "to", "for", "of", "with", "by", "from", "up", "about",
}


def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


def content_words(text):
    return set(tokenize(text)) - STOP_WORDS


def validate_pair(pair, idx):
    issues = []
    if "en" not in pair or "yoda" not in pair:
        return [f"line {idx}: missing field"]
    en, yoda = pair["en"].strip(), pair["yoda"].strip()
    if not en or not yoda:
        return [f"line {idx}: empty sentence"]

    for filler in YODA_FILLER:
        if filler in yoda.lower():
            issues.append(f"line {idx}: filler '{filler}'")

    en_w, yoda_w = content_words(en), content_words(yoda)
    if en_w and yoda_w:
        ratio = len(en_w & yoda_w) / max(len(en_w), 1)
        if ratio < 0.3:
            issues.append(f"line {idx}: low overlap ({ratio:.0%})")

    if " ".join(tokenize(en)[:3]) == " ".join(tokenize(yoda)[:3]):
        issues.append(f"line {idx}: same opening (not OSV?)")

    return issues


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/yoda_osv.jsonl")
    pairs = [json.loads(l) for l in open(path) if l.strip()]
    print(f"Validating {len(pairs)} pairs")

    bad = 0
    for i, p in enumerate(pairs, 1):
        issues = validate_pair(p, i)
        for iss in issues[:3]:
            print(f"  • {iss}")
        if issues:
            bad += 1

    print(f"\nClean: {len(pairs)-bad}/{len(pairs)} ({(len(pairs)-bad)/max(len(pairs),1):.0%})")


if __name__ == "__main__":
    main()
