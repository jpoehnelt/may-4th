"""
transform_with_claude.py — Build a high-quality Yoda OSV dataset by:

1. Loading natural English sentences from sentence-transformers/all-nli
2. Batching them through Claude Haiku via the claude CLI to get Yoda translations
3. Merging with dvgodoy/yoda_sentences (gold hand-crafted pairs)
4. Writing the result to data/yoda_osv.jsonl

Usage:
    uv run python data/transform_with_claude.py --n 5000 --batch 40
"""

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset

PROMPT_TEMPLATE = """Translate each English sentence to Yoda OSV speaking style. \
Front the object or prepositional phrase, sometimes invert verb-subject, \
sometimes append "did" or "is" at the end. Keep the meaning identical.

Output ONLY a numbered list of Yoda translations matching the input numbering. \
No commentary, no explanations, no filler words like "hmm" or "yes".

{numbered}"""


def call_claude(numbered_text: str, retries: int = 2) -> str:
    """Call claude CLI with Haiku. Returns stdout text."""
    for attempt in range(retries + 1):
        try:
            result = subprocess.run(
                ["claude", "--model", "haiku", "-p", PROMPT_TEMPLATE.format(numbered=numbered_text)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
            if attempt < retries:
                time.sleep(2)
                continue
            print(f"  CLI failed (rc={result.returncode}): {result.stderr[:200]}", file=sys.stderr)
            return ""
        except subprocess.TimeoutExpired:
            if attempt < retries:
                time.sleep(2)
                continue
            print("  CLI timeout", file=sys.stderr)
            return ""
    return ""


def parse_numbered_output(text: str, expected: int) -> list[str]:
    """Parse '1. line\n2. line\n...' style output. Returns list of length up to expected."""
    out: list[str | None] = [None] * expected
    pattern = re.compile(r"^\s*(\d+)[\.\)]\s*(.+?)\s*$")
    for line in text.splitlines():
        m = pattern.match(line)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if 0 <= idx < expected and out[idx] is None:
            out[idx] = m.group(2)
    return [o for o in out if o]


def load_natural_sentences(n: int, min_len: int = 30, max_len: int = 100) -> list[str]:
    """Pull diverse natural English sentences from sentence-transformers/all-nli."""
    print(f"Loading {n} sentences from sentence-transformers/all-nli...")
    ds = load_dataset("sentence-transformers/all-nli", "pair", split="train", streaming=True)
    seen = set()
    out = []
    for x in ds:
        for key in ("anchor", "positive"):
            s = x.get(key, "").strip()
            if not s or len(s) < min_len or len(s) > max_len:
                continue
            # require a final period for cleanliness
            if not s.endswith("."):
                s = s + "."
            # require alphabetic and basic structure
            if not s[0].isalpha():
                continue
            key_l = s.lower()
            if key_l in seen:
                continue
            seen.add(key_l)
            out.append(s)
            if len(out) >= n:
                return out
    return out


def load_dvgodoy_pairs() -> list[dict]:
    """Load the gold hand-crafted yoda_sentences dataset."""
    print("Loading dvgodoy/yoda_sentences...")
    ds = load_dataset("dvgodoy/yoda_sentences")["train"]
    pairs = []
    for x in ds:
        en = x["sentence"].strip()
        yoda = x["translation"].strip()
        if en and yoda:
            pairs.append({"en": en, "yoda": yoda})
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000, help="number of all-nli sentences to transform")
    parser.add_argument("--batch", type=int, default=40, help="sentences per claude call")
    parser.add_argument("--workers", type=int, default=6, help="parallel claude invocations")
    parser.add_argument("--out", default="data/yoda_osv.jsonl")
    args = parser.parse_args()

    sentences = load_natural_sentences(args.n)
    print(f"Loaded {len(sentences)} natural sentences.")

    # Split into batches
    batches = [sentences[i:i + args.batch] for i in range(0, len(sentences), args.batch)]
    print(f"Split into {len(batches)} batches; running {args.workers} parallel claude workers.")

    def run_one(batch):
        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(batch))
        text = call_claude(numbered)
        translations = parse_numbered_output(text, len(batch))
        pairs = []
        for en, yoda in zip(batch, translations):
            yoda = yoda.strip()
            if not yoda or yoda.lower() == en.lower():
                continue
            pairs.append({"en": en, "yoda": yoda})
        return len(batch), len(translations), pairs

    out_pairs: list[dict] = []
    t0 = time.time()
    done_batches = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, b): i for i, b in enumerate(batches)}
        for fut in as_completed(futures):
            try:
                in_n, out_n, pairs = fut.result()
            except Exception as e:
                print(f"  batch failed: {e}", file=sys.stderr)
                continue
            out_pairs.extend(pairs)
            done_batches += 1
            elapsed = time.time() - t0
            rate = done_batches / elapsed if elapsed > 0 else 0
            eta = (len(batches) - done_batches) / rate if rate > 0 else 0
            print(f"  batch {done_batches}/{len(batches)}: got {out_n}/{in_n} "
                  f"(total pairs: {len(out_pairs)}, {rate:.2f} batch/s, eta {eta:.0f}s)")

    gold = load_dvgodoy_pairs()
    print(f"Loaded {len(gold)} gold pairs from dvgodoy/yoda_sentences.")

    seen_en = set()
    final = []
    for pair in out_pairs + gold:
        key = pair["en"].lower()
        if key in seen_en:
            continue
        seen_en.add(key)
        final.append(pair)

    out_path = Path(args.out)
    with out_path.open("w") as f:
        for p in final:
            f.write(json.dumps(p) + "\n")
    print(f"\nWrote {len(final)} unique pairs to {out_path}")


if __name__ == "__main__":
    main()
