"""
prepare.py — Constants, data prep, tokenizer, dataloader, evaluation.
"""

import os
import re
import json
import time
import struct
import random
import numpy as np
import torch

# ---- Constants ----
MAX_SEQ_LEN = 256          # Yoda sentences are short
VOCAB_SIZE = 4096          # Larger vocab → more whole-word tokens
EVAL_TOKENS = 50_000        # Keep eval fast
TRAIN_BUDGET_SEC = 600       # 10 min per experiment
DATA_PATH = "data/yoda_osv.jsonl"
TOKENIZER_PATH = "data/tokenizer.json"
TRAIN_BIN = "data/train.bin"
VAL_BIN = "data/val.bin"
CHECKPOINT_DIR = "checkpoints"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ---- Special tokens (reserved at top of vocab) ----
PAD_ID = VOCAB_SIZE - 3   # 4093
BOS_ID = VOCAB_SIZE - 2   # 4094
EOS_ID = VOCAB_SIZE - 1   # 4095
SPECIAL_TOKEN_IDS = {PAD_ID, BOS_ID, EOS_ID}

# ---- Pre-tokenization regex (GPT-2 style, simplified) ----
# Splits text at word/punct/whitespace boundaries so BPE merges never cross them.
PRE_TOK_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z\d]+|\s+"
)

# ---- Byte-Pair Encoding Tokenizer (minimal) ----

class ByteLevelTokenizer:
    """
    Dead-simple byte-level tokenizer with BPE merges.
    For a tiny model on short sentences, byte-level with small vocab is fine.
    """

    def __init__(self, vocab_size=VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.merges = {}       # (int, int) -> int
        self.vocab = {}        # int -> bytes

    def _build_base_vocab(self):
        """256 byte-level tokens + 3 special tokens at the top of the vocab."""
        self.vocab = {i: bytes([i]) for i in range(256)}
        # Special tokens: produce no bytes when decoded
        self.vocab[PAD_ID] = b""
        self.vocab[BOS_ID] = b""
        self.vocab[EOS_ID] = b""

    def _pretokenize(self, text):
        """Split text at word/punct boundaries; return list of byte-id lists."""
        return [list(m.group().encode("utf-8")) for m in PRE_TOK_PATTERN.finditer(text)]

    def train(self, texts, num_merges=None):
        """Train BPE merges with pre-tokenization; merges never cross word boundaries."""
        if num_merges is None:
            num_merges = self.vocab_size - 256 - 3  # reserve 3 special tokens

        self._build_base_vocab()

        # Pre-tokenize everything into chunks (each chunk = list of byte ids).
        # Pairs are counted only WITHIN chunks, so e.g. " the" stays "the" but
        # "through·the" can never merge across the space.
        chunks = []
        for text in texts:
            chunks.extend(self._pretokenize(text))

        for i in range(num_merges):
            pair_counts = {}
            for seq in chunks:
                for j in range(len(seq) - 1):
                    pair = (seq[j], seq[j + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            new_id = 256 + i
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            for k, seq in enumerate(chunks):
                new_seq = []
                j = 0
                while j < len(seq):
                    if j < len(seq) - 1 and (seq[j], seq[j + 1]) == best_pair:
                        new_seq.append(new_id)
                        j += 2
                    else:
                        new_seq.append(seq[j])
                        j += 1
                chunks[k] = new_seq

            if (i + 1) % 200 == 0:
                print(f"  BPE merge {i + 1}/{num_merges}")

        print(f"Tokenizer trained: {len(self.vocab)} tokens, {len(self.merges)} merges")

    def encode(self, text):
        """Encode string with pre-tokenization, applying BPE merges within each chunk."""
        out = []
        for chunk_ids in self._pretokenize(text):
            ids = chunk_ids[:]
            while True:
                best_pair = None
                best_idx = float("inf")
                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i + 1])
                    if pair in self.merges and self.merges[pair] < best_idx:
                        best_pair = pair
                        best_idx = self.merges[pair]
                if best_pair is None:
                    break
                new_ids = []
                i = 0
                while i < len(ids):
                    if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best_pair:
                        new_ids.append(self.merges[best_pair])
                        i += 2
                    else:
                        new_ids.append(ids[i])
                        i += 1
                ids = new_ids
            out.extend(ids)
        return out

    def decode(self, ids):
        """Decode list of token IDs to string. Special tokens render as empty."""
        raw = b"".join(self.vocab.get(i, b"?") for i in ids)
        return raw.decode("utf-8", errors="replace")

    def save(self, path):
        """Save tokenizer to JSON."""
        data = {
            "vocab_size": self.vocab_size,
            "merges": {f"{a},{b}": v for (a, b), v in self.merges.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        """Load tokenizer from JSON."""
        with open(path) as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self._build_base_vocab()
        self.merges = {}
        for k, v in data["merges"].items():
            a, b = k.split(",")
            pair = (int(a), int(b))
            self.merges[pair] = int(v)
            self.vocab[int(v)] = self.vocab[pair[0]] + self.vocab[pair[1]]


# ---- Data Preparation ----

def load_raw_data(path=DATA_PATH):
    """Load JSONL dataset, return raw English + Yoda sentences (separately).
    Each sentence is its own training text — the tokenizer never learns merges
    across the en/yoda boundary, which would waste vocab.
    """
    texts = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["en"])
            texts.append(row["yoda"])
    return texts


def prepare_data():
    """One-time data preparation: train tokenizer, write binary shards."""
    print("=" * 60)
    print("PREPARING DATA")
    print("=" * 60)

    texts = load_raw_data()
    random.shuffle(texts)
    print(f"Loaded {len(texts)} examples from {DATA_PATH}")

    # Train tokenizer
    print("\nTraining tokenizer...")
    tok = ByteLevelTokenizer(vocab_size=VOCAB_SIZE)
    tok.train(texts)
    os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
    tok.save(TOKENIZER_PATH)

    # Tokenize and split
    print("\nTokenizing...")
    all_ids = []
    for text in texts:
        all_ids.extend(tok.encode(text))

    total = len(all_ids)
    val_size = min(EVAL_TOKENS, total // 10)
    train_ids = all_ids[:total - val_size]
    val_ids = all_ids[total - val_size:]

    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens:   {len(val_ids):,}")

    # Write to binary (uint16)
    os.makedirs(os.path.dirname(TRAIN_BIN), exist_ok=True)
    with open(TRAIN_BIN, "wb") as f:
        for tid in train_ids:
            f.write(struct.pack("<H", tid))
    with open(VAL_BIN, "wb") as f:
        for tid in val_ids:
            f.write(struct.pack("<H", tid))

    print(f"\nWrote {TRAIN_BIN} ({os.path.getsize(TRAIN_BIN):,} bytes)")
    print(f"Wrote {VAL_BIN} ({os.path.getsize(VAL_BIN):,} bytes)")
    print("Data preparation complete.")


# ---- Runtime Utilities (used by train.py) ----

def get_tokenizer():
    """Load the pre-trained tokenizer."""
    tok = ByteLevelTokenizer(vocab_size=VOCAB_SIZE)
    tok.load(TOKENIZER_PATH)
    return tok


def get_data(split="train"):
    """Load tokenized data from binary shard. Returns numpy array of uint16."""
    path = TRAIN_BIN if split == "train" else VAL_BIN
    with open(path, "rb") as f:
        raw = f.read()
    return np.frombuffer(raw, dtype=np.uint16)


def get_batch(data, batch_size, seq_len=MAX_SEQ_LEN, device=DEVICE):
    """Sample a random batch from the data."""
    # Clamp seq_len to fit available data
    seq_len = min(seq_len, len(data) - 2)
    if seq_len < 1:
        raise ValueError(f"Data too small ({len(data)} tokens) for any batch")
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+seq_len].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+seq_len].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, val_data, batch_size=8, seq_len=MAX_SEQ_LEN, device=DEVICE):
    """
    Evaluate model on validation data.
    Returns val_bpb (bits per byte) — lower is better, vocab-size-independent.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    tokens_evaluated = 0

    # Cap eval at available data
    eval_target = min(EVAL_TOKENS, len(val_data) * 4)

    while tokens_evaluated < eval_target:
        x, y = get_batch(val_data, batch_size, seq_len, device)
        logits = model(x)  # (B, T, vocab_size)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum",
        )
        n = y.numel()
        total_loss += loss.item()
        total_tokens += n
        tokens_evaluated += n

    # Convert from nats-per-token to bits-per-byte
    # bpb = (nats_per_token / ln(2)) * (avg_tokens_per_byte)
    # For byte-level BPE, avg_tokens_per_byte ≈ 1 (depends on compression)
    nats_per_token = total_loss / total_tokens
    bpb = nats_per_token / np.log(2)
    model.train()
    return bpb


def save_checkpoint(model, optimizer=None, step=0, val_bpb=0.0, path=None):
    """Save model checkpoint."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if path is None:
        path = os.path.join(CHECKPOINT_DIR, "best.pt")
        # Check if we are actually improving
        if os.path.exists(path):
            old_state = torch.load(path, map_location="cpu", weights_only=False)
            best_val = old_state.get("val_bpb", float("inf"))
            if val_bpb >= best_val:
                print(f"Not saving checkpoint. val_bpb ({val_bpb:.4f}) >= best ({best_val:.4f})")
                return

    state = {
        "model": model.state_dict(),
        "step": step,
        "val_bpb": val_bpb,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path)
    print(f"Checkpoint saved to {path} (val_bpb={val_bpb:.4f})")


def load_checkpoint(model, path=None, device=DEVICE):
    """Load model checkpoint. Returns (step, val_bpb)."""
    if path is None:
        path = os.path.join(CHECKPOINT_DIR, "best.pt")
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    return state.get("step", 0), state.get("val_bpb", float("inf"))


# ---- Main: run data prep if executed directly ----

if __name__ == "__main__":
    prepare_data()
