"""
train.py — Encoder-Decoder transformer for Yoda OSV translation.

Architecture pivot (4-hour demo): the prior decoder-only GPT had to regenerate
every word through softmax. Translation is ~90% reorder, so we now use a tiny
T5-style encoder-decoder where the decoder cross-attends to the source.

- Encoder: N bidirectional self-attention layers reading the English source
- Decoder: N (causal self-attn + cross-attn + FFN) layers producing Yoda
- Shared token embedding (also weight-tied to LM head)
- BOS = PAD = id 0 (byte 0x00, never appears in training); EOS = newline
"""

import json
import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from prepare import (
    VOCAB_SIZE, MAX_SEQ_LEN, TRAIN_BUDGET_SEC, DEVICE, DATA_PATH,
    PAD_ID, BOS_ID, EOS_ID,
    get_tokenizer, save_checkpoint,
)

# ---- Hyperparameters ----
DEPTH_ENC = 2
DEPTH_DEC = 3
D_MODEL = 64
N_HEADS = 2
D_FF_MULT = 2
DROPOUT = 0.2
LABEL_SMOOTHING = 0.0
LEARNING_RATE = 7e-4
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 300
DEVICE_BATCH_SIZE = 32
LOG_INTERVAL = 20

# Per-side max length (pairs are short — 99% under 32 tokens each)
MAX_SRC_LEN = 64
MAX_TGT_LEN = 64

VAL_FRAC = 0.1


# ---- Model components ----

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class MultiHeadAttention(nn.Module):
    """Generic attention; same module is used for self and cross-attention."""

    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, q_in, k_in, v_in, key_padding_mask=None, is_causal=False):
        B, Tq, C = q_in.shape
        Tk = k_in.size(1)
        q = self.q_proj(q_in).reshape(B, Tq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k_in).reshape(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v_in).reshape(B, Tk, self.n_heads, self.head_dim).transpose(1, 2)

        # key_padding_mask: (B, Tk) bool, True where PAD
        attn_mask = None
        if key_padding_mask is not None:
            # shape (B, 1, 1, Tk), broadcast over heads and queries
            attn_mask = key_padding_mask[:, None, None, :].to(torch.bool)
            # SDPA expects True = mask (block); convert to additive mask via 0/-inf
            attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill_(attn_mask, float("-inf"))

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).reshape(B, Tq, C)
        return self.out(y)


class FFN(nn.Module):
    def __init__(self, d_model, mult=4):
        super().__init__()
        d_ff = d_model * mult
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff_mult=4, dropout=0.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = RMSNorm(d_model)
        self.ffn = FFN(d_model, d_ff_mult)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_pad_mask):
        h = self.ln1(x)
        x = x + self.drop(self.attn(h, h, h, key_padding_mask=src_pad_mask, is_causal=False))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff_mult=4, dropout=0.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = RMSNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln3 = RMSNorm(d_model)
        self.ffn = FFN(d_model, d_ff_mult)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_pad_mask):
        h = self.ln1(x)
        x = x + self.drop(self.self_attn(h, h, h, is_causal=True))
        h = self.ln2(x)
        x = x + self.drop(self.cross_attn(h, enc_out, enc_out, key_padding_mask=src_pad_mask))
        x = x + self.drop(self.ffn(self.ln3(x)))
        return x


class EncoderDecoder(nn.Module):
    def __init__(self,
                 vocab_size=VOCAB_SIZE,
                 d_model=D_MODEL,
                 n_heads=N_HEADS,
                 depth_enc=DEPTH_ENC,
                 depth_dec=DEPTH_DEC,
                 d_ff_mult=D_FF_MULT,
                 dropout=DROPOUT,
                 max_src_len=MAX_SRC_LEN,
                 max_tgt_len=MAX_TGT_LEN):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.src_pos = nn.Embedding(max_src_len, d_model)
        self.tgt_pos = nn.Embedding(max_tgt_len, d_model)
        self.emb_drop = nn.Dropout(dropout)

        self.encoder = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff_mult, dropout) for _ in range(depth_enc)
        ])
        self.enc_ln = RMSNorm(d_model)

        self.decoder = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff_mult, dropout) for _ in range(depth_dec)
        ])
        self.dec_ln = RMSNorm(d_model)

        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def encode(self, src, src_pad_mask):
        B, T = src.shape
        pos = torch.arange(T, device=src.device).unsqueeze(0)
        x = self.emb_drop(self.tok_emb(src) + self.src_pos(pos))
        for blk in self.encoder:
            x = blk(x, src_pad_mask)
        return self.enc_ln(x)

    def decode(self, tgt, enc_out, src_pad_mask):
        B, T = tgt.shape
        pos = torch.arange(T, device=tgt.device).unsqueeze(0)
        x = self.emb_drop(self.tok_emb(tgt) + self.tgt_pos(pos))
        for blk in self.decoder:
            x = blk(x, enc_out, src_pad_mask)
        x = self.dec_ln(x)
        return self.head(x)

    def forward(self, src, tgt, src_pad_mask=None):
        if src_pad_mask is None:
            src_pad_mask = (src == PAD_ID)
        enc_out = self.encode(src, src_pad_mask)
        return self.decode(tgt, enc_out, src_pad_mask)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ---- Data ----

def load_pairs():
    """Read yoda_osv.jsonl, return list of (en_text, yoda_text)."""
    pairs = []
    with open(DATA_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            en = obj.get("en", "").strip()
            yoda = obj.get("yoda", "").strip()
            if en and yoda:
                pairs.append((en, yoda))
    return pairs


def tokenize_pairs(pairs, tokenizer):
    """Tokenize each pair. Returns list of (src_ids, tgt_ids) where:
    src_ids = en_tokens + [EOS]
    tgt_ids = [BOS] + yoda_tokens + [EOS]
    """
    out = []
    for en, yoda in pairs:
        s = tokenizer.encode(en) + [EOS_ID]
        t = [BOS_ID] + tokenizer.encode(yoda) + [EOS_ID]
        if len(s) > MAX_SRC_LEN or len(t) > MAX_TGT_LEN:
            continue
        out.append((s, t))
    return out


def make_batch(pairs_subset, device):
    """Pad a list of (src, tgt) into tensors. Returns (src, tgt_in, tgt_out, src_pad_mask)."""
    B = len(pairs_subset)
    src_lens = [len(p[0]) for p in pairs_subset]
    tgt_lens = [len(p[1]) for p in pairs_subset]
    max_s = max(src_lens)
    max_t = max(tgt_lens)
    src = torch.full((B, max_s), PAD_ID, dtype=torch.long)
    tgt = torch.full((B, max_t), PAD_ID, dtype=torch.long)
    for i, (s, t) in enumerate(pairs_subset):
        src[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        tgt[i, :len(t)] = torch.tensor(t, dtype=torch.long)
    # decoder input = tgt[:-1]; decoder target = tgt[1:]
    tgt_in = tgt[:, :-1].contiguous()
    tgt_out = tgt[:, 1:].contiguous()
    src_pad_mask = (src == PAD_ID)
    return src.to(device), tgt_in.to(device), tgt_out.to(device), src_pad_mask.to(device)


def iter_batches(pairs, batch_size, device, shuffle=True):
    indices = list(range(len(pairs)))
    if shuffle:
        random.shuffle(indices)
    for i in range(0, len(indices), batch_size):
        batch = [pairs[j] for j in indices[i:i + batch_size]]
        if not batch:
            continue
        yield make_batch(batch, device)


# ---- Evaluation ----

@torch.no_grad()
def evaluate_seq2seq(model, val_pairs, val_text_bytes, device, batch_size=32):
    """Compute val_bpb on (src, tgt) pairs, normalised by total target byte length."""
    model.eval()
    total_loss_nats = 0.0
    total_tgt_tokens = 0
    for src, tgt_in, tgt_out, src_pad_mask in iter_batches(val_pairs, batch_size, device, shuffle=False):
        logits = model(src, tgt_in, src_pad_mask=src_pad_mask)
        # mask PAD positions in target
        valid = (tgt_out != PAD_ID)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
            reduction="none",
        ).reshape(tgt_out.shape)
        total_loss_nats += (loss * valid).sum().item()
        total_tgt_tokens += valid.sum().item()
    model.train()
    if total_tgt_tokens == 0 or val_text_bytes == 0:
        return float("inf")
    nats_per_token = total_loss_nats / total_tgt_tokens
    nats_per_byte = (total_loss_nats / val_text_bytes)
    bpb = nats_per_byte / np.log(2)
    return bpb


# ---- Training ----

def train():
    print("=" * 60)
    print("TRAINING (Encoder-Decoder)")
    print(f"Device: {DEVICE}")
    print(f"Architecture: enc={DEPTH_ENC} dec={DEPTH_DEC} d_model={D_MODEL} heads={N_HEADS}")
    print(f"Budget: {TRAIN_BUDGET_SEC}s")
    print("=" * 60)

    tokenizer = get_tokenizer()
    print(f"Special tokens: PAD={PAD_ID} BOS={BOS_ID} EOS={EOS_ID}  vocab={VOCAB_SIZE}")

    raw_pairs = load_pairs()
    print(f"Raw pairs: {len(raw_pairs)}")
    pairs = tokenize_pairs(raw_pairs, tokenizer)
    print(f"Pairs after length filter: {len(pairs)} (dropped {len(raw_pairs) - len(pairs)})")

    random.seed(42)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_FRAC))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    # Sum of target byte length for bpb normalisation (decode tgt body, sans BOS/EOS)
    val_text_bytes = 0
    for s, t in val_pairs:
        body = t[1:-1] if len(t) >= 2 else t
        decoded = tokenizer.decode(body)
        val_text_bytes += len(decoded.encode("utf-8"))

    print(f"Train pairs: {len(train_pairs)}  Val pairs: {len(val_pairs)}  Val target bytes: {val_text_bytes}")

    model = EncoderDecoder().to(DEVICE)
    n_params = model.param_count()
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    model.train()
    step = 0
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed >= TRAIN_BUDGET_SEC:
            break

        for src, tgt_in, tgt_out, src_pad_mask in iter_batches(train_pairs, DEVICE_BATCH_SIZE, DEVICE):
            elapsed = time.time() - start_time
            if elapsed >= TRAIN_BUDGET_SEC:
                break

            # LR schedule
            if step < WARMUP_STEPS:
                lr = LEARNING_RATE * (step + 1) / WARMUP_STEPS
            else:
                progress = elapsed / TRAIN_BUDGET_SEC
                lr = LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
            logits = model(src, tgt_in, src_pad_mask=src_pad_mask)
            valid = (tgt_out != PAD_ID)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1),
                reduction="none",
                label_smoothing=LABEL_SMOOTHING,
            ).reshape(tgt_out.shape)
            loss = (loss * valid).sum() / valid.sum().clamp(min=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

            if step % LOG_INTERVAL == 0:
                elapsed = time.time() - start_time
                remaining = TRAIN_BUDGET_SEC - elapsed
                print(f"step={step:5d}  loss={loss.item():.4f}  lr={lr:.2e}  "
                      f"elapsed={elapsed:.0f}s  remaining={remaining:.0f}s")

    elapsed = time.time() - start_time
    val_bpb = evaluate_seq2seq(model, val_pairs, val_text_bytes, DEVICE)
    print(f"\n{'=' * 60}")
    print(f"RESULT val_bpb={val_bpb:.4f}")
    print(f"Training time: {elapsed:.1f}s  Steps: {step}")
    print(f"{'=' * 60}")

    save_checkpoint(model, optimizer, step, val_bpb)

    print(f"\n{'=' * 60}")
    print("Exporting model to ONNX...")
    try:
        from export_onnx import export as export_onnx_model
        export_onnx_model()
        print("ONNX export complete.")
    except Exception as e:
        print(f"ONNX export failed: {e}")
    print(f"{'=' * 60}\n")

    return val_bpb


# Backward compat shim — infer.py still imports `GPT`. Provide an alias.
GPT = EncoderDecoder


if __name__ == "__main__":
    train()
