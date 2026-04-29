"""
train.py — Model architecture + training loop.
THIS FILE IS MODIFIED BY THE AI AGENT. Do not edit manually.

Starting point: tiny GPT (4-layer transformer, 256 dim).
The agent will mutate architecture, hyperparameters, and optimizer.
"""

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    VOCAB_SIZE, MAX_SEQ_LEN, TRAIN_BUDGET_SEC, DEVICE,
    get_data, get_batch, evaluate, save_checkpoint,
)

# ---- Hyperparameters (agent tunes these) ----
DEPTH = 4                    # Number of transformer blocks
D_MODEL = 128                # Model dimension
N_HEADS = 4                  # Number of attention heads
D_FF_MULT = 4                # FFN expansion factor
DROPOUT = 0.3                # Dropout rate
LEARNING_RATE = 3e-4         # Peak learning rate
WEIGHT_DECAY = 0.1           # AdamW weight decay
WARMUP_STEPS = 100           # LR warmup steps
DEVICE_BATCH_SIZE = 16       # Micro-batch size
TOTAL_BATCH_SIZE = 2**14     # ~16K tokens per gradient step
LOG_INTERVAL = 10            # Print loss every N steps


# ---- Model Architecture ----

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # each (B, T, nh, hd)
        q = q.transpose(1, 2)    # (B, nh, T, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention with causal mask
        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.out(y)


class FFN(nn.Module):
    def __init__(self, d_model, mult=4):
        super().__init__()
        d_ff = d_model * mult
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # SwiGLU gate

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff_mult=4, dropout=0.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = RMSNorm(d_model)
        self.ffn = FFN(d_model, d_ff_mult)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.resid_drop(self.attn(self.ln1(x)))
        x = x + self.resid_drop(self.ffn(self.ln2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, seq_len=MAX_SEQ_LEN,
                 d_model=D_MODEL, n_heads=N_HEADS, depth=DEPTH,
                 d_ff_mult=D_FF_MULT, dropout=DROPOUT):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.emb_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, d_ff_mult, dropout) for _ in range(depth)
        ])
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.tok_emb.weight
        self.seq_len = seq_len
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.emb_drop(self.tok_emb(x) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ---- Training Loop ----

def train():
    print("=" * 60)
    print("TRAINING")
    print(f"Device: {DEVICE}")
    print(f"Architecture: GPT depth={DEPTH} d_model={D_MODEL} heads={N_HEADS}")
    print(f"Budget: {TRAIN_BUDGET_SEC}s")
    print("=" * 60)

    # Load data
    train_data = get_data("train")
    val_data = get_data("val")
    print(f"Train tokens: {len(train_data):,}  Val tokens: {len(val_data):,}")

    # Build model
    model = GPT().to(DEVICE)
    n_params = model.param_count()
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    # Gradient accumulation
    grad_accum_steps = max(1, TOTAL_BATCH_SIZE // (DEVICE_BATCH_SIZE * MAX_SEQ_LEN))
    print(f"Gradient accumulation: {grad_accum_steps} steps")

    # Training
    model.train()
    step = 0
    best_val_bpb = float("inf")
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed >= TRAIN_BUDGET_SEC:
            break

        # Learning rate schedule: linear warmup + cosine decay
        if step < WARMUP_STEPS:
            lr = LEARNING_RATE * (step + 1) / WARMUP_STEPS
        else:
            progress = (elapsed / TRAIN_BUDGET_SEC)
            lr = LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Accumulate gradients
        optimizer.zero_grad()
        total_loss = 0.0
        for _ in range(grad_accum_steps):
            x, y = get_batch(train_data, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, DEVICE)
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            ) / grad_accum_steps
            loss.backward()
            total_loss += loss.item()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

        if step % LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            remaining = TRAIN_BUDGET_SEC - elapsed
            print(f"step={step:4d}  loss={total_loss:.4f}  lr={lr:.2e}  "
                  f"elapsed={elapsed:.0f}s  remaining={remaining:.0f}s")

    # Final evaluation
    elapsed = time.time() - start_time
    val_bpb = evaluate(model, val_data, device=DEVICE)
    print(f"\n{'=' * 60}")
    print(f"RESULT val_bpb={val_bpb:.4f}")
    print(f"Training time: {elapsed:.1f}s  Steps: {step}")
    print(f"{'=' * 60}")

    # Save checkpoint
    save_checkpoint(model, optimizer, step, val_bpb)

    return val_bpb


if __name__ == "__main__":
    train()
