# Kaminoan — Autonomous ML Research Demo

> May the force be with your validation loss.

An autonomous ML research loop, inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch), that trains a tiny language model to speak in Yoda's Object-Subject-Verb syntax — on Apple Silicon.

## How It Works

An AI agent reads `program.md`, modifies `train.py`, runs a 5-minute training experiment, and commits or reverts based on whether `val_bpb` improved. Repeat overnight.

```
program.md  →  Agent reads instructions
train.py    →  Agent modifies architecture/hyperparams
prepare.py  →  Fixed data pipeline (never modified)
infer.py    →  TUI inference after training
```

## Quick Start

```bash
# Install dependencies
uv sync

# Generate more with Claude CLI
claude -p < data/prompt.md

# Prepare data (train tokenizer, write binary shards)
uv run python prepare.py

# Run a single training experiment (5 min)
uv run python train.py

# Run inference on the trained model
uv run python infer.py
```

## Running the Agent Loop

Point your coding agent at this repo and say:

> Have a look at program.md and let's kick off a new experiment.

The agent will autonomously:

1. Read `program.md` for instructions
2. Modify `train.py` (architecture, hyperparams, optimizer)
3. Run training (5-min budget)
4. Check val_bpb, commit if improved, revert if not
5. Repeat

## Files

| File               | Modified By | Purpose                                 |
| ------------------ | ----------- | --------------------------------------- |
| `prepare.py`       | Nobody      | Data loading, tokenizer, evaluation     |
| `train.py`         | The agent   | Model architecture + training loop      |
| `program.md`       | You         | Agent instructions                      |
| `infer.py`         | Nobody      | Terminal inference UI (rich)            |
| `data/generate.py` | Nobody      | Seed pairs + clean/dedupe/stats helpers |
| `data/validate.py` | Nobody      | Dataset quality checks                  |
