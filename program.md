# Autonomous Research Program: Yoda OSV Language Model

## Objective

Minimize `val_bpb` (validation bits per byte) on the Kaminoan dataset — English-to-Yoda Object-Subject-Verb syntax pairs.

## Instructions

You are an autonomous ML research agent. Follow this loop **repeatedly** until told to stop:

### 1. Check State

- Read `train.py` to understand the current architecture and hyperparameters.
- Read `experiments.jsonl` to see what has been tried and what worked.
- The current best `val_bpb` is the lowest value from any `"result": "IMPROVED"` entry (or the baseline).

### 2. Form a Hypothesis

State a 1–2 sentence hypothesis about a change to `train.py` that you believe will lower `val_bpb`. Be specific. Base it on what you've learned from prior experiments.

### 3. Modify `train.py`

Apply your proposed change. You may ONLY modify `train.py`. Never touch `prepare.py` or this file.

Constraints:

- `D_MODEL` must be divisible by `N_HEADS`
- Keep the same interface: `GPT` class with `param_count()`, `train()` function
- Maintain imports from `prepare.py`

### 4. Train

```bash
uv run python train.py
```

Training is budget-capped at `TRAIN_BUDGET_SEC` (set in `prepare.py`). Do not change it.
Look for the `RESULT val_bpb=X.XXXX` line in stdout.

### 5. Log the Experiment

**Always** append one JSON line to `experiments.jsonl` before committing or reverting:

```json
{"exp": NNN, "hypothesis": "...", "val_bpb": X.XXXX, "result": "IMPROVED|REGRESSED", "changes": "brief description of what changed"}
```

### 6. Evaluate & Decide

**If val_bpb IMPROVED** (lower than current best):

```bash
git add train.py experiments.jsonl && git commit -m "exp-NNN: <brief description>, val_bpb X.XX → Y.YY"
```

**If val_bpb REGRESSED** (higher or equal):

```bash
git checkout -- train.py
git add experiments.jsonl && git commit -m "exp-NNN: REGRESSED — <brief description>"
```

### 7. Repeat

Go back to step 1. Keep going until told to stop.

## Data Generation (optional, periodic)

If val_bpb has plateaued for 3+ experiments, expand the dataset: `data/yoda_osv.jsonl`.Then commit the expanded data and retrain.

## Files

| File                  | Role                                           | Mutable?               |
| --------------------- | ---------------------------------------------- | ---------------------- |
| `train.py`            | Model architecture + training loop             | ✅ You modify this     |
| `prepare.py`          | Constants, tokenizer, data loading, evaluation | ❌ Do not touch        |
| `infer.py`            | Interactive TUI for inference                  | ❌ Do not touch        |
| `data/yoda_osv.jsonl` | Training dataset                               | ✅ Expand via data gen |
| `data/prompt.md`      | Prompt for synthetic data generation           | ❌ Do not touch        |
| `data/generate.py`    | Dataset clean/dedup utility                    | ❌ Do not touch        |
| `experiments.jsonl`   | Experiment log (git-tracked)                   | ✅ You append to this  |
| `program.md`          | This file                                      | ❌ Do not touch        |

## Constraints

- Hardware: Apple Silicon (MPS backend), limited memory
- Dataset: ~3,000 pairs (~53K tokens)
- Overfitting is the primary risk (more params than tokens)
- Metric: `val_bpb` (lower = better)
