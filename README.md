# Kaminoan — Browser-Based Yoda Translator

> May the force be with your validation loss.

**[Live Demo: yoda-translator-1ml.pages.dev](https://yoda-translator-1ml.pages.dev)**

Kaminoan is an autonomous ML research project that trains a tiny 525K parameter encoder-decoder transformer to translate English to Yoda's Object-Subject-Verb (OSV) syntax. The resulting model is exported to ONNX and runs entirely client-side in the browser using a Svelte & Astro frontend.

## 🚀 Architecture & Stack

- **Model:** 525K Parameter custom Encoder-Decoder Transformer (PyTorch)
- **Training:** Automated agentic research loop (inspired by Karpathy's autoresearch) running on Apple Silicon (MPS)
- **Inference Pipeline:** ONNX Runtime Web (`onnxruntime-web`)
- **Frontend:** Astro + Svelte 5 (with Runes for reactive UI state)
- **Deployment:** Cloudflare Pages via GitHub Actions
- **Data Source:** [sentence-transformers/all-nli](https://huggingface.co/datasets/sentence-transformers/all-nli) (filtered for training pairs)

---

## 🧠 Part 1: Autonomous ML Research Loop

An AI agent reads `program.md`, modifies `train.py`, runs a training experiment, and commits or reverts based on whether the validation bits-per-byte (`val_bpb`) improved. 

```
program.md  →  Agent reads instructions
train.py    →  Agent modifies architecture/hyperparams
prepare.py  →  Fixed data pipeline (never modified)
infer.py    →  TUI inference after training
```

### Running the Agent Loop

Point your coding agent at this repo and say:
> Have a look at program.md and let's kick off a new experiment.

The agent will autonomously:
1. Read `program.md` for instructions.
2. Modify `train.py` (architecture, hyperparams, optimizer).
3. Run training (e.g. 5-min budget).
4. Check `val_bpb`, commit if improved, revert if not.
5. Repeat.

---

## 🌐 Part 2: Browser Deployment

To make the model usable by anyone without a backend GPU, we export the PyTorch model to ONNX with embedded weights.

### Exporting the Model
```bash
uv run python export_onnx.py
```
This generates `encoder.onnx` and `decoder.onnx` into `web/public/models/`. Because the models are extremely small (~11MB combined), they are tracked directly in Git to avoid long CI build times.

### Running the Web Frontend Locally

The web frontend uses Astro and Svelte 5. It handles tokenization manually in JavaScript and streams inference via WebAssembly.

```bash
cd web
pnpm install
pnpm run dev
```

### Deployment

The application is automatically built and deployed to Cloudflare Pages via GitHub Actions on every push to the `main` branch. 

To deploy manually via the Wrangler CLI:
```bash
cd web
pnpm run build
pnpm exec wrangler pages deploy dist --project-name yoda-translator-1ml
```

---

## 📂 Project Structure

| Directory/File       | Purpose                                 |
| ------------------ | --------------------------------------- |
| `prepare.py`       | Data loading, BPE tokenizer, evaluation |
| `train.py`         | Model architecture & training loop      |
| `export_onnx.py`   | Traces and exports PyTorch to ONNX      |
| `infer.py`         | Terminal inference UI (rich)            |
| `web/`             | Astro/Svelte frontend application       |
| `web/public/models`| Embedded `.onnx` files and `tokenizer.json` |
| `.github/workflows`| Cloudflare Pages deployment CI          |
