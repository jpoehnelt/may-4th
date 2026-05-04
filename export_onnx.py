"""
export_onnx.py — Export the trained EncoderDecoder model to ONNX for browser inference.

Produces two ONNX files:
  - web/public/models/encoder.onnx  (src_ids → encoder hidden states)
  - web/public/models/decoder.onnx  (tgt_ids + encoder hidden + src_pad_mask → logits)

Usage:
    uv run python export_onnx.py
    uv run python export_onnx.py --checkpoint checkpoints/best.pt
    uv run python export_onnx.py --output-dir onnx_out
"""

import argparse
import os
import onnx
import shutil
import json
import torch
import torch.nn as nn
import importlib.util

from prepare import get_tokenizer, DEVICE


class EncoderWrapper(nn.Module):
    """Wraps model.encode() for ONNX export.

    Inputs:  src_ids (int64 [B, S])
    Outputs: enc_out (float32 [B, S, D]), src_pad_mask (bool [B, S])
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, src_ids):
        src_pad_mask = (src_ids == 0)
        enc_out = self.model.encode(src_ids, src_pad_mask)
        return enc_out, src_pad_mask


class DecoderWrapper(nn.Module):
    """Wraps model.decode() for ONNX export.

    Inputs:  tgt_ids (int64 [B, T]), enc_out (float32 [B, S, D]), src_pad_mask (bool [B, S])
    Outputs: logits (float32 [B, T, V])
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tgt_ids, enc_out, src_pad_mask):
        return self.model.decode(tgt_ids, enc_out, src_pad_mask)


def load_model(checkpoint_path):
    """Load the EncoderDecoder model from checkpoint."""
    spec = importlib.util.spec_from_file_location("train", "train.py")
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = state["model"]

    # Infer architecture from checkpoint keys/shapes (handles train.py config drift)
    enc_layers = max(int(k.split(".")[1]) for k in state_dict if k.startswith("encoder.")) + 1
    dec_layers = max(int(k.split(".")[1]) for k in state_dict if k.startswith("decoder.")) + 1
    
    vocab_size, d_model = state_dict["tok_emb.weight"].shape
    d_ff = state_dict["encoder.0.ffn.w1.weight"].shape[0]
    d_ff_mult = d_ff // d_model
    n_heads = d_model // 32  # head_dim is implicitly 32 in this project

    print(f"Detected architecture: enc={enc_layers}, dec={dec_layers}, d_model={d_model}, n_heads={n_heads}, d_ff_mult={d_ff_mult}, vocab={vocab_size}")

    model = train_module.EncoderDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        depth_enc=enc_layers,
        depth_dec=dec_layers,
        d_ff_mult=d_ff_mult,
    ).to("cpu")
    model.load_state_dict(state_dict)
    model.eval()

    val_bpb = state.get("val_bpb", 0.0)
    step = state.get("step", 0)
    print(f"Loaded checkpoint: val_bpb={val_bpb:.4f}, step={step}, params={model.param_count():,}")
    return model, val_bpb, step


def export_encoder(model, output_dir):
    """Export encoder to ONNX."""
    encoder = EncoderWrapper(model)
    encoder.eval()

    # Dummy input: batch=1, source length=10
    dummy_src = torch.randint(1, 100, (1, 10), dtype=torch.long)

    path = os.path.join(output_dir, "encoder.onnx")
    print(f"Exporting encoder to {path}...")

    torch.onnx.export(
        encoder,
        (dummy_src,),
        path,
        input_names=["src_ids"],
        output_names=["enc_out", "src_pad_mask"],
        dynamic_axes={
            "src_ids": {0: "batch", 1: "src_len"},
            "enc_out": {0: "batch", 1: "src_len"},
            "src_pad_mask": {0: "batch", 1: "src_len"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Merge external data into single file (browser can't load .data files)
    _embed_external_data(path)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  → encoder.onnx: {size_mb:.2f} MB")
    return path


def export_decoder(model, output_dir):
    """Export decoder to ONNX."""
    decoder = DecoderWrapper(model)
    decoder.eval()

    # Dummy inputs: batch=1, target length=5, source length=10, d_model=128
    d_model = model.tok_emb.weight.shape[1]
    dummy_tgt = torch.randint(1, 100, (1, 5), dtype=torch.long)
    dummy_enc = torch.randn(1, 10, d_model)
    dummy_mask = torch.zeros(1, 10, dtype=torch.bool)

    path = os.path.join(output_dir, "decoder.onnx")
    print(f"Exporting decoder to {path}...")

    torch.onnx.export(
        decoder,
        (dummy_tgt, dummy_enc, dummy_mask),
        path,
        input_names=["tgt_ids", "enc_out", "src_pad_mask"],
        output_names=["logits"],
        dynamic_axes={
            "tgt_ids": {0: "batch", 1: "tgt_len"},
            "enc_out": {0: "batch", 1: "src_len"},
            "src_pad_mask": {0: "batch", 1: "src_len"},
            "logits": {0: "batch", 1: "tgt_len"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    # Merge external data into single file (browser can't load .data files)
    _embed_external_data(path)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  → decoder.onnx: {size_mb:.2f} MB")
    return path


def _embed_external_data(onnx_path):
    """Load an ONNX model with external data and re-save with everything embedded."""
    data_path = onnx_path + ".data"
    if not os.path.exists(data_path):
        return  # No external data to embed

    model = onnx.load(onnx_path, load_external_data=True)
    onnx.save_model(
        model,
        onnx_path,
        save_as_external_data=False,  # Embed all tensors
    )
    # Remove the now-unused .data file
    os.remove(data_path)
    print(f"    Embedded external data, removed {os.path.basename(data_path)}")


def export_metadata(model, tokenizer, val_bpb, step, output_dir):
    """Export model metadata + tokenizer for the web app."""
    # Tokenizer: copy the merges file
    tok_src = "data/tokenizer.json"
    tok_dst = os.path.join(output_dir, "tokenizer.json")
    shutil.copy2(tok_src, tok_dst)
    print(f"  → tokenizer.json copied")

    # Extract special token IDs (default to new tokenizer's fixed IDs if not set)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None: eos_id = 4095
    
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None: pad_id = 4093
    
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is None: bos_id = 4094

    # Calculate model sizes
    try:
        enc_size = os.path.getsize(os.path.join(output_dir, "encoder.onnx"))
        dec_size = os.path.getsize(os.path.join(output_dir, "decoder.onnx"))
        size_mb = round((enc_size + dec_size) / (1024 * 1024), 2)
    except FileNotFoundError:
        size_mb = 0.0

    # Model metadata
    meta = {
        "val_bpb": round(val_bpb, 4),
        "step": step,
        "params": model.param_count(),
        "size_mb": size_mb,
        "vocab_size": model.tok_emb.weight.shape[0],
        "d_model": model.tok_emb.weight.shape[1],
        "max_src_len": model.src_pos.weight.shape[0],
        "max_tgt_len": model.tgt_pos.weight.shape[0],
        "eos_id": eos_id,
        "pad_id": pad_id,
        "bos_id": bos_id,
    }
    meta_path = os.path.join(output_dir, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  → model_meta.json: {json.dumps(meta)}")


def verify_export(model, tokenizer, output_dir):
    """Quick verification: run ONNX models and compare to PyTorch output."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("\n⚠️  onnxruntime not installed — skipping verification.")
        print("  Install with: uv add onnxruntime")
        return

    print("\nVerifying ONNX export against PyTorch...")

    # Prepare test input
    test_text = "The dog ran to the tree."
    eos_ids = tokenizer.encode("\n")
    eos_id = eos_ids[-1] if eos_ids else 10
    src_ids = tokenizer.encode(test_text) + [eos_id]
    src_tensor = torch.tensor([src_ids], dtype=torch.long)

    # PyTorch encoder
    model.eval()
    with torch.no_grad():
        src_pad_mask = (src_tensor == 0)
        pt_enc = model.encode(src_tensor, src_pad_mask)

    # ONNX encoder
    enc_sess = ort.InferenceSession(os.path.join(output_dir, "encoder.onnx"))
    src_ids_np = src_tensor.numpy()
    ort_enc, ort_mask = enc_sess.run(None, {"src_ids": src_ids_np})

    # Compare
    diff = abs(pt_enc.numpy() - ort_enc).max()
    print(f"  Encoder max diff: {diff:.6f} {'✅' if diff < 0.001 else '❌'}")

    # PyTorch decoder (one step)
    tgt = torch.tensor([[0]], dtype=torch.long)  # BOS
    with torch.no_grad():
        pt_logits = model.decode(tgt, pt_enc, src_pad_mask)

    # ONNX decoder
    import numpy as np
    dec_sess = ort.InferenceSession(os.path.join(output_dir, "decoder.onnx"))
    ort_logits = dec_sess.run(None, {
        "tgt_ids": tgt.numpy(),
        "enc_out": ort_enc,
        "src_pad_mask": ort_mask,
    })[0]

    diff = abs(pt_logits.numpy() - ort_logits).max()
    print(f"  Decoder max diff: {diff:.6f} {'✅' if diff < 0.001 else '❌'}")

    # Full greedy decode comparison
    print(f"\n  Test input: \"{test_text}\"")

    # PyTorch greedy
    tgt = torch.tensor([[0]], dtype=torch.long)
    pt_tokens = []
    with torch.no_grad():
        for _ in range(40):
            logits = model.decode(tgt, pt_enc, src_pad_mask)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if next_id.item() == eos_id:
                break
            pt_tokens.append(next_id.item())
            tgt = torch.cat([tgt, next_id], dim=1)
    print(f"  PyTorch output: \"{tokenizer.decode(pt_tokens)}\"")

    # ONNX greedy
    tgt_np = np.array([[0]], dtype=np.int64)
    ort_tokens = []
    for _ in range(40):
        ort_logits = dec_sess.run(None, {
            "tgt_ids": tgt_np,
            "enc_out": ort_enc,
            "src_pad_mask": ort_mask,
        })[0]
        next_id = int(ort_logits[0, -1, :].argmax())
        if next_id == eos_id:
            break
        ort_tokens.append(next_id)
        tgt_np = np.concatenate([tgt_np, [[next_id]]], axis=1)
    print(f"  ONNX output:    \"{tokenizer.decode(ort_tokens)}\"")

    match = pt_tokens == ort_tokens
    print(f"  Outputs match: {'✅' if match else '❌'}")


def export(checkpoint="checkpoints/best.pt", output_dir="web/public/models", skip_verify=False, hf_repo=None):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("ONNX EXPORT")
    print("=" * 60)

    model, val_bpb, step = load_model(checkpoint)
    tokenizer = get_tokenizer()

    export_encoder(model, output_dir)
    export_decoder(model, output_dir)
    export_metadata(model, tokenizer, val_bpb, step, output_dir)

    if not skip_verify:
        verify_export(model, tokenizer, output_dir)

    if hf_repo:
        import subprocess
        print(f"\nUploading to Hugging Face Hub ({hf_repo})...")
        try:
            subprocess.run([
                "hf", "upload", 
                hf_repo, 
                output_dir, 
                ".", 
                "--commit-message", f"Update ONNX model (val_bpb: {val_bpb:.4f})"
            ], check=True)
            print(f"Successfully uploaded to https://huggingface.co/{hf_repo}")
        except FileNotFoundError:
            print("\n⚠️  `hf` CLI binary not found in PATH — skipping upload.")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Hugging Face upload failed: {e}")

    print(f"\n{'=' * 60}")
    print(f"Export complete → {output_dir}/")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Export Kaminoan model to ONNX")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--output-dir", default="web/public/models")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--hf-repo", default=None, help="Hugging Face repo ID (e.g. username/yoda-translator)")
    args = parser.parse_args()
    export(args.checkpoint, args.output_dir, args.skip_verify, args.hf_repo)


if __name__ == "__main__":
    main()
