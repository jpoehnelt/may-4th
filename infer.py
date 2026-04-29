"""
infer.py — Terminal UI for Yoda model inference.

Loads the best checkpoint and streams completions with a
rich terminal interface. No server needed — runs inline.

Usage:
    python infer.py                            # interactive mode
    python infer.py --prompt "The code"        # single-shot
    python infer.py --checkpoint checkpoints/best.pt
"""

import argparse
import time
import sys
import torch

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.columns import Columns
from rich import box

from prepare import get_tokenizer, DEVICE, MAX_SEQ_LEN

# Lazy-import model from train.py
import importlib.util

console = Console()

# ---- Amber/gold theme ----
STYLE_TITLE = "bold #f59e0b"
STYLE_PROMPT = "bold #e2e8f0"
STYLE_OUTPUT = "#fbbf24"
STYLE_DIM = "dim #94a3b8"
STYLE_STAT = "#22c55e"
STYLE_BORDER = "#f59e0b"


def load_model(checkpoint_path):
    """Load model from checkpoint, importing architecture from train.py."""
    spec = importlib.util.spec_from_file_location("train", "train.py")
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)

    model = train_module.GPT().to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()

    val_bpb = state.get("val_bpb", 0.0)
    step = state.get("step", 0)
    n_params = model.param_count()
    return model, val_bpb, step, n_params


@torch.no_grad()
def generate_tokens(model, tokenizer, prompt, max_tokens=80, temperature=0.8, top_k=40):
    """Yield (token_string, token_count, elapsed) tuples.
    Stops at the first newline to produce single-sentence output.
    """
    ids = tokenizer.encode(prompt)
    ids = ids[-(MAX_SEQ_LEN - 1):]
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    start = time.time()
    for i in range(max_tokens):
        logits = model(x)
        logits = logits[:, -1, :] / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        token_str = tokenizer.decode([next_id.item()])

        # Stop at sentence boundary (newline = end of example in training data)
        if "\n" in token_str:
            token_str = token_str.split("\n")[0]
            if token_str:
                elapsed = time.time() - start
                yield token_str, i + 1, elapsed
            break

        elapsed = time.time() - start
        yield token_str, i + 1, elapsed

        x = torch.cat([x, next_id], dim=1)
        if x.size(1) > MAX_SEQ_LEN:
            x = x[:, -MAX_SEQ_LEN:]


def render_header(val_bpb, n_params, step):
    """Render the header panel."""
    title = Text("⚔️  KAMINOAN", style=STYLE_TITLE)
    subtitle = Text("  Autonomous Research · Yoda OSV Model", style=STYLE_DIM)
    header_text = title + subtitle

    stats = Text()
    stats.append(f"  val_bpb ", style=STYLE_DIM)
    stats.append(f"{val_bpb:.4f}", style=STYLE_STAT)
    stats.append(f"  │  params ", style=STYLE_DIM)
    stats.append(f"{n_params/1e6:.1f}M", style=STYLE_STAT)
    stats.append(f"  │  step ", style=STYLE_DIM)
    stats.append(f"{step}", style=STYLE_STAT)

    return Panel(
        header_text + Text("\n") + stats,
        border_style=STYLE_BORDER,
        box=box.DOUBLE,
        padding=(0, 1),
    )


def run_interactive(model, tokenizer, val_bpb, step, n_params, args):
    """Interactive prompt loop."""
    console.print(render_header(val_bpb, n_params, step))
    console.print()

    while True:
        try:
            console.print("[dim]Type an English sentence. The model will convert it to Yoda-speak.[/dim]")
            console.print()
            user_input = console.input("[bold #f59e0b]❯ [/bold #f59e0b]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]May the force be with your validation loss.[/dim]")
            break

        if not user_input.strip():
            continue

        console.print()

        # Wrap in training format so the model sees the pattern it learned
        model_prompt = f"English: {user_input}\nYoda: "

        # Stream output with live display
        output_parts = []
        final_tokens = 0
        final_elapsed = 0.0

        with Live(console=console, refresh_per_second=15, transient=True) as live:
            for token_str, count, elapsed in generate_tokens(
                model, tokenizer, model_prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            ):
                output_parts.append(token_str)
                final_tokens = count
                final_elapsed = elapsed

                # Build display: show English input + Yoda output
                display = Text()
                display.append("English: ", style=STYLE_DIM)
                display.append(user_input, style=STYLE_PROMPT)
                display.append("\nYoda: ", style=STYLE_DIM)
                display.append("".join(output_parts), style=STYLE_OUTPUT)

                tps = count / max(elapsed, 0.001)
                status = Text(f"\n\n  {count} tokens  │  {elapsed:.1f}s  │  {tps:.0f} tok/s", style=STYLE_DIM)

                panel = Panel(
                    display + status,
                    title="[#f59e0b]generating[/#f59e0b]",
                    border_style="#1e293b",
                    box=box.ROUNDED,
                    padding=(1, 2),
                )
                live.update(panel)

        # Final static output
        tps = final_tokens / max(final_elapsed, 0.001)
        final_display = Text()
        final_display.append("English: ", style=STYLE_DIM)
        final_display.append(user_input, style=STYLE_PROMPT)
        final_display.append("\nYoda: ", style=STYLE_DIM)
        final_display.append("".join(output_parts), style=STYLE_OUTPUT)

        final_status = Text(
            f"\n\n  {final_tokens} tokens  │  {final_elapsed:.1f}s  │  {tps:.0f} tok/s",
            style=STYLE_STAT,
        )

        console.print(Panel(
            final_display + final_status,
            title="[#f59e0b]complete[/#f59e0b]",
            border_style=STYLE_BORDER,
            box=box.ROUNDED,
            padding=(1, 2),
        ))
        console.print()


def run_single(model, tokenizer, prompt, args):
    """Single-shot generation for piping / scripting."""
    model_prompt = f"English: {prompt}\nYoda: "
    for token_str, _, _ in generate_tokens(
        model, tokenizer, model_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    ):
        sys.stdout.write(token_str)
        sys.stdout.flush()
    sys.stdout.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Kaminoan Inference TUI")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--prompt", default=None, help="Single-shot prompt (skips interactive)")
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    console.print(f"\n[dim]Loading model from {args.checkpoint}...[/dim]")
    model, val_bpb, step, n_params = load_model(args.checkpoint)
    tokenizer = get_tokenizer()
    console.print(f"[dim]Ready. Device: {DEVICE}[/dim]\n")

    if args.prompt:
        run_single(model, tokenizer, args.prompt, args)
    else:
        run_interactive(model, tokenizer, val_bpb, step, n_params, args)


if __name__ == "__main__":
    main()
