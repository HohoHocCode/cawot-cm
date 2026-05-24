"""Step 4: evaluate a checkpoint on PAB test split.

Usage:
  python scripts/04_evaluate.py --config config.yaml --checkpoint outputs/checkpoints/v0.pt --name v0
  python scripts/04_evaluate.py --config config.yaml --checkpoint outputs/checkpoints/random.pt --name random

To evaluate the zero-shot pretrained CLIP (no finetuning):
  python scripts/04_evaluate.py --config config.yaml --checkpoint none --name zeroshot
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval import evaluate
from src.utils import load_config, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--checkpoint", type=str, default="none")
    ap.add_argument("--name", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    ckpt = None if args.checkpoint.lower() == "none" else args.checkpoint
    metrics = evaluate(cfg, ckpt, args.name)
    print(f"\n=== Results [{args.name}] ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
