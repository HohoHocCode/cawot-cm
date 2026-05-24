"""Step 3: fine-tune CLIP on the selected coreset.

Usage:
  python scripts/03_train.py --config config.yaml --coreset outputs/coreset_v0.npy --name v0
  python scripts/03_train.py --config config.yaml --coreset outputs/coreset_random.npy --name random
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import train_with_coreset
from src.utils import load_config, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--coreset", type=str, required=True, help="Path to coreset indices .npy")
    ap.add_argument("--name", type=str, required=True, help="Run name (used for checkpoint filename)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    ckpt = train_with_coreset(cfg, args.coreset, args.name)
    print(f"Checkpoint saved to: {ckpt}")


if __name__ == "__main__":
    main()
