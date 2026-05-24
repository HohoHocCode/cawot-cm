"""Step 1: extract CLIP embeddings of the train pool.

Usage:
  python scripts/01_extract_embeddings.py --config config.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embed import extract_embeddings
from src.utils import load_config, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    paths = extract_embeddings(cfg)
    print("Embeddings saved:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
