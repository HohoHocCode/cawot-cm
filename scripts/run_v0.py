"""End-to-end: extract → cluster+select (random AND v0) → train both → eval both.

Usage:
  python scripts/run_v0.py --config config.yaml

Runs back-to-back; safe to interrupt and restart (caches embeddings and clusters).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cluster import kmeans_faiss, load_clusters, save_clusters
from src.embed import extract_embeddings
from src.eval import evaluate
from src.select import select_random, select_v0
from src.train import train_with_coreset
from src.utils import ensure_dir, load_config, set_seed, setup_logger

logger = setup_logger("run_v0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--skip-train", action="store_true",
                    help="Skip training; eval zero-shot CLIP only")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    out_dir = ensure_dir(cfg["coreset"]["output_dir"])
    embed_dir = Path(cfg["embed"]["output_dir"])

    # === Step 1: embeddings (cached) ===
    img_emb_path = embed_dir / "image_embeddings.npy"
    if img_emb_path.exists():
        logger.info(f"Using cached embeddings at {img_emb_path}")
    else:
        extract_embeddings(cfg)
    img_emb = np.load(img_emb_path)
    N = img_emb.shape[0]
    budget = int(round(cfg["coreset"]["budget_ratio"] * N))
    logger.info(f"N={N}  budget={budget}")

    # === Step 2: clustering (cached) ===
    cluster_path = Path(cfg["cluster"]["output_path"])
    if cluster_path.exists():
        logger.info(f"Using cached clusters at {cluster_path}")
        centroids, assignments = load_clusters(cluster_path)
    else:
        centroids, assignments = kmeans_faiss(
            img_emb,
            k=cfg["cluster"]["k"],
            niter=cfg["cluster"]["niter"],
            spherical=cfg["cluster"]["spherical"],
            use_gpu=cfg["cluster"]["use_gpu"],
            seed=cfg["seed"],
        )
        save_clusters(cluster_path, centroids, assignments)

    # === Step 3: selections ===
    coresets = {}
    rand_path = Path(out_dir) / "coreset_random.npy"
    v0_path = Path(out_dir) / "coreset_v0.npy"
    if not rand_path.exists():
        np.save(rand_path, select_random(N, budget, seed=cfg["seed"]))
    if not v0_path.exists():
        np.save(v0_path, select_v0(img_emb, centroids, assignments, budget, seed=cfg["seed"]))
    coresets["random"] = str(rand_path)
    coresets["v0"] = str(v0_path)

    # === Step 4: train + eval each ===
    results = {}
    if args.skip_train:
        results["zeroshot"] = evaluate(cfg, None, "zeroshot")
    else:
        for name, path in coresets.items():
            ckpt = train_with_coreset(cfg, path, name)
            results[name] = evaluate(cfg, ckpt, name)

    summary_path = Path(cfg["eval"]["output_dir"]) / "summary.json"
    ensure_dir(summary_path.parent)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(results, indent=2))
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
