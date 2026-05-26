"""End-to-end V0 on the friend's qproxy subset.

Pipeline:
  1. Load friend's manifest (.parquet) + embeddings (.npy)
  2. Hold out a (image, caption) val split for image-text retrieval R@k
  3. Cluster the remaining train pool with FAISS k-means (spherical)
  4. Select two coresets at the same budget:
       - random (baseline)
       - v0 (farthest-from-centroid within each cluster)
  5. Fine-tune CLIP ViT-B/16 on each coreset
  6. Evaluate both checkpoints on the val split

Usage:
  python scripts/run_v0.py --config config.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cluster import kmeans_faiss, load_clusters, save_clusters
from src.data import load_embeddings, load_manifest, make_train_val_split
from src.eval import evaluate_val_split
from src.select import select_random, select_v0
from src.train import train_with_manifest
from src.utils import ensure_dir, load_config, set_seed, setup_logger

logger = setup_logger("run_v0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    out_dir = ensure_dir(cfg["coreset"]["output_dir"])
    cluster_path = Path(cfg["cluster"]["output_path"])

    # === Step 1: load friend's manifest + embeddings ===
    logger.info(f"loading manifest from {cfg['qproxy']['manifest_path']}")
    manifest = load_manifest(
        cfg["qproxy"]["manifest_path"],
        path_remap=cfg["qproxy"].get("path_remap") or None,
    )
    logger.info(f"manifest: {len(manifest)} rows")

    logger.info(f"loading embeddings from {cfg['qproxy']['embeddings_path']}")
    embeddings = load_embeddings(cfg["qproxy"]["embeddings_path"], expected_n=len(manifest))
    logger.info(f"embeddings: shape={embeddings.shape}")

    # === Step 2: held-out val split ===
    val_size = cfg["data"]["val_size"]
    train_idx, val_idx = make_train_val_split(len(manifest), val_size, seed=cfg["seed"])
    logger.info(f"split: {len(train_idx)} train pool / {len(val_idx)} val")

    train_emb = embeddings[train_idx]
    train_manifest = manifest.iloc[train_idx].reset_index(drop=True)
    budget = int(round(cfg["coreset"]["budget_ratio"] * len(train_idx)))
    logger.info(f"coreset budget: {budget} ({cfg['coreset']['budget_ratio']:.0%})")

    # === Step 3: cluster ===
    if cluster_path.exists():
        logger.info(f"using cached clusters at {cluster_path}")
        centroids, assignments = load_clusters(cluster_path)
    else:
        centroids, assignments = kmeans_faiss(
            train_emb,
            k=cfg["cluster"]["k"],
            niter=cfg["cluster"]["niter"],
            spherical=cfg["cluster"]["spherical"],
            use_gpu=cfg["cluster"]["use_gpu"],
            seed=cfg["seed"],
        )
        save_clusters(cluster_path, centroids, assignments)

    # === Step 4: two coresets ===
    rand_local = select_random(len(train_idx), budget, seed=cfg["seed"])
    v0_local = select_v0(train_emb, centroids, assignments, budget, seed=cfg["seed"])
    # Indices above are positions into `train_manifest` (post train/val split).
    np.save(out_dir / "coreset_random.npy", rand_local)
    np.save(out_dir / "coreset_v0.npy", v0_local)

    # === Step 5+6: train and eval each ===
    results = {}
    for name, indices in [("random", rand_local), ("v0", v0_local)]:
        ckpt = train_with_manifest(cfg, train_manifest, indices, name)
        results[name] = evaluate_val_split(cfg, manifest, val_idx, ckpt, name)

    # Also evaluate zero-shot (no fine-tune) as anchor
    results["zeroshot"] = evaluate_val_split(cfg, manifest, val_idx, None, "zeroshot")

    summary_path = Path(cfg["eval"]["output_dir"]) / "summary.json"
    ensure_dir(summary_path.parent)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(results, indent=2))
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
