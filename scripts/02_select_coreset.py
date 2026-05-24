"""Step 2: cluster + select coreset.

Methods:
  --method random   : uniform random baseline
  --method v0       : cluster + farthest-from-centroid (V0)

Usage:
  python scripts/02_select_coreset.py --config config.yaml --method v0
  python scripts/02_select_coreset.py --config config.yaml --method random
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cluster import kmeans_faiss, save_clusters, load_clusters
from src.select import select_random, select_v0
from src.utils import ensure_dir, load_config, set_seed, setup_logger

logger = setup_logger("select_main")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--method", type=str, choices=["random", "v0"], required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    out_dir = ensure_dir(cfg["coreset"]["output_dir"])
    embed_dir = Path(cfg["embed"]["output_dir"])

    # Load embeddings (needed for both v0 clustering and to know N)
    img_emb = np.load(embed_dir / "image_embeddings.npy")
    N = img_emb.shape[0]
    budget = int(round(cfg["coreset"]["budget_ratio"] * N))
    logger.info(f"N={N}  budget={budget}  ({cfg['coreset']['budget_ratio']:.1%})")

    if args.method == "random":
        indices = select_random(N, budget, seed=cfg["seed"])
        out_path = Path(out_dir) / "coreset_random.npy"
    else:  # v0
        cluster_path = Path(cfg["cluster"]["output_path"])
        if cluster_path.exists():
            logger.info(f"Loading cached clusters from {cluster_path}")
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
        indices = select_v0(img_emb, centroids, assignments, budget, seed=cfg["seed"])
        out_path = Path(out_dir) / "coreset_v0.npy"

    np.save(out_path, indices)
    logger.info(f"Saved {len(indices)} indices to {out_path}")


if __name__ == "__main__":
    main()
