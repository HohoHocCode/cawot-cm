"""End-to-end V0 — independent of friend's qproxy work.

Pipeline (matches the plan: V0 = diversity-only baseline):

  1. Load PAB JSONL annotations, discover image shards on disk, build a
     random sample of N (= cfg.data.sample_size) (image, caption) entries.
  2. Extract CLIP-B/16 image embeddings (cached to disk).
  3. Hold out V (= cfg.data.val_size) pairs for image-text retrieval R@k.
  4. FAISS k-means (spherical) over the remaining train pool.
  5. Two coresets at the same budget:
       - random       (baseline)
       - v0           (farthest-from-centroid within each cluster)
  6. Fine-tune CLIP-B/16 on each coreset.
  7. Evaluate each + zero-shot anchor on the val split.

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
from src.data import TrainPoolDataset, build_pool, make_train_val_split
from src.embed import extract_image_embeddings, load_clip
from src.eval import evaluate_val_split
from src.select import select_random, select_v0
from src.train import train_on_dataset
from src.utils import ensure_dir, get_device, load_config, set_seed, setup_logger

logger = setup_logger("run_v0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device("cuda")

    # === Step 1: build train pool (anns + shard map, no images loaded yet) ===
    logger.info(f"building pool from annotations_dir={cfg['data']['annotations_dir']}, "
                f"image_root={cfg['data']['image_root']}")
    anns, shard_roots = build_pool(
        annotations_dir=cfg["data"]["annotations_dir"],
        image_root=cfg["data"]["image_root"],
        sample_size=cfg["data"]["sample_size"],
        seed=cfg["seed"],
    )
    N = len(anns)
    logger.info(f"pool: {N} samples, shards on disk: {sorted(shard_roots)}")

    # === Step 2: extract embeddings (cached) ===
    model, preprocess, tokenizer = load_clip(
        cfg["embed"]["model"], cfg["embed"]["pretrained"], device
    )
    emb_dir = ensure_dir(cfg["embed"]["output_dir"])
    emb_path = Path(emb_dir) / "image_embeddings.npy"

    if emb_path.exists():
        embeddings = np.load(emb_path)
        if embeddings.shape[0] != N:
            logger.warning(f"cached embeddings have {embeddings.shape[0]} rows but pool is {N}; "
                           f"re-extracting")
            emb_path.unlink()

    if not emb_path.exists():
        embed_ds = TrainPoolDataset(
            anns=anns, shard_roots=shard_roots, image_transform=preprocess, tokenizer=None
        )
        embeddings = extract_image_embeddings(
            embed_ds, model, device,
            batch_size=cfg["embed"]["batch_size"],
            num_workers=cfg["data"]["num_workers"],
            amp=cfg["train"]["amp"],
        )
        np.save(emb_path, embeddings)
        logger.info(f"saved embeddings to {emb_path}  shape={embeddings.shape}")
    else:
        logger.info(f"using cached embeddings at {emb_path}  shape={embeddings.shape}")

    # Free the embedding-extraction model — we'll reload per-run for finetune
    del model
    import torch
    torch.cuda.empty_cache()

    # === Step 3: train/val split ===
    train_idx, val_idx = make_train_val_split(N, cfg["data"]["val_size"], seed=cfg["seed"])
    train_emb = embeddings[train_idx]
    logger.info(f"split: {len(train_idx)} train pool / {len(val_idx)} val")

    # === Step 4: cluster ===
    cluster_path = Path(cfg["cluster"]["output_path"])
    if cluster_path.exists():
        centroids, assignments = load_clusters(cluster_path)
        logger.info(f"using cached clusters at {cluster_path}")
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

    # === Step 5: select coresets ===
    budget = int(round(cfg["coreset"]["budget_ratio"] * len(train_idx)))
    logger.info(f"coreset budget: {budget} samples ({cfg['coreset']['budget_ratio']:.0%})")

    rand_local = select_random(len(train_idx), budget, seed=cfg["seed"])
    v0_local = select_v0(train_emb, centroids, assignments, budget, seed=cfg["seed"])

    out_dir = ensure_dir(cfg["coreset"]["output_dir"])
    np.save(Path(out_dir) / "coreset_random.npy", rand_local)
    np.save(Path(out_dir) / "coreset_v0.npy", v0_local)

    # === Step 6 + 7: train and eval each ===
    results: dict = {}

    val_dataset = TrainPoolDataset(
        anns=anns, shard_roots=shard_roots,
        image_transform=preprocess, tokenizer=tokenizer,
        indices=val_idx,
    )

    # Zero-shot anchor (no fine-tune)
    results["zeroshot"] = evaluate_val_split(cfg, val_dataset, None, "zeroshot")

    for name, local_indices in [("random", rand_local), ("v0", v0_local)]:
        # local_indices are positions into train_idx; convert to pool positions
        pool_indices = train_idx[local_indices]
        train_dataset = TrainPoolDataset(
            anns=anns, shard_roots=shard_roots,
            image_transform=preprocess, tokenizer=tokenizer,
            indices=pool_indices,
        )
        ckpt = train_on_dataset(cfg, train_dataset, name)
        results[name] = evaluate_val_split(cfg, val_dataset, ckpt, name)

    summary_path = Path(cfg["eval"]["output_dir"]) / "summary.json"
    ensure_dir(summary_path.parent)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(results, indent=2))
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
