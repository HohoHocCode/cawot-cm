"""End-to-end V0 — budget sweep over the friend-independent train pool.

Pipeline (V0 = diversity-only baseline):

  Fixed setup (done once):
    1. Load PAB JSONL annotations + discover image shards, random-sample N.
    2. Extract CLIP-B/16 image embeddings (cached).
    3. Hold out V pairs as a FIXED retrieval val set (same gallery for every
       run → numbers directly comparable).
    4. Zero-shot eval (anchor).

  Sweep (for each seed, for each budget):
    5. Cluster the train pool (FAISS spherical, this seed).
    6. Two coresets at this budget: Random vs V0 (farthest-from-centroid).
    7. Fine-tune CLIP-B/16 on each → eval on the val set.

  Aggregate:
    - summary.json : per (method, budget) mean ± std over seeds.
    - records.csv  : one row per (method, budget, seed) for plotting.

Usage:
  python scripts/run_v0.py --config config.yaml
"""
import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cluster import kmeans_faiss
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
    base_seed = cfg["seed"]
    set_seed(base_seed)
    device = get_device("cuda")

    budgets = cfg["coreset"]["budgets"]
    seeds = cfg["train"]["seeds"]
    keep_ckpts = cfg["train"].get("keep_checkpoints", False)
    logger.info(f"budgets={budgets}  seeds={seeds}")

    # === Fixed setup ===
    anns, shard_roots = build_pool(
        annotations_dir=cfg["data"]["annotations_dir"],
        image_root=cfg["data"]["image_root"],
        sample_size=cfg["data"]["sample_size"],
        seed=base_seed,
    )
    N = len(anns)
    logger.info(f"pool: {N} samples, shards: {sorted(shard_roots)}")

    model, preprocess, tokenizer = load_clip(
        cfg["embed"]["model"], cfg["embed"]["pretrained"], device
    )

    # Embeddings (cached)
    emb_dir = ensure_dir(cfg["embed"]["output_dir"])
    emb_path = Path(emb_dir) / "image_embeddings.npy"
    if emb_path.exists() and np.load(emb_path, mmap_mode="r").shape[0] == N:
        embeddings = np.load(emb_path)
        logger.info(f"using cached embeddings {embeddings.shape}")
    else:
        embed_ds = TrainPoolDataset(anns, shard_roots, image_transform=preprocess, tokenizer=None)
        embeddings = extract_image_embeddings(
            embed_ds, model, device,
            batch_size=cfg["embed"]["batch_size"],
            num_workers=cfg["data"]["num_workers"],
            amp=cfg["train"]["amp"],
        )
        np.save(emb_path, embeddings)
        logger.info(f"saved embeddings {embeddings.shape}")

    del model
    import torch
    torch.cuda.empty_cache()

    # FIXED train/val split — same val gallery for every run
    train_idx, val_idx = make_train_val_split(N, cfg["data"]["val_size"], seed=base_seed)
    train_emb = embeddings[train_idx]
    logger.info(f"split: {len(train_idx)} train pool / {len(val_idx)} val (FIXED)")

    val_dataset = TrainPoolDataset(
        anns, shard_roots, image_transform=preprocess, tokenizer=tokenizer, indices=val_idx
    )

    out_dir = ensure_dir(cfg["coreset"]["output_dir"])
    ckpt_dir = ensure_dir(cfg["train"]["output_dir"])

    # === Zero-shot anchor (once) ===
    zs = evaluate_val_split(cfg, val_dataset, None, "zeroshot")
    logger.info(f"zeroshot mean_R@1 = {zs['mean_R@1']:.2f}")

    # === Sweep ===
    records: list[dict] = []
    for seed in seeds:
        set_seed(seed)
        centroids, assignments = kmeans_faiss(
            train_emb,
            k=cfg["cluster"]["k"],
            niter=cfg["cluster"]["niter"],
            spherical=cfg["cluster"]["spherical"],
            use_gpu=cfg["cluster"]["use_gpu"],
            seed=seed,
        )
        for budget in budgets:
            b = int(round(budget * len(train_idx)))
            coresets = {
                "random": select_random(len(train_idx), b, seed=seed),
                "v0": select_v0(train_emb, centroids, assignments, b, seed=seed),
            }
            for method, local_idx in coresets.items():
                pool_idx = train_idx[local_idx]
                train_ds = TrainPoolDataset(
                    anns, shard_roots, image_transform=preprocess,
                    tokenizer=tokenizer, indices=pool_idx,
                )
                run_name = f"{method}_b{int(budget * 100)}_s{seed}"
                ckpt = train_on_dataset(cfg, train_ds, run_name)
                m = evaluate_val_split(cfg, val_dataset, ckpt, run_name)
                records.append({
                    "method": method, "budget": budget, "seed": seed,
                    "mean_R@1": m["mean_R@1"],
                    "t2i_R@1": m["t2i_R@1"], "i2t_R@1": m["i2t_R@1"],
                    "t2i_R@5": m["t2i_R@5"], "i2t_R@5": m["i2t_R@5"],
                })
                if not keep_ckpts:
                    Path(ckpt).unlink(missing_ok=True)
                logger.info(f"[{run_name}] mean_R@1 = {m['mean_R@1']:.2f}")

    # === Aggregate ===
    eval_dir = ensure_dir(cfg["eval"]["output_dir"])

    csv_path = Path(eval_dir) / "records.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    summary: dict = {"zeroshot": {"mean_R@1": zs["mean_R@1"]}}
    for method in ("random", "v0"):
        summary[method] = {}
        for budget in budgets:
            vals = [r["mean_R@1"] for r in records
                    if r["method"] == method and r["budget"] == budget]
            summary[method][f"{budget}"] = {
                "mean_R@1_mean": round(statistics.mean(vals), 3),
                "mean_R@1_std": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
                "n_seeds": len(vals),
            }

    summary_path = Path(eval_dir) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Console table
    print("\n=== V0 budget sweep ===")
    print(f"zeroshot mean_R@1 = {zs['mean_R@1']:.2f}\n")
    print(f"{'budget':>8} | {'random':>16} | {'v0':>16} | {'Δ(v0-rand)':>10}")
    print("-" * 60)
    for budget in budgets:
        r = summary["random"][f"{budget}"]
        v = summary["v0"][f"{budget}"]
        delta = v["mean_R@1_mean"] - r["mean_R@1_mean"]
        rstr = f"{r['mean_R@1_mean']:.2f}±{r['mean_R@1_std']:.2f}"
        vstr = f"{v['mean_R@1_mean']:.2f}±{v['mean_R@1_std']:.2f}"
        print(f"{budget:>8.0%} | {rstr:>16} | {vstr:>16} | {delta:>+10.2f}")
    print(f"\nSaved: {summary_path}\n        {csv_path}")


if __name__ == "__main__":
    main()
