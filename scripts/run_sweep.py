"""End-to-end coreset sweep: V0 family + V1, all budgets, all seeds.

Methods (config coreset.methods):
  random   - uniform baseline
  v0       - farthest-from-centroid (plan V0, diversity/atypical)
  v0_proto - closest-to-centroid (prototype/representative, diagnostic)
  v1       - cross-modal cost + facility location (plan V1, the method)

Pipeline:
  Fixed once:
    1. Load PAB JSONL annotations + discover shards, random-sample N.
    2. Extract CLIP-B/16 IMAGE + TEXT embeddings (cached). V1 needs text.
    3. Hold out V pairs as a FIXED retrieval val set (same gallery for all runs).
    4. Zero-shot eval (anchor).
  Sweep (per seed → cluster on image embeddings; per budget; per method):
    5. Select coreset.
    6. Fine-tune CLIP-B/16 on the coreset → eval on val.
  Aggregate:
    summary.json  : per (method, budget) mean ± std over seeds.
    records.csv   : one row per (method, budget, seed).

Usage:
  python scripts/run_sweep.py --config config.yaml
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
from src.embed import extract_image_text_embeddings, load_clip
from src.eval import evaluate_val_split
from src.select import select_random, select_v0, select_v0_proto, select_v1
from src.train import train_on_dataset
from src.utils import ensure_dir, get_device, load_config, set_seed, setup_logger

logger = setup_logger("run_sweep")


def select_coreset(method, n_train, b, zv, zt, centroids, assignments, seed):
    if method == "random":
        return select_random(n_train, b, seed=seed)
    if method == "v0":
        return select_v0(zv, centroids, assignments, b, seed=seed)
    if method == "v0_proto":
        return select_v0_proto(zv, centroids, assignments, b, seed=seed)
    if method == "v1":
        return select_v1(zv, zt, centroids, assignments, b, seed=seed)
    raise ValueError(f"unknown method: {method}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    base_seed = cfg["seed"]
    set_seed(base_seed)
    device = get_device("cuda")

    budgets = cfg["coreset"]["budgets"]
    methods = cfg["coreset"]["methods"]
    seeds = cfg["train"]["seeds"]
    keep_ckpts = cfg["train"].get("keep_checkpoints", False)
    needs_text = "v1" in methods
    logger.info(f"methods={methods}  budgets={budgets}  seeds={seeds}  needs_text={needs_text}")

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

    # Embeddings (cached). Always extract image; extract text when V1 is in play.
    emb_dir = ensure_dir(cfg["embed"]["output_dir"])
    img_path = Path(emb_dir) / "image_embeddings.npy"
    txt_path = Path(emb_dir) / "text_embeddings.npy"

    def _cache_ok(p):
        return p.exists() and np.load(p, mmap_mode="r").shape[0] == N

    have_img = _cache_ok(img_path)
    have_txt = _cache_ok(txt_path)
    if have_img and (have_txt or not needs_text):
        image_emb = np.load(img_path)
        text_emb = np.load(txt_path) if needs_text else None
        logger.info(f"using cached embeddings: image{image_emb.shape}"
                    + (f" + text{text_emb.shape}" if needs_text else ""))
    else:
        embed_ds = TrainPoolDataset(anns, shard_roots, image_transform=preprocess, tokenizer=tokenizer)
        image_emb, text_emb = extract_image_text_embeddings(
            embed_ds, model, device,
            batch_size=cfg["embed"]["batch_size"],
            num_workers=cfg["data"]["num_workers"],
            amp=cfg["train"]["amp"],
        )
        np.save(img_path, image_emb)
        np.save(txt_path, text_emb)
        logger.info(f"saved embeddings image{image_emb.shape} text{text_emb.shape}")
        if not needs_text:
            text_emb = None

    del model
    import torch
    torch.cuda.empty_cache()

    # FIXED train/val split — same val gallery for every run
    train_idx, val_idx = make_train_val_split(N, cfg["data"]["val_size"], seed=base_seed)
    train_zv = image_emb[train_idx]
    train_zt = text_emb[train_idx] if text_emb is not None else None
    n_train = len(train_idx)
    logger.info(f"split: {n_train} train pool / {len(val_idx)} val (FIXED)")

    val_dataset = TrainPoolDataset(
        anns, shard_roots, image_transform=preprocess, tokenizer=tokenizer, indices=val_idx
    )

    ensure_dir(cfg["coreset"]["output_dir"])
    ensure_dir(cfg["train"]["output_dir"])

    # === Zero-shot anchor (once) ===
    zs = evaluate_val_split(cfg, val_dataset, None, "zeroshot")
    logger.info(f"zeroshot mean_R@1 = {zs['mean_R@1']:.2f}")

    # === Sweep ===
    records: list[dict] = []
    for seed in seeds:
        set_seed(seed)
        centroids, assignments = kmeans_faiss(
            train_zv,
            k=cfg["cluster"]["k"],
            niter=cfg["cluster"]["niter"],
            spherical=cfg["cluster"]["spherical"],
            use_gpu=cfg["cluster"]["use_gpu"],
            seed=seed,
        )
        for budget in budgets:
            b = int(round(budget * n_train))
            for method in methods:
                local_idx = select_coreset(
                    method, n_train, b, train_zv, train_zt, centroids, assignments, seed
                )
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
                    "n_coreset": int(len(local_idx)),
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

    summary: dict = {"zeroshot": {"mean_R@1": round(zs["mean_R@1"], 3)}}
    for method in methods:
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

    # Console table: methods as columns, budgets as rows
    print("\n=== Coreset sweep (mean_R@1, val image-text retrieval) ===")
    print(f"zeroshot = {zs['mean_R@1']:.2f}\n")
    header = f"{'budget':>7} | " + " | ".join(f"{m:>14}" for m in methods)
    print(header)
    print("-" * len(header))
    for budget in budgets:
        cells = []
        for m in methods:
            s = summary[m][f"{budget}"]
            cells.append(f"{s['mean_R@1_mean']:.2f}±{s['mean_R@1_std']:.2f}")
        print(f"{budget:>6.0%} | " + " | ".join(f"{c:>14}" for c in cells))
    print(f"\nSaved: {summary_path}\n        {csv_path}")


if __name__ == "__main__":
    main()
