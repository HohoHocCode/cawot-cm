"""End-to-end coreset sweep: V0 family + V1, all budgets, all seeds, with
per-category eval (goal / full / wentwrong) and resumability for multi-session
Kaggle runs.

Methods (config coreset.methods):
  random   - uniform baseline
  v0       - farthest-from-centroid (plan V0, diversity/atypical)
  v0_proto - closest-to-centroid (prototype/representative, diagnostic)
  v1       - cross-modal cost + facility location (plan V1, the method)

Output:
  outputs/eval/records.csv    : 1 row per (method, budget, seed, category)
  outputs/eval/summary.json   : per (method, budget): mean ± std over seeds,
                                per category + overall
The script is RESUMABLE: it reads records.csv at start and skips any (method,
budget, seed) combination already complete. Safe to restart after a Kaggle
session timeout.

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

RECORD_FIELDS = [
    "method", "budget", "seed", "category", "n_queries",
    "t2i_R@1", "i2t_R@1", "t2i_R@5", "i2t_R@5", "t2i_R@10", "i2t_R@10", "mean_R@1",
]


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


def load_done(records_path: Path) -> set:
    """Return set of (method, budget, seed) tuples that already have ≥1 row."""
    done: set = set()
    if not records_path.exists():
        return done
    import csv as _csv
    with open(records_path, "r", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            try:
                done.add((row["method"], float(row["budget"]), int(row["seed"])))
            except (KeyError, ValueError):
                continue
    return done


def append_records(records_path: Path, rows: list[dict]):
    write_header = not records_path.exists()
    with open(records_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RECORD_FIELDS)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in RECORD_FIELDS})


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

    # Embeddings (cached)
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
        logger.info(f"using cached embeddings image{image_emb.shape}"
                    + (f" text{text_emb.shape}" if needs_text else ""))
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
    val_cat_counts = {c: int((val_dataset.categories == c).sum())
                      for c in sorted(set(val_dataset.categories.tolist()))}
    logger.info(f"val category counts: {val_cat_counts}")

    ensure_dir(cfg["coreset"]["output_dir"])
    ensure_dir(cfg["train"]["output_dir"])

    eval_dir = ensure_dir(cfg["eval"]["output_dir"])
    records_path = Path(eval_dir) / "records.csv"
    done = load_done(records_path)
    if done:
        logger.info(f"resuming — {len(done)} (method,budget,seed) combos already in records.csv")

    # === Zero-shot anchor (once per fresh records.csv) ===
    if ("zeroshot", -1.0, -1) not in done:
        zs = evaluate_val_split(cfg, val_dataset, None, "zeroshot")
        rows = []
        for cat, m in zs.items():
            rows.append({
                "method": "zeroshot", "budget": -1.0, "seed": -1, "category": cat,
                "n_queries": m["n_pairs"],
                "t2i_R@1": m.get("t2i_R@1", ""), "i2t_R@1": m.get("i2t_R@1", ""),
                "t2i_R@5": m.get("t2i_R@5", ""), "i2t_R@5": m.get("i2t_R@5", ""),
                "t2i_R@10": m.get("t2i_R@10", ""), "i2t_R@10": m.get("i2t_R@10", ""),
                "mean_R@1": m.get("mean_R@1", ""),
            })
        append_records(records_path, rows)
        done.add(("zeroshot", -1.0, -1))

    # === Sweep ===
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
                key = (method, float(budget), int(seed))
                if key in done:
                    logger.info(f"[skip] {method} b={budget} s={seed} already done")
                    continue
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
                m_dict = evaluate_val_split(cfg, val_dataset, ckpt, run_name)
                rows = []
                for cat, m in m_dict.items():
                    rows.append({
                        "method": method, "budget": float(budget), "seed": int(seed),
                        "category": cat, "n_queries": m["n_pairs"],
                        "t2i_R@1": m.get("t2i_R@1", ""), "i2t_R@1": m.get("i2t_R@1", ""),
                        "t2i_R@5": m.get("t2i_R@5", ""), "i2t_R@5": m.get("i2t_R@5", ""),
                        "t2i_R@10": m.get("t2i_R@10", ""), "i2t_R@10": m.get("i2t_R@10", ""),
                        "mean_R@1": m.get("mean_R@1", ""),
                    })
                append_records(records_path, rows)
                done.add(key)
                if not keep_ckpts:
                    Path(ckpt).unlink(missing_ok=True)

    # === Aggregate summary.json ===
    import csv as _csv
    all_rows = list(_csv.DictReader(open(records_path, "r", encoding="utf-8")))

    def collect(method, budget, category):
        return [float(r["mean_R@1"]) for r in all_rows
                if r["method"] == method and float(r["budget"]) == budget
                and r["category"] == category and r["mean_R@1"] != ""]

    summary: dict = {}
    zs_rows = [r for r in all_rows if r["method"] == "zeroshot"]
    summary["zeroshot"] = {r["category"]: round(float(r["mean_R@1"]), 3) for r in zs_rows if r["mean_R@1"]}

    categories = ["overall"] + sorted(set(val_dataset.categories.tolist()))
    for method in methods:
        summary[method] = {}
        for budget in budgets:
            summary[method][f"{budget}"] = {}
            for cat in categories:
                vals = collect(method, float(budget), cat)
                if not vals:
                    continue
                summary[method][f"{budget}"][cat] = {
                    "mean_R@1_mean": round(statistics.mean(vals), 3),
                    "mean_R@1_std": round(statistics.stdev(vals), 3) if len(vals) > 1 else 0.0,
                    "n_seeds": len(vals),
                }

    with open(Path(eval_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Console table — overall + wentwrong (the anomaly story)
    print("\n=== Sweep summary (mean_R@1) ===")
    print(f"zeroshot overall = {summary['zeroshot'].get('overall', '?')}")
    for category in ("overall", "wentwrong"):
        if category not in categories:
            continue
        print(f"\n[{category}]")
        header = f"{'budget':>7} | " + " | ".join(f"{m:>14}" for m in methods)
        print(header); print("-" * len(header))
        for budget in budgets:
            cells = []
            for m in methods:
                s = summary[m].get(f"{budget}", {}).get(category)
                cells.append(f"{s['mean_R@1_mean']:.2f}±{s['mean_R@1_std']:.2f}" if s else "    -    ")
            print(f"{budget:>6.0%} | " + " | ".join(f"{c:>14}" for c in cells))
    print(f"\nSaved: {records_path}\n        {Path(eval_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()
