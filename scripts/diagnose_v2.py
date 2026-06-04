"""Diagnose old V2 and new V2.1 distribution signals.

Outputs:
  outputs/diagnostic/v2_diagnosis.json
  outputs/diagnostic/v2_diagnosis.md

The report measures:
  - old V2 Gaussian-diag W spread on flat clusters,
  - V2.1 Sliced Wasserstein spread on coarse clusters,
  - V2/V2.1 budget shift magnitude relative to size-proportional allocation,
  - Q_proxy pairwise similarity and effective PCA dimensions,
  - flat/coarse/fine cluster size summaries.

Run after embeddings and qproxy cache exist:
  python scripts/run_sweep.py --config configs/smoke_v2_1.yaml
  python scripts/diagnose_v2.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cluster import hierarchical_kmeans_faiss, kmeans_faiss
from src.data import build_pool, make_train_val_split
from src.select import (
    _proportional_budgets,
    compute_cluster_wasserstein_gaps,
    wasserstein_aware_budgets,
)
from src.select_v2_1 import compute_coarse_sliced_wasserstein_gaps
from src.utils import ensure_dir, load_config, setup_logger

logger = setup_logger("diagnose_v2")


def _summary(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    mean = float(x.mean()) if len(x) else 0.0
    std = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    return {
        "min": float(x.min()) if len(x) else 0.0,
        "median": float(np.median(x)) if len(x) else 0.0,
        "max": float(x.max()) if len(x) else 0.0,
        "mean": mean,
        "std": std,
        "cv": float(std / mean) if mean else 0.0,
    }


def _budget_shift_summary(sizes: np.ndarray, gaps: np.ndarray, total_budget: int, alpha: float) -> dict:
    proportional = _proportional_budgets(sizes, total_budget)
    shifted = wasserstein_aware_budgets(sizes, gaps, total_budget, alpha=alpha)
    delta = shifted - proportional
    return {
        "budget": int(total_budget),
        "proportional_sum": int(proportional.sum()),
        "shifted_sum": int(shifted.sum()),
        "l1_shift": int(np.abs(delta).sum()),
        "max_abs_shift": int(np.abs(delta).max()) if len(delta) else 0,
        "nonzero_shift_bins": int((delta != 0).sum()),
    }


def _qproxy_quality(q: np.ndarray, seed: int = 42) -> dict:
    rng = np.random.RandomState(seed)
    n = len(q)
    n_pairs = min(100_000, max(0, n * (n - 1) // 2))
    if n_pairs == 0:
        return {"n": n, "n_pairs": 0}
    a = rng.randint(0, n, size=n_pairs)
    b = rng.randint(0, n, size=n_pairs)
    same = a == b
    if same.any():
        b[same] = (b[same] + 1) % n
    sims = np.sum(q[a] * q[b], axis=1)
    out = {
        "n": int(n),
        "n_pairs": int(n_pairs),
        "pairwise_cosine": _summary(sims),
    }
    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=min(256, n - 1, q.shape[1]))
        pca.fit(q)
        cum = np.cumsum(pca.explained_variance_ratio_)
        for target in (0.8, 0.9, 0.95):
            key = f"pca_components_for_{int(target * 100)}pct"
            out[key] = int(np.searchsorted(cum, target) + 1) if cum.max() >= target else None
    except Exception as e:
        out["pca_error"] = str(e)
    return out


def _write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# V2 / V2.1 Diagnosis",
        "",
        "## Wasserstein Signal",
        f"- Old V2 Gaussian W CV: {report['old_v2']['w_summary']['cv']:.4f}",
        f"- V2.1 coarse SW2 CV: {report['v2_1']['sw_summary']['cv']:.4f}",
        "",
        "## Budget Shift",
        f"- Old V2 L1 shift: {report['old_v2']['budget_shift']['l1_shift']} "
        f"(max bin {report['old_v2']['budget_shift']['max_abs_shift']})",
        f"- V2.1 L1 shift: {report['v2_1']['budget_shift']['l1_shift']} "
        f"(max bin {report['v2_1']['budget_shift']['max_abs_shift']})",
        "",
        "## Cluster Sizes",
        f"- Flat clusters: {report['old_v2']['cluster_size_summary']}",
        f"- Coarse clusters: {report['v2_1']['coarse_size_summary']}",
        f"- Fine leaves: {report['v2_1']['fine_leaf_size_summary']}",
        "",
        "## Q_proxy",
        f"- Queries: {report['qproxy']['n']}",
        f"- Pairwise cosine summary: {report['qproxy']['pairwise_cosine']}",
        "",
        "## Interpretation Guide",
        "- If old Gaussian W CV is very small but V2.1 SW2 CV is larger, the V2.1 "
        "replacement is doing the intended job.",
        "- If both CV values are small, Q_proxy expansion or a stronger distribution "
        "distance may be needed before spending training compute.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--budget", type=float, default=0.05)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_dir(Path(cfg["coreset"].get("output_dir", "outputs")) / "diagnostic")

    anns, shard_roots = build_pool(
        annotations_dir=cfg["data"]["annotations_dir"],
        image_root=cfg["data"]["image_root"],
        sample_size=cfg["data"]["sample_size"],
        seed=cfg["seed"],
    )
    del shard_roots
    emb_dir = Path(cfg["embed"]["output_dir"])
    image_emb = np.load(emb_dir / "image_embeddings.npy")
    text_emb = np.load(emb_dir / "text_embeddings.npy")
    q_proxy_emb = np.load(cfg["qproxy"]["cache_path"])
    train_idx, _ = make_train_val_split(len(anns), cfg["data"]["val_size"], seed=cfg["seed"])
    train_zv = image_emb[train_idx]
    train_zt = text_emb[train_idx]
    total_budget = int(round(args.budget * len(train_idx)))
    logger.info(f"diagnosing on train={len(train_idx)} budget={total_budget}")

    centroids, assignments = kmeans_faiss(
        train_zv,
        k=cfg["cluster"]["k"],
        niter=cfg["cluster"]["niter"],
        spherical=cfg["cluster"]["spherical"],
        use_gpu=cfg["cluster"]["use_gpu"],
        seed=cfg["seed"],
    )
    flat_sizes = np.bincount(assignments, minlength=centroids.shape[0]).astype(np.int64)
    old_w = compute_cluster_wasserstein_gaps(train_zt, assignments, q_proxy_emb, centroids.shape[0])

    v21 = cfg["coreset"].get("v2_1", {})
    hier = hierarchical_kmeans_faiss(
        train_zv,
        k_coarse=int(v21.get("k_coarse", 20)),
        k_fine=int(v21.get("k_fine", 10)),
        niter=cfg["cluster"]["niter"],
        spherical=cfg["cluster"]["spherical"],
        use_gpu=cfg["cluster"]["use_gpu"],
        seed=cfg["seed"],
    )
    coarse_sizes = np.bincount(
        hier["coarse_assignments"],
        minlength=hier["coarse_centroids"].shape[0],
    ).astype(np.int64)
    sw = compute_coarse_sliced_wasserstein_gaps(
        train_zt,
        hier["coarse_assignments"],
        q_proxy_emb,
        hier["coarse_centroids"].shape[0],
        num_projections=int(v21.get("num_projections", 128)),
        seed=cfg["seed"],
    )
    fine_sizes = []
    for c in range(hier["fine_valid"].shape[0]):
        idx_c = np.where(hier["coarse_assignments"] == c)[0]
        for f in np.where(hier["fine_valid"][c])[0]:
            fine_sizes.append(int((hier["fine_assignments"][idx_c] == f).sum()))
    fine_sizes = np.asarray(fine_sizes, dtype=np.int64)

    report = {
        "config": args.config,
        "budget_ratio": float(args.budget),
        "old_v2": {
            "w_summary": _summary(old_w),
            "cluster_size_summary": _summary(flat_sizes),
            "budget_shift": _budget_shift_summary(
                flat_sizes, old_w, total_budget, float(cfg["coreset"].get("v2_alpha", 0.5))
            ),
        },
        "v2_1": {
            "sw_summary": _summary(sw),
            "coarse_size_summary": _summary(coarse_sizes),
            "fine_leaf_size_summary": _summary(fine_sizes),
            "budget_shift": _budget_shift_summary(
                coarse_sizes, sw, total_budget, float(cfg["coreset"].get("v2_alpha", 0.5))
            ),
        },
        "qproxy": _qproxy_quality(q_proxy_emb, seed=cfg["seed"]),
    }

    json_path = Path(out_dir) / "v2_diagnosis.json"
    md_path = Path(out_dir) / "v2_diagnosis.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(md_path, report)
    logger.info(f"saved {json_path}")
    logger.info(f"saved {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
