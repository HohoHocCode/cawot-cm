"""Pre-sweep diagnostic — visualize what each selection method picks + Q_proxy quality.

Outputs go to outputs/diagnostic/:
  selection_pca.png            — 2D PCA scatter, colored by each method's coreset
  centroid_dist_hist.png       — distance-to-centroid distribution: selected vs pool
  image_grid_v0_vs_proto.png   — side-by-side 16-image grid: V0 (atypical) vs V0_proto (prototype)
  qproxy_quality.png           — pairwise cosine sim hist + PCA cumulative variance
  qproxy_themes.txt            — top-20 KMeans themes with example queries per theme

Run AFTER scripts/run_sweep.py has cached image+text embeddings (and optionally
Q_proxy embeddings). Cheap (~5 min, no training).

Usage:
  python scripts/diagnose_selection.py --config config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cluster import kmeans_faiss
from src.data import build_pool, make_train_val_split, TrainPoolDataset
from src.select import (select_random, select_v0, select_v0_proto, select_v1, select_v2)
from src.utils import ensure_dir, load_config, setup_logger

logger = setup_logger("diagnose")

DIAGNOSTIC_BUDGET = 0.05      # 5% — selection effect is largest here
SAMPLE_FOR_PCA = 5000          # subsample for plotting speed
N_THEMES = 20                  # Q_proxy KMeans themes
N_IMG_PER_METHOD = 16          # samples shown in image grid


# -----------------------------------------------------------------------------
# Selections
# -----------------------------------------------------------------------------

def run_selections(cfg, image_emb, text_emb, train_idx, q_proxy_emb):
    train_zv = image_emb[train_idx]
    train_zt = text_emb[train_idx]
    n_train = len(train_idx)
    budget = int(round(DIAGNOSTIC_BUDGET * n_train))
    seed = cfg["seed"]

    centroids, assignments = kmeans_faiss(
        train_zv, k=cfg["cluster"]["k"], niter=cfg["cluster"]["niter"],
        spherical=cfg["cluster"]["spherical"], use_gpu=cfg["cluster"]["use_gpu"],
        seed=seed,
    )

    sels = {
        "random": select_random(n_train, budget, seed=seed),
        "v0": select_v0(train_zv, centroids, assignments, budget, seed=seed),
        "v0_proto": select_v0_proto(train_zv, centroids, assignments, budget, seed=seed),
        "v1": select_v1(train_zv, text_emb[train_idx], centroids, assignments, budget, seed=seed),
    }
    if q_proxy_emb is not None:
        sels["v2"] = select_v2(
            train_zv, train_zt, centroids, assignments, q_proxy_emb,
            budget, alpha=cfg["coreset"].get("v2_alpha", 0.5), seed=seed,
        )
    return sels, centroids, assignments


# -----------------------------------------------------------------------------
# Plots: selection space + centroid-distance histogram
# -----------------------------------------------------------------------------

def plot_pca_selections(train_zv: np.ndarray, sels: dict, out_path: Path) -> None:
    from sklearn.decomposition import PCA
    rng = np.random.RandomState(0)
    n = train_zv.shape[0]
    sub = rng.choice(n, size=min(SAMPLE_FOR_PCA, n), replace=False)
    coords = PCA(n_components=2).fit_transform(train_zv[sub])
    methods = list(sels.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 4),
                             sharex=True, sharey=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, m in zip(axes, methods):
        sel = set(sels[m].tolist())
        is_sel = np.array([s in sel for s in sub])
        ax.scatter(coords[~is_sel, 0], coords[~is_sel, 1], s=2, c="lightgray", alpha=0.3)
        ax.scatter(coords[is_sel, 0], coords[is_sel, 1], s=10, c="red", alpha=0.7,
                   label=f"selected ({is_sel.sum()})")
        ax.set_title(m)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Selection in 2D PCA space — {DIAGNOSTIC_BUDGET:.0%} budget on a "
                 f"{SAMPLE_FOR_PCA}-sample subset")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"saved {out_path}")


def plot_centroid_dist_hist(train_zv: np.ndarray, centroids: np.ndarray,
                            assignments: np.ndarray, sels: dict, out_path: Path) -> None:
    dists = np.zeros(len(train_zv), dtype=np.float32)
    for k in range(len(centroids)):
        mask = assignments == k
        if not mask.any():
            continue
        dists[mask] = 1.0 - (train_zv[mask] @ centroids[k])
    methods = list(sels.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 3.5),
                             sharex=True, sharey=True)
    if len(methods) == 1:
        axes = [axes]
    for ax, m in zip(axes, methods):
        sm = np.zeros(len(train_zv), dtype=bool)
        sm[sels[m]] = True
        ax.hist(dists[~sm], bins=40, alpha=0.4, color="gray", density=True, label="pool")
        ax.hist(dists[sm], bins=40, alpha=0.6, color="red", density=True, label="selected")
        ax.set_title(m); ax.set_xlabel("1 − cos(x, centroid)")
        ax.legend(fontsize=8)
    fig.suptitle("Distance-to-centroid: selected vs full pool")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"saved {out_path}")


# -----------------------------------------------------------------------------
# Image grid: V0 vs V0_proto (the visual evidence)
# -----------------------------------------------------------------------------

def plot_image_grid(anns, shard_roots, sels, train_idx, out_path: Path,
                    n_per_method: int = N_IMG_PER_METHOD) -> None:
    methods_to_show = [m for m in ("v0", "v0_proto") if m in sels]
    if not methods_to_show:
        logger.warning("v0 / v0_proto not in selections — skipping image grid")
        return

    rows = 4
    cols_per_method = n_per_method // rows
    cols_total = cols_per_method * len(methods_to_show)
    fig, axes = plt.subplots(rows, cols_total, figsize=(2 * cols_total, 2 * rows + 0.5))

    rng = np.random.RandomState(0)
    for block, method in enumerate(methods_to_show):
        sel = sels[method]
        chosen = rng.choice(len(sel), size=min(n_per_method, len(sel)), replace=False)
        # Use TrainPoolDataset's image-path resolution. Identity transform → PIL.
        sub_anns = [anns[train_idx[sel[i]]] for i in chosen]
        ds = TrainPoolDataset(sub_anns, shard_roots, image_transform=lambda x: x)
        for i in range(n_per_method):
            r = i // cols_per_method
            c = block * cols_per_method + (i % cols_per_method)
            ax = axes[r, c]
            try:
                img = ds[i]["image"]
                ax.imshow(img)
            except Exception:
                pass
            ax.set_xticks([]); ax.set_yticks([])
        # Title in the middle column of each block
        mid_col = block * cols_per_method + cols_per_method // 2
        axes[0, mid_col].set_title(
            f"{method}\n({'farthest = atypical' if method == 'v0' else 'closest = prototype'})",
            fontsize=12,
        )

    fig.suptitle(f"Visual comparison: {N_IMG_PER_METHOD} random samples from each method's "
                 f"{DIAGNOSTIC_BUDGET:.0%}-budget coreset", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"saved {out_path}")


# -----------------------------------------------------------------------------
# Q_proxy quality: pairwise similarity, PCA dims, KMeans themes
# -----------------------------------------------------------------------------

def diagnose_qproxy(q_proxy_emb: np.ndarray, q_proxy_captions: list[str], out_dir: Path) -> None:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    n = q_proxy_emb.shape[0]

    rng = np.random.RandomState(0)
    n_pairs = min(100_000, n * (n - 1) // 2)
    a = rng.randint(0, n, size=n_pairs)
    b = rng.randint(0, n, size=n_pairs)
    same = a == b
    if same.any():
        b[same] = (b[same] + 1) % n
    sims = np.sum(q_proxy_emb[a] * q_proxy_emb[b], axis=1)

    pca = PCA(n_components=min(256, n - 1))
    pca.fit(q_proxy_emb)
    cum = np.cumsum(pca.explained_variance_ratio_)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(sims, bins=60, color="steelblue", alpha=0.85)
    axes[0].set_xlabel("pairwise cosine similarity")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Q_proxy pairwise similarity ({n_pairs:,} pairs from {n} queries)\n"
                      f"mean={sims.mean():.3f}, median={np.median(sims):.3f}")
    axes[1].plot(range(1, len(cum) + 1), cum, marker=".")
    for tgt in (0.80, 0.90, 0.95):
        k90 = int(np.searchsorted(cum, tgt) + 1) if cum.max() >= tgt else None
        if k90 is not None:
            axes[1].axhline(tgt, ls="--", c="gray", alpha=0.5)
            axes[1].annotate(f"{tgt*100:.0f}% @ k={k90}", xy=(k90, tgt),
                             xytext=(k90 + 5, tgt - 0.02), fontsize=8)
    axes[1].set_xlabel("# PCA components")
    axes[1].set_ylabel("cumulative explained variance")
    axes[1].set_title("Q_proxy effective dimensionality")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "qproxy_quality.png", dpi=120, bbox_inches="tight")
    plt.close()

    km = KMeans(n_clusters=N_THEMES, random_state=42, n_init=10)
    labels = km.fit_predict(q_proxy_emb)
    counts = np.bincount(labels)

    with open(out_dir / "qproxy_themes.txt", "w", encoding="utf-8") as f:
        f.write(f"Q_proxy: {n} queries clustered into {N_THEMES} themes\n")
        f.write(f"Cluster sizes: min={int(counts.min())}, median={int(np.median(counts))}, "
                f"max={int(counts.max())}\n")
        f.write(f"Pairwise sim: mean={sims.mean():.3f}, median={np.median(sims):.3f}, "
                f"std={sims.std():.3f}\n\n")
        for rank, k in enumerate(np.argsort(-counts)):
            mask = labels == k
            if not mask.any():
                continue
            cluster_emb = q_proxy_emb[mask]
            cluster_caps = [q_proxy_captions[i] for i in np.where(mask)[0]]
            cent = km.cluster_centers_[k]
            cent_n = cent / (np.linalg.norm(cent) + 1e-9)
            d = cluster_emb @ cent_n
            top3 = np.argsort(-d)[:3]
            f.write(f"=== Theme {rank+1}/{N_THEMES} (k={k}, n={int(counts[k])}) ===\n")
            for ti in top3:
                f.write(f"  - {cluster_caps[ti][:200]}\n")
            f.write("\n")
    logger.info(f"saved {out_dir / 'qproxy_quality.png'} + {out_dir / 'qproxy_themes.txt'}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = ensure_dir("outputs/diagnostic")

    logger.info("building pool + loading cached embeddings ...")
    anns, shard_roots = build_pool(
        annotations_dir=cfg["data"]["annotations_dir"],
        image_root=cfg["data"]["image_root"],
        sample_size=cfg["data"]["sample_size"],
        seed=cfg["seed"],
    )
    N = len(anns)

    emb_dir = Path(cfg["embed"]["output_dir"])
    img_path = emb_dir / "image_embeddings.npy"
    txt_path = emb_dir / "text_embeddings.npy"
    if not (img_path.exists() and txt_path.exists()):
        logger.error(f"Embeddings not cached at {emb_dir}. Run scripts/run_sweep.py "
                     f"(or the smoke test) once first so embeddings get extracted.")
        return 1
    image_emb = np.load(img_path)
    text_emb = np.load(txt_path)
    logger.info(f"loaded embeddings image{image_emb.shape} text{text_emb.shape}")

    train_idx, _ = make_train_val_split(N, cfg["data"]["val_size"], seed=cfg["seed"])

    q_proxy_emb = None
    q_proxy_captions = None
    try:
        qcache = Path(cfg["qproxy"]["cache_path"])
        if qcache.exists():
            q_proxy_emb = np.load(qcache)
            from src.qproxy import load_qproxy_captions
            try:
                q_proxy_captions = load_qproxy_captions(cfg["qproxy"]["queries_json_path"])
            except Exception:
                q_proxy_captions = [f"<query_{i}>" for i in range(q_proxy_emb.shape[0])]
            logger.info(f"Q_proxy: {q_proxy_emb.shape} ({len(q_proxy_captions)} captions)")
    except KeyError:
        pass

    sels, centroids, assignments = run_selections(cfg, image_emb, text_emb, train_idx, q_proxy_emb)
    logger.info(f"selections done: {list(sels.keys())}")

    train_zv = image_emb[train_idx]
    plot_pca_selections(train_zv, sels, out_dir / "selection_pca.png")
    plot_centroid_dist_hist(train_zv, centroids, assignments, sels, out_dir / "centroid_dist_hist.png")
    plot_image_grid(anns, shard_roots, sels, train_idx, out_dir / "image_grid_v0_vs_proto.png")

    if q_proxy_emb is not None and q_proxy_captions is not None:
        diagnose_qproxy(q_proxy_emb, q_proxy_captions, out_dir)
    else:
        logger.info("Q_proxy not available — skipping qproxy diagnostics")

    logger.info(f"\nDiagnostic outputs in: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
