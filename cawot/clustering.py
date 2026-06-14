"""
cawot/clustering.py
===================
Two-level approximate kernel-aware clustering (plan step 2).

We do NOT run exact kernel k-means. Instead we run mini-batch k-means on an
explicit feature map:
  * additive pair kernel  -> features [sqrt(lam) z_v, sqrt(1-lam) z_t]
  * (RBF variant)         -> RFF features of the pair feature map

This keeps clustering operational at 1M scale while remaining consistent with
the kernel used downstream. The feature vector here is a clustering feature
map, not a fused pair representation for scoring.
"""
from __future__ import annotations

import numpy as np

try:
    from sklearn.cluster import MiniBatchKMeans
    _HAS_SK = True
except Exception:  # pragma: no cover
    _HAS_SK = False


def _kmeans(X: np.ndarray, k: int, seed: int, batch_size: int = 4096,
            n_init: int = 3, max_iter: int = 100) -> np.ndarray:
    """Return integer labels in [0, k) for rows of X."""
    if not _HAS_SK:
        raise RuntimeError("scikit-learn is required for clustering.")
    k = int(min(k, X.shape[0]))
    km = MiniBatchKMeans(
        n_clusters=k, random_state=seed, batch_size=batch_size,
        n_init=n_init, max_iter=max_iter, reassignment_ratio=0.01,
    )
    return km.fit_predict(X).astype(np.int64)


def two_level_clustering(
    pair_features: np.ndarray,
    k_coarse: int = 20,
    k_fine: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Hierarchical clustering on a precomputed pair-feature matrix.

    Parameters
    ----------
    pair_features : (N, F) explicit feature map of each pair.
    k_coarse, k_fine : cluster counts per level.

    Returns
    -------
    coarse_labels : (N,) in [0, k_coarse)
    fine_labels   : (N,) global fine id in [0, k_coarse * k_fine), defined as
                    coarse_id * k_fine + local_fine_id.
    """
    N = pair_features.shape[0]
    coarse = _kmeans(pair_features, k_coarse, seed)
    fine = np.full(N, -1, dtype=np.int64)
    for c in range(int(coarse.max()) + 1):
        idx = np.where(coarse == c)[0]
        if idx.size == 0:
            continue
        sub = pair_features[idx]
        local = _kmeans(sub, min(k_fine, idx.size), seed + 1 + c)
        fine[idx] = c * k_fine + local
    return coarse, fine


def cluster_sizes(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """Counts per cluster id in [0, n_clusters)."""
    out = np.zeros(n_clusters, dtype=np.int64)
    u, cnt = np.unique(labels[labels >= 0], return_counts=True)
    out[u] = cnt
    return out
