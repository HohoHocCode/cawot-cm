"""Coreset selection methods.

V0:    cluster + proportional budget + farthest-from-centroid within cluster.
Random: uniform random sampling baseline.

Output: an int64 array of indices into the train pool.
"""
from __future__ import annotations

import numpy as np

from .utils import setup_logger

logger = setup_logger("select")


def select_random(n_total: int, budget: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n_total, size=budget, replace=False)).astype(np.int64)


def _proportional_budgets(cluster_sizes: np.ndarray, total_budget: int) -> np.ndarray:
    """Allocate total_budget across clusters proportional to size.

    Uses largest-remainder rounding so allocations sum exactly to total_budget,
    and caps each cluster's allocation at its size.
    """
    sizes = cluster_sizes.astype(np.float64)
    N = sizes.sum()
    raw = sizes / N * total_budget
    floor = np.floor(raw).astype(np.int64)
    remainder = raw - floor
    deficit = total_budget - floor.sum()
    if deficit > 0:
        # Largest remainders get +1 (but cap by size)
        order = np.argsort(-remainder)
        for i in order:
            if deficit == 0:
                break
            if floor[i] < int(sizes[i]):
                floor[i] += 1
                deficit -= 1
    floor = np.minimum(floor, sizes.astype(np.int64))
    return floor


def select_v0(
    embeddings: np.ndarray,
    centroids: np.ndarray,
    assignments: np.ndarray,
    budget: int,
    seed: int = 42,
) -> np.ndarray:
    """V0: farthest-from-centroid within each cluster.

    embeddings: (N, d) L2-normalized image embeddings.
    centroids:  (K, d) cluster centroids (from spherical k-means → normalized).
    assignments: (N,) int cluster id per sample.
    budget: int total samples to select.
    """
    N, d = embeddings.shape
    K = centroids.shape[0]
    rng = np.random.RandomState(seed)

    # Cluster sizes
    cluster_sizes = np.bincount(assignments, minlength=K)
    logger.info(f"Clusters: {K}, sizes min/median/max = "
                f"{cluster_sizes.min()}/{int(np.median(cluster_sizes))}/{cluster_sizes.max()}")

    budgets = _proportional_budgets(cluster_sizes, budget)
    logger.info(f"Allocated budgets: total={budgets.sum()} (target={budget}), "
                f"min/median/max = {budgets.min()}/{int(np.median(budgets))}/{budgets.max()}")

    # For each cluster, pick farthest-from-centroid samples.
    # We use cosine *distance* = 1 - <x, c> for normalized vectors.
    # Larger distance → farther from centroid → more diverse.
    selected: list[np.ndarray] = []
    for k in range(K):
        b_k = int(budgets[k])
        if b_k == 0:
            continue
        idx_k = np.where(assignments == k)[0]
        if len(idx_k) == 0:
            continue
        if b_k >= len(idx_k):
            selected.append(idx_k)
            continue
        sims = embeddings[idx_k] @ centroids[k]   # (n_k,)
        dists = 1.0 - sims
        # Tie-breaking with small jitter so equally-distant points don't bias
        jitter = rng.uniform(0, 1e-9, size=dists.shape)
        order = np.argsort(-(dists + jitter))     # descending distance
        selected.append(idx_k[order[:b_k]])

    out = np.concatenate(selected) if selected else np.empty((0,), dtype=np.int64)
    out = np.sort(out).astype(np.int64)
    logger.info(f"Selected {len(out)} indices (target {budget})")
    return out
