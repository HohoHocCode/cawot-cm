"""Coreset selection methods. All return int64 indices into the train pool.

Baselines / V0 family (cluster on IMAGE embeddings, proportional budget):
  - select_random       : uniform random.
  - select_v0           : farthest-from-centroid within each cluster
                          (the plan's "diversity-only" baseline — selects
                          atypical/boundary points).
  - select_v0_proto     : closest-to-centroid within each cluster
                          (prototype selection — selects representative points;
                          diagnostic for the Sorscher low-budget hypothesis).

V1 (the method):
  - select_v1           : cross-modal cost + submodular facility location
                          within each cluster. Picks samples that *cover* the
                          cluster in a joint image+text+alignment space, rather
                          than extremes.

Clustering is always on image embeddings (per plan). V1 additionally uses text
embeddings to build the cross-modal cost.
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

    Largest-remainder rounding so allocations sum exactly to total_budget,
    capped at each cluster's size.
    """
    sizes = cluster_sizes.astype(np.float64)
    N = sizes.sum()
    raw = sizes / N * total_budget
    floor = np.floor(raw).astype(np.int64)
    remainder = raw - floor
    deficit = total_budget - floor.sum()
    if deficit > 0:
        order = np.argsort(-remainder)
        for i in order:
            if deficit == 0:
                break
            if floor[i] < int(sizes[i]):
                floor[i] += 1
                deficit -= 1
    return np.minimum(floor, sizes.astype(np.int64))


# -----------------------------------------------------------------------------
# Centroid-distance family: V0 (farthest) and V0-proto (closest)
# -----------------------------------------------------------------------------


def _select_by_centroid(
    embeddings: np.ndarray,
    centroids: np.ndarray,
    assignments: np.ndarray,
    budget: int,
    seed: int,
    farthest: bool,
) -> np.ndarray:
    K = centroids.shape[0]
    rng = np.random.RandomState(seed)
    cluster_sizes = np.bincount(assignments, minlength=K)
    budgets = _proportional_budgets(cluster_sizes, budget)

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
        sims = embeddings[idx_k] @ centroids[k]          # cosine to centroid
        dists = 1.0 - sims
        jitter = rng.uniform(0, 1e-9, size=dists.shape)  # break ties
        order = np.argsort(-(dists + jitter)) if farthest else np.argsort(dists + jitter)
        selected.append(idx_k[order[:b_k]])

    out = np.sort(np.concatenate(selected)).astype(np.int64) if selected else np.empty((0,), np.int64)
    logger.info(f"[{'v0-farthest' if farthest else 'v0-proto'}] selected {len(out)} (target {budget})")
    return out


def select_v0(embeddings, centroids, assignments, budget, seed=42):
    """Farthest-from-centroid (diversity / atypical selection)."""
    return _select_by_centroid(embeddings, centroids, assignments, budget, seed, farthest=True)


def select_v0_proto(embeddings, centroids, assignments, budget, seed=42):
    """Closest-to-centroid (prototype / representative selection)."""
    return _select_by_centroid(embeddings, centroids, assignments, budget, seed, farthest=False)


# -----------------------------------------------------------------------------
# V1: cross-modal cost + facility location
# -----------------------------------------------------------------------------


def _rank_norm(D: np.ndarray) -> np.ndarray:
    """Rank-normalize matrix entries to [0, 1] (0 = smallest distance)."""
    flat = D.ravel()
    order = flat.argsort()
    ranks = np.empty(len(flat), dtype=np.float64)
    ranks[order] = np.arange(len(flat))
    return (ranks / max(1, len(flat) - 1)).reshape(D.shape)


def cross_modal_similarity(zv: np.ndarray, zt: np.ndarray) -> np.ndarray:
    """Cross-modal similarity kernel for facility location.

    Cost (plan):  c_ij = ( rank(d_v) + rank(d_t) + rank(d_a) ) / 3
      d_v = 1 - cos(image_i, image_j)
      d_t = 1 - cos(text_i,  text_j)
      d_a = |a_i - a_j|,  a_i = cos(image_i, text_i)   (alignment quality)
    Similarity = 1 - c, diagonal forced to 1 (a point represents itself).

    zv, zt must be L2-normalized (cosine = dot product).
    """
    Dv = 1.0 - zv @ zv.T
    Dt = 1.0 - zt @ zt.T
    a = np.sum(zv * zt, axis=1)                 # (n,) alignment per sample
    Da = np.abs(a[:, None] - a[None, :])
    c = (_rank_norm(Dv) + _rank_norm(Dt) + _rank_norm(Da)) / 3.0
    sim = 1.0 - c
    np.fill_diagonal(sim, 1.0)
    return sim


def facility_location_greedy(sim: np.ndarray, k: int) -> np.ndarray:
    """Greedy maximization of facility location  f(S) = Σ_i max_{s∈S} sim[i,s].

    Exact lazy-free vectorized greedy (1 − 1/e guarantee). O(k·n²); fine at the
    per-cluster scale (n ≤ ~1000). For a full-1M run, swap in submodlib's C++
    LazyGreedy with the same kernel.

    Returns local indices (length min(k, n)) in greedy-selection order.
    """
    n = sim.shape[0]
    k = min(k, n)
    coverage = np.zeros(n, dtype=np.float64)   # max sim to current selection
    available = np.ones(n, dtype=bool)
    selected: list[int] = []
    for _ in range(k):
        gains = np.maximum(sim - coverage[:, None], 0.0).sum(axis=0)
        gains[~available] = -np.inf
        j = int(np.argmax(gains))
        selected.append(j)
        available[j] = False
        coverage = np.maximum(coverage, sim[:, j])
    return np.asarray(selected, dtype=np.int64)


def select_v1(
    zv: np.ndarray,
    zt: np.ndarray,
    centroids: np.ndarray,
    assignments: np.ndarray,
    budget: int,
    seed: int = 42,
) -> np.ndarray:
    """V1: within each cluster, facility location on the cross-modal cost.

    zv, zt: (N, d) L2-normalized image / text embeddings.
    centroids/assignments: from k-means on image embeddings (same as V0).
    """
    K = centroids.shape[0]
    cluster_sizes = np.bincount(assignments, minlength=K)
    budgets = _proportional_budgets(cluster_sizes, budget)

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
        sim = cross_modal_similarity(zv[idx_k], zt[idx_k])
        local = facility_location_greedy(sim, b_k)
        selected.append(idx_k[local])

    out = np.sort(np.concatenate(selected)).astype(np.int64) if selected else np.empty((0,), np.int64)
    logger.info(f"[v1] selected {len(out)} (target {budget})")
    return out
