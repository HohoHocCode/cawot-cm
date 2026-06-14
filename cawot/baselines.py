"""
cawot/baselines.py
==================
Selection baselines for the Stage-B comparison (plan §7.3).

All selectors share the signature:
    select(budget, *, rng, **ctx) -> np.ndarray  (global indices)

Context (``ctx``) provides whatever a method needs:
    image_emb, text_emb, pair_features, proxy_text_emb (list), coarse_labels...

Included here:
    * random
    * clipscore         (mean cosine of caption to proxy-query centroid)
    * kcenter           (greedy farthest-point on pair features)
    * sw_cawot          (poster-era sliced-Wasserstein scoring + proportional
                         allocation; kept as an internal ablation/evolution)

The proposed method lives in scoring.py + selection.py and is wired in pipeline.py.
"""
from __future__ import annotations

import numpy as np


def select_random(budget: int, n: int, rng: np.random.Generator) -> np.ndarray:
    budget = int(min(budget, n))
    return np.sort(rng.choice(n, size=budget, replace=False))


def select_clipscore(
    budget: int,
    text_emb: np.ndarray,
    proxy_text_emb: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Top-budget captions by mean cosine similarity to proxy-query centroids.

    A sample-level relevance baseline (the thing our cluster-level distributional
    method must beat). Embeddings assumed L2-normalized.
    """
    n = text_emb.shape[0]
    budget = int(min(budget, n))
    centroids = np.stack([Q.mean(axis=0) for Q in proxy_text_emb], axis=0)
    centroids /= (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    sims = text_emb @ centroids.T            # (n, R)
    score = sims.mean(axis=1)
    return np.sort(np.argpartition(-score, kth=budget - 1)[:budget])


def select_kcenter(
    budget: int,
    pair_features: np.ndarray,
    rng: np.random.Generator,
    seed_point: int | None = None,
) -> np.ndarray:
    """Greedy k-center (farthest-point sampling) on pair features.

    O(n * budget). Uses squared Euclidean distance in feature space.
    """
    n = pair_features.shape[0]
    budget = int(min(budget, n))
    F = pair_features.astype(np.float32)
    first = int(rng.integers(n)) if seed_point is None else seed_point
    selected = [first]
    d2 = np.sum((F - F[first]) ** 2, axis=1)
    for _ in range(1, budget):
        nxt = int(np.argmax(d2))
        selected.append(nxt)
        nd = np.sum((F - F[nxt]) ** 2, axis=1)
        d2 = np.minimum(d2, nd)
    return np.sort(np.array(selected, dtype=np.int64))


# --------------------------------------------------------------------------- #
#  SW-CAWOT (poster-era) -- sliced-Wasserstein scoring + proportional alloc     #
# --------------------------------------------------------------------------- #
def _sliced_wasserstein_1d(a: np.ndarray, b: np.ndarray, n_proj: int,
                           rng: np.random.Generator) -> float:
    """Sliced 2-Wasserstein^2 between two point clouds in R^d via random 1D projections."""
    d = a.shape[1]
    proj = rng.normal(size=(d, n_proj)).astype(np.float32)
    proj /= (np.linalg.norm(proj, axis=0, keepdims=True) + 1e-12)
    pa = np.sort(a @ proj, axis=0)     # (na, n_proj)
    pb = np.sort(b @ proj, axis=0)     # (nb, n_proj)
    # interpolate to common quantile grid
    q = np.linspace(0, 1, num=min(len(pa), len(pb), 256))
    ia = (q * (len(pa) - 1)).astype(int)
    ib = (q * (len(pb) - 1)).astype(int)
    diff = pa[ia] - pb[ib]
    return float(np.mean(diff ** 2))


def select_sw_cawot(
    budget: int,
    pair_features_raw: np.ndarray,   # use a low-dim view (e.g. concat z_v,z_t)
    coarse_labels: np.ndarray,
    proxy_pair_features: np.ndarray,  # poster used pooled query rep in same space
    k_coarse: int,
    rng: np.random.Generator,
    n_proj: int = 128,
    alpha: float = 0.5,
):
    """Poster-era selector: score clusters by SW gap to the query rep, allocate
    budget proportional to |C|^alpha * (1 + SW_gap), select randomly within.

    This is intentionally faithful to the poster (proportional-to-gap, random
    within-cluster) so the paper can show the MMD method matches or beats it
    while being cleaner / more scalable.
    """
    from .clustering import cluster_sizes
    from .scoring import capped_largest_remainder

    n = pair_features_raw.shape[0]
    budget = int(min(budget, n))
    gaps = np.zeros(k_coarse)
    for c in range(k_coarse):
        idx = np.where(coarse_labels == c)[0]
        if idx.size == 0:
            continue
        gaps[c] = _sliced_wasserstein_1d(
            pair_features_raw[idx], proxy_pair_features, n_proj, rng)
    # normalize gaps to [0,1] for stability
    if gaps.max() > 0:
        gaps = gaps / gaps.max()
    sizes = cluster_sizes(coarse_labels, k_coarse)
    s = (sizes.astype(np.float64) ** alpha) * (1.0 + gaps)
    s[sizes <= 0] = 0.0
    target = budget * s / s.sum() if s.sum() > 0 else np.zeros_like(s)
    cbud = capped_largest_remainder(target, sizes, budget)

    selected: list[int] = []
    for c in range(k_coarse):
        b = int(cbud[c])
        if b <= 0:
            continue
        idx = np.where(coarse_labels == c)[0]
        pick = rng.choice(idx, size=min(b, idx.size), replace=False)
        selected.extend(pick.tolist())
    return np.array(sorted(set(selected)), dtype=np.int64)
