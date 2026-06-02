"""V2.1 coreset selection.

V2.1 implements the redesign in the Claude plan:
  1. coarse-cluster Sliced Wasserstein gap instead of Gaussian diagonal W2,
  2. hierarchical coarse/fine clusters,
  3. image+text weighted similarity without alignment rank cost,
  4. prototype-conditioned submodular greedy inside fine clusters.

The old V2 implementation remains in src.select for direct comparison.
"""
from __future__ import annotations

import numpy as np

from .select import _capped_largest_remainder, wasserstein_aware_budgets
from .utils import setup_logger

logger = setup_logger("select_v2_1")


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def make_random_projections(dim: int, num_projections: int, seed: int) -> np.ndarray:
    """Random unit directions for sliced Wasserstein projections."""
    if num_projections <= 0:
        raise ValueError(f"num_projections must be positive, got {num_projections}")
    rng = np.random.RandomState(seed)
    proj = rng.normal(size=(num_projections, dim)).astype(np.float32)
    return _normalize_rows(proj)


def exact_w2_1d_squared(x: np.ndarray, y: np.ndarray) -> float:
    """Exact squared W2 between two equally weighted empirical 1D samples.

    This integrates the squared difference of the two empirical quantile
    functions. It handles unequal sample counts without interpolation bias.
    """
    xs = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    ys = np.sort(np.asarray(y, dtype=np.float64).reshape(-1))
    n = len(xs)
    m = len(ys)
    if n == 0 or m == 0:
        raise ValueError("exact_w2_1d_squared needs two non-empty samples")

    i = 0
    j = 0
    u_prev = 0.0
    total = 0.0
    while i < n and j < m:
        u_next = min((i + 1) / n, (j + 1) / m)
        diff = xs[i] - ys[j]
        total += (u_next - u_prev) * diff * diff
        u_prev = u_next
        if (i + 1) / n <= u_next + 1e-15:
            i += 1
        if (j + 1) / m <= u_next + 1e-15:
            j += 1
    return float(total)


def sliced_wasserstein_distance(
    x: np.ndarray,
    y: np.ndarray,
    *,
    num_projections: int = 128,
    seed: int = 42,
    projections: np.ndarray | None = None,
) -> float:
    """Sliced W2 distance between two high-dimensional empirical sets."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError(f"x and y must be 2D with same dim, got {x.shape} and {y.shape}")
    if len(x) == 0 or len(y) == 0:
        raise ValueError("sliced_wasserstein_distance needs non-empty inputs")
    if projections is None:
        projections = make_random_projections(x.shape[1], num_projections, seed)
    projections = np.asarray(projections, dtype=np.float32)
    if projections.ndim != 2 or projections.shape[1] != x.shape[1]:
        raise ValueError(
            f"projections must have shape (L, {x.shape[1]}), got {projections.shape}"
        )

    xp = x @ projections.T
    yp = y @ projections.T
    sq = [exact_w2_1d_squared(xp[:, l], yp[:, l]) for l in range(projections.shape[0])]
    return float(np.sqrt(np.mean(sq)))


def compute_coarse_sliced_wasserstein_gaps(
    text_emb: np.ndarray,
    coarse_assignments: np.ndarray,
    q_proxy_emb: np.ndarray,
    k_coarse: int,
    *,
    num_projections: int = 128,
    seed: int = 42,
) -> np.ndarray:
    """Compute SW2(Q_proxy, cluster text embeddings) for each coarse cluster."""
    projections = make_random_projections(text_emb.shape[1], num_projections, seed)
    q_proj = q_proxy_emb @ projections.T
    gaps = np.zeros(k_coarse, dtype=np.float64)
    for k in range(k_coarse):
        idx = np.where(coarse_assignments == k)[0]
        if len(idx) == 0:
            gaps[k] = -1.0
            continue
        c_proj = text_emb[idx] @ projections.T
        sq = [exact_w2_1d_squared(c_proj[:, l], q_proj[:, l]) for l in range(num_projections)]
        gaps[k] = float(np.sqrt(np.mean(sq)))

    valid = gaps[gaps >= 0]
    fallback = float(valid.mean()) if len(valid) else 0.0
    gaps[gaps < 0] = fallback
    return gaps


def weighted_image_text_similarity(
    zv: np.ndarray,
    zt: np.ndarray,
    *,
    lambda_image: float = 0.7,
) -> np.ndarray:
    """Non-negative image/text similarity used by V2.1 facility coverage."""
    if not (0.0 <= lambda_image <= 1.0):
        raise ValueError(f"lambda_image must be in [0, 1], got {lambda_image}")
    lambda_text = 1.0 - lambda_image
    cos = lambda_image * (zv @ zv.T) + lambda_text * (zt @ zt.T)
    sim = np.clip(0.5 * (cos + 1.0), 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    return sim.astype(np.float64)


def prototype_conditioned_greedy(
    sim: np.ndarray,
    prototype_scores: np.ndarray,
    k: int,
    *,
    alpha: float = 0.75,
    seed: int = 42,
) -> np.ndarray:
    """Greedy for modular prototype centrality + facility-location coverage.

    alpha=1 selects top prototype scores (V0_proto behavior).
    alpha=0 is pure facility location on the provided image/text similarity.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    sim = np.asarray(sim, dtype=np.float64)
    prototype_scores = np.asarray(prototype_scores, dtype=np.float64).reshape(-1)
    n = sim.shape[0]
    if sim.shape != (n, n) or prototype_scores.shape[0] != n:
        raise ValueError("sim must be square and prototype_scores must match its size")
    k = min(int(k), n)
    if k <= 0:
        return np.empty((0,), dtype=np.int64)

    rng = np.random.RandomState(seed)
    coverage = np.zeros(n, dtype=np.float64)
    available = np.ones(n, dtype=bool)
    selected: list[int] = []
    jitter = rng.uniform(0, 1e-12, size=n)
    for _ in range(k):
        facility_gain = np.maximum(sim - coverage[:, None], 0.0).mean(axis=0)
        gains = alpha * prototype_scores + (1.0 - alpha) * facility_gain + jitter
        gains[~available] = -np.inf
        j = int(np.argmax(gains))
        selected.append(j)
        available[j] = False
        coverage = np.maximum(coverage, sim[:, j])
    return np.asarray(selected, dtype=np.int64)


def _fine_budgets_for_coarse(
    fine_assignments_for_coarse: np.ndarray,
    fine_valid_for_coarse: np.ndarray,
    coarse_budget: int,
) -> np.ndarray:
    sizes = np.array(
        [
            int((fine_assignments_for_coarse == f).sum()) if fine_valid_for_coarse[f] else 0
            for f in range(len(fine_valid_for_coarse))
        ],
        dtype=np.int64,
    )
    if coarse_budget == 0:
        return np.zeros_like(sizes)
    return _capped_largest_remainder(sizes.astype(np.float64), sizes, coarse_budget)


def select_v2_1(
    *,
    zv: np.ndarray,
    zt: np.ndarray,
    coarse_centroids: np.ndarray,
    coarse_assignments: np.ndarray,
    fine_centroids: np.ndarray,
    fine_assignments: np.ndarray,
    fine_valid: np.ndarray,
    q_proxy_emb: np.ndarray,
    budget: int,
    alpha: float = 0.75,
    lambda_image: float = 0.7,
    v2_alpha: float = 0.5,
    num_projections: int = 128,
    seed: int = 42,
) -> np.ndarray:
    """Select a V2.1 coreset from hierarchical cluster assignments."""
    n = len(zv)
    if budget < 0 or budget > n:
        raise ValueError(f"budget must be in [0, {n}], got {budget}")
    if budget == 0:
        return np.empty((0,), dtype=np.int64)

    k_coarse = coarse_centroids.shape[0]
    coarse_sizes = np.bincount(coarse_assignments, minlength=k_coarse).astype(np.int64)
    gaps = compute_coarse_sliced_wasserstein_gaps(
        zt,
        coarse_assignments,
        q_proxy_emb,
        k_coarse,
        num_projections=num_projections,
        seed=seed,
    )
    coarse_budgets = wasserstein_aware_budgets(
        coarse_sizes,
        gaps,
        total_budget=budget,
        alpha=v2_alpha,
    )

    selected: list[np.ndarray] = []
    for c in range(k_coarse):
        b_c = int(coarse_budgets[c])
        if b_c == 0:
            continue
        idx_c = np.where(coarse_assignments == c)[0]
        if len(idx_c) == 0:
            continue
        if b_c >= len(idx_c):
            selected.append(idx_c)
            continue

        fine_local = fine_assignments[idx_c]
        fine_budgets = _fine_budgets_for_coarse(fine_local, fine_valid[c], b_c)
        for f, b_f in enumerate(fine_budgets.tolist()):
            if b_f == 0:
                continue
            idx_cf = idx_c[fine_local == f]
            if len(idx_cf) == 0:
                continue
            if b_f >= len(idx_cf):
                selected.append(idx_cf)
                continue

            sim = weighted_image_text_similarity(
                zv[idx_cf],
                zt[idx_cf],
                lambda_image=lambda_image,
            )
            proto_cos = zv[idx_cf] @ fine_centroids[c, f]
            proto_scores = np.clip(0.5 * (proto_cos + 1.0), 0.0, 1.0)
            local = prototype_conditioned_greedy(
                sim,
                proto_scores,
                b_f,
                alpha=alpha,
                seed=seed + 1009 * c + f,
            )
            selected.append(idx_cf[local])

    out = np.sort(np.concatenate(selected)).astype(np.int64) if selected else np.empty((0,), np.int64)
    if len(out) != budget:
        raise RuntimeError(f"V2.1 selected {len(out)} samples, expected {budget}")
    logger.info(
        f"[v2_1] selected {len(out)} (target {budget}); "
        f"SW2 mean={gaps.mean():.4f} min={gaps.min():.4f} max={gaps.max():.4f}; "
        f"coarse budget min/median/max={coarse_budgets.min()}/"
        f"{int(np.median(coarse_budgets))}/{coarse_budgets.max()}"
    )
    return out
