"""
cawot/scoring.py
================
Text-side query-relevance scoring and budget allocation (plan §3, the heart of
the method).

Pipeline:
  1. For each coarse cluster c and proxy family r, compute the raw text-side
     squared MMD between the cluster caption distribution and the proxy query
     distribution, via RFF:
         d_{c,r} = MMD^2_{k_t}( P_c^t , Q_r )
  2. Normalize per proxy (across clusters) with robust z/IQR -> d~_{c,r}.
     (Plan §3 / ablation: prevents one proxy's scale from inflating u_c.)
  3. Convert to relevance, mean relevance, and proxy disagreement:
         a_{c,r} = exp(-tau * softplus(d~_{c,r}))   in (0, 1]
         abar_c  = sum_r pi_r a_{c,r}
         u_c     = sum_r pi_r (a_{c,r} - abar_c)^2
     (softplus keeps relevance bounded even when normalized d~ goes negative;
      for raw d >= 0 it preserves ordering.)
  4. Allocation score and budget:
         s_c = |C_c|^alpha * (1 + beta*abar_c + eta*u_c)
         b_c = capped_largest_remainder(B * s_c / sum_j s_j, caps=|C_c|)

Backbone is abar_c (mean relevance). u_c is an OPTIONAL uncertainty-aware bonus
controlled by eta; set eta=0 for relevance-only, beta=0 for uncertainty-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kernels import RFF, mmd2_rff


# --------------------------------------------------------------------------- #
#  Raw text-side MMD scores                                                     #
# --------------------------------------------------------------------------- #
def compute_raw_scores(
    text_embeddings: np.ndarray,
    coarse_labels: np.ndarray,
    proxy_embeddings: list[np.ndarray],
    rff_t: RFF,
    n_clusters: int,
) -> np.ndarray:
    """Return d[c, r] = MMD^2_{k_t}(cluster c caption dist, proxy r) via RFF.

    text_embeddings : (N, d) caption embeddings (CLIP text, normalized).
    coarse_labels   : (N,) cluster id per pair.
    proxy_embeddings: list of (m_r, d) query-text embedding arrays.
    """
    R = len(proxy_embeddings)
    d = np.full((n_clusters, R), np.nan, dtype=np.float64)

    # precompute proxy mean embeddings in feature space
    proxy_mu = [rff_t.mean_embedding(Q) for Q in proxy_embeddings]

    for c in range(n_clusters):
        idx = np.where(coarse_labels == c)[0]
        if idx.size == 0:
            continue
        mu_c = rff_t.mean_embedding(text_embeddings[idx])
        for r in range(R):
            d[c, r] = mmd2_rff(mu_c, proxy_mu[r])
    return d


# --------------------------------------------------------------------------- #
#  Per-proxy robust normalization                                               #
# --------------------------------------------------------------------------- #
def normalize_per_proxy(d: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Robust z/IQR normalization across clusters, within each proxy column.

        d~_{c,r} = (d_{c,r} - median_c d_{.,r}) / (IQR_c d_{.,r} + eps)

    Ignores NaN (empty clusters) when computing median/IQR; NaNs stay NaN.
    """
    d_tilde = np.full_like(d, np.nan)
    R = d.shape[1]
    for r in range(R):
        col = d[:, r]
        valid = ~np.isnan(col)
        if valid.sum() == 0:
            continue
        v = col[valid]
        med = np.median(v)
        q75, q25 = np.percentile(v, [75, 25])
        iqr = float(q75 - q25)
        d_tilde[valid, r] = (v - med) / (iqr + eps)
    return d_tilde


# --------------------------------------------------------------------------- #
#  Relevance, mean relevance, disagreement                                      #
# --------------------------------------------------------------------------- #
def relevance_and_uncertainty(
    d_for_relevance: np.ndarray,
    tau: float = 1.0,
    pi: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a_{c,r}, abar_c, u_c.

    d_for_relevance : (K, R) distances used to build relevance (normalized or raw).
    tau             : relevance temperature.
    pi              : (R,) proxy weights (default uniform).

    Relevance uses a BOUNDED transform a = exp(-tau * softplus(d)) in (0, 1].
    Rationale: after per-proxy z/IQR normalization, d~ can be negative (a cluster
    closer than the median), and a naive exp(-tau * d~) would exceed 1 and blow
    up. Passing d through softplus keeps the argument non-negative so relevance
    stays in (0, 1] and abar, u remain bounded and comparable across proxies.
    For raw (non-normalized) d >= 0, softplus is ~ monotone and preserves
    ordering, so behavior is unchanged in the relevance-only/raw setting.
    """
    K, R = d_for_relevance.shape
    if pi is None:
        pi = np.full(R, 1.0 / R)
    pi = np.asarray(pi, dtype=np.float64)
    pi = pi / pi.sum()

    # softplus keeps the argument >= 0 so a in (0, 1]; numerically stable form
    x = d_for_relevance
    softplus = np.where(x > 20.0, x, np.log1p(np.exp(np.clip(x, -50, 20))))
    a = np.exp(-tau * softplus)                   # (K, R), in (0, 1]
    abar = np.full(K, np.nan)
    u = np.full(K, np.nan)
    for c in range(K):
        row = a[c]
        valid = ~np.isnan(row)
        if valid.sum() == 0:
            continue
        w = pi[valid]
        w = w / w.sum()
        ar = row[valid]
        m = float(np.sum(w * ar))
        abar[c] = m
        u[c] = float(np.sum(w * (ar - m) ** 2))
    return a, abar, u


# --------------------------------------------------------------------------- #
#  Allocation score + capped largest-remainder rounding                         #
# --------------------------------------------------------------------------- #
def allocation_scores(
    sizes: np.ndarray,
    abar: np.ndarray,
    u: np.ndarray,
    alpha: float = 0.5,
    beta: float = 1.0,
    eta: float = 1.0,
) -> np.ndarray:
    """s_c = |C_c|^alpha * (1 + beta*abar_c + eta*u_c). Empty clusters -> 0."""
    sizes = np.asarray(sizes, dtype=np.float64)
    abar0 = np.nan_to_num(abar, nan=0.0)
    u0 = np.nan_to_num(u, nan=0.0)
    s = (sizes ** alpha) * (1.0 + beta * abar0 + eta * u0)
    s[sizes <= 0] = 0.0
    return s


def capped_largest_remainder(
    target: np.ndarray,
    caps: np.ndarray,
    total: int,
) -> np.ndarray:
    """Integer allocation summing to ``total``, with per-bin caps.

    Standard largest-remainder (Hamilton) apportionment with capacity limits:
      * floor each target, clip to cap;
      * distribute the remaining units by largest fractional remainder, skipping
        bins already at cap;
      * if caps make ``total`` infeasible, allocate as much as possible.
    """
    target = np.asarray(target, dtype=np.float64).copy()
    caps = np.asarray(caps, dtype=np.int64)
    K = target.size

    target = np.maximum(target, 0.0)
    base = np.floor(target).astype(np.int64)
    base = np.minimum(base, caps)
    remainder = target - np.floor(target)

    allocated = int(base.sum())
    leftover = int(total) - allocated
    if leftover <= 0:
        return base

    # candidates that still have capacity
    order = np.argsort(-remainder)  # largest remainder first
    i = 0
    guard = 0
    max_guard = 10 * K + leftover + 10
    while leftover > 0 and guard < max_guard:
        if i >= K:
            i = 0
            # re-sort by remaining capacity priority if all remainders exhausted
            order = np.argsort(-(caps - base).astype(np.float64))
        c = order[i]
        if base[c] < caps[c]:
            base[c] += 1
            leftover -= 1
        i += 1
        guard += 1
        if (caps - base).sum() <= 0:
            break
    return base


def split_to_fine(
    coarse_budget: np.ndarray,
    fine_sizes: np.ndarray,
    k_fine: int,
) -> np.ndarray:
    """Split each coarse budget across its fine sub-clusters proportionally.

    coarse_budget : (k_coarse,) integer budgets.
    fine_sizes    : (k_coarse * k_fine,) sizes, fine id = coarse*k_fine + local.
    Returns (k_coarse * k_fine,) integer budgets, capped by fine sizes.
    """
    k_coarse = coarse_budget.shape[0]
    out = np.zeros(k_coarse * k_fine, dtype=np.int64)
    for c in range(k_coarse):
        bc = int(coarse_budget[c])
        if bc <= 0:
            continue
        ids = np.arange(c * k_fine, (c + 1) * k_fine)
        sz = fine_sizes[ids].astype(np.float64)
        if sz.sum() <= 0:
            continue
        target = bc * sz / sz.sum()
        out[ids] = capped_largest_remainder(target, fine_sizes[ids], bc)
    return out


# --------------------------------------------------------------------------- #
#  End-to-end allocation bundle (with full logging)                             #
# --------------------------------------------------------------------------- #
@dataclass
class AllocationResult:
    d_raw: np.ndarray                  # (K, R)
    d_tilde: np.ndarray                # (K, R)
    a: np.ndarray                      # (K, R)
    abar: np.ndarray                   # (K,)
    u: np.ndarray                      # (K,)
    s: np.ndarray                      # (K,)
    coarse_budget: np.ndarray          # (K,)
    fine_budget: np.ndarray            # (K * k_fine,)
    sizes: np.ndarray                  # (K,)
    config: dict = field(default_factory=dict)


def allocate(
    text_embeddings: np.ndarray,
    coarse_labels: np.ndarray,
    fine_labels: np.ndarray,
    proxy_embeddings: list[np.ndarray],
    rff_t: RFF,
    k_coarse: int,
    k_fine: int,
    budget_total: int,
    *,
    alpha: float = 0.5,
    beta: float = 1.0,
    eta: float = 1.0,
    tau: float = 1.0,
    pi: np.ndarray | None = None,
    normalize: bool = True,
) -> AllocationResult:
    """Run the full scoring+allocation path and return everything for logging."""
    from .clustering import cluster_sizes

    d_raw = compute_raw_scores(
        text_embeddings, coarse_labels, proxy_embeddings, rff_t, k_coarse)
    d_tilde = normalize_per_proxy(d_raw) if normalize else d_raw.copy()
    d_for_rel = d_tilde if normalize else d_raw

    a, abar, u = relevance_and_uncertainty(d_for_rel, tau=tau, pi=pi)

    sizes = cluster_sizes(coarse_labels, k_coarse)
    s = allocation_scores(sizes, abar, u, alpha=alpha, beta=beta, eta=eta)

    denom = s.sum()
    if denom <= 0:
        target = np.zeros_like(s)
    else:
        target = budget_total * s / denom
    coarse_budget = capped_largest_remainder(target, sizes, budget_total)

    fine_sizes = cluster_sizes(fine_labels, k_coarse * k_fine)
    fine_budget = split_to_fine(coarse_budget, fine_sizes, k_fine)

    return AllocationResult(
        d_raw=d_raw, d_tilde=d_tilde, a=a, abar=abar, u=u, s=s,
        coarse_budget=coarse_budget, fine_budget=fine_budget, sizes=sizes,
        config=dict(alpha=alpha, beta=beta, eta=eta, tau=tau,
                    normalize=normalize, budget_total=int(budget_total),
                    k_coarse=k_coarse, k_fine=k_fine),
    )
