"""
cawot/selection.py
==================
Within-cluster sample selection (plan steps 6-7).

We replace the old O(m^3) dense facility location with:
  * step 6: k-NN sparsification (FAISS if available, else exact numpy kNN);
  * step 7: approximate kernel herding (Frank-Wolfe) on the pair feature map.

Honesty notes (plan §5, Lemma 4):
  * Kernel herding reduces MMD(coreset, cluster). Its O(1/T) rate holds only
    under a marginal-polytope-interior condition; otherwise it is O(1/sqrt(T)).
    We claim the safe O(1/sqrt(b)) and report faster decay if observed.
  * The MMD-herding guarantee is for the FULL-candidate version. Our default
    uses full-candidate herding on the (already small) sub-cluster, so the
    guarantee applies directly. The k-NN graph is used to bound memory/time
    when a sub-cluster is large; with sparse/local updates the objective becomes
    a sparse approximation -- we verify empirically it does not degrade R@k.
  * This is kernel herding, NOT kernel thinning / Compress++ (a separate line,
    related-work only).
"""
from __future__ import annotations

import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except Exception:  # pragma: no cover
    _HAS_FAISS = False


# --------------------------------------------------------------------------- #
#  Step 6 -- k-NN graph (used only to cap cost for large sub-clusters)          #
# --------------------------------------------------------------------------- #
def knn_graph(features: np.ndarray, k: int = 32) -> np.ndarray:
    """Return (n, k) indices of nearest neighbors (inner-product / cosine).

    Features are expected to be RFF pair features (already reflecting the
    kernel). With FAISS we use an inner-product index; otherwise exact numpy.
    """
    n = features.shape[0]
    k = int(min(k, max(1, n - 1)))
    f = np.ascontiguousarray(features.astype(np.float32))
    if _HAS_FAISS and n > 2048:
        index = faiss.IndexFlatIP(f.shape[1])
        index.add(f)
        _, idx = index.search(f, k + 1)  # +1 because self is included
        return idx[:, 1:]
    # exact fallback
    sims = f @ f.T
    np.fill_diagonal(sims, -np.inf)
    return np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]


# --------------------------------------------------------------------------- #
#  Step 7 -- approximate kernel herding (Frank-Wolfe)                           #
# --------------------------------------------------------------------------- #
def kernel_herding(
    features: np.ndarray,
    budget: int,
    candidate_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Greedy kernel herding to minimize MMD(selected, all) in feature space.

    Selects ``budget`` points from rows of ``features`` (a feature map such that
    <psi(x), psi(y)> approximates the kernel). At each step pick the point that
    best matches the residual between the target mean embedding and the current
    selection's mean embedding:

        x_{t+1} = argmax_x  < psi(x), mu - (1/t) sum_{i<=t} psi(x_i) >

    Parameters
    ----------
    features : (m, F) feature rows.
    budget   : number of points to select.
    candidate_idx : optional subset of row indices to restrict the search
                    (e.g. union of a k-NN neighborhood). If None, all rows are
                    candidates (full-candidate herding -- guarantee applies).

    Returns
    -------
    selected : (budget,) row indices into ``features``.
    """
    m = features.shape[0]
    budget = int(min(budget, m))
    if budget <= 0:
        return np.empty(0, dtype=np.int64)
    if budget == m:
        return np.arange(m, dtype=np.int64)

    F = features.astype(np.float32)
    mu = F.mean(axis=0)                       # target mean embedding (F,)

    if candidate_idx is None:
        cand = np.arange(m)
    else:
        cand = np.unique(candidate_idx.astype(np.int64))

    selected: list[int] = []
    running_sum = np.zeros(F.shape[1], dtype=np.float32)
    chosen_mask = np.zeros(m, dtype=bool)

    for t in range(budget):
        # standard kernel-herding objective:
        #   x_{t+1} = argmax_x  < psi(x),  mu - (1/t) sum_{i<=t} psi(x_i) >
        # at t == 0 there is no selection yet, so the residual is just mu.
        if t == 0:
            residual = mu
        else:
            residual = mu - running_sum / t
        scores = F[cand] @ residual
        # forbid re-selection
        scores[chosen_mask[cand]] = -np.inf
        best_local = int(np.argmax(scores))
        best = int(cand[best_local])
        selected.append(best)
        chosen_mask[best] = True
        running_sum += F[best]

    return np.asarray(selected, dtype=np.int64)


# --------------------------------------------------------------------------- #
#  Orchestration over sub-clusters                                              #
# --------------------------------------------------------------------------- #
def select_within_clusters(
    pair_features: np.ndarray,
    fine_labels: np.ndarray,
    fine_budget: np.ndarray,
    k_nn: int = 32,
    large_cluster_threshold: int = 5000,
) -> np.ndarray:
    """Select a coreset by running herding inside every fine sub-cluster.

    For small sub-clusters (<= threshold) we run full-candidate herding (the
    MMD guarantee applies). For larger ones we restrict candidates to a k-NN
    neighborhood union to cap cost (sparse approximation).

    Returns global row indices of the selected coreset (sorted).
    """
    selected_all: list[int] = []
    n_fine = fine_budget.shape[0]

    for f in range(n_fine):
        b = int(fine_budget[f])
        if b <= 0:
            continue
        idx = np.where(fine_labels == f)[0]
        if idx.size == 0:
            continue
        feats = pair_features[idx]

        if idx.size <= large_cluster_threshold:
            sel_local = kernel_herding(feats, b, candidate_idx=None)
        else:
            # sparse approximation for large sub-clusters
            nbr = knn_graph(feats, k=k_nn)
            cand = np.unique(nbr.reshape(-1))
            sel_local = kernel_herding(feats, b, candidate_idx=cand)

        selected_all.extend(idx[sel_local].tolist())

    return np.array(sorted(set(selected_all)), dtype=np.int64)
