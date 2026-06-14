"""
cawot/diagnostics.py
====================
Stage-A sanity diagnostics and retrieval metrics (plan §7.2, §5 checklist).

Everything a reviewer (or you, three weeks later) needs to understand WHY a
selector won or lost. Per the plan's logging checklist, we expose:
    d_{c,r}, d~_{c,r}, a_{c,r}, abar_c, u_c, s_c, b_c,
    cluster sizes, top clusters by abar / by u, selected ids, selector overlap,
    runtime + peak memory per step.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager

import numpy as np


# --------------------------------------------------------------------------- #
#  Timing / memory                                                              #
# --------------------------------------------------------------------------- #
@contextmanager
def timed(name: str, log: dict):
    """Context manager recording wall-clock seconds into ``log[name]``."""
    t0 = time.perf_counter()
    yield
    log.setdefault("runtime_sec", {})[name] = round(time.perf_counter() - t0, 4)


def peak_memory_mb() -> float:
    """Best-effort peak RSS in MB (resource module; 0.0 if unavailable)."""
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports kB, macOS reports bytes
        import sys
        return round(ru / (1024 if sys.platform.startswith("linux") else 1024 * 1024), 2)
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- #
#  Stage-A: allocation sanity                                                   #
# --------------------------------------------------------------------------- #
def allocation_report(alloc, top: int = 5) -> dict:
    """Summarize an AllocationResult for sanity checking before any fine-tuning."""
    sizes = alloc.sizes
    abar = np.nan_to_num(alloc.abar, nan=0.0)
    u = np.nan_to_num(alloc.u, nan=0.0)

    def topk(vec):
        order = np.argsort(-vec)[:top]
        return [(int(c), float(vec[c]), int(sizes[c])) for c in order]

    nonempty = int((sizes > 0).sum())
    rep = {
        "config": alloc.config,
        "n_clusters_nonempty": nonempty,
        "budget_total": int(alloc.coarse_budget.sum()),
        "coarse_budget": alloc.coarse_budget.tolist(),
        "cluster_sizes": sizes.tolist(),
        "top_clusters_by_mean_relevance": topk(abar),
        "top_clusters_by_uncertainty": topk(u),
        "budget_vs_size_corr": _safe_corr(alloc.coarse_budget, sizes),
        "budget_vs_relevance_corr": _safe_corr(alloc.coarse_budget, abar),
        "budget_vs_uncertainty_corr": _safe_corr(alloc.coarse_budget, u),
    }
    return rep


def _safe_corr(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return round(float(np.corrcoef(a, b)[0, 1]), 4)


def normalization_rank_shift(d_raw: np.ndarray, d_tilde: np.ndarray) -> dict:
    """How much does per-proxy normalization change cluster rankings?

    Reports mean Spearman-like rank correlation per proxy between raw and
    normalized scores. Low correlation => normalization matters a lot.
    """
    from scipy.stats import spearmanr  # type: ignore
    R = d_raw.shape[1]
    corrs = []
    for r in range(R):
        a, b = d_raw[:, r], d_tilde[:, r]
        valid = ~(np.isnan(a) | np.isnan(b))
        if valid.sum() < 3:
            continue
        rho, _ = spearmanr(a[valid], b[valid])
        if not np.isnan(rho):
            corrs.append(float(rho))
    return {"per_proxy_rank_corr_raw_vs_norm": corrs,
            "mean": round(float(np.mean(corrs)), 4) if corrs else None}


def selector_overlap(sel_a: np.ndarray, sel_b: np.ndarray) -> float:
    """Jaccard overlap between two selected index sets."""
    A, B = set(sel_a.tolist()), set(sel_b.tolist())
    if not A and not B:
        return 1.0
    return round(len(A & B) / max(1, len(A | B)), 4)


# --------------------------------------------------------------------------- #
#  Retrieval metrics (text -> image)                                            #
# --------------------------------------------------------------------------- #
def retrieval_metrics(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
    gt_index: np.ndarray,
    ks=(1, 5, 10),
) -> dict:
    """Standard text->image retrieval R@k and mAP (single positive per query).

    query_emb : (Q, d) query (text) embeddings, normalized.
    gallery_emb: (G, d) gallery (image) embeddings, normalized.
    gt_index  : (Q,) index into gallery of the correct match for each query.
    """
    sims = query_emb @ gallery_emb.T          # (Q, G)
    order = np.argsort(-sims, axis=1)          # ranked gallery ids per query
    Q = query_emb.shape[0]

    ranks = np.zeros(Q, dtype=np.int64)
    for q in range(Q):
        # position of the ground-truth item in the ranking (0-based)
        pos = np.where(order[q] == gt_index[q])[0]
        ranks[q] = pos[0] if pos.size else gallery_emb.shape[0]

    out = {}
    for k in ks:
        out[f"R@{k}"] = round(float(np.mean(ranks < k)), 4)
    # mAP with a single relevant item == mean reciprocal rank
    out["mAP"] = round(float(np.mean(1.0 / (ranks + 1.0))), 4)
    out["median_rank"] = int(np.median(ranks))
    return out


def bootstrap_ci(per_query_hits: np.ndarray, n_boot: int = 1000,
                 alpha: float = 0.05, rng: np.random.Generator | None = None):
    """Percentile bootstrap CI for a per-query mean (e.g. R@1 indicator)."""
    rng = np.random.default_rng() if rng is None else rng
    n = per_query_hits.shape[0]
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = per_query_hits[idx].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return round(float(lo), 4), round(float(hi), 4)


def save_json(obj: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
