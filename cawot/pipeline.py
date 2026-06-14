"""
cawot/pipeline.py
=================
End-to-end driver wiring kernels -> clustering -> scoring/allocation -> selection,
plus baselines and full diagnostic logging.

This module is deliberately framework-light: it works on numpy embedding arrays
(z_v, z_t) and a list of proxy text-embedding arrays. The (optional) CLIP
fine-tuning / retrieval evaluation (Stage C) is left as a thin hook
(`evaluate_coreset`) because it depends on your training code in the repo; the
selection + Stage-A/B logic here is fully runnable and testable on its own.

Stages (plan, ChatGPT four-tier strategy):
  A  build_features + score + allocate + allocation_report      (no fine-tune)
  B  run_selectors  -> coreset indices per method + overlaps     (no fine-tune)
  C  evaluate_coreset(...) hook for fine-tune + R@k/mAP           (your trainer)
  D  scale up by re-running with a larger embedding cache
"""
from __future__ import annotations

import os
import numpy as np

from .kernels import RFF, PairFeatureMap, median_heuristic_bandwidth
from .clustering import two_level_clustering, cluster_sizes
from .scoring import allocate, AllocationResult
from .selection import select_within_clusters
from . import baselines as B
from . import diagnostics as D


# --------------------------------------------------------------------------- #
#  Feature construction                                                         #
# --------------------------------------------------------------------------- #
def build_features(
    z_v: np.ndarray,
    z_t: np.ndarray,
    *,
    lam: float = 0.7,
    rff_dim: int = 1024,
    seed: int = 0,
    log: dict | None = None,
):
    """Build RFF maps and pair features. Returns (rff_v, rff_t, pair_map, pair_feats)."""
    log = {} if log is None else log
    rng = np.random.default_rng(seed)

    sig_v = median_heuristic_bandwidth(np.asarray(z_v[: min(4000, len(z_v))]), rng=rng)
    sig_t = median_heuristic_bandwidth(np.asarray(z_t[: min(4000, len(z_t))]), rng=rng)
    log.setdefault("bandwidth", {})["sigma_v"] = sig_v
    log["bandwidth"]["sigma_t"] = sig_t

    d = z_v.shape[1]
    rff_v = RFF(d, rff_dim, sig_v, rng=np.random.default_rng(seed + 1))
    rff_t = RFF(d, rff_dim, sig_t, rng=np.random.default_rng(seed + 2))
    pair_map = PairFeatureMap(rff_v, rff_t, lam=lam)

    with D.timed("build_pair_features", log):
        pair_feats = pair_map.transform(np.asarray(z_v), np.asarray(z_t))
    return rff_v, rff_t, pair_map, pair_feats


# --------------------------------------------------------------------------- #
#  Stage A + the proposed selector                                              #
# --------------------------------------------------------------------------- #
def run_proposed(
    z_v: np.ndarray,
    z_t: np.ndarray,
    proxy_text_emb: list[np.ndarray],
    *,
    budget_total: int,
    k_coarse: int = 20,
    k_fine: int = 10,
    lam: float = 0.7,
    rff_dim: int = 1024,
    alpha: float = 0.5,
    beta: float = 1.0,
    eta: float = 1.0,
    tau: float = 1.0,
    normalize: bool = True,
    k_nn: int = 32,
    seed: int = 0,
    out_dir: str | None = None,
) -> dict:
    """Run the proposed pipeline end-to-end (selection only) with full logging."""
    log: dict = {"method": "proposed", "seed": seed,
                 "params": dict(k_coarse=k_coarse, k_fine=k_fine, lam=lam,
                                rff_dim=rff_dim, alpha=alpha, beta=beta, eta=eta,
                                tau=tau, normalize=normalize, k_nn=k_nn,
                                budget_total=int(budget_total))}

    rff_v, rff_t, pair_map, pair_feats = build_features(
        z_v, z_t, lam=lam, rff_dim=rff_dim, seed=seed, log=log)

    with D.timed("clustering", log):
        coarse, fine = two_level_clustering(
            pair_feats, k_coarse=k_coarse, k_fine=k_fine, seed=seed)

    with D.timed("score_allocate", log):
        alloc: AllocationResult = allocate(
            np.asarray(z_t), coarse, fine, proxy_text_emb, rff_t,
            k_coarse, k_fine, budget_total,
            alpha=alpha, beta=beta, eta=eta, tau=tau, normalize=normalize)

    # Stage-A diagnostics
    log["allocation_report"] = D.allocation_report(alloc)
    log["normalization_rank_shift"] = _safe_norm_shift(alloc)

    with D.timed("within_cluster_selection", log):
        coreset = select_within_clusters(
            pair_feats, fine, alloc.fine_budget, k_nn=k_nn)

    log["coreset_size"] = int(coreset.size)
    log["peak_memory_mb"] = D.peak_memory_mb()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "coreset_indices.npy"), coreset)
        # dump all per-cluster quantities the plan's checklist requires
        np.savez(
            os.path.join(out_dir, "allocation_tables.npz"),
            d_raw=alloc.d_raw, d_tilde=alloc.d_tilde, a=alloc.a,
            abar=alloc.abar, u=alloc.u, s=alloc.s,
            coarse_budget=alloc.coarse_budget, fine_budget=alloc.fine_budget,
            sizes=alloc.sizes, coarse_labels=coarse, fine_labels=fine)
        D.save_json(log, os.path.join(out_dir, "log_proposed.json"))

    return {"coreset": coreset, "alloc": alloc, "coarse": coarse,
            "fine": fine, "pair_feats": pair_feats, "rff_t": rff_t, "log": log}


def _safe_norm_shift(alloc):
    try:
        return D.normalization_rank_shift(alloc.d_raw, alloc.d_tilde)
    except Exception as e:                       # scipy missing, etc.
        return {"error": str(e)}


# --------------------------------------------------------------------------- #
#  Stage B -- baselines + proposed, with overlaps                               #
# --------------------------------------------------------------------------- #
def run_selectors(
    z_v: np.ndarray,
    z_t: np.ndarray,
    proxy_text_emb: list[np.ndarray],
    *,
    budget_total: int,
    methods=("random", "clipscore", "kcenter", "sw_cawot", "proposed"),
    seed: int = 0,
    out_dir: str | None = None,
    proposed_kwargs: dict | None = None,
) -> dict:
    """Run a set of selectors at one budget; return coresets + pairwise overlap."""
    rng = np.random.default_rng(seed)
    n = z_v.shape[0]
    proposed_kwargs = proposed_kwargs or {}

    # shared structures for baselines that need clustering / features
    _, _, _, pair_feats = build_features(
        z_v, z_t, lam=proposed_kwargs.get("lam", 0.7),
        rff_dim=proposed_kwargs.get("rff_dim", 1024), seed=seed)
    k_coarse = proposed_kwargs.get("k_coarse", 20)
    k_fine = proposed_kwargs.get("k_fine", 10)
    coarse, _fine = two_level_clustering(pair_feats, k_coarse, k_fine, seed)
    # low-dim view for SW baseline: concat raw embeddings
    sw_view = np.concatenate([np.asarray(z_v), np.asarray(z_t)], axis=1).astype(np.float32)
    sw_proxy = np.concatenate(
        [np.concatenate([Q, Q], axis=1) for Q in proxy_text_emb], axis=0
    ).astype(np.float32)  # crude lift into the same concat space

    coresets: dict[str, np.ndarray] = {}
    for m in methods:
        if m == "random":
            coresets[m] = B.select_random(budget_total, n, np.random.default_rng(seed + 7))
        elif m == "clipscore":
            coresets[m] = B.select_clipscore(budget_total, np.asarray(z_t),
                                             proxy_text_emb, rng)
        elif m == "kcenter":
            coresets[m] = B.select_kcenter(budget_total, pair_feats,
                                           np.random.default_rng(seed + 9))
        elif m == "sw_cawot":
            coresets[m] = B.select_sw_cawot(
                budget_total, sw_view, coarse, sw_proxy, k_coarse,
                np.random.default_rng(seed + 11))
        elif m == "proposed":
            pk = {k: v for k, v in proposed_kwargs.items() if k != "seed"}
            res = run_proposed(z_v, z_t, proxy_text_emb,
                               budget_total=budget_total, seed=seed,
                               out_dir=out_dir, **pk)
            coresets[m] = res["coreset"]
        else:
            raise ValueError(f"unknown method {m}")

    # pairwise overlap matrix
    names = list(coresets)
    overlap = {a: {b: D.selector_overlap(coresets[a], coresets[b])
                   for b in names} for a in names}

    summary = {
        "budget_total": int(budget_total),
        "coreset_sizes": {m: int(v.size) for m, v in coresets.items()},
        "pairwise_overlap": overlap,
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for m, v in coresets.items():
            np.save(os.path.join(out_dir, f"coreset_{m}.npy"), v)
        D.save_json(summary, os.path.join(out_dir, "selector_summary.json"))
    return {"coresets": coresets, "summary": summary}


# --------------------------------------------------------------------------- #
#  Stage C -- evaluation hook (depends on your trainer)                         #
# --------------------------------------------------------------------------- #
def evaluate_coreset(coreset_indices, train_fn, eval_fn) -> dict:
    """Thin hook: fine-tune on the coreset then evaluate retrieval.

    Parameters
    ----------
    coreset_indices : indices into the training pool.
    train_fn(indices) -> model   : your CLIP fine-tuning routine.
    eval_fn(model) -> dict        : returns {'R@1':..,'R@5':..,'R@10':..,'mAP':..}

    Kept abstract so it plugs into the existing CAWOT training code rather than
    imposing a new trainer.
    """
    model = train_fn(coreset_indices)
    return eval_fn(model)
