#!/usr/bin/env python
"""
scripts/run_ablation.py
=======================
Stage A + Stage B runner for the 45K-scale ablation that decides the paper's
main story (plan §6.0, ChatGPT four-tier strategy).

Runs the three decisive ablations at a single budget (default 20%):
    * full           : s_c = |C|^a (1 + b*abar + e*u)
    * relevance_only : eta = 0
    * uncertainty_only: beta = 0
plus with/without per-proxy normalization, and the baseline selectors for
context. Everything is logged to --out.

USAGE (synthetic smoke test, no GPU/CLIP needed):
    python scripts/run_ablation.py --synthetic --n 45000 --out runs/smoke

USAGE (real embeddings cached as z_v.npy / z_t.npy + proxies):
    python scripts/run_ablation.py \
        --zv cache/z_v.npy --zt cache/z_t.npy \
        --proxies cache/Q1.npy cache/Q2.npy \
        --budget-frac 0.2 --out runs/pab45k
"""
from __future__ import annotations

import argparse
import os
import numpy as np

from cawot import diagnostics as D
from cawot.pipeline import run_proposed, run_selectors
from cawot.data import make_synthetic_dataset


def load_inputs(args):
    if args.synthetic:
        ds = make_synthetic_dataset(n=args.n, seed=args.seed)
        return ds["z_v"], ds["z_t"], ds["proxy_emb"], ds
    z_v = np.load(args.zv, mmap_mode="r")
    z_t = np.load(args.zt, mmap_mode="r")
    proxies = [np.load(p) for p in args.proxies]
    return z_v, z_t, proxies, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n", type=int, default=45000)
    ap.add_argument("--zv"); ap.add_argument("--zt")
    ap.add_argument("--proxies", nargs="*", default=[])
    ap.add_argument("--budget-frac", type=float, default=0.2)
    ap.add_argument("--k-coarse", type=int, default=20)
    ap.add_argument("--k-fine", type=int, default=10)
    ap.add_argument("--rff-dim", type=int, default=1024)
    ap.add_argument("--k-nn", type=int, default=32)
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/ablation")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    z_v, z_t, proxies, meta = load_inputs(args)
    N = z_v.shape[0]
    budget = int(round(args.budget_frac * N))
    print(f"[info] N={N}  budget={budget} ({args.budget_frac:.0%})  "
          f"proxies={len(proxies)}")

    common = dict(k_coarse=args.k_coarse, k_fine=args.k_fine,
                  rff_dim=args.rff_dim, k_nn=args.k_nn, lam=args.lam,
                  alpha=args.alpha, beta=args.beta, tau=args.tau, seed=args.seed)

    # ---- the three decisive ablations x normalization ----
    grid = [
        ("full_norm",            dict(eta=1.0, beta=args.beta, normalize=True)),
        ("relevance_only_norm",  dict(eta=0.0, beta=args.beta, normalize=True)),
        ("uncertainty_only_norm",dict(eta=1.0, beta=0.0,       normalize=True)),
        ("full_nonorm",          dict(eta=1.0, beta=args.beta, normalize=False)),
    ]
    ablation_index = {}
    for name, over in grid:
        sub = os.path.join(args.out, name)
        kw = dict(common); kw.update(over)
        res = run_proposed(z_v, z_t, proxies, budget_total=budget,
                           out_dir=sub, **kw)
        rep = res["log"]["allocation_report"]
        ablation_index[name] = {
            "coreset_size": int(res["coreset"].size),
            "budget_vs_size_corr": rep["budget_vs_size_corr"],
            "budget_vs_relevance_corr": rep["budget_vs_relevance_corr"],
            "budget_vs_uncertainty_corr": rep["budget_vs_uncertainty_corr"],
            "runtime_sec": res["log"].get("runtime_sec", {}),
        }
        print(f"[done] {name}: size={res['coreset'].size}  "
              f"corr(size,budget)={rep['budget_vs_size_corr']}")

    # ---- baselines + proposed overlap at the same budget ----
    sel = run_selectors(
        z_v, z_t, proxies, budget_total=budget,
        methods=("random", "clipscore", "kcenter", "sw_cawot", "proposed"),
        seed=args.seed, out_dir=os.path.join(args.out, "selectors"),
        proposed_kwargs=dict(common, eta=1.0, beta=args.beta, normalize=True))

    # ---- Stage-A noisy-group check (synthetic only) ----
    extra = {}
    if meta is not None:
        # does uncertainty concentrate on the injected noisy groups?
        full_npz = np.load(os.path.join(args.out, "full_norm",
                                        "allocation_tables.npz"))
        u = np.nan_to_num(full_npz["u"], nan=0.0)
        top_u = set(np.argsort(-u)[:meta["n_groups"] // 4].tolist())
        # NOTE: cluster ids != group ids; this is a loose sanity signal only.
        extra["synthetic_check"] = {
            "noisy_groups": meta["noisy_groups"],
            "relevant_groups": meta["relevant_groups"],
            "top_uncertainty_clusters": sorted(top_u),
            "note": "cluster ids are not group ids; inspect manually for overlap",
        }

    summary = {
        "N": int(N), "budget": int(budget), "budget_frac": args.budget_frac,
        "ablation_index": ablation_index,
        "selector_overlap": sel["summary"]["pairwise_overlap"],
        "coreset_sizes": sel["summary"]["coreset_sizes"],
        **extra,
    }
    D.save_json(summary, os.path.join(args.out, "ABLATION_SUMMARY.json"))
    print(f"[ok] wrote {os.path.join(args.out, 'ABLATION_SUMMARY.json')}")
    print("[next] run Stage C (fine-tune + R@k/mAP) via your trainer on the "
          "saved coreset_*.npy, comparing relevance_only vs full (the eta=0 "
          "life-or-death test).")


if __name__ == "__main__":
    main()
