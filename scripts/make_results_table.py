#!/usr/bin/env python
"""
scripts/make_results_table.py
=============================
Turn raw Stage-C metric JSONs into a paper-ready, reviewer-defensible results
table: methods x budgets, mean +/- std over seeds, Wilson 95% CI for R@1,
t2i / i2t reported separately, and an explicit list of MISSING cells so coverage
is never silently incomplete.

This is the "proof" apparatus: it does not run experiments, it standardizes and
aggregates their outputs into something publishable.

Input: a directory of Stage-C metric JSONs in the schema your eval emits, e.g.
    {
      "overall":   {"n_pairs":1978,"t2i_R@1":..,"i2t_R@1":..,"mean_R@1":..,
                    "t2i_R@5":..,"i2t_R@5":..,"t2i_R@10":..,"i2t_R@10":..,"mAP":..},
      "wentwrong": {...}, "goal": {...}, "full": {...}
    }

Filename convention (tolerant): tokens separated by '_' or '__'. Recognized:
    method = everything not matched as a budget/seed token
    budget = token like  b20  b5  b0.05  budget20
    seed   = token like  s0  s42  seed3
e.g.  proposed_b5_s0_metrics.json,  random__b5__s1.json,  full_data_b5_s2_stage_c_metrics.json

USAGE
    python scripts/make_results_table.py --dir runs/stage_c --metric t2i_R@1 \
        --subgroup overall --out runs/stage_c/results
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

# make stdout safe for non-ASCII (em-dash, ±, emoji) on Windows consoles
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# preferred display order; unknown methods appended
METHOD_ORDER = ["random", "clipscore", "semdedup", "kcenter", "sw_cawot",
                "proposed", "full_data", "fulldata", "full"]

_BUDGET_RE = re.compile(r"^(?:b|budget)?(\d+(?:\.\d+)?)$", re.I)
_SEED_RE = re.compile(r"^(?:s|seed)(\d+)$", re.I)
_STRIP = {"metrics", "stage", "c", "stagec", "eval", "norm"}


def parse_name(stem: str):
    """Return (method, budget|None, seed|None) from a filename stem."""
    toks = re.split(r"__|_", stem)
    budget = seed = None
    method_toks = []
    for t in toks:
        if not t or t.lower() in _STRIP:
            continue
        ms = _SEED_RE.match(t)
        if ms:
            seed = int(ms.group(1)); continue
        mb = _BUDGET_RE.match(t)
        if mb and any(ch.isdigit() for ch in t) and t.lower() not in METHOD_ORDER:
            budget = mb.group(1); continue
        method_toks.append(t)
    method = "_".join(method_toks) if method_toks else stem
    return method, budget, seed


def wilson_ci(p_pct: float, n: int, z: float = 1.96):
    """Wilson 95% CI for a proportion given R@1 in percent and n trials."""
    if not n or n <= 0:
        return (None, None)
    p = max(0.0, min(1.0, p_pct / 100.0))
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round(100 * (center - half), 2), round(100 * (center + half), 2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="dir of Stage-C metric JSONs")
    ap.add_argument("--glob", default="*.json")
    ap.add_argument("--metric", default="t2i_R@1",
                    help="t2i_R@1 (default, the deployment direction) / i2t_R@1 / mean_R@1 / mAP ...")
    ap.add_argument("--subgroup", default="overall",
                    help="overall / wentwrong (anomaly) / goal / full")
    ap.add_argument("--out", default="results", help="output path prefix (.md + .csv)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, args.glob)))
    if not files:
        raise SystemExit(f"no JSON found in {args.dir}/{args.glob}")

    # cell[(method, budget)] = list of (value, n_pairs, seed, file)
    cell = defaultdict(list)
    budgets, methods = set(), set()
    rows_csv = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        method, budget, seed = parse_name(stem)
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            print(f"[skip] {f}: {e}"); continue
        sub = d.get(args.subgroup)
        if not isinstance(sub, dict) or args.metric not in sub:
            print(f"[skip] {f}: no {args.subgroup}/{args.metric}"); continue
        val = float(sub[args.metric]); n = int(sub.get("n_pairs", 0))
        cell[(method, budget)].append((val, n, seed))
        budgets.add(budget); methods.add(method)
        rows_csv.append(dict(method=method, budget=budget, seed=seed,
                             subgroup=args.subgroup, metric=args.metric,
                             value=val, n_pairs=n, file=os.path.basename(f)))

    def bkey(b):
        try: return float(b)
        except Exception: return 1e9
    budgets = sorted(budgets, key=bkey)
    methods = sorted(methods, key=lambda m: (METHOD_ORDER.index(m) if m in METHOD_ORDER else 99, m))

    # ---- markdown table ----
    md = [f"# Results — {args.metric} ({args.subgroup})",
          "",
          f"Each cell: mean ± std over seeds (n_seeds). Single-seed cells show the "
          f"Wilson 95% CI for R@1. `—` = run missing.",
          "",
          "| method | " + " | ".join(f"budget {b}%" for b in budgets) + " |",
          "|" + "---|" * (len(budgets) + 1)]
    missing = []
    for m in methods:
        cells = []
        for b in budgets:
            vals = cell.get((m, b), [])
            if not vals:
                cells.append("—"); missing.append(f"{m} @ {b}%"); continue
            xs = [v for v, _, _ in vals]
            mean = sum(xs) / len(xs)
            if len(xs) == 1:
                _, n, _ = vals[0]
                lo, hi = wilson_ci(xs[0], n)
                ci = f" [{lo},{hi}]" if lo is not None else ""
                cells.append(f"{mean:.2f}{ci} (1)")
            else:
                std = (sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
                cells.append(f"{mean:.2f}±{std:.2f} ({len(xs)})")
        flag = " **(method)**" if m == "proposed" else ""
        md.append(f"| {m}{flag} | " + " | ".join(cells) + " |")

    md += ["", "## Coverage / what's missing", ""]
    if missing:
        md.append("⚠️ Missing runs (fill these before claiming a result):")
        md += [f"- {x}" for x in missing]
    else:
        md.append("✅ All method×budget cells present.")
    md += ["",
           "## Notes for a defensible claim",
           "- Report **t2i** (text→image, the deployment direction) separately from i2t.",
           "- Compare `proposed` against **every** baseline *and* **full_data** at each budget.",
           "- Wilson CI needs only (R@1, n_pairs); for paired significance, dump per-query "
           "hit vectors from eval and add a paired permutation test.",
           "- The coreset story is strongest at **low budget** (2–5%) on the **real** test set."]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".md", "w", encoding="utf-8") as fo:
        fo.write("\n".join(md) + "\n")
    with open(args.out + ".csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=["method", "budget", "seed", "subgroup",
                                           "metric", "value", "n_pairs", "file"])
        w.writeheader(); w.writerows(rows_csv)

    print("\n".join(md))
    print(f"\n[ok] wrote {args.out}.md and {args.out}.csv  "
          f"({len(rows_csv)} runs, {len(methods)} methods, {len(budgets)} budgets)")
    if missing:
        print(f"[warn] {len(missing)} missing cells — see table.")


if __name__ == "__main__":
    main()
