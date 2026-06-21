# CAWOT-CM

Query-aware **coreset selection** for **cross-modal (image–text) person anomaly
retrieval** on the PAB benchmark: pick a small, useful subset of a large
synthetic training pool, fine-tune CLIP on it, and keep retrieval quality high at
a fraction of the data budget.

The repo holds **two parallel versions** (the old one is kept intact as a fallback):

| Version | Code | Idea | Status |
|---|---|---|---|
| **V0–V2** (poster, sliced-Wasserstein) | [`src/`](src/), [`scripts/run_sweep.py`](scripts/run_sweep.py), [`result/`](result/) | hierarchical k-means → **SW gap** to a query proxy → budget allocation → greedy facility-location. | stable; docs: [docs/README_poster_v0_v2.md](docs/README_poster_v0_v2.md) |
| **V3** (current, kernel/MMD) | [`cawot/`](cawot/), [`scripts/`](scripts/), [`configs/`](configs/) | **text-side MMD** relevance to a finite family of proxy queries → mean-relevance allocation → **kernel herding** within clusters. | active |

---

## Method (V3) in one paragraph

We never build a fused image–text pair vector. Cluster **relevance** to the query
is measured on the **text axis** — MMD between a cluster's caption distribution
and each proxy-query distribution — because queries are text. MMD is estimated via
**random Fourier features** (not dense kernel matrices), so scoring stays
near-linear. Budget is allocated per coarse cluster as

```
s_c = |C_c|^alpha * (1 + beta * abar_c)
abar_c = mean over proxy families r of   a_{c,r} = exp(-tau * softplus(d~_{c,r}))
d_{c,r} = MMD^2_text( captions of cluster c ,  proxy family Q_r )   # via RFF, per-proxy z/IQR normalized
```

Within each fine cluster, **approximate kernel herding** selects representative
pairs. **The method is frozen at mean-relevance (`eta = 0`).** A proxy-disagreement
bonus `eta * u_c` was tried and **dropped** — it is empirically inert (see
[Findings](#current-findings)); `u_c` is retained only as a negative-result ablation.

---

## Repository layout (V3)

```
cawot/                 V3 package
  kernels.py           RBF-on-sphere, RFF, MMD via RFF, additive pair feature map
  clustering.py        two-level approximate kernel-aware clustering (mini-batch kmeans)
  scoring.py           text-side MMD scores, per-proxy normalization, abar/u,
                       capped largest-remainder allocation        <- the heart
  selection.py         kNN sparsification + approximate kernel herding
  baselines.py         random, clipscore, kcenter, sw_cawot (poster, internal ablation)
  diagnostics.py       Stage-A sanity, retrieval R@k/mAP, Wilson/bootstrap CI, logging
  data.py              CLIP embedding + cache, proxy-family construction, synthetic data
  pipeline.py          end-to-end driver (Stage A/B) + Stage-C hook
  qproxy_v3.py         LLM (DeepSeek) generator for proxy query families
scripts/
  run_ablation.py            Stage A/B: select coresets (baselines + proposed) + diagnostics
  embed_qproxy_families.py   encode LLM proxy-family JSON -> CLIP .npy
  make_results_table.py      aggregate Stage-C metrics -> paper-ready table (CI, missing cells)
configs/first_run_45k.yaml
docs/
  EXPERIMENTS.md             the reviewer-grade protocol + pre-registered win criteria
  PROJECT_STATE.md           living state map (where we are, what's left)
  PROXY_DATASET_GUIDE.md     how to create proxy query families (counts, DeepSeek prompts)
  README_poster_v0_v2.md     the original V0–V2 poster README
outputs/qproxy_v3_hybrid6_20k/   the 6 generated proxy families + manifest
```

---

## Install

```bash
pip install numpy scikit-learn scipy          # core (selection + smoke test)
# for real data / Stage C also: torch, open_clip_torch (or openai/CLIP), faiss-* (optional)
```

---

## Quickstart — the pipeline end to end

### 1. Smoke test (no GPU / no CLIP / no data, seconds)
```bash
PYTHONPATH=. python scripts/run_ablation.py --synthetic --n 45000 --out runs/smoke
```
Generates structured fake embeddings and runs the ablations + baseline selectors,
writing all per-cluster quantities (`d, d~, abar, u, s, b`), selector overlaps,
runtimes to `runs/smoke/`. Use it to confirm the wiring before spending GPU.

### 2. Encode the proxy query families (`qproxy_v3` JSON → `.npy`)
```bash
# (a) parse-check, no GPU:
python scripts/embed_qproxy_families.py --proxy-dir outputs/qproxy_v3_hybrid6_20k \
    --out-dir cache --dry-run
# (b) encode with the SAME CLIP model that produced z_v.npy/z_t.npy:
python scripts/embed_qproxy_families.py --proxy-dir outputs/qproxy_v3_hybrid6_20k \
    --out-dir cache --backend open_clip --model ViT-B-32 --pretrained openai
```
> ⚠️ **Encoder must match.** Proxies and the pool (`z_v/z_t.npy`) must come from the
> *same* CLIP weights. Both common backbones are 512-d, so a mismatch will **not**
> error — it silently produces garbage relevance.

### 3. Select coresets (baselines + proposed) at a budget
```bash
PYTHONPATH=. python scripts/run_ablation.py \
    --zv cache/z_v.npy --zt cache/z_t.npy \
    --proxies cache/Q1_templates.npy cache/Q2_deepseek_paraphrases.npy \
              cache/Q3_deepseek_anomaly.npy cache/Q4_deepseek_hard_cases.npy \
              cache/Q5_deepseek_normal.npy cache/Q6_deepseek_appearance.npy \
    --budget-frac 0.05 --out runs/pab45k_b5
```
Saves `runs/.../selectors/coreset_<method>.npy` for random / clipscore / kcenter /
sw_cawot / **proposed**, plus full diagnostics. (`run_ablation` accepts any number
of `--proxies`; family weights default to uniform.)

### 4. Stage C — fine-tune + evaluate on the real test
Plug your CLIP fine-tuning routine into the hook and evaluate each coreset on the
**real PAB test (1,978 pairs)**:
```python
from cawot.pipeline import evaluate_coreset
metrics = evaluate_coreset(coreset_indices, train_fn=my_finetune, eval_fn=my_eval)
```
Write one metrics JSON per run, named `<method>_b<budget>_s<seed>_metrics.json`
(schema in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) §5).

### 5. Build the results table
```bash
python scripts/make_results_table.py --dir runs/stage_c --metric t2i_R@1 --out runs/stage_c/results
```
→ `results.md` + `results.csv`: methods × budgets, mean ± std over seeds, **Wilson
95% CI** for R@1, and an explicit list of any **missing cells** (so no claim is
made on incomplete data).

---

## Proving the method (read before claiming a result)

Follow **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)** — it fixes the protocol *and*
the win criteria up front:

- **Regime:** low budget (**2–5%**) on the **real PAB test (1,978)**, not 20% on
  synthetic val (where every method ties at ~88%). Report **t2i** (text→image, the
  deployment direction) separately from i2t; also the **anomaly** subgroup.
- **Baselines:** random, clipscore, kcenter, sw_cawot (poster), **proposed**, and
  **full-data** (the key reference). Fix backbone + fine-tune pipeline; vary only selection.
- **Statistics:** ≥3 seeds, mean ± std, Wilson 95% CI for R@1; dump per-query hits to
  enable paired permutation + Holm.
- **Pre-registered win criteria:** `proposed` ≥ every baseline at 5% / real-test /
  t2i_R@1 with non-overlapping CI; **or** `proposed ≈ full-data` at low budget
  (efficiency claim). Either is publishable. If both fail, see EXPERIMENTS.md §7
  (change the method only then — start by reducing size dominance via `--alpha`).

---

## Current findings

- ✅ **Pipeline runs on real PAB** (45K pool @ 20%, `result/cawot_v3_2.zip`,
  `outputs/pab45k_6proxy/`): clustering ~20s, scoring ~0.5s, selection ~4s.
- 🔴 **`eta` (proxy-disagreement) is inert → dropped.** On the real 6-family run,
  `u` (≈0.0006) is ~**1000×** smaller than `abar` (0.17–0.70); `full` vs
  `relevance-only` coresets are **99.98% identical** (1 of 9000) even with 6 diverse
  families. The families agree on cluster relevance in CLIP space. Method = mean-relevance.
- 🟠 **First Stage-C R@1 (≈5% budget, 2250 train) is a NULL result** —
  `result/cawot_v3_3.zip`, `outputs/stage_c/`, single seed, eval on the **synthetic
  val split (5000)**. On `t2i_R@1` (the deployment direction) **no method beats
  random**; everything is within overlapping Wilson CIs (~±1.1):

  | kcenter | random | proposed | sw_cawot | clipscore |
  |---|---|---|---|---|
  | 79.52 | 79.02 | 78.94 | 78.94 | 78.50 |

  `proposed` is nominally top on *mean*/`i2t` (80.32 / 81.70) but that is within
  noise and **not** on `t2i` — do **not** cherry-pick it as a win.
- ⚠️ **Not yet the decisive test.** The eval above is `val_split` (synthetic,
  in-distribution, easy — every method ties at ~88% at higher budget), **single
  seed**, and **no `full_data`**. The pre-registered decisive run (see
  [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)) is still pending: **PAB real test
  (1,978) + ≥3 seeds + `full_data`, at 2% and 5%**. If `proposed` ties random there
  too, pivot the contribution (efficiency-vs-full-data / theory / other benchmark)
  per EXPERIMENTS.md §6.

---

## Honesty notes

- **Theorem 1** (uniform MMD-score concentration) is a *statistical guarantee*, not
  the novelty — the novelty is the *composition*. With RFF the score error has three
  parts: cluster sampling, proxy sampling, and an RFF term `eps_RFF(D)` kept general
  (not hard-bounded) pending a full proof. **A statistician should vet it.**
- **`u_c`** measures sensitivity to proxy *choice within the constructed family*, not
  true risk under the real deployment distribution.
- Complexity is for the *actual* algorithm: kNN `O(m log m)` + herding `O(b·k)` per
  sub-cluster. We do **not** claim Compress++'s `O(m log³ m)` (we use kernel
  *herding*, not kernel *thinning*).
- **Proxy naming:** `Q3_deepseek_anomaly` (a normal anomaly-behavior family, fine for
  the main method) is **not** the same as a `validation-failure` proxy (which would be
  ablation-only, never from test data). The six `qproxy_v3` families are all
  training-side and legitimate as main families.
- Report on the **real PAB test (1,978)**; the synthetic val split is in-distribution
  and easy — not the number to compare against SOTA.

---

## Docs

- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) — protocol + win criteria
- [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md) — full state map / roadmap
- [docs/PROXY_DATASET_GUIDE.md](docs/PROXY_DATASET_GUIDE.md) — creating proxy families
- [docs/README_poster_v0_v2.md](docs/README_poster_v0_v2.md) — the V0–V2 poster method

Research code for the CAWOT line of work (Sim2Real cross-modal anomaly retrieval,
PAB benchmark). V0–V2 was presented at the SNAB/VIASM workshop; V3 is the extended
kernel/MMD method under development.
