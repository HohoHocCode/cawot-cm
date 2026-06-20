# CAWOT-CM

Coreset selection for **cross-modal (image–text) person anomaly retrieval** on
the PAB benchmark: pick a small, useful subset of a large synthetic training
pool, then fine-tune CLIP on it and keep retrieval quality high.

This repo now holds **two parallel versions**. The old one is untouched so it
stays runnable as a fallback; the new one is the method to test going forward.

| Version | Where | Idea | Status |
|---|---|---|---|
| **V0–V2 (poster, sliced-Wasserstein)** | [`src/`](src/), [`scripts/run_sweep.py`](scripts/run_sweep.py), [`result/`](result/) | Hierarchical k-means → **SW gap** to a query proxy → budget allocation → greedy facility-location selection. | **Stable.** Poster results in `result/`. Docs: [docs/README_poster_v0_v2.md](docs/README_poster_v0_v2.md). |
| **V3 (new: finite-proxy kernel / MMD)** | [`cawot/`](cawot/), [`scripts/run_ablation.py`](scripts/run_ablation.py), [`configs/first_run_45k.yaml`](configs/first_run_45k.yaml) | Text-side **MMD** relevance to a *finite family* of proxy queries, RFF for scale, mean-relevance + optional disagreement allocation, **kernel herding** within clusters. | **New — run this tonight.** |

> Nothing from V0–V2 was deleted. The poster README is preserved verbatim at
> [docs/README_poster_v0_v2.md](docs/README_poster_v0_v2.md). If V3 fails, V0–V2
> still runs exactly as before.

---

## V3 in one paragraph

The poster version had two structural weaknesses: it scored a cluster's
*distance* to a single query proxy with sliced-Wasserstein on a fused image+text
vector (a domain mismatch — queries are text), and its final greedy
facility-location step is `O(m³)` and does not scale to 1M. **V3 fixes both.**
We never build a fused pair vector. Cluster **relevance** to the query is measured
on the **text axis** — MMD between a cluster's caption distribution and each
*proxy-query* distribution — because queries are text. Budget is allocated by

```
s_c = |C_c|^alpha * (1 + beta * mean_relevance_c + eta * disagreement_c)
```

where **mean relevance is the backbone** and **proxy-disagreement is an optional
uncertainty bonus** (`eta`). Within each cluster, **approximate kernel herding**
on a pair-kernel feature map picks representative pairs. MMD is estimated via
**random Fourier features** (not dense kernel matrices), so scoring stays
near-linear.

---

## V3 layout

```
cawot/
  kernels.py      RBF-on-sphere, RFF, MMD via RFF, additive pair feature map
  clustering.py   two-level approximate kernel-aware clustering (mini-batch kmeans)
  scoring.py      text-side MMD scores, per-proxy normalization, abar/u,
                  capped largest-remainder allocation  <-- the heart
  selection.py    kNN sparsification + approximate kernel herding (steps 6-7)
  baselines.py    random, clipscore, kcenter, sw_cawot (poster, internal ablation)
  diagnostics.py  Stage-A sanity, retrieval R@k/mAP, bootstrap CI, logging
  data.py         CLIP embedding + cache, proxy-family construction, synthetic data
  pipeline.py     end-to-end driver (Stage A/B) + Stage-C hook
configs/first_run_45k.yaml
scripts/run_ablation.py   the 45K-scale ablation that decides the paper's story
```

---

## Quick start

### 1. Smoke test — no GPU / no CLIP / no data (runs in seconds)

```bash
pip install numpy scikit-learn scipy
PYTHONPATH=. python scripts/run_ablation.py --synthetic --n 45000 --out runs/smoke
```

This generates structured fake embeddings and runs the three decisive ablations
(full / relevance-only / uncertainty-only) + with/without normalization + the
baseline selectors, writing **everything** (per-cluster `d, d~, a, abar, u, s, b`,
selector overlaps, runtimes, peak memory) to `runs/smoke/`. Use it to confirm the
pipeline is wired correctly before spending GPU time.

### 2. Real PAB embeddings

Encode the pairs **once** (cached to memmapped `.npy`, so later stages never
re-encode), build the proxy families, then run the ablation:

```python
from cawot.data import embed_pairs, build_proxy_families
import numpy as np

z_v, z_t = embed_pairs(image_paths, captions, cache_dir="cache/")   # CLIP ViT-B/16
proxies, names = build_proxy_families(training_attribute_phrases)   # Q1 + Q2 (text)
np.save("cache/Q1.npy", proxies[0]); np.save("cache/Q2.npy", proxies[1])
```

```bash
PYTHONPATH=. python scripts/run_ablation.py \
    --zv cache/z_v.npy --zt cache/z_t.npy \
    --proxies cache/Q1.npy cache/Q2.npy \
    --budget-frac 0.2 --out runs/pab45k
```

Outputs (`coreset_*.npy`, `runs/`) are git-ignored on purpose.

### 2b. Run with the LLM-generated 6-family proxy (`qproxy_v3`)

If you generated proxy families with `cawot/qproxy_v3.py` (e.g.
`outputs/qproxy_v3_hybrid6_20k/` with `Q1…Q6` JSON files of raw text queries),
first **encode them to `.npy`**, then feed all six to the ablation.

```bash
# (a) sanity-check parsing, no GPU:
python scripts/embed_qproxy_families.py \
    --proxy-dir outputs/qproxy_v3_hybrid6_20k --out-dir cache --dry-run

# (b) encode with the SAME CLIP model that produced z_v.npy/z_t.npy:
python scripts/embed_qproxy_families.py \
    --proxy-dir outputs/qproxy_v3_hybrid6_20k --out-dir cache \
    --backend open_clip --model ViT-B-32 --pretrained openai

# (c) run the 6-family ablation (it prints the exact --proxies line):
PYTHONPATH=. python scripts/run_ablation.py \
    --zv cache/z_v.npy --zt cache/z_t.npy \
    --proxies cache/Q1_templates.npy cache/Q2_deepseek_paraphrases.npy \
              cache/Q3_deepseek_anomaly.npy cache/Q4_deepseek_hard_cases.npy \
              cache/Q5_deepseek_normal.npy cache/Q6_deepseek_appearance.npy \
    --budget-frac 0.2 --out runs/pab45k_6proxy
```

> ⚠️ **Encoder must match.** Proxies and the pool (`z_v.npy`/`z_t.npy`) must be
> encoded by the *same* CLIP weights. Both ViT-B/16 (openai) and ViT-B-32
> (open_clip) are 512-d, so a mismatch will **not** error — it silently produces
> garbage relevance. `run_ablation.py` already accepts any number of `--proxies`;
> family weights `pi` default to uniform (1/R), which runs fine — tune later.

### 3. Stage C — fine-tune + R@1/R@5/R@10/mAP

Selection (Stages A/B) is fully runnable on its own. To turn a selected coreset
into retrieval numbers, plug your existing CLIP fine-tuning routine into the hook:

```python
from cawot.pipeline import evaluate_coreset
metrics = evaluate_coreset(coreset_indices, train_fn=my_finetune, eval_fn=my_retrieval_eval)
```

---

## Current status (read before running)

**Method is frozen: `proposed` = finite-proxy MEAN-RELEVANCE coreset (η = 0).**
`s_c = |C_c|^α · (1 + β·abar_c)`.

**Why η was dropped — confirmed on the real 6-family run** (`result/cawot_v3_2.zip`,
`outputs/pab45k_6proxy/`, PAB 45K @ 20%): the proxy-disagreement bonus `u_c` is
empirically inert — `u` (≈0.0006) is ~**1000× smaller** than `abar` (0.17–0.70),
so `full` vs `relevance-only` coresets are **99.98% identical** (differ by 1 of
9000) **even with 6 diverse families** (anomaly/normal/hard/appearance/…). The
families agree on cluster relevance in CLIP space → disagreement ≈ 0. `u` is kept
only as an honest **negative-result ablation**, not part of the method.

**What is NOT proven yet:** the 6-family run produced *coresets + diagnostics but
no retrieval numbers* (Stage C not run on it). And `proposed` has ~random-level
overlap (0.11) with **every** baseline, so downstream R@1 is genuinely unknown.
The earlier 5K Stage C was on the old 2-family setup — not evidence for V3.

**To actually prove the method**, follow **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**:
low budget (2–5%) on the **real PAB test (1,978)**, all baselines **+ full_data**,
≥3 seeds, t2i/i2t split, Wilson CI. Aggregate with
**`scripts/make_results_table.py`** → a paper-ready table that lists any missing
cells (so no claim is made on incomplete data).

The four-tier order still holds: **A** sanity → **B** selection benchmark →
**C** fine-tune (**at 5% first**, real test) → **D** scale up. **Do not jump to 1M.**

---

## Roadmap — what's left (priority order)

**Now — make the result conclusive (this is what the paper actually claims):**
- [ ] **Add the baselines + full-data** to Stage B/C at the same setup: Random,
      CLIPScore, k-Center, **SW-CAWOT (poster)**, and full-data. The core claim is
      "beats every selector at each budget" + "20% ≈ full-data at 1/5 the cost".
      (η-ablation is secondary to this.)
- [ ] **≥3–5 seeds + confidence intervals**: Wilson interval for R@1 (it is
      Bernoulli per query) + bootstrap CI — `cawot/diagnostics.py` already has
      `bootstrap_ci`. Only then is `full` vs `relevance_only` meaningful.
- [ ] **Report t2i and i2t separately** (not just mean): the direction split above
      is the whole nuance for text-based person search.
- [ ] **Log the run config** (budget %, scale, seed) into the metrics JSON — the
      current files don't say which setup produced them.

**Code ↔ theory consistency (quick):**
- [ ] In `scoring.normalize_per_proxy`, switch the denominator from `IQR + eps`
      to a **floor `max(IQR, γ₀)`** with `γ₀` a real, ablatable hyperparameter
      ({0.01, 0.05, 0.1}). This matches Lemma 2 in the theory and is ~30 min.

**Then — strengthen for submission:**
- [ ] Measure `γ₀, IQR, L_norm, ξ` empirically on PAB → fill the theory appendix.
- [ ] Send the 4-page theory note to a statistician (TrungTin Nguyen / Khai Nguyen)
      to vet **Theorem 1** (the `eps_RFF` term + sub-Gaussian-through-CLIP assumption).
- [ ] Widen the grid: budgets {5, 20, 50}%, datasets (CUHK-PEDES, ICFG-PEDES,
      COCO/Flickr30k for generality, UFineBench for robustness), full stats protocol
      (paired permutation + Holm + partial-conjunction + ASO).
- [ ] Write the paper — **theorem statement first**; R@k lives in experiments only.

**Do NOT rush:** 1M scale, full Wasserstein-DRO, ε-net, renaming the method, or
writing the abstract before the ablation numbers (with CI) are in.

---

## Honesty notes (carried from the design plan)

- **Theorem 1** (uniform MMD-score concentration) is a *statistical guarantee*,
  not the novelty. The novelty is the *composition*. With RFF the score error has
  three parts: cluster sampling, proxy sampling, and an RFF term
  `eps_RFF(D, delta)` kept general (not hard-bounded) pending a full proof.
- **`u_c`** measures sensitivity to proxy *choice within the constructed family*,
  not true risk under the real deployment query distribution.
- Complexity is stated for the *actual* algorithm: kNN `O(m log m)` + herding
  `O(b·k)` per sub-cluster. We do **not** claim Compress++'s `O(m log³ m)` for the
  whole pipeline (we use kernel **herding**, not kernel **thinning**).
- `Q_3` (validation-failure proxies) is **ablation-only**, never in the main
  method, and never derived from test data.
  - **Naming note:** this `Q_3` (validation-failure) is **not** the same thing as
    `Q3_deepseek_anomaly` in `qproxy_v3`'s hybrid set. The latter is an ordinary
    *anomaly-behavior* family built from training-side semantics and is fine to
    use in the main method — the shared "Q3" label is incidental. Treat the six
    `qproxy_v3` families (`template / paraphrase / anomaly / hard / normal /
    appearance`) as the main proxy families; the validation-failure proxy is a
    separate, ablation-only construct.

These mark exactly which claims are proven, which are empirical, and which still
need a statistician's eye before submission.

---

## Citation / context

Research code for the CAWOT line of work (Sim2Real cross-modal anomaly retrieval,
PAB benchmark). The poster version (V0–V2) was presented at the SNAB/VIASM
workshop; V3 is the extended kernel/MMD method under development.
