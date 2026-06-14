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

### 3. Stage C — fine-tune + R@1/R@5/R@10/mAP

Selection (Stages A/B) is fully runnable on its own. To turn a selected coreset
into retrieval numbers, plug your existing CLIP fine-tuning routine into the hook:

```python
from cawot.pipeline import evaluate_coreset
metrics = evaluate_coreset(coreset_indices, train_fn=my_finetune, eval_fn=my_retrieval_eval)
```

---

## The decisive experiment (run this first)

The **`eta=0` (relevance-only) vs full** comparison is the life-or-death test for
the proxy-disagreement idea. `run_ablation.py` already produces both coresets;
fine-tune **`relevance_only_norm` vs `full_norm`** first:

- full wins → the story is *"proxy-disagreement improves robustness"*;
- relevance-only wins → the story is *"finite-proxy mean-relevance allocation"*,
  and disagreement becomes a robustness diagnostic;
- either way the paper survives, because **mean relevance is the backbone**.

Run the four tiers in order — **A** sanity (no fine-tune) → **B** selection
benchmark → **C** fine-tune at **20% first** → **D** scale up
(45K → 100K → 250K → 1M). **Do not jump to 1M.**

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

These mark exactly which claims are proven, which are empirical, and which still
need a statistician's eye before submission.

---

## Citation / context

Research code for the CAWOT line of work (Sim2Real cross-modal anomaly retrieval,
PAB benchmark). The poster version (V0–V2) was presented at the SNAB/VIASM
workshop; V3 is the extended kernel/MMD method under development.
