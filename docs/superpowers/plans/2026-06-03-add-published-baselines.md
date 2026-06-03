# Published Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three paper-backed supplementary baselines before any full compute run: CLIPScore filtering, SemDeDup, and k-center greedy.

**Architecture:** Keep the current CAWOT-CM sweep pipeline intact. Add baseline selectors in one focused module, dispatch them from `scripts/run_sweep.py`, and run them through the same budgets, seeds, train/eval code, and resumable `records.csv` protocol as existing methods.

**Tech Stack:** Python, NumPy, FAISS k-means from existing `src/cluster.py`, existing CLIP image/text embeddings, `pytest`, Kaggle P100 runtime.

---

## Source Audit

Use these sources when writing Related Work and method notes:

| Baseline | Paper/source | Plan decision |
|---|---|---|
| `clipscore` | Hessel et al., EMNLP 2021, "CLIPScore: A Reference-free Evaluation Metric for Image Captioning" | Implement top-B image/text compatibility by `cos(z_v, z_t)`. It is a paper-derived filtering adaptation, not the original caption-eval protocol. |
| `semdedup` | Abbas et al., 2023, "SemDeDup: Data-efficient learning at web-scale through semantic deduplication" | Implement semantic duplicate removal on joint image/text embeddings inside existing image clusters. Keep exact budget by filling after thresholded dedup. |
| `k_center` | Sener and Savarese, ICLR 2018, "Active Learning for CNNs: A Core-Set Approach" | Implement exact greedy k-center inside existing image clusters for scalable fair comparison. Call it "clustered k-center" in paper text. |
| optional later: `cliploss` | Wang et al., 2024, "CLIPLoss and Norm-Based Data Selection Methods for Multimodal Contrastive Learning" | Do not include in first supplementary run. It needs contrastive-pair normalization and should be a separate ablation. |
| optional later: `jest` | Evans et al., NeurIPS 2024 Datasets and Benchmarks, "Data curation via joint example selection further accelerates multimodal learning" | Do not include in first supplementary run. It is batch-level online selection, not a simple fixed coreset selector. |

Stop condition before implementation: if any of the three Tier-2 baselines lacks a stable paper link or cannot be mapped to the current retrieval setup without changing the training objective, do not implement it in this pass.

---

## File Structure

Create:
- `src/select_published_baselines.py` - all new baseline selectors and small helpers.
- `tests/test_select_published_baselines.py` - deterministic unit tests for exact budget, ranking, dedup behavior, and k-center behavior.
- `configs/smoke_published_baselines.yaml` - one-budget, one-seed, one-epoch smoke config for the three new baselines.

Modify:
- `scripts/run_sweep.py` - import selectors, list methods in docstring, dispatch methods, compute flat clusters/text embeddings only when needed.
- `config.yaml` - document optional baseline config values; do not switch default full method list until smoke passes.
- `README.md` - add method definitions, paper source mapping, and run order.

Do not modify:
- `src/train.py`, `src/eval.py`, `src/embed.py`, or dataset loading unless smoke exposes a real integration bug.

---

## Method Definitions

### CLIPScore Filtering

Selection score:

```python
score_i = float(np.sum(zv[i] * zt[i]))
```

Pick top `budget` by descending score. Return sorted int64 indices for consistency with existing selectors.

### SemDeDup

Use joint embedding:

```python
pair_i = normalize(concat(sqrt(lambda_image) * zv_i, sqrt(1 - lambda_image) * zt_i))
```

Within each existing image k-means cluster:
1. Allocate cluster budget with `_proportional_budgets`.
2. Compute local joint centroid.
3. Sort candidates by ascending cosine to joint centroid when `keep="hard"`; this follows the common SemDeDup "keep harder representative" interpretation.
4. Greedily accept a candidate if its max similarity to already selected local candidates is `<= max_similarity`.
5. If thresholded dedup yields fewer than the exact local budget, fill from the remaining candidate order.

Default config:

```yaml
coreset:
  published_baselines:
    semdedup:
      lambda_image: 0.5
      max_similarity: 0.95
      keep: hard
```

### k-center Greedy

Within each existing image k-means cluster:
1. Allocate cluster budget with `_proportional_budgets`.
2. First center is closest to the cluster centroid.
3. Repeatedly add the point with maximum current minimum cosine distance to selected centers:

```python
dist_to_new = 1.0 - embeddings @ embeddings[new_center]
min_dist = np.minimum(min_dist, dist_to_new)
```

Return exact `budget` sorted int64 indices.

---

## Task 1: Add Failing Tests

**Files:**
- Create: `tests/test_select_published_baselines.py`

- [ ] **Step 1: Write CLIPScore test**

```python
import numpy as np

from src.select_published_baselines import (
    select_clipscore,
    select_clustered_k_center,
    select_semdedup,
)


def test_select_clipscore_takes_highest_image_text_alignment():
    zv = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    zt = np.array(
        [
            [0.9, 0.1],
            [1.0, 0.0],
            [-1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    selected = select_clipscore(zv, zt, budget=2, seed=0)

    assert selected.dtype == np.int64
    assert selected.tolist() == [0, 1]
```

- [ ] **Step 2: Write SemDeDup exact-budget test**

```python
def test_select_semdedup_skips_near_duplicates_then_fills_exact_budget():
    zv = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    zt = zv.copy()
    centroids = np.array([[1.0, 0.0]], dtype=np.float32)
    assignments = np.array([0, 0, 0, 0], dtype=np.int64)

    selected = select_semdedup(
        zv,
        zt,
        centroids,
        assignments,
        budget=3,
        seed=0,
        lambda_image=0.5,
        max_similarity=0.95,
        keep="hard",
    )

    assert selected.dtype == np.int64
    assert len(selected) == 3
    assert len(set(selected.tolist())) == 3
    assert not ({0, 1} <= set(selected.tolist()))
```

- [ ] **Step 3: Write k-center spread test**

```python
def test_select_clustered_k_center_spreads_points_within_cluster():
    emb = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )
    centroids = np.array([[0.0, 1.0]], dtype=np.float32)
    assignments = np.array([0, 0, 0, 0], dtype=np.int64)

    selected = select_clustered_k_center(
        emb,
        centroids,
        assignments,
        budget=2,
        seed=0,
    )

    assert selected.dtype == np.int64
    assert selected.tolist() == [2, 3]
```

- [ ] **Step 4: Run tests and verify they fail for missing module**

Run:

```bash
python -m pytest tests/test_select_published_baselines.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'src.select_published_baselines'
```

---

## Task 2: Implement Baseline Selectors

**Files:**
- Create: `src/select_published_baselines.py`
- Test: `tests/test_select_published_baselines.py`

- [ ] **Step 1: Add imports and validation helpers**

```python
from __future__ import annotations

import numpy as np

from .select import _proportional_budgets
from .utils import setup_logger

logger = setup_logger("select_published_baselines")


def _validate_budget(n_total: int, budget: int) -> int:
    budget = int(budget)
    if budget < 0 or budget > int(n_total):
        raise ValueError(f"budget must be in [0, {n_total}], got {budget}")
    return budget


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)
```

- [ ] **Step 2: Implement `select_clipscore`**

```python
def select_clipscore(zv: np.ndarray, zt: np.ndarray, budget: int, seed: int = 42) -> np.ndarray:
    zv = np.asarray(zv, dtype=np.float32)
    zt = np.asarray(zt, dtype=np.float32)
    if zv.ndim != 2 or zt.ndim != 2 or zv.shape != zt.shape:
        raise ValueError(f"zv and zt must have same 2D shape, got {zv.shape} and {zt.shape}")
    budget = _validate_budget(len(zv), budget)
    if budget == 0:
        return np.empty((0,), dtype=np.int64)
    rng = np.random.RandomState(seed)
    scores = np.sum(zv * zt, axis=1).astype(np.float64)
    jitter = rng.uniform(0.0, 1e-12, size=len(scores))
    selected = np.argsort(-(scores + jitter))[:budget]
    out = np.sort(selected).astype(np.int64)
    logger.info(f"[clipscore] selected {len(out)} (target {budget})")
    return out
```

- [ ] **Step 3: Implement SemDeDup helpers and selector**

```python
def _joint_embeddings(zv: np.ndarray, zt: np.ndarray, lambda_image: float) -> np.ndarray:
    if not (0.0 <= lambda_image <= 1.0):
        raise ValueError(f"lambda_image must be in [0, 1], got {lambda_image}")
    lambda_text = 1.0 - lambda_image
    joint = np.concatenate(
        [np.sqrt(lambda_image) * zv, np.sqrt(lambda_text) * zt],
        axis=1,
    )
    return _normalize_rows(joint)


def _dedup_local(pair_emb: np.ndarray, budget: int, max_similarity: float, keep: str) -> np.ndarray:
    n = len(pair_emb)
    budget = min(int(budget), n)
    if budget <= 0:
        return np.empty((0,), dtype=np.int64)
    if not (-1.0 <= max_similarity <= 1.0):
        raise ValueError(f"max_similarity must be in [-1, 1], got {max_similarity}")
    centroid = _normalize_rows(pair_emb.mean(axis=0, keepdims=True))[0]
    centroid_sim = pair_emb @ centroid
    if keep == "hard":
        order = np.argsort(centroid_sim)
    elif keep == "easy":
        order = np.argsort(-centroid_sim)
    else:
        raise ValueError(f"keep must be 'hard' or 'easy', got {keep}")

    selected: list[int] = []
    for j in order:
        if len(selected) == budget:
            break
        if not selected:
            selected.append(int(j))
            continue
        max_sim = float((pair_emb[selected] @ pair_emb[j]).max())
        if max_sim <= max_similarity:
            selected.append(int(j))

    if len(selected) < budget:
        used = set(selected)
        for j in order:
            if len(selected) == budget:
                break
            if int(j) not in used:
                selected.append(int(j))
                used.add(int(j))
    return np.asarray(selected, dtype=np.int64)


def select_semdedup(
    zv: np.ndarray,
    zt: np.ndarray,
    centroids: np.ndarray,
    assignments: np.ndarray,
    budget: int,
    seed: int = 42,
    *,
    lambda_image: float = 0.5,
    max_similarity: float = 0.95,
    keep: str = "hard",
) -> np.ndarray:
    del seed
    zv = np.asarray(zv, dtype=np.float32)
    zt = np.asarray(zt, dtype=np.float32)
    assignments = np.asarray(assignments, dtype=np.int64)
    if zv.ndim != 2 or zt.ndim != 2 or zv.shape != zt.shape:
        raise ValueError(f"zv and zt must have same 2D shape, got {zv.shape} and {zt.shape}")
    if assignments.shape[0] != len(zv):
        raise ValueError("assignments length must match embeddings")
    budget = _validate_budget(len(zv), budget)
    if budget == 0:
        return np.empty((0,), dtype=np.int64)

    k = int(centroids.shape[0])
    sizes = np.bincount(assignments, minlength=k)
    budgets = _proportional_budgets(sizes, budget)
    pair_emb = _joint_embeddings(zv, zt, lambda_image=lambda_image)
    selected: list[np.ndarray] = []
    for c in range(k):
        b_c = int(budgets[c])
        if b_c == 0:
            continue
        idx = np.where(assignments == c)[0]
        if b_c >= len(idx):
            selected.append(idx)
            continue
        local = _dedup_local(pair_emb[idx], b_c, max_similarity=max_similarity, keep=keep)
        selected.append(idx[local])
    out = np.sort(np.concatenate(selected)).astype(np.int64) if selected else np.empty((0,), np.int64)
    if len(out) != budget:
        raise RuntimeError(f"SemDeDup selected {len(out)} samples, expected {budget}")
    logger.info(f"[semdedup] selected {len(out)} (target {budget})")
    return out
```

- [ ] **Step 4: Implement clustered k-center**

```python
def _k_center_local(embeddings: np.ndarray, centroid: np.ndarray, budget: int) -> np.ndarray:
    n = len(embeddings)
    budget = min(int(budget), n)
    if budget <= 0:
        return np.empty((0,), dtype=np.int64)
    centroid = _normalize_rows(np.asarray(centroid, dtype=np.float32).reshape(1, -1))[0]
    selected = [int(np.argmax(embeddings @ centroid))]
    min_dist = 1.0 - embeddings @ embeddings[selected[0]]
    min_dist[selected[0]] = -np.inf
    while len(selected) < budget:
        j = int(np.argmax(min_dist))
        selected.append(j)
        dist = 1.0 - embeddings @ embeddings[j]
        min_dist = np.minimum(min_dist, dist)
        min_dist[selected] = -np.inf
    return np.asarray(selected, dtype=np.int64)


def select_clustered_k_center(
    embeddings: np.ndarray,
    centroids: np.ndarray,
    assignments: np.ndarray,
    budget: int,
    seed: int = 42,
) -> np.ndarray:
    del seed
    embeddings = np.asarray(embeddings, dtype=np.float32)
    centroids = np.asarray(centroids, dtype=np.float32)
    assignments = np.asarray(assignments, dtype=np.int64)
    budget = _validate_budget(len(embeddings), budget)
    if budget == 0:
        return np.empty((0,), dtype=np.int64)

    k = int(centroids.shape[0])
    sizes = np.bincount(assignments, minlength=k)
    budgets = _proportional_budgets(sizes, budget)
    selected: list[np.ndarray] = []
    for c in range(k):
        b_c = int(budgets[c])
        if b_c == 0:
            continue
        idx = np.where(assignments == c)[0]
        if b_c >= len(idx):
            selected.append(idx)
            continue
        local = _k_center_local(embeddings[idx], centroids[c], b_c)
        selected.append(idx[local])
    out = np.sort(np.concatenate(selected)).astype(np.int64) if selected else np.empty((0,), np.int64)
    if len(out) != budget:
        raise RuntimeError(f"k-center selected {len(out)} samples, expected {budget}")
    logger.info(f"[k_center] selected {len(out)} (target {budget})")
    return out
```

- [ ] **Step 5: Run tests**

Run:

```bash
python -m pytest tests/test_select_published_baselines.py -q
```

Expected:

```text
3 passed
```

---

## Task 3: Wire Baselines Into Sweep

**Files:**
- Modify: `scripts/run_sweep.py`
- Test: `tests/test_select_published_baselines.py`

- [ ] **Step 1: Update imports**

Add:

```python
from src.select_published_baselines import (
    select_clipscore,
    select_clustered_k_center,
    select_semdedup,
)
```

- [ ] **Step 2: Update docstring method list**

Add:

```text
  clipscore - top-B CLIP image/text compatibility filtering
  semdedup  - SemDeDup-style semantic duplicate pruning on joint embeddings
  k_center  - clustered k-center greedy on image embeddings
```

- [ ] **Step 3: Extend `select_coreset` signature**

Add keyword arguments:

```python
semdedup_lambda_image=0.5,
semdedup_max_similarity=0.95,
semdedup_keep="hard",
```

- [ ] **Step 4: Add dispatch cases**

Add before `v1`:

```python
    if method == "clipscore":
        return select_clipscore(zv, zt, b, seed=seed)
    if method == "semdedup":
        return select_semdedup(
            zv,
            zt,
            centroids,
            assignments,
            b,
            seed=seed,
            lambda_image=semdedup_lambda_image,
            max_similarity=semdedup_max_similarity,
            keep=semdedup_keep,
        )
    if method == "k_center":
        return select_clustered_k_center(zv, centroids, assignments, b, seed=seed)
```

- [ ] **Step 5: Update embedding/cluster dependency flags**

Use:

```python
needs_text = any(m in methods for m in ("v1", "v2", "v2_1", "clipscore", "semdedup"))
needs_qproxy = any(m in methods for m in ("v2", "v2_1"))
needs_flat = any(m in methods for m in ("v0", "v0_proto", "v1", "v2", "semdedup", "k_center"))
```

- [ ] **Step 6: Read SemDeDup config**

After V2.1 config parsing:

```python
published_cfg = cfg["coreset"].get("published_baselines", {})
semdedup_cfg = published_cfg.get("semdedup", {})
semdedup_lambda_image = float(semdedup_cfg.get("lambda_image", 0.5))
semdedup_max_similarity = float(semdedup_cfg.get("max_similarity", 0.95))
semdedup_keep = str(semdedup_cfg.get("keep", "hard"))
```

Pass those values into `select_coreset`.

- [ ] **Step 7: Run compile and tests**

Run:

```bash
python -m pytest tests/test_select_published_baselines.py tests/test_select_v2_1.py -q
python -m py_compile src/select_published_baselines.py scripts/run_sweep.py
```

Expected:

```text
all tests pass
py_compile exits 0
```

---

## Task 4: Add Smoke Config

**Files:**
- Create: `configs/smoke_published_baselines.yaml`
- Modify: `config.yaml`
- Modify: `README.md`

- [ ] **Step 1: Create smoke config**

Use this full file:

```yaml
# Smoke-test config for paper-backed supplementary baselines.
# Runs CLIPScore, SemDeDup, and clustered k-center at 5% budget, 1 seed, 1 epoch.

seed: 42

data:
  image_root: "/kaggle/working/pab_data/images"
  annotations_dir: "/kaggle/working/pab_data/annotations"
  sample_size: 50000
  val_size: 1000
  image_size: 224
  num_workers: 2

embed:
  model: "ViT-B-16"
  pretrained: "openai"
  batch_size: 128
  device: "cuda"
  output_dir: "/kaggle/working/outputs/embeddings"

cluster:
  k: 150
  niter: 25
  spherical: true
  use_gpu: true
  output_path: "/kaggle/working/outputs_smoke_published_baselines/clusters.npz"

coreset:
  budgets: [0.05]
  methods: [clipscore, semdedup, k_center]
  v2_alpha: 0.5
  published_baselines:
    semdedup:
      lambda_image: 0.5
      max_similarity: 0.95
      keep: hard
  output_dir: "/kaggle/working/outputs_smoke_published_baselines"

qproxy:
  queries_json_path: "/kaggle/working/qproxy/queries.json"
  cache_path: "/kaggle/working/outputs/embeddings/qproxy_clip_text_emb.npy"

train:
  model: "ViT-B-16"
  pretrained: "openai"
  unfreeze_last_n_layers: 4
  batch_size: 96
  num_epochs: 1
  lr: 1.0e-5
  weight_decay: 0.1
  warmup_steps: 50
  temperature: 0.07
  use_learned_temp: true
  grad_accum_steps: 1
  amp: true
  output_dir: "/kaggle/working/outputs_smoke_published_baselines/checkpoints"
  log_every: 25
  seeds: [42]
  keep_checkpoints: false

eval:
  batch_size: 128
  k_values: [1, 5]
  output_dir: "/kaggle/working/outputs_smoke_published_baselines/eval"
```

- [ ] **Step 2: Add default config documentation**

Add under `coreset` in `config.yaml`:

```yaml
  published_baselines:
    semdedup:
      lambda_image: 0.5
      max_similarity: 0.95
      keep: hard
```

Do not change `coreset.methods: [v2_1]` until the smoke baseline config passes.

- [ ] **Step 3: Add README method notes**

Add concise definitions for:
- `CLIPScore filtering`: source Hessel et al. EMNLP 2021; top-B image/text cosine.
- `SemDeDup`: source Abbas et al. 2023; joint image/text semantic duplicate pruning.
- `clustered k-center`: source Sener and Savarese ICLR 2018; max-min coverage on image embeddings.

- [ ] **Step 4: Verify YAML parses**

Run:

```powershell
@'
from src.utils import load_config
for p in ["config.yaml", "configs/smoke_published_baselines.yaml"]:
    cfg = load_config(p)
    print(p, cfg["coreset"]["methods"])
'@ | python -
```

Expected:

```text
config.yaml ['v2_1']
configs/smoke_published_baselines.yaml ['clipscore', 'semdedup', 'k_center']
```

---

## Task 5: Local Verification Before Kaggle

**Files:**
- No source edits unless commands fail.

- [ ] **Step 1: Run focused tests**

```bash
python -m pytest tests/test_select_published_baselines.py tests/test_select_v2_1.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Compile touched Python modules**

```bash
python -m py_compile src/select_published_baselines.py src/select.py src/select_v2_1.py scripts/run_sweep.py src/utils.py
```

Expected: exit 0.

- [ ] **Step 3: Check whitespace**

```bash
git diff --check
```

Expected: exit 0. CRLF warnings on Windows are acceptable; whitespace errors are not.

- [ ] **Step 4: Review diff scope**

```bash
git status --short
git diff --stat
```

Expected changed files:
- `src/select_published_baselines.py`
- `tests/test_select_published_baselines.py`
- `scripts/run_sweep.py`
- `configs/smoke_published_baselines.yaml`
- `config.yaml`
- `README.md`
- this plan file

---

## Task 6: Kaggle Smoke Run Gate

**Files:**
- No code edits during this gate unless smoke fails.

- [ ] **Step 1: Prepare data and Q_proxy**

Run on Kaggle:

```bash
python scripts/setup_qproxy.py --output ./qproxy --only-queries
python scripts/run_sweep.py --config configs/smoke_published_baselines.yaml
```

- [ ] **Step 2: Verify smoke records**

Run on Kaggle:

```bash
python - <<'PY'
import csv
from pathlib import Path
p = Path("/kaggle/working/outputs_smoke_published_baselines/eval/records.csv")
rows = list(csv.DictReader(open(p, encoding="utf-8")))
methods = sorted({r["method"] for r in rows if r["method"] != "zeroshot"})
keys = sorted({(r["method"], r["budget"], r["seed"]) for r in rows if r["method"] != "zeroshot"})
print("methods", methods)
print("combos", keys)
assert methods == ["clipscore", "k_center", "semdedup"]
assert len(keys) == 3
for r in rows:
    if r["method"] != "zeroshot":
        assert r["mean_R@1"] != ""
print("smoke records ok")
PY
```

Expected:

```text
methods ['clipscore', 'k_center', 'semdedup']
smoke records ok
```

- [ ] **Step 3: Stop/fix criteria**

Stop and fix before full sweep if any of these happens:
- selection count mismatch,
- `records.csv` lacks one of the three baseline methods,
- any `mean_R@1` field is empty or NaN,
- a baseline is slower than V2.1 selection by more than 3x before training starts,
- k-center OOMs or exceeds practical runtime.

---

## Task 7: Full Supplementary Baseline Sweep

**Files:**
- Create after smoke passes: `configs/baselines_published.yaml`

- [ ] **Step 1: Create full baseline config**

Copy `config.yaml` and change only:

```yaml
coreset:
  methods: [clipscore, semdedup, k_center]
  output_dir: "/kaggle/working/outputs_published_baselines"

eval:
  output_dir: "/kaggle/working/outputs_published_baselines/eval"

train:
  output_dir: "/kaggle/working/outputs_published_baselines/checkpoints"
```

Keep:

```yaml
coreset.budgets: [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
train.seeds: [42, 1, 2]
train.num_epochs: 3
```

- [ ] **Step 2: Run resumable full sweep**

```bash
python scripts/run_sweep.py --config configs/baselines_published.yaml
```

Expected total new combos: `3 methods x 6 budgets x 3 seeds = 54`.

- [ ] **Step 3: Verify full records**

```bash
python - <<'PY'
import csv
from pathlib import Path
p = Path("/kaggle/working/outputs_published_baselines/eval/records.csv")
rows = list(csv.DictReader(open(p, encoding="utf-8")))
keys = {(r["method"], r["budget"], r["seed"]) for r in rows if r["method"] != "zeroshot"}
print("num method-budget-seed combos", len(keys))
print(sorted({r["method"] for r in rows}))
assert len(keys) == 54
assert {"clipscore", "semdedup", "k_center"}.issubset({r["method"] for r in rows})
print("full baseline records ok")
PY
```

Expected:

```text
num method-budget-seed combos 54
full baseline records ok
```

---

## Task 8: Analysis Gate Before Paper Claims

**Files:**
- No code edits unless analysis scripts are missing required plots.

- [ ] **Step 1: Compare against existing baselines and V2.1**

Use `outputs/eval/summary.json` for V2.1 and `outputs_published_baselines/eval/summary.json` for new baselines. Compare on `overall.mean_R@1_mean` for every budget.

- [ ] **Step 2: Minimum report table**

Create a table with columns:

```text
method | 5% | 10% | 20% | 30% | 40% | 50% | avg rank
```

Include methods:

```text
random, v0, v0_proto, v1, v2, v2_1, clipscore, semdedup, k_center
```

- [ ] **Step 3: Claim gate**

Allowed claims:
- "V2.1 is competitive with paper-backed baselines" if it is within one standard deviation of the best method across most budgets.
- "V2.1 improves over V2" only if V2.1 beats V2 at most budgets or has higher average rank.
- "CLIPScore is a strong cross-modal filter" only if it beats random at low budgets.
- "SemDeDup helps redundancy control" only if it beats random or improves low-budget stability.

Do not claim SOTA. This is a coreset-efficiency study on PAB, not a retrieval architecture benchmark.

---

## Execution Order

1. Implement Tasks 1-5 locally.
2. Commit and push branch.
3. Run Task 6 smoke on Kaggle.
4. Only after smoke passes, run Task 7 full supplementary sweep.
5. Run Task 8 analysis before changing paper claims.
