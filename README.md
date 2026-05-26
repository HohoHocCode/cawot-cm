# CAWOT-CM V0: Diversity-Only Coreset Baseline

V0 baseline cho CAWOT-CM paper. Mục đích: sanity check rằng intelligent selection (cluster + farthest-from-centroid) beats random sampling cho text-based person retrieval trên PAB benchmark.

## Pipeline V0

```
1M synthetic pairs (PAB)
  → Extract CLIP embeddings
  → FAISS k-means (K=5000 clusters)
  → Budget allocation: PROPORTIONAL (B_k = budget × n_k / N)
  → Within cluster: Farthest-from-centroid sampling
  → Coreset (20% by default)
  → Fine-tune CLIP
  → Eval on 1978 test queries vs 36773 gallery
```

**V0 KHÔNG có**: cross-modal cost, Q_proxy, Wasserstein gap, submodular. Những thứ đó để V1/V2.

## Dataset

Source: [TruongVox/Cawot-dataset](https://huggingface.co/datasets/TruongVox/Cawot-dataset) on Hugging Face.

### Expected upstream layout (target spec)

```
TruongVox/Cawot-dataset/
├── train/
│   ├── imgs_0.zip        # ~1.4 GB, ~13k synthetic images
│   ├── imgs_0.jsonl      # newline-delimited annotations for imgs_0.zip
│   ├── imgs_1.zip
│   ├── imgs_1.jsonl
│   ├── ...               # 75 (zip, jsonl) pairs
│   ├── imgs_74.zip
│   └── imgs_74.jsonl
└── test/
    ├── gallery.zip       # 36,773 real gallery images
    ├── gallery.jsonl     # gallery metadata + person id
    └── queries.jsonl     # 1,978 text queries + target person id
```

### JSONL schemas

Each line is one JSON object.

**`train/imgs_N.jsonl`**
```json
{"image": "imgs_0/0.jpg", "caption": "A person wearing red...", "id": 42}
```
- **Required**: `image` (path inside the zip), `caption`, `id` (person identity int)
- Optional: `scene`, `action`, `image_id`, etc.

**`test/queries.jsonl`**
```json
{"caption": "An elderly woman in a blue coat with a cane", "id": 42}
```

**`test/gallery.jsonl`**
```json
{"image": "0.jpg", "id": 42}
```

The `id` field is the **person identity**. A retrieved gallery image counts as a correct match for a query when their `id` values are equal. Without this field, R@k / mAP cannot be computed — the dataset is unusable for retrieval evaluation.

### On-disk layout after `download_pab.py`

```
pab_data/
├── images/                 # flat union of train + gallery images (basename only)
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
├── train.jsonl             # concatenation of all train/imgs_N.jsonl shards
├── test_query.jsonl
└── test_gallery.jsonl
```

The data loader resolves image paths by basename against `<root>/images/`, so any JSON `image` value — `"0.jpg"`, `"imgs_0/0.jpg"`, `"train/imgs_0/goal/0.jpg"` — all work.

> **Status**: the upstream HF dataset is still being assembled. As of this commit, `train/imgs_N.json` exists in JSON-Lines format **without** a person-`id` field, and there is no `test/` split yet. `download_pab.py` will be updated once the dataset matches the spec above.

## Quick start (Colab)

Mở `notebooks/colab_v0.ipynb` lên Colab → Run all. Có 2 mode:
- `MODE = "sanity"` — dummy data, ~5-10 phút trên T4 free, kiểm tra pipeline.
- `MODE = "pab"` — auto-download N zips từ HF + chạy end-to-end.

## Quick start (local / ThunderCompute)

```bash
pip install -r requirements.txt

# 0. Download dataset from Hugging Face
#    Full (~104 GB):
python scripts/download_pab.py --root ./pab_data
#    Or subset for testing (e.g. 3 zips ≈ 4 GB):
python scripts/download_pab.py --root ./pab_data --num-zips 3
#    For partial downloads, filter annotations to entries with local images:
python scripts/filter_annotations.py --root ./pab_data
#    then in config.yaml set train_json: "train_local.json"

# 1. Extract embeddings (~30 min on A100, ~2hr on T4)
python scripts/01_extract_embeddings.py --config config.yaml

# 2. Select coreset (V0 = diversity-only, also runs Random baseline)
python scripts/02_select_coreset.py --config config.yaml --method v0
python scripts/02_select_coreset.py --config config.yaml --method random

# 3. Fine-tune
python scripts/03_train.py --config config.yaml --coreset outputs/coreset_v0.npy  --name v0
python scripts/03_train.py --config config.yaml --coreset outputs/coreset_random.npy --name random

# 4. Evaluate
python scripts/04_evaluate.py --config config.yaml --checkpoint outputs/checkpoints/v0.pt    --name v0
python scripts/04_evaluate.py --config config.yaml --checkpoint outputs/checkpoints/random.pt --name random

# Or do steps 1-4 in one shot:
python scripts/run_v0.py --config config.yaml
```

## Project structure

```
cawot-cm-v0/
├── config.yaml          # All hyperparameters in one place
├── src/
│   ├── data.py          # PAB dataset loader (resolves images via <root>/images/)
│   ├── embed.py         # CLIP embedding extraction
│   ├── cluster.py       # FAISS k-means clustering
│   ├── select.py        # Random + V0 (farthest-from-centroid) selection
│   ├── train.py         # CLIP fine-tuning with InfoNCE
│   ├── eval.py          # Retrieval metrics (R@1/5/10, mAP)
│   └── utils.py         # logger, seed, etc.
├── scripts/
│   ├── download_pab.py        # HF → ./pab_data with flat images/
│   ├── filter_annotations.py  # Keep only entries with locally available images
│   ├── make_dummy_data.py     # Pipeline sanity check without PAB
│   ├── 01_extract_embeddings.py
│   ├── 02_select_coreset.py
│   ├── 03_train.py
│   ├── 04_evaluate.py
│   └── run_v0.py              # End-to-end runner
└── notebooks/
    └── colab_v0.ipynb         # End-to-end Colab demo
```

## Dataset expected format

PAB / CMP format: JSON with list of dicts:
```json
[
  {"image_path": "synthetic/0001.jpg", "captions": ["A person walking..."], "id": 1},
  ...
]
```

Edit `src/data.py:PABDataset` if your format differs.

## Notes for V0 → V1 transition

V0 uses **frozen CLIP backbone + tunable last 4 layers**. For V1 sẽ:
- Switch to IRRA backbone (BERT + ViT + SDM loss)
- Add cross-modal cost matrix (rank-normalized image + text + alignment)
- Replace farthest-point with submodular facility location (submodlib)

V0 expected results (rough estimate at 20% budget):
- Random:  ~78-79% R@1
- V0:      ~79-80% R@1 (+0.5-1.5%)

Nếu V0 không beat Random → có bug ở pipeline cơ bản, fix trước khi sang V1.
