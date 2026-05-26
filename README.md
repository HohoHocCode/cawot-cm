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

Source: [TruongVox/Cawot-dataset](https://huggingface.co/datasets/TruongVox/Cawot-dataset) on Hugging Face — 75 image zip archives (~104 GB total) + 3 JSON annotation files (`train.json`, `test_query.json`, `test_gallery.json`).

After running the download script the on-disk layout is:

```
pab_data/
├── images/                 # ALL images flat (synthetic train + real gallery)
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
├── train.json
├── test_query.json
└── test_gallery.json
```

The data loader (`src/data.py`) resolves image paths by basename against `<root>/images/`, so JSON entries with paths like `"000001.jpg"`, `"images/000001.jpg"`, or `"imgs/000001.jpg"` all work.

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
