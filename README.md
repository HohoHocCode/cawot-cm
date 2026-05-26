# CAWOT-CM V0: Diversity-Only Coreset Baseline

V0 baseline cho CAWOT-CM paper. Sanity check rằng intelligent selection (cluster + farthest-from-centroid) beats random sampling cho text-based person retrieval trên **PAB ECCV'26 Workshop Track 4**.

## Pipeline V0

```
Friend's qproxy outputs (~47K filtered subset, EVA02-E-14 embeddings)
  → Hold out 2K (image, caption) pairs for image-text retrieval val
  → FAISS k-means (K=150, spherical) trên ~45K train pool
  → 2 coresets at 20% budget:
        Random (baseline)
        V0 (farthest-from-centroid within each cluster)
  → Fine-tune CLIP ViT-B/16 trên mỗi coreset (last 4 layers + InfoNCE)
  → Image-text retrieval R@1/5/10 trên val split
  → Expect: V0 > Random by ~0.5-1.5% mean R@1
```

V0 không có: cross-modal cost, Q_proxy budget allocation, Wasserstein gap, submodular optimization — đó là V1/V2.

## Tại sao dùng 2 model khác nhau?

- **Selection model = EVA02-E-14** (~5B params, 1024-dim): friend đã pre-compute embeddings trên 47K subset. Chỉ dùng cho clustering + selection.
- **Task model = CLIP ViT-B/16** (~150M params, 512-dim): cái mà chúng ta fine-tune trên coreset. Fits Kaggle P100 (16GB VRAM), trong khi EVA02-E-14 cần A100 80GB.

Selection model ≠ task model là standard practice. Reviewer accept được.

## Inputs cần có

Trên máy chạy V0:

1. **Friend's qproxy outputs** (~213MB) — folder chứa:
   - `image_subset_manifest.parquet` — schema: `row_id, image_id, image_path, caption, scene, action, ...`
   - `image_subset_embeddings.npy` — (47K, 1024) EVA02-E-14 embeddings, parallel với manifest
   - `filter_metadata.json` — config friend's filter

2. **Train images** (~10GB, webp) — friend's Kaggle dataset `vnhtbo/pab-eccv26-track4-train-webp-part-01-05` (Parts 1-5). Manifest `image_path` column đã trỏ tới `/kaggle/input/datasets/vnhtbo/pab-eccv26-track4-train-webp-part-01-05/Part X/...`. Nếu Kaggle dataset slug khác, dùng `qproxy.path_remap` trong `config.yaml`.

## Quick start (Kaggle)

1. New Kaggle notebook, Settings → Accelerator = GPU P100, Internet ON
2. Add datasets:
   - Friend's image dataset (e.g. `vnhtbo/pab-eccv26-track4-train-webp-part-01-05`)
   - Friend's qproxy dataset (upload từ Drive)
3. Clone repo + install:
   ```python
   !git clone https://github.com/HohoHocCode/cawot-cm.git
   %cd cawot-cm
   !pip install -q open_clip_torch faiss-gpu-cu12 pyarrow einops
   ```
4. Edit `config.yaml`:
   - `qproxy.manifest_path` — path đến `image_subset_manifest.parquet`
   - `qproxy.embeddings_path` — path đến `image_subset_embeddings.npy`
   - `qproxy.path_remap` — nếu Kaggle dataset slug khác với friend's
5. Run:
   ```python
   !python scripts/run_v0.py --config config.yaml
   ```

Kết quả ở `outputs/eval/summary.json`.

Có template notebook `notebooks/kaggle_v0.ipynb` để copy thẳng.

## Quick start (local / ThunderCompute)

```bash
pip install -r requirements.txt

# Edit config.yaml: set absolute paths to friend's parquet + npy,
# and qproxy.path_remap nếu image_path trong manifest cần rewrite.

python scripts/run_v0.py --config config.yaml
```

## Project structure

```
cawot-cm-v0/
├── config.yaml                 # qproxy paths, cluster/coreset/train/eval config
├── src/
│   ├── data.py                 # FriendSubsetDataset (parquet-based) + legacy JSON loaders
│   ├── embed.py                # CLIP embedding extraction (legacy path only)
│   ├── cluster.py              # FAISS k-means spherical
│   ├── select.py               # select_random + select_v0 (farthest-from-centroid)
│   ├── train.py                # train_with_manifest (friend-data) + train_with_coreset (legacy)
│   ├── eval.py                 # evaluate_val_split (image-text R@k) + legacy person-id eval
│   └── utils.py
├── scripts/
│   ├── run_v0.py               # ★ END-TO-END runner — primary entry point
│   ├── make_dummy_data.py      # tạo PAB-like data giả cho pipeline sanity (không dùng EVA02 path)
│   ├── 01_extract_embeddings.py # legacy: extract CLIP embeddings từ JSON (cho dummy mode)
│   ├── 02_select_coreset.py    # legacy
│   ├── 03_train.py             # legacy
│   ├── 04_evaluate.py          # legacy
│   ├── download_pab.py         # legacy: download từ HF (không dùng nữa, data chuyển sang Kaggle)
│   └── filter_annotations.py   # legacy
└── notebooks/
    ├── kaggle_v0.ipynb         # ★ Kaggle end-to-end demo
    └── colab_v0.ipynb          # legacy Colab demo (dummy mode)
```

## Eval setup

V0 dùng **image-text retrieval R@k trên held-out val** thay vì person-id retrieval. Lý do:
- PAB ECCV'26 Track 4 test set có ground-truth person id **bị mask cố ý** (competition track) → không eval local được.
- Image-text retrieval (mỗi caption matches đúng image của nó) chỉ cần (image, caption) pairs từ train pool → đủ cho V0 sanity.
- Real eval cho competition: sau khi V0 work, generate predictions trên `name-masked_test_set/gallery.zip` và submit lên ECCV leaderboard.

Metrics output:
- `t2i_R@1/5/10`: text → image retrieval
- `i2t_R@1/5/10`: image → text retrieval
- `mean_R@1`: average của hai chiều

## Expected results (Kaggle P100, V0 vs Random, 20% budget on ~45K pool)

| Method | Expected mean R@1 |
|---|---|
| Zero-shot CLIP-B/16 | ~30-40% (baseline anchor) |
| Random | ~40-50% |
| V0 | ~41-52% (+0.5-1.5% over Random) |

Numbers tuyệt đối phụ thuộc vào pool size + epochs + LR. Cái quan trọng: **V0 > Random consistently**.

Nếu V0 ≤ Random → có bug ở pipeline hoặc K_clusters sai scale.

## V0 → V1 → V2

V0 chỉ là sanity baseline. Plan tiếp theo:

| Version | Thêm gì |
|---|---|
| V0 (current) | cluster + farthest-from-centroid, proportional budget |
| V1 | + cross-modal cost (image+text+alignment, rank-normalized) + submodular facility location (submodlib) |
| V2 | + Q_proxy (friend đã có queries.json + text_embeddings.npy + max_sim_raw.npy) + Wasserstein-aware budget allocation |

Friend's qproxy folder đã có đủ data cho cả 3 versions — V1/V2 sẽ tái sử dụng cùng manifest + embeddings.
