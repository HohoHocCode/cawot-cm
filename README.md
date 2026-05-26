# CAWOT-CM V0: Diversity-Only Coreset Baseline

V0 baseline cho CAWOT-CM paper. Sanity check rằng cluster + farthest-from-centroid selection beats random sampling cho text-based person retrieval trên **PAB ECCV'26 Workshop Track 4**.

## V0 spec

V0 = **clean baseline, hoàn toàn độc lập với V1/V2 prep work**.

```
Download N shards từ HF (TruongVox/Cawot-dataset)
  → Random sample M (image, caption) entries
  → Extract CLIP ViT-B/16 image embeddings
  → Hold out V (image, caption) pairs cho val
  → FAISS k-means (K, spherical) trên (M − V) train pool
  → 2 coresets cùng budget 20%:
        Random (baseline)
        V0 (farthest-from-centroid trong mỗi cluster)
  → Fine-tune CLIP-B/16 (last 4 layers + InfoNCE) trên mỗi coreset
  → Image-text retrieval R@1/5/10 trên val
```

**V0 KHÔNG có** (đó là V1/V2):
- ❌ Q_proxy / friend's filtered subset
- ❌ EVA02-E-14 hay model selection khác CLIP-B/16
- ❌ Cross-modal cost / submodular optimization
- ❌ Wasserstein-aware budget

## Single source of truth: HuggingFace

[`TruongVox/Cawot-dataset`](https://huggingface.co/datasets/TruongVox/Cawot-dataset) chứa cả images và annotations:

```
TruongVox/Cawot-dataset/
├── imgs_0.zip ... imgs_74.zip   # 75 shards × ~1.4 GB = 104 GB total
└── train/
    ├── imgs_0.json ... imgs_74.json   # JSONL annotations, ~5 MB each
```

V0 không cần download toàn bộ — chạy trên N shards là đủ (default N=5 ≈ 65k samples, 7 GB download, 14 GB extracted).

---

## Quick start (Kaggle)

1. New Kaggle notebook, **Settings → GPU P100 + Internet ON**
2. Copy nội dung `notebooks/kaggle_v0.ipynb`
3. **Run all** — notebook tự:
   - Clone repo
   - Install deps
   - Download 5 shards từ HF
   - Sanity check 1 image resolve
   - Chạy V0 end-to-end (~50-70 phút)
   - In summary

Không cần Add Data (không phụ thuộc Kaggle dataset nào).

## Quick start (local / ThunderCompute)

```bash
pip install -r requirements.txt

# Download N shards from HF
python scripts/setup_data.py --output ./pab_data --num-shards 5
# Or specific shards:
# python scripts/setup_data.py --output ./pab_data --shards 0,1,2,5,10

# Edit config.yaml: data.image_root + data.annotations_dir trỏ vào ./pab_data/...

python scripts/run_v0.py --config config.yaml
```

---

## Required inputs

V0 cần 2 thứ trên máy chạy — cả 2 đều download tự động bởi `scripts/setup_data.py`:

| Input | Mô tả | Nguồn |
|---|---|---|
| **Image shards** (~1.4 GB / shard) | `imgs_N.zip` extract thành `images/imgs_N/<action>/*.jpg` | HF `TruongVox/Cawot-dataset/imgs_N.zip` |
| **Annotations** (~5 MB / shard) | `imgs_N.json` JSONL, 1 entry/dòng | HF `TruongVox/Cawot-dataset/train/imgs_N.json` |

Annotation JSONL schema (1 entry / dòng):
```json
{"image": "train/imgs_0/goal/0.jpg", "caption": "...", "image_id": "0_0", "scene": "...", "normal": "..."}
```

Code tự handle `.jpg` ↔ `.webp` fallback (HF zips có thể là `.jpg`, friend's Kaggle re-encode là `.webp`).

---

## Layout sau khi setup_data.py chạy

```
pab_data/
├── images/
│   ├── imgs_0/
│   │   └── goal/
│   │       ├── 0.jpg
│   │       └── ...
│   ├── imgs_1/
│   └── ...
└── annotations/
    ├── imgs_0.json
    ├── imgs_1.json
    └── ...
```

---

## Config knobs

`config.yaml`:
- `data.image_root` / `data.annotations_dir` — trỏ vào output của setup_data.py
- `data.sample_size: 50000` — random sample size từ pool đã download
- `data.val_size: 2000` — held-out pairs cho R@k eval
- `cluster.k: 150` — rule of thumb √(N/2)
- `coreset.budget_ratio: 0.20` — coreset size = 20% train pool sau val split
- `train.num_epochs: 3` — đủ cho V0 sanity
- `train.batch_size: 96` — fit Kaggle P100 16GB với amp

---

## Project structure

```
cawot-cm-v0/
├── config.yaml                  # All knobs
├── src/
│   ├── data.py                  # TrainPoolDataset + build_pool (JSONL → images)
│   ├── embed.py                 # extract_image_embeddings (CLIP-B/16)
│   ├── cluster.py               # FAISS k-means spherical
│   ├── select.py                # select_random + select_v0 (farthest-from-centroid)
│   ├── train.py                 # train_on_dataset (CLIP last-4-layers + InfoNCE)
│   ├── eval.py                  # image-text retrieval R@k
│   └── utils.py
├── scripts/
│   ├── setup_data.py            # ★ Download + extract N shards từ HF
│   ├── run_v0.py                # ★ END-TO-END runner
│   └── make_dummy_data.py       # Pipeline sanity với dummy data (no HF needed)
└── notebooks/
    └── kaggle_v0.ipynb          # ★ Kaggle template
```

---

## Eval — tại sao không phải person-ID retrieval?

PAB Track 4 test set có person ID **bị mask cố ý** (competition track) → local không có ground truth. Thay vào đó V0 dùng:

**Image-text retrieval R@k trên held-out val split**:
- 2K (image, caption) pairs tách từ train pool
- Ground truth: `image[i]` matches `text[i]` (1-to-1 theo index)
- Compute cosine sim matrix → rank
- 6 numbers: `t2i_R@{1,5,10}`, `i2t_R@{1,5,10}`, plus `mean_R@1`

Đây là eval chuẩn trong CLIP/BLIP papers cho cross-modal alignment.

Real test set: sau khi V0 confirmed work, generate predictions trên `name-masked_test_set/gallery.zip` và submit lên ECCV leaderboard.

---

## Troubleshooting

| Vấn đề | Nguyên nhân + fix |
|---|---|
| `No shard folders found` | Chưa chạy `setup_data.py`, hoặc `--output` khác với `config.yaml`. |
| `Loaded 0 annotations` | `annotations_dir` không có `imgs_*.json`. Re-run setup_data.py. |
| Disk full khi extract | Giảm `--num-shards`. Mỗi shard ~3 GB sau extract. |
| Out of memory khi training | Giảm `train.batch_size` (96 → 64). |
| `mean_R@1` của V0 ≤ Random | Bug ở select.py hoặc K wrong scale. Sanity check K ≈ √(N/2). |
| Cả V0 và Random ~zeroshot | Training collapsed. Check LR (1e-5 default), check loss curve. |

---

## V0 → V1 → V2 roadmap

| Version | Thêm | Reuse từ V0 |
|---|---|---|
| **V0 (current)** | cluster + farthest-from-centroid trên CLIP embeddings | — |
| **V1** | + cross-modal cost (image+text+alignment, rank-normalized) + submodular facility location | Pipeline, eval, data loaders |
| **V2** | + friend's `qproxy/` (LLM queries + EVA02 embeddings + max_sim) + Wasserstein-aware budget | V1 method |

Friend's `qproxy/` folder vẫn còn nguyên trên Drive — sẽ dùng cho V2 sau khi V0/V1 confirmed.
