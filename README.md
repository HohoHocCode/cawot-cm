# CAWOT-CM V0: Diversity-Only Coreset Baseline

V0 baseline cho CAWOT-CM paper. Sanity check rằng cluster + farthest-from-centroid selection beats random sampling cho text-based person retrieval trên **PAB ECCV'26 Workshop Track 4**.

## V0 spec

V0 = **clean baseline, hoàn toàn độc lập với V1/V2 prep work**.

```
Random sample N images từ PAB train pool
  → Extract CLIP ViT-B/16 image embeddings
  → Hold out V (image, caption) pairs cho val
  → FAISS k-means (K, spherical) trên (N − V) train pool
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

V0 self-contained: chỉ cần train images + JSONL annotations.

---

## Friend cần làm gì để chạy

### Bước 1: Kaggle setup (1 lần)

1. Tạo Kaggle notebook mới (hoặc copy template `notebooks/kaggle_v0.ipynb` của repo)
2. **Settings → Accelerator**: GPU P100
3. **Settings → Internet**: ON (cần để download annotations từ HuggingFace)
4. **Add Data**: thêm dataset `vnhtbo/pab-eccv26-track4-train-webp-part-01-05` (cái friend đã upload trước đây)

### Bước 2: Open notebook và Run all

Mở `notebooks/kaggle_v0.ipynb` từ repo này. Notebook làm tất cả:

1. Clone repo
2. Install deps (`open_clip_torch`, `faiss-gpu-cu12`, ...)
3. Download annotations (75 JSONL files, ~390MB) từ HuggingFace `TruongVox/Cawot-dataset/train/`
4. Patch config với đường dẫn Kaggle
5. Sanity check resolve 1 image
6. Chạy `scripts/run_v0.py` end-to-end (~50-70 phút trên P100)
7. Show summary table

### Bước 3: Xem kết quả

Output ở `/kaggle/working/outputs/eval/summary.json`:

```json
{
  "zeroshot": {"t2i_R@1": ..., "i2t_R@1": ..., "mean_R@1": ...},
  "random":   {"t2i_R@1": ..., "i2t_R@1": ..., "mean_R@1": ...},
  "v0":       {"t2i_R@1": ..., "i2t_R@1": ..., "mean_R@1": ...}
}
```

**Mong đợi**: `v0.mean_R@1 > random.mean_R@1 > zeroshot.mean_R@1`. Gap V0-Random ~0.5-1.5%.

Friend nên save Kaggle notebook version (Save & Run All) — sẽ persistent toàn bộ `outputs/` để mình review checkpoint + numbers sau.

---

## Required inputs

V0 cần 2 thứ trên máy chạy:

| Input | Mô tả | Cách lấy |
|---|---|---|
| **Train images** (~50GB unzipped) | Webp images trong `Part 1/imgs_N/imgs_N/<action>/*.webp` | Friend's Kaggle dataset `vnhtbo/pab-eccv26-track4-train-webp-part-01-05` (Parts 1-5) |
| **Train annotations** (~390MB) | 75 JSONL files (`imgs_0.json` ... `imgs_74.json`) | Notebook auto-download từ HF `TruongVox/Cawot-dataset/train/` |

Annotation JSONL schema (1 entry / dòng):
```json
{"image": "train/imgs_0/goal/0.jpg", "caption": "...", "image_id": "0_0", "scene": "...", "normal": "..."}
```

Code tự rewrite `.jpg` extension → `.webp` và Part-N structure khi resolve image trên disk.

---

## Config knobs

`config.yaml`:
- `data.sample_size: 50000` — random sample size. Tăng nếu có thời gian (full pool ~500K trên Parts 1-5).
- `data.val_size: 2000` — held-out pairs cho R@k eval.
- `cluster.k: 150` — rule of thumb √(N/2). Tăng nếu sample_size lớn hơn.
- `coreset.budget_ratio: 0.20` — coreset size = 20% của train pool sau val split.
- `train.num_epochs: 3` — đủ cho V0 sanity. Tăng nếu loss chưa converged.
- `train.batch_size: 96` — fit Kaggle P100 16GB với amp.

---

## Project structure

```
cawot-cm-v0/
├── config.yaml                  # All knobs in one file
├── src/
│   ├── data.py                  # ★ TrainPoolDataset + build_pool (JSONL annotations → images)
│   ├── embed.py                 # extract_image_embeddings (CLIP-B/16)
│   ├── cluster.py               # FAISS k-means spherical
│   ├── select.py                # select_random + select_v0 (farthest-from-centroid)
│   ├── train.py                 # train_on_dataset (CLIP last-4-layers + InfoNCE)
│   ├── eval.py                  # image-text retrieval R@k on val split
│   └── utils.py
├── scripts/
│   ├── run_v0.py                # ★ END-TO-END runner — primary entry point
│   ├── make_dummy_data.py       # Pipeline sanity với data giả (không cần PAB)
│   └── 01-04 + others           # Legacy (dummy-data path), không dùng cho V0 chính
└── notebooks/
    └── kaggle_v0.ipynb          # ★ Kaggle template
```

`★` = file friend cần biết. Còn lại là internals.

---

## Eval setup — tại sao không phải retrieval với person ID?

PAB Track 4 test set có person ID **bị mask cố ý** (competition track) → local không có ground truth. Thay vào đó V0 dùng:

**Image-text retrieval R@k trên held-out val split**:
- 2K (image, caption) pairs tách từ train pool
- Ground truth: `image[i]` matches `text[i]` (1-to-1 theo index)
- Compute cosine sim matrix → rank
- 6 numbers: `t2i_R@{1,5,10}`, `i2t_R@{1,5,10}`, plus `mean_R@1`

Đây là eval chuẩn trong CLIP/BLIP papers cho cross-modal alignment. Đủ rigorous để confirm V0 > Random.

Khi nào dùng real test set: sau khi V0 work, generate predictions trên `name-masked_test_set/gallery.zip` của Track 4 và submit lên ECCV leaderboard.

---

## Troubleshooting

| Vấn đề | Nguyên nhân + fix |
|---|---|
| `No shard folders found under <image_root>` | Kaggle image dataset chưa được add, hoặc slug khác. Edit `IMAGE_ROOT` ở cell 5 của notebook. |
| `Loaded 0 annotations` | Annotations dir trống. Re-run cell 4 (HF download). |
| Out of memory khi training | Giảm `train.batch_size` (96 → 64). |
| `mean_R@1` của V0 ≤ Random | Bug ở select.py hoặc K wrong scale. Sanity check K = √(N/2). |
| Cả V0 và Random ~zeroshot | Training collapsed. Check LR (1e-5 default), check loss curve trong stdout. |

---

## V0 → V1 → V2 roadmap

V0 ở đây là sanity baseline. Khi V0 confirmed work:

| Version | Thêm | Reuse từ V0 |
|---|---|---|
| **V0 (current)** | cluster + farthest-from-centroid trên CLIP embeddings | — |
| **V1** | + cross-modal cost (image+text+alignment, rank-normalized) + submodular facility location | Pipeline, eval, data loaders |
| **V2** | + friend's `qproxy/` (queries.json + EVA02 embeddings + max_sim) + Wasserstein-aware budget | + V1 method |

Friend's `qproxy/` folder vẫn còn nguyên — sẽ dùng cho V2 sau khi V0 + V1 confirmed.
