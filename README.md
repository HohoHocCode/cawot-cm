# CAWOT-CM — Coreset Selection cho Text-based Person Anomaly Retrieval (V0 + V1 + V2)

Codebase cho phần **coreset selection** của paper CAWOT-CM, chạy trên benchmark **PAB ECCV'26 Workshop Track 4** (Pedestrian Anomaly Behavior). Mục tiêu: chọn một tập con (coreset) nhỏ từ pool ảnh-caption synthetic sao cho fine-tune trên coreset đó cho kết quả retrieval tốt nhất với cùng một ngân sách dữ liệu (data budget).

README này đủ chi tiết để viết phần **Method + Experiments** của báo cáo. Mọi định nghĩa, công thức, protocol, và cách đọc số đều ở đây.

---

## 1. Bài toán & vì sao coreset

- **Task**: text-based person retrieval. Cho 1 query text mô tả người + hành vi, retrieve ảnh đúng trong gallery.
- **Dữ liệu train**: ~1M cặp (image, caption) synthetic (diffusion-generated). Train trên toàn bộ 1M tốn compute.
- **Câu hỏi coreset**: nếu chỉ được train trên `B%` dữ liệu, chọn cặp nào để model generalize tốt nhất? Coreset tốt → đạt gần full-data performance với ít compute hơn nhiều ("efficiency story").
- **So sánh cốt lõi**: method chọn coreset thông minh (V0/V1) vs **Random** (bốc ngẫu nhiên cùng số lượng).

---

## 2. Các phương pháp selection

Tất cả đều: **cluster pool trên image-embedding bằng FAISS k-means (spherical)** → **chia budget cho mỗi cluster tỉ lệ với size** (proportional) → **chọn trong mỗi cluster theo method**. Khác nhau ở bước chọn-trong-cluster.

Ký hiệu: trong 1 cluster có các sample, `z_v` = image embedding (CLIP, L2-normalized), `z_t` = text embedding, centroid = tâm cluster.

### Random (baseline)
Bốc ngẫu nhiên `B` sample đều trên toàn pool. Không cluster. Đây là mốc để mọi method phải vượt qua.

### V0 — farthest-from-centroid (plan V0, "diversity-only")
Trong mỗi cluster, chọn các sample **xa centroid nhất** (cosine distance lớn nhất):
```
d_i = 1 − cos(z_v_i, centroid_k)
chọn top-B_k theo d_i giảm dần
```
Ý tưởng ban đầu: "đa dạng". Thực chất: chọn **điểm biên/atypical** của cluster.

### V0-proto — closest-to-centroid (diagnostic)
Ngược V0: chọn các sample **gần centroid nhất** = **điểm điển hình/đại diện (prototype)**. Dùng để kiểm chứng giả thuyết Sorscher (xem §6).

### V1 — cross-modal cost + facility location (plan V1)
Trong mỗi cluster, thay vì dựa vào khoảng cách tới centroid, chọn tập **phủ (cover)** cluster trong không gian **đa phương thức (image + text + alignment)** bằng tối ưu submodular **facility location**.

**Cross-modal cost** giữa 2 sample i, j (theo plan):
```
c_ij = ( rank(d_v) + rank(d_t) + rank(d_a) ) / 3
  d_v = 1 − cos(z_v_i, z_v_j)              # khoảng cách ảnh
  d_t = 1 − cos(z_t_i, z_t_j)              # khoảng cách text
  d_a = | a_i − a_j |,  a_i = cos(z_v_i, z_t_i)   # chênh lệch "độ khớp ảnh-caption"
rank(·) = rank-normalize toàn bộ entry của ma trận khoảng cách về [0,1]
```
Similarity kernel cho facility location: `s_ij = 1 − c_ij` (đường chéo set = 1).

**Facility location** chọn tập S tối đa hóa:
```
f(S) = Σ_i  max_{s ∈ S}  s_{i,s}
```
tức S "đại diện" tốt cho toàn cluster. Giải bằng **greedy** (đảm bảo xấp xỉ `1 − 1/e`). Implement bằng numpy thuần trong [src/select.py](src/select.py) (`facility_location_greedy`) — vì ở quy mô per-cluster (≤ ~1000 điểm) greedy vectorized là đủ nhanh và minh bạch. Khi scale lên full-1M có thể thay bằng submodlib (C++ LazyGreedy) với cùng kernel.

### V2 — Wasserstein-aware budget + cross-modal facility location (plan V2, **the method**)
V2 giữ y nguyên cách chọn TRONG cluster của V1, nhưng đổi cách chia ngân sách GIỮA cluster.

**Q_proxy**: tập 2,593 LLM-generated text queries (từ folder Drive của teammate). Mỗi caption được **re-encode bằng CLIP-B/16 text encoder** để cùng không gian với text embeddings của pool (vì friend's `text_embeddings.npy` đang ở EVA02-1024d, không match).

**Wasserstein gap mỗi cluster**: với cluster k có text embeddings `T_k`, tính khoảng cách Wasserstein-2 tới Q_proxy:
```
W_k = W2( T_k , Q_proxy )
```
Implement: **Gaussian fit + diagonal covariance** closed-form (xem `gaussian_w2_diag` trong [src/select.py](src/select.py)):
```
W2² ≈ ||μ_T − μ_Q||²  +  Σ_d (σ_{T,d} − σ_{Q,d})²
```
Lý do dùng diagonal thay vì full covariance:
- Full Gaussian W2 cần `Σ^{1/2}` (O(d³)); với d=512, K=150 sẽ chậm.
- Khi cluster size < d (rank-deficient), full Σ không invertible; diagonal ổn.
- GORACS (KDD'25) + FDMat (AAAI'24) cũng dùng closed-form xấp xỉ tương tự.

**Budget allocation**:
```
B_k  ∝  n_k^α  ×  (1 + W_k / W_avg)
```
với α = 0.5 (plan default, `coreset.v2_alpha`):
- `n_k^α` sub-linear: tránh cluster lớn "ăn" hết budget.
- `(1 + W_k/W_avg)` boost cluster xa Q_proxy: chưa được cover bởi loại query test-like → cần thêm sample.

Largest-remainder rounding, capped at cluster size. Final coreset same selection logic V1 (facility location on cross-modal cost) within each cluster.

### Tóm tắt method matrix

| Component | random | v0 | v0_proto | v1 | **v2** |
|---|---|---|---|---|---|
| Cluster (image) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Budget ∝ size | ✓ | ✓ | ✓ | ✓ | ✗ |
| Budget Wasserstein-aware | ✗ | ✗ | ✗ | ✗ | **✓** |
| Dùng text embedding | ✗ | ✗ | ✗ | ✓ | ✓ |
| Q_proxy | ✗ | ✗ | ✗ | ✗ | **✓** |
| Chọn trong cluster | — (toàn cục) | xa centroid | gần centroid | facility location | facility location |

---

## 3. Dữ liệu (single source: HuggingFace)

[`TruongVox/Cawot-dataset`](https://huggingface.co/datasets/TruongVox/Cawot-dataset):
```
├── imgs_0.zip ... imgs_74.zip       # 75 shards × ~1.4 GB (ảnh, ~13K/shard)
└── train/imgs_0.json ... imgs_74.json   # JSONL annotations, 1 entry/dòng
```
JSONL schema mỗi dòng:
```json
{"image": "train/imgs_0/goal/0.jpg", "caption": "...", "image_id": "0_0", "scene": "...", "normal": "..."}
```
`scripts/setup_data.py` tải N shards, giải nén thành `pab_data/images/imgs_N/...` + `pab_data/annotations/imgs_N.json`. Loader tự xử lý `.jpg`/`.webp`.

Cho V0/V1 sanity + sweep: **5 shards (~65K cặp)** là đủ. Full-1M chỉ cần cho số liệu cuối paper.

---

## 4. Protocol thực nghiệm (để mô tả trong báo cáo)

1. **Pool**: random-sample `sample_size = 50,000` cặp từ các shard đã tải (seed cố định).
2. **Embeddings**: trích CLIP ViT-B/16 (OpenAI) image + text embedding (512-d, L2-normalized). Cache lại.
3. **Val set (FIXED)**: hold-out `val_size = 5,000` cặp làm tập eval retrieval. **Giữ nguyên qua mọi method/budget/seed** → số liệu so sánh trực tiếp được. → train pool = 45,000.
4. **Sweep đầy đủ** (plan-faithful, V0+V1+V2):
   - **Methods**: random, v0, v0_proto, v1, **v2** (5) — V2 là method chính paper
   - **Budgets**: 5%, 10%, 20%, 30%, 40%, 50% (6) — full scaling-law range của plan Part B2
   - **Seeds**: 42, 1, 2 (3) — error bars (plan Tuần 4-5)
   - Tổng: **90 fine-tune runs** với error bars chuẩn
5. **Cluster**: cho mỗi seed, k-means spherical với `k=150` trên train_zv.
6. **Fine-tune**: unfreeze 4 transformer block cuối (image + text) + projection, InfoNCE đối xứng, 3 epoch, lr 1e-5, batch 96, AMP. (~28% params trainable.)
7. **Eval**: image-text retrieval R@1/5/10 hai chiều trên val 5K (xem §5), **kèm per-category breakdown** (goal / full / wentwrong).
8. **Resumability**: `run_sweep.py` đọc `records.csv` lúc khởi động, skip combo đã làm → an toàn khi Kaggle session timeout, multi-session free.

**Selection model = task model = CLIP ViT-B/16** ở V0/V1 (V2 tương lai có thể dùng EVA02 cho selection).

---

## 5. Eval metric (quan trọng — giải thích trong báo cáo)

PAB Track 4 test set có **person id bị mask** (competition) → không có ground-truth để eval local. Thay bằng **image-text retrieval trên val hold-out**:
- 5,000 cặp (image, caption); với mỗi caption i, ground-truth là image i (1-to-1 theo index).
- Tính ma trận cosine similarity (5000×5000), với mỗi query đếm rank của cặp đúng.
- Báo cáo: `t2i_R@{1,5,10}` (text→image), `i2t_R@{1,5,10}` (image→text), và `mean_R@1` = trung bình R@1 hai chiều.

Đây là eval chuẩn cho cross-modal alignment (CLIP/BLIP/ALBEF). Gallery 5,000 (không 2,000) để tránh bão hòa.

### Per-category breakdown — quan trọng cho rigor

Train data PAB có 3 loại theo cấu trúc folder của image path (`train/imgs_N/<category>/X.jpg`):

| Category | Tỉ lệ trên train | Semantic |
|---|---|---|
| `goal` | 37% | Hành động được kỳ vọng (normal-intent) |
| `full` | 34% | Chuỗi đầy đủ của hành động (normal-ish) |
| `wentwrong` | **29%** | **Hành động bị sai → ANOMALY** |

Pipeline tự parse `category` từ image path. Val 5K (random từ train) → có ~3550 normal-ish (goal+full) + ~1450 anomaly (wentwrong).

Mỗi run, eval report **R@k riêng cho từng category** (queries filter theo category, gallery vẫn là full 5K → realistic retrieval setting). Đây là cách kiểm chứng method có giúp tìm **anomaly** không, không chỉ overall.

**Caveat ghi trong báo cáo**: vẫn là proxy retrieval nội bộ synthetic↔synthetic exact-pair, không phải Sim2Real real-gallery retrieval. Số liệu official cho leaderboard ECCV cần submit prediction trên `name-masked_test_set/gallery.zip` sau khi method ổn định. Nhưng per-category breakdown đã cho biết method có generalize sang anomaly hay không.

---

## 6. Kết quả đã có & cách diễn giải

### Sweep V0 family + V1 — overall R@1, 5K val, **1 seed (chưa có error bar)**

```
budget |  random  |  v0 (farthest)  |  v0_proto (closest)  |  v1 (facility loc)
  5%   |  79.94   |     75.23       |       82.05          |      80.65
 10%   |  84.65   |     80.81       |       87.08          |      85.52
 20%   |  88.93   |     86.50       |       89.84          |      88.04
 40%   |  90.87   |     90.25       |       91.99          |      90.56
zero-shot mean_R@1 = 59.03
```

**Diễn giải (dùng cho báo cáo)**:

1. **Trục prototypicality ↔ atypicality quyết định kết quả**:
   - `v0` (farthest = atypical) → tệ nhất, thua Random mọi budget (max −4.71 @ 5%)
   - `v1` (facility location coverage) → middle, thắng Random ở 5/10% (+0.7/+0.9), thua ở 20/40%
   - `v0_proto` (closest = prototypical) → **tốt nhất**, thắng Random mọi budget (+0.9 đến +2.4)
2. **Khớp Sorscher et al., NeurIPS 2022**: ở budget nhỏ, sample điển hình thắng. v0_proto thắng vì chọn đúng sample điển hình; v1 facility location bị kéo về phía biên do objective coverage → kém v0_proto.
3. **Caveat quan trọng**: val random hold-out trộn 3 category (goal/full/wentwrong). v0_proto thắng OVERALL — chưa biết có thắng anomaly (wentwrong) không. Đây là lý do protocol mới (§4-5) thêm **per-category eval** + **3 seeds**.

**Kết quả overall trên là 1 seed (seed=42). Sweep đầy đủ 3 seeds × 6 budgets × 4 methods × 4 categories đang trong protocol mới — sẽ thay thế bảng này khi có.**

---

## 7. Cách chạy

### Kaggle (khuyến nghị)
1. New notebook → GPU **P100** + Internet **ON**.
2. Copy [notebooks/kaggle_v0.ipynb](notebooks/kaggle_v0.ipynb) → **Run All**.
3. Notebook tự: clone repo → tải 5 shards từ HF → sanity check → chạy `scripts/run_sweep.py` → in bảng + vẽ đường cong.
4. Thời gian: ~3-3.5 h (4 method × 4 budget × 1 seed = 16 fine-tune runs).

### Local / ThunderCompute
```bash
pip install -r requirements.txt
python scripts/setup_data.py --output ./pab_data --num-shards 5
# sửa config.yaml: data.image_root, data.annotations_dir
python scripts/run_sweep.py --config config.yaml
```

### Output
- `outputs/eval/summary.json` — per (method, budget): `mean_R@1_mean`, `mean_R@1_std`, `n_seeds`
- `outputs/eval/records.csv` — 1 dòng mỗi (method, budget, seed) với đầy đủ t2i/i2t R@k
- `outputs/eval/sweep_curve.png` — đường cong R@1 vs budget

---

## 8. Số lần run & multi-session (full-rigor protocol)

**Cấu hình mặc định = full rigor (V0+V1+V2)**:
- 5 methods × 6 budgets × 3 seeds = **90 fine-tune runs**
- + 1 zero-shot eval (anchor) + Q_proxy text encoding (~10 sec)
- Total compute: ~20-21 h trên Kaggle P100

**Multi-session — không lo timeout**: `run_sweep.py` resumable qua `records.csv`. Chỉ cần Kaggle quota tuần ≥ 17h (Kaggle cho 30h GPU/tuần).

Có 2 cách chia session:

| Strategy | Cách làm | Pros/Cons |
|---|---|---|
| **A. Tự động** | Giữ `train.seeds: [42, 1, 2]`. Mỗi session restart cell "Run sweep" → tiếp tục từ chỗ dừng. | Đơn giản, không cần đụng config. |
| **B. Thủ công** | Session 1: `seeds: [42]`. Session 2: `seeds: [1]`. Session 3: `seeds: [2]`. | Kiểm soát chính xác mỗi session làm gì. |

Output sau khi xong: `records.csv` chứa **4 method × 6 budget × 3 seed × 4 category = 288 rows** (cộng zero-shot rows). `summary.json` aggregate theo mean ± std. Quá đủ rigor cho cả workshop lẫn conference.

---

## 9. Config reference ([config.yaml](config.yaml))

```yaml
data.sample_size: 50000      # pool size lấy từ shard đã tải
data.val_size: 5000          # gallery hold-out (lớn → tránh bão hòa)
cluster.k: 150               # ~ sqrt(N/2) cho 45K pool
coreset.budgets: [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]  # full scaling-law range
coreset.methods: [random, v0, v0_proto, v1, v2]   # v2 = full plan method
coreset.v2_alpha: 0.5         # exponent in B_k ∝ n_k^α × (1+W_k/W_avg)
qproxy.queries_json_path: ".../queries.json"   # downloaded by setup_qproxy.py
qproxy.cache_path: ".../qproxy_clip_text_emb.npy"
train.seeds: [42, 1, 2]      # error bars (plan-faithful)
train.num_epochs: 3
train.batch_size: 96         # fit P100 16GB + AMP
train.keep_checkpoints: false  # xóa ckpt sau eval (72 ckpts × 600MB không fit Kaggle disk)
```

---

## 10. Cấu trúc project

```
cawot-cm-v0/
├── config.yaml                  # mọi hyperparameter
├── src/
│   ├── data.py                  # TrainPoolDataset + build_pool (JSONL → ảnh)
│   ├── embed.py                 # extract image / image+text CLIP embeddings
│   ├── cluster.py               # FAISS k-means spherical
│   ├── select.py                # random / v0 / v0_proto / v1 / v2 (+ helpers)
│   ├── qproxy.py                # Q_proxy loading + CLIP text re-encoding (V2)
│   ├── train.py                 # train_on_dataset (CLIP last-4-layer + InfoNCE)
│   ├── eval.py                  # image-text retrieval R@k + per-category split
│   └── utils.py
├── scripts/
│   ├── setup_data.py            # ★ tải + giải nén N shards từ HF
│   ├── setup_qproxy.py          # ★ tải Q_proxy queries.json từ Drive (V2 only)
│   └── run_sweep.py             # ★ END-TO-END: V0 family + V1 + V2
└── notebooks/
    └── kaggle_v0.ipynb          # ★ Kaggle template (clone → tải → sweep → plot)
```

---

## 11. Troubleshooting

| Vấn đề | Fix |
|---|---|
| `No shard folders found` | chưa chạy setup_data.py / sai `--output` vs config |
| `Loaded 0 annotations` | `annotations_dir` thiếu `imgs_*.json` |
| OOM khi train | giảm `train.batch_size` (96 → 64) |
| OOM/chậm khi V1 select | cluster quá to; tăng `cluster.k` để cluster nhỏ lại (facility location là O(n²)/cluster) |
| Disk đầy | giảm `--num-shards` |
| R@1 ~99% (bão hòa) | tăng `data.val_size` |
| V1 ≤ Random mọi budget | báo lại — cần xem lại cost weighting / cluster granularity |

---

## 12. Roadmap

| Version | Thêm | Trạng thái |
|---|---|---|
| V0 | farthest-from-centroid | ✅ chạy xong (negative result, motivate V1) |
| V0-proto | closest-to-centroid | ✅ chạy xong (Sorscher-aligned, prototype thắng) |
| V1 | cross-modal cost + facility location | ✅ chạy xong (mixed result vs v0_proto overall) |
| **V2** | + Q_proxy + Wasserstein-aware budget allocation | ✅ **code xong, cần chạy full sweep** |

### Tốc độ training & lý do không dùng unsloth

**Unsloth không support CLIP ViT-B/16** (chỉ support LLM/VLM như Llama/Qwen/LLaVA — custom Triton kernels cho attention không apply cho CLIP). Để fair so sánh V0/V1/V2, cả 3 đều dùng cùng setup: last-4-layer fine-tune + InfoNCE + AMP + batch 96. Không đổi giữa các version.

**Tùy chọn tăng tốc (future work, không bắt buộc cho paper hiện tại)**:
- LoRA via PEFT (r=16): giảm VRAM ~40%, tốc độ +20-40%, nhưng yêu cầu re-run **TOÀN BỘ** V0/V1/V2 với cùng setup mới để giữ ablation fair.
- `torch.compile` (PyTorch 2.x): +30-50% throughput, compile overhead ~1 min/model. Drop-in nếu cần.
- Đổi backbone sang VLM (LLaVA, Qwen2-VL) để dùng unsloth: major design change, không khuyến nghị cho workshop paper.
