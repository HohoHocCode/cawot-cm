# CAWOT-CM — Coreset Selection cho Text-based Person Anomaly Retrieval (V0 + V1)

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

### V1 — cross-modal cost + facility location (plan V1, **the method**)
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

| Component | random | v0 | v0_proto | v1 |
|---|---|---|---|---|
| Cluster (image) + proportional budget | ✓ | ✓ | ✓ | ✓ |
| Dùng text embedding | ✗ | ✗ | ✗ | ✓ |
| Chọn trong cluster | — (toàn cục) | xa centroid | gần centroid | facility location (cross-modal) |
| Q_proxy / Wasserstein budget | ✗ | ✗ | ✗ | ✗ (để V2) |

**V1 KHÔNG có**: Q_proxy, Wasserstein-aware budget — đó là V2.

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
4. **Sweep**: cho mỗi `seed`, cluster pool (k-means, `k=150`). Cho mỗi `budget ∈ {5,10,20,40}%`, mỗi method ∈ {random, v0, v0_proto, v1}: chọn coreset → fine-tune CLIP-B/16 → eval.
5. **Fine-tune**: unfreeze 4 transformer block cuối (image + text) + projection, InfoNCE đối xứng, 3 epoch, lr 1e-5, batch 96, AMP. (~28% params trainable.)
6. **Eval**: image-text retrieval R@1/5/10 hai chiều trên val 5K (xem §5).

**Selection model = task model = CLIP ViT-B/16** ở V0/V1 (khác V2 tương lai có thể dùng EVA02 cho selection).

---

## 5. Eval metric (quan trọng — giải thích trong báo cáo)

PAB Track 4 test set có **person id bị mask** (competition) → không có ground-truth để eval local. Thay bằng **image-text retrieval trên val hold-out**:
- 5,000 cặp (image, caption); với mỗi caption i, ground-truth là image i (1-to-1 theo index).
- Tính ma trận cosine similarity (5000×5000), với mỗi query đếm rank của cặp đúng.
- Báo cáo: `t2i_R@{1,5,10}` (text→image), `i2t_R@{1,5,10}` (image→text), và `mean_R@1` = trung bình R@1 hai chiều.

Đây là eval chuẩn cho cross-modal alignment (CLIP/BLIP/ALBEF). **Gallery 5,000** (không phải 2,000) để tránh bão hòa — với 2,000 mọi method đều ~94% R@1, không phân biệt được.

**Caveat để ghi trong báo cáo**: đây là proxy retrieval nội bộ (synthetic→synthetic, exact-pair), không phải full Sim2Real person-id retrieval. Số liệu chính thức cho leaderboard ECCV cần submit prediction trên `name-masked_test_set/gallery.zip` (làm sau khi method ổn định).

---

## 6. Kết quả đã có & cách diễn giải

### V0 (đã chạy, 5K gallery, 1 seed) — **clean negative result**

```
budget |  random | v0 (farthest) | Δ(v0 − random)
  5%   |  79.94  |    75.23      |   −4.71
 10%   |  84.25  |    81.29      |   −2.96
 20%   |  88.23  |    86.44      |   −1.79
 40%   |  91.13  |    89.93      |   −1.20
zero-shot mean_R@1 = 59.03
```

**Diễn giải (dùng cho báo cáo)**:
- V0 (farthest-from-centroid) **thua Random ở mọi budget**, và thua **nhiều nhất ở budget thấp** (−4.71 ở 5%), thu hẹp khi budget tăng.
- Cơ chế: farthest-from-centroid = chọn **outlier/atypical**. Ở budget thấp, coreset toàn sample cực biên → kém đại diện → model generalize kém trên val điển hình. Budget tăng → buộc lấy thêm sample trung tâm → hội tụ về Random.
- **Khớp Sorscher et al., NeurIPS 2022** ("Beyond Neural Scaling Laws"): ở budget nhỏ nên giữ sample **prototypical**, chỉ budget lớn mới nên giữ sample hard/atypical. V0 làm ngược → đúng dấu hiệu quan sát được.
- → Đây là **negative result sạch, có lý thuyết**, và là **động lực trực tiếp cho V1** (chọn sample đại diện thay vì biên).

### V0-proto & V1 (sẽ có sau khi chạy sweep) — kỳ vọng
- `v0_proto > random` ở budget thấp (xác nhận Sorscher từ chiều ngược lại).
- `v1 ≳ v0_proto > random > v0` ở budget thấp; mọi method hội tụ ở 40%.
- Headline: **V1 vượt Random nhiều nhất ở 5%** = "cross-modal representative coverage giúp khi data khan hiếm".

Nếu V1 **không** vượt Random → báo lại; có thể cần điều chỉnh cost weighting hoặc cluster granularity.

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

## 8. Số lần run & multi-seed (theo plan)

- **Plan** (Tuần 4-5): headline budget 20% chạy **3 seeds**; budget khác 1 seed; scaling curve 5-50%.
- **Mặc định hiện tại**: `train.seeds: [42]` (1 seed) để confirm V1 trend nhanh (~3.5 h).
- **Để có error bar đúng plan**: đổi `train.seeds: [42, 1, 2]`. Lưu ý 3 seeds × 16 runs ≈ 8-9 h > giới hạn 1 session Kaggle (12h nhưng rủi ro). **Khuyến nghị**: hoặc (a) chạy full sweep 1 seed trước rồi chạy riêng 3 seeds chỉ ở 20%; hoặc (b) tách 3 session Kaggle, mỗi session 1 seed (cache embeddings giúp seed sau nhanh hơn vì không extract lại).

Báo cáo workshop: 1 seed full sweep + 3 seeds ở 20% headline là đủ thuyết phục. Conference: 3 seeds toàn bộ.

---

## 9. Config reference ([config.yaml](config.yaml))

```yaml
data.sample_size: 50000      # pool size lấy từ shard đã tải
data.val_size: 5000          # gallery hold-out (lớn → tránh bão hòa)
cluster.k: 150               # ~ sqrt(N/2) cho 45K pool
coreset.budgets: [0.05, 0.10, 0.20, 0.40]
coreset.methods: [random, v0, v0_proto, v1]
train.seeds: [42]            # → [42,1,2] cho error bar
train.num_epochs: 3
train.batch_size: 96         # fit P100 16GB + AMP
train.keep_checkpoints: false  # xóa ckpt sau eval (tiết kiệm disk)
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
│   ├── select.py                # random / v0 / v0_proto / v1 (+ facility_location_greedy, cross_modal_similarity)
│   ├── train.py                 # train_on_dataset (CLIP last-4-layer + InfoNCE)
│   ├── eval.py                  # image-text retrieval R@k
│   └── utils.py
├── scripts/
│   ├── setup_data.py            # ★ tải + giải nén N shards từ HF
│   └── run_sweep.py             # ★ END-TO-END: V0 family + V1, mọi budget/seed
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

## 12. Roadmap → V2

| Version | Thêm | Trạng thái |
|---|---|---|
| V0 | farthest-from-centroid | ✅ chạy xong (negative result, motivate V1) |
| V0-proto | closest-to-centroid | ✅ code xong, chạy trong sweep |
| **V1** | cross-modal cost + facility location | ✅ code xong, **cần chạy sweep** |
| V2 | + Q_proxy (LLM queries, EVA02 embeddings) + Wasserstein-aware budget allocation | ⏳ folder `qproxy/` của teammate đã sẵn sàng |

V2 sẽ tái dùng toàn bộ harness này (data, eval, sweep), chỉ thêm bước budget allocation theo Wasserstein gap tới Q_proxy.
