# EXPERIMENTS — protocol đạt chuẩn để chứng minh method

> Mục tiêu của file này: biến "đã chạy được" thành "**đã chứng minh được**" — một
> bảng kết quả mà reviewer AISTATS/NeurIPS không bắt bẻ được. Định nghĩa tiêu chí
> THẮNG **trước khi** chạy (tránh p-hacking), rồi để dữ liệu quyết định.

---

## 0. Method đã chốt (đừng mở lại)

**`proposed` = finite-proxy MEAN-RELEVANCE coreset** (η = 0).
`s_c = |C_c|^α · (1 + β·abar_c)`, abar = mean text-side MMD-relevance qua các proxy family.

- Bonus disagreement `u_c` (η>0) đã được xác nhận **vô dụng thực nghiệm**: `u`
  nhỏ hơn `abar` ~1000×, coreset full vs relevance-only trùng 99.98% **ngay cả
  với 6 family đa dạng**. → η bị bỏ khỏi method, chỉ giữ làm **ablation/negative
  result trung thực** (các proxy family đồng thuận trong không gian CLIP).

---

## 1. Regime đo (quan trọng nhất — đừng đo sai chỗ)

| Trục | Dùng gì | Vì sao |
|---|---|---|
| **Budget** | **2%, 5%, 10%, 20%** | Coreset "cắn" ở **budget thấp**; 20% là chỗ mọi method ≈ nhau (~88%). Headline nên ở 5%. |
| **Eval set** | **PAB real test (1.978 cặp)** = chính; synthetic val = phụ | Sim2Real chỉ chứng minh được trên test thật, không phải synthetic val (in-distribution, dễ). |
| **Pool** | 45K trước → 100K → 1M sau | Chứng minh ở 45K trước; scale sau khi thắng. |
| **Metric** | **t2i_R@1** (chính) tách khỏi i2t; + R@5/10, mAP | Bài toán = text query → tìm ảnh ⇒ **t2i** là metric of record. |
| **Subgroup** | overall + **wentwrong (anomaly)** | Anomaly là chỗ query-aware đáng thắng nhất. |

## 2. Baselines bắt buộc (so SELECTOR, không so model)

`random`, `clipscore`, `kcenter`, `sw_cawot` (poster V0–V2), **`proposed`**, và
**`full_data`** (đối thủ tham chiếu chính). Nếu kịp: `semdedup`, `ClipCov`/`VAS`
adapt sang retrieval. Cố định backbone + pipeline fine-tune, chỉ đổi chiến lược chọn.

## 3. Thống kê (đây là phần làm nên "chuyên nghiệp")

- **≥ 3 seeds** mỗi (method, budget) — 5 nếu kịp. Báo **mean ± std**.
- **Wilson 95% CI** cho R@1 (R@1 là Bernoulli/query; chỉ cần `R@1` + `n_pairs`).
- **Paired permutation test** + **Holm–Bonferroni** giữa `proposed` và từng baseline
  → cần **dump per-query hit vector** trong eval (thêm 1 mảng `t2i_hits` (0/1) vào
  metric JSON). Không có nó thì chỉ kết luận được bằng CI chồng/không chồng.
- Xuyên dataset (sau này): partial-conjunction (Dror et al.).

## 4. Quy trình chạy (đầu→cuối)

```
1) encode 6 proxy family  ->  scripts/embed_qproxy_families.py        (đã có)
2) chọn coreset mỗi (budget, seed):
       scripts/run_ablation.py --budget-frac 0.05 ... --out runs/b5_s0
   (lặp budget ∈ {0.02,0.05,0.10,0.20}, seed ∈ {0,1,2})
3) Stage C: fine-tune CLIP trên từng coreset_<method>.npy, eval trên REAL test
       -> ghi metric JSON tên chuẩn (xem §5)
4) gom bảng:  scripts/make_results_table.py --dir runs/stage_c --metric t2i_R@1
       -> results.md + results.csv (mean±std, Wilson CI, liệt kê cell thiếu)
```

Coreset cho mỗi selector đã được `run_ablation.py` lưu ở
`runs/<...>/selectors/coreset_<method>.npy` — Stage C chỉ cần load và fine-tune.

## 5. Naming convention cho metric JSON (để `make_results_table.py` parse đúng)

```
<method>_b<budget>_s<seed>_metrics.json
ví dụ:  proposed_b5_s0_metrics.json   random_b5_s1_metrics.json
        full_data_b5_s2_metrics.json  proposed_b20_s0_metrics.json
```
Schema mỗi file (đúng cái Stage C của bạn đang xuất):
```json
{ "overall":   {"n_pairs":1978,"t2i_R@1":..,"i2t_R@1":..,"mean_R@1":..,
                "t2i_R@5":..,"i2t_R@5":..,"t2i_R@10":..,"i2t_R@10":..,"mAP":..},
  "wentwrong": {"n_pairs":989, ...} }
```
(Khuyến nghị thêm `"t2i_hits":[0,1,...]` để chạy được paired test.)

## 6. Tiêu chí THẮNG — định nghĩa trước khi nhìn số

- **Primary (mạnh nhất):** ở **budget 5%, real test, t2i_R@1**, `proposed` ≥ **mọi**
  selector baseline, CI **không chồng** (hoặc paired test p<0.05 sau Holm), trên ≥3 seeds.
- **Secondary (efficiency, fallback an toàn):** `proposed` (hoặc selector tốt nhất)
  đạt **≈ full_data** ở budget thấp ("đạt full-data với 1/20 dữ liệu") — vẫn publishable.
- **Negative trung thực:** nếu `proposed ≈ random` kể cả ở 5%/real-test → **báo cáo
  thẳng**, chuyển trọng tâm đóng góp sang **lý thuyết** (concentration/stability của
  query-aware scoring) hoặc benchmark khác. Không cố reframe method vô tận.

## 7. Nếu primary trượt — chỉ khi đó mới chỉnh method (theo thứ tự)

Chẩn đoán đã có từ allocation: **size lấn relevance** (`budget_vs_size_corr=0.975`
> `budget_vs_relevance_corr=0.88`) ⇒ coreset ≈ size-proportional ≈ stratified random.
Cách chỉnh, rẻ → đắt:
1. **Sweep α** (`--alpha 0.5 / 0.25 / 0`): α nhỏ ⇒ relevance lấn size. Rẻ, thử trước.
2. Relevance **nhân thay vì cộng** / sharpen `τ` ⇒ dồn budget vào vùng query-relevant
   mà random under-sample.
3. Đổi within-cluster từ **representative (herding)** sang **informative/hard** —
   vì distribution-matching ≈ random; muốn thắng random phải lệch có chủ đích.

Tất cả chỉ làm **sau** khi §6 primary trượt trên 5%/real-test — không làm trước.

---

## 8. Checklist "đã chuyên nghiệp / chứng minh được" ✅

- [ ] Eval trên **PAB real test 1.978** (không chỉ synthetic val).
- [ ] Budget **5%** (+ 2/10/20) — không chỉ 20%.
- [ ] Đủ baselines **+ full_data**.
- [ ] **≥3 seeds**, mean±std, **Wilson CI** cho R@1.
- [ ] **t2i tách i2t**; báo cả subgroup anomaly.
- [ ] Bảng sinh bằng `make_results_table.py` (tự liệt kê cell thiếu → không claim trên dữ liệu khuyết).
- [ ] η-inert ghi ở mục ablation, không phải main claim.
