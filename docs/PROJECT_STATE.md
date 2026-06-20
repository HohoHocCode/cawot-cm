# CAWOT-CM — Bản đồ trạng thái dự án (State of the Project)

> Một file duy nhất để không lạc giữa lý thuyết và thực nghiệm. Cập nhật khi có
> kết quả mới. Đọc TL;DR trước, rồi nhảy tới phần cần.

---

## 0. TL;DR — bạn đang ở đâu

- **Pipeline V3 (finite-proxy kernel/MMD) đã chốt và đã push, chạy được end-to-end**
  (đã verify trên synthetic). Code ở `cawot/`.
- **Lý thuyết = 7 kết quả, chuỗi đã LIỀN về logic** (vá xong mắt xích đứt
  raw→normalized). Nhưng "liền logic" ≠ "chặt định lượng": vài hằng số cần đo
  thực nghiệm + một nhà thống kê đọc Theorem 1.
- **Thực nghiệm = chiến lược 4 tầng (A→B→C→D), CHƯA có số thật.** Stage A/B chạy
  được; Stage C cần cắm trainer của bạn vào.
- **Việc kế tiếp duy nhất quan trọng:** encode PAB 45K → chạy Stage A/B → fine-tune
  `relevance_only` vs `full` (phép thử `eta=0` sống-chết). **Đừng làm gì khác trước cái này.**

---

## 1. Bài toán (đã reframe — đây là gốc)

**Cũ:** "chúng tôi có heuristic chọn coreset tốt hơn CLIPScore." → yếu, dễ bị xem là importance sampling trá hình.

**Mới:** chọn coreset *query-aware* để fine-tune CLIP cho cross-modal anomaly
retrieval, **khi phân phối truy vấn thật chỉ biết xấp xỉ** qua một *họ hữu hạn*
proxy queries.

> Câu hỏi paper: *"Can a query-aware coreset improve efficient fine-tuning for
> cross-modal anomaly retrieval when the query distribution is only approximately
> known?"*

Reframe này trả lời đúng **2 câu hỏi của thầy**:
1. *"Sao cộng mà không nối z_v, z_t?"* → **không tạo pair vector nào cả.** Cặp được
   biểu diễn qua một *kernel tích*; relevance đo trên **trục text** (vì query là text).
2. *"What is your statistical inference?"* → đại lượng đích là rủi ro truy hồi dưới
   phân phối truy vấn không quan sát được; ta **ước lượng MMD kèm concentration +
   confidence interval** (Theorem 1) thay vì một con số trần.

---

## 2. Pipeline mới — 7 bước (+ bước 0), kèm trạng thái code

| Bước | Làm gì | Input → Output | File | Trạng thái |
|---|---|---|---|---|
| **0** | Chọn 1 kernel tích `k = λ·k_v + (1−λ)·k_t`; RBF-on-sphere + RFF | — → hàm kernel | `kernels.py` | ✅ code xong |
| **1** | Encode CLIP, **giữ z_v, z_t riêng** (KHÔNG tạo e_i) | ảnh+caption → z_v, z_t (cache memmap) | `data.py` | ✅ (cần CLIP cho data thật) |
| **2** | Two-level kernel-aware clustering (mini-batch kmeans trên feature map) | z → coarse (20) × fine (10) | `clustering.py` | ✅ code xong |
| **3** | **Text-side** MMD: `d_{c,r}=MMD²(caption-dist của cluster, proxy r)` qua RFF | clusters + proxies → ma trận d[K,R] | `scoring.py` | ✅ code xong |
| **4** | Chuẩn hóa per-proxy (median/IQR) → `d~` | d → d~ | `scoring.py` | ⚠️ code dùng `iqr+eps`, **lý thuyết muốn `max{iqr, γ₀}`** — xem §6 |
| **5** | `a=exp(−τ·softplus(d~))`, `ābar` (xương sống), `u` (bonus); `s_c=\|C\|^α(1+β·ābar+η·u)`; capped largest-remainder | d~ → ngân sách b_c | `scoring.py` | ✅ code xong (softplus đã có) |
| **6** | kNN sparsify cho cluster lớn (cap chi phí) | cluster lớn → candidate set | `selection.py` | ✅ code xong |
| **7** | **Kernel herding** trong mỗi fine-cluster (full-candidate nếu nhỏ) | candidates + b_c → coreset | `selection.py` | ✅ code xong |

Driver: `pipeline.py` (Stage A/B). Ablation runner: `scripts/run_ablation.py`.

**So với poster cũ:** bỏ SW + pair-vector + greedy O(m³). Bước 1–2 giữ skeleton, bước 3–7 thay mới.

---

## 3. Theory stack — 7 kết quả & trạng thái từng cái

Chuỗi suy luận: `d̂ ≈ d → d̃̂ ≈ d̃ → âbar,û ≈ ābar,u → ŝ ≈ s → b̂ ≈ b`.

| # | Kết quả | Nói gì | Trạng thái |
|---|---|---|---|
| **T1** | Concentration của MMD score (raw `d`) | `\|d̂^RFF − d\| ≤ sampling + ε_RFF(D)` | Phần **sampling proven** (McDiarmid + union bound). Phần **ε_RFF để dạng tổng quát**, chưa bound chặt. |
| **L2** | Ổn định chuẩn hóa median/IQR (với floor `γ₀`) | `\|d̃̂ − d̃\| ≤ L_norm·ε_d`, `L_norm = 2/γ₀ + 8κ/γ₀²` | Đại số **đúng**. Hằng số **có thể rộng** khi IQR nhỏ. |
| **P3** | Ổn định relevance softplus | `\|âbar−ābar\|≤τε`, `\|û−u\|≤4τε` | **Proven**, sạch (đã kiểm hằng số). |
| **L4** | `ābar,u → s` (relative) | `\|ŝ/s − 1\| ≤ ξ`; **C_max biến mất** nhờ chia chuẩn | **Proven** (nhờ số hạng "+1" → h_c ≥ 1). |
| **P5** | q-stability | `‖q̂−q‖₁ ≤ 2ξ/(1−ξ)` | **Proven** (đã khai triển đầy đủ). |
| **C6** | Rounding nguyên | `‖b̂−b‖₁/B ≤ 2ξ/(1−ξ) + 2K/B` | Đúng; `2K/B` lỏng nhưng **nhỏ thực tế** (K=20, B lớn). |
| **L7** | Kernel herding | MMD coreset = `O(1/√b)` (an toàn), `O(1/b)` có điều kiện | **Cite** Bach et al. 2012 + Chen-Welling 2010, không tự chứng minh. |

**Kết luận lý thuyết:** chuỗi **liền về logic** (mục tiêu đã đổi từ `b̂=b` cứng sang
`‖b̂−b‖/B nhỏ` — đúng bản chất coreset). Nhưng **liền logic ≠ chặt định lượng**:
tích `L_norm` (T1→L2→...) có thể lớn khi IQR nhỏ → bound end-to-end lỏng. Cách trình
bày trung thực: phát biểu "tồn tại ngưỡng, dưới đó ổn định" + **báo cáo thực nghiệm**
thay vì dựa bound xấu nhất.

---

## 4. Thực nghiệm — kế hoạch & trạng thái

**Chiến lược 4 tầng (làm đúng thứ tự):**
- **A — Sanity (không fine-tune):** đọc `allocation_report` — cluster nào nhiều budget,
  `ābar`/`u` có hợp lý không, `u` có dồn vào cluster noisy không. → code chạy được ✅
- **B — Selection benchmark (không fine-tune):** chạy mọi selector ở 20%, đo overlap. ✅
- **C — Fine-tune + R@k/mAP:** cắm trainer qua `evaluate_coreset(...)`. → **cần hook** ⬜
- **D — Scale:** 45K → 100K → 250K → 1M. **Đừng nhảy thẳng 1M.** ⬜

**Datasets:** PAB (headline) + CUHK-PEDES + ICFG-PEDES (core, uy tín) + COCO/Flickr30k
(generality check — chống câu hỏi "sao toàn person retrieval?") + UFineBench (robustness).
Optional: RSTPReid, UCFCrime-AR.

**Metrics — 3 tầng:** (a) theory: surrogate risk Lipschitz; (b) benchmark: R@1/5/10 + mAP
(để so với văn liệu); (c) DRO-style: CVaR@α, worst-group recall, đường cong theo độ
dịch proxy. R@k **chỉ là evaluation**, không phải theory target.

**Baselines (so selector, KHÔNG so retrieval model):** Random, CLIPScore, SemDeDup,
k-Center, **SW-CAWOT (poster cũ, V0–V2)**, ClipCov + VAS *adapt sang retrieval*,
CCS/D2-Pruning. Đối thủ tham chiếu = **full-data** ("20% ≈ full, 1/5 chi phí").

**Stats protocol:** ≥5 seeds, bootstrap BCa CI, Wilson cho R@1, paired permutation
test, Holm-Bonferroni, partial-conjunction xuyên dataset, ASO ε_min.

---

## 5. Điểm mạnh đã đạt được (wins — đừng quên mình đã đi xa)

1. **Reframe sạch:** heuristic → bài toán có statistical inference (trả lời được thầy).
2. **Một ngôn ngữ MMD xuyên suốt** — hết mâu thuẫn "SW chỗ này, MMD chỗ kia".
3. **Không còn pair-vector e_i** — đóng câu hỏi cộng-vs-nối của thầy về bản chất.
4. **Sửa domain mismatch:** relevance đo trên trục text (query là text).
5. **Scalable:** O(m³) greedy → herding + RFF (near-linear), không còn O(n²) ẩn ở scoring.
6. **Mean-relevance là xương sống, `u` là bonus** → paper sống dù ablation `eta=0` ra sao.
7. **Cắt đúng thứ cần cắt:** bỏ R@k khỏi theory, bỏ full-DRO + ε-net khỏi main → **kham nổi 1 kỳ**.
8. **Trung thực rõ ràng:** biết chính xác cái gì proven / empirical / cần người đọc.
9. **Code:** đã push, chạy end-to-end, logging đầy đủ; **V0–V2 giữ nguyên làm fallback**.

---

## 6. Vấn đề mới / điểm mở (cần address)

| Vấn đề | Mức độ | Hướng xử lý |
|---|---|---|
| **Code §4 dùng `iqr+eps`, lý thuyết L2 muốn `max{iqr, γ₀}`** | 🔴 sửa code | Đổi `normalize_per_proxy` sang denominator floor `max{IQR, γ₀}`; `γ₀` thành hyperparameter có ablate ({0.01,0.05,0.1}). |
| `ε_RFF` chưa bound chặt | 🟡 cần statistician | Bernstein vector-valued / empirical-process; hoặc để dạng tổng quát + đo thực nghiệm. |
| `L_norm` rộng khi IQR nhỏ | 🟡 đo + trình bày | Đo `γ₀`, IQR, `L_norm`, `ξ` thực nghiệm trên PAB; trình bày bound suy biến ở IQR nhỏ. |
| `u_c` đo "sensitivity to proxy choice", **KHÔNG phải true risk** | 🟡 viết thật | Nói thẳng trong paper; đừng overclaim. |
| Finite-proxy **không tự kế thừa** DRO | 🟢 remark | Để ý tưởng nối-DRO ở 1 remark điều kiện (future work), không claim ở main. |
| Sparse herding ≠ guarantee full herding | 🟢 verify | Main theory cho full-candidate; sparse là approximation, **verify empirical** không tụt R@k. |
| `2K/B` lỏng | 🟢 OK thực tế | K nhỏ, B lớn → bỏ qua; ghi remark "biến mất khi không có ties". |
| **Chưa ai (nhà thống kê) đọc Theorem 1** | 🟡 quan trọng | Gửi 4-trang cho TrungTin Nguyen / Khai Nguyen. |

---

## 7. Việc cần làm — theo thứ tự ưu tiên

**Ngay (tuần này) — để DỮ LIỆU quyết định câu chuyện:**
1. ⬜ Sửa `normalize_per_proxy` → floor `max{IQR, γ₀}` (khớp L2). *(~30 phút code)*
2. ⬜ Encode PAB 45K bằng CLIP, build Q1 (templates) + Q2 (paraphrases), cache `.npy`.
3. ⬜ Chạy **Stage A** — đọc allocation_report, kiểm `u` vs cluster noisy.
4. ⬜ Chạy **Stage B** — selection benchmark @20%, đọc overlap.
5. ⬜ Cắm trainer vào `evaluate_coreset` → **fine-tune `relevance_only_norm` vs `full_norm`**
   = phép thử `eta=0` SỐNG-CHẾT. Đọc số → chốt kịch bản A/B/C (xem README §"decisive").

**Sau khi có tín hiệu:**
6. ⬜ Đo `γ₀, IQR, L_norm, ξ` thực nghiệm → điền vào appendix theory.
7. ⬜ Gửi bản 4-trang theory cho 1 nhà thống kê (TrungTin Nguyen / Khai Nguyen).
8. ⬜ Mở rộng: budget 5/50%, thêm dataset, ≥5 seeds, full stats protocol.
9. ⬜ Viết paper — **theorem statement first**, R@k chỉ ở experiments.

**KHÔNG làm vội:** 1M, full-DRO, ε-net, đổi tên method, viết abstract trước khi có số ablation.

---

## 8. Quyết định đã CHỐT (đừng mở lại — tránh vòng lặp vô tận)

- MMD (không SW) ở theory; **SW-CAWOT giữ làm baseline/ablation**, không phải main.
- Relevance đo **text-side**; **không tạo pair vector**.
- Kernel **additive** (`λ·k_v+(1−λ)·k_t`), không product (product để ablation).
- Relevance = `exp(−τ·softplus(d~))`.
- Denominator **floor `γ₀`** (không phải eps số học) — *cần đưa vào code*.
- **Mean-relevance = xương sống; `u` = bonus** (η điều khiển).
- **Không** R@k theorem; **không** full-DRO / ε-net ở main paper.
- `Q_3` (validation-failure) **chỉ ablation**, không bao giờ từ test data.
- Giữ tên CAWOT (chưa đổi tên method).

---

## 9. Một câu để tự định hướng mỗi khi lạc

> *Method đã đủ vững để **chạy**. Việc tiếp theo không phải nghĩ thêm framework mới,
> mà là **để dữ liệu lên tiếng** (ablation `eta=0`) và **để một nhà thống kê đọc
> Theorem 1**. Hai việc đó không AI nào làm thay được.*
