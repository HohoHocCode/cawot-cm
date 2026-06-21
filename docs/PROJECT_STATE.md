# CAWOT-CM — Bản đồ trạng thái dự án (State of the Project)

> Một file duy nhất để không lạc. Cập nhật khi có kết quả mới. Đọc §0 trước.
> Cập nhật gần nhất: sau Stage C lần 1 (≈5% budget) — kết quả NULL trên synthetic val.

---

## 0. TL;DR — đang ở đâu

- **Method đã CHỐT: `proposed` = finite-proxy mean-relevance coreset (η=0).** MMD
  text-side + RFF + kernel herding. Pipeline chạy thật trên PAB, nhanh, scale ổn.
- **η (proxy-disagreement) đã bị BỎ** — xác nhận inert (u ~1000× nhỏ hơn abar; full vs
  relevance-only trùng 99.98% ngay cả với 6 family đa dạng). Giữ làm negative-result.
- **Bộ apparatus chứng minh đã sẵn sàng**: `run_ablation` (selection + baselines),
  `embed_qproxy_families` (proxy→npy), `make_results_table` (bảng + Wilson CI),
  `docs/EXPERIMENTS.md` (protocol + tiêu chí thắng pre-registered).
- **Kết quả R@1 đầu tiên (≈5%, synthetic val, 1 seed) = NULL**: không method nào
  thắng random trên t2i; mọi CI chồng nhau.
- **Việc duy nhất quyết định còn lại:** chạy **PAB real test 1.978 + ≥3 seeds +
  full_data** ở 2%/5%. Đó mới là phép thử pre-registered. Chưa chạy = chưa kết luận.

---

## 1. Bài toán (đã reframe)

Chọn coreset *query-aware* để fine-tune CLIP cho cross-modal anomaly retrieval, khi
phân phối truy vấn thật chỉ biết xấp xỉ qua một **họ hữu hạn proxy queries** (text).
Trả lời 2 câu hỏi của thầy: (1) không tạo pair vector — dùng kernel, relevance đo
trên trục text; (2) statistical inference — MMD concentration + CI.

---

## 2. Pipeline (đã chốt) — 7 bước

| Bước | Làm gì | File | Trạng thái |
|---|---|---|---|
| 0 | kernel tích (RBF-on-sphere + RFF) | `kernels.py` | ✅ |
| 1 | encode CLIP, giữ z_v/z_t riêng (không e_i) | `data.py` | ✅ |
| 2 | two-level kernel-aware clustering | `clustering.py` | ✅ |
| 3 | text-side MMD `d_{c,r}` qua RFF | `scoring.py` | ✅ |
| 4 | normalize per-proxy (median/IQR) | `scoring.py` | ✅ (xem §6: nên đổi `IQR+eps`→`max(IQR,γ₀)`) |
| 5 | `s_c=|C|^α(1+β·abar)`, **η=0**; capped largest-remainder | `scoring.py` | ✅ |
| 6 | kNN sparsify cho cluster lớn | `selection.py` | ✅ |
| 7 | kernel herding within cluster | `selection.py` | ✅ |

---

## 3. Theory (7 kết quả, chuỗi liền logic)

`d̂≈d → d̃̂≈d̃ → âbar≈abar → ŝ≈s → b̂≈b`. Trạng thái: **liền về logic**, nhưng *liền
logic ≠ chặt định lượng* (hằng số `L_norm` có thể rộng khi IQR nhỏ; `ε_RFF` để dạng
tổng quát). **Cần một nhà thống kê đọc Theorem 1** (giả thiết sub-Gaussian qua CLIP,
ε_RFF cho MMD²). Đây là giới hạn AI không thay được.

---

## 4. Thực nghiệm — trạng thái & kết quả

**Đã chạy:**
- Selection 6-family @20% và @5% trên PAB 45K — pipeline OK (`pab45k_6proxy`, `pab45k_b5`).
- **Stage C @≈5% (2250 train), baselines đầy đủ, eval synthetic val (5000), 1 seed**
  (`result/cawot_v3_3.zip`, `outputs/stage_c/`, dùng `make_results_table`):

  | metric | kcenter | random | proposed | sw_cawot | clipscore |
  |---|---|---|---|---|---|
  | t2i_R@1 | **79.52** | 79.02 | 78.94 | 78.94 | 78.50 |
  | mean_R@1 | 80.10 | 80.06 | **80.32** | 80.01 | 79.76 |

  → **NULL: không method nào thắng random trên t2i; mọi Wilson CI chồng (~±1.1).**
  `proposed` nhỉnh ở mean/i2t nhưng trong noise và không phải trên t2i → **không
  cherry-pick.**

**Chưa chạy (phép thử quyết định):** PAB **real test 1.978** + ≥3 seeds + `full_data`
ở 2%/5%, t2i tách i2t. (Hiện mới: synthetic val, 1 seed, thiếu full_data.)

**Đọc:** ở budget thấp — nơi selection đáng lẽ tách nhất — proposed ≈ random trên
synthetic val. Khớp lịch sử V0–V2. Real-test là cơ hội cuối (Sim2Real có thể tách).

---

## 5. Điểm mạnh đã đạt (wins)

1. Reframe sạch: heuristic → bài toán có statistical inference.
2. Một ngôn ngữ MMD xuyên suốt; không còn pair-vector e_i; sửa domain mismatch.
3. Scalable: O(m³) greedy → herding + RFF (đo thật: cluster ~20s ở 45K).
4. Method gọn, trung thực: η inert → bỏ; mean-relevance là backbone.
5. **Bộ apparatus reviewer-grade**: baselines + Wilson CI + bảng tự báo cell thiếu +
   protocol pre-registered. Đây là phần "chuyên nghiệp" đã xong.
6. Code + tài liệu trên repo; V0–V2 giữ nguyên làm fallback.

---

## 6. Vấn đề mở / rủi ro

| Vấn đề | Mức | Xử lý |
|---|---|---|
| **proposed chưa thắng random** (5%/synthetic val) | 🔴 | chạy real-test/seeds/full_data; nếu vẫn hòa → đổi trục (§7) |
| eval mới là synthetic val, 1 seed, thiếu full_data | 🟠 | chạy đúng EXPERIMENTS.md |
| tag budget "b4" nhưng 2250/45000 = **5%** | 🟢 | sửa tag |
| `ε_RFF` chưa bound chặt; `L_norm` rộng khi IQR nhỏ | 🟡 | đo γ₀/IQR thực nghiệm + nhà thống kê |
| code §4 dùng `IQR+eps`, theory muốn `max(IQR,γ₀)` | 🟡 | đổi 1 dòng + ablate γ₀ |
| size lấn relevance (`corr` 0.975 vs 0.88) | 🟡 | nếu cần thắng: sweep α (0.5/0.25/0) |

---

## 7. Việc cần làm — ưu tiên

**Ngay (phép thử quyết định):**
1. ⬜ Stage C trên **PAB real test 1.978** + **≥3 seeds** + **full_data**, ở **2% và 5%**,
   báo **t2i tách i2t** + Wilson CI → `make_results_table`.
2. ⬜ (rẻ, làm kèm) sweep **α ∈ {0.5,0.25,0}** vì size đang lấn relevance.

**Quyết theo kết quả #1:**
- proposed **thắng** baselines (real test, t2i, CI không chồng) → giữ method, viết paper.
- proposed **≈ full_data** ở budget thấp → claim efficiency (cần full_data).
- proposed **≈ random** kể cả real test → **đổi trục**: (a) theory (concentration/
  stability), (b) benchmark khác nơi selection quan trọng, (c) chỉnh method (sweep α →
  relevance-dominant → within-cluster *informative* thay vì representative). KHÔNG
  reframe vô tận.

**Sau:** đo γ₀/IQR/L_norm thực nghiệm; gửi theory cho TrungTin/Khai Nguyen; scale 1M.

**KHÔNG vội:** 1M, full-DRO, ε-net, đổi tên, viết abstract trước khi có số real-test.

---

## 8. Quyết định đã CHỐT (đừng mở lại)

- Method = **mean-relevance (η=0)**; η chỉ là ablation/negative-result.
- MMD (không SW) ở theory; SW-CAWOT là baseline. Relevance **text-side**, không pair-vector.
- Kernel **additive**; relevance `exp(−τ·softplus(d̃))`.
- **Không** R@k theorem; **không** full-DRO/ε-net ở main. `Q_3` validation-failure chỉ ablation.
- Metric of record = **t2i_R@1 trên real test**; synthetic val chỉ là phụ.

---

## 9. Một câu để tự định hướng

> Method đã chốt và chạy được; apparatus chứng minh đã xong. Việc còn lại **không phải
> nghĩ thêm method**, mà là **chạy đúng 1 phép thử** (real test + seeds + full_data) để
> biết giữ hay đổi trục. Kết quả 5% hiện tại là NULL trên regime dễ — chưa đóng cửa,
> nhưng là cảnh báo thật.
