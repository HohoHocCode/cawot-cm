# Hướng dẫn tạo Proxy Query Dataset — cho bản V3 (finite-proxy MMD)

> Dành cho người tạo proxy. Đọc §0 + §1 trước (vì cách làm cho V3 **khác hẳn**
> poster cũ), rồi theo §2–§5 để tạo, §6–§7 để kiểm.

---

## 0. TL;DR

- Tạo **3 HỌ (family) proxy KHÁC NHAU**, mỗi họ ~**500–1000** câu query text
  (tổng ~1500–3000). **Không phải 1 đống query đồng nhất.**
- Mỗi họ = **một "góc nhìn" khác** về cách người dùng mô tả hành vi bất thường
  (taxonomy / câu người dùng / mô tả cảnh).
- **Tất cả chỉ lấy từ ngữ nghĩa training-side** (action/anomaly/scene của PAB).
  **TUYỆT ĐỐI không** từ test captions / identities / labels.
- Output: mỗi họ 1 file `.txt` (1 query/dòng) → embed bằng CLIP text encoder →
  `.npy` → đưa vào `--proxies`.

---

## 1. Vì sao V3 cần NHIỀU HỌ ĐA DẠNG (đọc kỹ — đây là khác biệt chính)

**Proxy là gì:** một phân phối các câu truy vấn text, mô phỏng "người dùng cuối sẽ
hỏi gì". Nó là *anchor* để method biết vùng nào trong pool đáng chọn — vì ta
**không quan sát được** phân phối truy vấn thật lúc chọn dữ liệu.

**Poster cũ (V0–V2):** dùng **1 set** proxy (~2,593 query LLM) làm 1 anchor duy nhất.

**V3 dùng MỘT HỌ `{Q_1, …, Q_R}`** và sinh ra 2 đại lượng:
- `abar_c` (**mean relevance**) = trung bình độ liên quan của cluster *c* qua các họ
  → **xương sống** của allocation.
- `u_c` (**disagreement**) = phương sai độ liên quan **giữa các họ** → **bonus** (η điều khiển).

> **Mấu chốt:** `u_c` chỉ có ý nghĩa khi **các họ BẤT ĐỒNG** với nhau. Nếu các họ
> gần như giống nhau, `u_c ≈ 0` ở mọi cluster → bonus η không có gì để bắt.
>
> Đây **rất có thể là lý do η bị "hòa"** trong kết quả Stage C vừa rồi: bản cũ đặt
> `Q_2 = paraphrase của Q_1` → hai họ **đồng nhất** → phép thử η không công bằng.

→ **Bản V3 yêu cầu các họ đa dạng theo một trục rõ ràng**, không phải viết lại cùng
một thứ. Tạo đúng proxy ở đây = điều kiện để biết bonus η thật sự có giá trị hay không.

---

## 2. Tạo bao nhiêu? (counts + lý do)

| Tham số | Khuyến nghị | Lý do |
|---|---|---|
| **Số họ `R`** | **3** (tối thiểu 2, tối đa ~5) | Cần ≥2 để `u_c` tồn tại; 3–5 đủ để ước lượng disagreement mà không quá khó quản. |
| **Query / họ `m_r`** | **500–1000** | MMD ổn định cần `m` ≥ vài trăm (sai số ~ `m^(−1/2)`). Dưới ~200 thì MMD nhiễu. |
| **Tổng** | ~**1500–3000** | Khớp scale poster cũ (~2,593). |
| **M-sweep (ablation)** | {500, 1000, 2593, 5000} | Để báo cáo độ nhạy theo *kích thước* proxy (mục limitation của report). |

Lưu ý: query **chất lượng + đa dạng** quan trọng hơn số lượng thô. 600 query phủ
rộng + 3 họ khác nhau **tốt hơn** 5000 query na ná nhau.

---

## 3. Ba họ đề xuất — mỗi họ MỘT góc nhìn

Cả ba **đều** dựng từ ngữ nghĩa hành vi bất thường ở **training-side** của PAB
(action/anomaly/scene), nhưng khác nhau về *phong cách & độ cụ thể* để tạo bất đồng:

| Họ | Góc nhìn | Ví dụ query | Vai trò |
|---|---|---|---|
| **`q_taxonomy`** | Có cấu trúc, theo taxonomy hành vi | "a person falling on the stairs" | Phủ rộng, formal |
| **`q_user`** | Câu người dùng gõ thật, ngắn, đời thường | "guy collapsing on sidewalk" | Sát deployment, colloquial |
| **`q_scene`** | Mô tả cảnh/bối cảnh, dài hơn | "an elderly man loses balance and falls while climbing a staircase in a mall" | Chi tiết, giàu ngữ cảnh |

Ba họ này **khác style/độ cụ thể** → cùng một cluster có thể "liên quan" với họ này
nhưng "ít liên quan" với họ kia → `u_c` mới có cái để đo.

> **`q_valfail` (validation-failure): CHỈ DÙNG TRONG ABLATION**, dựng từ một
> validation split **sạch**, **không bao giờ** từ test. Không nằm trong họ chính.

---

## 4. Tạo bằng DeepSeek API — prompt cụ thể

### Bước 4.1 — Lấy "seed attributes" từ PAB training

Trích các cụm hành vi/bất thường/bối cảnh từ **caption/annotation phần train** của
PAB (action: "falling", "lying on the ground", "climbing a fence"; scene:
"stairs", "sidewalk", "mall"…). Khoảng **100–200 attribute** là đủ để bung ra hàng
nghìn query. *(Chỉ lấy từ train. Không đụng test.)*

### Bước 4.2 — Sinh từng họ bằng DeepSeek (mỗi họ một system prompt KHÁC nhau)

Điểm quan trọng: **dùng system prompt khác nhau cho mỗi họ** để chúng đa dạng.
Query nên **bằng tiếng Anh** (CLIP/PAB là tiếng Anh).

**Họ `q_taxonomy`** (formal, theo taxonomy):
```
System: You generate concise, formal image-search queries describing a person's
abnormal behavior, in the style of a dataset taxonomy. One query per line, no
numbering. Each query is a clean noun phrase like "a person <action> <where>".
User: Generate 8 distinct formal queries for the behavior attribute: "{attribute}".
Vary the location/scene but keep the phrasing formal and templated.
```

**Họ `q_user`** (câu người dùng thật, ngắn, đời thường):
```
System: You generate SHORT, casual search queries the way a real user would type
them into a search box — informal, sometimes incomplete grammar, lowercase,
3–8 words. One per line, no numbering, no quotes.
User: Write 8 casual user-style search queries a person might type to find someone
"{attribute}". Use everyday words (e.g. "guy", "old man", "kid"), not formal terms.
```

**Họ `q_scene`** (mô tả cảnh dài, giàu ngữ cảnh):
```
System: You write one-sentence DESCRIPTIVE captions of a scene where a person shows
abnormal behavior, including subject, action, and setting. 12–25 words, natural
English. One per line, no numbering.
User: Write 8 descriptive one-sentence scene captions for the behavior:
"{attribute}". Vary subject (age/clothing), setting, and time of day.
```

Lặp prompt qua **mọi attribute** → gộp lại theo họ. Với ~150 attribute × 8 query =
~1200 query/họ (cắt/đa dạng hóa xuống ~500–1000 sau dedup).

### Bước 4.3 — Hậu xử lý (bắt buộc)
- **Dedup** trong từng họ (bỏ trùng exact + gần-trùng).
- **Chuẩn hóa** nhẹ: strip, bỏ số thứ tự/bullet/dấu ngoặc thừa.
- **Giữ khác biệt giữa các họ** — KHÔNG dedup chéo đến mức làm chúng giống nhau.
- **Lọc rò rỉ:** bỏ bất kỳ câu nào trùng/đạo từ test caption (nếu có nghi ngờ).

---

## 5. Output format & cắm vào pipeline

**Người tạo proxy giao:** 3 file text (1 query/dòng):
```
proxies/q_taxonomy.txt
proxies/q_user.txt
proxies/q_scene.txt
```

**Embed sang `.npy`** (CLIP text encoder, đã chuẩn hóa) — dùng helper có sẵn:
```python
from cawot.data import embed_texts
import numpy as np
for name in ["q_taxonomy", "q_user", "q_scene"]:
    texts = [l.strip() for l in open(f"proxies/{name}.txt") if l.strip()]
    np.save(f"cache/{name}.npy", embed_texts(texts))   # (m_r, 512) float32, L2-normalized
```

**Chạy ablation với 3 họ:**
```bash
PYTHONPATH=. python scripts/run_ablation.py \
    --zv cache/z_v.npy --zt cache/z_t.npy \
    --proxies cache/q_taxonomy.npy cache/q_user.npy cache/q_scene.npy \
    --budget-frac 0.2 --out runs/pab45k_3proxy
```
> `--proxies` nhận **số họ tùy ý** — không cần sửa code. (`cawot.data.build_proxy_families`
> có sẵn Q1+Q2 nếu muốn làm nhanh, nhưng để **đa dạng** thì nên tự tạo text per-family
> bằng DeepSeek như trên.)

---

## 6. Quy tắc CỨNG (vi phạm = phải làm lại)

1. **KHÔNG** dùng test captions / identities / labels / phân phối truy vấn test để
   tạo proxy chính. Rò rỉ test = kết quả vô giá trị.
2. `q_valfail` **chỉ** từ validation split sạch, **chỉ** dùng trong ablation.
3. Mỗi họ **coherent nội bộ** nhưng **khác** các họ khác — đó chính là điểm.
4. Query bằng **tiếng Anh**, chuẩn hóa, dedup trong-họ.

---

## 7. Kiểm tra chất lượng TRƯỚC khi giao

- **Coverage:** các họ có chạm đủ loại hành vi bất thường của PAB không (đừng chỉ
  toàn "falling"). Đối chiếu với danh sách attribute đã trích.
- **Diversity giữa họ (quan trọng nhất):** đo độ khác nhau giữa các họ. Cách nhanh:
  ```python
  from cawot.kernels import RFF, median_heuristic_bandwidth, mmd2_rff
  import numpy as np
  Q = {n: np.load(f"cache/{n}.npy") for n in ["q_taxonomy","q_user","q_scene"]}
  allq = np.concatenate(list(Q.values())); rng = np.random.default_rng(0)
  sig = median_heuristic_bandwidth(allq, rng=rng)
  rff = RFF(allq.shape[1], 1024, sig, rng=rng)
  mu = {n: rff.mean_embedding(v) for n, v in Q.items()}
  for a in Q:
      for b in Q:
          if a < b: print(a, b, round(mmd2_rff(mu[a], mu[b]), 4))
  ```
  Nếu MMD giữa các họ **≈ 0** → chúng quá giống → `u_c` sẽ vô dụng → **phải làm khác đi**
  (đổi style mạnh hơn). Nếu MMD **> 0 rõ rệt** → tốt, các họ thật sự là góc nhìn khác.
- **Sanity ngữ nghĩa:** embed thử vài query, tìm caption gần nhất trong pool — xem có
  hợp lý không (query "falling" → kéo ra ảnh người ngã, không phải người đi bộ).

---

## 8. Vì sao việc này đáng làm NGAY

Kết quả Stage C cho η "hòa" — nhưng **có thể do proxy cũ đồng nhất** (Q2 = paraphrase
Q1), chứ không phải do bonus η vô dụng. Tạo **3 họ thật sự đa dạng** rồi **chạy lại
η-ablation** (`relevance_only` vs `full`) chính là **phép thử công bằng** cho η:
- nếu η>0 thắng với proxy đa dạng → câu chuyện "disagreement giúp robustness" sống lại;
- nếu vẫn hòa → lúc đó mới kết luận chắc chắn "mean-relevance là đủ, η chỉ là diagnostic".

Hoặc cách nào thì cũng **trả lời dứt điểm** một câu hỏi đang treo — đáng giá hơn nhiều
so với chạy thêm data ở proxy cũ.
