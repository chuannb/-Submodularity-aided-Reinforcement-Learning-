# Experiment Notes — exp-smaller-dataset

Branch: `exp-smaller-dataset` (fork từ `bert4rec-backbone`)

---

## 0. Môi trường thực nghiệm

- **GPU**: NVIDIA GeForce RTX 3090 (23 GB VRAM)
- **Quy tắc**: Mọi lần chạy pipeline phải dùng `--device cuda` để có kết quả đủ nhanh
- **Smoke test**: `--max_users 200 --epochs 1 --steps_per_epoch 50 --device cuda`
- **Full run**: Không giới hạn max_users, ít nhất 5 epochs

---

## 1. Vấn đề cốt lõi được xác định

### Root cause: Dataset quá lớn

Pipeline hiện tại train trên **toàn bộ Amazon cross-category** không lọc category:

| | Hiện tại (full Amazon) | Standard benchmark |
|--|--|--|
| Items | **2,611,529** | 11K–18K |
| Item embedding params | **332M** (2.6M × 128) | 1.5–2.3M |
| BM25 recall@100 | ~5% | ~30–50% |
| Reward rate (r_t > 0) | ~1% | ~10–20% |
| RL gradient signal | Gần như toàn 0 | Đủ để học |

**Hệ quả:**
- BERT4Rec/SASRec item embedding quá sparse → không học được (mỗi item gặp rất ít lần/epoch)
- Recall thấp → submodular greedy nhận candidates kém → RL reward gần như luôn = 0
- Diversity embeddings `item_emb` (2.6M × 64 = 167M params) không hội tụ

### Kết quả thực nghiệm trước (full Amazon, 50K users)
```
Hit@10 = 0.0100   (epoch 1, best checkpoint)
NDCG@10 = 0.0100
MRR@10  = 0.0100
Coverage = 1.3%
```
Model tốt hơn random ~33×, nhưng còn xa SOTA (SASRec ~5–15%).

---

## 2. Benchmark numbers — các model có thể so sánh

> Nguồn: ICSRec (WSDM 2024 Oral, [arxiv:2310.14318](https://arxiv.org/abs/2310.14318))  
> Evaluation protocol: **leave-one-out** (last item làm test), **full ranking** (không sample negatives)  
> Dataset: Amazon 2014 5-core

### Amazon Beauty (22K users, 12K items)

| Model | HR@10 | NDCG@10 | Ghi chú |
|-------|------:|--------:|---------|
| BPR | 0.0296 | 0.0147 | MF baseline |
| GRU4Rec | 0.0284 | 0.0150 | RNN-based |
| Caser | 0.0342 | 0.0226 | CNN-based |
| **SASRec** | **0.0624** | **0.0342** | Transformer causal |
| **BERT4Rec** | **0.0601** | **0.0300** | Transformer bidirectional |
| CL4SRec | 0.0642 | 0.0345 | SASRec + contrastive |
| CoSeRec | 0.0725 | 0.0410 | augmentation-based |
| DuoRec | 0.0851 | 0.0441 | structural contrastive |
| ICLRec | 0.0744 | 0.0403 | intent contrastive |
| IOCRec | 0.0774 | 0.0396 | — |
| **ICSRec** | **0.0960** | **0.0579** | SOTA contrastive (2024) |
| *Bạn (hiện tại)* | *0.0100* | *0.0100* | full Amazon, 2.6M items |

### Amazon Sports (35K users, 18K items)

| Model | HR@10 | NDCG@10 |
|-------|------:|--------:|
| GRU4Rec | 0.0258 | 0.0142 |
| Caser | 0.0261 | 0.0135 |
| **SASRec** | **0.0333** | **0.0177** |
| **BERT4Rec** | **0.0359** | **0.0190** |
| CL4SRec | 0.0369 | 0.0191 |
| CoSeRec | 0.0439 | 0.0244 |
| DuoRec | 0.0466 | 0.0244 |
| ICLRec | 0.0437 | 0.0238 |
| **ICSRec** | **0.0565** | **0.0335** |

### Amazon Toys (19K users, 12K items)

| Model | HR@10 | NDCG@10 |
|-------|------:|--------:|
| GRU4Rec | 0.0184 | 0.0097 |
| Caser | 0.0333 | 0.0168 |
| **SASRec** | **0.0652** | **0.0320** |
| **BERT4Rec** | **0.0524** | **0.0309** |
| CoSeRec | 0.0755 | 0.0442 |
| DuoRec | 0.0959 | 0.0490 |
| **ICSRec** | **0.1055** | **0.0657** |

### MovieLens-1M (6K users, 3.7K items)

| Model | HR@10 | NDCG@10 |
|-------|------:|--------:|
| GRU4Rec | 0.1344 | 0.0649 |
| Caser | 0.1442 | 0.0734 |
| **SASRec** | **0.1810** | **0.0948** |
| **BERT4Rec** | **0.2219** | **0.1097** |
| DuoRec | 0.3078 | 0.1749 |
| **ICSRec** | **0.3368** | **0.2007** |

### Beauty — semantic/LLM models (nguồn: LIGER, [arxiv:2411.18814](https://arxiv.org/abs/2411.18814))

| Model | NDCG@10 | Recall@10 | Ghi chú |
|-------|--------:|----------:|---------|
| SASRec | 0.0218 | 0.0511 | ID-based, cold-start = 0 |
| UniSRec | 0.0335 | 0.0694 | text-based |
| RecFormer | 0.0288 | 0.0627 | text-based |
| TIGER | 0.0322 | 0.0601 | generative |
| **LIGER** | **0.0402** | **0.0745** | generative+dense |

> Note: Con số SASRec ở đây thấp hơn bảng trên vì dùng evaluation protocol khác (temporal split, full ranking trên toàn catalog kể cả cold-start items).

### Diversity metrics (nguồn: TRIER, ACM TOIS 2024; CatDive, PLOS ONE 2025)

TRIER report cải thiện so với SASRec baseline:
- Steam: ILD@5 tăng **+11.36%**
- Yelp: ILD@5 tăng **+3.43%**, HR@5 tăng **+7.62%**, NDCG@5 tăng **+8.63%**
- Beauty: CC@5 (Category Coverage) tăng **+3.77%**

CatDive (Amazon Books, Kindle — khác dataset nhưng có ILD):
- SASRec baseline: HR@10=0.1477, NDCG@10=0.0807, ILD@10≈0.514, Coverage@10≈0.101
- CatDive: HR@10=0.1938 (+31%), NDCG@10=0.1066 (+32%), ILD@10=0.5947 (+15.6%), Cov@10=0.1213 (+20%)

---

### Mục tiêu cần đạt để paper competitive

| Dataset | Hit@10 baseline (SASRec) | Mục tiêu tối thiểu | SOTA (2024) |
|---------|------------------------:|------------------:|------------:|
| Beauty | 0.0624 | **≥ 0.0624** + ILD cao hơn | 0.0960 (ICSRec) |
| Sports | 0.0333 | **≥ 0.0333** + ILD cao hơn | 0.0565 (ICSRec) |
| Toys | 0.0652 | **≥ 0.0652** + ILD cao hơn | 0.1055 (ICSRec) |
| ML-1M | 0.1810 | **≥ 0.1810** + ILD cao hơn | 0.3368 (ICSRec) |

> Paper của bạn không cần beat ICSRec về accuracy. Claim là: **với accuracy tương đương SASRec, hệ thống đạt diversity (ILD) cao hơn đáng kể** — đây là trade-off mà accuracy-only models không tối ưu.

---

## 3. Dataset chuẩn các paper benchmark

Số liệu xác thực từ [RecSys 2024 — "Does It Look Sequential?"](https://arxiv.org/pdf/2408.12008):

| Dataset | Users | Items | Interactions | So với hiện tại |
|---------|------:|------:|-------------:|----------------|
| Amazon Beauty (2014 5-core) | 22,363 | **12,101** | 198,502 | Items × 215 nhỏ hơn |
| Amazon Sports (2014 5-core) | 35,598 | **18,357** | 296,337 | Items × 142 nhỏ hơn |
| Amazon Toys (2014 5-core) | 19,412 | **11,924** | 167,597 | Items × 219 nhỏ hơn |
| MovieLens-1M | 6,040 | **3,706** | 1,000,209 | Items × 704 nhỏ hơn |
| Steam | 281,210 | **7,931** | 3,484,694 | — |

**Các paper quan trọng và dataset họ dùng:**
- **HistLLM** (2504.10150, Apr 2025): Movie 605 users/2,400 items, Netflix 803/3,219, News 7,199/14,599
- **RL-LLM Rec** (2403.16948, 2024): LFM 11K seq/18K items, Industry 10K seq/5.8K items
- **TRIER** (ACM TOIS): Beauty, Sports, Steam (standard 5-core)
- **DivGCL** (AAAI 2025): ML-1M, Beauty, AMiner, Yelp2018
- **Non-monotone Submodular** (AAAI 2024): Video recommendation, small datasets

---

## 3. Dataset đã tải về

```
/workspace/datasets/
├── beauty_2014/
│   └── reviews_5core.json.gz     # 22,363 users, 12,101 items, 198,502 interactions
├── sports_2014/
│   └── reviews_5core.json.gz     # 35,598 users, 18,357 items, 296,337 interactions
├── toys_2014/
│   └── reviews_5core.json.gz     # 19,412 users, 11,924 items, 167,597 interactions
├── ml-1m/ml-1m/
│   ├── ratings.dat                # 6,040 users, 3,706 items, 1,000,209 interactions
│   ├── movies.dat
│   └── users.dat
├── beauty/                        # Amazon Beauty 2023 (full, backup)
│   ├── reviews.jsonl.gz           # 631,986 users, 112,565 items, 701,528 interactions
│   └── meta.jsonl.gz
└── sports/                        # Amazon Sports 2023 (full, backup)
    ├── reviews.jsonl.gz
    └── meta.jsonl.gz
```

**Format Amazon 2014 (fields):**
```json
{"reviewerID": "A...", "asin": "B...", "overall": 4.0,
 "reviewText": "...", "summary": "...", "unixReviewTime": 1234567890}
```

**Lưu ý:** Amazon 2014 KHÔNG có `also_buy`/`also_view` trong review file.
Nếu cần hard negatives cho DPO, phải tải thêm metadata từ:
`https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz`

---

## 4. Submodular vẫn hợp lý với dataset nhỏ

Submodular greedy chạy trên **candidate set** (top-100 sau retrieval), **không** phải toàn catalog.
Catalog size không ảnh hưởng computation. Nhưng diversity embeddings thì có:

| | Full Amazon | Beauty 2014 |
|--|--|--|
| `item_emb` shape | 2.6M × 64 = **167M params** | 12K × 64 = **768K params** |
| Gradient update/epoch | Cực sparse | Dense, học được |

Với dataset nhỏ: recall cao hơn → reward ít sparse → submodular+RL học được thực sự.

---

## 5. Novelty analysis — dùng BERT4Rec/SASRec làm retriever

**Kết luận: Không làm mất novelty. Cần frame đúng.**

Pipeline claim:
```
[BERT4Rec/SASRec]  →  [Qwen3-Reranker + DPO]  →  [Submodular f_θ + RL]
   ← standard →          ← khá novel →              ← novel nhất →
```

**Retriever là plug-in component** — giống RAG papers không claim novelty ở vector database.

**Claim chính xác:**
> "A retriever-agnostic framework for diverse slate optimization, combining a learnable
> submodular function with an RL policy that adaptively balances relevance and diversity
> based on user history."

**So sánh với MMR (Maximal Marginal Relevance) — câu hỏi reviewer hay hỏi:**

| | MMR | Ours |
|--|--|--|
| λ (relevance weight) | Fixed | RL học adaptive theo user state |
| Similarity metric | Cosine cố định | Learned item_emb + RBF kernel (θ) |
| DPO signal | Không | Finetune reranker từ preference pairs |

**Ablation bắt buộc để defend novelty:**

| Retriever | Stage 2+3 | Hit@10 | ILD | Coverage |
|-----------|-----------|--------|-----|----------|
| BM25 only | ✗ | baseline | — | — |
| BERT4Rec only | ✗ | baseline | — | — |
| BM25 | ✓ (ours) | +?% | +?% | +?% |
| BERT4Rec | ✓ (ours) | +?% | +?% | +?% |
| SASRec | ✓ (ours) | +?% | +?% | +?% |

---

## 6. Paper liên quan cần đọc kỹ

| Paper | Link | Lý do cần đọc |
|-------|------|--------------|
| HistLLM (2025) | [arxiv:2504.10150](https://arxiv.org/abs/2504.10150) | LLM + history encoding gần nhất với ours |
| RL-LLM Rec (2024) | [arxiv:2403.16948](https://arxiv.org/abs/2403.16948) | RL + LLM reranker, compare methodology |
| TRIER (ACM TOIS) | [doi:10.1145/3653016](https://dl.acm.org/doi/10.1145/3653016) | Diversity sequential rec, same datasets |
| DivGCL (AAAI 2025) | [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/33852) | Diversity baseline to compare |
| Non-monotone Submodular | [arxiv:2308.08641](https://arxiv.org/abs/2308.08641) | Submodular theory baseline |
| Bayesian Diversity (2025) | [arxiv:2506.21617](https://arxiv.org/abs/2506.21617) | Sequential diversity sampling |
| ReRec (2025) | [arxiv:2604.07851](https://arxiv.org/abs/2604.07851) | RL finetuning LLM rec |
| Does It Look Sequential? | [arxiv:2408.12008](https://arxiv.org/abs/2408.12008) | Dataset stats reference |

---

## 7. Việc cần làm trên branch này

- [x] ~~Viết `data/beauty_loader.py`~~ — không cần, `amazon_loader.py` đã xử lý đúng format 2014 (`reviewerID`, `asin`, `overall`, `unixReviewTime`). `build_meta_from_reviews()` tự tạo catalog khi không có meta file.
- [x] ~~Dataset verify~~ — tất cả 4 dataset khớp với paper (Beauty: 22,363u/12,101i, Sports: 35,598u/18,357i, Toys: 19,412u/11,924i, ML-1M: 6,040u/3,706i)
- [ ] Adapter `run_amazon.py` → `run_beauty.py` — chạy pipeline trên Beauty/Sports/Toys với defaults phù hợp
- [ ] Benchmark BERT4Rec standalone trên Beauty 2014 → lấy số recall@k baseline
- [ ] Benchmark SASRec standalone trên Beauty 2014
- [ ] Chạy full pipeline (BERT4Rec → Reranker → Submodular+RL) trên Beauty 2014
- [ ] So sánh ILD, Hit@10, NDCG@10, Coverage với baselines
- [ ] Tải meta_Beauty.json.gz nếu cần DPO pairs (also_buy/also_view)

---

## 8. Bugs cần fix trước khi chạy (từ REPORT.md)

```python
# algorithms/unified_trainer.py, dòng 346-353
# ĐÃ FIX: next_state được tính đúng từ history[t+1]
next_hist = (step.history_ids + [step.item_id])[-self.pipeline.history_length:]
next_state = self.pipeline.encode_state(next_hist, next_ext)  # ĐÚNG
```

- [x] ~~`next_state` bug~~ — ĐÃ FIX trong unified_trainer.py dòng 348-353

Sparse reward fix (chưa implement):
```python
# HIỆN TẠI (trajectory_builder.py):
r_t = (stars / 5.0) if target in slate else 0.0   # không có penalty, không có rank bonus

# CẦN SỬA THÀNH:
r_t = (stars/5) * (1 + 0.2 * (k - rank_in_slate) / k)  if hit else -0.01
```

- [ ] Fix sparse reward — cần implement shaped reward với miss penalty (-0.01) và rank bonus
