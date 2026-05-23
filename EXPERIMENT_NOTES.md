# Experiment Notes — exp-smaller-dataset

Branch: `exp-smaller-dataset` (fork từ `bert4rec-backbone`)

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

## 2. Dataset chuẩn các paper benchmark

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

- [ ] Viết `data/beauty_loader.py` — load Amazon 2014 format (`reviewerID`, `asin`, `overall`, `unixReviewTime`)
- [ ] Adapter `run_amazon.py` → `run_beauty.py` — chạy pipeline trên Beauty/Sports/Toys
- [ ] Benchmark BERT4Rec standalone trên Beauty 2014 → lấy số recall@k baseline
- [ ] Benchmark SASRec standalone trên Beauty 2014
- [ ] Chạy full pipeline (BERT4Rec → Reranker → Submodular+RL) trên Beauty 2014
- [ ] So sánh ILD, Hit@10, NDCG@10, Coverage với baselines
- [ ] Tải meta_Beauty.json.gz nếu cần DPO pairs (also_buy/also_view)

---

## 8. Bugs cần fix trước khi chạy (từ REPORT.md)

```python
# algorithms/unified_trainer.py, dòng 332
# BUG: next_state dùng state hiện tại thay vì state thực của bước tiếp theo
next_state=trans_dict["state"],   # SAI — phải là next step's actual state

# FIX cần implement:
# Khi build trajectory, lưu next_state = encode(history[t+1])
```

Sparse reward fix:
```python
# Thay vì: r_t = stars/5 if hit else 0
# Dùng shaped reward:
r_t = (stars/5) * (1 + rank_bonus * (k - rank_in_slate) / k)  if hit else -0.01
```
