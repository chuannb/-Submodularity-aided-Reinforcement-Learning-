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

## 3b. Dataset đã tải về

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

- [x] ~~Viết `data/beauty_loader.py`~~ — không cần, `amazon_loader.py` đã xử lý đúng format 2014.
- [x] ~~Dataset verify~~ — tất cả 4 dataset khớp với paper (Beauty: 22,363u/12,101i, Sports: 35,598u/18,357i, Toys: 19,412u/11,924i, ML-1M: 6,040u/3,706i)
- [x] ~~Adapter `run_amazon.py` → `run_beauty.py`~~ — đã tạo `run_beauty.py` (legacy BM25 mode) và `run_beauty_bert4rec.py` (BERT4Rec mode — **recommended**)
- [ ] Chạy full 100-epoch Beauty 2014 với BERT4Rec pipeline → lấy số hit@10, ndcg@10, ILD
- [ ] Benchmark SASRec standalone trên Beauty 2014
- [ ] So sánh ILD, Hit@10, NDCG@10, Coverage với baselines (SASRec, BERT4Rec-only)
- [ ] Chạy Sports + Toys sau khi Beauty converge
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
- [x] ~~Sparse reward~~ — ĐÃ FIX trong trajectory_builder.py `proxy_reward_amazon()`:

```python
# ĐÃ FIX (trajectory_builder.py):
def proxy_reward_amazon(slate, target, stars=None, rank_bonus=0.2, miss_penalty=-0.01):
    if target not in slate:
        return miss_penalty                              # -0.01: gradient âm nhỏ khi miss
    k    = len(slate)
    rank = slate.index(target)                          # 0-indexed, thấp = vị trí tốt hơn
    weight         = (stars / 5.0) if stars else 1.0
    position_bonus = rank_bonus * (k - 1 - rank) / k   # 0 ở vị trí cuối, max ở đầu
    return float(weight * (1.0 + position_bonus))
```

- [x] ~~**Bug trong trainer** (`unified_trainer.py` dòng 334–335)~~: khi `step.reward is not None`,
  trainer dùng binary hit/miss thay vì shaped reward — ĐÃ FIX, nay luôn gọi `proxy_reward_amazon()`
  với `stars = step.reward * 5` (xem Section 9 để hiểu lý do).

---

## 9. History length và Reward function — Design decisions

### 9.1 History length — nên dùng bao nhiêu item?

Các paper benchmark trên Amazon 2014 5-core đều dùng **20–50 item gần nhất**, không phải 5:

| Paper | maxlen (Beauty/Sports/Toys) | Ghi chú |
|-------|-----------------------------|---------|
| SASRec (Tang & Wang 2018) | **50** | Causal transformer |
| BERT4Rec (Sun et al. 2019) | **50–200** | Bidirectional, thường dùng 50 |
| S3-Rec (AAAI 2021) | **50** | Self-supervised pre-training |
| GRU4Rec | **20** | RNN, ít ổn định hơn |
| REINFORCE-Rec (Chen et al. 2019) | **50** | RL-based |
| SQN (Xin et al. 2020, SIGIR) | **10–20** | RL state = GRU(last 10) |

**Quyết định cho pipeline này:**
- **StateEncoder (RL policy)**: `history_length = 20` — đủ context, không quá nặng
- **BERT4Rec retriever**: `maxlen = 50` — nhiều context hơn cho recall tốt hơn
- **Không dùng "5 item gần nhất"** — không có paper nào làm vậy, quá ít context

### 9.2 Reward function — các paper dùng gì?

**Ngữ cảnh**: Amazon 5-core là **explicit feedback** — toàn bộ tương tác đều là sản phẩm user
thực sự mua VÀ để lại đánh giá (selection bias: user chỉ review khi đủ quan tâm).

| Paper | Dataset | Reward | Notes |
|-------|---------|--------|-------|
| **SQN** (Xin et al. 2020) | Amazon Beauty, Sports | Binary: r=1 nếu hit, 0 nếu không | Implicit feedback |
| **REINFORCE-Rec** (Chen et al. 2019) | Taobao click log | r=click/no-click, purchase=higher | Multi-step env |
| **KERL** (2020) | Amazon | Binary với threshold: rating≥4 → r=1 | Chặn negative reviews |
| **CausalRec** (Yuan et al. 2021) | Amazon 5-core | r=1 nếu target trong top-K | Binary |
| **RLREC** (Zhao et al. 2018) | MovieLens, LastFM | r = rating/max_rating nếu hit | Rating-weighted |

**Phổ biến nhất**: binary hit (r=1) hoặc NDCG-style (r=1/log2(rank+2)).

**Current approach (shaped reward):**
```
r_t = (stars/5) × (1 + 0.2×(k-1-rank)/k)  if hit   →  [0.2, 1.18]  range với k=10
r_t = -0.01                                  if miss
```
Đây là rating-weighted + rank bonus — hợp lý nhưng phức tạp hơn cần thiết.

**✅ Đã fix trong unified_trainer.py dòng 334–342:**
```python
# ĐÃ FIX:
if step.reward is not None and dataset_type == "amazon":
    # step.reward = target_rating/5 → dùng shaped reward với rating thực
    reward = proxy_reward_amazon(slate, step.item_id, stars=step.reward * 5)
elif step.reward is not None:
    # Non-amazon (RetailRocket, ML-1M): giữ raw reward
    reward = float(step.reward) if step.item_id in slate else 0.0
elif dataset_type == "amazon":
    # Legacy: không có target rating → proxy bằng last history star
    stars = step.history_extras[-1] if step.history_extras else None
    reward = proxy_reward_amazon(slate, step.item_id, stars * 5 if stars else None)
```

Khi build trajectories từ 5-core data, `step.reward = target_rating/5` → shaped reward đúng.

**Kết luận — giữ shaped reward hay đổi sang binary?**

Shaped reward (hiện tại, sau khi fix bug) hợp lý vì:
1. Rating-weighted ưu tiên recommend item chất lượng cao
2. Rank bonus tạo gradient để model học đặt target lên đầu slate
3. Miss penalty (-0.01) ngăn pure-zero gradients

**Không cần đổi sang binary** — fix bug là đủ.

---

## 10. BERT4Rec Pipeline — Implementation & Optimization

### 10.1 Files được tạo mới

| File | Mô tả |
|------|-------|
| `retrieval_models/bert4rec/train_5core.py` | BERT4Rec trainer đọc `reviews_5core.json.gz`, leave-last-2-out split, eval recall@K |
| `retrieval/bert4rec_retriever.py` | Wrapper BERT4Rec+FAISS, sequential + **batched** search interface |
| `retrieval/bert4rec_pipeline.py` | Drop-in thay `UnifiedPipeline`, dùng BERT4Rec thay BM25+Reranker, có `batch_evaluate()` |
| `run_beauty_bert4rec.py` | End-to-end runner: BERT4Rec pretraining → RL training → eval |

### 10.2 Tại sao bỏ BM25 + Qwen3-Reranker?

3 vấn đề cốt lõi với BM25+Reranker trên Amazon 5-core:
1. **Không có product metadata** — `build_meta_from_reviews()` dùng review summaries làm "title" → BM25 query meaningless
2. **Train/eval distribution mismatch** — training dùng `fast_mode=True` (BM25 scores), eval dùng Qwen3-Reranker → RL policy train trên distribution A, eval trên distribution B → hit@10=0
3. **Tốc độ** — Qwen3-Reranker 0.6B: ~2s/query → không thể train online

BERT4Rec:
- Không cần text metadata, chỉ cần interaction history
- Train/eval cùng scorer (cosine similarity) → không mismatch
- ~5ms/query (400x nhanh hơn Qwen3-Reranker)

### 10.3 Indexing convention — BUG đã fix

**Bug**: `RerankerBackedSubmodular(num_items=item_num)` tạo `Embedding(item_num)` với valid indices `0..item_num-1`. Nhưng BERT4Rec dùng **1-indexed** item_id (1..item_num), nên index `item_num` là out-of-bounds → CUDA assertion error.

**Fix**: `num_items=item_num + 1` để accommodate 1-indexed ids. `StateEncoder` đã làm đúng (`Embedding(num_items+1, ..., padding_idx=0)`), submodular cần nhất quán.

```python
# ĐÚNG:
submodular = RerankerBackedSubmodular(num_items=item_num + 1, ...)  # index 0=padding, 1..item_num=real
state_encoder = StateEncoder(num_items=item_num, ...)               # StateEncoder đã tự +1 bên trong
```

### 10.4 GPU Optimization — Profiling results (RTX 3090, 23GB)

**BERT4Rec training batch size** (hidden_dim=128, maxlen=50, 2 blocks):

| Batch | VRAM | ms/batch | Ghi chú |
|-------|------|----------|---------|
| 512 | 0.87GB | 325ms | CUDA warmup lần đầu |
| 1024 | **2.38GB** | **33ms** | **Sweet spot** ✓ |
| 2048 | 7.33GB | 77ms | Tốn VRAM, logit matrix (M,M) lớn |
| 4096 | OOM | — | InfoNCE (M,M) ~ 20K² floats = 1.6GB |

Lý do InfoNCE tốn memory: logit matrix `(M, M)` trong `infonce_loss()`, với M = số masked positions trong batch. M ≈ B × maxlen × mask_prob. Tại B=1024: M ≈ 10K → matrix 400MB.

**Eval bottleneck — sequential vs batched retrieval:**

| Mode | ms/query | 22K users | Ghi chú |
|------|----------|-----------|---------|
| Sequential `search_by_history()` | 1.80ms | 40s/epoch | Current default |
| Batched `batch_search_by_history()` | **0.04ms** | **0.9s/epoch** | **45x speedup** |
| Greedy selector (unavoidable) | 1.68ms | 37s/epoch | Không thể batch |

**Kết luận**: User embedding + FAISS có thể batch 45x faster, nhưng greedy selector vẫn là bottleneck (sequential, per-user). Tổng eval vẫn ~38s/epoch với 22K users.

**Giải pháp thực tế**: `--eval_steps 2000` → cap eval tại 2000 users = 3.4s/epoch.

### 10.5 Estimated runtime — Full Beauty run (22K users)

| Component | Time |
|-----------|------|
| BERT4Rec pretraining (10 epochs, B=1024) | ~1 phút |
| RL training (100 epochs × 2000 steps) | ~18 phút |
| Eval (100 epochs × 2000 users) | ~6 phút |
| **Tổng** | **~25 phút** |

### 10.6 Command chuẩn cho full run

```bash
tmux new -s beauty100

python run_beauty_bert4rec.py --dataset beauty \
    --bert4rec_epochs 10 \
    --bert4rec_batch 1024 \
    --bert4rec_dim 128 \
    --epochs 100 \
    --steps_per_epoch 2000 \
    --eval_steps 2000 \
    --batch_size 256 \
    --buffer_size 50000 \
    --min_buffer 1000 \
    --n_retrieve 200 \
    --device cuda \
    2>&1 | tee output_beauty_bert4rec/run_100ep.log

# Detach: Ctrl+B D
# Attach lại: tmux attach -t beauty100
# Xem log: tail -f output_beauty_bert4rec/run_100ep.log
```

### 10.7 Smoke test kết quả (500 users, 3 BERT4Rec epochs, 2 RL epochs)

```
BERT4Rec recall@50 = 0.006  (500 users — quá ít để học 12K items)
RL epoch 1: (warmup → losses bắt đầu)
RL epoch 2: rl/critic_loss=0.41  actor_loss=0.73  sub/total=0.65
Val hit@10 = 0.0020  coverage=0.054  ILD=1.047
Test hit@10 = 0.0000  (expected — BERT4Rec recall quá thấp với 500 users)
```

Pipeline chạy không lỗi. Với full 22K users + 10 BERT4Rec epochs, BERT4Rec recall@50 dự kiến ~10-20%.

---

## 11. Phân tích training dynamics — Beauty 100 epoch

### 11.1 Run 1: gamma=0.99, inject_target=True (FAILED — diverge)

**Log**: `output_beauty_bert4rec/run_100ep.log`

**Quan sát (CLAIM — có log làm bằng chứng):**
- critic_loss tăng mũ không dừng: ep9=49 → ep14=628 → ep20=4017
- actor_loss âm và tăng dần magnitude: -0.14 → -7.14
- val hit@10 stuck 0.0035–0.0050 suốt 20 epoch

**Nguyên nhân (CLAIM):**
- inject_target → training hit_rate ~91% → reward trung bình ≈ 0.91
- V* = r/(1-γ) = 0.91/0.01 = **91** — quá lớn
- Critic bootstrap target tăng nhanh hơn critic có thể học (gradient clipping 1.0 giới hạn step size)
- Kết quả: divergence thật, không phải bootstrapping bình thường

**Kết luận**: gamma=0.99 không phù hợp với reward scale hiện tại khi có inject_target.

---

### 11.2 Run 2: gamma=0.9, inject_target=True (PARTIAL — actor collapse)

**Log**: `output_beauty_bert4rec/run_100ep_g09.log`
**Thay đổi**: `--gamma 0.9` (default mới trong run_beauty_bert4rec.py)

**Quan sát (CLAIM — có log làm bằng chứng):**
- critic_loss không diverge: peak ở ep~55 (~110), giảm dần về ~70 ở ep80 ✓
- actor pg_loss: oscillate gần 0 suốt từ ep1 (-0.01 đến +0.08)
- sub/reinforce_loss: flat 0.31–0.32 suốt 80 epoch (không học)
- val hit@10: stuck 0.003–0.005

**Cơ chế actor collapse (CLAIM):**
```
inject_target → reward ≈ 0.91 uniform
critic hội tụ → V_pred → V* ≈ 9.1 cho mọi state
advantage = target - V_pred ≈ 9.1 - 9.1 ≈ 0
pg_loss = -mean(log_prob × advantage) ≈ 0
→ actor không nhận gradient có nghĩa
```

**Kết luận**: gamma=0.9 fix divergence nhưng lộ ra vấn đề sâu hơn: inject_target tạo reward uniform → advantages ≈ 0 → actor/submodular không học được.

---

### 11.3 Run 3: gamma=0.9, inject_target=False + advantage normalization (EXPERIMENT)

**Log**: `output_beauty_bert4rec/run_100ep_noinject.log`

**Thay đổi code:**
1. `unified_trainer.py`: `inject_target=True` → `inject_target=False`
2. `unified_pipeline.py`: thêm advantage normalization:
   ```python
   advantages = targets - v_pred.detach()
   advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
   ```

**Hypothesis (EXPERIMENT — chưa có bằng chứng):**
- Training hit_rate giảm xuống ~35-40% (BERT4Rec recall@200 thực)
- Reward không còn uniform → advantages có variance thực → actor học được
- Advantage normalization đảm bảo gradient ổn định dù reward sparse
- V* = 0.4 × 0.91 / 0.1 ≈ 3.6 — critic converge nhanh hơn
- val hit@10 tăng rõ so với Run 2 (nếu hypothesis đúng)

**Nếu val hit@10 vẫn stuck**: bottleneck là BERT4Rec recall@10 ceiling, không phải RL.
**Nếu val hit@10 tăng**: inject_target là nguyên nhân chính gây actor collapse.

**Command:**
```bash
python run_beauty_bert4rec.py --dataset beauty \
    --skip_bert4rec \
    --bert4rec_batch 1024 --bert4rec_dim 128 \
    --epochs 100 --steps_per_epoch 2000 --eval_steps 2000 \
    --batch_size 1024 --buffer_size 200000 --min_buffer 2000 \
    --n_retrieve 200 --device cuda \
    2>&1 | tee output_beauty_bert4rec/run_100ep_noinject.log
```
