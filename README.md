# SRS — Submodularity-aided RL for Product Slate Recommendation

Three-stage recommendation pipeline:

| Stage | Component | Role |
|-------|-----------|------|
| 1 | **ICSRec-SAS** (frozen) | Sequential dense retriever; returns top-m candidates via FAISS |
| 2 | *(dense retriever — no separate reranker)* | ICSRec similarity scores used directly as r_u(i) |
| 3 | **Submodular RL** (trained) | Actor-critic policy learns per-user (α_t, η_t); submodular greedy builds slate of k items |

The RL action is 2-dimensional:
- **α_t** — relevance-diversity trade-off weight in the submodular objective F_θ(S | α_t)
- **η_t** — training-time exploration intensity (ε-greedy + softmax temperature)

## Requirements

```bash
pip install -r requirements.txt
# ICSRec repo must be at /workspace/repos/ICSRec
# ICSRec data must be at /workspace/repos/ICSRec/data/
```

## How to Run

### Beauty (main experiment)

```bash
# Smoke test (CPU, ~2 min)
python run_beauty_icsr.py \
    --dataset beauty \
    --max_users 200 \
    --epochs 1 \
    --steps_per_epoch 50 \
    --device cpu

# Full run (GPU, 100 epochs)
python run_beauty_icsr.py \
    --dataset beauty \
    --epochs 100 \
    --steps_per_epoch 500 \
    --device cuda \
    2>&1 | tee run_beauty.log

# Resume with a specific ICSRec checkpoint
python run_beauty_icsr.py \
    --dataset beauty \
    --epochs 100 \
    --ckpt_path /workspace/repos/ICSRec/src/output/ICSRec-SAS-Beauty-ep80.pt \
    --output_dir output_beauty_ep80 \
    --device cuda
```

### Sports / Toys

```bash
python run_beauty_icsr.py --dataset sports --epochs 100 --device cuda
python run_beauty_icsr.py --dataset toys   --epochs 100 --device cuda
```

### Key arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--dataset` | `beauty` | `beauty` / `sports` / `toys` |
| `--ckpt_path` | ICSRec default | Override ICSRec checkpoint path |
| `--output_dir` | `output_{dataset}_icsr` | Directory for checkpoints + results |
| `--epochs` | `10` | Training epochs |
| `--steps_per_epoch` | `500` | Sampled training steps per epoch |
| `--n_retrieve` | `200` | Candidate pool size m (FAISS top-m) |
| `--slate_size` | `10` | Final slate size k |
| `--history_length` | `20` | User history window h |
| `--device` | `cuda` | `cpu` or `cuda` |

## Folder Structure

```
LM_Submodular_RL/
├── run_beauty_icsr.py              # Main entry point (ICSRec pipeline)
│
├── retrieval/
│   ├── icsr_retriever.py           # ICSRec-SAS + FAISS retriever (Stage 1)
│   ├── unified_pipeline.py         # RL policy (UnifiedRLPolicy) + typed containers
│   │                               #   ScoredCandidate, UnifiedSearchResult
│   ├── bert4rec_pipeline.py        # Alternative pipeline: BERT4Rec retriever
│   │                               #   (same search/collect_transition interface)
│   ├── bert4rec_retriever.py       # BERT4Rec ANN retriever
│   └── bm25_retriever.py           # BM25 retriever (legacy, unused)
│
├── models/
│   ├── submodular.py               # RerankerBackedSubmodular
│   │                               #   F_θ(S) = α·Σr_u(i) − (1−α)·Σκ_θ(i,j)
│   └── rl_policy.py                # Actor / Critic; action dims ALPHA_DIM + KAPPA_DIM
│
├── algorithms/
│   ├── greedy_selector.py          # Vectorized budgeted submodular greedy
│   ├── trajectory_builder.py       # Build TrajectoryStep from dataset splits
│   └── unified_trainer.py          # Joint RL + submodular trainer + evaluator
│
├── utils/
│   ├── encoders.py                 # StateEncoder (GRU + mean-pool), pad_history
│   └── metrics.py                  # HR@k, NDCG@k, MRR@k, ILD, Coverage
│
├── data/                           # Dataset loaders (Amazon 5-core, RetailRocket)
├── retrieval_models/               # BERT4Rec training scripts
└── tests/
```

## Data Dependencies

```
/workspace/repos/ICSRec/
├── src/output/ICSRec-SAS-Beauty-0.pt   # Trained ICSRec checkpoint
└── data/
    ├── Beauty.txt                       # User interaction sequences (ICSRec format)
    ├── beauty_asin2id.json              # ASIN → integer item index
    └── beauty_id2asin.json              # Integer item index → ASIN
```

## Output

Each run writes to `output_dir/`:

```
output_beauty_icsr/
├── best_unified.pt     # Best validation checkpoint (submodular + encoder + RL)
└── results.json        # Final test metrics: HR@k, NDCG@k, MRR@k, ILD, Coverage
```

## Key Design Notes

**Why no reranker (Stage 2)?**  
ICSRec-SAS is a dense retriever trained with a full-softmax objective, so its inner-product similarity scores are already calibrated relevance estimates. Stage 2 is collapsed into Stage 1.

**α_t vs η_t naming.**  
The RL policy action has two dimensions. In `models/rl_policy.py` these are named `ALPHA_DIM` (α_t) and `KAPPA_DIM` (η_t — confusingly named `kappa` in the model to avoid clash with the kernel κ_θ). In pipeline code, the second action dimension is always referred to as `eta_t`.

**Target injection during training.**  
`collect_transition` optionally injects the target item into the candidate pool at `top_score + 0.01` to ensure the hit signal is reachable even when retrieval recall is low (~20–25% at top-200).
