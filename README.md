# SRS — Submodularity-aided RL for Product Slate Recommendation

Three-stage recommendation pipeline:

| Stage | Component | Role |
|-------|-----------|------|
| 1 | **ICSRec-SAS** (frozen) | Sequential dense retriever; returns top-m candidates via FAISS |
| 2 | *(dense retriever — no separate reranker)* | ICSRec similarity scores used directly as r_u(i) |
| 3 | **Submodular RL** (trained) | Actor-critic policy learns per-user (α_t, κ_t); submodular greedy builds slate of k items |

The RL action is 2-dimensional:
- **α_t** — relevance-diversity trade-off weight in the submodular objective F_θ(S | α_t)
- **κ_t** — training-time exploration intensity (ε-greedy + softmax temperature)

## Requirements

```bash
pip install -r requirements.txt
# ICSRec repo must be at /workspace/repos/ICSRec
# ICSRec data must be at /workspace/repos/ICSRec/data/
```

## How to Run

```bash
# Smoke test (CPU, ~2 min)
python run_beauty_icsr.py \
    --max_users 200 \
    --epochs 1 \
    --steps_per_epoch 50 \
    --device cpu

# Full run (GPU, 100 epochs)
python run_beauty_icsr.py \
    --epochs 100 \
    --steps_per_epoch 500 \
    --device cuda \
    2>&1 | tee run_beauty.log

# Resume with a specific ICSRec checkpoint
python run_beauty_icsr.py \
    --epochs 100 \
    --ckpt_path /workspace/repos/ICSRec/src/output/ICSRec-SAS-Beauty-ep80.pt \
    --output_dir output_beauty_ep80 \
    --device cuda

# Evaluate best checkpoint only (no training)
python run_beauty_icsr.py --epochs 0 --output_dir <checkpoint_dir> --device cuda
```

### Key arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--ckpt_path` | ICSRec default | Override ICSRec checkpoint path |
| `--output_dir` | `output_beauty_icsr` | Directory for checkpoints + results |
| `--epochs` | `10` | Training epochs (0 = eval only) |
| `--steps_per_epoch` | `500` | Sampled training steps per epoch |
| `--n_retrieve` | `200` | Candidate pool size m (FAISS top-m) |
| `--slate_size` | `10` | Final slate size k |
| `--history_length` | `20` | User history window h |
| `--actor_alpha_bias` | `None` | Init actor α-head bias to logit(value) |
| `--fixed_alpha` | `None` | Bypass RL α entirely, use fixed value |
| `--device` | `cuda` | `cpu` or `cuda` |

## Folder Structure

```
LM_Submodular_RL/
├── run_beauty_icsr.py              # Main entry point
│
├── retrieval/
│   ├── icsr_retriever.py           # ICSRec-SAS + FAISS retriever (Stage 1)
│   ├── two_stage_pipeline.py       # Generic two-stage pipeline (any retriever)
│   ├── unified_pipeline.py         # RL policy (UnifiedRLPolicy) + typed containers
│   │                               #   ScoredCandidate, UnifiedSearchResult
│   ├── bert4rec_retriever.py       # BERT4Rec ANN retriever (alternative Stage 1)
│   └── bm25_retriever.py           # BM25 retriever (alternative Stage 1)
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
├── tests/
│   ├── test_two_stage_pipeline.py  # Smoke tests for TwoStageRLPipeline
│   └── test_amazon_weights.py
│
└── stuff/                          # Experiment outputs (gitignored)
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

Each run writes to `output_dir/` (stored under `stuff/`, which is gitignored):

```
stuff/output_beauty_icsr/
├── best_unified.pt     # Best validation checkpoint (submodular + encoder + RL)
└── results.json        # Final test metrics: HR@k, NDCG@k, MRR@k, ILD, Coverage
```
## Checkpoint
Save at huggingface: https://huggingface.co/mrr1ha/SRS-beauty-checkpoints

## Results (Amazon Beauty 2014 5-core, k=10)

| Method | HR@10 | NDCG@10 | MRR@10 | ILD | Coverage |
|--------|-------|---------|--------|-----|----------|
| ICSRec top-10 greedy *(same pipeline)* | 0.0959 | — | — | 0.5126 | — |
| **SRS (ours)** | 0.0899 | **0.0522** | **0.0407** | **0.5622** | **0.844** |

*Full-catalogue baselines (different protocol): SASRec 0.0624, BERT4Rec 0.0601, ICSRec paper 0.0963.*

## Key Design Notes

**Why no reranker (Stage 2)?**  
ICSRec-SAS is a dense retriever trained with a full-softmax objective, so its inner-product similarity scores are already calibrated relevance estimates. Stage 2 is collapsed into Stage 1.

**Pluggable retriever.**  
`TwoStageRLPipeline` accepts any retriever that implements `search_by_history` and `batch_search_by_history`. Swap `ICSRecRetriever` for `BERT4RecRetriever` or `BM25Retriever` without changing the RL/submodular code.

**Target injection during training.**  
`collect_transition` optionally injects the target item into the candidate pool at `top_score + 0.01` to ensure the hit signal is reachable even when retrieval recall is low.
