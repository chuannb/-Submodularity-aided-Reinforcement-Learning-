# SRS — Submodularity-aided RL for Product Slate Recommendation

Two-stage recommendation framework:
1. **ICSRec-SAS** (frozen) — sequential dense retriever, top-200 candidates via FAISS
2. **Submodular + RL** (trained) — actor-critic policy learns per-user α_t; greedy submodular selector builds final slate of k items

## Requirements

```bash
pip install -r requirements.txt
# Also needs ICSRec repo at /workspace/repos/ICSRec
# and ICSRec data at /workspace/repos/ICSRec/data/
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

# Full run (GPU, ~20 epochs)
python run_beauty_icsr.py \
    --dataset beauty \
    --epochs 20 \
    --steps_per_epoch 500 \
    --device cuda \
    2>&1 | tee run_beauty.log

# Resume with specific ICSRec checkpoint
python run_beauty_icsr.py \
    --dataset beauty \
    --epochs 20 \
    --ckpt_path /workspace/repos/ICSRec/src/output/ICSRec-SAS-Beauty-ep80.pt \
    --output_dir output_beauty_icsr_ep80 \
    --device cuda
```

### Sports / Toys

```bash
python run_beauty_icsr.py --dataset sports --epochs 20 --device cuda
python run_beauty_icsr.py --dataset toys   --epochs 20 --device cuda
```

### Key arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--dataset` | `beauty` | `beauty` / `sports` / `toys` |
| `--ckpt_path` | ICSRec default | Override ICSRec checkpoint |
| `--output_dir` | `output_{dataset}_icsr` | Where to save checkpoint + results |
| `--epochs` | `10` | Training epochs |
| `--steps_per_epoch` | `500` | Sampled train steps per epoch |
| `--n_retrieve` | `200` | FAISS candidates per user |
| `--slate_size` | `10` | Final slate size k |
| `--history_length` | `20` | User history window |
| `--device` | `cuda` | `cpu` or `cuda` |

## Folder Structure

```
LM_Submodular_RL/
├── run_beauty_icsr.py          # Main entry point (ICSRec pipeline)
│
├── retrieval/
│   ├── icsr_retriever.py       # ICSRec-SAS + FAISS retriever
│   ├── bert4rec_pipeline.py    # Full pipeline: retriever → RL → submodular
│   ├── bert4rec_retriever.py   # BERT4Rec retriever (alternative)
│   ├── bm25_retriever.py       # BM25 retriever (legacy)
│   └── unified_pipeline.py     # RL policy + scored candidate types
│
├── models/
│   └── submodular.py           # RerankerBackedSubmodular: f(S) = α·rel + (1-α)·div
│
├── algorithms/
│   ├── greedy_selector.py      # Vectorized budgeted submodular greedy
│   ├── trajectory_builder.py   # Build TrajectoryStep from dataset
│   └── unified_trainer.py      # Joint RL + submodular trainer + evaluator
│
├── utils/
│   ├── encoders.py             # StateEncoder, pad_history
│   └── metrics.py              # Hit@k, NDCG@k, MRR@k, ILD, Coverage
│
├── data/                       # Dataset loaders (Amazon, RetailRocket)
├── retrieval_models/           # BERT4Rec training scripts
├── tests/
│
├── stuff/                      # Logs, experiment notes (gitignored)
└── output_{dataset}_icsr/      # Checkpoints + results.json (gitignored)
```

## Data Dependencies

```
/workspace/repos/ICSRec/
├── src/output/ICSRec-SAS-Beauty-0.pt   # Trained ICSRec checkpoint
└── data/
    ├── Beauty.txt                       # User sequences (ICSRec format)
    ├── beauty_asin2id.json
    └── beauty_id2asin.json
```

## Output

```
output_beauty_icsr_ep80/
├── best_unified.pt     # Best val checkpoint (submodular + encoder + RL weights)
└── results.json        # Final test metrics
```
