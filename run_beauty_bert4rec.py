"""
End-to-end pipeline: BERT4Rec → Submodular+RL
Cho Amazon 2014 5-core datasets (Beauty / Sports / Toys)

Steps:
  1. Load 5-core data (reviews_5core.json.gz) → asin2iid, user_seqs
  2. Train BERT4Rec (hoặc load từ checkpoint)
  3. Build BERT4RecRetriever + BERT4RecPipeline
  4. Build trajectory steps từ user_seqs
  5. Train Submodular+RL (UnifiedJointTrainer)
  6. Evaluate val / test → print metrics

Usage:
  # Smoke test (1 epoch BERT4Rec + 1 epoch RL, 200 users)
  python run_beauty_bert4rec.py --dataset beauty --max_users 200 \\
      --bert4rec_epochs 2 --epochs 1 --steps_per_epoch 50 --device cuda

  # Full Beauty run
  python run_beauty_bert4rec.py --dataset beauty \\
      --bert4rec_epochs 10 --epochs 5 --device cuda

  # Skip BERT4Rec training (dùng checkpoint có sẵn)
  python run_beauty_bert4rec.py --dataset beauty --skip_bert4rec \\
      --bert4rec_dir retrieval_models/bert4rec/output_beauty \\
      --epochs 5 --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

RECSYS_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(RECSYS_DIR))

DATASET_PATHS = {
    "beauty": "/workspace/datasets/beauty_2014/reviews_5core.json.gz",
    "sports": "/workspace/datasets/sports_2014/reviews_5core.json.gz",
    "toys":   "/workspace/datasets/toys_2014/reviews_5core.json.gz",
}

DATASET_SIZES = {
    "beauty": (22_363, 12_101),
    "sports": (35_598, 18_357),
    "toys":   (19_412, 11_924),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BERT4Rec → Submodular+RL pipeline on Amazon 2014 5-core"
    )
    p.add_argument("--dataset", default="beauty", choices=list(DATASET_PATHS))
    p.add_argument("--review_path", default=None, help="Override review file path")
    p.add_argument("--output_dir",  default=None, help="Default: output_<dataset>_bert4rec")
    p.add_argument("--max_users",   type=int, default=None)

    # BERT4Rec training
    p.add_argument("--bert4rec_dir",    default=None,
                   help="Pre-trained BERT4Rec dir. Default: retrieval_models/bert4rec/output_<dataset>")
    p.add_argument("--skip_bert4rec",   action="store_true",
                   help="Skip BERT4Rec training (requires --bert4rec_dir with checkpoint)")
    p.add_argument("--bert4rec_epochs", type=int,   default=10)
    p.add_argument("--bert4rec_batch",  type=int,   default=512)
    p.add_argument("--bert4rec_lr",     type=float, default=1e-3)
    p.add_argument("--bert4rec_maxlen", type=int,   default=50)
    p.add_argument("--bert4rec_dim",    type=int,   default=64)
    p.add_argument("--bert4rec_temp",   type=float, default=0.07)

    # Retrieval
    p.add_argument("--n_retrieve",      type=int, default=200, help="Top-K from BERT4Rec")
    p.add_argument("--slate_size",      type=int, default=10)
    p.add_argument("--history_length",  type=int, default=20, help="StateEncoder history length")

    # RL training
    p.add_argument("--epochs",          type=int,   default=5)
    p.add_argument("--steps_per_epoch", type=int,   default=500)
    p.add_argument("--eval_steps",      type=int,   default=None)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--buffer_size",     type=int,   default=10_000)
    p.add_argument("--min_buffer",      type=int,   default=64)
    p.add_argument("--log_every",       type=int,   default=50)

    # Optimiser
    p.add_argument("--lr_rl",       type=float, default=3e-4)
    p.add_argument("--lr_sub",      type=float, default=1e-3)
    p.add_argument("--lr_encoder",  type=float, default=1e-3)
    p.add_argument("--gamma",       type=float, default=0.9)
    p.add_argument("--lambda_sub",  type=float, default=0.5)
    p.add_argument("--lambda_rank", type=float, default=0.1)
    p.add_argument("--alpha_init",  type=float, default=0.7)

    # Misc
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",   type=int, default=42)

    return p.parse_args()


# ===========================================================================
# Trajectory building từ user_seqs (leave-last-2-out)
# ===========================================================================

def build_trajectories_5core(
    user_seqs: Dict[str, List[int]],
    split: str,
    history_length: int = 20,
    slate_size: int = 10,
    target_ratings: Optional[Dict[str, Dict[int, float]]] = None,
):
    """
    Build TrajectoryStep list từ user_seqs (5-core leave-last-2-out split).

    split = "train" : sliding window trên toàn lịch sử (trừ 2 item cuối)
    split = "val"   : history = all[:-2], target = item thứ 2 từ cuối
    split = "test"  : history = all[:-1], target = item cuối

    step.reward = target_rating / 5 nếu có, else None
    """
    from algorithms.trajectory_builder import TrajectoryStep

    steps = []
    for uid, seq in user_seqs.items():
        if len(seq) < 3:
            continue

        if split == "test":
            history = seq[:-1][-history_length:]
            target  = seq[-1]
            t_rating = None
            if target_ratings:
                t_rating = target_ratings.get(uid, {}).get(target)
            steps.append(TrajectoryStep(
                user_id=uid, item_id=target,
                history_ids=history,
                budget=float(slate_size),
                split="test",
                reward=t_rating / 5.0 if t_rating else None,
            ))

        elif split == "val":
            history = seq[:-2][-history_length:]
            target  = seq[-2]
            t_rating = None
            if target_ratings:
                t_rating = target_ratings.get(uid, {}).get(target)
            steps.append(TrajectoryStep(
                user_id=uid, item_id=target,
                history_ids=history,
                budget=float(slate_size),
                split="val",
                reward=t_rating / 5.0 if t_rating else None,
            ))

        else:  # train — sliding window
            train_seq = seq[:-2]
            for end in range(2, len(train_seq) + 1):
                hist   = train_seq[max(0, end - history_length): end - 1]
                target = train_seq[end - 1]
                t_rating = None
                if target_ratings:
                    t_rating = target_ratings.get(uid, {}).get(target)
                steps.append(TrajectoryStep(
                    user_id=uid, item_id=target,
                    history_ids=hist,
                    budget=float(slate_size),
                    split="train",
                    reward=t_rating / 5.0 if t_rating else None,
                ))

    return steps


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    review_path = args.review_path or DATASET_PATHS[args.dataset]
    output_dir  = Path(args.output_dir or f"output_{args.dataset}_bert4rec")
    bert4rec_dir = Path(args.bert4rec_dir or f"retrieval_models/bert4rec/output_{args.dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_users, expected_items = DATASET_SIZES[args.dataset]
    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset.upper()} 2014 5-core")
    print(f"  Expected: {expected_users:,} users, {expected_items:,} items")
    print(f"  Review:   {review_path}")
    print(f"  Output:   {output_dir}")
    print(f"  Device:   {device}")
    if args.max_users:
        print(f"  [LIMITED] max_users={args.max_users} (smoke-test mode)")
    print(f"{'='*60}")

    # ──────────────────────────────────────────────────────────────────────
    # Step 1: Load 5-core data
    # ──────────────────────────────────────────────────────────────────────
    print(f"\nStep 1: Loading {args.dataset} 5-core data...")
    from retrieval_models.bert4rec.train_5core import load_5core, split_sequences

    asin2iid, item_num, user_seqs = load_5core(review_path, args.max_users)
    print(f"  Loaded: {len(user_seqs):,} users, {item_num:,} items")

    # id_map: str(1-indexed item_idx) → int — BERT4RecRetriever stores str(item_idx)
    id_map = {str(i): i for i in range(1, item_num + 1)}

    # ──────────────────────────────────────────────────────────────────────
    # Step 2: Train BERT4Rec (or load checkpoint)
    # ──────────────────────────────────────────────────────────────────────
    print(f"\nStep 2: BERT4Rec retriever")

    if args.skip_bert4rec and (bert4rec_dir / "best_model.pt").exists():
        print(f"  Skipping training — loading from {bert4rec_dir}")
    else:
        if args.skip_bert4rec:
            print(f"  WARNING: --skip_bert4rec set but no checkpoint at {bert4rec_dir}. Training...")

        print(f"  Training BERT4Rec ({args.bert4rec_epochs} epochs, batch={args.bert4rec_batch})")
        bert4rec_dir.mkdir(parents=True, exist_ok=True)

        # Build train/val/test numpy arrays for BERT4Rec training
        from retrieval_models.bert4rec.train_5core import (
            split_sequences, BERT4RecDataset5Core,
            build_item_embs, evaluate_recall
        )
        from torch.utils.data import DataLoader
        import torch.nn as nn
        from retrieval_models.bert4rec.model import BERT4Rec
        from tqdm import tqdm
        import time

        pre_dir = bert4rec_dir / "preprocessed"
        pre_dir.mkdir(exist_ok=True)
        seqs_cache = pre_dir / f"splits_L{args.bert4rec_maxlen}.npz"

        if seqs_cache.exists():
            print("  Loading cached splits...", flush=True)
            d = np.load(seqs_cache)
            train_seqs, train_targets = d["train_seqs"], d["train_targets"]
            val_seqs,   val_targets   = d["val_seqs"],   d["val_targets"]
            test_seqs,  test_targets  = d["test_seqs"],  d["test_targets"]
        else:
            (train_seqs, train_targets,
             val_seqs,   val_targets,
             test_seqs,  test_targets) = split_sequences(user_seqs, args.bert4rec_maxlen)
            np.savez_compressed(
                seqs_cache,
                train_seqs=train_seqs, train_targets=train_targets,
                val_seqs=val_seqs,     val_targets=val_targets,
                test_seqs=test_seqs,   test_targets=test_targets,
            )

        print(f"  Train: {len(train_targets):,}  Val: {len(val_targets):,}  Test: {len(test_targets):,}")

        b4r_model = BERT4Rec(
            item_num=item_num,
            maxlen=args.bert4rec_maxlen,
            hidden_dim=args.bert4rec_dim,
            num_heads=2,
            num_blocks=2,
            dropout_rate=0.2,
            mask_prob=0.2,
        ).to(device)
        n_params = sum(p.numel() for p in b4r_model.parameters())
        print(f"  BERT4Rec params={n_params:,}", flush=True)

        dataset_b4r = BERT4RecDataset5Core(
            train_seqs, train_targets, item_num,
            mask_prob=0.2, mask_token=b4r_model.mask_token,
        )
        loader_b4r = DataLoader(
            dataset_b4r, batch_size=args.bert4rec_batch, shuffle=True,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=(device.type == "cuda"), persistent_workers=True,
        )
        optimizer_b4r = torch.optim.Adam(b4r_model.parameters(),
                                          lr=args.bert4rec_lr, betas=(0.9, 0.98))

        best_r = 0.0
        for epoch in range(1, args.bert4rec_epochs + 1):
            b4r_model.train()
            total_loss = 0.0
            t0 = time.perf_counter()
            pbar = tqdm(loader_b4r, desc=f"  BERT4Rec epoch {epoch}/{args.bert4rec_epochs}",
                        unit="batch", dynamic_ncols=True)
            for masked_seqs, pos_ids in pbar:
                masked_seqs = masked_seqs.to(device)
                pos_ids     = pos_ids.to(device)
                loss = b4r_model.infonce_loss(masked_seqs, pos_ids, temperature=args.bert4rec_temp)
                if loss == 0:
                    continue
                optimizer_b4r.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(b4r_model.parameters(), 5.0)
                optimizer_b4r.step()
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = total_loss / max(len(loader_b4r), 1)
            print(f"\n  Epoch {epoch}  loss={avg_loss:.4f}  {time.perf_counter()-t0:.0f}s", flush=True)

            item_embs_b4r = build_item_embs(b4r_model, item_num, device)
            val_rec = evaluate_recall(b4r_model, item_embs_b4r, val_seqs, val_targets,
                                       [10, 20, 50], device)
            print("  Val  Recall: " + "  ".join(f"@{k}={v:.4f}" for k, v in sorted(val_rec.items())), flush=True)

            r50 = val_rec.get(50, 0.0)
            if r50 > best_r:
                best_r = r50
                torch.save({
                    "epoch": epoch, "model_state": b4r_model.state_dict(),
                    "args": {
                        "hidden_dim": args.bert4rec_dim, "maxlen": args.bert4rec_maxlen,
                        "num_heads": 2, "num_blocks": 2, "dropout": 0.2, "mask_prob": 0.2,
                    },
                    "item_num": item_num,
                }, bert4rec_dir / "best_model.pt")
                print(f"  ✓ Best recall@50={best_r:.4f} saved", flush=True)
            del item_embs_b4r

        # Build FAISS index from best checkpoint
        print("\n  Building FAISS index...", flush=True)
        ckpt_b4r = torch.load(bert4rec_dir / "best_model.pt", map_location=device, weights_only=True)
        b4r_model.load_state_dict(ckpt_b4r["model_state"])
        item_embs_b4r = build_item_embs(b4r_model, item_num, device)
        try:
            import faiss
            arr   = item_embs_b4r.cpu().numpy().astype(np.float32)
            index = faiss.IndexFlatIP(arr.shape[1])
            index.add(arr)
            faiss.write_index(index, str(bert4rec_dir / "item_faiss.index"))
            print(f"  FAISS saved ({(bert4rec_dir/'item_faiss.index').stat().st_size/1e6:.0f} MB)", flush=True)
        except ImportError:
            np.save(bert4rec_dir / "item_embs.npy", item_embs_b4r.cpu().numpy())
            print("  item_embs.npy saved (faiss not installed)", flush=True)
        del item_embs_b4r

    # ──────────────────────────────────────────────────────────────────────
    # Step 3: Build BERT4RecRetriever + Pipeline
    # ──────────────────────────────────────────────────────────────────────
    print(f"\nStep 3: Building BERT4RecRetriever...")
    from retrieval.bert4rec_retriever import BERT4RecRetriever
    from retrieval.bert4rec_pipeline import BERT4RecPipeline
    from retrieval.unified_pipeline import UnifiedRLPolicy
    from models.submodular import RerankerBackedSubmodular
    from utils.encoders import StateEncoder

    retriever = BERT4RecRetriever(
        model_dir=str(bert4rec_dir),
        item_num=item_num,
        device=device,
        id_map_inv={i: str(i) for i in range(1, item_num + 1)},
    )

    embed_dim  = 128
    # +1 vì item_idx là 1-indexed (1..item_num); index 0 = padding không dùng.
    # StateEncoder đã làm Embedding(num_items+1) sẵn; submodular cần làm tương tự.
    submodular = RerankerBackedSubmodular(
        num_items=item_num + 1, embed_dim=64, alpha_init=args.alpha_init,
    ).to(device)
    state_encoder = StateEncoder(num_items=item_num, embed_dim=embed_dim).to(device)
    rl_policy     = UnifiedRLPolicy(
        state_dim=embed_dim, hidden_dim=256, lr=args.lr_rl, gamma=args.gamma,
    ).to(device)

    pipeline = BERT4RecPipeline(
        retriever=retriever,
        submodular=submodular,
        rl_policy=rl_policy,
        state_encoder=state_encoder,
        id_map=id_map,
        device=device,
        n_retrieve=args.n_retrieve,
        slate_size=args.slate_size,
        history_length=args.history_length,
    )

    total_params = sum(p.numel() for p in (
        list(submodular.parameters()) +
        list(state_encoder.parameters()) +
        list(rl_policy.parameters())
    ))
    print(f"  Trainable (submodular+encoder+RL): {total_params:,}")
    print(f"  n_retrieve={args.n_retrieve}  slate_size={args.slate_size}  history_length={args.history_length}")

    # ──────────────────────────────────────────────────────────────────────
    # Step 4: Build trajectories
    # ──────────────────────────────────────────────────────────────────────
    print(f"\nStep 4: Building trajectory steps (leave-last-2-out)...")

    # Load target ratings for reward shaping
    import gzip
    from collections import defaultdict
    target_ratings: Dict[str, Dict[int, float]] = defaultdict(dict)
    opener = gzip.open if review_path.endswith(".gz") else open
    with opener(review_path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line.strip())
                uid  = r.get("reviewerID", "")
                asin = r.get("asin", "")
                rating = float(r.get("overall", 3.0))
                if uid and asin in asin2iid:
                    target_ratings[uid][asin2iid[asin]] = rating
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    train_steps = build_trajectories_5core(
        user_seqs, "train", args.history_length, args.slate_size, target_ratings)
    val_steps   = build_trajectories_5core(
        user_seqs, "val",   args.history_length, args.slate_size, target_ratings)
    test_steps  = build_trajectories_5core(
        user_seqs, "test",  args.history_length, args.slate_size, target_ratings)
    print(f"  Train: {len(train_steps):,}  Val: {len(val_steps):,}  Test: {len(test_steps):,}")

    # dummy query fn (BERT4Rec ignores query text)
    make_query = lambda step: ""

    # ──────────────────────────────────────────────────────────────────────
    # Step 5: Train RL + Submodular
    # ──────────────────────────────────────────────────────────────────────
    print(f"\nStep 5: Training Submodular+RL ({args.epochs} epochs)...")
    from algorithms.unified_trainer import UnifiedJointTrainer

    trainer = UnifiedJointTrainer(
        pipeline=pipeline,
        rl_policy=rl_policy,
        submodular=submodular,
        state_encoder=state_encoder,
        lambda_sub=args.lambda_sub,
        lambda_rank=args.lambda_rank,
        lr_sub=args.lr_sub,
        lr_encoder=args.lr_encoder,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        min_buffer=args.min_buffer,
        gamma=args.gamma,
        device=device,
    )

    best_hit  = 0.0
    ckpt_path = output_dir / "best_unified.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")
        losses = trainer.train_epoch(
            trajectory_steps=train_steps,
            pipeline_query_fn=make_query,
            steps_per_epoch=args.steps_per_epoch,
            log_every=args.log_every,
            dataset_type="amazon",
            fast_mode=False,   # BERT4Rec is already fast — no reranker to skip
        )

        eval_steps = val_steps[:args.eval_steps] if args.eval_steps else val_steps
        val_metrics = trainer.evaluate(
            eval_steps=eval_steps,
            pipeline_query_fn=make_query,
            dataset_type="amazon",
        )

        loss_str = "  ".join(f"{k}={v:.4f}" for k, v in losses.items()) or "(warmup)"
        print(f"  Losses: {loss_str}")
        print(f"  Val hit@{args.slate_size}={val_metrics['hit@k']:.4f}  "
              f"ndcg@{args.slate_size}={val_metrics['ndcg@k']:.4f}  "
              f"mrr@{args.slate_size}={val_metrics.get('mrr@k', 0.0):.4f}  "
              f"coverage={val_metrics['coverage']:.4f}  "
              f"n={val_metrics['n_samples']}")

        if val_metrics["hit@k"] > best_hit:
            best_hit = val_metrics["hit@k"]
            torch.save({
                "submodular":     submodular.state_dict(),
                "state_encoder":  state_encoder.state_dict(),
                "rl_actor":       rl_policy.actor.state_dict(),
                "rl_critic":      rl_policy.critic.state_dict(),
                "epoch":          epoch,
                "best_hit":       best_hit,
            }, ckpt_path)
            print(f"  *** New best hit@{args.slate_size}={best_hit:.4f} saved ***")

    # ──────────────────────────────────────────────────────────────────────
    # Step 6: Final evaluation on TEST set
    # ──────────────────────────────────────────────────────────────────────
    print(f"\nStep 6: Final evaluation on TEST set...")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        submodular.load_state_dict(ckpt["submodular"])
        state_encoder.load_state_dict(ckpt["state_encoder"])
        rl_policy.actor.load_state_dict(ckpt["rl_actor"])
        rl_policy.critic.load_state_dict(ckpt["rl_critic"])
        print(f"  Loaded best checkpoint (epoch {ckpt['epoch']}, val hit@k={ckpt['best_hit']:.4f})")

    eval_test = test_steps[:args.eval_steps] if args.eval_steps else test_steps
    test_metrics = trainer.evaluate(
        eval_steps=eval_test,
        pipeline_query_fn=make_query,
        dataset_type="amazon",
    )

    # ILD
    all_slates = trainer.metrics.all_slates
    emb_weight = submodular.item_emb.weight.detach().cpu()
    from utils.metrics import diversity_score
    ild_scores = [diversity_score(s, emb_weight) for s in all_slates if len(s) >= 2]
    ild = float(np.mean(ild_scores)) if ild_scores else 0.0

    print(f"\n{'='*60}")
    print(f"FINAL TEST RESULTS  [{args.dataset.upper()} 2014 5-core]  (k={args.slate_size})")
    print(f"{'='*60}")
    print(f"  Hit@{args.slate_size}      = {test_metrics['hit@k']:.4f}")
    print(f"  NDCG@{args.slate_size}     = {test_metrics['ndcg@k']:.4f}")
    print(f"  MRR@{args.slate_size}      = {test_metrics.get('mrr@k', 0.0):.4f}")
    print(f"  Coverage   = {test_metrics['coverage']:.4f}")
    print(f"  ILD        = {ild:.4f}")
    print(f"  N samples  = {test_metrics['n_samples']}")
    print(f"{'='*60}")

    # Baseline reference
    baselines = {
        "beauty": {"SASRec": 0.0624, "BERT4Rec": 0.0601},
        "sports": {"SASRec": 0.0333, "BERT4Rec": 0.0359},
        "toys":   {"SASRec": 0.0652, "BERT4Rec": 0.0524},
    }
    if args.dataset in baselines:
        print(f"\n  Comparison (Hit@10):")
        for name, val in baselines[args.dataset].items():
            diff = test_metrics["hit@k"] - val
            sign = "+" if diff >= 0 else ""
            print(f"    {name:10s}: {val:.4f}  (ours: {sign}{diff:.4f})")

    results = {
        "dataset": args.dataset,
        "item_num": item_num,
        "n_users": len(user_seqs),
        "retriever": "BERT4Rec",
        "n_retrieve": args.n_retrieve,
        "slate_size": args.slate_size,
        "test_hit_k": test_metrics["hit@k"],
        "test_ndcg_k": test_metrics["ndcg@k"],
        "test_mrr_k": test_metrics.get("mrr@k", 0.0),
        "test_coverage": test_metrics["coverage"],
        "test_ild": ild,
        "n_samples": test_metrics["n_samples"],
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {output_dir}/results.json")


if __name__ == "__main__":
    main()
