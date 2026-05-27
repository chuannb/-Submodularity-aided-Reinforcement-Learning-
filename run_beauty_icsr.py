"""
End-to-end pipeline: ICSRec-SAS → Submodular+RL
Cho Amazon Beauty 2014 5-core dataset

Thay BERT4Rec retriever bằng ICSRec-SAS (WSDM 2024 Oral):
  - Full-softmax CE loss → item embeddings tốt hơn
  - Recall@200 ~28% (vs ~2% với BERT4Rec InfoNCE)

Steps:
  1. Load ICSRec-format data từ /workspace/repos/ICSRec/data/
  2. Build ICSRecRetriever (FAISS inner-product index)
  3. Build pipeline với Submodular+RL
  4. Build trajectory steps (leave-last-2-out)
  5. Train RL + Submodular
  6. Evaluate val/test

Usage:
  # Smoke test
  python run_beauty_icsr.py --max_users 200 --epochs 1 --steps_per_epoch 50 --device cuda

  # Full run
  python run_beauty_icsr.py --epochs 10 --device cuda
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

ICSREC_DATA   = Path("/workspace/repos/ICSRec/data")
DATASET       = "beauty"
REVIEW_PATH   = "/workspace/datasets/beauty_2014/reviews_5core.json.gz"
DATASET_TXT   = "Beauty"
EXPECTED_USERS, EXPECTED_ITEMS = 22_363, 12_101


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ICSRec-SAS → Submodular+RL pipeline on Amazon Beauty 2014 5-core"
    )
    p.add_argument("--output_dir", default=None, help="Default: output_beauty_icsr")
    p.add_argument("--max_users",  type=int, default=None)

    # ICSRec checkpoint (optional override)
    p.add_argument("--ckpt_path", default=None,
                   help="Override ICSRec checkpoint path (default: ICSRec-SAS-{Dataset}-0.pt)")

    # Retrieval
    p.add_argument("--n_retrieve",     type=int, default=200)
    p.add_argument("--slate_size",     type=int, default=10)
    p.add_argument("--history_length", type=int, default=20)

    # RL training
    p.add_argument("--epochs",          type=int,   default=10)
    p.add_argument("--steps_per_epoch", type=int,   default=500)
    p.add_argument("--eval_steps",      type=int,   default=None)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--buffer_size",     type=int,   default=10_000)
    p.add_argument("--min_buffer",      type=int,   default=64)
    p.add_argument("--log_every",       type=int,   default=50)
    p.add_argument("--test_every",      type=int,   default=10,
                   help="Run TEST evaluation every N epochs (val runs every epoch)")

    # Optimiser
    p.add_argument("--lr_rl",       type=float, default=3e-4)
    p.add_argument("--lr_sub",      type=float, default=1e-3)
    p.add_argument("--lr_encoder",  type=float, default=1e-3)
    p.add_argument("--gamma",       type=float, default=0.9)
    p.add_argument("--lambda_sub",  type=float, default=0.5)
    p.add_argument("--lambda_rank", type=float, default=0.1)
    p.add_argument("--alpha_init",  type=float, default=0.7)
    p.add_argument("--bc_coeff",    type=float, default=0.01,
                   help="BC coefficient for kappa only (alpha is free from BC)")
    p.add_argument("--ent_coeff",   type=float, default=0.05,
                   help="Entropy bonus coefficient to prevent alpha collapse")
    p.add_argument("--fixed_alpha", type=float, default=None,
                   help="Bypass RL alpha entirely — use this fixed value (0-1) for all decisions")
    p.add_argument("--actor_alpha_bias", type=float, default=None,
                   help="Initialize actor mean_head bias[0] to logit(this value) to center alpha here")

    # Misc
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",   type=int, default=42)

    return p.parse_args()


# ===========================================================================
# Load ICSRec-format data
# ===========================================================================

def load_icsrec_data(dataset: str, max_users: Optional[int] = None):
    """
    Load từ /workspace/repos/ICSRec/data/{Dataset}.txt + asin2id/id2asin JSON.

    Returns:
        asin2iid   : {asin_str: item_int_id}  (1-indexed)
        id2asin    : {item_int_id: asin_str}
        user_seqs  : {user_idx_str: [item_int_ids]}  (train+val+test items, sorted by time)
        item_num   : max item int id
    """
    txt_path = ICSREC_DATA / f"{DATASET_TXT}.txt"
    asin2id_path = ICSREC_DATA / f"{dataset}_asin2id.json"
    id2asin_path = ICSREC_DATA / f"{dataset}_id2asin.json"

    if not txt_path.exists():
        raise FileNotFoundError(f"ICSRec data not found: {txt_path}")
    if not asin2id_path.exists():
        raise FileNotFoundError(f"ASIN map not found: {asin2id_path}")

    asin2iid = {k: int(v) for k, v in json.loads(asin2id_path.read_text()).items()}
    id2asin  = {int(k): v for k, v in json.loads(id2asin_path.read_text()).items()}

    lines = txt_path.read_text().splitlines()
    if max_users:
        lines = lines[:max_users]

    user_seqs: Dict[str, List[int]] = {}
    max_item = 0
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        uid   = parts[0]
        items = [int(x) for x in parts[1:]]
        user_seqs[uid] = items
        max_item = max(max_item, max(items))

    return asin2iid, id2asin, user_seqs, max_item


# ===========================================================================
# Trajectory building (leave-last-2-out, same as bert4rec pipeline)
# ===========================================================================

def build_trajectories_icsr(
    user_seqs: Dict[str, List[int]],
    split: str,
    history_length: int = 20,
    slate_size: int = 10,
    target_ratings: Optional[Dict[str, Dict[int, float]]] = None,
):
    """
    Build TrajectoryStep list.  History_ids = ICSRec 1-indexed int item IDs.

    split = "train": sliding window (trừ 2 item cuối)
    split = "val"  : history = all[:-2], target = item[-2]
    split = "test" : history = all[:-1], target = item[-1]

    target_ratings: {user_str → {item_int_id → rating}}  (từ review JSON)
    """
    from algorithms.trajectory_builder import TrajectoryStep

    steps = []
    for uid, seq in user_seqs.items():
        if len(seq) < 3:
            continue

        if split == "test":
            history = seq[:-1][-history_length:]
            target  = seq[-1]
            reward  = None
            if target_ratings:
                r = target_ratings.get(uid, {}).get(target)
                reward = r / 5.0 if r else None
            steps.append(TrajectoryStep(
                user_id=uid, item_id=target,
                history_ids=history,
                budget=float(slate_size),
                split="test",
                reward=reward,
                seen_ids=list(seq[:-1]),   # exclude train+val items (ICSRec protocol)
            ))

        elif split == "val":
            history = seq[:-2][-history_length:]
            target  = seq[-2]
            reward  = None
            if target_ratings:
                r = target_ratings.get(uid, {}).get(target)
                reward = r / 5.0 if r else None
            steps.append(TrajectoryStep(
                user_id=uid, item_id=target,
                history_ids=history,
                budget=float(slate_size),
                split="val",
                reward=reward,
                seen_ids=list(seq[:-2]),   # exclude train items only
            ))

        else:  # train — sliding window
            train_seq = seq[:-2]
            for end in range(2, len(train_seq) + 1):
                hist   = train_seq[max(0, end - history_length): end - 1]
                target = train_seq[end - 1]
                reward = None
                if target_ratings:
                    r = target_ratings.get(uid, {}).get(target)
                    reward = r / 5.0 if r else None
                steps.append(TrajectoryStep(
                    user_id=uid, item_id=target,
                    history_ids=hist,
                    budget=float(slate_size),
                    split="train",
                    reward=reward,
                ))

    return steps


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    output_dir = Path(args.output_dir or "output_beauty_icsr")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset: BEAUTY 2014 5-core  [ICSRec retriever]")
    print(f"  Expected: {EXPECTED_USERS:,} users, {EXPECTED_ITEMS:,} items")
    print(f"  ICSRec data: {ICSREC_DATA / DATASET_TXT}.txt")
    print(f"  Output:  {output_dir}")
    print(f"  Device:  {device}")
    if args.max_users:
        print(f"  [LIMITED] max_users={args.max_users}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Step 1: Load ICSRec-format data
    # ------------------------------------------------------------------
    print(f"\nStep 1: Loading beauty ICSRec data...")
    asin2iid, id2asin, user_seqs, item_num = load_icsrec_data(DATASET, args.max_users)
    # Always use full catalog size (retriever returns items up to this ID regardless of max_users)
    full_item_num = len(asin2iid)
    print(f"  Users: {len(user_seqs):,}  Items in seqs: {item_num:,}  Catalog: {full_item_num:,}")
    item_num = full_item_num  # submodular/state_encoder must cover full catalog

    # id_map: asin_str → item_int_id (for pipeline candidate lookup)
    id_map = asin2iid  # {asin: 1-indexed int}

    # ------------------------------------------------------------------
    # Step 2: Build ICSRecRetriever
    # ------------------------------------------------------------------
    print(f"\nStep 2: Building ICSRecRetriever (FAISS inner-product)...")
    from retrieval.icsr_retriever import ICSRecRetriever

    retriever = ICSRecRetriever(
        dataset=DATASET,
        device=device,
        id_map_inv=id2asin,
        ckpt_path=args.ckpt_path,
    )

    # ------------------------------------------------------------------
    # Step 3: Build pipeline
    # ------------------------------------------------------------------
    print(f"\nStep 3: Building pipeline components...")
    from retrieval.two_stage_pipeline import TwoStageRLPipeline
    from retrieval.unified_pipeline import UnifiedRLPolicy
    from models.submodular import RerankerBackedSubmodular
    from utils.encoders import StateEncoder

    embed_dim  = 128
    submodular = RerankerBackedSubmodular(
        num_items=item_num + 1, embed_dim=64, alpha_init=args.alpha_init,
    ).to(device)
    state_encoder = StateEncoder(num_items=item_num, embed_dim=embed_dim).to(device)
    rl_policy     = UnifiedRLPolicy(
        state_dim=embed_dim, hidden_dim=256, lr=args.lr_rl, gamma=args.gamma,
        ent_coeff=args.ent_coeff,
    ).to(device)

    if args.actor_alpha_bias is not None:
        import math
        bias_val = math.log(args.actor_alpha_bias / (1.0 - args.actor_alpha_bias))
        with torch.no_grad():
            rl_policy.actor.mean_head.bias[0].fill_(bias_val)
        print(f"  actor mean_head bias[0] set to {bias_val:.3f} → sigmoid={args.actor_alpha_bias}")

    pipeline = TwoStageRLPipeline(
        retriever=retriever,
        submodular=submodular,
        rl_policy=rl_policy,
        state_encoder=state_encoder,
        item_id_map=id_map,
        device=device,
        n_retrieve=args.n_retrieve,
        slate_size=args.slate_size,
        history_length=args.history_length,
    )
    if args.fixed_alpha is not None:
        pipeline.fixed_alpha = args.fixed_alpha
        print(f"  fixed_alpha={args.fixed_alpha} — RL alpha bypassed")

    total_params = sum(p.numel() for p in (
        list(submodular.parameters()) +
        list(state_encoder.parameters()) +
        list(rl_policy.parameters())
    ))
    print(f"  Trainable (submodular+encoder+RL): {total_params:,}")
    print(f"  n_retrieve={args.n_retrieve}  slate_size={args.slate_size}  history_length={args.history_length}")

    # ------------------------------------------------------------------
    # Step 4: Build trajectories
    # ------------------------------------------------------------------
    print(f"\nStep 4: Building trajectory steps (leave-last-2-out)...")

    # Load per-user target ratings from original 5-core JSON
    import gzip
    from collections import defaultdict
    target_ratings: Dict[str, Dict[int, float]] = defaultdict(dict)
    opener = gzip.open if REVIEW_PATH.endswith(".gz") else open
    with opener(REVIEW_PATH, "rt", encoding="utf-8") as f:
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

    # ICSRec user indices are "1", "2", ... (string). Remap to match ICSRec uid.
    # user_seqs keys are ICSRec user_idx strings; ratings keys are reviewerID strings.
    # Build mapping: ICSRec_uid → reviewerID using sequential user ordering from data.
    # (We won't have rating rewards unless we track this mapping at convert time)
    # For now, pass None target_ratings (reward=None → computed from hit signal only)
    print("  Note: rating rewards not mapped (ICSRec uid ≠ reviewerID). Rewards from hit signal.")

    train_steps = build_trajectories_icsr(
        user_seqs, "train", args.history_length, args.slate_size, None)
    val_steps   = build_trajectories_icsr(
        user_seqs, "val",   args.history_length, args.slate_size, None)
    test_steps  = build_trajectories_icsr(
        user_seqs, "test",  args.history_length, args.slate_size, None)
    print(f"  Train: {len(train_steps):,}  Val: {len(val_steps):,}  Test: {len(test_steps):,}")

    make_query = lambda step: ""

    # ------------------------------------------------------------------
    # Step 5: Train RL + Submodular
    # ------------------------------------------------------------------
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
        bc_coeff=args.bc_coeff,
        device=device,
    )

    best_hit  = 0.0
    ckpt_path = output_dir / "best_unified.pt"

    # Epoch-0 baseline eval (runs before any training, useful for --epochs 0)
    print(f"\n[Epoch 0 — pre-training baseline]")
    _eval0 = val_steps[:args.eval_steps] if args.eval_steps else val_steps
    _val0  = trainer.evaluate(_eval0, make_query, dataset_type="amazon")
    print(f"  Val  hit@{args.slate_size}={_val0['hit@k']:.4f}  "
          f"ndcg@{args.slate_size}={_val0['ndcg@k']:.4f}  "
          f"mrr@{args.slate_size}={_val0.get('mrr@k', 0.0):.4f}  "
          f"coverage={_val0['coverage']:.4f}")
    _test0 = test_steps[:args.eval_steps] if args.eval_steps else test_steps
    _t0    = trainer.evaluate(_test0, make_query, dataset_type="amazon")
    print(f"  Test hit@{args.slate_size}={_t0['hit@k']:.4f}  "
          f"ndcg@{args.slate_size}={_t0['ndcg@k']:.4f}  "
          f"mrr@{args.slate_size}={_t0.get('mrr@k', 0.0):.4f}  "
          f"ild={_t0.get('ild', 0.0):.4f}  [epoch 0 baseline]")

    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")
        losses = trainer.train_epoch(
            trajectory_steps=train_steps,
            pipeline_query_fn=make_query,
            steps_per_epoch=args.steps_per_epoch,
            log_every=args.log_every,
            dataset_type="amazon",
            fast_mode=False,
        )

        loss_str = "  ".join(f"{k}={v:.4f}" for k, v in losses.items()) or "(warmup)"
        print(f"  Losses: {loss_str}")

        # Val evaluation — every epoch (for best checkpoint tracking)
        eval_steps  = val_steps[:args.eval_steps] if args.eval_steps else val_steps
        val_metrics = trainer.evaluate(
            eval_steps=eval_steps,
            pipeline_query_fn=make_query,
            dataset_type="amazon",
        )
        print(f"  Val  hit@{args.slate_size}={val_metrics['hit@k']:.4f}  "
              f"ndcg@{args.slate_size}={val_metrics['ndcg@k']:.4f}  "
              f"mrr@{args.slate_size}={val_metrics.get('mrr@k', 0.0):.4f}  "
              f"coverage={val_metrics['coverage']:.4f}")

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

        # Test evaluation — every test_every epochs
        if epoch % args.test_every == 0 or epoch == args.epochs:
            eval_test_mid   = test_steps[:args.eval_steps] if args.eval_steps else test_steps
            test_mid = trainer.evaluate(
                eval_steps=eval_test_mid,
                pipeline_query_fn=make_query,
                dataset_type="amazon",
            )
            all_slates_mid  = trainer.metrics.all_slates
            icsrec_embs_mid = retriever.model.item_embeddings.weight.detach().cpu()
            from utils.metrics import diversity_score
            ild_mid = float(np.mean([
                diversity_score(s, icsrec_embs_mid) for s in all_slates_mid if len(s) >= 2
            ])) if any(len(s) >= 2 for s in all_slates_mid) else 0.0
            print(f"  Test hit@{args.slate_size}={test_mid['hit@k']:.4f}  "
                  f"ndcg@{args.slate_size}={test_mid['ndcg@k']:.4f}  "
                  f"mrr@{args.slate_size}={test_mid.get('mrr@k', 0.0):.4f}  "
                  f"ild={ild_mid:.4f}  [epoch {epoch}]")

    # ------------------------------------------------------------------
    # Step 6: Final evaluation on TEST set
    # ------------------------------------------------------------------
    print(f"\nStep 6: Final evaluation on TEST set...")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        submodular.load_state_dict(ckpt["submodular"])
        state_encoder.load_state_dict(ckpt["state_encoder"])
        rl_policy.actor.load_state_dict(ckpt["rl_actor"])
        rl_policy.critic.load_state_dict(ckpt["rl_critic"])
        print(f"  Loaded best checkpoint (epoch {ckpt['epoch']}, val hit@k={ckpt['best_hit']:.4f})")

    eval_test    = test_steps[:args.eval_steps] if args.eval_steps else test_steps
    test_metrics = trainer.evaluate(
        eval_steps=eval_test,
        pipeline_query_fn=make_query,
        dataset_type="amazon",
    )

    # ILD — use ICSRec frozen embeddings (semantic) so the metric reflects
    # actual content diversity, not the trained diversity-module embeddings
    # which are optimised to minimise pairwise κ and can diverge.
    all_slates  = trainer.metrics.all_slates
    icsrec_embs = retriever.model.item_embeddings.weight.detach().cpu()  # (N+2, 64)
    from utils.metrics import diversity_score
    ild_scores = [diversity_score(s, icsrec_embs) for s in all_slates if len(s) >= 2]
    ild = float(np.mean(ild_scores)) if ild_scores else 0.0

    print(f"\n{'='*60}")
    print(f"FINAL TEST RESULTS  [BEAUTY 2014 5-core]  ICSRec retriever  (k={args.slate_size})")
    print(f"{'='*60}")
    print(f"  Hit@{args.slate_size}      = {test_metrics['hit@k']:.4f}")
    print(f"  NDCG@{args.slate_size}     = {test_metrics['ndcg@k']:.4f}")
    print(f"  MRR@{args.slate_size}      = {test_metrics.get('mrr@k', 0.0):.4f}")
    print(f"  Coverage   = {test_metrics['coverage']:.4f}")
    print(f"  ILD        = {ild:.4f}")
    print(f"  N samples  = {test_metrics['n_samples']}")
    print(f"{'='*60}")

    # Same-pipeline baseline (ICSRec FAISS-200 top-10, raw dot, excl seen, epochs=0)
    # Full-cat baselines are from ICSRec paper Table 2 (different protocol: full-catalogue)
    baselines = {
        "ICSRec FAISS-200 top-10 [same pipeline]": 0.0883,
        "ICSRec full-cat [paper, diff protocol]":  0.0963,
        "SASRec  full-cat [ICSRec paper]":         0.0624,
        "BERT4Rec full-cat [ICSRec paper]":        0.0601,
    }
    print(f"\n  Comparison (Hit@10):")
    for name, val in baselines.items():
        diff = test_metrics["hit@k"] - val
        sign = "+" if diff >= 0 else ""
        print(f"    {name:35s}: {val:.4f}  (ours: {sign}{diff:.4f})")

    results = {
        "dataset":     DATASET,
        "item_num":    item_num,
        "n_users":     len(user_seqs),
        "retriever":   "ICSRec-SAS",
        "n_retrieve":  args.n_retrieve,
        "slate_size":  args.slate_size,
        "test_hit_k":  test_metrics["hit@k"],
        "test_ndcg_k": test_metrics["ndcg@k"],
        "test_mrr_k":  test_metrics.get("mrr@k", 0.0),
        "test_coverage": test_metrics["coverage"],
        "test_ild":    ild,
        "n_samples":   test_metrics["n_samples"],
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {output_dir}/results.json")


if __name__ == "__main__":
    main()
