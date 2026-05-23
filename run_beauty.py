"""
Pipeline runner for Amazon 2014 5-core benchmark datasets.

Supported datasets: beauty | sports | toys
(ML-1M needs a separate loader due to different format)

The key difference from run_amazon.py:
  - Uses legacy review-only mode (no V2 dataset_dir)
  - Defaults to full dataset (all users, no max_users cap)
  - Output isolated per dataset in output_<dataset>/

Usage:
  # Full run on Beauty (all 22K users)
  python run_beauty.py --dataset beauty --device cuda

  # Quick smoke test (200 users, 1 epoch)
  python run_beauty.py --dataset beauty --max_users 200 --epochs 1 --steps_per_epoch 50

  # Sports
  python run_beauty.py --dataset sports --device cuda

  # All datasets sequentially
  for ds in beauty sports toys; do
      python run_beauty.py --dataset $ds --device cuda
  done
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Amazon 2014 5-core benchmark pipeline (Beauty / Sports / Toys)"
    )

    p.add_argument("--dataset", default="beauty",
                   choices=list(DATASET_PATHS),
                   help="Which benchmark dataset to use")

    # Override review path (auto-set from --dataset, but can be overridden)
    p.add_argument("--review_path", default=None,
                   help="Override review file path (default: auto from --dataset)")

    p.add_argument("--max_users", type=int, default=None,
                   help="Max users to load (None = all). Use small value for quick smoke test")
    p.add_argument("--history_length", type=int, default=20)

    # BM25
    p.add_argument("--bm25_backend", default="bm25s",
                   choices=["bm25s", "rank_bm25", "pyserini"])
    p.add_argument("--rebuild_bm25", action="store_true")

    # Retrieval
    p.add_argument("--build_dense",  action="store_true")
    p.add_argument("--n_bm25",       type=int, default=100)
    p.add_argument("--n_dense",      type=int, default=50)
    p.add_argument("--n_fuse",       type=int, default=200)
    p.add_argument("--embed_batch_size",    type=int, default=16)
    p.add_argument("--reranker_batch_size", type=int, default=8)
    p.add_argument("--slate_size",   type=int, default=10)

    # Training
    p.add_argument("--epochs",           type=int,   default=5)
    p.add_argument("--steps_per_epoch",  type=int,   default=500)
    p.add_argument("--eval_steps",       type=int,   default=None,
                   help="Max eval steps per epoch (None = all)")
    p.add_argument("--batch_size",       type=int,   default=32)
    p.add_argument("--buffer_size",      type=int,   default=10_000)
    p.add_argument("--min_buffer",       type=int,   default=64)
    p.add_argument("--log_every",        type=int,   default=50)

    # Optimiser
    p.add_argument("--lr_rl",        type=float, default=3e-4)
    p.add_argument("--lr_sub",       type=float, default=1e-3)
    p.add_argument("--lr_encoder",   type=float, default=1e-3)
    p.add_argument("--gamma",        type=float, default=0.99)
    p.add_argument("--lambda_sub",   type=float, default=0.5)
    p.add_argument("--lambda_rank",  type=float, default=0.1)
    p.add_argument("--alpha_init",   type=float, default=0.7)

    # Misc
    p.add_argument("--device",   default="cpu")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--output_dir", default=None,
                   help="Output directory (default: output_<dataset>)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve paths
    review_path = args.review_path or DATASET_PATHS[args.dataset]
    output_dir  = args.output_dir  or f"output_{args.dataset}"

    expected_users, expected_items = DATASET_SIZES[args.dataset]
    print(f"\nDataset: {args.dataset.upper()} 2014 5-core")
    print(f"  Expected: {expected_users:,} users, {expected_items:,} items")
    print(f"  Review path: {review_path}")
    print(f"  Output dir:  {output_dir}")
    if args.max_users:
        print(f"  [LIMITED] max_users={args.max_users} (smoke-test mode)")

    # Build a namespace compatible with run_amazon.main()
    # Set dataset_dir=None to force legacy (review-file) mode.
    import argparse as _ap
    amazon_args = _ap.Namespace(
        # V2 path — disabled
        dataset_dir=None,
        meta_v2_path=None,
        max_train_samples=None,
        # Legacy path
        review_path=review_path,
        meta_path=None,          # no separate meta file for 2014 5-core
        max_items=50_000,
        max_users=args.max_users,
        history_length=args.history_length,
        # BM25
        bm25_backend=args.bm25_backend,
        rebuild_bm25=args.rebuild_bm25,
        # Retrieval
        build_dense=args.build_dense,
        n_bm25=args.n_bm25,
        n_dense=args.n_dense,
        n_fuse=args.n_fuse,
        embed_batch_size=args.embed_batch_size,
        reranker_batch_size=args.reranker_batch_size,
        slate_size=args.slate_size,
        # Training
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        eval_steps=args.eval_steps,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        min_buffer=args.min_buffer,
        log_every=args.log_every,
        # Optimiser
        lr_rl=args.lr_rl,
        lr_sub=args.lr_sub,
        lr_encoder=args.lr_encoder,
        gamma=args.gamma,
        lambda_sub=args.lambda_sub,
        lambda_rank=args.lambda_rank,
        alpha_init=args.alpha_init,
        # Misc
        device=args.device,
        seed=args.seed,
        output_dir=output_dir,
    )

    from run_amazon import main as amazon_main
    amazon_main(amazon_args)


if __name__ == "__main__":
    main()
