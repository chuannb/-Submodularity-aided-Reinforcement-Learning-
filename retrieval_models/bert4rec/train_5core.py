"""
BERT4Rec trainer cho Amazon 2014 5-core datasets (Beauty / Sports / Toys).

Đọc trực tiếp từ reviews_5core.json.gz — không cần V2 format hay products.jsonl.

Format review:
  {"reviewerID": str, "asin": str, "overall": float, "unixReviewTime": int, ...}

Split strategy: leave-last-2-out
  test  = last interaction per user
  val   = second-to-last
  train = all others (same as run_amazon.py)

Usage:
  # Beauty, quick test (2 epochs)
  python -m retrieval_models.bert4rec.train_5core \\
      --review_path /workspace/datasets/beauty_2014/reviews_5core.json.gz \\
      --dataset beauty --epochs 2 --max_train 50000

  # Full Beauty run (5 epochs, GPU)
  python -m retrieval_models.bert4rec.train_5core \\
      --review_path /workspace/datasets/beauty_2014/reviews_5core.json.gz \\
      --dataset beauty --epochs 5 --device cuda

  # Sports
  python -m retrieval_models.bert4rec.train_5core \\
      --review_path /workspace/datasets/sports_2014/reviews_5core.json.gz \\
      --dataset sports --epochs 5 --device cuda
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from retrieval_models.bert4rec.model import BERT4Rec


# ===========================================================================
# 1. Data loading — reads reviews_5core.json.gz
# ===========================================================================

def load_5core(review_path: str, max_users: Optional[int] = None):
    """
    Load Amazon 5-core reviews, build:
      - asin2iid  : {asin: int}  (1-indexed, 0=padding)
      - user_seqs : {user_id: [(item_id, timestamp), ...]}  sorted by time
    """
    print(f"\nLoading {review_path} ...", flush=True)
    opener = gzip.open if review_path.endswith(".gz") else open

    raw: Dict[str, List[Tuple[int, int]]] = defaultdict(list)  # user -> [(iid, ts)]
    asin_set: set = set()

    with opener(review_path, "rt", encoding="utf-8") as f:
        for line in tqdm(f, desc="reading", unit="rec", dynamic_ncols=True):
            try:
                r = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            uid  = r.get("reviewerID", "")
            asin = r.get("asin", "")
            ts   = int(r.get("unixReviewTime", 0))
            if uid and asin:
                raw[uid].append((asin, ts))
                asin_set.add(asin)

    # Build asin → item_id (1-indexed, sorted for reproducibility)
    asin2iid = {a: i + 1 for i, a in enumerate(sorted(asin_set))}
    item_num  = len(asin2iid)
    print(f"  {len(raw):,} users, {item_num:,} items", flush=True)

    # Sort each user's interactions by timestamp
    user_ids = sorted(raw.keys())
    if max_users:
        user_ids = user_ids[:max_users]

    user_seqs = {}
    for uid in user_ids:
        interactions = sorted(raw[uid], key=lambda x: x[1])
        # Convert asin → item_id
        iid_seq = [asin2iid[a] for a, _ in interactions if a in asin2iid]
        if len(iid_seq) >= 3:   # need ≥3 for leave-last-2-out
            user_seqs[uid] = iid_seq

    print(f"  {len(user_seqs):,} users with ≥3 interactions", flush=True)
    return asin2iid, item_num, user_seqs


def split_sequences(user_seqs: Dict[str, List[int]], maxlen: int):
    """
    Leave-last-2-out split.
    Returns (train_seqs, train_targets, val_seqs, val_targets, test_seqs, test_targets)
    Each seq is left-padded to maxlen (0=padding).
    """
    train_seqs, train_targets = [], []
    val_seqs,   val_targets   = [], []
    test_seqs,  test_targets  = [], []

    for uid, seq in user_seqs.items():
        if len(seq) < 3:
            continue

        # Test: last item
        test_hist = seq[:-1][-maxlen:]
        s = np.zeros(maxlen, dtype=np.int32)
        s[-len(test_hist):] = test_hist
        test_seqs.append(s)
        test_targets.append(seq[-1])

        # Val: second-to-last
        val_hist = seq[:-2][-maxlen:]
        s = np.zeros(maxlen, dtype=np.int32)
        s[-len(val_hist):] = val_hist
        val_seqs.append(s)
        val_targets.append(seq[-2])

        # Train: all items except last 2, use a sliding window
        train_hist = seq[:-2]
        if len(train_hist) >= 2:
            for end in range(2, len(train_hist) + 1):
                h = train_hist[max(0, end - maxlen): end]
                s = np.zeros(maxlen, dtype=np.int32)
                s[-len(h):] = h
                train_seqs.append(s)
                train_targets.append(train_hist[end - 1])

    return (
        np.array(train_seqs,  dtype=np.int32),  np.array(train_targets,  dtype=np.int32),
        np.array(val_seqs,    dtype=np.int32),  np.array(val_targets,    dtype=np.int32),
        np.array(test_seqs,   dtype=np.int32),  np.array(test_targets,   dtype=np.int32),
    )


# ===========================================================================
# 2. Dataset
# ===========================================================================

class BERT4RecDataset5Core(Dataset):
    """Cloze masking dataset for 5-core data (same logic as train.py)."""

    def __init__(self, seqs: np.ndarray, targets: np.ndarray,
                 item_num: int, mask_prob: float = 0.2, mask_token: int = None):
        self.seqs       = seqs
        self.targets    = targets
        self.item_num   = item_num
        self.mask_prob  = mask_prob
        self.mask_token = mask_token if mask_token is not None else item_num + 1
        self.maxlen     = seqs.shape[1]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        seq    = self.seqs[idx].astype(np.int64)
        target = int(self.targets[idx])

        # Append target as last item
        seq_t = np.empty(self.maxlen, dtype=np.int64)
        seq_t[:-1] = seq[1:]
        seq_t[-1]  = target

        masked_seq = seq_t.copy()
        pos_ids    = np.zeros(self.maxlen, dtype=np.int64)

        non_pad = np.where(seq_t != 0)[0]
        if len(non_pad) == 0:
            return torch.zeros(self.maxlen, dtype=torch.long), torch.zeros(self.maxlen, dtype=torch.long)

        will_mask = [p for p in non_pad if random.random() < self.mask_prob]
        if not will_mask:
            will_mask = [non_pad[-1]]

        for pos in will_mask:
            orig = seq_t[pos]
            r = random.random()
            if r < 0.8:
                masked_seq[pos] = self.mask_token
            elif r < 0.9:
                masked_seq[pos] = random.randint(1, self.item_num)
            pos_ids[pos] = orig

        return torch.from_numpy(masked_seq), torch.from_numpy(pos_ids)


# ===========================================================================
# 3. Evaluation — Recall@K on val/test sets
# ===========================================================================

@torch.no_grad()
def evaluate_recall(model, item_embs, seqs, targets, topk_list, device, batch_size=512):
    """GPU-accelerated recall@K evaluation."""
    model.eval()
    item_embs_gpu = item_embs.to(device)
    max_k  = max(topk_list)
    hits   = {k: 0 for k in topk_list}
    valid  = 0
    N      = len(targets)

    for start in range(0, N, batch_size):
        end    = min(start + batch_size, N)
        seqs_t = torch.from_numpy(seqs[start:end].astype(np.int64)).to(device)
        tgts   = targets[start:end]

        uembs  = model.user_embedding(seqs_t)               # (B, d)
        scores = torch.matmul(uembs, item_embs_gpu.T)       # (B, item_num)
        _, topk_idx = scores.topk(max_k, dim=1)             # (B, max_k)
        topk_np = (topk_idx + 1).cpu().numpy()              # 1-indexed item_id

        for row, tgt in zip(topk_np, tgts):
            for k in topk_list:
                if tgt in row[:k]:
                    hits[k] += 1
            valid += 1

    return {k: hits[k] / max(valid, 1) for k in topk_list}


def build_item_embs(model, item_num, device, batch_size=8192):
    model.eval()
    parts = []
    ids_full = torch.arange(1, item_num + 1, device=device)
    with torch.no_grad():
        for s in range(0, item_num, batch_size):
            chunk = ids_full[s: s + batch_size]
            emb   = nn.functional.normalize(model.embedding.token(chunk), dim=-1)
            parts.append(emb)
    return torch.cat(parts, dim=0)   # (item_num, d)


# ===========================================================================
# 4. Args
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="BERT4Rec for Amazon 2014 5-core datasets")
    p.add_argument("--review_path", required=True,
                   help="Path to reviews_5core.json.gz")
    p.add_argument("--dataset", default="beauty",
                   choices=["beauty", "sports", "toys"],
                   help="Dataset name (used for output directory naming)")
    p.add_argument("--output_dir", default=None,
                   help="Output dir. Default: retrieval_models/bert4rec/output_<dataset>")
    p.add_argument("--max_users", type=int, default=None)

    # Model
    p.add_argument("--maxlen",     type=int,   default=50)
    p.add_argument("--hidden_dim", type=int,   default=64)
    p.add_argument("--num_heads",  type=int,   default=2)
    p.add_argument("--num_blocks", type=int,   default=2)
    p.add_argument("--dropout",    type=float, default=0.2)
    p.add_argument("--mask_prob",  type=float, default=0.2)

    # Training
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch_size", type=int,   default=512)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--temperature",type=float, default=0.07)
    p.add_argument("--max_train",  type=int,   default=None)

    # Eval
    p.add_argument("--topk",       type=int, nargs="+", default=[10, 20, 50])
    p.add_argument("--eval_every", type=int, default=1, help="Eval every N epochs")

    # Misc
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


# ===========================================================================
# 5. Main
# ===========================================================================

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    out_dir = Path(args.output_dir or f"retrieval_models/bert4rec/output_{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ────────────────────────────────────────────────────────
    cache_dir  = out_dir / "preprocessed"
    cache_dir.mkdir(exist_ok=True)
    asin2iid_f = cache_dir / "asin2iid.json"
    seqs_cache = cache_dir / f"splits_L{args.maxlen}.npz"

    if seqs_cache.exists() and asin2iid_f.exists():
        print("Loading cached splits...", flush=True)
        with open(asin2iid_f) as f:
            asin2iid = json.load(f)
        item_num = max(asin2iid.values())
        d = np.load(seqs_cache)
        train_seqs,  train_targets  = d["train_seqs"],  d["train_targets"]
        val_seqs,    val_targets    = d["val_seqs"],    d["val_targets"]
        test_seqs,   test_targets   = d["test_seqs"],   d["test_targets"]
    else:
        asin2iid, item_num, user_seqs = load_5core(args.review_path, args.max_users)
        (train_seqs, train_targets,
         val_seqs,   val_targets,
         test_seqs,  test_targets) = split_sequences(user_seqs, args.maxlen)
        np.savez_compressed(
            seqs_cache,
            train_seqs=train_seqs,   train_targets=train_targets,
            val_seqs=val_seqs,       val_targets=val_targets,
            test_seqs=test_seqs,     test_targets=test_targets,
        )
        with open(asin2iid_f, "w") as f:
            json.dump(asin2iid, f)

    print(f"\nitem_num={item_num:,}")
    print(f"  train: {len(train_targets):,} samples")
    print(f"  val  : {len(val_targets):,} users")
    print(f"  test : {len(test_targets):,} users", flush=True)

    if args.max_train and len(train_targets) > args.max_train:
        idx = np.random.permutation(len(train_targets))[:args.max_train]
        train_seqs, train_targets = train_seqs[idx], train_targets[idx]
        print(f"  Capped train to {len(train_targets):,}", flush=True)

    # ── 2. Model ─────────────────────────────────────────────────────────────
    model = BERT4Rec(
        item_num    = item_num,
        maxlen      = args.maxlen,
        hidden_dim  = args.hidden_dim,
        num_heads   = args.num_heads,
        num_blocks  = args.num_blocks,
        dropout_rate= args.dropout,
        mask_prob   = args.mask_prob,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nBERT4Rec  params={n_params:,}  mask_token={model.mask_token}", flush=True)

    dataset = BERT4RecDataset5Core(
        train_seqs, train_targets, item_num,
        mask_prob=args.mask_prob, mask_token=model.mask_token,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    start_epoch = 1
    best_recall = 0.0
    history     = []

    if args.resume:
        ckpt_path = out_dir / "last_checkpoint.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch  = ckpt["epoch"] + 1
            best_recall  = ckpt["best_recall"]
            history      = ckpt.get("history", [])
            print(f"Resumed from epoch {ckpt['epoch']}, best_recall@{max(args.topk)}={best_recall:.4f}", flush=True)

    # ── 3. Training loop ──────────────────────────────────────────────────────
    print(f"\nTraining: {args.epochs} epochs × {len(dataset):,} samples  (batch={args.batch_size})\n", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.perf_counter()

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch", dynamic_ncols=True)
        for masked_seqs, pos_ids in pbar:
            masked_seqs = masked_seqs.to(device)
            pos_ids     = pos_ids.to(device)

            loss = model.infonce_loss(masked_seqs, pos_ids, temperature=args.temperature)
            if loss == 0:
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(len(loader), 1)
        elapsed  = time.perf_counter() - t0
        print(f"\nEpoch {epoch}  loss={avg_loss:.4f}  {elapsed:.0f}s", flush=True)

        # Eval
        recall = {}
        if epoch % args.eval_every == 0:
            item_embs = build_item_embs(model, item_num, device)
            val_recall  = evaluate_recall(model, item_embs, val_seqs,  val_targets,  args.topk, device)
            test_recall = evaluate_recall(model, item_embs, test_seqs, test_targets, args.topk, device)
            recall = {"val": val_recall, "test": test_recall}

            print(f"  Val  Recall: " + "  ".join(f"@{k}={v:.4f}" for k, v in sorted(val_recall.items())), flush=True)
            print(f"  Test Recall: " + "  ".join(f"@{k}={v:.4f}" for k, v in sorted(test_recall.items())), flush=True)

            best_k  = max(args.topk)
            r_best  = test_recall.get(best_k, 0.0)
            if r_best > best_recall:
                best_recall = r_best
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "recall": recall,
                    "args": vars(args),
                    "asin2iid_path": str(asin2iid_f),
                    "item_num": item_num,
                }, out_dir / "best_model.pt")
                print(f"  ✓ best recall@{best_k}={r_best:.4f} saved → {out_dir}/best_model.pt", flush=True)

            del item_embs

        history.append({"epoch": epoch, "loss": avg_loss, "recall": recall})
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_recall": best_recall,
            "history": history,
        }, out_dir / "last_checkpoint.pt")

    # ── 4. Final: build FAISS index from best checkpoint ─────────────────────
    print("\nBuilding FAISS index from best checkpoint...", flush=True)
    best_ckpt_path = out_dir / "best_model.pt"
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded best model from epoch {ckpt['epoch']}", flush=True)

    item_embs = build_item_embs(model, item_num, device)
    try:
        import faiss
        arr   = item_embs.cpu().numpy().astype(np.float32)
        index = faiss.IndexFlatIP(arr.shape[1])
        index.add(arr)
        faiss.write_index(index, str(out_dir / "item_faiss.index"))
        print(f"  FAISS saved: {(out_dir/'item_faiss.index').stat().st_size/1e6:.0f} MB", flush=True)
    except ImportError:
        print("  faiss not installed — skipping FAISS index build", flush=True)
        np.save(out_dir / "item_embs.npy", item_embs.cpu().numpy())
        print(f"  Item embeddings saved as item_embs.npy", flush=True)

    # Final test recall
    test_recall = evaluate_recall(model, item_embs, test_seqs, test_targets, args.topk, device)

    print(f"\n{'═'*55}")
    print(f"  BERT4Rec — {args.dataset.upper()} 2014 5-core")
    print(f"{'═'*55}")
    for k in sorted(args.topk):
        print(f"  Recall@{k:<4}: {test_recall[k]:.4f}")
    print(f"{'═'*55}")
    print(f"  Output: {out_dir}", flush=True)

    # Save asin2iid alongside model
    out_asin2iid = out_dir / "asin2iid.json"
    if not out_asin2iid.exists():
        import shutil
        shutil.copy2(str(asin2iid_f), str(out_asin2iid))

    results = {
        "dataset": args.dataset,
        "item_num": item_num,
        "model": "BERT4Rec",
        "hidden_dim": args.hidden_dim,
        "maxlen": args.maxlen,
        "epochs_trained": args.epochs,
        "test_recall": test_recall,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Artifacts in {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
