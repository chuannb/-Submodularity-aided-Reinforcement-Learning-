"""
eval_icsrec_fullcat.py
======================
Reproduce ICSRec paper Table 2 evaluation:
  - Full-catalogue ranking (all items scored, no FAISS)
  - Leave-one-out split (last item = test, second-to-last = val)
  - No seen-item exclusion (paper does NOT exclude)
  - HR@{5,10,20}, NDCG@{5,10,20}

Usage:
  python eval_icsrec_fullcat.py --dataset beauty --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT   = Path(__file__).parent.resolve()
ICSREC_DATA = Path("/workspace/repos/ICSRec/data")
ICSREC_SRC  = Path("/workspace/repos/ICSRec/src")
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Load ICSRec model (same logic as icsr_retriever.py)
# ---------------------------------------------------------------------------
def load_model(dataset: str, device):
    import importlib.util as _ilu

    name_map = {"beauty": "Beauty", "sports": "Sports_and_Outdoors", "toys": "Toys_and_Games"}
    ics_name = name_map.get(dataset.lower(), dataset)

    txt_path = ICSREC_DATA / f"{ics_name}.txt"
    lines = txt_path.read_text().splitlines()
    max_item = max(max(int(x) for x in l.strip().split()[1:]) for l in lines if l.strip())
    item_size = max_item + 2  # 0=pad, 1..max_item=items, max_item+1=mask

    ckpt_path = ICSREC_SRC / "output" / f"ICSRec-SAS-{ics_name}-0.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    print(f"item_size={item_size}  (max_item={max_item})")

    def _load_mod(name, path):
        if name in sys.modules:
            return sys.modules[name]
        spec = _ilu.spec_from_file_location(name, path)
        mod  = _ilu.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    _load_mod("icsrec_modules", str(ICSREC_SRC / "modules.py"))
    _old = sys.modules.get("modules")
    sys.modules["modules"] = sys.modules["icsrec_modules"]
    try:
        _m = _load_mod("icsrec_models", str(ICSREC_SRC / "models.py"))
    finally:
        if _old is not None:
            sys.modules["modules"] = _old
        else:
            sys.modules.pop("modules", None)

    import argparse as _ap
    args = _ap.Namespace(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
        hidden_act="gelu", attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0, initializer_range=0.02,
        max_seq_length=50, item_size=item_size,
        mask_id=item_size - 1, cuda_condition=(device.type == "cuda"),
    )
    model = _m.SASRecModel(args).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    return model, max_item


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_seqs(dataset: str):
    name_map = {"beauty": "Beauty", "sports": "Sports_and_Outdoors", "toys": "Toys_and_Games"}
    txt_path = ICSREC_DATA / f"{name_map.get(dataset.lower(), dataset)}.txt"
    seqs = {}
    for line in txt_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        uid   = parts[0]
        items = [int(x) for x in parts[1:]]
        seqs[uid] = items
    return seqs


# ---------------------------------------------------------------------------
# Encode sequence → user embedding
# ---------------------------------------------------------------------------
def encode_seq(model, item_ids, maxlen, device):
    """Return last-position output as user embedding. Shape: (d,)"""
    seq = item_ids[-maxlen:]
    pad = maxlen - len(seq)
    inp = [0] * pad + seq
    inp_t = torch.LongTensor([inp]).to(device)
    with torch.no_grad():
        out = model(inp_t)   # (1, maxlen, d)
    return out[0, -1, :]     # (d,)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def hit_at_k(rank, k):
    return 1.0 if rank < k else 0.0

def ndcg_at_k(rank, k):
    import math
    return 1.0 / math.log2(rank + 2) if rank < k else 0.0


# ---------------------------------------------------------------------------
# Full-catalogue evaluation
# ---------------------------------------------------------------------------
def evaluate(model, seqs, max_item, device, maxlen=50, batch_size=256,
             split="test", exclude_seen=False):
    """
    split: "test" → target=last item, history=all[:-1]
           "val"  → target=second-last, history=all[:-2]
    exclude_seen: if True, mask out seen items from scores (not in paper protocol)
    """
    # Item embedding matrix: shape (item_size, d)
    item_emb = model.item_embeddings.weight.detach()  # (item_size, d)
    # Only real items: indices 1..max_item
    real_item_emb = item_emb[1: max_item + 1]         # (max_item, d)
    real_item_emb = F.normalize(real_item_emb, dim=-1)

    metrics = {k: [] for k in [5, 10, 20]}
    ndcg    = {k: [] for k in [5, 10, 20]}

    users = [uid for uid, seq in seqs.items() if len(seq) >= 3]
    print(f"Evaluating {len(users)} users [{split}, exclude_seen={exclude_seen}]...")

    for i in range(0, len(users), batch_size):
        batch_uids = users[i: i + batch_size]
        user_vecs  = []
        targets    = []
        seen_sets  = []

        for uid in batch_uids:
            seq = seqs[uid]
            if split == "test":
                history = seq[:-1]
                target  = seq[-1]
            else:
                history = seq[:-2]
                target  = seq[-2]

            uv = encode_seq(model, history, maxlen, device)
            user_vecs.append(uv)
            targets.append(target)
            seen_sets.append(set(history) if exclude_seen else set())

        user_mat = torch.stack(user_vecs, dim=0)           # (B, d)
        user_mat = F.normalize(user_mat, dim=-1)
        # Scores: (B, max_item)  — inner product after L2 norm = cosine sim
        scores = (user_mat @ real_item_emb.T).cpu().numpy()

        for j, (uid, target) in enumerate(zip(batch_uids, targets)):
            sc = scores[j].copy()

            # Optionally mask seen items
            if exclude_seen:
                for sid in seen_sets[j]:
                    si = sid - 1   # 0-indexed into real_item_emb
                    if 0 <= si < len(sc):
                        sc[si] = -1e9

            # target index in real_item_emb (0-indexed)
            target_idx = target - 1
            if target_idx < 0 or target_idx >= len(sc):
                continue

            target_score = sc[target_idx]
            rank = int((sc > target_score).sum())  # number of items scoring higher

            for k in [5, 10, 20]:
                metrics[k].append(hit_at_k(rank, k))
                ndcg[k].append(ndcg_at_k(rank, k))

        if (i // batch_size) % 10 == 0:
            done = min(i + batch_size, len(users))
            print(f"  {done}/{len(users)} users processed...", end="\r")

    print()
    return {
        f"HR@{k}":   np.mean(metrics[k]) for k in [5, 10, 20]
    } | {
        f"NDCG@{k}": np.mean(ndcg[k])   for k in [5, 10, 20]
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",      default="beauty")
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--split",        default="test", choices=["test", "val"])
    p.add_argument("--exclude_seen", action="store_true",
                   help="Exclude seen items from candidate set (NOT paper protocol)")
    p.add_argument("--batch_size",   type=int, default=256)
    args = p.parse_args()

    device = torch.device(args.device)
    model, max_item = load_model(args.dataset, device)
    seqs = load_seqs(args.dataset)

    print(f"\nDataset: {args.dataset}  |  Users: {len(seqs)}  |  Items: {max_item}")
    print(f"Paper protocol:   full-catalogue, NO seen exclusion")
    print(f"Our protocol:     FAISS top-200, WITH seen exclusion")
    print()

    # Paper protocol (no exclusion)
    res_paper = evaluate(model, seqs, max_item, device,
                         split=args.split, exclude_seen=False,
                         batch_size=args.batch_size)
    print("\n=== Full-catalogue, NO seen exclusion (Paper protocol) ===")
    for k, v in res_paper.items():
        print(f"  {k} = {v:.4f}")

    # Our protocol (with exclusion)
    res_ours = evaluate(model, seqs, max_item, device,
                        split=args.split, exclude_seen=True,
                        batch_size=args.batch_size)
    print("\n=== Full-catalogue, WITH seen exclusion (our-protocol variant) ===")
    for k, v in res_ours.items():
        print(f"  {k} = {v:.4f}")

    print(f"\nPaper reports ICSRec HR@10 (Beauty) = 0.1298")
    print(f"Reproduced   HR@10 (no excl)        = {res_paper['HR@10']:.4f}")
    print(f"Reproduced   HR@10 (with excl)       = {res_ours['HR@10']:.4f}")


if __name__ == "__main__":
    main()
