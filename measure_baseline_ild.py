"""
Measure ILD and HR@10 for ICSRec greedy baseline using ICSRec frozen embeddings.

Three variants:
  A. Pure FAISS top-10: just take top-10 by FAISS score (no submodular)
  B. Pipeline epoch-0 (kappa=0.5): submodular greedy with random embs, eps=0.25 exploration
  C. Pipeline epoch-0 (kappa=0.0): submodular greedy with random embs, deterministic

Same settings as run_alpha_bias09: FAISS-200, raw dot, h=20, seen-item exclusion.
"""
import json, sys, random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

RECSYS_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(RECSYS_DIR))

from retrieval.icsr_retriever import ICSRecRetriever
from models.submodular import RerankerBackedSubmodular
from utils.metrics import diversity_score

ICSREC_DATA   = Path("/workspace/repos/ICSRec/data")
HISTORY_LENGTH = 20
SLATE_SIZE    = 10
N_RETRIEVE    = 200
DATASET       = "beauty"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# -- Load data
asin2id_path = ICSREC_DATA / f"{DATASET}_asin2id.json"
id2asin_path  = ICSREC_DATA / f"{DATASET}_id2asin.json"
asin2iid = {k: int(v) for k, v in json.loads(asin2id_path.read_text()).items()}
id2asin  = {int(k): v for k, v in json.loads(id2asin_path.read_text()).items()}

txt_path = ICSREC_DATA / "Beauty.txt"
user_seqs = {}
for line in txt_path.read_text().splitlines():
    parts = line.strip().split()
    if len(parts) >= 3:
        user_seqs[parts[0]] = [int(x) for x in parts[1:]]

print(f"Users: {len(user_seqs):,}")

# -- Build retriever + submodular (random init)
retriever = ICSRecRetriever(dataset=DATASET, device=device, id_map_inv=id2asin)
icsrec_embs = retriever.model.item_embeddings.weight.detach().cpu()
item_num = retriever.item_num

submodular = RerankerBackedSubmodular(num_items=item_num + 1, embed_dim=64).to(device)
# random init at epoch 0 — no training

# -- Test steps
test_steps = []
for uid, seq in user_seqs.items():
    if len(seq) < 3:
        continue
    test_steps.append({
        "history": seq[:-1][-HISTORY_LENGTH:],
        "target":  seq[-1],
        "seen":    set(seq[:-1]),
    })
print(f"Test steps: {len(test_steps):,}")

# -- Helper: run greedy reranker (inline, vectorized)
def greedy_reranker(candidates, rel_scores, k_np, alpha, kappa, slate_size):
    from algorithms.greedy_selector import eps_from_kappa, tau_from_kappa
    n = len(candidates)
    rel_arr  = np.array([rel_scores.get(c, 0.0) for c in candidates], dtype=np.float32)
    cost_arr = np.ones(n, dtype=np.float32)
    slate_local = []
    in_slate    = np.zeros(n, dtype=bool)
    k_cur = 0
    eps = eps_from_kappa(kappa)
    tau = tau_from_kappa(kappa)
    for _ in range(slate_size):
        feasible = np.where(~in_slate)[0]
        if feasible.size == 0:
            break
        if k_cur == 0:
            benefit = alpha * rel_arr[feasible]
        else:
            sum_k = k_np[np.ix_(feasible, slate_local)].sum(axis=1)
            benefit = alpha * rel_arr[feasible] - (1.0 - alpha) * sum_k
        if random.random() < eps:
            q = (benefit / tau).astype(np.float64); q -= q.max()
            probs = np.exp(q); probs /= probs.sum()
            li = int(np.random.choice(len(feasible), p=probs))
        else:
            li = int(np.argmax(benefit))
        chosen = int(feasible[li])
        slate_local.append(chosen)
        in_slate[chosen] = True
        k_cur += 1
    return [candidates[i] for i in slate_local]

# -- Precompute full k_mat for entire catalog (for submodular use)
@torch.no_grad()
def get_k_mat(candidates, sub, device):
    if not candidates:
        return np.zeros((0,0), dtype=np.float32)
    cand_t = torch.tensor(candidates, dtype=torch.long, device=device)
    embs   = F.normalize(sub.item_emb(cand_t), dim=-1)
    sim    = embs @ embs.T
    bw     = torch.exp(sub.log_bandwidth).clamp(min=1e-3)
    k_mat  = torch.exp(-(1.0 - sim) / bw)
    return k_mat.cpu().numpy()

# -- Run all three variants
BATCH = 512
results = {
    "pure_top10": {"hits": [], "slates": []},
    "greedy_kappa05": {"hits": [], "slates": []},  # epoch-0: kappa=0.5
    "greedy_kappa00": {"hits": [], "slates": []},  # deterministic alpha=0.9
}

# For reproducibility in eps-greedy, fix seed
random.seed(42)
np.random.seed(42)

for start in range(0, len(test_steps), BATCH):
    batch = test_steps[start:start + BATCH]
    hists = [s["history"] for s in batch]
    all_results = retriever.batch_search_by_history(hists, top_k=N_RETRIEVE)

    for step, res_list in zip(batch, all_results):
        seen = step["seen"]
        target = step["target"]

        # Filter seen, build candidates
        candidates = []
        rel_map = {}
        for r in res_list:
            item_id_str = r.item_id
            iid = asin2iid.get(item_id_str)
            if iid is None:
                try: iid = int(item_id_str)
                except: continue
            if iid <= 0 or iid in seen:
                continue
            candidates.append(iid)
            rel_map[iid] = float(r.score)

        # A. Pure FAISS top-10
        slate_a = candidates[:SLATE_SIZE]
        results["pure_top10"]["hits"].append(1.0 if target in slate_a else 0.0)
        results["pure_top10"]["slates"].append(slate_a)

        if not candidates:
            for k in ["greedy_kappa05", "greedy_kappa00"]:
                results[k]["hits"].append(0.0)
                results[k]["slates"].append([])
            continue

        # Precompute k_mat for this user's candidates
        k_np = get_k_mat(candidates, submodular, device)

        # B. kappa=0.5 (epoch-0 pipeline default: sigmoid(0)=0.5)
        slate_b = greedy_reranker(candidates, rel_map, k_np, alpha=0.9, kappa=0.5, slate_size=SLATE_SIZE)
        results["greedy_kappa05"]["hits"].append(1.0 if target in slate_b else 0.0)
        results["greedy_kappa05"]["slates"].append(slate_b)

        # C. kappa=0.0 (deterministic greedy)
        slate_c = greedy_reranker(candidates, rel_map, k_np, alpha=0.9, kappa=0.0, slate_size=SLATE_SIZE)
        results["greedy_kappa00"]["hits"].append(1.0 if target in slate_c else 0.0)
        results["greedy_kappa00"]["slates"].append(slate_c)

    if (start // BATCH) % 10 == 0:
        print(f"  {min(start+BATCH, len(test_steps)):,}/{len(test_steps):,}", flush=True)

print(f"\n{'='*60}")
for name, data in results.items():
    hr10 = float(np.mean(data["hits"]))
    ild_scores = [diversity_score(s, icsrec_embs) for s in data["slates"] if len(s) >= 2]
    ild = float(np.mean(ild_scores)) if ild_scores else 0.0
    print(f"{name:22s}  HR@10={hr10:.4f}  ILD={ild:.4f}")
print(f"{'='*60}")
print("\nPaper claims baseline: HR@10=0.0883, ILD=0.5127")
print("Paper claims SRS:      HR@10=0.0893, ILD=0.5651")
