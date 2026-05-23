"""
BERT4RecRetriever — Sequential retriever thay thế BM25.

Nhận user history (danh sách item_id int) → trả về top-K candidates.
Interface tương tự BM25Retriever.search() nhưng dùng BERT4Rec + FAISS.

Không cần query string hay metadata — chỉ cần user history.
Phù hợp cho Amazon 2014 5-core (không có product metadata).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from retrieval.bm25_retriever import SearchResult


# ---------------------------------------------------------------------------
# BERT4RecRetriever
# ---------------------------------------------------------------------------

class BERT4RecRetriever:
    """
    Stage-1 retriever backed by a trained BERT4Rec model + FAISS (or numpy).

    Parameters
    ----------
    model_dir   : directory containing best_model.pt, item_embs.npy (or item_faiss.index)
    item_num    : number of items (1-indexed)
    maxlen      : sequence length for BERT4Rec (default: 50)
    device      : torch device
    id_map_inv  : {item_idx_int: item_id_str} — to build SearchResult.item_id
    """

    def __init__(
        self,
        model_dir: str,
        item_num: int,
        maxlen: int = 50,
        device: torch.device = torch.device("cpu"),
        id_map_inv: Optional[Dict[int, str]] = None,
    ):
        from retrieval_models.bert4rec.model import BERT4Rec

        self.device      = device
        self.maxlen      = maxlen
        self.item_num    = item_num
        self.id_map_inv  = id_map_inv or {}

        model_path = Path(model_dir) / "best_model.pt"
        ckpt = torch.load(str(model_path), map_location=device, weights_only=True)

        # Restore model config from checkpoint
        saved_args  = ckpt.get("args", {})
        hidden_dim  = saved_args.get("hidden_dim",  64)
        num_heads   = saved_args.get("num_heads",   2)
        num_blocks  = saved_args.get("num_blocks",  2)
        dropout     = saved_args.get("dropout",     0.2)
        mask_prob   = saved_args.get("mask_prob",   0.2)
        model_maxlen = saved_args.get("maxlen",     maxlen)

        self.model = BERT4Rec(
            item_num     = item_num,
            maxlen       = model_maxlen,
            hidden_dim   = hidden_dim,
            num_heads    = num_heads,
            num_blocks   = num_blocks,
            dropout_rate = dropout,
            mask_prob    = mask_prob,
        ).to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.maxlen = model_maxlen

        # Load item embeddings
        faiss_path = Path(model_dir) / "item_faiss.index"
        npy_path   = Path(model_dir) / "item_embs.npy"

        self.faiss_index = None
        self.item_embs   = None   # (item_num, d) GPU tensor

        if faiss_path.exists():
            import faiss
            self.faiss_index = faiss.read_index(str(faiss_path))
            print(f"BERT4RecRetriever: loaded FAISS index ({faiss_path.stat().st_size/1e6:.0f} MB)", flush=True)
        elif npy_path.exists():
            arr = np.load(str(npy_path)).astype(np.float32)
            self.item_embs = torch.from_numpy(arr).to(device)
            print(f"BERT4RecRetriever: loaded item embs {arr.shape} on {device}", flush=True)
        else:
            # Build from model weights at load time
            print("BERT4RecRetriever: no pre-built index found, building from model...", flush=True)
            self.item_embs = self._build_item_embs()

        print(f"BERT4RecRetriever ready — item_num={item_num:,}", flush=True)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def _build_item_embs(self) -> torch.Tensor:
        parts = []
        ids_full = torch.arange(1, self.item_num + 1, device=self.device)
        for s in range(0, self.item_num, 8192):
            chunk = ids_full[s: s + 8192]
            emb   = F.normalize(self.model.embedding.token(chunk), dim=-1)
            parts.append(emb)
        return torch.cat(parts, dim=0)   # (item_num, d)

    # ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def get_user_embedding(self, history_ids: List[int]) -> torch.Tensor:
        """Encode user history → normalized (1, d) user vector."""
        seq = np.zeros(self.maxlen, dtype=np.int64)
        hist = history_ids[-self.maxlen:]
        seq[-len(hist):] = hist
        seq_t = torch.from_numpy(seq).unsqueeze(0).to(self.device)   # (1, L)
        return self.model.user_embedding(seq_t)   # (1, d)

    @torch.no_grad()
    def batch_get_user_embeddings(self, batch_history: List[List[int]]) -> np.ndarray:
        """
        Batch user embedding — 40x faster than sequential calls.
        Returns (B, d) float32 numpy array (already on CPU for FAISS).
        """
        B = len(batch_history)
        seqs = np.zeros((B, self.maxlen), dtype=np.int64)
        for i, hist in enumerate(batch_history):
            h = hist[-self.maxlen:]
            seqs[i, -len(h):] = h
        seqs_t = torch.from_numpy(seqs).to(self.device)   # (B, L)
        uembs  = self.model.user_embedding(seqs_t)         # (B, d)
        return uembs.cpu().numpy().astype(np.float32)

    def batch_search_by_history(
        self,
        batch_history: List[List[int]],
        top_k: int = 100,
    ) -> List[List[SearchResult]]:
        """
        Batch retrieval — compute all user embeddings at once, then batch FAISS.
        ~40x faster than calling search_by_history() in a loop.
        Returns List[List[SearchResult]], one list per user.
        """
        if not batch_history:
            return []
        uembs = self.batch_get_user_embeddings(batch_history)   # (B, d)
        if self.faiss_index is not None:
            all_scores, all_indices = self.faiss_index.search(uembs, top_k)
        else:
            uembs_gpu = torch.from_numpy(uembs).to(self.device)
            sims = torch.matmul(uembs_gpu, self.item_embs.T)   # (B, item_num)
            vals, idx = sims.topk(top_k, dim=1)
            all_scores  = vals.cpu().numpy()
            all_indices = idx.cpu().numpy()

        results_batch = []
        for i in range(len(batch_history)):
            results = []
            for sc, idx in zip(all_scores[i], all_indices[i]):
                if idx < 0:
                    continue
                item_idx = int(idx) + 1
                item_id  = self.id_map_inv.get(item_idx, str(item_idx))
                results.append(SearchResult(item_id=item_id, score=float(sc), title="", text=""))
            if not results and not batch_history[i]:
                results = [SearchResult(item_id=str(j), score=0.0, title="", text="")
                           for j in range(1, top_k + 1)]
            results_batch.append(results)
        return results_batch

    # ──────────────────────────────────────────────────────────────────────
    def search_by_history(
        self,
        history_ids: List[int],
        top_k: int = 100,
    ) -> List[SearchResult]:
        """
        Retrieve top-K items given user interaction history.

        Returns List[SearchResult] with:
          item_id = str(item_idx)   (1-indexed int as string)
          score   = cosine similarity
          title   = ""  (no metadata available)
          text    = ""
        """
        if not history_ids:
            # Cold-start: return items 1..top_k with uniform scores
            return [
                SearchResult(item_id=str(i), score=0.0, title="", text="")
                for i in range(1, top_k + 1)
            ]

        uemb = self.get_user_embedding(history_ids)   # (1, d)

        if self.faiss_index is not None:
            arr = uemb.cpu().numpy().astype(np.float32)
            scores, indices = self.faiss_index.search(arr, top_k)
            # FAISS returns 0-indexed positions → add 1 for item_id
            results = []
            for sc, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                item_idx = int(idx) + 1   # 1-indexed
                item_id  = self.id_map_inv.get(item_idx, str(item_idx))
                results.append(SearchResult(item_id=item_id, score=float(sc), title="", text=""))
            return results

        else:
            # numpy/GPU matmul fallback
            sims = torch.matmul(uemb, self.item_embs.T)[0]   # (item_num,)
            vals, idx = sims.topk(top_k)
            results = []
            for sc, item_idx0 in zip(vals.tolist(), idx.tolist()):
                item_idx = item_idx0 + 1   # 1-indexed
                item_id  = self.id_map_inv.get(item_idx, str(item_idx))
                results.append(SearchResult(item_id=item_id, score=float(sc), title="", text=""))
            return results

    def search(self, query: str, top_k: int = 100) -> List[SearchResult]:
        """BM25-compatible interface — query is ignored (no text metadata)."""
        # Called with empty query in no-metadata mode; returns cold-start
        return []
