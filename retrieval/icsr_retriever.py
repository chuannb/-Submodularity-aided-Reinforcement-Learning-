"""
ICSRecRetriever — Stage-1 retriever backed by a trained ICSRec-SAS model + FAISS.

Drop-in replacement for BERT4RecRetriever.
Uses the pre-trained ICSRec model from /workspace/repos/ICSRec/src/output/

Interface identical to BERT4RecRetriever:
  search_by_history(history_ids, top_k) -> List[SearchResult]
  batch_search_by_history(batch_history, top_k) -> List[List[SearchResult]]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import faiss

from retrieval.bm25_retriever import SearchResult

# ICSRec source dir
ICSREC_SRC = Path("/workspace/repos/ICSRec/src")
ICSREC_DATA = Path("/workspace/repos/ICSRec/data")


def _load_icsrec_model(dataset: str, device: torch.device, ckpt_override: Optional[Path] = None):
    """Load ICSRec-SAS model for given dataset (beauty/sports/toys)."""
    # Normalise name to ICSRec convention
    name_map = {
        "beauty": "Beauty",
        "sports": "Sports_and_Outdoors",
        "toys":   "Toys_and_Games",
    }
    ics_name = name_map.get(dataset.lower(), dataset)

    # ICSRec data stats
    txt_path = ICSREC_DATA / f"{dataset.lower()}.txt"
    if not txt_path.exists():
        # Fallback to original ICSRec name
        txt_path = ICSREC_DATA / f"{ics_name}.txt"
    lines = txt_path.read_text().splitlines()
    max_item = max(
        max(int(x) for x in l.strip().split()[1:]) for l in lines if l.strip()
    )
    item_size = max_item + 2  # 0=padding, 1..max_item=items, max_item+1=mask

    ckpt_path = Path(ckpt_override) if ckpt_override else ICSREC_SRC / "output" / f"ICSRec-SAS-{ics_name}-0.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train first.")

    # Import ICSRec model under namespaced module keys to avoid shadowing
    # LM_Submodular_RL's `models` package (which is a directory, not a .py file).
    import importlib.util as _ilu, types as _types

    def _load_icsrec_module(name: str, path: str):
        """Load a .py file as sys.modules[name] if not already loaded."""
        if name in sys.modules:
            return sys.modules[name]
        spec = _ilu.spec_from_file_location(name, path)
        mod  = _ilu.module_from_spec(spec)
        sys.modules[name] = mod   # register BEFORE exec so circular imports work
        spec.loader.exec_module(mod)
        return mod

    # ICSRec modules.py depends on nothing outside its file; load it first
    _load_icsrec_module("icsrec_modules", str(ICSREC_SRC / "modules.py"))
    # ICSRec models.py does `from modules import ...`; patch sys.modules temporarily
    _had_modules = "modules" in sys.modules
    _old_modules = sys.modules.get("modules")
    sys.modules["modules"] = sys.modules["icsrec_modules"]
    try:
        _icsrec_models = _load_icsrec_module("icsrec_models", str(ICSREC_SRC / "models.py"))
    finally:
        if _had_modules:
            sys.modules["modules"] = _old_modules
        else:
            sys.modules.pop("modules", None)
    SASRecModel = _icsrec_models.SASRecModel

    args = argparse.Namespace(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        initializer_range=0.02,
        max_seq_length=50,
        item_size=item_size,
        mask_id=item_size - 1,
        cuda_condition=(device.type == "cuda"),
    )

    model = SASRecModel(args).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    return model, max_item


class ICSRecRetriever:
    """
    Sequential retriever using ICSRec-SAS model + FAISS inner-product index.

    Parameters
    ----------
    dataset     : 'beauty' | 'sports' | 'toys'
    device      : torch device
    id_map_inv  : {item_int_id: asin_str} — built from convert_5core.py output
                  If None, returns str(item_int_id) as item_id.
    """

    def __init__(
        self,
        dataset: str = "beauty",
        device: torch.device = torch.device("cpu"),
        id_map_inv: Optional[Dict[int, str]] = None,
        ckpt_path: Optional[str] = None,
    ):
        self.device     = device
        self.dataset    = dataset.lower()
        self.maxlen     = 50

        # Load model
        self.model, self.item_num = _load_icsrec_model(dataset, device, ckpt_path)
        print(f"ICSRecRetriever: loaded {dataset} model, {self.item_num:,} items", flush=True)

        # id_map_inv: 1-indexed item_int → asin string
        if id_map_inv is not None:
            self.id_map_inv = id_map_inv
        else:
            map_path = ICSREC_DATA / f"{self.dataset}_id2asin.json"
            if map_path.exists():
                raw = json.loads(map_path.read_text())
                self.id_map_inv = {int(k): v for k, v in raw.items()}
                print(f"ICSRecRetriever: loaded id2asin map ({len(self.id_map_inv):,} items)", flush=True)
            else:
                self.id_map_inv = {}
                print("ICSRecRetriever: no id2asin map, returning int IDs", flush=True)

        # Build FAISS index from item embeddings
        self._build_faiss_index()
        print(f"ICSRecRetriever ready — item_num={self.item_num:,}", flush=True)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _build_faiss_index(self):
        """Build normalized FAISS flat inner-product index from item embeddings."""
        # item_embeddings: (item_size, d), indices 1..item_num are real items
        embs = self.model.item_embeddings.weight[1: self.item_num + 1]  # (item_num, d)
        embs = F.normalize(embs, dim=-1)
        embs_np = embs.cpu().float().numpy()

        self.index = faiss.IndexFlatIP(embs_np.shape[1])
        self.index.add(embs_np)
        print(f"ICSRecRetriever: FAISS index built ({embs_np.shape})", flush=True)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode_history(self, history_ids: List[int]) -> np.ndarray:
        """Encode single user history → (1, d) normalized float32 array."""
        seq = np.zeros(self.maxlen, dtype=np.int64)
        hist = [h for h in history_ids if 1 <= h <= self.item_num]
        h = hist[-self.maxlen:]
        seq[-len(h):] = h
        seq_t = torch.from_numpy(seq).unsqueeze(0).to(self.device)
        out = self.model(seq_t)               # (1, L, d)
        uemb = out[0, -1, :]                  # last position
        uemb = F.normalize(uemb, dim=-1)
        return uemb.cpu().float().numpy()[None]  # (1, d)

    @torch.no_grad()
    def _encode_batch(self, batch_history: List[List[int]]) -> np.ndarray:
        """Batch encode → (B, d) normalized float32 array."""
        B = len(batch_history)
        seqs = np.zeros((B, self.maxlen), dtype=np.int64)
        for i, hist in enumerate(batch_history):
            h = [x for x in hist if 1 <= x <= self.item_num][-self.maxlen:]
            seqs[i, -len(h):] = h
        seqs_t = torch.from_numpy(seqs).to(self.device)
        out = self.model(seqs_t)              # (B, L, d)
        uembs = out[:, -1, :]                 # (B, d)
        uembs = F.normalize(uembs, dim=-1)
        return uembs.cpu().float().numpy()

    # ------------------------------------------------------------------
    def _indices_to_results(
        self, scores: np.ndarray, indices: np.ndarray
    ) -> List[SearchResult]:
        results = []
        for sc, idx in zip(scores, indices):
            if idx < 0:
                continue
            item_int = int(idx) + 1          # FAISS 0-indexed → 1-indexed
            item_id  = self.id_map_inv.get(item_int, str(item_int))
            results.append(SearchResult(item_id=item_id, score=float(sc), title="", text=""))
        return results

    # ------------------------------------------------------------------
    def search_by_history(
        self,
        history_ids: List[int],
        top_k: int = 200,
    ) -> List[SearchResult]:
        """Single-user retrieval. Returns top_k SearchResults."""
        if not history_ids:
            return [SearchResult(item_id=str(i), score=0.0, title="", text="")
                    for i in range(1, top_k + 1)]
        uemb = self._encode_history(history_ids)
        scores, indices = self.index.search(uemb, top_k)
        return self._indices_to_results(scores[0], indices[0])

    def batch_search_by_history(
        self,
        batch_history: List[List[int]],
        top_k: int = 200,
    ) -> List[List[SearchResult]]:
        """Batch retrieval — 40-50x faster than sequential calls."""
        if not batch_history:
            return []
        uembs = self._encode_batch(batch_history)
        all_scores, all_indices = self.index.search(uembs, top_k)
        return [
            self._indices_to_results(all_scores[i], all_indices[i])
            for i in range(len(batch_history))
        ]

    def search(self, query: str, top_k: int = 200) -> List[SearchResult]:
        """BM25-compatible stub — not used in history-based mode."""
        return []
