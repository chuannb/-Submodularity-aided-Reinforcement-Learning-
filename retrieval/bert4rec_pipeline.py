"""
BERT4RecPipeline
================

Thay thế BM25 + Qwen3-Reranker bằng BERT4Rec cho stage retrieval.
Giữ nguyên Submodular + RL (stage 2).

Pipeline flow:
  User history → BERT4Rec (ANN recall + cosine similarity score)
              → Submodular f_θ + RL policy
              → Slate S_t

Ưu điểm so với BM25 + Reranker:
  1. Không cần metadata (phù hợp Amazon 5-core không có product title)
  2. Train/eval dùng cùng scorer (BERT4Rec cosine sim) — không mismatch
  3. Nhanh hơn Qwen3-Reranker (~50ms vs ~2s per query)

Interface tương tự UnifiedPipeline — dùng được với UnifiedJointTrainer.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from algorithms.greedy_selector import budgeted_submodular_greedy_reranker
from models.submodular import RerankerBackedSubmodular
from retrieval.bert4rec_retriever import BERT4RecRetriever
from retrieval.unified_pipeline import ScoredCandidate, UnifiedRLPolicy, UnifiedSearchResult
from utils.encoders import StateEncoder, pad_history


class BERT4RecPipeline:
    """
    Full pipeline: BERT4Rec retrieval → Submodular(RL) → Slate.

    Drop-in replacement cho UnifiedPipeline:
      search(query, history_ids, ...) → List[UnifiedSearchResult]
      collect_transition(query, history_ids, ...) → dict
      encode_state(history_ids, ...) → np.ndarray
    """

    def __init__(
        self,
        retriever: BERT4RecRetriever,
        submodular: RerankerBackedSubmodular,
        rl_policy: UnifiedRLPolicy,
        state_encoder: StateEncoder,
        id_map: Dict[str, int],           # str_id → int_idx (1-indexed)
        device: torch.device = torch.device("cpu"),
        n_retrieve: int = 200,            # top-K từ BERT4Rec
        slate_size: int = 10,
        history_length: int = 20,
        costs_map: Optional[Dict[int, float]] = None,
    ):
        self.retriever      = retriever
        self.submodular     = submodular
        self.rl_policy      = rl_policy
        self.state_encoder  = state_encoder
        self.id_map         = id_map
        self.id_map_inv     = {v: k for k, v in id_map.items()}
        self.device         = device
        self.n_retrieve     = n_retrieve
        self.slate_size     = slate_size
        self.history_length = history_length
        self.costs_map      = costs_map or {}

    # ------------------------------------------------------------------
    def _retrieve(self, history_ids: List[int]) -> List[ScoredCandidate]:
        """BERT4Rec ANN recall — trả về candidates sorted by cosine sim."""
        results = self.retriever.search_by_history(history_ids, top_k=self.n_retrieve)
        candidates = []
        for r in results:
            # Thử tra id_map trước, sau đó parse int từ str
            item_idx = self.id_map.get(r.item_id, -1)
            if item_idx < 0:
                try:
                    item_idx = int(r.item_id)
                except ValueError:
                    continue
            if item_idx <= 0:
                continue
            candidates.append(ScoredCandidate(
                item_id=r.item_id,
                item_idx=item_idx,
                rel_score=float(r.score),
                title="",
                text="",
            ))
        candidates.sort(key=lambda c: c.rel_score, reverse=True)
        return candidates

    # ------------------------------------------------------------------
    def _encode_state(
        self,
        history_ids: List[int],
        history_extras: Optional[List[float]] = None,
    ) -> torch.Tensor:
        ids_t, ext_t = pad_history(
            [history_ids],
            [history_extras] if history_extras else None,
            self.history_length,
            self.device,
        )
        with torch.no_grad():
            return self.state_encoder(ids_t, ext_t)   # (1, embed_dim)

    def encode_state(
        self,
        history_ids: List[int],
        history_extras: Optional[List[float]] = None,
    ) -> np.ndarray:
        return self._encode_state(history_ids, history_extras).detach().cpu().numpy()[0]

    # ------------------------------------------------------------------
    def search(
        self,
        query: str = "",                  # ignored — không có text metadata
        history_ids: Optional[List[int]] = None,
        history_extras: Optional[List[float]] = None,
        session_id: Optional[str] = None,
        budget: Optional[float] = None,
        page: int = 1,
        deterministic: bool = True,
        **kwargs,
    ) -> Tuple[List[UnifiedSearchResult], Optional[str]]:
        """
        Full pipeline eval: BERT4Rec recall → RL → submodular select.
        `query` bị bỏ qua — dùng history_ids cho BERT4Rec.
        """
        history_ids = history_ids or []
        budget      = budget or float(self.slate_size)

        candidates = self._retrieve(history_ids)
        if not candidates:
            return [], None

        state   = self._encode_state(history_ids, history_extras)
        knobs   = self.rl_policy.act(state, deterministic=deterministic)
        alpha_t = float(knobs["alpha"].item())
        kappa_t = float(knobs["kappa"].item())

        cand_idx_list = [c.item_idx for c in candidates]
        rel_score_map = {c.item_idx: c.rel_score for c in candidates}
        cost_map      = {c.item_idx: self.costs_map.get(c.item_idx, 1.0) for c in candidates}

        slate_idx, final_sub_score = budgeted_submodular_greedy_reranker(
            candidates=cand_idx_list,
            reranker_score_map=rel_score_map,
            utility=self.submodular,
            slate_size=self.slate_size,
            budget=budget,
            costs=cost_map,
            alpha_override=alpha_t,
            kappa=kappa_t,
        )

        idx_to_cand = {c.item_idx: c for c in candidates}
        results = [
            UnifiedSearchResult(
                item_id=idx_to_cand[idx].item_id if idx in idx_to_cand else str(idx),
                item_idx=idx,
                rel_score=rel_score_map.get(idx, 0.0),
                submodular_score=final_sub_score,
                title="",
                slate_position=pos,
            )
            for pos, idx in enumerate(slate_idx)
            if idx in rel_score_map
        ]
        return results, None

    # ------------------------------------------------------------------
    def batch_evaluate(
        self,
        steps: list,
        retrieval_batch_size: int = 512,
    ) -> list:
        """
        Batch-optimized eval: groups user embedding computation + FAISS,
        then runs greedy per-user.  ~2x faster than sequential search().
        Returns List[List[int]] — one slate (item_idx list) per step.
        """
        all_slates = []
        for start in range(0, len(steps), retrieval_batch_size):
            batch = steps[start: start + retrieval_batch_size]
            # 1. Batch BERT4Rec retrieval
            histories = [s.history_ids for s in batch]
            all_results = self.retriever.batch_search_by_history(
                histories, top_k=self.n_retrieve
            )
            # 2. Per-user: state encode + RL + greedy
            for step, results in zip(batch, all_results):
                candidates = []
                for r in results:
                    item_idx = self.id_map.get(r.item_id, -1)
                    if item_idx < 0:
                        try:
                            item_idx = int(r.item_id)
                        except ValueError:
                            continue
                    if item_idx <= 0:
                        continue
                    candidates.append(ScoredCandidate(
                        item_id=r.item_id, item_idx=item_idx,
                        rel_score=float(r.score), title="", text="",
                    ))
                candidates.sort(key=lambda c: c.rel_score, reverse=True)

                if not candidates:
                    all_slates.append([])
                    continue

                state   = self._encode_state(step.history_ids, step.history_extras)
                knobs   = self.rl_policy.act(state, deterministic=True)
                alpha_t = float(knobs["alpha"].item())
                kappa_t = float(knobs["kappa"].item())

                cand_idx_list = [c.item_idx for c in candidates]
                rel_score_map = {c.item_idx: c.rel_score for c in candidates}
                cost_map      = {c.item_idx: self.costs_map.get(c.item_idx, 1.0) for c in candidates}
                budget        = step.budget or float(self.slate_size)

                slate_idx, _ = budgeted_submodular_greedy_reranker(
                    candidates=cand_idx_list,
                    reranker_score_map=rel_score_map,
                    utility=self.submodular,
                    slate_size=self.slate_size,
                    budget=budget,
                    costs=cost_map,
                    alpha_override=alpha_t,
                    kappa=kappa_t,
                )
                all_slates.append(slate_idx)
        return all_slates

    # ------------------------------------------------------------------
    def collect_transition(
        self,
        query: str = "",                # ignored
        history_ids: Optional[List[int]] = None,
        history_extras: Optional[List[float]] = None,
        budget: float = 10.0,
        target_item_idx: int = -1,
        reward: float = 0.0,
        dataset_type: str = "amazon",
        stars: float = None,
        fast_mode: bool = False,        # ignored — BERT4Rec is always fast
        inject_target: bool = True,
        **kwargs,
    ) -> Optional[dict]:
        """
        Training step: BERT4Rec recall → inject target → RL → submodular.
        Trả về transition dict cho replay buffer.
        """
        history_ids = history_ids or []
        candidates  = self._retrieve(history_ids)
        if not candidates:
            return None

        if inject_target and target_item_idx > 0:
            if not any(c.item_idx == target_item_idx for c in candidates):
                target_str_id = self.id_map_inv.get(target_item_idx, str(target_item_idx))
                max_score     = max((c.rel_score for c in candidates), default=1.0)
                candidates.append(ScoredCandidate(
                    item_id=target_str_id,
                    item_idx=target_item_idx,
                    rel_score=max_score + 0.01,
                    title="",
                    text="",
                ))

        state   = self._encode_state(history_ids, history_extras)
        knobs   = self.rl_policy.act(state, deterministic=False)
        alpha_t = float(knobs["alpha"].item())
        kappa_t = float(knobs["kappa"].item())
        raw_action = knobs["raw"].detach().cpu().numpy()[0]

        cand_idx_list = [c.item_idx for c in candidates]
        rel_score_map = {c.item_idx: c.rel_score for c in candidates}
        cost_map      = {c.item_idx: self.costs_map.get(c.item_idx, 1.0) for c in candidates}

        slate_idx, _ = budgeted_submodular_greedy_reranker(
            candidates=cand_idx_list,
            reranker_score_map=rel_score_map,
            utility=self.submodular,
            slate_size=self.slate_size,
            budget=budget,
            costs=cost_map,
            alpha_override=alpha_t,
            kappa=kappa_t,
        )

        return {
            "state": state.detach().cpu().numpy()[0],
            "action": raw_action,
            "slate": slate_idx,
            "rel_scores": [rel_score_map[i] for i in slate_idx if i in rel_score_map],
            "slate_cands_idx": cand_idx_list,
            "slate_cands_scores": list(rel_score_map.values()),
            "target": target_item_idx,
            "reward": reward,
            "alpha": alpha_t,
            "kappa": kappa_t,
            "budget": budget,
        }
