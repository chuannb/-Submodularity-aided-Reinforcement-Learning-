"""
TwoStageRLPipeline
==================

Full recommendation pipeline: Retrieval Stage → Submodular RL Reranking → Slate.

Pipeline flow:
    User history → Retriever (ANN recall or search)
                 → RL policy  (outputs α_t, η_t)
                 → Submodular greedy selector (F_θ(S | α_t))
                 → Slate S_t  (k items)

Stage mapping:
    Stage 1 — retrieval:  Generic Retriever (must implement search_by_history)
    Stage 2 — selection:  actor-critic outputs (α_t, κ_t); submodular greedy builds slate
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from algorithms.greedy_selector import budgeted_submodular_greedy_reranker
from models.submodular import RerankerBackedSubmodular
from retrieval.unified_pipeline import ScoredCandidate, UnifiedRLPolicy, UnifiedSearchResult
from utils.encoders import StateEncoder, pad_history


class TwoStageRLPipeline:
    """
    Wires a Retriever, RerankerBackedSubmodular, and UnifiedRLPolicy
    into a single search/training interface.

    The RL policy outputs a 2-D action (α_t, κ_t):
        α_t — relevance-diversity trade-off weight in F_θ(S | α_t)
        κ_t — training-time exploration intensity (ε-greedy + softmax temperature)
    """

    def __init__(
        self,
        retriever: Any,                       # Must implement search_by_history
        submodular: RerankerBackedSubmodular,
        rl_policy: UnifiedRLPolicy,
        state_encoder: StateEncoder,
        item_id_map: Dict[str, int],          # str_id → int_idx (1-indexed; 0 = padding)
        device: torch.device = torch.device("cpu"),
        n_retrieve: int = 200,                # candidate pool size (m in the paper)
        slate_size: int = 10,                 # final slate size k
        history_length: int = 20,             # user history window h
        item_costs: Optional[Dict[int, float]] = None,
    ):
        self.retriever      = retriever
        self.submodular     = submodular
        self.rl_policy      = rl_policy
        self.state_encoder  = state_encoder
        self.item_id_map    = item_id_map
        self.item_idx_map   = {v: k for k, v in item_id_map.items()}   # int_idx → str_id
        self.device         = device
        self.n_retrieve     = n_retrieve
        self.slate_size     = slate_size
        self.history_length = history_length
        self.item_costs     = item_costs or {}
        self.fixed_alpha: Optional[float] = None   # override RL α; set externally if needed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_candidates(self, raw_results) -> List[ScoredCandidate]:
        """Convert retriever results to ScoredCandidate, dropping invalid indices."""
        candidates = []
        for r in raw_results:
            item_idx = self.item_id_map.get(r.item_id, -1)
            if item_idx < 0:
                try:
                    item_idx = int(r.item_id)
                except ValueError:
                    continue
            if item_idx <= 0:   # 0 is the padding token; skip
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

    def _retrieve(self, history_ids: List[int]) -> List[ScoredCandidate]:
        """Stage 1: ANN recall → sorted candidate list."""
        raw = self.retriever.search_by_history(history_ids, top_k=self.n_retrieve)
        return self._parse_candidates(raw)

    def _encode_state(
        self,
        history_ids: List[int],
        history_extras: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """Encode user history into a state vector (1, state_dim)."""
        ids_t, ext_t = pad_history(
            [history_ids],
            [history_extras] if history_extras else None,
            self.history_length,
            self.device,
        )
        with torch.no_grad():
            return self.state_encoder(ids_t, ext_t)

    def _run_policy(
        self,
        history_ids: List[int],
        history_extras: Optional[List[float]],
        deterministic: bool,
    ) -> Tuple[torch.Tensor, float, float, np.ndarray]:
        """
        Encode state and sample action from the RL policy.

        Returns:
            state       — encoded state tensor (1, state_dim)
            alpha_t     — relevance-diversity trade-off ∈ [0, 1]
            kappa_t     — exploration intensity ∈ [0, 1]
            raw_action  — pre-squash latent action (for replay buffer)
        """
        state         = self._encode_state(history_ids, history_extras)
        policy_action = self.rl_policy.act(state, deterministic=deterministic)
        alpha_t       = (
            self.fixed_alpha
            if self.fixed_alpha is not None
            else float(policy_action["alpha"].item())
        )
        kappa_t       = float(policy_action["kappa"].item())
        raw_action    = policy_action["raw"].detach().cpu().numpy()[0]
        return state, alpha_t, kappa_t, raw_action

    def _build_slate(
        self,
        candidates: List[ScoredCandidate],
        alpha_t: float,
        kappa_t: float,
        budget: float,
    ) -> Tuple[List[int], float]:
        """
        Stage 2: run the submodular greedy selector.

        Returns:
            selected_indices — item_idx list (length ≤ slate_size)
            slate_score      — final submodular objective value
        """
        candidate_indices = [c.item_idx for c in candidates]
        relevance_scores  = {c.item_idx: c.rel_score for c in candidates}
        costs             = {c.item_idx: self.item_costs.get(c.item_idx, 1.0)
                             for c in candidates}

        selected_indices, slate_score = budgeted_submodular_greedy_reranker(
            candidates=candidate_indices,
            reranker_score_map=relevance_scores,
            utility=self.submodular,
            slate_size=self.slate_size,
            budget=budget,
            costs=costs,
            alpha_override=alpha_t,
            kappa=kappa_t,
        )
        return selected_indices, slate_score

    def encode_state(
        self,
        history_ids: List[int],
        history_extras: Optional[List[float]] = None,
    ) -> np.ndarray:
        return self._encode_state(history_ids, history_extras).detach().cpu().numpy()[0]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(
        self,
        query: str = "",                   # unused — no text metadata in Amazon 5-core
        history_ids: Optional[List[int]] = None,
        history_extras: Optional[List[float]] = None,
        session_id: Optional[str] = None,
        budget: Optional[float] = None,
        page: int = 1,
        deterministic: bool = True,
        **kwargs,
    ) -> Tuple[List[UnifiedSearchResult], Optional[str]]:
        """
        Evaluation-mode inference: retrieve → policy → greedy → slate.
        Returns (results, session_id_passthrough).
        """
        history_ids = history_ids or []
        budget      = budget or float(self.slate_size)

        candidates = self._retrieve(history_ids)
        if not candidates:
            return [], None

        state, alpha_t, kappa_t, _ = self._run_policy(
            history_ids, history_extras, deterministic=deterministic
        )

        selected_indices, slate_score = self._build_slate(
            candidates, alpha_t, kappa_t, budget
        )

        relevance_scores  = {c.item_idx: c.rel_score for c in candidates}
        candidates_by_idx = {c.item_idx: c for c in candidates}

        results = [
            UnifiedSearchResult(
                item_id=candidates_by_idx[idx].item_id
                        if idx in candidates_by_idx else str(idx),
                item_idx=idx,
                rel_score=relevance_scores.get(idx, 0.0),
                submodular_score=slate_score,
                title="",
                slate_position=pos,
            )
            for pos, idx in enumerate(selected_indices)
            if idx in relevance_scores
        ]
        return results, None

    # ------------------------------------------------------------------

    def batch_evaluate(
        self,
        steps: list,
        retrieval_batch_size: int = 512,
    ) -> list:
        """
        Batched evaluation: groups queries for throughput (~2× faster
        than sequential search()).

        Args:
            steps                — list of TrajectoryStep-like objects with
                                   .history_ids, .history_extras, .seen_ids, .budget
            retrieval_batch_size — batch size for retriever ANN search

        Returns:
            List[List[int]] — one item_idx slate per step.
        """
        all_slates = []
        for start in range(0, len(steps), retrieval_batch_size):
            batch = steps[start: start + retrieval_batch_size]

            # Stage 1: batch retrieval
            histories = [s.history_ids for s in batch]
            batch_retrieval_results = self.retriever.batch_search_by_history(
                histories, top_k=self.n_retrieve
            )

            # Stage 2: per-user state encoding, policy, greedy selection
            for step, retrieval_results in zip(batch, batch_retrieval_results):
                seen_items = set(step.seen_ids) if getattr(step, "seen_ids", None) else set()

                candidates = [
                    c for c in self._parse_candidates(retrieval_results)
                    if c.item_idx not in seen_items
                ]

                if not candidates:
                    all_slates.append([])
                    continue

                _, alpha_t, kappa_t, _ = self._run_policy(
                    step.history_ids,
                    getattr(step, "history_extras", None),
                    deterministic=True,
                )
                budget = getattr(step, "budget", None) or float(self.slate_size)

                selected_indices, _ = self._build_slate(
                    candidates, alpha_t, kappa_t, budget
                )
                all_slates.append(selected_indices)

        return all_slates

    # ------------------------------------------------------------------

    def collect_transition(
        self,
        query: str = "",                   # unused
        history_ids: Optional[List[int]] = None,
        history_extras: Optional[List[float]] = None,
        budget: float = 10.0,
        target_item_idx: int = -1,
        reward: float = 0.0,
        inject_target: bool = True,
        **kwargs,
    ) -> Optional[dict]:
        """
        Training step: retrieve → (optionally inject target) → policy → greedy.
        Returns a transition dict ready for the replay buffer, or None if
        retrieval returns no candidates.
        """
        history_ids = history_ids or []
        candidates  = self._retrieve(history_ids)
        if not candidates:
            return None

        # Guarantee the target item is reachable during training
        if inject_target and target_item_idx > 0:
            if not any(c.item_idx == target_item_idx for c in candidates):
                target_str_id = self.item_idx_map.get(target_item_idx, str(target_item_idx))
                top_score     = max((c.rel_score for c in candidates), default=1.0)
                candidates.append(ScoredCandidate(
                    item_id=target_str_id,
                    item_idx=target_item_idx,
                    rel_score=top_score + 0.01,
                    title="",
                    text="",
                ))

        state, alpha_t, kappa_t, raw_action = self._run_policy(
            history_ids, history_extras, deterministic=False
        )

        selected_indices, _ = self._build_slate(candidates, alpha_t, kappa_t, budget)

        relevance_scores = {c.item_idx: c.rel_score for c in candidates}

        return {
            "state":              state.detach().cpu().numpy()[0],
            "action":             raw_action,
            "slate":              selected_indices,
            "rel_scores":         [relevance_scores[i] for i in selected_indices
                                   if i in relevance_scores],
            "slate_cands_idx":    [c.item_idx for c in candidates],
            "slate_cands_scores": [c.rel_score for c in candidates],
            "target":             target_item_idx,
            "reward":             reward,
            "alpha":              alpha_t,
            "kappa":              kappa_t,
            "budget":             budget,
        }
