"""
Smoke tests for TwoStageRLPipeline.

Dùng fake retriever + lightweight models — không cần data thật hay checkpoint.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from retrieval.bm25_retriever import SearchResult
from retrieval.two_stage_pipeline import TwoStageRLPipeline
from retrieval.unified_pipeline import UnifiedRLPolicy
from models.submodular import RerankerBackedSubmodular
from utils.encoders import StateEncoder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUM_ITEMS    = 50
EMBED_DIM    = 16
SLATE_SIZE   = 4
N_RETRIEVE   = 20
HISTORY_LEN  = 10
DEVICE       = torch.device("cpu")

# item_id_map: str_id "1".."NUM_ITEMS" → int_idx 1..NUM_ITEMS
ITEM_ID_MAP = {str(i): i for i in range(1, NUM_ITEMS + 1)}


class FakeRetriever:
    """Returns top_k fake SearchResults with deterministic scores."""

    def search_by_history(self, history_ids: List[int], top_k: int) -> List[SearchResult]:
        top_k = min(top_k, NUM_ITEMS)
        return [
            SearchResult(item_id=str(i), score=float(NUM_ITEMS - i), title="", text="")
            for i in range(1, top_k + 1)
        ]

    def batch_search_by_history(
        self, batch_histories: List[List[int]], top_k: int
    ) -> List[List[SearchResult]]:
        return [self.search_by_history(h, top_k) for h in batch_histories]


def make_pipeline() -> TwoStageRLPipeline:
    submodular    = RerankerBackedSubmodular(num_items=NUM_ITEMS + 1, embed_dim=EMBED_DIM)
    state_encoder = StateEncoder(num_items=NUM_ITEMS, embed_dim=EMBED_DIM)
    rl_policy     = UnifiedRLPolicy(state_dim=EMBED_DIM, hidden_dim=32)

    return TwoStageRLPipeline(
        retriever=FakeRetriever(),
        submodular=submodular,
        rl_policy=rl_policy,
        state_encoder=state_encoder,
        item_id_map=ITEM_ID_MAP,
        device=DEVICE,
        n_retrieve=N_RETRIEVE,
        slate_size=SLATE_SIZE,
        history_length=HISTORY_LEN,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_search_returns_slate():
    pipeline = make_pipeline()
    history = list(range(1, 6))
    results, session_id = pipeline.search(history_ids=history)

    assert session_id is None
    assert len(results) <= SLATE_SIZE
    assert len(results) > 0, "slate should be non-empty"
    for r in results:
        assert r.item_idx > 0
        assert 0.0 <= r.rel_score


def test_search_empty_history():
    pipeline = make_pipeline()
    results, _ = pipeline.search(history_ids=[])
    assert isinstance(results, list)
    assert len(results) <= SLATE_SIZE


def test_search_deterministic_same_slate():
    import random as _random
    pipeline = make_pipeline()
    history = list(range(1, 6))
    # Seed both torch and python random to neutralise ε-greedy randomness
    torch.manual_seed(0); _random.seed(0)
    r1, _ = pipeline.search(history_ids=history, deterministic=True)
    torch.manual_seed(0); _random.seed(0)
    r2, _ = pipeline.search(history_ids=history, deterministic=True)
    assert [r.item_idx for r in r1] == [r.item_idx for r in r2]


def test_search_no_duplicate_items():
    pipeline = make_pipeline()
    results, _ = pipeline.search(history_ids=[1, 2, 3])
    indices = [r.item_idx for r in results]
    assert len(indices) == len(set(indices)), "slate must not have duplicates"


def test_fixed_alpha_bypasses_rl():
    pipeline = make_pipeline()
    pipeline.fixed_alpha = 0.9
    history = list(range(1, 6))
    results, _ = pipeline.search(history_ids=history)
    assert len(results) > 0


def test_encode_state_shape():
    pipeline = make_pipeline()
    state = pipeline.encode_state([1, 2, 3, 4])
    assert state.ndim == 1
    assert state.shape[0] == EMBED_DIM


def test_collect_transition_keys():
    pipeline = make_pipeline()
    trans = pipeline.collect_transition(
        history_ids=[1, 2, 3],
        budget=float(SLATE_SIZE),
        target_item_idx=5,
        reward=1.0,
        inject_target=True,
    )
    assert trans is not None
    required_keys = {"state", "action", "slate", "rel_scores",
                     "slate_cands_idx", "slate_cands_scores",
                     "target", "reward", "alpha", "kappa", "budget"}
    assert required_keys <= trans.keys(), f"Missing keys: {required_keys - trans.keys()}"


def test_collect_transition_inject_target():
    pipeline = make_pipeline()
    # Pick a valid item idx that FakeRetriever won't return (> N_RETRIEVE) but
    # still within the embedding table (< NUM_ITEMS).
    target = NUM_ITEMS - 1   # idx 49, not in retrieval (retriever returns 1..N_RETRIEVE=20)

    trans = pipeline.collect_transition(
        history_ids=[1, 2],
        target_item_idx=target,
        inject_target=True,
    )
    assert trans is not None
    assert target in trans["slate_cands_idx"], "target should be injected into candidates"


def test_collect_transition_no_inject():
    pipeline = make_pipeline()
    trans = pipeline.collect_transition(
        history_ids=[1, 2, 3],
        target_item_idx=99,
        inject_target=False,
    )
    assert trans is not None
    assert isinstance(trans["slate"], list)


@dataclass
class FakeStep:
    history_ids: List[int]
    history_extras: Optional[List[float]] = None
    seen_ids: List[int] = field(default_factory=list)
    budget: float = float(SLATE_SIZE)
    item_id: int = 1
    reward: Optional[float] = None
    event: Optional[str] = None


def test_batch_evaluate_length():
    pipeline = make_pipeline()
    steps = [FakeStep(history_ids=list(range(1, 6))) for _ in range(5)]
    slates = pipeline.batch_evaluate(steps, retrieval_batch_size=3)
    assert len(slates) == 5
    for s in slates:
        assert len(s) <= SLATE_SIZE


def test_batch_evaluate_seen_ids_filtered():
    pipeline = make_pipeline()
    # Block first N_RETRIEVE-1 items as seen
    seen = list(range(1, N_RETRIEVE))
    steps = [FakeStep(history_ids=[1, 2], seen_ids=seen)]
    slates = pipeline.batch_evaluate(steps)
    # remaining candidate is item N_RETRIEVE (score 0 or low)
    for idx in slates[0]:
        assert idx not in seen, f"seen item {idx} appeared in slate"


def test_batch_evaluate_all_seen_returns_empty():
    pipeline = make_pipeline()
    seen = list(range(1, NUM_ITEMS + 1))  # every possible item
    steps = [FakeStep(history_ids=[1, 2], seen_ids=seen)]
    slates = pipeline.batch_evaluate(steps)
    assert slates[0] == []
