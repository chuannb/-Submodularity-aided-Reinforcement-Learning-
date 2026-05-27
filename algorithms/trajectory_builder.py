
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from utils.encoders import StateEncoder, encode_history


# ---------------------------------------------------------------------------
# Data class for a single trajectory step
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryStep:
    user_id: int
    item_id: int             # y_t = ground truth next item
    history_ids: List[int]   # last L item ids
    budget: float            # B_t
    split: str               # "train" | "val" | "test"

    # Optional fields
    history_extras: Optional[List[float]] = None  # stars or event weights
    event: Optional[str] = None                   # RetailRocket event type
    reward: Optional[float] = None                # explicit reward
    seen_ids: Optional[List[int]] = None          # items to exclude from candidates (train+val)


# ---------------------------------------------------------------------------
# Algorithm 1 implementation
# ---------------------------------------------------------------------------


def build_trajectories_v2(
    samples: list,
    item2id: dict,
    price_map: dict,
    history_length: int = 20,
    slate_size: int = 10,
) -> List[TrajectoryStep]:
    """
    Build trajectory steps from V2 pre-split dataset.

    Each sample dict:
      user_id, history:[{asin, stars, ts}], target_asin,
      target_stars, r_hit (user-normalized reward), user_mean_stars, ts

    r_hit is stored in step.reward for direct use during training.
    Budget = mean_price * slate_size so greedy can always fill a full slate.
    Falls back to slate_size when no price data is available.
    """
    steps: List[TrajectoryStep] = []
    for rec in samples:
        target_asin = rec.get("target_asin", "")
        if not target_asin or target_asin not in item2id:
            continue

        target_id = item2id[target_asin]
        history   = rec.get("history", [])

        # Trim to history_length most recent
        history = history[-history_length:]

        history_ids   = [item2id[h["asin"]] for h in history if h.get("asin") in item2id]
        history_stars = [float(h.get("stars", 3)) / 5.0 for h in history if h.get("asin") in item2id]

        if not history_ids:
            continue

        # Budget = mean price * slate_size so full slate is affordable.
        # Without this, budget=1.0 with uniform costs=1.0 limits slate to 1 item.
        prices = [price_map.get(i, 0.0) for i in history_ids]
        valid  = [p for p in prices if p > 0]
        budget = float(np.mean(valid)) * slate_size if valid else float(slate_size)

        steps.append(TrajectoryStep(
            user_id       = rec.get("user_id", ""),
            item_id       = target_id,
            history_ids   = history_ids,
            history_extras= history_stars,
            budget        = budget,
            split         = "train",   # caller sets correct split label
            reward        = float(rec.get("r_hit", 0.5)),
        ))
    return steps


# ---------------------------------------------------------------------------
# Encode trajectories to state tensors
# ---------------------------------------------------------------------------

def encode_trajectories(
    steps: List[TrajectoryStep],
    encoder: StateEncoder,
    history_length: int,
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[torch.Tensor, List[float], List[int]]:
    """
    Encode all trajectory steps into state vectors.

    Returns
    -------
    states    : (N, embed_dim) FloatTensor
    budgets   : List[float] of length N
    targets   : List[int] – item_id y_t for each step
    """
    states_list = []
    budgets = []
    targets = []

    for start in range(0, len(steps), batch_size):
        batch = steps[start: start + batch_size]
        hist_ids = [s.history_ids for s in batch]
        hist_ext = [s.history_extras for s in batch]

        state_batch = encode_history(hist_ids, hist_ext, encoder, history_length, device)
        states_list.append(state_batch)
        budgets.extend([s.budget for s in batch])
        targets.extend([s.item_id for s in batch])

    states = torch.cat(states_list, dim=0)   # (N, embed_dim)
    return states, budgets, targets


# ---------------------------------------------------------------------------
# Proxy reward functions (used when no simulator is available)
# ---------------------------------------------------------------------------

def proxy_reward_amazon(
    slate: List[int],
    target: int,
    stars: Optional[float] = None,
    rank_bonus: float = 0.2,
    miss_penalty: float = -0.01,
) -> float:
    """
    Amazon: shaped reward = stars-weighted hit + rank bonus + miss penalty.

    r_hit  = (stars/5) * (1 + rank_bonus * (k - rank) / k)
    r_miss = miss_penalty   (small negative to push gradient away from 0)

    rank_bonus pushes the model to place the target higher in the slate.
    miss_penalty prevents pure-zero gradients on misses.
    """
    if target not in slate:
        return miss_penalty
    k = len(slate)
    rank = slate.index(target)          # 0-indexed, lower = better position
    weight = (stars / 5.0) if stars is not None else 1.0
    position_bonus = rank_bonus * (k - 1 - rank) / k  # 0 at last, rank_bonus*(k-1)/k at first
    return float(weight * (1.0 + position_bonus))


def proxy_reward_retailrocket(
    slate: List[int],
    target: int,
    event: Optional[str] = None,
) -> float:
    """
    RetailRocket: match next event/item.
    r(view)=0.1  r(addtocart)=0.5  r(transaction)=1.0
    """
    if target not in slate:
        return 0.0
    event_weight = {"view": 0.1, "addtocart": 0.5, "transaction": 1.0}
    return event_weight.get(event or "view", 0.1)
