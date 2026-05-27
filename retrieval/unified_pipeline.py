"""
unified_pipeline.py — RL policy and candidate types for the ICSRec+Submodular pipeline.

Retrieval is handled by ICSRecRetriever (FAISS).  This module owns:
  - ScoredCandidate / UnifiedSearchResult  (typed containers)
  - UnifiedRLPolicy                         (actor-critic, action = α_t)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from models.rl_policy import ALPHA_DIM, KAPPA_DIM
from utils.encoders import StateEncoder, pad_history


# ---------------------------------------------------------------------------
# Candidate containers
# ---------------------------------------------------------------------------

@dataclass
class ScoredCandidate:
    item_id: str          # original string id (from product DB)
    item_idx: int         # integer index used by models
    rel_score: float      # retriever / reranker score
    title: str
    text: str


@dataclass
class UnifiedSearchResult:
    item_id: str
    item_idx: int
    rel_score: float
    submodular_score: float
    title: str
    slate_position: int


# ---------------------------------------------------------------------------
# RL action space
# ---------------------------------------------------------------------------
UNIFIED_ACTION_DIM = ALPHA_DIM + KAPPA_DIM   # 2


class UnifiedRLPolicy(torch.nn.Module):
    """
    Actor-critic policy.  Action: a_t = (α_t, κ_t).
    α_t controls relevance-vs-diversity trade-off in the submodular objective.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256, lr: float = 3e-4,
                 gamma: float = 0.99, ent_coeff: float = 0.05):
        super().__init__()
        from models.rl_policy import Actor, Critic
        import math
        self.actor = Actor(state_dim, hidden_dim, num_layers=2, z_dim=0)
        self.actor.mean_head = torch.nn.Linear(hidden_dim, UNIFIED_ACTION_DIM)
        self.actor.log_std = torch.nn.Parameter(
            torch.full((UNIFIED_ACTION_DIM,), math.log(0.5))
        )
        self.critic = Critic(state_dim, hidden_dim)
        self.target_critic = Critic(state_dim, hidden_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.ent_coeff = ent_coeff

        self.gamma = gamma
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, state: torch.Tensor, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        if deterministic:
            mean, _ = self.actor.forward(state)
            action = mean
        else:
            action, _ = self.actor.sample(state)
        alpha = torch.sigmoid(action[:, 0:1])
        kappa = torch.sigmoid(action[:, 1:2])
        return {"alpha": alpha, "kappa": kappa, "raw": action}

    def soft_update_target(self, tau: float = 0.005) -> None:
        for p, pt in zip(self.critic.parameters(), self.target_critic.parameters()):
            pt.data.copy_(tau * p.data + (1 - tau) * pt.data)

    def update(self, states, actions, rewards, next_states, dones, bc_coeff=0.01):
        import torch.nn.functional as F
        with torch.no_grad():
            v_next = self.target_critic(next_states)
            targets = rewards + self.gamma * v_next * (~dones).float()

        v_pred = self.critic(states)
        critic_loss = F.mse_loss(v_pred, targets)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        new_actions, log_probs = self.actor.sample(states)
        advantages = targets - v_pred.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        pg_loss = -(log_probs * advantages).mean()

        # BC only on κ (kappa, dim 1) — NOT on α (dim 0).
        # α must be driven purely by RL signal; BC on α was pulling it to 0.5 forever.
        bc_loss = F.mse_loss(new_actions[:, 1:2], actions[:, 1:2])

        # Entropy bonus keeps α from collapsing under sparse rewards.
        ent_loss = self.ent_coeff * log_probs.mean()
        actor_loss = pg_loss + bc_coeff * bc_loss + ent_loss

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        with torch.no_grad():
            alpha_mean = torch.sigmoid(new_actions[:, 0:1]).mean().item()
            alpha_std  = torch.sigmoid(new_actions[:, 0:1]).std().item()

        return {
            "rl/critic_loss": critic_loss.item(),
            "rl/actor_loss":  actor_loss.item(),
            "rl/pg_loss":     pg_loss.item(),
            "rl/ent_loss":    ent_loss.item(),
            "rl/bc_loss":     bc_loss.item(),
            "rl/alpha_mean":  alpha_mean,
            "rl/alpha_std":   alpha_std,
        }
