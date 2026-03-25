"""
AgentGrid — PPO Actor-Critic Agent
====================================
Pure-PyTorch implementation of Proximal Policy Optimization (PPO)
with Generalised Advantage Estimation (GAE).

Architecture
------------
SharedBackbone  →  [actor_head → π(a|s)]
                →  [critic_head → V(s)]

Both heads share a 2-layer MLP backbone. Sharing features between
actor and critic reduces sample complexity and is standard in
on-policy RL (Schulman et al., 2017).

Key algorithmic components
--------------------------
- Clipped surrogate objective  (PPO-Clip, ε=0.2)
- Generalised Advantage Estimation  (GAE, λ=0.95)
- Entropy bonus  (β=0.01 — encourages exploration)
- Value function loss  (MSE, coefficient c₁=0.5)
- Gradient clipping  (max_norm=0.5)
- Mini-batch updates over collected rollout

References
----------
Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms.
    arXiv:1707.06347
Schulman, J. et al. (2015). High-Dimensional Continuous Control Using
    Generalised Advantage Estimation. arXiv:1506.02438
"""

from __future__ import annotations

from typing import Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    _TORCH_AVAILABLE = False


# ── Network ───────────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Shared-backbone actor-critic network.

    Parameters
    ----------
    obs_dim     : int   — flattened observation size (25 for 5×5 window)
    n_actions   : int   — number of discrete actions (4)
    hidden_dim  : int   — width of hidden layers (default 128)
    n_layers    : int   — number of shared hidden layers (default 2)
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
    ):
        super().__init__()
        self.obs_dim   = obs_dim
        self.n_actions = n_actions

        # ── Shared backbone ───────────────────────────────────────────────────
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # ── Actor head  (outputs logits over actions) ─────────────────────────
        self.actor_head  = nn.Linear(hidden_dim, n_actions)

        # ── Critic head (outputs scalar state value) ──────────────────────────
        self.critic_head = nn.Linear(hidden_dim, 1)

        # ── Orthogonal weight initialisation (standard for PPO) ───────────────
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Smaller gain for the actor output → more uniform initial policy
        nn.init.orthogonal_(self.actor_head.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, obs: torch.Tensor):
        """
        Forward pass.

        Parameters
        ----------
        obs : Tensor  shape (batch, obs_dim)

        Returns
        -------
        logits : Tensor  shape (batch, n_actions)
        values : Tensor  shape (batch,)
        """
        features = self.backbone(obs)
        logits   = self.actor_head(features)
        values   = self.critic_head(features).squeeze(-1)
        return logits, values

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Sample (or greedily select) an action.

        Returns
        -------
        action   : Tensor  shape (batch,)
        log_prob : Tensor  shape (batch,)
        entropy  : Tensor  shape (batch,)
        value    : Tensor  shape (batch,)
        """
        logits, values = self.forward(obs)
        dist           = Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), values

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Re-evaluate stored (obs, action) pairs during the PPO update.
        Used to compute the probability ratio r_t(θ).

        Returns
        -------
        log_prob : Tensor  shape (batch,)
        entropy  : Tensor  shape (batch,)
        value    : Tensor  shape (batch,)
        """
        logits, values = self.forward(obs)
        dist           = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), values


# ── Rollout buffer ─────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Stores a single on-policy rollout for PPO updates.

    Rollout data is collected in numpy arrays for speed,
    then converted to tensors when get_batches() is called.

    Parameters
    ----------
    n_steps   : int    — rollout length (steps per update)
    obs_dim   : int    — observation dimensionality
    n_envs    : int    — number of parallel environments (default 1)
    gae_lambda: float  — λ for GAE (default 0.95)
    gamma     : float  — discount factor (default 0.99)
    device    : str    — 'cpu' or 'cuda'
    """

    def __init__(
        self,
        n_steps: int,
        obs_dim: int,
        n_envs: int = 1,
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.n_steps    = n_steps
        self.obs_dim    = obs_dim
        self.n_envs     = n_envs
        self.gae_lambda = gae_lambda
        self.gamma      = gamma
        self.device     = device
        self.reset()

    def reset(self):
        """Clear all stored data."""
        N, E, D = self.n_steps, self.n_envs, self.obs_dim
        self.obs       = np.zeros((N, E, D), dtype=np.float32)
        self.actions   = np.zeros((N, E),    dtype=np.int64)
        self.rewards   = np.zeros((N, E),    dtype=np.float32)
        self.values    = np.zeros((N, E),    dtype=np.float32)
        self.log_probs = np.zeros((N, E),    dtype=np.float32)
        self.dones     = np.zeros((N, E),    dtype=np.float32)
        self.advantages = np.zeros((N, E),   dtype=np.float32)
        self.returns    = np.zeros((N, E),   dtype=np.float32)
        self._ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
        done: np.ndarray,
    ):
        """Store one timestep of experience."""
        assert self._ptr < self.n_steps, "Buffer is full — call compute_returns() then reset()"
        self.obs[self._ptr]       = obs
        self.actions[self._ptr]   = action
        self.rewards[self._ptr]   = reward
        self.values[self._ptr]    = value
        self.log_probs[self._ptr] = log_prob
        self.dones[self._ptr]     = done
        self._ptr += 1

    def compute_returns_and_advantages(self, last_value: np.ndarray, last_done: np.ndarray):
        """
        Compute GAE advantages and discounted returns.

        GAE(λ) formula:
            δ_t   = r_t + γ·V(s_{t+1})·(1 − done_t) − V(s_t)
            A_t   = δ_t + (γλ)·(1 − done_t)·A_{t+1}
            R_t   = A_t + V(s_t)

        Parameters
        ----------
        last_value : np.ndarray  shape (n_envs,)
            Value estimate for the state AFTER the last stored step.
        last_done  : np.ndarray  shape (n_envs,)
            Whether each env was done after the last step.
        """
        gae = np.zeros(self.n_envs, dtype=np.float32)

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_value        = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value        = self.values[t + 1]

            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int):
        """
        Flatten and shuffle rollout data, yield mini-batches.

        Yields
        ------
        Tuple of tensors: (obs, actions, log_probs, advantages, returns)
        Each with shape (batch_size, ...).
        Advantages are normalised per mini-batch for training stability.
        """
        # Flatten (n_steps, n_envs, ...) → (n_steps * n_envs, ...)
        total   = self.n_steps * self.n_envs
        indices = np.random.permutation(total)

        obs       = self.obs.reshape(total, self.obs_dim)
        actions   = self.actions.reshape(total)
        log_probs = self.log_probs.reshape(total)
        advantages = self.advantages.reshape(total)
        returns   = self.returns.reshape(total)

        for start in range(0, total, batch_size):
            idx = indices[start : start + batch_size]
            adv = advantages[idx]
            # Normalise advantages within the mini-batch
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            yield (
                torch.tensor(obs[idx],       dtype=torch.float32, device=self.device),
                torch.tensor(actions[idx],   dtype=torch.int64,   device=self.device),
                torch.tensor(log_probs[idx], dtype=torch.float32, device=self.device),
                torch.tensor(adv,            dtype=torch.float32, device=self.device),
                torch.tensor(returns[idx],   dtype=torch.float32, device=self.device),
            )


# ── PPO Trainer ───────────────────────────────────────────────────────────────

class PPOTrainer:
    """
    Proximal Policy Optimization trainer.

    Parameters
    ----------
    policy        : ActorCritic
    lr            : float  — Adam learning rate          (default 3e-4)
    clip_eps      : float  — PPO clipping parameter ε    (default 0.2)
    value_coef    : float  — value loss coefficient c₁   (default 0.5)
    entropy_coef  : float  — entropy bonus coefficient β (default 0.01)
    max_grad_norm : float  — gradient clip norm          (default 0.5)
    n_epochs      : int    — PPO update epochs per rollout (default 4)
    batch_size    : int    — mini-batch size             (default 64)
    device        : str
    """

    def __init__(
        self,
        policy: ActorCritic,
        lr: float            = 3e-4,
        clip_eps: float      = 0.2,
        value_coef: float    = 0.5,
        entropy_coef: float  = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int        = 4,
        batch_size: int      = 64,
        device: str          = "cpu",
    ):
        self.policy        = policy
        self.clip_eps      = clip_eps
        self.value_coef    = value_coef
        self.entropy_coef  = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.device        = device

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

        # Metrics accumulated per update call
        self._metrics: dict[str, list[float]] = {
            "policy_loss": [], "value_loss": [], "entropy": [],
            "approx_kl": [], "clip_fraction": [],
        }

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """
        Run n_epochs of mini-batch PPO updates on the collected rollout.

        Returns
        -------
        dict of mean losses/metrics for logging.
        """
        for key in self._metrics:
            self._metrics[key].clear()

        for _ in range(self.n_epochs):
            for obs, actions, old_log_probs, advantages, returns in buffer.get_batches(self.batch_size):
                # ── Re-evaluate actions under current policy ──────────────────
                new_log_probs, entropy, values = self.policy.evaluate_actions(obs, actions)

                # ── Probability ratio r_t(θ) = π_θ(a|s) / π_θ_old(a|s) ───────
                ratio = torch.exp(new_log_probs - old_log_probs)

                # ── PPO-Clip surrogate loss ────────────────────────────────────
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # ── Value function loss (MSE) ──────────────────────────────────
                value_loss = F.mse_loss(values, returns)

                # ── Entropy bonus (encourages exploration) ────────────────────
                entropy_loss = -entropy.mean()

                # ── Combined loss ─────────────────────────────────────────────
                loss = (
                    policy_loss
                    + self.value_coef   * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # ── Diagnostics ───────────────────────────────────────────────
                with torch.no_grad():
                    approx_kl   = ((ratio - 1) - (new_log_probs - old_log_probs)).mean().item()
                    clip_frac   = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()

                self._metrics["policy_loss"].append(policy_loss.item())
                self._metrics["value_loss"].append(value_loss.item())
                self._metrics["entropy"].append(-entropy_loss.item())
                self._metrics["approx_kl"].append(approx_kl)
                self._metrics["clip_fraction"].append(clip_frac)

        return {k: float(np.mean(v)) for k, v in self._metrics.items()}

    def save(self, path: str):
        """Save policy weights and optimiser state."""
        torch.save({
            "policy_state":    self.policy.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved → {path}")

    def load(self, path: str):
        """Load policy weights and optimiser state."""
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"Checkpoint loaded ← {path}")