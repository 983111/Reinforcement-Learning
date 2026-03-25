# AgentGrid — PPO Autonomous Navigation Agent

> A from-scratch implementation of **Proximal Policy Optimization (PPO)** on a custom procedurally-generated GridWorld environment, built for RL research and portfolio demonstration.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

AgentGrid trains an autonomous agent to navigate a procedurally generated grid world with walls, trap tiles, and a goal. The agent learns entirely from reward signals using **PPO with GAE** — no hand-crafted heuristics, no imitation learning.

| Metric | Value |
|---|---|
| Environment | Custom `gymnasium`-compatible `GridWorldEnv` |
| Algorithm | PPO-Clip with GAE (Schulman et al., 2017) |
| Architecture | Shared MLP Actor-Critic (2 × 128) |
| Training budget | 300,000 timesteps |
| Typical success rate (10×10 grid) | ~85% after training |

---

## Project Structure

```
agentgrid/
├── envs/
│   └── grid_world.py        # Custom Gym environment
├── agents/
│   └── ppo.py               # ActorCritic, RolloutBuffer, PPOTrainer
├── utils/
│   └── visualise.py         # Plotting helpers
├── tests/
│   └── test_grid_world.py   # 24 unit tests
├── train.py                 # Training loop
├── eval.py                  # Evaluation + GIF generation
├── gymnasium_shim.py        # Offline Gym fallback
├── requirements.txt
└── outputs/
    ├── checkpoints/         # Saved model weights
    ├── plots/               # Training curves, heatmaps
    └── gifs/                # Agent replay GIFs
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the agent

```bash
python train.py
```

Default config: 10×10 grid, 300k timesteps, ~5–10 minutes on CPU.

```bash
# Larger grid, longer training
python train.py --grid-size 14 --total-timesteps 600000

# Custom reward shaping
python train.py --entropy-coef 0.02 --clip-eps 0.15
```

### 3. Evaluate + generate GIF

```bash
python eval.py --n-episodes 20
```

### 4. Run tests

```bash
python -m pytest tests/ -v
```

---

## Environment: `GridWorldEnv`

### Grid layout

```
█ █ █ █ █ █ █ █ █ █
█ A · · · · · · · █
█ · · ✗ · █ · · · █
█ · █ · █ █ ✗ █ · █
█ ✗ · · · · · · · █
█ █ · █ · · · · · █
█ · · · · · · · · █
█ · · · ✗ · · · · █
█ · █ · · · █ ✗ ★ █
█ █ █ █ █ █ █ █ █ █
```

| Symbol | Meaning |
|---|---|
| `A` | Agent (current position) |
| `★` | Goal (+10 reward, episode ends) |
| `✗` | Trap (−1 reward, continues) |
| `█` | Wall (impassable, −0.5 bump penalty) |
| `·` | Empty floor (−0.05 step penalty) |

### Observation space

A **5×5 egocentric window** centred on the agent, flattened to shape `(25,)` float32. Out-of-bounds cells are padded as walls. The agent's own cell is encoded as `4`.

```
[ wall | wall | wall | wall | wall ]
[ wall |  .   |  .   |  .   | wall ]
[ wall |  .   |  A   |  .   |  .  ]   ← agent at centre (index 12)
[ wall |  .   |  .   |  .   |  .  ]
[ wall |  .   |  ✗   |  .   |  .  ]
```

### Action space

`Discrete(4)` → `{0: UP, 1: DOWN, 2: LEFT, 3: RIGHT}`

### Reward structure

| Event | Reward |
|---|---|
| Step on empty floor | −0.05 |
| Step on trap | −1.00 |
| Bump into wall | −0.50 |
| Reach goal | +10.00 |

### Map generation

Every episode generates a new map via:
1. Border walls
2. Random interior walls (density configurable)
3. Random traps
4. BFS reachability check — retries up to 20 times until goal is reachable

---

## Algorithm: PPO with GAE

### Actor-Critic architecture

```
obs (25,)
    ↓
[Linear(25→128) → Tanh]   ← shared backbone
[Linear(128→128) → Tanh]
    ↓              ↓
actor_head      critic_head
(128→4 logits)  (128→1 value)
    ↓
Categorical(logits) → action, log_prob, entropy
```

### PPO update

For each collected rollout, run `n_epochs=4` passes of mini-batch updates:

**1. GAE Advantage Estimation**
```
δ_t   = r_t + γ · V(s_{t+1}) · (1 − done_t) − V(s_t)
A_t   = δ_t + (γλ) · (1 − done_t) · A_{t+1}
R_t   = A_t + V(s_t)
```

**2. PPO-Clip Surrogate Loss**
```
r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)

L^CLIP = E[ min(r_t · A_t,  clip(r_t, 1−ε, 1+ε) · A_t) ]
```

**3. Value Function Loss**
```
L^VF = E[ (V_θ(s_t) − R_t)² ]
```

**4. Total Loss**
```
L = −L^CLIP + c₁ · L^VF − β · H[π_θ]
```

Where `H[π_θ]` is the policy entropy bonus (encourages exploration).

### Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `lr` | 3e-4 | Adam learning rate |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE smoothing |
| `clip_eps` | 0.2 | PPO clipping ε |
| `n_epochs` | 4 | Update epochs per rollout |
| `batch_size` | 64 | Mini-batch size |
| `n_steps` | 512 | Steps per rollout |
| `value_coef` | 0.5 | c₁ value loss weight |
| `entropy_coef` | 0.01 | β entropy bonus |
| `max_grad_norm` | 0.5 | Gradient clip norm |
| `hidden_dim` | 128 | MLP hidden width |

---

## Outputs

After training:

```
outputs/
├── checkpoints/
│   ├── ppo_update_00050.pt
│   ├── ppo_update_00100.pt
│   └── ppo_final.pt
├── plots/
│   ├── training_curves.png    # 6-panel metric dashboard
│   ├── grid_snapshot.png      # env render
│   └── value_heatmap.png      # critic V(s) across grid
├── gifs/
│   └── agent_eval.gif         # agent navigating the grid
└── training_log.csv           # full per-update log
```

---

## References

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
   **Proximal Policy Optimization Algorithms**. *arXiv:1707.06347*

2. Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015).
   **High-Dimensional Continuous Control Using Generalised Advantage Estimation**.
   *arXiv:1506.02438*

3. Mnih, V. et al. (2016). **Asynchronous Methods for Deep Reinforcement Learning**.
   *ICML 2016* (foundational actor-critic reference)

---

## License

MIT — see [LICENSE](LICENSE).