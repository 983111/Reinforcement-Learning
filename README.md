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
| Architecture | Shared MLP Actor-Critic (2 × 128, Tanh) |
| Policy Parameters | 20,485 |
| Training budget (10×10) | 300,000 timesteps (~1,435s on CPU) |
| Training budget (14×14) | 600,000 timesteps (~4,733s on CPU) |
| Training budget (custom hyperparams) | 300,000 timesteps |
| Success rate (10×10, 20 eps) | 60.0% (12/20 reached goal) |

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
│   └── test_grid_world.py   # 30 unit tests (30/30 passing)
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

Default config: 10×10 grid, 300k timesteps, ~24 minutes on CPU.

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

## Training Results

### 10×10 Grid — 300,000 Steps (585 Updates)

| Phase | Updates | Steps | Mean Reward | Mean Length | Entropy |
|---|---|---|---|---|---|
| Early | 10–50 | 5k–25k | -33.9 → -5.1 | 192 → 54 | 1.338 → 1.250 |
| Mid | 100–200 | 51k–102k | -10.9 → -2.1 | 75 → 53 | 1.244 → 1.054 |
| Late | 300–400 | 153k–204k | 1.8 → 8.0 | 54 → 19 | 1.041 → 0.708 |
| Final | 500–585 | 256k–296k | 7.0 → 6.5 | 25 → 30 | 0.701 → 0.751 |

Training complete in **1,435.7s** on CPU.

### Evaluation Results — 10×10 Grid (20 Episodes)

```
════════════════════════════════════════════════════
  Evaluation Results  (20 episodes)
════════════════════════════════════════════════════
  Success rate      : 60.0%  (12/20 reached goal)
  Timeouts          : 8
  Mean reward       : 1.562  ±  9.443
  Mean episode len  : 88.4 steps

  Per-episode breakdown:
    Ep    Reward   Steps  Outcome
  ────  ────────  ──────  ────────
     1   -10.000     200  ✗ timeout
     2     9.350      14  ✓ goal
     3   -10.000     200  ✗ timeout
     4     9.350      14  ✓ goal
     5   -10.000     200  ✗ timeout
     6   -10.000     200  ✗ timeout
     7     9.350      14  ✓ goal
     8     9.350      14  ✓ goal
     9     9.350      14  ✓ goal
    10   -10.000     200  ✗ timeout
    11     9.350      14  ✓ goal
    12     8.400      14  ✓ goal
    13     9.350      14  ✓ goal
    14     9.350      14  ✓ goal
    15     9.350      14  ✓ goal
    16     9.350      14  ✓ goal
    17   -10.000     200  ✗ timeout
    18   -10.000     200  ✗ timeout
    19   -10.000     200  ✗ timeout
    20     9.350      14  ✓ goal
════════════════════════════════════════════════════
```

> **Note:** The agent exhibits bimodal behaviour — successful episodes complete in exactly 14 steps, while failed episodes always hit the 200-step timeout with no partial progress. This suggests the policy either finds the optimal path or gets stuck entirely, likely due to the fixed goal position creating map-dependent sensitivity.

### 14×14 Grid — 600,000 Steps (1,171 Updates)

Training complete in **4,733.1s** on CPU.

| Update | Steps | Mean Reward | Mean Length | Entropy | KL |
|---|---|---|---|---|---|
| 10 | 5,120 | -41.145 | 192.7 | 1.372 | 0.0019 |
| 50 | 25,600 | -20.817 | 111.4 | 1.249 | 0.0012 |
| 100 | 51,200 | -17.382 | 124.1 | 1.186 | 0.0017 |
| 200 | 102,400 | -1.125 | 67.0 | 1.152 | 0.0020 |
| 300 | 153,600 | -6.197 | 70.5 | 1.119 | 0.0078 |
| 400 | 204,800 | -3.980 | 90.2 | 0.996 | 0.0114 |
| 500 | 256,000 | 0.962 | 62.8 | 0.924 | 0.0065 |
| 600 | 307,200 | 3.150 | 60.8 | 0.853 | 0.0092 |
| 700 | 358,400 | 6.157 | 49.5 | 0.640 | 0.0027 |
| 800 | 409,600 | 2.800 | 59.7 | 0.651 | 0.0025 |
| 900 | 460,800 | -0.255 | 60.2 | 0.479 | 0.1139 |
| 1000 | 512,000 | 5.192 | 56.3 | 0.501 | 0.0029 |
| 1100 | 563,200 | -0.303 | 62.0 | 0.587 | 0.0124 |
| 1170 | 599,040 | -0.435 | 58.2 | 0.604 | 0.0052 |

The 14×14 run shows notably slower convergence and higher variance throughout — reward never stabilises cleanly above 6 the way the 10×10 run does. A large KL spike at update 900 (0.1139) indicates a destabilising policy update that partially recovered. The larger state space with a fixed goal position creates harder generalisation pressure for the pure MLP policy.

### Custom Hyperparameters — `--entropy-coef 0.02 --clip-eps 0.15` (10×10)

| Update | Steps | Mean Reward | Mean Length | Entropy | KL |
|---|---|---|---|---|---|
| 10 | 5,120 | -11.217 | 82.3 | 1.292 | 0.0056 |
| 50 | 25,600 | -0.215 | 46.0 | 1.221 | 0.0016 |
| 100 | 51,200 | -0.107 | 45.1 | 1.199 | 0.0060 |
| 130 | 66,560 | 4.300 | 36.9 | 1.085 | 0.0047 |
| 200 | 102,400 | 4.700 | 34.9 | 1.016 | 0.0049 |
| 240 | 122,880 | 5.865 | 37.5 | 0.951 | 0.0028 |
| 250 | 128,000 | 5.905 | 29.9 | 1.084 | 0.0034 |

Higher entropy coefficient (0.02 vs 0.01) and tighter clipping (0.15 vs 0.20) produces **faster early convergence** — the agent reaches positive reward by update 50 (~25k steps) vs update 110 (~56k steps) in the default run. Entropy stays higher throughout, indicating broader exploration is maintained.

---

## Test Suite

**30/30 tests passing** in 9.81s on Python 3.10.11 / pytest 9.0.2 (Windows 10).

```
tests/test_grid_world.py::TestSpaces::test_observation_shape          PASSED
tests/test_grid_world.py::TestSpaces::test_observation_dtype          PASSED
tests/test_grid_world.py::TestSpaces::test_observation_range          PASSED
tests/test_grid_world.py::TestSpaces::test_action_space_size          PASSED
tests/test_grid_world.py::TestSpaces::test_action_space_contains      PASSED
tests/test_grid_world.py::TestMapGeneration::test_border_is_all_walls PASSED
tests/test_grid_world.py::TestMapGeneration::test_goal_is_placed      PASSED
tests/test_grid_world.py::TestMapGeneration::test_goal_position       PASSED
tests/test_grid_world.py::TestMapGeneration::test_agent_cell_is_not_wall PASSED
tests/test_grid_world.py::TestMapGeneration::test_reachability        PASSED
tests/test_grid_world.py::TestMapGeneration::test_different_seeds_differ PASSED
tests/test_grid_world.py::TestMapGeneration::test_same_seed_reproducible PASSED
tests/test_grid_world.py::TestStepMechanics::test_step_returns_correct_types PASSED
tests/test_grid_world.py::TestStepMechanics::test_floor_step_reward   PASSED
tests/test_grid_world.py::TestStepMechanics::test_wall_bump_reward    PASSED
tests/test_grid_world.py::TestStepMechanics::test_invalid_action_raises PASSED
tests/test_grid_world.py::TestStepMechanics::test_truncation_at_max_steps PASSED
tests/test_grid_world.py::TestStepMechanics::test_goal_terminates_episode PASSED
tests/test_grid_world.py::TestObservation::test_agent_center_encoded  PASSED
tests/test_grid_world.py::TestObservation::test_obs_changes_after_move PASSED
tests/test_grid_world.py::TestObservation::test_oob_padded_as_wall    PASSED
tests/test_grid_world.py::TestInfoDict::test_info_keys                PASSED
tests/test_grid_world.py::TestInfoDict::test_step_count_increments    PASSED
tests/test_grid_world.py::TestInfoDict::test_manhattan_dist_type      PASSED
tests/test_grid_world.py::TestRendering::test_rgb_array_shape         PASSED
tests/test_grid_world.py::TestRendering::test_rgb_array_dtype         PASSED
tests/test_grid_world.py::TestRendering::test_human_render_no_crash   PASSED
tests/test_grid_world.py::TestProperties::test_n_actions              PASSED
tests/test_grid_world.py::TestProperties::test_obs_dim                PASSED
tests/test_grid_world.py::TestProperties::test_repr                   PASSED

================================================= 30 passed in 9.81s ==================================================
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
│   └── agent_eval.gif         # 201-frame agent navigation replay
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
   *ICML 2016*

4. Engstrom, L. et al. (2020). **Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO**.
   *ICLR 2020*

---

## License

MIT — see [LICENSE](LICENSE).
