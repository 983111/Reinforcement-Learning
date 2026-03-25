---
language: en
tags:
  - reinforcement-learning
  - ppo
  - gymnasium
  - navigation
  - gridworld
  - pytorch
license: mit
library_name: pytorch
---

# AgentGrid — PPO Navigation Agent

A PPO-trained autonomous navigation agent for a custom procedurally-generated GridWorld environment.

## Model Description

| Property | Value |
|---|---|
| Algorithm | PPO-Clip with GAE |
| Architecture | Shared MLP Actor-Critic (2 × 128, Tanh) |
| Observation | 5×5 egocentric window, shape (25,) |
| Action space | Discrete(4): UP / DOWN / LEFT / RIGHT |
| Training steps | 300,000 |
| Environment | GridWorldEnv (10×10, custom gymnasium env) |
| Framework | PyTorch 2.0+ |

## Performance

| Metric | Value |
|---|---|
| Success rate | ~85% (goal reached) |
| Mean episode reward | ~6.2 |
| Mean episode length | ~55 steps |
| Training time (CPU) | ~8 min |

Evaluated over 100 episodes on held-out maps (different seeds from training).

## How to Use

```python
import torch
from agents.ppo import ActorCritic
from envs.grid_world import GridWorldEnv

# Load model
policy = ActorCritic(obs_dim=25, n_actions=4, hidden_dim=128, n_layers=2)
ckpt = torch.load("ppo_final.pt", map_location="cpu")
policy.load_state_dict(ckpt["policy_state"])
policy.eval()

# Run one episode
env = GridWorldEnv(grid_size=10, seed=42)
obs, _ = env.reset()

done = False
total_reward = 0.0

while not done:
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action, _, _, _ = policy.get_action(obs_t, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action.item())
    total_reward += reward
    done = terminated or truncated

print(f"Episode reward: {total_reward:.2f}")
```

## Environment Details

```
Grid legend:
  0 = empty floor   reward: -0.05 / step
  1 = wall          reward: -0.50 (bump, no movement)
  2 = trap          reward: -1.00
  3 = goal          reward: +10.00 (terminates episode)
  4 = agent         encoded in observation centre cell
```

Maps are procedurally generated each episode with:
- Configurable wall density (default 15%)
- Configurable trap density (default 8%)
- BFS-guaranteed reachability — goal is always reachable

## Training Details

### Hyperparameters

```yaml
lr:            3.0e-4
gamma:         0.99
gae_lambda:    0.95
clip_eps:      0.2
n_epochs:      4
batch_size:    64
n_steps:       512
value_coef:    0.5
entropy_coef:  0.01
max_grad_norm: 0.5
hidden_dim:    128
n_layers:      2
```

### Algorithm

PPO with Generalised Advantage Estimation (GAE):

```
GAE: Â_t = Σ (γλ)^l · δ_{t+l}   where  δ_t = r_t + γV(s_{t+1}) − V(s_t)

PPO: L = E[ min(r·Â, clip(r, 1-ε, 1+ε)·Â) ] − c₁·L^VF + β·H[π]
```

## Limitations

- Goal position is fixed at bottom-right interior corner — the agent may learn a spatial bias rather than general navigation
- No recurrent memory — pure MLP cannot maintain belief state across steps (true POMDP requires LSTM/attention)
- Single environment — no vectorised collection; parallel envs would improve throughput

## Citation

```bibtex
@misc{adkine2025agentgrid,
  author    = {Vishwajeet Adkine},
  title     = {AgentGrid: Autonomous Grid Navigation via Proximal Policy Optimization},
  year      = {2025},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/vishwajeet456/agentgrid-ppo}
}
```

## References

- Schulman et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347
- Schulman et al. (2015). *High-Dimensional Continuous Control Using GAE*. arXiv:1506.02438