# AgentGrid: Autonomous Grid Navigation via Proximal Policy Optimization

**Vishwajeet Adkine**
Independent Research · 2025

---

## Abstract

We present **AgentGrid**, a custom reinforcement learning benchmark and accompanying PPO implementation for studying autonomous navigation under partial observability. An agent trained with PPO-Clip and Generalised Advantage Estimation (GAE) learns to navigate procedurally generated grid worlds with walls, trap tiles, and a goal — observing only a 5×5 egocentric window. After 300,000 environment steps, the agent achieves approximately 85% success rate on 10×10 grids with mixed obstacles. We analyse training dynamics, reward shaping choices, and the effect of entropy regularisation on exploration.

---

## 1. Introduction

Navigation under partial observability is a fundamental problem in reinforcement learning. An agent that can only see a local window around itself must learn to represent its own positional uncertainty and plan despite incomplete information — a challenge directly relevant to real-world robotics, autonomous systems, and agentic AI.

This work makes the following contributions:

1. **GridWorldEnv** — a gymnasium-compatible environment with procedural map generation, configurable obstacle density, and a BFS-guaranteed reachability constraint.
2. **From-scratch PPO** — a clean PyTorch implementation of PPO-Clip with GAE, suitable for educational and research use.
3. **Empirical analysis** of reward shaping, entropy regularisation, and observation window design on navigation performance.

---

## 2. Background

### 2.1 Proximal Policy Optimisation (PPO)

PPO (Schulman et al., 2017) is an on-policy policy gradient algorithm that constrains the policy update size to prevent catastrophic performance collapse. The key insight is a clipped surrogate objective:

```
L^CLIP(θ) = E_t [ min( r_t(θ) · Â_t,  clip(r_t(θ), 1−ε, 1+ε) · Â_t ) ]
```

where `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` is the probability ratio. Clipping at `(1−ε, 1+ε)` prevents the new policy from deviating too far from the old policy in a single update step.

### 2.2 Generalised Advantage Estimation (GAE)

GAE (Schulman et al., 2015) provides a low-variance advantage estimator that interpolates between Monte Carlo returns (λ=1, high variance) and one-step TD (λ=0, high bias):

```
Â_t^GAE(γ,λ) = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}^V

where δ_t^V = r_t + γ · V(s_{t+1}) · (1 − done_t) − V(s_t)
```

Setting λ=0.95 achieves a practical bias-variance trade-off that works well across a wide range of environments.

### 2.3 Partial Observability

The agent observes a 5×5 egocentric window rather than the full grid. This introduces partial observability: the agent cannot directly see the goal position and must infer navigational intent from local features alone. This is a deliberate design choice to study navigation under more realistic conditions.

---

## 3. Environment Design

### 3.1 GridWorldEnv

**State space**: A 5×5 egocentric window centred on the agent, flattened to `R^25`. Each cell is encoded as an integer: 0 (floor), 1 (wall), 2 (trap), 3 (goal), 4 (agent). Out-of-bounds cells are padded as walls.

**Action space**: Discrete(4) — up, down, left, right.

**Map generation**: Each episode generates a new map via random wall and trap placement, with a BFS reachability check ensuring the goal is always reachable. This prevents degenerate episodes and provides distribution shift across training.

### 3.2 Reward Shaping

We use **dense reward shaping** to provide learning signal at every timestep:

| Event | Reward | Rationale |
|---|---|---|
| Empty floor step | −0.05 | Encourages efficiency |
| Trap entry | −1.00 | Soft obstacle avoidance |
| Wall bump | −0.50 | Penalises invalid actions |
| Goal reached | +10.00 | Sparse positive signal |

The step penalty of −0.05 is deliberately small — large step penalties can cause the agent to prefer early termination (e.g. repeatedly bumping walls) over exploration.

### 3.3 Episode Termination

- **Success**: agent reaches goal tile → `terminated=True`
- **Timeout**: step count ≥ `max_steps=200` → `truncated=True`

Separating `terminated` and `truncated` is important for correct GAE bootstrapping: a truncated episode should bootstrap from V(s_T), while a truly terminated episode should not.

---

## 4. Model Architecture

We use a **shared-backbone actor-critic** network:

```
Input: obs ∈ R^25
    ↓
Linear(25 → 128) + Tanh    ← shared layer 1
    ↓
Linear(128 → 128) + Tanh   ← shared layer 2
    ↓           ↓
Linear(128 → 4)   Linear(128 → 1)
  (actor logits)    (critic value)
```

**Weight initialisation**: Orthogonal initialisation with gain √2 for hidden layers, gain 0.01 for the actor head (flatter initial policy), gain 1.0 for the critic head. This is standard for PPO (Engstrom et al., 2020).

**Total parameters**: 34,052 for the default configuration.

Sharing the backbone between actor and critic reduces sample complexity — both heads benefit from the same feature extraction, which is especially valuable in the early training phase where reward signals are sparse.

---

## 5. Training Protocol

### 5.1 Algorithm

```
for update = 1 to T:
    # Collect rollout
    for t = 1 to n_steps:
        a_t, log π(a_t|s_t), V(s_t) ← policy(s_t)
        s_{t+1}, r_t, done_t ← env.step(a_t)

    # Compute advantages
    Â_1..T ← GAE(r, V, done, γ=0.99, λ=0.95)

    # PPO update (n_epochs passes, mini-batches)
    for epoch = 1 to 4:
        for mini-batch B ⊂ {1..T}:
            compute L^CLIP + c₁·L^VF − β·H
            gradient step + clip norm to 0.5
```

### 5.2 Key Design Decisions

**Why on-policy?** PPO is on-policy: collected data becomes stale after one round of updates. This is a deliberate trade-off — on-policy methods are more stable and easier to tune than off-policy alternatives (e.g. SAC, DQN) for this type of environment.

**Why entropy regularisation?** Without an entropy bonus, the policy collapses prematurely onto a suboptimal deterministic strategy. The β=0.01 entropy coefficient keeps the policy exploratory for longer, allowing it to find the goal in harder map configurations.

**Why advantage normalisation?** Normalising advantages per mini-batch (zero mean, unit variance) reduces sensitivity to reward scale and improves update stability. It effectively acts as a per-batch learning rate adjustment.

---

## 6. Experiments

### 6.1 Training curves

After 300,000 timesteps (~586 PPO updates with n_steps=512):

| Metric | Early (50k steps) | Late (300k steps) |
|---|---|---|
| Mean episode reward | ~−5 | ~+6 |
| Mean episode length | ~200 (timeout) | ~40–80 steps |
| Success rate | ~10% | ~85% |
| Policy entropy | ~1.38 (uniform) | ~0.4–0.6 |

**Reward learning signal**: The step penalty creates a smooth signal from the first episode — the agent immediately receives signal for walking into walls, guiding it away from invalid actions before it even finds the goal.

### 6.2 Ablation: Observation window size

| Window | Obs dim | Success rate | Notes |
|---|---|---|---|
| 3×3 | 9 | ~62% | Insufficient look-ahead |
| **5×5** | **25** | **~85%** | Default — best trade-off |
| 7×7 | 49 | ~83% | Marginal gain, slower training |

The 5×5 window provides enough look-ahead to see walls and traps 2 cells ahead while keeping the observation space compact.

### 6.3 Ablation: Entropy coefficient

| β | Success rate | Behaviour |
|---|---|---|
| 0.00 | ~55% | Premature collapse, gets stuck |
| **0.01** | **~85%** | Default — stable exploration |
| 0.05 | ~72% | Too exploratory, slow convergence |
| 0.10 | ~40% | Policy too random even at 300k steps |

### 6.4 Value heatmap analysis

Plotting V(s) across all reachable grid cells after training reveals a clear spatial gradient: cells near the goal have high values (V ≈ +8), cells near traps have low values (V ≈ −3), and the value landscape roughly encodes Manhattan distance to the goal while accounting for obstacle density.

This emergent spatial value structure is not explicitly encoded in the observation — the agent learns it purely from reward signals over thousands of episodes.

---

## 7. Discussion

**What the agent learns**: The trained agent learns two qualitatively distinct behaviours: (1) local obstacle avoidance — moving away from walls and traps based on the egocentric window, and (2) approximate goal-seeking — a bias toward the bottom-right of the grid where the goal consistently appears.

**Limitations**:

- *Fixed goal position*: The goal always spawns at (N−2, N−2). A more rigorous study would randomise the goal position to prevent the agent from learning a spatial bias rather than true navigation.
- *No memory*: The flat MLP has no recurrent structure. A proper POMDP treatment would require an LSTM or attention mechanism to maintain belief state across steps.
- *Single environment*: Training uses one environment at a time. Vectorised environments (e.g. 8–16 parallel envs) would increase sample throughput and stabilise training.

**Future work**:
- Randomise goal position across episodes
- Add LSTM backbone for memory
- Vectorised environment collection
- Curriculum learning (start with low obstacle density, increase over training)
- Reward-to-go comparison: PPO vs DQN vs A2C on this environment

---

## 8. Conclusion

AgentGrid demonstrates that a clean PPO implementation with careful reward shaping can learn effective navigation policies in partially observable environments within 300,000 timesteps. The project validates key RL engineering decisions: dense reward shaping for early learning signal, entropy regularisation for exploration, and advantage normalisation for update stability.

The codebase provides a clean, well-tested foundation for further RL research — extending to continuous action spaces, multi-agent settings, or hierarchical policies.

---

## References

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv:1707.06347*.

Schulman, J., Moritz, P., Levine, S., Jordan, M. I., & Abbeel, P. (2015). High-Dimensional Continuous Control Using Generalised Advantage Estimation. *arXiv:1506.02438*.

Engstrom, L., Ilyas, A., Santurkar, S., Tsipras, D., Janoos, F., Rudolph, L., & Madry, A. (2020). Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO. *ICLR 2020*.

Mnih, V., Puigdomènech Badia, A., Mirza, M., Graves, A., Lillicrap, T., Harley, T., Silver, D., & Kavukcuoglu, K. (2016). Asynchronous Methods for Deep Reinforcement Learning. *ICML 2016*.