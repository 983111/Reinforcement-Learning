"""
AgentGrid — train.py
=====================
Main training script. Runs PPO on GridWorldEnv and logs metrics.

Usage
-----
    python train.py                          # default config
    python train.py --grid-size 12 --steps 500000
    python train.py --no-checkpoint          # skip saving

Logged metrics (printed + saved to outputs/training_log.csv)
--------------------------------------------------------------
  update      — PPO update number
  timesteps   — total env steps so far
  mean_reward — mean episode reward over last 20 episodes
  mean_len    — mean episode length over last 20 episodes
  policy_loss — PPO clipped surrogate loss
  value_loss  — MSE value function loss
  entropy     — mean policy entropy
  approx_kl   — approximate KL divergence
  clip_frac   — fraction of clipped probability ratios
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import torch

from agents.ppo import ActorCritic, PPOTrainer, RolloutBuffer
from envs.grid_world import GridWorldEnv


# ── Default hyperparameters ───────────────────────────────────────────────────

DEFAULTS = dict(
    # Environment
    grid_size     = 10,
    wall_density  = 0.15,
    trap_density  = 0.08,
    max_ep_steps  = 200,
    seed          = 0,
    # PPO rollout
    n_steps       = 512,     # steps collected per update
    # PPO hyperparams
    lr            = 3e-4,
    clip_eps      = 0.2,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    n_epochs      = 4,
    batch_size    = 64,
    value_coef    = 0.5,
    entropy_coef  = 0.01,
    max_grad_norm = 0.5,
    # Training budget
    total_timesteps = 300_000,
    # Logging & checkpointing
    log_interval  = 10,      # print every N updates
    save_interval = 50,      # checkpoint every N updates
    hidden_dim    = 128,
    n_layers      = 2,
)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg: dict) -> dict:
    """
    Run the full PPO training loop.

    Parameters
    ----------
    cfg : dict  — hyperparameter config (merged from DEFAULTS + CLI args)

    Returns
    -------
    history : dict  — lists of logged scalars per update, suitable for plotting
    """
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Environment ───────────────────────────────────────────────────────────
    env = GridWorldEnv(
        grid_size    = cfg["grid_size"],
        wall_density = cfg["wall_density"],
        trap_density = cfg["trap_density"],
        max_steps    = cfg["max_ep_steps"],
        seed         = cfg["seed"],
    )
    obs, _ = env.reset(seed=cfg["seed"])
    obs_dim   = env.obs_dim
    n_actions = env.n_actions

    # ── Policy ────────────────────────────────────────────────────────────────
    policy = ActorCritic(
        obs_dim    = obs_dim,
        n_actions  = n_actions,
        hidden_dim = cfg["hidden_dim"],
        n_layers   = cfg["n_layers"],
    ).to(device)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy parameters: {n_params:,}")

    # ── PPO trainer + rollout buffer ──────────────────────────────────────────
    trainer = PPOTrainer(
        policy        = policy,
        lr            = cfg["lr"],
        clip_eps      = cfg["clip_eps"],
        value_coef    = cfg["value_coef"],
        entropy_coef  = cfg["entropy_coef"],
        max_grad_norm = cfg["max_grad_norm"],
        n_epochs      = cfg["n_epochs"],
        batch_size    = cfg["batch_size"],
        device        = device,
    )

    buffer = RolloutBuffer(
        n_steps    = cfg["n_steps"],
        obs_dim    = obs_dim,
        n_envs     = 1,
        gae_lambda = cfg["gae_lambda"],
        gamma      = cfg["gamma"],
        device     = device,
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/plots",       exist_ok=True)

    ep_rewards  = deque(maxlen=20)
    ep_lengths  = deque(maxlen=20)
    ep_reward   = 0.0
    ep_length   = 0

    history: dict[str, list] = {
        "update": [], "timesteps": [], "mean_reward": [], "mean_len": [],
        "policy_loss": [], "value_loss": [], "entropy": [],
        "approx_kl": [], "clip_fraction": [],
    }

    csv_path = "outputs/training_log.csv"
    csv_file = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=list(history.keys()))
    writer.writeheader()

    total_updates = cfg["total_timesteps"] // cfg["n_steps"]
    total_steps   = 0
    start_time    = time.time()

    print(f"\n{'─'*60}")
    print(f"  AgentGrid PPO Training")
    print(f"  Grid: {cfg['grid_size']}×{cfg['grid_size']}  |  "
          f"Budget: {cfg['total_timesteps']:,} steps  |  "
          f"Updates: {total_updates:,}")
    print(f"{'─'*60}\n")

    # ── Collect → update loop ─────────────────────────────────────────────────
    for update in range(1, total_updates + 1):

        # ── Rollout collection ────────────────────────────────────────────────
        policy.eval()
        buffer.reset()

        for _ in range(cfg["n_steps"]):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, _, value = policy.get_action(obs_tensor)

            a         = action.item()
            lp        = log_prob.item()
            v         = value.item()

            next_obs, reward, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            buffer.add(
                obs    = obs.reshape(1, -1),
                action = np.array([a]),
                reward = np.array([reward], dtype=np.float32),
                value  = np.array([v],      dtype=np.float32),
                log_prob = np.array([lp],   dtype=np.float32),
                done   = np.array([float(done)], dtype=np.float32),
            )

            ep_reward += reward
            ep_length += 1
            total_steps += 1

            if done:
                ep_rewards.append(ep_reward)
                ep_lengths.append(ep_length)
                ep_reward = 0.0
                ep_length = 0
                obs, _ = env.reset()
            else:
                obs = next_obs

        # ── Bootstrap last value ──────────────────────────────────────────────
        with torch.no_grad():
            obs_tensor  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            _, last_val = policy.forward(obs_tensor)
            last_val    = last_val.cpu().numpy()
        done_flag = np.array([float(done)])
        buffer.compute_returns_and_advantages(last_val, done_flag)

        # ── PPO update ────────────────────────────────────────────────────────
        policy.train()
        metrics = trainer.update(buffer)

        # ── Logging ───────────────────────────────────────────────────────────
        mean_rew = float(np.mean(ep_rewards)) if ep_rewards else float("nan")
        mean_len = float(np.mean(ep_lengths)) if ep_lengths else float("nan")

        row = {
            "update":       update,
            "timesteps":    total_steps,
            "mean_reward":  round(mean_rew, 4),
            "mean_len":     round(mean_len, 2),
            "policy_loss":  round(metrics["policy_loss"], 6),
            "value_loss":   round(metrics["value_loss"],  6),
            "entropy":      round(metrics["entropy"],     6),
            "approx_kl":    round(metrics["approx_kl"],  6),
            "clip_fraction":round(metrics["clip_fraction"], 4),
        }
        writer.writerow(row)
        csv_file.flush()

        for k, v in row.items():
            history[k].append(v)

        if update % cfg["log_interval"] == 0:
            elapsed = time.time() - start_time
            fps     = total_steps / elapsed
            print(
                f"  Update {update:>5}/{total_updates}  |  "
                f"Steps {total_steps:>8,}  |  "
                f"Rew {mean_rew:>7.3f}  |  "
                f"Len {mean_len:>6.1f}  |  "
                f"Entropy {metrics['entropy']:.3f}  |  "
                f"KL {metrics['approx_kl']:.4f}  |  "
                f"{fps:.0f} fps"
            )

        if update % cfg["save_interval"] == 0 and cfg.get("checkpoint", True):
            ckpt_path = f"outputs/checkpoints/ppo_update_{update:05d}.pt"
            trainer.save(ckpt_path)

    # ── Final checkpoint ──────────────────────────────────────────────────────
    if cfg.get("checkpoint", True):
        trainer.save("outputs/checkpoints/ppo_final.pt")

    csv_file.close()
    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s  |  Log → {csv_path}")

    return history, policy, env


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> dict:
    p = argparse.ArgumentParser(description="Train PPO on GridWorldEnv")
    p.add_argument("--grid-size",        type=int,   default=DEFAULTS["grid_size"])
    p.add_argument("--wall-density",     type=float, default=DEFAULTS["wall_density"])
    p.add_argument("--trap-density",     type=float, default=DEFAULTS["trap_density"])
    p.add_argument("--total-timesteps",  type=int,   default=DEFAULTS["total_timesteps"])
    p.add_argument("--lr",               type=float, default=DEFAULTS["lr"])
    p.add_argument("--n-steps",          type=int,   default=DEFAULTS["n_steps"])
    p.add_argument("--n-epochs",         type=int,   default=DEFAULTS["n_epochs"])
    p.add_argument("--batch-size",       type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--entropy-coef",     type=float, default=DEFAULTS["entropy_coef"])
    p.add_argument("--clip-eps",         type=float, default=DEFAULTS["clip_eps"])
    p.add_argument("--hidden-dim",       type=int,   default=DEFAULTS["hidden_dim"])
    p.add_argument("--seed",             type=int,   default=DEFAULTS["seed"])
    p.add_argument("--no-checkpoint",    action="store_true")
    args = p.parse_args()

    cfg = {**DEFAULTS}
    cfg.update({k.replace("-", "_"): v for k, v in vars(args).items()})
    cfg["checkpoint"] = not args.no_checkpoint
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    history, policy, env = train(cfg)

    # Auto-generate plots after training
    try:
        from utils.visualise import plot_training_curves
        plot_training_curves(history, save_path="outputs/plots/training_curves.png")
        print("Training curves saved → outputs/plots/training_curves.png")
    except Exception as e:
        print(f"Plot generation skipped: {e}")