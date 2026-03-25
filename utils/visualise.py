"""
AgentGrid — utils/visualise.py
================================
Plotting helpers for training curves and grid snapshots.
Requires matplotlib. All functions save to disk and optionally display.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np


def plot_training_curves(
    history: dict,
    save_path: str = "outputs/plots/training_curves.png",
    show: bool = False,
):
    """
    Plot PPO training metrics in a 2×3 grid of subplots.

    Parameters
    ----------
    history   : dict  — keys match train.py's history dict
    save_path : str
    show      : bool  — call plt.show() after saving
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("AgentGrid — PPO Training Curves", fontsize=14, fontweight="bold")
    fig.patch.set_facecolor("#f8f8f8")

    plots = [
        ("mean_reward",   "Mean Episode Reward",  "#2196F3", True),
        ("mean_len",      "Mean Episode Length",  "#4CAF50", False),
        ("policy_loss",   "Policy Loss",          "#F44336", False),
        ("value_loss",    "Value Loss",           "#FF9800", False),
        ("entropy",       "Policy Entropy",       "#9C27B0", False),
        ("approx_kl",     "Approx KL Divergence", "#00BCD4", False),
    ]

    steps = history.get("timesteps", list(range(len(history["mean_reward"]))))

    for ax, (key, title, color, fill) in zip(axes.flat, plots):
        if key not in history or not history[key]:
            ax.set_visible(False)
            continue

        y = np.array(history[key], dtype=float)
        # Remove NaNs for plotting
        mask = ~np.isnan(y)
        x    = np.array(steps)[mask]
        y    = y[mask]

        ax.plot(x, y, color=color, linewidth=1.5, alpha=0.9)

        if fill and len(y) > 5:
            # Rolling mean ± std band
            window = max(1, len(y) // 20)
            rolled_mean = np.convolve(y, np.ones(window) / window, mode="valid")
            pad = len(y) - len(rolled_mean)
            ax.fill_between(
                x[pad:], rolled_mean - rolled_mean.std(),
                rolled_mean + rolled_mean.std(),
                alpha=0.15, color=color,
            )

        ax.set_title(title, fontsize=11, fontweight="500")
        ax.set_xlabel("Timesteps", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_facecolor("#fdfdfd")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        # Annotate final value
        if len(y) > 0:
            ax.annotate(
                f"{y[-1]:.3f}",
                xy=(x[-1], y[-1]),
                fontsize=8,
                color=color,
                ha="right",
                va="bottom",
            )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Training curves saved → {save_path}")

    if show:
        plt.show()
    plt.close()


def render_grid_snapshot(
    env,
    save_path: str = "outputs/plots/grid_snapshot.png",
    title: str = "GridWorld — agent snapshot",
):
    """
    Save a single RGB render of the current grid state as PNG.

    Parameters
    ----------
    env       : GridWorldEnv  — must support _render_rgb_array()
    save_path : str
    title     : str
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    img = env._render_rgb_array()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    ax.set_title(title, fontsize=11)
    ax.axis("off")

    from envs.grid_world import EMPTY, WALL, TRAP, GOAL, AGENT
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=np.array(env.TILE_COLORS[EMPTY])/255, label="Floor"),
        Patch(facecolor=np.array(env.TILE_COLORS[WALL]) /255, label="Wall"),
        Patch(facecolor=np.array(env.TILE_COLORS[TRAP]) /255, label="Trap"),
        Patch(facecolor=np.array(env.TILE_COLORS[GOAL]) /255, label="Goal"),
        Patch(facecolor=np.array(env.TILE_COLORS[AGENT])/255, label="Agent"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper right",
        fontsize=7, framealpha=0.85,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Grid snapshot saved → {save_path}")
    plt.close()


def plot_value_heatmap(
    policy,
    env,
    save_path: str = "outputs/plots/value_heatmap.png",
    device: str = "cpu",
):
    """
    Visualise the critic's value estimates across every reachable grid cell.
    Shows which positions the agent considers high-value (close to goal).

    Parameters
    ----------
    policy    : ActorCritic
    env       : GridWorldEnv
    save_path : str
    device    : str
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import torch
    except ImportError:
        print("matplotlib/torch not available")
        return

    from envs.grid_world import WALL
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    G = env.grid_size
    value_grid = np.full((G, G), np.nan)

    original_pos = env.agent_pos

    for r in range(G):
        for c in range(G):
            if env.grid[r, c] == WALL:
                continue
            env.agent_pos = (r, c)
            obs = env._get_obs()
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, value = policy.forward(obs_t)
            value_grid[r, c] = value.item()

    env.agent_pos = original_pos

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(value_grid, cmap="RdYlGn", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="V(s)")
    ax.set_title("Critic Value Heatmap\n(green = high value, red = low)", fontsize=11)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    # Mark goal
    gr, gc = env.goal_pos
    ax.plot(gc, gr, "w*", markersize=14, label="Goal")
    ar, ac = env.agent_pos
    ax.plot(ac, ar, "b^", markersize=10, label="Agent start")
    ax.legend(fontsize=8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Value heatmap saved → {save_path}")
    plt.close()