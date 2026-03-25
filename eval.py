"""
AgentGrid — eval.py
====================
Evaluate a trained PPO agent and produce:
  • Per-episode stats (reward, steps, outcome)
  • A GIF of the agent navigating the grid
  • A summary table printed to stdout

Usage
-----
    python eval.py                                   # uses ppo_final.pt
    python eval.py --checkpoint outputs/checkpoints/ppo_update_00050.pt
    python eval.py --n-episodes 20 --deterministic
    python eval.py --no-gif                          # skip GIF generation
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from agents.ppo import ActorCritic
from envs.grid_world import GridWorldEnv
from train import DEFAULTS


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate(
    policy: ActorCritic,
    env: GridWorldEnv,
    n_episodes: int      = 10,
    deterministic: bool  = True,
    device: str          = "cpu",
    render_gif: bool     = True,
    gif_episode: int     = 0,   # which episode to record
    gif_path: str        = "outputs/gifs/agent_eval.gif",
) -> dict:
    """
    Run n_episodes greedy (or stochastic) rollouts.

    Returns
    -------
    dict with keys:
        rewards      — list of episode total rewards
        lengths      — list of episode lengths
        success_rate — fraction of episodes that reached the goal
        mean_reward  — mean episode reward
        std_reward   — std episode reward
    """
    policy.eval()
    rewards  = []
    lengths  = []
    outcomes = []    # 'goal' | 'trap_loop' | 'timeout'
    frames   = []   # collected for GIF

    for ep in range(n_episodes):
        obs, _    = env.reset(seed=ep)
        ep_reward = 0.0
        ep_steps  = 0
        record    = render_gif and (ep == gif_episode)

        if record:
            frame = env._render_rgb_array()
            frames.append(frame)

        while True:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = policy.get_action(obs_t, deterministic=deterministic)
            a = action.item()

            obs, reward, terminated, truncated, _ = env.step(a)
            ep_reward += reward
            ep_steps  += 1

            if record:
                frames.append(env._render_rgb_array())

            if terminated:
                outcomes.append("goal")
                break
            if truncated:
                outcomes.append("timeout")
                break

        rewards.append(ep_reward)
        lengths.append(ep_steps)

    # ── Save GIF ──────────────────────────────────────────────────────────────
    if render_gif and frames:
        _save_gif(frames, gif_path)

    success_rate = outcomes.count("goal") / n_episodes

    return {
        "rewards":      rewards,
        "lengths":      lengths,
        "outcomes":     outcomes,
        "success_rate": success_rate,
        "mean_reward":  float(np.mean(rewards)),
        "std_reward":   float(np.std(rewards)),
        "mean_length":  float(np.mean(lengths)),
    }


def _save_gif(frames: list, path: str, fps: int = 6):
    """Save a list of (H, W, 3) uint8 numpy arrays as an animated GIF."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import imageio
        imageio.mimsave(path, frames, fps=fps)
        print(f"GIF saved → {path}  ({len(frames)} frames)")
    except ImportError:
        # Fallback: save individual frames as PNG strip
        try:
            import struct, zlib

            def _write_png(img: np.ndarray, fpath: str):
                """Minimal PNG writer — no external deps."""
                H, W, C = img.shape
                raw = b""
                for row in img:
                    raw += b"\x00" + row.tobytes()
                def chunk(tag, data):
                    import struct, zlib
                    c = struct.pack(">I", len(data)) + tag + data
                    return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
                header   = b"\x89PNG\r\n\x1a\n"
                ihdr     = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)
                idat_raw = zlib.compress(raw)
                png_data = header + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat_raw) + chunk(b"IEND", b"")
                with open(fpath, "wb") as f:
                    f.write(png_data)

            # Save first frame as PNG
            png_path = path.replace(".gif", "_frame0.png")
            _write_png(frames[0], png_path)
            print(f"imageio not available — first frame saved as PNG → {png_path}")
        except Exception as e:
            print(f"GIF/PNG save failed: {e}")


# ── Pretty print results ──────────────────────────────────────────────────────

def print_results(results: dict, n_episodes: int):
    outcomes = results["outcomes"]
    goal_eps    = outcomes.count("goal")
    timeout_eps = outcomes.count("timeout")

    print(f"\n{'═'*52}")
    print(f"  Evaluation Results  ({n_episodes} episodes)")
    print(f"{'═'*52}")
    print(f"  Success rate      : {results['success_rate']*100:.1f}%  ({goal_eps}/{n_episodes} reached goal)")
    print(f"  Timeouts          : {timeout_eps}")
    print(f"  Mean reward       : {results['mean_reward']:.3f}  ±  {results['std_reward']:.3f}")
    print(f"  Mean episode len  : {results['mean_length']:.1f} steps")
    print(f"\n  Per-episode breakdown:")
    print(f"  {'Ep':>4}  {'Reward':>8}  {'Steps':>6}  {'Outcome'}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*6}  {'─'*8}")
    for i, (r, l, o) in enumerate(zip(results["rewards"], results["lengths"], outcomes)):
        marker = "✓" if o == "goal" else "✗"
        print(f"  {i+1:>4}  {r:>8.3f}  {l:>6}  {marker} {o}")
    print(f"{'═'*52}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained PPO agent")
    p.add_argument("--checkpoint",    default="outputs/checkpoints/ppo_final.pt")
    p.add_argument("--n-episodes",    type=int,   default=10)
    p.add_argument("--grid-size",     type=int,   default=DEFAULTS["grid_size"])
    p.add_argument("--wall-density",  type=float, default=DEFAULTS["wall_density"])
    p.add_argument("--trap-density",  type=float, default=DEFAULTS["trap_density"])
    p.add_argument("--max-ep-steps",  type=int,   default=DEFAULTS["max_ep_steps"])
    p.add_argument("--hidden-dim",    type=int,   default=DEFAULTS["hidden_dim"])
    p.add_argument("--n-layers",      type=int,   default=DEFAULTS["n_layers"])
    p.add_argument("--stochastic",    action="store_true", help="Use stochastic policy")
    p.add_argument("--no-gif",        action="store_true")
    p.add_argument("--gif-path",      default="outputs/gifs/agent_eval.gif")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = GridWorldEnv(
        grid_size    = args.grid_size,
        wall_density = args.wall_density,
        trap_density = args.trap_density,
        max_steps    = args.max_ep_steps,
        render_mode  = "rgb_array",
    )

    policy = ActorCritic(
        obs_dim    = env.obs_dim,
        n_actions  = env.n_actions,
        hidden_dim = args.hidden_dim,
        n_layers   = args.n_layers,
    ).to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        policy.load_state_dict(ckpt["policy_state"])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"Checkpoint not found: {args.checkpoint}  — using random policy")

    results = evaluate(
        policy        = policy,
        env           = env,
        n_episodes    = args.n_episodes,
        deterministic = not args.stochastic,
        device        = device,
        render_gif    = not args.no_gif,
        gif_path      = args.gif_path,
    )
    print_results(results, args.n_episodes)