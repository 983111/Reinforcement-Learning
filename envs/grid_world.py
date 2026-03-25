"""
AgentGrid — GridWorldEnv
========================
A fully gymnasium-compatible custom environment for training
PPO-based autonomous navigation agents.

Grid legend
-----------
0  — empty floor
1  — wall (impassable)
2  — trap tile  (penalty on entry, episode continues)
3  — goal tile  (large reward, episode ends)
4  — agent      (current position; overlaid at render time)

Observation
-----------
A flattened egocentric 5×5 window centred on the agent.
Out-of-bounds cells are treated as walls (value = 1).
Shape: (25,)  dtype: np.float32

Action space
------------
Discrete(4):  0=UP  1=DOWN  2=LEFT  3=RIGHT

Reward shaping
--------------
+10.0   reaching the goal
-0.05   each step taken             (encourages efficiency)
-1.0    stepping into a trap
-0.5    bumping into a wall         (no movement occurs)
 0.0    normal floor step

Episode termination
-------------------
- Agent reaches the goal                  → truncated=False, terminated=True
- Step count exceeds max_steps            → truncated=True,  terminated=False
"""

from __future__ import annotations

import random
import sys
import os
from typing import Optional

import numpy as np

# ── Gymnasium with offline fallback ──────────────────────────────────────────
try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:
    # Add project root to path so the shim is importable
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    import gymnasium_shim as gym          # type: ignore[no-redef]
    from gymnasium_shim import spaces     # type: ignore[no-redef]

# ── Tile constants ────────────────────────────────────────────────────────────
EMPTY = 0
WALL  = 1
TRAP  = 2
GOAL  = 3
AGENT = 4  # used only for rendering, never stored in self.grid

# ── Reward constants ──────────────────────────────────────────────────────────
R_GOAL       =  10.0
R_STEP       =  -0.05
R_TRAP       =  -1.0
R_WALL_BUMP  =  -0.5

# ── Action → (row_delta, col_delta) ──────────────────────────────────────────
ACTION_DELTAS = {
    0: (-1,  0),   # UP
    1: ( 1,  0),   # DOWN
    2: ( 0, -1),   # LEFT
    3: ( 0,  1),   # RIGHT
}
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

# ── View window half-size ─────────────────────────────────────────────────────
VIEW_HALF = 2   # produces a (2*VIEW_HALF+1)² = 5×5 window


class GridWorldEnv(gym.Env):
    """
    Custom procedurally-generated GridWorld for RL research.

    Parameters
    ----------
    grid_size : int
        Side-length of the square grid (default 10).
    wall_density : float
        Fraction of non-border cells that become walls (default 0.15).
    trap_density : float
        Fraction of non-border cells that become traps (default 0.08).
    max_steps : int
        Episode step limit before truncation (default 200).
    seed : int | None
        Fixed seed for reproducible map generation.
    render_mode : str | None
        'human' for terminal rendering, 'rgb_array' for pixel array.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    # ── Tile colours for rgb_array rendering (RGB tuples) ─────────────────────
    TILE_COLORS = {
        EMPTY: (240, 240, 240),
        WALL:  ( 50,  50,  50),
        TRAP:  (220,  80,  80),
        GOAL:  ( 80, 200, 120),
        AGENT: ( 70, 130, 220),
    }
    CELL_PX = 32   # pixels per cell in rgb_array mode

    def __init__(
        self,
        grid_size: int = 10,
        wall_density: float = 0.15,
        trap_density: float = 0.08,
        max_steps: int = 200,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        assert 5 <= grid_size <= 50, "grid_size must be between 5 and 50"
        assert 0.0 <= wall_density <= 0.4, "wall_density must be in [0, 0.4]"
        assert 0.0 <= trap_density <= 0.3, "trap_density must be in [0, 0.3]"
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.grid_size    = grid_size
        self.wall_density = wall_density
        self.trap_density = trap_density
        self.max_steps    = max_steps
        self.render_mode  = render_mode

        # ── Spaces ────────────────────────────────────────────────────────────
        obs_dim = (2 * VIEW_HALF + 1) ** 2   # 25 for a 5×5 window
        self.observation_space = spaces.Box(
            low=0.0, high=float(AGENT),
            shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)

        # ── Internal state (initialised in reset) ─────────────────────────────
        self.grid: np.ndarray           = np.zeros((grid_size, grid_size), dtype=np.int32)
        self.agent_pos: tuple[int, int] = (1, 1)
        self.goal_pos:  tuple[int, int] = (grid_size - 2, grid_size - 2)
        self._step_count: int           = 0
        self._episode_reward: float     = 0.0

        # ── Seeding ───────────────────────────────────────────────────────────
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    # ── Core API ──────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Generate a new map and place the agent.

        Returns
        -------
        observation : np.ndarray  shape (25,)
        info        : dict        metadata (agent_pos, goal_pos, grid_size)
        """
        if seed is not None:
            self._rng = random.Random(seed)
            self._np_rng = np.random.default_rng(seed)

        self.grid = self._generate_map()
        self._step_count    = 0
        self._episode_reward = 0.0

        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Apply action and advance the environment by one timestep.

        Returns
        -------
        observation  : np.ndarray
        reward       : float
        terminated   : bool   — True if goal reached
        truncated    : bool   — True if step limit exceeded
        info         : dict
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        dr, dc = ACTION_DELTAS[action]
        r, c   = self.agent_pos
        nr, nc = r + dr, c + dc

        # ── Boundary / wall collision ─────────────────────────────────────────
        if not self._in_bounds(nr, nc) or self.grid[nr, nc] == WALL:
            reward = R_WALL_BUMP
            # Agent does not move
        else:
            self.agent_pos = (nr, nc)
            tile = self.grid[nr, nc]

            if tile == GOAL:
                reward     = R_GOAL
                terminated = True
                self._episode_reward += reward
                self._step_count     += 1
                obs  = self._get_obs()
                info = self._get_info()
                info["episode_reward"] = self._episode_reward
                if self.render_mode == "human":
                    self._render_human()
                return obs, reward, True, False, info

            elif tile == TRAP:
                reward = R_TRAP
            else:
                reward = R_STEP

        self._step_count     += 1
        self._episode_reward += reward

        terminated = False
        truncated  = self._step_count >= self.max_steps

        obs  = self._get_obs()
        info = self._get_info()
        if truncated:
            info["episode_reward"] = self._episode_reward

        if self.render_mode == "human":
            self._render_human()

        return obs, reward, terminated, truncated, info

    def render(self):
        """Dispatch to the chosen render mode."""
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()

    def close(self):
        pass  # no window/resource cleanup needed

    # ── Map generation ────────────────────────────────────────────────────────

    def _generate_map(self) -> np.ndarray:
        """
        Procedurally generate a grid map.

        Strategy
        --------
        1. Fill border with walls.
        2. Scatter interior walls randomly.
        3. Scatter traps randomly (no overlap with walls).
        4. Place goal at bottom-right interior corner.
        5. Place agent at top-left interior corner.
        6. Verify a path exists with BFS; if not, retry (max 20 attempts).
        """
        for attempt in range(20):
            grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

            # ── Border walls ──────────────────────────────────────────────────
            grid[0, :]  = WALL
            grid[-1, :] = WALL
            grid[:, 0]  = WALL
            grid[:, -1] = WALL

            # ── Interior positions (excluding borders and fixed positions) ────
            interior = [
                (r, c)
                for r in range(1, self.grid_size - 1)
                for c in range(1, self.grid_size - 1)
            ]
            agent_pos = (1, 1)
            goal_pos  = (self.grid_size - 2, self.grid_size - 2)
            reserved  = {agent_pos, goal_pos}

            candidates = [p for p in interior if p not in reserved]
            self._rng.shuffle(candidates)

            # ── Place walls ───────────────────────────────────────────────────
            n_walls = int(len(candidates) * self.wall_density)
            wall_cells = candidates[:n_walls]
            for r, c in wall_cells:
                grid[r, c] = WALL

            # ── Place traps ───────────────────────────────────────────────────
            remaining = [p for p in candidates[n_walls:]]
            n_traps = int(len(interior) * self.trap_density)
            trap_cells = remaining[:n_traps]
            for r, c in trap_cells:
                grid[r, c] = TRAP

            # ── Fix goal ──────────────────────────────────────────────────────
            grid[goal_pos]  = GOAL
            grid[agent_pos] = EMPTY  # agent is not stored in grid

            # ── BFS reachability check ────────────────────────────────────────
            if self._is_reachable(grid, agent_pos, goal_pos):
                self.agent_pos = agent_pos
                self.goal_pos  = goal_pos
                return grid

        # ── Fallback: open map with just goal ─────────────────────────────────
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        grid[0, :]  = WALL
        grid[-1, :] = WALL
        grid[:, 0]  = WALL
        grid[:, -1] = WALL
        grid[goal_pos] = GOAL
        self.agent_pos = agent_pos
        self.goal_pos  = goal_pos
        return grid

    def _is_reachable(
        self,
        grid: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> bool:
        """BFS to check that `goal` is reachable from `start`."""
        from collections import deque
        visited = set()
        queue   = deque([start])
        visited.add(start)

        while queue:
            r, c = queue.popleft()
            if (r, c) == goal:
                return True
            for dr, dc in ACTION_DELTAS.values():
                nr, nc = r + dr, c + dc
                if (
                    self._in_bounds(nr, nc)
                    and (nr, nc) not in visited
                    and grid[nr, nc] != WALL
                ):
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return False

    # ── Observation ───────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """
        Return a flattened 5×5 egocentric view centred on the agent.
        Out-of-bounds cells are padded with WALL (1).
        The agent's own cell is encoded as AGENT (4).
        """
        r, c   = self.agent_pos
        half   = VIEW_HALF
        window = np.full((2 * half + 1, 2 * half + 1), WALL, dtype=np.float32)

        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                nr, nc = r + dr, c + dc
                if self._in_bounds(nr, nc):
                    window[dr + half, dc + half] = float(self.grid[nr, nc])

        # Mark the agent's own cell
        window[half, half] = float(AGENT)
        return window.flatten()

    # ── Info dict ─────────────────────────────────────────────────────────────

    def _get_info(self) -> dict:
        ar, ac = self.agent_pos
        gr, gc = self.goal_pos
        return {
            "agent_pos":       self.agent_pos,
            "goal_pos":        self.goal_pos,
            "step_count":      self._step_count,
            "grid_size":       self.grid_size,
            "manhattan_dist":  abs(ar - gr) + abs(ac - gc),
        }

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _render_human(self) -> None:
        """Print a coloured ASCII map to stdout."""
        ICONS = {EMPTY: "·", WALL: "█", TRAP: "✗", GOAL: "★", AGENT: "A"}
        lines = []
        for r in range(self.grid_size):
            row = []
            for c in range(self.grid_size):
                if (r, c) == self.agent_pos:
                    row.append(ICONS[AGENT])
                else:
                    row.append(ICONS[self.grid[r, c]])
            lines.append(" ".join(row))
        print("\n".join(lines))
        print(
            f"Step: {self._step_count}  |  "
            f"Agent: {self.agent_pos}  |  "
            f"Goal: {self.goal_pos}  |  "
            f"Reward so far: {self._episode_reward:.2f}"
        )
        print()

    def _render_rgb_array(self) -> np.ndarray:
        """
        Return an (H, W, 3) uint8 RGB image of the grid.
        Used for GIF generation during evaluation.
        """
        px   = self.CELL_PX
        H    = self.grid_size * px
        W    = self.grid_size * px
        img  = np.zeros((H, W, 3), dtype=np.uint8)

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if (r, c) == self.agent_pos:
                    color = self.TILE_COLORS[AGENT]
                else:
                    color = self.TILE_COLORS[self.grid[r, c]]

                y0, y1 = r * px, (r + 1) * px
                x0, x1 = c * px, (c + 1) * px
                img[y0:y1, x0:x1] = color

                # ── Cell border ───────────────────────────────────────────────
                img[y0, x0:x1]   = (180, 180, 180)  # top edge
                img[y0:y1, x0]   = (180, 180, 180)  # left edge

        return img

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def n_actions(self) -> int:
        return int(self.action_space.n)

    @property
    def obs_dim(self) -> int:
        return int(self.observation_space.shape[0])

    def __repr__(self) -> str:
        return (
            f"GridWorldEnv("
            f"grid_size={self.grid_size}, "
            f"wall_density={self.wall_density}, "
            f"trap_density={self.trap_density}, "
            f"max_steps={self.max_steps})"
        )
