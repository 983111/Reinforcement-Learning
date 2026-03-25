"""
tests/test_grid_world.py
------------------------
Unit tests for GridWorldEnv.
Run with:  python -m pytest tests/ -v
"""

import numpy as np
import pytest
from envs.grid_world import (
    GridWorldEnv,
    EMPTY, WALL, TRAP, GOAL, AGENT,
    R_GOAL, R_STEP, R_TRAP, R_WALL_BUMP,
    VIEW_HALF,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    """Default 10×10 env with fixed seed."""
    e = GridWorldEnv(grid_size=10, seed=42)
    e.reset(seed=42)
    return e


@pytest.fixture
def small_env():
    """Minimal 5×5 env for deterministic tests."""
    e = GridWorldEnv(
        grid_size=5,
        wall_density=0.0,
        trap_density=0.0,
        max_steps=50,
        seed=0,
    )
    e.reset(seed=0)
    return e


# ── Spaces ────────────────────────────────────────────────────────────────────

class TestSpaces:
    def test_observation_shape(self, env):
        obs, _ = env.reset()
        expected_dim = (2 * VIEW_HALF + 1) ** 2
        assert obs.shape == (expected_dim,), f"Expected ({expected_dim},), got {obs.shape}"

    def test_observation_dtype(self, env):
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_observation_range(self, env):
        obs, _ = env.reset()
        assert obs.min() >= 0.0
        assert obs.max() <= float(AGENT)

    def test_action_space_size(self, env):
        assert env.action_space.n == 4

    def test_action_space_contains(self, env):
        for a in range(4):
            assert env.action_space.contains(a)
        assert not env.action_space.contains(4)


# ── Map generation ────────────────────────────────────────────────────────────

class TestMapGeneration:
    def test_border_is_all_walls(self, env):
        g = env.grid
        assert np.all(g[0, :]  == WALL), "Top border should be walls"
        assert np.all(g[-1, :] == WALL), "Bottom border should be walls"
        assert np.all(g[:, 0]  == WALL), "Left border should be walls"
        assert np.all(g[:, -1] == WALL), "Right border should be walls"

    def test_goal_is_placed(self, env):
        assert GOAL in env.grid, "Goal tile must exist in grid"

    def test_goal_position(self, env):
        gr, gc = env.goal_pos
        assert env.grid[gr, gc] == GOAL

    def test_agent_cell_is_not_wall(self, env):
        r, c = env.agent_pos
        assert env.grid[r, c] != WALL

    def test_reachability(self):
        """Every generated map must have a reachable goal."""
        for seed in range(20):
            e = GridWorldEnv(grid_size=10, seed=seed)
            e.reset(seed=seed)
            assert e._is_reachable(e.grid, e.agent_pos, e.goal_pos), \
                f"Map with seed={seed} has unreachable goal"

    def test_different_seeds_differ(self):
        e1 = GridWorldEnv(seed=1); e1.reset(seed=1)
        e2 = GridWorldEnv(seed=2); e2.reset(seed=2)
        assert not np.array_equal(e1.grid, e2.grid), \
            "Different seeds should produce different maps"

    def test_same_seed_reproducible(self):
        e1 = GridWorldEnv(seed=99); e1.reset(seed=99)
        e2 = GridWorldEnv(seed=99); e2.reset(seed=99)
        assert np.array_equal(e1.grid, e2.grid)


# ── Step mechanics ────────────────────────────────────────────────────────────

class TestStepMechanics:
    def test_step_returns_correct_types(self, env):
        obs, rew, terminated, truncated, info = env.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(rew, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_floor_step_reward(self, small_env):
        """Moving onto an empty floor tile gives R_STEP."""
        # Agent starts at (1,1). Move right → (1,2) which is EMPTY in 0-density map
        _, rew, term, trunc, _ = small_env.step(3)  # RIGHT
        if not term:
            assert rew == pytest.approx(R_STEP)

    def test_wall_bump_reward(self, small_env):
        """Bumping a wall gives R_WALL_BUMP and agent doesn't move."""
        pos_before = small_env.agent_pos
        # Move UP from (1,1) hits the top wall (row 0)
        _, rew, _, _, _ = small_env.step(0)  # UP
        assert rew == pytest.approx(R_WALL_BUMP), f"Expected {R_WALL_BUMP}, got {rew}"
        # Agent didn't move (hit border wall from (1,1))
        assert small_env.agent_pos == pos_before

    def test_invalid_action_raises(self, env):
        with pytest.raises(AssertionError):
            env.step(99)

    def test_truncation_at_max_steps(self):
        """Episode truncates exactly at max_steps."""
        e = GridWorldEnv(grid_size=10, max_steps=5, wall_density=0.0, trap_density=0.0, seed=0)
        e.reset(seed=0)
        truncated = False
        for _ in range(5):
            _, _, terminated, truncated, _ = e.step(0)  # keep bumping UP
            if terminated:
                break
        assert truncated or terminated

    def test_goal_terminates_episode(self):
        """Stepping onto the goal terminates with terminated=True."""
        # 5×5, no walls/traps; agent at (1,1), goal at (3,3)
        e = GridWorldEnv(
            grid_size=5, wall_density=0.0, trap_density=0.0, max_steps=100, seed=0
        )
        e.reset(seed=0)
        # Force agent next to goal
        e.agent_pos = (3, 2)
        _, rew, terminated, truncated, _ = e.step(3)  # RIGHT → (3,3) = GOAL
        assert terminated, "Goal tile should terminate episode"
        assert not truncated
        assert rew == pytest.approx(R_GOAL)


# ── Observation ───────────────────────────────────────────────────────────────

class TestObservation:
    def test_agent_center_encoded(self, env):
        obs, _ = env.reset()
        center = VIEW_HALF * (2 * VIEW_HALF + 1) + VIEW_HALF  # index 12 for 5×5
        assert obs[center] == float(AGENT), "Centre cell must encode AGENT"

    def test_obs_changes_after_move(self, env):
        obs1, _ = env.reset()
        obs2, _, _, _, _ = env.step(3)  # RIGHT
        assert not np.array_equal(obs1, obs2) or env.agent_pos == (1, 1), \
            "Observation should change (or agent was wall-blocked)"

    def test_oob_padded_as_wall(self):
        """Cells outside the grid appear as WALL in the observation."""
        e = GridWorldEnv(
            grid_size=5, wall_density=0.0, trap_density=0.0, seed=0
        )
        e.reset(seed=0)
        # Agent is at (1,1). Top-left corner of window is (-1,-1) → OOB → WALL
        obs = e._get_obs()
        assert obs[0] == float(WALL), "OOB top-left cell should be WALL"


# ── Info dict ─────────────────────────────────────────────────────────────────

class TestInfoDict:
    def test_info_keys(self, env):
        _, info = env.reset()
        for key in ("agent_pos", "goal_pos", "step_count", "grid_size", "manhattan_dist"):
            assert key in info, f"Missing key: {key}"

    def test_step_count_increments(self, env):
        env.reset()
        for i in range(1, 4):
            _, _, _, _, info = env.step(0)
            assert info["step_count"] == i

    def test_manhattan_dist_type(self, env):
        _, info = env.reset()
        assert isinstance(info["manhattan_dist"], (int, np.integer))


# ── Rendering ─────────────────────────────────────────────────────────────────

class TestRendering:
    def test_rgb_array_shape(self):
        e = GridWorldEnv(grid_size=8, render_mode="rgb_array", seed=0)
        e.reset(seed=0)
        img = e.render()
        px = GridWorldEnv.CELL_PX
        assert img.shape == (8 * px, 8 * px, 3)

    def test_rgb_array_dtype(self):
        e = GridWorldEnv(grid_size=6, render_mode="rgb_array", seed=1)
        e.reset(seed=1)
        img = e.render()
        assert img.dtype == np.uint8

    def test_human_render_no_crash(self, capsys):
        e = GridWorldEnv(grid_size=6, render_mode="human", seed=2)
        e.reset(seed=2)
        e.render()
        captured = capsys.readouterr()
        assert "Step:" in captured.out


# ── Properties ───────────────────────────────────────────────────────────────

class TestProperties:
    def test_n_actions(self, env):
        assert env.n_actions == 4

    def test_obs_dim(self, env):
        assert env.obs_dim == (2 * VIEW_HALF + 1) ** 2

    def test_repr(self, env):
        r = repr(env)
        assert "GridWorldEnv" in r
        assert "grid_size" in r


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
