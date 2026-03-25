"""
gymnasium_shim.py
-----------------
Minimal stub that replicates the gymnasium API surface used by GridWorldEnv.
This allows the environment to run without installing the full gymnasium package.

In production (with gymnasium installed), this file is never imported —
GridWorldEnv imports from the real `gymnasium` package.

Usage: only needed if `gymnasium` is not available (e.g. offline dev / CI).
"""

import numpy as np


class Space:
    """Abstract base for observation and action spaces."""
    def contains(self, x) -> bool:
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError


class Box(Space):
    """Continuous box space."""
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low   = np.full(shape, low,  dtype=dtype)
        self.high  = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

    def contains(self, x) -> bool:
        x = np.asarray(x)
        return (
            x.shape == self.shape
            and np.all(x >= self.low)
            and np.all(x <= self.high)
        )

    def sample(self) -> np.ndarray:
        return np.random.uniform(
            self.low, self.high, size=self.shape
        ).astype(self.dtype)


class Discrete(Space):
    """Discrete integer action space {0, 1, …, n-1}."""
    def __init__(self, n: int):
        self.n = n

    def contains(self, x) -> bool:
        return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n

    def sample(self) -> int:
        return int(np.random.randint(self.n))


# Expose the `spaces` sub-module interface
class spaces:
    Box      = Box
    Discrete = Discrete


class Env:
    """Minimal gymnasium.Env base class."""
    metadata         = {}
    observation_space: Space = None
    action_space:      Space = None

    def reset(self, *, seed=None, options=None):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self):
        pass

    def close(self):
        pass
