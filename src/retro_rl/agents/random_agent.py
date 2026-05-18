"""Uniform-random baseline.

Used as a sanity floor for evaluation: any trained policy must clear this on
mean episode return, otherwise something is broken upstream of the algorithm.

Persistence
-----------
The action space and seed are serialized to a small JSON sidecar. We avoid
pickle because (a) the agent has no learnable state, (b) JSON makes the
artifact diffable and human-readable, (c) loading is cheap.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces


class RandomAgent:
    """Samples uniformly from the action space. Conforms to :class:`Agent`."""

    def __init__(self, action_space: spaces.Space, seed: int | None = None):
        self.action_space = action_space
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        # Seed the gymnasium space itself so action_space.sample() is
        # reproducible too (we don't rely on it but it costs nothing).
        if seed is not None:
            action_space.seed(seed)

    def predict(
        self,
        observation: np.ndarray,
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """Return a random action. ``deterministic`` is ignored (no policy).

        Batch detection: image observations are 3D single (H, W, C) or 4D
        batched (N, H, W, C); we treat ``ndim == 4`` as the batched-image case
        and anything else as a single observation. This matches the shapes
        produced by our env wrappers and SB3's VecEnv.
        """
        obs = np.asarray(observation)
        batched = obs.ndim == 4

        if isinstance(self.action_space, spaces.Discrete):
            n = int(self.action_space.n)
            if batched:
                return (
                    self._rng.integers(0, n, size=(obs.shape[0],), dtype=np.int64),
                    None,
                )
            return np.array(self._rng.integers(0, n), dtype=np.int64), None

        # Non-discrete: defer to gymnasium's sampler (one sample per batch row).
        if batched:
            samples = np.stack(
                [np.asarray(self.action_space.sample()) for _ in range(obs.shape[0])]
            )
            return samples, None
        return np.asarray(self.action_space.sample()), None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _action_space_to_json(self.action_space) | {"seed": self._seed}
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "RandomAgent":
        data = json.loads(Path(path).read_text())
        action_space = _action_space_from_json(data)
        return cls(action_space, seed=data.get("seed"))


def _action_space_to_json(space: spaces.Space) -> dict[str, Any]:
    if isinstance(space, spaces.Discrete):
        return {"type": "Discrete", "n": int(space.n)}
    if isinstance(space, spaces.MultiDiscrete):
        return {"type": "MultiDiscrete", "nvec": [int(x) for x in space.nvec]}
    if isinstance(space, spaces.MultiBinary):
        return {"type": "MultiBinary", "n": int(space.n)}
    raise NotImplementedError(
        f"RandomAgent.save: unsupported action space {type(space).__name__}"
    )


def _action_space_from_json(data: dict[str, Any]) -> spaces.Space:
    t = data["type"]
    if t == "Discrete":
        return spaces.Discrete(int(data["n"]))
    if t == "MultiDiscrete":
        return spaces.MultiDiscrete(np.asarray(data["nvec"], dtype=np.int64))
    if t == "MultiBinary":
        return spaces.MultiBinary(int(data["n"]))
    raise NotImplementedError(f"RandomAgent.load: unsupported action space type {t!r}")


__all__ = ["RandomAgent"]
