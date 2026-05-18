"""Agent protocol — structural interface implemented by every algorithm.

Both SB3's :class:`PPO` and our :class:`RandomAgent` satisfy this protocol
without explicit inheritance; using :class:`typing.Protocol` lets the trainer
and backend treat them uniformly without forcing SB3 classes into a custom
hierarchy.

Method shapes mirror SB3's ``BaseAlgorithm.predict`` so the protocol is a
no-op contract for any SB3 model and a minimal target for our hand-rolled
baselines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Agent(Protocol):
    """Minimal agent surface consumed by trainer, evaluator, and backend."""

    def predict(
        self,
        observation: np.ndarray,
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """Return ``(action, next_state)`` for a single (or batched) observation."""
        ...

    def save(self, path: str | Path) -> None:
        """Persist the agent to ``path``. Format is implementation-defined."""
        ...

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "Agent":
        """Restore an agent previously saved to ``path``."""
        ...


__all__ = ["Agent"]
