"""Evaluation metrics — pure computation over rollout data.

No env, no agent, no I/O. All functions are stateless and trivially testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EpisodeResult:
    """Single-episode outcome collected by :func:`evaluate`."""

    return_: float  # cumulative reward received during the episode
    length: int  # number of env steps taken
    stage_cleared: bool  # True if the stage-clear flag fired at any step
    deaths: int  # count of ``terminated=True`` events (not truncations)


@dataclass(frozen=True)
class EvalMetrics:
    """Aggregate statistics over N evaluation episodes."""

    n_episodes: int
    mean_return: float
    std_return: float
    min_return: float
    max_return: float
    mean_length: float
    std_length: float
    stage_clear_rate: float  # fraction of episodes where stage was cleared
    mean_deaths: float  # mean deaths per episode


def compute_metrics(episodes: list[EpisodeResult]) -> EvalMetrics:
    """Aggregate a list of episode results into :class:`EvalMetrics`.

    Raises
    ------
    ValueError
        If *episodes* is empty.
    """
    if not episodes:
        raise ValueError("episodes must be non-empty")
    returns = [e.return_ for e in episodes]
    lengths = [e.length for e in episodes]
    return EvalMetrics(
        n_episodes=len(episodes),
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        min_return=float(np.min(returns)),
        max_return=float(np.max(returns)),
        mean_length=float(np.mean(lengths)),
        std_length=float(np.std(lengths)),
        stage_clear_rate=float(np.mean([e.stage_cleared for e in episodes])),
        mean_deaths=float(np.mean([e.deaths for e in episodes])),
    )


__all__ = ["EpisodeResult", "EvalMetrics", "compute_metrics"]
