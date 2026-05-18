"""retro_rl.evaluation — deterministic rollouts and metrics.

Public surface:

* :func:`evaluate` — run N episodes, return (EvalMetrics, frames).
* :class:`EpisodeResult` — per-episode outcome.
* :class:`EvalMetrics` — aggregate statistics over N episodes.
* :func:`compute_metrics` — pure aggregation over a list of EpisodeResult.
"""

from retro_rl.evaluation.evaluator import evaluate
from retro_rl.evaluation.metrics import EpisodeResult, EvalMetrics, compute_metrics

__all__ = ["evaluate", "EpisodeResult", "EvalMetrics", "compute_metrics"]
