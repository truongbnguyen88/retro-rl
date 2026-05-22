"""Deterministic evaluation rollouts.

:func:`evaluate` runs N episodes on a single ``gym.Env`` and returns
aggregate metrics plus optional first-episode frames for video.

Agent interface
---------------
Any object satisfying the :class:`~retro_rl.agents.base.Agent` protocol —
``predict(obs, state, episode_start, deterministic) -> (action, next_state)``.
The returned ``next_state`` is threaded back across steps so recurrent policies
(``RecurrentPPO``) keep their LSTM hidden state; plain ``PPO`` and
``RandomAgent`` ignore it and return ``None``.

Death counting
--------------
A "death" is counted when ``terminated=True`` (not truncation). This aligns
with the ``EndOnLifeLost`` wrapper behaviour in the env stack.

Stage-clear detection
---------------------
``info_keys["stage_clear"]`` is sampled at each step; if it is truthy at any
point during the episode, the episode is flagged as cleared. Defaults to
``DEFAULT_INFO_KEYS`` from ``reward_shaping`` when not supplied.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from retro_rl.env.reward_shaping import DEFAULT_INFO_KEYS
from retro_rl.evaluation.metrics import EpisodeResult, EvalMetrics, compute_metrics


def evaluate(
    agent: Any,
    env: gym.Env,
    n_episodes: int = 20,
    *,
    deterministic: bool = True,
    record_video: bool = False,
    info_keys: dict[str, str] | None = None,
) -> tuple[EvalMetrics, list[np.ndarray]]:
    """Run *n_episodes* rollouts and return aggregate metrics.

    Parameters
    ----------
    agent
        Object satisfying the :class:`~retro_rl.agents.base.Agent` protocol.
        Its returned recurrent state is threaded across steps within an episode
        and reset at each episode boundary.
    env
        Single (non-vectorised) ``gym.Env``. Must be constructed with
        ``render_mode='rgb_array'`` when ``record_video=True``.
    n_episodes
        Number of episodes to roll out. Must be >= 1.
    deterministic
        Passed through to ``agent.predict``.
    record_video
        If True, collect rendered frames for episode 0 only.
    info_keys
        Map of semantic name → info dict key. Defaults to
        ``DEFAULT_INFO_KEYS`` from :mod:`retro_rl.env.reward_shaping`.

    Returns
    -------
    metrics
        :class:`~retro_rl.evaluation.metrics.EvalMetrics` aggregate.
    frames
        List of RGB ``np.ndarray`` frames from episode 0.
        Empty when ``record_video=False`` or env returns no render output.

    Raises
    ------
    ValueError
        If *n_episodes* < 1.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")

    _info_keys = info_keys if info_keys is not None else DEFAULT_INFO_KEYS
    stage_clear_key = _info_keys.get("stage_clear", "stage_clear")

    episode_results: list[EpisodeResult] = []
    frames: list[np.ndarray] = []

    for ep_i in range(n_episodes):
        obs, _ = env.reset()
        ep_return = 0.0
        ep_length = 0
        ep_deaths = 0
        ep_stage_cleared = False
        done = False

        # Thread recurrent state across the episode. ``state`` carries the LSTM
        # hidden/cell state; ``episode_start`` is True only on the first step so
        # a recurrent policy (RecurrentPPO) zeroes its state at the boundary and
        # then accumulates memory. Both are reset per episode. Plain PPO and
        # RandomAgent ignore both args and return state=None, so this is correct
        # for every Agent. Discarding the returned state (the previous bug) made
        # recurrent policies run with a perpetually-zeroed hidden state — i.e.
        # evaluated as if memoryless, underreporting their true return.
        state: tuple[np.ndarray, ...] | None = None
        episode_start = np.ones((1,), dtype=bool)

        while not done:
            action, state = agent.predict(
                obs,
                state=state,
                episode_start=episode_start,
                deterministic=deterministic,
            )
            episode_start = np.zeros((1,), dtype=bool)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated) or bool(truncated)
            ep_return += float(reward)
            ep_length += 1

            if terminated:
                ep_deaths += 1
            if info.get(stage_clear_key):
                ep_stage_cleared = True

            if record_video and ep_i == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(np.asarray(frame))

        episode_results.append(
            EpisodeResult(
                return_=ep_return,
                length=ep_length,
                stage_cleared=ep_stage_cleared,
                deaths=ep_deaths,
            )
        )

    return compute_metrics(episode_results), frames


__all__ = ["evaluate"]
