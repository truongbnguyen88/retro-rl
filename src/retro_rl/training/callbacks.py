"""SB3 callbacks for periodic checkpointing + deterministic eval with video.

Two callbacks, both stateless except for a cursor of the next firing step.

:class:`PeriodicCheckpointCallback`
    Fires every ``every_steps`` env steps. Calls ``manager.save`` with
    ``eval_return=None`` — contributes to last-K rotation but not to best.

:class:`EvalAndVideoCallback`
    Fires every ``every_steps`` env steps. Runs ``n_episodes`` deterministic
    rollouts on a single eval env (built lazily on first eval to avoid env
    construction before training starts). Logs ``eval/mean_return``,
    ``eval/std_return``, ``eval/mean_length`` to TB. Calls ``manager.save``
    with the mean return — updates best when applicable. Optionally records an
    mp4 of the first eval episode each cycle.

Cadence overlap: if eval and periodic-checkpoint cadences land on the same
step, both will call ``manager.save`` and the second write wins on disk.
Configure cadences so this is rare (e.g. eval at 100k, ckpt at 250k).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from retro_rl.training.checkpoint import CheckpointManager
from retro_rl.utils.video import write_mp4


class PeriodicCheckpointCallback(BaseCallback):
    """Saves a step checkpoint every ``every_steps`` env steps."""

    def __init__(self, manager: CheckpointManager, every_steps: int, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        if every_steps < 1:
            raise ValueError(f"every_steps must be >= 1, got {every_steps}")
        self.manager = manager
        self.every_steps = every_steps
        self._next_save = every_steps

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_save:
            return True
        self.manager.save(self.model, self.num_timesteps, eval_return=None)
        self._next_save = ((self.num_timesteps // self.every_steps) + 1) * self.every_steps
        return True


class EntCoefLinearSchedule(BaseCallback):
    """Linearly anneal ``model.ent_coef`` from ``initial`` to ``final``.

    Motivation
    ----------
    SB3's ``PPO`` accepts ``ent_coef`` only as a float, not a callable schedule.
    For Atari-style policies with small Discrete action spaces, a constant
    ent_coef large enough to encourage exploration early tends to also prevent
    the policy from committing late: gradients shrink as the value function
    fits, and the entropy regulariser becomes the dominant loss term, pulling
    the policy back toward uniform. We observed this empirically on v5 — eval
    return plateaued at 1.4M then de-committed by 2M while ``approx_kl → 0``.

    Implementation
    --------------
    PPO reads ``self.ent_coef`` afresh at every optimisation step. We update it
    on ``_on_rollout_end`` (i.e. just before the next ``train()`` call) so the
    value PPO sees during a given gradient-update pass is consistent. The
    schedule is linear in ``num_timesteps / total_timesteps``, clamped to
    ``[0, 1]``. We also log the current value to TensorBoard under
    ``train/ent_coef`` so the schedule is auditable.
    """

    def __init__(
        self,
        initial: float,
        final: float,
        total_timesteps: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        if total_timesteps <= 0:
            raise ValueError(f"total_timesteps must be > 0, got {total_timesteps}")
        if initial < 0 or final < 0:
            raise ValueError(f"ent_coef bounds must be >= 0; got initial={initial}, final={final}")
        self.initial = float(initial)
        self.final = float(final)
        self.total_timesteps = int(total_timesteps)

    def _current_value(self) -> float:
        # Read directly from the model. ``BaseCallback.num_timesteps`` is a
        # cached attribute that's only refreshed inside ``on_step``; bypassing
        # the cache makes the schedule robust to which lifecycle hook fired.
        steps = int(self.model.num_timesteps)
        progress = min(1.0, max(0.0, steps / self.total_timesteps))
        return self.initial + progress * (self.final - self.initial)

    def _on_training_start(self) -> None:
        self.model.ent_coef = self._current_value()
        self.logger.record("train/ent_coef", float(self.model.ent_coef))

    def _on_rollout_end(self) -> None:
        self.model.ent_coef = self._current_value()
        self.logger.record("train/ent_coef", float(self.model.ent_coef))

    def _on_step(self) -> bool:
        return True


class EvalAndVideoCallback(BaseCallback):
    """Deterministic eval + optional video; updates best via manager.

    Parameters
    ----------
    eval_env_factory
        ``(record_video: bool) -> gym.Env``. When ``record_video`` is True the
        factory must build the env with ``render_mode='rgb_array'`` so
        ``env.render()`` returns frames. Single env only (no VecEnv).
    n_episodes
        Number of deterministic episodes per eval. Mean/std reported.
    every_steps
        Env-step cadence between evals.
    manager
        :class:`CheckpointManager`. Receives ``save(model, step, eval_return)``
        after each eval — the only path that updates ``best.zip``.
    video_dir
        If set, writes ``eval-step-<N>.mp4`` of the FIRST eval episode per
        cycle. None disables video.
    video_fps
        Output mp4 framerate. 30 matches Genesis native.
    """

    def __init__(
        self,
        eval_env_factory: Callable[[bool], gym.Env],
        n_episodes: int,
        every_steps: int,
        manager: CheckpointManager,
        video_dir: Path | None = None,
        video_fps: int = 30,
        eval_seed: int | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        if every_steps < 1:
            raise ValueError(f"every_steps must be >= 1, got {every_steps}")
        if n_episodes < 1:
            raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")
        self.eval_env_factory = eval_env_factory
        self.n_episodes = n_episodes
        self.every_steps = every_steps
        self.manager = manager
        self.video_dir = Path(video_dir) if video_dir is not None else None
        self.video_fps = video_fps
        # Each episode resets with seed = eval_seed + ep_i. Gives different
        # env RNG states per episode so std_return is non-zero for stochastic
        # envs. For Airstriker (deterministic save-state), this only helps if
        # sticky_action_prob > 0 or similar env-level noise is added.
        self.eval_seed = eval_seed

        self._eval_env: gym.Env | None = None
        self._next_eval = every_steps

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval:
            return True

        if self._eval_env is None:
            self._eval_env = self.eval_env_factory(self.video_dir is not None)

        returns, lengths, frames = self._run_eval()
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns))
        mean_length = float(np.mean(lengths))

        self.logger.record("eval/mean_return", mean_return)
        self.logger.record("eval/std_return", std_return)
        self.logger.record("eval/mean_length", mean_length)
        self.logger.dump(self.num_timesteps)

        self.manager.save(self.model, self.num_timesteps, eval_return=mean_return)

        if self.video_dir is not None and frames:
            write_mp4(
                frames, self.video_dir / f"eval-step-{self.num_timesteps}.mp4", fps=self.video_fps
            )

        self._next_eval = ((self.num_timesteps // self.every_steps) + 1) * self.every_steps
        return True

    def _on_training_end(self) -> None:
        if self._eval_env is not None:
            self._eval_env.close()
            self._eval_env = None

    # -------------------------------------------------------------- helpers

    def _run_eval(self) -> tuple[list[float], list[int], list[np.ndarray]]:
        assert self._eval_env is not None
        returns: list[float] = []
        lengths: list[int] = []
        frames: list[np.ndarray] = []

        for ep_i in range(self.n_episodes):
            seed = (self.eval_seed + ep_i) if self.eval_seed is not None else None
            obs, _ = self._eval_env.reset(seed=seed)
            ep_return = 0.0
            ep_length = 0
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self._eval_env.step(action)
                done = bool(terminated) or bool(truncated)
                ep_return += float(reward)
                ep_length += 1
                if self.video_dir is not None and ep_i == 0:
                    frame = self._eval_env.render()
                    if frame is not None:
                        frames.append(np.asarray(frame))
            returns.append(ep_return)
            lengths.append(ep_length)

        return returns, lengths, frames


__all__ = ["PeriodicCheckpointCallback", "EvalAndVideoCallback"]
