"""Thin SB3 PPO wrapper — config plumbing + linear schedules.

The function :func:`build_ppo` is the only public surface. It exists to:

1. Pull hyperparameters out of :class:`TrainConfig` once, in one place,
   so the trainer doesn't sprout config-knowledge.
2. Apply linear schedules to ``learning_rate`` and ``clip_range``
   (see ``configs/ppo.yaml`` comments — Atari PPO benefits from these and
   SB3 supports them out of the box via callables on a [0, 1] progress var).
3. Wire our :class:`RetroCNN` feature extractor via
   :func:`retro_rl.models.policies.policy_kwargs`.

We deliberately do **not** subclass :class:`PPO`. SB3 already exposes
``save``/``load``/``predict`` matching our :class:`Agent` protocol.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from retro_rl.models.policies import policy_kwargs as build_policy_kwargs
from retro_rl.utils.config import TrainConfig


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Return an SB3-compatible schedule that linearly anneals to 0.

    SB3 calls schedules with ``progress_remaining`` in [1.0, 0.0]. Multiplying
    by ``initial_value`` produces a linear ramp from ``initial_value`` (start)
    to 0 (end).
    """

    def _fn(progress_remaining: float) -> float:
        return float(progress_remaining * initial_value)

    return _fn


def build_ppo(
    vec_env: VecEnv,
    cfg: TrainConfig,
    tb_log_path: Path | None = None,
    features_dim: int = 512,
) -> PPO:
    """Construct an SB3 PPO model from :class:`TrainConfig`.

    Parameters
    ----------
    vec_env
        Vectorized env (e.g. ``SubprocVecEnv(make_env_fn(...))``). PPO needs a
        ``VecEnv``; passing a single env trips a noisy SB3 warning + wrap.
    cfg
        Top-level training config; only ``cfg.ppo``, ``cfg.seed``,
        ``cfg.policy`` are read here.
    tb_log_path
        If provided, SB3 writes TensorBoard event files under this directory.
    features_dim
        CNN feature dimension (forwarded to :class:`RetroCNN`).

    Returns
    -------
    stable_baselines3.PPO
        Ready to call ``.learn(total_timesteps=cfg.total_timesteps, ...)``.
    """
    if cfg.policy != "cnn":
        raise ValueError(
            f"build_ppo: only 'cnn' policy is wired today; got {cfg.policy!r}. "
            f"Add a registry when a second policy lands."
        )

    ppo_cfg = cfg.ppo
    return PPO(
        policy="CnnPolicy",
        env=vec_env,
        learning_rate=linear_schedule(ppo_cfg.learning_rate),
        n_steps=ppo_cfg.n_steps,
        batch_size=ppo_cfg.batch_size,
        n_epochs=ppo_cfg.n_epochs,
        gamma=ppo_cfg.gamma,
        gae_lambda=ppo_cfg.gae_lambda,
        clip_range=linear_schedule(ppo_cfg.clip_range),
        ent_coef=ppo_cfg.ent_coef,
        vf_coef=ppo_cfg.vf_coef,
        max_grad_norm=ppo_cfg.max_grad_norm,
        normalize_advantage=ppo_cfg.normalize_advantage,
        policy_kwargs=build_policy_kwargs(features_dim=features_dim),
        tensorboard_log=str(tb_log_path) if tb_log_path is not None else None,
        seed=cfg.seed,
        verbose=1,
    )


__all__ = ["build_ppo", "linear_schedule"]
