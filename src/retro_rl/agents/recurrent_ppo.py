"""Thin sb3-contrib RecurrentPPO wrapper — the v10 algorithm.

Sibling of :func:`retro_rl.agents.ppo.build_ppo`. RecurrentPPO inserts an LSTM
after the feature extractor (so the hidden state owns temporal context), which
is why v10 drops ``frame_stack`` to 1. The feature extractor is wired exactly
as for plain PPO — :func:`retro_rl.models.policies.policy_kwargs` — plus
``lstm_hidden_size`` for the recurrent layer.

We reuse :func:`retro_rl.agents.ppo.linear_schedule` for the learning-rate and
clip-range ramps so v10 inherits v9's schedule behaviour unchanged.
"""

from __future__ import annotations

from pathlib import Path

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecEnv

from retro_rl.agents.ppo import linear_schedule
from retro_rl.models.policies import policy_kwargs as build_policy_kwargs
from retro_rl.utils.config import TrainConfig


def build_recurrent_ppo(
    vec_env: VecEnv,
    cfg: TrainConfig,
    tb_log_path: Path | None = None,
    features_dim: int | None = None,
    features_extractor: str | None = None,
) -> RecurrentPPO:
    """Construct an sb3-contrib RecurrentPPO model from :class:`TrainConfig`.

    Mirrors :func:`retro_rl.agents.ppo.build_ppo`: same feature-extractor
    wiring and linear schedules, but ``policy="CnnLstmPolicy"`` and an LSTM of
    width ``cfg.lstm_hidden_size`` after the extractor.
    """
    if cfg.policy != "cnn":
        raise ValueError(
            f"build_recurrent_ppo: only 'cnn' policy is wired today; got {cfg.policy!r}."
        )

    resolved_dim = features_dim if features_dim is not None else cfg.features_dim
    resolved_extractor = (
        features_extractor if features_extractor is not None else cfg.features_extractor
    )

    kwargs = build_policy_kwargs(
        features_dim=resolved_dim,
        features_extractor=resolved_extractor,
    )
    kwargs["lstm_hidden_size"] = cfg.lstm_hidden_size

    ppo_cfg = cfg.ppo
    return RecurrentPPO(
        policy="CnnLstmPolicy",
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
        policy_kwargs=kwargs,
        tensorboard_log=str(tb_log_path) if tb_log_path is not None else None,
        seed=cfg.seed,
        verbose=1,
    )


__all__ = ["build_recurrent_ppo"]
