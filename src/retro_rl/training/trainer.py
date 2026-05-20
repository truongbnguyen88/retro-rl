"""Top-level PPO training orchestration.

:func:`train` wires together env construction, agent build (or resume), callbacks
(periodic checkpoint + eval-with-video), and SB3's ``model.learn`` loop. Returns
the path to the best checkpoint produced (or latest, if no eval set a best).

Resume semantics
----------------
``PPO.load(path, env=vec_env)`` restores weights + optimizer + ``num_timesteps``.
We pass ``reset_num_timesteps=False`` to ``learn`` so the step counter continues
from the resumed value.

Known quirk: changing ``cfg.total_timesteps`` between original and resumed run
shifts the linear LR/clip schedule (progress = ``num_timesteps / total_timesteps``).
Keep ``total_timesteps`` constant across resumes, or accept the schedule jump.

Train env is always ``SubprocVecEnv`` (even at ``n_envs=1``) so the main
process stays free for the eval env. Stable-retro enforces one emulator
instance per process; co-locating train + eval emulators in the main process
would fail with ``RuntimeError("Cannot create multiple emulator instances per
process")``. The IPC cost at ``n_envs=1`` is negligible vs the env stepping
cost.
"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from retro_rl.agents.ppo import build_ppo
from retro_rl.env import make_env, make_env_fn
from retro_rl.training.callbacks import (
    EntCoefLinearSchedule,
    EvalAndVideoCallback,
    PeriodicCheckpointCallback,
)
from retro_rl.training.checkpoint import CheckpointManager
from retro_rl.utils.config import TrainConfig
from retro_rl.utils.logging import get_logger
from retro_rl.utils.seeding import set_global_seed


def train(cfg: TrainConfig, resume_from: Path | None = None) -> Path:
    """Run a full training session. Returns path to best (or latest) checkpoint."""
    set_global_seed(cfg.seed)

    run_dir = cfg.checkpoint_dir / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    video_dir = cfg.video_dir / cfg.run_name if cfg.eval.record_video else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)

    log = get_logger(name=f"retro_rl.train.{cfg.run_name}", run_dir=run_dir)
    log.info(
        "run_name=%s seed=%d total_timesteps=%d n_envs=%d resume=%s",
        cfg.run_name,
        cfg.seed,
        cfg.total_timesteps,
        cfg.n_envs,
        resume_from,
    )

    config_snapshot_path = run_dir / "config_snapshot.json"
    config_snapshot_path.write_text(json.dumps(cfg.model_dump(mode="json"), indent=2, default=str))

    vecnorm_stats: Path | None = None
    if resume_from is not None and cfg.normalize_reward:
        candidate = resume_from.with_suffix(".pkl")
        if candidate.exists():
            vecnorm_stats = candidate
            log.info("restoring VecNormalize stats from %s", candidate)
        else:
            log.warning(
                "normalize_reward set but no VecNormalize stats at %s; "
                "resuming with fresh running averages (reward scale will re-estimate)",
                candidate,
            )
    vec_env = _build_vec_env(cfg, vecnormalize_stats=vecnorm_stats)
    if cfg.normalize_reward:
        log.info("VecNormalize active: norm_reward=True norm_obs=False gamma=%.4g", cfg.ppo.gamma)

    if resume_from is None:
        log.info(
            "features_extractor=%s features_dim=%d",
            cfg.features_extractor,
            cfg.features_dim,
        )
        model = build_ppo(vec_env, cfg, tb_log_path=cfg.log_dir)
    else:
        log.info("resuming from %s", resume_from)
        model = PPO.load(
            str(resume_from),
            env=vec_env,
            tensorboard_log=str(cfg.log_dir),
            verbose=1,
        )

    manager = CheckpointManager(
        root=cfg.checkpoint_dir,
        run_name=cfg.run_name,
        keep_last_k=cfg.checkpoint.keep_last_k,
        keep_best=cfg.checkpoint.keep_best,
        config_snapshot_path=config_snapshot_path,
    )

    eval_seed = cfg.seed + 10_000
    eval_env_factory = _build_eval_env_factory(cfg, eval_seed)
    callback_list: list = [
        PeriodicCheckpointCallback(
            manager=manager,
            every_steps=cfg.checkpoint.every_steps,
        ),
        EvalAndVideoCallback(
            eval_env_factory=eval_env_factory,
            n_episodes=cfg.eval.n_episodes,
            every_steps=cfg.eval.every_steps,
            manager=manager,
            video_dir=video_dir,
            eval_seed=eval_seed,
        ),
    ]
    if cfg.ppo.ent_coef_final is not None:
        callback_list.append(
            EntCoefLinearSchedule(
                initial=cfg.ppo.ent_coef,
                final=cfg.ppo.ent_coef_final,
                total_timesteps=cfg.total_timesteps,
            )
        )
        log.info(
            "ent_coef schedule: linear %.4g → %.4g over %d steps",
            cfg.ppo.ent_coef,
            cfg.ppo.ent_coef_final,
            cfg.total_timesteps,
        )
    callbacks = CallbackList(callback_list)

    try:
        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=callbacks,
            tb_log_name=cfg.run_name,
            log_interval=cfg.log_interval,
            reset_num_timesteps=resume_from is None,
        )
    finally:
        vec_env.close()

    # Always emit a final checkpoint so resume is possible even if no eval fired.
    final_step = int(model.num_timesteps)
    manager.save(model, step=final_step, eval_return=None)
    log.info("training complete; final_step=%d best_return=%s", final_step, manager.best_return)

    ckpt = manager.best() or manager.latest()
    if ckpt is None:
        raise RuntimeError(f"no checkpoint produced in {run_dir}")
    return ckpt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_vec_env(cfg: TrainConfig, vecnormalize_stats: Path | None = None):
    """Build train VecEnv. Always SubprocVecEnv (see module docstring).

    When ``cfg.normalize_reward`` is set, the stack is
    ``VecNormalize(VecMonitor(SubprocVecEnv(...)))``: VecMonitor sits *inside*
    VecNormalize so ``rollout/ep_rew_mean`` is logged in raw reward units
    (comparable across runs), while PPO sees the normalized reward. Only the
    reward is normalized — ``norm_obs=False`` because images are scaled in the
    CNN forward and the bare eval env would otherwise see a different obs scale.
    ``gamma`` must match PPO's gamma: the return-variance estimate is
    gamma-dependent.

    ``vecnormalize_stats`` (resume only): path to a saved ``.pkl`` whose running
    averages are loaded so the reward scale continues seamlessly instead of
    re-estimating from scratch. The saved object already carries ``gamma`` /
    ``norm_obs`` / ``norm_reward``; ``training`` is forced True so resumed runs
    keep updating the statistics.
    """
    env_fns = [make_env_fn(cfg.env, seed=cfg.seed, rank=i) for i in range(cfg.n_envs)]
    venv = VecMonitor(SubprocVecEnv(env_fns, start_method="spawn"))
    if cfg.normalize_reward:
        if vecnormalize_stats is not None:
            venv = VecNormalize.load(str(vecnormalize_stats), venv)
            venv.training = True
            venv.norm_reward = True
        else:
            venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=cfg.ppo.gamma)
    return venv


def _build_eval_env_factory(cfg: TrainConfig, eval_seed: int):
    """Closure: (record_video: bool) -> single gym.Env for deterministic eval."""

    def _factory(record_video: bool) -> gym.Env:
        return make_env(
            cfg.env,
            seed=eval_seed,
            render_mode="rgb_array" if record_video else None,
        )

    return _factory


__all__ = ["train"]
