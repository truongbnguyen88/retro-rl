"""Config models + YAML loader.

Pydantic v2 schemas mirror the YAML files in ``configs/``. The loader handles
two composition mechanisms:

* ``extends: <path>`` — inherit from a base YAML, deep-merge with this file
  overriding. Chains are followed recursively.
* ``env_config: <path>`` — string field in train configs that points at a
  separate env YAML; resolved to an :class:`EnvConfig` instance.

Paths in YAML are resolved relative to the repo root (current working
directory). The loader is intentionally minimal — no Jinja, no env-var
interpolation. Add those only if a concrete need shows up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Env / reward schemas
# ---------------------------------------------------------------------------


class RewardConfig(BaseModel):
    """Reward-shaping weights. See ``configs/env.yaml`` for the canonical doc."""

    model_config = ConfigDict(extra="forbid")

    score_delta: float = 1.0
    x_progress: float = 0.01
    x_regress_penalty: float = 0.005
    life_loss: float = -25.0
    death: float = -100.0
    stage_clear: float = 200.0
    survival_bonus: float = 0.0  # additive per-step reward; encourages staying alive
    clip: tuple[float, float] = (-10.0, 10.0)

    @field_validator("clip", mode="before")
    @classmethod
    def _coerce_clip(cls, v: Any) -> tuple[float, float]:
        # YAML lists deserialize to list; pydantic v2 accepts tuples only.
        if isinstance(v, list):
            if len(v) != 2:
                raise ValueError(f"reward.clip must be [lo, hi], got {v!r}")
            return (float(v[0]), float(v[1]))
        return v


class AutoFireConfig(BaseModel):
    """Tap-fire generator for games where holding the fire button fires only
    one bullet on the rising edge (Airstriker and most retro shooters).

    Wraps the raw env so it sees every emulator frame: overrides the fire
    button's bit to follow a periodic 1-on / (period-1)-off pattern. The
    pattern produces a press-release-press sequence the emulator interprets
    as repeated shots, regardless of what the policy emits for that bit.

    Empirically verified for Airstriker: holding B → 0 score in 600 frames;
    1-on / 3-off pattern → consistent score accumulation.
    """

    model_config = ConfigDict(extra="forbid")

    button_index: int = 0  # 0 = B for stable-retro Genesis button order
    period: int = 4  # one press every `period` frames (1 on, period-1 off)

    @field_validator("period")
    @classmethod
    def _period_min(cls, v: int) -> int:
        if v < 2:
            raise ValueError(f"auto_fire period must be >= 2 (1-on, ≥1-off); got {v}")
        return v

    @field_validator("button_index")
    @classmethod
    def _button_index_range(cls, v: int) -> int:
        if not 0 <= v < 12:
            raise ValueError(f"auto_fire button_index must be in [0, 12); got {v}")
        return v


class EnvConfig(BaseModel):
    """Stable-retro env configuration consumed by :func:`retro_rl.env.make_env`."""

    model_config = ConfigDict(extra="forbid")

    # stable-retro identifiers
    game: str = "Airstriker-Genesis-v0"
    state: str = "Level1"
    scenario: str = "scenario"
    record: bool = False

    # Preprocessing
    grayscale: bool = True
    resize: tuple[int, int] = (84, 84)
    frame_stack: int = 4
    action_repeat: int = 4
    sticky_action_prob: float = 0.0
    max_episode_steps: int = 4500

    # Episode boundaries
    end_on_life_lost: bool = True

    # Shaping
    reward: RewardConfig = Field(default_factory=RewardConfig)

    # Per-integration override for stable-retro info dict keys.
    # None → use retro_rl.env.reward_shaping.DEFAULT_INFO_KEYS (generic).
    # Set per game to match the integration's data.json variable names.
    info_keys: dict[str, str] | None = None

    # Optional Discrete(N) action space defined by a list of length-12
    # 0/1 button-combo vectors. None → raw MultiBinary(12). See
    # `DiscreteActionWrapper` for the rationale (Bernoulli threshold
    # problem on the fire button for shooters).
    action_combos: list[list[int]] | None = None

    # Optional frame-level tap-fire generator for the fire button. None →
    # disabled (policy controls fire bit directly). When set, the fire bit
    # is overridden inside the wrapper stack so the policy effectively only
    # learns movement; firing happens at the configured cadence.
    auto_fire: AutoFireConfig | None = None

    @field_validator("action_combos")
    @classmethod
    def _validate_action_combos(cls, v: list[list[int]] | None) -> list[list[int]] | None:
        if v is None:
            return None
        if not v:
            raise ValueError("action_combos must be non-empty if provided")
        for i, combo in enumerate(v):
            if not all(b in (0, 1) for b in combo):
                raise ValueError(f"action_combos[{i}] must be 0/1; got {combo}")
        return v

    @field_validator("resize", mode="before")
    @classmethod
    def _coerce_resize(cls, v: Any) -> tuple[int, int]:
        if isinstance(v, list):
            if len(v) != 2:
                raise ValueError(f"resize must be [H, W], got {v!r}")
            return (int(v[0]), int(v[1]))
        return v

    @field_validator("frame_stack", "action_repeat", "max_episode_steps")
    @classmethod
    def _positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"must be >= 1, got {v}")
        return v

    @field_validator("sticky_action_prob")
    @classmethod
    def _prob(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"sticky_action_prob must be in [0, 1], got {v}")
        return v


# ---------------------------------------------------------------------------
# Training / PPO schemas
# ---------------------------------------------------------------------------


class PPOHyperparams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    learning_rate: float = 2.5e-4
    n_steps: int = 128
    batch_size: int = 256
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.1
    ent_coef: float = 0.01
    # When set, linearly anneal ent_coef from ``ent_coef`` (start) to
    # ``ent_coef_final`` (end) over ``total_timesteps``. None disables the
    # schedule and keeps ent_coef constant.
    ent_coef_final: float | None = None
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True

    @field_validator("ent_coef_final")
    @classmethod
    def _ent_coef_final_nonneg(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError(f"ent_coef_final must be >= 0 if set, got {v}")
        return v


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    every_steps: int = 100_000
    n_episodes: int = 5
    deterministic: bool = True
    record_video: bool = True


class CheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    every_steps: int = 250_000
    keep_last_k: int = 5
    keep_best: bool = True
    keep_top_k: int = 5  # retain top-K-by-return AND top-K-by-length (union, on top of recent-K)


class TrainConfig(BaseModel):
    """Top-level training config (ppo.yaml shape)."""

    model_config = ConfigDict(extra="forbid")

    run_name: str
    seed: int = 42

    # ``env_config`` is a path on disk; ``env`` holds the resolved EnvConfig.
    # We accept either form: if the YAML supplies a path, the loader replaces
    # it with the parsed object and stores the path in ``env_config_path``.
    env: EnvConfig
    env_config_path: Path | None = None

    n_envs: int = 8
    algorithm: str = "ppo"
    policy: str = "cnn"
    total_timesteps: int = 10_000_000

    # Feature-extractor selection. Names are validated against the registry in
    # retro_rl.models.policies (imported lazily in the validator to avoid a
    # config→models import cycle and to keep torch off the config import path).
    # Defaults reproduce v8 (Nature-CNN, 512-dim) so older snapshots that omit
    # these fields validate unchanged.
    features_extractor: str = "nature_cnn"
    features_dim: int = 512

    # Wrap the train VecEnv in SB3 VecNormalize with norm_reward=True (returns
    # rescaled to ~unit variance; observations are NOT normalized — images are
    # scaled in the CNN forward). Shrinks the value-target scale so the value
    # head fits faster (higher explained_variance, lower value_loss). Eval is
    # unaffected: it runs on a bare env and reports raw returns. Default False
    # reproduces v8/v9 behaviour and keeps older snapshots valid.
    normalize_reward: bool = False

    ppo: PPOHyperparams = Field(default_factory=PPOHyperparams)

    log_dir: Path = Path("outputs/tensorboard")
    checkpoint_dir: Path = Path("outputs/checkpoints")
    video_dir: Path = Path("outputs/videos")
    log_interval: int = 10

    eval: EvalConfig = Field(default_factory=EvalConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

    @field_validator("features_extractor")
    @classmethod
    def _known_extractor(cls, v: str) -> str:
        # Lazy import: retro_rl.models.policies pulls in torch, which we keep
        # off the config import path. The registry is the single source of
        # truth for valid names.
        from retro_rl.models.policies import FEATURE_EXTRACTORS

        if v not in FEATURE_EXTRACTORS:
            raise ValueError(
                f"features_extractor must be one of {sorted(FEATURE_EXTRACTORS)}; got {v!r}"
            )
        return v

    @field_validator("features_dim")
    @classmethod
    def _features_dim_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"features_dim must be >= 1, got {v}")
        return v


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base``. Override wins on leaves."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml_with_extends(path: Path, _seen: set[Path] | None = None) -> dict:
    """Load YAML and resolve any ``extends:`` chain.

    Cycle-safe via ``_seen``. ``extends`` paths are repo-root-relative.
    """
    path = path.resolve()
    seen = _seen or set()
    if path in seen:
        raise ValueError(f"cycle in extends chain at {path}")
    seen = seen | {path}

    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"top-level YAML at {path} must be a mapping")

    extends = data.pop("extends", None)
    if extends is None:
        return data

    base_path = Path(extends)
    if not base_path.is_absolute():
        base_path = Path.cwd() / base_path
    base = _load_yaml_with_extends(base_path, seen)
    return _deep_merge(base, data)


def load_env_config(path: str | Path) -> EnvConfig:
    """Load an env YAML into an :class:`EnvConfig`."""
    data = _load_yaml_with_extends(Path(path))
    return EnvConfig.model_validate(data)


def load_train_config(path: str | Path) -> TrainConfig:
    """Load a training YAML into a :class:`TrainConfig`.

    Resolves the ``env_config`` field (a string path) into a nested
    :class:`EnvConfig` before validation.
    """
    data = _load_yaml_with_extends(Path(path))

    env_path_raw = data.pop("env_config", None)
    if env_path_raw is None:
        raise ValueError(f"train config at {path} is missing required field 'env_config'")
    env_path = Path(env_path_raw)
    if not env_path.is_absolute():
        env_path = Path.cwd() / env_path

    data["env"] = load_env_config(env_path).model_dump()
    data["env_config_path"] = env_path

    return TrainConfig.model_validate(data)


__all__ = [
    "RewardConfig",
    "AutoFireConfig",
    "EnvConfig",
    "PPOHyperparams",
    "EvalConfig",
    "CheckpointConfig",
    "TrainConfig",
    "load_env_config",
    "load_train_config",
]
