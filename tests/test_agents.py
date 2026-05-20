"""Milestone-2 agent tests.

Covers the :class:`RandomAgent` baseline and the PPO factory. We avoid running
``model.learn`` — constructing PPO already builds the policy network, asserts
SB3 accepts our ``policy_kwargs``, and lets us exercise predict + save/load.

The stub env produces the *wrapped* observation shape (channels-last, uint8)
that the env layer normally emits. SB3 detects this as an image space and
auto-wraps the VecEnv with :class:`VecTransposeImage` — same code path as a
real training run, minus the stable-retro dependency.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from retro_rl.agents.base import Agent
from retro_rl.agents.ppo import build_ppo, linear_schedule
from retro_rl.agents.random_agent import RandomAgent
from retro_rl.utils.config import EnvConfig, PPOHyperparams, TrainConfig

# ---------------------------------------------------------------------------
# Stub env — post-wrapper shape, no stable-retro
# ---------------------------------------------------------------------------


class StubWrappedEnv(gym.Env):
    """Emits (84, 84, 4) uint8 obs and Discrete(8) actions."""

    metadata = {"render_modes": []}

    def __init__(self, episode_len: int = 32):
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8)
        self.action_space = spaces.Discrete(8)
        self._rng = np.random.default_rng(0)
        self._t = 0
        self._episode_len = episode_len

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        terminated = self._t >= self._episode_len
        return self._obs(), 0.0, terminated, False, {}

    def _obs(self) -> np.ndarray:
        return self._rng.integers(0, 256, size=(84, 84, 4), dtype=np.uint8)


def _make_vec_env(n: int = 2) -> DummyVecEnv:
    return DummyVecEnv([lambda: StubWrappedEnv() for _ in range(n)])


def _train_cfg(tmp_path: Path) -> TrainConfig:
    env_cfg = EnvConfig(game="stub", state="stub")
    # n_steps tiny so any future learn() call in this test file is cheap.
    return TrainConfig(
        run_name="test",
        seed=7,
        env=env_cfg,
        n_envs=2,
        total_timesteps=64,
        ppo=PPOHyperparams(n_steps=16, batch_size=16, n_epochs=1),
        log_dir=tmp_path / "tb",
        checkpoint_dir=tmp_path / "ckpt",
        video_dir=tmp_path / "videos",
    )


# ---------------------------------------------------------------------------
# Linear schedule
# ---------------------------------------------------------------------------


def test_linear_schedule_anneals_to_zero():
    sched = linear_schedule(3.0)
    assert sched(1.0) == pytest.approx(3.0)
    assert sched(0.5) == pytest.approx(1.5)
    assert sched(0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# RandomAgent
# ---------------------------------------------------------------------------


def test_random_agent_predict_shape_discrete():
    agent = RandomAgent(spaces.Discrete(8), seed=0)
    obs = np.zeros((84, 84, 4), dtype=np.uint8)
    action, state = agent.predict(obs)
    assert state is None
    assert 0 <= int(action) < 8


def test_random_agent_predict_batched():
    agent = RandomAgent(spaces.Discrete(8), seed=0)
    obs = np.zeros((4, 84, 84, 4), dtype=np.uint8)
    action, _ = agent.predict(obs)
    assert action.shape == (4,)
    assert action.dtype == np.int64


def test_random_agent_seeded_reproducible():
    a = RandomAgent(spaces.Discrete(8), seed=42)
    b = RandomAgent(spaces.Discrete(8), seed=42)
    obs = np.zeros((84, 84, 4), dtype=np.uint8)
    for _ in range(10):
        assert int(a.predict(obs)[0]) == int(b.predict(obs)[0])


def test_random_agent_save_load_roundtrip(tmp_path: Path):
    agent = RandomAgent(spaces.Discrete(8), seed=123)
    path = tmp_path / "random.json"
    agent.save(path)
    loaded = RandomAgent.load(path)
    assert isinstance(loaded.action_space, spaces.Discrete)
    assert int(loaded.action_space.n) == 8
    # Reproducibility carries through save/load.
    obs = np.zeros((84, 84, 4), dtype=np.uint8)
    fresh = RandomAgent(spaces.Discrete(8), seed=123)
    assert int(loaded.predict(obs)[0]) == int(fresh.predict(obs)[0])


def test_random_agent_satisfies_agent_protocol():
    agent = RandomAgent(spaces.Discrete(8))
    assert isinstance(agent, Agent)


# ---------------------------------------------------------------------------
# PPO factory
# ---------------------------------------------------------------------------


def test_build_ppo_constructs_model(tmp_path: Path):
    vec_env = _make_vec_env()
    cfg = _train_cfg(tmp_path)
    model = build_ppo(vec_env, cfg, tb_log_path=cfg.log_dir)
    assert isinstance(model, PPO)
    assert model.n_steps == cfg.ppo.n_steps
    assert model.n_epochs == cfg.ppo.n_epochs
    assert model.seed == cfg.seed
    vec_env.close()


def test_build_ppo_uses_retro_cnn(tmp_path: Path):
    from retro_rl.models.cnn import RetroCNN

    vec_env = _make_vec_env()
    cfg = _train_cfg(tmp_path)
    model = build_ppo(vec_env, cfg, features_dim=256)
    assert isinstance(model.policy.features_extractor, RetroCNN)
    assert model.policy.features_extractor.features_dim == 256
    vec_env.close()


def test_build_ppo_uses_impala_from_config(tmp_path: Path):
    """v9 path: cfg.features_extractor='impala' wires ImpalaCNN end-to-end."""
    from retro_rl.models.impala import ImpalaCNN

    vec_env = _make_vec_env()
    cfg = _train_cfg(tmp_path)
    cfg.features_extractor = "impala"
    cfg.features_dim = 256
    model = build_ppo(vec_env, cfg, tb_log_path=cfg.log_dir)
    assert isinstance(model.policy.features_extractor, ImpalaCNN)
    assert model.policy.features_extractor.features_dim == 256
    vec_env.close()


def test_build_ppo_rejects_unknown_policy(tmp_path: Path):
    vec_env = _make_vec_env()
    cfg = _train_cfg(tmp_path)
    cfg.policy = "mlp"  # not wired
    with pytest.raises(ValueError, match="cnn"):
        build_ppo(vec_env, cfg)
    vec_env.close()


def test_ppo_predict_returns_valid_action(tmp_path: Path):
    vec_env = _make_vec_env(n=1)
    cfg = _train_cfg(tmp_path)
    model = build_ppo(vec_env, cfg)
    obs = vec_env.reset()
    action, _ = model.predict(obs, deterministic=True)
    # After VecTransposeImage SB3 still returns batch-shaped actions.
    assert action.shape == (1,)
    assert 0 <= int(action[0]) < 8
    vec_env.close()


def test_ppo_save_load_roundtrip(tmp_path: Path):
    vec_env = _make_vec_env(n=1)
    cfg = _train_cfg(tmp_path)
    model = build_ppo(vec_env, cfg)
    save_path = tmp_path / "ppo_model"
    model.save(save_path)

    loaded = PPO.load(save_path, env=vec_env)
    # Same weights ⇒ same deterministic action for same obs.
    obs = vec_env.reset()
    a_orig, _ = model.predict(obs, deterministic=True)
    a_loaded, _ = loaded.predict(obs, deterministic=True)
    np.testing.assert_array_equal(a_orig, a_loaded)
    vec_env.close()
