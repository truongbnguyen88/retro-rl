"""Milestone-1 env layer tests.

ROM-independent tests use a tiny fake gym env that emits random uint8 frames
and a configurable info dict — enough surface area to exercise every wrapper.
The ROM-gated smoke test skips automatically when stable-retro can't find the
imported Contra ROM.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from contra_rl.env.reward_shaping import ShapingState, shape_reward
from contra_rl.env.wrappers import (
    ActionRepeat,
    EndOnLifeLost,
    FrameStack,
    GrayscaleResize,
    RewardShapingWrapper,
    StickyAction,
    apply_wrappers,
)
from contra_rl.utils.config import EnvConfig, RewardConfig, load_env_config
from contra_rl.utils.seeding import set_global_seed


# ---------------------------------------------------------------------------
# Fake env — stand-in for stable-retro
# ---------------------------------------------------------------------------


class FakeContraEnv(gym.Env):
    """Tiny env: 240x256x3 uint8 obs, 8 discrete actions, scripted info."""

    metadata = {"render_modes": []}

    def __init__(self, episode_len: int = 50, lives_start: int = 3):
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(240, 256, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(8)
        self._t = 0
        self._episode_len = episode_len
        self._lives_start = lives_start
        self._lives = lives_start
        self._score = 0
        self._x = 0
        self._rng = np.random.default_rng(0)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._lives = self._lives_start
        self._score = 0
        self._x = 0
        obs = self._rng.integers(0, 256, size=(240, 256, 3), dtype=np.uint8)
        return obs, self._info()

    def step(self, action):
        self._t += 1
        self._score += int(action)  # deterministic-ish score growth
        self._x += 1
        if self._t == self._episode_len // 2:
            self._lives -= 1
        terminated = self._t >= self._episode_len
        obs = self._rng.integers(0, 256, size=(240, 256, 3), dtype=np.uint8)
        return obs, 0.0, terminated, False, self._info()

    def _info(self) -> dict:
        return {"score": self._score, "xpos": self._x, "lives": self._lives}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_load_env_config_from_repo_yaml():
    cfg = load_env_config(Path("configs/env.yaml"))
    assert cfg.game == "Contra-Nes"
    assert cfg.resize == (84, 84)
    assert cfg.frame_stack == 4
    assert cfg.action_repeat == 4
    assert cfg.reward.clip == (-10.0, 10.0)


def test_env_config_validation_rejects_unknown_field():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        EnvConfig.model_validate({"game": "X", "bogus": 1})


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def test_set_global_seed_is_deterministic():
    import torch

    set_global_seed(123)
    a_np = np.random.rand(5)
    a_t = torch.randn(5)

    set_global_seed(123)
    b_np = np.random.rand(5)
    b_t = torch.randn(5)

    np.testing.assert_array_equal(a_np, b_np)
    assert torch.equal(a_t, b_t)


# ---------------------------------------------------------------------------
# Reward shaping (pure function)
# ---------------------------------------------------------------------------


def test_shape_reward_first_step_yields_zero_for_deltas():
    cfg = RewardConfig()
    state = ShapingState()
    r = shape_reward({"score": 100, "xpos": 10, "lives": 3}, state, cfg)
    # No prior frame → no deltas → r == 0.
    assert r == 0.0
    assert state.prev_score == 100


def test_shape_reward_score_delta_and_x_progress():
    # Widen clip so the 10 + 2 total isn't truncated.
    cfg = RewardConfig(
        score_delta=1.0,
        x_progress=0.5,
        x_regress_penalty=0.1,
        clip=(-1000.0, 1000.0),
    )
    state = ShapingState()
    shape_reward({"score": 0, "xpos": 0, "lives": 3}, state, cfg)
    r = shape_reward({"score": 10, "xpos": 4, "lives": 3}, state, cfg)
    # score delta = 10 * 1.0; x progress = 4 * 0.5 = 2.0
    assert r == pytest.approx(10 + 2.0)


def test_shape_reward_life_loss_and_death():
    cfg = RewardConfig(life_loss=-25.0, death=-100.0, clip=(-1000.0, 1000.0))
    state = ShapingState()
    shape_reward({"score": 0, "xpos": 0, "lives": 3}, state, cfg)
    r = shape_reward({"score": 0, "xpos": 0, "lives": 2}, state, cfg, terminated=True)
    assert r == pytest.approx(-25.0 + -100.0)


def test_shape_reward_clips():
    cfg = RewardConfig(score_delta=1.0, clip=(-1.0, 1.0))
    state = ShapingState()
    shape_reward({"score": 0, "xpos": 0, "lives": 3}, state, cfg)
    r = shape_reward({"score": 1_000_000, "xpos": 0, "lives": 3}, state, cfg)
    assert r == 1.0


def test_shape_reward_missing_key_does_not_raise():
    cfg = RewardConfig()
    state = ShapingState()
    # Drop the 'xpos' key entirely.
    r = shape_reward({"score": 0, "lives": 3}, state, cfg)
    assert r == 0.0


# ---------------------------------------------------------------------------
# Wrappers — shape/dtype, behavior
# ---------------------------------------------------------------------------


def test_action_repeat_sums_reward_and_maxpools():
    env = ActionRepeat(FakeContraEnv(episode_len=100), skip=4)
    obs, _ = env.reset(seed=0)
    next_obs, reward, term, trunc, info = env.step(3)
    # Fake env returns 0 reward per step; sum still 0.0.
    assert reward == 0.0
    assert next_obs.shape == (240, 256, 3)
    assert next_obs.dtype == np.uint8


def test_grayscale_resize_observation_space_and_dtype():
    env = GrayscaleResize(FakeContraEnv(), size=(84, 84))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (84, 84, 1)
    assert obs.dtype == np.uint8
    assert env.observation_space.shape == (84, 84, 1)


def test_frame_stack_shape():
    env = GrayscaleResize(FakeContraEnv(), size=(84, 84))
    env = FrameStack(env, n=4)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (84, 84, 4)
    # Step once — still 4 stacked.
    obs2, _, _, _, _ = env.step(0)
    assert obs2.shape == (84, 84, 4)


def test_end_on_life_lost_truncates_episode():
    # FakeContraEnv decrements lives at t == episode_len // 2.
    env = EndOnLifeLost(FakeContraEnv(episode_len=10))
    env.reset(seed=0)
    terminated = False
    for _ in range(20):
        _, _, terminated, _, _ = env.step(0)
        if terminated:
            break
    assert terminated


def test_sticky_action_repeats_with_prob_1():
    env = StickyAction(FakeContraEnv(), p=1.0)
    env.reset(seed=0)
    # First step records action; subsequent steps should always re-use it
    # regardless of what we pass — observable via FakeContraEnv's score delta
    # (== action). After the first step, score should grow by the first action
    # each step.
    _, _, _, _, info0 = env.step(7)
    _, _, _, _, info1 = env.step(0)
    _, _, _, _, info2 = env.step(0)
    assert info1["score"] - info0["score"] == 7
    assert info2["score"] - info1["score"] == 7


def test_reward_shaping_wrapper_adds_to_native_reward():
    cfg = RewardConfig(
        score_delta=2.0,
        x_progress=0.0,
        life_loss=0.0,
        death=0.0,
        clip=(-1000.0, 1000.0),
    )
    env = RewardShapingWrapper(FakeContraEnv(), cfg)
    env.reset(seed=0)
    # First step seeds prev_* state; shaped reward is 0 because there's no
    # prior frame to diff against. Step again to observe the actual delta.
    env.step(5)  # seed; score now 5
    _, reward, _, _, info = env.step(5)  # score now 10, delta=5 → shaped 2.0*5
    assert info["shaped_reward"] == pytest.approx(10.0)
    assert reward == pytest.approx(10.0)  # native reward is 0


def test_apply_wrappers_end_to_end_shape():
    cfg = EnvConfig(
        game="dummy",
        state="dummy",
        action_repeat=2,
        frame_stack=4,
        resize=(84, 84),
        max_episode_steps=20,
        end_on_life_lost=False,
    )
    env = apply_wrappers(FakeContraEnv(episode_len=1000), cfg)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (84, 84, 4)
    assert obs.dtype == np.uint8


def test_apply_wrappers_seeded_determinism():
    cfg = EnvConfig(
        game="dummy",
        state="dummy",
        action_repeat=2,
        frame_stack=4,
        resize=(84, 84),
        max_episode_steps=50,
        end_on_life_lost=False,
        sticky_action_prob=0.0,
    )
    env_a = apply_wrappers(FakeContraEnv(episode_len=1000), cfg)
    env_b = apply_wrappers(FakeContraEnv(episode_len=1000), cfg)
    obs_a, _ = env_a.reset(seed=42)
    obs_b, _ = env_b.reset(seed=42)
    np.testing.assert_array_equal(obs_a, obs_b)
    for action in [1, 2, 3, 4, 5]:
        oa, ra, _, _, _ = env_a.step(action)
        ob, rb, _, _, _ = env_b.step(action)
        np.testing.assert_array_equal(oa, ob)
        assert ra == rb


# ---------------------------------------------------------------------------
# Smoke tests against real stable-retro envs
# ---------------------------------------------------------------------------


def _rom_available(game: str) -> bool:
    try:
        import retro

        retro.data.get_romfile_path(game)
        return True
    except Exception:
        return False


def test_make_env_airstriker_smoke():
    """End-to-end smoke against Airstriker — the ROM ships with stable-retro,
    so this runs unconditionally on any clean install."""
    retro = pytest.importorskip("retro")
    if not _rom_available("Airstriker-Genesis-v0"):
        pytest.skip("Airstriker-Genesis-v0 ROM not found in stable-retro data dir")

    from contra_rl.env import make_env

    cfg = load_env_config(Path("configs/env-airstriker.yaml"))
    env = make_env(cfg, seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape[-1] == cfg.frame_stack  # channels-last stack
    assert obs.dtype == np.uint8
    assert obs.shape[:2] == cfg.resize
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    # env.close() triggers pyglet's Cocoa event-loop teardown which has a
    # known AttributeError in pyglet 1.5.x on macOS. The env contents we care
    # about (factory, wrappers, stepping) already validated above; the leaked
    # subprocess is reaped at pytest teardown anyway.
    try:
        env.close()
    except AttributeError:
        pass


@pytest.mark.rom
@pytest.mark.skipif(not _rom_available("Contra-Nes"), reason="Contra ROM not imported")
def test_make_env_contra_smoke():
    """Same shape as the Airstriker smoke — runs only if the user has dumped
    and imported a Contra ROM. Validates the production target."""
    from contra_rl.env import make_env

    cfg = load_env_config(Path("configs/env.yaml"))
    env = make_env(cfg, seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape[-1] == cfg.frame_stack
    assert obs.dtype == np.uint8
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    try:
        env.close()
    except AttributeError:
        pass  # pyglet 1.5 Cocoa teardown bug on macOS; see Airstriker test
