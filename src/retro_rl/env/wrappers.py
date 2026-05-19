"""Gymnasium wrappers for the stable-retro env.

Composition order (outermost last) matches Atari-DQN convention:

    base env (MultiBinary(12) — raw Genesis buttons)
    └─ AutoFireWrapper         (optional, frame-level tap-fire on fire bit)
       └─ DiscreteActionWrapper (optional, Discrete(N) → MultiBinary combo)
          └─ ActionRepeat          (frame-skip k, max-pool last 2 frames)
             └─ StickyAction       (optional, sticky prob)
                └─ EndOnLifeLost   (optional, ends ep on life decrement)
                   └─ RewardShaping (additive, info-based)
                      └─ GrayscaleResize (84x84 uint8 single-channel)
                         └─ FrameStack    (channel-first stack of last n)
                            └─ TimeLimit  (max_episode_steps)

We use gymnasium's built-in ``TimeLimit`` rather than rolling our own.

Action repeat returns the element-wise max of the last two raw frames before
downstream visual wrappers — this suppresses NES flicker artifacts where
sprites alternate visibility across frames.

DiscreteActionWrapper sits just above AutoFireWrapper so the categorical
combos are translated to MultiBinary once; AutoFireWrapper then overrides
the fire bit on a per-emulator-frame basis (it sees every frame, even those
inside an ActionRepeat skip loop).
"""

from __future__ import annotations

from collections import deque

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from retro_rl.env.reward_shaping import DEFAULT_INFO_KEYS, ShapingState, shape_reward
from retro_rl.utils.config import AutoFireConfig, EnvConfig, RewardConfig

# ---------------------------------------------------------------------------
# Action space wrappers
# ---------------------------------------------------------------------------


class DiscreteActionWrapper(gym.ActionWrapper):
    """Map ``Discrete(N)`` actions to a fixed ``MultiBinary`` button-combo table.

    Retro Genesis envs expose ``MultiBinary(12)`` — each of 12 buttons as an
    independent bit. SB3's PPO models this as 12 Bernoulli outputs, so the
    deterministic policy fires button *i* only when ``P(button_i=1) > 0.5``.
    For sparse-reward shooters like Airstriker, the fire button's probability
    typically converges below 0.5 (delayed kill rewards yield a weaker
    gradient than immediate per-step survival rewards), leaving the
    deterministic policy unable to commit to firing even when the stochastic
    policy fires often. Empirically observed in v2 training: stochastic
    rollout return 1174 vs deterministic eval frozen at -35.75 for 700K
    consecutive steps.

    Replacing the raw button space with a curated ``Discrete(N)`` over a
    handful of meaningful combos turns the policy head into a single
    categorical distribution. The deterministic policy then takes ``argmax``
    over combos directly — fire is selected or not based on its Q-value, no
    probability threshold required.

    Parameters
    ----------
    env
        Inner env whose ``action_space`` is ``MultiBinary``.
    combos
        List of length-``n_buttons`` 0/1 vectors. Length determines ``N``;
        ``combos[i]`` is the button vector emitted when the agent picks action
        ``i``.
    """

    def __init__(self, env: gym.Env, combos: list[list[int]]):
        super().__init__(env)
        inner = env.action_space
        if not isinstance(inner, spaces.MultiBinary):
            raise TypeError(
                f"DiscreteActionWrapper expects MultiBinary inner space; got {type(inner).__name__}"
            )
        if not combos:
            raise ValueError("combos must be non-empty")
        n_buttons = int(inner.n)
        for i, combo in enumerate(combos):
            if len(combo) != n_buttons:
                raise ValueError(f"combo {i} has length {len(combo)}; expected {n_buttons}")
            if any(b not in (0, 1) for b in combo):
                raise ValueError(f"combo {i} must be all 0/1; got {combo}")
        self._combos = np.array(combos, dtype=np.int8)
        self.action_space = spaces.Discrete(len(combos))

    def action(self, action) -> np.ndarray:
        return self._combos[int(action)].copy()


class AutoFireWrapper(gym.Wrapper):
    """Inject a tap-fire pattern on the fire button at the emulator-frame level.

    Many retro shooters — Airstriker among them — fire only on the *rising
    edge* of the fire button. Holding the button continuously fires one
    bullet on press and then nothing for the rest of the episode. v3/v4
    training failed for exactly this reason: every action_combo had ``B=1``,
    so the agent held B forever and emitted one bullet per life.

    This wrapper overrides the fire-button bit of every action passed to the
    inner env, following a periodic 1-on / (period-1)-off pattern. Because
    it wraps the *raw* env (innermost in our stack), it sees every emulator
    frame, including the ones inside an ``ActionRepeat`` skip loop — so the
    tap rate is independent of action_repeat.

    The fire bit emitted by upstream wrappers (or the policy) is ignored
    entirely: firing is taken out of the policy's action set and turned
    into a fixed cadence.
    """

    def __init__(self, env: gym.Env, cfg: AutoFireConfig):
        super().__init__(env)
        if not isinstance(env.action_space, spaces.MultiBinary):
            raise TypeError(
                f"AutoFireWrapper expects MultiBinary inner space; got {type(env.action_space).__name__}"
            )
        n = int(env.action_space.n)
        if not 0 <= cfg.button_index < n:
            raise ValueError(f"button_index {cfg.button_index} out of range for MultiBinary({n})")
        self._fire_idx = cfg.button_index
        self._period = cfg.period
        self._counter = 0

    def reset(self, *, seed: int | None = None, options=None):
        self._counter = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        # Copy so we don't mutate the caller's buffer (matters when the
        # outer ActionRepeat hands us the same numpy view repeatedly).
        action = np.array(action, dtype=np.int8, copy=True)
        action[self._fire_idx] = 1 if (self._counter % self._period) == 0 else 0
        self._counter += 1
        return self.env.step(action)


# ---------------------------------------------------------------------------
# Temporal wrappers
# ---------------------------------------------------------------------------


class ActionRepeat(gym.Wrapper):
    """Repeat each action ``skip`` times; return the max-pooled last 2 frames.

    Reward is summed; ``terminated``/``truncated`` short-circuit the loop.
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        if skip < 1:
            raise ValueError(f"action_repeat must be >= 1, got {skip}")
        self._skip = skip
        # Buffer for the last two raw observations.
        self._obs_buf = np.zeros(
            (2, *env.observation_space.shape), dtype=env.observation_space.dtype
        )

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        info: dict = {}
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buf[0] = obs
            if i == self._skip - 1:
                self._obs_buf[1] = obs
            total_reward += float(reward)
            if terminated or truncated:
                break
        max_frame = self._obs_buf.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


class StickyAction(gym.Wrapper):
    """With probability ``p``, repeat the previous action instead of the new one."""

    def __init__(self, env: gym.Env, p: float = 0.0):
        super().__init__(env)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"sticky prob must be in [0, 1], got {p}")
        self._p = p
        self._last_action = None
        self._rng = np.random.default_rng()

    def reset(self, *, seed: int | None = None, options=None):
        self._last_action = None
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        if self._last_action is not None and self._rng.random() < self._p:
            action = self._last_action
        self._last_action = action
        return self.env.step(action)


class EndOnLifeLost(gym.Wrapper):
    """Terminate the episode on the first life decrement.

    The underlying retro env keeps running until all lives are exhausted; we
    surface a terminal signal early to tighten credit assignment. We do *not*
    reset the underlying env here — SB3's vec env handles that on the next
    ``reset()`` call.
    """

    def __init__(self, env: gym.Env, lives_key: str = "lives"):
        super().__init__(env)
        self._key = lives_key
        self._prev_lives: int | None = None

    def reset(self, *, seed: int | None = None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_lives = info.get(self._key)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        lives = info.get(self._key)
        if lives is not None and self._prev_lives is not None and lives < self._prev_lives:
            terminated = True
        self._prev_lives = lives
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------


class RewardShapingWrapper(gym.Wrapper):
    """Add config-driven shaping on top of the native env reward."""

    def __init__(
        self,
        env: gym.Env,
        cfg: RewardConfig,
        info_keys: dict[str, str] | None = None,
    ):
        super().__init__(env)
        self._cfg = cfg
        self._info_keys = info_keys or DEFAULT_INFO_KEYS
        self._state = ShapingState()

    def reset(self, *, seed: int | None = None, options=None):
        self._state.reset()
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = shape_reward(
            info,
            self._state,
            self._cfg,
            terminated=terminated,
            info_keys=self._info_keys,
        )
        info = dict(info)  # avoid mutating shared dict
        info["native_reward"] = float(reward)
        info["shaped_reward"] = float(shaped)
        info["cumulative_shaped"] = self._state.cumulative_shaped
        return obs, float(reward) + shaped, terminated, truncated, info


# ---------------------------------------------------------------------------
# Visual wrappers
# ---------------------------------------------------------------------------


class GrayscaleResize(gym.ObservationWrapper):
    """RGB (H,W,3) uint8 → grayscale resized (H', W', 1) uint8."""

    def __init__(self, env: gym.Env, size: tuple[int, int] = (84, 84)):
        super().__init__(env)
        h, w = size
        self._size = (w, h)  # cv2 uses (W, H)
        self.observation_space = spaces.Box(low=0, high=255, shape=(h, w, 1), dtype=np.uint8)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs.squeeze()
        resized = cv2.resize(gray, self._size, interpolation=cv2.INTER_AREA)
        return resized[:, :, None].astype(np.uint8)


class FrameStack(gym.Wrapper):
    """Stack the last ``n`` frames along the channel axis.

    Output shape: ``(H, W, C * n)`` to match SB3's CNN expectations when
    ``channels_first=False``. We keep channels-last; SB3's policy transposes
    internally when needed.
    """

    def __init__(self, env: gym.Env, n: int = 4):
        super().__init__(env)
        if n < 1:
            raise ValueError(f"frame_stack must be >= 1, got {n}")
        self._n = n
        self._frames: deque[np.ndarray] = deque(maxlen=n)

        low = env.observation_space.low
        high = env.observation_space.high
        h, w, c = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=np.repeat(low, n, axis=-1) if low.ndim == 3 else 0,
            high=np.repeat(high, n, axis=-1) if high.ndim == 3 else 255,
            shape=(h, w, c * n),
            dtype=env.observation_space.dtype,
        )

    def reset(self, *, seed: int | None = None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(self._n):
            self._frames.append(obs)
        return self._stack(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._stack(), reward, terminated, truncated, info

    def _stack(self) -> np.ndarray:
        return np.concatenate(list(self._frames), axis=-1)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def apply_wrappers(env: gym.Env, cfg: EnvConfig) -> gym.Env:
    """Apply the standard wrapper stack as documented at module top."""
    info_keys = cfg.info_keys or DEFAULT_INFO_KEYS
    if cfg.auto_fire is not None:
        env = AutoFireWrapper(env, cfg.auto_fire)
    if cfg.action_combos:
        env = DiscreteActionWrapper(env, cfg.action_combos)
    env = ActionRepeat(env, skip=cfg.action_repeat)
    if cfg.sticky_action_prob > 0.0:
        env = StickyAction(env, p=cfg.sticky_action_prob)
    if cfg.end_on_life_lost:
        env = EndOnLifeLost(env, lives_key=info_keys.get("lives", "lives"))
    env = RewardShapingWrapper(env, cfg.reward, info_keys=cfg.info_keys)
    if cfg.grayscale:
        env = GrayscaleResize(env, size=cfg.resize)
    if cfg.frame_stack > 1:
        env = FrameStack(env, n=cfg.frame_stack)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=cfg.max_episode_steps)
    return env


__all__ = [
    "DiscreteActionWrapper",
    "AutoFireWrapper",
    "ActionRepeat",
    "StickyAction",
    "EndOnLifeLost",
    "RewardShapingWrapper",
    "GrayscaleResize",
    "FrameStack",
    "apply_wrappers",
]
