"""retro_rl.env — env factory + wrappers + reward shaping.

Public surface:

* :func:`make_env` — top-level factory, returns a wrapped single env.
* :func:`make_env_fn` — returns a *thunk* (no-arg callable) for use with
  SB3's ``SubprocVecEnv`` / ``DummyVecEnv``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import gymnasium as gym

from retro_rl.utils.config import EnvConfig


def make_env(
    cfg: EnvConfig,
    seed: int | None = None,
    record_dir: Path | None = None,
    render_mode: str | None = None,
) -> gym.Env:
    """Build a single fully-wrapped stable-retro env.

    ``render_mode='human'`` opens a viewer window (for ``scripts/play_random.py``);
    ``None`` (default) is headless.
    """
    # Lazy import: retro_env pulls in wrappers.py → cv2 at module level.
    # Deferring to call time keeps `import retro_rl.env` lightweight so
    # pytest collection and non-emulator tests don't require a working cv2/retro.
    from retro_rl.env.retro_env import make_retro_env

    return make_retro_env(cfg, seed=seed, record_dir=record_dir, render_mode=render_mode)


def make_env_fn(
    cfg: EnvConfig,
    seed: int | None = None,
    rank: int = 0,
    record_dir: Path | None = None,
) -> Callable[[], gym.Env]:
    """Return a thunk that builds an env. Used by SB3 vec env constructors.

    ``rank`` is added to ``seed`` to give each worker a distinct stream while
    keeping the full set of seeds derivable from one base seed.
    """
    from retro_rl.env.retro_env import make_retro_env

    worker_seed = None if seed is None else seed + rank

    def _thunk() -> gym.Env:
        return make_retro_env(cfg, seed=worker_seed, record_dir=record_dir)

    return _thunk


__all__ = ["make_env", "make_env_fn"]
