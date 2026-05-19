"""stable-retro env factory + ROM import sanity check.

Public entrypoint: :func:`make_retro_env`. Internals here are responsible for
two things:

1. **Failing fast with an actionable error** when the requested game's ROM
   isn't available to stable-retro. For the default target (Airstriker), the
   ROM ships with the stable-retro distribution and this check is a no-op;
   the message remains in place so swapping ``cfg.game`` to a user-supplied
   ROM produces an immediately useful error if the import step was skipped.
2. **Adapting the gym-vs-gymnasium API** via shimmy when stable-retro returns
   the legacy gym Env shape.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym

from retro_rl.env.wrappers import apply_wrappers
from retro_rl.utils.config import EnvConfig

ROM_HELP = (
    "\n"
    "How to fix:\n"
    "  - For games that ship with stable-retro (e.g. Airstriker-Genesis-v0),\n"
    "    reinstall stable-retro — the ROM lives at\n"
    "    <site-packages>/stable_retro/data/stable/<game>/.\n"
    "  - For user-supplied ROMs, place the file under roms/ and run:\n"
    "        python -m retro.import roms/\n"
    "  - Verify with:\n"
    "        python -c \"import retro; print(retro.data.get_romfile_path('{game}'))\"\n"
)


def _check_rom_imported(game: str) -> None:
    """Raise RuntimeError with actionable guidance if the ROM is missing."""
    import retro

    try:
        rom_path = retro.data.get_romfile_path(game)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"stable-retro has no ROM imported for game {game!r}.\n"
            f"Underlying error: {e}" + ROM_HELP.format(game=game)
        ) from e

    if not Path(rom_path).exists():
        raise RuntimeError(
            f"stable-retro reports a ROM path for {game!r} that doesn't exist on disk: "
            f"{rom_path}\n"
            f"Re-run the import step." + ROM_HELP.format(game=game)
        )


def _to_gymnasium(env) -> gym.Env:
    """Wrap a legacy gym.Env in shimmy's compatibility shim if needed.

    stable-retro >= 0.9.2 returns a gymnasium-native env, but we hedge against
    version drift by sniffing the API surface rather than the version string.
    """
    if isinstance(env, gym.Env):
        return env
    try:
        from shimmy import GymV21CompatibilityV0
    except ImportError as e:
        raise RuntimeError(
            "shimmy is required to adapt stable-retro's legacy gym Env to "
            "gymnasium. Install with: pip install 'shimmy>=1.3'"
        ) from e
    return GymV21CompatibilityV0(env=env)


def make_retro_env(
    cfg: EnvConfig,
    seed: int | None = None,
    record_dir: Path | None = None,
    render_mode: str | None = None,
) -> gym.Env:
    """Build a single wrapped stable-retro env.

    Parameters
    ----------
    cfg
        Env configuration. Sets game/state/scenario plus wrapper params.
    seed
        If given, threads through ``env.reset(seed=...)`` and any stochastic
        wrappers.
    record_dir
        If given, stable-retro writes .bk2 replays here. Overrides
        ``cfg.record``.
    render_mode
        Forwarded to ``retro.make``. ``"human"`` opens a viewer window;
        ``"rgb_array"`` returns frames from ``env.render()``; ``None``
        (default) is headless. Training uses ``None``; ``scripts/play_random.py``
        uses ``"human"``.
    """
    import retro

    _check_rom_imported(cfg.game)

    retro_kwargs: dict = {
        "game": cfg.game,
        "state": cfg.state,
        "scenario": cfg.scenario,
    }
    if render_mode is not None:
        retro_kwargs["render_mode"] = render_mode
    if record_dir is not None:
        record_dir = Path(record_dir)
        record_dir.mkdir(parents=True, exist_ok=True)
        retro_kwargs["record"] = str(record_dir)
    elif cfg.record:
        retro_kwargs["record"] = True

    raw_env = retro.make(**retro_kwargs)
    env = _to_gymnasium(raw_env)
    env = apply_wrappers(env, cfg)

    if seed is not None:
        env.reset(seed=seed)
    return env


__all__ = ["make_retro_env"]
