"""Reward shaping — pure function over the stable-retro ``info`` dict.

Design
------
The function is intentionally state-free: callers thread a :class:`ShapingState`
through successive calls. This makes the shaping trivially unit-testable and
keeps the wrapper layer (which owns env state) thin.

Stable-retro exposes RAM-derived variables through ``info`` based on the
integration's ``data.json``. ``DEFAULT_INFO_KEYS`` below is a sensible
generic mapping; per-game overrides come in through ``EnvConfig.info_keys``
(see ``configs/env.yaml`` for the Airstriker mapping).

Missing keys
------------
A missing key is treated as "value unchanged" — the corresponding shaping
term contributes zero. We log a single WARNING per missing key per process so
divergence between expected integration vars and reality is surfaced loudly
on the first step but never spams the log.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

from retro_rl.utils.config import RewardConfig

logger = logging.getLogger(__name__)

# Generic default info-key mapping. Most stable-retro integrations expose
# ``score`` and ``lives`` under those names; ``x_pos`` and ``stage_clear`` are
# game-specific and treated as "missing → 0 contribution" when absent.
# Per-game mappings live in ``configs/<game>.yaml`` under ``info_keys:``.
DEFAULT_INFO_KEYS = {
    "score": "score",
    "x_pos": "xpos",
    "lives": "lives",
    "stage_clear": "stage_clear",  # 0/1 flag, may not exist by default
}


@dataclass
class ShapingState:
    """Per-episode shaping state. Reset on env reset."""

    prev_score: int | None = None
    prev_x: int | None = None
    prev_lives: int | None = None
    cumulative_shaped: float = 0.0
    _warned_missing: set[str] = field(default_factory=set)

    def reset(self) -> None:
        self.prev_score = None
        self.prev_x = None
        self.prev_lives = None
        self.cumulative_shaped = 0.0
        # Keep _warned_missing across episodes — one warning per process.


def shape_reward(
    info: Mapping[str, float],
    state: ShapingState,
    cfg: RewardConfig,
    *,
    terminated: bool = False,
    info_keys: Mapping[str, str] = DEFAULT_INFO_KEYS,
) -> float:
    """Compute shaped reward for a single step.

    Mutates ``state`` to track deltas across calls. Returns the shaped reward
    *before* combining with the env's native reward — the caller decides
    whether to add or replace.

    Parameters
    ----------
    info
        ``info`` dict from ``env.step``.
    state
        Per-episode :class:`ShapingState`; mutated in place.
    cfg
        :class:`RewardConfig` weights.
    terminated
        True if this step ended the episode via game-over (not truncation).
        Triggers the ``death`` term.
    info_keys
        Map of semantic name → ``info`` dict key. Override per-integration.

    Returns
    -------
    float
        Shaped reward, clipped to ``cfg.clip``.
    """
    r = 0.0

    score = _get(info, info_keys["score"], state)
    x_pos = _get(info, info_keys["x_pos"], state)
    lives = _get(info, info_keys["lives"], state)
    stage_clear = _get(info, info_keys["stage_clear"], state)

    if score is not None:
        if state.prev_score is not None:
            r += cfg.score_delta * float(score - state.prev_score)
        state.prev_score = int(score)

    if x_pos is not None:
        if state.prev_x is not None:
            dx = float(x_pos - state.prev_x)
            if dx > 0:
                r += cfg.x_progress * dx
            elif dx < 0:
                r += cfg.x_regress_penalty * dx  # dx<0; penalty stored positive
        state.prev_x = int(x_pos)

    if lives is not None:
        if state.prev_lives is not None and lives < state.prev_lives:
            r += cfg.life_loss * float(state.prev_lives - lives)
        state.prev_lives = int(lives)

    if stage_clear is not None and stage_clear:
        r += cfg.stage_clear

    if terminated:
        r += cfg.death
    else:
        # Survival bonus only on non-terminal steps; otherwise it would
        # silently offset the death penalty on the dying frame.
        r += cfg.survival_bonus

    lo, hi = cfg.clip
    r = max(lo, min(hi, r))
    state.cumulative_shaped += r
    return r


def _get(info: Mapping[str, float], key: str, state: ShapingState) -> float | None:
    if key in info:
        return info[key]
    if key not in state._warned_missing:
        state._warned_missing.add(key)
        logger.warning(
            "reward_shaping: info key %r not present; corresponding term will be zero. "
            "Check the stable-retro integration's data.json.",
            key,
        )
    return None


__all__ = ["ShapingState", "shape_reward", "DEFAULT_INFO_KEYS"]
