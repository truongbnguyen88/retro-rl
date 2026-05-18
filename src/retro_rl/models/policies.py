"""Policy wiring for SB3.

Rather than subclassing :class:`ActorCriticCnnPolicy` (which would duplicate
SB3's action/value head plumbing for no gain), we expose a single helper that
builds the ``policy_kwargs`` dict consumed by SB3's PPO. The custom CNN is
plugged in via ``features_extractor_class`` / ``features_extractor_kwargs``;
SB3 handles the rest (orthogonal init, action head, value head, schedule).

If a second policy variant shows up (e.g. recurrent), we'll introduce a real
subclass at that point — premature abstraction is explicitly called out in
CLAUDE.md.
"""

from __future__ import annotations

from typing import Any

from retro_rl.models.cnn import RetroCNN


def policy_kwargs(features_dim: int = 512) -> dict[str, Any]:
    """Return the SB3 ``policy_kwargs`` dict wiring :class:`RetroCNN`.

    The returned dict is meant to be passed directly to ``PPO(... policy_kwargs=...)``
    alongside ``policy="CnnPolicy"``.
    """
    return {
        "features_extractor_class": RetroCNN,
        "features_extractor_kwargs": {"features_dim": features_dim},
        # Keep SB3's default MLP head (net_arch=None ⇒ small shared head).
        # Leaving normalize_images=False would skip SB3's /255 step; we want
        # SB3 to leave normalization to RetroCNN.forward, so set False here.
        "normalize_images": False,
    }


__all__ = ["policy_kwargs"]
