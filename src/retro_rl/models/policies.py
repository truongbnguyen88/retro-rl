"""Policy wiring for SB3.

Rather than subclassing :class:`ActorCriticCnnPolicy` (which would duplicate
SB3's action/value head plumbing for no gain), we expose a single helper that
builds the ``policy_kwargs`` dict consumed by SB3's PPO. The custom feature
extractor is plugged in via ``features_extractor_class`` /
``features_extractor_kwargs``; SB3 handles the rest (orthogonal init, action
head, value head, schedule).

Feature-extractor registry
---------------------------
Two extractors ship today — the Nature-CNN (:class:`RetroCNN`) and the IMPALA
ResNet (:class:`ImpalaCNN`). ``FEATURE_EXTRACTORS`` maps the config-facing name
(``TrainConfig.features_extractor``) to the class. This is the "second variant"
case CLAUDE.md anticipates: a flat dict, not a plugin framework. Add an entry
when a third extractor lands.

We still avoid a custom policy *subclass* — that abstraction only earns its
keep when a non-CNN policy (e.g. recurrent) arrives, at which point the trainer
already branches on algorithm.
"""

from __future__ import annotations

from typing import Any

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from retro_rl.models.cnn import RetroCNN
from retro_rl.models.impala import ImpalaCNN

FEATURE_EXTRACTORS: dict[str, type[BaseFeaturesExtractor]] = {
    "nature_cnn": RetroCNN,
    "impala": ImpalaCNN,
}


def policy_kwargs(
    features_dim: int = 512,
    features_extractor: str = "nature_cnn",
) -> dict[str, Any]:
    """Return the SB3 ``policy_kwargs`` dict wiring the chosen feature extractor.

    Parameters
    ----------
    features_dim
        Output dimension of the extractor's final linear layer.
    features_extractor
        Key into :data:`FEATURE_EXTRACTORS` (``"nature_cnn"`` or ``"impala"``).

    The returned dict is meant to be passed directly to
    ``PPO(... policy_kwargs=...)`` alongside ``policy="CnnPolicy"``.
    """
    try:
        extractor_cls = FEATURE_EXTRACTORS[features_extractor]
    except KeyError as e:
        raise ValueError(
            f"unknown features_extractor {features_extractor!r}; "
            f"valid options: {sorted(FEATURE_EXTRACTORS)}"
        ) from e

    return {
        "features_extractor_class": extractor_cls,
        "features_extractor_kwargs": {"features_dim": features_dim},
        # normalize_images=False because both extractors do their own /255
        # inside forward(); leaving it True would double-normalize.
        "normalize_images": False,
    }


__all__ = ["policy_kwargs", "FEATURE_EXTRACTORS"]
