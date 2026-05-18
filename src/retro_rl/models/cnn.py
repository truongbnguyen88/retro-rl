"""Nature-CNN feature extractor (SB3-compatible).

Architecture is the canonical DQN/Nature paper stack:

    Conv2d(C, 32, kernel=8, stride=4) ─► ReLU
    Conv2d(32, 64, kernel=4, stride=2) ─► ReLU
    Conv2d(64, 64, kernel=3, stride=1) ─► ReLU
    Flatten ─► Linear(?, features_dim) ─► ReLU

We subclass SB3's :class:`BaseFeaturesExtractor` rather than re-using
``stable_baselines3.common.torch_layers.NatureCNN`` directly so that:

* the feature dimension is an explicit, type-checked config knob,
* future variants (LSTM head, dueling head, attention) drop in without
  re-plumbing the PPO factory.

Channels convention
-------------------
SB3 detects the observation space as a channels-last image
(``shape=(H, W, C*frame_stack)``) and inserts ``VecTransposeImage`` at the
vec-env layer, so by the time tensors reach :meth:`forward` they are
channels-first ``(B, C, H, W)``. We assert this in :meth:`__init__` against
the *transposed* observation space SB3 hands us.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class RetroCNN(BaseFeaturesExtractor):
    """Nature-CNN feature extractor with a configurable output dimension.

    Named ``RetroCNN`` to avoid collision with
    :class:`stable_baselines3.common.torch_layers.NatureCNN` (same architecture,
    fixed ``features_dim=512``; we want it parameterized).
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        if len(observation_space.shape) != 3:
            raise ValueError(
                f"RetroCNN expects a 3D image observation space (C, H, W) after "
                f"VecTransposeImage; got shape {observation_space.shape}"
            )

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # SB3 passes uint8 tensors; normalize to [0, 1] inside the extractor.
        x = observations.float() / 255.0
        return self.linear(self.cnn(x))


__all__ = ["RetroCNN"]
