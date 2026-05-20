"""IMPALA ResNet feature extractor (SB3-compatible).

Architecture from Espeholt et al. 2018 ("IMPALA: Scalable Distributed
Deep-RL", Fig. 3, the "large" / deep variant). Three convolutional stacks,
each:

    Conv2d(in, C, kernel=3, stride=1, padding=1)
    MaxPool2d(kernel=3, stride=2, padding=1)
    ResidualBlock(C)   ×2

with channel widths 16 → 32 → 32. A residual block is

    x ─► ReLU ─► Conv3x3 ─► ReLU ─► Conv3x3 ─► (+x)

(pre-activation; the skip connection adds the block input back). After the
three stacks: ReLU ─► Flatten ─► Linear(?, features_dim) ─► ReLU.

Rationale vs :class:`~retro_rl.models.cnn.RetroCNN` (Nature-CNN)
---------------------------------------------------------------
~2× the parameters and ~1.3× the compute, but the residual stacks give a
deeper effective receptive field and gradient path, which tends to help when
the policy must track many small, fast-moving sprites (bullets, enemies) — the
regime where the Nature-CNN's three-conv stack plateaus. Kept as a *sibling*
extractor, selectable via ``TrainConfig.features_extractor`` / the registry in
:mod:`retro_rl.models.policies`, so rolling back to ``RetroCNN`` is a one-line
config change.

Channels convention
-------------------
SB3 inserts ``VecTransposeImage`` at the vec-env layer, so tensors arrive
channels-first ``(B, C, H, W)``. We validate this against the transposed
observation space in :meth:`__init__`, mirroring ``RetroCNN``.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class _ResidualBlock(nn.Module):
    """Pre-activation residual block: two 3×3 convs with a skip connection.

    Channel count is preserved (in == out), so the skip is a plain add with no
    projection.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = torch.relu(x)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        return x + residual


class _ConvSequence(nn.Module):
    """One IMPALA conv stack: conv → max-pool(/2) → 2 residual blocks."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = _ResidualBlock(out_channels)
        self.res2 = _ResidualBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class ImpalaCNN(BaseFeaturesExtractor):
    """IMPALA ResNet feature extractor with a configurable output dimension.

    Drop-in replacement for :class:`~retro_rl.models.cnn.RetroCNN`: same
    ``(observation_space, features_dim)`` constructor signature, same uint8→
    float normalization inside :meth:`forward`, same channels-first contract.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        channels: tuple[int, ...] = (16, 32, 32),
    ):
        super().__init__(observation_space, features_dim)

        if len(observation_space.shape) != 3:
            raise ValueError(
                f"ImpalaCNN expects a 3D image observation space (C, H, W) after "
                f"VecTransposeImage; got shape {observation_space.shape}"
            )

        n_input_channels = observation_space.shape[0]

        stacks: list[nn.Module] = []
        in_ch = n_input_channels
        for out_ch in channels:
            stacks.append(_ConvSequence(in_ch, out_ch))
            in_ch = out_ch

        self.conv = nn.Sequential(*stacks, nn.ReLU(inplace=True), nn.Flatten())

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.conv(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # SB3 passes uint8 tensors; normalize to [0, 1] inside the extractor.
        x = observations.float() / 255.0
        return self.linear(self.conv(x))


__all__ = ["ImpalaCNN"]
