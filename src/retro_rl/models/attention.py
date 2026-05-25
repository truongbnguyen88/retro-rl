"""Temporal-attention feature extractor (SB3-compatible).

Encodes each frame of the stack as a token, then applies self-attention over
the K-frame sequence — a transformer alternative to the recurrent memory tried
in v10/v11. The key property vs an LSTM: there is **no hidden state to
initialize**, so the cold-start blindness that capped v11 (``(h, c)=0`` at every
episode reset) does not exist here. At reset the FrameStack wrapper fills all K
slots with the first frame (see ``retro_rl.env.wrappers.FrameStack.reset``), so
the window is always populated — mildly degenerate for the first few steps, but
never a zero-information state.

Architecture
------------
::

    obs (B, K, 84, 84) uint8         # channel dim IS the K-frame sequence
      │  /255, reshape → (B*K, 1, 84, 84)
      ▼
    per-frame Nature-CNN  (shared)   # one frame → one token of features_dim
      │  reshape → (B, K, features_dim)
      ▼
    + learned positional embedding (1, K, features_dim)
      ▼
    TransformerEncoder (n_layers, n_heads, full self-attention, batch_first)
      ▼
    take last token  z[:, -1, :]     # most-recent frame, contextualized by all K
      ▼
    (B, features_dim)

Design notes
------------
* **``d_model = features_dim``** — attention runs in the extractor's output
  width; no separate token-dim knob. ``n_heads`` must divide ``features_dim``.
* **Last-token readout** — full (non-causal) self-attention is used because we
  only read the final token, which attends over all K frames regardless of a
  causal mask; full attention is the simpler equivalent.
* **Per-frame encoder = Nature-CNN** (not IMPALA) — keeps the K-pass cost in the
  ballpark of v9's single IMPALA pass. The temporal modelling lives in the
  attention layers, so the per-frame encoder need not be deep.
* **Attention hyperparameters are baked in** (``n_heads``, ``n_layers``,
  ``dim_feedforward``) rather than config-exposed: this is the first run with
  the extractor and these are used in exactly one place. Promote to config knobs
  only if a smoke shows they need tuning.

Channels convention
-------------------
SB3 inserts ``VecTransposeImage`` (H=W=84 ≫ K, so channels-last is detected and
transposed), handing the extractor channels-first ``(B, C, H, W)`` where
``C == frame_stack == K`` under grayscale (1 channel/frame). We treat
``observation_space.shape[0]`` as the token count K and assume 1 channel/frame.
"""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class _PerFrameCNN(nn.Module):
    """Nature-CNN encoder over a single grayscale frame → ``features_dim`` token.

    Same conv stack as :class:`~retro_rl.models.cnn.RetroCNN`, but fixed to a
    single input channel (one frame) since the temporal dimension is handled by
    attention downstream, not by channel stacking.
    """

    def __init__(self, features_dim: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, 1, 84, 84)).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(x))


class TemporalAttentionExtractor(BaseFeaturesExtractor):
    """Self-attention over a K-frame window. Drop-in for the CNN extractors.

    Same ``(observation_space, features_dim)`` constructor contract as
    :class:`~retro_rl.models.impala.ImpalaCNN` /
    :class:`~retro_rl.models.cnn.RetroCNN`; selectable via the
    ``FEATURE_EXTRACTORS`` registry in :mod:`retro_rl.models.policies`.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__(observation_space, features_dim)

        if len(observation_space.shape) != 3:
            raise ValueError(
                f"TemporalAttentionExtractor expects a 3D image observation space "
                f"(K, H, W) after VecTransposeImage; got shape {observation_space.shape}"
            )
        if features_dim % n_heads != 0:
            raise ValueError(
                f"features_dim ({features_dim}) must be divisible by n_heads ({n_heads})"
            )

        # Channel dim is the frame sequence (grayscale → 1 channel/frame).
        self._num_frames = observation_space.shape[0]

        self.per_frame = _PerFrameCNN(features_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, self._num_frames, features_dim))
        nn.init.normal_(self.pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=features_dim,
            nhead=n_heads,
            dim_feedforward=2 * features_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor=False: incompatible with norm_first (pre-LN) and
        # would otherwise emit a UserWarning on every construction.
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        b, k, h, w = observations.shape
        # SB3 passes uint8 tensors; normalize to [0, 1] inside the extractor.
        x = observations.float() / 255.0
        # Tokenize: one frame per token, encoded by the shared per-frame CNN.
        x = x.reshape(b * k, 1, h, w)
        tokens = self.per_frame(x).reshape(b, k, -1)
        tokens = tokens + self.pos_emb
        z = self.transformer(tokens)
        # Most-recent-frame token; attends over all K frames via self-attention.
        return z[:, -1, :]


__all__ = ["TemporalAttentionExtractor"]
