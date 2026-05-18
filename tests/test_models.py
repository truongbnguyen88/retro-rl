"""Milestone-2 model tests.

These exercise the CNN feature extractor in isolation — no env, no SB3
training loop. The aim is to lock the public surface: input shape, output
shape, dtype handling, configurable feature dimension.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from retro_rl.models.cnn import RetroCNN
from retro_rl.models.policies import policy_kwargs


# After VecTransposeImage SB3 hands the extractor a channels-first space.
def _cnn_obs_space(c: int = 4, h: int = 84, w: int = 84) -> spaces.Box:
    return spaces.Box(low=0, high=255, shape=(c, h, w), dtype=np.uint8)


def test_retro_cnn_forward_default_features_dim():
    obs_space = _cnn_obs_space()
    cnn = RetroCNN(obs_space)
    x = torch.zeros((2, 4, 84, 84), dtype=torch.uint8)
    out = cnn(x)
    assert out.shape == (2, 512)
    assert out.dtype == torch.float32


def test_retro_cnn_custom_features_dim():
    cnn = RetroCNN(_cnn_obs_space(), features_dim=256)
    out = cnn(torch.zeros((1, 4, 84, 84), dtype=torch.uint8))
    assert out.shape == (1, 256)


def test_retro_cnn_rejects_non_image_space():
    flat = spaces.Box(low=0, high=255, shape=(4 * 84 * 84,), dtype=np.uint8)
    with pytest.raises(ValueError):
        RetroCNN(flat)


def test_retro_cnn_handles_various_frame_stack_sizes():
    # Frame stack = 1 (no temporal context) should still produce features.
    cnn = RetroCNN(_cnn_obs_space(c=1))
    out = cnn(torch.zeros((1, 1, 84, 84), dtype=torch.uint8))
    assert out.shape == (1, 512)


def test_retro_cnn_normalizes_uint8_to_float():
    """Forward must accept uint8 input and internally cast to float32 / 255."""
    cnn = RetroCNN(_cnn_obs_space())
    x_zeros = torch.zeros((1, 4, 84, 84), dtype=torch.uint8)
    x_max = torch.full((1, 4, 84, 84), 255, dtype=torch.uint8)
    # No crash, finite outputs in both extremes.
    assert torch.isfinite(cnn(x_zeros)).all()
    assert torch.isfinite(cnn(x_max)).all()


def test_policy_kwargs_wires_retro_cnn():
    kw = policy_kwargs(features_dim=384)
    assert kw["features_extractor_class"] is RetroCNN
    assert kw["features_extractor_kwargs"] == {"features_dim": 384}
    assert kw["normalize_images"] is False
