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

from retro_rl.models.attention import TemporalAttentionExtractor
from retro_rl.models.cnn import RetroCNN
from retro_rl.models.impala import ImpalaCNN
from retro_rl.models.policies import FEATURE_EXTRACTORS, policy_kwargs


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


# ---- IMPALA ResNet --------------------------------------------------------


def test_impala_forward_default_features_dim():
    cnn = ImpalaCNN(_cnn_obs_space())
    out = cnn(torch.zeros((2, 4, 84, 84), dtype=torch.uint8))
    assert out.shape == (2, 256)
    assert out.dtype == torch.float32


def test_impala_custom_features_dim():
    cnn = ImpalaCNN(_cnn_obs_space(), features_dim=512)
    out = cnn(torch.zeros((1, 4, 84, 84), dtype=torch.uint8))
    assert out.shape == (1, 512)


def test_impala_rejects_non_image_space():
    flat = spaces.Box(low=0, high=255, shape=(4 * 84 * 84,), dtype=np.uint8)
    with pytest.raises(ValueError):
        ImpalaCNN(flat)


def test_impala_handles_frame_stack_one():
    cnn = ImpalaCNN(_cnn_obs_space(c=1))
    out = cnn(torch.zeros((1, 1, 84, 84), dtype=torch.uint8))
    assert out.shape == (1, 256)


def test_impala_normalizes_uint8_to_float():
    cnn = ImpalaCNN(_cnn_obs_space())
    x_zeros = torch.zeros((1, 4, 84, 84), dtype=torch.uint8)
    x_max = torch.full((1, 4, 84, 84), 255, dtype=torch.uint8)
    assert torch.isfinite(cnn(x_zeros)).all()
    assert torch.isfinite(cnn(x_max)).all()


# ---- Temporal attention ---------------------------------------------------


def test_temporal_attn_forward_default_features_dim():
    # K=8 frames (v12 default frame_stack).
    extractor = TemporalAttentionExtractor(_cnn_obs_space(c=8))
    out = extractor(torch.zeros((2, 8, 84, 84), dtype=torch.uint8))
    assert out.shape == (2, 256)
    assert out.dtype == torch.float32


def test_temporal_attn_custom_features_dim():
    extractor = TemporalAttentionExtractor(_cnn_obs_space(c=8), features_dim=512)
    out = extractor(torch.zeros((1, 8, 84, 84), dtype=torch.uint8))
    assert out.shape == (1, 512)


def test_temporal_attn_handles_other_frame_counts():
    # K=4 (v9-style window) must also produce features.
    extractor = TemporalAttentionExtractor(_cnn_obs_space(c=4))
    out = extractor(torch.zeros((3, 4, 84, 84), dtype=torch.uint8))
    assert out.shape == (3, 256)


def test_temporal_attn_rejects_non_image_space():
    flat = spaces.Box(low=0, high=255, shape=(8 * 84 * 84,), dtype=np.uint8)
    with pytest.raises(ValueError):
        TemporalAttentionExtractor(flat)


def test_temporal_attn_rejects_indivisible_heads():
    # features_dim must be divisible by n_heads.
    with pytest.raises(ValueError):
        TemporalAttentionExtractor(_cnn_obs_space(c=8), features_dim=250, n_heads=4)


def test_temporal_attn_normalizes_uint8_to_float():
    extractor = TemporalAttentionExtractor(_cnn_obs_space(c=8))
    x_zeros = torch.zeros((1, 8, 84, 84), dtype=torch.uint8)
    x_max = torch.full((1, 8, 84, 84), 255, dtype=torch.uint8)
    assert torch.isfinite(extractor(x_zeros)).all()
    assert torch.isfinite(extractor(x_max)).all()


# ---- extractor selection --------------------------------------------------


def test_policy_kwargs_selects_impala():
    kw = policy_kwargs(features_dim=256, features_extractor="impala")
    assert kw["features_extractor_class"] is ImpalaCNN
    assert kw["features_extractor_kwargs"] == {"features_dim": 256}
    assert kw["normalize_images"] is False


def test_policy_kwargs_selects_temporal_attn():
    kw = policy_kwargs(features_dim=256, features_extractor="temporal_attn")
    assert kw["features_extractor_class"] is TemporalAttentionExtractor
    assert kw["features_extractor_kwargs"] == {"features_dim": 256}
    assert kw["normalize_images"] is False


def test_policy_kwargs_rejects_unknown_extractor():
    with pytest.raises(ValueError):
        policy_kwargs(features_extractor="transformer")


def test_feature_extractor_registry_contents():
    assert FEATURE_EXTRACTORS["nature_cnn"] is RetroCNN
    assert FEATURE_EXTRACTORS["impala"] is ImpalaCNN
    assert FEATURE_EXTRACTORS["temporal_attn"] is TemporalAttentionExtractor
