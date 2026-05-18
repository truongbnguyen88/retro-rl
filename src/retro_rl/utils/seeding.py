"""Global seeding helper.

One function: :func:`set_global_seed`. Seeds python ``random``, numpy, torch
(CPU + all CUDA devices), and exports the seed as ``PYTHONHASHSEED`` for
subprocess workers.

We deliberately do *not* call ``torch.use_deterministic_algorithms(True)`` —
it slows training meaningfully and forces ``CUBLAS_WORKSPACE_CONFIG`` config
that's only worth the cost when bit-exact reproducibility is required.
Env-side determinism (the policy and rollouts) is achieved via per-env seeds
passed to ``env.reset(seed=...)``.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed python/numpy/torch RNGs. Idempotent."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


__all__ = ["set_global_seed"]
