"""CLI: train a PPO agent on a stable-retro env.

Usage
-----

    python scripts/train.py --config configs/ppo.yaml
    python scripts/train.py --config configs/ppo.yaml --resume outputs/checkpoints/<run>/latest-or-best.zip

The config schema lives in ``retro_rl.utils.config.TrainConfig``. Override
``total_timesteps`` etc. by composing a child YAML with ``extends:`` rather
than adding CLI flags.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from retro_rl.training.trainer import train
from retro_rl.utils.config import load_train_config


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a PPO agent (stable-retro).")
    p.add_argument("--config", required=True, type=Path, help="path to training YAML")
    p.add_argument("--resume", type=Path, default=None, help="checkpoint .zip to resume from")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_train_config(args.config)
    best = train(cfg, resume_from=args.resume)
    print(f"best checkpoint: {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
