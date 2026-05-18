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
import os
import sys
from pathlib import Path

# sys.path shim: macOS auto-applies UF_HIDDEN to files in `.venv/lib/.../site-packages/`,
# and CPython 3.12.5+ skips hidden .pth files for security. We seed sys.path here
# (main process) and PYTHONPATH (so SubprocVecEnv spawn-method workers inherit it).
_repo_root = Path(__file__).resolve().parents[1]
_src_path = str(_repo_root / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)
_existing_pp = os.environ.get("PYTHONPATH", "")
if _src_path not in _existing_pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = (
        _src_path + (os.pathsep + _existing_pp if _existing_pp else "")
    )

from retro_rl.training.trainer import train  # noqa: E402
from retro_rl.utils.config import load_train_config  # noqa: E402


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
