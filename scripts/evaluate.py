#!/usr/bin/env python
"""Evaluate a trained checkpoint on N deterministic episodes.

Usage
-----
python scripts/evaluate.py \\
    --checkpoint outputs/checkpoints/ppo_airstriker_smoke/best.zip \\
    --config configs/ppo.yaml \\
    --episodes 20 \\
    [--seed 42] \\
    [--no-video] \\
    [--output-dir outputs/eval/my_run/]

Outputs
-------
<output-dir>/metrics.json     — EvalMetrics dict
<output-dir>/episode_0.mp4   — rendered first episode (unless --no-video)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# sys.path shim: mirrors conftest.py — needed when run as a plain script.
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from stable_baselines3 import PPO

from retro_rl.env import make_env
from retro_rl.evaluation import evaluate
from retro_rl.utils.config import load_train_config
from retro_rl.utils.logging import get_logger
from retro_rl.utils.seeding import set_global_seed
from retro_rl.utils.video import write_mp4


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a retro-rl checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Path to a .zip checkpoint produced by train.py.")
    parser.add_argument("--config", required=True, type=Path,
                        help="Training config YAML (used for env settings).")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of deterministic evaluation episodes (default: 20).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Env seed for evaluation (default: 42).")
    parser.add_argument("--no-video", action="store_true", dest="no_video",
                        help="Disable video recording.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for metrics.json and episode_0.mp4.")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    record_video = not args.no_video
    train_cfg = load_train_config(args.config)
    set_global_seed(args.seed)

    out_dir: Path = args.output_dir or (Path("outputs/eval") / train_cfg.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = get_logger(name="retro_rl.evaluate", run_dir=out_dir)
    log.info(
        "checkpoint=%s episodes=%d seed=%d video=%s output=%s",
        args.checkpoint, args.episodes, args.seed, record_video, out_dir,
    )

    render_mode = "rgb_array" if record_video else None
    env = make_env(train_cfg.env, seed=args.seed, render_mode=render_mode)

    log.info("loading checkpoint %s", args.checkpoint)
    # env=None is valid for inference-only PPO.load — no training env needed.
    model = PPO.load(str(args.checkpoint), env=None)

    try:
        metrics, frames = evaluate(
            agent=model,
            env=env,
            n_episodes=args.episodes,
            deterministic=True,
            record_video=record_video,
            info_keys=train_cfg.env.info_keys,
        )
    finally:
        env.close()

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(asdict(metrics), indent=2))
    log.info("metrics → %s", metrics_path)
    log.info(
        "mean_return=%.2f std=%.2f mean_length=%.1f "
        "stage_clear_rate=%.2f deaths/ep=%.2f",
        metrics.mean_return, metrics.std_return, metrics.mean_length,
        metrics.stage_clear_rate, metrics.mean_deaths,
    )

    if record_video and frames:
        video_path = out_dir / "episode_0.mp4"
        write_mp4(frames, video_path, fps=30)
        log.info("video → %s", video_path)
    elif record_video:
        log.warning(
            "record_video=True but no frames collected "
            "(env render_mode may not be 'rgb_array')"
        )

    print(f"\n=== Evaluation complete ({args.episodes} episodes) ===")
    print(f"  mean_return:       {metrics.mean_return:.2f} ± {metrics.std_return:.2f}")
    print(f"  [min, max]:        [{metrics.min_return:.2f}, {metrics.max_return:.2f}]")
    print(f"  mean_length:       {metrics.mean_length:.1f} steps")
    print(f"  stage_clear_rate:  {metrics.stage_clear_rate:.0%}")
    print(f"  deaths/episode:    {metrics.mean_deaths:.2f}")
    print(f"\nOutputs → {out_dir}")


if __name__ == "__main__":
    main()
