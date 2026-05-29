#!/usr/bin/env python
"""Watch a trained checkpoint play the game.

Opens a live viewer window by default and saves an mp4 to outputs/videos/.
Use --no-render to run headless (video only, faster).

Usage
-----
    # Watch best v12 checkpoint (live window + save video)
    python scripts/play.py --checkpoint outputs/checkpoints/ppo_airstriker_v12/best.zip

    # Headless — just save the video, no window
    python scripts/play.py --checkpoint outputs/checkpoints/ppo_airstriker_v12/best.zip \\
        --no-render

    # Multiple episodes, custom FPS, explicit config
    python scripts/play.py --checkpoint outputs/checkpoints/ppo_airstriker_v9/best.zip \\
        --episodes 3 --fps 30 --config configs/ppo_v9.yaml

Outputs
-------
outputs/videos/<run_name>/play-ep<N>.mp4   — one mp4 per episode

Notes
-----
* Config is inferred from the checkpoint's config_snapshot.json sidecar when
  --config is omitted — no need to remember which config matches which run.
* Evaluation is always deterministic (argmax policy, fixed seeds).
* Reward normalization (VecNormalize) is not applied at inference — the env
  reports raw game scores, consistent with eval/backend behaviour.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from stable_baselines3 import PPO  # noqa: E402

from retro_rl.env import make_env  # noqa: E402
from retro_rl.utils.config import load_train_config  # noqa: E402
from retro_rl.utils.logging import get_logger  # noqa: E402
from retro_rl.utils.seeding import set_global_seed  # noqa: E402
from retro_rl.utils.video import write_mp4  # noqa: E402


def _infer_config(checkpoint: Path) -> Path:
    """Return the config path from the sidecar, or raise a helpful error."""
    snapshot = checkpoint.parent / "config_snapshot.json"
    if not snapshot.exists():
        raise FileNotFoundError(
            f"No config_snapshot.json found next to {checkpoint}. Pass --config explicitly."
        )
    data = json.loads(snapshot.read_text())
    # Walk configs/ to find a ppo_*.yaml whose stem matches the run_name.
    run_name = data.get("run_name", "")
    configs_dir = _repo_root / "configs"
    for candidate in configs_dir.glob("ppo_*.yaml"):
        if run_name.replace("ppo_airstriker_", "") in candidate.stem:
            return candidate
    # Fallback: try to reconstruct from run_name
    guess = configs_dir / f"{run_name.replace('ppo_airstriker_', 'ppo_')}.yaml"
    if guess.exists():
        return guess
    raise FileNotFoundError(
        f"Could not infer config for run '{run_name}' from {snapshot}. Pass --config explicitly."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a retro-rl checkpoint play.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to a .zip checkpoint (e.g. outputs/checkpoints/<run>/best.zip).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Training config YAML. Inferred from config_snapshot.json if omitted.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to play (default: 1).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Viewer/video frame rate in steps/sec (default: 15). 0 = unthrottled.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed; each episode uses seed+episode_index (default: 42).",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Skip the live viewer window; still saves mp4.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip mp4 recording (useful with the live viewer for zero overhead).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for mp4 files. Default: outputs/videos/<run_name>/",
    )
    args = parser.parse_args()

    if not args.checkpoint.exists():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    config_path = args.config or _infer_config(args.checkpoint)
    if not config_path.exists():
        parser.error(f"config not found: {config_path}")

    train_cfg = load_train_config(config_path)
    set_global_seed(args.seed)

    out_dir: Path = args.output_dir or (Path("outputs/videos") / train_cfg.run_name)
    if not args.no_video:
        out_dir.mkdir(parents=True, exist_ok=True)

    log = get_logger(name="retro_rl.play", run_dir=out_dir if not args.no_video else None)
    log.info(
        "checkpoint=%s  run=%s  episodes=%d  fps=%g  render=%s  video=%s",
        args.checkpoint,
        train_cfg.run_name,
        args.episodes,
        args.fps,
        not args.no_render,
        not args.no_video,
    )

    # render_mode: "human" opens a viewer window; "rgb_array" captures frames silently.
    # When the live viewer is on, use --no-render + no --no-video to get a clean mp4.
    render_mode: str | None = (
        ("rgb_array" if not args.no_video else None) if args.no_render else "human"
    )

    model = PPO.load(str(args.checkpoint), env=None)
    log.info("loaded checkpoint %s", args.checkpoint)

    frame_period = 1.0 / args.fps if args.fps > 0 else 0.0

    for ep in range(args.episodes):
        ep_seed = args.seed + ep
        env = make_env(train_cfg.env, seed=ep_seed, render_mode=render_mode)

        try:
            obs, _ = env.reset(seed=ep_seed)
            cum_reward = 0.0
            steps = 0
            frames: list = []
            terminated = truncated = False
            log.info("episode %d/%d  seed=%d", ep + 1, args.episodes, ep_seed)

            next_frame_t = time.perf_counter()

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                cum_reward += float(reward)
                steps += 1

                if render_mode == "rgb_array" and not args.no_video:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)

                if frame_period > 0 and not args.no_render:
                    next_frame_t += frame_period
                    sleep_for = next_frame_t - time.perf_counter()
                    if sleep_for > 0:
                        time.sleep(sleep_for)

        finally:
            with contextlib.suppress(AttributeError):  # pyglet Cocoa teardown bug on macOS
                env.close()

        log.info(
            "episode %d done: steps=%d  return=%.2f  reason=%s",
            ep + 1,
            steps,
            cum_reward,
            "terminated" if terminated else "truncated",
        )
        print(f"  Episode {ep + 1}: {steps} steps  return={cum_reward:.2f}")

        if frames and not args.no_video:
            video_path = out_dir / f"play-ep{ep + 1}.mp4"
            write_mp4(frames, video_path, fps=int(args.fps) or 30)
            log.info("video → %s", video_path)
            print(f"  Saved → {video_path}")

    if not args.no_video:
        print(f"\nVideos → {out_dir}")


if __name__ == "__main__":
    main()
