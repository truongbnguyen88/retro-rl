"""Visual env sanity check via random-action rollouts.

Use this to *watch* the env actually run — a slower, longer cousin of the
unit-test smoke that opens a viewer window. Decoupled from pytest so it
doesn't slow CI.

Usage
-----
    # Default config (Airstriker)
    python scripts/play_random.py --config configs/env.yaml

    # Headless smoke (e.g., from a script) — no window, max throughput
    python scripts/play_random.py --config configs/env.yaml --no-render --fps 0

Notes
-----
* Default FPS is 15 — matches native rate post action-repeat (60Hz / 4).
* Random agent typically dies within 5-30 seconds per episode; with
  ``end_on_life_lost=true`` (the default), one life lost ends the episode.
* This is a *visual* check, not a benchmark. Throughput is throttled to FPS.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from retro_rl.env import make_env
from retro_rl.utils import get_logger, load_env_config, set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="path to env YAML")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="rendered fps (steps/sec, post action-repeat). 0 = unthrottled.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="skip the viewer window (headless)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="print step progress every N steps",
    )
    args = parser.parse_args()

    log = get_logger("play_random")
    set_global_seed(args.seed)

    cfg = load_env_config(args.config)
    render_mode = None if args.no_render else "human"
    log.info(
        "config=%s  game=%s  state=%s  render=%s  fps=%g",
        args.config,
        cfg.game,
        cfg.state,
        render_mode or "off",
        args.fps,
    )

    env = make_env(cfg, seed=args.seed, render_mode=render_mode)
    frame_period = 1.0 / args.fps if args.fps > 0 else 0.0

    try:
        for ep in range(args.episodes):
            ep_seed = args.seed + ep
            obs, info = env.reset(seed=ep_seed)
            cum_reward = 0.0
            cum_shaped = 0.0
            steps = 0
            terminated = truncated = False
            log.info("episode %d/%d start (seed=%d)", ep + 1, args.episodes, ep_seed)

            t0 = time.perf_counter()
            next_frame = t0

            while not (terminated or truncated):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                cum_reward += float(reward)
                cum_shaped += float(info.get("shaped_reward", 0.0))
                steps += 1

                if args.log_every > 0 and steps % args.log_every == 0:
                    log.info(
                        "  step=%d  cum_total=%.2f  cum_shaped=%.2f",
                        steps,
                        cum_reward,
                        cum_shaped,
                    )

                if frame_period > 0:
                    next_frame += frame_period
                    sleep_for = next_frame - time.perf_counter()
                    if sleep_for > 0:
                        time.sleep(sleep_for)

            elapsed = time.perf_counter() - t0
            reason = "terminated" if terminated else "truncated"
            log.info(
                "episode %d done: steps=%d  cum_total=%.2f  cum_shaped=%.2f  "
                "duration=%.1fs  reason=%s",
                ep + 1,
                steps,
                cum_reward,
                cum_shaped,
                elapsed,
                reason,
            )
    finally:
        try:
            env.close()
        except AttributeError:
            # pyglet 1.5 Cocoa event-loop teardown bug on macOS.
            pass


if __name__ == "__main__":
    main()
