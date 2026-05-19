"""Inference-time sweep: does slower AutoFire raise return without retraining?

The standalone rate-vs-state test (`diagnose_fire_rate_vs_state.py`) showed
that with a stationary ship, slower fire produces more kills:

  period   displayed_score   kills
       2                 0       0   ← game completely jams
       4                40       2   ← current training setting
      12                60       3
      60                60       3

The policy in v6 doesn't control fire — only movement. So we can re-evaluate
any trained checkpoint with a *different* AutoFire period at inference time,
no retraining needed. If period > 4 raises return at inference, that's
direct evidence the training-time AutoFire rate is throttling us.

Run
---
    PYTHONPATH=src python scripts/eval_period_sweep.py \
        --checkpoint outputs/checkpoints/ppo_airstriker_v6/best.zip \
        --periods 4 8 12 16 24 \
        --episodes 3

Each episode uses a different env seed (eval_seed + ep_i) so we get some
variance instead of the std=0 single-trajectory replay the training-time
eval reports.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stable_baselines3 import PPO  # noqa: E402

from retro_rl.env import make_env  # noqa: E402
from retro_rl.utils.config import load_train_config  # noqa: E402

SCORE_OFFSET_REAL = 0x0250  # data.json says 0x024E; the real byte is 2 bytes higher


def read_displayed_score(env) -> int:
    """Read the actual displayed score byte. Walks down the wrapper chain to
    the raw retro env to call get_ram()."""
    e = env
    while hasattr(e, "env"):
        e = e.env
    if not hasattr(e, "get_ram"):
        return -1
    ram = e.get_ram()
    return int(ram[SCORE_OFFSET_REAL])


def run_episode(env, model, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    ep_return = 0.0
    ep_length = 0
    score_progression = []
    last_score = read_displayed_score(env)
    score_progression.append(last_score)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        ep_return += float(reward)
        ep_length += 1
        cur_score = read_displayed_score(env)
        if cur_score != last_score:
            score_progression.append(cur_score)
            last_score = cur_score
        done = bool(term) or bool(trunc)
    # Approximate kills as the number of *positive* score deltas.
    score_deltas = np.diff(score_progression)
    kills = int((score_deltas > 0).sum())
    return {
        "return": ep_return,
        "length": ep_length,
        "final_score": last_score,
        "kills": kills,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--config", default=Path("configs/ppo.yaml"), type=Path)
    ap.add_argument("--periods", nargs="+", type=int, default=[4, 8, 12, 16, 24])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument(
        "--seed", type=int, default=10042, help="base eval seed (matches train.py eval_seed)"
    )
    args = ap.parse_args()

    cfg = load_train_config(args.config)
    print(f"Loading {args.checkpoint}")
    model = PPO.load(str(args.checkpoint).replace(".zip", ""), device="cpu")

    # Build the env ONCE — on macOS, repeated retro.make() in the same process
    # trips a pyglet/cocoa bug. Instead we reach into the wrapper chain and
    # monkey-patch AutoFireWrapper._period before each sweep.
    from retro_rl.env.wrappers import AutoFireWrapper  # local import

    env = make_env(cfg.env, seed=args.seed)
    autofire = None
    e = env
    while hasattr(e, "env"):
        if isinstance(e, AutoFireWrapper):
            autofire = e
            break
        e = e.env
    if autofire is None:
        raise RuntimeError("No AutoFireWrapper found in env stack")

    print(
        f"\nSweep: {len(args.periods)} periods × {args.episodes} episodes each, "
        f"seeds {args.seed}..{args.seed + args.episodes - 1}"
    )
    print(
        f"\n{'period':>6}  {'rate(Hz)':>8}    "
        f"{'mean_ret':>9}  {'std_ret':>8}    "
        f"{'mean_kills':>11}  {'mean_score':>11}  {'mean_length':>11}    "
        f"per-episode kills"
    )
    rows = []
    try:
        for p in args.periods:
            autofire._period = p  # patch the live wrapper

            ep_results = []
            for ep_i in range(args.episodes):
                ep_seed = args.seed + ep_i
                r = run_episode(env, model, seed=ep_seed)
                ep_results.append(r)

            returns = [r["return"] for r in ep_results]
            kills = [r["kills"] for r in ep_results]
            scores = [r["final_score"] for r in ep_results]
            lengths = [r["length"] for r in ep_results]
            std_ret = statistics.stdev(returns) if len(returns) > 1 else 0.0
            print(
                f"{p:>6}  {60 / p:>8.2f}    "
                f"{statistics.mean(returns):>9.2f}  {std_ret:>8.2f}    "
                f"{statistics.mean(kills):>11.2f}  {statistics.mean(scores):>11.2f}  "
                f"{statistics.mean(lengths):>11.1f}    {kills}"
            )
            rows.append((p, returns, kills, scores, lengths))
    finally:
        try:
            env.close()
        except Exception:
            pass

    print()
    best = max(rows, key=lambda r: statistics.mean(r[1]))
    base = next(r for r in rows if r[0] == 4)
    base_mean = statistics.mean(base[1])
    best_mean = statistics.mean(best[1])
    delta = best_mean - base_mean
    print(f"baseline period=4:  mean_return={base_mean:.2f}")
    print(f"best period={best[0]:>3}:    mean_return={best_mean:.2f}   ({delta:+.2f} vs baseline)")
    if delta > 10:
        print(
            f"\nVERDICT: changing AutoFire period to {best[0]} at inference time RAISES return\n"
            f"         by {delta:.1f} without retraining. Retraining v7 with this period should\n"
            f"         lift the ceiling further."
        )
    elif delta < -10:
        print(
            "\nVERDICT: the trained policy is *tuned* to period=4. Inference-time changes hurt.\n"
            "         A v7 retrain with the new period is required to validate the slower-fire idea."
        )
    else:
        print(
            f"\nVERDICT: inference-time period change has no meaningful effect (Δ={delta:.2f}).\n"
            f"         Retrain v7 if you want to test slower fire under matching train conditions."
        )


if __name__ == "__main__":
    main()
