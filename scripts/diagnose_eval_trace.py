"""Per-step diagnostic trace of a deterministic eval rollout.

Loads a checkpoint, runs one deterministic episode against the env, and
records per-policy-step:
  - action index chosen by the policy
  - score (cumulative + delta from prior step)
  - lives (current + flag when it drops)
  - reward, cumulative reward
  - terminated/truncated

Use this to correlate observations in an eval video against actual game
state — specifically, whether visual "stops firing" moments coincide
with life-loss events (Hypothesis 1: respawn invulnerability).

Each policy step corresponds to `action_repeat` raw emulator frames. With
the default action_repeat=4 and Genesis @ 60 fps, policy step N maps to
game frames [N*4 .. N*4+3] (~ N/15 seconds of gameplay).

Run:
    python scripts/diagnose_eval_trace.py \\
        --checkpoint outputs/checkpoints/ppo_airstriker_v5/step-100000.zip \\
        --output outputs/diagnostics/v5_100k_eval_trace.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Make src/ importable when run as a plain script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stable_baselines3 import PPO

from retro_rl.env import make_env
from retro_rl.utils.config import load_env_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/ppo_airstriker_v5/step-100000.zip",
    )
    ap.add_argument("--env-config", default="configs/env.yaml")
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Eval seed. Match the eval callback's seed for an exact "
        "trajectory replay; otherwise any fixed seed reveals the "
        "kill/life-loss correlation we care about.",
    )
    ap.add_argument(
        "--output",
        default="outputs/diagnostics/v5_100k_eval_trace.csv",
    )
    ap.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Safety cap on policy steps (default 5000; > max_episode_steps).",
    )
    args = ap.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model = PPO.load(args.checkpoint, device="cpu")

    print(f"Loading env config: {args.env_config}")
    cfg = load_env_config(Path(args.env_config))

    env = make_env(cfg, seed=args.seed)
    obs, info = env.reset(seed=args.seed)
    print(f"Initial: lives={info.get('lives', '?')}  score={info.get('score', '?')}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    prev_score = int(info.get("score", 0) or 0)
    prev_lives = int(info.get("lives", 3) or 3)
    cum_reward = 0.0

    for step in range(args.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        cum_reward += float(reward)

        score = int(info.get("score", prev_score) or prev_score)
        lives = int(info.get("lives", prev_lives) or prev_lives)
        score_delta = score - prev_score
        lives_dropped = lives < prev_lives

        frame_start = step * cfg.action_repeat
        frame_end = (step + 1) * cfg.action_repeat - 1

        rows.append(
            {
                "step": step,
                "frame_range": f"{frame_start}-{frame_end}",
                "game_sec": round(frame_start / 60.0, 2),
                "action": int(action),
                "score": score,
                "score_delta": score_delta,
                "lives": lives,
                "lives_dropped": "YES" if lives_dropped else "",
                "reward": round(float(reward), 3),
                "cum_reward": round(cum_reward, 2),
                "terminated": terminated,
                "truncated": truncated,
            }
        )

        prev_score = score
        prev_lives = lives

        if terminated or truncated:
            print(f"\nEpisode ended at step {step}: terminated={terminated}, truncated={truncated}")
            break

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nPer-step trace written to: {args.output}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("SUMMARY")
    print("=" * 76)
    total = len(rows)
    print(f"Policy steps:       {total}")
    print(f"Game frames:        {total * cfg.action_repeat}")
    print(f"Game seconds:       {total * cfg.action_repeat / 60.0:.2f}")
    print(f"Final score:        {rows[-1]['score']}")
    print(f"Final lives:        {rows[-1]['lives']}")
    print(f"Cumulative reward:  {rows[-1]['cum_reward']}")

    # Life-loss events
    life_losses = [r for r in rows if r["lives_dropped"]]
    print(f"\nLife-loss events: {len(life_losses)}")
    for r in life_losses:
        print(
            f"  step {r['step']:>4}  "
            f"frames {r['frame_range']:>9}  "
            f"~{r['game_sec']:>5.2f}s  "
            f"lives→{r['lives']}  score={r['score']}"
        )

    # Kill events (score increases)
    kills = [r for r in rows if r["score_delta"] > 0]
    print(f"\nKill events (score delta > 0): {len(kills)}")
    if kills:
        print("  First 15:")
        for r in kills[:15]:
            flag = "  ←← LIFE LOST" if r["lives_dropped"] else ""
            print(
                f"  step {r['step']:>4}  "
                f"frames {r['frame_range']:>9}  "
                f"~{r['game_sec']:>5.2f}s  "
                f"+{r['score_delta']}  →  total {r['score']}{flag}"
            )

    # The critical check: kills + life-loss on the same step
    print("\n" + "-" * 76)
    coinciding = [r for r in rows if r["score_delta"] > 0 and r["lives_dropped"]]
    print(f"Kills coinciding with life loss (same policy step): {len(coinciding)}")
    for r in coinciding:
        print(
            f"  step {r['step']}  frames {r['frame_range']}  "
            f"~{r['game_sec']}s  +{r['score_delta']} score, life lost"
        )

    # Near-coinciding: kill within 5 steps before life loss
    near = []
    for r in life_losses:
        s = r["step"]
        recent_kills = [k for k in kills if 0 < s - k["step"] <= 5 and k is not r]
        if recent_kills:
            near.append((r, recent_kills))
    print(f"\nLife losses preceded by a kill within 5 steps (~0.33s): {len(near)}")
    for life_row, recent in near:
        prev_kill = recent[-1]
        gap_steps = life_row["step"] - prev_kill["step"]
        print(
            f"  life lost at step {life_row['step']} (~{life_row['game_sec']}s); "
            f"last kill {gap_steps} steps prior at step {prev_kill['step']} "
            f"(+{prev_kill['score_delta']})"
        )

    print("\n" + "=" * 76)
    if coinciding or near:
        print(
            "INTERPRETATION: kills occur near life-loss events → Hypothesis 1\n"
            "                (respawn invulnerability) is consistent with the\n"
            "                'stops firing' visual after kills."
        )
    else:
        print(
            "INTERPRETATION: kills and life-losses are temporally decoupled.\n"
            "                If the video still shows fire stopping after\n"
            "                kills, look at Hypothesis 2 (explosion sprite\n"
            "                occluding bullets) or 3 (on-screen bullet cap)."
        )

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
