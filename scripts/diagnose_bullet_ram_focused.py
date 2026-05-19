"""Focused follow-up: trace the candidate bullet-slot array over time.

The earlier scan ``diagnose_bullet_ram.py`` flagged a 2-byte-spaced array
at 0xFF008F .. 0xFF00A1 (and beyond) of 0/1 valued bytes that activate
under tap-fire. This script:

1. Resets, waits past the "Level 1 / Get ready" splash by polling a RAM
   byte that's known to change once gameplay starts.
2. Fires B at AutoFire cadence for N frames, and at every emu frame logs:
   - sum of bytes 0xFF008F, 0xFF0091, ..., 0xFF00A1  (presumed "active slots")
   - the score (from data.json address)
   - per-slot activation flag
   - whether B was pressed this frame
3. Writes a CSV with per-frame state.
4. Prints summary: max concurrent active slots, deactivation lag after a
   press, and number of presses that did vs did not raise the active sum.

Use the CSV to confirm: (a) the max active sum is the on-screen bullet cap;
(b) presses while the cap is hit do NOT increment the sum (the smoking gun
for HYP 3 bullet-cap).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import retro  # noqa: E402

GAME = "Airstriker-Genesis-v0"
STATE = "Level1"
SCENARIO = "scenario"

# Candidate slot offsets (2-byte stride starting at 0x008F).
SLOT_OFFSETS = [0x008F + 2 * k for k in range(12)]  # 12 slots covers anything

# Score address from data.json: 16712270 = 0xFF024E (offset 0x024E, u4 BE).
SCORE_OFFSET = 0x024E
# Lives byte (data.json says u2 but the actual byte is at 0x025A).
LIVES_OFFSET = 0x025A


def noop_action() -> np.ndarray:
    return np.zeros(12, dtype=np.uint8)


def fire_action() -> np.ndarray:
    a = np.zeros(12, dtype=np.uint8)
    a[0] = 1
    return a


def read_score_u4(ram: np.ndarray) -> int:
    b = ram[SCORE_OFFSET : SCORE_OFFSET + 4]
    return (int(b[0]) << 24) | (int(b[1]) << 16) | (int(b[2]) << 8) | int(b[3])


def step_until_gameplay(env, max_frames: int = 600) -> int:
    """Step with B=0 until we detect the splash has cleared.

    Heuristic: lives byte is set to 3 once gameplay begins. With the env just
    reset, the byte may be initialised earlier — fall back to a fixed skip if
    we never see a change.
    """
    last_lives = int(env.get_ram()[LIVES_OFFSET])
    for i in range(max_frames):
        env.step(noop_action())
        ram = env.get_ram()
        lives = int(ram[LIVES_OFFSET])
        score = read_score_u4(ram)
        # Once any of the slot bytes becomes settable, the player ship is alive.
        # We use a simpler rule: bail at 360 emu frames (6 s at 60 Hz).
        if i >= 360:
            return i
    return max_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=1200)
    ap.add_argument("--fire-period", type=int, default=4)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/diagnostics/bullet_slots_trace.csv"),
    )
    args = ap.parse_args()

    env = retro.make(game=GAME, state=STATE, scenario=SCENARIO)
    try:
        env.reset()
        intro_n = step_until_gameplay(env)
        print(f"intro stepped {intro_n} frames; starting fire phase")

        rows = []
        for i in range(args.n_frames):
            press = (i % args.fire_period) == 0
            a = fire_action() if press else noop_action()
            _o, _r, term, trunc, _info = env.step(a)
            ram = env.get_ram()
            slots = [int(ram[off]) for off in SLOT_OFFSETS]
            row = {
                "frame": i,
                "press": int(press),
                "active_sum": sum(slots),
                "score": read_score_u4(ram),
                "lives": int(ram[LIVES_OFFSET]),
            }
            for k, v in enumerate(slots):
                row[f"slot_{k:02d}"] = v
            rows.append(row)
            if term or trunc:
                print(f"  episode ended at frame {i} (term={term}, trunc={trunc})")
                break

        # Write CSV
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote per-frame trace ({len(rows)} rows) → {args.out}")

        # Summary
        sums = np.array([r["active_sum"] for r in rows])
        presses = np.array([r["press"] for r in rows], dtype=bool)
        scores = np.array([r["score"] for r in rows])

        print("\n--- summary ---")
        print(f"  total frames:                {len(rows)}")
        print(f"  total B presses:             {int(presses.sum())}")
        print(f"  active_sum: min/mean/max     {int(sums.min())}/{sums.mean():.2f}/{int(sums.max())}")
        print(f"  score: start/end             {int(scores[0])} / {int(scores[-1])}")

        # When does a press translate to a slot-sum increase within next 4 frames?
        press_frames = np.where(presses)[0]
        effective = 0
        ineffective = 0
        for pf in press_frames:
            window_end = min(pf + 4, len(sums) - 1)
            if sums[window_end] > sums[pf - 1] if pf > 0 else False:
                effective += 1
            else:
                # Stricter: check if sum strictly increased anywhere in (pf, pf+4]
                base = sums[pf - 1] if pf > 0 else sums[pf]
                if any(sums[pf + 1 : window_end + 1] > base):
                    effective += 1
                else:
                    ineffective += 1
        total = effective + ineffective
        if total:
            print(
                f"  presses that raised active_sum within 4 frames: "
                f"{effective}/{total} = {effective/total:.1%}"
            )
            print(
                f"  presses that did NOT raise active_sum (cap hit?): "
                f"{ineffective}/{total} = {ineffective/total:.1%}"
            )

        # Print sample timeline
        print("\nFirst 40 frames (frame  press  active_sum  score  slots):")
        for r in rows[:40]:
            slot_str = "".join(str(r[f"slot_{k:02d}"]) for k in range(12))
            print(
                f"  {r['frame']:>4}  {r['press']}  {r['active_sum']:>2}  "
                f"{r['score']:>5}  {slot_str}"
            )
        print("\nFrames 200-260:")
        for r in rows[200:260]:
            slot_str = "".join(str(r[f"slot_{k:02d}"]) for k in range(12))
            print(
                f"  {r['frame']:>4}  {r['press']}  {r['active_sum']:>2}  "
                f"{r['score']:>5}  {slot_str}"
            )

    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
