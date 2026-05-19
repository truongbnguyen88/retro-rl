"""Is the mid-life fire-freeze rate-driven or state-driven?

Run via:
    for p in 2 4 12 60; do
      python scripts/diagnose_fire_rate_vs_state.py --period $p \
          --out outputs/diagnostics/fire_rvs_$p.json
    done
    python scripts/diagnose_fire_rate_vs_state.py --summarise \
        --inputs outputs/diagnostics/fire_rvs_*.json

Each run is its own Python process (one retro.make per process — works around
pyglet/cocoa state leaking between repeated env constructions on macOS).

Detection
---------
"Freeze onset" = first emu frame at which slots 0..9 stop changing for
FREEZE_WIN consecutive frames despite B being pressed at least once. We use
slots 0..9 (the actual sprite slots) instead of slot 11 (muzzle flash) —
at high press rates muzzle flash saturates to constantly-on and looks frozen
even when the game is responding fine.

Verdict
-------
If freeze_emu_frame is roughly constant across periods (and ~coincides with
the first lives drop) ⇒ state-driven, slowing AutoFire WILL NOT FIX IT.
If freeze_emu_frame varies systematically with period ⇒ rate-driven, knob.
"""

from __future__ import annotations

import argparse
import json
import sys
from glob import glob
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

GAME = "Airstriker-Genesis-v0"
STATE = "Level1"
SCENARIO = "scenario"

SLOT_OFFSETS = [0x008F + 2 * k for k in range(12)]
SCORE_OFFSET = 0x024E
LIVES_OFFSET = 0x025A

FREEZE_WIN = 60  # ≥60 emu frames with no slot[0..9] change while B pressed = freeze
INTRO_SKIP = 360
RUN_FRAMES = 1500  # well past first life


def noop() -> np.ndarray:
    return np.zeros(12, dtype=np.uint8)


def fire() -> np.ndarray:
    a = np.zeros(12, dtype=np.uint8)
    a[0] = 1
    return a


def run_period(period: int) -> dict:
    import retro  # local import — each subprocess gets a clean module

    env = retro.make(game=GAME, state=STATE, scenario=SCENARIO)
    try:
        env.reset()
        for _ in range(INTRO_SKIP):
            env.step(noop())

        bits = np.zeros(RUN_FRAMES, dtype=np.uint8)
        # Bitmask of slots 0..9 (the actual sprite slots, not muzzle flash).
        slot09_mask = np.zeros(RUN_FRAMES, dtype=np.int32)
        slot_str_arr = [""] * RUN_FRAMES
        lives = np.zeros(RUN_FRAMES, dtype=np.int16)
        score = np.zeros(RUN_FRAMES, dtype=np.int64)
        T = RUN_FRAMES

        for i in range(RUN_FRAMES):
            press = (i % period) == 0
            a = fire() if press else noop()
            bits[i] = a[0]
            _o, _r, term, trunc, _info = env.step(a)
            ram = env.get_ram()
            mask = 0
            for k in range(10):
                if int(ram[SLOT_OFFSETS[k]]) > 0:
                    mask |= 1 << k
            slot09_mask[i] = mask
            slots = [1 if int(ram[off]) > 0 else 0 for off in SLOT_OFFSETS]
            slot_str_arr[i] = "".join(str(s) for s in slots)
            lives[i] = int(ram[LIVES_OFFSET])
            sb = ram[SCORE_OFFSET : SCORE_OFFSET + 4]
            score[i] = (int(sb[0]) << 24) | (int(sb[1]) << 16) | (int(sb[2]) << 8) | int(sb[3])
            if term or trunc:
                T = i + 1
                bits = bits[:T]
                slot09_mask = slot09_mask[:T]
                slot_str_arr = slot_str_arr[:T]
                lives = lives[:T]
                score = score[:T]
                break

        # Freeze detector: first window of FREEZE_WIN consecutive frames where
        # slot09 bitmask is *constant* AND B was pressed at least once in the
        # window. Use bitmask so a slot turning on then off in the same window
        # doesn't count as constant (different masks).
        freeze_onset = -1
        for start in range(T - FREEZE_WIN):
            window_mask = slot09_mask[start : start + FREEZE_WIN]
            window_bits = bits[start : start + FREEZE_WIN]
            if int(window_bits.sum()) == 0:
                continue
            if int(window_mask.max()) == int(window_mask.min()):
                freeze_onset = start
                break

        lives_drops = [int(i) for i in range(1, T) if lives[i] != lives[i - 1]]
        presses_before_freeze = (
            int(bits[:freeze_onset].sum()) if freeze_onset >= 0 else int(bits.sum())
        )

        # Score timeline: first frame at which score equals its final value
        final_score = int(score[-1])
        last_score_change = (
            int(np.where(np.diff(score) != 0)[0].max()) + 1 if (np.diff(score) != 0).any() else -1
        )

        return {
            "period": period,
            "press_rate_hz": 60.0 / period,
            "run_frames": T,
            "presses_total": int(bits.sum()),
            "presses_before_freeze": presses_before_freeze,
            "freeze_onset_frame": int(freeze_onset),
            "last_score_change_frame": last_score_change,
            "first_lives_drop_frame": lives_drops[0] if lives_drops else -1,
            "n_lives_drops": len(lives_drops),
            "final_lives": int(lives[-1]),
            "final_score": final_score,
            "slot_pattern_at_freeze": slot_str_arr[freeze_onset] if freeze_onset >= 0 else "—",
            "slot_pattern_at_run_end": slot_str_arr[-1],
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


def cmd_run(args: argparse.Namespace) -> None:
    print(f"\n  period={args.period:>3}  (B every {args.period} frames, {60 / args.period:.1f} Hz)")
    r = run_period(args.period)
    for k, v in r.items():
        print(f"    {k:>26} = {v}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2))
    print(f"\n  json → {out}")


def cmd_summarise(args: argparse.Namespace) -> None:
    paths = sorted(glob(args.inputs))
    results = [json.loads(Path(p).read_text()) for p in paths]
    results.sort(key=lambda r: r["period"])
    print("\n" + "=" * 96)
    print("SUMMARY")
    print("=" * 96)
    print(
        f"{'period':>6}  {'rate(Hz)':>8}  {'freeze@':>8}  {'lastScore@':>11}  "
        f"{'1stDeath@':>10}  {'presses_b4':>11}  {'finalScore':>11}  {'finalLives':>10}"
    )
    for r in results:
        print(
            f"{r['period']:>6}  {r['press_rate_hz']:>8.2f}  "
            f"{r['freeze_onset_frame']:>8}  {r['last_score_change_frame']:>11}  "
            f"{r['first_lives_drop_frame']:>10}  {r['presses_before_freeze']:>11}  "
            f"{r['final_score']:>11}  {r['final_lives']:>10}"
        )

    fr = [r["freeze_onset_frame"] for r in results if r["freeze_onset_frame"] >= 0]
    ld = [r["first_lives_drop_frame"] for r in results if r["first_lives_drop_frame"] >= 0]
    print("\n" + "-" * 96)
    if not fr:
        print("VERDICT: no freeze observed in any pattern.")
        return
    fr_arr = np.array(fr)
    spread = int(fr_arr.max() - fr_arr.min())
    print(f"freeze frames: {fr}   spread={spread}")
    if ld:
        print(f"first-lives-drop frames: {ld}")
        for r in results:
            if r["freeze_onset_frame"] >= 0 and r["first_lives_drop_frame"] >= 0:
                d = r["freeze_onset_frame"] - r["first_lives_drop_frame"]
                print(f"  period {r['period']:>3}: freeze - 1stDeath = {d:+d} frames")
    if spread < 30:
        print("\nVERDICT: STATE-DRIVEN — freeze frame is ~constant across periods.")
        print("         Slowing AutoFire will NOT fix it. Need to find the trigger state.")
    elif np.array_equal(
        fr_arr.argsort()[::-1], np.array([list(range(len(fr_arr)))]).flatten()[::-1]
    ):
        print("\nVERDICT: RATE-DRIVEN — faster B presses produce earlier freeze.")
    else:
        print("\nVERDICT: MIXED / NOISY — inspect per-period rows above.")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=False)

    run = sub.add_parser("run", help="Run a single period")
    run.add_argument("--period", type=int, required=True)
    run.add_argument("--out", type=Path, required=True)

    summ = sub.add_parser("summarise", help="Summarise multiple JSON outputs")
    summ.add_argument(
        "--inputs", required=True, help="Glob pattern, e.g. 'outputs/diagnostics/fire_rvs_*.json'"
    )

    # Backward-compat: also accept --period / --out directly for the run subcommand
    ap.add_argument("--period", type=int)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--inputs", default="outputs/diagnostics/fire_rvs_*.json")

    args = ap.parse_args()
    if args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "summarise" or args.summarise:
        cmd_summarise(args)
    elif args.period is not None and args.out is not None:
        cmd_run(args)
    else:
        ap.error("specify either 'run --period N --out PATH' or '--summarise --inputs GLOB'")


if __name__ == "__main__":
    main()
