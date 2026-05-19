"""Find the player-bullet-count address in Airstriker's 68K work RAM.

Strategy
--------
Run two controlled rollouts from the Level1 save state, with no movement and
no policy in the loop. The only difference is the fire input:

* Phase A (NO FIRE): 600 emulator frames with B=0 every frame.
* Phase B (TAP FIRE): 600 emulator frames with B=1 every Nth frame
  (matching AutoFire's 1-on / 3-off cadence).

After each step we dump the full 64K of 68K work RAM (``env.get_ram()``).

A RAM byte that holds the *player bullet count* should satisfy:

1. Small bounded range during Phase B (0 .. K_max where K_max <= ~8).
2. Mean during Phase B is strictly greater than mean during Phase A
   (firing should keep the count above zero on average).
3. The byte's value crosses zero in Phase A (i.e. it sits at 0 with no fire).

A candidate that also *increases* shortly after a B press is the strongest
match. We compute, for each candidate, the per-frame ``Δvalue`` and check
how often it's positive on a frame where B was pressed in the previous 1-3
emulator frames.

Outputs
-------
* CSV at ``--out`` with one row per candidate address sorted by score.
* Stdout summary with the top 5 candidates and their first 60 Phase-B values
  for eyeball verification (a true bullet count should oscillate in a small
  range while bullets cycle through the screen).
* Confirms that ``ram[lives_addr_offset]`` matches the value reported by
  ``info['lives']`` — sanity check that ``get_ram()`` indexing is what we
  expect.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Make src/ importable so stable_retro is found via the project venv
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import retro  # noqa: E402

GAME = "Airstriker-Genesis-v0"
STATE = "Level1"
SCENARIO = "scenario"

# stable-retro Genesis button order, B is index 0
B_BUTTON = 0

# 68K work RAM base address as exposed in data.json. lives is at 16712282
# (0xFF031A); env.get_ram() returns a flat 64K buffer covering 0xFF0000+,
# so the offset into the array is 0x031A.
RAM_BASE = 0xFF0000
LIVES_ADDR_ABS = 16712282
LIVES_OFFSET = LIVES_ADDR_ABS - RAM_BASE  # 0x031A


def noop_action() -> np.ndarray:
    return np.zeros(12, dtype=np.uint8)


def fire_action() -> np.ndarray:
    a = np.zeros(12, dtype=np.uint8)
    a[B_BUTTON] = 1
    return a


def run_phase(
    env,
    n_frames: int,
    fire_period: int | None,
    skip_intro_frames: int = 240,
) -> tuple[np.ndarray, np.ndarray]:
    """Reset env, skip intro, then run n_frames recording RAM + fire bit.

    Returns
    -------
    ram_history : (n_frames, 65536) uint8
    fire_bits   : (n_frames,) uint8 — whether B was pressed on that frame
    """
    env.reset()
    # Step through the "Level 1 / Get ready" splash with B=0.
    for _ in range(skip_intro_frames):
        env.step(noop_action())

    ram_history = np.zeros((n_frames, 65536), dtype=np.uint8)
    fire_bits = np.zeros(n_frames, dtype=np.uint8)
    for i in range(n_frames):
        if fire_period is not None and (i % fire_period) == 0:
            a = fire_action()
        else:
            a = noop_action()
        fire_bits[i] = a[B_BUTTON]
        _o, _r, term, trunc, _info = env.step(a)
        ram_history[i] = env.get_ram()
        if term or trunc:
            ram_history = ram_history[: i + 1]
            fire_bits = fire_bits[: i + 1]
            print(f"  [warn] episode ended at frame {i} (term={term}, trunc={trunc})")
            break
    return ram_history, fire_bits


def score_candidates(
    no_fire: np.ndarray,
    fire: np.ndarray,
    fire_bits: np.ndarray,
) -> list[dict]:
    """Identify and score addresses that look like a player-bullet counter.

    Filters (must all hold):
      - fire phase byte range in [1, 12]
      - fire phase max byte value <= 16
      - fire phase mean > no-fire phase mean
      - no-fire phase min is 0 (count sits at 0 with no firing)
    Score = fire_mean - no_fire_mean (larger = stronger firing signal).
    Also report cross-correlation between B-press events and value increases.
    """
    nf_min = no_fire.min(0)
    nf_max = no_fire.max(0)
    nf_mean = no_fire.mean(0)

    f_min = fire.min(0)
    f_max = fire.max(0)
    f_mean = fire.mean(0)
    f_range = f_max - f_min

    mask = (
        (f_range >= 1) & (f_range <= 12) & (f_max <= 16) & (f_mean > nf_mean + 0.05) & (nf_min == 0)
    )

    addrs = np.where(mask)[0]
    if len(addrs) == 0:
        return []

    # ΔRAM[t] = fire[t] - fire[t-1]; positive means value went up.
    fire_int = fire.astype(np.int16)
    delta = np.diff(fire_int, axis=0)  # (T-1, 65536)
    fire_press_prev = fire_bits[:-1].astype(bool)  # B pressed on frame t-1

    candidates = []
    for addr in addrs:
        d = delta[:, addr]
        pos_when_press = int(((d > 0) & fire_press_prev).sum())
        pos_total = int((d > 0).sum())
        press_total = int(fire_press_prev.sum())
        # Fraction of "value-up" events that coincide with a recent press.
        precision = (pos_when_press / pos_total) if pos_total > 0 else 0.0
        # Fraction of press events followed by a value-up.
        recall = (pos_when_press / press_total) if press_total > 0 else 0.0
        candidates.append(
            {
                "addr_offset": int(addr),
                "addr_abs": int(RAM_BASE + addr),
                "no_fire_mean": float(nf_mean[addr]),
                "no_fire_range": (int(nf_min[addr]), int(nf_max[addr])),
                "fire_mean": float(f_mean[addr]),
                "fire_range": (int(f_min[addr]), int(f_max[addr])),
                "score": float(f_mean[addr] - nf_mean[addr]),
                "press_align_precision": precision,
                "press_align_recall": recall,
            }
        )
    candidates.sort(key=lambda c: (c["score"], c["press_align_recall"]), reverse=True)
    return candidates


def verify_known_addresses(env, ram: np.ndarray) -> None:
    """Sanity-check that get_ram()[LIVES_OFFSET..+2] matches info['lives']."""
    env.reset()
    _o, _r, _t, _tr, info = env.step(noop_action())
    live_ram = env.get_ram()
    # u2 big-endian over LIVES_OFFSET .. LIVES_OFFSET+1
    lives_ram_val = (int(live_ram[LIVES_OFFSET]) << 8) | int(live_ram[LIVES_OFFSET + 1])
    print(
        f"sanity: info['lives']={info.get('lives')!r}  "
        f"RAM[0x{LIVES_OFFSET:04x}:+2 as >u2]={lives_ram_val}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=600)
    ap.add_argument("--fire-period", type=int, default=4)
    ap.add_argument("--intro-skip", type=int, default=240)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/diagnostics/bullet_ram_candidates.csv"),
    )
    args = ap.parse_args()

    print(f"Game: {GAME}  state: {STATE}  scenario: {SCENARIO}")
    env = retro.make(game=GAME, state=STATE, scenario=SCENARIO)
    try:
        print(f"Action space: {env.action_space}")

        verify_known_addresses(env, np.zeros(1))

        print(f"\nPhase A (NO FIRE): {args.n_frames} frames, intro_skip={args.intro_skip}")
        no_fire, _nf_bits = run_phase(
            env, args.n_frames, fire_period=None, skip_intro_frames=args.intro_skip
        )

        print(f"Phase B (TAP FIRE every {args.fire_period}): {args.n_frames} frames")
        fire, fire_bits = run_phase(
            env,
            args.n_frames,
            fire_period=args.fire_period,
            skip_intro_frames=args.intro_skip,
        )

        # Truncate to the shorter of the two to keep array shapes aligned.
        n = min(len(no_fire), len(fire))
        no_fire = no_fire[:n]
        fire = fire[:n]
        fire_bits = fire_bits[:n]
        print(f"\nAnalysing {n} aligned frames per phase ...")

        candidates = score_candidates(no_fire, fire, fire_bits)
        print(f"Candidate addresses passing filters: {len(candidates)}\n")

        # Write CSV
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            cols = [
                "addr_offset",
                "addr_abs",
                "score",
                "fire_mean",
                "no_fire_mean",
                "fire_min",
                "fire_max",
                "no_fire_min",
                "no_fire_max",
                "press_align_precision",
                "press_align_recall",
            ]
            w = csv.writer(f)
            w.writerow(cols)
            for c in candidates:
                w.writerow(
                    [
                        f"0x{c['addr_offset']:04x}",
                        f"0x{c['addr_abs']:08x}",
                        f"{c['score']:.4f}",
                        f"{c['fire_mean']:.4f}",
                        f"{c['no_fire_mean']:.4f}",
                        c["fire_range"][0],
                        c["fire_range"][1],
                        c["no_fire_range"][0],
                        c["no_fire_range"][1],
                        f"{c['press_align_precision']:.4f}",
                        f"{c['press_align_recall']:.4f}",
                    ]
                )
        print(f"wrote candidates → {args.out}")

        # Stdout summary: top 10 by score with first 80 fire-phase values
        print("\nTop candidates (ordered by fire-vs-nofire mean diff):")
        for c in candidates[:10]:
            addr = c["addr_offset"]
            sample = fire[:80, addr].tolist()
            print(
                f"  0x{addr:04x} (abs 0x{c['addr_abs']:08x})  "
                f"fire_range={c['fire_range']}  no_fire_range={c['no_fire_range']}  "
                f"score={c['score']:.3f}  precision={c['press_align_precision']:.2f}  "
                f"recall={c['press_align_recall']:.2f}"
            )
            print(f"    first 80 fire-phase values: {sample}")

    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
