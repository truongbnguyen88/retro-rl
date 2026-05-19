"""Visually verify whether AutoFire-pattern B presses produce bullets on-screen
over a long horizon — independent of any RAM-interpretation guess.

Setup: stationary player ship at Level1 start. AutoFire pattern (B every 4
frames) for 600 emu frames. Save the rendered frame every 30 emu frames
(twice a second) as a numbered PNG. Also save the slot bytes per frame in a
CSV alongside for cross-reference.

If bullets are visibly present in the screen below the top after frame 60+,
AutoFire is producing fresh bullets and the user's "stops firing" is about
something else (sparse density, dim sprites, occlusion). If frames are empty
of bullets after the initial burst, AutoFire stopped producing visible
bullets — that's the actual bug.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import retro  # noqa: E402

GAME = "Airstriker-Genesis-v0"
STATE = "Level1"
SCENARIO = "scenario"

SLOT_OFFSETS = [0x008F + 2 * k for k in range(12)]
SCORE_OFFSET = 0x024E
LIVES_OFFSET = 0x025A

OUT_DIR = Path("outputs/diagnostics/visual_fire")


def noop() -> np.ndarray:
    return np.zeros(12, dtype=np.uint8)


def fire() -> np.ndarray:
    a = np.zeros(12, dtype=np.uint8)
    a[0] = 1
    return a


def read_score(ram: np.ndarray) -> int:
    b = ram[SCORE_OFFSET : SCORE_OFFSET + 4]
    return (int(b[0]) << 24) | (int(b[1]) << 16) | (int(b[2]) << 8) | int(b[3])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = retro.make(game=GAME, state=STATE, scenario=SCENARIO, render_mode="rgb_array")
    try:
        env.reset()
        # Skip the splash screen
        for _ in range(360):
            env.step(noop())

        # Continuous AutoFire-pattern for 600 emu frames
        n_frames = 600
        save_every = 15  # 15 emu frames = 0.25 s

        rows: list[dict] = []
        for i in range(n_frames):
            press = (i % 4) == 0
            a = fire() if press else noop()
            _o, _r, term, trunc, _info = env.step(a)
            ram = env.get_ram()
            score = read_score(ram)
            lives = int(ram[LIVES_OFFSET])
            slots = [int(ram[off]) for off in SLOT_OFFSETS]
            slot_str = "".join("1" if v > 0 else "0" for v in slots)
            active_sum = sum(1 for v in slots if v > 0)
            rows.append(
                {
                    "frame": i,
                    "press": int(press),
                    "active_sum": active_sum,
                    "slots": slot_str,
                    "score": score,
                    "lives": lives,
                }
            )

            if i % save_every == 0:
                rgb = env.render()
                if rgb is not None:
                    # Overlay annotations on the frame for at-a-glance review
                    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
                    label = (
                        f"t={i:>3}  fire_press={'P' if press else '.'}  "
                        f"slots={slot_str}  sum={active_sum}  "
                        f"score={score}  lives={lives}"
                    )
                    cv2.putText(
                        bgr,
                        label,
                        (4, bgr.shape[0] - 6),
                        cv2.FONT_HERSHEY_PLAIN,
                        0.7,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.imwrite(str(OUT_DIR / f"frame_{i:04d}.png"), bgr)

            if term or trunc:
                print(f"  episode ended at frame {i}")
                break

        csv_path = OUT_DIR / "trace.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows → {csv_path}")
        print(f"wrote ~{n_frames // save_every} rendered frames → {OUT_DIR}")

    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
