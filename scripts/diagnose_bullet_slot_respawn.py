"""Do the bullet-slot tickets at 0xFF008F+2k reset when the player respawns?

Sequence
--------
1. Reset, skip the ``Level 1 / Get ready`` splash (~360 emu frames of B=0).
2. Phase A — burn 5 fire tickets (B every 4th frame for 20 frames).
3. Phase B — long observation with B=0 while the stationary ship is killed
   by an oncoming enemy and the respawn animation plays. Until ~600 frames.
4. Phase C — after we see a lives decrement (proof of respawn), burst-fire
   another 5x to see whether new tickets can be claimed.
5. Phase D — another long B=0 observation.

What to look for
----------------
* If slots 0–4 transition 1 → 0 around the same emu frame that the lives
  byte decrements: slots ARE per-life tickets, and respawn frees them.
* If slots 0–4 stay 1 even after the lives byte decrements: tickets are
  per-EPISODE, which would be the worst case — the agent has 10 bullets
  total across all 3 lives.
* In Phase C, if slots 5–9 fill but 0–4 are still 1, that's consistent with
  per-life tickets that reset (the post-respawn fires populate the next
  available slots; old slots that cleared became reusable). If a different
  slot block fills, the slot indexing is more nuanced than a simple stack.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import retro  # noqa: E402

GAME = "Airstriker-Genesis-v0"
STATE = "Level1"
SCENARIO = "scenario"

SLOT_OFFSETS = [0x008F + 2 * k for k in range(12)]
SCORE_OFFSET = 0x024E
LIVES_OFFSET = 0x025A


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
    env = retro.make(game=GAME, state=STATE, scenario=SCENARIO)
    try:
        env.reset()
        for _ in range(360):
            env.step(noop())

        # Phase A: burn 5 fire tickets at period=4 (20 emu frames)
        phase_a_frames = 20
        # Phase B: long no-fire observation to let the stationary ship die
        phase_b_frames = 600
        # Phase C: burst-fire 5 more times (20 frames)
        phase_c_frames = 20
        # Phase D: tail observation
        phase_d_frames = 200

        total = phase_a_frames + phase_b_frames + phase_c_frames + phase_d_frames

        slot_h = np.zeros((total, 12), dtype=np.uint8)
        press_h = np.zeros(total, dtype=bool)
        score_h = np.zeros(total, dtype=np.int64)
        lives_h = np.zeros(total, dtype=np.int16)
        phase_h = np.zeros(total, dtype=np.int8)

        ended_early = False
        for i in range(total):
            if i < phase_a_frames:
                phase = 0  # A
                press = (i % 4) == 0
            elif i < phase_a_frames + phase_b_frames:
                phase = 1  # B
                press = False
            elif i < phase_a_frames + phase_b_frames + phase_c_frames:
                phase = 2  # C
                j = i - (phase_a_frames + phase_b_frames)
                press = (j % 4) == 0
            else:
                phase = 3  # D
                press = False

            a = fire() if press else noop()
            phase_h[i] = phase
            press_h[i] = press
            _o, _r, term, trunc, _info = env.step(a)
            ram = env.get_ram()
            for k, off in enumerate(SLOT_OFFSETS):
                slot_h[i, k] = int(ram[off])
            score_h[i] = read_score(ram)
            lives_h[i] = int(ram[LIVES_OFFSET])
            if term or trunc:
                slot_h = slot_h[: i + 1]
                press_h = press_h[: i + 1]
                score_h = score_h[: i + 1]
                lives_h = lives_h[: i + 1]
                phase_h = phase_h[: i + 1]
                ended_early = True
                print(f"  episode terminated at frame {i} (term={term}, trunc={trunc})")
                break

        # ---- Identify lives transitions
        n = len(lives_h)
        lives_drops = []
        for i in range(1, n):
            if lives_h[i] < lives_h[i - 1]:
                lives_drops.append(i)

        print(f"\nlives series: start={int(lives_h[0])} end={int(lives_h[-1])}")
        if not lives_drops:
            print(
                "!! No lives drop observed in the trace. Increase phase B duration "
                "or take damage faster."
            )
        else:
            for t in lives_drops:
                print(f"  lives {int(lives_h[t - 1])} → {int(lives_h[t])} at emu frame {t}")

        # ---- Slot bit transitions per slot
        print("\n  slot   0→1   1→0   first_set@   last_set@   value@final")
        for k in range(12):
            col = (slot_h[:, k] > 0).astype(np.int8)
            d = np.diff(col)
            up = int((d > 0).sum())
            dn = int((d < 0).sum())
            first_set = int(np.argmax(col > 0)) if col.any() else -1
            last_set = int(np.where(col > 0)[0].max()) if col.any() else -1
            print(
                f"  {k:>4}   {up:>3}   {dn:>3}   {first_set:>10}   "
                f"{last_set:>9}   {int(col[-1]):>11}"
            )

        # ---- Print the timeline around lives drops + phase boundaries
        markers = set()
        markers.add(0)
        markers.add(phase_a_frames - 1)
        markers.add(phase_a_frames)
        markers.add(phase_a_frames + phase_b_frames - 1)
        markers.add(phase_a_frames + phase_b_frames)
        for t in lives_drops:
            for d in (-3, -2, -1, 0, 1, 2, 3):
                markers.add(t + d)
        markers = sorted(m for m in markers if 0 <= m < n)

        phases = "ABCD"
        print("\n  t      phase  press  slots         sum  score  lives")
        for t in markers:
            bits = "".join("1" if v > 0 else "0" for v in slot_h[t])
            sum_ = int((slot_h[t] > 0).sum())
            print(
                f"  {t:>5}    {phases[int(phase_h[t])]}    {'P' if press_h[t] else '.'}    "
                f"{bits}  {sum_:>3}  {int(score_h[t]):>5}  {int(lives_h[t])}"
            )

        # ---- Verdict
        print("\n--- verdict ---")
        if not lives_drops:
            print("  inconclusive (no lives drop observed)")
        else:
            t_drop = lives_drops[0]
            # Did any of slots 0..9 transition 1 → 0 within ±20 frames of the drop?
            window = slice(max(0, t_drop - 20), min(n, t_drop + 20))
            cleared = []
            for k in range(10):
                col = (slot_h[window, k] > 0).astype(np.int8)
                if (np.diff(col) < 0).any():
                    cleared.append(k)
            if cleared:
                print(
                    f"  RESPAWN CLEARS TICKETS: slots {cleared} dropped 1→0 within "
                    f"20 emu frames of the lives decrement at t={t_drop}."
                )
            else:
                print(
                    f"  RESPAWN DOES NOT CLEAR TICKETS: no slot 0..9 went 1→0 within "
                    f"20 emu frames of the lives drop at t={t_drop}."
                )

            # Did Phase C presses claim new tickets?
            phase_c_start = phase_a_frames + phase_b_frames
            phase_c_end = phase_c_start + phase_c_frames
            if phase_c_end <= n:
                pre = slot_h[phase_c_start - 1, :10]
                post = slot_h[phase_c_end - 1, :10]
                newly_set = [k for k in range(10) if int(post[k]) > 0 and int(pre[k]) == 0]
                if newly_set:
                    print(
                        f"  POST-RESPAWN FIRE WORKS: slots {newly_set} went 0→1 "
                        f"during phase C presses."
                    )
                else:
                    print(
                        "  POST-RESPAWN FIRE FAILED: no new slot 0..9 set during "
                        "phase C — tickets do not refill after respawn."
                    )
            else:
                print("  (episode ended before phase C completed)")

    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
