"""Are 0xFF008F+2k slots transient (bullet-in-flight) or persistent (per-life)?

Earlier scan: 12 bytes at 0xFF008F, 0xFF0091, ..., 0xFF00A5 fill up sequentially
under AutoFire. Open question: once a bullet exits the screen or hits an
enemy, does its slot byte go back to 0 (transient) or stay at 1 (persistent)?

Test
----
1. Reset, skip intro (~360 frames of B=0).
2. Fire one B press (single emu frame).
3. Step with B=0 for 240 emu frames (~4 s — plenty of time for any bullet to
   traverse the 240-px Genesis screen at typical shooter speeds).
4. Record (a) all 12 slot bytes per frame, (b) score, (c) lives.

Then repeat with 3, 5, 9, 11 fires (spread one per 4-frame cycle) — to see
whether each fire permanently consumes a slot ticket.

Output
------
For each fire-count phase, print the slot byte timeline and the slot-bit
transitions (count of 0→1 and 1→0 events per slot). A persistent-ticket
mechanic shows many 0→1 events and zero 1→0 events. A transient mechanic
shows 0→1 followed by 1→0 events as bullets despawn.
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


def run_phase(env, n_fires: int, intro_skip: int, observe_frames: int = 240):
    """Reset, skip intro, fire ``n_fires`` times at period=4, observe with B=0.

    Returns
    -------
    slot_history : (T, 12) uint8 — slot bytes per emu frame
    fire_events  : (T,) bool — whether B was pressed on that frame
    scores       : (T,) int
    lives        : (T,) int
    """
    env.reset()
    for _ in range(intro_skip):
        env.step(noop())

    fire_phase = 4 * n_fires  # frames during which fires happen
    total = fire_phase + observe_frames

    slot_h = np.zeros((total, 12), dtype=np.uint8)
    fire_e = np.zeros(total, dtype=bool)
    scores = np.zeros(total, dtype=np.int64)
    lives = np.zeros(total, dtype=np.uint8)

    for i in range(total):
        if i < fire_phase and (i % 4) == 0:
            a = fire()
        else:
            a = noop()
        fire_e[i] = a[0] == 1
        _o, _r, term, trunc, _info = env.step(a)
        ram = env.get_ram()
        for k, off in enumerate(SLOT_OFFSETS):
            slot_h[i, k] = int(ram[off])
        scores[i] = read_score(ram)
        lives[i] = int(ram[LIVES_OFFSET])
        if term or trunc:
            slot_h = slot_h[: i + 1]
            fire_e = fire_e[: i + 1]
            scores = scores[: i + 1]
            lives = lives[: i + 1]
            break

    return slot_h, fire_e, scores, lives


def transitions(slot_series: np.ndarray) -> tuple[int, int]:
    """Return (count_of_0_to_1, count_of_1_to_0) transitions in a 1-D series."""
    s = (slot_series > 0).astype(np.int8)
    d = np.diff(s)
    return int((d > 0).sum()), int((d < 0).sum())


def summarise(label: str, slot_h, fire_e, scores, lives) -> None:
    n_fires = int(fire_e.sum())
    print(f"\n=== {label}  (n_fires={n_fires}, observed={len(slot_h)} emu frames) ===")
    print(
        f"  lives: start={int(lives[0])}  end={int(lives[-1])}    "
        f"score: start={int(scores[0])}  end={int(scores[-1])}"
    )

    # Per-slot transitions
    print(f"  {'slot':>4}  {'0→1':>5}  {'1→0':>5}  "
          f"{'max':>3}  {'first_set@':>11}  {'last_clr@':>10}")
    for k in range(12):
        up, dn = transitions(slot_h[:, k])
        col = slot_h[:, k]
        first_set = int(np.argmax(col > 0)) if (col > 0).any() else -1
        # Last frame where slot was non-zero; if it's still set at the end,
        # there was no "clear" → mark as "—".
        nonzero = np.where(col > 0)[0]
        if len(nonzero) and nonzero[-1] < len(col) - 1:
            last_clr_str = f"{int(nonzero[-1]) + 1}"
        elif len(nonzero):
            last_clr_str = "—"  # never cleared
        else:
            last_clr_str = "n/a"
        print(
            f"  {k:>4}  {up:>5}  {dn:>5}  {int(col.max()):>3}  "
            f"{first_set:>11}  {last_clr_str:>10}"
        )

    # Aggregate verdict
    total_up = sum(transitions(slot_h[:, k])[0] for k in range(12))
    total_dn = sum(transitions(slot_h[:, k])[1] for k in range(12))
    print(f"  TOTAL: 0→1={total_up}  1→0={total_dn}")
    if total_dn == 0 and total_up > 0:
        print("  VERDICT: slots are PERSISTENT (per-life fire tickets) — no slot ever cleared.")
    elif total_dn >= total_up:
        print("  VERDICT: slots are TRANSIENT (bullets-in-flight) — slots cycle 0→1→0.")
    else:
        print(f"  VERDICT: mixed — {total_up} sets vs {total_dn} clears. Inspect per-slot above.")


def main() -> None:
    env = retro.make(game=GAME, state=STATE, scenario=SCENARIO)
    try:
        for n_fires in (1, 3, 5, 9, 11, 20):
            slot_h, fire_e, scores, lives = run_phase(
                env, n_fires=n_fires, intro_skip=360, observe_frames=240
            )
            summarise(f"Fire {n_fires}x then observe 240 emu frames", slot_h, fire_e, scores, lives)

            # Print first 30 frames + frame-where-cap-might-have-been-hit + last 10
            print("  -- slot-byte timeline (12 cols, one row per emu frame) --")
            T = len(slot_h)
            sample_idxs = list(range(0, min(30, T)))
            # plus a strided sample of later frames
            if T > 30:
                sample_idxs += list(range(30, T, max(1, T // 30)))
            sample_idxs = sorted(set(sample_idxs))[:60]
            for i in sample_idxs:
                bits = "".join("1" if v > 0 else "0" for v in slot_h[i])
                press = "P" if fire_e[i] else "."
                print(
                    f"    t={i:>4}  {press}  {bits}  sum={int((slot_h[i]>0).sum()):>2}  "
                    f"score={int(scores[i]):>5}  lives={int(lives[i])}"
                )
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
