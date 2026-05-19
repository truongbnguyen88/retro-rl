"""Identify the fire mechanic in Airstriker.

v4 hypothesis: button must be TAPPED (pressed-then-released), not held.
Holding the button continuously fires one bullet on the press event, then
nothing. Random actions naturally toggle, so they fire many bullets.

Test: for each candidate button, run a toggle pattern (press 1 frame, off
N frames, repeat) and see if score rises.
"""

from __future__ import annotations

import numpy as np
import retro

BUTTON_NAMES = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
BUTTON_INDEX = {n: i for i, n in enumerate(BUTTON_NAMES)}
CANDIDATE_FIRE = ["B", "A", "C", "X", "Y", "Z"]


def make_action(buttons: list[str]) -> np.ndarray:
    a = np.zeros(12, dtype=np.uint8)
    for b in buttons:
        a[BUTTON_INDEX[b]] = 1
    return a


def probe_toggle(env, btn: str, on_frames: int, off_frames: int, n: int = 600) -> int:
    """Tap `btn` ON for on_frames, OFF for off_frames, repeating, for n frames total."""
    env.reset()
    max_score = 0
    period = on_frames + off_frames
    for i in range(n):
        is_on = (i % period) < on_frames
        action = make_action([btn]) if is_on else make_action([])
        _o, _r, term, trunc, info = env.step(action)
        max_score = max(max_score, info.get("score", 0))
        if term or trunc:
            break
    return max_score


def main():
    print("=" * 76)
    print("Diagnostic v4: tap-fire vs hold-fire")
    print("=" * 76)

    env = retro.make(
        game="Airstriker-Genesis-v0",
        state="Level1",
        scenario="scenario",
    )
    print(f"Action space: {env.action_space}")
    print()

    # Reference: random toggles many times → many bullets
    env.reset()
    max_score = 0
    for _ in range(600):
        _o, _r, term, trunc, info = env.step(env.action_space.sample())
        max_score = max(max_score, info.get("score", 0))
        if term or trunc:
            break
    print(f"Random actions baseline: max_score = {max_score}")
    print()

    print("Tap pattern: 1 frame ON, N frames OFF, 600 frames total")
    print("-" * 76)
    print(
        f"  {'button':>8}  {'on=1,off=1':>14}  {'on=1,off=3':>14}  {'on=1,off=7':>14}  {'on=2,off=2':>14}"
    )
    for btn in CANDIDATE_FIRE:
        scores = []
        for on, off in [(1, 1), (1, 3), (1, 7), (2, 2)]:
            s = probe_toggle(env, btn, on, off, n=600)
            scores.append(s)
        score_strs = [f"{s:>14}" for s in scores]
        print(f"  {btn:>8}  " + "  ".join(score_strs))

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
