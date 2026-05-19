"""Run a trained policy through the full wrapper stack and trace AutoFire.

Earlier tests used the RAW stable-retro env and showed continuous bullets.
The v6 eval videos clearly show a per-life dense-then-sparse fire pattern.
Something about the wrapper stack + trained-policy interaction is different.

This script reproduces the eval pipeline (full wrapper stack, deterministic
policy from a checkpoint) but tees off three quantities per *emulator*
frame:
* the B bit AutoFireWrapper actually sent to the raw env that frame
* the 12 slot bytes at 0xFF008F+2k
* whether the lives byte changed this frame (respawn marker)

It also saves rendered frames every N emu frames. If, during a visually
"no-fire" gap, the B bit is still 1 every 4th frame, the wrapper is firing
but the game suppresses the shots — game-side mechanic. If the B bit is 0
during the gap, the wrapper itself is failing in some interaction — our
bug.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stable_baselines3 import PPO  # noqa: E402

from retro_rl.env.wrappers import AutoFireWrapper  # noqa: E402
from retro_rl.utils.config import load_env_config  # noqa: E402

SLOT_OFFSETS = [0x008F + 2 * k for k in range(12)]
LIVES_OFFSET = 0x025A

OUT_DIR = Path("outputs/diagnostics/wrapper_fire_trace")


class AutoFireSpy(AutoFireWrapper):
    """AutoFireWrapper subclass that also records the (counter, fire_bit) it
    emitted on every step — i.e. once per emulator frame."""

    def __init__(self, env, cfg, log: list):
        super().__init__(env, cfg)
        self._log = log

    def step(self, action):
        # Mirror the parent logic so we can record the actual bit emitted.
        action = np.array(action, dtype=np.int8, copy=True)
        bit = 1 if (self._counter % self._period) == 0 else 0
        action[self._fire_idx] = bit
        ctr = int(self._counter)
        self._counter += 1
        obs, reward, term, trunc, info = self.env.step(action)
        # We can't reach the raw retro env from here without rummaging — but
        # the outer trainer wires this AutoFireWrapper directly onto the
        # raw env, so ``self.env`` IS the raw retro env. Grab RAM directly.
        try:
            ram = self.env.get_ram()
            lives = int(ram[LIVES_OFFSET])
            slot_bits = "".join("1" if int(ram[off]) > 0 else "0" for off in SLOT_OFFSETS)
        except AttributeError:
            lives = -1
            slot_bits = ""
        self._log.append(
            {
                "counter": ctr,
                "bit": int(bit),
                "lives": lives,
                "slots": slot_bits,
            }
        )
        return obs, reward, term, trunc, info


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_env_config(Path("configs/env.yaml"))

    # Build the env directly so we can splice in the spy AutoFire instead of
    # the standard one. We replicate apply_wrappers' order verbatim.
    import retro  # noqa: PLC0415

    raw = retro.make(game=cfg.game, state=cfg.state, scenario=cfg.scenario, render_mode="rgb_array")

    fire_log: list[dict] = []
    env = AutoFireSpy(raw, cfg.auto_fire, log=fire_log)
    # Apply the rest of the standard wrapper stack manually.
    from retro_rl.env.wrappers import (
        ActionRepeat,
        DiscreteActionWrapper,
        FrameStack,
        GrayscaleResize,
        RewardShapingWrapper,
    )

    env = DiscreteActionWrapper(env, cfg.action_combos)
    env = ActionRepeat(env, skip=cfg.action_repeat)
    env = RewardShapingWrapper(env, cfg.reward, info_keys=cfg.info_keys)
    if cfg.grayscale:
        env = GrayscaleResize(env, size=cfg.resize)
    if cfg.frame_stack > 1:
        env = FrameStack(env, n=cfg.frame_stack)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=cfg.max_episode_steps)

    # We need to render the raw retro env's frame — find it back in the chain.
    def get_raw():
        e = env
        while hasattr(e, "env"):
            e = e.env
        return e

    raw_ref = get_raw()

    # Load v6 1.1M checkpoint (closest available to the 1M eval video).
    model = PPO.load("outputs/checkpoints/ppo_airstriker_v6/step-1200000", device="cpu")

    obs, info = env.reset(seed=42 + 10_000)  # match training eval_seed
    print(f"Initial lives: {info.get('lives', '?')}")

    rendered_every = 4  # save one frame every N emu frames
    rows = []
    step_i = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)

        # Save a render snapshot every Nth outer step
        if step_i % 5 == 0:
            frame_rgb = raw_ref.render()
            if frame_rgb is not None:
                bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    bgr,
                    f"step={step_i} action={int(action)}",
                    (4, 18),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.8,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imwrite(str(OUT_DIR / f"step_{step_i:04d}.png"), bgr)

        rows.append(
            {
                "outer_step": step_i,
                "action": int(action),
                "lives_info": info.get("lives"),
                "score_info": info.get("score"),
            }
        )
        step_i += 1
        if term or trunc:
            print(f"Episode ended at outer_step={step_i}")
            break

    # Write CSVs
    with open(OUT_DIR / "outer_trace.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(OUT_DIR / "emu_fire_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fire_log[0].keys()))
        w.writeheader()
        w.writerows(fire_log)

    # ---- Summary
    print(f"\nemu frames: {len(fire_log)}")
    print(f"outer steps: {len(rows)}")
    bits = np.array([r["bit"] for r in fire_log])
    print(f"AutoFire B=1 fraction: {bits.mean():.3f} (expected ~{1 / cfg.auto_fire.period:.3f})")
    # Per-life fire pattern
    lives_arr = np.array([r["lives"] for r in fire_log])
    life_breaks = [0]
    for i in range(1, len(lives_arr)):
        if lives_arr[i] != lives_arr[i - 1]:
            life_breaks.append(i)
    life_breaks.append(len(lives_arr))
    print(f"\nlives transitions at emu frames: {life_breaks[1:-1]}")

    print("\n  life_seg   emu_frames   B=1_count   first_B   last_B   slots@start  slots@end")
    for k in range(len(life_breaks) - 1):
        a, b = life_breaks[k], life_breaks[k + 1]
        seg_bits = bits[a:b]
        b1 = int(seg_bits.sum())
        if b1 > 0:
            local_idx = np.where(seg_bits == 1)[0]
            first_B = int(local_idx[0]) + a
            last_B = int(local_idx[-1]) + a
        else:
            first_B = last_B = -1
        slots_a = fire_log[a]["slots"]
        slots_b = fire_log[b - 1]["slots"]
        print(
            f"  {k:>4}     {a:>5}-{b:>5}   {b1:>9}   {first_B:>7}   {last_B:>6}   "
            f"{slots_a}  {slots_b}"
        )


if __name__ == "__main__":
    main()
